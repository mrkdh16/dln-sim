#!/usr/bin/env python3
"""
Precompute training trajectories for the Linear vs Nonlinear widget.

Runs gradient descent for each preset and saves loss + W₁ singular value
history to linear_nonlinear_data.json.

Usage:
    python precompute_ln.py

Output format:
    {
      "depth1": {
        "loss": [{"iter": 0, "loss": 1.23}, ...],
        "svs":  [{"iter": 0, "svs": [0.8, 0.6, ...]}, ...]
      },
      ...
    }
"""

import json
import time

import numpy as np

# ---------------------------------------------------------------------------
# Configuration (matches JS widget-linear-nonlinear.js PRESETS)
# ---------------------------------------------------------------------------

NUM_DATA     = 64
RECORD_EVERY = 50     # record metrics every N gradient steps
MAX_ITERS    = 80_000

PRESETS = {
    "depth1": {
        "dims": [6, 6],        "lr": 0.01,  "init_scale": 0.05,
        "activation": None,    "use_teacher": False,
    },
    "depth2": {
        "dims": [6, 6, 6],     "lr": 0.005, "init_scale": 0.01,
        "activation": None,    "use_teacher": False,
    },
    "depth3": {
        "dims": [6, 6, 6, 6],  "lr": 0.005, "init_scale": 0.01,
        "activation": None,    "use_teacher": False,
    },
    "relu_shallow": {
        "dims": [6, 6, 6],     "lr": 0.005, "init_scale": 0.01,
        "activation": "relu",  "use_teacher": True,
    },
    "relu_deep": {
        "dims": [6, 6, 6, 6],  "lr": 0.005, "init_scale": 0.01,
        "activation": "relu",  "use_teacher": True,
    },
    "tanh": {
        "dims": [6, 6, 6, 6],  "lr": 0.005, "init_scale": 0.01,
        "activation": "tanh",  "use_teacher": True,
    },
}


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _act(name, x):
    if name == "relu": return np.maximum(0.0, x)
    if name == "tanh": return np.tanh(x)
    return x


def _d_act(name, pre):
    """Elementwise derivative of activation at pre-activation values."""
    if name == "relu": return (pre > 0.0).astype(np.float64)
    if name == "tanh": return 1.0 - np.tanh(pre) ** 2
    return np.ones_like(pre)


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _forward(weights, X, activation):
    h = X
    for i, W in enumerate(weights):
        h = W @ h
        if i < len(weights) - 1 and activation:
            h = _act(activation, h)
    return h


def _make_target(dims):
    """Diagonal target matrix with SVs (n-i)/(n+1), i=0..n-1."""
    out_d, in_d = dims[-1], dims[0]
    n = min(out_d, in_d)
    T = np.zeros((out_d, in_d))
    for i in range(n):
        T[i, i] = (n - i) / (n + 1)
    return T


def _gd_step(weights, X, Y, lr, activation):
    """
    One in-place gradient-descent step.
    Returns the MSE loss *before* the weight update.
    """
    N = X.shape[1]

    # ── Forward pass (cache layer inputs and pre-activations) ─────────────
    layer_ins = [X]   # layer_ins[i]  = input fed into weights[i]
    pre_acts  = []    # pre_acts[i]   = weights[i] @ layer_ins[i]  (pre-activation)
    h = X
    for i, W in enumerate(weights):
        pre = W @ h
        pre_acts.append(pre)
        if i < len(weights) - 1 and activation:
            h = _act(activation, pre)
        else:
            h = pre
        layer_ins.append(h)

    output = layer_ins[-1]
    loss   = float(np.sum((output - Y) ** 2) / N)

    # ── Backward pass ──────────────────────────────────────────────────────
    residual = 2.0 * (output - Y) / N
    grads = [None] * len(weights)
    for i in range(len(weights) - 1, -1, -1):
        grads[i] = residual @ layer_ins[i].T
        if i > 0:
            residual = weights[i].T @ residual   # uses original (pre-update) W
            if activation:
                residual = residual * _d_act(activation, pre_acts[i - 1])

    # ── Weight update ──────────────────────────────────────────────────────
    for i in range(len(weights)):
        weights[i] = weights[i] - lr * grads[i]

    return loss


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(key, cfg, rng):
    dims        = cfg["dims"]
    lr          = cfg["lr"]
    init_scale  = cfg["init_scale"]
    activation  = cfg["activation"]
    use_teacher = cfg["use_teacher"]
    depth       = len(dims) - 1

    # ── Weights (random normal, same as JS widget) ─────────────────────────
    weights = [
        rng.normal(0.0, init_scale, (dims[i + 1], dims[i]))
        for i in range(depth)
    ]

    # ── Fixed input data ───────────────────────────────────────────────────
    X = rng.standard_normal((dims[0], NUM_DATA))

    # ── Target outputs ─────────────────────────────────────────────────────
    target_mat = _make_target(dims)
    if use_teacher:
        # He-initialised teacher network (1/√c per layer)
        teacher = [
            rng.normal(0.0, 1.0 / np.sqrt(dims[i]), (dims[i + 1], dims[i]))
            for i in range(depth)
        ]
        Y = _forward(teacher, X, activation)
    else:
        Y = target_mat @ X

    # ── Training loop ──────────────────────────────────────────────────────
    loss_history = []
    sv_history   = []
    stop_at      = None
    initial_loss = None

    t0 = time.perf_counter()

    for step in range(MAX_ITERS):
        loss = _gd_step(weights, X, Y, lr, activation)

        if step % RECORD_EVERY == 0:
            if initial_loss is None:
                initial_loss = loss

            loss_history.append({"iter": step, "loss": round(loss, 6)})

            svs = np.linalg.svd(weights[0], compute_uv=False).tolist()
            sv_history.append({"iter": step, "svs": [round(v, 6) for v in svs]})

            # Stop criterion: run 2× longer after loss drops below 1% of initial
            threshold = (initial_loss or 1.0) * 0.01
            if stop_at is None and loss < threshold:
                stop_at = step * 2
            if stop_at is not None and step >= stop_at:
                break

    elapsed = time.perf_counter() - t0
    print(f"  {key:14s}: {step + 1:6d} steps, {len(loss_history):4d} pts, "
          f"final loss={loss:.4g}, {elapsed:.1f}s")

    return {"loss": loss_history, "svs": sv_history}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    rng    = np.random.default_rng(42)
    result = {}

    for key, cfg in PRESETS.items():
        print(f"Simulating {key}…")
        result[key] = simulate(key, cfg, rng)

    out_path = "linear_nonlinear_data.json"
    with open(out_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    size_kb = len(json.dumps(result)) // 1024
    print(f"\nSaved {out_path}  (~{size_kb} KB)")


if __name__ == "__main__":
    main()
