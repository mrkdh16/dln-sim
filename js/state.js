// ============================================================================
// GLOBAL STATE
// ============================================================================
//
// A deep linear network is:
//   f(x) = W_L · W_{L-1} · … · W_1 · x
//
// Dimensions are stored as:
//   dims = [d_0, d_1, …, d_L]
//   dims[0]  = input dimension
//   dims[L]  = output dimension
//   depth L  = dims.length - 1
//
// Weight matrix i (0-indexed) has shape [dims[i+1], dims[i]].
// The end-to-end (e2e) product has shape [dims[L], dims[0]].
// ============================================================================

// Network dimensions (mutable when user applies a new config)
let dims = [6, 6, 6]; // default: depth=2, all dims=6 (matches depth2 preset)

// TF.js trainable variables — one per weight matrix
// weightVars[i].shape = [dims[i+1], dims[i]]
let weightVars = [];

// Fixed random target matrix, shape [dims[L], dims[0]]
let targetMatrix = null;

// Training history
let lossHistory = [];   // [{iter, loss}, …]

// Singular value histories: one entry per weight matrix + one for e2e
// svHistories[i]     = [{iter, svs: float[]}, …]  for i < depth
// svHistories[depth] = [{iter, svs: float[]}, …]  for the e2e product
let svHistories = [];

// Simulation control
let isRunning     = false;
let iterationCount = 0;
let animFrameId   = null;
let stepTimestamps = []; // rolling window for steps/sec

// Hyperparameters (match depth2 preset defaults)
let learningRate = 0.005;
let initScale    = 0.01;

// Activation function key for nonlinear presets: null = linear, 'relu'/'tanh'/'sigmoid'
let activation = null;

// UI — which matrix is currently selected on the canvas
// null = none, 0…depth-1 = weight matrix W_{i+1}, depth = e2e product
let selectedMatrixIdx = 2; // default: W_e2e (depth=2 for dims=[6,6,6])

// ── Saxe et al. theoretical comparison ──────────────────────────────────────
// Singular values of the target matrix W* (computed at init, descending order)
let targetSVs  = [];
// Initial effective (e2e) singular values u_α(0) (computed at init, descending order)
let initE2ESVs = [];
// SV chart curve-group visibility (controlled by toggle pills)
let showDeep    = true;   // empirical deep network curves
let showTheory  = true;   // Saxe et al. theoretical overlay

// ── Depth-1 (shallow) comparison network ─────────────────────────────────────
// A single-layer network trained in parallel so users can compare dynamics.
// Same target matrix, same learning rate, same init scale as the deep network.
let weightVar1L  = null;   // tf.Variable, shape [d_L, d_0]
let svHistory1L  = [];     // [{iter, svs}, …]

// ── Input covariance (Saxe et al. assumption) ─────────────────────────────────
// Saxe et al. assume Σ_x = (1/P)∑xxᵀ = I (isotropic inputs).
// When this is disabled we sample a random A and use Σ_x = AAᵀ ≠ I.
// The loss becomes ||(W_e2e − W*)A||²_F, and the theory target SVs become
// SVD(W*A) instead of SVD(W*).  TF.js autograd handles the gradient change.
let alignedInit        = true;  // toggle: use decoupled/aligned initial conditions
let inputCovIsIdentity = true;  // toggle: true → Σ_x = I
let sqrtSigmaX         = null;  // tf.Tensor [inDim, inDim], null when identity

// ── Derived helpers ──────────────────────────────────────────────────────────

function getDepth()      { return dims.length - 1; }
function getWeightShape(i) { return [dims[i + 1], dims[i]]; }
function getE2EShape()   { return [dims[dims.length - 1], dims[0]]; }

// Number of singular values for item i (i < depth → weight matrix, i == depth → e2e)
function numSVsFor(i) {
  const shape = (i < getDepth()) ? getWeightShape(i) : getE2EShape();
  return Math.min(shape[0], shape[1]);
}
