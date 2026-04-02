// ============================================================================
// WIDGET UTILITIES — shared pure helpers for all standalone DLN widgets.
// Requires: TF.js, numeric.js, Chart.js  (loaded before widget scripts).
// ============================================================================

// ── Colors ───────────────────────────────────────────────────────────────────

const DLN_SV_COLORS = [
  '#0969da', '#1a7f37', '#d1242f', '#8250df',
  '#bc4c00', '#005cc5', '#2da44e', '#cf222e',
  '#6e40c9', '#953800',
];

// ── Single-active compute manager ────────────────────────────────────────────
//
// Only one widget may be running at a time.  Call WidgetManager.register(uid,
// pauseFn) at widget bootstrap, then WidgetManager.requestStart(uid) at the
// top of each widget's _start() function.

const WidgetManager = (() => {
  const registry = {};   // uid → pauseFn
  let activeId   = null;

  return {
    register(uid, pauseFn) {
      registry[uid] = pauseFn;
    },
    requestStart(uid) {
      if (activeId && activeId !== uid && registry[activeId]) {
        registry[activeId]();  // pause the currently active widget
      }
      activeId = uid;
    },
    notifyStop(uid) {
      if (activeId === uid) activeId = null;
    },
  };
})();

// ── SVD via numeric.js ────────────────────────────────────────────────────────

function dlnComputeSVs(t) {
  let mat = t.arraySync();
  if (mat.length < mat[0].length) mat = numeric.transpose(mat);
  try { return numeric.svd(mat).S; } catch (_) { return []; }
}

// Returns { U, S, V } where mat = U diag(S) V^T, from a 2D JS array.
function dlnSVD(mat2d) {
  let mat = mat2d;
  let swapped = false;
  if (mat.length < mat[0].length) { mat = numeric.transpose(mat); swapped = true; }
  const { U, S, V } = numeric.svd(mat);
  return swapped ? { U: V, S, V: U } : { U, S, V };
}

// ── Downsampling ──────────────────────────────────────────────────────────────

function dlnDownsample(arr, maxPts) {
  if (arr.length <= maxPts) return arr;
  const stride = Math.pow(2, Math.ceil(Math.log2(arr.length / maxPts)));
  const out = [];
  for (let i = 0; i < arr.length; i++) {
    if (i % stride === 0 || i === arr.length - 1) out.push(arr[i]);
  }
  return out;
}

// ── Saxe theoretical singular value trajectory ────────────────────────────────
//
// Pure function — takes all inputs, no globals.
// cacheMap (optional Map) memoises the RK4 integration per (sA,u0A,L,lr) key.
//
// Returns array of theory values evaluated at iterations in `iters` (sorted ↑).

function dlnSaxeTheory(sA, u0A, depth, lr, iters, cacheMap) {
  if (sA == null || u0A == null || sA < 0 || iters.length === 0) return iters.map(() => 0);

  // L = 1: exponential approach
  if (depth === 1) {
    return iters.map(k => sA + (u0A - sA) * Math.exp(-2 * lr * k));
  }
  if (u0A <= 0) return iters.map(() => 0);

  // L = 2: exact logistic (Saxe Eq. 12)
  if (depth === 2) {
    return iters.map(k => {
      const e = Math.exp(-4 * sA * lr * k);
      return sA / (1 + (sA / u0A - 1) * e);
    });
  }

  // L ≥ 3: RK4 integration of du/dk = 2*lr*L*u^{2-2/L}*(s-u)
  const exponent = 2 - 2 / depth;
  function dudk(u) {
    if (u <= 0) return 0;
    return 2 * lr * depth * Math.pow(u, exponent) * (sA - u);
  }

  const key = cacheMap
    ? `${sA.toPrecision(10)}_${u0A.toPrecision(10)}_${depth}_${lr.toPrecision(10)}`
    : null;
  let traj = (cacheMap && key) ? cacheMap.get(key) : null;

  const maxIter = iters[iters.length - 1];
  if (traj && traj.length > maxIter) {
    return iters.map(k => traj[Math.min(k, traj.length - 1)]);
  }

  const startIdx = traj ? traj.length : 0;
  traj = traj ? traj.slice() : [u0A];
  let u = traj[traj.length - 1];

  for (let k = startIdx; k < maxIter; k++) {
    const k1 = dudk(u);
    const k2 = dudk(Math.max(0, u + 0.5 * k1));
    const k3 = dudk(Math.max(0, u + 0.5 * k2));
    const k4 = dudk(Math.max(0, u + k3));
    u = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    u = Math.max(0, Math.min(sA + 1e-6, u));
    traj.push(u);
  }

  if (cacheMap && key) cacheMap.set(key, traj);
  return iters.map(k => traj[Math.min(k, traj.length - 1)]);
}

// ── Aligned (balanced) initialisation ────────────────────────────────────────
//
// Sets each W_l = R[l+1] @ (initScale * I) @ R[l]^T
// where R[0]=V_T and R[depth]=U_T come from the SVD of the target matrix,
// and intermediate R are random orthogonal matrices.
// Modifies weightVars in-place.

function dlnAlignedInit(weightVars, dims, targetMatrix, initScale) {
  const depth = dims.length - 1;
  const { U: U_T, V: V_T } = dlnSVD(targetMatrix.arraySync());

  const R = [V_T];
  for (let l = 1; l < depth; l++) {
    const d    = dims[l];
    const rand = Array.from({ length: d }, () =>
      Array.from({ length: d }, () => (Math.random() - 0.5) * 2));
    R.push(numeric.svd(rand).U);
  }
  R.push(U_T);

  weightVars.forEach((wv, l) => {
    const outD = dims[l + 1], inD = dims[l], k = Math.min(outD, inD);
    const D = Array.from({ length: outD }, (_, i) =>
      Array.from({ length: inD }, (_, j) => (i === j && i < k) ? initScale : 0));
    wv.assign(tf.tensor2d(numeric.dot(R[l + 1], numeric.dot(D, numeric.transpose(R[l])))));
  });
}

// ── Chart options ─────────────────────────────────────────────────────────────
// Distill.pub style: subdued gridlines, clean axes, no legend.

function dlnChartOptions(xLabel, yLabel, extra) {
  return Object.assign({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        title: { display: !!xLabel, text: xLabel || '', color: '#8c959f', font: { size: 10 } },
        ticks: { color: '#8c959f', maxTicksLimit: 5, maxRotation: 0, font: { size: 9 } },
        grid:  { color: 'rgba(0,0,0,0.05)', lineWidth: 1 },
        border:{ color: '#d0d7de', dash: [] },
      },
      y: {
        title: { display: !!yLabel, text: yLabel || '', color: '#8c959f', font: { size: 10 } },
        ticks: { color: '#8c959f', font: { size: 9 } },
        grid:  { color: 'rgba(0,0,0,0.05)', lineWidth: 1 },
        border:{ color: '#d0d7de', dash: [] },
        beginAtZero: true,
      },
    },
    plugins: { legend: { display: false } },
  }, extra || {});
}

// ── Scroll-triggered autoplay ─────────────────────────────────────────────────
//
// Calls startFn() the first time `element` is at least `threshold` visible
// in the viewport.  Uses IntersectionObserver (no-op in environments without it).
// Returns the observer so callers can disconnect it if needed.

function dlnScrollAutoplay(element, startFn, threshold) {
  threshold = threshold === undefined ? 0.55 : threshold;
  if (!element || typeof IntersectionObserver === 'undefined') return null;
  let triggered = false;
  const obs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting && !triggered) {
      triggered = true;
      startFn();
    }
  }, { threshold });
  obs.observe(element);
  return obs;
}

// ── Tiny UID for unique canvas IDs ────────────────────────────────────────────

let _dlnWidgetCounter = 0;
function dlnUID() { return 'dln' + (++_dlnWidgetCounter); }
