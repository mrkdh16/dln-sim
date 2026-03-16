// ============================================================================
// GRADIENT DESCENT SIMULATION
// ============================================================================

const STEPS_PER_FRAME  = 3;  // gradient steps taken each animation frame
const SV_RECORD_EVERY  = 5;  // record singular values every N iterations
const STEPS_SEC_WINDOW = 60; // timestamps kept for steps/sec rolling average
const CHART_UPDATE_MS  = 25; // throttle chart redraws to ~40 fps

let lastChartUpdate = 0;

// ── Initialisation ───────────────────────────────────────────────────────────

function initSimulation() {
  // Dispose previous TF.js tensors
  weightVars.forEach(v => { try { v.dispose(); } catch (_) {} });
  if (targetMatrix)  { try { targetMatrix.dispose();  } catch (_) {} }
  if (weightVar1L)   { try { weightVar1L.dispose();   } catch (_) {} }
  if (sqrtSigmaX)    { try { sqrtSigmaX.dispose();    } catch (_) {} sqrtSigmaX = null; }

  const depth = getDepth();

  // Create weight variables (small random initialisation)
  weightVars = [];
  for (let i = 0; i < depth; i++) {
    const [rows, cols] = getWeightShape(i);
    const init = tf.randomNormal([rows, cols], 0, initScale);
    weightVars.push(tf.variable(init));
    init.dispose();
  }

  // Create random target matrix (standard normal)
  const [outDim, inDim] = getE2EShape();
  targetMatrix = tf.randomNormal([outDim, inDim]);

  // Input covariance: A ∈ ℝ^{inDim×inDim}, Σ_x = AAᵀ, scaled so E[Σ_x] = I.
  // null when the isotropic assumption (Σ_x = I) is active.
  if (!inputCovIsIdentity) {
    sqrtSigmaX = tf.randomNormal([inDim, inDim], 0, 1 / Math.sqrt(inDim));
  }

  // Target SVs for Saxe theory: SVD(W*A) when Σ_x ≠ I, SVD(W*) otherwise.
  targetSVs = sqrtSigmaX
    ? tf.tidy(() => computeSVs(tf.matMul(targetMatrix, sqrtSigmaX)))
    : computeSVs(targetMatrix);

  // Initial effective e2e SVs u_α(0)
  initE2ESVs = tf.tidy(() => {
    const e2e = computeE2E();
    if (sqrtSigmaX) return computeSVs(tf.matMul(e2e, sqrtSigmaX));
    return computeSVs(e2e);
  });

  // Depth-1 comparison network: single weight matrix, same target + hypers
  const init1L = tf.randomNormal([outDim, inDim], 0, initScale);
  weightVar1L  = tf.variable(init1L);
  init1L.dispose();

  // Reset histories and counters
  lossHistory    = [];
  svHistories    = Array.from({ length: depth + 1 }, () => []);
  svHistory1L    = [];
  iterationCount = 0;
  stepTimestamps = [];
  lastChartUpdate = 0;
  _deepODECache   = new Map(); // invalidate deep-ODE theory cache
}

// ── Saxe et al. (2014) theoretical singular value trajectory ─────────────────
//
// Depth L = 1 (single weight matrix, linear ODE):
//   u_α(k) = s_α + (u_α^0 − s_α) · exp(−2ηk)
//
// Depth L = 2 (Saxe et al. Eq. 12, exact closed form, balanced init):
//   u_α(k) = s_α / (1 + (s_α/u_α^0 − 1) · exp(−2 s_α η k))
//
// Depth L ≥ 3 (Saxe et al. Eq. 15, numerical integration):
//   τ du/dt = L · u^{2−2/L} · (s − u)
//
//   where L = depth = number of weight matrices (paper's N_l − 1),
//   τ = 1/η, and the exponent 2−2/L comes from the balanced-init
//   constraint a_i = u^{1/L}.
//
// s_α = α-th SV of target W*, u_α^0 = initial e2e SV, η = learning rate,
// k = step (discrete iteration count).
//
// Returns an array of u_α values evaluated at the iterations in `iters`.

function saxeTheoretical(alpha, iters) {
  const sA    = targetSVs[alpha];
  const u0A   = initE2ESVs[alpha];
  const depth = getDepth();  // L = number of weight matrices

  if (sA == null || u0A == null || sA < 0) return iters.map(() => 0);

  // ── L = 1: linear (exponential approach) ────────────────────────────────
  if (depth === 1) {
    return iters.map(k => sA + (u0A - sA) * Math.exp(-2 * learningRate * k));
  }

  if (u0A <= 0) return iters.map(() => 0);

  // ── L = 2: exact logistic (Saxe Eq. 12) ────────────────────────────────
  if (depth === 2) {
    return iters.map(k => {
      const e = Math.exp(-4 * sA * learningRate * k);
      return sA / (1 + (sA / u0A - 1) * e);
    });
  }

  // ── L ≥ 3: numerical integration of the deep ODE (Saxe Eq. 15) ─────────
  //
  //   du/dk = η · L · u^{2−2/L} · (s − u)
  //
  // We integrate forward with RK4 from k=0 to max(iters), sampling at each
  // requested iteration.  The step size is 1 (one gradient-descent iteration),
  // which is fine because η is small.

  return _integrateDeepODE(sA, u0A, depth, learningRate, iters);
}

// Cache for the deep-ODE integration so we don't re-integrate every chart
// frame for every SV index.  Keyed by (alpha, depth, lr) — invalidated on
// reset because initE2ESVs / targetSVs change.
let _deepODECache = new Map();

function _deepODECacheKey(sA, u0A, depth, lr) {
  // Use truncated floats as key (plenty of precision for cache hits)
  return `${sA.toPrecision(10)}_${u0A.toPrecision(10)}_${depth}_${lr.toPrecision(10)}`;
}

function _integrateDeepODE(sA, u0A, L, lr, iters) {
  if (iters.length === 0) return [];

  const key = _deepODECacheKey(sA, u0A, L, lr);
  let cached = _deepODECache.get(key);

  const maxIter = iters[iters.length - 1]; // iters assumed sorted ascending

  // If cache exists and covers enough iterations, just sample from it
  if (cached && cached.length > maxIter) {
    return iters.map(k => cached[Math.min(k, cached.length - 1)]);
  }

  // RHS of the ODE:  du/dk = lr * L * u^{2-2/L} * (s - u)
  const exponent = 2 - 2 / L;
  function dudk(u) {
    if (u <= 0) return 0;
    return lr * L * Math.pow(u, exponent) * (sA - u);
  }

  // Integrate with RK4, step size = 1 iteration
  const startIdx = cached ? cached.length : 0;
  const trajectory = cached ? cached.slice() : [u0A]; // trajectory[k] = u(k)
  let u = trajectory[trajectory.length - 1];

  const h = 1; // step size = 1 iteration
  for (let k = startIdx; k < maxIter; k++) {
    // Clamp to [0, sA] for numerical safety (u should stay in this range)
    const k1 = dudk(u);
    const k2 = dudk(Math.max(0, u + 0.5 * h * k1));
    const k3 = dudk(Math.max(0, u + 0.5 * h * k2));
    const k4 = dudk(Math.max(0, u + h * k3));
    u = u + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    u = Math.max(0, Math.min(sA + 1e-6, u)); // soft clamp
    trajectory.push(u);
  }

  // Store in cache
  _deepODECache.set(key, trajectory);

  return iters.map(k => trajectory[Math.min(k, trajectory.length - 1)]);
}

// ── Forward pass helpers ─────────────────────────────────────────────────────

// Compute W_L · … · W_1 inside a tf.tidy scope.
// Caller owns the returned tensor (must dispose or it will be disposed by
// the enclosing tidy if one is active).
function computeE2E() {
  let product = weightVars[0];
  for (let i = 1; i < weightVars.length; i++) {
    product = tf.matMul(weightVars[i], product);
  }
  return product; // chain of matMuls, intermediate tensors tracked by tidy
}

// ── Single gradient step ─────────────────────────────────────────────────────

// Helper: compute the loss scalar given a diff tensor and current Σ_x setting.
// diff has shape [outDim, inDim].  When sqrtSigmaX is set, loss = ||diff A||²_F.
function lossFromDiff(diff) {
  if (sqrtSigmaX) return tf.sum(tf.square(tf.matMul(diff, sqrtSigmaX)));
  return tf.sum(tf.square(diff));
}

// Depth-1 comparison: single W minimising the same Σ_x-weighted loss.
function gradientStep1L() {
  tf.tidy(() => {
    const result = tf.variableGrads(() => {
      return lossFromDiff(tf.sub(weightVar1L, targetMatrix));
    });
    const g = result.grads[weightVar1L.name];
    if (g) weightVar1L.assign(tf.sub(weightVar1L, tf.mul(g, learningRate)));
  });
}

function gradientStep() {
  // Wrap everything in tf.tidy so intermediate tensors and gradient tensors
  // are automatically freed.  Variable tensors are never disposed by tidy.
  tf.tidy(() => {
    const result = tf.variableGrads(() => {
      return lossFromDiff(tf.sub(computeE2E(), targetMatrix));
    });

    // Gradient descent: w ← w − lr · ∇w
    weightVars.forEach(v => {
      const g = result.grads[v.name];
      if (g) v.assign(tf.sub(v, tf.mul(g, learningRate)));
    });
  });
}

// ── Loss readout ─────────────────────────────────────────────────────────────

function getCurrentLoss() {
  return tf.tidy(() => {
    return lossFromDiff(tf.sub(computeE2E(), targetMatrix)).dataSync()[0];
  });
}

// ── SVD recording ────────────────────────────────────────────────────────────

// Compute singular values of a 2D tf.Variable / tf.Tensor using numeric.js.
function computeSVs(t) {
  let mat = t.arraySync(); // 2D JS array
  if (mat.length < mat[0].length) mat = numeric.transpose(mat); // numeric.svd needs rows ≥ cols
  try {
    return numeric.svd(mat).S;
  } catch (_) {
    return [];
  }
}

function recordSVs() {
  const depth = getDepth();

  // Singular values of each weight matrix
  for (let i = 0; i < depth; i++) {
    const svs = computeSVs(weightVars[i]);
    svHistories[i].push({ iter: iterationCount, svs });
  }

  // Singular values of the end-to-end product (projected through Σ_x^{1/2}
  // when the isotropic assumption is off, to match the theory's basis).
  const e2eSVs = tf.tidy(() => {
    const e2e = computeE2E();
    if (sqrtSigmaX) return computeSVs(tf.matMul(e2e, sqrtSigmaX));
    return computeSVs(e2e);
  });
  svHistories[depth].push({ iter: iterationCount, svs: e2eSVs });

  // Depth-1 comparison network SVs (same projection for fair comparison)
  const svs1L = sqrtSigmaX
    ? tf.tidy(() => computeSVs(tf.matMul(weightVar1L, sqrtSigmaX)))
    : computeSVs(weightVar1L);
  svHistory1L.push({ iter: iterationCount, svs: svs1L });
}

// ── Animation loop ───────────────────────────────────────────────────────────

function simulationLoop() {
  if (!isRunning) return;

  // Multiple gradient steps per frame for better throughput
  for (let s = 0; s < STEPS_PER_FRAME; s++) {
    gradientStep();
    gradientStep1L();
    iterationCount++;
    stepTimestamps.push(performance.now());
    if (stepTimestamps.length > STEPS_SEC_WINDOW) stepTimestamps.shift();
  }

  // Record loss after this batch of steps
  const loss = getCurrentLoss();
  lossHistory.push({ iter: iterationCount, loss });

  // Record singular values periodically
  if (iterationCount % SV_RECORD_EVERY === 0) {
    recordSVs();
    // If a matrix is selected but its chart was hidden (no prior data),
    // now that we have data, bring the panel up.
    if (selectedMatrixIdx !== null) refreshSVPanel();
  }

  // Update UI
  updateMetricsDisplay(loss);

  const now = performance.now();
  if (now - lastChartUpdate >= CHART_UPDATE_MS) {
    updateLossChart();
    if (selectedMatrixIdx !== null) updateSVChart();
    lastChartUpdate = now;
  }

  draw();

  animFrameId = requestAnimationFrame(simulationLoop);
}

// ── Public controls ───────────────────────────────────────────────────────────

function startSim() {
  if (isRunning) return;
  if (weightVars.length === 0) initSimulation();
  isRunning = true;
  document.getElementById('start-btn').textContent = 'pause';
  document.getElementById('start-btn').classList.add('running');
  simulationLoop();
}

function pauseSim() {
  isRunning = false;
  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
  document.getElementById('start-btn').textContent = 'start';
  document.getElementById('start-btn').classList.remove('running');
}

function resetSim() {
  pauseSim();
  initSimulation();
  selectedMatrixIdx = getDepth(); // restore e2e selection (initial state)
  updateMetricsDisplay(null);
  clearCharts();
  refreshSVPanel();
  draw();
}