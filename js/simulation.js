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
}

// ── Saxe et al. (2014) theoretical singular value trajectory ─────────────────
//
// Depth L = 1 (single weight matrix):
//   u_α(k) = s_α + (u_α^0 − s_α) · exp(−2ηk)          [linear approach]
//
// Depth L = 2 (Saxe et al. exact formula, balanced initialisation):
//   u_α(k) = s_α / (1 + (s_α/u_α^0 − 1) · exp(−2 s_α η k))  [logistic]
//
// Depth L ≥ 3:  above formula used as approximation; convergence time scale
//               changes but qualitative sigmoidal shape persists.
//
// s_α = α-th SV of target W*, u_α^0 = initial e2e SV, η = learning rate, k = step.
// Returns an array of u_α values evaluated at the iterations in `iters`.
function saxeTheoretical(alpha, iters) {
  const sA    = targetSVs[alpha];
  const u0A   = initE2ESVs[alpha];
  const depth = getDepth();

  if (sA == null || u0A == null || sA < 0) return iters.map(() => 0);

  if (depth === 1) {
    // u_α(k) = sA + (u0A - sA) * exp(-2 * η * k)
    return iters.map(k => sA + (u0A - sA) * Math.exp(-2 * learningRate * k));
  }

  // depth ≥ 2: logistic (exact for L=2, approximate for L>2)
  if (u0A <= 0) return iters.map(() => 0);
  return iters.map(k => {
    const e = Math.exp(-4 * sA * learningRate * k);
    return sA / (1 + (sA / u0A - 1) * e);
  });
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
