// ============================================================================
// SIMPLE WIDGET — independent simulation, completely separate from the bottom
// widget's simulation state.  All variables and functions are prefixed s_.
// ============================================================================

const S_STEPS_PER_FRAME    = 10;
const S_SV_RECORD_EVERY    = 5;
const S_CHART_UPDATE_EVERY = 2;

// ── State ─────────────────────────────────────────────────────────────────────

let s_dims         = [6, 6, 6];
let s_activation   = null;
let s_learningRate = 0.01;
let s_initScale    = 0.01;

let s_weightVars   = [];
let s_targetMatrix = null;

let s_lossHistory  = [];
let s_svHistories  = [];   // s_svHistories[s_getDepth()] = e2e SVs

let s_isRunning    = false;
let s_iterCount    = 0;
let s_frameCount   = 0;
let s_animFrameId  = null;

// ── Helpers ───────────────────────────────────────────────────────────────────

function s_getDepth()        { return s_dims.length - 1; }
function s_getWeightShape(i) { return [s_dims[i + 1], s_dims[i]]; }
function s_getE2EShape()     { return [s_dims[s_dims.length - 1], s_dims[0]]; }

// ── Initialisation ────────────────────────────────────────────────────────────

function s_initSim() {
  s_weightVars.forEach(v => { try { v.dispose(); } catch (_) {} });
  if (s_targetMatrix) { try { s_targetMatrix.dispose(); } catch (_) {} }

  const depth = s_getDepth();
  s_weightVars = [];
  for (let i = 0; i < depth; i++) {
    const [rows, cols] = s_getWeightShape(i);
    const init = tf.randomNormal([rows, cols], 0, s_initScale);
    s_weightVars.push(tf.variable(init));
    init.dispose();
  }

  const [outDim, inDim] = s_getE2EShape();
  const n = Math.min(outDim, inDim);
  // Equally spaced in (0, 1): σ_i = (n - i) / (n + 1) for i = 0, 1, ..., n-1
  const diagArr  = Array.from({ length: n }, (_, i) => (n - i) / (n + 1));
  const diagVals = tf.tensor1d(diagArr);
  s_targetMatrix = tf.pad(tf.diag(diagVals), [[0, outDim - n], [0, inDim - n]]);
  diagVals.dispose();

  s_lossHistory = [];
  s_svHistories = Array.from({ length: depth + 1 }, () => []);
  s_iterCount   = 0;
  s_frameCount  = 0;
}

// ── Forward pass ──────────────────────────────────────────────────────────────
// Uses identity as implicit input (same convention as the main simulation).

function s_computeForward() {
  if (!s_activation) {
    let p = s_weightVars[0];
    for (let i = 1; i < s_weightVars.length; i++) {
      p = tf.matMul(s_weightVars[i], p);
    }
    return p;
  }
  const act = s_activation === 'relu'    ? tf.relu
            : s_activation === 'tanh'    ? tf.tanh
            : s_activation === 'sigmoid' ? tf.sigmoid : null;
  let h = s_weightVars[0];
  for (let i = 1; i < s_weightVars.length; i++) {
    h = act(h);
    h = tf.matMul(s_weightVars[i], h);
  }
  return h;
}

// ── Gradient step ─────────────────────────────────────────────────────────────

function s_gradientStep() {
  tf.tidy(() => {
    const result = tf.variableGrads(() => {
      return tf.sum(tf.square(tf.sub(s_computeForward(), s_targetMatrix)));
    });
    s_weightVars.forEach(v => {
      const g = result.grads[v.name];
      if (g) v.assign(tf.sub(v, tf.mul(g, s_learningRate)));
    });
  });
}

function s_getCurrentLoss() {
  return tf.tidy(() => {
    return tf.sum(tf.square(tf.sub(s_computeForward(), s_targetMatrix))).dataSync()[0];
  });
}

// ── SV recording ─────────────────────────────────────────────────────────────

function s_recordSVs() {
  const depth  = s_getDepth();
  const e2eSVs = tf.tidy(() => computeSVs(s_computeForward()));
  s_svHistories[depth].push({ iter: s_iterCount, svs: e2eSVs });
}

// ── Animation loop ────────────────────────────────────────────────────────────

function s_simulationLoop() {
  if (!s_isRunning) return;

  for (let i = 0; i < S_STEPS_PER_FRAME; i++) {
    s_gradientStep();
    s_iterCount++;
  }

  const currentLoss = s_getCurrentLoss();
  s_lossHistory.push({ iter: s_iterCount, loss: currentLoss });

  if (s_iterCount % S_SV_RECORD_EVERY === 0) s_recordSVs();

  s_frameCount++;
  if (s_frameCount % S_CHART_UPDATE_EVERY === 0) updateSimpleCharts();

  if (currentLoss < 0.1) {
    updateSimpleCharts();
    s_pauseSim();
    return;
  }

  s_animFrameId = requestAnimationFrame(s_simulationLoop);
}

// ── Start / pause / reset ─────────────────────────────────────────────────────

function s_startSim() {
  if (s_isRunning) return;
  if (s_weightVars.length === 0) s_initSim();
  s_isRunning = true;
  updateCardPlayIcon(true);
  s_simulationLoop();
}

function s_pauseSim() {
  s_isRunning = false;
  if (s_animFrameId) { cancelAnimationFrame(s_animFrameId); s_animFrameId = null; }
  updateCardPlayIcon(false);
}

function s_resetSim() {
  s_pauseSim();
  s_initSim();
  clearSimpleCharts();
}

// ── Apply preset ──────────────────────────────────────────────────────────────

function s_applyPreset(key) {
  const p = PRESETS[key];
  if (!p) return;

  s_dims         = p.dims.slice();
  s_learningRate = p.learningRate;
  s_initScale    = p.initScale;
  s_activation   = p.activation || null;

  updateEquation(key);

  document.querySelectorAll('.preset-card').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.preset === key);
  });

  s_resetSim();
}
