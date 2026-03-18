// ============================================================================
// SIMPLE WIDGET — independent simulation, completely separate from the bottom
// widget's simulation state.  All variables and functions are prefixed s_.
// ============================================================================

const S_STEPS_PER_FRAME    = 10;
const S_SV_RECORD_EVERY    = 5;
const S_CHART_UPDATE_EVERY = 2;
const S_NUM_DATA           = 64;  // number of data points (x^μ, y^μ = W* x^μ)

// ── State ─────────────────────────────────────────────────────────────────────

let s_dims         = [6, 6, 6];
let s_activation   = null;
let s_learningRate = 0.01;
let s_initScale    = 0.01;

let s_weightVars   = [];
let s_targetMatrix = null;
let s_inputData    = null;  // fixed random inputs X,    shape [d_in,  S_NUM_DATA]
let s_targetData   = null;  // targets Y = W* X,         shape [d_out, S_NUM_DATA]

let s_lossHistory  = [];
let s_svHistories  = [];   // s_svHistories[s_getDepth()] = e2e SVs

let s_isRunning    = false;
let s_completed    = false; // true once the sim finishes naturally
let s_iterCount    = 0;
let s_frameCount   = 0;
let s_animFrameId  = null;
let s_stopAt       = null; // set to 3T when loss first crosses 0.1

// ── Helpers ───────────────────────────────────────────────────────────────────

function s_getDepth()        { return s_dims.length - 1; }
function s_getWeightShape(i) { return [s_dims[i + 1], s_dims[i]]; }
function s_getE2EShape()     { return [s_dims[s_dims.length - 1], s_dims[0]]; }

// ── Initialisation ────────────────────────────────────────────────────────────

function s_initSim() {
  s_weightVars.forEach(v => { try { v.dispose(); } catch (_) {} });
  if (s_targetMatrix) { try { s_targetMatrix.dispose(); } catch (_) {} }
  if (s_inputData)    { try { s_inputData.dispose();    } catch (_) {} }
  if (s_targetData)   { try { s_targetData.dispose();   } catch (_) {} }

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

  // Fixed random dataset: x^μ ~ N(0, I), y^μ = W* x^μ
  // Inputs drawn iid N(0,1) so E[X Xᵀ / P] = I (whitened), keeping the
  // input-output correlation matrix f(X) Xᵀ / P in the same scale as W*.
  s_inputData  = tf.randomNormal([inDim, S_NUM_DATA]);
  s_targetData = tf.matMul(s_targetMatrix, s_inputData);

  s_lossHistory = [];
  s_svHistories = Array.from({ length: depth + 1 }, () => []);
  s_iterCount   = 0;
  s_frameCount  = 0;
  s_stopAt      = null;
  s_completed   = false;
  s_hideAllCardResets();
}

// ── Forward pass ──────────────────────────────────────────────────────────────
// Takes an explicit input matrix X of shape [d_in, P].

function s_computeForwardOn(X) {
  if (!s_activation) {
    let p = tf.matMul(s_weightVars[0], X);
    for (let i = 1; i < s_weightVars.length; i++) {
      p = tf.matMul(s_weightVars[i], p);
    }
    return p;
  }
  const act = s_activation === 'relu'    ? tf.relu
            : s_activation === 'tanh'    ? tf.tanh
            : s_activation === 'sigmoid' ? tf.sigmoid : null;
  let h = tf.matMul(s_weightVars[0], X);
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
      const pred = s_computeForwardOn(s_inputData);
      return tf.div(tf.sum(tf.square(tf.sub(pred, s_targetData))), S_NUM_DATA);
    });
    s_weightVars.forEach(v => {
      const g = result.grads[v.name];
      if (g) v.assign(tf.sub(v, tf.mul(g, s_learningRate)));
    });
  });
}

function s_getCurrentLoss() {
  return tf.tidy(() => {
    const pred = s_computeForwardOn(s_inputData);
    return tf.div(tf.sum(tf.square(tf.sub(pred, s_targetData))), S_NUM_DATA).dataSync()[0];
  });
}

// ── SV recording ─────────────────────────────────────────────────────────────
// Plots singular values of the input-output correlation matrix f(X) Xᵀ / P.
// For a linear network this equals W_e2e (X Xᵀ / P) ≈ W_e2e under whitened inputs.
// For a nonlinear network it captures the effective linear readout of the learned map.

function s_recordSVs() {
  const depth  = s_getDepth();
  const e2eSVs = tf.tidy(() => {
    const fX   = s_computeForwardOn(s_inputData);
    const corr = tf.div(tf.matMul(fX, tf.transpose(s_inputData)), S_NUM_DATA);
    return computeSVs(corr);
  });
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

  if (s_stopAt === null && currentLoss < 0.1) {
    s_stopAt = 3 * s_iterCount; // stop at 3T
  }

  if (s_stopAt !== null && s_iterCount >= s_stopAt) {
    updateSimpleCharts();
    s_completed = true;
    s_pauseSim();
    // Override the play icon to replay
    const activeCard = document.querySelector('.preset-card.active');
    if (activeCard) {
      const icon = activeCard.querySelector('.preset-card-play-icon i');
      if (icon) icon.className = 'fas fa-rotate-right';
    }
    return;
  }

  s_animFrameId = requestAnimationFrame(s_simulationLoop);
}

// ── Start / pause / reset ─────────────────────────────────────────────────────

function s_showCardReset() {
  const activeCard = document.querySelector('.preset-card.active');
  if (activeCard) {
    const btn = activeCard.querySelector('.preset-card-reset');
    if (btn) btn.style.display = 'block';
  }
}

function s_hideAllCardResets() {
  document.querySelectorAll('.preset-card-reset').forEach(btn => {
    btn.style.display = 'none';
  });
}

function s_startSim() {
  if (s_isRunning) return;
  if (s_weightVars.length === 0) s_initSim();
  s_isRunning = true;
  s_hideAllCardResets();
  updateCardPlayIcon(true);
  s_simulationLoop();
}

function s_pauseSim() {
  s_isRunning = false;
  if (s_animFrameId) { cancelAnimationFrame(s_animFrameId); s_animFrameId = null; }
  updateCardPlayIcon(false);
  if (!s_completed) s_showCardReset();
}

function s_resetSim() {
  s_pauseSim();
  s_initSim();
  clearSimpleCharts();
}

// ── Apply preset ──────────────────────────────────────────────────────────────

function s_updateParamsDisplay() {
  const el = document.getElementById('simple-params');
  if (!el) return;
  const depth = s_getDepth();
  const width = depth > 1 ? s_dims[1] : s_dims[0];
  const lr    = s_learningRate % 1 === 0 ? s_learningRate : s_learningRate.toPrecision(2);
  const scale = s_initScale    % 1 === 0 ? s_initScale    : s_initScale.toPrecision(2);
  el.textContent = `lr = ${lr}  ·  width = ${width}  ·  init ~ N(0, ${scale})`;
}

function s_applyPreset(key) {
  const p = PRESETS[key];
  if (!p) return;

  s_dims         = p.dims.slice();
  s_learningRate = p.learningRate;
  s_initScale    = p.initScale;
  s_activation   = p.activation || null;

  updateEquation(key);
  s_updateParamsDisplay();

  document.querySelectorAll('.preset-card').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.preset === key);
  });

  s_resetSim();
}
