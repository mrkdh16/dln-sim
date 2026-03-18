// ============================================================================
// UI — controls panel, sliders, metrics display, presets
// ============================================================================

// ── Logarithmic slider utilities ──────────────────────────────────────────────

const LR_VALUES         = generateLogValues(-4, -1).concat([1]);   // 10⁻⁴ … 1
const SCALE_VALUES      = generateLogValues(-4, -1).concat([10]);  // 10⁻⁵ … 10
const HIDDEN_DIM_VALUES = [1,2,3,4,5,6,7,8,10,12,16,20,24,32,40,48,64,80,96,128,160,192,256];

function generateLogValues(minExp, maxExp) {
  const out = [];
  for (let k = minExp; k <= maxExp; k++) {
    const p = Math.pow(10, k);
    for (let m = 1; m <= 9; m++) out.push(m * p);
  }
  return out;
}

function sliderIndexToValue(sliderPos, values) {
  const idx = Math.round((sliderPos / 100) * (values.length - 1));
  return values[Math.max(0, Math.min(values.length - 1, idx))];
}

function valueToSliderPos(value, values) {
  let closestIdx = 0;
  let minDiff = Math.abs(values[0] - value);
  for (let i = 1; i < values.length; i++) {
    const diff = Math.abs(values[i] - value);
    if (diff < minDiff) { minDiff = diff; closestIdx = i; }
  }
  return (closestIdx / (values.length - 1)) * 100;
}

// Format a small number in scientific notation using HTML <sup>
function fmtScientific(v) {
  if (v >= 1) return v.toString();
  const exp  = Math.floor(Math.log10(v));
  const mant = Math.round((v / Math.pow(10, exp)) * 1e9) / 1e9;
  if (Math.abs(mant - 1) < 1e-9) return `10<sup>${exp}</sup>`;
  return `${mant}×10<sup>${exp}</sup>`;
}

// ── Hidden dim value (kept in sync with slider) ───────────────────────────────

let hiddenDimVal = 6;

// ── Presets ───────────────────────────────────────────────────────────────────

const PRESETS = {
  'depth1': {
    dims: [6, 6],
    learningRate: 0.005,
    initScale: 0.05,
    inputCovIsIdentity: true,
  },
  'depth2': {
    dims: [6, 6, 6],
    learningRate: 0.005,
    initScale: 0.01,
    inputCovIsIdentity: true,
  },
  'plateau': {
    dims: [6, 6, 6, 6],
    learningRate: 0.005,
    initScale: 0.01,
    inputCovIsIdentity: true,
  },
   'relu_shallow': {
    dims: [6, 6, 6],
    learningRate: 0.005,
    initScale: 0.01,
    inputCovIsIdentity: true,
    activation: 'relu',
  },
  'relu_deep': {
    dims: [6, 6, 6, 6],
    learningRate: 0.005,
    initScale: 0.01,
    inputCovIsIdentity: true,
    activation: 'relu',
  },
  'tanh': {
    dims: [6, 6, 6, 6],
    learningRate: 0.005,
    initScale: 0.01,
    inputCovIsIdentity: true,
    activation: 'tanh',
  },
  'sigmoid': {
    dims: [6, 6, 6, 6],
    learningRate: 0.005,
    initScale: 0.01,
    inputCovIsIdentity: true,
    activation: 'sigmoid',
  },
};

// Apply preset settings + reset, but do NOT start — used on page load
// and when user switches presets before pressing start.
function updateEquation(key) {
  const el = document.getElementById('simple-model-eq');
  if (!el) return;
  const eqs = {
    depth1:  '$y = W_1 x$',
    depth2:  '$y = W_2 W_1 x = W_{\\text{total}} x$',
    plateau: '$y = W_3 W_2 W_1 x = W_{\\text{total}} x$',
    relu_shallow:    '$y = W_2\\,\\text{ReLU}(W_1 x)$',
    relu_deep:    '$y = W_3\\,\\text{ReLU}(W_2\\,\\text{ReLU}(W_1 x))$',
    tanh:    '$y = W_3\\,\\tanh(W_2\\,\\tanh(W_1 x))$',
    sigmoid: '$y = W_3\\,\\sigma(W_2\\,\\sigma(W_1 x))$',
  };
  el.textContent = eqs[key] || '$y = W_L \\cdots W_1 x$';
  if (typeof renderMathInElement !== 'undefined') {
    renderMathInElement(el, KATEX_OPTS);
  }
}

function applyPresetConfig(key) {
  const p = PRESETS[key];
  if (!p) return;

  dims               = p.dims.slice();
  learningRate       = p.learningRate;
  initScale          = p.initScale;
  inputCovIsIdentity = p.inputCovIsIdentity;
  activation         = p.activation || null;

  updateEquation(key);
  syncControlsToState();

  document.querySelectorAll('.preset-card').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.preset === key);
  });

  resetSim();
}

// Apply preset settings + reset (clicking a preset button calls this)
function applyPreset(key) {
  applyPresetConfig(key);
}

// ── Sync DOM controls to match current global state ──────────────────────────

function syncControlsToState() {
  const depth = getDepth();

  // Depth input
  const depthEl = document.getElementById('depth-input');
  if (depthEl) depthEl.value = depth;

  // Input / output dims
  const inEl  = document.getElementById('input-dim');
  const outEl = document.getElementById('output-dim');
  if (inEl)  inEl.value  = dims[0];
  if (outEl) outEl.value = dims[dims.length - 1];

  // Hidden dim (use dims[1] when depth > 1)
  if (depth > 1) {
    hiddenDimVal = dims[1];
    const hdDisplay = document.getElementById('hidden-dim-display');
    const hdSlider  = document.getElementById('hidden-dim-slider');
    if (hdDisplay) hdDisplay.textContent = String(hiddenDimVal);
    if (hdSlider) {
      const idx = HIDDEN_DIM_VALUES.reduce((best, val, i) =>
        Math.abs(val - hiddenDimVal) < Math.abs(HIDDEN_DIM_VALUES[best] - hiddenDimVal) ? i : best, 0);
      hdSlider.value = (idx / (HIDDEN_DIM_VALUES.length - 1)) * 100;
    }
  }

  // LR slider
  const lrSlider  = document.getElementById('lr-slider');
  const lrDisplay = document.getElementById('lr-display');
  if (lrSlider) {
    lrSlider.value = valueToSliderPos(learningRate, LR_VALUES);
    if (lrDisplay) lrDisplay.innerHTML = fmtScientific(learningRate);
  }

  // Init scale slider
  const scaleSlider  = document.getElementById('scale-slider');
  const scaleDisplay = document.getElementById('scale-display');
  if (scaleSlider) {
    scaleSlider.value = valueToSliderPos(initScale, SCALE_VALUES);
    if (scaleDisplay) scaleDisplay.innerHTML = fmtScientific(initScale);
  }

  // Isotropic toggle
  const isoEl = document.getElementById('isotropic-toggle');
  if (isoEl) isoEl.checked = inputCovIsIdentity;
}

// ── Read the form and rebuild `dims` ─────────────────────────────────────────

function applyNetworkConfig() {
  const d0    = Math.max(1, parseInt(document.getElementById('input-dim').value)  || 6);
  const dL    = Math.max(1, parseInt(document.getElementById('output-dim').value) || 6);
  const depth = Math.max(1, Math.min(10, parseInt(document.getElementById('depth-input').value) || 2));

  // All hidden layers share the same dimension from the slider
  dims = [d0, ...Array(depth - 1).fill(hiddenDimVal), dL];

  // Bottom widget always runs a linear network
  activation = null;

  // Clear active preset (manual config overrides preset)
  document.querySelectorAll('.preset-card').forEach(btn => btn.classList.remove('active'));

  resetSim();
}

// ── Metrics display ───────────────────────────────────────────────────────────

function fmtLoss(v) {
  if (v === null || isNaN(v)) return '—';
  if (v === 0) return '0';
  const exp  = Math.floor(Math.log10(Math.abs(v)));
  const mant = Math.round((v / Math.pow(10, exp)) * 1000) / 1000;
  if (exp === 0) return mant.toString();
  if (Math.abs(mant - 1) < 1e-9) return `10<sup>${exp}</sup>`;
  return `${mant}×10<sup>${exp}</sup>`;
}

function updateMetricsDisplay(loss) {
  const lossEl = document.getElementById('loss-display');
  const iterEl = document.getElementById('iter-display');
  const rateEl = document.getElementById('steprate-display');
  if (lossEl) lossEl.innerHTML   = fmtLoss(loss);
  if (iterEl) iterEl.textContent = loss !== null ? iterationCount : '—';

  if (rateEl) {
    if (stepTimestamps.length >= 2) {
      const span        = stepTimestamps[stepTimestamps.length - 1] - stepTimestamps[0];
      const stepsPerSec = ((stepTimestamps.length - 1) / (span / 1000)) * STEPS_PER_FRAME;
      rateEl.textContent = Math.round(stepsPerSec);
    } else {
      rateEl.textContent = '—';
    }
  }
}

// ── Wire up all UI handlers ───────────────────────────────────────────────────

function setupUIHandlers() {
  // Preset cards drive the SIMPLE widget's independent simulation
  document.querySelectorAll('.preset-card').forEach(btn => {
    btn.addEventListener('click', () => {
      if (btn.classList.contains('active')) {
        s_isRunning ? s_pauseSim() : s_startSim();
      } else {
        s_applyPreset(btn.dataset.preset);
        s_startSim();
      }
    });
  });

  // Start / Pause
  document.getElementById('start-btn').addEventListener('click', () => {
    isRunning ? pauseSim() : startSim();
  });

  // Reset
  document.getElementById('reset-btn').addEventListener('click', resetSim);

  // Depth input — apply on commit (Enter or blur)
  document.getElementById('depth-input').addEventListener('change', applyNetworkConfig);

  // More options toggle
  const moreBtn   = document.getElementById('more-options-btn');
  const morePanel = document.getElementById('more-options-panel');
  if (moreBtn && morePanel) {
    moreBtn.addEventListener('click', () => {
      const open = morePanel.style.display !== 'none';
      morePanel.style.display = open ? 'none' : '';
      moreBtn.textContent     = open ? '+ more hyperparameters' : '− fewer hyperparameters';
    });
  }

  // Hidden dim slider — sync display on drag, apply on release
  const hdSlider  = document.getElementById('hidden-dim-slider');
  const hdDisplay = document.getElementById('hidden-dim-display');
  function syncHD() {
    hiddenDimVal = sliderIndexToValue(parseFloat(hdSlider.value), HIDDEN_DIM_VALUES);
    hdDisplay.textContent = String(hiddenDimVal);
  }
  hdSlider.addEventListener('input', syncHD);
  hdSlider.addEventListener('change', applyNetworkConfig);
  syncHD();

  // Input/output dim — apply on commit
  document.getElementById('input-dim').addEventListener('change', applyNetworkConfig);
  document.getElementById('output-dim').addEventListener('change', applyNetworkConfig);

  // Learning-rate slider
  const lrSlider  = document.getElementById('lr-slider');
  const lrDisplay = document.getElementById('lr-display');
  function syncLR() {
    learningRate = sliderIndexToValue(parseFloat(lrSlider.value), LR_VALUES);
    lrDisplay.innerHTML = fmtScientific(learningRate);
  }
  lrSlider.addEventListener('input', syncLR);
  syncLR();

  // Init-scale slider
  const scaleSlider  = document.getElementById('scale-slider');
  const scaleDisplay = document.getElementById('scale-display');
  function syncScale() {
    initScale = sliderIndexToValue(parseFloat(scaleSlider.value), SCALE_VALUES);
    scaleDisplay.innerHTML = fmtScientific(initScale);
  }
  scaleSlider.addEventListener('input', syncScale);
  syncScale();

  // ── SV curve-group toggle pills ─────────────────────────────────────────

  document.getElementById('show-deep-toggle').addEventListener('change', e => {
    showDeep = e.target.checked;
    if (selectedMatrixIdx !== null) updateSVChart();
  });

  document.getElementById('show-theory-toggle').addEventListener('change', e => {
    showTheory = e.target.checked;
    if (selectedMatrixIdx !== null) updateSVChart();
  });


  // Σ_x = I toggle
  document.getElementById('isotropic-toggle').addEventListener('change', e => {
    inputCovIsIdentity = e.target.checked;
    resetSim();
  });
}
