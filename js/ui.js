// ============================================================================
// UI — controls panel, sliders, metrics display
// ============================================================================

// ── Logarithmic slider utilities ──────────────────────────────────────────────

const LR_VALUES         = generateLogValues(-4, -1).concat([1]);   // 10⁻⁴ … 1
const SCALE_VALUES      = generateLogValues(-5,  0).concat([10]);  // 10⁻⁵ … 10
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

// Format a small number in scientific notation using HTML <sup>
function fmtScientific(v) {
  if (v >= 1) return v.toString();
  const exp  = Math.floor(Math.log10(v));
  const mant = Math.round((v / Math.pow(10, exp)) * 1e9) / 1e9;
  if (Math.abs(mant - 1) < 1e-9) return `10<sup>${exp}</sup>`;
  return `${mant}×10<sup>${exp}</sup>`;
}

// ── Hidden dim value (kept in sync with slider) ───────────────────────────────

let hiddenDimVal = 8; // slider value=32 → index 7 → 8

// ── Read the form and rebuild `dims` ─────────────────────────────────────────

function applyNetworkConfig() {
  const d0    = Math.max(1, parseInt(document.getElementById('input-dim').value)  || 8);
  const dL    = Math.max(1, parseInt(document.getElementById('output-dim').value) || 8);
  const depth = Math.max(1, Math.min(10, parseInt(document.getElementById('depth-input').value) || 2));

  // All hidden layers share the same dimension from the slider
  dims = [d0, ...Array(depth - 1).fill(hiddenDimVal), dL];

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
  document.getElementById('loss-display').innerHTML   = fmtLoss(loss);
  document.getElementById('iter-display').textContent = loss !== null ? iterationCount : '—';

  if (stepTimestamps.length >= 2) {
    const span        = stepTimestamps[stepTimestamps.length - 1] - stepTimestamps[0];
    const stepsPerSec = ((stepTimestamps.length - 1) / (span / 1000)) * STEPS_PER_FRAME;
    document.getElementById('steprate-display').textContent = Math.round(stepsPerSec);
  } else {
    document.getElementById('steprate-display').textContent = '—';
  }
}

// ── Wire up all UI handlers ───────────────────────────────────────────────────

function setupUIHandlers() {
  // Start / Pause
  document.getElementById('start-btn').addEventListener('click', () => {
    isRunning ? pauseSim() : startSim();
  });

  // Reset
  document.getElementById('reset-btn').addEventListener('click', resetSim);

  // Apply config
  document.getElementById('apply-btn').addEventListener('click', applyNetworkConfig);

  // Hidden dim slider
  const hdSlider  = document.getElementById('hidden-dim-slider');
  const hdDisplay = document.getElementById('hidden-dim-display');
  function syncHD() {
    hiddenDimVal = sliderIndexToValue(parseFloat(hdSlider.value), HIDDEN_DIM_VALUES);
    hdDisplay.textContent = String(hiddenDimVal);
  }
  hdSlider.addEventListener('input', syncHD);
  syncHD(); // set initial display

  // Learning-rate slider
  const lrSlider  = document.getElementById('lr-slider');
  const lrDisplay = document.getElementById('lr-display');
  function syncLR() {
    learningRate = sliderIndexToValue(parseFloat(lrSlider.value), LR_VALUES);
    lrDisplay.innerHTML = fmtScientific(learningRate);
  }
  lrSlider.addEventListener('input', syncLR);
  syncLR(); // set initial display

  // Init-scale slider
  const scaleSlider  = document.getElementById('scale-slider');
  const scaleDisplay = document.getElementById('scale-display');
  function syncScale() {
    initScale = sliderIndexToValue(parseFloat(scaleSlider.value), SCALE_VALUES);
    scaleDisplay.innerHTML = fmtScientific(initScale);
  }
  scaleSlider.addEventListener('input', syncScale);
  syncScale(); // set initial display

  // ── SV curve-group toggle pills ─────────────────────────────────────────

  // Deep network curves
  document.getElementById('show-deep-toggle').addEventListener('change', e => {
    showDeep = e.target.checked;
    if (selectedMatrixIdx !== null) updateSVChart();
  });

  // Theory (Saxe et al.) overlay — replaces the old inline theory-toggle
  document.getElementById('show-theory-toggle').addEventListener('change', e => {
    showTheory = e.target.checked;
    if (selectedMatrixIdx !== null) updateSVChart();
  });

  // Shallow (depth-1 comparison) curves
  document.getElementById('show-shallow-toggle').addEventListener('change', e => {
    showShallow = e.target.checked;
    if (selectedMatrixIdx !== null) updateSVChart();
  });

  // Σ_x = I toggle — changing input covariance redefines the problem, so reset.
  document.getElementById('isotropic-toggle').addEventListener('change', e => {
    inputCovIsIdentity = e.target.checked;
    resetSim();
  });
}
