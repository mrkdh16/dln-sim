// ============================================================================
// WIDGET: STRONG ALIGNMENT (Pre-computed / Scrubbable)
// ============================================================================

function createAlignmentWidget(containerId, opts) {
  opts = opts || {};

  // ── Regime definitions ────────────────────────────────────────────────────
  const REGIMES = [
    { key: 'lazy', label: 'lazy',   sub: 'σ₀ = 2.0, η = 0.01',   initScale: 2.0,   lr: 0.01,  color: '#d1242f' },
    { key: 'mid',  label: 'middle', sub: 'σ₀ = 0.3, η = 0.01',   initScale: 0.3,   lr: 0.01,  color: '#8250df' },
    { key: 'rich', label: 'rich',   sub: 'σ₀ = 0.005, η = 0.001', initScale: 0.005, lr: 0.001, color: '#0969da' },
  ];

  const MAT_LABELS = ['U₂ᵀU', 'V₁ᵀV', 'V₂ᵀU₁'];

  // ── Simulation constants ──────────────────────────────────────────────────
  const DEPTH           = 2;
  const DIMS            = 6;
  const MAX_ITERS       = 25000;
  const RECORD_EVERY    = 100;   // snapshot interval (iterations)
  const CHUNK_SIZE      = 100;   // Gradient steps per UI yield

  // ── State ─────────────────────────────────────────────────────────────────
  let regStates = REGIMES.map(r => ({
    cfg:         r,
    weightVars:  [],
    targetMatrix: null,
    lossHistory: [],
    iterCount:   0,
  }));

  let snapshots    = [];
  let currentSnap  = 0;
  let isComputing  = false;
  let simDone      = false;

  let lossChart    = null;
  let uid          = dlnUID();
  let root, sliderEl, iterLabel, statusEl;

  // ── DOM ───────────────────────────────────────────────────────────────────
  function buildDOM() {
    root = document.getElementById(containerId);
    if (!root) { console.error('AlignmentWidget: container not found:', containerId); return; }
    root.classList.add('dln-widget');

    const colHeaders = REGIMES.map(r => `
      <div class="dln-align-col-hdr">
        <div class="dln-align-col-name" style="color:${r.color}">${r.label}</div>
        <div class="dln-align-col-sub">${r.sub}</div>
      </div>`).join('');

    const hmRows = MAT_LABELS.map((label, row) => `
      <div class="dln-align-hm-row">
        <div class="dln-align-row-label">${label}</div>
        ${REGIMES.map((r, col) => `<div id="${uid}-hm-${row}-${col}" class="dln-align-hm-cell"></div>`).join('')}
      </div>`).join('');

    const legendSpans = REGIMES.map(r =>
      `<span style="display:inline-flex;align-items:center;gap:4px;font-size:11px;color:${r.color}">` +
      `<span style="display:inline-block;width:16px;height:2px;background:${r.color}"></span>${r.label}</span>`
    ).join('');

    root.innerHTML = `
      <div class="dln-widget-controls" style="display: flex; justify-content: space-between; align-items: center;">
        <button class="dln-btn dln-reset-btn">&#8635; recompute</button>
        <span class="dln-status-text" style="font-size: 12px; color: #666;"></span>
      </div>

      <div class="dln-align-top">
        <div class="dln-align-loss-block">
          <div class="dln-ln-chart-title" style="display:flex;gap:10px;align-items:center">
            <span>loss</span>${legendSpans}
          </div>
          <div class="dln-chart-wrap" style="height:160px">
            <canvas id="${uid}-loss"></canvas>
          </div>
        </div>
      </div>

      <div class="dln-align-slider-row" style="margin-top: 15px; display: flex; align-items: center; gap: 10px;">
        <span class="dln-ctrl-label">time</span>
        <input type="range" class="dln-slider" style="flex:1" min="0" max="0" value="0" disabled>
        <span class="dln-align-iter-label" style="min-width: 60px; text-align: right;">step 0</span>
      </div>

      <div class="dln-align-hm-section">
        <div class="dln-align-col-hdrs">
          <div></div>${colHeaders}
        </div>
        ${hmRows}
      </div>`;

    sliderEl  = root.querySelector('input[type=range]');
    statusEl  = root.querySelector('.dln-status-text');
    iterLabel = root.querySelector('.dln-align-iter-label');

    sliderEl.addEventListener('input', () => {
      currentSnap = parseInt(sliderEl.value);
      renderSnapshot(currentSnap);
    });

    root.querySelector('.dln-reset-btn').addEventListener('click', () => {
      if (!isComputing) _resetAndCompute();
    });
  }

  // ── Loss chart ─────────────────────────────────────────────────────────────
  function initChart() {
    const el = document.getElementById(`${uid}-loss`);
    if (!el) return;
    lossChart = new Chart(el.getContext('2d'), {
      type: 'line',
      data: {
        datasets: [
          ...REGIMES.map(r => ({
            label: r.label, data: [],
            borderColor: r.color, backgroundColor: 'transparent',
            borderWidth: 1.5, pointRadius: 0, tension: 0, parsing: false,
          })),
          {
            label: '_indicator', data: [],
            borderColor: 'rgba(0,0,0,0.35)', backgroundColor: 'transparent',
            borderWidth: 1, pointRadius: 0, tension: 0,
            borderDash: [4, 3], parsing: false,
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
          x: { type: 'linear', ticks: { color: '#888', maxTicksLimit: 5 }, grid: { color: '#eaeef2' } },
          y: { type: 'logarithmic', ticks: { color: '#888', callback: v => v >= 0.01 ? v.toFixed(2) : v.toExponential(0) }, grid: { color: '#eaeef2' } },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  function updateLossChart() {
    if (!lossChart) return;
    REGIMES.forEach((r, ri) => {
      const pts = dlnDownsample(regStates[ri].lossHistory, 400);
      lossChart.data.datasets[ri].data = pts.map(p => ({ x: p.iter, y: Math.max(p.loss, 1e-8) }));
    });
    lossChart.update('none');
  }

  function updateIndicator(iter) {
    if (!lossChart) return;
    let yMin = Infinity, yMax = -Infinity;
    REGIMES.forEach((r, ri) => {
      const h = regStates[ri].lossHistory;
      if (h.length > 0) {
        yMin = Math.min(yMin, Math.min(...h.map(p => p.loss)));
        yMax = Math.max(yMax, h[0].loss);
      }
    });
    lossChart.data.datasets[3].data = [
      { x: iter, y: Math.max(yMin * 0.5, 1e-9) },
      { x: iter, y: yMax * 2 },
    ];
    lossChart.update('none');
  }

  // ── Plotly heatmaps ────────────────────────────────────────────────────────
  const HM_COLORSCALE = 'Blues';
  const HM_LAYOUT = {
    margin: { t: 0, b: 0, l: 0, r: 0 },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: { showticklabels: false, showgrid: false, zeroline: false, fixedrange: true },
    yaxis: { showticklabels: false, showgrid: false, zeroline: false, fixedrange: true, scaleanchor: 'x' },
  };
  const HM_CONFIG = { responsive: true, displayModeBar: false, staticPlot: true };

  function emptyZ() { return Array.from({ length: DIMS }, () => Array(DIMS).fill(0)); }

  function initHeatmaps() {
    if (typeof Plotly === 'undefined') return;
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        const divId = `${uid}-hm-${row}-${col}`;
        const el = document.getElementById(divId);
        if (!el) continue;
        Plotly.newPlot(divId, [{ type: 'heatmap', z: emptyZ(), zmin: 0, zmax: 1, colorscale: HM_COLORSCALE, showscale: false, xgap: 1, ygap: 1 }], HM_LAYOUT, HM_CONFIG);
      }
    }
  }

  function renderSnapshot(snapIdx) {
    if (snapshots.length === 0) return;
    const snap = snapshots[Math.min(snapIdx, snapshots.length - 1)];
    if (!snap) return;

    if (typeof Plotly !== 'undefined') {
      for (let row = 0; row < 3; row++) {
        for (let col = 0; col < 3; col++) {
          const divId = `${uid}-hm-${row}-${col}`;
          const mat = snap.mats[col] && snap.mats[col][row];
          if (!mat) continue;
          Plotly.react(divId, [{ type: 'heatmap', z: mat.slice().reverse(), zmin: 0, zmax: 1, colorscale: HM_COLORSCALE, showscale: false, xgap: 1, ygap: 1 }], HM_LAYOUT, HM_CONFIG);
        }
      }
    }

    updateIndicator(snap.iter);
    if (iterLabel) iterLabel.textContent = `step ${snap.iter.toLocaleString()}`;
  }

  // ── SVD utilities (numeric.js) ─────────────────────────────────────────────
  function sortedSVD(mat2d) {
    const { U, S, V } = numeric.svd(mat2d);
    const idx = S.map((s, i) => i).sort((a, b) => S[b] - S[a]);
    return {
      U: U.map(row => idx.map(i => row[i])),
      S: idx.map(i => S[i]),
      V: V.map(row => idx.map(i => row[i])),
    };
  }

  function computeAlignMats(wv) {
    const W1 = wv[0].arraySync();
    const W2 = wv[1].arraySync();
    let svd1, svd2;
    try {
      svd1 = sortedSVD(W1);
      svd2 = sortedSVD(W2);
    } catch (_) {
      const eye = Array.from({ length: DIMS }, (_, i) => Array.from({ length: DIMS }, (_, j) => (i === j ? 1 : 0)));
      return [eye, eye, eye];
    }
    const { U: U1, V: V1 } = svd1;
    const { U: U2, V: V2 } = svd2;

    const A = numeric.transpose(U2).map(r => r.map(Math.abs));
    const B = numeric.transpose(V1).map(r => r.map(Math.abs));
    const C = numeric.dot(numeric.transpose(V2), U1).map(r => r.map(Math.abs));
    return [A, B, C];
  }

  // ── Simulation ─────────────────────────────────────────────────────────────
  function initSim() {
    regStates.forEach(st => {
      st.weightVars.forEach(v => { try { v.dispose(); } catch (_) {} });
      if (st.targetMatrix) { try { st.targetMatrix.dispose(); } catch (_) {} }
      st.weightVars   = [];
      st.targetMatrix = null;
      st.lossHistory  = [];
      st.iterCount    = 0;
    });
    snapshots    = [];
    currentSnap  = 0;
    simDone      = false;

    regStates.forEach(st => {
      const σ = st.cfg.initScale;
      for (let i = 0; i < DEPTH; i++) {
        const init = tf.randomNormal([DIMS, DIMS], 0, σ);
        st.weightVars.push(tf.variable(init));
        init.dispose();
      }
      const dv = tf.tensor1d(Array.from({ length: DIMS }, (_, i) => (DIMS - i) / (DIMS + 1)));
      st.targetMatrix = tf.diag(dv);
      dv.dispose();
    });
  }

  function stepRegime(st) {
    tf.tidy(() => {
      const result = tf.variableGrads(() => {
        let e2e = st.weightVars[0];
        for (let i = 1; i < st.weightVars.length; i++) e2e = tf.matMul(st.weightVars[i], e2e);
        return tf.sum(tf.square(tf.sub(e2e, st.targetMatrix)));
      });
      st.weightVars.forEach(v => {
        const g = result.grads[v.name];
        if (g) v.assign(tf.sub(v, tf.mul(g, st.cfg.lr)));
      });
    });
  }

  function getLoss(st) {
    return tf.tidy(() => {
      let e2e = st.weightVars[0];
      for (let i = 1; i < st.weightVars.length; i++) e2e = tf.matMul(st.weightVars[i], e2e);
      return tf.sum(tf.square(tf.sub(e2e, st.targetMatrix))).dataSync()[0];
    });
  }

  function recordSnapshot() {
    const iter  = regStates[0].iterCount;
    const loss  = regStates.map(st => {
      const h = st.lossHistory;
      return h.length > 0 ? h[h.length - 1].loss : NaN;
    });
    const mats  = regStates.map(st => st.weightVars.length >= 2 ? computeAlignMats(st.weightVars) : [null, null, null]);
    snapshots.push({ iter, loss, mats });
  }

  // ── Async Compute Loop ─────────────────────────────────────────────────────
  async function _computeAll() {
    if (isComputing) return;
    if (regStates[0].weightVars.length === 0) initSim();
    
    isComputing = true;
    statusEl.textContent = 'Computing SVDs & Trajectory...';
    sliderEl.disabled = true;

    // Record initial step 0
    recordSnapshot();
    
    while (!simDone) {
      const prevIter = regStates[0].iterCount;

      for (let s = 0; s < CHUNK_SIZE; s++) {
        regStates.forEach(st => { stepRegime(st); st.iterCount++; });
      }

      regStates.forEach(st => {
        const loss = getLoss(st);
        st.lossHistory.push({ iter: st.iterCount, loss });
      });

      const newIter = regStates[0].iterCount;
      if (Math.floor(prevIter / RECORD_EVERY) < Math.floor(newIter / RECORD_EVERY)) {
        recordSnapshot();
      }

      const pct = Math.round((newIter / MAX_ITERS) * 100);
      statusEl.textContent = `Computing... ${pct}%`;

      if (newIter >= MAX_ITERS) {
        simDone = true;
        // ensure final step is recorded if it didn't align cleanly
        if (newIter % RECORD_EVERY !== 0) recordSnapshot();
      }

      await tf.nextFrame();
    }

    isComputing = false;
    statusEl.textContent = 'Done. Drag the slider to explore.';
    
    // Setup slider bounds
    sliderEl.max = snapshots.length - 1;
    sliderEl.value = snapshots.length - 1;
    sliderEl.disabled = false;
    
    // Draw fully populated loss chart and latest matrices
    updateLossChart();
    currentSnap = snapshots.length - 1;
    renderSnapshot(currentSnap);
  }

  function _resetAndCompute() {
    initSim();
    
    if (lossChart) {
      REGIMES.forEach((r, ri) => { lossChart.data.datasets[ri].data = []; });
      lossChart.data.datasets[3].data = [];
      lossChart.update('none');
    }

    const blank = emptyZ();
    if (typeof Plotly !== 'undefined') {
      for (let row = 0; row < 3; row++) {
        for (let col = 0; col < 3; col++) {
          Plotly.react(`${uid}-hm-${row}-${col}`, [{ type: 'heatmap', z: blank, zmin: 0, zmax: 1, colorscale: HM_COLORSCALE, showscale: false, xgap: 1, ygap: 1 }], HM_LAYOUT, HM_CONFIG);
        }
      }
    }
    
    _computeAll();
  }

  // ── Bootstrap ──────────────────────────────────────────────────────────────
  buildDOM();
  initChart();
  initHeatmaps();
  if (opts.autoStart !== false) _resetAndCompute();

  return { reset: _resetAndCompute };
}