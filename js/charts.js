// ============================================================================
// CHART.JS CHARTS  (loss + singular value history)
// ============================================================================

let lossChart = null;
let svChart   = null;

// Simple-widget charts (always show e2e SVs, no controls)
let simpleLossChart = null;
let simpleSVChart   = null;

// ── Color helpers ─────────────────────────────────────────────────────────────

// Convert a #rrggbb hex color to rgba(r,g,b,a) string
function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Common chart options ──────────────────────────────────────────────────────

function baseChartOptions(xLabel, yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        title: { display: true, text: xLabel, color: '#57606a', font: { size: 10 } },
        ticks: { color: '#57606a', maxTicksLimit: 6, maxRotation: 0 },
        grid:  { color: '#eaeef2' },
        border: { color: '#d0d7de' },
      },
      y: {
        title: { display: true, text: yLabel, color: '#57606a', font: { size: 10 } },
        ticks: { color: '#57606a' },
        grid:  { color: '#eaeef2' },
        border: { color: '#d0d7de' },
        beginAtZero: true,
      },
    },
    plugins: {
      legend: { display: false },
    },
  };
}

// Minimal chart options (no axis labels) for the simple top widget
function simpleChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        ticks: { color: '#888', maxTicksLimit: 5, maxRotation: 0, font: { size: 9 } },
        grid:  { color: '#eaeef2' },
        border: { color: '#d0d7de' },
      },
      y: {
        ticks: { color: '#888', font: { size: 9 } },
        grid:  { color: '#eaeef2' },
        border: { color: '#d0d7de' },
        beginAtZero: true,
      },
    },
    plugins: { legend: { display: false } },
  };
}

// ── Init ─────────────────────────────────────────────────────────────────────

function initCharts() {
  const lossCtx = document.getElementById('loss-chart').getContext('2d');
  lossChart = new Chart(lossCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Loss',
        data: [],
        borderColor: '#0969da',
        backgroundColor: 'rgba(9,105,218,0.06)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0,
        fill: true,
      }],
    },
    options: baseChartOptions('iteration', 'loss'),
  });

  const svCtx = document.getElementById('sv-chart').getContext('2d');
  svChart = new Chart(svCtx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      ...baseChartOptions('iteration', 'singular value'),
      plugins: { legend: { display: false } },
    },
  });
}

function initSimpleCharts() {
  const lossEl = document.getElementById('simple-loss-chart');
  const svEl   = document.getElementById('simple-sv-chart');
  if (!lossEl || !svEl) return;

  simpleLossChart = new Chart(lossEl.getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        data: [],
        borderColor: '#0969da',
        backgroundColor: 'rgba(9,105,218,0.06)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0,
        fill: true,
      }],
    },
    options: simpleChartOptions(),
  });

  simpleSVChart = new Chart(svEl.getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: simpleChartOptions(),
  });
}

// ── Update: loss ──────────────────────────────────────────────────────────────

function updateLossChart() {
  if (!lossChart || lossHistory.length === 0) return;

  const pts = downsample(lossHistory, 300);
  lossChart.data.labels           = pts.map(p => p.iter);
  lossChart.data.datasets[0].data = pts.map(p => p.loss);
  lossChart.update('none');
}

// ── Update: singular values ───────────────────────────────────────────────────

// SV_COLORS is defined in drawing.js (loaded before this file)

function updateSVChart() {
  if (!svChart || selectedMatrixIdx === null) return;

  const history = svHistories[selectedMatrixIdx];
  if (!history || history.length === 0) return;

  const depth  = getDepth();
  const isE2E  = (selectedMatrixIdx === depth);
  const pts    = downsample(history, 300);
  const iters  = pts.map(p => p.iter);
  const numSVs = pts[pts.length - 1].svs.length;

  // Saxe theory overlay: only valid for linear networks (e2e chart, theory arrays populated)
  const needTheory = isE2E && targetSVs.length > 0 && initE2ESVs.length > 0;

  // Total datasets: deep + theory
  const totalNeeded = numSVs + (needTheory ? numSVs : 0);

  // Rebuild dataset list whenever the count changes
  if (svChart.data.datasets.length !== totalNeeded) {
    svChart.data.datasets = [];

    // ── Group 1: Deep network — solid, 2px ──────────────────────────────────
    for (let i = 0; i < numSVs; i++) {
      const col   = SV_COLORS[i % SV_COLORS.length];
      const label = depth > 1 ? `σ${sub(i + 1)} (L=${depth})` : `σ${sub(i + 1)}`;
      svChart.data.datasets.push({
        label,
        data: [],
        borderColor: col,
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        borderDash: [],
        _group: 'deep',
      });
    }

    // ── Group 2: Saxe theory — dotted, 1.2px, same hue as deep ─────────────
    if (needTheory) {
      for (let i = 0; i < numSVs; i++) {
        const col = SV_COLORS[i % SV_COLORS.length];
        svChart.data.datasets.push({
          label: `σ${sub(i + 1)} (theory)`,
          data: [],
          borderColor: col,
          backgroundColor: 'transparent',
          borderWidth: 1.2,
          pointRadius: 0,
          tension: 0,
          borderDash: [2, 2],
          _group: 'theory',
        });
      }
    }
  }

  // ── Fill data ────────────────────────────────────────────────────────────

  svChart.data.labels = iters;

  for (let i = 0; i < numSVs; i++) {
    svChart.data.datasets[i].data = pts.map(p => p.svs[i] ?? 0);
  }

  if (needTheory) {
    for (let i = 0; i < numSVs; i++) {
      svChart.data.datasets[numSVs + i].data = saxeTheoretical(i, iters);
    }
  }

  // ── Apply toggle-pill visibility ──────────────────────────────────────────

  for (const ds of svChart.data.datasets) {
    if (ds._group === 'deep')   ds.hidden = !showDeep;
    if (ds._group === 'theory') ds.hidden = !showTheory;
  }

  svChart.update('none');
}

// ── Update: simple top-widget charts (always e2e, deep + theory only) ────────

function updateSimpleCharts() {
  // Loss chart — reads from simple widget's own history
  if (simpleLossChart && s_lossHistory.length > 0) {
    const pts = downsample(s_lossHistory, 300);
    simpleLossChart.data.labels           = pts.map(p => p.iter);
    simpleLossChart.data.datasets[0].data = pts.map(p => p.loss);
    simpleLossChart.update('none');
  }

  // SV chart — always show e2e of the simple simulation
  if (!simpleSVChart) return;
  const depth   = s_getDepth();
  const history = s_svHistories[depth];
  if (!history || history.length === 0) return;

  const pts    = downsample(history, 300);
  const iters  = pts.map(p => p.iter);
  const numSVs = pts[pts.length - 1].svs.length;

  if (simpleSVChart.data.datasets.length !== numSVs) {
    simpleSVChart.data.datasets = [];
    for (let i = 0; i < numSVs; i++) {
      const col = SV_COLORS[i % SV_COLORS.length];
      simpleSVChart.data.datasets.push({
        data: [], borderColor: col, backgroundColor: 'transparent',
        borderWidth: 2, pointRadius: 0, tension: 0,
      });
    }
  }

  simpleSVChart.data.labels = iters;
  for (let i = 0; i < numSVs; i++) {
    simpleSVChart.data.datasets[i].data = pts.map(p => p.svs[i] ?? 0);
  }

  simpleSVChart.update('none');
}

// ── Clear ─────────────────────────────────────────────────────────────────────

// Clears only the bottom widget charts (called by main resetSim)
function clearCharts() {
  if (lossChart) {
    lossChart.data.labels           = [];
    lossChart.data.datasets[0].data = [];
    lossChart.update('none');
  }
  if (svChart) {
    svChart.data.labels   = [];
    svChart.data.datasets = [];
    svChart.update('none');
  }
}

// Clears only the simple widget charts (called by s_resetSim)
function clearSimpleCharts() {
  if (simpleLossChart) {
    simpleLossChart.data.labels           = [];
    simpleLossChart.data.datasets[0].data = [];
    simpleLossChart.update('none');
  }
  if (simpleSVChart) {
    simpleSVChart.data.labels   = [];
    simpleSVChart.data.datasets = [];
    simpleSVChart.update('none');
  }
}

// ── Downsampling ──────────────────────────────────────────────────────────────

function downsample(arr, maxPts) {
  if (arr.length <= maxPts) return arr;
  const rawStride = arr.length / maxPts;
  const stride    = Math.pow(2, Math.ceil(Math.log2(rawStride)));
  const out = [];
  for (let i = 0; i < arr.length; i++) {
    if (i % stride === 0 || i === arr.length - 1) out.push(arr[i]);
  }
  return out;
}
