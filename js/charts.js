// ============================================================================
// CHART.JS CHARTS  (loss + singular value history)
// ============================================================================

let lossChart = null;
let svChart   = null;

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
      // Legend is replaced by the toggle pills in the UI
      plugins: { legend: { display: false } },
    },
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

  // Depth-1 comparison: shown on the e2e chart when main depth > 1
  const show1L  = isE2E && depth > 1 && svHistory1L.length > 0;
  const pts1L   = show1L ? downsample(svHistory1L, 300) : [];
  const numSVs1 = show1L && pts1L.length > 0 ? pts1L[pts1L.length - 1].svs.length : 0;

  // Saxe theory overlay: only valid for depth=2
  const needTheory = isE2E && depth === 2
                  && targetSVs.length > 0 && initE2ESVs.length > 0;

  // Total datasets: deep + shallow(1L) + theory
  const totalNeeded = numSVs + numSVs1 + (needTheory ? numSVs : 0);

  // Rebuild dataset list whenever the count changes
  if (svChart.data.datasets.length !== totalNeeded) {
    svChart.data.datasets = [];

    // ── Group 1: Deep network — solid, 2px, full-saturation colors ──────────
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

    // ── Group 2: Shallow (depth-1) — dashed, 1.5px, desaturated (55% opacity) ─
    for (let i = 0; i < numSVs1; i++) {
      const col = SV_COLORS[i % SV_COLORS.length];
      svChart.data.datasets.push({
        label: `σ${sub(i + 1)} (L=1)`,
        data: [],
        borderColor: hexToRgba(col, 0.55),
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0,
        borderDash: [5, 3],
        _group: 'shallow',
      });
    }

    // ── Group 3: Saxe theory — dotted, 1.2px, same hue as deep ─────────────
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

  // Deep network
  for (let i = 0; i < numSVs; i++) {
    svChart.data.datasets[i].data = pts.map(p => p.svs[i] ?? 0);
  }

  // Shallow (depth-1) comparison
  for (let i = 0; i < numSVs1; i++) {
    svChart.data.datasets[numSVs + i].data = pts1L.map(p => p.svs[i] ?? 0);
  }

  // Saxe theory
  if (needTheory) {
    for (let i = 0; i < numSVs; i++) {
      svChart.data.datasets[numSVs + numSVs1 + i].data = saxeTheoretical(i, iters);
    }
  }

  // ── Apply toggle-pill visibility ──────────────────────────────────────────

  for (const ds of svChart.data.datasets) {
    if (ds._group === 'deep')    ds.hidden = !showDeep;
    if (ds._group === 'shallow') ds.hidden = !showShallow;
    if (ds._group === 'theory')  ds.hidden = !showTheory;
  }

  svChart.update('none');
}

// ── Clear ─────────────────────────────────────────────────────────────────────

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
