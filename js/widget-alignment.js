// ============================================================================
// WIDGET: STRONG ALIGNMENT (Data-Driven / Logarithmic / RdBu)
// ============================================================================

function createAlignmentWidget(containerId, dataUrl) {
  const REGIMES = [
    { key: 'unbalanced', label: 'unbalanced',   sub: 'σ₀ = 2.0, η = 0.01',   color: '#d1242f' },
    { key: 'mid',  label: 'middle', sub: 'σ₀ = 0.08, η = 0.01',   color: '#8250df' },
    { key: 'balanced', label: 'balanced',   sub: 'σ₀ = 0.005, η = 0.001', color: '#0969da' },
  ];

  const MAT_LABELS = ['U₂ᵀU', 'V₁ᵀV', 'V₂ᵀU₁'];
  const DIMS = 6;
  const SLIDER_RES     = 1000;  // slider resolution
  const PLAY_DURATION_MS = 10000; // ms to animate from current pos to end

  let snapshots = [];
  let lossChart = null;
  let uid = dlnUID();
  let root, sliderEl, iterLabel, statusEl, playBtn;

  // ── Playback state ────────────────────────────────────────────────────────
  let isPlaying     = false;
  let animFrameId   = null;
  let playStartTime = null;
  let playStartVal  = 0;

  // ── DOM ───────────────────────────────────────────────────────────────────

  function buildDOM() {
    root = document.getElementById(containerId);
    if (!root) return;
    root.classList.add('dln-widget');

    const colHeaders = REGIMES.map(r => `
      <div class="dln-align-col-hdr">
        <div class="dln-align-col-name" style="color:${r.color}">${r.label}</div>
        <div class="dln-align-col-sub" style="font-size: 10px;">${r.sub}</div>
      </div>`).join('');

    const hmRows = MAT_LABELS.map((label, row) => `
      <div class="dln-align-hm-row" style="align-items: center;">
        <div class="dln-align-row-label" style="width: 40px; text-align: right; margin-right: 10px;">${label}</div>
        ${REGIMES.map((r, col) => `<div id="${uid}-hm-${row}-${col}" class="dln-align-hm-cell" style="width: 100px; height: 100px;"></div>`).join('')}
      </div>`).join('');

    const legendSpans = REGIMES.map(r =>
      `<span style="display:inline-flex;align-items:center;gap:4px;font-size:11px;color:${r.color}">` +
      `<span style="display:inline-block;width:16px;height:2px;background:${r.color}"></span>${r.label}</span>`
    ).join('');

    root.innerHTML = `
      <div class="dln-widget-controls" style="display:flex;justify-content:space-between;align-items:center;">
        <span class="dln-status-text" style="font-size:12px;color:#8c959f;">Loading data…</span>
        <button class="dln-btn dln-play-btn" id="${uid}-play" disabled>&#9654; Play</button>
      </div>

      <div class="dln-align-top">
        <div class="dln-align-loss-block">
          <div class="dln-ln-chart-title" style="display:flex;gap:10px;align-items:center">
            <span>loss (log-log)</span>${legendSpans}
          </div>
          <div class="dln-chart-wrap" style="height:160px">
            <canvas id="${uid}-loss"></canvas>
          </div>
        </div>
      </div>

      <div class="dln-align-slider-row" style="margin-top:15px;display:flex;align-items:center;gap:10px;">
        <span class="dln-ctrl-label">time</span>
        <input type="range" class="dln-slider" style="flex:1" min="0" max="${SLIDER_RES}" value="0" disabled>
        <span class="dln-align-iter-label" style="min-width:60px;text-align:right;font-variant-numeric:tabular-nums;">step 0</span>
      </div>

      <div class="dln-align-hm-section" style="max-width:400px;margin:15px auto 0 auto;">
        <div class="dln-align-col-hdrs">
          <div style="width:40px;margin-right:10px;"></div>${colHeaders}
        </div>
        ${hmRows}
      </div>

      <div class="dln-heatmap-legend" style="display:flex;align-items:center;justify-content:center;margin-top:15px;font-size:11px;color:#8c959f;">
        <span>-1</span>
        <div style="width:120px;height:8px;margin:0 10px;background:linear-gradient(to right,#b2182b,#f7f7f7,#2166ac);border-radius:4px;border:1px solid #ddd;"></div>
        <span>+1</span>
      </div>`;

    playBtn   = root.querySelector(`#${uid}-play`);
    sliderEl  = root.querySelector('input[type=range]');
    statusEl  = root.querySelector('.dln-status-text');
    iterLabel = root.querySelector('.dln-align-iter-label');

    // Slider scrub — pause playback if dragged manually
    sliderEl.addEventListener('input', () => {
      if (isPlaying) _stopPlay();
      _renderFromSlider();
    });

    playBtn.addEventListener('click', () => {
      if (isPlaying) {
        _stopPlay();
      } else {
        // Restart from beginning if already at the end
        if (parseInt(sliderEl.value, 10) >= SLIDER_RES) {
          sliderEl.value = 0;
          _renderFromSlider();
        }
        _startPlay();
      }
    });
  }

  // ── Loss chart ────────────────────────────────────────────────────────────

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
            borderColor: 'rgba(0,0,0,0.3)', backgroundColor: 'transparent',
            borderWidth: 1, pointRadius: 0, tension: 0,
            borderDash: [4, 3], parsing: false,
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
          x: {
            type: 'logarithmic', min: 1,
            ticks: { color: '#8c959f', maxTicksLimit: 6,
                     callback: v => v >= 1000 ? (v / 1000) + 'k' : v },
            grid: { color: 'rgba(0,0,0,0.05)' },
          },
          y: {
            type: 'logarithmic',
            ticks: { color: '#8c959f', callback: v => v >= 0.01 ? v.toFixed(2) : v.toExponential(0) },
            grid: { color: 'rgba(0,0,0,0.05)' },
          },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  function populateLossChart() {
    if (!lossChart || snapshots.length === 0) return;
    const pts = dlnDownsample(snapshots, 400);
    REGIMES.forEach((r, ri) => {
      lossChart.data.datasets[ri].data = pts.map(snap => ({
        x: snap.iter === 0 ? 1 : snap.iter,
        y: Math.max(snap.loss[ri], 1e-8),
      }));
    });
    lossChart.update('none');
  }

  function updateIndicator(iter) {
    if (!lossChart || snapshots.length === 0) return;
    let yMin = Infinity, yMax = -Infinity;
    snapshots.forEach(snap => {
      snap.loss.forEach(l => {
        if (l < yMin) yMin = l;
        if (l > yMax) yMax = l;
      });
    });
    const safeIter = iter === 0 ? 1 : iter;
    lossChart.data.datasets[3].data = [
      { x: safeIter, y: Math.max(yMin * 0.5, 1e-9) },
      { x: safeIter, y: yMax * 2 },
    ];
    lossChart.update('none');
  }

  // ── Heatmaps ──────────────────────────────────────────────────────────────

  const HM_COLORSCALE = 'RdBu';
  const HM_LAYOUT = {
    margin: { t: 0, b: 0, l: 0, r: 0 },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: { showticklabels: false, showgrid: false, zeroline: false, fixedrange: true },
    yaxis: { showticklabels: false, showgrid: false, zeroline: false, fixedrange: true, scaleanchor: 'x' },
  };
  const HM_CONFIG = { responsive: true, displayModeBar: false, staticPlot: true };

  function emptyZ() {
    return Array.from({ length: DIMS }, () => Array(DIMS).fill(0));
  }

  function initHeatmaps() {
    if (typeof Plotly === 'undefined') return;
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        const divId = `${uid}-hm-${row}-${col}`;
        const el = document.getElementById(divId);
        if (!el) continue;
        Plotly.newPlot(divId, [{
          type: 'heatmap', z: emptyZ(),
          zmin: -1, zmax: 1,
          colorscale: HM_COLORSCALE, reversescale: true,
          showscale: false, xgap: 1, ygap: 1,
        }], HM_LAYOUT, HM_CONFIG);
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
          Plotly.react(divId, [{
            type: 'heatmap',
            z: mat.slice().reverse(),
            zmin: -1, zmax: 1,
            colorscale: HM_COLORSCALE,
            reversescale: true,
            showscale: false,
            xgap: 1, ygap: 1,
          }], HM_LAYOUT, HM_CONFIG);
        }
      }
    }

    updateIndicator(snap.iter);
    if (iterLabel) iterLabel.textContent = `step ${snap.iter.toLocaleString()}`;
  }

  // Translate current slider value → closest snapshot and render
  function _renderFromSlider() {
    if (snapshots.length === 0) return;
    const fraction = parseInt(sliderEl.value, 10) / SLIDER_RES;
    const maxIter  = snapshots[snapshots.length - 1].iter;
    const minIter  = snapshots[1] ? snapshots[1].iter : 100;

    let targetIter = 0;
    if (fraction > 0) {
      targetIter = minIter * Math.pow(maxIter / minIter, fraction);
    }

    let bestIdx = 0, minDiff = Infinity;
    for (let i = 0; i < snapshots.length; i++) {
      const diff = Math.abs(snapshots[i].iter - targetIter);
      if (diff < minDiff) { minDiff = diff; bestIdx = i; }
    }
    renderSnapshot(bestIdx);
  }

  // ── Playback ──────────────────────────────────────────────────────────────

  // Convert the current log-scale slider position to the nearest snapshot index.
  function _sliderToSnapIdx() {
    const fraction = parseInt(sliderEl.value, 10) / SLIDER_RES;
    const maxIter  = snapshots[snapshots.length - 1].iter;
    const minIter  = snapshots[1] ? snapshots[1].iter : 100;
    const targetIter = fraction > 0
      ? minIter * Math.pow(maxIter / minIter, fraction)
      : 0;
    let best = 0, minDiff = Infinity;
    for (let i = 0; i < snapshots.length; i++) {
      const d = Math.abs(snapshots[i].iter - targetIter);
      if (d < minDiff) { minDiff = d; best = i; }
    }
    return best;
  }

  // Convert a snapshot index back to the log-scale slider value for display.
  function _snapIdxToSlider(snapIdx) {
    const iter    = snapshots[snapIdx].iter;
    const maxIter = snapshots[snapshots.length - 1].iter;
    const minIter = snapshots[1] ? snapshots[1].iter : 100;
    if (iter <= 0) return 0;
    const fraction = Math.log(iter / minIter) / Math.log(maxIter / minIter);
    return Math.round(Math.max(0, Math.min(1, fraction)) * SLIDER_RES);
  }

  function _syncPlayBtn() {
    if (!playBtn) return;
    const atEnd = parseInt(sliderEl.value, 10) >= SLIDER_RES;
    playBtn.innerHTML = isPlaying ? '&#9646;&#9646; Pause'
                      : atEnd    ? '&#8635; Replay'
                      :            '&#9654; Play';
  }

   function _startPlay() {
    if (isPlaying || snapshots.length === 0) return;
    WidgetManager.requestStart(uid);
    isPlaying        = true;
    
    // FIX: Store the current slider value, NOT the snapshot index
    playStartVal     = parseInt(sliderEl.value, 10); 
    playStartTime    = null;
    animFrameId      = requestAnimationFrame(_playLoop);
    _syncPlayBtn();
  }

  function _playLoop(timestamp) {
    if (!isPlaying) return;
    if (playStartTime === null) playStartTime = timestamp;

    const fraction = Math.min((timestamp - playStartTime) / PLAY_DURATION_MS, 1);
    
    // FIX: Advance linearly through the slider's log-space
    const currentSliderVal = Math.round(playStartVal + fraction * (SLIDER_RES - playStartVal));
    sliderEl.value = currentSliderVal; 

    // Look up the nearest snapshot for this log-scaled position
    const snapIdx = _sliderToSnapIdx();
    renderSnapshot(snapIdx);

    if (fraction >= 1) { _stopPlay(); return; }
    animFrameId = requestAnimationFrame(_playLoop);
  }

  function _stopPlay() {
    isPlaying = false;
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    WidgetManager.notifyStop(uid);
    _syncPlayBtn();
  }

  // ── Data loading ──────────────────────────────────────────────────────────

  async function loadData() {
    try {
      const response = await fetch(dataUrl);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      snapshots = data.snapshots;

      populateLossChart();

      // Show final snapshot by default
      sliderEl.value    = SLIDER_RES;
      sliderEl.disabled = false;
      playBtn.disabled  = false;

      renderSnapshot(snapshots.length - 1);
      if (statusEl) statusEl.textContent = 'Drag the slider or press Play to explore.';
      _syncPlayBtn();

    } catch (error) {
      console.error('Failed to load alignment data:', error);
      if (statusEl) {
        statusEl.textContent = 'Error loading data.';
        statusEl.style.color = '#d1242f';
      }
    }
  }

  // ── Scroll autoplay ───────────────────────────────────────────────────────

  function _autoplayStart() {
    if (snapshots.length === 0) return;
    sliderEl.value = 0;
    _renderFromSlider();
    _startPlay();
  }

  // ── Bootstrap ─────────────────────────────────────────────────────────────

  buildDOM();
  initChart();
  initHeatmaps();
  WidgetManager.register(uid, _stopPlay);

  if (dataUrl) {
    loadData().then(() => {
      dlnScrollAutoplay(root, _autoplayStart);
    });
  }
}
