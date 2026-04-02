// ============================================================================
// WIDGET: SV THEORY
// Shows empirical e2e singular value curves alongside Saxe et al. theory
// curves for a linear DLN.  Depth is selectable; lr and init scale are
// chosen automatically to keep convergence visible and theory clean.
//
// Usage:
//   <div id="my-widget"></div>
//   <script> createSVTheoryWidget('my-widget'); </script>
//
// Requires (loaded first): widget-utils.js, TF.js, numeric.js, Chart.js
// ============================================================================

function createSVTheoryWidget(containerId, opts) {
  opts = opts || {};

  // ── Constants ──────────────────────────────────────────────────────────────
  const STEPS_PER_FRAME = 20;
  const SV_RECORD_EVERY = 40;
  const DIMS            = 6;
  const MAX_ITERS       = 80000;

  const DEPTH_CFG = {
    1: { lr: 0.01,  initScale: 0.01 },
    2: { lr: 0.005, initScale: 0.01 },
    3: { lr: 0.005, initScale: 0.01 },
    4: { lr: 0.005, initScale: 0.01 },
  };

  // ── State ──────────────────────────────────────────────────────────────────
  let depth      = opts.depth || 3;
  let lr         = null;
  let initScale  = null;
  let dims       = null;

  let weightVars   = [];
  let targetMatrix = null;
  let targetSVs    = [];
  let initE2ESVs   = [];
  let svHistory    = [];
  let iterCount    = 0;
  let isRunning    = false;
  let animFrameId  = null;
  let completed    = false;
  let stopAt       = null;
  let stopAtWall   = null;
  let odeCache     = new Map();

  let chart = null;
  const uid = dlnUID();
  let root  = null;

  // ── DOM ────────────────────────────────────────────────────────────────────

  function buildDOM() {
    root = document.getElementById(containerId);
    if (!root) { console.error('SVTheoryWidget: container not found:', containerId); return; }
    root.classList.add('dln-widget');

    const depthBtns = [1, 2, 3, 4].map(d =>
      `<button class="dln-btn dln-depth-btn${d === depth ? ' active' : ''}" data-depth="${d}">${d}</button>`
    ).join('');

    root.innerHTML = `
      <div class="dln-widget-controls">
        <div class="dln-ctrl-group">
          <span class="dln-ctrl-label">Depth</span>
          <div class="dln-btn-row">${depthBtns}</div>
        </div>
        <div class="dln-ctrl-group" style="margin-left:auto;gap:6px;">
          <button class="dln-btn dln-play-btn">&#9654; Start</button>
          <button class="dln-btn dln-reset-btn">Reset</button>
        </div>
      </div>
      <div class="dln-chart-wrap">
        <canvas id="${uid}-sv"></canvas>
      </div>
      <div class="dln-legend">
        <span class="dln-legend-item"><span class="dln-legend-solid"></span>empirical</span>
        <span class="dln-legend-item"><span class="dln-legend-dashed"></span>theory</span>
      </div>`;

    root.querySelectorAll('.dln-depth-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        depth = parseInt(btn.dataset.depth);
        root.querySelectorAll('.dln-depth-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        _reset();
      });
    });

    root.querySelector('.dln-play-btn').addEventListener('click', () => {
      if (completed) { _reset(); _start(); return; }
      isRunning ? _pause() : _start();
    });

    root.querySelector('.dln-reset-btn').addEventListener('click', _reset);
  }

  // ── Chart ──────────────────────────────────────────────────────────────────

  function _initChart() {
    if (chart) { chart.destroy(); chart = null; }
    const ctx = document.getElementById(`${uid}-sv`);
    if (!ctx) return;
    chart = new Chart(ctx.getContext('2d'), {
      type: 'line',
      data: { labels: [], datasets: [] },
      options: dlnChartOptions('iteration', 'singular value'),
    });
  }

  function _updateChart() {
    if (!chart || svHistory.length === 0) return;
    const pts     = dlnDownsample(svHistory, 300);
    const iters   = pts.map(p => p.iter);
    const numSVs  = Math.min(pts[pts.length - 1].svs.length, DIMS);
    const hasTheory = targetSVs.length > 0 && initE2ESVs.length > 0;
    const need    = numSVs * (hasTheory ? 2 : 1);

    if (chart.data.datasets.length !== need) {
      chart.data.datasets = [];
      // Empirical: solid lines
      for (let i = 0; i < numSVs; i++) {
        chart.data.datasets.push({
          data: [], borderColor: DLN_SV_COLORS[i % DLN_SV_COLORS.length],
          backgroundColor: 'transparent', borderWidth: 2,
          pointRadius: 0, tension: 0, borderDash: [],
        });
      }
      // Theory: dashed lines, same color, thinner
      if (hasTheory) {
        for (let i = 0; i < numSVs; i++) {
          chart.data.datasets.push({
            data: [], borderColor: DLN_SV_COLORS[i % DLN_SV_COLORS.length],
            backgroundColor: 'transparent', borderWidth: 1.2,
            pointRadius: 0, tension: 0, borderDash: [5, 4],
          });
        }
      }
    }

    chart.data.labels = iters;
    for (let i = 0; i < numSVs; i++) {
      chart.data.datasets[i].data = pts.map(p => p.svs[i] ?? 0);
    }
    if (hasTheory) {
      for (let i = 0; i < numSVs; i++) {
        chart.data.datasets[numSVs + i].data =
          dlnSaxeTheory(targetSVs[i], initE2ESVs[i], depth, lr, iters, odeCache);
      }
    }
    chart.update('none');
  }

  function _clearChart() {
    if (!chart) return;
    chart.data.labels   = [];
    chart.data.datasets = [];
    chart.update('none');
  }

  // ── Simulation ─────────────────────────────────────────────────────────────

  function _initSim() {
    weightVars.forEach(v => { try { v.dispose(); } catch (_) {} });
    if (targetMatrix) { try { targetMatrix.dispose(); } catch (_) {} }
    weightVars   = [];
    targetMatrix = null;
    targetSVs    = [];
    initE2ESVs   = [];
    svHistory    = [];
    iterCount    = 0;
    stopAt       = null;
    stopAtWall   = null;
    completed    = false;
    odeCache     = new Map();

    const cfg = DEPTH_CFG[depth] || DEPTH_CFG[3];
    lr        = cfg.lr;
    initScale = cfg.initScale;
    dims      = [DIMS, ...Array(depth - 1).fill(DIMS), DIMS];

    for (let i = 0; i < depth; i++) {
      const init = tf.randomNormal([DIMS, DIMS], 0, initScale);
      weightVars.push(tf.variable(init));
      init.dispose();
    }

    const diagVals = tf.tensor1d(
      Array.from({ length: DIMS }, (_, i) => (DIMS - i) / (DIMS + 1))
    );
    targetMatrix = tf.diag(diagVals);
    diagVals.dispose();

    dlnAlignedInit(weightVars, dims, targetMatrix, initScale);

    targetSVs  = dlnComputeSVs(targetMatrix);
    initE2ESVs = tf.tidy(() => {
      let e2e = weightVars[0];
      for (let i = 1; i < weightVars.length; i++) e2e = tf.matMul(weightVars[i], e2e);
      return dlnComputeSVs(e2e);
    });
  }

  function _e2e() {
    let p = weightVars[0];
    for (let i = 1; i < weightVars.length; i++) p = tf.matMul(weightVars[i], p);
    return p;
  }

  function _step() {
    tf.tidy(() => {
      const result = tf.variableGrads(() => tf.sum(tf.square(tf.sub(_e2e(), targetMatrix))));
      weightVars.forEach(v => {
        const g = result.grads[v.name];
        if (g) v.assign(tf.sub(v, tf.mul(g, lr)));
      });
    });
  }

  function _recordSVs() {
    const svs = tf.tidy(() => dlnComputeSVs(_e2e()));
    svHistory.push({ iter: iterCount, svs });
  }

  function _checkStop() {
    if (iterCount > MAX_ITERS) { _finish(); return true; }
    if (svHistory.length === 0) return false;
    const last = svHistory[svHistory.length - 1].svs;
    const smallestTarget = targetSVs[targetSVs.length - 1] ?? 0;
    if (stopAt === null && last[last.length - 1] > smallestTarget * 0.95) {
      stopAt     = iterCount * 3;
      stopAtWall = performance.now();
    }
    if (stopAt !== null && iterCount >= stopAt &&
        (performance.now() - stopAtWall) >= 8000) {
      _finish(); return true;
    }
    return false;
  }

  function _finish() {
    _updateChart();
    completed = true;
    _pause();
    _syncPlayBtn();
  }

  function _loop() {
    if (!isRunning) return;
    for (let i = 0; i < STEPS_PER_FRAME; i++) { _step(); iterCount++; }
    if (iterCount % SV_RECORD_EVERY === 0) _recordSVs();
    _updateChart();
    if (_checkStop()) return;
    animFrameId = requestAnimationFrame(_loop);
  }

  // ── Controls ───────────────────────────────────────────────────────────────

  function _syncPlayBtn() {
    const btn = root && root.querySelector('.dln-play-btn');
    if (!btn) return;
    btn.innerHTML = completed ? '&#8635; Replay'
                  : isRunning ? '&#9646;&#9646; Pause'
                  : '&#9654; Start';
  }

  function _start() {
    if (isRunning) return;
    WidgetManager.requestStart(uid);
    if (weightVars.length === 0) _initSim();
    isRunning = true;
    _syncPlayBtn();
    _loop();
  }

  function _pause() {
    isRunning = false;
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    WidgetManager.notifyStop(uid);
    _syncPlayBtn();
  }

  function _reset() {
    _pause();
    _initSim();
    _clearChart();
    _syncPlayBtn();
  }

  // ── Bootstrap ──────────────────────────────────────────────────────────────

  buildDOM();
  _initChart();
  _initSim();
  WidgetManager.register(uid, _pause);
  if (opts.autoStart !== false) {
    _start();
  } else {
    dlnScrollAutoplay(root, _start);
  }

  return { start: _start, pause: _pause, reset: _reset };
}
