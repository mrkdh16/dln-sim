// ============================================================================
// WIDGET: LOSS
// Shows the training loss curve for a linear DLN to illustrate plateaus and
// sudden drops.  Depth is selectable; a "target scale" slider lets the reader
// adjust the magnitude of the target singular values (larger scale → more
// dramatic initial plateaus).
//
// Usage:
//   <div id="my-widget"></div>
//   <script> createLossWidget('my-widget'); </script>
//
// Requires (loaded first): widget-utils.js, TF.js, numeric.js, Chart.js
// ============================================================================

function createLossWidget(containerId, opts) {
  opts = opts || {};

  // ── Constants ──────────────────────────────────────────────────────────────
  const STEPS_PER_FRAME = 10;
  const DIMS            = 6;
  const LR              = 0.005;
  const INIT_SCALE      = 0.01;
  const MAX_ITERS       = 100000;

  // ── State ──────────────────────────────────────────────────────────────────
  let depth       = opts.depth || 3;
  let targetScale = opts.targetScale || 1.0;

  let weightVars   = [];
  let targetMatrix = null;
  let lossHistory  = [];
  let iterCount    = 0;
  let isRunning    = false;
  let animFrameId  = null;
  let completed    = false;
  let stopAt       = null;
  let stopAtWall   = null;

  let chart = null;
  let uid   = dlnUID();
  let root  = null;

  // ── DOM ────────────────────────────────────────────────────────────────────

  function buildDOM() {
    root = document.getElementById(containerId);
    if (!root) { console.error('LossWidget: container not found:', containerId); return; }
    root.classList.add('dln-widget');

    const depthBtns = [1, 2, 3].map(d =>
      `<button class="dln-btn dln-depth-btn${d === depth ? ' active' : ''}" data-depth="${d}">${d}</button>`
    ).join('');

    // Target scale slider: linear 0.1→1.0 displayed value
    const initSliderVal = Math.round((targetScale - 0.1) / 0.9 * 100);

    root.innerHTML = `
      <div class="dln-widget-controls">
        <div class="dln-ctrl-group">
          <span class="dln-ctrl-label">depth</span>
          <div class="dln-btn-row">${depthBtns}</div>
        </div>
        <div class="dln-ctrl-group">
          <span class="dln-ctrl-label">target scale</span>
          <input class="dln-slider dln-scale-slider" type="range"
                 min="0" max="100" value="${initSliderVal}">
          <span class="dln-scale-display">${targetScale.toFixed(1)}</span>
        </div>
        <div class="dln-ctrl-group dln-ctrl-right">
          <button class="dln-btn dln-play-btn">&#9654; start</button>
          <button class="dln-btn dln-reset-btn">&#8635; reset</button>
        </div>
      </div>
      <div class="dln-chart-wrap">
        <canvas id="${uid}-loss"></canvas>
      </div>`;

    // Depth buttons
    root.querySelectorAll('.dln-depth-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        depth = parseInt(btn.dataset.depth);
        root.querySelectorAll('.dln-depth-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        _reset();
        if (opts.autoStart !== false) _start();
      });
    });

    // Scale slider
    const slider  = root.querySelector('.dln-scale-slider');
    const display = root.querySelector('.dln-scale-display');
    slider.addEventListener('input', () => {
      targetScale = parseFloat((0.1 + slider.value / 100 * 0.9).toFixed(2));
      display.textContent = targetScale.toFixed(1);
    });
    slider.addEventListener('change', () => {
      _reset();
      if (opts.autoStart !== false) _start();
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
    const el = document.getElementById(`${uid}-loss`);
    if (!el) return;
    chart = new Chart(el.getContext('2d'), {
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
      options: dlnChartOptions('iteration', 'loss'),
    });
  }

  function _updateChart() {
    if (!chart || lossHistory.length === 0) return;
    const pts = dlnDownsample(lossHistory, 400);
    chart.data.labels           = pts.map(p => p.iter);
    chart.data.datasets[0].data = pts.map(p => p.loss);
    chart.update('none');
  }

  function _clearChart() {
    if (!chart) return;
    chart.data.labels           = [];
    chart.data.datasets[0].data = [];
    chart.update('none');
  }

  // ── Simulation ─────────────────────────────────────────────────────────────

  function _initSim() {
    weightVars.forEach(v => { try { v.dispose(); } catch (_) {} });
    if (targetMatrix) { try { targetMatrix.dispose(); } catch (_) {} }
    weightVars   = [];
    targetMatrix = null;
    lossHistory  = [];
    iterCount    = 0;
    stopAt       = null;
    stopAtWall   = null;
    completed    = false;

    const dims = [DIMS, ...Array(depth - 1).fill(DIMS), DIMS];

    for (let i = 0; i < depth; i++) {
      const init = tf.randomNormal([DIMS, DIMS], 0, INIT_SCALE);
      weightVars.push(tf.variable(init));
      init.dispose();
    }

    // Target: equally spaced SVs, all scaled by targetScale
    const diagVals = tf.tensor1d(
      Array.from({ length: DIMS }, (_, i) => targetScale * (DIMS - i) / (DIMS + 1))
    );
    targetMatrix = tf.diag(diagVals);
    diagVals.dispose();

    // Aligned init so theory would apply (makes plateaus clean)
    dlnAlignedInit(weightVars, dims, targetMatrix, INIT_SCALE);
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
        if (g) v.assign(tf.sub(v, tf.mul(g, LR)));
      });
    });
  }

  function _getLoss() {
    return tf.tidy(() => tf.sum(tf.square(tf.sub(_e2e(), targetMatrix))).dataSync()[0]);
  }

  function _checkStop(loss) {
    if (iterCount > MAX_ITERS) { _finish(); return true; }
    // Initial loss ≈ sum(s_i^2) for small init; threshold at 0.5% of initial range
    const threshold = targetScale * targetScale * 0.005;
    if (stopAt === null && loss < threshold) {
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
    const loss = _getLoss();
    lossHistory.push({ iter: iterCount, loss });
    _updateChart();
    if (_checkStop(loss)) return;
    animFrameId = requestAnimationFrame(_loop);
  }

  // ── Controls ───────────────────────────────────────────────────────────────

  function _syncPlayBtn() {
    const btn = root && root.querySelector('.dln-play-btn');
    if (!btn) return;
    btn.innerHTML = completed ? '&#8635; replay'
                  : isRunning ? '&#9646;&#9646; pause'
                  : '&#9654; start';
  }

  function _start() {
    if (isRunning) return;
    if (weightVars.length === 0) _initSim();
    isRunning = true;
    _syncPlayBtn();
    _loop();
  }

  function _pause() {
    isRunning = false;
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
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
  if (opts.autoStart !== false) _start();

  return { start: _start, pause: _pause, reset: _reset };
}
