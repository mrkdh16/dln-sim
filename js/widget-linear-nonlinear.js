// ============================================================================
// WIDGET: LINEAR vs NONLINEAR
// A self-contained version of the simple preset-card widget from index.html.
// Shows loss and first-layer singular values for various network presets.
//
// Usage:
//   <div id="my-widget"></div>
//   <script> createLinearNonlinearWidget('my-widget'); </script>
//
// Requires (loaded first): widget-utils.js, TF.js, numeric.js, Chart.js
// KaTeX is used if available globally (renderMathInElement / katex).
// ============================================================================

function createLinearNonlinearWidget(containerId, opts) {
  opts = opts || {};

  // ── Preset definitions ────────────────────────────────────────────────────

  const NUM_DATA = 64;

  const PRESETS = {
    depth1: {
      column: 'linear', depthLabel: 'depth = 1',
      name: 'Linear Regression', desc: 'exponential loss decay',
      dims: [6, 6], lr: 0.01, initScale: 0.05,
      eq: '$\\hat{f}(x) = W_1 x$',
    },
    depth2: {
      column: 'linear', depthLabel: 'depth = 2',
      name: 'Minimally Deep', desc: 'analytically tractable',
      dims: [6, 6, 6], lr: 0.005, initScale: 0.01,
      eq: '$\\hat{f}(x) = W_2 W_1 x$',
    },
    depth3: {
      column: 'linear', depthLabel: 'depth = 3',
      name: 'Deep Linear', desc: 'stepwise decrease in loss',
      dims: [6, 6, 6, 6], lr: 0.005, initScale: 0.01,
      eq: '$\\hat{f}(x) = W_3 W_2 W_1 x$',
    },
    relu_shallow: {
      column: 'nonlinear', depthLabel: 'depth = 2',
      name: 'Shallow ReLU', desc: 'piecewise linear',
      dims: [6, 6, 6], lr: 0.005, initScale: 0.01,
      activation: 'relu', useTeacher: true,
      eq: '$\\hat{f}(x) = W_2\\,\\text{ReLU}(W_1 x)$',
    },
    relu_deep: {
      column: 'nonlinear', depthLabel: 'depth = 3',
      name: 'Deep ReLU', desc: 'piecewise linear',
      dims: [6, 6, 6, 6], lr: 0.005, initScale: 0.01,
      activation: 'relu', useTeacher: true,
      eq: '$\\hat{f}(x) = W_3\\,\\text{ReLU}(W_2\\,\\text{ReLU}(W_1 x))$',
    },
    tanh: {
      column: 'nonlinear', depthLabel: 'depth = 3',
      name: 'Deep tanh', desc: 'smoothly saturating',
      dims: [6, 6, 6, 6], lr: 0.005, initScale: 0.01,
      activation: 'tanh', useTeacher: true,
      eq: '$\\hat{f}(x) = W_3\\,\\tanh(W_2\\,\\tanh(W_1 x))$',
    },
  };

  const LINEAR_KEYS    = ['depth1', 'depth2', 'depth3'];
  const NONLINEAR_KEYS = ['relu_shallow', 'relu_deep', 'tanh'];

  // ── State ─────────────────────────────────────────────────────────────────

  const STEPS_PER_FRAME = 20;
  const SV_RECORD_EVERY = 40;
  const MAX_ITERS       = 80000;

  let activeKey    = opts.defaultPreset || 'tanh';
  let weightVars   = [];
  let teacherVars  = [];
  let targetMatrix = null;
  let inputData    = null;
  let targetData   = null;
  let lossHistory  = [];
  let svHistory    = [];   // SVs of first weight matrix W_1
  let iterCount    = 0;
  let isRunning    = false;
  let animFrameId  = null;
  let completed    = false;
  let stopAt       = null;
  let stopAtWall   = null;

  // current preset config (copied from PRESETS[activeKey])
  let dims       = null;
  let lr         = null;
  let initScale  = null;
  let activation = null;
  let useTeacher = false;

  let lossChart = null;
  let svChart   = null;
  let uid       = dlnUID();
  let root      = null;

  // ── DOM ───────────────────────────────────────────────────────────────────

  function _cardHTML(key) {
    const p = PRESETS[key];
    return `
      <button class="dln-preset-card${key === activeKey ? ' active' : ''}" data-preset="${key}">
        <div class="dln-preset-header">
          <span class="dln-preset-depth">${p.depthLabel}</span>
          <span class="dln-preset-icon">&#9654;</span>
        </div>
        <div class="dln-preset-name">${p.name}</div>
        <div class="dln-preset-desc">${p.desc}</div>
        <span class="dln-preset-reset" style="display:none">&#8635;</span>
      </button>`;
  }

  function buildDOM() {
    root = document.getElementById(containerId);
    if (!root) { console.error('LinearNonlinearWidget: container not found:', containerId); return; }
    root.classList.add('dln-widget', 'dln-ln-widget');

    root.innerHTML = `
      <div class="dln-ln-columns">
        <div class="dln-preset-col">
          <div class="dln-preset-col-label">linear</div>
          ${LINEAR_KEYS.map(_cardHTML).join('')}
        </div>
        <div class="dln-ln-center">
          <div class="dln-ln-charts">
            <div class="dln-ln-chart-section">
              <div class="dln-ln-chart-title">loss</div>
              <div class="dln-chart-wrap"><canvas id="${uid}-loss"></canvas></div>
            </div>
            <div class="dln-ln-chart-section">
              <div class="dln-ln-chart-title">singular values of $W_1$</div>
              <div class="dln-chart-wrap"><canvas id="${uid}-sv"></canvas></div>
            </div>
          </div>
          <div id="${uid}-eq" class="dln-ln-eq"></div>
        </div>
        <div class="dln-preset-col">
          <div class="dln-preset-col-label">nonlinear</div>
          ${NONLINEAR_KEYS.map(_cardHTML).join('')}
        </div>
      </div>`;

    // Wire preset cards
    root.querySelectorAll('.dln-preset-card').forEach(btn => {
      btn.addEventListener('click', e => {
        // If clicking the reset icon inside, reset+replay
        if (e.target.classList.contains('dln-preset-reset')) {
          e.stopPropagation();
          _applyPreset(btn.dataset.preset);
          _start();
          return;
        }
        const key = btn.dataset.preset;
        if (key === activeKey) {
          if (completed) { _reset(); _start(); }
          else           { isRunning ? _pause() : _start(); }
        } else {
          _applyPreset(key);
          _start();
        }
      });
    });

    // Render math in eq display if KaTeX available
    _renderEq(activeKey);
  }

  function _renderEq(key) {
    const el = document.getElementById(`${uid}-eq`);
    if (!el) return;
    const eq = PRESETS[key]?.eq || '';
    el.textContent = eq;
    if (typeof renderMathInElement !== 'undefined' && typeof KATEX_OPTS !== 'undefined') {
      renderMathInElement(el, KATEX_OPTS);
    } else if (typeof katex !== 'undefined') {
      try { el.innerHTML = katex.renderToString(eq.replace(/^\$|\$$/g, ''), { throwOnError: false }); } catch (_) {}
    }
  }

  // ── Charts ────────────────────────────────────────────────────────────────

  function _initCharts() {
    if (lossChart) { lossChart.destroy(); lossChart = null; }
    if (svChart)   { svChart.destroy();   svChart   = null; }

    const lossEl = document.getElementById(`${uid}-loss`);
    const svEl   = document.getElementById(`${uid}-sv`);
    if (!lossEl || !svEl) return;

    lossChart = new Chart(lossEl.getContext('2d'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          data: [],
          borderColor: '#0969da', backgroundColor: 'rgba(9,105,218,0.06)',
          borderWidth: 1.5, pointRadius: 0, tension: 0, fill: true,
        }],
      },
      options: dlnChartOptions(null, null),
    });

    svChart = new Chart(svEl.getContext('2d'), {
      type: 'line',
      data: { labels: [], datasets: [] },
      options: dlnChartOptions(null, null),
    });
  }

  function _updateCharts() {
    if (lossChart && lossHistory.length > 0) {
      const pts = dlnDownsample(lossHistory, 300);
      lossChart.data.labels           = pts.map(p => p.iter);
      lossChart.data.datasets[0].data = pts.map(p => p.loss);
      lossChart.update('none');
    }

    if (svChart && svHistory.length > 0) {
      const pts    = dlnDownsample(svHistory, 300);
      const numSVs = pts[pts.length - 1].svs.length;
      if (svChart.data.datasets.length !== numSVs) {
        svChart.data.datasets = [];
        for (let i = 0; i < numSVs; i++) {
          svChart.data.datasets.push({
            data: [], borderColor: DLN_SV_COLORS[i % DLN_SV_COLORS.length],
            backgroundColor: 'transparent', borderWidth: 2,
            pointRadius: 0, tension: 0,
          });
        }
      }
      svChart.data.labels = pts.map(p => p.iter);
      for (let i = 0; i < numSVs; i++) {
        svChart.data.datasets[i].data = pts.map(p => p.svs[i] ?? 0);
      }
      svChart.update('none');
    }
  }

  function _clearCharts() {
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

  // ── Simulation ────────────────────────────────────────────────────────────

  function _initSim() {
    weightVars.forEach(v => { try { v.dispose(); } catch (_) {} });
    teacherVars.forEach(v => { try { v.dispose(); } catch (_) {} });
    if (targetMatrix) { try { targetMatrix.dispose(); } catch (_) {} }
    if (inputData)    { try { inputData.dispose();    } catch (_) {} }
    if (targetData)   { try { targetData.dispose();   } catch (_) {} }

    weightVars  = [];
    teacherVars = [];
    targetMatrix = inputData = targetData = null;
    lossHistory  = [];
    svHistory    = [];
    iterCount    = 0;
    stopAt       = null;
    stopAtWall   = null;
    completed    = false;

    const depth = dims.length - 1;
    const inDim = dims[0];

    for (let i = 0; i < depth; i++) {
      const [r, c] = [dims[i + 1], dims[i]];
      const init = tf.randomNormal([r, c], 0, initScale);
      weightVars.push(tf.variable(init));
      init.dispose();
    }

    // Fixed target matrix (diagonal, SVs in (0,1))
    const n = Math.min(dims[dims.length - 1], dims[0]);
    const dv = tf.tensor1d(Array.from({ length: n }, (_, i) => (n - i) / (n + 1)));
    targetMatrix = tf.pad(tf.diag(dv), [[0, dims[dims.length-1]-n], [0, dims[0]-n]]);
    dv.dispose();

    // Fixed dataset
    inputData = tf.randomNormal([inDim, NUM_DATA]);

    if (useTeacher) {
      // Teacher network with same architecture
      const act = _actFn(activation);
      for (let i = 0; i < depth; i++) {
        const [r, c] = [dims[i + 1], dims[i]];
        teacherVars.push(tf.mul(tf.randomNormal([r, c]), 1 / Math.sqrt(c)));
      }
      targetData = tf.tidy(() => {
        let h = tf.matMul(teacherVars[0], inputData);
        for (let i = 1; i < teacherVars.length; i++) {
          h = act(h);
          h = tf.matMul(teacherVars[i], h);
        }
        return h;
      });
    } else {
      targetData = tf.matMul(targetMatrix, inputData);
    }
  }

  function _actFn(key) {
    return key === 'relu'    ? tf.relu
         : key === 'tanh'    ? tf.tanh
         : key === 'sigmoid' ? tf.sigmoid
         : key === 'sin'     ? tf.sin
         : (x => x); // identity / linear
  }

  function _forward(X) {
    if (!activation) {
      let p = tf.matMul(weightVars[0], X);
      for (let i = 1; i < weightVars.length; i++) p = tf.matMul(weightVars[i], p);
      return p;
    }
    const act = _actFn(activation);
    let h = tf.matMul(weightVars[0], X);
    for (let i = 1; i < weightVars.length; i++) { h = act(h); h = tf.matMul(weightVars[i], h); }
    return h;
  }

  function _step() {
    tf.tidy(() => {
      const result = tf.variableGrads(() => {
        const pred = _forward(inputData);
        return tf.div(tf.sum(tf.square(tf.sub(pred, targetData))), NUM_DATA);
      });
      weightVars.forEach(v => {
        const g = result.grads[v.name];
        if (g) v.assign(tf.sub(v, tf.mul(g, lr)));
      });
    });
  }

  function _getLoss() {
    return tf.tidy(() => {
      const pred = _forward(inputData);
      return tf.div(tf.sum(tf.square(tf.sub(pred, targetData))), NUM_DATA).dataSync()[0];
    });
  }

  function _recordSVs() {
    // Record SVs of first weight matrix W_1
    const svs = tf.tidy(() => dlnComputeSVs(weightVars[0]));
    svHistory.push({ iter: iterCount, svs });
  }

  function _checkStop(loss) {
    if (iterCount > MAX_ITERS) { _finish(); return true; }
    if (stopAt === null && loss < 0.05) {
      stopAt     = iterCount * 3;
      stopAtWall = performance.now();
    }
    if (stopAt !== null && iterCount >= stopAt &&
        (performance.now() - stopAtWall) >= 10000) {
      _finish(); return true;
    }
    return false;
  }

  function _finish() {
    _updateCharts();
    completed = true;
    _pause();
    _syncCards();
  }

  function _loop() {
    if (!isRunning) return;
    for (let i = 0; i < STEPS_PER_FRAME; i++) { _step(); iterCount++; }
    const loss = _getLoss();
    lossHistory.push({ iter: iterCount, loss });
    if (iterCount % SV_RECORD_EVERY === 0) _recordSVs();
    _updateCharts();
    if (_checkStop(loss)) return;
    animFrameId = requestAnimationFrame(_loop);
  }

  // ── Preset / Controls ──────────────────────────────────────────────────────

  function _applyPreset(key) {
    const p = PRESETS[key];
    if (!p) return;
    activeKey  = key;
    dims       = p.dims.slice();
    lr         = p.lr;
    initScale  = p.initScale;
    activation = p.activation || null;
    useTeacher = p.useTeacher || false;
    root.querySelectorAll('.dln-preset-card').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.preset === key);
    });
    _renderEq(key);
    _reset();
  }

  function _syncCards() {
    // Update play icons
    root.querySelectorAll('.dln-preset-card').forEach(btn => {
      const icon  = btn.querySelector('.dln-preset-icon');
      const reset = btn.querySelector('.dln-preset-reset');
      if (!icon) return;
      if (btn.dataset.preset === activeKey) {
        icon.innerHTML = completed ? '&#8635;' : isRunning ? '&#9646;&#9646;' : '&#9654;';
        if (reset) reset.style.display = (!completed && !isRunning) ? '' : 'none';
      } else {
        icon.innerHTML = '&#9654;';
        if (reset) reset.style.display = 'none';
      }
    });
  }

  function _start() {
    if (isRunning) return;
    if (weightVars.length === 0) _initSim();
    isRunning = true;
    _syncCards();
    _loop();
  }

  function _pause() {
    isRunning = false;
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    _syncCards();
  }

  function _reset() {
    _pause();
    _initSim();
    _clearCharts();
    _syncCards();
  }

  // ── Bootstrap ─────────────────────────────────────────────────────────────

  buildDOM();
  _initCharts();
  _applyPreset(activeKey);
  if (opts.autoStart !== false) _start();

  return { start: _start, pause: _pause, reset: _reset };
}
