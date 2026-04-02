// ============================================================================
// WIDGET: LINEAR vs NONLINEAR  (data-driven version)
//
// Loads pre-computed data from a JSON file (produced by precompute_ln.py).
// Supports animated playback with play/pause and a time scrubber slider.
//
// Expected JSON format:
//   {
//     "depth1":       { "loss": [{iter, loss}, ...], "svs": [{iter, svs:[...]}, ...] },
//     "depth2":       { ... },
//     "depth3":       { ... },
//     "relu_shallow": { ... },
//     "relu_deep":    { ... },
//     "tanh":         { ... }
//   }
//
// Usage:
//   <div id="my-widget"></div>
//   <script>
//     createLinearNonlinearWidget('my-widget', { dataUrl: 'linear_nonlinear_data.json' });
//   </script>
//
// Requires (loaded first): widget-utils.js, Chart.js
// KaTeX is used if available globally.
// ============================================================================

function createLinearNonlinearWidget(containerId, opts) {
  opts = opts || {};

  // ── Preset metadata ───────────────────────────────────────────────────────

  const PRESETS = {
    depth1: {
      group: 'linear', name: 'Linear Regression', depthLabel: 'd = 1',
      eq: '$\\hat{f}(x) = W_1 x$',
    },
    depth2: {
      group: 'linear', name: 'Minimally Deep', depthLabel: 'd = 2',
      eq: '$\\hat{f}(x) = W_2 W_1 x$',
    },
    depth3: {
      group: 'linear', name: 'Deep Linear', depthLabel: 'd = 3',
      eq: '$\\hat{f}(x) = W_3 W_2 W_1 x$',
    },
    relu_shallow: {
      group: 'nonlinear', name: 'Shallow ReLU', depthLabel: 'd = 2',
      eq: '$\\hat{f}(x) = W_2\\,\\text{ReLU}(W_1 x)$',
    },
    relu_deep: {
      group: 'nonlinear', name: 'Deep ReLU', depthLabel: 'd = 3',
      eq: '$\\hat{f}(x) = W_3\\,\\text{ReLU}(W_2\\,\\text{ReLU}(W_1 x))$',
    },
    tanh: {
      group: 'nonlinear', name: 'Deep tanh', depthLabel: 'd = 3',
      eq: '$\\hat{f}(x) = W_3\\,\\tanh(W_2\\,\\tanh(W_1 x))$',
    },
  };

  const LINEAR_KEYS    = ['depth1', 'depth2', 'depth3'];
  const NONLINEAR_KEYS = ['relu_shallow', 'relu_deep', 'tanh'];

  // Animation plays through the data over this many milliseconds
  const PLAY_DURATION_MS = 7000;

  // ── State ─────────────────────────────────────────────────────────────────

  let activeKey      = opts.defaultPreset || 'depth3';
  let data           = null;   // full JSON payload after load
  let playIdx        = 0;
  let maxIdx         = 0;
  let isPlaying      = false;
  let animFrameId    = null;
  let playStartTime  = null;
  let playStartIdx   = 0;

  let lossChart = null;
  let svChart   = null;
  const uid     = dlnUID();
  let root      = null;
  let playBtn   = null;
  let slider    = null;

  // ── DOM ───────────────────────────────────────────────────────────────────

  function _itemHTML(key) {
    const p = PRESETS[key];
    return `
      <button class="dln-preset-item${key === activeKey ? ' active' : ''}" data-preset="${key}">
        <span class="dln-preset-item-name">${p.name}</span>
        <span class="dln-preset-item-depth">${p.depthLabel}</span>
      </button>`;
  }

  function buildDOM() {
    root = document.getElementById(containerId);
    if (!root) { console.error('LinearNonlinearWidget: container not found:', containerId); return; }
    root.classList.add('dln-widget', 'dln-ln-widget');

    root.innerHTML = `
      <div class="dln-ln-columns">

        <div class="dln-preset-col">
          <div class="dln-preset-col-label">Linear</div>
          ${LINEAR_KEYS.map(_itemHTML).join('')}
        </div>

        <div class="dln-ln-center">
          <div id="${uid}-status" class="dln-status-text" style="min-height:1.2em;margin-bottom:4px;"></div>
          <div class="dln-ln-charts">
            <div class="dln-ln-chart-section">
              <div class="dln-chart-title">loss</div>
              <div class="dln-chart-wrap"><canvas id="${uid}-loss"></canvas></div>
            </div>
            <div class="dln-ln-chart-section">
              <div class="dln-chart-title">singular values of $W_1$</div>
              <div class="dln-chart-wrap"><canvas id="${uid}-sv"></canvas></div>
            </div>
          </div>
          <div class="dln-ln-playbar">
            <button class="dln-btn dln-play-btn" id="${uid}-play" disabled>&#9654; Play</button>
            <input type="range" class="dln-slider" id="${uid}-slider"
                   min="0" max="100" value="100" step="1" disabled>
          </div>
          <div id="${uid}-eq" class="dln-ln-eq"></div>
        </div>

        <div class="dln-preset-col">
          <div class="dln-preset-col-label">Nonlinear</div>
          ${NONLINEAR_KEYS.map(_itemHTML).join('')}
        </div>

      </div>`;

    playBtn = root.querySelector(`#${uid}-play`);
    slider  = root.querySelector(`#${uid}-slider`);

    root.querySelectorAll('.dln-preset-item').forEach(btn => {
      btn.addEventListener('click', () => _selectPreset(btn.dataset.preset));
    });

    playBtn.addEventListener('click', () => {
      if (isPlaying) {
        _stopPlay();
      } else {
        if (playIdx >= maxIdx) playIdx = 0; // replay from start when done
        _startPlay();
      }
    });

    slider.addEventListener('input', () => {
      if (isPlaying) _stopPlay();
      playIdx = parseInt(slider.value, 10);
      _renderAtIdx();
      _syncPlayBtn();
    });

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
      try {
        el.innerHTML = katex.renderToString(eq.replace(/^\$|\$$/g, ''), { throwOnError: false });
      } catch (_) {}
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
          data: [], borderColor: '#0969da',
          backgroundColor: 'rgba(9,105,218,0.05)',
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

  // Render charts up to (and including) playIdx in the current preset's data
  function _renderAtIdx() {
    if (!data || !data[activeKey]) return;
    const preset = data[activeKey];
    const end    = Math.min(playIdx, preset.loss.length - 1);

    // Loss chart
    if (lossChart) {
      const slice = preset.loss.slice(0, end + 1);
      lossChart.data.labels           = slice.map(p => p.iter);
      lossChart.data.datasets[0].data = slice.map(p => p.loss);
      lossChart.update('none');
    }

    // SV chart
    if (svChart) {
      const svSlice = preset.svs.slice(0, end + 1);
      if (svSlice.length > 0) {
        const numSVs = svSlice[svSlice.length - 1].svs.length;
        if (svChart.data.datasets.length !== numSVs) {
          svChart.data.datasets = Array.from({ length: numSVs }, (_, i) => ({
            data: [], borderColor: DLN_SV_COLORS[i % DLN_SV_COLORS.length],
            backgroundColor: 'transparent', borderWidth: 1.8,
            pointRadius: 0, tension: 0,
          }));
        }
        svChart.data.labels = svSlice.map(p => p.iter);
        for (let i = 0; i < numSVs; i++) {
          svChart.data.datasets[i].data = svSlice.map(p => p.svs[i] ?? 0);
        }
      } else {
        svChart.data.labels   = [];
        svChart.data.datasets = [];
      }
      svChart.update('none');
    }

    // Sync slider thumb position
    if (slider) slider.value = end;
  }

  // ── Preset selection ──────────────────────────────────────────────────────

  function _selectPreset(key) {
    _stopPlay();
    activeKey = key;

    root.querySelectorAll('.dln-preset-item').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.preset === key);
    });
    _renderEq(key);

    if (data && data[key]) {
      maxIdx   = data[key].loss.length - 1;
      playIdx  = maxIdx;  // show full data immediately
      if (slider) { slider.max = maxIdx; slider.value = maxIdx; }
      _renderAtIdx();
    }
  }

  // ── Playback ──────────────────────────────────────────────────────────────

  function _startPlay() {
    if (isPlaying || !data || !data[activeKey]) return;
    WidgetManager.requestStart(uid);
    isPlaying     = true;
    playStartIdx  = playIdx;
    playStartTime = null;
    animFrameId   = requestAnimationFrame(_playLoop);
    _syncPlayBtn();
  }

  function _playLoop(timestamp) {
    if (!isPlaying) return;
    if (playStartTime === null) playStartTime = timestamp;

    const fraction = Math.min((timestamp - playStartTime) / PLAY_DURATION_MS, 1);
    playIdx = Math.round(playStartIdx + fraction * (maxIdx - playStartIdx));
    _renderAtIdx();

    if (fraction >= 1) {
      _stopPlay();
      return;
    }
    animFrameId = requestAnimationFrame(_playLoop);
  }

  function _stopPlay() {
    if (!isPlaying && animFrameId === null) return;
    isPlaying = false;
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    WidgetManager.notifyStop(uid);
    _syncPlayBtn();
  }

  function _syncPlayBtn() {
    if (!playBtn) return;
    if (isPlaying) {
      playBtn.innerHTML = '&#9646;&#9646; Pause';
    } else if (playIdx >= maxIdx && maxIdx > 0) {
      playBtn.innerHTML = '&#8635; Replay';
    } else {
      playBtn.innerHTML = '&#9654; Play';
    }
  }

  // ── Data loading ──────────────────────────────────────────────────────────

  function _setStatus(msg) {
    const el = document.getElementById(`${uid}-status`);
    if (el) el.textContent = msg;
  }

  async function loadData() {
    const dataUrl = opts.dataUrl || 'linear_nonlinear_data.json';
    _setStatus('Loading data\u2026');
    try {
      const res = await fetch(dataUrl);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      data = await res.json();
      _setStatus('');

      // Enable controls
      if (playBtn) playBtn.disabled = false;
      if (slider)  slider.disabled  = false;

      // Show the default preset
      _selectPreset(activeKey);

    } catch (err) {
      console.error('LinearNonlinearWidget: failed to load data:', err);
      _setStatus('Error loading data.');
    }
  }

  // ── Scroll autoplay ───────────────────────────────────────────────────────

  function _autoplayStart() {
    if (!data) return; // data not yet loaded; loadData will check pendingAutoplay
    playIdx = 0;
    _renderAtIdx();
    _startPlay();
  }

  // ── Bootstrap ─────────────────────────────────────────────────────────────

  buildDOM();
  _initCharts();
  WidgetManager.register(uid, _stopPlay);
  loadData().then(() => {
    dlnScrollAutoplay(root, _autoplayStart);
  });
}
