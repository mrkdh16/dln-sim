// ============================================================================
// DETAIL TABS — tabbed panel below the simple widget
// ============================================================================

const _STAB_KEYS   = ['hyperparams', 'iomap', 'target'];
const _STAB_LABELS = {
  hyperparams: 'Hyperparameters',
  iomap:       'Input-Output Map',
  target:      'Input Data & Target Function',
};

// ── Content generators ────────────────────────────────────────────────────────

function _fmtNum(v) {
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v))   return String(v);
  if (v < 0.001) return v.toExponential(0);
  return parseFloat(v.toPrecision(3)).toString();
}

function _kv(key, val) {
  return `<div class="stab-kv-pair"><span class="stab-kv-key">${key}</span><span class="stab-kv-val">${val}</span></div>`;
}

function _hyperparamsContent() {
  const dims  = typeof s_dims         !== 'undefined' ? s_dims         : [];
  const lr    = typeof s_learningRate !== 'undefined' ? s_learningRate : '—';
  const scale = typeof s_initScale    !== 'undefined' ? s_initScale    : '—';
  const depth = dims.length - 1;
  const inDim = dims[0]  || '—';
  const outDim = dims[dims.length - 1] || '—';

  const items = [
    _kv('Optimizer',        'Gradient descent (full-batch)'),
    _kv('Learning rate',    _fmtNum(lr)),
    _kv('Initialization',   `Gaussian, σ = ${_fmtNum(scale)}`),
    _kv('Loss function',    'Mean squared error'),
    _kv('Dataset size',     `P = ${typeof S_NUM_DATA !== 'undefined' ? S_NUM_DATA : 64} examples`),
    _kv('Input dim',        inDim),
    _kv('Output dim',       outDim),
  ];
  if (depth > 1) items.push(_kv('Hidden width', dims[1]));
  items.push(_kv('Depth', depth));

  return `<div class="stab-kv-grid">${items.join('')}</div>`;
}

function _iomapContent() {
  const isLinear = typeof s_activation !== 'undefined' && !s_activation;
  if (isLinear) {
    return `<p>The input-output map is defined as the evolving input-output correlation matrix $\\tfrac{1}{P}\\sum_\\mu f(x^\\mu){x^\\mu}^{\\top}$ (For linear networks, this matrix is equivalent to $W_{\\text{total}}$ given whitened input data).</p>`;
  } else {
    return `<p>The input-output map is defined as the evolving input-output correlation matrix $\\tfrac{1}{P}\\sum_\\mu f(x^\\mu){x^\\mu}^{\\top}$.</p>`;
  }
}

function _targetContent() {
  const dims   = typeof s_dims !== 'undefined' ? s_dims : [6, 6];
  const actRaw = typeof s_activation !== 'undefined' ? s_activation : null;
  const inDim  = dims[0];
  if (!actRaw) {
    const P = typeof S_NUM_DATA !== 'undefined' ? S_NUM_DATA : 64;
    return `<p>The target function is $f^*(x) = U x$ for some fixed diagonal matrix $U \\in \\mathbb{R}^{${inDim} \\times ${inDim}}$ with singular values equally spaced in $(0,1)$. Inputs are drawn i.i.d. from $x^\\mu \\sim \\mathcal{N}(0, I_{${inDim}})$. The network is trained to match this linear map over a fixed dataset of $P = ${P}$ examples.</p>`;
  } else {
    // Nonlinear: g(Ux)
    const gName = typeof s_targetActivation !== 'undefined' && s_targetActivation ? s_targetActivation : actRaw;
    const p     = typeof s_projDim !== 'undefined' && s_projDim ? s_projDim : '?';
    const actDisplay = gName.charAt(0).toUpperCase() + gName.slice(1);
    return `<p>The target function is $f^*(x) = \\mathrm{${gName}}(Ux)$, where $U \\in \\mathbb{R}^{${p} \\times ${inDim}}$ is a fixed random Gaussian projection matrix and $\\mathrm{${gName}}(\\cdot)$ is applied elementwise.</p>
    <p>Inputs are drawn i.i.d. from $x^\\mu \\sim \\mathcal{N}(0, I_{${inDim}})$. The network is trained to match this nonlinear map over a fixed dataset of $P = ${typeof S_NUM_DATA !== 'undefined' ? S_NUM_DATA : 64}$ examples.</p>`;
  }
}

// ── Tab switching ─────────────────────────────────────────────────────────────

let _activeStab = null;

function _selectStab(key, animate) {
  // Toggle: clicking the active tab closes it
  if (_activeStab === key) {
    _activeStab = null;
    document.querySelectorAll('#stab-bar [role="tab"]').forEach(btn => {
      btn.setAttribute('aria-selected', 'false');
      btn.tabIndex = btn.dataset.tab === key ? 0 : -1;
    });
    document.querySelectorAll('.stab-panel').forEach(p => {
      p.classList.remove('active');
      p.style.opacity = '';
    });
    return;
  }

  _activeStab = key;

  // Update buttons
  document.querySelectorAll('#stab-bar [role="tab"]').forEach(btn => {
    const sel = btn.dataset.tab === key;
    btn.setAttribute('aria-selected', sel);
    btn.tabIndex = sel ? 0 : -1;
  });

  // Update panels
  document.querySelectorAll('.stab-panel').forEach(panel => {
    const isTarget = panel.id === `stab-panel-${key}`;
    if (isTarget) {
      panel.classList.add('active');
      if (animate) {
        panel.style.opacity = '0';
        requestAnimationFrame(() => requestAnimationFrame(() => {
          panel.style.opacity = '1';
        }));
      } else {
        panel.style.opacity = '1';
      }
    } else {
      panel.classList.remove('active');
      panel.style.opacity = '';
    }
  });
}

function openIomapTab() {
  _selectStab('iomap', true);
}

// ── Public API ────────────────────────────────────────────────────────────────

function updateSimpleTabs() {
  // Re-render all panel contents (preserves current tab selection)
  _STAB_KEYS.forEach(key => {
    const el = document.getElementById(`stab-content-${key}`);
    if (!el) return;
    el.innerHTML = _buildStabContent(key);
    if (typeof renderMathInElement !== 'undefined' && typeof KATEX_OPTS !== 'undefined') {
      renderMathInElement(el, KATEX_OPTS);
    }
  });
}

function _buildStabContent(key) {
  switch (key) {
    case 'hyperparams': return _hyperparamsContent();
    case 'iomap':       return _iomapContent();
    case 'target':      return _targetContent();
    default:            return '';
  }
}

function initSimpleTabs() {
  const tabBar = document.getElementById('stab-bar');
  if (!tabBar) return;

  // Click handlers
  tabBar.querySelectorAll('[role="tab"]').forEach(btn => {
    btn.addEventListener('click', () => _selectStab(btn.dataset.tab, true));
  });

  // Keyboard navigation (arrow keys)
  tabBar.addEventListener('keydown', e => {
    const tabs = [...tabBar.querySelectorAll('[role="tab"]')];
    const idx  = tabs.indexOf(document.activeElement);
    if (idx === -1) return;
    let next = -1;
    if (e.key === 'ArrowRight') { next = (idx + 1) % tabs.length; e.preventDefault(); }
    if (e.key === 'ArrowLeft')  { next = (idx - 1 + tabs.length) % tabs.length; e.preventDefault(); }
    if (next !== -1) { tabs[next].focus(); _selectStab(tabs[next].dataset.tab, true); }
  });
}
