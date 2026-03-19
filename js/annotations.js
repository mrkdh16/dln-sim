// ============================================================================
// ANNOTATIONS — click-to-open informational popovers
// ============================================================================

const _ICON_SVG = '<svg class="anno-icon-svg" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3" aria-hidden="true" focusable="false"><circle cx="8" cy="8" r="6.5"/><line x1="8" y1="7.5" x2="8" y2="11" stroke-linecap="round"/><circle cx="8" cy="5" r="0.8" fill="currentColor" stroke="none"/></svg>';

// ── Content ───────────────────────────────────────────────────────────────────

const ANNO_INFO = {

  loss: {
    title: 'Training Details',
    body() {
      const lr    = typeof s_learningRate !== 'undefined' ? s_learningRate : '—';
      const scale = typeof s_initScale    !== 'undefined' ? s_initScale    : '—';
      const fmt   = v => typeof v === 'number' ? (v % 1 === 0 ? v : v.toPrecision(2)) : v;
      return `<table class="anno-table"><tbody>
        <tr><td>Optimizer</td><td>gradient descent (full-batch)</td></tr>
        <tr><td>Learning rate</td><td>${fmt(lr)}</td></tr>
        <tr><td>Init scale</td><td>N(0, ${fmt(scale)}) per weight</td></tr>
        <tr><td>Dataset</td><td>P = 64 fixed examples</td></tr>
      </tbody></table>
      <p class="anno-note">In deep linear networks, loss drops in discrete steps — each step corresponds to one singular value mode switching on.</p>`;
    }
  },

  sv: {
    title: 'Input–Output Map',
    body() {
      const isLinear = typeof s_activation !== 'undefined' && !s_activation;
      if (isLinear) {
        return `<p>The input–output map is the matrix product $W_L \\cdots W_1$. Its singular values decompose the learned transformation into independent modes.</p>
        <p class="anno-note">The largest mode is learned first; smaller ones switch on sequentially.</p>`;
      } else {
        return `<p>We plot singular values of the correlation matrix $\\tfrac{1}{P}\\,f(X)X^T$, where $f(x^\\mu)$ is the network output on input $x^\\mu$.</p>
        <p>This captures the effective linear structure of the learned nonlinear map — it equals $W_\\mathrm{total}$ for linear networks.</p>`;
      }
    }
  },

  eq: {
    title: 'Network Architecture',
    body() {
      const dims    = typeof s_dims !== 'undefined' ? s_dims : [];
      const depth   = dims.length - 1;
      const actRaw  = typeof s_activation !== 'undefined' ? s_activation : null;
      const actName = actRaw ? actRaw.charAt(0).toUpperCase() + actRaw.slice(1) : 'none (linear)';
      let html = `<table class="anno-table"><tbody>
        <tr><td>Dimensions</td><td>${dims.join(' → ')}</td></tr>
        <tr><td>Depth</td><td>${depth}</td></tr>
        <tr><td>Activation</td><td>${actName}</td></tr>
      </tbody></table>`;
      if (typeof s_targetActivation !== 'undefined' && s_targetActivation
          && typeof s_projDim !== 'undefined' && s_projDim) {
        html += `<p style="margin-top:8px">Target: $y^\\mu = \\mathrm{${s_targetActivation}}(Ux^\\mu)$, $\\;U \\in \\mathbb{R}^{${s_projDim} \\times ${dims[0]}}$ fixed random projection.</p>`;
      } else if (!actRaw) {
        html += `<p class="anno-note" style="margin-top:8px">Without activation, $W_L\\cdots W_1$ is a single linear map. Depth affects optimization <em>dynamics</em>, not the function class.</p>`;
      }
      return html;
    }
  },

  depth1: {
    title: 'Depth 1 — Linear Regression',
    body: () => `<p>A single weight matrix. Loss decays exponentially with no phase transitions — gradient descent converges smoothly. This is the baseline against which deeper dynamics are compared.</p>`
  },

  depth2: {
    title: 'Depth 2 — Minimally Deep',
    body: () => `<p>The simplest network that exhibits sequential singular value learning. Each mode follows a sigmoidal trajectory that can be derived analytically.</p>
    <p class="anno-note">See Saxe, McClelland & Ganguli (2014) for the exact solution.</p>`
  },

  plateau: {
    title: 'Depth 3 — Deep Linear',
    body: () => `<p>Deeper networks show sharper, more stepwise loss drops. Plateau lengths grow with depth — each drop corresponds to one singular value switching on.</p>`
  },

  relu_shallow: {
    title: 'Depth 2 — Shallow ReLU',
    body: () => `<p>A single hidden layer with ReLU activation. The learned input–output map is piecewise linear. Compared to linear networks, the loss landscape has richer geometry.</p>`
  },

  relu_deep: {
    title: 'Depth 3 — Deep ReLU',
    body: () => `<p>Two hidden layers with ReLU. Deeper nonlinear networks can exhibit qualitatively different dynamics — representations in early layers may reorganize substantially during training.</p>`
  },

  tanh: {
    title: 'Depth 3 — Deep tanh',
    body: () => `<p>Two hidden layers with tanh activation. Unlike ReLU, tanh saturates — this can slow learning but produces smoother gradients and output surfaces.</p>`
  },
};

// ── Popover DOM ───────────────────────────────────────────────────────────────

let _activeKey = null;
let _popEl     = null;
let _titleEl   = null;
let _bodyEl    = null;

function _buildPopover() {
  if (_popEl) return;

  _popEl = document.createElement('div');
  _popEl.id = 'anno-pop';
  _popEl.setAttribute('role', 'dialog');
  _popEl.setAttribute('aria-modal', 'false');
  _popEl.style.display = 'none';

  _titleEl = document.createElement('div');
  _titleEl.id = 'anno-pop-title';
  _popEl.appendChild(_titleEl);

  _bodyEl = document.createElement('div');
  _bodyEl.id = 'anno-pop-body';
  _popEl.appendChild(_bodyEl);

  document.body.appendChild(_popEl);
  _popEl.addEventListener('click', e => e.stopPropagation());
}

// ── Open / close ──────────────────────────────────────────────────────────────

function openAnnotation(key, anchorEl) {
  _buildPopover();
  const info = ANNO_INFO[key];
  if (!info) return;

  if (_activeKey === key) { closeAnnotation(); return; }

  const wasOpen = _popEl.style.display !== 'none';
  _activeKey = key;

  _titleEl.textContent = info.title;
  _bodyEl.innerHTML    = info.body();

  if (typeof renderMathInElement !== 'undefined' && typeof KATEX_OPTS !== 'undefined') {
    renderMathInElement(_bodyEl, KATEX_OPTS);
  }

  _popEl.style.display    = 'block';
  _popEl.style.opacity    = wasOpen ? '1' : '0';
  _popEl.style.transition = wasOpen ? 'none' : 'opacity 0.15s ease';

  _positionPopover(anchorEl);

  if (!wasOpen) {
    requestAnimationFrame(() => { _popEl.style.opacity = '1'; });
  }
}

function closeAnnotation() {
  if (!_popEl || _popEl.style.display === 'none') return;
  _activeKey = null;
  _popEl.style.transition = 'opacity 0.15s ease';
  _popEl.style.opacity    = '0';
  setTimeout(() => {
    if (_popEl && _activeKey === null) _popEl.style.display = 'none';
  }, 160);
}

// ── Positioning ───────────────────────────────────────────────────────────────

function _positionPopover(anchorEl) {
  const rect   = anchorEl.getBoundingClientRect();
  const scrollY = window.scrollY || window.pageYOffset;
  const scrollX = window.scrollX || window.pageXOffset;
  const vpW    = window.innerWidth;
  const vpH    = window.innerHeight;
  const GAP    = 10;
  const MAXW   = 320;

  const popW = Math.min(MAXW, vpW - 20);
  _popEl.style.width    = popW + 'px';
  _popEl.style.maxWidth = MAXW + 'px';

  const popH = _popEl.offsetHeight;
  const anchorCX = rect.left + rect.width  / 2;

  // Prefer below; flip to above if insufficient room
  const placeBelow = (vpH - rect.bottom) >= popH + GAP || (vpH - rect.bottom) >= rect.top;

  let top;
  if (placeBelow) {
    top = rect.bottom + scrollY + GAP;
    _popEl.className = 'arr-up';
  } else {
    top = rect.top + scrollY - popH - GAP;
    _popEl.className = 'arr-down';
  }

  let left = anchorCX + scrollX - popW / 2;
  left = Math.max(scrollX + 8, Math.min(left, scrollX + vpW - popW - 8));

  _popEl.style.top  = top  + 'px';
  _popEl.style.left = left + 'px';

  // Arrow position relative to popover
  const arrowLeft = anchorCX + scrollX - left - 7; // 7 = half arrow base width
  _popEl.style.setProperty('--anno-arrow', Math.max(12, Math.min(arrowLeft, popW - 22)) + 'px');
}

// ── Init ──────────────────────────────────────────────────────────────────────

function initAnnotations() {
  _buildPopover();

  document.querySelectorAll('[data-annotation]').forEach(el => {
    el.addEventListener('click', e => {
      e.stopPropagation();
      openAnnotation(el.dataset.annotation, el);
    });
  });

  document.addEventListener('click', () => closeAnnotation());
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeAnnotation(); });
}
