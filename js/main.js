// ============================================================================
// ENTRY POINT
// ============================================================================

window.addEventListener('DOMContentLoaded', () => {
  // 1. Size the canvas to fill its container
  resizeCanvas();

  // 2. Create Chart.js chart instances
  initCharts();

  // 3. Wire up canvas click events (matrix selection)
  setupCanvasEvents();

  // 4. Wire up controls bar (sliders, buttons, config)
  setupUIHandlers();

  // 5. Wire up panel drag-resize handles
  setupPanelResizers();

  // 6. Init simulation so the canvas shows the network skeleton
  initSimulation();
  draw();

  // 7. Reflect the default selectedMatrixIdx (W_e2e) in the SV panel
  refreshSVPanel();
});

// ── Panel resizers ────────────────────────────────────────────────────────────

function setupPanelResizers() {
  const mainDivider  = document.getElementById('main-divider');
  const leftDivider  = document.getElementById('left-divider');
  const leftPanel    = document.getElementById('left-panel');
  const canvasPanel  = document.getElementById('canvas-panel');
  const mainEl       = document.getElementById('main');

  // ── Horizontal: left ↔ right ─────────────────────────────────────────────
  let hDragging = false;

  mainDivider.addEventListener('mousedown', e => {
    hDragging = true;
    mainDivider.classList.add('dragging');
    document.body.classList.add('resizing-h');
    e.preventDefault();
  });

  // ── Vertical: canvas ↔ controls ──────────────────────────────────────────
  let vDragging = false;

  leftDivider.addEventListener('mousedown', e => {
    vDragging = true;
    leftDivider.classList.add('dragging');
    document.body.classList.add('resizing-v');
    e.preventDefault();
  });

  document.addEventListener('mousemove', e => {
    if (hDragging) {
      const rect = mainEl.getBoundingClientRect();
      const newW = Math.max(180, Math.min(rect.width - 260, e.clientX - rect.left));
      leftPanel.style.width = newW + 'px';
      resizeCanvas();
      draw();
      if (lossChart) lossChart.resize();
      if (svChart)   svChart.resize();
    }
    if (vDragging) {
      const rect = leftPanel.getBoundingClientRect();
      const newH = Math.max(60, Math.min(rect.height - 100, e.clientY - rect.top));
      canvasPanel.style.flex = `0 0 ${newH}px`;
      resizeCanvas();
      draw();
    }
  });

  document.addEventListener('mouseup', () => {
    if (hDragging) {
      hDragging = false;
      mainDivider.classList.remove('dragging');
      document.body.classList.remove('resizing-h');
    }
    if (vDragging) {
      vDragging = false;
      leftDivider.classList.remove('dragging');
      document.body.classList.remove('resizing-v');
    }
  });
}

// Redraw on window resize
window.addEventListener('resize', () => {
  resizeCanvas();
  draw();
});
