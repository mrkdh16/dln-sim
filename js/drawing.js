// ============================================================================
// CANVAS DRAWING — proportional trapezoid chain, light theme
// ============================================================================
//
// Layout:
//
//   dims[0]   dims[1]   dims[2]        dims[L]
//     |    W₁   |    W₂   |   …    W_L   |
//  x ──[trapezoid]──[trapezoid]──…──[trapezoid]──▶ y
//       SV bars    SV bars             SV bars
//
//               [  W_e2e trapezoid  ]
//                    SV bars
//
// Each trapezoid reflects the actual input/output dimensions of that
// weight matrix: left-edge height ∝ dims[i], right-edge height ∝ dims[i+1].
// Consecutive trapezoids share edges — creating a seamless ribbon.
// ============================================================================

// Hit-test records, rebuilt each frame
let matrixBoxes = []; // [{x,y,w,h,idx}]

// Logical canvas dimensions
let canvasW = 800;
let canvasH = 500;

// ── Unicode subscript helper ─────────────────────────────────────────────────

const SUBS = '₀₁₂₃₄₅₆₇₈₉';
function sub(n) {
  return n.toString().split('').map(d => SUBS[+d]).join('');
}

// ── Resize ───────────────────────────────────────────────────────────────────

function resizeCanvas() {
  const canvas = document.getElementById('main-canvas');
  const wrap   = document.getElementById('canvas-panel');
  const dpr    = window.devicePixelRatio || 1;

  canvasW = wrap.clientWidth;
  canvasH = wrap.clientHeight;

  canvas.width  = canvasW * dpr;
  canvas.height = canvasH * dpr;

  canvas.style.width  = canvasW + 'px';
  canvas.style.height = canvasH + 'px';

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// ── Dimension → pixel height ─────────────────────────────────────────────────

function dimScale() {
  const maxDim = Math.max(...dims);
  const maxH   = Math.min(canvasH * 0.3, 100);
  return maxH / maxDim;
}

function dimH(d) {
  return Math.max(12, d * dimScale());
}

// ── Color palette (works on light background) ────────────────────────────────

const SV_COLORS = [
  '#0969da', '#1a7f37', '#d1242f', '#8250df',
  '#bc4c00', '#005cc5', '#2da44e', '#cf222e',
  '#6e40c9', '#953800',
];

// ── Main draw ────────────────────────────────────────────────────────────────

function draw() {
  const canvas = document.getElementById('main-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  ctx.clearRect(0, 0, canvasW, canvasH);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvasW, canvasH);

  matrixBoxes = [];

  const depth  = getDepth();
  const margin = 44;

  // Trapezoid geometry
  const availW   = canvasW - 2 * margin;
  const trapW    = Math.max(54, Math.min(110, availW / depth * 0.85));
  const totalW   = trapW * depth;
  const startX   = canvasW / 2 - totalW / 2;
  const maxHalfH = dimH(Math.max(...dims)) / 2;
  const chainY   = maxHalfH + 44; // leave room above for labels + instruction

  // ── Instruction hint ───────────────────────────────────────────────────────
  ctx.fillStyle = '#8c959f';
  ctx.font      = '11px -apple-system, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Click a matrix to inspect its singular value history', margin, 18);

  // ── "x" label + arrow ─────────────────────────────────────────────────────
  ctx.fillStyle = '#57606a';
  ctx.font      = '13px monospace';
  ctx.textAlign = 'right';
  ctx.fillText('x', startX - 16, chainY + 5);
  drawArrow(ctx, startX - 14, chainY, startX, chainY);

  // ── Trapezoid chain ────────────────────────────────────────────────────────
  for (let i = 0; i < depth; i++) {
    const lH = dimH(dims[i]);
    const rH = dimH(dims[i + 1]);
    const tx = startX + i * trapW;
    const isSelected = (selectedMatrixIdx === i);

    drawTrapezoid(ctx, tx, chainY, trapW, lH, rH, isSelected, false);

    // Bounding box for hit-testing
    const boxTop = chainY - Math.max(lH, rH) / 2;
    const boxH   = Math.max(lH, rH);
    matrixBoxes.push({ x: tx, y: boxTop, w: trapW, h: boxH, idx: i });

    // Matrix label
    const innerH = Math.min(lH, rH);
    const fontSize = Math.min(12, Math.max(8, innerH * 0.28));
    ctx.fillStyle = isSelected ? '#ffffff' : '#24292f';
    ctx.font      = `bold ${fontSize}px monospace`;
    ctx.textAlign = 'center';
    ctx.fillText(`W${sub(i + 1)}`, tx + trapW / 2, chainY + 2);

    if (innerH > 28) {
      ctx.fillStyle = isSelected ? 'rgba(255,255,255,0.8)' : '#57606a';
      ctx.font      = `${Math.max(8, fontSize - 2)}px monospace`;
      ctx.fillText(`${dims[i + 1]}×${dims[i]}`, tx + trapW / 2, chainY + fontSize + 4);
    }

    // SV bars below chain
    const svs = latestSVs(i);
    if (svs) {
      drawSVBars(ctx, tx, chainY + maxHalfH + 12, trapW, 44, svs);
    }
  }

  // ── Dimension labels at junctions (above chain) ────────────────────────────
  for (let i = 0; i <= depth; i++) {
    const jx = startX + i * trapW;
    const jH = dimH(dims[i]);
    ctx.fillStyle = '#8c959f';
    ctx.font      = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(String(dims[i]), jx, chainY - jH / 2 - 7);
  }

  // ── "y" label + arrow ─────────────────────────────────────────────────────
  const endX = startX + totalW;
  drawArrow(ctx, endX, chainY, endX + 14, chainY);
  ctx.fillStyle = '#57606a';
  ctx.font      = '13px monospace';
  ctx.textAlign = 'left';
  ctx.fillText('y', endX + 18, chainY + 5);

  // ── End-to-end product ────────────────────────────────────────────────────
  const e2eLH     = dimH(dims[0]);
  const e2eRH     = dimH(dims[depth]);
  const e2eMaxH   = Math.max(e2eLH, e2eRH);
  const e2eTrapW  = Math.min(trapW * 1.15, 120);
  const e2eX      = canvasW / 2 - e2eTrapW / 2;
  const svBotY    = chainY + maxHalfH + 12 + 44; // bottom of chain SV bars
  const e2eY      = svBotY + 16 + e2eMaxH / 2;   // center of e2e trapezoid
  const isE2ESel  = (selectedMatrixIdx === depth);

  // "W_e2e = W_L ··· W₁" label above the e2e box
  ctx.fillStyle = '#8c959f';
  ctx.font      = '10px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('W_e2e = W_L ··· W₁', canvasW / 2, e2eY - e2eMaxH / 2 - 7);

  drawTrapezoid(ctx, e2eX, e2eY, e2eTrapW, e2eLH, e2eRH, isE2ESel, true);

  matrixBoxes.push({
    x: e2eX, y: e2eY - e2eMaxH / 2,
    w: e2eTrapW, h: e2eMaxH, idx: depth,
  });

  // e2e labels
  const e2eInner = Math.min(e2eLH, e2eRH);
  ctx.fillStyle = isE2ESel ? '#ffffff' : '#24292f';
  ctx.font      = `bold ${Math.min(12, Math.max(8, e2eInner * 0.28))}px monospace`;
  ctx.textAlign = 'center';
  ctx.fillText('W_e2e', canvasW / 2, e2eY + 2);
  ctx.fillStyle = isE2ESel ? 'rgba(255,255,255,0.8)' : '#57606a';
  ctx.font      = '9px monospace';
  ctx.fillText(`${dims[depth]}×${dims[0]}`, canvasW / 2, e2eY + 13);

  const e2eSVs = latestSVs(depth);
  if (e2eSVs) {
    drawSVBars(ctx, e2eX, e2eY + e2eMaxH / 2 + 10, e2eTrapW, 44, e2eSVs);
  }
}

// ── Drawing primitives ────────────────────────────────────────────────────────

function drawTrapezoid(ctx, x, cy, w, lH, rH, selected, isE2E) {
  const baseCol = isE2E ? '#6e40c9' : '#0969da';
  const fillCol = selected
    ? (isE2E ? 'rgba(110,64,201,0.55)' : 'rgba(9,105,218,0.55)')
    : (isE2E ? 'rgba(110,64,201,0.10)' : 'rgba(9,105,218,0.10)');

  ctx.fillStyle   = fillCol;
  ctx.strokeStyle = baseCol;
  ctx.lineWidth   = selected ? 2 : 1.5;

  ctx.beginPath();
  ctx.moveTo(x,     cy - lH / 2); // top-left
  ctx.lineTo(x + w, cy - rH / 2); // top-right
  ctx.lineTo(x + w, cy + rH / 2); // bottom-right
  ctx.lineTo(x,     cy + lH / 2); // bottom-left
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function drawSVBars(ctx, x, y, w, h, svs) {
  if (!svs || svs.length === 0) return;
  const n    = svs.length;
  const gap  = 2;
  const barW = Math.max(1, (w - gap * (n - 1)) / n);
  const maxV = Math.max(...svs, 1e-9);

  for (let i = 0; i < n; i++) {
    const bh = Math.max(1, (svs[i] / maxV) * h);
    const bx = x + i * (barW + gap);
    const by = y + h - bh;
    ctx.fillStyle = SV_COLORS[i % SV_COLORS.length];
    ctx.fillRect(bx, by, barW, bh);
  }

  ctx.strokeStyle = '#d0d7de';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(x,     y + h);
  ctx.lineTo(x + w, y + h);
  ctx.stroke();
}

function drawArrow(ctx, x1, y1, x2, y2) {
  const headLen = 6;
  const angle   = Math.atan2(y2 - y1, x2 - x1);

  ctx.strokeStyle = '#8c959f';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  ctx.fillStyle = '#8c959f';
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6),
             y2 - headLen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6),
             y2 - headLen * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function latestSVs(idx) {
  if (!svHistories[idx] || svHistories[idx].length === 0) return null;
  return svHistories[idx][svHistories[idx].length - 1].svs;
}

// ── Canvas click → selection ─────────────────────────────────────────────────

function setupCanvasEvents() {
  const canvas = document.getElementById('main-canvas');

  canvas.addEventListener('click', e => {
    const rect   = canvas.getBoundingClientRect();
    const scaleX = canvasW / rect.width;
    const scaleY = canvasH / rect.height;
    const mx     = (e.clientX - rect.left) * scaleX;
    const my     = (e.clientY - rect.top)  * scaleY;

    let hit = null;
    for (const box of matrixBoxes) {
      if (mx >= box.x && mx <= box.x + box.w &&
          my >= box.y && my <= box.y + box.h) {
        hit = box.idx;
      }
    }

    selectedMatrixIdx = hit;
    refreshSVPanel();
    draw();
  });
}

function refreshSVPanel() {
  const hintEl    = document.getElementById('sv-hint');
  const wrapEl    = document.getElementById('sv-chart-wrap');
  const titleEl   = document.getElementById('sv-chart-title');
  const theoryRow = document.getElementById('theory-row');
  const togglesEl = document.getElementById('sv-toggles');
  const depth     = getDepth();

  if (selectedMatrixIdx === null) {
    hintEl.style.display    = '';
    hintEl.textContent      = 'Click a matrix on the canvas to view its singular value history.';
    wrapEl.style.display    = 'none';
    theoryRow.style.display = 'none';
    titleEl.textContent     = 'singular values';
    if (togglesEl) togglesEl.style.display = 'none';
    return;
  }

  // Show toggle pills whenever a matrix is selected
  if (togglesEl) togglesEl.style.display = '';

  const isE2E     = (selectedMatrixIdx === depth);
  const hasHistory = svHistories[selectedMatrixIdx] &&
                     svHistories[selectedMatrixIdx].length > 0;

  // Update title
  titleEl.textContent = isE2E
    ? 'W_e2e — effective singular values'
    : `W${sub(selectedMatrixIdx + 1)} — singular values`;

  // Theory formula row: visible only for e2e with depth=2
  // (Saxe et al. formula is exact only for the depth-2 case)
  theoryRow.style.display = (isE2E && depth === 2) ? '' : 'none';

  if (hasHistory) {
    hintEl.style.display = 'none';
    wrapEl.style.display = '';
    updateSVChart();
  } else {
    hintEl.style.display = '';
    hintEl.textContent   = 'Start the simulation to collect singular value data.';
    wrapEl.style.display = 'none';
  }
}
