/**
 * @file "Spy" plot: draw the sparsity pattern of a matrix (one square per
 *       stored non-zero), optionally colouring rows/columns by block.
 */
import { theme } from './canvas.js';

/**
 * @param {{ctx: CanvasRenderingContext2D, w: number, h: number}} surf
 * @param {import('../core/la/sparse.js').CSR} A
 * @param {{color?: string, blocks?: number[], blockColors?: string[], title?: string, magnitude?: boolean}} [o]
 *   blocks: increasing split indices (e.g. [nInterior, n]) to draw separator lines
 */
export function drawSpy(surf, A, o = {}) {
  const { ctx, w, h } = surf, T = theme();
  ctx.save();
  ctx.fillStyle = T.bg; ctx.fillRect(0, 0, w, h);
  const pad = 6, size = Math.min(w, h) - 2 * pad;
  const ox = (w - size) / 2, oy = pad, cell = size / Math.max(A.n, A.m);
  ctx.fillStyle = T.elev; ctx.fillRect(ox, oy, cell * A.m, cell * A.n);
  let vmax = 0;
  if (o.magnitude) for (let k = 0; k < A.vals.length; k++) vmax = Math.max(vmax, Math.abs(A.vals[k]));
  const s = Math.max(cell, 1.2);
  ctx.fillStyle = o.color || T.accent;
  for (let i = 0; i < A.n; i++)
    for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      if (A.vals[k] === 0) continue;
      if (o.magnitude) ctx.globalAlpha = 0.25 + 0.75 * Math.min(1, Math.abs(A.vals[k]) / vmax) ** 0.3;
      ctx.fillRect(ox + A.colIdx[k] * cell, oy + i * cell, s, s);
    }
  ctx.globalAlpha = 1;
  if (o.blocks) {
    ctx.strokeStyle = T.accent2; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    for (const b of o.blocks) {
      if (b <= 0 || b >= Math.max(A.n, A.m)) continue;
      ctx.beginPath();
      ctx.moveTo(ox + b * cell, oy); ctx.lineTo(ox + b * cell, oy + A.n * cell);
      ctx.moveTo(ox, oy + b * cell); ctx.lineTo(ox + A.m * cell, oy + b * cell);
      ctx.stroke();
    }
  }
  ctx.setLineDash([]);
  ctx.strokeStyle = T.faint; ctx.strokeRect(ox + 0.5, oy + 0.5, cell * A.m, cell * A.n);
  ctx.restore();
}
