/**
 * @file Drawing meshes: triangle/quad edges, highlighted cells, vertices,
 *       vector glyphs.
 */
import { theme, arrow } from './canvas.js';

/**
 * Stroke all triangle edges.
 * @param {CanvasRenderingContext2D} ctx
 * @param {import('./canvas.js').View2D} view
 * @param {Float64Array} nodes
 * @param {Int32Array} tris
 * @param {{color?: string, width?: number, alpha?: number, skip?: (t:number)=>boolean}} [o]
 */
export function drawTriMesh(ctx, view, nodes, tris, o = {}) {
  ctx.save();
  ctx.strokeStyle = o.color || theme().faint; ctx.lineWidth = o.width ?? 0.7; ctx.globalAlpha = o.alpha ?? 1;
  ctx.beginPath();
  for (let t = 0; t < tris.length / 3; t++) {
    if (o.skip && o.skip(t)) continue;
    for (let k = 0; k < 3; k++) {
      const a = tris[3 * t + k], b = tris[3 * t + ((k + 1) % 3)];
      ctx.moveTo(view.X(nodes[2 * a]), view.Y(nodes[2 * a + 1]));
      ctx.lineTo(view.X(nodes[2 * b]), view.Y(nodes[2 * b + 1]));
    }
  }
  ctx.stroke();
  ctx.restore();
}

/**
 * Fill one triangle.
 */
export function fillTri(ctx, view, nodes, tris, t, color, alpha = 1) {
  ctx.save();
  ctx.fillStyle = color; ctx.globalAlpha = alpha;
  ctx.beginPath();
  for (let k = 0; k < 3; k++) {
    const v = tris[3 * t + k];
    const X = view.X(nodes[2 * v]), Y = view.Y(nodes[2 * v + 1]);
    if (k === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
  }
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

/**
 * Grid lines of a structured nx×ny grid over box.
 */
export function drawGrid(ctx, view, nx, ny, box, o = {}) {
  const [x0, x1, y0, y1] = box;
  ctx.save();
  ctx.strokeStyle = o.color || theme().faint; ctx.lineWidth = o.width ?? 0.7; ctx.globalAlpha = o.alpha ?? 1;
  ctx.beginPath();
  for (let i = 0; i <= nx; i++) { const X = view.X(x0 + (x1 - x0) * i / nx); ctx.moveTo(X, view.Y(y0)); ctx.lineTo(X, view.Y(y1)); }
  for (let j = 0; j <= ny; j++) { const Y = view.Y(y0 + (y1 - y0) * j / ny); ctx.moveTo(view.X(x0), Y); ctx.lineTo(view.X(x1), Y); }
  ctx.stroke();
  ctx.restore();
}

/**
 * Vector glyphs (arrows) at points.
 * @param {CanvasRenderingContext2D} ctx
 * @param {import('./canvas.js').View2D} view
 * @param {ArrayLike<number>} xs
 * @param {ArrayLike<number>} ys
 * @param {ArrayLike<number>} us
 * @param {ArrayLike<number>} vs
 * @param {{scale?: number, color?: string, width?: number, head?: number}} [o] scale: world length per unit vector
 */
export function drawVectors(ctx, view, xs, ys, us, vs, o = {}) {
  ctx.save();
  ctx.strokeStyle = ctx.fillStyle = o.color || theme().ink; ctx.lineWidth = o.width ?? 1.2;
  const s = o.scale ?? 1;
  for (let i = 0; i < xs.length; i++) {
    const X0 = view.X(xs[i]), Y0 = view.Y(ys[i]);
    arrow(ctx, X0, Y0, X0 + us[i] * s * view.sx, Y0 - vs[i] * s * view.sy, o.head ?? 5);
  }
  ctx.restore();
}
