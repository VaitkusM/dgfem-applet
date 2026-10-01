/**
 * @file Iso-lines of a scalar function by marching squares (the 2D cousin of
 *       marching cubes): sample f on a grid, and in every grid square connect
 *       the points where f crosses the level (found by linear interpolation
 *       along the square's edges).
 */

/**
 * Draw contour lines of f(x,y) over the view.
 * @param {CanvasRenderingContext2D} ctx
 * @param {import('./canvas.js').View2D} view
 * @param {(x:number, y:number) => number} f
 * @param {number[]} levels
 * @param {{n?: number, color?: string, width?: number, alpha?: number}} [o] n = grid resolution
 */
export function drawContours(ctx, view, f, levels, o = {}) {
  const n = o.n ?? 80;
  const { x0, x1, y0, y1 } = view;
  const xs = (i) => x0 + (x1 - x0) * i / n, ys = (j) => y0 + (y1 - y0) * j / n;
  const F = new Float64Array((n + 1) * (n + 1));
  for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) F[j * (n + 1) + i] = f(xs(i), ys(j));
  ctx.save();
  ctx.strokeStyle = o.color || 'rgba(255,255,255,0.8)'; ctx.lineWidth = o.width ?? 1; ctx.globalAlpha = o.alpha ?? 1;
  ctx.beginPath();
  for (const lev of levels) {
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      // corners: 0 (i,j), 1 (i+1,j), 2 (i+1,j+1), 3 (i,j+1)
      const v = [F[j * (n + 1) + i], F[j * (n + 1) + i + 1], F[(j + 1) * (n + 1) + i + 1], F[(j + 1) * (n + 1) + i]];
      const P = [[xs(i), ys(j)], [xs(i + 1), ys(j)], [xs(i + 1), ys(j + 1)], [xs(i), ys(j + 1)]];
      const pts = [];
      for (let e = 0; e < 4; e++) {
        const a = v[e] - lev, b = v[(e + 1) % 4] - lev;
        if ((a < 0) !== (b < 0)) {
          const t = a / (a - b), A = P[e], B = P[(e + 1) % 4];
          pts.push([A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])]);
        }
      }
      // 2 crossings: one segment; 4 crossings (saddle): pair them in order
      for (let k = 0; k + 1 < pts.length; k += 2) {
        ctx.moveTo(view.X(pts[k][0]), view.Y(pts[k][1]));
        ctx.lineTo(view.X(pts[k + 1][0]), view.Y(pts[k + 1][1]));
      }
    }
  }
  ctx.stroke();
  ctx.restore();
}
