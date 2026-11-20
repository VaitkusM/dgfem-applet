/**
 * @file Both sides of the divergence theorem on a polygon,
 *         ∮_∂P F·n ds  =  ∫_P ∇·F dA,
 *       computed in two completely independent ways (edge quadrature vs.
 *       area quadrature on a triangulation). Used by the chapter-0 playground
 *       and by the tests.
 *
 * A polygon is an array of vertices [[x0,y0],[x1,y1],…] in either orientation;
 * edge i joins vertex i and vertex (i+1) mod n. It should be simple (not
 * self-intersecting); zero-length edges (two coincident consecutive vertices)
 * are allowed and contribute nothing.
 */
import { gaussLegendre } from '../quad/gauss1d.js';
import { triangleRule } from '../quad/simplex.js';

/**
 * Twice the signed area of a polygon (shoelace formula): positive for
 * counter-clockwise vertex order.
 * @param {number[][]} poly vertices [[x,y],…]
 * @returns {number}
 */
export function polygonArea2(poly) {
  const n = poly.length;
  let area2 = 0;
  for (let i = 0; i < n; i++) { const [ax, ay] = poly[i], [bx, by] = poly[(i + 1) % n]; area2 += ax * by - bx * ay; }
  return area2;
}

/**
 * Outward flux ∫ F·n ds through each polygon edge (composite Gauss rule:
 * 8 sub-segments × 8 points per edge). Works for either orientation: the
 * normals are flipped for clockwise polygons so that they always point out.
 * A zero-length edge gets flux 0 and the normal (0,0).
 * @param {number[][]} poly vertices [[x,y],…]
 * @param {(x:number, y:number)=>number[]} F vector field, returns [Fx, Fy]
 * @returns {{fl: {flux:number, nx:number, ny:number, mx:number, my:number}[], orient: number}}
 *   per edge: flux, outward unit normal (nx,ny) and midpoint (mx,my);
 *   orient = +1 for a counter-clockwise polygon, −1 for clockwise
 */
export function polygonEdgeFluxes(poly, F) {
  const n = poly.length, G = gaussLegendre(8), SUB = 8;
  const orient = polygonArea2(poly) >= 0 ? 1 : -1;
  const fl = [];
  for (let i = 0; i < n; i++) {
    const [ax, ay] = poly[i], [bx, by] = poly[(i + 1) % n];
    const len = Math.hypot(bx - ax, by - ay), mx = (ax + bx) / 2, my = (ay + by) / 2;
    if (!(len > 1e-12)) { fl.push({ flux: 0, nx: 0, ny: 0, mx, my }); continue; } // degenerate edge
    const nx = orient * (by - ay) / len, ny = -orient * (bx - ax) / len; // outward normal
    let s = 0;
    for (let k = 0; k < SUB; k++)
      for (let q = 0; q < G.x.length; q++) {
        const t = (k + (G.x[q] + 1) / 2) / SUB, [fx, fy] = F(ax + t * (bx - ax), ay + t * (by - ay));
        s += G.w[q] / 2 / SUB * len * (fx * nx + fy * ny);
      }
    fl.push({ flux: s, nx, ny, mx, my });
  }
  return { fl, orient };
}

/**
 * ∫_P f dA for a simple polygon P via a signed fan triangulation from vertex
 * 0: fan triangles with negative orientation subtract, so the result is
 * correct for any simple (also non-convex) polygon in either orientation.
 * Each fan triangle is refined into m² = 64 sub-triangles with a degree-12
 * rule on each.
 * @param {number[][]} poly vertices [[x,y],…]
 * @param {(x:number, y:number)=>number} f integrand
 * @returns {number}
 */
export function polygonAreaIntegral(poly, f) {
  const R = triangleRule(12), m = 8;
  let total = 0, area2 = 0;
  const [ox, oy] = poly[0];
  for (let i = 1; i + 1 < poly.length; i++) {
    const [ax, ay] = poly[i], [bx, by] = poly[i + 1];
    const det = (ax - ox) * (by - oy) - (bx - ox) * (ay - oy); // signed 2×area
    let s = 0;
    // uniform refinement of the reference triangle: "upright" and "flipped" sub-triangles
    for (let I = 0; I < m; I++) for (let J = 0; I + J < m; J++) {
      const subs = [[[I, J], [I + 1, J], [I, J + 1]]];
      if (I + J < m - 1) subs.push([[I + 1, J], [I + 1, J + 1], [I, J + 1]]);
      for (const [a, b, c] of subs)
        for (let q = 0; q < R.n; q++) {
          const sx = (a[0] + (b[0] - a[0]) * R.x[q] + (c[0] - a[0]) * R.y[q]) / m;
          const sy = (a[1] + (b[1] - a[1]) * R.x[q] + (c[1] - a[1]) * R.y[q]) / m;
          s += R.w[q] / (m * m) * f(ox + (ax - ox) * sx + (bx - ox) * sy, oy + (ay - oy) * sx + (by - oy) * sy);
        }
    }
    total += s * det; area2 += det;
  }
  return Math.sign(area2 || 1) * total;
}
