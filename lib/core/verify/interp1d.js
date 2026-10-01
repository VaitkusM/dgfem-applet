/**
 * @file Piecewise polynomial interpolation in 1D and its error — the
 *       simplest setting in which to see "order of convergence".
 *
 * [a,b] is cut into N cells of width h = (b−a)/N; on each cell we
 * interpolate f at p+1 nodes (GLL nodes mapped to the cell; for p = 1 these
 * are the cell end points, i.e. ordinary piecewise-linear interpolation).
 * For smooth f the interpolation error behaves like C h^{p+1} in the L² and
 * max norms; for non-smooth f the rate is limited by the smoothness.
 */
import { gaussLobatto, gaussLegendre } from '../quad/gauss1d.js';
import { lagrangeValues } from '../basis/lagrange.js';

/**
 * Nodes on [-1,1] used for degree p (p = 0: midpoint).
 * @param {number} p
 * @returns {Float64Array}
 */
export function interpNodes(p) {
  return p === 0 ? Float64Array.of(0) : gaussLobatto(p + 1).x;
}

/**
 * Evaluate the piecewise interpolant of f at x.
 * @param {(x:number)=>number} f
 * @param {number} N cells
 * @param {number} p degree
 * @param {number} x
 * @param {number} [a=0]
 * @param {number} [b=1]
 */
export function interpolantAt(f, N, p, x, a = 0, b = 1) {
  const h = (b - a) / N;
  const e = Math.min(N - 1, Math.max(0, Math.floor((x - a) / h)));
  const xl = a + e * h, nodes = interpNodes(p);
  const xi = 2 * (x - xl) / h - 1;
  const L = lagrangeValues(nodes, xi);
  let s = 0;
  for (let j = 0; j < nodes.length; j++) s += L[j] * f(xl + (nodes[j] + 1) * h / 2);
  return s;
}

/**
 * L² and max-norm errors of the piecewise interpolant (high-order quadrature
 * with sub-sampling so that kinks/jumps inside a cell are integrated well).
 * @param {(x:number)=>number} f
 * @param {number} N
 * @param {number} p
 * @param {number} [a=0]
 * @param {number} [b=1]
 * @returns {{L2: number, Linf: number}}
 */
export function interpError(f, N, p, a = 0, b = 1) {
  const h = (b - a) / N, nodes = interpNodes(p), np = nodes.length;
  const G = gaussLegendre(p + 4), sub = 8; // 8 sub-intervals per cell
  let s2 = 0, mx = 0;
  const fv = new Float64Array(np), L = new Float64Array(np);
  for (let e = 0; e < N; e++) {
    const xl = a + e * h;
    for (let j = 0; j < np; j++) fv[j] = f(xl + (nodes[j] + 1) * h / 2);
    for (let s = 0; s < sub; s++) {
      const x0 = xl + s * h / sub, hs = h / sub;
      for (let q = 0; q < G.x.length; q++) {
        const x = x0 + (G.x[q] + 1) * hs / 2;
        const xi = 2 * (x - xl) / h - 1;
        lagrangeValues(nodes, xi, L);
        let ih = 0;
        for (let j = 0; j < np; j++) ih += L[j] * fv[j];
        const err = f(x) - ih;
        s2 += G.w[q] * hs / 2 * err * err;
        mx = Math.max(mx, Math.abs(err));
      }
    }
  }
  return { L2: Math.sqrt(s2), Linf: mx };
}
