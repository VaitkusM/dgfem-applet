/**
 * @file One-dimensional quadrature rules on the reference interval [-1, 1].
 *
 * A quadrature rule approximates ∫_{-1}^{1} f(x) dx ≈ Σ_q w_q f(x_q).
 *
 *  - Gauss–Legendre (n points): exact for polynomials of degree ≤ 2n − 1.
 *    Nodes are the roots of P_n; no endpoint is included.
 *  - Gauss–Lobatto–Legendre, "GLL" (n points, n ≥ 2): includes both endpoints
 *    ±1; interior nodes are roots of P'_{n−1}. Exact for degree ≤ 2n − 3.
 *    GLL nodes are the standard interpolation nodes of nodal DG / spectral
 *    elements because they contain the element boundary.
 *
 * Nodes are returned in ascending order.
 */
import { legendreAll } from '../basis/legendre.js';

/** @typedef {{x: Float64Array, w: Float64Array}} Rule1D */

const cacheG = new Map();
const cacheL = new Map();

/**
 * Gauss–Legendre rule with n points (Newton iteration on P_n).
 * @param {number} n number of points ≥ 1
 * @returns {Rule1D}
 */
export function gaussLegendre(n) {
  if (cacheG.has(n)) return cacheG.get(n);
  const x = new Float64Array(n), w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    // Chebyshev-like initial guess, descending; we fill ascending afterwards.
    let z = Math.cos(Math.PI * (i + 0.75) / (n + 0.5));
    for (let it = 0; it < 100; it++) {
      const { P, dP } = legendreAll(n, z);
      const dz = P[n] / dP[n];
      z -= dz;
      if (Math.abs(dz) < 1e-16) break;
    }
    const { dP } = legendreAll(n, z);
    x[n - 1 - i] = z;
    w[n - 1 - i] = 2 / ((1 - z * z) * dP[n] * dP[n]);
  }
  const rule = { x, w };
  cacheG.set(n, rule);
  return rule;
}

/**
 * Gauss–Lobatto–Legendre rule with n points (n ≥ 2), following the classic
 * Newton iteration of G. von Winckel ("lglnodes").
 * @param {number} n number of points ≥ 2
 * @returns {Rule1D}
 */
export function gaussLobatto(n) {
  if (n < 2) throw new Error('gaussLobatto: need n >= 2');
  if (cacheL.has(n)) return cacheL.get(n);
  const N = n - 1; // polynomial degree
  const x = new Float64Array(n), w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let z = Math.cos(Math.PI * i / N);
    for (let it = 0; it < 100; it++) {
      const { P } = legendreAll(N, z);
      // Newton step for (1 − z²) P'_N(z) = 0, written via P_N and P_{N−1}.
      const dz = (z * P[N] - P[N - 1]) / (n * P[N]);
      z -= dz;
      if (Math.abs(dz) < 1e-16) break;
    }
    const { P } = legendreAll(N, z);
    x[N - i] = z;
    w[N - i] = 2 / (N * n * P[N] * P[N]);
  }
  x[0] = -1; x[N] = 1; // exact endpoints
  const rule = { x, w };
  cacheL.set(n, rule);
  return rule;
}

/**
 * Map a reference rule on [-1,1] to [a,b].
 * @param {Rule1D} rule
 * @param {number} a
 * @param {number} b
 * @returns {Rule1D}
 */
export function mapRule(rule, a, b) {
  const n = rule.x.length, x = new Float64Array(n), w = new Float64Array(n);
  const J = (b - a) / 2;
  for (let i = 0; i < n; i++) {
    x[i] = a + (rule.x[i] + 1) * J;
    w[i] = rule.w[i] * J;
  }
  return { x, w };
}

/**
 * Gauss–Legendre rule on [0,1] that integrates degree `deg` exactly.
 * @param {number} deg polynomial degree to integrate exactly
 * @returns {Rule1D}
 */
export function gaussForDegree01(deg) {
  const n = Math.max(1, Math.ceil((deg + 1) / 2));
  return mapRule(gaussLegendre(n), 0, 1);
}
