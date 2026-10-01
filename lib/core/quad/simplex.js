/**
 * @file Quadrature on the reference triangle T̂ = {(x,y): x ≥ 0, y ≥ 0, x + y ≤ 1}.
 *
 * Construction ("Duffy collapse"): the unit square (s,t) ∈ [0,1]² is mapped
 * onto T̂ by   x = s(1 − t),  y = t,   with Jacobian det = (1 − t).
 * (Graphics analogy: the top edge of the square is squashed to the vertex
 * (0,1), like a degenerate bilinear patch.)
 * A polynomial of total degree d in (x,y) becomes degree ≤ d in s and
 * ≤ d + 1 in t (including the Jacobian), so tensor Gauss rules with
 * ⌈(d+1)/2⌉ points in s and ⌈(d+2)/2⌉ points in t integrate it exactly.
 *
 * Weights sum to |T̂| = 1/2.
 */
import { gaussForDegree01 } from './gauss1d.js';

/** @typedef {{x: Float64Array, y: Float64Array, w: Float64Array, n: number}} Rule2D */

const cache = new Map();

/**
 * Collapsed-Gauss rule on the reference triangle, exact for total degree `deg`.
 * @param {number} deg
 * @returns {Rule2D}
 */
export function triangleRule(deg) {
  if (cache.has(deg)) return cache.get(deg);
  const rs = gaussForDegree01(deg), rt = gaussForDegree01(deg + 1);
  const n = rs.x.length * rt.x.length;
  const x = new Float64Array(n), y = new Float64Array(n), w = new Float64Array(n);
  let q = 0;
  for (let j = 0; j < rt.x.length; j++) {
    const t = rt.x[j];
    for (let i = 0; i < rs.x.length; i++) {
      const s = rs.x[i];
      x[q] = s * (1 - t);
      y[q] = t;
      w[q] = rs.w[i] * rt.w[j] * (1 - t);
      q++;
    }
  }
  const rule = { x, y, w, n };
  cache.set(deg, rule);
  return rule;
}

/**
 * Map the reference rule to a physical triangle with vertices a, b, c
 * (affine map  X = a + (b−a) x̂ + (c−a) ŷ).
 * @param {Rule2D} rule
 * @param {number[]} a [ax, ay]
 * @param {number[]} b
 * @param {number[]} c
 * @returns {Rule2D} physical points and weights (weights include |det J|)
 */
export function mapTriangleRule(rule, a, b, c) {
  const n = rule.n;
  const x = new Float64Array(n), y = new Float64Array(n), w = new Float64Array(n);
  const e1x = b[0] - a[0], e1y = b[1] - a[1], e2x = c[0] - a[0], e2y = c[1] - a[1];
  const det = Math.abs(e1x * e2y - e1y * e2x);
  for (let q = 0; q < n; q++) {
    x[q] = a[0] + e1x * rule.x[q] + e2x * rule.y[q];
    y[q] = a[1] + e1y * rule.x[q] + e2y * rule.y[q];
    w[q] = rule.w[q] * det;
  }
  return { x, y, w, n };
}
