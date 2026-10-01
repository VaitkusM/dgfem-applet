/**
 * @file Polynomial bases on the reference triangle T̂ with vertices
 *       v0 = (0,0), v1 = (1,0), v2 = (0,1).
 *
 * Barycentric coordinates (λ0, λ1, λ2) of a point (x,y) in T̂:
 *   λ1 = x, λ2 = y, λ0 = 1 − x − y.
 * (Exactly the barycentric coordinates used for colour interpolation in a
 * rasteriser!)
 *
 *  - P1 Lagrange: φ_i = λ_i                         (3 functions, "hat" functions)
 *  - P2 Lagrange: φ_i = λ_i(2λ_i − 1)  at vertices  (i = 0,1,2)
 *                 φ_{3+k} = 4 λ_a λ_b  at the midpoint of local edge k = (a,b),
 *                 with local edges 0:(0,1), 1:(1,2), 2:(2,0).
 *  - Dubiner (orthonormal modal) basis of degree ≤ p: L²(T̂)-orthonormal,
 *    ∫_T̂ ψ_m ψ_n = δ_mn. Built from Jacobi polynomials in "collapsed"
 *    coordinates (Hesthaven & Warburton 2008, Simplex2DP / GradSimplex2DP).
 *
 * Every evaluator returns values and reference gradients (∂/∂x̂, ∂/∂ŷ).
 */
import { jacobiP, jacobiDP } from './legendre.js';

/** Number of polynomials of total degree ≤ p in 2D: (p+1)(p+2)/2. */
export const dimP = (p) => ((p + 1) * (p + 2)) / 2;

/**
 * P1 Lagrange basis on T̂.
 * @param {number} x
 * @param {number} y
 * @returns {{v: Float64Array, dx: Float64Array, dy: Float64Array}}
 */
export function p1Basis(x, y) {
  return {
    v: Float64Array.of(1 - x - y, x, y),
    dx: Float64Array.of(-1, 1, 0),
    dy: Float64Array.of(-1, 0, 1),
  };
}

/** Local edge k of a triangle joins local vertices EDGE_VERTS[k]. */
export const EDGE_VERTS = [[0, 1], [1, 2], [2, 0]];

/**
 * P2 Lagrange basis on T̂ (6 functions: 3 vertices then 3 edge midpoints).
 * @param {number} x
 * @param {number} y
 * @returns {{v: Float64Array, dx: Float64Array, dy: Float64Array}}
 */
export function p2Basis(x, y) {
  const L = [1 - x - y, x, y];
  const Lx = [-1, 1, 0], Ly = [-1, 0, 1];
  const v = new Float64Array(6), dx = new Float64Array(6), dy = new Float64Array(6);
  for (let i = 0; i < 3; i++) {
    v[i] = L[i] * (2 * L[i] - 1);
    dx[i] = (4 * L[i] - 1) * Lx[i];
    dy[i] = (4 * L[i] - 1) * Ly[i];
  }
  for (let k = 0; k < 3; k++) {
    const [a, b] = EDGE_VERTS[k];
    v[3 + k] = 4 * L[a] * L[b];
    dx[3 + k] = 4 * (Lx[a] * L[b] + L[a] * Lx[b]);
    dy[3 + k] = 4 * (Ly[a] * L[b] + L[a] * Ly[b]);
  }
  return { v, dx, dy };
}

/** Reference coordinates of the P2 nodes (vertices, then edge midpoints). */
export const P2_NODES = [[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]];

/**
 * Ordering of Dubiner modes: list of (i, j) with i + j ≤ p, ordered by total
 * degree so that the first dimP(q) modes span P_q for every q ≤ p.
 * @param {number} p
 * @returns {Array<[number, number]>}
 */
export function dubinerModes(p) {
  const modes = [];
  for (let d = 0; d <= p; d++) for (let i = d; i >= 0; i--) modes.push([i, d - i]);
  return modes;
}

/**
 * Orthonormal Dubiner basis of degree ≤ p on T̂ (unit right triangle).
 * Internally maps to Hesthaven's triangle (r,s) ∈ {(-1,-1),(1,-1),(-1,1)} by
 * r = 2x − 1, s = 2y − 1 (area ratio 4 ⇒ factor 2 to stay orthonormal), and
 * to collapsed coordinates a = 2(1+r)/(1−s) − 1, b = s.
 * @param {number} p degree
 * @param {number} x
 * @param {number} y
 * @param {{v?: Float64Array, dx?: Float64Array, dy?: Float64Array}} [out]
 * @returns {{v: Float64Array, dx: Float64Array, dy: Float64Array}}
 */
export function dubinerBasis(p, x, y, out) {
  const N = dimP(p);
  out = out || {};
  const v = out.v || new Float64Array(N), dx = out.dx || new Float64Array(N), dy = out.dy || new Float64Array(N);
  const r = 2 * x - 1, s = 2 * y - 1;
  const a = Math.abs(s - 1) < 1e-14 ? -1 : (2 * (1 + r)) / (1 - s) - 1;
  const b = s;
  const modes = dubinerModes(p);
  for (let m = 0; m < N; m++) {
    const [i, j] = modes[m];
    const fa = jacobiP(a, 0, 0, i), dfa = jacobiDP(a, 0, 0, i);
    const gb = jacobiP(b, 2 * i + 1, 0, j), dgb = jacobiDP(b, 2 * i + 1, 0, j);
    const omb = 0.5 * (1 - b);
    // value (Hesthaven Simplex2DP), times 2 for the area ratio
    v[m] = 2 * Math.SQRT2 * fa * gb * Math.pow(1 - b, i);
    // gradient w.r.t. (r,s) (Hesthaven GradSimplex2DP)
    let dr = dfa * gb;
    if (i > 0) dr *= Math.pow(omb, i - 1);
    let ds = dfa * (gb * (0.5 * (1 + a)));
    if (i > 0) ds *= Math.pow(omb, i - 1);
    let tmp = dgb * Math.pow(omb, i);
    if (i > 0) tmp -= 0.5 * i * gb * Math.pow(omb, i - 1);
    ds += fa * tmp;
    const scale = Math.pow(2, i + 0.5);
    dr *= scale; ds *= scale;
    // chain rule: ∂/∂x = 2 ∂/∂r, ∂/∂y = 2 ∂/∂s; times 2 for normalisation
    dx[m] = 4 * dr;
    dy[m] = 4 * ds;
  }
  return { v, dx, dy };
}
