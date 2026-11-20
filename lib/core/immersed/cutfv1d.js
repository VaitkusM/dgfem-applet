/**
 * @file The small-cell problem of explicit cut-cell finite volume methods, in
 *       its simplest form: linear advection in 1D on a grid with ONE small cell.
 *
 * Model problem (the 1D model of Berger & Giuliani, J. Comput. Phys. 428 (2021),
 * Section 2; here with periodic boundaries so that mass is conserved exactly):
 *
 *     u_t + a u_x = 0,  a > 0,  on the periodic interval [0, 1),
 *
 * discretised on N "full" cells of width h and one "cut" cell of width α h,
 * 0 < α ≤ 1 (so h = 1/(N + α)). α plays the role of the volume fraction α_K of
 * a cut cell in 2D (chapter 11). The base scheme is first-order upwind,
 *
 *     Û_i = U_i − (a Δt / h_i) (U_i − U_{i−1}),          (flux a·U_{i−1} in, a·U_i out)
 *
 * with ONE time step Δt for all cells. With the full-cell CFL number
 * ν = a Δt / h the small cell sees the local CFL number ν/α:
 *
 *     Û_s = (1 − ν/α) U_s + (ν/α) U_{s−1}.
 *
 *  - ν ≤ α        : every update is a convex combination ⇒ monotone
 *                   (discrete maximum principle), as on a uniform grid;
 *  - α < ν < 2α   : |1 − ν/α| < 1: the small cell over-/undershoots but stays bounded;
 *  - ν > 2α       : the small-cell value is multiplied by 1 − ν/α < −1 every
 *                   step: exponential blow-up (which then spreads downstream).
 *                   (ν = 2α is the marginal case, factor −1: for ν < 1 the small cell
 *                   oscillates without growing or decaying; see smallCellFactor.)
 *
 * Two cures that keep the full-cell time step (ν ≤ 1) are implemented as
 * POSTPROCESSING steps applied to the provisional update Û:
 *
 *  - 'merge' (cell merging): the small cell and its left neighbour are replaced
 *    by their volume-weighted mean. This is identical to running upwind on a grid
 *    in which the two cells are one cell of width (1 + α) h, hence monotone for
 *    ν ≤ 1 (the limit of the remaining full cells).
 *
 *  - 'srd' (state redistribution, Berger & Giuliani 2021, piecewise-constant
 *    version): every cell i has a merging NEIGHBOURHOOD M_i — itself if it is
 *    large enough (h_i ≥ threshold·h, threshold = ½), otherwise itself plus its
 *    left and right neighbours (widened until the neighbourhood is at least
 *    threshold·h long). Neighbourhoods may overlap; N_j = number of
 *    neighbourhoods containing cell j. Then
 *        Q̂_i = ( Σ_{j ∈ M_i} (h_j / N_j) Û_j ) / ( Σ_{j ∈ M_i} h_j / N_j ),
 *        U_j^{n+1} = (1 / N_j) Σ_{i : j ∈ M_i} Q̂_i .
 *    This is conservative (Σ_j h_j U_j^{n+1} = Σ_j h_j Û_j) and, in our tests,
 *    bounded for ν < 1 and every α, but — as Berger & Giuliani point out — not
 *    monotone: some weights are negative for small α.
 *
 * Sign/array conventions: cells are numbered left to right, cell i = [x[i], x[i+1]],
 * widths w[i] = x[i+1] − x[i]; all cell data are Float64Array of length n = N + 1;
 * index arithmetic is periodic.
 */
import { gaussForDegree01 } from '../quad/gauss1d.js';

/**
 * @typedef {Object} CutGrid1D
 * @property {number} N       number of full cells
 * @property {number} n       number of cells (N + 1)
 * @property {number} alpha   width of the small cell in units of h (0 < α ≤ 1)
 * @property {number} small   index of the small cell
 * @property {number} h       full cell width 1/(N + α)
 * @property {Float64Array} w cell widths (length n)
 * @property {Float64Array} x cell edges (length n + 1), x[0] = 0, x[n] = 1
 */

/**
 * Periodic grid on [0, 1) with N full cells and one small cell of width α h.
 * @param {number} N number of full cells (≥ 3)
 * @param {number} alpha volume fraction of the small cell, 0 < α ≤ 1
 * @param {number} [small] index of the small cell (default ⌊N/2⌋)
 * @returns {CutGrid1D}
 */
export function cutGrid1D(N, alpha, small = Math.floor(N / 2)) {
  if (!(alpha > 0 && alpha <= 1)) throw new Error('cutGrid1D: need 0 < α ≤ 1');
  if (!(N >= 3) || small < 0 || small > N) throw new Error('cutGrid1D: need N ≥ 3 and 0 ≤ small ≤ N');
  const n = N + 1, h = 1 / (N + alpha);
  const w = new Float64Array(n).fill(h), x = new Float64Array(n + 1);
  w[small] = alpha * h;
  for (let i = 0; i < n; i++) x[i + 1] = x[i] + w[i];
  x[n] = 1;
  return { N, n, alpha, small, h, w, x };
}

/**
 * Exact cell averages of the 1-periodic function f(x − shift) (4-point Gauss
 * rule on 8 sub-intervals per cell; exact for the travelling exact solution
 * u(x, t) = u₀(x − a t) up to quadrature error).
 * @param {CutGrid1D} grid
 * @param {(x: number) => number} f 1-periodic function
 * @param {number} [shift=0]
 * @returns {Float64Array} length grid.n
 */
export function cellAverages(grid, f, shift = 0) {
  const G = gaussForDegree01(7), U = new Float64Array(grid.n), S = 8;
  for (let i = 0; i < grid.n; i++) {
    let s = 0;
    for (let k = 0; k < S; k++) for (let q = 0; q < G.x.length; q++) {
      const xq = grid.x[i] + grid.w[i] * (k + G.x[q]) / S - shift;
      s += G.w[q] * f(xq - Math.floor(xq));
    }
    U[i] = s / S;
  }
  return U;
}

/**
 * Total mass Σ_i h_i U_i.
 * @param {CutGrid1D} grid
 * @param {ArrayLike<number>} U
 * @returns {number}
 */
export function mass(grid, U) {
  let m = 0;
  for (let i = 0; i < grid.n; i++) m += grid.w[i] * U[i];
  return m;
}

/**
 * Provisional first-order upwind update with one time step for all cells:
 * Û_i = U_i − (ν h / h_i)(U_i − U_{i−1}), periodic.
 * @param {CutGrid1D} grid
 * @param {Float64Array} U cell averages (length n)
 * @param {number} nu full-cell CFL number ν = a Δt / h
 * @param {Float64Array} [out] output (must not alias U)
 * @returns {Float64Array} Û
 */
export function upwindStep(grid, U, nu, out = new Float64Array(grid.n)) {
  const { n, w, h } = grid;
  for (let i = 0; i < n; i++) out[i] = U[i] - (nu * h / w[i]) * (U[i] - U[(i + n - 1) % n]);
  return out;
}

/**
 * @typedef {Object} CutFvScheme
 * @property {CutGrid1D} grid
 * @property {'none'|'merge'|'srd'} method
 * @property {Int32Array} nbrPtr  CSR-like pointers into nbrIdx: neighbourhood M_i = nbrIdx[nbrPtr[i] … nbrPtr[i+1])
 * @property {Int32Array} nbrIdx  cell indices of the merging neighbourhoods
 * @property {Int32Array} count   N_j = number of neighbourhoods containing cell j
 */

/**
 * Precompute the merging neighbourhoods of a stabilisation method.
 *  - 'none' : M_i = {i} for all cells (the postprocessing is the identity);
 *  - 'merge': the small cell s and its left neighbour both get M = {s − 1, s}
 *             (a non-overlapping merged cell), if h_s < threshold·h;
 *  - 'srd'  : a small cell gets M_s = {s − k, …, s + k} with the smallest k ≥ 1
 *             for which the neighbourhood is at least threshold·h long; all
 *             other cells keep M_i = {i} (overlapping neighbourhoods).
 * @param {CutGrid1D} grid
 * @param {'none'|'merge'|'srd'} method
 * @param {{threshold?: number}} [o] cells with h_i < threshold·h are "small". Defaults: 1 for 'merge' (every
 *   cell narrower than a full cell is merged, which makes the scheme monotone for all ν ≤ 1) and ½ for 'srd'
 *   (the choice of Berger & Giuliani; cells with ½ ≤ α < 1 are left alone and are bounded only for ν < 2α).
 * @returns {CutFvScheme}
 */
export function cutFvScheme(grid, method, o = {}) {
  if (method !== 'none' && method !== 'merge' && method !== 'srd') throw new Error(`cutFvScheme: unknown method ${method}`);
  const { n, w, h } = grid, thr = (o.threshold ?? (method === 'merge' ? 1 : 0.5)) * (1 - 1e-12) * h;
  const lists = [];
  for (let i = 0; i < n; i++) lists.push([i]);
  for (let i = 0; i < n; i++) {
    if (method === 'none' || w[i] >= thr) continue;
    if (method === 'merge') {
      const l = (i + n - 1) % n;
      lists[i] = [l, i]; lists[l] = [l, i];
    } else {
      let k = 0, len = w[i];
      const M = [i];
      while (len < thr && 2 * k + 3 <= n) {
        k++;
        const l = (i - k + n * k) % n, r = (i + k) % n;
        M.unshift(l); M.push(r); len += w[l] + w[r];
      }
      lists[i] = M;
    }
  }
  const nbrPtr = new Int32Array(n + 1), count = new Int32Array(n);
  for (let i = 0; i < n; i++) nbrPtr[i + 1] = nbrPtr[i] + lists[i].length;
  const nbrIdx = new Int32Array(nbrPtr[n]);
  for (let i = 0; i < n; i++) lists[i].forEach((j, k) => { nbrIdx[nbrPtr[i] + k] = j; });
  if (method === 'merge') count.fill(1); // merged cells are disjoint: each cell is in exactly one merged cell
  else for (let k = 0; k < nbrIdx.length; k++) count[nbrIdx[k]]++;
  return { grid, method, nbrPtr, nbrIdx, count };
}

/**
 * One time step: provisional upwind update with the full-cell CFL number ν,
 * followed by the scheme's conservative postprocessing (see the file header).
 * @param {CutFvScheme} sch
 * @param {Float64Array} U cell averages at tⁿ (length n; not modified)
 * @param {number} nu full-cell CFL number ν = a Δt / h
 * @returns {Float64Array} cell averages at tⁿ⁺¹
 */
export function cutFvStep(sch, U, nu) {
  const { grid, method, nbrPtr, nbrIdx, count } = sch, { n, w } = grid;
  const Uh = upwindStep(grid, U, nu);
  if (method === 'none') return Uh;
  // weighted neighbourhood averages Q̂_i
  const Q = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let m = 0, v = 0;
    for (let k = nbrPtr[i]; k < nbrPtr[i + 1]; k++) { const j = nbrIdx[k], c = w[j] / count[j]; m += c * Uh[j]; v += c; }
    Q[i] = m / v;
  }
  if (method === 'merge') return Q; // both members of a merged cell carry the merged average
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) for (let k = nbrPtr[i]; k < nbrPtr[i + 1]; k++) { const j = nbrIdx[k]; out[j] += Q[i] / count[j]; }
  return out;
}

/**
 * Amplification factor g = 1 − ν/α of the small-cell value in the unstabilised
 * scheme (ν ≤ 1 assumed, so the full cells are fine), and the regime it implies:
 *   'monotone' : ν ≤ α        (g ≥ 0: all weights non-negative, discrete maximum principle)
 *   'bounded'  : α < ν < 2α   (−1 < g < 0: over-/undershoots that die out)
 *   'marginal' : ν = 2α       (g = −1: an oscillation in the small cell that neither grows nor decays)
 *   'unstable' : ν > 2α       (g < −1: exponential growth by |g| per step)
 * Equality is decided with a relative tolerance of 1e−9 on g, so that slider values such
 * as α = 10^{−1}, ν = 0.2 (g = −1 up to rounding) are classified as marginal, not unstable.
 * @param {number} alpha volume fraction of the small cell
 * @param {number} nu full-cell CFL number
 * @returns {{factor: number, regime: 'monotone'|'bounded'|'marginal'|'unstable', monotone: boolean, bounded: boolean,
 *            stepsTo1e6: number}}
 *   monotone / bounded: regime === 'monotone' / regime is 'monotone' or 'bounded' (strictly inside the stable range);
 *   stepsTo1e6: number of steps in which |g|ⁿ reaches 10⁶ (Infinity unless unstable) — how long the blow-up takes
 *   once the small cell has been disturbed
 */
export function smallCellFactor(alpha, nu) {
  const factor = 1 - nu / alpha, eps = 1e-9;
  const regime = factor >= -eps ? 'monotone' : factor > -1 + eps ? 'bounded' : factor >= -1 - eps ? 'marginal' : 'unstable';
  return {
    factor, regime, monotone: regime === 'monotone', bounded: regime === 'monotone' || regime === 'bounded',
    stepsTo1e6: regime === 'unstable' ? Math.ceil(Math.log(1e6) / Math.log(-factor)) : Infinity,
  };
}

/**
 * Run `steps` time steps from the cell averages of u₀ and compare with the
 * exact solution u₀(x − ν h · steps) (a = 1, Δt = ν h).
 * @param {{N: number, alpha: number, nu: number, method: 'none'|'merge'|'srd', steps: number,
 *          u0: (x: number) => number, small?: number, threshold?: number}} o
 * @returns {{grid: CutGrid1D, U: Float64Array, exact: Float64Array, t: number, l1: number, max: number,
 *            min: number, massDrift: number}}
 *   l1 = Σ h_i |U_i − exact average|; massDrift = |mass(U) − mass(U⁰)|; max/min over all steps and cells
 */
export function cutFvRun(o) {
  const grid = cutGrid1D(o.N, o.alpha, o.small), sch = cutFvScheme(grid, o.method, { threshold: o.threshold });
  let U = cellAverages(grid, o.u0);
  const m0 = mass(grid, U);
  let max = -Infinity, min = Infinity;
  for (const v of U) { max = Math.max(max, v); min = Math.min(min, v); }
  for (let s = 0; s < o.steps; s++) {
    U = cutFvStep(sch, U, o.nu);
    for (const v of U) { max = Math.max(max, v); min = Math.min(min, v); }
  }
  const t = o.steps * o.nu * grid.h, exact = cellAverages(grid, o.u0, t);
  let l1 = 0;
  for (let i = 0; i < grid.n; i++) l1 += grid.w[i] * Math.abs(U[i] - exact[i]);
  return { grid, U, exact, t, l1, max, min, massDrift: Math.abs(mass(grid, U) - m0) };
}
