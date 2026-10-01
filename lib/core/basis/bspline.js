/**
 * @file Uniform (cardinal) B-splines and their tensor products on a grid.
 *
 * The cardinal B-spline N_n of degree n is the n-fold convolution of the box
 * function χ_[0,1) with itself — the same kernel as an (n+1)-tap box blur
 * repeated, or the cubic B-spline filter of image resampling (n = 3). It is
 *   - a piecewise polynomial of degree n on the unit intervals [m, m+1],
 *   - C^{n−1} smooth, non-negative, supported on [0, n+1],
 *   - and its integer shifts form a partition of unity: Σ_k N_n(t − k) = 1.
 * Cox–de Boor recursion (uniform knots):
 *   N_0(t) = 1 on [0,1), 0 otherwise,
 *   N_n(t) = ( t N_{n−1}(t) + (n + 1 − t) N_{n−1}(t − 1) ) / n,
 *   N_n'(t) = N_{n−1}(t) − N_{n−1}(t − 1).
 *
 * Grid B-splines: on a grid of width h with origin (x0, y0) and index-space
 * coordinates ξ = (x − x0)/h, η = (y − y0)/h,
 *   b_k(x, y) = N_n(ξ − k1) N_n(η − k2),  k = (k1, k2),
 * so b_k is supported on the (n+1)×(n+1) grid cells with lower-left indices
 * k1 … k1+n, k2 … k2+n. On the cell [l, l+1] the non-zero 1D B-splines are
 * those with k = l − m, m = 0 … n, and their values are N_n(s + m) with the
 * cell-local coordinate s = ξ − l ∈ [0, 1).
 *
 * Marsden's identity (uniform knots t_k = k): for every τ,
 *   (ξ − τ)^n = Σ_k ψ_k(τ) N_n(ξ − k),   ψ_k(τ) = Π_{μ=1}^{n} (k + μ − τ).
 * Hence the B-spline coefficients of any polynomial of degree ≤ n are
 * POLYNOMIALS of degree ≤ n in the index k — the fact behind WEB-spline
 * extension (lib/core/immersed/webspline.js).
 */

/**
 * Cardinal B-spline N_n(t) (support [0, n+1]).
 * @param {number} n degree ≥ 0
 * @param {number} t
 * @returns {number}
 */
export function cardinalBspline(n, t) {
  if (t < 0 || t >= n + 1) return 0;
  if (n === 0) return 1;
  return (t * cardinalBspline(n - 1, t) + (n + 1 - t) * cardinalBspline(n - 1, t - 1)) / n;
}

/**
 * Derivative N_n'(t) = N_{n−1}(t) − N_{n−1}(t − 1) (n ≥ 1; 0 for n = 0).
 * @param {number} n
 * @param {number} t
 */
export function cardinalBsplineDeriv(n, t) {
  if (n === 0) return 0;
  return cardinalBspline(n - 1, t) - cardinalBspline(n - 1, t - 1);
}

/**
 * Values and derivatives of the n+1 B-splines that are non-zero on a cell, at
 * cell-local coordinate s ∈ [0,1): vals[m] = N_n(s + m), ders[m] = N_n'(s + m),
 * belonging to the B-spline with index k = l − m on cell l.
 * @param {number} n
 * @param {number} s
 * @param {Float64Array} vals length n+1 (output)
 * @param {Float64Array} ders length n+1 (output)
 */
export function bsplineCellValues(n, s, vals, ders) {
  for (let m = 0; m <= n; m++) { vals[m] = cardinalBspline(n, s + m); ders[m] = cardinalBsplineDeriv(n, s + m); }
}

/**
 * Marsden coefficient ψ_k(τ) = Π_{μ=1}^{n} (k + μ − τ): the B-spline
 * coefficient of (ξ − τ)^n (index-space variable ξ).
 * @param {number} n
 * @param {number} k
 * @param {number} tau
 */
export function marsden(n, k, tau) {
  let p = 1;
  for (let mu = 1; mu <= n; mu++) p *= k + mu - tau;
  return p;
}

/**
 * Tensor-product B-spline grid of degree n with N×N cells of width h on
 * [x0, x0 + N h] × [y0, y0 + N h]. Indices k_ν ∈ {−n, …, N−1} (all B-splines
 * whose support meets the box); linear index (k1 + n) + M (k2 + n), M = N + n.
 * @param {number} n
 * @param {number} N
 * @param {number[]} [box=[0,1,0,1]] square box
 * @returns {{n: number, N: number, h: number, x0: number, y0: number, M: number, nB: number,
 *            index: (k1:number,k2:number)=>number, kOf: (idx:number)=>[number,number]}}
 */
export function bsplineGrid(n, N, box = [0, 1, 0, 1]) {
  const h = (box[1] - box[0]) / N, M = N + n;
  return {
    n, N, h, x0: box[0], y0: box[2], M, nB: M * M,
    index: (k1, k2) => (k1 + n) + M * (k2 + n),
    kOf: (idx) => [(idx % M) - n, Math.floor(idx / M) - n],
  };
}
