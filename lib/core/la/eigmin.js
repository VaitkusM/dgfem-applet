/**
 * @file Certified smallest eigenvalue of a sparse symmetric (possibly
 *       indefinite) matrix — "is this matrix positive definite, and by how much?"
 *
 * Why not just Lanczos? The smallest Ritz value of a Krylov method is only an
 * UPPER bound of λ_min (it is a Rayleigh quotient), and for matrices whose
 * spectrum spans many orders of magnitude (an under-penalised DG matrix has
 * λ_max ~ 10⁴ and a slightly negative λ_min) it converges very slowly. The
 * missing ingredient is a LOWER bound, and Sylvester's law of inertia gives
 * one cheaply: the number of negative pivots of the factorisation
 *     A − s I = L D Lᵀ
 * equals the number of eigenvalues of A below the shift s ("spectrum
 * slicing"). After a reverse Cuthill–McKee renumbering the factorisation only
 * touches the band, so one count costs O(n · bandwidth²).
 *
 * Algorithm of smallestEigenvalueSym:
 *  1. a short Lanczos run on A gives an upper bound r₀ ≥ λ_min;
 *  2. counts locate a shift s₀ ≤ λ_min with no eigenvalue below it, not much
 *     further away from λ_min than λ_min is from the rest of the spectrum
 *     (s₀ = 0 for a positive definite matrix);
 *  3. Lanczos on (A − s₀ I)⁻¹ ("shift-invert"): its largest eigenvalue is
 *     1/(λ_min − s₀), well separated, so it converges in a few steps;
 *  4. two more counts certify the result: no eigenvalue below λ(1 ∓ tol), at
 *     least one below λ(1 ± tol). If the certificate fails, bisection on the
 *     count finishes the job.
 */
import { csrPermute, csrBandwidth, SparseBuilder } from './sparse.js';
import { rcm, sparseLU } from './direct.js';
import { lanczosRitz } from './eig.js';
import { asOperator } from './krylov.js';

/**
 * Prepare a symmetric sparse matrix for repeated eigenvalue counts: RCM
 * renumbering and extraction of the lower band.
 * @param {import('./sparse.js').CSR} A symmetric n×n
 * @returns {{n: number, kl: number, band: Float64Array, norm: number}}
 *   band[i*(kl+1) + (j − i + kl)] = entry (i,j) of the permuted matrix for i − kl ≤ j ≤ i;
 *   norm = ‖A‖_∞ (largest absolute row sum; every eigenvalue satisfies |λ| ≤ norm).
 */
export function bandSym(A) {
  const n = A.n, B = csrPermute(A, rcm(A)), { kl, ku } = csrBandwidth(B), k = Math.max(kl, ku), W = k + 1;
  const band = new Float64Array(n * W);
  let norm = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let q = B.rowPtr[i]; q < B.rowPtr[i + 1]; q++) {
      const j = B.colIdx[q], v = B.vals[q];
      s += Math.abs(v);
      if (j <= i) band[i * W + (j - i + k)] += v;
    }
    norm = Math.max(norm, s);
  }
  return { n, kl: k, band, norm };
}

/**
 * Number of eigenvalues of A that are smaller than the shift s, as the number
 * of negative pivots of the banded factorisation A − sI = L D Lᵀ (no pivoting).
 * @param {ReturnType<typeof bandSym>} S prepared matrix
 * @param {number} s shift
 * @returns {{neg: number, ok: boolean}} ok = false if a pivot was numerically zero
 *   (s is then an eigenvalue of a leading block to working precision: move s slightly and repeat)
 */
export function countBelow(S, s) {
  const { n, kl, norm } = S, W = kl + 1, a = Float64Array.from(S.band);
  const tiny = 1e-13 * (norm || 1);
  // row-oriented LDLᵀ: after the loop a[i,j] (j < i) holds L_ij·d_j, a[i,i] holds d_i
  const d = new Float64Array(n);
  let neg = 0, ok = true;
  for (let i = 0; i < n; i++) {
    const j0 = Math.max(0, i - kl), ri = i * W - i + kl; // a[ri + j] = entry (i,j)
    for (let j = j0; j <= i; j++) {
      // v = a_ij − Σ_{m<j} (L_im d_m) L_jm, with L_jm = a[j,m]/d_m
      let v = a[ri + j] - (j === i ? s : 0);
      const rj = j * W - j + kl, m0 = Math.max(j0, j - kl);
      for (let m = m0; m < j; m++) v -= a[ri + m] * a[rj + m] / d[m];
      a[ri + j] = v;
    }
    let piv = a[ri + i];
    if (Math.abs(piv) <= tiny) { ok = false; piv = tiny; }
    if (piv < 0) neg++;
    d[i] = piv;
  }
  return { neg, ok };
}

/**
 * Smallest eigenvalue of a sparse symmetric matrix with a certificate.
 * @param {import('./sparse.js').CSR} A symmetric n×n (positive definite or indefinite)
 * @param {{k?: number, tol?: number, maxBisect?: number}} [o] Krylov dimension of the Lanczos runs (default 60),
 *   relative tolerance tol of the certificate (default 1e-6) and the maximal number of bisection steps of the
 *   fallback (default 80, always enough in double precision; a small value makes the routine give up early and
 *   return certified = false together with the bounds it has proved so far)
 * @returns {{value: number, lower: number, upper: number, certified: boolean, counts: number}}
 *   value ≈ λ_min; lower ≤ λ_min ≤ upper are the bounds established by the eigenvalue counts, with
 *   upper − lower ≤ 2·max(tol·|λ_min|, 1e-11·‖A‖_∞) when certified (the second term is the resolution
 *   of a count in floating-point arithmetic); counts = number of factorisations used.
 */
export function smallestEigenvalueSym(A, o = {}) {
  const k = o.k ?? 60, tol = o.tol ?? 1e-6, maxBisect = o.maxBisect ?? 80, n = A.n;
  const S = bandSym(A), nrm = S.norm || 1;
  // absolute resolution of an eigenvalue count in floating point (pivots below 1e-13·‖A‖ are "zero")
  const res = 1e-11 * nrm;
  let counts = 0;
  /** true if no eigenvalue lies below s (a numerically zero pivot is resolved by nudging s down) */
  const noneBelow = (s) => {
    let r = countBelow(S, s); counts++;
    for (let t = 1; !r.ok && t <= 3; t++) { r = countBelow(S, s - t * 4e-13 * nrm); counts++; }
    return r.neg === 0;
  };
  // 1. upper bound from Lanczos on A (a Ritz value is a Rayleigh quotient, so r0 ≥ λ_min)
  const r0 = lanczosRitz(asOperator(A), n, Math.min(k, 40))[0];
  // 2. a shift lo ≤ λ_min: 0 if A is positive definite, otherwise search downwards from r0
  let lo, hi = r0;
  if (r0 > 0 && noneBelow(0)) lo = 0;
  else {
    hi = Math.min(r0, 0);
    let g = Math.max(Math.abs(hi), 1e-6 * nrm);
    lo = hi - g;
    while (!noneBelow(lo) && lo > -2 * S.norm - 1) { hi = lo; g *= 2; lo -= g; }
    // tighten until the distance to the shift is comparable with |λ_min| (good shift-invert separation)
    for (let it = 0; it < 3 && hi - lo > 0.5 * Math.max(Math.abs(hi), res); it++) {
      const mid = 0.5 * (lo + hi);
      if (noneBelow(mid)) lo = mid; else hi = mid;
    }
  }
  // 3. shift-invert Lanczos at lo: the eigenvalue of A closest to lo (from above) is λ_min
  const shifted = new SparseBuilder(n, n, A.vals.length + n);
  for (let i = 0; i < n; i++) {
    shifted.add(i, i, -lo);
    for (let q = A.rowPtr[i]; q < A.rowPtr[i + 1]; q++) shifted.add(i, A.colIdx[q], A.vals[q]);
  }
  const F = sparseLU(shifted.toCSR());
  const mu = lanczosRitz((x, y) => y.set(F.solve(x)), n, k);
  let value = lo + 1 / mu[mu.length - 1];
  if (!Number.isFinite(value) || value < lo) value = hi;
  // 4. certificate: nothing below value − δ, something below value + δ
  const delta = () => Math.max(tol * Math.abs(value), res);
  let certified = noneBelow(value - delta()) && !noneBelow(value + delta());
  if (certified) { lo = value - delta(); hi = value + delta(); } else {
    // fall back to bisection on the count (lo: none below; hi: at least one below, or the Lanczos bound)
    if (noneBelow(value - delta())) lo = Math.max(lo, value - delta()); else hi = Math.min(hi, value - delta());
    const width = () => 2 * Math.max(tol * Math.max(Math.abs(lo), Math.abs(hi)), res);
    for (let it = 0; it < maxBisect && hi - lo > width(); it++) {
      const mid = 0.5 * (lo + hi);
      if (noneBelow(mid)) lo = mid; else hi = mid;
    }
    value = 0.5 * (lo + hi);
    certified = hi - lo <= width() * (1 + 1e-9);
  }
  return { value, lower: lo, upper: hi, certified, counts };
}
