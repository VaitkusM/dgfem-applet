/**
 * @file Spectra of linear semi-discrete operators and the resulting maximal
 *       stable CFL numbers of explicit Runge–Kutta methods.
 *
 * For the linear advection equation u_t + a u_x = 0 (a = 1) with periodic
 * boundary conditions, a DG (or FR) discretisation is a linear ODE system
 * du/dt = L u. Its eigenvalues λ ("spectrum") decide explicit stability: one
 * RK step multiplies an eigen-component by R(λΔt), where R is the method's
 * stability function, so we need |R(λΔt)| ≤ 1 for every λ.
 *
 * Two ways to get the spectrum:
 *  1. operatorMatrix: apply the right-hand side to every unit vector
 *     (column j of L = L e_j) and compute all eigenvalues of the n×n matrix.
 *  2. Bloch (Fourier) analysis: on a uniform periodic mesh every element sees
 *     the same coupling blocks A_{−1}, A_0, A_{+1} to its left neighbour, itself
 *     and its right neighbour. A discrete Fourier mode u_k = û e^{ikθ} turns the
 *     system into the small (p+1)×(p+1) complex matrix
 *          S(θ) = A_{−1} e^{−iθ} + A_0 + A_{+1} e^{iθ},   θ ∈ [0, 2π),
 *     whose eigenvalues, for θ = 2πm/N, are exactly the eigenvalues of the full
 *     matrix with N elements. Sampling θ densely gives the N → ∞ limit.
 *
 * CFL number convention: ν = |a| Δt / h, h = element width (NOT h/(p+1)).
 * Eigenvalues are computed for a = 1, h = 1, so λΔt = ν λ.
 */
import { eigGeneral } from '../la/dense.js';
import { stabilityAmp } from '../time/rk.js';
import { makeDG1D } from './dg1d.js';
import { advection } from '../models/scalar.js';

/**
 * Dense matrix of a linear right-hand side: L[i*n+j] = (rhs(e_j))_i.
 * @param {(u: Float64Array, t: number, out: Float64Array) => void} rhs linear in u
 * @param {number} n system size
 * @returns {Float64Array} row-major n×n
 */
export function operatorMatrix(rhs, n) {
  const L = new Float64Array(n * n), e = new Float64Array(n), col = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    e.fill(0); e[j] = 1;
    rhs(e, 0, col);
    for (let i = 0; i < n; i++) L[i * n + j] = col[i];
  }
  return L;
}

/**
 * All eigenvalues of a linear right-hand side.
 * @param {(u: Float64Array, t: number, out: Float64Array) => void} rhs
 * @param {number} n
 * @returns {{re: Float64Array, im: Float64Array}}
 */
export function operatorEigs(rhs, n) {
  return eigGeneral(operatorMatrix(rhs, n), n);
}

/**
 * Coupling blocks of a periodic element-local scheme, extracted from a
 * 3-element periodic discretisation (the middle element's block row).
 * @param {(N: number) => {rhs: Function, n: number}} make factory: N elements → linear discretisation
 *   whose unknowns are stored element-major with np per element
 * @param {number} np unknowns per element
 * @returns {{np: number, Am: Float64Array, A0: Float64Array, Ap: Float64Array}} np×np row-major blocks
 */
export function blochBlocks(make, np) {
  const d = make(3), n = 3 * np, L = operatorMatrix(d.rhs, n);
  const blk = (c) => {
    const B = new Float64Array(np * np);
    for (let i = 0; i < np; i++) for (let j = 0; j < np; j++) B[i * np + j] = L[(np + i) * n + c * np + j];
    return B;
  };
  return { np, Am: blk(0), A0: blk(1), Ap: blk(2) };
}

/**
 * Eigenvalues of the Bloch symbol S(θ) for nTheta equispaced θ in [0, 2π).
 * The complex np×np matrix S = X + iY is embedded as the real 2np×2np matrix
 * [[X, −Y], [Y, X]], whose eigenvalues are those of S together with those of
 * conj(S) = S(−θ); both belong to the spectrum, so the union is the same set.
 * @param {{np: number, Am: Float64Array, A0: Float64Array, Ap: Float64Array}} B
 * @param {number} [nTheta=256]
 * @returns {{re: Float64Array, im: Float64Array}}
 */
export function blochEigs(B, nTheta = 256) {
  const np = B.np, m = 2 * np, re = [], im = [];
  const E = new Float64Array(m * m);
  for (let t = 0; t < nTheta; t++) {
    const th = 2 * Math.PI * t / nTheta, c = Math.cos(th), s = Math.sin(th);
    for (let i = 0; i < np; i++) for (let j = 0; j < np; j++) {
      const k = i * np + j;
      // e^{−iθ} Am + A0 + e^{iθ} Ap
      const X = B.A0[k] + c * (B.Am[k] + B.Ap[k]);
      const Y = s * (B.Ap[k] - B.Am[k]);
      E[i * m + j] = X; E[i * m + j + np] = -Y;
      E[(i + np) * m + j] = Y; E[(i + np) * m + j + np] = X;
    }
    const ev = eigGeneral(E, m);
    for (let k = 0; k < m; k++) { re.push(ev.re[k]); im.push(ev.im[k]); }
  }
  return { re: Float64Array.from(re), im: Float64Array.from(im) };
}

/**
 * Is ν·λ inside the stability region for every eigenvalue?
 * @param {{re: ArrayLike<number>, im: ArrayLike<number>}} ev
 * @param {number} nu
 * @param {string} method RK method name (see rk.js)
 * @param {number} [tol=1e-9] allowed |R| − 1 (round-off; neutral modes have |R| = 1 − O(ν⁴))
 */
export function isStable(ev, nu, method, tol = 1e-9) {
  for (let k = 0; k < ev.re.length; k++) if (stabilityAmp(method, nu * ev.re[k], nu * ev.im[k]) > 1 + tol) return false;
  return true;
}

/**
 * Largest ν such that all ν λ lie in the stability region of `method`.
 * Scans ν upward in steps of `dnu` until the first unstable value, then bisects.
 * (Assumes the stable set of ν is an interval [0, ν_max], true for the
 * spectra considered here; returns 0 if even ν = dnu/1000 is unstable.)
 * @param {{re: ArrayLike<number>, im: ArrayLike<number>}} ev eigenvalues for a = 1, h = 1
 * @param {string} method
 * @param {{dnu?: number, numax?: number, tol?: number}} [o]
 * @returns {number} ν_max (|a| Δt / h)
 */
export function maxStableCFL(ev, method, o = {}) {
  const dnu = o.dnu ?? 0.02, numax = o.numax ?? 4, tol = o.tol ?? 1e-9;
  if (!isStable(ev, dnu / 1000, method, tol)) return 0;
  let lo = dnu / 1000, hi = null;
  for (let nu = dnu; nu <= numax; nu += dnu) {
    if (isStable(ev, nu, method, tol)) lo = nu; else { hi = nu; break; }
  }
  if (hi === null) return lo;
  for (let it = 0; it < 40; it++) {
    const mid = 0.5 * (lo + hi);
    if (isStable(ev, mid, method, tol)) lo = mid; else hi = mid;
  }
  return lo;
}

/**
 * Spectral radius max |λ| (useful for the Δt ∝ 1/|λ|_max heuristic).
 * @param {{re: ArrayLike<number>, im: ArrayLike<number>}} ev
 */
export function spectralRadius(ev) {
  let r = 0;
  for (let k = 0; k < ev.re.length; k++) r = Math.max(r, Math.hypot(ev.re[k], ev.im[k]));
  return r;
}

/**
 * Safety factor the widgets apply to the linear stability limit: Δt = CFL_SAFETY · ν_max · h / max|f'|.
 * (ν_max is computed for linear advection; 20 % head-room covers nonlinear fluxes and the
 * finite sampling of the Bloch spectrum.)
 */
export const CFL_SAFETY = 0.8;

const cflCache = new Map();
/**
 * Largest stable CFL number ν_max = |a| Δt / h of the 1D DG scheme of dg1d.js (GLL nodes) for
 * periodic linear advection, from the Bloch spectrum (N → ∞). Results are cached.
 * This is the number a time step must respect; the popular rule of thumb 1/(2p+1) is only an
 * approximation of it (e.g. SSP-RK3, upwind, exact mass: ν_max (2p+1) = 1.23, 1.05, 0.91, 0.81, 0.73
 * for p = 1..5, so "0.8/(2p+1)" is UNSTABLE for p = 5).
 * @param {number} p degree ≥ 0
 * @param {{flux?: 'upwind'|'central', quad?: 'exact'|'collocated', method?: string, nTheta?: number}} [o]
 *   flux: 'upwind' (also valid for Rusanov / Godunov, which coincide with it for advection) or 'central';
 *   quad: mass matrix / quadrature variant of makeDG1D; method: RK method (default 'ssprk3').
 * @returns {number} ν_max (0 if the method is unstable for every Δt, e.g. forward Euler with p ≥ 1)
 */
export function dgStableCFL(p, o = {}) {
  const flux = o.flux === 'central' ? 'central' : 'upwind', quad = o.quad || 'exact';
  const method = o.method || 'ssprk3', nTheta = o.nTheta ?? 128;
  const key = `${p}|${flux}|${quad}|${method}|${nTheta}`;
  if (!cflCache.has(key)) {
    const B = blochBlocks((N) => makeDG1D({ p, N, a: 0, b: N, model: advection(1), flux, nodes: 'gll', quad }), p + 1);
    cflCache.set(key, maxStableCFL(blochEigs(B, nTheta), method, { dnu: 0.005 }));
  }
  return cflCache.get(key);
}

/**
 * Does the stability region of an RK method contain an interval [−iy, iy], y > 0, of the imaginary
 * axis? (True for SSP-RK3, RK4, LSRK4; false for forward Euler and SSP-RK2, for which
 * |R(iy)| > 1 for every y ≠ 0.) Purely imaginary spectra (central flux) are stable only if it does.
 * @param {string} method RK method name (see rk.js)
 * @returns {boolean}
 */
export function containsImaginaryAxis(method) {
  return stabilityAmp(method, 0, 0.05) <= 1;
}
