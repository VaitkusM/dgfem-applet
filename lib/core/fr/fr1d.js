/**
 * @file Flux reconstruction (FR, Huynh 2007) for scalar conservation laws
 *       u_t + f(u)_x = 0 in 1D.
 *
 * Same unknowns as nodal DG (dg1d.js): u[k*(p+1)+i] = value at solution point
 * ξ_i of element k (Gauss or GLL points). One evaluation of du/dt:
 *
 *  1. Discontinuous flux: F^D_k(ξ) = Σ_i f(u_{k,i}) ℓ_i(ξ), the degree-p interpolant
 *     of the flux values at the solution points (element by element, so F^D jumps
 *     across faces).
 *  2. Interface (common) fluxes: F̂_{k+½} = F̂(u⁻, u⁺) with the extrapolated
 *     solution u⁻ = u_k(1), u⁺ = u_{k+1}(−1) (same numerical fluxes as DG).
 *  3. Corrected flux of degree p+1:
 *        F_k(ξ) = F^D_k(ξ) + (F̂_{k−½} − F^D_k(−1)) g_L(ξ) + (F̂_{k+½} − F^D_k(1)) g_R(ξ),
 *     which takes the common values at both ends (g_L(−1) = 1, g_R(1) = 1, and
 *     the other one vanishes there), so the global flux is continuous.
 *  4. du_{k,i}/dt = −(2/h) F_k'(ξ_i)
 *               = −(2/h) [ (D F^D)_i + (F̂_{k−½} − F^D_k(−1)) g_L'(ξ_i) + (F̂_{k+½} − F^D_k(1)) g_R'(ξ_i) ].
 *
 * No integrals at all: FR is a "differential" (collocation) formulation.
 * With g_L = g_DG = R_{R,p+1} it reproduces nodal DG in strong form with exact
 * mass matrix and collocated flux interpolant (dg1d: form 'strong', quad 'exact',
 * same solution points) — identically, for any flux f.
 *
 * Boundary conditions as in dg1d.js ('periodic' or weak 'inflow').
 */
import { refNodes, DG_FLUXES, makeNodalTools } from '../dg/dg1d.js';
import { diffMatrix, lagrangeValues } from '../basis/lagrange.js';
import { correctionCoeffs, evalCorrection, cHU } from './corrections.js';
import { blochBlocks, blochEigs, maxStableCFL } from '../dg/spectrum.js';
import { advection } from '../models/scalar.js';

/**
 * Create a 1D FR discretisation.
 * @param {{p: number, N: number, a?: number, b?: number, model: object,
 *          flux?: string|Function, points?: 'gauss'|'gll',
 *          correction?: 'dg'|'g2'|'ga'|'vcjh', c?: number,
 *          bc?: 'periodic'|'inflow', bcValue?: (x: number, t: number) => number}} o
 * @returns {object} { p, N, np, n, h, a, b, xf, x, r, D, gLd, gRd (g_L', g_R' at solution points), coef,
 *   rhs, traces, faceFlux, project, interpolate, evalRef, evalAt, l2Error, mass, maxSpeed }
 */
export function makeFR1D(o) {
  const p = o.p, N = o.N, a = o.a ?? 0, b = o.b ?? 1, h = (b - a) / N;
  const np = p + 1, n = N * np, m = o.model;
  const Ffn = typeof o.flux === 'function' ? o.flux : DG_FLUXES[o.flux || 'upwind'].fn;
  const r = refNodes(p, p === 0 ? 'gauss' : (o.points || 'gauss')).x;
  const D = np > 1 ? diffMatrix(r) : new Float64Array(1);
  const lL = lagrangeValues(r, -1), lR = lagrangeValues(r, 1);
  const coef = correctionCoeffs(o.correction || 'dg', p, o.c ?? 0);
  const gLd = new Float64Array(np), gRd = new Float64Array(np);
  for (let i = 0; i < np; i++) { const g = evalCorrection(coef, r[i]); gLd[i] = g.dgL; gRd[i] = g.dgR; }
  const periodic = (o.bc || 'periodic') === 'periodic';
  const bcValue = o.bcValue || (() => 0);

  const xf = new Float64Array(N + 1), x = new Float64Array(n);
  for (let k = 0; k <= N; k++) xf[k] = a + k * h;
  for (let k = 0; k < N; k++) for (let i = 0; i < np; i++) x[k * np + i] = xf[k] + (r[i] + 1) * h / 2;

  const uM = new Float64Array(N + 1), uP = new Float64Array(N + 1), faceFlux = new Float64Array(N + 1);
  const fn = new Float64Array(np);

  function traces(u, t = 0) {
    for (let k = 0; k < N; k++) {
      let sl = 0, sr = 0;
      for (let i = 0; i < np; i++) { sl += lL[i] * u[k * np + i]; sr += lR[i] * u[k * np + i]; }
      uP[k] = sl; uM[k + 1] = sr;
    }
    if (periodic) { uM[0] = uM[N]; uP[N] = uP[0]; }
    else { uM[0] = bcValue(a, t); uP[N] = bcValue(b, t); }
    return { uMinus: uM, uPlus: uP };
  }

  /**
   * du/dt by flux reconstruction.
   * @param {Float64Array} u
   * @param {number} t
   * @param {Float64Array} out
   */
  function rhs(u, t, out) {
    traces(u, t);
    for (let f = 0; f <= N; f++) faceFlux[f] = Ffn(m, uM[f], uP[f]);
    const s = 2 / h;
    for (let k = 0; k < N; k++) {
      const off = k * np;
      let fl = 0, fr = 0;
      for (let j = 0; j < np; j++) { fn[j] = m.f(u[off + j]); fl += lL[j] * fn[j]; fr += lR[j] * fn[j]; }
      const jl = faceFlux[k] - fl, jr = faceFlux[k + 1] - fr; // flux mismatches at the two ends
      for (let i = 0; i < np; i++) {
        let d = 0;
        for (let j = 0; j < np; j++) d += D[i * np + j] * fn[j];
        out[off + i] = -s * (d + jl * gLd[i] + jr * gRd[i]);
      }
    }
  }

  // projection / evaluation / error helpers: the same as for DG (exact L² projection onto P_p)
  const { project, interpolate, evalRef, evalAt, l2Error, mass } = makeNodalTools({ r, N, a, h, xf, x });
  const maxSpeed = (u) => { let s = 0; for (let i = 0; i < n; i++) s = Math.max(s, Math.abs(m.df(u[i]))); return s; };

  return { p, N, np, n, h, a, b, xf, x, r, D, lL, lR, gLd, gRd, coef, periodic, model: m,
    rhs, traces, faceFlux, project, interpolate, evalRef, evalAt, l2Error, mass, maxSpeed };
}

/**
 * Bloch eigenvalues of FR for linear advection (a = 1, h = 1, upwind flux) — the
 * N → ∞ spectrum used for CFL limits. See dg/spectrum.js.
 * @param {number} p
 * @param {{correction?: string, c?: number, points?: string, flux?: string}} [o]
 * @param {number} [nTheta=128]
 * @returns {{re: Float64Array, im: Float64Array}}
 */
export function frBlochEigs(p, o = {}, nTheta = 128) {
  const B = blochBlocks((N) => makeFR1D({ p, N, a: 0, b: N, model: advection(1), flux: o.flux || 'upwind',
    correction: o.correction || 'vcjh', c: o.c ?? 0, points: o.points || 'gauss' }), p + 1);
  return blochEigs(B, nTheta);
}

/**
 * Golden-section search for the VCJH parameter c that maximises the stable CFL
 * number of a given RK method (this is how the "c_+" values of Vincent et al.
 * are defined). The search runs over log10(c) in [lo, hi].
 * @param {number} p ≥ 1
 * @param {string} method RK method
 * @param {{lo?: number, hi?: number, iters?: number, nTheta?: number}} [o] log10 bounds
 * @returns {{c: number, cfl: number}}
 */
export function frOptimalC(p, method, o = {}) {
  const f = (lc) => maxStableCFL(frBlochEigs(p, { c: 10 ** lc }, o.nTheta ?? 64), method, { dnu: 0.01 });
  let a = o.lo ?? Math.log10(cHU(p)) - 1, b = o.hi ?? Math.log10(cHU(p)) + 1.5;
  const g = (Math.sqrt(5) - 1) / 2;
  let x1 = b - g * (b - a), x2 = a + g * (b - a), f1 = f(x1), f2 = f(x2);
  for (let it = 0; it < (o.iters ?? 25); it++) {
    if (f1 < f2) { a = x1; x1 = x2; f1 = f2; x2 = a + g * (b - a); f2 = f(x2); }
    else { b = x2; x2 = x1; f2 = f1; x1 = b - g * (b - a); f1 = f(x1); }
  }
  const lc = 0.5 * (a + b);
  return { c: 10 ** lc, cfl: f(lc) };
}

/**
 * Is the FR scheme with the given correction function the SAME scheme as a DG variant of dg1d.js?
 * (Used by the chapter-7 comparison widget; verified against the actual right-hand sides in the tests.)
 *  - g_DG  ≡ DG in strong form with exact mass matrix and collocated flux: any flux, any solution points;
 *  - g_DG  ≡ DG in weak form with exact integration: linear flux only (a nonlinear flux is over-integrated there);
 *  - g_2   ≡ DG-SEM (GLL collocation, lumped mass): on GLL solution points for any flux, and for a LINEAR
 *            flux on any solution points (the FR polynomial is then independent of the points).
 * @param {{correction: 'dg'|'g2'|'ga', points: 'gauss'|'gll', dg: 'strong'|'weak'|'sem', linear: boolean}} o
 *   points = FR solution points; dg = DG variant ('strong': strong form, exact mass, collocated flux on the
 *   same points; 'weak': weak form, over-integrated; 'sem': GLL collocation); linear = is the flux linear?
 * @returns {boolean}
 */
export function frMatchesDG(o) {
  if (o.correction === 'dg') return o.dg === 'strong' || (o.dg === 'weak' && o.linear);
  if (o.correction === 'g2') return o.dg === 'sem' && (o.points === 'gll' || o.linear);
  return false;
}
