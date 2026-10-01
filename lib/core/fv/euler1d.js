/**
 * @file Finite volume method for the 1D Euler equations (method of lines).
 *
 * Unknowns: cell averages of the conservative variables U = (ρ, ρu, E),
 * interleaved: U[3i + k] for cell i = 0..N−1, component k = 0, 1, 2.
 * Semi-discrete scheme (exact for cell averages if the face fluxes were exact):
 *     dŪ_i/dt = −( F̂_{i+½} − F̂_{i−½} ) / h,
 *     F̂_{i+½} = F̂(U⁻_{i+½}, U⁺_{i+½})   (a numerical flux from models/euler.js).
 * Face states:
 *  - first order:  U⁻_{i+½} = Ū_i, U⁺_{i+½} = Ū_{i+1}  (Godunov's piecewise constants)
 *  - MUSCL:        piecewise-linear reconstruction of the PRIMITIVE variables
 *                  W = (ρ, u, p), each limited separately with a slope limiter
 *                  φ(a, b) of the one-sided differences (see fv1d.js LIMITERS):
 *                  W⁻_{i+½} = W_i + ½σ_i,  W⁺_{i−½} = W_i − ½σ_i.
 *    Primitive (not conservative, not characteristic) limiting is a common,
 *    simple and robust choice; characteristic limiting is slightly sharper
 *    but needs the eigenvectors of the flux Jacobian per cell.
 *    Positivity safeguard: if a reconstructed face value of ρ or p would be
 *    ≤ 0, that cell falls back to first order (σ_i = 0).
 * Boundary conditions via 2 ghost cells per side: 'outflow' (transmissive,
 * zero-gradient copy) or 'periodic'.
 * Time stepping is left to the caller (lib/core/time/rk.js, SSP-RK).
 */
import { EULER_FLUXES, GAMMA, maxWaveSpeed } from '../models/euler.js';
import { LIMITERS } from './fv1d.js';

/**
 * Create the Euler FV discretisation.
 * @param {{N: number, a?: number, b?: number, flux?: string, recon?: 'none'|'muscl', limiter?: string,
 *          bc?: 'outflow'|'periodic', gamma?: number}} o
 * @returns {{N: number, h: number, xc: Float64Array, xf: Float64Array,
 *   rhs: (U: Float64Array, t: number, out: Float64Array) => void,
 *   maxSpeed: (U: Float64Array) => number,
 *   faceFlux: Float64Array, fallbacks: () => number}}
 *   faceFlux[3f + k]: flux component k through face f (f = 0..N) of the last rhs call;
 *   fallbacks(): number of cells that used the positivity fallback in the last rhs call.
 */
export function makeEuler1D(o) {
  const N = o.N, a = o.a ?? 0, b = o.b ?? 1, h = (b - a) / N, g = o.gamma ?? GAMMA;
  const F = EULER_FLUXES[o.flux || 'hllc'].fn;
  const lim = LIMITERS[o.limiter || 'minmod'].fn;
  const muscl = o.recon === 'muscl';
  const periodic = o.bc === 'periodic';
  const xc = new Float64Array(N), xf = new Float64Array(N + 1);
  for (let i = 0; i <= N; i++) xf[i] = a + i * h;
  for (let i = 0; i < N; i++) xc[i] = a + (i + 0.5) * h;
  const G = 2, NE = N + 2 * G;
  const W = new Float64Array(3 * NE), S = new Float64Array(3 * NE); // primitive values and slopes (extended)
  const faceFlux = new Float64Array(3 * (N + 1));
  const UL = new Float64Array(3), UR = new Float64Array(3), fl = new Float64Array(3);
  let nFallback = 0;
  const toCons = (r, u, p, out) => { out[0] = r; out[1] = r * u; out[2] = p / (g - 1) + 0.5 * r * u * u; };

  function rhs(U, t, out) {
    // 1. primitive variables in all cells incl. ghosts
    for (let e = 0; e < NE; e++) {
      let i = e - G;
      if (i < 0) i = periodic ? i + N : 0;
      else if (i >= N) i = periodic ? i - N : N - 1;
      const r = U[3 * i], u = U[3 * i + 1] / r;
      W[3 * e] = r; W[3 * e + 1] = u; W[3 * e + 2] = (g - 1) * (U[3 * i + 2] - 0.5 * r * u * u);
    }
    // 2. limited slopes (per primitive component), with positivity fallback
    S.fill(0); nFallback = 0;
    if (muscl) {
      for (let e = 1; e < NE - 1; e++) {
        for (let k = 0; k < 3; k++) S[3 * e + k] = lim(W[3 * e + k] - W[3 * (e - 1) + k], W[3 * (e + 1) + k] - W[3 * e + k]);
        const r = W[3 * e], p = W[3 * e + 2], sr = 0.5 * Math.abs(S[3 * e]), sp = 0.5 * Math.abs(S[3 * e + 2]);
        if (r - sr <= 0 || p - sp <= 0) { S[3 * e] = S[3 * e + 1] = S[3 * e + 2] = 0; nFallback++; }
      }
    }
    // 3. face fluxes
    for (let f = 0; f <= N; f++) {
      const L = f - 1 + G, R = f + G;
      toCons(W[3 * L] + 0.5 * S[3 * L], W[3 * L + 1] + 0.5 * S[3 * L + 1], W[3 * L + 2] + 0.5 * S[3 * L + 2], UL);
      toCons(W[3 * R] - 0.5 * S[3 * R], W[3 * R + 1] - 0.5 * S[3 * R + 1], W[3 * R + 2] - 0.5 * S[3 * R + 2], UR);
      F(UL, UR, fl);
      faceFlux[3 * f] = fl[0]; faceFlux[3 * f + 1] = fl[1]; faceFlux[3 * f + 2] = fl[2];
    }
    // 4. flux differences
    for (let i = 0; i < N; i++)
      for (let k = 0; k < 3; k++) out[3 * i + k] = -(faceFlux[3 * (i + 1) + k] - faceFlux[3 * i + k]) / h;
  }

  const maxSpeed = (U) => { let s = 0; for (let i = 0; i < N; i++) s = Math.max(s, maxWaveSpeed(U, 3 * i, g)); return s; };
  return { N, h, xc, xf, rhs, maxSpeed, faceFlux, fallbacks: () => nFallback };
}

/**
 * Cell averages of a primitive initial state given as a function of x
 * (5-point Gauss per cell, averaging the CONSERVATIVE variables).
 * @param {(x: number) => number[]} W0 x ↦ [ρ, u, p]
 * @param {Float64Array} xf faces (N+1)
 * @param {number} [g=GAMMA]
 * @returns {Float64Array} interleaved U (3N)
 */
export function eulerCellAverages(W0, xf, g = GAMMA) {
  const N = xf.length - 1, U = new Float64Array(3 * N);
  const gx = [-0.9061798459386640, -0.5384693101056831, 0, 0.5384693101056831, 0.9061798459386640];
  const gw = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891];
  for (let i = 0; i < N; i++) {
    const c = 0.5 * (xf[i] + xf[i + 1]), r = 0.5 * (xf[i + 1] - xf[i]);
    for (let q = 0; q < 5; q++) {
      const [rho, u, p] = W0(c + r * gx[q]), w = 0.5 * gw[q];
      U[3 * i] += w * rho; U[3 * i + 1] += w * rho * u; U[3 * i + 2] += w * (p / (g - 1) + 0.5 * rho * u * u);
    }
  }
  return U;
}

/**
 * Diagnostics of a grid state: minima of ρ and p (positivity) and totals Σ U_i h.
 * @param {Float64Array} U interleaved (3N)
 * @param {number} h
 * @param {number} [g=GAMMA]
 * @returns {{minRho: number, minP: number, mass: number, momentum: number, energy: number, finite: boolean}}
 */
export function eulerStats(U, h, g = GAMMA) {
  const N = U.length / 3;
  let minRho = Infinity, minP = Infinity, mass = 0, mom = 0, en = 0, finite = true;
  for (let i = 0; i < N; i++) {
    const r = U[3 * i], m = U[3 * i + 1], E = U[3 * i + 2], p = (g - 1) * (E - 0.5 * m * m / r);
    if (!Number.isFinite(r) || !Number.isFinite(p)) finite = false;
    minRho = Math.min(minRho, r); minP = Math.min(minP, p);
    mass += r * h; mom += m * h; en += E * h;
  }
  return { minRho, minP, mass, momentum: mom, energy: en, finite };
}

/**
 * Classic shock-tube problems on [0, 1]: left/right primitive states, the
 * diaphragm position x0 and a final time T (Sod 1978; Toro 2009, Tables 4.1 and 10.1; Lax 1954).
 * @type {Record<string, {label: string, WL: number[], WR: number[], x0: number, T: number}>}
 */
export const SHOCK_TUBES = {
  sod: { label: 'Sod shock tube', WL: [1, 0, 1], WR: [0.125, 0, 0.1], x0: 0.5, T: 0.2 },
  sonic: { label: 'Sod with sonic rarefaction (Toro test 1)', WL: [1, 0.75, 1], WR: [0.125, 0, 0.1], x0: 0.3, T: 0.2 },
  lax: { label: 'Lax problem', WL: [0.445, 0.698, 3.528], WR: [0.5, 0, 0.571], x0: 0.5, T: 0.13 },
  '123': { label: '123 problem: two rarefactions, near vacuum', WL: [1, -2, 0.4], WR: [1, 2, 0.4], x0: 0.5, T: 0.15 },
  contact: { label: 'Stationary contact (ρ jumps, u = 0, p = 1)', WL: [1, 0, 1], WR: [0.2, 0, 1], x0: 0.5, T: 0.2 },
  blast: { label: 'Left blast wave (pressure ratio 10⁵)', WL: [1, 0, 1000], WR: [1, 0, 0.01], x0: 0.5, T: 0.012 },
};
