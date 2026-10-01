/**
 * @file Finite volume method for scalar conservation laws u_t + f(u)_x = 0 in 1D.
 *
 * Unknowns are cell averages ū_i over cells [x_{i−½}, x_{i+½}] of width h.
 * Integrating the PDE over cell i exactly gives
 *     dū_i/dt = −( F_{i+½} − F_{i−½} ) / h,
 * where F_{i+½} is the flux through the right face. The method is defined by
 * how F_{i+½} is computed from the averages: a numerical flux F̂(u⁻, u⁺) of
 * the states just left/right of the face.
 *  - first order:   u⁻ = ū_i,  u⁺ = ū_{i+1}
 *  - MUSCL (2nd):   piecewise-linear reconstruction with a limited slope σ_i,
 *                   u⁻ = ū_i + σ_i h/2,  u⁺ = ū_{i+1} − σ_{i+1} h/2.
 * Boundary conditions via ghost cells: 'periodic' or 'outflow' (zero-gradient
 * copy of the boundary cell, a.k.a. transmissive).
 */
import { SCALAR_FLUXES } from '../models/scalar.js';

/** Slope limiters φ(a, b) for one-sided differences a = ū_i − ū_{i−1}, b = ū_{i+1} − ū_i. */
export const LIMITERS = {
  none: { label: 'none (unlimited central slope)', fn: (a, b) => 0.5 * (a + b) },
  minmod: { label: 'minmod', fn: (a, b) => (a * b <= 0 ? 0 : Math.abs(a) < Math.abs(b) ? a : b) },
  mc: {
    label: 'monotonised central (MC)',
    fn: (a, b) => (a * b <= 0 ? 0 : Math.sign(a) * Math.min(2 * Math.abs(a), 2 * Math.abs(b), 0.5 * Math.abs(a + b))),
  },
  vanleer: { label: 'van Leer', fn: (a, b) => (a * b <= 0 ? 0 : (2 * a * b) / (a + b)) },
  superbee: {
    label: 'superbee',
    fn: (a, b) => {
      if (a * b <= 0) return 0;
      const s = Math.sign(a), A = Math.abs(a), B = Math.abs(b);
      return s * Math.max(Math.min(2 * A, B), Math.min(A, 2 * B));
    },
  },
};

/**
 * Create a 1D scalar FV discretisation.
 * @param {{model: import('../models/scalar.js').ScalarModel, N: number, a?: number, b?: number,
 *          flux?: string, recon?: 'none'|'muscl', limiter?: string, bc?: 'periodic'|'outflow'}} o
 * @returns {{N: number, h: number, xc: Float64Array, xf: Float64Array, rhs: (u: Float64Array, t: number, out: Float64Array) => void,
 *            maxSpeed: (u: Float64Array) => number, faceFlux: Float64Array}}
 */
export function makeFV1D(o) {
  const N = o.N, a = o.a ?? 0, b = o.b ?? 1, h = (b - a) / N;
  const m = o.model, F = SCALAR_FLUXES[o.flux || 'godunov'].fn;
  const lim = LIMITERS[o.limiter || 'minmod'].fn;
  const muscl = o.recon === 'muscl';
  const periodic = (o.bc || 'periodic') === 'periodic';
  const xc = new Float64Array(N), xf = new Float64Array(N + 1);
  for (let i = 0; i <= N; i++) xf[i] = a + i * h;
  for (let i = 0; i < N; i++) xc[i] = a + (i + 0.5) * h;
  const G = 2; // ghost cells per side
  const ue = new Float64Array(N + 2 * G), slope = new Float64Array(N + 2 * G);
  const faceFlux = new Float64Array(N + 1);
  const fill = (u) => {
    ue.set(u, G);
    for (let g = 0; g < G; g++) {
      ue[g] = periodic ? u[N - G + g] : u[0];
      ue[N + G + g] = periodic ? u[g] : u[N - 1];
    }
  };
  function rhs(u, t, out) {
    fill(u);
    if (muscl) for (let i = 1; i < N + 2 * G - 1; i++) slope[i] = lim(ue[i] - ue[i - 1], ue[i + 1] - ue[i]);
    for (let f = 0; f <= N; f++) {
      const L = f - 1 + G, R = f + G; // extended indices of the cells left/right of face f
      const uL = muscl ? ue[L] + 0.5 * slope[L] : ue[L];
      const uR = muscl ? ue[R] - 0.5 * slope[R] : ue[R];
      faceFlux[f] = F(m, uL, uR);
    }
    for (let i = 0; i < N; i++) out[i] = -(faceFlux[i + 1] - faceFlux[i]) / h;
  }
  const maxSpeed = (u) => { let s = 0; for (let i = 0; i < N; i++) s = Math.max(s, Math.abs(m.df(u[i]))); return s; };
  return { N, h, xc, xf, rhs, maxSpeed, faceFlux };
}

/**
 * Exact cell averages of a function (Gauss quadrature per cell).
 * @param {(x:number)=>number} f
 * @param {Float64Array} xf face coordinates (N+1)
 * @returns {Float64Array}
 */
export function cellAverages(f, xf) {
  const N = xf.length - 1, u = new Float64Array(N);
  // 5-point Gauss–Legendre on each cell
  const gx = [-0.9061798459386640, -0.5384693101056831, 0, 0.5384693101056831, 0.9061798459386640];
  const gw = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891];
  for (let i = 0; i < N; i++) {
    const c = 0.5 * (xf[i] + xf[i + 1]), r = 0.5 * (xf[i + 1] - xf[i]);
    let s = 0;
    for (let q = 0; q < 5; q++) s += gw[q] * f(c + r * gx[q]);
    u[i] = 0.5 * s;
  }
  return u;
}
