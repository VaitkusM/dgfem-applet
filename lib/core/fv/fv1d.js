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
import { makeStepper } from '../time/rk.js';

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
 * The same limiters in Sweby's flux-limiter form φ(r) of the smoothness ratio
 * r = (ū_i − ū_{i−1}) / (ū_{i+1} − ū_i). For these symmetric limiters
 * (φ(r)/r = φ(1/r)) the limited slope is  σ_i = φ(r) · (ū_{i+1} − ū_i) = LIMITERS[k].fn(a, b).
 * Sweby's TVD region: φ(r) = 0 for r ≤ 0 and 0 ≤ φ(r) ≤ min(2r, 2) for r > 0.
 * 'none' (central slope, Fromm's scheme) is φ = (1 + r)/2: not TVD.
 */
export const LIMITER_PHI = {
  none: (r) => 0.5 * (1 + r),
  minmod: (r) => Math.max(0, Math.min(1, r)),
  mc: (r) => Math.max(0, Math.min(2 * r, 0.5 * (1 + r), 2)),
  vanleer: (r) => (r + Math.abs(r)) / (1 + Math.abs(r)),
  superbee: (r) => Math.max(0, Math.min(2 * r, 1), Math.min(r, 2)),
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

/**
 * Advance a scalar FV discretisation from u to time T with a fixed CFL number
 * ν = Δt · max|f'(u)| / h (recomputed every step, last step shortened).
 * @param {ReturnType<typeof makeFV1D>} fv
 * @param {Float64Array} u cell averages (modified in place)
 * @param {number} T final time (starting from 0)
 * @param {number} cfl
 * @param {string} [method='ssprk3'] RK method (lib/core/time/rk.js)
 * @returns {{steps: number}}
 */
export function advanceFV1D(fv, u, T, cfl, method = 'ssprk3') {
  const step = makeStepper(method, fv.N);
  let t = 0, steps = 0;
  while (t < T - 1e-14) {
    const dt = Math.min(cfl * fv.h / Math.max(fv.maxSpeed(u), 1e-12), T - t);
    step(u, t, dt, fv.rhs); t += dt; steps++;
  }
  return { steps };
}

/** Total variation TV(u) = Σ_i |u_{i+1} − u_i| (periodic: includes the wrap-around pair). */
export function totalVariation(u, periodic = true) {
  let s = 0;
  for (let i = 0; i + 1 < u.length; i++) s += Math.abs(u[i + 1] - u[i]);
  if (periodic) s += Math.abs(u[0] - u[u.length - 1]);
  return s;
}

/**
 * Error function, Abramowitz & Stegun 7.1.26 (|error| ≤ 1.5e−7) — enough for plotting.
 * @param {number} x
 */
export function erf(x) {
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}

/**
 * Modified equation of the first-order upwind scheme with forward Euler for
 * u_t + a u_x = 0 (a > 0), ν = a Δt / h:
 *   u_t + a u_x = D u_xx + O(h²),   D = (a h / 2)(1 − ν).
 * (Taylor-expand u_i^{n+1} = u_i − ν(u_i − u_{i−1}) and use u_tt = a² u_xx.)
 * @param {number} a speed (> 0) @param {number} h cell size @param {number} nu CFL number
 * @returns {number} the numerical diffusion coefficient D
 */
export function upwindNumericalDiffusion(a, h, nu) {
  return 0.5 * a * h * (1 - nu);
}

/**
 * Exact solution of u_t + a u_x = D u_xx on the periodic unit interval for the
 * square pulse u0 = 1 on [x1, x2] (0 elsewhere): a sum of error functions,
 *   u = ½ Σ_k [erf((x − x1 − at + k)/√(4Dt)) − erf((x − x2 − at + k)/√(4Dt))].
 * Used to show that upwind behaves like advection–diffusion.
 * @param {number} x1 @param {number} x2 @param {number} a @param {number} D @param {number} t
 * @returns {(x: number) => number}
 */
export function diffusedSquare(x1, x2, a, D, t) {
  const s = Math.sqrt(4 * D * t);
  return (x) => {
    if (s === 0) { const y = ((x - a * t) % 1 + 1) % 1; return y > x1 && y < x2 ? 1 : 0; }
    let v = 0;
    for (let k = -3; k <= 3; k++) v += 0.5 * (erf((x - x1 - a * t + k) / s) - erf((x - x2 - a * t + k) / s));
    return v;
  };
}
