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
 *                   (σ_i = difference per cell, i.e. slope × h)
 *                   u⁻ = ū_i + ½σ_i,  u⁺ = ū_{i+1} − ½σ_{i+1}.
 * Boundary conditions via ghost cells: 'periodic' or 'outflow' (zero-gradient
 * copy of the boundary cell, a.k.a. transmissive).
 */
import { SCALAR_FLUXES, maxWaveSpeed } from '../models/scalar.js';
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
 *   maxSpeed(u): the wave speed for the CFL condition Δt = ν h / maxSpeed. Linear and convex fluxes:
 *   max_i |f'(ū_i)|. Non-convex fluxes: max |f'| over the whole RANGE [min ū, max ū] of the data, because the
 *   fastest wave between two neighbouring cells can be faster than f' at either of them (Buckley–Leverett:
 *   f'(0) = f'(1) = 0, so a jump from 1 to 0 would otherwise report speed 0).
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
  const maxSpeed = (u) => {
    if (m.linear || m.convex) { let s = 0; for (let i = 0; i < N; i++) s = Math.max(s, Math.abs(m.df(u[i]))); return s; }
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < N; i++) { if (u[i] < lo) lo = u[i]; if (u[i] > hi) hi = u[i]; }
    return lo <= hi ? maxWaveSpeed(m, lo, hi) : NaN; // NaN data: let the caller's blow-up guard see it
  };
  return { N, h, xc, xf, rhs, maxSpeed, faceFlux };
}

// 5-point Gauss–Legendre rule on [−1, 1] (exact for polynomials of degree ≤ 9)
const G5X = [-0.9061798459386640, -0.5384693101056831, 0, 0.5384693101056831, 0.9061798459386640];
const G5W = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891];

/**
 * End points of the smooth pieces of the cell [a, b]: a, the break points strictly inside it, b.
 * @param {number} a left face
 * @param {number} b right face (> a)
 * @param {ArrayLike<number>} [breaks] points where the integrand jumps or has a kink (any order)
 * @returns {number[]} increasing, first = a, last = b
 */
export function cellPieces(a, b, breaks) {
  const e = [a];
  if (breaks && breaks.length) {
    const tol = 1e-14 * Math.max(1, Math.abs(a), Math.abs(b));
    const inside = [];
    for (let k = 0; k < breaks.length; k++) if (breaks[k] > a + tol && breaks[k] < b - tol) inside.push(breaks[k]);
    inside.sort((x, y) => x - y);
    for (const x of inside) if (x > e[e.length - 1] + tol) e.push(x);
  }
  e.push(b);
  return e;
}

/**
 * Break points of a 1-periodic function shifted by `shift` (u(x) = u0(x − shift)), reduced to [0, 1).
 * @param {ArrayLike<number>} breaks break points of u0
 * @param {number} shift
 * @returns {number[]}
 */
export function shiftedBreaks(breaks, shift) {
  return Array.from(breaks, (b) => (((b + shift) % 1) + 1) % 1);
}

/**
 * Cell averages of a function: 5-point Gauss quadrature on every smooth piece of every cell.
 * For a smooth f (no break points) this is one Gauss rule per cell, accurate to round-off for the smooth data
 * used in this course. For PIECEWISE smooth data (square pulse, Riemann data, exact solutions with shocks) the
 * positions of the jumps/kinks must be given — a Gauss rule straddling a jump is wrong by O(jump) in that cell
 * (square pulse on 30 cells: total mass 0.34526 instead of 0.35). With the break points the averages are exact
 * for piecewise polynomials of degree ≤ 9, wherever the jumps lie relative to the mesh.
 * @param {((x:number)=>number) & {breaks?: number[]}} f
 * @param {Float64Array} xf face coordinates (N+1)
 * @param {ArrayLike<number>} [breaks=f.breaks] break points of f (any order; points outside the mesh are ignored)
 * @returns {Float64Array}
 */
export function cellAverages(f, xf, breaks = f.breaks) {
  const N = xf.length - 1, u = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const e = cellPieces(xf[i], xf[i + 1], breaks), h = xf[i + 1] - xf[i];
    for (let k = 0; k + 1 < e.length; k++) {
      const c = 0.5 * (e[k] + e[k + 1]), r = 0.5 * (e[k + 1] - e[k]);
      let s = 0;
      for (let q = 0; q < 5; q++) s += G5W[q] * f(c + r * G5X[q]);
      u[i] += (e.length === 2 ? 1 : 2 * r / h) * 0.5 * s; // one piece: exactly the plain Gauss average
    }
  }
  return u;
}

/**
 * Averages of a fine-grid FV solution over the cells of a coarser grid on the SAME interval
 * (exact overlap integration of the piecewise-constant fine data; the cell counts need not divide).
 * Used to compare a coarse solution with a fine-grid reference.
 * @param {Float64Array} fine cell averages on M uniform cells
 * @param {number} N number of coarse cells (N ≤ M is not required)
 * @returns {Float64Array} N cell averages; conserves the mean: Σ out/N = Σ fine/M
 */
export function coarsenAverages(fine, N) {
  const M = fine.length, out = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const a = i * M / N, b = (i + 1) * M / N; // coarse cell i in units of fine cells
    let s = 0;
    for (let k = Math.floor(a); k < M && k < b; k++) s += fine[k] * (Math.min(b, k + 1) - Math.max(a, k));
    out[i] = s / (b - a);
  }
  return out;
}

/**
 * Advance a scalar FV discretisation from u to time T with a fixed CFL number
 * ν = Δt · max|f'(u)| / h (recomputed every step, last step shortened).
 * @param {ReturnType<typeof makeFV1D>} fv
 * @param {Float64Array} u cell averages (modified in place)
 * @param {number} T final time (starting from 0)
 * @param {number} cfl
 * @param {string} [method='ssprk3'] RK method (lib/core/time/rk.js)
 * @param {number} [maxSteps=5e6] safety cap on the number of steps
 * @returns {{steps: number}}
 * @throws {Error} if T or cfl is not finite/positive, if the time step becomes zero or not a number (the
 *   solution has blown up), or if more than maxSteps steps would be needed — never loops forever
 */
export function advanceFV1D(fv, u, T, cfl, method = 'ssprk3', maxSteps = 5e6) {
  if (!Number.isFinite(T) || !(cfl > 0) || !Number.isFinite(cfl)) throw new Error(`advanceFV1D: need finite T and cfl > 0 (got T = ${T}, cfl = ${cfl})`);
  const step = makeStepper(method, fv.N);
  let t = 0, steps = 0;
  while (t < T - 1e-14) {
    const dt = Math.min(cfl * fv.h / Math.max(fv.maxSpeed(u), 1e-12), T - t);
    if (!(dt > 0) || !Number.isFinite(dt)) throw new Error(`advanceFV1D: time step ${dt} at t = ${t} (non-finite solution?)`);
    if (steps >= maxSteps) throw new Error(`advanceFV1D: more than ${maxSteps} steps (t = ${t} of ${T})`);
    step(u, t, dt, fv.rhs); t += dt; steps++;
  }
  return { steps };
}

/**
 * Total variation TV(u) = Σ_i |u_{i+1} − u_i|.
 * @param {ArrayLike<number>} u cell averages
 * @param {boolean} [periodic=true] include the wrap-around pair |u_0 − u_{N−1}|
 * @returns {number}
 */
export function totalVariation(u, periodic = true) {
  let s = 0;
  for (let i = 0; i + 1 < u.length; i++) s += Math.abs(u[i + 1] - u[i]);
  if (periodic) s += Math.abs(u[0] - u[u.length - 1]);
  return s;
}

/**
 * One reconstruct–evolve–average (REA) step for u_t + a u_x = 0, a > 0, on a periodic row of cells, done
 * LITERALLY: the piecewise-linear reconstruction ũ = ū_i + σ_i (x − x_i)/h is shifted by ν cells and the
 * shifted function is integrated exactly over every cell (two linear pieces per cell, 2-point Gauss each).
 * @param {ArrayLike<number>} u cell averages ū_i
 * @param {ArrayLike<number>} sigma slopes σ_i as differences per cell (all 0: Godunov's piecewise constants)
 * @param {number} nu CFL number ν = aΔt/h, 0 ≤ ν ≤ 1
 * @returns {Float64Array} new cell averages
 */
export function reaStepExact(u, sigma, nu) {
  const M = u.length, out = new Float64Array(M), g = 1 / Math.sqrt(3);
  // value of the reconstruction at x (in cell units, periodic)
  const recon = (x) => { const y = ((x % M) + M) % M, i = Math.min(M - 1, Math.floor(y)); return u[i] + sigma[i] * (y - i - 0.5); };
  for (let i = 0; i < M; i++) {
    let v = 0;
    for (const [a, b] of [[i, i + nu], [i + nu, i + 1]]) // split at the kink x = i + ν
      for (const q of [-g, g]) v += 0.5 * (b - a) * recon(0.5 * (a + b) + 0.5 * (b - a) * q - nu);
    out[i] = v;
  }
  return out;
}

/**
 * The same REA step in closed form (flux through face i+½ = area of the trapezoid that crosses it):
 *   ū_i^{n+1} = ū_i − ν (ū_i − ū_{i−1}) − ½ ν (1 − ν) (σ_i − σ_{i−1}),   periodic.
 * σ_i = ū_{i+1} − ū_i gives Lax–Wendroff, σ_i = ū_i − ū_{i−1} Beam–Warming, the central slope Fromm's scheme.
 * @param {ArrayLike<number>} u cell averages
 * @param {ArrayLike<number>} sigma slopes (differences per cell)
 * @param {number} nu CFL number, 0 ≤ ν ≤ 1
 * @returns {Float64Array} new cell averages
 */
export function reaStepFormula(u, sigma, nu) {
  const M = u.length, out = new Float64Array(M);
  for (let i = 0; i < M; i++) {
    const im = (i - 1 + M) % M;
    out[i] = u[i] - nu * (u[i] - u[im]) - 0.5 * nu * (1 - nu) * (sigma[i] - sigma[im]);
  }
  return out;
}

/**
 * Error function, Abramowitz & Stegun 7.1.26 (|error| ≤ 1.5e−7) — enough for plotting.
 * @param {number} x
 * @returns {number}
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
