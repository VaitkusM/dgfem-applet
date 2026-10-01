/**
 * @file Explicit Runge–Kutta time integrators for semi-discrete systems
 *       du/dt = L(u, t)   (the "method of lines").
 *
 * After space discretisation (FV, DG, FR, …) a PDE becomes a large system of
 * ODEs. We advance it with explicit RK schemes:
 *  - 'fe'     forward Euler, order 1
 *  - 'ssprk2' Heun / SSP-RK2 (Shu–Osher form), order 2
 *  - 'ssprk3' SSP-RK3 (Shu–Osher 1988), order 3 — the workhorse for DG + limiters
 *  - 'rk4'    classical RK4, order 4
 *  - 'lsrk4'  5-stage, 4th-order low-storage RK of Carpenter & Kennedy (1994)
 * "SSP" (strong-stability preserving) schemes are convex combinations of
 * forward-Euler steps, so any property forward Euler has under a CFL limit
 * (e.g. no new oscillations with a TVD limiter) is kept.
 *
 * L is called as L(u, t, out) and must write du/dt into `out`.
 */

/** @typedef {(u: Float64Array, t: number, out: Float64Array) => void} RHS */

export const RK_METHODS = ['fe', 'ssprk2', 'ssprk3', 'rk4', 'lsrk4'];
export const RK_ORDER = { fe: 1, ssprk2: 2, ssprk3: 3, rk4: 4, lsrk4: 4 };
export const RK_LABEL = {
  fe: 'Forward Euler (1st order)',
  ssprk2: 'SSP-RK2 (Heun)',
  ssprk3: 'SSP-RK3 (Shu–Osher)',
  rk4: 'Classical RK4',
  lsrk4: 'Low-storage RK4 (5 stages)',
};

// Carpenter & Kennedy (1994) 5-stage 4th-order 2N-storage coefficients
const LS_A = [0, -567301805773 / 1357537059087, -2404267990393 / 2016746695238,
  -3550918686646 / 2091501179385, -1275806237668 / 842570457699];
const LS_B = [1432997174477 / 9575080441755, 5161836677717 / 13612068292357,
  1720146321549 / 2090206949498, 3134564353537 / 4481467310338, 2277821191437 / 14882151754819];
const LS_C = [0, 1432997174477 / 9575080441755, 2526269341429 / 6820363962896,
  2006345519317 / 3224310063776, 2802321613138 / 2924317926251];

/**
 * Create a stepper with pre-allocated work arrays.
 * @param {string} method one of RK_METHODS
 * @param {number} n system size
 * @param {(u: Float64Array) => void} [post] optional post-stage hook (e.g. a slope limiter)
 * @returns {(u: Float64Array, t: number, dt: number, L: RHS) => void} advances u in place
 */
export function makeStepper(method, n, post) {
  const k = new Float64Array(n), u1 = new Float64Array(n), u2 = new Float64Array(n);
  const k2 = new Float64Array(n), k3 = new Float64Array(n), k4 = new Float64Array(n);
  const P = post || (() => {});
  switch (method) {
    case 'fe':
      return (u, t, dt, L) => {
        L(u, t, k);
        for (let i = 0; i < n; i++) u[i] += dt * k[i];
        P(u);
      };
    case 'ssprk2':
      return (u, t, dt, L) => {
        L(u, t, k);
        for (let i = 0; i < n; i++) u1[i] = u[i] + dt * k[i];
        P(u1);
        L(u1, t + dt, k);
        for (let i = 0; i < n; i++) u[i] = 0.5 * u[i] + 0.5 * (u1[i] + dt * k[i]);
        P(u);
      };
    case 'ssprk3':
      return (u, t, dt, L) => {
        L(u, t, k);
        for (let i = 0; i < n; i++) u1[i] = u[i] + dt * k[i];
        P(u1);
        L(u1, t + dt, k);
        for (let i = 0; i < n; i++) u2[i] = 0.75 * u[i] + 0.25 * (u1[i] + dt * k[i]);
        P(u2);
        L(u2, t + 0.5 * dt, k);
        for (let i = 0; i < n; i++) u[i] = u[i] / 3 + (2 / 3) * (u2[i] + dt * k[i]);
        P(u);
      };
    case 'rk4':
      return (u, t, dt, L) => {
        L(u, t, k);
        for (let i = 0; i < n; i++) u1[i] = u[i] + 0.5 * dt * k[i];
        L(u1, t + 0.5 * dt, k2);
        for (let i = 0; i < n; i++) u1[i] = u[i] + 0.5 * dt * k2[i];
        L(u1, t + 0.5 * dt, k3);
        for (let i = 0; i < n; i++) u1[i] = u[i] + dt * k3[i];
        L(u1, t + dt, k4);
        for (let i = 0; i < n; i++) u[i] += (dt / 6) * (k[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        P(u);
      };
    case 'lsrk4':
      return (u, t, dt, L) => {
        k2.fill(0); // residual register
        for (let s = 0; s < 5; s++) {
          L(u, t + LS_C[s] * dt, k);
          for (let i = 0; i < n; i++) {
            k2[i] = LS_A[s] * k2[i] + dt * k[i];
            u[i] += LS_B[s] * k2[i];
          }
        }
        P(u);
      };
    default:
      throw new Error(`unknown RK method ${method}`);
  }
}

/**
 * Stability function R(z) of a method: applying one step to y' = λ y with
 * z = λ dt gives y₁ = R(z) y₀. The method is stable for λ dt where |R(z)| ≤ 1.
 * Computed by running the actual stepper on a 2-vector (Re, Im) — so the plot
 * is guaranteed to match the integrator used in the simulations.
 * @param {string} method
 * @param {number} zr Re z
 * @param {number} zi Im z
 * @returns {number} |R(z)|
 */
export function stabilityAmp(method, zr, zi) {
  const step = stabilityAmp.cache[method] || (stabilityAmp.cache[method] = makeStepper(method, 2));
  const y = Float64Array.of(1, 0);
  // complex multiplication by z, real 2×2 form
  step(y, 0, 1, (u, t, out) => { out[0] = zr * u[0] - zi * u[1]; out[1] = zi * u[0] + zr * u[1]; });
  return Math.hypot(y[0], y[1]);
}
stabilityAmp.cache = {};
