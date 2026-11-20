/**
 * @file Closed-form facts quoted in chapter 1 (fluxes & conservation laws),
 *       kept here so that the widgets and the tests use the same formulas.
 */

/**
 * Exact solution of the 1D advection–diffusion equation
 *   u_t + a u_x = k u_xx   on the real line
 * with the Gaussian initial profile u(x,0) = exp(−(x−x0)²/σ²):
 *   u(x,t) = σ/√(σ²+4kt) · exp(−(x − x0 − a t)² / (σ² + 4kt)).
 * The pulse moves with speed a and spreads; its mass ∫u dx = σ√π is constant.
 * @param {number} x
 * @param {number} t time ≥ 0
 * @param {{a: number, k: number, sigma: number, x0: number}} prm velocity a, diffusivity k ≥ 0, initial width σ > 0, initial centre x0
 * @returns {{u: number, ux: number}} value and x-derivative
 */
export function advDiffGaussian(x, t, prm) {
  const { a, k, sigma, x0 } = prm;
  const s2 = sigma * sigma + 4 * k * t, d = x - x0 - a * t;
  const u = sigma / Math.sqrt(s2) * Math.exp(-(d * d) / s2);
  return { u, ux: -2 * d / s2 * u };
}

/**
 * Breaking time of u_t + u u_x = 0 (Burgers) for smooth initial data u0:
 * the characteristics x = x0 + u0(x0) t first cross at
 *   t* = −1 / min_x u0'(x)      (t* = ∞ if u0 is non-decreasing).
 * The minimum is searched on a uniform sample of [a,b].
 * @param {(x:number)=>number} du0 derivative u0'
 * @param {number} [a=0]
 * @param {number} [b=1]
 * @param {number} [n=4000] number of sample intervals
 * @returns {number}
 */
export function burgersBreakingTime(du0, a = 0, b = 1, n = 4000) {
  let m = Infinity;
  for (let i = 0; i <= n; i++) m = Math.min(m, du0(a + (b - a) * i / n));
  return m < 0 ? -1 / m : Infinity;
}

/**
 * Type of the entropy solution of the Burgers Riemann problem
 * (u = uL for x < 0, u = uR for x > 0):
 *  - uL > uR: 'shock' moving with the Rankine–Hugoniot speed s = (uL+uR)/2,
 *  - uL < uR: 'rarefaction' fan u = x/t for uL ≤ x/t ≤ uR,
 *  - uL = uR: 'constant' (no wave at all).
 * @param {number} uL
 * @param {number} uR
 * @returns {{kind: 'shock'|'rarefaction'|'constant', speed: number}} speed = (uL+uR)/2
 *   (shock speed; for the other two cases just the mean characteristic speed)
 */
export function burgersRiemannKind(uL, uR) {
  const speed = 0.5 * (uL + uR);
  return { kind: uL > uR ? 'shock' : uL < uR ? 'rarefaction' : 'constant', speed };
}
