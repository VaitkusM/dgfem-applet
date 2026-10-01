/**
 * @file Exact Riemann solver for the 1D Euler equations of an ideal gas,
 *       following E. F. Toro, "Riemann Solvers and Numerical Methods for
 *       Fluid Dynamics", 3rd ed. (2009), chapter 4.
 *
 * Riemann problem: primitive states WL = (ρL, uL, pL) for x < 0 and
 * WR = (ρR, uR, pR) for x > 0 at t = 0. The solution is self-similar,
 * W(x, t) = W(x/t), and consists of three waves:
 *   left  wave (u − c family): shock or rarefaction,
 *   contact (u family): ρ jumps, u and p are continuous,
 *   right wave (u + c family): shock or rarefaction.
 * Between them lie the "star" states (ρ*L, u*, p*) and (ρ*R, u*, p*).
 *
 * p* is the root of the pressure function [Toro, eq. 4.5]
 *   f(p) = fL(p) + fR(p) + (uR − uL) = 0,
 * where for K ∈ {L, R}, with AK = 2/((γ+1)ρK), BK = (γ−1)/(γ+1) pK [eqs. 4.6–4.7]:
 *   fK(p) = (p − pK) √(AK / (p + BK))                          p > pK (shock)
 *   fK(p) = 2cK/(γ−1) · ((p/pK)^((γ−1)/(2γ)) − 1)               p ≤ pK (rarefaction)
 * f is monotone increasing and concave, so Newton's method converges from
 * a good guess (we use the two-rarefaction approximation, eq. 4.46).
 * Then u* = ½(uL + uR) + ½(fR(p*) − fL(p*))   [eq. 4.9].
 * Vacuum generation (uR − uL ≥ 2(cL + cR)/(γ−1), eq. 4.40) is not supported.
 */
/** Default ratio of specific heats (air); same value as GAMMA in euler.js. */
const GAMMA_DEFAULT = 1.4;

/**
 * Pressure function fK(p) and its derivative [Toro, eqs. 4.6, 4.7, 4.37].
 * @returns {number[]} [f, f']
 */
function prefun(p, rK, pK, cK, g) {
  if (p > pK) { // shock
    const A = 2 / ((g + 1) * rK), B = (g - 1) / (g + 1) * pK, q = Math.sqrt(A / (B + p));
    return [(p - pK) * q, q * (1 - 0.5 * (p - pK) / (B + p))];
  }
  const pr = p / pK; // rarefaction
  return [2 * cK / (g - 1) * (Math.pow(pr, (g - 1) / (2 * g)) - 1), Math.pow(pr, -(g + 1) / (2 * g)) / (rK * cK)];
}

/**
 * @typedef {Object} EulerRiemannSolution
 * @property {number} pStar  pressure in the star region
 * @property {number} uStar  velocity in the star region (= contact speed)
 * @property {number} rhoStarL density between left wave and contact
 * @property {number} rhoStarR density between contact and right wave
 * @property {'shock'|'rarefaction'} left  type of the left wave
 * @property {'shock'|'rarefaction'} right type of the right wave
 * @property {number[]} leftSpeeds  [shock speed] or [head, tail] speeds of the left fan
 * @property {number[]} rightSpeeds [shock speed] or [tail, head] speeds of the right fan
 * @property {number} iterations Newton iterations used
 * @property {(xi: number, out?: number[]) => number[]} sample primitive state [ρ, u, p] at ξ = x/t
 */

/**
 * Solve the Riemann problem exactly.
 * @param {number[]} WL left primitive state [ρ, u, p]
 * @param {number[]} WR right primitive state [ρ, u, p]
 * @param {number} [g=1.4] ratio of specific heats
 * @param {number} [tol=1e-12] relative Newton tolerance on p
 * @returns {EulerRiemannSolution}
 */
export function exactRiemann(WL, WR, g = GAMMA_DEFAULT, tol = 1e-12) {
  const [rL, uL, pL] = WL, [rR, uR, pR] = WR;
  const cL = Math.sqrt(g * pL / rL), cR = Math.sqrt(g * pR / rR), du = uR - uL;
  if (2 * (cL + cR) / (g - 1) <= du) throw new Error('exactRiemann: initial data generate vacuum');
  // two-rarefaction initial guess [Toro, eq. 4.46]
  const z = (g - 1) / (2 * g);
  let p = Math.pow((cL + cR - 0.5 * (g - 1) * du) / (cL / Math.pow(pL, z) + cR / Math.pow(pR, z)), 1 / z);
  if (!(p > 0)) p = 1e-8;
  let it = 0;
  for (; it < 100; it++) {
    const [fL, dL] = prefun(p, rL, pL, cL, g), [fR, dR] = prefun(p, rR, pR, cR, g);
    let pn = p - (fL + fR + du) / (dL + dR);
    if (pn <= 0) pn = 0.1 * p; // stay positive (never needed for non-vacuum data in practice)
    const change = 2 * Math.abs(pn - p) / (pn + p);
    p = pn;
    if (change < tol) break;
  }
  const pS = p;
  const uS = 0.5 * (uL + uR) + 0.5 * (prefun(pS, rR, pR, cR, g)[0] - prefun(pS, rL, pL, cL, g)[0]);
  const gm = (g - 1) / (g + 1);
  // star densities [Toro, eqs. 4.50 (shock), 4.53 (rarefaction)]
  const starRho = (rK, pK) => (pS > pK
    ? rK * (pS / pK + gm) / (gm * pS / pK + 1)
    : rK * Math.pow(pS / pK, 1 / g));
  const rSL = starRho(rL, pL), rSR = starRho(rR, pR);
  const left = pS > pL ? 'shock' : 'rarefaction', right = pS > pR ? 'shock' : 'rarefaction';
  // wave speeds [Toro, eqs. 4.52, 4.55 (left); 4.59, 4.62 (right)]
  const cSL = Math.sqrt(g * pS / rSL), cSR = Math.sqrt(g * pS / rSR);
  const leftSpeeds = left === 'shock'
    ? [uL - cL * Math.sqrt((g + 1) / (2 * g) * pS / pL + (g - 1) / (2 * g))]
    : [uL - cL, uS - cSL];
  const rightSpeeds = right === 'shock'
    ? [uR + cR * Math.sqrt((g + 1) / (2 * g) * pS / pR + (g - 1) / (2 * g))]
    : [uS + cSR, uR + cR];

  /** Sample at ξ = x/t [Toro, section 4.5, Fig. 4.14]. */
  function sample(xi, out = [0, 0, 0]) {
    if (xi <= uS) { // left of the contact
      if (left === 'shock') {
        if (xi <= leftSpeeds[0]) { out[0] = rL; out[1] = uL; out[2] = pL; } else { out[0] = rSL; out[1] = uS; out[2] = pS; }
      } else if (xi <= leftSpeeds[0]) { out[0] = rL; out[1] = uL; out[2] = pL; }
      else if (xi >= leftSpeeds[1]) { out[0] = rSL; out[1] = uS; out[2] = pS; }
      else { // inside the left fan [eq. 4.56]
        const f = 2 / (g + 1) + (g - 1) / ((g + 1) * cL) * (uL - xi);
        out[0] = rL * Math.pow(f, 2 / (g - 1));
        out[1] = 2 / (g + 1) * (cL + 0.5 * (g - 1) * uL + xi);
        out[2] = pL * Math.pow(f, 2 * g / (g - 1));
      }
    } else if (right === 'shock') {
      if (xi >= rightSpeeds[0]) { out[0] = rR; out[1] = uR; out[2] = pR; } else { out[0] = rSR; out[1] = uS; out[2] = pS; }
    } else if (xi >= rightSpeeds[1]) { out[0] = rR; out[1] = uR; out[2] = pR; }
    else if (xi <= rightSpeeds[0]) { out[0] = rSR; out[1] = uS; out[2] = pS; }
    else { // inside the right fan [eq. 4.63]
      const f = 2 / (g + 1) - (g - 1) / ((g + 1) * cR) * (uR - xi);
      out[0] = rR * Math.pow(f, 2 / (g - 1));
      out[1] = 2 / (g + 1) * (-cR + 0.5 * (g - 1) * uR + xi);
      out[2] = pR * Math.pow(f, 2 * g / (g - 1));
    }
    return out;
  }
  return { pStar: pS, uStar: uS, rhoStarL: rSL, rhoStarR: rSR, left, right, leftSpeeds, rightSpeeds, iterations: it + 1, sample };
}

/**
 * Exact Godunov flux for the Euler equations: F(W(0)) where W(ξ) solves the
 * Riemann problem between the conservative states UL and UR.
 * @param {ArrayLike<number>} UL @param {ArrayLike<number>} UR @param {number[]|Float64Array} out
 * @param {number} [g=1.4]
 */
export function godunovFluxEuler(UL, UR, out, g = GAMMA_DEFAULT) {
  const prim = (U) => { const r = U[0], u = U[1] / r; return [r, u, (g - 1) * (U[2] - 0.5 * r * u * u)]; };
  const W = exactRiemann(prim(UL), prim(UR), g, 1e-10).sample(0);
  const [r, u, p] = W, E = p / (g - 1) + 0.5 * r * u * u;
  out[0] = r * u; out[1] = r * u * u + p; out[2] = u * (E + p);
  return out;
}
