/**
 * @file Legendre and Jacobi polynomials on the reference interval [-1, 1].
 *
 * Legendre polynomials P_n are the orthogonal polynomials for the weight 1 on
 * [-1,1]:  ∫ P_m P_n dx = 2/(2n+1) δ_mn.  They are the natural "modal" basis
 * of DG methods (like a Fourier basis, but for polynomials on an interval).
 *
 * Three-term recurrence (Bonnet):  (n+1) P_{n+1} = (2n+1) x P_n − n P_{n−1}.
 * Derivative recurrence:           P'_{n+1} = P'_{n−1} + (2n+1) P_n.
 *
 * Jacobi polynomials P_n^{(α,β)} are orthogonal for the weight (1−x)^α(1+x)^β;
 * they appear in the orthonormal (Dubiner) basis on triangles.
 *
 * All functions are pure and DOM-free.
 */

/**
 * Value of the Legendre polynomial P_n at x.
 * @param {number} n degree ≥ 0
 * @param {number} x point in [-1,1] (any real works)
 * @returns {number}
 */
export function legendreP(n, x) {
  if (n === 0) return 1;
  let p0 = 1, p1 = x;
  for (let k = 1; k < n; k++) {
    const p2 = ((2 * k + 1) * x * p1 - k * p0) / (k + 1);
    p0 = p1; p1 = p2;
  }
  return p1;
}

/**
 * All Legendre values P_0..P_N and derivatives P'_0..P'_N at x.
 * @param {number} N max degree
 * @param {number} x
 * @returns {{P: Float64Array, dP: Float64Array}}
 */
export function legendreAll(N, x) {
  const P = new Float64Array(N + 1);
  const dP = new Float64Array(N + 1);
  P[0] = 1;
  if (N >= 1) { P[1] = x; dP[1] = 1; }
  for (let k = 1; k < N; k++) {
    P[k + 1] = ((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1);
    dP[k + 1] = dP[k - 1] + (2 * k + 1) * P[k];
  }
  return { P, dP };
}

/**
 * Derivative P'_n(x).
 * @param {number} n
 * @param {number} x
 */
export function legendreDP(n, x) {
  return legendreAll(n, x).dP[n];
}

/**
 * Right Radau polynomial of degree k (Huynh 2007):
 *   R_{R,k}(x) = (−1)^k / 2 · (P_k(x) − P_{k−1}(x)).
 * It satisfies R_{R,k}(−1) = 1 and R_{R,k}(1) = 0 — it "lives" at the left end
 * and vanishes at the right end. Used for FR correction functions.
 * @param {number} k degree ≥ 1
 * @param {number} x
 * @returns {{v: number, d: number}} value and derivative
 */
export function radauRight(k, x) {
  const { P, dP } = legendreAll(k, x);
  const s = (k % 2 === 0 ? 1 : -1) / 2;
  return { v: s * (P[k] - P[k - 1]), d: s * (dP[k] - dP[k - 1]) };
}

/* ------------------------------------------------------------------------ */
/*  Jacobi polynomials (normalised), after Hesthaven & Warburton (2008),     */
/*  "Nodal Discontinuous Galerkin Methods", Appendix A (JacobiP, GradJacobiP) */
/* ------------------------------------------------------------------------ */

/** log Γ via Lanczos approximation (only used for small integer-ish args). */
function gammaFn(z) {
  // For the arguments we need (positive integers and half-integers ≤ ~30)
  // a direct product is exact enough; use recursion down to (0,1].
  let r = 1;
  while (z > 1) { z -= 1; r *= z; }
  // now z in (0,1]; Γ(1) = 1, Γ(1/2) = √π
  if (Math.abs(z - 1) < 1e-14) return r;
  if (Math.abs(z - 0.5) < 1e-14) return r * Math.sqrt(Math.PI);
  throw new Error('gammaFn: unsupported argument');
}

/**
 * Normalised Jacobi polynomial  P̃_n^{(α,β)}(x), orthonormal w.r.t. the weight
 * (1−x)^α (1+x)^β on [−1,1].
 * @param {number} x
 * @param {number} alpha α > −1
 * @param {number} beta β > −1
 * @param {number} n degree
 */
export function jacobiP(x, alpha, beta, n) {
  const ab = alpha + beta;
  const gamma0 = Math.pow(2, ab + 1) / (ab + 1) * gammaFn(alpha + 1) * gammaFn(beta + 1) / gammaFn(ab + 1);
  const p0 = 1 / Math.sqrt(gamma0);
  if (n === 0) return p0;
  const gamma1 = (alpha + 1) * (beta + 1) / (ab + 3) * gamma0;
  const p1 = ((ab + 2) * x / 2 + (alpha - beta) / 2) / Math.sqrt(gamma1);
  if (n === 1) return p1;
  let aold = 2 / (2 + ab) * Math.sqrt((alpha + 1) * (beta + 1) / (ab + 3));
  let pm = p0, pc = p1;
  for (let i = 1; i < n; i++) {
    const h1 = 2 * i + ab;
    const anew = 2 / (h1 + 2) * Math.sqrt((i + 1) * (i + 1 + ab) * (i + 1 + alpha) * (i + 1 + beta) / (h1 + 1) / (h1 + 3));
    const bnew = -(alpha * alpha - beta * beta) / h1 / (h1 + 2);
    const pn = (-aold * pm + (x - bnew) * pc) / anew;
    pm = pc; pc = pn; aold = anew;
  }
  return pc;
}

/**
 * Derivative of the normalised Jacobi polynomial:
 *   d/dx P̃_n^{(α,β)} = √(n(n+α+β+1)) · P̃_{n−1}^{(α+1,β+1)}.
 */
export function jacobiDP(x, alpha, beta, n) {
  if (n === 0) return 0;
  return Math.sqrt(n * (n + alpha + beta + 1)) * jacobiP(x, alpha + 1, beta + 1, n - 1);
}
