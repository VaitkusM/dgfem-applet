/**
 * @file Correction functions for flux reconstruction (FR), after
 *       H. T. Huynh (2007) and Vincent, Castonguay & Jameson (2011).
 *
 * A LEFT correction function g_L is a polynomial of degree p+1 on [−1,1] with
 *     g_L(−1) = 1,   g_L(1) = 0,
 * and the RIGHT one is its mirror image  g_R(ξ) = g_L(−ξ)  (so g_R(−1) = 0, g_R(1) = 1).
 * All families below are combinations of Legendre polynomials P_k.
 *
 * Right Radau polynomial (basis/legendre.js):  R_{R,k} = (−1)^k/2 · (P_k − P_{k−1}),
 *   R_{R,k}(−1) = 1, R_{R,k}(1) = 0; it vanishes at the k right-Radau points.
 *
 *  - 'dg'   g_DG = R_{R,p+1}. Since P_{p+1} − P_p ⟂ P_{p−1}, ∫ g_DG' v dξ = −v(−1) for every
 *           v ∈ P_p; this is exactly the DG "lifting" of a left face term → FR ≡ nodal DG.
 *  - 'g2'   g_2 = ( p R_{R,p+1} + (p+1) R_{R,p} ) / (2p+1)    (Huynh's g_2).
 *  - 'ga'   g_Ga = ( (p+1) R_{R,p+1} + p R_{R,p} ) / (2p+1) = (−1)^p/2 · (1 − ξ) P_p(ξ):
 *           zero at the p Gauss points (roots of P_p) and at ξ = 1.
 *  - 'vcjh' Vincent–Castonguay–Jameson–Huynh one-parameter family (c ≥ 0):
 *           g_L = (−1)^p/2 · [ P_p − (η_p P_{p−1} + P_{p+1}) / (1 + η_p) ],
 *           η_p = c (2p+1) (a_p p!)² / 2,   a_p = (2p)! / (2^p (p!)²)
 *           (a_p = leading coefficient of P_p). Special values:
 *             c = 0                                  → g_DG
 *             c_SD = 2p / ((2p+1)(p+1)(a_p p!)²)       → η_p = p/(p+1)   → g_Ga
 *             c_HU = 2(p+1) / ((2p+1) p (a_p p!)²)     → η_p = (p+1)/p   → g_2
 *           c → ∞ gives (−1)^p/2 (P_p − P_{p−1}) = R_{R,p}, a correction of degree p only.
 *
 * p = 0 (piecewise constants): every family reduces to g_L = (1 − ξ)/2.
 */
import { legendreAll } from '../basis/legendre.js';

/** n! for small n. */
function fact(n) { let r = 1; for (let k = 2; k <= n; k++) r *= k; return r; }

/** Leading coefficient a_p of the Legendre polynomial P_p: (2p)!/(2^p (p!)²). */
export function legendreLeading(p) { return fact(2 * p) / (2 ** p * fact(p) ** 2); }

/** (a_p p!)², the constant appearing in the VCJH formulas. */
function apPf2(p) { const v = legendreLeading(p) * fact(p); return v * v; }

/** c_SD (VCJH value recovering g_Ga / spectral difference). @param {number} p ≥ 1 */
export function cSD(p) { return 2 * p / ((2 * p + 1) * (p + 1) * apPf2(p)); }
/** c_HU (VCJH value recovering Huynh's g_2). @param {number} p ≥ 1 */
export function cHU(p) { return 2 * (p + 1) / ((2 * p + 1) * p * apPf2(p)); }
/** η_p(c) of the VCJH family. */
export function vcjhEta(p, c) { return c * (2 * p + 1) * apPf2(p) / 2; }

/**
 * Left correction function as Legendre coefficients: g_L = Σ_k coef[k] P_k, k = 0..p+1.
 * @param {'dg'|'g2'|'ga'|'vcjh'} type
 * @param {number} p solution degree ≥ 0
 * @param {number} [c=0] VCJH parameter (type 'vcjh' only)
 * @returns {Float64Array} length p+2
 */
export function correctionCoeffs(type, p, c = 0) {
  const a = new Float64Array(p + 2);
  if (p === 0) { a[0] = 0.5; a[1] = -0.5; return a; } // (1 − ξ)/2
  const s = (p % 2 === 0 ? 1 : -1) / 2; // (−1)^p / 2
  // R_{R,k} = (−1)^k/2 (P_k − P_{k−1});  R_{R,p+1} = −s (P_{p+1} − P_p),  R_{R,p} = s (P_p − P_{p−1})
  const addRR = (k, wgt) => {
    const sk = (k % 2 === 0 ? 1 : -1) / 2;
    a[k] += wgt * sk; a[k - 1] -= wgt * sk;
  };
  switch (type) {
    case 'dg': addRR(p + 1, 1); break;
    case 'g2': addRR(p + 1, p / (2 * p + 1)); addRR(p, (p + 1) / (2 * p + 1)); break;
    case 'ga': addRR(p + 1, (p + 1) / (2 * p + 1)); addRR(p, p / (2 * p + 1)); break;
    case 'vcjh': {
      const eta = vcjhEta(p, c);
      a[p] = s; a[p - 1] = -s * eta / (1 + eta); a[p + 1] = -s / (1 + eta);
      break;
    }
    default: throw new Error(`unknown correction function ${type}`);
  }
  return a;
}

/**
 * Evaluate a correction function pair at ξ.
 * @param {Float64Array} coef Legendre coefficients of g_L (from correctionCoeffs)
 * @param {number} xi
 * @returns {{gL: number, dgL: number, gR: number, dgR: number}}
 *   g_R(ξ) = g_L(−ξ), so g_R'(ξ) = −g_L'(−ξ).
 */
export function evalCorrection(coef, xi) {
  const K = coef.length - 1;
  const A = legendreAll(K, xi), B = legendreAll(K, -xi);
  let gL = 0, dgL = 0, gR = 0, dgR = 0;
  for (let k = 0; k <= K; k++) {
    gL += coef[k] * A.P[k]; dgL += coef[k] * A.dP[k];
    gR += coef[k] * B.P[k]; dgR -= coef[k] * B.dP[k];
  }
  return { gL, dgL, gR, dgR };
}

/** Labels for the UI. */
export const CORRECTION_LABELS = {
  dg: '$g_{DG}$ (Radau, recovers DG)',
  g2: '$g_2$ (Huynh)',
  ga: '$g_{Ga}$ (zeros at Gauss points)',
  vcjh: 'VCJH family, parameter $c$',
};
