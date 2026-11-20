/**
 * @file Small helpers shared by the elliptic chapters 8 (interior penalty DG)
 *       and 9 (mixed methods, hybridization): the list of manufactured
 *       solutions offered by the widgets and a fixed colour range for them.
 */

/** Manufactured solutions of lib/core/verify/mms.js (POISSON_MMS keys) offered in the selects. */
export const PROBLEMS = [
  { value: 'wave', label: 'u = cos(πx)·eʸ' },
  { value: 'sinsin', label: 'u = sin(πx)·sin(πy)' },
  { value: 'peak', label: 'Gaussian peak' },
];

/**
 * Min/max of a function on a 61×61 grid of the unit square (a fixed colour
 * range per problem, so that colours do not change with the mesh or method).
 * @param {(x: number, y: number) => number} u
 * @returns {[number, number]} [min, max]
 */
export function exactRange(u) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i <= 60; i++) for (let j = 0; j <= 60; j++) { const v = u(i / 60, j / 60); lo = Math.min(lo, v); hi = Math.max(hi, v); }
  return [lo, hi];
}

