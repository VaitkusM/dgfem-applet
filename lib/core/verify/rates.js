/**
 * @file Convergence-rate utilities.
 *
 * If a method is "of order r", its error behaves like e(h) ≈ C h^r for small
 * mesh size h. On a log–log plot this is a straight line of slope r:
 *   log e = log C + r log h.
 * We estimate r from computed (h, e) pairs either pairwise,
 *   r_k = log(e_k / e_{k−1}) / log(h_k / h_{k−1}),
 * or by a least-squares line fit through the last few points.
 */

/**
 * Pairwise observed rates.
 * @param {number[]} hs mesh sizes (decreasing)
 * @param {number[]} es errors
 * @returns {number[]} rates (length n−1)
 */
export function pairwiseRates(hs, es) {
  const r = [];
  for (let k = 1; k < hs.length; k++) r.push(Math.log(es[k] / es[k - 1]) / Math.log(hs[k] / hs[k - 1]));
  return r;
}

/**
 * Least-squares slope of log e versus log h through the last `last` points.
 * @param {number[]} hs
 * @param {number[]} es
 * @param {number} [last=3]
 * @returns {number}
 */
export function fitRate(hs, es, last = 3) {
  const n = hs.length, s = Math.max(0, n - last);
  let sx = 0, sy = 0, sxx = 0, sxy = 0, m = 0;
  for (let k = s; k < n; k++) {
    const x = Math.log(hs[k]), y = Math.log(es[k]);
    sx += x; sy += y; sxx += x * x; sxy += x * y; m++;
  }
  return (m * sxy - sx * sy) / (m * sxx - sx * sx);
}
