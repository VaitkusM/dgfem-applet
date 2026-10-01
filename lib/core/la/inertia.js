/**
 * @file Inertia of a symmetric matrix: how many eigenvalues are negative, zero
 *       and positive — without computing any eigenvalue.
 *
 * Sylvester's law of inertia: a congruence A ↦ L D Lᵀ (L invertible) does not
 * change the number of negative/zero/positive eigenvalues. Gaussian elimination
 * without pivoting produces exactly such a factorisation A = L D Lᵀ with D
 * diagonal, so the signs of the pivots d_k tell the inertia. (This is the
 * classical way to count eigenvalues below a shift σ: factor A − σI.)
 * Without pivoting the factorisation needs non-zero leading minors; a pivot
 * that is (numerically) zero is counted as zero and replaced by a tiny value.
 */

/**
 * Inertia of a dense symmetric matrix via LDLᵀ without pivoting. O(n³/3).
 * @param {Float64Array} A row-major n×n symmetric (not modified)
 * @param {number} n
 * @returns {{neg: number, zero: number, pos: number, pivots: Float64Array}}
 */
export function inertia(A, n) {
  const a = Float64Array.from(A), d = new Float64Array(n);
  let scale = 0;
  for (let i = 0; i < n * n; i++) scale = Math.max(scale, Math.abs(a[i]));
  const tiny = 1e-14 * (scale || 1);
  let neg = 0, zero = 0, pos = 0;
  for (let k = 0; k < n; k++) {
    let piv = a[k * n + k];
    if (Math.abs(piv) <= tiny) { zero++; piv = tiny; } else if (piv < 0) neg++; else pos++;
    d[k] = piv;
    for (let i = k + 1; i < n; i++) {
      const l = a[i * n + k] / piv;
      if (l === 0) continue;
      for (let j = k + 1; j <= i; j++) a[i * n + j] -= l * a[j * n + k]; // lower triangle only
    }
  }
  return { neg, zero, pos, pivots: d };
}
