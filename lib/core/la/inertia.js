/**
 * @file Inertia of a symmetric matrix: how many eigenvalues are negative, zero
 *       and positive — without computing any eigenvalue.
 *
 * Sylvester's law of inertia: a congruence A ↦ S A Sᵀ (S invertible) does not
 * change the number of negative/zero/positive eigenvalues. Gaussian elimination
 * without pivoting produces exactly such a factorisation A = L D Lᵀ with D
 * diagonal, so the signs of the pivots d_k tell the inertia. (This is the
 * classical way to count eigenvalues below a shift σ: factor A − σI.)
 *
 * Without pivoting the factorisation needs non-zero leading minors. A zero
 * pivot does NOT mean a zero eigenvalue (think of [0 1; 1 0], eigenvalues ±1),
 * so as soon as a (numerically) zero pivot shows up we restart with a
 * symmetrically pivoted factorisation P A Pᵀ = L D Lᵀ in which D has 1×1 and
 * 2×2 blocks (Bunch–Parlett complete pivoting): the largest remaining diagonal
 * entry is used as a 1×1 pivot, or — if the off-diagonal entries dominate —
 * a 2×2 block [a b; b c] with b² > a c, which always contributes exactly one
 * positive and one negative eigenvalue. Symmetric row/column swaps are
 * congruences too, so the inertia is still read off D.
 */

/**
 * LDLᵀ without pivoting on the lower triangle; stops at the first tiny pivot.
 * @param {Float64Array} A row-major n×n symmetric (not modified)
 * @param {number} n
 * @param {number} tiny pivots with |d| ≤ tiny count as zero
 * @returns {{neg: number, zero: number, pos: number, pivots: Float64Array}|null} null if a tiny pivot occurred
 */
function inertiaNoPivot(A, n, tiny) {
  const a = Float64Array.from(A), d = new Float64Array(n);
  let neg = 0, pos = 0;
  for (let k = 0; k < n; k++) {
    const piv = a[k * n + k];
    if (Math.abs(piv) <= tiny) return null;
    if (piv < 0) neg++; else pos++;
    d[k] = piv;
    for (let i = k + 1; i < n; i++) {
      const l = a[i * n + k] / piv;
      if (l === 0) continue;
      for (let j = k + 1; j <= i; j++) a[i * n + j] -= l * a[j * n + k]; // lower triangle only
    }
  }
  return { neg, zero: 0, pos, pivots: d };
}

/**
 * Inertia by LDLᵀ with Bunch–Parlett (complete) pivoting, 1×1 and 2×2 blocks.
 * @param {Float64Array} A row-major n×n symmetric (not modified)
 * @param {number} n
 * @param {number} tiny remaining blocks whose largest entry is ≤ tiny count as zero
 * @returns {{neg: number, zero: number, pos: number, pivots: Float64Array}}
 */
function inertiaPivoted(A, n, tiny) {
  const a = Float64Array.from(A), d = new Float64Array(n);
  const ALPHA = (1 + Math.sqrt(17)) / 8; // Bunch–Parlett threshold ≈ 0.64
  /** symmetric swap of rows/columns p and q (full storage) */
  const swap = (p, q) => {
    if (p === q) return;
    for (let j = 0; j < n; j++) { const t = a[p * n + j]; a[p * n + j] = a[q * n + j]; a[q * n + j] = t; }
    for (let i = 0; i < n; i++) { const t = a[i * n + p]; a[i * n + p] = a[i * n + q]; a[i * n + q] = t; }
  };
  let neg = 0, zero = 0, pos = 0, k = 0;
  while (k < n) {
    // largest diagonal and largest off-diagonal entry of the remaining block
    let mu1 = 0, r = k, mu0 = 0, p = k, q = k;
    for (let i = k; i < n; i++) {
      const v = Math.abs(a[i * n + i]);
      if (v > mu1) { mu1 = v; r = i; }
      for (let j = k; j < i; j++) { const w = Math.abs(a[i * n + j]); if (w > mu0) { mu0 = w; p = i; q = j; } }
    }
    if (Math.max(mu0, mu1) <= tiny) { zero += n - k; break; } // remaining block is zero
    if (mu1 >= ALPHA * mu0) {
      // 1×1 pivot
      swap(k, r);
      const piv = a[k * n + k];
      if (piv < 0) neg++; else pos++;
      d[k] = piv;
      for (let i = k + 1; i < n; i++) {
        const l = a[i * n + k] / piv;
        if (l === 0) continue;
        for (let j = k + 1; j < n; j++) a[i * n + j] -= l * a[k * n + j];
      }
      k += 1;
    } else {
      // 2×2 pivot [e b; b c] on rows/columns (q, p), q < p; det = e c − b² < 0
      swap(k, q); swap(k + 1, p);
      const e = a[k * n + k], b = a[(k + 1) * n + k], c = a[(k + 1) * n + k + 1];
      const det = e * c - b * b;
      neg++; pos++;
      // store the two eigenvalues of the block (one negative, one positive)
      const m = 0.5 * (e + c), rad = Math.sqrt(0.25 * (e - c) * (e - c) + b * b);
      d[k] = m - rad; d[k + 1] = m + rad;
      for (let i = k + 2; i < n; i++) {
        // [l1 l2] = [a_ik a_i,k+1] · block⁻¹
        const x = a[i * n + k], y = a[i * n + k + 1];
        const l1 = (x * c - y * b) / det, l2 = (y * e - x * b) / det;
        if (l1 === 0 && l2 === 0) continue;
        for (let j = k + 2; j < n; j++) a[i * n + j] -= l1 * a[k * n + j] + l2 * a[(k + 1) * n + j];
      }
      k += 2;
    }
  }
  return { neg, zero, pos, pivots: d };
}

/**
 * Inertia of a dense symmetric matrix via LDLᵀ. O(n³/3) without pivoting;
 * if a (numerically) zero pivot is met, the factorisation is redone with
 * symmetric Bunch–Parlett pivoting (1×1 and 2×2 blocks), which is correct for
 * every symmetric matrix, including singular ones.
 * @param {Float64Array} A row-major n×n symmetric (not modified)
 * @param {number} n
 * @returns {{neg: number, zero: number, pos: number, pivots: Float64Array}}
 *   counts of negative / zero / positive eigenvalues; `pivots` are the diagonal
 *   entries of D in elimination order (for a 2×2 block: its two eigenvalues;
 *   entries belonging to a zero block are 0)
 */
export function inertia(A, n) {
  let scale = 0;
  for (let i = 0; i < n * n; i++) scale = Math.max(scale, Math.abs(A[i]));
  const tiny = 1e-14 * (scale || 1);
  return inertiaNoPivot(A, n, tiny) || inertiaPivoted(A, n, tiny);
}
