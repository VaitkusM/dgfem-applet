/**
 * @file Sparse matrices in CSR (compressed sparse row) format.
 *
 * Finite element matrices are sparse: row i only has entries for unknowns
 * that share an element with unknown i. CSR stores, row after row, only the
 * non-zeros:
 *   rowPtr[i] .. rowPtr[i+1]-1  index the entries of row i,
 *   colIdx[k]  is the column of entry k,  vals[k]  its value.
 * (Think of it as an adjacency list of the mesh graph with weights.)
 *
 * Matrices are assembled through a COO ("coordinate", triplet) builder that
 * accepts repeated (i,j) pairs and sums them — exactly what element-by-element
 * assembly produces.
 */

/** @typedef {{n: number, m: number, rowPtr: Int32Array, colIdx: Int32Array, vals: Float64Array}} CSR */

/** Triplet accumulator: add(i, j, v) any number of times, then toCSR(). */
export class SparseBuilder {
  /**
   * @param {number} n rows
   * @param {number} [m=n] columns
   * @param {number} [capacity=16]
   */
  constructor(n, m = n, capacity = 16) {
    this.n = n; this.m = m;
    this.I = new Int32Array(capacity);
    this.J = new Int32Array(capacity);
    this.V = new Float64Array(capacity);
    this.len = 0;
  }
  /** Add v to entry (i, j). */
  add(i, j, v) {
    if (this.len === this.I.length) {
      const c = this.I.length * 2;
      const I = new Int32Array(c), J = new Int32Array(c), V = new Float64Array(c);
      I.set(this.I); J.set(this.J); V.set(this.V);
      this.I = I; this.J = J; this.V = V;
    }
    this.I[this.len] = i; this.J[this.len] = j; this.V[this.len] = v; this.len++;
  }
  /**
   * Add a dense local block: entries K[a*nl+b] go to (rows[a], cols[b]).
   * Negative indices are skipped (convenient for eliminated DOFs).
   */
  addBlock(rows, cols, K) {
    const nr = rows.length, nc = cols.length;
    for (let a = 0; a < nr; a++) {
      const i = rows[a];
      if (i < 0) continue;
      for (let b = 0; b < nc; b++) {
        const j = cols[b];
        if (j < 0) continue;
        const v = K[a * nc + b];
        if (v !== 0) this.add(i, j, v);
      }
    }
  }
  /** Convert to CSR, summing duplicates and sorting columns within rows. */
  toCSR() {
    const { n, m, len, I, J, V } = this;
    const count = new Int32Array(n + 1);
    for (let k = 0; k < len; k++) count[I[k] + 1]++;
    for (let i = 0; i < n; i++) count[i + 1] += count[i];
    const pos = Int32Array.from(count);
    const cj = new Int32Array(len), cv = new Float64Array(len);
    for (let k = 0; k < len; k++) { const p = pos[I[k]]++; cj[p] = J[k]; cv[p] = V[k]; }
    // sort each row by column and merge duplicates
    const rowPtr = new Int32Array(n + 1);
    const outJ = new Int32Array(len), outV = new Float64Array(len);
    let nnz = 0;
    const order = [];
    for (let i = 0; i < n; i++) {
      const s = count[i], e = count[i + 1];
      order.length = 0;
      for (let k = s; k < e; k++) order.push(k);
      order.sort((a, b) => cj[a] - cj[b]);
      let last = -1;
      for (const k of order) {
        if (cj[k] === last) outV[nnz - 1] += cv[k];
        else { outJ[nnz] = cj[k]; outV[nnz] = cv[k]; nnz++; last = cj[k]; }
      }
      rowPtr[i + 1] = nnz;
    }
    return { n, m, rowPtr, colIdx: outJ.slice(0, nnz), vals: outV.slice(0, nnz) };
  }
}

/** y = A x */
export function csrMatVec(A, x, y) {
  y = y || new Float64Array(A.n);
  const { rowPtr, colIdx, vals } = A;
  for (let i = 0; i < A.n; i++) {
    let s = 0;
    for (let k = rowPtr[i]; k < rowPtr[i + 1]; k++) s += vals[k] * x[colIdx[k]];
    y[i] = s;
  }
  return y;
}

/** Entry A_ij (0 if not stored). */
export function csrGet(A, i, j) {
  for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) if (A.colIdx[k] === j) return A.vals[k];
  return 0;
}

/** Diagonal of A. */
export function csrDiag(A) {
  const d = new Float64Array(A.n);
  for (let i = 0; i < A.n; i++) d[i] = csrGet(A, i, i);
  return d;
}

/** Dense copy (row-major) — for tests and small illustrations only. */
export function csrToDense(A) {
  const D = new Float64Array(A.n * A.m);
  for (let i = 0; i < A.n; i++)
    for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) D[i * A.m + A.colIdx[k]] += A.vals[k];
  return D;
}

/** Build CSR from a dense matrix (drops exact zeros). */
export function denseToCsr(D, n, m = n) {
  const B = new SparseBuilder(n, m, n * 4);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) if (D[i * m + j] !== 0) B.add(i, j, D[i * m + j]);
  return B.toCSR();
}

/** Transpose. */
export function csrTranspose(A) {
  const B = new SparseBuilder(A.m, A.n, A.vals.length);
  for (let i = 0; i < A.n; i++)
    for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) B.add(A.colIdx[k], i, A.vals[k]);
  return B.toCSR();
}

/** max |A_ij − A_ji| / max |A_ij| — 0 for a symmetric matrix. */
export function csrSymmetryError(A) {
  const T = csrTranspose(A);
  let num = 0, den = 0;
  for (let i = 0; i < A.n; i++)
    for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      const j = A.colIdx[k];
      num = Math.max(num, Math.abs(A.vals[k] - csrGet(T, i, j)));
      den = Math.max(den, Math.abs(A.vals[k]));
    }
  for (let i = 0; i < T.n; i++)
    for (let k = T.rowPtr[i]; k < T.rowPtr[i + 1]; k++) {
      num = Math.max(num, Math.abs(T.vals[k] - csrGet(A, i, T.colIdx[k])));
    }
  return den === 0 ? 0 : num / den;
}

/**
 * Extract the sub-matrix A[rows, cols].
 * @param {CSR} A
 * @param {ArrayLike<number>} rows
 * @param {ArrayLike<number>} cols
 * @returns {CSR}
 */
export function csrExtract(A, rows, cols) {
  const map = new Int32Array(A.m).fill(-1);
  for (let c = 0; c < cols.length; c++) map[cols[c]] = c;
  const B = new SparseBuilder(rows.length, cols.length, A.vals.length);
  for (let r = 0; r < rows.length; r++) {
    const i = rows[r];
    for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      const c = map[A.colIdx[k]];
      if (c >= 0) B.add(r, c, A.vals[k]);
    }
  }
  return B.toCSR();
}

/** Symmetric permutation  B = P A Pᵀ  with  B[i][j] = A[perm[i]][perm[j]]. */
export function csrPermute(A, perm) {
  const inv = new Int32Array(A.n);
  for (let i = 0; i < A.n; i++) inv[perm[i]] = i;
  const B = new SparseBuilder(A.n, A.m, A.vals.length);
  for (let i = 0; i < A.n; i++)
    for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) B.add(inv[i], inv[A.colIdx[k]], A.vals[k]);
  return B.toCSR();
}

/** Lower and upper bandwidth: max(i−j) and max(j−i) over stored entries. */
export function csrBandwidth(A) {
  let kl = 0, ku = 0;
  for (let i = 0; i < A.n; i++)
    for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      const d = A.colIdx[k] - i;
      if (d > ku) ku = d;
      if (-d > kl) kl = -d;
    }
  return { kl, ku };
}

/** C = αA + βB (same shape). */
export function csrAdd(A, B, alpha = 1, beta = 1) {
  const S = new SparseBuilder(A.n, A.m, A.vals.length + B.vals.length);
  for (let i = 0; i < A.n; i++) {
    for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) S.add(i, A.colIdx[k], alpha * A.vals[k]);
    for (let k = B.rowPtr[i]; k < B.rowPtr[i + 1]; k++) S.add(i, B.colIdx[k], beta * B.vals[k]);
  }
  return S.toCSR();
}

/** Number of stored non-zeros. */
export const nnz = (A) => A.rowPtr[A.n];
