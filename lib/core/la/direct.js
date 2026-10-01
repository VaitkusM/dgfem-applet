/**
 * @file Sparse direct solver: reverse Cuthill–McKee reordering + banded LU
 *       with partial pivoting.
 *
 * Why: Gaussian elimination on a sparse matrix only creates fill-in inside
 * the band |i − j| ≤ bandwidth. Renumbering the unknowns (RCM = breadth-first
 * search of the matrix graph, like a flood fill, then reversed) makes the
 * band narrow, so the cost drops from O(n³) to O(n · kl · (kl + ku)).
 *
 * The band LU follows LAPACK's dgbtrf idea: row interchanges are restricted to
 * the kl rows below the pivot, which can grow the upper bandwidth to ku + kl.
 * Works for symmetric positive definite AND indefinite systems (saddle-point
 * problems of mixed methods, under-penalised DG, …).
 */
import { csrPermute, csrBandwidth } from './sparse.js';

/**
 * Reverse Cuthill–McKee ordering of the (symmetrised) graph of A.
 * @param {import('./sparse.js').CSR} A
 * @returns {Int32Array} perm, new index i ↦ old index perm[i]
 */
export function rcm(A) {
  const n = A.n;
  // symmetric adjacency lists
  const adj = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i++)
    for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) {
      const j = A.colIdx[k];
      if (j !== i) { adj[i].push(j); adj[j].push(i); }
    }
  for (let i = 0; i < n; i++) adj[i] = [...new Set(adj[i])];
  const deg = adj.map((a) => a.length);
  const visited = new Uint8Array(n);
  const order = [];
  const bfs = (start) => {
    // returns the BFS levels' last node (used to find a pseudo-peripheral node)
    const dist = new Map([[start, 0]]);
    const q = [start];
    let last = start;
    for (let h = 0; h < q.length; h++) {
      const v = q[h];
      for (const w of adj[v]) if (!dist.has(w)) { dist.set(w, dist.get(v) + 1); q.push(w); last = w; }
    }
    return { last, ecc: dist.get(last) };
  };
  const nodesByDeg = Array.from({ length: n }, (_, i) => i).sort((a, b) => deg[a] - deg[b]);
  for (const seed of nodesByDeg) {
    if (visited[seed]) continue;
    // pseudo-peripheral start: repeat BFS from farthest node while eccentricity grows
    let start = seed, { last, ecc } = bfs(seed);
    for (let it = 0; it < 5; it++) {
      const r = bfs(last);
      if (r.ecc <= ecc) break;
      start = last; last = r.last; ecc = r.ecc;
    }
    const q = [start];
    visited[start] = 1;
    for (let h = 0; h < q.length; h++) {
      const v = q[h];
      order.push(v);
      const nb = adj[v].filter((w) => !visited[w]).sort((a, b) => deg[a] - deg[b]);
      for (const w of nb) { visited[w] = 1; q.push(w); }
    }
  }
  order.reverse();
  return Int32Array.from(order);
}

/**
 * Banded LU factorisation with partial pivoting.
 * Row i is stored in a window of width W = 2kl + ku + 1 starting at column i − kl.
 * @param {import('./sparse.js').CSR} A
 * @returns {{n:number, kl:number, ku:number, W:number, band:Float64Array, piv:Int32Array, singular:boolean}}
 */
export function bandLUFactor(A) {
  const n = A.n;
  const { kl, ku } = csrBandwidth(A);
  const W = 2 * kl + ku + 1;
  const band = new Float64Array(n * W);
  const at = (i, j) => i * W + (j - i + kl); // index of (i,j) inside row i's window
  for (let i = 0; i < n; i++)
    for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) band[at(i, A.colIdx[k])] += A.vals[k];
  const piv = new Int32Array(n);
  let singular = false;
  for (let k = 0; k < n; k++) {
    const rmax = Math.min(n - 1, k + kl);
    let p = k, best = Math.abs(band[at(k, k)]);
    for (let r = k + 1; r <= rmax; r++) {
      const v = Math.abs(band[at(r, k)]);
      if (v > best) { best = v; p = r; }
    }
    piv[k] = p;
    const cmax = Math.min(n - 1, k + ku + kl);
    if (p !== k) {
      for (let j = k; j <= cmax; j++) {
        const a = at(k, j), b = at(p, j);
        const t = band[a]; band[a] = band[b]; band[b] = t;
      }
    }
    const d = band[at(k, k)];
    if (d === 0) { singular = true; continue; }
    for (let r = k + 1; r <= rmax; r++) {
      const irk = at(r, k);
      const f = band[irk] / d;
      band[irk] = f; // store multiplier L_rk
      if (f === 0) continue;
      const rowK = k * W - k + kl, rowR = r * W - r + kl; // add j to get index
      for (let j = k + 1; j <= cmax; j++) band[rowR + j] -= f * band[rowK + j];
    }
  }
  return { n, kl, ku, W, band, piv, singular };
}

/**
 * Solve with a banded LU factorisation.
 * @param {ReturnType<typeof bandLUFactor>} F
 * @param {ArrayLike<number>} b
 * @returns {Float64Array}
 */
export function bandLUSolve(F, b) {
  const { n, kl, ku, W, band, piv } = F;
  const x = Float64Array.from(b);
  const at = (i, j) => i * W + (j - i + kl);
  for (let k = 0; k < n; k++) {
    const p = piv[k];
    if (p !== k) { const t = x[k]; x[k] = x[p]; x[p] = t; }
    const xk = x[k];
    if (xk === 0) continue;
    const rmax = Math.min(n - 1, k + kl);
    for (let r = k + 1; r <= rmax; r++) x[r] -= band[at(r, k)] * xk;
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    const cmax = Math.min(n - 1, i + ku + kl);
    for (let j = i + 1; j <= cmax; j++) s -= band[at(i, j)] * x[j];
    x[i] = s / band[at(i, i)];
  }
  return x;
}

/**
 * Factorise a sparse matrix (with RCM reordering) and return a solver.
 * @param {import('./sparse.js').CSR} A
 * @param {{reorder?: boolean}} [opts]
 * @returns {{solve: (b: ArrayLike<number>) => Float64Array, singular: boolean, kl: number, ku: number, perm: Int32Array|null}}
 */
export function sparseLU(A, opts = {}) {
  const reorder = opts.reorder !== false;
  const perm = reorder ? rcm(A) : null;
  const B = perm ? csrPermute(A, perm) : A;
  const F = bandLUFactor(B);
  const n = A.n;
  return {
    singular: F.singular, kl: F.kl, ku: F.ku, perm,
    solve(b) {
      if (!perm) return bandLUSolve(F, b);
      const bp = new Float64Array(n);
      for (let i = 0; i < n; i++) bp[i] = b[perm[i]];
      const xp = bandLUSolve(F, bp);
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) x[perm[i]] = xp[i];
      return x;
    },
  };
}

/** Convenience: solve A x = b with RCM + banded LU. */
export function sparseSolve(A, b) {
  return sparseLU(A).solve(b);
}
