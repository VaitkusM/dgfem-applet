/**
 * @file Extreme eigenvalues and condition numbers of sparse symmetric matrices.
 *
 * The (spectral) condition number κ(A) = λ_max / λ_min of an SPD matrix
 * measures how much a linear solve can amplify errors, and how slowly CG
 * converges (≈ √κ iterations). Several chapters plot κ(A) (e.g. CutFEM with
 * and without ghost penalty), so we estimate it with the Lanczos method:
 *  - λ_max  from Lanczos on A,
 *  - λ_min  from Lanczos on A⁻¹ (each step = one sparse LU solve).
 * Lanczos builds an orthonormal Krylov basis in which A is tridiagonal; the
 * extreme eigenvalues of that small tridiagonal matrix converge very fast to
 * the extreme eigenvalues of A.
 */
import { asOperator } from './krylov.js';
import { sparseLU } from './direct.js';
import { symEig } from './dense.js';

/**
 * Lanczos with full re-orthogonalisation; returns Ritz values (ascending).
 * @param {(x: Float64Array, y: Float64Array) => void} op symmetric operator
 * @param {number} n size
 * @param {number} [k=60] Krylov dimension
 * @returns {Float64Array}
 */
export function lanczosRitz(op, n, k = 60) {
  k = Math.min(k, n);
  const Q = [];
  const alpha = [], beta = [];
  let q = new Float64Array(n);
  // deterministic pseudo-random start vector
  let seed = 12345;
  for (let i = 0; i < n; i++) { seed = (seed * 1103515245 + 12345) & 0x7fffffff; q[i] = seed / 0x7fffffff - 0.5; }
  let nq = Math.sqrt(q.reduce((s, v) => s + v * v, 0));
  for (let i = 0; i < n; i++) q[i] /= nq;
  const w = new Float64Array(n);
  for (let j = 0; j < k; j++) {
    Q.push(q);
    op(q, w);
    let a = 0;
    for (let i = 0; i < n; i++) a += w[i] * q[i];
    alpha.push(a);
    // full re-orthogonalisation (twice for safety)
    for (let pass = 0; pass < 2; pass++)
      for (const v of Q) {
        let h = 0;
        for (let i = 0; i < n; i++) h += w[i] * v[i];
        for (let i = 0; i < n; i++) w[i] -= h * v[i];
      }
    const b = Math.sqrt(w.reduce((s, v) => s + v * v, 0));
    if (j === k - 1 || b < 1e-14 * Math.abs(a)) break;
    beta.push(b);
    q = new Float64Array(n);
    for (let i = 0; i < n; i++) q[i] = w[i] / b;
  }
  const m = alpha.length, T = new Float64Array(m * m);
  for (let i = 0; i < m; i++) {
    T[i * m + i] = alpha[i];
    if (i + 1 < m) { T[i * m + i + 1] = beta[i]; T[(i + 1) * m + i] = beta[i]; }
  }
  return symEig(T, m).values;
}

/**
 * Estimate λ_min, λ_max and κ = λ_max/λ_min of a sparse SPD matrix.
 * If A is indefinite the returned λ_min is the eigenvalue closest to zero
 * (possibly negative) and κ uses absolute values.
 * @param {import('./sparse.js').CSR} A
 * @param {{k?: number}} [opts]
 * @returns {{lmin: number, lmax: number, kappa: number}}
 */
export function conditionEstimate(A, opts = {}) {
  const k = opts.k ?? 60;
  const op = asOperator(A);
  const rMax = lanczosRitz(op, A.n, k);
  const lmax = Math.max(Math.abs(rMax[0]), Math.abs(rMax[rMax.length - 1]));
  const F = sparseLU(A);
  const inv = (x, y) => y.set(F.solve(x));
  const rInv = lanczosRitz(inv, A.n, k);
  // largest |eigenvalue| of A⁻¹ ↔ eigenvalue of A closest to 0
  const muLo = rInv[0], muHi = rInv[rInv.length - 1];
  const mu = Math.abs(muLo) > Math.abs(muHi) ? muLo : muHi;
  const lmin = 1 / mu;
  return { lmin, lmax, kappa: lmax / Math.abs(lmin) };
}
