/**
 * @file Krylov subspace iterative solvers: CG / PCG and restarted GMRES.
 *
 * Instead of factorising A, Krylov methods only need the action x ↦ A x
 * (a "matvec"). After k steps the iterate lies in span{b, Ab, …, A^{k−1} b}.
 *  - CG (conjugate gradients) needs A symmetric positive definite (SPD);
 *    its convergence speed is governed by the condition number κ(A).
 *  - GMRES works for any non-singular A; it minimises the residual over the
 *    Krylov space (restarted every `restart` steps to bound memory).
 * A preconditioner M ≈ A⁻¹ (cheap to apply) clusters the spectrum and speeds
 * things up — Jacobi (diagonal), Schwarz domain decomposition, …
 *
 * Operators are passed either as CSR matrices or as functions (x, y) => void
 * writing y = A x.
 */
import { csrMatVec, csrDiag } from './sparse.js';

/** @typedef {import('./sparse.js').CSR | ((x: Float64Array, y: Float64Array) => void)} Operator */
/** @typedef {(r: Float64Array, z: Float64Array) => void} Precond  writes z = M⁻¹ r */

/** Wrap a CSR matrix or function into a function (x, y) => void. */
export function asOperator(A) {
  if (typeof A === 'function') return A;
  return (x, y) => csrMatVec(A, x, y);
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const norm = (a) => Math.sqrt(dot(a, a));

/** Jacobi (diagonal) preconditioner for a CSR matrix. */
export function jacobiPrecond(A) {
  const d = csrDiag(A);
  return (r, z) => { for (let i = 0; i < r.length; i++) z[i] = d[i] !== 0 ? r[i] / d[i] : r[i]; };
}

/**
 * (Preconditioned) conjugate gradients.
 * @param {Operator} A SPD operator
 * @param {ArrayLike<number>} b
 * @param {{x0?: ArrayLike<number>, tol?: number, maxIter?: number, precond?: Precond, onIter?: (k: number, relres: number, x: Float64Array) => void}} [opts]
 * @returns {{x: Float64Array, iters: number, converged: boolean, history: number[]}}
 *          history = relative residual ‖r_k‖/‖b‖ after each iteration (index 0 = initial)
 */
export function cg(A, b, opts = {}) {
  const n = b.length, op = asOperator(A);
  const tol = opts.tol ?? 1e-10, maxIter = opts.maxIter ?? 10 * n;
  const x = opts.x0 ? Float64Array.from(opts.x0) : new Float64Array(n);
  const r = new Float64Array(n), z = new Float64Array(n), p = new Float64Array(n), Ap = new Float64Array(n);
  op(x, Ap);
  for (let i = 0; i < n; i++) r[i] = b[i] - Ap[i];
  const bn = norm(b) || 1;
  const history = [norm(r) / bn];
  if (history[0] <= tol) return { x, iters: 0, converged: true, history };
  const M = opts.precond || ((rr, zz) => zz.set(rr));
  M(r, z);
  p.set(z);
  let rz = dot(r, z);
  for (let k = 1; k <= maxIter; k++) {
    op(p, Ap);
    const pAp = dot(p, Ap);
    if (pAp === 0) break;
    const alpha = rz / pAp;
    for (let i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
    const rel = norm(r) / bn;
    history.push(rel);
    opts.onIter?.(k, rel, x);
    if (rel <= tol) return { x, iters: k, converged: true, history };
    M(r, z);
    const rzNew = dot(r, z);
    const beta = rzNew / rz;
    rz = rzNew;
    for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
  }
  return { x, iters: history.length - 1, converged: false, history };
}

/**
 * Restarted GMRES(m) with right preconditioning: solves A M⁻¹ y = b, x = M⁻¹ y,
 * so the monitored residual is the true residual ‖b − A x‖.
 * @param {Operator} A
 * @param {ArrayLike<number>} b
 * @param {{x0?: ArrayLike<number>, tol?: number, maxIter?: number, restart?: number, precond?: Precond, onIter?: (k: number, relres: number) => void}} [opts]
 * @returns {{x: Float64Array, iters: number, converged: boolean, history: number[]}}
 */
export function gmres(A, b, opts = {}) {
  const n = b.length, op = asOperator(A);
  const tol = opts.tol ?? 1e-10, maxIter = opts.maxIter ?? 1000, m = Math.min(opts.restart ?? 50, n);
  const M = opts.precond || ((rr, zz) => zz.set(rr));
  const x = opts.x0 ? Float64Array.from(opts.x0) : new Float64Array(n);
  const bn = norm(b) || 1;
  const r = new Float64Array(n), w = new Float64Array(n), z = new Float64Array(n);
  const V = Array.from({ length: m + 1 }, () => new Float64Array(n));
  const Z = Array.from({ length: m }, () => new Float64Array(n));
  const H = new Float64Array((m + 1) * m);
  const cs = new Float64Array(m), sn = new Float64Array(m), g = new Float64Array(m + 1);
  const history = [];
  let total = 0;
  const residual = () => { op(x, w); for (let i = 0; i < n; i++) r[i] = b[i] - w[i]; return norm(r); };
  let beta = residual();
  history.push(beta / bn);
  if (beta / bn <= tol) return { x, iters: 0, converged: true, history };
  while (total < maxIter) {
    for (let i = 0; i < n; i++) V[0][i] = r[i] / beta;
    g.fill(0); g[0] = beta; H.fill(0);
    let j = 0;
    for (; j < m && total < maxIter; j++) {
      total++;
      M(V[j], Z[j]);
      op(Z[j], w);
      for (let i = 0; i <= j; i++) { // modified Gram–Schmidt
        const h = dot(w, V[i]);
        H[i * m + j] = h;
        for (let l = 0; l < n; l++) w[l] -= h * V[i][l];
      }
      const hn = norm(w);
      H[(j + 1) * m + j] = hn;
      if (hn > 0) for (let l = 0; l < n; l++) V[j + 1][l] = w[l] / hn;
      for (let i = 0; i < j; i++) { // apply previous Givens rotations
        const a = H[i * m + j], c = H[(i + 1) * m + j];
        H[i * m + j] = cs[i] * a + sn[i] * c;
        H[(i + 1) * m + j] = -sn[i] * a + cs[i] * c;
      }
      const a = H[j * m + j], c = H[(j + 1) * m + j];
      const d = Math.hypot(a, c) || 1;
      cs[j] = a / d; sn[j] = c / d;
      H[j * m + j] = d; H[(j + 1) * m + j] = 0;
      g[j + 1] = -sn[j] * g[j];
      g[j] = cs[j] * g[j];
      const rel = Math.abs(g[j + 1]) / bn;
      history.push(rel);
      opts.onIter?.(total, rel);
      if (rel <= tol || hn === 0) { j++; break; }
    }
    // back substitution  H y = g  and update x += Z y
    const y = new Float64Array(j);
    for (let i = j - 1; i >= 0; i--) {
      let s = g[i];
      for (let l = i + 1; l < j; l++) s -= H[i * m + l] * y[l];
      y[i] = s / H[i * m + i];
    }
    for (let i = 0; i < j; i++) for (let l = 0; l < n; l++) x[l] += y[i] * Z[i][l];
    beta = residual();
    if (beta / bn <= tol * 1.0000001) return { x, iters: total, converged: true, history };
  }
  return { x, iters: total, converged: false, history };
}
