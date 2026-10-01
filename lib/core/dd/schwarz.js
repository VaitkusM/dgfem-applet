/**
 * @file Overlapping Schwarz domain decomposition methods (chapter 10).
 *
 * Notation (A u = b is the global n×n SPD system of lib/core/dd/problem.js):
 *  - R_k   : restriction to the unknowns of the overlapping subdomain Ω'_k
 *            (a 0/1 matrix picking the entries `dofs`); R_kᵀ = extension by zero.
 *  - A_k = R_k A R_kᵀ : the local stiffness matrix = Poisson problem on Ω'_k with
 *            homogeneous Dirichlet conditions on ∂Ω'_k. Factorised once (sparse LU).
 *  - D_k   : 0/1 diagonal "partition of unity" (core flags), Σ_k R_kᵀ D_k R_k = I.
 *  - R_0ᵀ = Z : n × n_c matrix whose columns are coarse basis functions
 *            (interpolated to the fine grid); A_0 = Zᵀ A Z (dense, small).
 *
 * Methods implemented:
 *  - multiplicative Schwarz (stationary; for 2 subdomains = Schwarz's 1870
 *    alternating method):   for k = 1..K:  u ← u + R_kᵀ A_k⁻¹ R_k (b − A u)
 *    (with a coarse space, the coarse correction is applied first);
 *  - additive Schwarz preconditioner (one or two level), used inside CG:
 *        M_ASM⁻¹ = Σ_k R_kᵀ A_k⁻¹ R_k  (+ R_0ᵀ A_0⁻¹ R_0);
 *    it is symmetric positive definite, so PCG applies;
 *  - restricted additive Schwarz (RAS), used inside GMRES (it is NOT symmetric):
 *        M_RAS⁻¹ = Σ_k R_kᵀ D_k A_k⁻¹ R_k  (+ R_0ᵀ A_0⁻¹ R_0).
 *
 * Coarse space ('q1'): bilinear (Q1) hat functions on the coarse grid whose nodes are
 * the subdomain corners (only interior coarse nodes, since u = 0 on ∂Ω), evaluated
 * at the fine vertices (= nodal interpolation of the coarse function on the fine grid).
 * The coarse grid has no interior node if Kx = 1 or Ky = 1 (strips); then n_c = 0
 * and the "two-level" method silently reduces to the one-level one.
 */
import { sparseLU } from '../la/direct.js';
import { csrExtract, csrMatVec } from '../la/sparse.js';
import { luFactor, luSolve } from '../la/dense.js';
import { cg, gmres } from '../la/krylov.js';
import { freeIndex } from './problem.js';

/**
 * Q1 coarse basis Z = R_0ᵀ: one column per interior coarse node (subdomain corner
 * (cutsX[mx], cutsY[my]), 1 ≤ mx < Kx, 1 ≤ my < Ky), ordered my-major.
 * Column = bilinear hat function of that node evaluated at the free fine vertices.
 * @param {import('./problem.js').DDProblem} prob
 * @param {import('./partition.js').BoxPartition} part
 * @returns {Float64Array[]} columns, each of length prob.n; (Kx−1)(Ky−1) of them
 */
export function coarseBasis(prob, part) {
  const { N, n } = prob, cols = [];
  // 1D hat on the coarse breakpoints c[0..K]: value at fine index i of the hat centred at c[m]
  const hat = (c, m, i) => {
    if (i <= c[m - 1] || i >= c[m + 1]) return 0;
    return i <= c[m] ? (i - c[m - 1]) / (c[m] - c[m - 1]) : (c[m + 1] - i) / (c[m + 1] - c[m]);
  };
  for (let my = 1; my < part.Ky; my++)
    for (let mx = 1; mx < part.Kx; mx++) {
      const z = new Float64Array(n);
      for (let j = Math.max(1, part.cutsY[my - 1] + 1); j < Math.min(N, part.cutsY[my + 1]); j++)
        for (let i = Math.max(1, part.cutsX[mx - 1] + 1); i < Math.min(N, part.cutsX[mx + 1]); i++)
          z[freeIndex(N, i, j)] = hat(part.cutsX, mx, i) * hat(part.cutsY, my, j);
      cols.push(z);
    }
  return cols;
}

/**
 * @typedef {Object} SchwarzSetup
 * @property {import('./problem.js').DDProblem} prob
 * @property {import('./partition.js').BoxPartition} part
 * @property {{dofs: Int32Array, core: Uint8Array, solve: (r: ArrayLike<number>) => Float64Array}[]} locals
 * @property {{Z: Float64Array[], nc: number, F: object}|null} coarse
 */

/**
 * Factorise the local problems (and the coarse problem).
 * @param {import('./problem.js').DDProblem} prob
 * @param {import('./partition.js').BoxPartition} part
 * @param {{coarse?: 'none'|'q1'}} [opts]
 * @returns {SchwarzSetup}
 */
export function schwarzSetup(prob, part, opts = {}) {
  const locals = part.subs.map((s) => {
    const F = sparseLU(csrExtract(prob.A, s.dofs, s.dofs));
    return { dofs: s.dofs, core: s.core, solve: (r) => F.solve(r) };
  });
  let coarse = null;
  const type = opts.coarse ?? 'none';
  if (type !== 'none') {
    if (type !== 'q1') throw new Error(`unknown coarse space ${type}`);
    const Z = coarseBasis(prob, part), nc = Z.length;
    if (nc > 0) {
      const AZ = Z.map((z) => csrMatVec(prob.A, z));
      const A0 = new Float64Array(nc * nc);
      for (let a = 0; a < nc; a++) for (let c = 0; c < nc; c++) {
        let s = 0;
        for (let i = 0; i < prob.n; i++) s += Z[a][i] * AZ[c][i];
        A0[a * nc + c] = s;
      }
      coarse = { Z, nc, F: luFactor(A0, nc) };
    }
  }
  return { prob, part, locals, coarse };
}

/** z += R_0ᵀ A_0⁻¹ R_0 r  (R_0 = Zᵀ). */
function addCoarse(dd, r, z) {
  const { Z, nc, F } = dd.coarse;
  const r0 = new Float64Array(nc);
  for (let a = 0; a < nc; a++) { let s = 0; const za = Z[a]; for (let i = 0; i < r.length; i++) s += za[i] * r[i]; r0[a] = s; }
  const u0 = luSolve(F, r0);
  for (let a = 0; a < nc; a++) { const za = Z[a], c = u0[a]; if (c !== 0) for (let i = 0; i < z.length; i++) z[i] += c * za[i]; }
}

/**
 * Additive Schwarz preconditioner  z = M_ASM⁻¹ r = Σ_k R_kᵀ A_k⁻¹ R_k r (+ coarse).
 * @param {SchwarzSetup} dd
 * @param {Float64Array} r
 * @param {Float64Array} z output
 */
export function applyASM(dd, r, z) {
  z.fill(0);
  for (const L of dd.locals) {
    const rl = Float64Array.from(L.dofs, (f) => r[f]);
    const ul = L.solve(rl);
    for (let l = 0; l < L.dofs.length; l++) z[L.dofs[l]] += ul[l];
  }
  if (dd.coarse) addCoarse(dd, r, z);
}

/**
 * Restricted additive Schwarz  z = M_RAS⁻¹ r = Σ_k R_kᵀ D_k A_k⁻¹ R_k r (+ coarse):
 * every subdomain solves on its overlapping region but only writes back the
 * values it owns (like a renderer tile that reads its halo but writes only its tile).
 * @param {SchwarzSetup} dd
 * @param {Float64Array} r
 * @param {Float64Array} z output
 */
export function applyRAS(dd, r, z) {
  z.fill(0);
  for (const L of dd.locals) {
    const rl = Float64Array.from(L.dofs, (f) => r[f]);
    const ul = L.solve(rl);
    for (let l = 0; l < L.dofs.length; l++) if (L.core[l]) z[L.dofs[l]] += ul[l];
  }
  if (dd.coarse) addCoarse(dd, r, z);
}

/** Local residual R_k (b − A x), computed only on the rows of subdomain k. */
function localResidual(A, b, x, dofs) {
  const r = new Float64Array(dofs.length);
  for (let l = 0; l < dofs.length; l++) {
    const i = dofs[l];
    let s = b[i];
    for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) s -= A.vals[k] * x[A.colIdx[k]];
    r[l] = s;
  }
  return r;
}

/**
 * One sweep of multiplicative Schwarz (in place):
 *   (coarse correction first, if any) then for k = 1..K:  x ← x + R_kᵀ A_k⁻¹ R_k (b − A x).
 * Each local solve uses the newest values of its neighbours, like Gauss–Seidel.
 * @param {SchwarzSetup} dd
 * @param {ArrayLike<number>} b
 * @param {Float64Array} x
 */
export function multiplicativeSweep(dd, b, x) {
  const A = dd.prob.A;
  if (dd.coarse) {
    const r = new Float64Array(x.length);
    csrMatVec(A, x, r);
    for (let i = 0; i < r.length; i++) r[i] = b[i] - r[i];
    addCoarse(dd, r, x);
  }
  for (const L of dd.locals) {
    const ul = L.solve(localResidual(A, b, x, L.dofs));
    for (let l = 0; l < L.dofs.length; l++) x[L.dofs[l]] += ul[l];
  }
}

const norm = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0));

/**
 * Solve A x = b with a Schwarz method, starting from x = 0.
 * @param {SchwarzSetup} dd
 * @param {ArrayLike<number>} b
 * @param {{method: 'multiplicative'|'asm-cg'|'ras-gmres', tol?: number, maxIter?: number, keepIterates?: boolean}} o
 * @returns {{x: Float64Array, iters: number, converged: boolean, history: number[], iterates?: Float64Array[]}}
 *   history[k] = ‖b − A x_k‖ / ‖b‖ (k = 0 initial); iterates[k] = x_k if requested
 */
export function solveSchwarz(dd, b, o) {
  const tol = o.tol ?? 1e-8, maxIter = o.maxIter ?? 500, n = b.length;
  const A = dd.prob.A;
  if (o.method === 'multiplicative') {
    const x = new Float64Array(n), r = new Float64Array(n), bn = norm(b) || 1;
    const history = [1], iterates = o.keepIterates ? [x.slice()] : undefined;
    let k = 0;
    while (k < maxIter && history[k] > tol) {
      multiplicativeSweep(dd, b, x);
      csrMatVec(A, x, r);
      for (let i = 0; i < n; i++) r[i] = b[i] - r[i];
      history.push(norm(r) / bn);
      iterates?.push(x.slice());
      k++;
    }
    return { x, iters: k, converged: history[k] <= tol, history, iterates };
  }
  if (o.method === 'asm-cg') {
    const iterates = o.keepIterates ? [new Float64Array(n)] : undefined;
    const res = cg(A, b, { tol, maxIter, precond: (r, z) => applyASM(dd, r, z), onIter: iterates ? (k, rel, x) => iterates.push(x.slice()) : undefined });
    return { ...res, iterates };
  }
  if (o.method === 'ras-gmres') {
    const precond = (r, z) => applyRAS(dd, r, z);
    const res = gmres(A, b, { tol, maxIter, restart: Math.max(maxIter, 1), precond });
    let iterates;
    if (o.keepIterates) {
      // restarted GMRES only forms x at the end of a cycle: recompute x_k by running k steps
      iterates = [new Float64Array(n)];
      for (let k = 1; k <= res.iters; k++) iterates.push(gmres(A, b, { tol: 0, maxIter: k, restart: k, precond }).x);
    }
    return { ...res, iterates };
  }
  throw new Error(`unknown method ${o.method}`);
}

/**
 * Schwarz's alternating method for  −u'' = f₀ (constant) on (0, 1),
 * u(0) = uL, u(1) = uR, with overlapping subdomains Ω₁ = (0, β), Ω₂ = (α, 1), α < β.
 * Local problems are solved exactly (u is a quadratic on each subdomain):
 *   u₁ⁿ on (0, β):  −u'' = f₀,  u(0) = uL,  u(β) = u₂ⁿ⁻¹(β)
 *   u₂ⁿ on (α, 1):  −u'' = f₀,  u(α) = u₁ⁿ(α),  u(1) = uR.
 * The error is linear on each subdomain and shrinks by the factor
 *   ρ = α(1 − β) / (β(1 − α))  per iteration (both half-steps).
 * @param {{alpha: number, beta: number, f0?: number, uL?: number, uR?: number, g0?: number, iters: number}} o
 *   g0 = initial guess for u at x = β
 * @returns {{u1: number[][], u2: number[][], exact: number[], rho: number, gammaB: number[]}}
 *   quadratics as coefficient triples [q, c1, c0] (u = q x² + c1 x + c0); gammaB[n] = u₂ⁿ(β) (gammaB[0] = g0)
 */
export function alternatingSchwarz1D(o) {
  const { alpha, beta, iters } = o, f0 = o.f0 ?? 0, uL = o.uL ?? 0, uR = o.uR ?? 0;
  const q = -f0 / 2;
  // quadratic with leading coefficient q through (x0, v0) and (x1, v1)
  const fit = (x0, v0, x1, v1) => {
    const c1 = (v1 - q * x1 * x1 - (v0 - q * x0 * x0)) / (x1 - x0);
    return [q, c1, v0 - q * x0 * x0 - c1 * x0];
  };
  const ev = (c, x) => c[0] * x * x + c[1] * x + c[2];
  const u1 = [], u2 = [], gammaB = [o.g0 ?? 0];
  let g = gammaB[0];
  for (let n = 0; n < iters; n++) {
    const a = fit(0, uL, beta, g);
    const c = fit(alpha, ev(a, alpha), 1, uR);
    u1.push(a); u2.push(c);
    g = ev(c, beta); gammaB.push(g);
  }
  return { u1, u2, exact: fit(0, uL, 1, uR), rho: (alpha * (1 - beta)) / (beta * (1 - alpha)), gammaB };
}
