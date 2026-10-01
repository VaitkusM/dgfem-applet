/**
 * @file Non-overlapping domain decomposition: Schur complement (interface)
 *       system and the Dirichlet–Neumann iteration (chapter 10).
 *
 * Split the unknowns of A u = b (lib/core/dd/problem.js) into the interiors
 * I_1, …, I_K of the K non-overlapping subdomains and the interface Γ
 * (lib/core/dd/partition.js interfacePartition). Interior unknowns of different
 * subdomains are never neighbours, so after reordering
 *
 *       ⎡ A_{I₁I₁}             A_{I₁Γ} ⎤ ⎡u_{I₁}⎤   ⎡b_{I₁}⎤
 *       ⎢          ⋱             ⋮     ⎥ ⎢  ⋮   ⎥ = ⎢  ⋮   ⎥
 *       ⎢             A_{I_KI_K} A_{I_KΓ}⎥ ⎢u_{I_K}⎥   ⎢b_{I_K}⎥
 *       ⎣ A_{ΓI₁}  ⋯  A_{ΓI_K}   A_{ΓΓ} ⎦ ⎣ u_Γ  ⎦   ⎣ b_Γ  ⎦
 *
 * (block "arrow" matrix). Eliminating the interiors (block Gaussian elimination
 * = static condensation of the subdomain interiors) gives the interface problem
 *
 *     S u_Γ = g,   S = A_ΓΓ − Σ_k A_{ΓI_k} A_{I_kI_k}⁻¹ A_{I_kΓ},
 *                  g = b_Γ − Σ_k A_{ΓI_k} A_{I_kI_k}⁻¹ b_{I_k},
 *
 * after which u_{I_k} = A_{I_kI_k}⁻¹ (b_{I_k} − A_{I_kΓ} u_Γ) — K independent solves.
 * S is SPD; applying it costs one Dirichlet solve per subdomain.
 *
 * Local Schur complements. With the "Neumann" matrix A^{(k)} assembled from the
 * elements of subdomain k only,  S^{(k)} = A^{(k)}_{ΓΓ} − A^{(k)}_{ΓI} (A_{I_kI_k})⁻¹ A^{(k)}_{IΓ}
 * is the discrete Dirichlet-to-Neumann (Steklov–Poincaré) map of subdomain k, and
 * S = Σ_k R_{Γ_k}ᵀ S^{(k)} R_{Γ_k}. For a "floating" subdomain (not touching ∂Ω)
 * S^{(k)} 1 = 0: it is singular.
 */
import { sparseLU } from '../la/direct.js';
import { csrExtract, csrMatVec } from '../la/sparse.js';
import { symEig } from '../la/dense.js';
import { cg } from '../la/krylov.js';
import { interfacePartition } from './partition.js';
import { freeIndex } from './problem.js';

/**
 * @typedef {Object} SchurSetup
 * @property {import('./problem.js').DDProblem} prob
 * @property {number} Kx
 * @property {number} Ky
 * @property {Int32Array} gamma        interface free indices
 * @property {Int32Array[]} interiors  interior free indices per subdomain
 * @property {Int32Array} perm         reordering [I_1, …, I_K, Γ] (new → old free index)
 * @property {number[]} blocks         block boundaries in the reordered numbering
 * @property {(x: Float64Array, y: Float64Array) => void} applyS  y = S x (matrix-free)
 */

/**
 * Factorise the subdomain interior problems and set up the interface operator.
 * @param {import('./problem.js').DDProblem} prob
 * @param {number} Kx
 * @param {number} Ky
 * @returns {SchurSetup & {cellSub: (ci:number,cj:number)=>number, interiorSolve: (k:number, r:ArrayLike<number>)=>Float64Array,
 *          AIG: import('../la/sparse.js').CSR[], AGI: import('../la/sparse.js').CSR[], AGG: import('../la/sparse.js').CSR}}
 */
export function schurSetup(prob, Kx, Ky) {
  const P = interfacePartition(prob.N, Kx, Ky);
  const { gamma, interiors } = P;
  const LU = interiors.map((I) => sparseLU(csrExtract(prob.A, I, I)));
  const AIG = interiors.map((I) => csrExtract(prob.A, I, gamma));
  const AGI = interiors.map((I) => csrExtract(prob.A, gamma, I));
  const AGG = csrExtract(prob.A, gamma, gamma);
  const interiorSolve = (k, r) => LU[k].solve(r);
  const nG = gamma.length;
  const applyS = (x, y) => {
    csrMatVec(AGG, x, y);
    for (let k = 0; k < interiors.length; k++) {
      const t = interiorSolve(k, csrMatVec(AIG[k], x));
      const s = csrMatVec(AGI[k], t);
      for (let i = 0; i < nG; i++) y[i] -= s[i];
    }
  };
  const perm = Int32Array.from([...interiors.flatMap((I) => [...I]), ...gamma]);
  const blocks = [];
  let acc = 0;
  for (const I of interiors) { acc += I.length; blocks.push(acc); }
  return { prob, Kx, Ky, gamma, interiors, perm, blocks, applyS, interiorSolve, AIG, AGI, AGG, cellSub: P.cellSub };
}

/**
 * Dense Schur complement S (nΓ × nΓ, row-major), column by column: S e_j.
 * Only for small problems (illustration, condition numbers).
 * @param {SchurSetup} su
 * @returns {Float64Array}
 */
export function denseSchur(su) {
  const nG = su.gamma.length, S = new Float64Array(nG * nG);
  const e = new Float64Array(nG), y = new Float64Array(nG);
  for (let j = 0; j < nG; j++) {
    e.fill(0); e[j] = 1;
    su.applyS(e, y);
    for (let i = 0; i < nG; i++) S[i * nG + j] = y[i];
  }
  return S;
}

/**
 * Solve A u = b by (unpreconditioned) CG on the interface system S u_Γ = g,
 * then back-substitute the interiors.
 * @param {SchurSetup} su
 * @param {ArrayLike<number>} b global load vector (free unknowns)
 * @param {{tol?: number, maxIter?: number}} [o]
 * @returns {{x: Float64Array, iters: number, converged: boolean, history: number[]}} x = full solution (free unknowns)
 */
export function schurSolve(su, b, o = {}) {
  const { gamma, interiors } = su, nG = gamma.length;
  const g = Float64Array.from(gamma, (f) => b[f]);
  const bI = interiors.map((I) => Float64Array.from(I, (f) => b[f]));
  for (let k = 0; k < interiors.length; k++) {
    const s = csrMatVec(su.AGI[k], su.interiorSolve(k, bI[k]));
    for (let i = 0; i < nG; i++) g[i] -= s[i];
  }
  const res = cg(su.applyS, g, { tol: o.tol ?? 1e-10, maxIter: o.maxIter ?? 1000 });
  const x = new Float64Array(b.length);
  gamma.forEach((f, i) => { x[f] = res.x[i]; });
  for (let k = 0; k < interiors.length; k++) {
    const r = csrMatVec(su.AIG[k], res.x);
    for (let i = 0; i < r.length; i++) r[i] = bI[k][i] - r[i];
    const uI = su.interiorSolve(k, r);
    interiors[k].forEach((f, i) => { x[f] = uI[i]; });
  }
  return { x, iters: res.iters, converged: res.converged, history: res.history };
}

/**
 * Local Schur complement (discrete Dirichlet-to-Neumann map) of subdomain k:
 *   S^{(k)} = A^{(k)}_{Γ_kΓ_k} − A^{(k)}_{Γ_k I_k} A_{I_kI_k}⁻¹ A^{(k)}_{I_kΓ_k}
 * on the interface unknowns Γ_k lying on the boundary of subdomain k.
 * @param {SchurSetup} su
 * @param {number} k subdomain index ky Kx + kx
 * @returns {{gammaLocal: Int32Array, gammaPos: Int32Array, S: Float64Array, floating: boolean}}
 *   gammaLocal = free indices of Γ_k, gammaPos = their positions in su.gamma, S dense row-major
 */
export function localSchur(su, k) {
  const { prob } = su;
  const { A: Ak } = prob.localMatrix((ci, cj) => su.cellSub(ci, cj) === k);
  const I = su.interiors[k];
  // Γ_k = interface unknowns touched by the elements of subdomain k (non-zero diagonal of A^{(k)})
  const posInGamma = new Map();
  su.gamma.forEach((f, i) => posInGamma.set(f, i));
  const gl = [], gp = [];
  for (const f of su.gamma) {
    let d = 0;
    for (let q = Ak.rowPtr[f]; q < Ak.rowPtr[f + 1]; q++) if (Ak.colIdx[q] === f) d = Ak.vals[q];
    if (d !== 0) { gl.push(f); gp.push(posInGamma.get(f)); }
  }
  const gammaLocal = Int32Array.from(gl), n = gl.length;
  const AGG = csrExtract(Ak, gammaLocal, gammaLocal), AGI = csrExtract(Ak, gammaLocal, I), AIG = csrExtract(Ak, I, gammaLocal);
  const S = new Float64Array(n * n), e = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    e.fill(0); e[j] = 1;
    const y = csrMatVec(AGG, e);
    const s = csrMatVec(AGI, su.interiorSolve(k, csrMatVec(AIG, e)));
    for (let i = 0; i < n; i++) S[i * n + j] = y[i] - s[i];
  }
  // floating = no vertex of the subdomain's closure on ∂Ω
  const kx = k % su.Kx, ky = Math.floor(k / su.Kx);
  const floating = kx > 0 && kx < su.Kx - 1 && ky > 0 && ky < su.Ky - 1;
  return { gammaLocal, gammaPos: Int32Array.from(gp), S, floating };
}

/**
 * Extreme eigenvalues and condition number of a dense symmetric matrix.
 * @param {Float64Array} M n×n
 * @param {number} n
 * @returns {{lmin: number, lmax: number, kappa: number}}
 */
export function denseCondition(M, n) {
  const ev = symEig(M, n).values;
  return { lmin: ev[0], lmax: ev[n - 1], kappa: ev[n - 1] / ev[0] };
}

/**
 * Two subdomains Ω₁ = (0, s h) × (0,1) and Ω₂ = (s h, 1) × (0,1) separated by the
 * vertical interface Γ = {x = s h}: set up the Dirichlet–Neumann iteration.
 * @param {import('./problem.js').DDProblem} prob
 * @param {number} s interface column, 1 ≤ s ≤ N−1
 * @returns {{I1: Int32Array, I2: Int32Array, G: Int32Array, step: (lambda: Float64Array, theta: number) => {lambda: Float64Array, x: Float64Array},
 *            S1: Float64Array, S2: Float64Array}} S1, S2 = dense local Schur complements on Γ (ordered by j)
 */
export function dirichletNeumannSetup(prob, s) {
  const { N, A, b } = prob;
  const I1 = [], I2 = [], G = [];
  for (let j = 1; j < N; j++) for (let i = 1; i < N; i++) (i < s ? I1 : i > s ? I2 : G).push(freeIndex(N, i, j));
  const i1 = Int32Array.from(I1), i2 = Int32Array.from(I2), g = Int32Array.from(G), nG = g.length;
  const A1 = prob.localMatrix((ci) => ci < s).A, A2 = prob.localMatrix((ci) => ci >= s).A;
  // Dirichlet problem on Ω₁: interior unknowns I1, data λ on Γ
  const F1 = sparseLU(csrExtract(A, i1, i1));
  const A1IG = csrExtract(A, i1, g), A1GI = csrExtract(A1, g, i1), A1GG = csrExtract(A1, g, g);
  // Neumann problem on Ω₂: unknowns [I2, Γ], local matrix A^{(2)}
  const n2 = [...i2, ...g];
  const F2 = sparseLU(csrExtract(A2, n2, n2));
  const bI1 = Float64Array.from(i1, (f) => b[f]), bI2 = Float64Array.from(i2, (f) => b[f]), bG = Float64Array.from(g, (f) => b[f]);
  /**
   * One Dirichlet–Neumann step from interface values λ:
   *  1. Dirichlet solve in Ω₁:  A_{I₁I₁} u_{I₁} = b_{I₁} − A_{I₁Γ} λ
   *  2. Neumann solve in Ω₂ with the flux (residual) of Ω₁ as data:
   *       A^{(2)} [u_{I₂}; u_Γ] = [b_{I₂}; b_Γ − A^{(1)}_{ΓI₁} u_{I₁} − A^{(1)}_{ΓΓ} λ]
   *  3. relax: λ ← θ u_Γ + (1 − θ) λ.
   */
  const step = (lambda, theta) => {
    const r1 = csrMatVec(A1IG, lambda);
    for (let i = 0; i < r1.length; i++) r1[i] = bI1[i] - r1[i];
    const u1 = F1.solve(r1);
    const flux = csrMatVec(A1GI, u1), aGG = csrMatVec(A1GG, lambda);
    const rhs = new Float64Array(n2.length);
    rhs.set(bI2);
    for (let i = 0; i < nG; i++) rhs[i2.length + i] = bG[i] - flux[i] - aGG[i];
    const u2 = F2.solve(rhs);
    const out = new Float64Array(nG);
    for (let i = 0; i < nG; i++) out[i] = theta * u2[i2.length + i] + (1 - theta) * lambda[i];
    const x = new Float64Array(prob.n);
    i1.forEach((f, k) => { x[f] = u1[k]; });
    i2.forEach((f, k) => { x[f] = u2[k]; });
    g.forEach((f, k) => { x[f] = u2[i2.length + k]; });
    return { lambda: out, x };
  };
  // dense local Schur complements on Γ (for the convergence analysis)
  const localS = (Aloc, I) => {
    const F = I === i1 ? F1 : sparseLU(csrExtract(A, I, I));
    const GG = csrExtract(Aloc, g, g), GI = csrExtract(Aloc, g, I), IG = csrExtract(Aloc, I, g);
    const S = new Float64Array(nG * nG), e = new Float64Array(nG);
    for (let j = 0; j < nG; j++) {
      e.fill(0); e[j] = 1;
      const y = csrMatVec(GG, e), t = csrMatVec(GI, F.solve(csrMatVec(IG, e)));
      for (let i = 0; i < nG; i++) S[i * nG + j] = y[i] - t[i];
    }
    return S;
  };
  return { I1: i1, I2: i2, G: g, step, S1: localS(A1, i1), S2: localS(A2, i2) };
}

/**
 * Spectrum of the Dirichlet–Neumann iteration. The step is the preconditioned
 * Richardson iteration  λ ← λ + θ S₂⁻¹ (g − S λ),  S = S₁ + S₂, so the error is
 * multiplied by  I − θ S₂⁻¹ S = I − θ (I + S₂⁻¹ S₁).  With μ ∈ [μ_min, μ_max] the
 * (real, positive) eigenvalues of S₂⁻¹ S₁, the contraction factor is
 *   ρ(θ) = max(|1 − θ(1 + μ_min)|, |1 − θ(1 + μ_max)|),
 * minimised by θ_opt = 2 / (2 + μ_min + μ_max).
 * @param {Float64Array} S1 dense n×n SPD
 * @param {Float64Array} S2 dense n×n SPD
 * @param {number} n
 * @returns {{muMin: number, muMax: number, thetaOpt: number, rho: (theta: number) => number}}
 */
export function dnSpectrum(S1, S2, n) {
  // S₂⁻¹S₁ is similar to the symmetric matrix C = Wᵀ S₁ W with W = V Λ^{-1/2},
  // where S₂ = V Λ Vᵀ (indeed W Wᵀ = S₂⁻¹ and W⁻¹ S₂⁻¹ S₁ W = Wᵀ S₁ W),
  // so its eigenvalues are real and positive.
  const E = symEig(S2, n), V = E.vectors, lam = E.values;
  const W = new Float64Array(n * n); // W = V Λ^{-1/2}
  for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) W[i * n + k] = V[i * n + k] / Math.sqrt(lam[k]);
  // C = Wᵀ S₁ W
  const T = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += S1[i * n + j] * W[j * n + k];
    T[i * n + k] = s;
  }
  const C = new Float64Array(n * n);
  for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += W[i * n + a] * T[i * n + c];
    C[a * n + c] = s;
  }
  for (let a = 0; a < n; a++) for (let c = a + 1; c < n; c++) { const m = 0.5 * (C[a * n + c] + C[c * n + a]); C[a * n + c] = C[c * n + a] = m; }
  const mu = symEig(C, n).values, muMin = mu[0], muMax = mu[n - 1];
  const rho = (th) => Math.max(Math.abs(1 - th * (1 + muMin)), Math.abs(1 - th * (1 + muMax)));
  return { muMin, muMax, thetaOpt: 2 / (2 + muMin + muMax), rho };
}
