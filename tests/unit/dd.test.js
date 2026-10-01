/**
 * Chapter 10 (domain decomposition): partitions, identities and exact solves.
 */
import { test, assert, assertClose } from '../harness.js';
import { ddPoissonProblem, freeIndex } from '../../lib/core/dd/problem.js';
import { boxPartition, interfacePartition } from '../../lib/core/dd/partition.js';
import { schwarzSetup, coarseBasis, applyASM, applyRAS, alternatingSchwarz1D, solveSchwarz } from '../../lib/core/dd/schwarz.js';
import { schurSetup, denseSchur, schurSolve, localSchur } from '../../lib/core/dd/schur.js';
import { triGrid } from '../../lib/core/mesh/structured.js';
import { solvePoissonCG } from '../../lib/core/elliptic/cg.js';
import { sparseSolve, sparseLU } from '../../lib/core/la/direct.js';
import { csrGet, csrSymmetryError, csrMatVec } from '../../lib/core/la/sparse.js';

const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const f = (x, y) => Math.sin(3 * x) * (1 + y * y) + 1;

test('DD model problem = solvePoissonCG (strong BC, g = 0) and A is the 5-point stencil', () => {
  const N = 10, prob = ddPoissonProblem(N, f);
  const ref = solvePoissonCG(triGrid(N, N, [0, 1, 0, 1], { diag: 'right' }), { p: 1, f, g: () => 0 });
  assert(prob.n === ref.free.length && prob.free.every((v, k) => v === ref.free[k]), 'same free ordering');
  const u = sparseSolve(prob.A, prob.b);
  assert(maxDiff(u, Float64Array.from(prob.free, (v) => ref.U[v])) < 1e-13, 'same solution');
  assert(csrSymmetryError(prob.A) < 1e-14, 'symmetric');
  for (let j = 2; j < N - 1; j++) for (let i = 2; i < N - 1; i++) {
    const r = freeIndex(N, i, j);
    assertClose(csrGet(prob.A, r, r), 4, 1e-13);
    for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) assertClose(csrGet(prob.A, r, freeIndex(N, i + di, j + dj)), -1, 1e-13);
    for (const [di, dj] of [[1, 1], [-1, -1], [1, -1], [-1, 1]]) assertClose(csrGet(prob.A, r, freeIndex(N, i + di, j + dj)), 0, 1e-13);
  }
});

test('box partition: overlapping subdomains cover everything, cores form a partition of unity', () => {
  for (const [N, Kx, Ky, ell] of [[12, 3, 2, 1], [16, 4, 4, 2], [15, 2, 3, 3]]) {
    const P = boxPartition(N, Kx, Ky, ell), n = (N - 1) ** 2;
    const cover = new Int32Array(n), owned = new Int32Array(n);
    P.subs.forEach((s, k) => s.dofs.forEach((d, l) => { cover[d]++; if (s.core[l]) { owned[d]++; assert(P.owner[d] === k, 'owner'); } }));
    assert(owned.every((c) => c === 1), 'Σ R_kᵀ D_k R_k = I');
    assert(cover.every((c) => c >= 1), 'covering');
    // overlap width: two horizontal neighbours share exactly 2ℓ − 1 vertex columns (strip of width 2ℓh)
    const a = P.subs[0], b = P.subs[1];
    const colsA = new Set([...a.dofs].map((d) => d % (N - 1))), colsB = new Set([...b.dofs].map((d) => d % (N - 1)));
    assert([...colsA].filter((c) => colsB.has(c)).length === 2 * ell - 1, 'overlap columns');
    assertClose(a.box[1] - b.box[0], 2 * ell / N, 1e-14, 0, 'geometric overlap δ = 2ℓh');
  }
});

test('Q1 coarse basis: n_c = (Kx−1)(Ky−1) hat functions summing to 1 on the interior coarse cells', () => {
  const N = 12, prob = ddPoissonProblem(N), P = boxPartition(N, 3, 4, 1);
  const Z = coarseBasis(prob, P);
  assert(Z.length === 2 * 3, 'n_c = (Kx−1)(Ky−1)');
  // Σ_c z_c = 1 on the interior coarse cells [cutsX[1], cutsX[2]] × [cutsY[1], cutsY[3]]
  for (let j = P.cutsY[1]; j <= P.cutsY[3]; j++) for (let i = P.cutsX[1]; i <= P.cutsX[2]; i++) {
    let s = 0; for (const z of Z) s += z[freeIndex(N, i, j)];
    assertClose(s, 1, 1e-14);
  }
});

test('ASM preconditioner is symmetric positive definite; RAS is not symmetric', () => {
  const N = 12, prob = ddPoissonProblem(N), n = prob.n;
  for (const coarse of ['none', 'q1']) {
    const dd = schwarzSetup(prob, boxPartition(N, 3, 3, 1), { coarse });
    const M = new Float64Array(n * n), R = new Float64Array(n * n), e = new Float64Array(n), z = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      e.fill(0); e[j] = 1;
      applyASM(dd, e, z); for (let i = 0; i < n; i++) M[i * n + j] = z[i];
      applyRAS(dd, e, z); for (let i = 0; i < n; i++) R[i * n + j] = z[i];
    }
    let asym = 0, rasym = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { asym = Math.max(asym, Math.abs(M[i * n + j] - M[j * n + i])); rasym = Math.max(rasym, Math.abs(R[i * n + j] - R[j * n + i])); }
    assert(asym < 1e-13, `ASM symmetric (${coarse})`);
    assert(rasym > 1e-3, `RAS non-symmetric (${coarse})`);
    // positive: rᵀ M r > 0 for some random vectors
    let seed = 3;
    for (let t = 0; t < 5; t++) {
      for (let i = 0; i < n; i++) { seed = (seed * 16807) % 2147483647; e[i] = seed / 2147483647 - 0.5; }
      applyASM(dd, e, z);
      assert(e.reduce((s, v, i) => s + v * z[i], 0) > 0, 'positive');
    }
  }
});

test('Schur complement: interface solve = monolithic solve; S = Σ_k R_kᵀ S^(k) R_k; floating subdomain has S^(k) 1 = 0', () => {
  const N = 12, prob = ddPoissonProblem(N, f), u = sparseSolve(prob.A, prob.b);
  for (const [Kx, Ky] of [[2, 1], [2, 2], [3, 3]]) {
    const su = schurSetup(prob, Kx, Ky);
    const r = schurSolve(su, prob.b, { tol: 1e-13 });
    assert(r.converged && maxDiff(r.x, u) < 1e-11, `schur solve ${Kx}x${Ky}`);
    // interface = the cut lines; interiors are decoupled (no A entries between different interiors)
    const P = interfacePartition(N, Kx, Ky), sub = new Int32Array(prob.n).fill(-1);
    P.interiors.forEach((I, k) => I.forEach((d) => { sub[d] = k; }));
    for (let i = 0; i < prob.n; i++) for (let q = prob.A.rowPtr[i]; q < prob.A.rowPtr[i + 1]; q++) {
      const j = prob.A.colIdx[q];
      if (sub[i] >= 0 && sub[j] >= 0) assert(sub[i] === sub[j], 'interiors decoupled');
    }
  }
  const su = schurSetup(prob, 3, 3), S = denseSchur(su), nG = su.gamma.length, Ssum = new Float64Array(nG * nG);
  for (let k = 0; k < 9; k++) {
    const L = localSchur(su, k), m = L.gammaLocal.length;
    let rowSum = 0;
    for (let i = 0; i < m; i++) { let s = 0; for (let j = 0; j < m; j++) s += L.S[i * m + j]; rowSum = Math.max(rowSum, Math.abs(s)); }
    assert(L.floating === (k === 4), 'only the centre subdomain floats');
    if (L.floating) assert(rowSum < 1e-12, 'S^(k) 1 = 0 for the floating subdomain');
    else assert(rowSum > 0.1, 'non-floating S^(k) is non-singular on constants');
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) Ssum[L.gammaPos[i] * nG + L.gammaPos[j]] += L.S[i * m + j];
  }
  assert(maxDiff(S, Ssum) < 1e-12, 'S = sum of local Schur complements');
  assert(maxDiff(S, Float64Array.from({ length: nG * nG }, (_, k) => S[(k % nG) * nG + Math.floor(k / nG)])) < 1e-12, 'S symmetric');
});

test('1D alternating Schwarz: interface error shrinks by exactly ρ = α(1−β)/(β(1−α))', () => {
  for (const [alpha, beta, f0] of [[0.4, 0.6, 0], [0.3, 0.5, 2], [0.45, 0.55, -1]]) {
    const r = alternatingSchwarz1D({ alpha, beta, f0, uL: 0.2, uR: 1, g0: -0.5, iters: 8 });
    const ex = r.exact[0] * beta * beta + r.exact[1] * beta + r.exact[2];
    for (let n = 1; n < r.gammaB.length; n++) assertClose((r.gammaB[n] - ex) / (r.gammaB[n - 1] - ex), r.rho, 1e-9, 0, 'contraction');
    // both local solutions satisfy their boundary conditions
    const ev = (c, x) => c[0] * x * x + c[1] * x + c[2];
    assertClose(ev(r.u1[0], 0), 0.2, 1e-14); assertClose(ev(r.u2[0], 1), 1, 1e-14);
    assertClose(ev(r.u2[2], alpha), ev(r.u1[2], alpha), 1e-14);
  }
});

test('stationary additive Schwarz (no damping) diverges, while CG with the same M converges', () => {
  const N = 16, prob = ddPoissonProblem(N), dd = schwarzSetup(prob, boxPartition(N, 2, 2, 2));
  const x = new Float64Array(prob.n), r = new Float64Array(prob.n), z = new Float64Array(prob.n);
  const res = () => { csrMatVec(prob.A, x, r); for (let i = 0; i < r.length; i++) r[i] = prob.b[i] - r[i]; return Math.hypot(...r); };
  const r0 = res();
  for (let k = 0; k < 60; k++) { res(); applyASM(dd, r, z); for (let i = 0; i < x.length; i++) x[i] += z[i]; }
  assert(res() > 10 * r0, 'Richardson with M_ASM diverges');
  const c = solveSchwarz(dd, prob.b, { method: 'asm-cg', tol: 1e-10 });
  assert(c.converged && c.iters < 30, 'ASM-PCG converges');
});

test('banded LU of the N×N grid problem has bandwidth N − 1 = √n after RCM', () => {
  for (const N of [16, 32]) {
    const F = sparseLU(ddPoissonProblem(N).A);
    assert(F.kl === N - 1 && F.ku === N - 1, `bandwidth ${F.kl}, ${F.ku}`);
  }
});
