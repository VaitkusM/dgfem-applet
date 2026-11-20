import { test, assert, assertClose, assertRate } from '../harness.js';
import { triGrid } from '../../lib/core/mesh/structured.js';
import { solvePoissonCG, errorsCG, boundaryError } from '../../lib/core/elliptic/cg.js';
import { POISSON_MMS } from '../../lib/core/verify/mms.js';
import { fitRate } from '../../lib/core/verify/rates.js';
import { csrSymmetryError } from '../../lib/core/la/sparse.js';

const run = (p, bc, Ns, sol = 'wave', extra = {}) => {
  const S = POISSON_MMS[sol], hs = [], L2 = [], H1 = [];
  for (const N of Ns) {
    const mesh = triGrid(N, N, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.15 });
    const r = solvePoissonCG(mesh, { p, f: S.f, g: S.u, bc, ...extra });
    const e = errorsCG(r.space, r.U, S.u, S.grad);
    hs.push(1 / N); L2.push(e.L2); H1.push(e.H1);
  }
  return { L2: fitRate(hs, L2), H1: fitRate(hs, H1), errs: L2 };
};

test('CG P1/P2 with strong Dirichlet BCs: L2 rate p+1, H1 rate p', () => {
  for (const p of [1, 2]) {
    const r = run(p, 'strong', [8, 16, 32]);
    assertRate(r.L2, p + 1, `P${p} L2`); assertRate(r.H1, p, `P${p} H1`);
  }
});

test('CG with symmetric Nitsche BCs: optimal rates; matrix symmetric', () => {
  for (const p of [1, 2]) {
    const r = run(p, 'nitsche', [8, 16, 32], 'wave', { gamma: 10 * p * p });
    assertRate(r.L2, p + 1, `Nitsche P${p} L2`); assertRate(r.H1, p, `Nitsche P${p} H1`);
  }
  const S = POISSON_MMS.wave;
  const res = solvePoissonCG(triGrid(6, 6), { p: 2, f: S.f, g: S.u, bc: 'nitsche', gamma: 40 });
  assert(csrSymmetryError(res.A) < 1e-13, 'symmetric');
});

test('non-symmetric Nitsche converges (H1 rate p); penalty-only loses L2 order for P1', () => {
  for (const p of [1, 2]) {
    const r = run(p, 'nitsche-nonsym', [8, 16, 32], 'wave', { gamma: 10 });
    assertRate(r.H1, p, `nonsym P${p} H1`);
  }
  const pen = run(1, 'penalty', [8, 16, 32, 64], 'wave', { gamma: 10 });
  assert(pen.L2 < 1.4, `penalty L2 rate ${pen.L2} should be about 1`);
  assertRate(pen.L2, 1, 'penalty L2', 0.3, 0.3);
});

test('strong BCs interpolate g exactly at boundary nodes; Nitsche only approximately', () => {
  const S = POISSON_MMS.wave, mesh = triGrid(8, 8);
  const rs = solvePoissonCG(mesh, { p: 1, f: S.f, g: S.u, bc: 'strong' });
  const rn = solvePoissonCG(mesh, { p: 1, f: S.f, g: S.u, bc: 'nitsche', gamma: 10 });
  for (let i = 0; i < rs.space.nDof; i++) if (rs.space.onBoundary[i])
    assert(Math.abs(rs.U[i] - S.u(rs.space.dofXY[2 * i], rs.space.dofXY[2 * i + 1])) < 1e-14);
  const be = boundaryError(rn.space, rn.U, S.u);
  assert(be > 1e-5 && be < 0.05, `Nitsche boundary mismatch ${be}`);
});

/* ---------------------------------------------------------------------- */
/* Tests added by the audit: chapter 3 and 4 claims that had no test        */
/* ---------------------------------------------------------------------- */
import { cgSpace, assembleVolume } from '../../lib/core/elliptic/cg.js';
import { csrToDense, nnz } from '../../lib/core/la/sparse.js';
import { symEig } from '../../lib/core/la/dense.js';
import { cg as cgSolve, jacobiPrecond } from '../../lib/core/la/krylov.js';

/** extreme eigenvalues of the symmetric part (A + Aᵀ)/2 */
const symEigRange = (A) => {
  const n = A.n, D = csrToDense(A), Sm = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Sm[i * n + j] = 0.5 * (D[i * n + j] + D[j * n + i]);
  const ev = symEig(Sm, n).values;
  return [ev[0], ev[n - 1]];
};
const bcMesh = (N) => triGrid(N, N, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.1 }); // the mesh of the chapter-4 widgets

test('P1 stiffness matrix = cotangent Laplacian: A_ij = −½(cot α + cot β), zero row sums (distorted mesh)', () => {
  const mesh = triGrid(5, 4, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.25, seed: 3 });
  const space = cgSpace(mesh, 1), nV = space.nDof;
  const A = csrToDense(assembleVolume(space, () => 0).builder.toCSR());
  // reference: sum over triangles of −½ cot(angle opposite to the edge)
  const C = new Float64Array(nV * nV), { nodes, tris } = mesh;
  for (let t = 0; t < tris.length / 3; t++) for (let k = 0; k < 3; k++) {
    const c = tris[3 * t + k], a = tris[3 * t + (k + 1) % 3], b = tris[3 * t + (k + 2) % 3];
    const ux = nodes[2 * a] - nodes[2 * c], uy = nodes[2 * a + 1] - nodes[2 * c + 1];
    const vx = nodes[2 * b] - nodes[2 * c], vy = nodes[2 * b + 1] - nodes[2 * c + 1];
    const w = -0.5 * (ux * vx + uy * vy) / Math.abs(ux * vy - uy * vx);
    C[a * nV + b] += w; C[b * nV + a] += w; C[a * nV + a] -= w; C[b * nV + b] -= w;
  }
  for (let i = 0; i < nV; i++) {
    let rs = 0;
    for (let j = 0; j < nV; j++) { assertClose(A[i * nV + j], C[i * nV + j], 1e-12, 0, `A[${i},${j}]`); rs += A[i * nV + j]; }
    assertClose(rs, 0, 1e-12, 0, `row sum ${i}`);
  }
});

test('reduced P1 system: κ ~ h⁻², CG iterations ~ h⁻¹, Jacobi does not help on a quasi-uniform mesh; nonzeros per row', () => {
  const S = POISSON_MMS.wave, kap = [], its = [], itsJ = [];
  for (const N of [8, 16, 32]) {
    const r = solvePoissonCG(triGrid(N, N, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.15 }), { p: 1, f: S.f, g: S.u });
    if (N <= 16) { const [lo, hi] = symEigRange(r.A); assert(lo > 0, 'SPD'); kap.push(hi / lo); }
    its.push(cgSolve(r.A, r.b, { tol: 1e-10, maxIter: 5000 }).iters);
    itsJ.push(cgSolve(r.A, r.b, { tol: 1e-10, maxIter: 5000, precond: jacobiPrecond(r.A) }).iters);
  }
  assert(kap[1] / kap[0] > 3.6 && kap[1] / kap[0] < 4.4, `κ ratio for h → h/2: ${kap[1] / kap[0]}`);
  for (let k = 1; k < 3; k++) assert(its[k] / its[k - 1] > 1.6 && its[k] / its[k - 1] < 2.4, `CG iterations ${its}`);
  for (let k = 0; k < 3; k++) assert(Math.abs(itsJ[k] - its[k]) <= 0.1 * its[k], `Jacobi ${itsJ} vs plain ${its}`);
  // interior rows: 5 on the right-triangle grid (cot 90° = 0 removes the diagonal couplings), → 7 on a general mesh
  const perRow = (jiggle) => { const A = solvePoissonCG(triGrid(40, 40, [0, 1, 0, 1], { diag: 'alt', jiggle }), { p: 1, f: S.f, g: S.u }).A; return nnz(A) / A.n; };
  const reg = perRow(0), gen = perRow(0.15);
  assert(reg > 4.8 && reg <= 5, `regular grid: ${reg} per row`);
  assert(gen > 6.6 && gen <= 7, `distorted grid: ${gen} per row`);
});

test('symmetric Nitsche: coercive only above a γ threshold (≈1.6 for P1, ≈5.6 for P2), independent of h; flat above it', () => {
  const S = POISSON_MMS.wave;
  const lmin = (p, N, bc, gamma) => symEigRange(solvePoissonCG(bcMesh(N), { p, f: S.f, g: S.u, bc, gamma }).A)[0];
  // P1: threshold between 1 and 3 on N = 8 and N = 16; P2: between 4 and 8 on N = 4 and N = 8
  for (const N of [8, 16]) { assert(lmin(1, N, 'nitsche', 1) < 0, `P1 N=${N} γ=1 indefinite`); assert(lmin(1, N, 'nitsche', 3) > 0, `P1 N=${N} γ=3 SPD`); }
  for (const N of [4, 8]) { assert(lmin(2, N, 'nitsche', 4) < 0, `P2 N=${N} γ=4 indefinite`); assert(lmin(2, N, 'nitsche', 8) > 0, `P2 N=${N} γ=8 SPD`); }
  // bisection for the widget meshes
  const thr = (p, N) => { let lo = 0.1, hi = 50; for (let k = 0; k < 30; k++) { const m = Math.sqrt(lo * hi); if (lmin(p, N, 'nitsche', m) > 0) hi = m; else lo = m; } return hi; };
  const t1 = thr(1, 8), t2 = thr(2, 4);
  assert(t1 > 1.4 && t1 < 1.8, `P1 threshold ${t1}`);
  assert(t2 > 5.2 && t2 < 6.0, `P2 threshold ${t2}`);
  // above the threshold λ_min is set by the interior (≈ that of strong imposition) and no longer depends on γ
  const strong = symEigRange(solvePoissonCG(bcMesh(8), { p: 1, f: S.f, g: S.u, bc: 'strong' }).A)[0];
  for (const g of [10, 100, 1e4]) assertClose(lmin(1, 8, 'nitsche', g), strong, 0, 0.02, `flat at γ=${g}`);
});

test('non-symmetric Nitsche and penalty: positive definite for every γ > 0; at γ = 0 only semi-definite but still solvable', () => {
  const S = POISSON_MMS.wave;
  for (const p of [1, 2]) for (const bc of ['nitsche-nonsym', 'penalty']) for (const gamma of [0.1, 1, 10, 1e4]) {
    const l = symEigRange(solvePoissonCG(bcMesh(p === 1 ? 8 : 4), { p, f: S.f, g: S.u, bc, gamma }).A)[0];
    assert(l > 1e-3, `${bc} P${p} γ=${gamma}: λ_min = ${l}`);
  }
  // γ = 0: a_h(v, v) = ‖∇v‖² vanishes for constants → not coercive; Burman's penalty-free method is inf-sup stable
  const r = solvePoissonCG(bcMesh(8), { p: 1, f: S.f, g: S.u, bc: 'nitsche-nonsym', gamma: 0 });
  assert(Math.abs(symEigRange(r.A)[0]) < 1e-12, 'γ = 0: zero eigenvalue of the symmetric part');
  const e = errorsCG(r.space, r.U, S.u, S.grad);
  assert(!r.singular && e.L2 < 0.1, `γ = 0 still gives a solution, L2 error ${e.L2}`);
});

test('penalty method: L2 rate → 1 for P1 and P2, H1 rate of P2 also capped at 1; larger γ only postpones it', () => {
  const p1 = run(1, 'penalty', [16, 32, 64], 'wave', { gamma: 10 });
  assert(p1.L2 > 0.9 && p1.L2 < 1.4, `P1 L2 rate ${p1.L2}`);
  const p2 = run(2, 'penalty', [8, 16, 24], 'wave', { gamma: 10 });
  assert(p2.L2 > 0.8 && p2.L2 < 1.2, `P2 L2 rate ${p2.L2}`);
  assert(p2.H1 > 0.7 && p2.H1 < 1.2, `P2 H1 rate ${p2.H1}`);
  // with γ = 100 the same grids still look optimal for P1 (pre-asymptotic): the cap is O(h/γ)
  const big = run(1, 'penalty', [8, 16, 32], 'wave', { gamma: 100 });
  assert(big.L2 > 1.8, `P1, γ = 100: L2 rate ${big.L2}`);
});

test('boundary mismatch max|u_h − g|: Nitsche converges at the optimal rate p + 1, penalty only at rate 1', () => {
  const S = POISSON_MMS.wave;
  const rate = (p, bc, Ns) => {
    const be = Ns.map((N) => { const r = solvePoissonCG(bcMesh(N), { p, f: S.f, g: S.u, bc, gamma: 10 * p * p }); return boundaryError(r.space, r.U, S.u); });
    return { rate: fitRate(Ns.map((N) => 1 / N), be), last: be[be.length - 1] };
  };
  const n1 = rate(1, 'nitsche', [8, 16, 32]), n2 = rate(2, 'nitsche', [8, 16, 24]);
  assert(n1.rate > 1.7 && n1.rate < 2.3, `Nitsche P1 ${n1.rate}`);
  assert(n2.rate > 2.6 && n2.rate < 3.3, `Nitsche P2 ${n2.rate}`);
  const pen = rate(1, 'penalty', [16, 32, 64]);
  assert(pen.rate > 0.8 && pen.rate < 1.4, `penalty P1 ${pen.rate}`);
});

test('numbers quoted in the chapter-4 "Try this" box (widget meshes, fit over the last three grids)', () => {
  const S = POISSON_MMS.wave;
  const err = (bc, p, N, gamma) => { const r = solvePoissonCG(bcMesh(N), { p, f: S.f, g: S.u, bc, gamma }); return errorsCG(r.space, r.U, S.u, S.grad); };
  // symmetric Nitsche, P1, N = 8: γ = 1 (indefinite) is more than ten times worse than γ = 10
  assert(err('nitsche', 1, 8, 1).L2 > 10 * err('nitsche', 1, 8, 10).L2, 'γ = 1 vs γ = 10');
  // penalty, γ = 10: the study N = 4…32 shows ≈ 1.5 for P1 (pre-asymptotic), ≈ 1 for P2 on N = 4…24
  const rate = (p, Ns, gamma, key) => fitRate(Ns.map((N) => 1 / N), Ns.map((N) => err('penalty', p, N, gamma)[key]));
  const r1 = rate(1, [4, 8, 16, 32], 10, 'L2'), r2 = rate(2, [4, 8, 16, 24], 10, 'L2'), r2h = rate(2, [4, 8, 16, 24], 10, 'H1');
  assert(r1 > 1.35 && r1 < 1.65, `P1 penalty L2 rate ${r1}`);
  assert(r2 > 0.85 && r2 < 1.15, `P2 penalty L2 rate ${r2}`);
  assert(r2h > 0.75 && r2h < 1.1, `P2 penalty H1 rate ${r2h}`);
  assert(rate(1, [4, 8, 16, 32], 100, 'L2') > 1.85, 'P1 penalty, γ = 100 still looks optimal');
});
