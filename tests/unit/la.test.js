import { test, assert, assertClose, assertArrayClose } from '../harness.js';
import { luFactor, luSolve, inverse, matMul, matVec, transpose, solve, eye, cholesky, symEig, eigGeneral, det } from '../../lib/core/la/dense.js';
import { SparseBuilder, csrMatVec, csrToDense, csrTranspose, csrSymmetryError, csrPermute, csrBandwidth, csrExtract, csrAdd, csrGet, csrDiag, denseToCsr, nnz } from '../../lib/core/la/sparse.js';
import { rcm, sparseLU, sparseSolve } from '../../lib/core/la/direct.js';
import { cg, gmres, jacobiPrecond } from '../../lib/core/la/krylov.js';
import { conditionEstimate } from '../../lib/core/la/eig.js';

/** deterministic pseudo-random numbers in [-0.5, 0.5) */
function rng(seed = 1) { return () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; }; }

/** 2D 5-point Laplacian on an m×m interior grid (SPD). */
function laplace2D(m) {
  const n = m * m, B = new SparseBuilder(n);
  for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) {
    const k = j * m + i;
    B.add(k, k, 4);
    if (i > 0) B.add(k, k - 1, -1);
    if (i < m - 1) B.add(k, k + 1, -1);
    if (j > 0) B.add(k, k - m, -1);
    if (j < m - 1) B.add(k, k + m, -1);
  }
  return B.toCSR();
}

test('dense LU solve, inverse, determinant', () => {
  const r = rng(3), n = 7, A = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) A[i] = r();
  const x = Float64Array.from({ length: n }, (_, i) => i - 2);
  const b = new Float64Array(n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) b[i] += A[i * n + j] * x[j];
  assertArrayClose(luSolve(luFactor(A, n), b), x, 1e-11);
  assertArrayClose(matMul(A, inverse(A, n), n, n, n), eye(n), 1e-11);
  assertClose(det(Float64Array.of(2, 1, 1, 3), 2), 5, 1e-14);
  assertClose(det(Float64Array.of(0, 1, 1, 0), 2), -1, 1e-14);
});

test('Cholesky reproduces A and rejects indefinite matrices', () => {
  const A = Float64Array.of(4, 2, 0, 2, 5, 1, 0, 1, 3);
  const L = cholesky(A, 3);
  const LT = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) LT[i * 3 + j] = L[j * 3 + i];
  assertArrayClose(matMul(L, LT, 3, 3, 3), A, 1e-14);
  assert(cholesky(Float64Array.of(1, 2, 2, 1), 2) === null);
});

test('symmetric Jacobi eigenvalues of 1D Laplacian match 2-2cos(kπ/(n+1))', () => {
  const n = 12, A = new Float64Array(n * n);
  for (let i = 0; i < n; i++) { A[i * n + i] = 2; if (i > 0) A[i * n + i - 1] = A[(i - 1) * n + i] = -1; }
  const { values, vectors } = symEig(A, n);
  for (let k = 1; k <= n; k++) assertClose(values[k - 1], 2 - 2 * Math.cos(k * Math.PI / (n + 1)), 1e-12);
  // A v = λ v for the first eigenvector
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += A[i * n + j] * vectors[j * n];
    assertClose(s, values[0] * vectors[i * n], 1e-12);
  }
});

test('general eigenvalues: similarity transform of known spectrum incl. complex pairs', () => {
  const n = 6;
  // block diagonal D with eigenvalues 3, -1, 0.5±2i, -2±0.25i
  const D = new Float64Array(n * n);
  D[0] = 3; D[7] = -1;
  D[2 * n + 2] = 0.5; D[2 * n + 3] = 2; D[3 * n + 2] = -2; D[3 * n + 3] = 0.5;
  D[4 * n + 4] = -2; D[4 * n + 5] = 0.25; D[5 * n + 4] = -0.25; D[5 * n + 5] = -2;
  const r = rng(11), S = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) S[i] = r() + (i % (n + 1) === 0 ? 2 : 0);
  const A = matMul(matMul(S, D, n, n, n), inverse(S, n), n, n, n);
  const { re, im } = eigGeneral(A, n);
  const got = Array.from(re, (v, i) => [v, im[i]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const want = [[-2, -0.25], [-2, 0.25], [-1, 0], [0.5, -2], [0.5, 2], [3, 0]];
  for (let i = 0; i < n; i++) { assertClose(got[i][0], want[i][0], 1e-9); assertClose(got[i][1], want[i][1], 1e-9); }
});

test('general eigenvalues of a larger random matrix: trace and determinant', () => {
  const n = 40, r = rng(5), A = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) A[i] = r();
  const { re, im } = eigGeneral(A, n);
  let tr = 0; for (let i = 0; i < n; i++) tr += A[i * n + i];
  assertClose(re.reduce((a, b) => a + b, 0), tr, 1e-9);
  assertClose(im.reduce((a, b) => a + b, 0), 0, 1e-9);
  // product of eigenvalues (complex) = det
  let pr = 1, pi = 0;
  for (let i = 0; i < n; i++) { const a = pr * re[i] - pi * im[i], b = pr * im[i] + pi * re[i]; pr = a; pi = b; }
  assertClose(pr, det(A, n), 1e-8, 1e-8);
});

test('CSR builder sums duplicates; matvec, transpose, permute, extract', () => {
  const B = new SparseBuilder(3, 3);
  B.add(0, 0, 1); B.add(0, 0, 2); B.add(2, 1, 5); B.add(1, 2, -1); B.add(0, 2, 4);
  const A = B.toCSR();
  assertArrayClose(csrToDense(A), Float64Array.of(3, 0, 4, 0, 0, -1, 0, 5, 0));
  assertArrayClose(csrMatVec(A, Float64Array.of(1, 2, 3)), Float64Array.of(15, -3, 10));
  assertArrayClose(csrToDense(csrTranspose(A)), Float64Array.of(3, 0, 0, 0, 0, 5, 4, -1, 0));
  const P = csrPermute(A, Int32Array.of(2, 0, 1)); // new i ↦ old perm[i]
  assertArrayClose(csrToDense(P), Float64Array.of(0, 0, 5, 4, 3, 0, -1, 0, 0));
  assertArrayClose(csrToDense(csrExtract(A, [0, 2], [0, 1])), Float64Array.of(3, 0, 0, 5));
  assertClose(csrSymmetryError(laplace2D(4)), 0);
});

test('RCM reduces bandwidth of a shuffled Laplacian; banded LU solves', () => {
  const A0 = laplace2D(15), n = A0.n, r = rng(2);
  const shuffle = Int32Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor((r() + 0.5) * (i + 1)); [shuffle[i], shuffle[j]] = [shuffle[j], shuffle[i]]; }
  const A = csrPermute(A0, shuffle);
  const before = csrBandwidth(A).kl, after = csrBandwidth(csrPermute(A, rcm(A))).kl;
  assert(after <= 20 && after < before, `bandwidth ${before} -> ${after}`);
  const x = Float64Array.from({ length: n }, (_, i) => Math.sin(i));
  const b = csrMatVec(A, x);
  assertArrayClose(sparseSolve(A, b), x, 1e-10);
});

test('banded LU with pivoting solves an indefinite saddle-point system', () => {
  // [K Bᵀ; B 0] with K = Laplacian, B = sum constraint rows
  const m = 6, K = laplace2D(m), n = K.n, S = new SparseBuilder(n + 2);
  for (let i = 0; i < n; i++) for (let k = K.rowPtr[i]; k < K.rowPtr[i + 1]; k++) S.add(i, K.colIdx[k], K.vals[k]);
  for (let i = 0; i < n; i++) {
    const c = i < n / 2 ? n : n + 1;
    S.add(c, i, 1); S.add(i, c, 1);
  }
  const A = S.toCSR();
  const x = Float64Array.from({ length: n + 2 }, (_, i) => Math.cos(i));
  const b = csrMatVec(A, x);
  const F = sparseLU(A);
  assert(!F.singular);
  assertArrayClose(F.solve(b), x, 1e-10);
});

test('CG converges on SPD Laplacian; Jacobi PCG and GMRES agree', () => {
  const A = laplace2D(20), n = A.n;
  const x = Float64Array.from({ length: n }, (_, i) => Math.sin(0.1 * i));
  const b = csrMatVec(A, x);
  const r1 = cg(A, b, { tol: 1e-12 });
  assert(r1.converged); assertArrayClose(r1.x, x, 1e-9);
  const r2 = cg(A, b, { tol: 1e-12, precond: jacobiPrecond(A) });
  assert(r2.converged); assertArrayClose(r2.x, x, 1e-9);
  const r3 = gmres(A, b, { tol: 1e-12, restart: 30, maxIter: 2000 });
  assert(r3.converged); assertArrayClose(r3.x, x, 1e-8);
  // GMRES on a non-symmetric matrix (convection-diffusion-like)
  const B = new SparseBuilder(n);
  for (let i = 0; i < n; i++) for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) B.add(i, A.colIdx[k], A.vals[k]);
  for (let i = 1; i < n; i++) B.add(i, i - 1, -0.7);
  const N = B.toCSR(), bn = csrMatVec(N, x);
  const r4 = gmres(N, bn, { tol: 1e-12, restart: 250, maxIter: 3000, precond: jacobiPrecond(N) });
  assert(r4.converged); assertArrayClose(r4.x, x, 1e-8);
});

test('condition estimate of 2D Laplacian', () => {
  const m = 12, A = laplace2D(m);
  const lmin = 8 * Math.sin(Math.PI / (2 * (m + 1))) ** 2;
  const lmax = 8 * Math.cos(Math.PI / (2 * (m + 1))) ** 2;
  const c = conditionEstimate(A);
  assertClose(c.lmin, lmin, 0, 1e-8); assertClose(c.lmax, lmax, 0, 1e-8);
  assertClose(c.kappa, lmax / lmin, 0, 1e-7);
});

test('small dense helpers: matVec, transpose, solve; singular matrices are flagged', () => {
  const A = Float64Array.of(1, 2, 3, 4, 5, 6); // 2×3
  assertArrayClose(matVec(A, 2, 3, [1, 0, -1]), Float64Array.of(-2, -2));
  assertArrayClose(transpose(A, 2, 3), Float64Array.of(1, 4, 2, 5, 3, 6));
  assertArrayClose(solve(Float64Array.of(0, 2, 1, 1), 2, [4, 3]), Float64Array.of(1, 2), 1e-14); // needs the row swap
  assert(luFactor(Float64Array.of(1, 2, 2, 4), 2).singular === true);
  assert(luFactor(Float64Array.of(1, 2, 2, 5), 2).singular === false);
});

test('small CSR helpers: get, diag, add, dense round trip, nnz, bandwidth', () => {
  const D = Float64Array.of(2, 0, 1, 0, 3, 0, -1, 0, 0), A = denseToCsr(D, 3);
  assert(nnz(A) === 4);
  assertArrayClose(csrToDense(A), D);
  assert(csrGet(A, 0, 2) === 1 && csrGet(A, 1, 0) === 0);
  assertArrayClose(csrDiag(A), Float64Array.of(2, 3, 0));
  assertArrayClose(csrToDense(csrAdd(A, csrTranspose(A), 2, -1)), Float64Array.of(2, 0, 3, 0, 3, 0, -3, 0, 0));
  const bw = csrBandwidth(A);
  assert(bw.kl === 2 && bw.ku === 2);
  assertClose(csrSymmetryError(A), 2 / 3, 1e-15); // entries 1 and −1 differ by 2, max |A_ij| = 3
});

test('banded LU: random non-symmetric systems with weak diagonals agree with dense LU (with and without RCM)', () => {
  const r = rng(5);
  let checked = 0;
  for (let rep = 0; rep < 120; rep++) {
    const n = 5 + Math.floor((r() + 0.5) * 30), B = new SparseBuilder(n);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) B.add(i, Math.floor((r() + 0.5) * n), r());
      if (r() > 0) B.add(i, i, r() * 0.01); // tiny or missing diagonal ⇒ pivoting is essential
      B.add(i, (i + 1) % n, 1 + r());
    }
    const A = B.toCSR(), x = Float64Array.from({ length: n }, () => r()), b = csrMatVec(A, x);
    const xd = solve(csrToDense(A), n, b);
    let ed = 0;
    for (let i = 0; i < n; i++) ed = Math.max(ed, Math.abs(xd[i] - x[i]));
    if (ed > 1e-8) continue; // too ill-conditioned to be a fair comparison
    checked++;
    for (const reorder of [true, false]) assertArrayClose(sparseLU(A, { reorder }).solve(b), x, 1e-7, 0, `rep ${rep} reorder=${reorder}`);
  }
  assert(checked > 80, `only ${checked} systems were well conditioned`);
});

test('general eigenvalues: degenerate cases (zero, Jordan block, rotation, cyclic shift, 1×1)', () => {
  let e = eigGeneral(new Float64Array(9), 3);
  assertArrayClose(e.re, new Float64Array(3)); assertArrayClose(e.im, new Float64Array(3));
  e = eigGeneral(Float64Array.of(2, 1, 0, 0, 2, 1, 0, 0, 2), 3);
  assertArrayClose(e.re, Float64Array.of(2, 2, 2), 1e-12); assertArrayClose(e.im, new Float64Array(3), 1e-12);
  e = eigGeneral(Float64Array.of(0, -1, 1, 0), 2);
  assertArrayClose(e.re, Float64Array.of(0, 0), 1e-14); assertClose(Math.abs(e.im[0]), 1, 1e-14); assertClose(e.im[0] + e.im[1], 0, 1e-14);
  e = eigGeneral(Float64Array.of(7), 1);
  assert(e.re[0] === 7 && e.im[0] === 0);
  // cyclic shift of size 20: eigenvalues are the 20th roots of unity (needs the exceptional shifts)
  const n = 20, C = new Float64Array(n * n);
  for (let i = 0; i < n; i++) C[i * n + (i + 1) % n] = 1;
  e = eigGeneral(C, n);
  const ang = Array.from(e.re, (v, i) => { assertClose(Math.hypot(v, e.im[i]), 1, 1e-12); return Math.atan2(e.im[i], v); }).sort((a, b) => a - b);
  for (let i = 1; i < n; i++) assertClose(ang[i] - ang[i - 1], 2 * Math.PI / n, 1e-10);
});
