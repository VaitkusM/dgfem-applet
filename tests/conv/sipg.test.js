/**
 * Interior penalty DG (SIPG / NIPG / IIPG) for −Δu = f: convergence rates,
 * exactness, symmetry and the role of the penalty for coercivity.
 */
import { test, assert, assertRate } from '../harness.js';
import { triGrid } from '../../lib/core/mesh/structured.js';
import { solveIP, assembleIP, ipErrors, edgeJumps, smallestEigenvalue } from '../../lib/core/elliptic/sipg.js';
import { POISSON_MMS } from '../../lib/core/verify/mms.js';
import { fitRate } from '../../lib/core/verify/rates.js';
import { csrSymmetryError, csrToDense } from '../../lib/core/la/sparse.js';
import { symEig, cholesky } from '../../lib/core/la/dense.js';

const S = POISSON_MMS.wave;
const mesh = (N) => triGrid(N, N, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.15 });
const run = (p, variant, Ns, Cip = 2) => {
  const hs = [], L2 = [], En = [];
  for (const N of Ns) {
    const r = solveIP(mesh(N), { p, f: S.f, g: S.u, variant, Cip });
    const e = ipErrors(r, S.u, S.grad);
    hs.push(1 / N); L2.push(e.L2); En.push(e.energy);
  }
  return { L2: fitRate(hs, L2), En: fitRate(hs, En) };
};

test('SIPG: L2 rate p+1 and DG-energy rate p for p = 1, 2, 3', () => {
  for (const p of [1, 2, 3]) {
    const r = run(p, 'sipg', p === 1 ? [8, 16, 32] : [4, 8, 16]);
    assertRate(r.L2, p + 1, `SIPG p=${p} L2`);
    assertRate(r.En, p, `SIPG p=${p} energy`);
  }
});

test('NIPG / IIPG: L2 rate p+1 for odd p, only p for p = 2; energy rate p', () => {
  for (const variant of ['nipg', 'iipg']) {
    const r1 = run(1, variant, [8, 16, 32], 0.5);
    assertRate(r1.L2, 2, `${variant} p=1 L2`); assertRate(r1.En, 1, `${variant} p=1 energy`);
    const r2 = run(2, variant, [8, 16, 32], 0.5);
    assertRate(r2.L2, 2, `${variant} p=2 L2 (suboptimal)`, 0.2, 0.35); assertRate(r2.En, 2, `${variant} p=2 energy`);
    const r3 = run(3, variant, [4, 8, 16], 0.5);
    assertRate(r3.L2, 4, `${variant} p=3 L2`); assertRate(r3.En, 3, `${variant} p=3 energy`);
  }
});

test('IPDG is consistent: a polynomial solution of degree ≤ p is reproduced exactly (all variants)', () => {
  // u = x² − xy + 2y² + x,  −Δu = −(2 + 4) = −6
  const u = (x, y) => x * x - x * y + 2 * y * y + x, grad = (x, y) => [2 * x - y + 1, -x + 4 * y];
  for (const variant of ['sipg', 'nipg', 'iipg']) for (const p of [2, 3]) {
    const r = solveIP(mesh(4), { p, f: () => -6, g: u, variant, Cip: 3 });
    const e = ipErrors(r, u, grad);
    assert(e.L2 < 1e-12 && e.H1 < 1e-11, `${variant} p=${p}: errors ${e.L2}, ${e.H1}`);
    assert(Math.max(...edgeJumps(r, u).rms) < 1e-12, 'jumps vanish');
  }
});

test('SIPG matrix is symmetric, NIPG/IIPG are not', () => {
  const o = { p: 2, f: S.f, g: S.u, Cip: 2 };
  assert(csrSymmetryError(assembleIP(mesh(4), { ...o, variant: 'sipg' }).A) < 1e-13, 'SIPG symmetric');
  assert(csrSymmetryError(assembleIP(mesh(4), { ...o, variant: 'nipg' }).A) > 1e-3, 'NIPG non-symmetric');
  assert(csrSymmetryError(assembleIP(mesh(4), { ...o, variant: 'iipg' }).A) > 1e-3, 'IIPG non-symmetric');
});

test('SIPG: SPD for large penalty, indefinite for tiny penalty; threshold ~ independent of h', () => {
  for (const p of [1, 2, 3]) {
    const m = triGrid(2, 2);
    const lmin = (Cip) => { const A = assembleIP(m, { p, f: S.f, Cip }).A; return symEig(csrToDense(A), A.n).values[0]; };
    assert(lmin(2) > 0, `p=${p}: SPD for C_IP = 2`);
    assert(lmin(0.01) < 0, `p=${p}: indefinite for C_IP = 0.01`);
  }
  // with σ_F ∝ (p+1)(p+2)/h_F the critical C_IP stays in (0.3, 0.65) on all meshes tested
  for (const N of [2, 3]) for (const p of [1, 2, 3]) {
    const m = triGrid(N, N, [0, 1, 0, 1], { diag: 'alt' });
    const l = (Cip) => { const A = assembleIP(m, { p, f: S.f, Cip }).A; return symEig(csrToDense(A), A.n).values[0]; };
    assert(l(0.3) < 0 && l(0.65) > 0, `N=${N} p=${p} threshold in (0.3, 0.65)`);
  }
});

test('Lanczos smallestEigenvalue agrees with the dense eigen-solver (sign and value)', () => {
  const m = triGrid(3, 3, [0, 1, 0, 1], { diag: 'alt' });
  for (const Cip of [0.05, 0.4, 2]) {
    const A = assembleIP(m, { p: 2, f: S.f, Cip }).A;
    const exact = symEig(csrToDense(A), A.n).values[0], est = smallestEigenvalue(A);
    assert(Math.sign(exact) === Math.sign(est), `sign at C=${Cip}`);
    assert(Math.abs(est - exact) <= 1e-6 * Math.max(1, Math.abs(exact)), `value at C=${Cip}: ${est} vs ${exact}`);
  }
});

test('IP solves have tiny residuals (banded LU with pivoting)', () => {
  for (const variant of ['sipg', 'nipg', 'iipg']) {
    const r = solveIP(mesh(6), { p: 2, f: S.f, g: S.u, variant, Cip: 0.1 });
    assert(r.residual < 1e-11, `${variant} residual ${r.residual}`);
  }
});

test('penalty scaling: without 1/h the critical constant doubles with N; without the p-factor it grows like p(p+1)', () => {
  const crit = (N, p, sigmaFn) => {
    const m = triGrid(N, N, [0, 1, 0, 1], { diag: 'alt' });
    const spd = (C) => { const A = assembleIP(m, { p, f: S.f, Cip: C, sigmaFn }).A; return cholesky(csrToDense(A), A.n) !== null; };
    let lo = 1e-3, hi = 1e3;
    for (let i = 0; i < 28; i++) { const c = Math.sqrt(lo * hi); if (spd(c)) hi = c; else lo = c; }
    return hi;
  };
  const noH = (hF, p, C) => C * (p + 1) * (p + 2), noP = (hF, p, C) => C / hF;
  const r = crit(4, 1, noH) / crit(2, 1, noH);
  assert(r > 1.8 && r < 2.3, `no-h threshold ratio N=4/N=2: ${r}`);
  for (const p of [1, 2, 3]) {
    const c = crit(2, p, noP);
    assert(Math.abs(c / (p * (p + 1)) - 1) < 0.1, `no-p threshold p=${p}: ${c}`);
  }
});

test('large penalty: λ_max of the SIPG matrix grows proportionally to C_IP', () => {
  const m = triGrid(3, 3, [0, 1, 0, 1], { diag: 'alt' });
  const lmax = (Cip) => { const A = assembleIP(m, { p: 2, f: S.f, Cip }).A; const v = symEig(csrToDense(A), A.n).values; return v[v.length - 1]; };
  const r = lmax(200) / lmax(100);
  assert(r > 1.9 && r < 2.01, `ratio ${r}`);
});
