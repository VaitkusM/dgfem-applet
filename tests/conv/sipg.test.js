/**
 * Interior penalty DG (SIPG / NIPG / IIPG) for −Δu = f: convergence rates,
 * exactness, symmetry and the role of the penalty for coercivity.
 */
import { test, assert, assertRate, assertClose } from '../harness.js';
import { triGrid } from '../../lib/core/mesh/structured.js';
import { solveIP, assembleIP, ipErrors, edgeJumps, smallestEigenvalue, ipFluxBalance } from '../../lib/core/elliptic/sipg.js';
import { bandSym, countBelow, smallestEigenvalueSym } from '../../lib/core/la/eigmin.js';
import { POISSON_MMS } from '../../lib/core/verify/mms.js';
import { fitRate } from '../../lib/core/verify/rates.js';
import { csrSymmetryError, csrToDense, csrAdd, csrTranspose } from '../../lib/core/la/sparse.js';
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

// ---------------------------------------------------------------------------------------------
// Claims made in chapter 8 (prose and widget captions) — AGENTS.md rule 3
// ---------------------------------------------------------------------------------------------
/** the mesh family of the chapter-8 widgets */
const widgetMesh = (N) => triGrid(N, N, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.12 });
/** symmetric part ½(A + Aᵀ) */
const symPart = (A) => csrAdd(A, csrTranspose(A), 0.5, 0.5);
/** true if the symmetric matrix A is positive definite (no eigenvalue below 0, by an LDLᵀ eigenvalue count) */
const isSPD = (A) => { const r = countBelow(bandSym(A), 0); assert(r.ok, 'pivot'); return r.neg === 0; };

test('IPDG is locally conservative: the numerical flux −{{∂n u_h}} + σ⟦u_h⟧ balances ∫_K f on every triangle (all variants); one-sided fluxes do not', () => {
  for (const variant of ['sipg', 'nipg', 'iipg']) for (const p of [1, 2, 3]) for (const key of ['wave', 'peak']) {
    const P = POISSON_MMS[key], r = solveIP(mesh(5), { p, f: P.f, g: P.u, variant, Cip: 1.5 });
    const B = ipFluxBalance(r, P.f, P.u);
    let res = 0, one = 0, scale = 0;
    for (let t = 0; t < B.source.length; t++) {
      res = Math.max(res, Math.abs(B.outflow[t] - B.source[t]));
      one = Math.max(one, Math.abs(B.oneSided[t] - B.source[t]));
      scale = Math.max(scale, Math.abs(B.source[t]), Math.abs(B.outflow[t]));
    }
    assert(res < 1e-10 * (1 + scale), `${variant} p=${p} ${key}: flux balance residual ${res} (scale ${scale})`);
    assert(one > 1e-4 * scale, `${variant} p=${p} ${key}: one-sided fluxes are not conservative (${one})`);
  }
});

test('λ_min is certified in the widget regime (500–1400 unknowns, λ_min tiny compared with λ_max)', () => {
  // independent check by dense Cholesky: A − (λ − δ)I is positive definite, A − (λ + δ)I is not
  const check = (p, N, Cip, variant = 'sipg') => {
    const A0 = assembleIP(widgetMesh(N), { p, f: S.f, Cip, variant }).A, A = variant === 'sipg' ? A0 : symPart(A0);
    const r = smallestEigenvalueSym(A, { k: 60 }), lam = smallestEigenvalue(A, { k: 60 }), n = A.n, D = csrToDense(A);
    assert(r.certified && lam === r.value && r.lower <= lam && lam <= r.upper, `certificate p=${p} N=${N} C=${Cip}`);
    const shifted = (s) => { const M = Float64Array.from(D); for (let i = 0; i < n; i++) M[i * n + i] -= s; return M; };
    const d = 1e-4 * Math.abs(lam);
    assert(cholesky(shifted(lam - d), n) !== null, `p=${p} N=${N} C=${Cip}: an eigenvalue lies below λ_min = ${lam}`);
    assert(cholesky(shifted(lam + d), n) === null, `p=${p} N=${N} C=${Cip}: no eigenvalue at λ_min = ${lam}`);
    return lam;
  };
  // p = 3, N = 5 (500 unknowns), C = 0.591: plain Lanczos on A with k = 60 is off by tens of percent here
  assertClose(check(3, 5, 0.591), -0.883, 2e-3, 0, 'p=3 N=5 just below the threshold');
  assert(check(2, 9, 0.45) < 0, 'p=2 N=9 C=0.45 indefinite');           // 972 unknowns
  assert(check(1, 15, 0.3) < 0, 'p=1 N=15 C=0.3 indefinite');           // 1350 unknowns
  assert(check(2, 9, 2) > 0, 'p=2 N=9 C=2 SPD');
  assert(check(2, 8, 0.01, 'nipg') > 0, 'NIPG symmetric part SPD');     // 768 unknowns
  assert(check(2, 8, 0.01, 'iipg') < 0, 'IIPG symmetric part indefinite');
});

test('coercivity thresholds of the default penalty: C* = p/(p+2) on the 8-triangle mesh; ≈ 0.35 / 0.5 / 0.6 independent of h', () => {
  const crit = (m, p) => {
    let lo = 0.05, hi = 2;
    for (let i = 0; i < 30; i++) { const c = Math.sqrt(lo * hi); if (isSPD(assembleIP(m, { p, f: S.f, Cip: c }).A)) hi = c; else lo = c; }
    return hi;
  };
  const m8 = triGrid(2, 2, [0, 1, 0, 1], { diag: 'alt' });
  for (const p of [1, 2, 3]) assertClose(crit(m8, p), p / (p + 2), 1e-3, 0, `8 triangles p=${p}`);
  // widget meshes N = 4, 6, 8 (h varies by a factor 2): λ_min changes sign at about 0.35 (p=1), 0.5 (p=2), 0.6 (p=3)
  const win = { 1: [0.34, 0.36], 2: [0.485, 0.505], 3: [0.585, 0.605] };
  for (const N of [4, 6, 8]) for (const p of [1, 2, 3]) {
    const c = crit(widgetMesh(N), p);
    assert(c > win[p][0] && c < win[p][1], `N=${N} p=${p}: C* = ${c}`);
  }
});

test('σ_F = 10/h_F (no p-factor) is coercive for p = 1, 2 but not for p = 3 (widget meshes)', () => {
  const noP = (hF, p, C) => C / hF;
  for (const N of [2, 4, 8]) for (const p of [1, 2, 3])
    assert(isSPD(assembleIP(widgetMesh(N), { p, f: S.f, Cip: 10, sigmaFn: noP }).A) === (p <= 2), `N=${N} p=${p}`);
});

test('NIPG is coercive for any positive penalty, IIPG is not: symmetric part at C_IP = 0.01', () => {
  for (const p of [1, 2, 3]) {
    const o = { p, f: S.f, Cip: 0.01 };
    assert(isSPD(symPart(assembleIP(widgetMesh(6), { ...o, variant: 'nipg' }).A)), `NIPG p=${p}`);
    assert(!isSPD(symPart(assembleIP(widgetMesh(6), { ...o, variant: 'iipg' }).A)), `IIPG p=${p}`);
  }
});

test('under-penalised SIPG is erratic: p = 1, C_IP = 0.05, default solution: L² error > 10 for N = 12, < 0.2 for N = 11, 13', () => {
  const err = (N) => { const r = solveIP(widgetMesh(N), { p: 1, f: S.f, g: S.u, variant: 'sipg', Cip: 0.05 }); return ipErrors(r, S.u, S.grad).L2; };
  assert(err(12) > 10, `N=12: ${err(12)}`);
  assert(err(11) < 0.2 && err(13) < 0.2, `N=11: ${err(11)}, N=13: ${err(13)}`);
});

test('convergence widget default (p = 2, C_IP = 1): NIPG and IIPG drift towards L² rate 2, SIPG keeps 3', () => {
  const rate = (variant) => {
    const e = [16, 32].map((N) => ipErrors(solveIP(widgetMesh(N), { p: 2, f: S.f, g: S.u, variant, Cip: 1 }), S.u, S.grad).L2);
    return Math.log2(e[0] / e[1]);
  };
  assertRate(rate('sipg'), 3, 'SIPG', 0.15, 0.15);
  const rn = rate('nipg'), ri = rate('iipg');
  assert(rn > 1.9 && rn < 2.4, `NIPG ${rn}`); assert(ri > 1.9 && ri < 2.6, `IIPG ${ri}`);
});

test('SIPG rates also hold for a second manufactured solution (sin·sin, homogeneous boundary data)', () => {
  const P = POISSON_MMS.sinsin;
  for (const p of [1, 2]) {
    const hs = [], L2 = [], En = [];
    for (const N of p === 1 ? [8, 16, 32] : [4, 8, 16]) {
      const e = ipErrors(solveIP(mesh(N), { p, f: P.f, g: P.u, Cip: 2 }), P.u, P.grad);
      hs.push(1 / N); L2.push(e.L2); En.push(e.energy);
    }
    assertRate(fitRate(hs, L2), p + 1, `sinsin p=${p} L2`); assertRate(fitRate(hs, En), p, `sinsin p=${p} energy`);
  }
});

test('adjoint consistency defect: a_h(v, z) − ∫(−Δz)v = (1 − θ) Σ_F ∫_F {{∂n z}}⟦v⟧ for smooth z vanishing on ∂Ω', () => {
  // z = sin(πx) sin(πy) (zero on ∂Ω) is represented by its accurate p = 3 SIPG approximation z_h (error O(h⁴));
  // v is an arbitrary broken polynomial with O(1) jumps. Then a_h(v, z_h) − ∫(−Δz) v ≈ (1 − θ) Σ_F ∫ {{∂n z}}⟦v⟧:
  // ≈ 0 for SIPG (θ = 1), D for IIPG (θ = 0), 2D for NIPG (θ = −1).
  const P = POISSON_MMS.sinsin, m = mesh(4), p = 3;
  const zs = solveIP(m, { p, f: P.f, g: P.u, variant: 'sipg', Cip: 4 });     // z_h ≈ z, error O(h⁴)
  const v = new Float64Array(zs.nDof).map((_, i) => Math.sin(1.7 * i + 0.3));  // arbitrary broken test function
  const defect = (variant) => {
    const S2 = assembleIP(m, { p, f: P.f, g: P.u, variant, Cip: 4 });
    // a_h(v, z_h) = z_hᵀ A v  (A_ij = a_h(φ_j, φ_i): first argument = column) ;  ∫(−Δz) v = ∫ f v = bᵀv (g = 0)
    let a = 0, l = 0;
    for (let i = 0; i < S2.nDof; i++) {
      for (let k = S2.A.rowPtr[i]; k < S2.A.rowPtr[i + 1]; k++) a += zs.U[i] * S2.A.vals[k] * v[S2.A.colIdx[k]];
      l += S2.b[i] * v[i];
    }
    return a - l;
  };
  const d1 = defect('sipg'), d0 = defect('iipg'), dm = defect('nipg');
  // SIPG: a_h(v, z_h) = a_h(z_h, v) = ℓ(v) exactly (symmetry + the discrete equation) ⇒ defect = round-off
  assert(Math.abs(d1) < 1e-9 * Math.abs(d0) && Math.abs(d0) > 0.1, `SIPG defect ${d1}, IIPG defect ${d0}`);
  assertClose(dm / d0, 2, 0.05, 0, 'NIPG defect = 2 × IIPG defect (factor 1 − θ)');
});
