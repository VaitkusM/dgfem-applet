/**
 * Tests for flux reconstruction (lib/core/fr/fr1d.js): equivalences with DG,
 * convergence rates, conservation, linear stability and CFL limits vs c.
 */
import { test, assert, assertClose, assertRate } from '../harness.js';
import { makeFR1D, frBlochEigs, frOptimalC } from '../../lib/core/fr/fr1d.js';
import { makeDG1D, integrate } from '../../lib/core/dg/dg1d.js';
import { cSD, cHU } from '../../lib/core/fr/corrections.js';
import { advection, burgers } from '../../lib/core/models/scalar.js';
import { makeStepper } from '../../lib/core/time/rk.js';
import { maxStableCFL } from '../../lib/core/dg/spectrum.js';
import { fitRate } from '../../lib/core/verify/rates.js';

const TWO_PI = 2 * Math.PI;
function rnd(n, seed = 5) {
  const u = new Float64Array(n);
  for (let i = 0; i < n; i++) { seed = (seed * 16807) % 2147483647; u[i] = seed / 2147483647 - 0.3; }
  return u;
}
const maxDiff = (a, b) => { let d = 0; for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i])); return d; };

test('FR with g_DG = nodal DG (strong form, exact mass, collocated flux) to round-off — any flux, Gauss or GLL points', () => {
  for (const model of [advection(1), burgers]) for (const pts of ['gauss', 'gll']) for (let p = 1; p <= 5; p++) {
    const fr = makeFR1D({ p, N: 7, model, flux: 'rusanov', points: pts, correction: 'dg' });
    const dg = makeDG1D({ p, N: 7, model, flux: 'rusanov', nodes: pts, quad: 'exact', form: 'strong' });
    const u = rnd(fr.n), a = new Float64Array(fr.n), b = new Float64Array(fr.n);
    fr.rhs(u, 0, a); dg.rhs(u, 0, b);
    assert(maxDiff(a, b) < 1e-11, `${model.name} ${pts} p=${p}: ${maxDiff(a, b)}`);
  }
});

test('FR(g_DG) and weak-form DG with exact integration stay identical over a full advection run', () => {
  const p = 3, N = 12;
  const fr = makeFR1D({ p, N, model: advection(1), flux: 'upwind', correction: 'dg' });
  const dg = makeDG1D({ p, N, model: advection(1), flux: 'upwind', nodes: 'gauss', quad: 'exact', form: 'weak' });
  const u1 = fr.project((x) => (x > 0.2 && x < 0.5 ? 1 : 0)), u2 = Float64Array.from(u1);
  integrate(makeStepper('ssprk3', fr.n), u1, fr.rhs, 1, 0.1 * fr.h);
  integrate(makeStepper('ssprk3', dg.n), u2, dg.rhs, 1, 0.1 * dg.h);
  assert(maxDiff(u1, u2) < 1e-12, `difference ${maxDiff(u1, u2)}`);
});

test('FR with g_2 on GLL points = DG spectral element method (GLL collocation, lumped mass)', () => {
  for (const model of [advection(1), burgers]) for (let p = 1; p <= 5; p++) {
    const fr = makeFR1D({ p, N: 7, model, flux: 'rusanov', points: 'gll', correction: 'g2' });
    const dg = makeDG1D({ p, N: 7, model, flux: 'rusanov', nodes: 'gll', quad: 'collocated' });
    const u = rnd(fr.n), a = new Float64Array(fr.n), b = new Float64Array(fr.n);
    fr.rhs(u, 0, a); dg.rhs(u, 0, b);
    assert(maxDiff(a, b) < 1e-11, `${model.name} p=${p}: ${maxDiff(a, b)}`);
  }
});

test('FR converges at rate p+1 for g_2, g_Ga and VCJH (c = c_HU/2, 2 c_HU)', () => {
  const u0 = (x) => Math.sin(TWO_PI * x);
  for (let p = 2; p <= 4; p++) for (const [corr, c] of [['g2', 0], ['ga', 0], ['vcjh', 0.5 * cHU(p)], ['vcjh', 2 * cHU(p)]]) {
    const hs = [], es = [];
    for (const N of [8, 16, 32]) {
      const fr = makeFR1D({ p, N, model: advection(1), flux: 'upwind', correction: corr, c });
      const u = fr.project(u0);
      integrate(makeStepper('lsrk4', fr.n), u, fr.rhs, 1, 0.03 * fr.h / (2 * p + 1));
      hs.push(1 / N); es.push(fr.l2Error(u, u0));
    }
    assertRate(fitRate(hs, es), p + 1, `${corr} c=${c} p=${p}`);
  }
});

test('FR conserves mass exactly (Burgers, any correction function)', () => {
  for (const [corr, c] of [['dg', 0], ['g2', 0], ['ga', 0], ['vcjh', 0.1]]) {
    const fr = makeFR1D({ p: 3, N: 16, model: burgers, flux: 'rusanov', correction: corr, c });
    const u = fr.project((x) => 0.5 + Math.sin(TWO_PI * x));
    const m0 = fr.mass(u);
    integrate(makeStepper('ssprk3', fr.n), u, fr.rhs, 0.1, 0.002);
    assertClose(fr.mass(u), m0, 1e-13, 0, corr);
  }
});

test('VCJH with upwind flux is linearly stable (Re λ ≤ 0) for c ≥ 0', () => {
  for (let p = 1; p <= 4; p++) for (const c of [0, cSD(p), cHU(p), 10 * cHU(p), 1e3 * cHU(p)]) {
    const ev = frBlochEigs(p, { c }, 64);
    for (let k = 0; k < ev.re.length; k++) assert(ev.re[k] < 1e-9, `p=${p} c=${c} Re=${ev.re[k]}`);
  }
});

test('Larger c allows larger time steps: CFL(c_DG) < CFL(c_SD) < CFL(c_HU) < CFL(c_+), c_+ > c_HU, and CFL decreases again for c ≫ c_+', () => {
  for (let p = 1; p <= 4; p++) {
    const v = [0, cSD(p), cHU(p)].map((c) => maxStableCFL(frBlochEigs(p, { c }, 128), 'rk4'));
    const opt = frOptimalC(p, 'rk4');
    assert(v[0] < v[1] && v[1] < v[2] && v[2] < opt.cfl, `p=${p}: ${v} ${opt.cfl}`);
    assert(opt.c > cHU(p), `p=${p}: c+ = ${opt.c}`);
    if (p === 3) assert(opt.cfl > 2 * v[0], 'p=3: c+ more than doubles the DG time step');
    // beyond c+ the CFL limit decreases again
    assert(maxStableCFL(frBlochEigs(p, { c: 1e3 * cHU(p) }, 128), 'rk4') < opt.cfl - 0.02, `p=${p}: large c`);
  }
});

test('Accuracy price of large c: error at c = 64 c_HU is > 10× the DG (c = 0) error (p = 3, N = 16)', () => {
  const p = 3, u0 = (x) => Math.sin(TWO_PI * x);
  const err = (c) => {
    const fr = makeFR1D({ p, N: 16, model: advection(1), flux: 'upwind', correction: 'vcjh', c });
    const u = fr.project(u0);
    integrate(makeStepper('lsrk4', fr.n), u, fr.rhs, 1, 0.03 * fr.h / (2 * p + 1));
    return fr.l2Error(u, u0);
  };
  const e0 = err(0), eBig = err(64 * cHU(p));
  assert(eBig > 10 * e0, `${eBig} vs ${e0}`);
});

test('Linear flux: FR evolves the same polynomial whatever the solution points (Gauss vs GLL)', () => {
  const p = 3, N = 10, f0 = (x) => Math.exp(-40 * (x - 0.4) ** 2);
  for (const [corr, c] of [['g2', 0], ['vcjh', 0.01]]) {
    const A = makeFR1D({ p, N, model: advection(1), flux: 'upwind', points: 'gauss', correction: corr, c });
    const B = makeFR1D({ p, N, model: advection(1), flux: 'upwind', points: 'gll', correction: corr, c });
    const ua = A.project(f0), ub = B.project(f0);
    integrate(makeStepper('ssprk3', A.n), ua, A.rhs, 0.5, 0.05 * A.h);
    integrate(makeStepper('ssprk3', B.n), ub, B.rhs, 0.5, 0.05 * B.h);
    let d = 0;
    for (let k = 0; k <= 200; k++) { const x = (k + 0.5) / 201; d = Math.max(d, Math.abs(A.evalAt(ua, x) - B.evalAt(ub, x))); }
    assert(d < 1e-12, `${corr}: ${d}`);
  }
});


test('Over 0 ≤ c ≤ 64 c_HU the error grows monotonically with c (p = 1..4, N = 16)', () => {
  const u0 = (x) => Math.sin(TWO_PI * x);
  for (let p = 1; p <= 4; p++) {
    let prev = 0;
    for (const f of [0, 0.5, 1, 2, 4, 8, 16, 64]) {
      const fr = makeFR1D({ p, N: 16, model: advection(1), flux: 'upwind', correction: 'vcjh', c: f * cHU(p) });
      const u = fr.project(u0);
      integrate(makeStepper('lsrk4', fr.n), u, fr.rhs, 1, 0.05 * fr.h / (2 * p + 1));
      const e = fr.l2Error(u, u0);
      assert(e > prev, `p=${p} c=${f} c_HU: ${e} <= ${prev}`);
      prev = e;
    }
  }
});

/* ---------------------------------------------------------------------------------------------
 * Added by the chapter 6/7 audit
 * ------------------------------------------------------------------------------------------- */
import { frMatchesDG } from '../../lib/core/fr/fr1d.js';
import { interpMatrix, diffMatrix, lagrangeValues } from '../../lib/core/basis/lagrange.js';
import { operatorMatrix } from '../../lib/core/dg/spectrum.js';
import { symEig } from '../../lib/core/la/dense.js';
import { gaussLegendre } from '../../lib/core/quad/gauss1d.js';

test('frMatchesDG (labels of the FR-vs-DG widget) agrees with the actual right-hand sides for every pairing', () => {
  const DGV = { strong: { form: 'strong', quad: 'exact' }, weak: { form: 'weak', quad: 'exact' }, sem: { form: 'strong', quad: 'collocated', nodes: 'gll' } };
  const p = 3, N = 6, np = p + 1;
  for (const model of [advection(1), burgers]) for (const corr of ['dg', 'g2', 'ga']) for (const pts of ['gauss', 'gll']) for (const dgv of ['strong', 'weak', 'sem']) {
    const flux = model.linear ? 'upwind' : 'rusanov';
    const fr = makeFR1D({ p, N, model, flux, points: pts, correction: corr });
    const dg = makeDG1D({ p, N, model, flux, nodes: pts, ...DGV[dgv] });
    // the same piecewise polynomial in both nodal bases
    const uF = rnd(fr.n, 17), uD = new Float64Array(dg.n), I = interpMatrix(fr.r, dg.r);
    for (let k = 0; k < N; k++) for (let i = 0; i < np; i++) { let s = 0; for (let j = 0; j < np; j++) s += I[i * np + j] * uF[k * np + j]; uD[k * np + i] = s; }
    const a = new Float64Array(fr.n), b = new Float64Array(dg.n);
    fr.rhs(uF, 0, a); dg.rhs(uD, 0, b);
    // compare the two du/dt polynomials at sample points
    let d = 0, scale = 0;
    const la = new Float64Array(np), lb = new Float64Array(np);
    for (let k = 0; k < N; k++) for (const xi of [-0.9, -0.3, 0.1, 0.6, 0.95]) {
      lagrangeValues(fr.r, xi, la); lagrangeValues(dg.r, xi, lb);
      let va = 0, vb = 0; for (let i = 0; i < np; i++) { va += la[i] * a[k * np + i]; vb += lb[i] * b[k * np + i]; }
      d = Math.max(d, Math.abs(va - vb)); scale = Math.max(scale, Math.abs(va));
    }
    const same = d < 1e-10 * scale, expected = frMatchesDG({ correction: corr, points: pts, dg: dgv, linear: !!model.linear });
    assert(same === expected, `${model.name} ${corr} ${pts} vs ${dgv}: relative difference ${d / scale}, frMatchesDG = ${expected}`);
    if (!same) assert(d > 1e-4 * scale, `${model.name} ${corr} ${pts} vs ${dgv}: ambiguous difference ${d / scale}`);
  }
});

test('Linear flux: FR(g_2) on GAUSS points and DG-SEM (GLL) evolve the same polynomial over a whole run', () => {
  const p = 3, N = 10, f0 = (x) => 0.3 + Math.sin(TWO_PI * x);
  const fr = makeFR1D({ p, N, model: advection(1), flux: 'upwind', points: 'gauss', correction: 'g2' });
  const dg = makeDG1D({ p, N, model: advection(1), flux: 'upwind', nodes: 'gll', quad: 'collocated' });
  const a = fr.project(f0), b = dg.project(f0);
  integrate(makeStepper('ssprk3', fr.n), a, fr.rhs, 2, 0.5 * fr.h / (2 * p + 1));
  integrate(makeStepper('ssprk3', dg.n), b, dg.rhs, 2, 0.5 * dg.h / (2 * p + 1));
  let d = 0;
  for (let k = 0; k <= 300; k++) { const x = (k + 0.5) / 301; d = Math.max(d, Math.abs(fr.evalAt(a, x) - dg.evalAt(b, x))); }
  assert(d < 1e-12, `difference ${d}`);
});

test('Burgers, matching pair FR(g_DG) / strong-form DG on Gauss points: identical to round-off over a run through the shock', () => {
  const p = 3, N = 10, f0 = (x) => 0.3 + Math.sin(TWO_PI * x);
  const fr = makeFR1D({ p, N, model: burgers, flux: 'rusanov', points: 'gauss', correction: 'dg' });
  const dg = makeDG1D({ p, N, model: burgers, flux: 'rusanov', nodes: 'gauss', quad: 'exact', form: 'strong' });
  const a = fr.project(f0), b = dg.project(f0), sa = makeStepper('ssprk3', fr.n), sb = makeStepper('ssprk3', dg.n);
  let t = 0, d = 0, big = 0;
  while (t < 2 - 1e-12) {
    const dt = Math.min(0.5 * fr.h / (Math.max(fr.maxSpeed(a), 1e-3) * (2 * p + 1)), 2 - t);
    sa(a, t, dt, fr.rhs); sb(b, t, dt, dg.rhs); t += dt;
    for (let i = 0; i < fr.n; i++) { d = Math.max(d, Math.abs(a[i] - b[i])); big = Math.max(big, Math.abs(a[i])); }
  }
  assert(big < 3, `solution should stay bounded, max |u| = ${big}`);
  assert(d < 1e-11, `difference ${d}`);
});

test('VCJH energy stability: sym(K L) ≤ 0 in the norm K = M + (c/2)∫(∂_ξ^p u)², but NOT in the plain L2 norm for c > 0', () => {
  for (let p = 1; p <= 4; p++) for (const cf of [0, 0.5, 1, 10]) {
    const c = cf * cHU(p), N = 5, np = p + 1;
    const fr = makeFR1D({ p, N, a: 0, b: N, model: advection(1), flux: 'upwind', correction: 'vcjh', c, points: 'gauss' });
    const n = fr.n, L = operatorMatrix(fr.rhs, n), w = gaussLegendre(np).w, D = diffMatrix(fr.r);
    // row vector d with d·u = ∂_ξ^p u (a constant on the element): first row of D^p
    let Dp = new Float64Array(np * np); for (let i = 0; i < np; i++) Dp[i * np + i] = 1;
    for (let k = 0; k < p; k++) {
      const T = new Float64Array(np * np);
      for (let i = 0; i < np; i++) for (let j = 0; j < np; j++) { let s = 0; for (let m = 0; m < np; m++) s += D[i * np + m] * Dp[m * np + j]; T[i * np + j] = s; }
      Dp = T;
    }
    const maxEig = (withC) => {
      // K = diag(w) (exact Gauss mass) + (c/2)·∫_{-1}^{1} (d·u)² dξ = diag(w) + c d dᵀ
      const K = new Float64Array(np * np);
      for (let i = 0; i < np; i++) { K[i * np + i] = w[i]; if (withC) for (let j = 0; j < np; j++) K[i * np + j] += c * Dp[i] * Dp[j]; }
      const B = new Float64Array(n * n);
      for (let k = 0; k < N; k++) for (let i = 0; i < np; i++) for (let j = 0; j < n; j++) { let s = 0; for (let m = 0; m < np; m++) s += K[i * np + m] * L[(k * np + m) * n + j]; B[(k * np + i) * n + j] = s; }
      const S = new Float64Array(n * n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) S[i * n + j] = 0.5 * (B[i * n + j] + B[j * n + i]);
      return symEig(S, n).values[n - 1];
    };
    assert(maxEig(true) < 1e-11, `p=${p} c=${cf} c_HU: VCJH-norm energy rate ${maxEig(true)}`);
    if (cf > 0) assert(maxEig(false) > 0.1, `p=${p} c=${cf} c_HU: plain L2 should not be monotone (${maxEig(false)})`);
  }
});

test('CFL vs c for the other RK methods of the widget (SSP-RK3, LSRK4) and p = 5: DG < SD < HU < c_+, c_+ > c_HU, p = 3 gains > 2×', () => {
  for (const rk of ['ssprk3', 'lsrk4']) for (const p of [1, 3, 5]) {
    const v = [0, cSD(p), cHU(p)].map((c) => maxStableCFL(frBlochEigs(p, { c }, 96), rk));
    const opt = frOptimalC(p, rk, { iters: 16 });
    assert(v[0] < v[1] && v[1] < v[2] && v[2] < opt.cfl, `${rk} p=${p}: ${v} ${opt.cfl}`);
    assert(opt.c > cHU(p), `${rk} p=${p}: c+ = ${opt.c}`);
    if (p === 3) assert(opt.cfl > 2 * v[0], `${rk} p=3: ${opt.cfl} vs ${v[0]}`);
    assert(maxStableCFL(frBlochEigs(p, { c: 1e3 * cHU(p) }, 96), rk) < opt.cfl - 0.02, `${rk} p=${p}: large c`);
  }
  const p = 5, v = [0, cSD(p), cHU(p)].map((c) => maxStableCFL(frBlochEigs(p, { c }, 96), 'rk4'));
  assert(v[0] < v[1] && v[1] < v[2], `rk4 p=5: ${v}`);
});

test('FR with p = 1 also reaches rate 2 for g_2, g_Ga, c_HU/2 and 2 c_HU (finer meshes: pre-asymptotic for large c)', () => {
  const u0 = (x) => Math.sin(TWO_PI * x), p = 1;
  for (const [corr, c] of [['g2', 0], ['ga', 0], ['vcjh', 0.5 * cHU(p)], ['vcjh', 2 * cHU(p)]]) {
    const hs = [], es = [];
    for (const N of [32, 64, 128]) {
      const fr = makeFR1D({ p, N, model: advection(1), flux: 'upwind', correction: corr, c });
      const u = fr.project(u0);
      integrate(makeStepper('lsrk4', fr.n), u, fr.rhs, 1, 0.03 * fr.h / 3);
      hs.push(1 / N); es.push(fr.l2Error(u, u0));
    }
    assertRate(fitRate(hs, es), 2, `${corr} c=${c}`);
  }
});
