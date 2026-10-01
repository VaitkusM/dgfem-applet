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
