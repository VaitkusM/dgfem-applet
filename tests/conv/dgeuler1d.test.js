/**
 * Tests for the 1D Euler DG method (lib/core/dg/dgeuler1d.js): the "Systems" section of chapter 6.
 * Conservation, convergence on the smooth density wave, robustness on Sod's shock tube for every
 * setting the widget offers, and the two stages of the limiter.
 */
import { test, assert, assertClose, assertRate } from '../harness.js';
import { makeDGEuler1D, makeEulerLimiter, EULER_DG_PROBLEMS } from '../../lib/core/dg/dgeuler1d.js';
import { dgStableCFL, CFL_SAFETY } from '../../lib/core/dg/spectrum.js';
import { makeStepper } from '../../lib/core/time/rk.js';
import { exactRiemann } from '../../lib/core/models/eulerriemann.js';
import { physFlux, primToCons } from '../../lib/core/models/euler.js';
import { fitRate } from '../../lib/core/verify/rates.js';

/**
 * Run exactly like the chapter-6 widget: SSP-RK3, limiter after every stage,
 * Δt = frac · ν_max(p) · h / max(|u| + c). Stops at the first inadmissible state.
 */
function run(o) {
  const P = o.problem || EULER_DG_PROBLEMS[o.prob];
  const dg = makeDGEuler1D({ p: o.p, N: o.N, flux: o.flux, bc: P.bc });
  const U = dg.project(P.init);
  const lim = makeEulerLimiter(dg, { limit: o.lim !== 'none', M: o.lim === 'tvb' ? 50 : 0 });
  lim.apply(U);
  const step = makeStepper('ssprk3', dg.n, lim.apply);
  const nu = dgStableCFL(o.p) * (o.frac ?? CFL_SAFETY), T = o.T ?? P.T, T0 = dg.totals(U);
  let t = 0, bad = false;
  while (t < T - 1e-13) {
    const dt = Math.min(nu * dg.h / dg.maxSpeed(U), T - t);
    step(U, t, dt, dg.rhs); t += dt;
    const m = dg.minima(U);
    if (!m.finite || lim.stats.failed || !(m.rho > 0) || !(m.p > 0)) { bad = true; break; }
  }
  return { dg, U, t, bad, lim, T0, T1: dg.totals(U), min: dg.minima(U) };
}

test('Euler DG: consistency — a constant state has zero right-hand side (every flux, p = 0..3)', () => {
  for (const flux of ['rusanov', 'hll', 'hllc']) for (let p = 0; p <= 3; p++) for (const bc of ['periodic', 'outflow']) {
    const dg = makeDGEuler1D({ p, N: 5, flux, bc });
    const U = dg.project(() => [0.7, -0.4, 1.3]), out = new Float64Array(dg.n);
    dg.rhs(U, 0, out);
    for (let i = 0; i < dg.n; i++) assertClose(out[i], 0, 1e-12, 0, `${flux} p=${p} ${bc}`);
    // and the face flux is the physical flux of that state
    const F = physFlux(primToCons(0.7, -0.4, 1.3), [0, 0, 0]);
    for (let c = 0; c < 3; c++) assertClose(dg.faceFlux[3 * 2 + c], F[c], 1e-13);
  }
});

test('Euler DG conserves mass, momentum and energy to round-off (periodic, shocks, with and without limiter)', () => {
  const twoTubes = { bc: 'periodic', T: 0.1, init: (x) => (x > 0.25 && x < 0.75 ? [1, 0, 1] : [0.125, 0, 0.1]) };
  for (const flux of ['rusanov', 'hll', 'hllc']) for (const lim of ['none', 'minmod', 'tvb']) for (const p of [0, 1, 2, 3]) {
    const r = run({ problem: twoTubes, p, N: 40, flux, lim });
    assert(!r.bad, `${flux} ${lim} p=${p}: inadmissible state`);
    for (let c = 0; c < 3; c++) assertClose(r.T1[c], r.T0[c], 2e-13, 0, `${flux} ${lim} p=${p} component ${c}`);
  }
});

test('Euler DG: smooth density wave converges at rate p+1 (p = 1..3), p = 0 at first order', () => {
  const P = EULER_DG_PROBLEMS.wave;
  for (const flux of ['rusanov', 'hllc']) {
    for (let p = 1; p <= 3; p++) {
      const hs = [], es = [];
      for (const N of [8, 16, 32, 64]) {
        const r = run({ prob: 'wave', p, N, flux, lim: 'none', frac: 0.2 });
        hs.push(1 / N); es.push(r.dg.l1Error(r.U, 0, (x) => P.exact(x, r.t)[0]));
      }
      assertRate(fitRate(hs, es), p + 1, `${flux} p=${p}`, 0.35, 0.35);
    }
    const e = [128, 256].map((N) => { const r = run({ prob: 'wave', p: 0, N, flux, lim: 'none' }); return r.dg.l1Error(r.U, 0, (x) => P.exact(x, r.t)[0]); });
    const rate = Math.log2(e[0] / e[1]);
    assert(rate > 0.75 && rate < 1.15, `${flux} p=0 rate ${rate}`);
  }
});

test('Euler DG: TVB limiter (M = 50) leaves the smooth wave untouched, minmod (M = 0) costs accuracy', () => {
  const P = EULER_DG_PROBLEMS.wave, err = (lim) => { const r = run({ prob: 'wave', p: 2, N: 20, flux: 'hllc', lim }); return r.dg.l1Error(r.U, 0, (x) => P.exact(x, r.t)[0]); };
  const eNone = err('none'), eTVB = err('tvb'), eMM = err('minmod');
  assertClose(eTVB, eNone, 1e-12, 0, 'TVB = unlimited on smooth data');
  assert(eMM > 20 * eNone, `minmod ${eMM} vs none ${eNone}`);
});

test('Euler DG, Sod shock tube: every widget setting stays admissible (ρ, p > 0) and the L1 error of ρ decreases under refinement', () => {
  const P = EULER_DG_PROBLEMS.sod, ex = exactRiemann(P.WL, P.WR);
  for (const flux of ['rusanov', 'hll', 'hllc']) for (const lim of ['none', 'minmod', 'tvb']) for (let p = 0; p <= 3; p++) {
    const es = [];
    for (const N of [10, 30, 90]) {
      const r = run({ prob: 'sod', p, N, flux, lim });
      assert(!r.bad, `${flux} ${lim} p=${p} N=${N}: failed at t=${r.t}`);
      assert(r.min.rho > 0 && r.min.p > 0, `${flux} ${lim} p=${p} N=${N}: min ρ ${r.min.rho}, min p ${r.min.p}`);
      // with the minmod limiter there is no visible undershoot below the right state (ρ = 0.125, p = 0.1)
      if (lim === 'minmod') assert(r.min.rho > 0.12 && r.min.p > 0.095, `${flux} minmod p=${p} N=${N}: min ρ ${r.min.rho}, min p ${r.min.p}`);
      es.push(r.dg.l1Error(r.U, 0, (x) => ex.sample((x - P.x0) / r.t)[0]));
    }
    assert(es[1] < 0.8 * es[0] && es[2] < 0.8 * es[1], `${flux} ${lim} p=${p}: L1 errors ${es}`);
  }
});

test('Euler DG, Sod with minmod limiter: no new extrema in the density (within 2 %)', () => {
  for (let p = 1; p <= 3; p++) {
    const r = run({ prob: 'sod', p, N: 60, flux: 'hllc', lim: 'minmod' });
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < r.dg.n; i += 3) { lo = Math.min(lo, r.U[i]); hi = Math.max(hi, r.U[i]); }
    assert(lo > 0.125 - 0.02 && hi < 1 + 0.02, `p=${p}: density range ${lo} .. ${hi}`);
  }
});

test('Euler limiter: keeps cell averages; linear data untouched; positivity fallback repairs a negative nodal density', () => {
  const dg = makeDGEuler1D({ p: 3, N: 12, flux: 'hllc', bc: 'outflow' });
  // (a) conserved variables linear in x: untouched by the slope limiter (one-sided differences at the boundary)
  const lin = new Float64Array(dg.n);
  for (let i = 0; i < dg.n / 3; i++) { lin[3 * i] = 1 + 0.5 * dg.x[i]; lin[3 * i + 1] = 0.2 - 0.1 * dg.x[i]; lin[3 * i + 2] = 3 + dg.x[i]; }
  const c = Float64Array.from(lin), L = makeEulerLimiter(dg, { M: 0 });
  assert(L.apply(c) === 0 && L.stats.squeezed === 0, 'linear data must not be limited');
  for (let i = 0; i < dg.n; i++) assertClose(c[i], lin[i], 1e-13);
  // (b) a step: averages preserved exactly, result within the data range
  const st = dg.project((x) => (x < 0.47 ? [1, 0, 1] : [0.125, 0, 0.1]));
  const A0 = dg.cellAverages(st);
  assert(L.apply(st) > 0, 'the element containing the jump is troubled');
  const A1 = dg.cellAverages(st);
  for (let i = 0; i < A0.length; i++) assertClose(A1[i], A0[i], 1e-14);
  for (let i = 0; i < dg.n; i += 3) assert(st[i] > 0.125 - 1e-12 && st[i] < 1 + 1e-12, `density ${st[i]}`);
  // (c) positivity fallback alone (limit: false): one node pushed to negative density, mean still positive
  const U = dg.project(() => [1, 0, 1]);
  U[3 * (5 * dg.np + 1)] = -0.5;
  const B0 = dg.cellAverages(U), Pos = makeEulerLimiter(dg, { limit: false });
  Pos.apply(U);
  assert(Pos.stats.squeezed === 1 && Pos.squeezed[5] === 1 && !Pos.stats.failed, 'exactly one element squeezed');
  const m = dg.minima(U), B1 = dg.cellAverages(U);
  assert(m.rho > 0 && m.p > 0, `after fallback: min ρ ${m.rho}, min p ${m.p}`);
  for (let i = 0; i < B0.length; i++) assertClose(B1[i], B0[i], 1e-14);
  // (d) an inadmissible cell AVERAGE cannot be repaired: the limiter reports failure
  const W = dg.project(() => [1, 0, 1]);
  for (let i = 0; i < dg.np; i++) W[3 * (2 * dg.np + i)] = -1;
  const F = makeEulerLimiter(dg);
  F.apply(W);
  assert(F.stats.failed, 'negative mean density must be reported');
});

/* ---- round 3: the claims of the widget caption, at the widget's own settings (Δt = CFL_SAFETY·ν_max·h/max(|u|+c)) ---- */
test('Euler widget, smooth wave, HLLC, widget time step: order p+1 only without interfering limiter; p = 3 loses order to the RK3 time error; minmod ≈ 2.3', () => {
  const P = EULER_DG_PROBLEMS.wave;
  const err = (p, N, lim) => { const r = run({ prob: 'wave', p, N, flux: 'hllc', lim }); assert(!r.bad, 'admissible'); return r.dg.l1Error(r.U, 0, (x) => P.exact(x, r.t)[0]); };
  const order = (p, N, lim) => Math.log2(err(p, N, lim) / err(p, 2 * N, lim));
  for (const lim of ['none', 'tvb']) {
    assertRate(order(1, 20, lim), 2.1, `${lim} p=1`, 0.15, 0.15);
    assertRate(order(2, 20, lim), 3.0, `${lim} p=2`, 0.15, 0.15);
    assertRate(order(3, 20, lim), 3.85, `${lim} p=3, N = 20 → 40`, 0.15, 0.15);
  }
  assertRate(order(3, 80, 'none'), 3.2, 'p=3, N = 80 → 160 (time error dominates)', 0.15, 0.15);
  assertRate(order(0, 80, 'none'), 0.9, 'p=0, N = 80 → 160', 0.1, 0.1);
  // default limiter of the widget: minmod, M = 0
  for (let p = 1; p <= 3; p++) assertRate(order(p, 40, 'minmod'), 2.3, `minmod p=${p}`, 0.15, 0.15);
  assert(err(2, 40, 'minmod') > 1.5 * err(1, 40, 'minmod') && err(3, 40, 'minmod') > 1.5 * err(1, 40, 'minmod'), 'minmod: p = 2, 3 worse than p = 1');
  // "Try this": p = 3, N = 20 — tiny error without limiter, more than four orders of magnitude worse with minmod
  const e0 = err(3, 20, 'none');
  assert(e0 < 4e-7 && err(3, 20, 'minmod') > 1e4 * e0, `p=3 N=20: ${e0} vs ${err(3, 20, 'minmod')}`);
});

test('Euler widget: Sod for EVERY offered combination (p = 0..3, 3 fluxes, 3 limiters, N = 10, 20, …, 160: 576 runs) stays admissible', () => {
  let n = 0;
  for (const flux of ['rusanov', 'hll', 'hllc']) for (const lim of ['none', 'minmod', 'tvb']) for (let p = 0; p <= 3; p++) for (let N = 10; N <= 160; N += 10) {
    const r = run({ prob: 'sod', p, N, flux, lim });
    assert(!r.bad && r.min.rho > 0 && r.min.p > 0, `${flux} ${lim} p=${p} N=${N}: failed at t=${r.t} (min ρ ${r.min.rho}, min p ${r.min.p})`);
    n++;
  }
  assert(n === 576, 'sweep size');
});

test('Euler widget: smooth wave for every degree, flux and limiter on N = 10, 60, 160 stays admissible and conserves mass, momentum, energy', () => {
  for (const flux of ['rusanov', 'hll', 'hllc']) for (const lim of ['none', 'minmod', 'tvb']) for (let p = 0; p <= 3; p++) for (const N of [10, 60, 160]) {
    const r = run({ prob: 'wave', p, N, flux, lim });
    assert(!r.bad && r.min.rho > 0.75 && r.min.p > 0.99, `${flux} ${lim} p=${p} N=${N}: min ρ ${r.min.rho}, min p ${r.min.p}`);
    for (let c = 0; c < 3; c++) assertClose(r.T1[c], r.T0[c], 1e-11, 0, `${flux} ${lim} p=${p} N=${N} component ${c}`); // the widget's own tolerance
  }
});
