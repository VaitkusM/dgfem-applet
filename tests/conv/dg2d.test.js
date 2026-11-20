/**
 * Tests for the 2D DG spectral element method on periodic quads (lib/core/dg/dg2d.js).
 */
import { test, assert, assertClose, assertRate } from '../harness.js';
import { makeDG2D } from '../../lib/core/dg/dg2d.js';
import { integrate } from '../../lib/core/dg/dg1d.js';
import { makeStepper } from '../../lib/core/time/rk.js';
import { fitRate } from '../../lib/core/verify/rates.js';

const TWO_PI = 2 * Math.PI;

test('2D DG-SEM, constant-velocity advection: L2 rate p+1 (p = 1..4)', () => {
  const ax = 1, ay = 0.5, T = 0.5;
  const ex = (t) => (x, y) => Math.sin(TWO_PI * (x - ax * t)) * Math.sin(TWO_PI * (y - ay * t));
  for (let p = 1; p <= 4; p++) {
    const Ns = p === 1 ? [8, 16, 32] : p === 4 ? [3, 6, 12] : [4, 8, 16];
    const hs = [], es = [];
    for (const N of Ns) {
      const dg = makeDG2D({ p, nx: N, ny: N, velocity: () => [ax, ay] });
      const u = dg.project(ex(0));
      integrate(makeStepper('lsrk4', dg.n), u, dg.rhs, T, dg.dtStable(u, 0.3));
      hs.push(1 / N); es.push(dg.l2Error(u, ex(T)));
    }
    assertRate(fitRate(hs, es), p + 1, `p=${p}`, 0.3, 0.3);
  }
});

test('2D DG-SEM conserves mass exactly (rotating flow and Burgers)', () => {
  const blob = (x, y) => Math.exp(-((x - 0.7) ** 2 + (y - 0.5) ** 2) / 0.01);
  for (const o of [{ velocity: (x, y) => [-TWO_PI * (y - 0.5), TWO_PI * (x - 0.5)] }, { problem: 'burgers' }]) {
    const dg = makeDG2D({ p: 3, nx: 8, ny: 8, ...o });
    const u = dg.project(o.problem ? (x, y) => 0.5 + 0.5 * Math.sin(TWO_PI * (x + y)) : blob);
    const m0 = dg.mass(u);
    const step = makeStepper('ssprk3', dg.n);
    for (let k = 0; k < 40; k++) step(u, 0, dg.dtStable(u, 0.5), dg.rhs);
    assertClose(dg.mass(u), m0, 1e-13);
    for (let i = 0; i < dg.n; i++) assert(Number.isFinite(u[i]), 'finite');
  }
});

test('2D rotating blob: one revolution returns the blob, error decreases with p', () => {
  const blob = (x, y) => Math.exp(-((x - 0.7) ** 2 + (y - 0.5) ** 2) / 0.01);
  const vel = (x, y) => [-TWO_PI * (y - 0.5), TWO_PI * (x - 0.5)];
  const errs = [];
  for (const p of [1, 2, 3, 4]) {
    const dg = makeDG2D({ p, nx: 10, ny: 10, velocity: vel });
    const u = dg.project(blob);
    integrate(makeStepper('ssprk3', dg.n), u, dg.rhs, 1, dg.dtStable(u, 0.5));
    errs.push(dg.l2Error(u, blob));
  }
  for (let k = 1; k < errs.length; k++) assert(errs[k] < 0.6 * errs[k - 1], `errors ${errs}`);
});

/* ---- added by the chapter 6/7 audit ---- */
import { blownUp } from '../../lib/core/dg/dg1d.js';

/**
 * The time loop of the chapter-6 2D widget for Burgers, including its guard (stop at |u| > 5).
 * @returns {{stop: number|null, max: number}} time at which the guard fires (or null) and the largest |u| seen before
 */
function burgers2d(p, N, T = 0.4) {
  const dg = makeDG2D({ p, nx: N, ny: N, flux: 'rusanov', problem: 'burgers' });
  const u = dg.project((x, y) => 0.5 + 0.7 * Math.sin(TWO_PI * (x + y)));
  const step = makeStepper('ssprk3', dg.n);
  let t = 0, max = 0;
  while (t < T - 1e-12) {
    const dt = Math.min(dg.dtStable(u, 0.5), T - t);
    step(u, t, dt, dg.rhs); t += dt;
    if (blownUp(u, 5)) return { stop: t, max };
    for (let i = 0; i < dg.n; i++) max = Math.max(max, Math.abs(u[i]));
  }
  return { stop: null, max };
}

test('2D Burgers without limiter, every mesh of the widget (N = 3..16): p ≤ 3 stays within |u| ≤ 3.3; the guard |u| > 5 stops p = 4 for N = 8 and N ≥ 11 and p = 5 for N ≥ 4', () => {
  for (const p of [1, 2, 3]) for (let N = 3; N <= 16; N++) {
    const r = burgers2d(p, N);
    assert(r.stop === null && r.max < 3.3, `p=${p} N=${N}: ${JSON.stringify(r)}`);
  }
  for (let N = 3; N <= 16; N++) {
    const r4 = burgers2d(4, N), stops4 = N === 8 || N >= 11;
    assert((r4.stop !== null) === stops4, `p=4 N=${N}: ${JSON.stringify(r4)}`);
    const r5 = burgers2d(5, N);
    assert((r5.stop !== null) === (N >= 4), `p=5 N=${N}: ${JSON.stringify(r5)}`);
    // the guard never fires before the shock has formed (t = 1/(2.8π) ≈ 0.114), and a completed run stayed ≤ 5
    for (const r of [r4, r5]) assert(r.stop === null ? r.max <= 5 : r.stop > 0.1137, `N=${N}: ${JSON.stringify(r)}`);
  }
});

test('dtStable never returns NaN or 0 for finite data, and blownUp flags the state that would make it NaN', () => {
  const dg = makeDG2D({ p: 2, nx: 4, ny: 4, problem: 'burgers' });
  const u = dg.project(() => 1);
  assert(Number.isFinite(dg.dtStable(u, 0.5)) && dg.dtStable(u, 0.5) > 0, 'finite dt');
  u[3] = NaN;
  assert(blownUp(u), 'NaN must be detected before the next time step is computed');
});
