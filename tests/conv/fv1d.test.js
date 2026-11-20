import { test, assert, assertClose, assertRate } from '../harness.js';
import { makeFV1D, cellAverages } from '../../lib/core/fv/fv1d.js';
import { advection, burgers, burgersRiemann, godunovFlux, rusanovFlux, roeFlux, upwindFlux, centralFlux, buckley } from '../../lib/core/models/scalar.js';
import { makeStepper } from '../../lib/core/time/rk.js';
import { fitRate } from '../../lib/core/verify/rates.js';

/** run FV to time T with fixed CFL; returns {u, fv} */
function run(o, u0, T, cfl, method = 'ssprk3') {
  const fv = makeFV1D(o);
  const u = cellAverages(u0, fv.xf);
  const step = makeStepper(method, fv.N);
  let t = 0;
  while (t < T - 1e-14) {
    const s = Math.max(fv.maxSpeed(u), 1e-12);
    const dt = Math.min(cfl * fv.h / s, T - t);
    step(u, t, dt, fv.rhs);
    t += dt;
  }
  return { u, fv };
}

const l1 = (u, ex, fv) => { let s = 0; for (let i = 0; i < fv.N; i++) s += Math.abs(u[i] - ex[i]) * fv.h; return s; };

test('FV upwind is first order, MUSCL (unlimited) second order for smooth advection', () => {
  const u0 = (x) => Math.sin(2 * Math.PI * x);
  for (const [recon, expected] of [['none', 1], ['muscl', 2]]) {
    const hs = [], es = [];
    for (const N of [40, 80, 160, 320]) {
      const { u, fv } = run({ model: advection(1), N, flux: 'upwind', recon, limiter: 'none' }, u0, 1, 0.4);
      hs.push(fv.h); es.push(l1(u, cellAverages(u0, fv.xf), fv));
    }
    assertRate(fitRate(hs, es), expected, recon);
  }
});

test('FV conserves the total mass to round-off (periodic, nonlinear)', () => {
  const u0 = (x) => 1 + 0.5 * Math.sin(2 * Math.PI * x);
  for (const flux of ['godunov', 'rusanov', 'roefix']) {
    const fv = makeFV1D({ model: burgers, N: 100, flux, recon: 'muscl', limiter: 'mc' });
    const u = cellAverages(u0, fv.xf);
    const m0 = u.reduce((a, b) => a + b, 0);
    const step = makeStepper('ssprk3', 100);
    for (let k = 0; k < 200; k++) step(u, 0, 0.002, fv.rhs);
    assertClose(u.reduce((a, b) => a + b, 0), m0, 1e-11);
  }
});

test('Godunov FV converges to the exact Burgers Riemann solution (shock and rarefaction)', () => {
  // Expected L1 rates of a first-order monotone scheme: ≈ 1 for a shock (self-sharpening),
  // about 0.75–0.8 for a rarefaction on these grids (kinks at the fan edges; ~ h|log h| behaviour).
  // Measured: shock 1.00; fans 0.77 (−0.5|1) and 0.79 (−1|1). The fan window excludes rate 1 on purpose.
  for (const [uL, uR, lo, hi] of [[1, -0.5, 0.9, 1.1], [-0.5, 1, 0.68, 0.9], [-1, 1, 0.68, 0.9]]) {
    const errs = [], hs = [];
    for (const N of [400, 800, 1600]) {
      const { u, fv } = run({ model: burgers, N, a: -1, b: 1, flux: 'godunov', bc: 'outflow' }, (x) => (x < 0 ? uL : uR), 0.5, 0.5);
      const ex = cellAverages((x) => burgersRiemann(uL, uR, x / 0.5), fv.xf);
      errs.push(l1(u, ex, fv)); hs.push(fv.h);
    }
    const r = fitRate(hs, errs);
    assert(r > lo && r < hi, `uL=${uL} uR=${uR}: L1 rate ${r}`);
  }
});

test('Roe flux without entropy fix keeps a non-physical expansion shock; with fix it does not', () => {
  const opts = { model: burgers, N: 200, a: -1, b: 1, bc: 'outflow' };
  const u0 = (x) => (x < 0 ? -1 : 1);
  const bad = run({ ...opts, flux: 'roe' }, u0, 0.5, 0.4).u;
  const good = run({ ...opts, flux: 'roefix' }, u0, 0.5, 0.4).u;
  // exact solution is a fan: u(±0.25) = ±0.5. Expansion shock: u stays ±1 there.
  const i = Math.floor((0.25 + 1) / 0.01);
  assert(Math.abs(bad[i] - 1) < 1e-6, `roe gives ${bad[i]}`);
  assert(Math.abs(good[i] - 0.5) < 0.1, `roe+fix gives ${good[i]}`);
});

test('numerical fluxes are consistent: F(u,u) = f(u)', () => {
  for (const m of [advection(-0.7), burgers, buckley(0.5)])
    for (const u of [-0.8, 0, 0.3, 0.9]) {
      assertClose(godunovFlux(m, u, u), m.f(u), 1e-14);
      assertClose(rusanovFlux(m, u, u), m.f(u), 1e-14);
      assertClose(roeFlux(m, u, u, 0.3), m.f(u), 1e-14);
      assertClose(upwindFlux(m, u, u), m.f(u), 1e-14);
      assertClose(centralFlux(m, u, u), m.f(u), 1e-14);
    }
});

test('upwind flux for a nonlinear f is Roe\'s flux without entropy fix (Burgers), and the Godunov flux for linear advection', () => {
  for (const uL of [-1.2, -0.4, 0.1, 0.7]) for (const uR of [-0.9, -0.1, 0.3, 1.5]) {
    assertClose(upwindFlux(burgers, uL, uR), roeFlux(burgers, uL, uR, 0), 1e-14, 0, `Burgers ${uL}|${uR}`);
    for (const a of [1, -0.7]) assertClose(upwindFlux(advection(a), uL, uR), godunovFlux(advection(a), uL, uR), 1e-15, 0, 'advection');
  }
});
