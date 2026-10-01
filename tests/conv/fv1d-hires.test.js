import { test, assert, assertClose, assertRate } from '../harness.js';
import {
  makeFV1D, cellAverages, LIMITERS, LIMITER_PHI, advanceFV1D, totalVariation,
  erf, upwindNumericalDiffusion, diffusedSquare,
} from '../../lib/core/fv/fv1d.js';
import { advection, burgers, buckley, burgersRiemann, scalarRiemann } from '../../lib/core/models/scalar.js';
import { makeStepper } from '../../lib/core/time/rk.js';
import { fitRate } from '../../lib/core/verify/rates.js';

const square = (x) => (x > 0.25 && x < 0.6 ? 1 : 0);
const l1 = (u, v) => { let s = 0; for (let i = 0; i < u.length; i++) s += Math.abs(u[i] - v[i]); return s / u.length; };

test('slope limiters σ(a, b) = φ(r)·b with r = a/b, and Sweby\'s TVD region 0 ≤ φ ≤ min(2r, 2)', () => {
  for (const k of ['minmod', 'mc', 'vanleer', 'superbee', 'none'])
    for (const a of [-1.3, -0.2, 0.1, 0.5, 1, 2.5])
      for (const b of [-0.7, 0.3, 1, 1.9]) assertClose(LIMITERS[k].fn(a, b), LIMITER_PHI[k](a / b) * b, 1e-14, 1e-14, `${k} a=${a} b=${b}`);
  for (const k of ['minmod', 'mc', 'vanleer', 'superbee'])
    for (let r = -2; r <= 6; r += 0.01) {
      const p = LIMITER_PHI[k](r);
      assert(p >= 0 && p <= Math.max(0, Math.min(2 * r, 2)) + 1e-14, `${k} leaves TVD region at r=${r}`);
      if (r > 0) assert(p >= Math.min(r, 1) - 1e-14 && p <= Math.max(r, 1) * (k === 'superbee' ? 2 : 1) + 1e-14, `${k} 2nd-order region`);
    }
  assertClose(LIMITER_PHI.minmod(1), 1); assertClose(LIMITER_PHI.vanleer(1), 1); assertClose(LIMITER_PHI.mc(1), 1); assertClose(LIMITER_PHI.superbee(1), 1);
  assert(LIMITER_PHI.none(-1) === 0 && LIMITER_PHI.none(5) > 2, 'Fromm (central slope) is outside the TVD region');
});

test('MUSCL + TVD limiter: total variation never increases (ν = 0.45, FE and SSP-RK3); unlimited slope creates new extrema', () => {
  for (const method of ['fe', 'ssprk2', 'ssprk3'])
    for (const lim of ['minmod', 'vanleer', 'mc', 'superbee']) {
      const fv = makeFV1D({ model: advection(1), N: 100, flux: 'upwind', recon: 'muscl', limiter: lim });
      const u = cellAverages(square, fv.xf), step = makeStepper(method, 100);
      let tv = totalVariation(u);
      for (let k = 0; k < 300; k++) {
        step(u, 0, 0.45 * fv.h, fv.rhs);
        const tv1 = totalVariation(u);
        assert(tv1 <= tv + 1e-12, `${method}/${lim}: TV grew ${tv} → ${tv1}`);
        tv = tv1;
      }
    }
  // the linear second-order scheme (Fromm, unlimited central slope): overshoots — Godunov's theorem in action
  const fv = makeFV1D({ model: advection(1), N: 100, flux: 'upwind', recon: 'muscl', limiter: 'none' });
  const u = cellAverages(square, fv.xf);
  advanceFV1D(fv, u, 0.3, 0.45, 'ssprk3');
  assert(Math.max(...u) > 1.01 && Math.min(...u) < -0.01, 'unlimited MUSCL produces over/undershoots');
  assert(totalVariation(u) > 2.02, 'TV increased above 2');
  // first-order upwind (a monotone scheme) keeps 0 ≤ u ≤ 1
  const fv1 = makeFV1D({ model: advection(1), N: 100, flux: 'upwind' });
  const v = cellAverages(square, fv1.xf);
  advanceFV1D(fv1, v, 0.3, 0.9, 'fe');
  assert(Math.max(...v) <= 1 + 1e-14 && Math.min(...v) >= -1e-14, 'upwind is monotone');
});

test('CFL: upwind + forward Euler is exact at ν = 1, bounded for ν ≤ 1, blows up for ν > 1', () => {
  const N = 50, fv = makeFV1D({ model: advection(1), N, flux: 'upwind' });
  const u0 = cellAverages(square, fv.xf);
  const step = makeStepper('fe', N);
  const u = Float64Array.from(u0);
  for (let k = 0; k < N; k++) step(u, 0, fv.h, fv.rhs); // ν = 1: one cell per step, N steps = one period
  for (let i = 0; i < N; i++) assertClose(u[i], u0[i], 1e-12);
  const v = Float64Array.from(u0);
  for (let k = 0; k < 200; k++) step(v, 0, 0.9 * fv.h, fv.rhs);
  assert(Math.max(...v) <= 1 + 1e-14 && Math.min(...v) >= -1e-14, 'ν = 0.9 stays in [0, 1]');
  const w = Float64Array.from(u0);
  for (let k = 0; k < 200; k++) step(w, 0, 1.2 * fv.h, fv.rhs);
  assert(Math.max(...w.map(Math.abs)) > 1e3, 'ν = 1.2 must blow up');
});

test('modified equation: upwind behaves like advection–diffusion with D = (a h / 2)(1 − ν)', () => {
  // Distance to the advection–diffusion solution is ~ 1e-3 of the distance to the exact solution and shrinks like h^1.5
  const hs = [], es = [];
  for (const N of [100, 200, 400]) {
    const nu = 0.5, fv = makeFV1D({ model: advection(1), N, flux: 'upwind' });
    const u = cellAverages(square, fv.xf);
    advanceFV1D(fv, u, 1, nu, 'fe');
    const D = upwindNumericalDiffusion(1, fv.h, nu);
    const ad = cellAverages(diffusedSquare(0.25, 0.6, 1, D, 1), fv.xf), ex = cellAverages(square, fv.xf);
    const eAD = l1(u, ad), eEx = l1(u, ex);
    assert(eAD < 2e-3 * eEx, `N=${N}: ${eAD} vs ${eEx}`);
    hs.push(fv.h); es.push(eAD);
  }
  assertRate(fitRate(hs, es), 1.5, 'distance to modified equation', 0.2, 0.3);
  assertClose(erf(0.5), 0.5204998778, 2e-7); assertClose(erf(-1.5), -0.9661051465, 2e-7);
});

test('L1 rates for advection: smooth ≈ 1 (upwind) and ≈ 2 (MUSCL, all limiters); square pulse ≈ 1/2 (upwind), ≈ 2/3 (MUSCL)', () => {
  const sine = (x) => Math.sin(2 * Math.PI * x);
  const rate = (u0, recon, lim, method) => {
    const hs = [], es = [];
    for (const N of [200, 400, 800]) {
      const fv = makeFV1D({ model: advection(1), N, flux: 'upwind', recon, limiter: lim });
      const u = cellAverages(u0, fv.xf);
      advanceFV1D(fv, u, 1, 0.45, method);
      hs.push(fv.h); es.push(l1(u, cellAverages(u0, fv.xf)));
    }
    return fitRate(hs, es);
  };
  assertRate(rate(sine, 'none', 'minmod', 'fe'), 1, 'smooth upwind', 0.1, 0.1);
  for (const lim of ['none', 'minmod', 'vanleer', 'mc', 'superbee']) assertRate(rate(sine, 'muscl', lim, 'ssprk3'), 2, `smooth ${lim}`, 0.15, 0.15);
  assertRate(rate(square, 'none', 'minmod', 'fe'), 0.5, 'square upwind', 0.05, 0.05);
  for (const lim of ['minmod', 'vanleer', 'mc']) assertRate(rate(square, 'muscl', lim, 'ssprk3'), 2 / 3, `square ${lim}`, 0.06, 0.06);
  // superbee is "compressive": it steepens fronts and keeps the jump within a few cells → rate ≈ 1
  assertRate(rate(square, 'muscl', 'superbee', 'ssprk3'), 1, 'square superbee', 0.1, 0.1);
});

test('scalar Riemann solver (convex hull): matches Burgers exactly, Welge tangent for Buckley–Leverett', () => {
  for (const [uL, uR] of [[1, -0.5], [-0.5, 1], [-1, 1], [0.3, 0.3]]) {
    const R = scalarRiemann(burgers, uL, uR, 4000);
    for (let xi = -1.2; xi <= 1.2; xi += 0.0137) {
      const ex = burgersRiemann(uL, uR, xi);
      // fan values are resolved to the sampling resolution 1/2000
      assert(Math.abs(R(xi) - ex) < 2e-3, `Burgers ${uL}|${uR} at ${xi}: ${R(xi)} vs ${ex}`);
    }
  }
  // Buckley–Leverett uL = 1, uR = 0: rarefaction from 1 down to u*, then a shock to 0 with speed f(u*)/u* = f'(u*)
  const m = buckley(0.5), R = scalarRiemann(m, 1, 0, 20000);
  let us = 1, sShock = 0;
  for (let xi = 0; xi < 3; xi += 1e-4) { const v = R(xi); if (v === 0) { sShock = xi; break; } us = v; }
  assertClose(m.f(us) / us, m.df(us), 2e-3, 0, 'Welge tangent');
  assertClose(sShock, m.f(us) / us, 2e-3, 0, 'shock speed = chord slope');
});
