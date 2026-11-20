import { test, assert, assertClose, assertRate } from '../harness.js';
import {
  makeFV1D, cellAverages, LIMITERS, LIMITER_PHI, advanceFV1D, totalVariation,
  erf, upwindNumericalDiffusion, diffusedSquare,
} from '../../lib/core/fv/fv1d.js';
import { advection, burgers, buckley, burgersRiemann, scalarRiemann } from '../../lib/core/models/scalar.js';
import { makeStepper } from '../../lib/core/time/rk.js';
import { fitRate } from '../../lib/core/verify/rates.js';

// the jumps are declared, so cellAverages integrates piecewise (exact averages on any mesh)
const square = Object.assign((x) => (x > 0.25 && x < 0.6 ? 1 : 0), { breaks: [0.25, 0.6] });
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
  assertClose(sShock, 1.366, 2e-3, 0, 'shock speed quoted in chapter 5 (≈ 1.37)'); // M = ½: u* = 1/√3, speed (1 + √3)/2
  assertClose(us, 1 / Math.sqrt(3), 2e-3, 0, 'post-shock state');
});

/* ---------------------------------------------------------------------- */
/* Regression tests added by the audit                                     */
/* ---------------------------------------------------------------------- */
import { maxWaveSpeed, rusanovFlux, godunovFlux } from '../../lib/core/models/scalar.js';
import { coarsenAverages, reaStepExact, reaStepFormula, cellPieces, shiftedBreaks } from '../../lib/core/fv/fv1d.js';

test('maxWaveSpeed: max |f\'| over the interval between two states (non-convex: interior maximum)', () => {
  const m = buckley(0.5);
  assert(m.df(0) === 0 && m.df(1) === 0, 'Buckley–Leverett: f\'(0) = f\'(1) = 0');
  assertClose(maxWaveSpeed(m, 1, 0), 2.0807933, 1e-6, 0, 'BL 1|0');
  assertClose(maxWaveSpeed(m, 0, 1), 2.0807933, 1e-6, 0, 'BL 0|1');
  // never below a brute-force maximum, on random intervals (also outside [0, 1], where MUSCL overshoots can go)
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let n = 0; n < 300; n++) {
    const a = -3 + 7 * rnd(), b = -3 + 7 * rnd();
    let brute = 0; for (let k = 0; k <= 4000; k++) brute = Math.max(brute, Math.abs(m.df(a + (b - a) * k / 4000)));
    const v = maxWaveSpeed(m, a, b);
    assert(v >= brute - 1e-12 && v <= brute * (1 + 1e-5) + 1e-12, `[${a}, ${b}]: ${v} vs brute force ${brute}`);
  }
  // convex and linear models: the end states suffice (unchanged behaviour)
  assertClose(maxWaveSpeed(burgers, -0.3, 0.8), 0.8, 0, 1e-15); assertClose(maxWaveSpeed(advection(-0.7), 5, 9), 0.7, 0, 1e-15);
});

test('Rusanov flux is monotone and α ≥ |Rankine–Hugoniot speed| also for the non-convex Buckley–Leverett flux', () => {
  const m = buckley(0.5), e = 1e-6;
  for (let i = 0; i <= 70; i++) for (let j = 0; j <= 70; j++) {
    const uL = -0.2 + 0.02 * i, uR = -0.2 + 0.02 * j;
    const dL = (rusanovFlux(m, uL + e, uR) - rusanovFlux(m, uL - e, uR)) / (2 * e);
    const dR = (rusanovFlux(m, uL, uR + e) - rusanovFlux(m, uL, uR - e)) / (2 * e);
    assert(dL > -1e-5 && dR < 1e-5, `not monotone at (${uL}, ${uR}): ∂F/∂uL = ${dL}, ∂F/∂uR = ${dR}`);
    if (i !== j) assert(maxWaveSpeed(m, uL, uR) >= Math.abs((m.f(uR) - m.f(uL)) / (uR - uL)) - 1e-12, 'α ≥ |â|');
  }
  // the textbook failure of "max over the two end states": between 1 and 0 that would be the central flux
  assert(rusanovFlux(m, 1, 0) > 0.5 * (m.f(1) + m.f(0)) + 1, 'dissipation is present for 1|0');
});

test('CFL speed of a non-convex flux: Buckley–Leverett Riemann problems and pulses made of 0s and 1s run with a finite Δt', () => {
  const m = buckley(0.5);
  // maxSpeed must see the fast wave hidden between the cell values 0 and 1
  const fv0 = makeFV1D({ model: m, N: 100, a: -1, b: 1, flux: 'godunov', bc: 'outflow' });
  const jump = cellAverages((x) => (x < 0 ? 1 : 0), fv0.xf);
  assertClose(fv0.maxSpeed(jump), 2.0807933, 1e-6, 0, 'maxSpeed of the 1|0 jump');
  // Riemann A (1 → 0): rarefaction + shock (compound wave); first-order schemes converge to the convex-hull solution
  const T = 0.3, R = scalarRiemann(m, 1, 0, 8000);
  for (const flux of ['godunov', 'rusanov', 'roefix']) {
    const hs = [], es = [];
    for (const N of [100, 200, 400]) {
      const fv = makeFV1D({ model: m, N, a: -1, b: 1, flux, bc: 'outflow' });
      const u = cellAverages((x) => (x < 0 ? 1 : 0), fv.xf);
      const { steps } = advanceFV1D(fv, u, T, 0.8, 'fe');
      assert(steps >= 0.9 * T * 2.08 / (0.8 * fv.h), `${flux} N=${N}: only ${steps} steps`);
      assert(Math.max(...u) <= 1 + 1e-12 && Math.min(...u) >= -1e-12, `${flux} N=${N}: range [${Math.min(...u)}, ${Math.max(...u)}]`);
      const ex = cellAverages((x) => R(x / T), fv.xf);
      let e = 0; for (let i = 0; i < N; i++) e += Math.abs(u[i] - ex[i]) * fv.h;
      hs.push(fv.h); es.push(e);
    }
    assert(es[2] < 0.012, `${flux}: L1 error ${es[2]}`);
    const r = fitRate(hs, es);
    assert(r > 0.6 && r < 1.1, `${flux}: L1 rate ${r}`);
  }
  // periodic square pulse (cell values exactly 0 or 1): values stay in [0, 1], mass conserved
  const fv = makeFV1D({ model: m, N: 100, flux: 'godunov' });
  const u = cellAverages(square, fv.xf), m0 = u.reduce((a, b) => a + b, 0);
  const { steps } = advanceFV1D(fv, u, 0.2, 0.8, 'fe');
  assert(steps > 40, `square pulse: ${steps} steps`);
  assert(Math.max(...u) <= 1 + 1e-12 && Math.min(...u) >= -1e-12, 'square pulse stays in [0, 1]');
  assertClose(u.reduce((a, b) => a + b, 0), m0, 1e-11);
});

test('Riemann problems with transmissive boundaries: the mass changes by exactly (f(uL) − f(uR)) t', () => {
  for (const [m, uL, uR] of [[advection(1), 1, 0], [burgers, 1, 0], [burgers, -1, 1], [buckley(0.5), 1, 0]])
    for (const [recon, method] of [['none', 'fe'], ['muscl', 'ssprk3']]) {
      const fv = makeFV1D({ model: m, N: 100, a: -1, b: 1, flux: 'godunov', recon, limiter: 'mc', bc: 'outflow' });
      const u = cellAverages((x) => (x < 0 ? uL : uR), fv.xf), m0 = u.reduce((a, b) => a + b, 0) * fv.h;
      const smax = maxWaveSpeed(m, uL, uR);
      assert(smax > 0.5 && smax < 10, `${m.name}: wave speed ${smax}`); // a zero speed must fail here, not give T = ∞
      const T = 0.5 / smax; // waves stay well inside (−1, 1)
      advanceFV1D(fv, u, T, 0.4, method);
      assertClose(u.reduce((a, b) => a + b, 0) * fv.h - m0, (m.f(uL) - m.f(uR)) * T, 1e-12, 0, `${m.name} ${uL}|${uR} ${recon}`);
    }
});

test('coarsenAverages: exact overlap averaging for any pair of cell counts', () => {
  const fine = Float64Array.from({ length: 2000 }, (_, k) => Math.sin(0.01 * k) + (k % 7));
  const mean = fine.reduce((a, b) => a + b, 0) / 2000;
  for (const N of [20, 30, 70, 110, 130, 160, 400, 2000]) {
    const c = coarsenAverages(fine, N);
    assert(c.every(Number.isFinite), `N=${N}: finite`);
    assertClose(c.reduce((a, b) => a + b, 0) / N, mean, 1e-12, 0, `N=${N}: mean conserved`);
  }
  // divisor case = plain block average; non-divisor case by hand: 3 fine cells → 2 coarse cells
  const c4 = coarsenAverages(Float64Array.of(1, 3, 5, 7), 2); assertClose(c4[0], 2); assertClose(c4[1], 6);
  const c3 = coarsenAverages(Float64Array.of(1, 4, 10), 2); assertClose(c3[0], (1 + 0.5 * 4) / 1.5); assertClose(c3[1], (0.5 * 4 + 10) / 1.5);
});

test('REA with linear reconstruction: exact averaging = closed form; Lax–Wendroff, Beam–Warming and Fromm are special slopes', () => {
  const u = Float64Array.of(0.25, 0.3, 0.45, 0.95, 1.0, 0.7, 0.35, 0.3, 0.6, 0.35), M = u.length;
  const at = (i) => u[((i % M) + M) % M];
  const slopes = {
    const: () => 0, lw: (i) => at(i + 1) - at(i), bw: (i) => at(i) - at(i - 1), fromm: (i) => 0.5 * (at(i + 1) - at(i - 1)),
    minmod: (i) => LIMITERS.minmod.fn(at(i) - at(i - 1), at(i + 1) - at(i)),
  };
  for (const nu of [0, 0.05, 0.4, 0.75, 1]) {
    for (const k of Object.keys(slopes)) {
      const sig = Float64Array.from({ length: M }, (_, i) => slopes[k](i));
      const a = reaStepExact(u, sig, nu), b = reaStepFormula(u, sig, nu);
      for (let i = 0; i < M; i++) assertClose(a[i], b[i], 1e-14, 0, `${k} ν=${nu} cell ${i}`);
      assertClose(a.reduce((x, y) => x + y, 0), u.reduce((x, y) => x + y, 0), 1e-13, 0, 'conservative');
    }
    const sig = (k) => Float64Array.from({ length: M }, (_, i) => slopes[k](i));
    const lw = reaStepFormula(u, sig('lw'), nu), bw = reaStepFormula(u, sig('bw'), nu), up = reaStepFormula(u, sig('const'), nu);
    for (let i = 0; i < M; i++) {
      // classical stencils
      assertClose(up[i], (1 - nu) * at(i) + nu * at(i - 1), 1e-14, 0, 'upwind');
      assertClose(lw[i], at(i) - 0.5 * nu * (at(i + 1) - at(i - 1)) + 0.5 * nu * nu * (at(i + 1) - 2 * at(i) + at(i - 1)), 1e-14, 0, 'Lax–Wendroff');
      assertClose(bw[i], at(i) - 0.5 * nu * (3 * at(i) - 4 * at(i - 1) + at(i - 2)) + 0.5 * nu * nu * (at(i) - 2 * at(i - 1) + at(i - 2)), 1e-14, 0, 'Beam–Warming');
    }
  }
});

test('von Neumann analysis of upwind: |g(θ)|² = 1 − 2ν(1−ν)(1 − cos θ); the sawtooth grows by |1 − 2ν| per step', () => {
  const N = 64, fv = makeFV1D({ model: advection(1), N, flux: 'upwind' }), step = makeStepper('fe', N);
  const amp = (u, k) => { let re = 0, im = 0; for (let j = 0; j < N; j++) { re += u[j] * Math.cos(2 * Math.PI * k * j / N); im -= u[j] * Math.sin(2 * Math.PI * k * j / N); } return Math.hypot(re, im); };
  for (const nu of [0.3, 0.8, 1, 1.2])
    for (const k of [1, 5, 16, 32]) {
      const theta = 2 * Math.PI * k / N, u = Float64Array.from({ length: N }, (_, j) => Math.cos(theta * j));
      const a0 = amp(u, k);
      step(u, 0, nu * fv.h, fv.rhs);
      assertClose(amp(u, k) / a0, Math.sqrt(1 - 2 * nu * (1 - nu) * (1 - Math.cos(theta))), 1e-12, 0, `ν=${nu} θ=${theta}`);
    }
  const saw = Float64Array.from({ length: N }, (_, j) => (j % 2 ? -1 : 1));
  for (let n = 0; n < 10; n++) step(saw, 0, 1.2 * fv.h, fv.rhs);
  assertClose(Math.max(...saw.map(Math.abs)), 1.4 ** 10, 0, 1e-10, 'ν = 1.2: factor 1.4 per step');
});

test('Godunov flux by sampling (non-convex f) agrees with Osher\'s min/max formula evaluated finely', () => {
  const m = buckley(0.5);
  for (const [uL, uR] of [[1, 0], [0, 1], [0.2, 0.9], [0.9, 0.2], [-0.1, 0.4], [1.1, -0.1]]) {
    let best = uL <= uR ? Infinity : -Infinity;
    for (let k = 0; k <= 20000; k++) { const v = m.f(uL + (uR - uL) * k / 20000); best = uL <= uR ? Math.min(best, v) : Math.max(best, v); }
    assertClose(godunovFlux(m, uL, uR), best, 1e-4, 0, `${uL}|${uR}`);
  }
});

test('cellAverages with declared break points: exact averages of piecewise-smooth data on any mesh (square pulse N = 20…400)', () => {
  const plain = (x) => (x > 0.25 && x < 0.6 ? 1 : 0); // the same pulse WITHOUT declared jumps
  let worstPlain = 0;
  for (let N = 20; N <= 400; N++) {
    const xf = Float64Array.from({ length: N + 1 }, (_, i) => i / N), h = 1 / N;
    const u = cellAverages(square, xf);
    let mass = 0;
    for (let i = 0; i < N; i++) {
      mass += u[i] * h;
      // exact average = covered fraction of the cell
      const frac = Math.max(0, Math.min(xf[i + 1], 0.6) - Math.max(xf[i], 0.25)) / h;
      assertClose(u[i], frac, 1e-13, 0, `N=${N} cell ${i}`);
    }
    assertClose(mass, 0.35, 1e-14, 0, `N=${N}: initial mass`);
    worstPlain = Math.max(worstPlain, Math.abs(cellAverages(plain, xf).reduce((a, b) => a + b, 0) * h - 0.35));
    // the pulse shifted by t (exact solution of advection): still exact, wherever the jumps are
    const t = 0.123 + 0.01 * (N % 7), sh = (x) => square((((x - t) % 1) + 1) % 1);
    assertClose(cellAverages(sh, xf, shiftedBreaks(square.breaks, t)).reduce((a, b) => a + b, 0) * h, 0.35, 1e-13, 0, `N=${N}: shifted mass`);
  }
  assert(worstPlain > 1e-3, `without break points the mass is wrong by up to ${worstPlain}`); // the old behaviour (N = 30: 4.7e-3)
  // piecewise polynomial with a jump and a kink inside cells: exact
  const f = Object.assign((x) => (x < 0.37 ? x ** 3 : x < 0.71 ? 2 - x * x : 2 - 0.71 ** 2 + 5 * (x - 0.71)), { breaks: [0.71, 0.37] });
  const xf = Float64Array.from({ length: 8 }, (_, i) => i / 7);
  const I = cellAverages(f, xf).reduce((a, b) => a + b, 0) / 7;
  const exact = 0.37 ** 4 / 4 + (2 * (0.71 - 0.37) - (0.71 ** 3 - 0.37 ** 3) / 3) + (2 - 0.71 ** 2) * 0.29 + 2.5 * 0.29 ** 2;
  assertClose(I, exact, 1e-14, 0, 'piecewise polynomial');
  // smooth f: bit-identical to one 5-point Gauss rule per cell (backward compatible)
  const gx = [-0.9061798459386640, -0.5384693101056831, 0, 0.5384693101056831, 0.9061798459386640];
  const gw = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891];
  const g = (x) => Math.sin(3 * x) + x * x, a = cellAverages(g, xf);
  for (let i = 0; i < 7; i++) {
    const c = 0.5 * (xf[i] + xf[i + 1]), r = 0.5 * (xf[i + 1] - xf[i]); let s = 0;
    for (let q = 0; q < 5; q++) s += gw[q] * g(c + r * gx[q]);
    assert(a[i] === 0.5 * s, `smooth cell ${i} unchanged`);
  }
  // cellPieces: sorted, de-duplicated, only interior points
  const pc = cellPieces(0, 1, [0.5, 2, 0.2, 0.5, 0, 1, -3]);
  assert(pc.length === 4 && pc[0] === 0 && pc[1] === 0.2 && pc[2] === 0.5 && pc[3] === 1, `cellPieces ${pc}`);
});

test('exact Riemann solutions report their break points: averages of the exact solution conserve mass exactly', () => {
  // Burgers shock 1|0 (speed ½) and Buckley–Leverett 1|0 (shock speed 1.366): ∫ u dx over (−1, 1) = 1 + f(uL) t
  for (const [m, n] of [[burgers, 4000], [buckley(0.5), 20000]]) {
    const R = scalarRiemann(m, 1, 0, n), T = 0.3;
    assert(R.shocks.length >= 1 && R.breaks.length >= 2, 'a shock is reported');
    assertClose(R.shocks[R.shocks.length - 1], m.name === 'burgers' ? 0.5 : 1.366, 2e-3, 0, `${m.name} shock speed`);
    for (const N of [30, 100, 170]) {
      const xf = Float64Array.from({ length: N + 1 }, (_, i) => -1 + 2 * i / N);
      const mass = cellAverages((x) => R(x / T), xf, R.breaks.map((s) => s * T)).reduce((a, b) => a + b, 0) * 2 / N;
      assertClose(mass, 1 + m.f(1) * T, m.name === 'burgers' ? 1e-12 : 2e-4, 0, `${m.name} N=${N}`); // BL fan: staircase of n samples
    }
  }
  const fan = scalarRiemann(burgers, -1, 1, 2000);
  assert(fan.shocks.length === 0, 'a pure rarefaction has no shock');
  assertClose(fan.breaks[0], -1, 2e-3); assertClose(fan.breaks[fan.breaks.length - 1], 1, 2e-3);
});

test('advanceFV1D never hangs: it throws on a non-finite final time, on a blown-up solution and beyond maxSteps', () => {
  const fv = makeFV1D({ model: advection(1), N: 20, flux: 'upwind' });
  const throws = (fn, what) => { let ok = false; try { fn(); } catch (_) { ok = true; } assert(ok, `${what} must throw`); };
  throws(() => advanceFV1D(fv, cellAverages(square, fv.xf), Infinity, 0.5, 'fe'), 'T = Infinity');
  throws(() => advanceFV1D(fv, cellAverages(square, fv.xf), NaN, 0.5, 'fe'), 'T = NaN');
  throws(() => advanceFV1D(fv, cellAverages(square, fv.xf), 1, 0, 'fe'), 'cfl = 0');
  const fn = makeFV1D({ model: burgers, N: 20, flux: 'godunov' }); // nonlinear: the wave speed of NaN data is NaN
  throws(() => advanceFV1D(fn, new Float64Array(20).fill(NaN), 1, 0.5, 'fe'), 'NaN data');
  throws(() => advanceFV1D(fv, cellAverages(square, fv.xf), 1, 0.5, 'fe', 10), 'maxSteps = 10');
  // Burgers data that blow up (central flux, forward Euler): ends with an exception or a finite step count
  const fb = makeFV1D({ model: burgers, N: 40, flux: 'central' }), u = cellAverages((x) => Math.sin(2 * Math.PI * x), fb.xf);
  let steps = -1; try { steps = advanceFV1D(fb, u, 50, 0.9, 'fe', 1e5).steps; } catch (_) { steps = 0; }
  assert(steps >= 0 && steps <= 1e5, 'terminates');
  // normal use is unchanged
  assert(advanceFV1D(fv, cellAverages(square, fv.xf), 1, 0.5, 'fe').steps === 40, '40 steps at ν = ½ on 20 cells');
});

test('modified-equation widget numbers: ratio (distance to exact) : (distance to advection–diffusion) for N = 25, 100 and ν = 0.5, 0.95', () => {
  // exactly what the chapter-5 widget computes: an integer number of forward-Euler steps to t = 1
  const ratio = (N, nu0) => {
    const fv = makeFV1D({ model: advection(1), N, flux: 'upwind' }), u = cellAverages(square, fv.xf), step = makeStepper('fe', N);
    const n = Math.round(1 / (nu0 * fv.h)), dt = 1 / n, nu = dt / fv.h;
    for (let k = 0; k < n; k++) step(u, k * dt, dt, fv.rhs);
    const ad = cellAverages(diffusedSquare(0.25, 0.6, 1, upwindNumericalDiffusion(1, fv.h, nu), 1), fv.xf), ex = cellAverages(square, fv.xf);
    return l1(u, ex) / l1(u, ad);
  };
  const r = [ratio(100, 0.5), ratio(25, 0.5), ratio(100, 0.95), ratio(25, 0.95)];
  assert(r[0] > 650 && r[0] < 950, `N=100, ν=0.5: 1 : ${r[0]}`);   // caption: about 800
  assert(r[1] > 300 && r[1] < 450, `N=25, ν=0.5: 1 : ${r[1]}`);    // caption: about 370 (0.25 is not a face: needs exact averages)
  assert(r[2] > 9 && r[2] < 14, `N=100, ν≈0.95: 1 : ${r[2]}`);     // caption: about 11
  assert(r[3] > 3 && r[3] < 5.5, `N=25, ν≈0.95: 1 : ${r[3]}`);     // caption: about 4
});
