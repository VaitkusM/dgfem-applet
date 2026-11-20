import { test, assert, assertClose, assertRate } from '../harness.js';
import { makeFV2D, cellAverages2D, rotationVelocity, rotated, zalesakDisk, zalesakArea } from '../../lib/core/fv/fv2d.js';
import { makeStepper } from '../../lib/core/time/rk.js';
import { fitRate, pairwiseRates } from '../../lib/core/verify/rates.js';

const sum = (u) => u.reduce((a, b) => a + b, 0);

/** Advance to T with SSP-RK3 at the given CFL; returns {u, fv}. */
function run(o, u0, T, cfl = 0.4, refine = 0) {
  const fv = makeFV2D(o);
  const u = cellAverages2D(u0, o.nx, o.ny, fv.box, 4, refine); // refine > 0 for discontinuous data
  const step = makeStepper('ssprk3', o.nx * o.ny);
  const n = Math.ceil(T / fv.stableDt(cfl)), dt = T / n;
  assert(n >= 1 && n < 1e6, `run: ${n} steps`); // bounded loop
  for (let k = 0; k < n; k++) step(u, k * dt, dt, fv.rhs);
  return { u, fv };
}

test('2D FV: total mass conserved to round-off (periodic) and balanced by the boundary flux (inflow)', () => {
  const u0 = (x, y) => Math.exp(-40 * ((x - 0.6) ** 2 + (y - 0.4) ** 2));
  for (const recon of ['none', 'muscl']) {
    // periodic, variable divergence-free velocity a = (sin 2πy, cos 2πx)
    const o = { nx: 40, ny: 30, vel: (x, y) => [Math.sin(2 * Math.PI * y), Math.cos(2 * Math.PI * x)], recon, limiter: 'mc' };
    const { u } = run(o, u0, 0.5);
    assertClose(sum(u), sum(cellAverages2D(u0, 40, 30, [0, 1, 0, 1])), 1e-11, 0, `periodic ${recon}`);
    // inflow BC with rotation: one forward-Euler step changes the mass by exactly −dt · (net outflow)
    const fv = makeFV2D({ nx: 40, ny: 40, vel: rotationVelocity(), recon, limiter: 'mc', bc: 'inflow' });
    const v = cellAverages2D((x, y) => Math.exp(-10 * ((x - 0.8) ** 2 + (y - 0.5) ** 2)), 40, 40, fv.box), r = new Float64Array(1600);
    const m0 = sum(v) * fv.hx * fv.hy;
    fv.rhs(v, 0, r);
    const dt = fv.stableDt(0.5);
    for (let c = 0; c < 1600; c++) v[c] += dt * r[c];
    assert(fv.netOutflow() > 1e-6, 'something leaves the box');
    assertClose(sum(v) * fv.hx * fv.hy, m0 - dt * fv.netOutflow(), 1e-14, 0, `inflow ${recon}`);
  }
});

test('2D FV rates on a smoothly translated profile: upwind ≈ 1, MUSCL ≈ 2', () => {
  const u0 = (x, y) => Math.sin(2 * Math.PI * x) * Math.sin(2 * Math.PI * y);
  const ex = (x, y) => u0(x - 0.5, y - 0.25); // a = (1, 0.5), T = 0.5
  for (const [recon, limiter, expected, lo] of [['none', 'minmod', 1, 0.2], ['muscl', 'none', 2, 0.2], ['muscl', 'vanleer', 2, 0.4]]) {
    const hs = [], es = [];
    for (const n of [16, 32, 64, 128]) {
      const { u, fv } = run({ nx: n, ny: n, vel: () => [1, 0.5], recon, limiter }, u0, 0.5);
      const e0 = cellAverages2D(ex, n, n, fv.box);
      let e = 0; for (let c = 0; c < n * n; c++) e += Math.abs(u[c] - e0[c]) * fv.hx * fv.hy;
      hs.push(1 / n); es.push(e);
    }
    assertRate(fitRate(hs, es), expected, `${recon}/${limiter}`, lo, 0.3);
  }
});

test('2D FV rates for solid-body rotation of a smooth hump (inflow BC): upwind → 1 (from below), MUSCL ≈ 2', () => {
  // The first-order scheme is pre-asymptotic on these grids (its numerical diffusion ~ h/2 is large compared with the
  // hump width): observed pairwise rates 0.66, 0.79, 0.88 on 32→64→128→256. We check 32→64→128 (rates increasing, last > 0.75).
  const u0 = (x, y) => Math.exp(-((x - 0.5) ** 2 + (y - 0.75) ** 2) / 0.01);
  const T = 0.25, ex = rotated(u0, 2 * Math.PI, T); // quarter revolution
  for (const recon of ['none', 'muscl']) {
    const hs = [], es = [];
    for (const n of [32, 64, 128]) {
      const { u, fv } = run({ nx: n, ny: n, vel: rotationVelocity(), recon, limiter: 'none', bc: 'inflow' }, u0, T);
      const e0 = cellAverages2D(ex, n, n, fv.box);
      let e = 0; for (let c = 0; c < n * n; c++) e += Math.abs(u[c] - e0[c]) * fv.hx * fv.hy;
      hs.push(1 / n); es.push(e);
    }
    const r = pairwiseRates(hs, es);
    if (recon === 'none') assert(r[1] > r[0] && r[1] > 0.75 && r[1] < 1.1, `upwind rates ${r}`);
    else assertRate(fitRate(hs, es), 2, 'rotation MUSCL');
  }
});

test('Zalesak disk after one revolution (100²): MUSCL+MC much more accurate than upwind, both bounded in [0, 1]', () => {
  const errs = {};
  for (const [recon, limiter] of [['none', 'minmod'], ['muscl', 'mc']]) {
    const { u, fv } = run({ nx: 100, ny: 100, vel: rotationVelocity(), recon, limiter, bc: 'inflow' }, zalesakDisk, 1, 0.45, 6);
    const e0 = cellAverages2D(zalesakDisk, 100, 100, fv.box, 4, 6);
    let e = 0, lo = Infinity, hi = -Infinity;
    for (let c = 0; c < u.length; c++) { e += Math.abs(u[c] - e0[c]) * fv.hx * fv.hy; lo = Math.min(lo, u[c]); hi = Math.max(hi, u[c]); }
    assert(lo > -1e-12 && hi < 1 + 1e-12, `${recon}: range [${lo}, ${hi}]`);
    errs[recon] = e;
  }
  assert(errs.muscl < 0.6 * errs.none, `L1 errors ${errs.muscl} vs ${errs.none}`);
});

test('cellAverages2D with refinement: the mass of Zalesak\'s disk is exact to 2e-6 on every widget grid, also when rotated', () => {
  const exact = zalesakArea();
  // independent check of the closed form: disk minus the slot, the slot integrated by the midpoint rule
  let slot = 0; const K = 20000;
  for (let k = 0; k < K; k++) { const x = -0.025 + 0.05 * (k + 0.5) / K; slot += (0.1 + Math.sqrt(0.15 ** 2 - x * x)) * 0.05 / K; }
  assertClose(exact, Math.PI * 0.15 ** 2 - slot, 1e-11, 0, 'closed-form area');
  for (const n of [50, 100, 150]) {
    const mass = (f, q, r) => cellAverages2D(f, n, n, [0, 1, 0, 1], q, r).reduce((a, b) => a + b, 0) / (n * n);
    assertClose(mass(zalesakDisk, 4, 6), exact, 2e-6, 0, `n=${n} initial`);
    for (const t of [0.1, 0.37, 0.75]) assertClose(mass(rotated(zalesakDisk, 2 * Math.PI, t), 4, 4), exact, 1.5e-5, 0, `n=${n} rotated t=${t}`);
  }
  // the plain rule (what the widget used before) is off by up to 6e-4 on these grids
  const plain = [50, 100, 150].map((n) => Math.abs(cellAverages2D(zalesakDisk, n, n, [0, 1, 0, 1], 6).reduce((a, b) => a + b, 0) / (n * n) - exact));
  assert(Math.max(...plain) > 4e-5, `plain Gauss mass errors ${plain}`);
  // smooth data: refine = 0 is the default and unchanged; values in [0, 1] for the indicator
  const u = cellAverages2D(zalesakDisk, 100, 100, [0, 1, 0, 1], 4, 6);
  assert(Math.min(...u) >= 0 && Math.max(...u) <= 1 + 1e-14, 'averages of an indicator lie in [0, 1]');
});
