import { test, assert } from '../harness.js';
import { makeAdvBC1D } from '../../lib/core/fv/advbc1d.js';
import { makeStepper } from '../../lib/core/time/rk.js';
import { fitRate } from '../../lib/core/verify/rates.js';

const u0 = (x) => Math.sin(2 * Math.PI * x);
const gIn = (t) => Math.sin(-2 * Math.PI * t); // consistent with u0: exact u = sin(2π(x − t))
const exact = (x, t) => Math.sin(2 * Math.PI * (x - t));

function run(opts, N, T = 0.5, cfl = 0.4) {
  const d = makeAdvBC1D({ N, a: 1, gIn, gOut: () => 0, ...opts });
  const u = Float64Array.from(d.xc, (x) => u0(x));
  let t = 0;
  const step = makeStepper('ssprk3', N, (v) => d.enforce(v, t));
  const dt0 = cfl * d.h;
  while (t < T - 1e-14) { const dt = Math.min(dt0, T - t); step(u, t, dt, d.rhs); t += dt; d.enforce(u, t); }
  let e = 0, eLast = 0;
  for (let i = 0; i < N; i++) e += Math.abs(u[i] - exact(d.xc[i], T)) * d.h;
  eLast = Math.abs(u[N - 1] - exact(d.xc[N - 1], T));
  return { u, e, eLast };
}

test('weak inflow + upwind converges at first order', () => {
  const Ns = [50, 100, 200, 400];
  const es = Ns.map((N) => run({ flux: 'upwind', inflow: 'weak', outflow: 'extrapolate' }, N).e);
  const r = fitRate(Ns.map((N) => 1 / N), es);
  assert(r > 0.85 && r < 1.2, `rate ${r}`);
});

test('upwind flux ignores data offered at the outflow boundary', () => {
  const a = run({ flux: 'upwind', inflow: 'weak', outflow: 'extrapolate' }, 80).u;
  const b = run({ flux: 'upwind', inflow: 'weak', outflow: 'weak' }, 80).u;
  for (let i = 0; i < 80; i++) assert(a[i] === b[i], 'identical');
});

test('strong outflow over-specification produces an O(1) boundary error', () => {
  const ok = run({ flux: 'upwind', inflow: 'weak', outflow: 'extrapolate' }, 200);
  const bad = run({ flux: 'upwind', inflow: 'weak', outflow: 'strong', gOut: () => 1 }, 200);
  assert(ok.eLast < 0.05, `ok ${ok.eLast}`);
  assert(bad.eLast > 0.3, `bad ${bad.eLast}`);
});

test('central flux: wrong outflow data pollutes the interior', () => {
  const ok = run({ flux: 'central', inflow: 'weak', outflow: 'extrapolate' }, 200, 0.5, 0.2);
  const bad = run({ flux: 'central', inflow: 'weak', outflow: 'weak', gOut: () => 1 }, 200, 0.5, 0.2);
  assert(bad.e > 3 * ok.e, `central: ${ok.e} vs ${bad.e}`);
});

test('strong and weak inflow both converge', () => {
  for (const inflow of ['strong', 'weak']) {
    const r = run({ flux: 'upwind', inflow, outflow: 'extrapolate' }, 400);
    assert(r.e < 0.02, `${inflow}: ${r.e}`);
  }
});
