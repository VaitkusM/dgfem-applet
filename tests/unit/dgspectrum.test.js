/**
 * Tests for lib/core/dg/spectrum.js: spectra of DG for periodic linear advection
 * and maximal stable CFL numbers ν = |a| Δt / h (h = element width).
 */
import { test, assert, assertClose } from '../harness.js';
import { makeDG1D } from '../../lib/core/dg/dg1d.js';
import { advection } from '../../lib/core/models/scalar.js';
import { blochBlocks, blochEigs, maxStableCFL, operatorEigs } from '../../lib/core/dg/spectrum.js';

const make = (p, flux = 'upwind', extra = {}) => (N) => makeDG1D({ p, N, a: 0, b: N, model: advection(1), flux, ...extra });
const bloch = (p, flux = 'upwind', extra = {}, nT = 256) => blochEigs(blochBlocks(make(p, flux, extra), p + 1), nT);

test('Bloch eigenvalues at θ = 2πm/N coincide with the full periodic operator spectrum', () => {
  for (const p of [1, 3]) {
    const N = 8, full = operatorEigs(make(p)(N).rhs, N * (p + 1)), bl = bloch(p, 'upwind', {}, N);
    // every full eigenvalue has a Bloch eigenvalue within 1e-9 (and vice versa)
    const near = (A, B) => { for (let i = 0; i < A.re.length; i++) {
      let d = Infinity; for (let j = 0; j < B.re.length; j++) d = Math.min(d, Math.hypot(A.re[i] - B.re[j], A.im[i] - B.im[j]));
      assert(d < 1e-9, `eigenvalue ${A.re[i]}+${A.im[i]}i unmatched (${d})`);
    } };
    near(full, bl); near(bl, full);
  }
});

test('Upwind DG spectrum lies in the closed left half plane; central flux on the imaginary axis', () => {
  for (let p = 0; p <= 5; p++) {
    const up = bloch(p, 'upwind', {}, 64), ce = bloch(p, 'central', {}, 64);
    for (let k = 0; k < up.re.length; k++) assert(up.re[k] < 1e-10, `upwind p=${p} Re=${up.re[k]}`);
    for (let k = 0; k < ce.re.length; k++) assert(Math.abs(ce.re[k]) < 1e-9, `central p=${p} Re=${ce.re[k]}`);
  }
});

test('p = 0 recovers first-order upwind FV: forward Euler stable up to ν = 1, SSPRK3 up to 1.2563', () => {
  const ev = bloch(0);
  assertClose(maxStableCFL(ev, 'fe'), 1, 2e-3);
  assertClose(maxStableCFL(ev, 'ssprk3'), 1.2563, 2e-3);
});

test('Max stable CFL of upwind DG: Cockburn–Shu values (p=1 RK2: 1/3, p=2 RK3: 0.209, p=3 RK4: 0.145)', () => {
  assertClose(maxStableCFL(bloch(1), 'ssprk2'), 1 / 3, 2e-3, 0, 'p=1 ssprk2');
  assertClose(maxStableCFL(bloch(2), 'ssprk3'), 0.209, 2e-3, 0, 'p=2 ssprk3');
  assertClose(maxStableCFL(bloch(3), 'rk4'), 0.145, 2e-3, 0, 'p=3 rk4');
  // SSPRK3 for p = 1, 3
  assertClose(maxStableCFL(bloch(1), 'ssprk3'), 0.409, 2e-3, 0, 'p=1 ssprk3');
  assertClose(maxStableCFL(bloch(3), 'ssprk3'), 0.130, 2e-3, 0, 'p=3 ssprk3');
});

test('CFL limit scales roughly like 1/(2p+1): 0.7 ≤ ν_max (2p+1) ≤ 1.3 for SSPRK3, p = 1..5', () => {
  for (let p = 1; p <= 5; p++) {
    const v = maxStableCFL(bloch(p, 'upwind', {}, 128), 'ssprk3') * (2 * p + 1);
    assert(v > 0.7 && v < 1.3, `p=${p}: ${v}`);
  }
});

test('GLL collocation (lumped mass) allows a larger time step than exact DG', () => {
  for (let p = 1; p <= 4; p++) {
    const ex = maxStableCFL(bloch(p, 'upwind', { nodes: 'gll', quad: 'exact' }, 128), 'ssprk3');
    const lu = maxStableCFL(bloch(p, 'upwind', { nodes: 'gll', quad: 'collocated' }, 128), 'ssprk3');
    assert(lu > 1.5 * ex, `p=${p}: lumped ${lu} vs exact ${ex}`);
  }
});

test('Forward Euler is (essentially) unstable for upwind DG with p ≥ 1: ν_max < 0.01', () => {
  for (const p of [1, 2]) {
    const v64 = maxStableCFL(bloch(p, 'upwind', {}, 64), 'fe'), v512 = maxStableCFL(bloch(p, 'upwind', {}, 512), 'fe');
    assert(v64 < 0.01 && v512 <= v64, `p=${p}: ${v64} ${v512}`);
  }
});

test('Imaginary-axis stability limits: SSPRK3 up to √3, RK4 up to 2√2, SSPRK2 and forward Euler none', async () => {
  const { stabilityAmp } = await import('../../lib/core/time/rk.js');
  const ok = (m, y) => stabilityAmp(m, 0, y) <= 1 + 1e-12;
  assert(ok('ssprk3', Math.sqrt(3) - 1e-6) && !ok('ssprk3', Math.sqrt(3) + 1e-3), 'ssprk3');
  assert(ok('rk4', 2 * Math.SQRT2 - 1e-6) && !ok('rk4', 2 * Math.SQRT2 + 1e-3), 'rk4');
  assert(!ok('ssprk2', 0.01) && !ok('fe', 0.01), 'ssprk2/fe');
});
