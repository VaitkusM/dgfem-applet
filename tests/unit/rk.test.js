import { test, assert, assertClose } from '../harness.js';
import { makeStepper, RK_ORDER, RK_METHODS, RK_LABEL, stabilityAmp } from '../../lib/core/time/rk.js';
import { fitRate } from '../../lib/core/verify/rates.js';
import { assertRate } from '../harness.js';

test('RK observed orders on y\' = -y + sin t (nonautonomous)', () => {
  // exact solution of y' = -y + sin t, y(0)=1: y = 1.5 e^{-t} + (sin t - cos t)/2
  const exact = (t) => 1.5 * Math.exp(-t) + (Math.sin(t) - Math.cos(t)) / 2;
  const L = (u, t, out) => { out[0] = -u[0] + Math.sin(t); };
  for (const m of Object.keys(RK_ORDER)) {
    const hs = [], es = [];
    for (const N of [10, 20, 40, 80]) {
      const dt = 1 / N, step = makeStepper(m, 1), u = Float64Array.of(1);
      for (let k = 0; k < N; k++) step(u, k * dt, dt, L);
      hs.push(dt); es.push(Math.abs(u[0] - exact(1)));
    }
    assertRate(fitRate(hs, es), RK_ORDER[m], m);
  }
});

test('stability functions: |R(z)| matches Taylor polynomials', () => {
  const z = [-1.3, 0.7];
  const poly = (k) => { // Σ_{j≤k} z^j/j!
    let re = 1, im = 0, tr = 1, ti = 0;
    for (let j = 1; j <= k; j++) { const a = (tr * z[0] - ti * z[1]) / j, b = (tr * z[1] + ti * z[0]) / j; tr = a; ti = b; re += tr; im += ti; }
    return Math.hypot(re, im);
  };
  assertClose(stabilityAmp('fe', ...z), poly(1), 1e-14);
  assertClose(stabilityAmp('ssprk2', ...z), poly(2), 1e-14);
  assertClose(stabilityAmp('ssprk3', ...z), poly(3), 1e-14);
  assertClose(stabilityAmp('rk4', ...z), poly(4), 1e-14);
  // SSPRK3 is stable on the imaginary axis up to |z| = sqrt(3)
  assertClose(stabilityAmp('ssprk3', 0, Math.sqrt(3)), 1, 1e-12);
});

test('low-storage RK4: stability polynomial agrees with exp(z) to 4th order; stable on the imaginary axis beyond RK4', () => {
  // complex R(z) of lsrk4 by running the stepper on y' = z y
  const R = (zr, zi) => {
    const y = Float64Array.of(1, 0);
    makeStepper('lsrk4', 2)(y, 0, 1, (u, t, out) => { out[0] = zr * u[0] - zi * u[1]; out[1] = zi * u[0] + zr * u[1]; });
    return y;
  };
  // |R(z) − e^z| = C |z|⁵ + …  ⇒ halving z divides the defect by 2⁵ = 32
  for (const [dr, di] of [[-1, 0], [0, 1], [-0.6, 0.8]]) {
    const defect = (s) => { const y = R(s * dr, s * di), e = Math.exp(s * dr); return Math.hypot(y[0] - e * Math.cos(s * di), y[1] - e * Math.sin(s * di)); };
    const ratio = defect(0.2) / defect(0.1);
    assert(ratio > 26 && ratio < 38, `defect ratio ${ratio} (expected ≈ 32)`);
    assert(defect(0.1) < 2e-7, `defect ${defect(0.1)}`);
  }
  // imaginary-axis stability limit: classical RK4 2√2 ≈ 2.83; Carpenter–Kennedy ≈ 3.34 (their paper: 3.34)
  assert(stabilityAmp('rk4', 0, 2.82) <= 1 && stabilityAmp('rk4', 0, 2.84) > 1);
  assert(stabilityAmp('lsrk4', 0, 3.3) <= 1 && stabilityAmp('lsrk4', 0, 3.4) > 1, `${stabilityAmp('lsrk4', 0, 3.3)} ${stabilityAmp('lsrk4', 0, 3.4)}`);
});

test('RK bookkeeping: every method has an order and a label; steppers do not share state; post hook runs per stage', () => {
  assert(RK_METHODS.length === 5);
  for (const m of RK_METHODS) assert(RK_ORDER[m] >= 1 && typeof RK_LABEL[m] === 'string', m);
  let threw = false;
  try { makeStepper('nope', 1); } catch { threw = true; }
  assert(threw, 'unknown method must throw');
  const L = (u, t, out) => { out[0] = -u[0]; out[1] = 2 * u[1]; };
  for (const m of RK_METHODS) {
    // two independent steppers interleaved give the same result as each one alone
    const s1 = makeStepper(m, 2), s2 = makeStepper(m, 2), a = Float64Array.of(1, 1), b = Float64Array.of(3, -2), ref = Float64Array.of(1, 1);
    for (let k = 0; k < 5; k++) { s1(a, 0, 0.1, L); s2(b, 0, 0.1, L); }
    const s3 = makeStepper(m, 2);
    for (let k = 0; k < 5; k++) s3(ref, 0, 0.1, L);
    assertClose(a[0], ref[0], 0); assertClose(a[1], ref[1], 0);
    let calls = 0;
    makeStepper(m, 2, () => { calls++; })(Float64Array.of(1, 1), 0, 0.1, L);
    assert(calls === ({ fe: 1, ssprk2: 2, ssprk3: 3, rk4: 1, lsrk4: 1 })[m], `${m}: post hook called ${calls} times`);
  }
});
