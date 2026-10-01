import { test, assertClose } from '../harness.js';
import { makeStepper, RK_ORDER, stabilityAmp } from '../../lib/core/time/rk.js';
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
