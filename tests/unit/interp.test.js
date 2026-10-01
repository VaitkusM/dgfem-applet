import { test, assertRate } from '../harness.js';
import { interpError } from '../../lib/core/verify/interp1d.js';
import { fitRate } from '../../lib/core/verify/rates.js';

test('piecewise interpolation: L2 rate p+1 (smooth), 1.5 (kink), 0.5 (jump)', () => {
  const Ns = [16, 32, 64, 128], hs = Ns.map((N) => 1 / N);
  const smooth = (x) => Math.sin(2 * Math.PI * x) + x * x;
  const kink = (x) => Math.abs(x - 1 / 3);
  const jump = (x) => (x < 1 / 3 ? 0 : 1);
  for (let p = 1; p <= 4; p++) {
    const Ns2 = p >= 4 ? [4, 8, 16, 32] : Ns;
    assertRate(fitRate(Ns2.map((N) => 1 / N), Ns2.map((N) => interpError(smooth, N, p).L2)), p + 1, `smooth p=${p}`);
    assertRate(fitRate(hs, Ns.map((N) => interpError(kink, N, p).L2)), 1.5, `kink p=${p}`, 0.25, 0.25);
    assertRate(fitRate(hs, Ns.map((N) => interpError(jump, N, p).L2)), 0.5, `jump p=${p}`, 0.25, 0.25);
  }
});
