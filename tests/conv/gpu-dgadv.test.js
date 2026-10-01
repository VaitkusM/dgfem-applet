/**
 * Convergence of the CPU twin of the chapter-12 GPU solver:
 * L2 error vs the exact translated solution decays like h^{p+1}.
 */
import { test, assert, assertRate } from '../harness.js';
import { makeDGAdv2D, makeCpuSolver, INITIAL, translated } from '../../lib/gpu/cpuTwin.js';
import { fitRate } from '../../lib/core/verify/rates.js';

test('DG-SEM advection (CPU twin): L2 error rate p+1, p = 1..3', () => {
  const a = [1, 0.5], T = 0.25;
  for (let p = 1; p <= 3; p++) {
    const hs = [], es = [];
    for (const N of (p === 1 ? [8, 16, 32, 64] : [4, 8, 16, 32])) { // p = 1 is still pre-asymptotic on coarse grids
      const d = makeDGAdv2D({ N, p, a });
      const nSteps = Math.ceil(T / d.stableDt(0.5)), dt = T / nSteps;
      const S = makeCpuSolver(d, d.interpolate(INITIAL.sines));
      S.run(nSteps, dt);
      hs.push(d.h); es.push(d.l2Error(S.u(), translated(INITIAL.sines, a[0], a[1], T)));
    }
    assertRate(fitRate(hs, es), p + 1, `p=${p} errors ${es.map((x) => x.toExponential(2))}`);
    if (p === 3) assert(es[3] < 1e-6, 'chapter 12 quotes: p = 3, N = 32, T = 0.25 error below 1e-6');
  }
});
