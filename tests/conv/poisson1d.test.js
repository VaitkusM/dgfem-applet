import { test, assert, assertRate } from '../harness.js';
import { solvePoisson1D, errors1D } from '../../lib/core/elliptic/poisson1d.js';
import { fitRate } from '../../lib/core/verify/rates.js';

const u = (x) => Math.sin(3 * x) + x * x, du = (x) => 3 * Math.cos(3 * x) + 2 * x, f = (x) => 9 * Math.sin(3 * x) - 2;
const Ns = [8, 16, 32, 64], hs = Ns.map((N) => 1 / N);

test('1D Poisson: FD and CG are 2nd order in u, 1st order in the flux; CG is nodally exact', () => {
  for (const form of ['fd', 'cg']) {
    const e = Ns.map((N) => errors1D(form, solvePoisson1D(form, N, f, u(0), u(1)), u, du));
    assertRate(fitRate(hs, e.map((x) => x.uL2)), 2, `${form} u`);
    assertRate(fitRate(hs, e.map((x) => x.sigmaL2)), 1, `${form} sigma`);
  }
  const s = solvePoisson1D('cg', 10, f, u(0), u(1));
  for (let i = 0; i <= 10; i++) assert(Math.abs(s.U[i] - u(s.x[i])) < 1e-13, 'nodal exactness');
});

test('1D mixed method: u_h (P0) 1st order, flux sigma_h 2nd order, exact cell balance', () => {
  const e = Ns.map((N) => errors1D('mixed', solvePoisson1D('mixed', N, f, u(0), u(1)), u, du));
  assertRate(fitRate(hs, e.map((x) => x.uL2)), 1, 'mixed u');
  assertRate(fitRate(hs, e.map((x) => x.sigmaL2)), 2, 'mixed sigma');
  // σ(x_{i+1}) − σ(x_i) = ∫_cell f  (flux balance) — check against exact integral of f = [−u']
  const N = 12, s = solvePoisson1D('mixed', N, f, u(0), u(1));
  for (let i = 0; i < N; i++) {
    const exact = -du(s.x[i + 1]) + du(s.x[i]);
    assert(Math.abs(s.sigma[i + 1] - s.sigma[i] - exact) < 1e-12, 'cell balance');
  }
});
