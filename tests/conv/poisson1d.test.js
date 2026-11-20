import { test, assert, assertClose, assertRate } from '../harness.js';
import { solvePoisson1D, errors1D, integrate, energyP1 } from '../../lib/core/elliptic/poisson1d.js';
import { csrSymmetryError, csrToDense } from '../../lib/core/la/sparse.js';
import { inertia } from '../../lib/core/la/inertia.js';
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

// "heat source near x = 0.3" of chapter 2: a narrow Gaussian that a single 8-point Gauss rule per cell
// cannot integrate on very coarse meshes (regression: nodal error 2.3e-6 at N = 2 before the composite rule)
const bump = {
  u: (x) => x * (1 - x) * Math.exp(-30 * (x - 0.3) ** 2),
  du: (x) => { const e = Math.exp(-30 * (x - 0.3) ** 2); return (1 - 2 * x) * e + x * (1 - x) * (-60 * (x - 0.3)) * e; },
  f: (x) => { const e = Math.exp(-30 * (x - 0.3) ** 2), p = x * (1 - x), dp = 1 - 2 * x, s = -60 * (x - 0.3); return -(-2 + 2 * dp * s - 60 * p + p * s * s) * e; },
};
const sine = { u: (x) => Math.sin(Math.PI * x), du: (x) => Math.PI * Math.cos(Math.PI * x), f: (x) => Math.PI ** 2 * Math.sin(Math.PI * x) };
const PROBLEMS = { sine, mixed: { u, du, f }, bump };

test('1D CG is nodally exact for every widget problem and every slider value N = 2..40', () => {
  for (const [name, pr] of Object.entries(PROBLEMS)) for (let N = 2; N <= 40; N++) {
    const s = solvePoisson1D('cg', N, pr.f, pr.u(0), pr.u(1));
    for (let i = 0; i <= N; i++) assert(Math.abs(s.U[i] - pr.u(s.x[i])) < 1e-12, `${name} N=${N} node ${i}: ${Math.abs(s.U[i] - pr.u(s.x[i]))}`);
  }
});

test('composite load quadrature: exact for degree 15, resolves the narrow Gaussian on one cell', () => {
  assertClose(integrate(0, 2, (x) => x ** 15), 2 ** 16 / 16, 0, 1e-13);
  // ∫_0^1 exp(−30(x−0.3)²) dx = √(π/30)/2 · (erf(0.7√30) + erf(0.3√30)) = 0.31628742…
  // (reference by an independent 4000-interval Simpson rule)
  let simpson = 0; const n = 4000, g = (x) => Math.exp(-30 * (x - 0.3) ** 2);
  for (let i = 0; i <= n; i++) simpson += (i === 0 || i === n ? 1 : i % 2 ? 4 : 2) * g(i / n);
  assertClose(integrate(0, 1, g), simpson / (3 * n), 1e-12);
});

test('all three 1D forms stay finite and converge for every widget problem (N = 2..40)', () => {
  for (const [name, pr] of Object.entries(PROBLEMS)) {
    for (let N = 2; N <= 40; N++) for (const form of ['fd', 'cg', 'mixed']) {
      const e = errors1D(form, solvePoisson1D(form, N, pr.f, pr.u(0), pr.u(1)), pr.u, pr.du);
      assert(Number.isFinite(e.uL2) && Number.isFinite(e.sigmaL2) && e.uL2 < 1 && e.sigmaL2 < 3, `${name} ${form} N=${N}`);
    }
    // the rate table of the widget (N = 8..64): FD/CG 2 and 1, mixed 1 and 2
    for (const [form, ru, rs] of [['fd', 2, 1], ['cg', 2, 1], ['mixed', 1, 2]]) {
      const e = Ns.map((N) => errors1D(form, solvePoisson1D(form, N, pr.f, pr.u(0), pr.u(1)), pr.u, pr.du));
      assertRate(fitRate(hs, e.map((x) => x.uL2)), ru, `${name} ${form} u`, 0.15, 0.15);
      assertRate(fitRate(hs, e.map((x) => x.sigmaL2)), rs, `${name} ${form} sigma`, 0.15, 0.15);
    }
  }
});

test('1D mixed method: u_h is 2nd-order accurate as cell averages (superconvergence)', () => {
  for (const [name, pr] of Object.entries(PROBLEMS)) {
    const e = Ns.map((N) => errors1D('mixed', solvePoisson1D('mixed', N, pr.f, pr.u(0), pr.u(1)), pr.u, pr.du).uAvgMax);
    assertRate(fitRate(hs, e), 2, `${name} cell averages`, 0.15, 0.15);
  }
});

test('1D mixed matrix is a symmetric saddle point with N+1 positive and N negative eigenvalues', () => {
  for (const N of [2, 5, 12]) {
    const s = solvePoisson1D('mixed', N, f, u(0), u(1));
    assert(csrSymmetryError(s.A) === 0, 'mixed matrix not symmetric');
    const I = inertia(csrToDense(s.A), s.A.n);
    assert(I.pos === N + 1 && I.neg === N && I.zero === 0, `N=${N}: inertia ${I.pos}/${I.neg}/${I.zero}`);
  }
});

test('Dirichlet principle: the CG solution minimises the discrete energy J', () => {
  let seed = 9;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
  for (const [name, pr] of Object.entries(PROBLEMS)) for (const N of [2, 8, 24]) {
    const s = solvePoisson1D('cg', N, pr.f, pr.u(0), pr.u(1)), Jmin = energyP1(s.U, pr.f);
    for (let rep = 0; rep < 20; rep++) {
      // perturb interior nodes only (boundary values are fixed); J must grow by exactly ½ a(w,w)
      const v = Float64Array.from(s.U), amp = 10 ** (-rep / 3);
      let aww = 0;
      const w = new Float64Array(N + 1);
      for (let i = 1; i < N; i++) { w[i] = amp * rnd(); v[i] += w[i]; }
      for (let i = 0; i < N; i++) aww += (w[i + 1] - w[i]) ** 2 * N;
      const dJ = energyP1(v, pr.f) - Jmin;
      assert(dJ >= -1e-13, `${name} N=${N}: J decreased by ${dJ}`);
      assertClose(dJ, 0.5 * aww, 1e-12, 1e-9, `${name} N=${N} quadratic growth`);
    }
  }
  // continuous value for −u'' = π² sin πx: J(u) = −½∫u'² = −π²/4; the discrete minimum lies above and converges to it
  const J8 = energyP1(solvePoisson1D('cg', 8, sine.f, 0, 0).U, sine.f), J64 = energyP1(solvePoisson1D('cg', 64, sine.f, 0, 0).U, sine.f);
  assert(J8 > J64 && J64 > -(Math.PI ** 2) / 4 && J64 + Math.PI ** 2 / 4 < 1e-3, `J8=${J8} J64=${J64}`);
});

test('chapter 2 "try this": narrow source on 4 cells — FD (samples f) is far off, CG (integrates f) is nodally exact', () => {
  const fd4 = solvePoisson1D('fd', 4, bump.f, 0, 0), cg4 = solvePoisson1D('cg', 4, bump.f, 0, 0);
  const nodalErr = (s) => Math.max(...Array.from(s.U, (v, i) => Math.abs(v - bump.u(s.x[i]))));
  assert(nodalErr(fd4) > 0.15 && fd4.U[2] < 0 && fd4.U[3] < 0, `FD N=4: error ${nodalErr(fd4)}, U = ${Array.from(fd4.U)}`); // exact max of u is 0.22
  assert(nodalErr(cg4) < 1e-13);
  assert(nodalErr(solvePoisson1D('fd', 16, bump.f, 0, 0)) < 0.006, 'FD has caught up at N = 16');
  // flux comparison quoted in the text: at N = 16 the mixed flux is about 4× better than the CG flux, at N = 6 they are equally poor
  const sig = (form, N) => errors1D(form, solvePoisson1D(form, N, bump.f, 0, 0), bump.u, bump.du).sigmaL2;
  const r16 = sig('cg', 16) / sig('mixed', 16), r6 = sig('cg', 6) / sig('mixed', 6);
  assert(r16 > 3.5 && r16 < 4.5, `N=16 ratio ${r16}`);
  assert(r6 > 0.8 && r6 < 1.3, `N=6 ratio ${r6}`);
});
