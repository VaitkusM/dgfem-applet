/**
 * Chapter 10 (domain decomposition): convergence of the DD solvers, scalability
 * with/without coarse space, effect of overlap, condition numbers of A and S,
 * Dirichlet–Neumann relaxation.
 */
import { test, assert, assertRate } from '../harness.js';
import { ddPoissonProblem, pointLoad } from '../../lib/core/dd/problem.js';
import { boxPartition, oneBoxPerIteration } from '../../lib/core/dd/partition.js';
import { schwarzSetup, solveSchwarz } from '../../lib/core/dd/schwarz.js';
import { schurSetup, denseSchur, denseCondition, dirichletNeumannSetup, dnSpectrum } from '../../lib/core/dd/schur.js';
import { sparseSolve } from '../../lib/core/la/direct.js';
import { conditionEstimate } from '../../lib/core/la/eig.js';
import { fitRate } from '../../lib/core/verify/rates.js';

const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const f = (x, y) => 1 + 4 * Math.exp(-40 * ((x - 0.2) ** 2 + (y - 0.3) ** 2));

test('all Schwarz methods converge to the monolithic solution', () => {
  const N = 24, prob = ddPoissonProblem(N, f), u = sparseSolve(prob.A, prob.b);
  const umax = Math.max(...u);
  for (const [Kx, Ky] of [[2, 1], [3, 3], [4, 2]])
    for (const coarse of ['none', 'q1']) {
      const dd = schwarzSetup(prob, boxPartition(N, Kx, Ky, 1), { coarse });
      for (const method of ['multiplicative', 'asm-cg', 'ras-gmres']) {
        const r = solveSchwarz(dd, prob.b, { method, tol: 1e-10, maxIter: 1000 });
        assert(r.converged, `${method} ${coarse} ${Kx}x${Ky} converged`);
        assert(maxDiff(r.x, u) < 1e-8 * umax, `${method} ${coarse} ${Kx}x${Ky} = monolithic`);
      }
    }
});

test('weak scaling (H/h = 8, δ = 2h): one-level ASM-PCG iterations grow with K, two-level stay bounded', () => {
  const Ks = [2, 4, 6, 8], one = [], two = [], ras1 = [], ras2 = [];
  for (const K of Ks) {
    const N = 8 * K, prob = ddPoissonProblem(N), part = boxPartition(N, K, K, 1);
    for (const [coarse, cgList, rasList] of [['none', one, ras1], ['q1', two, ras2]]) {
      const dd = schwarzSetup(prob, part, { coarse });
      cgList.push(solveSchwarz(dd, prob.b, { method: 'asm-cg', tol: 1e-8 }).iters);
      rasList.push(solveSchwarz(dd, prob.b, { method: 'ras-gmres', tol: 1e-8, maxIter: 300 }).iters);
    }
  }
  for (let k = 1; k < Ks.length; k++) assert(one[k] > one[k - 1], `one-level grows ${one}`);
  assert(one[Ks.length - 1] >= 2.5 * one[0], `one-level grows substantially ${one}`);
  assert(Math.max(...two) <= 1.5 * two[1] && two[3] - two[2] <= 2, `two-level bounded ${two}`);
  assert(two[3] < 0.75 * one[3], `coarse space pays off at K = 8: ${two} vs ${one}`);
  for (let k = 1; k < Ks.length; k++) assert(ras1[k] > ras1[k - 1], `RAS one-level grows ${ras1}`);
  assert(Math.max(...ras2) <= 1.5 * ras2[1], `RAS two-level bounded ${ras2}`);
});

test('more overlap → fewer iterations (ASM-PCG, RAS-GMRES and multiplicative)', () => {
  const N = 32, prob = ddPoissonProblem(N);
  for (const method of ['asm-cg', 'ras-gmres', 'multiplicative']) {
    const its = [1, 2, 3, 4].map((ell) => solveSchwarz(schwarzSetup(prob, boxPartition(N, 4, 4, ell)), prob.b, { method, tol: 1e-8, maxIter: 1000 }).iters);
    for (let k = 1; k < its.length; k++) assert(its[k] < its[k - 1], `${method}: ${its}`);
  }
});

test('κ(A) ~ h⁻² (= cot²(πh/2) exactly) while κ(S) ~ h⁻¹', () => {
  const Ns = [8, 16, 32, 64], hs = Ns.map((N) => 1 / N), kA = [], kS = [], lmin = [], lmax = [];
  for (const N of Ns) {
    const prob = ddPoissonProblem(N);
    const c = conditionEstimate(prob.A, { k: 80 });
    const exact = 1 / Math.tan(Math.PI / (2 * N)) ** 2;
    assert(Math.abs(c.kappa - exact) < 2e-3 * exact, `Lanczos κ(A) ${c.kappa} vs exact ${exact}`);
    kA.push(exact);
    const su = schurSetup(prob, 2, 2);
    const c2 = denseCondition(denseSchur(su), su.gamma.length);
    kS.push(c2.kappa); lmin.push(c2.lmin); lmax.push(c2.lmax);
  }
  // λ_max(S) bounded, λ_min(S) ∝ h
  assert(Math.max(...lmax) < 1.1 * Math.min(...lmax), `λ_max(S) bounded ${lmax}`);
  assertRate(fitRate(hs, lmin), 1, 'λ_min(S) ∝ h', 0.1, 0.1);
  assertRate(-fitRate(hs, kA), 2, 'κ(A) exponent', 0.1, 0.1);
  assertRate(-fitRate(hs, kS), 1, 'κ(S) exponent', 0.1, 0.1);
});

test('Dirichlet–Neumann: θ = ½ converges, observed rate = predicted ρ(θ); θ = 1 diverges when the Neumann side is larger', () => {
  const N = 32, prob = ddPoissonProblem(N, f), u = sparseSolve(prob.A, prob.b);
  const run = (dn, theta, iters) => {
    let lam = new Float64Array(dn.G.length);
    const err = [];
    for (let it = 0; it < iters; it++) {
      const r = dn.step(lam, theta); lam = r.lambda;
      err.push(maxDiff(Float64Array.from(dn.G, (g) => u[g]), lam));
      if (it === iters - 1) return { err, x: r.x };
    }
  };
  // interface at x = 1/4: Ω₁ (Dirichlet) small, Ω₂ (Neumann) large ⇒ μ = eig(S₂⁻¹S₁) ≥ 1
  const dn = dirichletNeumannSetup(prob, 8), sp = dnSpectrum(dn.S1, dn.S2, dn.G.length);
  assert(sp.muMin > 0.99 && sp.muMin < 1.01, `high frequencies: μ ≈ 1 (${sp.muMin})`);
  const coth = (x) => 1 / Math.tanh(x);
  assert(Math.abs(sp.muMax - coth(Math.PI / 4) / coth(3 * Math.PI / 4)) < 0.01, `lowest mode μ ≈ coth(π/4)/coth(3π/4): ${sp.muMax}`);
  assert(Math.abs(sp.rho(0.5) - 0.25) < 0.01, `ρ(½) ≈ 0.25: ${sp.rho(0.5)}`);
  const a = run(dn, 0.5, 12);
  assert(a.err[11] < 1e-6 * a.err[0], 'θ = ½ converges');
  const obs = a.err[6] / a.err[5];
  assert(Math.abs(obs - sp.rho(0.5)) < 0.02, `observed ${obs} vs predicted ${sp.rho(0.5)}`);
  assert(maxDiff(run(dn, 0.5, 30).x, u) < 1e-10, 'DN limit = monolithic solution');
  const b = run(dn, 1, 12);
  assert(sp.rho(1) > 1.3 && b.err[11] > 10 * b.err[0], 'θ = 1 diverges');
  const c = run(dn, sp.thetaOpt, 12);
  assert(sp.rho(sp.thetaOpt) < sp.rho(0.5) && c.err[11] < a.err[11], 'θ_opt is faster');
  // symmetric split: S₁ = S₂, θ = ½ gives the exact interface values after one step
  const dnS = dirichletNeumannSetup(prob, 16);
  assert(run(dnS, 0.5, 1).err[0] < 1e-12, 'symmetric split converges in one step');
});

test('one-level additive Schwarz moves information at most one subdomain per iteration; the coarse space reaches everywhere at once', () => {
  const N = 40, K = 5, prob = ddPoissonProblem(N), part = boxPartition(N, K, K, 1);
  const b = new Float64Array(prob.n);
  b[(4 - 1) * (N - 1) + (4 - 1)] = 1; // point load at vertex (4,4), deep inside the core of subdomain (0,0)
  // max |x| over the core (owned vertices) of each subdomain
  const coreMax = (x, s) => { let m = 0; s.dofs.forEach((d, l) => { if (s.core[l]) m = Math.max(m, Math.abs(x[d])); }); return m; };
  for (const method of ['asm-cg', 'ras-gmres']) {
    const one = solveSchwarz(schwarzSetup(prob, part), b, { method, tol: 1e-14, maxIter: 12, keepIterates: true });
    for (let k = 1; k < one.iterates.length; k++)
      for (const s of part.subs) {
        const dist = Math.max(s.kx, s.ky); // Chebyshev distance (in subdomains) from subdomain (0,0)
        if (dist > k) assert(coreMax(one.iterates[k], s) === 0, `${method}: subdomain (${s.kx},${s.ky}) untouched at iteration ${k}`);
      }
    assert(coreMax(one.iterates[one.iterates.length - 1], part.subs[K * K - 1]) > 0, `${method}: far corner eventually reached`);
    const two = solveSchwarz(schwarzSetup(prob, part, { coarse: 'q1' }), b, { method, tol: 1e-14, maxIter: 2, keepIterates: true });
    assert(part.subs.every((s) => coreMax(two.iterates[1], s) > 0), `${method}: two-level reaches every subdomain at iteration 1`);
  }
});

test('information front at the widget settings (N = 32, point source at vertex (1,1)): exactly zero beyond n boxes iff δ ≤ H', () => {
  // All 6·6·4 partitions the Schwarz widget can show. oneBoxPerIteration(part) (every box with two neighbours is at
  // least 2ℓ cells wide) predicts EXACTLY when the one-level ASM+CG iterate obeys "subdomains more than n boxes
  // away are still zero after n iterations"; RAS+GMRES obeys it whenever the criterion holds.
  const N = 32, prob = ddPoissonProblem(N), b = pointLoad(prob, 1, 1);
  assert(b.reduce((s, v) => s + (v !== 0), 0) === 1 && b[0] === 1, 'point load = one non-zero entry');
  const coreMax = (x, s) => { let m = 0; s.dofs.forEach((d, l) => { if (s.core[l]) m = Math.max(m, Math.abs(x[d])); }); return m; };
  const violations = (part, method) => {
    const r = solveSchwarz(schwarzSetup(prob, part), b, { method, tol: 1e-14, maxIter: 7, keepIterates: true });
    let v = 0;
    for (let k = 1; k < r.iterates.length; k++) for (const s of part.subs) if (Math.max(s.kx, s.ky) > k && coreMax(r.iterates[k], s) !== 0) v++;
    return v;
  };
  let nOk = 0, nWide = 0;
  for (let Kx = 1; Kx <= 6; Kx++) for (let Ky = 1; Ky <= 6; Ky++) for (let ell = 1; ell <= 4; ell++) {
    const part = boxPartition(N, Kx, Ky, ell), ok = oneBoxPerIteration(part);
    const vA = violations(part, 'asm-cg');
    assert((vA === 0) === ok, `ASM+CG ${Kx}x${Ky} ℓ=${ell}: ${vA} violations, criterion ${ok}`);
    if (ok) { nOk++; assert(violations(part, 'ras-gmres') === 0, `RAS+GMRES ${Kx}x${Ky} ℓ=${ell}`); } else nWide++;
  }
  assert(nOk === 113 && nWide === 31, `${nOk} partitions with δ ≤ H, ${nWide} with wider overlap`);
  // the minimal overlap ℓ = 1 always qualifies; 6×6 boxes (5–6 cells) with ℓ = 3 (δ = 6h) do not
  assert(oneBoxPerIteration(boxPartition(N, 6, 6, 1)) && oneBoxPerIteration(boxPartition(N, 6, 6, 2)), 'δ ≤ H');
  assert(!oneBoxPerIteration(boxPartition(N, 6, 6, 3)) && !oneBoxPerIteration(boxPartition(N, 6, 6, 4)), 'δ > H');
  // with the coarse space the first iterate is non-zero in every subdomain
  for (const method of ['asm-cg', 'ras-gmres']) {
    const part = boxPartition(N, 6, 6, 1);
    const two = solveSchwarz(schwarzSetup(prob, part, { coarse: 'q1' }), b, { method, tol: 1e-14, maxIter: 1, keepIterates: true });
    assert(part.subs.every((s) => coreMax(two.iterates[1], s) > 0), `${method}: two-level reaches all 36 subdomains at once`);
  }
});

test('scaling experiment of the widget: with minimal overlap (ℓ = 1) two-level ASM+CG beats one-level at K = 8, weak and strong', () => {
  // weak: N = 8K; strong: N = 48. Measured: weak 35 vs 22, strong 30 vs 20 iterations.
  // (With ℓ = 3 in the strong experiment the boxes are 6 cells wide and δ = 6h = H: one-level 19, two-level 20 —
  //  the coarse space no longer pays, which is why the caption restricts the claim to small overlap.)
  for (const N of [64, 48]) {
    const prob = ddPoissonProblem(N), part = boxPartition(N, 8, 8, 1);
    const one = solveSchwarz(schwarzSetup(prob, part), prob.b, { method: 'asm-cg', tol: 1e-8 }).iters;
    const two = solveSchwarz(schwarzSetup(prob, part, { coarse: 'q1' }), prob.b, { method: 'asm-cg', tol: 1e-8 }).iters;
    assert(two <= 0.75 * one, `N=${N}: one-level ${one}, two-level ${two}`);
  }
  const prob = ddPoissonProblem(48), part = boxPartition(48, 8, 8, 3);
  const one = solveSchwarz(schwarzSetup(prob, part), prob.b, { method: 'asm-cg', tol: 1e-8 }).iters;
  const two = solveSchwarz(schwarzSetup(prob, part, { coarse: 'q1' }), prob.b, { method: 'asm-cg', tol: 1e-8 }).iters;
  assert(Math.abs(one - two) <= 2 && one <= 22, `generous overlap: one-level ${one} ≈ two-level ${two}`);
});
