import { test, assert, assertClose, assertRate } from '../harness.js';
import { exactRiemann } from '../../lib/core/models/eulerRiemann.js';
import { EULER_FLUXES, physFlux, primToCons, consToPrim } from '../../lib/core/models/euler.js';
import { makeEuler1D, eulerCellAverages, eulerStats, SHOCK_TUBES } from '../../lib/core/fv/euler1d.js';
import { makeStepper } from '../../lib/core/time/rk.js';
import { fitRate } from '../../lib/core/verify/rates.js';

/** Run a shock tube to its final time T; returns {U, fv, ex, err (L1 density error vs exact cell averages)}. */
function shockTube(key, N, flux, recon, limiter = 'minmod', cfl = 0.5) {
  const P = SHOCK_TUBES[key];
  const fv = makeEuler1D({ N, flux, recon, limiter });
  const U = eulerCellAverages((x) => (x < P.x0 ? P.WL : P.WR), fv.xf);
  const step = makeStepper(recon === 'muscl' ? 'ssprk2' : 'fe', 3 * N);
  let t = 0, ok = true;
  while (t < P.T - 1e-14) {
    const dt = Math.min(cfl * fv.h / fv.maxSpeed(U), P.T - t);
    step(U, t, dt, fv.rhs); t += dt;
    const s = eulerStats(U, fv.h);
    if (!s.finite || s.minRho <= 0 || s.minP <= 0) { ok = false; break; }
  }
  const ex = exactRiemann(P.WL, P.WR);
  const R = eulerCellAverages((x) => ex.sample((x - P.x0) / P.T), fv.xf);
  let err = 0;
  for (let i = 0; i < N; i++) err += Math.abs(U[3 * i] - R[3 * i]) * fv.h;
  return { U, fv, ex, err, ok };
}

test('exact Riemann solver reproduces Toro\'s star states (Table 4.3)', () => {
  // [WL, WR, p*, u*, rho*L, rho*R] — Toro (2009), tests 1–5 of chapter 4
  const cases = [
    [[1, 0, 1], [0.125, 0, 0.1], 0.30313, 0.92745, 0.42632, 0.26557],
    [[1, -2, 0.4], [1, 2, 0.4], 0.00189, 0, 0.02185, 0.02185],
    [[1, 0, 1000], [1, 0, 0.01], 460.894, 19.5975, 0.57506, 5.99924],
    [[1, 0, 0.01], [1, 0, 100], 46.0950, -6.19633, 5.99242, 0.57511],
    [[5.99924, 19.5975, 460.894], [5.99242, -6.19633, 46.0950], 1691.64, 8.68975, 14.2823, 31.0426],
  ];
  for (const [WL, WR, p, u, rL, rR] of cases) {
    const s = exactRiemann(WL, WR);
    assertClose(s.pStar, p, 1e-5, 5e-6, 'p*');
    assertClose(s.uStar, u, 1e-5, 5e-6, 'u*');
    assertClose(s.rhoStarL, rL, 1e-5, 5e-6, 'rho*L');
    assertClose(s.rhoStarR, rR, 1e-5, 5e-6, 'rho*R');
  }
});

test('exact Riemann solution: Sod wave structure, Rankine–Hugoniot across the shock, isentropic fan', () => {
  const g = 1.4, s = exactRiemann([1, 0, 1], [0.125, 0, 0.1]);
  assert(s.left === 'rarefaction' && s.right === 'shock', 'Sod = left rarefaction, contact, right shock');
  // Rankine–Hugoniot for all three conservation laws across the right shock: S [U] = [F]
  const S = s.rightSpeeds[0];
  const Ua = primToCons(s.rhoStarR, s.uStar, s.pStar), Ub = primToCons(0.125, 0, 0.1);
  const Fa = physFlux(Ua, [0, 0, 0]), Fb = physFlux(Ub, [0, 0, 0]);
  for (let k = 0; k < 3; k++) assertClose(S * (Ua[k] - Ub[k]), Fa[k] - Fb[k], 1e-12, 1e-11, `RH comp ${k}`);
  // inside the fan: p/ρ^γ constant and the left Riemann invariant u + 2c/(γ−1) constant
  for (const xi of [-1.1, -0.8, -0.3, -0.1]) {
    const [r, u, p] = s.sample(xi);
    assertClose(p / r ** g, 1, 1e-12, 1e-12, 'entropy');
    assertClose(u + 2 * Math.sqrt(g * p / r) / (g - 1), 2 * Math.sqrt(g) / (g - 1), 1e-12, 1e-12, 'Riemann invariant');
    assertClose(xi, u - Math.sqrt(g * p / r), 1e-12, 1e-12, 'fan: xi = u - c');
  }
  // star region: p and u continuous across the contact
  const a = s.sample(s.uStar - 1e-9), b = s.sample(s.uStar + 1e-9);
  assertClose(a[1], b[1], 1e-14); assertClose(a[2], b[2], 1e-14);
});

test('Euler numerical fluxes are consistent: F̂(U, U) = F(U)', () => {
  const states = [[1, 0, 1], [0.125, -0.3, 0.1], [2, 1.5, 3], [0.5, -2, 0.2]];
  const out = [0, 0, 0], F = [0, 0, 0];
  for (const W of states) {
    const U = primToCons(...W);
    physFlux(U, F);
    for (const key of Object.keys(EULER_FLUXES)) {
      EULER_FLUXES[key].fn(U, U, out);
      for (let k = 0; k < 3; k++) assertClose(out[k], F[k], 1e-12, 1e-12, `${key}[${k}]`);
    }
    const back = consToPrim(U);
    for (let k = 0; k < 3; k++) assertClose(back[k], W[k], 1e-14, 1e-14, 'prim↔cons');
  }
});

test('Roe flux resolves an isolated stationary contact and a Rankine–Hugoniot shock exactly', () => {
  // stationary contact: same u = 0, same p, different ρ → F̂ = F(UL) = F(UR) = (0, p, 0)
  const out = [0, 0, 0];
  for (const key of ['roe', 'hllc', 'godunov']) {
    EULER_FLUXES[key].fn(primToCons(1, 0, 1), primToCons(0.2, 0, 1), out);
    assertClose(out[0], 0, 1e-13, 0, key); assertClose(out[1], 1, 1e-13, 0, key); assertClose(out[2], 0, 1e-13, 0, key);
  }
  // HLL smears it (nonzero mass flux): the reason HLLC was invented
  EULER_FLUXES.hll.fn(primToCons(1, 0, 1), primToCons(0.2, 0, 1), out);
  assert(Math.abs(out[0]) > 1e-3, 'HLL diffuses a stationary contact');
  // an isolated right-moving shock (Sod's shock, speed S > 0): Roe gives the upwind flux F(UL) exactly
  const s = exactRiemann([1, 0, 1], [0.125, 0, 0.1]);
  const UL = primToCons(s.rhoStarR, s.uStar, s.pStar), UR = primToCons(0.125, 0, 0.1), FL = physFlux(UL, [0, 0, 0]);
  EULER_FLUXES.roe.fn(UL, UR, out);
  for (let k = 0; k < 3; k++) assertClose(out[k], FL[k], 1e-12, 1e-12, `Roe shock comp ${k}`);
});

test('FV Sod: L1 density error decreases (1st order ≈ 0.6–0.75, MUSCL ≈ 0.8–0.95), MUSCL more accurate', () => {
  for (const flux of ['rusanov', 'hllc', 'roefix']) {
    const res = {};
    for (const recon of ['none', 'muscl']) {
      const hs = [], es = [];
      for (const N of [200, 400, 800]) { const r = shockTube('sod', N, flux, recon); hs.push(1 / N); es.push(r.err); assert(r.ok, 'positivity'); }
      res[recon] = { rate: fitRate(hs, es), err: es[es.length - 1] };
    }
    assert(res.none.rate > 0.55 && res.none.rate < 0.8, `${flux} 1st order rate ${res.none.rate}`);
    assert(res.muscl.rate > 0.75 && res.muscl.rate < 1.0, `${flux} MUSCL rate ${res.muscl.rate}`);
    assert(res.muscl.err < 0.5 * res.none.err, `${flux}: MUSCL error ${res.muscl.err} vs ${res.none.err}`);
  }
});

test('FV Euler conservation: mass and energy to round-off, momentum changes exactly by (pL − pR) t', () => {
  // Waves have not reached the boundary at T = 0.2, so the boundary fluxes are F(UL) and F(UR):
  // d/dt Σ ρ h = 0, d/dt Σ E h = 0 (u = 0 at both ends) and d/dt Σ ρu h = pL − pR = 0.9.
  for (const recon of ['none', 'muscl']) {
    const r = shockTube('sod', 200, 'hllc', recon);
    const s = eulerStats(r.U, r.fv.h);
    assertClose(s.mass, 0.5 * 1 + 0.5 * 0.125, 1e-13, 0, 'mass');
    assertClose(s.energy, 0.5 * 2.5 + 0.5 * 0.25, 1e-13, 0, 'energy');
    assertClose(s.momentum, 0.9 * 0.2, 1e-13, 0, 'momentum');
  }
});

test('positivity: HLL/HLLC/Rusanov keep ρ, p > 0 on the 123 problem; Roe (with or without fix) fails', () => {
  for (const flux of ['rusanov', 'hll', 'hllc'])
    for (const recon of ['none', 'muscl']) assert(shockTube('123', 200, flux, recon).ok, `${flux} ${recon}`);
  for (const flux of ['roe', 'roefix']) assert(!shockTube('123', 200, flux, 'none').ok, `${flux} should fail`);
});

test('Roe without entropy fix: expansion shock in the sonic rarefaction (Toro test 1); the fix removes it', () => {
  const bad = shockTube('sonic', 400, 'roe', 'none'), good = shockTube('sonic', 400, 'roefix', 'none');
  assert(bad.err > 1.3 * good.err, `errors ${bad.err} vs ${good.err}`);
  // the expansion shock sits at the sonic point x = x0 (ξ = 0): a large jump between neighbouring cells
  const jump = (U, N) => { let m = 0; const i0 = Math.round(0.3 * N); for (let i = i0 - 3; i < i0 + 3; i++) m = Math.max(m, Math.abs(U[3 * i + 3] - U[3 * i])); return m; };
  assert(jump(bad.U, 400) > 3 * jump(good.U, 400), `jump ${jump(bad.U, 400)} vs ${jump(good.U, 400)}`);
});

test('MUSCL is second order on a smooth Euler density wave (exact solution = translation)', () => {
  // ρ = 1 + 0.2 sin(2πx), u = 1, p = 1, periodic: the exact solution is ρ(x − t), so at T = 1 it equals the initial data.
  const W0 = (x) => [1 + 0.2 * Math.sin(2 * Math.PI * x), 1, 1];
  for (const [recon, limiter, expected] of [['none', 'minmod', 1], ['muscl', 'none', 2], ['muscl', 'vanleer', 2]]) {
    const hs = [], es = [];
    for (const N of [32, 64, 128, 256]) {
      const fv = makeEuler1D({ N, flux: 'hllc', recon, limiter, bc: 'periodic' });
      const U = eulerCellAverages(W0, fv.xf), U0 = Float64Array.from(U);
      const step = makeStepper('ssprk3', 3 * N);
      let t = 0;
      while (t < 1 - 1e-14) { const dt = Math.min(0.4 * fv.h / fv.maxSpeed(U), 1 - t); step(U, t, dt, fv.rhs); t += dt; }
      let e = 0; for (let i = 0; i < N; i++) e += Math.abs(U[3 * i] - U0[3 * i]) * fv.h;
      hs.push(fv.h); es.push(e);
      // periodic: totals conserved to round-off
      const a = eulerStats(U, fv.h), b = eulerStats(U0, fv.h);
      assertClose(a.mass, b.mass, 1e-13); assertClose(a.energy, b.energy, 1e-12);
    }
    // van Leer clips the two smooth extrema: still close to 2 in L1
    assertRate(fitRate(hs, es), expected, `${recon}/${limiter}`, limiter === 'vanleer' ? 0.35 : 0.2, 0.3);
  }
});

test('stationary contact: HLLC, Roe and Godunov keep it exactly; HLL and Rusanov smear it', () => {
  for (const flux of ['hllc', 'roe', 'roefix', 'godunov']) {
    const r = shockTube('contact', 100, flux, 'none');
    assert(r.err < 1e-13, `${flux} error ${r.err}`);
  }
  for (const flux of ['hll', 'rusanov']) assert(shockTube('contact', 100, flux, 'none').err > 1e-3, flux);
});
