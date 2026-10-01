/**
 * Tests for the 1D nodal DG method (lib/core/dg/dg1d.js) and the slope limiter
 * (lib/core/dg/limiters.js): convergence rates, conservation, L²-stability,
 * weak/strong equivalence, aliasing with an entropy-conservative flux.
 */
import { test, assert, assertClose, assertRate } from '../harness.js';
import { makeDG1D, integrate } from '../../lib/core/dg/dg1d.js';
import { makeSlopeLimiter } from '../../lib/core/dg/limiters.js';
import { advection, burgers } from '../../lib/core/models/scalar.js';
import { makeStepper } from '../../lib/core/time/rk.js';
import { operatorMatrix } from '../../lib/core/dg/spectrum.js';
import { symEig } from '../../lib/core/la/dense.js';
import { fitRate } from '../../lib/core/verify/rates.js';

const TWO_PI = 2 * Math.PI;
const u0 = (x) => Math.sin(TWO_PI * x);
const VARIANTS = [
  { nodes: 'gll', quad: 'exact' },
  { nodes: 'gauss', quad: 'collocated' },
  { nodes: 'gll', quad: 'collocated' }, // DG-SEM (lumped mass)
];

/** Run periodic linear advection (a = 1) to T with LSRK4 and a small time step; return L² error. */
function advError(o, N, T = 1, flux = 'upwind', f0 = u0) {
  const dg = makeDG1D({ ...o, N, model: advection(1), flux });
  const u = dg.project(f0);
  integrate(makeStepper('lsrk4', dg.n), u, dg.rhs, T, 0.05 * dg.h / (2 * o.p + 1));
  return dg.l2Error(u, (x) => f0(x - T));
}

/** pseudo-random vector (deterministic) */
function rnd(n, seed = 3) {
  const u = new Float64Array(n);
  for (let i = 0; i < n; i++) { seed = (seed * 16807) % 2147483647; u[i] = seed / 2147483647 - 0.4; }
  return u;
}

test('DG upwind: L2 error converges at rate p+1 (p = 1..4), smooth periodic advection', () => {
  for (const v of VARIANTS) for (let p = 1; p <= 4; p++) {
    // GLL lumped mass with p = 1 is pre-asymptotic on coarse meshes: use finer meshes there
    const Ns = v.quad === 'collocated' && v.nodes === 'gll' && p === 1 ? [16, 32, 64, 128] : [4, 8, 16, 32];
    const hs = [], es = [];
    for (const N of Ns) { hs.push(1 / N); es.push(advError({ ...v, p }, N)); }
    assertRate(fitRate(hs, es), p + 1, `${v.nodes}/${v.quad} p=${p}`);
  }
});

test('DG central flux: converges at least at rate p (rates are irregular)', () => {
  const f0 = (x) => Math.exp(Math.sin(TWO_PI * x));
  for (let p = 1; p <= 4; p++) {
    const hs = [], es = [];
    for (const N of [10, 20, 40, 80]) { hs.push(1 / N); es.push(advError({ p, nodes: 'gll', quad: 'exact' }, N, 1, 'central', f0)); }
    const r = fitRate(hs, es);
    assert(r >= p - 0.2, `central p=${p}: rate ${r}`);
  }
});

test('DG p = 0 is the first-order upwind finite volume method', () => {
  const hs = [], es = [];
  for (const N of [40, 80, 160, 320]) { hs.push(1 / N); es.push(advError({ p: 0 }, N)); }
  assertRate(fitRate(hs, es), 1, 'p=0');
});

test('DG weak and strong forms coincide for linear advection (all quadrature variants)', () => {
  for (const v of VARIANTS) for (let p = 1; p <= 5; p++) {
    const W = makeDG1D({ ...v, p, N: 6, model: advection(1.3), flux: 'upwind', form: 'weak' });
    const S = makeDG1D({ ...v, p, N: 6, model: advection(1.3), flux: 'upwind', form: 'strong' });
    const u = rnd(W.n), a = new Float64Array(W.n), b = new Float64Array(W.n);
    W.rhs(u, 0, a); S.rhs(u, 0, b);
    for (let i = 0; i < W.n; i++) assertClose(a[i], b[i], 1e-11, 0, `${v.nodes}/${v.quad} p=${p}`);
  }
});

test('DG conserves the total mass exactly (Burgers, Rusanov flux, limiter on)', () => {
  for (const v of VARIANTS) {
    const dg = makeDG1D({ ...v, p: 3, N: 20, model: burgers, flux: 'rusanov' });
    const lim = makeSlopeLimiter(dg, { M: 0 });
    const u = dg.project((x) => 0.5 + Math.sin(TWO_PI * x));
    const m0 = dg.mass(u);
    const step = makeStepper('ssprk3', dg.n, lim.apply);
    integrate(step, u, dg.rhs, 0.5, 0.002); // shock forms at t = 1/(2π) ≈ 0.16
    assertClose(dg.mass(u), m0, 1e-13, 0, `${v.nodes}/${v.quad}`);
  }
});

test('DG is L2-stable: symmetric part of M L is negative semidefinite (upwind) / zero (central)', () => {
  for (const v of VARIANTS) for (let p = 1; p <= 4; p++) for (const flux of ['upwind', 'central']) {
    const N = 5;
    const dg = makeDG1D({ ...v, p, N, model: advection(1), flux });
    const n = dg.n, L = operatorMatrix(dg.rhs, n), np = dg.np, M = dg.ref.M;
    // B = (h/2) M_block L,  energy rate = uᵀ B u = uᵀ sym(B) u
    const B = new Float64Array(n * n);
    for (let k = 0; k < N; k++) for (let i = 0; i < np; i++) for (let j = 0; j < n; j++) {
      let s = 0;
      for (let m = 0; m < np; m++) s += M[i * np + m] * L[(k * np + m) * n + j];
      B[(k * np + i) * n + j] = s * dg.h / 2;
    }
    const Sym = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Sym[i * n + j] = 0.5 * (B[i * n + j] + B[j * n + i]);
    const { values } = symEig(Sym, n);
    const lo = values[0], hi = values[n - 1];
    if (flux === 'upwind') { assert(hi < 1e-12, `upwind max eig ${hi}`); assert(lo < -1e-3, 'upwind must dissipate jumps'); }
    else { assert(Math.abs(lo) < 1e-12 && Math.abs(hi) < 1e-12, `central sym part ${lo}, ${hi}`); }
  }
});

test('Burgers, entropy-conservative flux: exact integration conserves energy, GLL collocation does not (aliasing)', () => {
  // generic (under-resolved) piecewise polynomial data: random nodal values
  const u = rnd(8 * 5, 11);
  const rate = (o) => {
    const dg = makeDG1D({ p: 4, N: 8, model: burgers, flux: 'ec', ...o });
    const du = new Float64Array(dg.n); dg.rhs(u, 0, du);
    // dE/dt = uᵀ (h/2) M du
    let s = 0; const np = dg.np, M = dg.ref.M;
    for (let k = 0; k < dg.N; k++) for (let i = 0; i < np; i++) for (let j = 0; j < np; j++) s += u[k * np + i] * M[i * np + j] * du[k * np + j];
    return s * dg.h / 2;
  };
  assert(Math.abs(rate({ nodes: 'gll', quad: 'exact' })) < 1e-12, 'exact integration');
  assert(Math.abs(rate({ nodes: 'gll', quad: 'collocated' })) > 1e-4, 'collocation should show an aliasing energy error');
});

test('Inflow boundary condition: rate p+1 on a non-periodic domain', () => {
  const ex = (x, t) => Math.sin(TWO_PI * (x - t));
  for (let p = 1; p <= 3; p++) {
    const hs = [], es = [];
    for (const N of [4, 8, 16, 32]) {
      const dg = makeDG1D({ p, N, model: advection(1), flux: 'upwind', bc: 'inflow', bcValue: (x, t) => ex(x, t) });
      const u = dg.project((x) => ex(x, 0));
      integrate(makeStepper('lsrk4', dg.n), u, dg.rhs, 0.7, 0.05 * dg.h / (2 * p + 1));
      hs.push(1 / N); es.push(dg.l2Error(u, (x) => ex(x, 0.7)));
    }
    assertRate(fitRate(hs, es), p + 1, `inflow p=${p}`);
  }
});

test('Slope limiter: keeps averages and linear functions, flattens a jump, TVB spares smooth extrema', () => {
  const dg = makeDG1D({ p: 3, N: 20, model: advection(1) });
  // linear function: untouched
  const lin = dg.project((x) => 2 * x - 0.3);
  const c = Float64Array.from(lin);
  // non-periodic-looking data on a periodic mesh: the wrap-around element sees a jump; exclude it
  const L0 = makeSlopeLimiter(dg, { M: 0 });
  L0.apply(c);
  for (let k = 1; k < dg.N - 1; k++) for (let i = 0; i < dg.np; i++) assertClose(c[k * dg.np + i], lin[k * dg.np + i], 1e-13, 0, 'linear');
  // step function: averages kept, result non-oscillatory (within [0,1])
  const st = dg.project((x) => (x > 0.33 && x < 0.71 ? 1 : 0));
  const before = dg.mass(st);
  const n1 = L0.apply(st);
  assert(n1 > 0, 'some element must be troubled');
  assertClose(dg.mass(st), before, 1e-14, 0, 'mass');
  for (let i = 0; i < dg.n; i++) assert(st[i] > -1e-12 && st[i] < 1 + 1e-12, `overshoot ${st[i]}`);
  // smooth sine: M = 0 clips the extrema, large M leaves everything untouched
  const s0 = dg.project((x) => Math.sin(TWO_PI * x)), s1 = Float64Array.from(s0), s2 = Float64Array.from(s0);
  const nClip = L0.apply(s1);
  const nTVB = makeSlopeLimiter(dg, { M: 50 }).apply(s2);
  assert(nClip > 0, 'minmod should flag the extrema of a sine');
  assert(nTVB === 0, 'TVB with M = 50 should leave the sine untouched');
});

test('Limited DG on a square wave stays (almost) within the initial bounds', () => {
  const dg = makeDG1D({ p: 2, N: 50, model: advection(1), flux: 'upwind' });
  const lim = makeSlopeLimiter(dg, { M: 0 });
  const u = dg.project((x) => (x > 0.25 && x < 0.6 ? 1 : 0));
  lim.apply(u);
  const step = makeStepper('ssprk3', dg.n, lim.apply);
  integrate(step, u, dg.rhs, 1, 0.1 * dg.h);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < dg.n; i++) { lo = Math.min(lo, u[i]); hi = Math.max(hi, u[i]); }
  assert(lo > -0.02 && hi < 1.02, `bounds ${lo} ${hi}`);
});

test('DG with p = 0 has exactly the right-hand side of first-order upwind finite volumes', async () => {
  const { makeFV1D } = await import('../../lib/core/fv/fv1d.js');
  for (const [model, flux] of [[advection(1), 'upwind'], [burgers, 'godunov']]) {
    const dg = makeDG1D({ p: 0, N: 17, model, flux });
    const fv = makeFV1D({ model, N: 17, flux, recon: 'none' });
    const u = rnd(17, 9), a = new Float64Array(17), b = new Float64Array(17);
    dg.rhs(u, 0, a); fv.rhs(u, 0, b);
    for (let i = 0; i < 17; i++) assertClose(a[i], b[i], 1e-12);
  }
});
