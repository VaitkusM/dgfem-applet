import { test, assert, assertClose } from '../harness.js';
import { polygonEdgeFluxes, polygonAreaIntegral, polygonArea2 } from '../../lib/core/verify/divthm.js';
import { advDiffGaussian, burgersBreakingTime, burgersRiemannKind } from '../../lib/core/verify/closedforms.js';
import { burgersRiemann } from '../../lib/core/models/scalar.js';
import { gaussLegendre } from '../../lib/core/quad/gauss1d.js';

// the vector fields of the chapter-0 divergence-theorem playground
const VFIELDS = {
  source: { F: (x, y) => [x / 2, y / 2], div: () => 1 },
  vortex: { F: (x, y) => [-y, x], div: () => 0 },
  wind: { F: () => [1, 0.4], div: () => 0 },
  gauss: { F: (x, y) => { const e = Math.exp(-(x * x + y * y) / 2); return [x * e, y * e]; }, div: (x, y) => { const r2 = x * x + y * y; return (2 - r2) * Math.exp(-r2 / 2); } },
  waves: { F: (x, y) => [Math.sin(2 * x) / 2, Math.sin(2 * y) / 2], div: (x, y) => Math.cos(2 * x) + Math.cos(2 * y) },
};
const POLYS = {
  start: [[-1.2, -0.8], [0.6, -1.3], [1.4, 0.2], [0.5, 1.2], [-0.9, 0.9]], // initial polygon of the widget (CCW)
  clockwise: [[-0.9, 0.9], [0.5, 1.2], [1.4, 0.2], [0.6, -1.3], [-1.2, -0.8]],
  nonconvex: [[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [0.2, -0.4], [-1.5, 1.5]],
  corner: [[2.1, 2.1], [-2.1, 2.1], [-2.1, -2.1], [2.1, -2.1], [0, 0.3]], // vertices at the clamp limit ±2.1
};
const sumFlux = (poly, F) => polygonEdgeFluxes(poly, F).fl.reduce((s, e) => s + e.flux, 0);

test('divergence theorem: Σ edge fluxes = ∫ div F for every field of the playground (convex, non-convex, clockwise)', () => {
  for (const [pn, poly] of Object.entries(POLYS)) for (const [fn, V] of Object.entries(VFIELDS)) {
    const flux = sumFlux(poly, V.F), vol = polygonAreaIntegral(poly, V.div);
    assertClose(flux, vol, 1e-8, 0, `${pn}/${fn}`);
  }
  // div-free vortex: zero net flux although single edges carry flux; the flux through the edge a→b of a
  // counter-clockwise polygon is exactly (|a|² − |b|²)/2 (small for the start polygon, whose vertices are
  // almost equally far from the origin: max 0.23; large once a vertex is dragged to the origin)
  for (const poly of [POLYS.start, POLYS.nonconvex]) {
    const { fl } = polygonEdgeFluxes(poly, VFIELDS.vortex.F);
    fl.forEach((e, i) => { const a = poly[i], b = poly[(i + 1) % poly.length]; assertClose(e.flux, (a[0] ** 2 + a[1] ** 2 - b[0] ** 2 - b[1] ** 2) / 2, 1e-13); });
    assertClose(fl.reduce((s, e) => s + e.flux, 0), 0, 1e-13);
  }
  assertClose(Math.max(...polygonEdgeFluxes(POLYS.start, VFIELDS.vortex.F).fl.map((e) => Math.abs(e.flux))), 0.23, 1e-12);
  assert(Math.max(...polygonEdgeFluxes(POLYS.nonconvex, VFIELDS.vortex.F).fl.map((e) => Math.abs(e.flux))) > 2);
  // F = (x,y)/2 has div 1: net flux = area
  for (const poly of Object.values(POLYS)) assertClose(sumFlux(poly, VFIELDS.source.F), Math.abs(polygonArea2(poly)) / 2, 1e-12);
  // localised source: div = (2 − r²) e^{−r²/2} changes sign on the circle r = √2
  assert(VFIELDS.gauss.div(1.41, 0) > 0 && VFIELDS.gauss.div(0, 1.42) < 0);
});

test('divergence theorem: outward normals and orientation', () => {
  const sq = [[0, 0], [2, 0], [2, 1], [0, 1]];
  const ccw = polygonEdgeFluxes(sq, () => [1, 0]), cw = polygonEdgeFluxes([...sq].reverse(), () => [1, 0]);
  assert(ccw.orient === 1 && cw.orient === -1);
  assertClose(ccw.fl[0].ny, -1); assertClose(ccw.fl[1].nx, 1); assertClose(ccw.fl[2].ny, 1); assertClose(ccw.fl[3].nx, -1);
  assertClose(ccw.fl[1].flux, 1); assertClose(ccw.fl[3].flux, -1); // uniform wind: out on the right, in on the left
  for (const e of cw.fl) { const c = [1, 0.5]; assert((e.mx - c[0]) * e.nx + (e.my - c[1]) * e.ny > 0, 'normal not outward'); }
});

test('divergence theorem: coincident vertices (zero-length edges) give finite, consistent results', () => {
  // regression: dragging two neighbouring vertices onto the same corner made the normal 0/0 = NaN
  const poly = [[-1.2, -0.8], [2.1, 2.1], [2.1, 2.1], [0.5, 1.2], [-0.9, 0.9]];
  for (const [fn, V] of Object.entries(VFIELDS)) {
    const { fl } = polygonEdgeFluxes(poly, V.F);
    assert(fl.every((e) => Number.isFinite(e.flux) && Number.isFinite(e.nx) && Number.isFinite(e.ny)), `${fn}: non-finite edge data`);
    assert(fl[1].flux === 0 && fl[1].nx === 0 && fl[1].ny === 0, 'degenerate edge must contribute nothing');
    assertClose(fl.reduce((s, e) => s + e.flux, 0), polygonAreaIntegral(poly, V.div), 1e-8, 0, fn);
  }
  // all vertices in one point: everything is zero
  const pt = [[1, 1], [1, 1], [1, 1]];
  assert(sumFlux(pt, VFIELDS.gauss.F) === 0 && polygonAreaIntegral(pt, VFIELDS.gauss.div) === 0);
});

test('advection–diffusion Gaussian: satisfies u_t + a u_x = k u_xx, conserves mass σ√π, peak moves with speed a', () => {
  const G = gaussLegendre(20);
  for (const a of [-1, 0, 0.6, 2]) for (const k of [0, 0.002, 0.02, 0.1]) {
    const prm = { a, k, sigma: 0.25, x0: 1 }, U = (x, t) => advDiffGaussian(x, t, prm).u;
    for (const t of [0, 0.3, 1.5]) {
      // PDE residual by central differences (h = 1e-3 ⇒ truncation error ≪ 1e-3)
      for (const x of [-0.4, 0.3, 0.9, 1.0, 1.7, 3.1]) {
        const h = 1e-3, tt = Math.max(t, h);
        const ut = (U(x, tt + h) - U(x, tt - h)) / (2 * h), ux = (U(x + h, tt) - U(x - h, tt)) / (2 * h);
        const uxx = (U(x + h, tt) - 2 * U(x, tt) + U(x - h, tt)) / (h * h);
        assertClose(ut + a * ux - k * uxx, 0, 2e-3, 0, `residual a=${a} k=${k} t=${tt} x=${x}`);
        assertClose(advDiffGaussian(x, tt, prm).ux, ux, 1e-4, 0, 'ux');
      }
      let mass = 0;
      for (let e = 0; e < 60; e++) for (let q = 0; q < 20; q++) mass += G.w[q] / 2 * 15 / 60 * U(-5 + 15 * (e + (G.x[q] + 1) / 2) / 60, t);
      assertClose(mass, 0.25 * Math.sqrt(Math.PI), 1e-9, 0, `mass a=${a} k=${k} t=${t}`);
      // peak at x0 + a t with height σ/√(σ²+4kt); u_x = 0 there
      assertClose(U(1 + a * t, t), 0.25 / Math.sqrt(0.0625 + 4 * k * t), 1e-14);
      assertClose(advDiffGaussian(1 + a * t, t, prm).ux, 0, 1e-14);
      // diffusive flux −k u_x is ≥ 0 right of the peak and ≤ 0 left of it, whatever the sign of a
      assert(-k * advDiffGaussian(1 + a * t + 0.2, t, prm).ux >= 0 && -k * advDiffGaussian(1 + a * t - 0.2, t, prm).ux <= 0);
    }
  }
});

test('Burgers breaking time t* = −1/min u0\' ; characteristics cross exactly after t*', () => {
  const u0 = (x) => 0.25 + 0.5 * Math.sin(2 * Math.PI * x), du0 = (x) => Math.PI * Math.cos(2 * Math.PI * x);
  const ts = burgersBreakingTime(du0);
  assertClose(ts, 1 / Math.PI, 1e-12);
  assert(burgersBreakingTime(() => 0.3) === Infinity, 'increasing data never breaks');
  // the map x0 ↦ x0 + u0(x0) t is monotone (no crossing) for t < t*, and not monotone for t > t*
  const monotone = (t) => { let ok = true; for (let i = 0; i < 4000; i++) { const a = i / 4000, b = (i + 1) / 4000; if (b + u0(b) * t <= a + u0(a) * t) ok = false; } return ok; };
  assert(monotone(0.99 * ts), 'crossing before t*');
  assert(!monotone(1.01 * ts), 'no crossing after t*');
});

test('Burgers Riemann problem: shock / rarefaction / constant state over the whole slider range', () => {
  for (let i = 0; i <= 44; i++) for (let j = 0; j <= 44; j++) {
    const uL = -1 + 0.05 * i, uR = -1 + 0.05 * j, { kind, speed } = burgersRiemannKind(uL, uR);
    assert(kind === (i > j ? 'shock' : i < j ? 'rarefaction' : 'constant') || Math.abs(uL - uR) < 1e-12, `kind uL=${uL} uR=${uR}`);
    assertClose(speed, (uL + uR) / 2, 1e-15);
    for (const xi of [-1.5, -0.3, 0, 0.3, 1.5]) {
      const u = burgersRiemann(uL, uR, xi);
      assert(Number.isFinite(u) && u >= Math.min(uL, uR) - 1e-15 && u <= Math.max(uL, uR) + 1e-15, 'solution outside [uL,uR]');
    }
    if (kind === 'shock') {
      // Rankine–Hugoniot: s (uL − uR) = f(uL) − f(uR) with f = u²/2; Lax: uL > s > uR
      assertClose(speed * (uL - uR), uL * uL / 2 - uR * uR / 2, 1e-14);
      assert(uL > speed && speed > uR);
      assert(burgersRiemann(uL, uR, speed - 1e-9) === uL && burgersRiemann(uL, uR, speed + 1e-9) === uR);
    }
  }
  assert(burgersRiemannKind(0.3, 0.3).kind === 'constant' && burgersRiemann(0.3, 0.3, 5) === 0.3);
  assert(burgersRiemann(-0.5, 1, 0) === 0, 'sonic point u = 0 inside the fan');
});
