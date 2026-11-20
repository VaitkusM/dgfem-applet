import { test, assert, assertClose, assertRate } from '../harness.js';
import { circleLS, flowerLS } from '../../lib/core/mesh/levelset.js';
import { triGrid } from '../../lib/core/mesh/structured.js';
import { classifyMesh, cutTriangle, cutVolumeRule, cutInterfaceRule, cutArcRule, INSIDE, CUT } from '../../lib/core/quad/cut.js';
import { fitRate } from '../../lib/core/verify/rates.js';

const FLOWER = { cx: 0.51, cy: 0.47, R: 0.3, a: 0.2, k: 5, rot: 0.3 };

test('level sets: circle is an exact SDF; flower area = πR²(1 + a²/2), perimeter matches a fine polyline', () => {
  const c = circleLS({ cx: 0.4, cy: 0.6, R: 0.25 });
  for (const [x, y] of [[0.1, 0.1], [0.5, 0.6], [0.9, 0.3]]) {
    const p = c.closest(x, y);
    assertClose(c.phi(p.x, p.y), 0, 1e-15);
    assertClose(Math.abs(c.phi(x, y)), p.dist, 1e-14, 0, 'SDF');
  }
  const f = flowerLS(FLOWER);
  // ½∫R(θ)² dθ by the periodic trapezoidal rule
  let A = 0; const M = 2000;
  for (let i = 0; i < M; i++) A += 0.5 * f.radius(2 * Math.PI * i / M) ** 2 * 2 * Math.PI / M;
  assertClose(f.area, A, 1e-13, 0, 'flower area');
  const P = f.polyline(20000); let L = 0;
  for (let i = 0; i < 20000; i++) L += Math.hypot(P[2 * i + 2] - P[2 * i], P[2 * i + 3] - P[2 * i + 1]);
  assertClose(f.perimeter, L, 1e-6, 0, 'flower perimeter');
  // φ has the right sign but is not a distance: |φ| ≠ dist somewhere
  let maxDiff = 0;
  for (let i = 0; i < 50; i++) { const x = 0.05 + 0.9 * ((i * 37) % 50) / 50, y = 0.05 + 0.9 * ((i * 11) % 50) / 50; maxDiff = Math.max(maxDiff, Math.abs(Math.abs(f.phi(x, y)) - f.closest(x, y).dist)); }
  assert(maxDiff > 1e-3, 'flower level set is not an SDF');
});

test('flower closest point (Newton, several initial guesses): on Γ, orthogonal, globally closest', () => {
  const f = flowerLS(FLOWER);
  let seed = 3;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const samples = f.polyline(8000);
  for (let it = 0; it < 60; it++) {
    const x = rnd(), y = rnd(), p = f.closest(x, y);
    assert(Math.abs(f.phi(p.x, p.y)) < 1e-13, `residual ${f.phi(p.x, p.y)}`);
    const [tx, ty] = f.tangent(p.t);
    assert(Math.abs((x - p.x) * tx + (y - p.y) * ty) / Math.hypot(tx, ty) < 1e-11, 'x − x_Γ ⟂ tangent');
    let dmin = Infinity;
    for (let i = 0; i < 8000; i++) dmin = Math.min(dmin, Math.hypot(samples[2 * i] - x, samples[2 * i + 1] - y));
    assert(p.dist <= dmin + 1e-12, 'globally closest');
    const [dx, dy] = f.distVec(x, y);
    assertClose(Math.hypot(dx, dy), p.dist, 1e-14);
    // outward normal: φ increases along n
    assert(f.phi(p.x + 1e-6 * p.nx, p.y + 1e-6 * p.ny) > 0 && f.phi(p.x - 1e-6 * p.nx, p.y - 1e-6 * p.ny) < 0, 'outward normal');
  }
});

test('linear cut is exact for a straight interface (area, moments, length, normal)', () => {
  // φ = x + 0.5y − 0.6 on [0,1]²: Ω = {x < 0.6 − 0.5y}, area = ∫₀¹ (0.6 − 0.5y) dy = 0.35
  const phi = (x, y) => x + 0.5 * y - 0.6, m = triGrid(5, 7, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.2 });
  const C = classifyMesh(m, phi);
  let A = 0, Mx = 0, L = 0;
  for (let t = 0; t < m.nTri; t++) {
    if (C.cls[t] === 0) continue;
    const V = [0, 1, 2].map((k) => [m.nodes[2 * m.tris[3 * t + k]], m.nodes[2 * m.tris[3 * t + k] + 1]]);
    const cut = C.cls[t] === CUT ? C.cuts[t] : cutTriangle(phi, V[0], V[1], V[2]);
    const R = cutVolumeRule(cut, 2);
    for (let q = 0; q < R.n; q++) { A += R.w[q]; Mx += R.w[q] * R.x[q] * R.x[q]; }
    if (C.cls[t] === CUT) {
      const I = cutInterfaceRule(cut, 2);
      for (let q = 0; q < I.n; q++) { L += I.w[q]; assertClose(I.nx[q], 2 / Math.sqrt(5), 1e-12); assertClose(I.ny[q], 1 / Math.sqrt(5), 1e-12); }
    }
  }
  assertClose(A, 0.35, 1e-14, 0, 'area');
  // ∫∫ x² = ∫₀¹ (0.6 − 0.5y)³/3 dy = [(0.6⁴ − 0.1⁴)/(4·0.5·3)]
  assertClose(Mx, (0.6 ** 4 - 0.1 ** 4) / 6, 1e-14, 0, 'second moment');
  assertClose(L, Math.hypot(0.5, 1), 1e-14, 0, 'interface length');
});

test('cut quadrature: area and length errors O(h²); each refinement level gains ≈ 4×; curved cut exact', () => {
  for (const ls of [circleLS({ cx: 0.513, cy: 0.478, R: 0.33 }), flowerLS(FLOWER)]) {
    const meas = (N, levels, curved) => {
      const m = triGrid(N, N), C = classifyMesh(m, ls.phi, { levels, lip: ls.isSDF ? 1 : 2, curved: curved ? ls : undefined });
      let A = 0, L = 0;
      for (let t = 0; t < m.nTri; t++) {
        if (C.cls[t] === INSIDE) A += 0.5 / N / N;
        else if (C.cls[t] === CUT) {
          A += C.cuts[t].area;
          if (curved) { const r = cutArcRule(C.cuts[t], 8); for (let q = 0; q < r.n; q++) L += r.w[q]; } else L += C.cuts[t].length;
        }
      }
      return { dA: Math.abs(A - ls.area), dL: Math.abs(L - ls.perimeter) };
    };
    const Ns = [16, 32, 64, 128], hs = Ns.map((N) => 1 / N);
    const r0 = Ns.map((N) => meas(N, 0));
    assertRate(fitRate(hs, r0.map((r) => r.dA)), 2, `${ls.kind} area`);
    assertRate(fitRate(hs, r0.map((r) => r.dL)), 2, `${ls.kind} length`);
    const e0 = meas(16, 0).dA, e1 = meas(16, 1).dA, e2 = meas(16, 2).dA;
    assert(e0 / e1 > 2.5 && e1 / e2 > 2.5, `${ls.kind} refinement gains ${e0 / e1}, ${e1 / e2}`);
    for (const N of [8, 32]) {
      const c = meas(N, 1, true);
      assert(c.dA < 1e-12 && c.dL < 1e-11, `${ls.kind} curved cut N=${N}: ${c.dA}, ${c.dL}`); // measured ≤ 2.4e-14, ≤ 4.9e-13
    }
  }
});

test('classification: cut elements have volume fraction in (0, 1], inside elements exactly 1', () => {
  const ls = circleLS({ cx: 0.47, cy: 0.52, R: 0.31 }), m = triGrid(20, 20), C = classifyMesh(m, ls.phi);
  for (let t = 0; t < m.nTri; t++) {
    if (C.cls[t] === CUT) assert(C.frac[t] > 0 && C.frac[t] <= 1 + 1e-14, 'fraction');
    if (C.cls[t] === INSIDE) assert(C.frac[t] === 1);
  }
});

test('cut data are typed arrays with the documented layouts (6 per sub-triangle / segment, 4 per curved piece, 2 per arc)', () => {
  const ls = circleLS({ cx: 0.5, cy: 0.5, R: 0.3 });
  // A = (0.75, 0.5) inside, B = (0.95, 0.5) and C = (0.75, 0.7) outside
  const A = [0.75, 0.5], B = [0.95, 0.5], C = [0.75, 0.7];
  const lin = cutTriangle(ls.phi, A, B, C);
  assert(lin.tris instanceof Float64Array && lin.segs instanceof Float64Array, 'typed');
  assert(lin.tris.length === 6 && lin.segs.length === 6 && lin.curved.length === 0, 'one vertex inside: one sub-triangle, one segment');
  // sub-triangles are CCW, the segment normal is the outward normal (φ increases along it)
  const [ax, ay, bx, by, cx, cy] = lin.tris;
  assert((bx - ax) * (cy - ay) - (cx - ax) * (by - ay) > 0, 'CCW');
  const [px, py, qx, qy, nx, ny] = lin.segs, mx = (px + qx) / 2, my = (py + qy) / 2;
  assert(ls.phi(mx + 1e-3 * nx, my + 1e-3 * ny) > ls.phi(mx, my), 'outward normal');
  assertClose(Math.hypot(nx, ny), 1, 1e-14);
  assertClose(lin.area, 0.5 * Math.hypot(px - 0.75, py - 0.5) * Math.hypot(qx - 0.75, qy - 0.5), 1e-15, 0, 'area of the corner triangle');
  const cur = cutTriangle(ls.phi, A, B, C, { curved: ls });
  assert(cur.curved instanceof Float64Array && cur.curved.length === 4 && cur.arcs.length === 2 && cur.tris.length === 0, 'curved piece');
  // exact: Γ crosses AB at P* = (0.8, 0.5) and AC at Q* = (0.75, 0.5 + √(0.3² − 0.25²));
  // K ∩ Ω = triangle A P* Q* + circular segment over the chord P*Q*
  const yq = 0.5 + Math.sqrt(0.09 - 0.0625), chord = Math.hypot(0.05, yq - 0.5), th = 2 * Math.asin(chord / 0.6);
  assertClose(cur.area, 0.5 * 0.05 * (yq - 0.5) + 0.5 * 0.09 * (th - Math.sin(th)), 1e-14, 0, 'curved area = triangle A P* Q* + circular segment');
});
