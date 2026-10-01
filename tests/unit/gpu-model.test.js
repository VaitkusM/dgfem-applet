/**
 * Tests for the chapter-12 helper models: race-condition simulation, element
 * colouring, face-buffer gather (lib/gpu/races.js) and the cost / memory-transaction
 * model (lib/gpu/cost.js).
 */
import { test, assert, assertClose } from '../harness.js';
import { lockstepScatter, colorElements, isValidColoring, gridFaces, faceBufferGather } from '../../lib/gpu/races.js';
import { kernelCosts, stageTotals, derivativeFlops, roofline, transactions, warpAddresses } from '../../lib/gpu/cost.js';
import { quadGrid } from '../../lib/core/mesh/structured.js';

const toLists = (conn, k) => Array.from({ length: conn.length / k }, (_, e) => Array.from(conn.slice(e * k, e * k + k)));

test('races: lockstep scatter-add over all elements loses updates; colouring fixes it', () => {
  const g = quadGrid(4, 4), k = 4, T = toLists(g.quads, k), V = T.map((t) => t.map(() => 1));
  // exact result: vertex valence (number of elements touching the vertex)
  const valence = new Float64Array(g.nVert); for (const v of g.quads) valence[v]++;
  const all = [Array.from({ length: g.nElem }, (_, e) => e)];
  const naive = lockstepScatter(g.nVert, T, V, all);
  let wrong = 0; for (let v = 0; v < g.nVert; v++) if (naive.out[v] !== valence[v]) wrong++;
  assert(naive.lost > 0 && wrong > 0, 'naive scatter must lose updates');
  // every vertex shared by c > 1 elements ends with 1 instead of c: c − 1 updates lost
  assertClose(wrong, 9 + 12, 0, 0, 'wrong vertices = 9 interior + 12 edge-interior');
  assertClose(naive.lost, 4 * 16 - 25, 0, 0, 'lost = Σ (valence − 1)');
  for (let v = 0; v < g.nVert; v++) assertClose(naive.out[v], 1, 0);
  const { color, nColors } = colorElements(g.quads, k);
  assertClose(nColors, 4, 0);
  assert(isValidColoring(g.quads, k, color), 'valid colouring');
  const groups = Array.from({ length: nColors }, (_, c) => [...color.keys()].filter((e) => color[e] === c));
  const col = lockstepScatter(g.nVert, T, V, groups);
  assertClose(col.lost, 0, 0);
  for (let v = 0; v < g.nVert; v++) assertClose(col.out[v], valence[v], 0);
  // one element per group = sequential = also correct
  const seq = lockstepScatter(g.nVert, T, V, T.map((_, e) => [e]));
  for (let v = 0; v < g.nVert; v++) assertClose(seq.out[v], valence[v], 0);
});

test('races: greedy colouring of a triangle mesh is valid', () => {
  // two triangles per square, "/" diagonals
  const nx = 5, tris = [];
  for (let j = 0; j < nx; j++) for (let i = 0; i < nx; i++) { const v = j * (nx + 1) + i; tris.push(v, v + 1, v + nx + 2, v, v + nx + 2, v + nx + 1); }
  const { color, nColors } = colorElements(tris, 3);
  assert(isValidColoring(tris, 3, color), 'valid');
  assert(nColors >= 6, `a vertex with 6 triangles needs ≥ 6 colours, got ${nColors}`);
});

test('races: DG face buffer + gather = sequential per-face scatter; net flux sums to zero', () => {
  const nx = 4, ny = 3, F = gridFaces(nx, ny);
  assertClose(F.nFace, (nx - 1) * ny + nx * (ny - 1), 0);
  const flux = Float64Array.from({ length: F.nFace }, (_, f) => Math.sin(1 + 3 * f));
  const R = faceBufferGather(nx * ny, F, flux);
  const ref = new Float64Array(nx * ny);
  for (let f = 0; f < F.nFace; f++) { ref[F.minus[f]] += flux[f]; ref[F.plus[f]] -= flux[f]; }
  for (let e = 0; e < nx * ny; e++) assertClose(R[e], ref[e], 1e-15);
  assertClose(R.reduce((s, x) => s + x, 0), 0, 1e-13, 0, 'conservation');
  // naive lockstep per-face scatter into element accumulators loses updates
  const T = [], V = [];
  for (let f = 0; f < F.nFace; f++) { T.push([F.minus[f], F.plus[f]]); V.push([flux[f], -flux[f]]); }
  const naive = lockstepScatter(nx * ny, T, V, [Array.from({ length: F.nFace }, (_, f) => f)]);
  assert(naive.lost > 0, 'per-face scatter races');
});

test('cost model: FLOP/byte counts and sum factorisation', () => {
  const k1 = kernelCosts(1), k3 = kernelCosts(3);
  const by = (L, n) => L.find((x) => x.name === n);
  assertClose(by(k1, 'volume').flops, 13, 0); assertClose(by(k3, 'volume').flops, 21, 0);
  assertClose(by(k3, 'volume').bytes, 8, 0);
  assertClose(by(k3, 'rkStage').ai, 0.25, 1e-15);
  for (let p = 1; p <= 8; p++) for (const k of kernelCosts(p)) assert(k.ai < 6, `${k.name} p=${p} ai=${k.ai} — all kernels below a ridge point of ~10–60 FLOP/B`);
  // numbers quoted in chapter 12: p = 3, unfused ≈ 0.68 FLOP/B, fused ≈ 2.06 FLOP/B
  const t3 = stageTotals(3);
  assertClose(t3.flops, 32.5, 1e-12); assertClose(t3.bytes, 48, 1e-12);
  assertClose(by(k3, 'fused').flops, 33, 1e-12); assertClose(by(k3, 'fused').bytes, 16, 0);
  const t = stageTotals(2);
  assertClose(t.bytes, 8 + 3 * 4 * 2 / 3 + 12 + 4 * 2 / 3 + 20, 1e-12);
  // per element: dense O(n^{2d}) vs sum factorised O(n^{d+1})
  assertClose(derivativeFlops(3, 2).dense, 2 * 256, 0); assertClose(derivativeFlops(3, 2).sumfact, 2 * 64, 0);
  assertClose(derivativeFlops(3, 3).dense / derivativeFlops(3, 3).sumfact, 16, 0); // n^{d-1} = 4² = 16
  assertClose(roofline(0.25, 1e13, 5e11), 1.25e11, 1);
  assertClose(roofline(100, 1e13, 5e11), 1e13, 1);
});

test('memory transactions: SoA coalesces, AoS and per-element mapping do not', () => {
  const base = { nFields: 4, nItems: 1024, field: 0, warp: 32 };
  // 128-byte segments of f32 = 32 values
  assertClose(transactions(warpAddresses({ ...base, layout: 'soa', stride: 1 }), 32), 1, 0);
  assertClose(transactions(warpAddresses({ ...base, layout: 'aos', stride: 1 }), 32), 4, 0);
  assertClose(transactions(warpAddresses({ ...base, layout: 'soa', stride: 16 }), 32), 16, 0); // thread per element, p = 3
  assertClose(transactions(warpAddresses({ ...base, layout: 'soa', stride: 1, first: 16 }), 32), 2, 0); // misaligned start
});
