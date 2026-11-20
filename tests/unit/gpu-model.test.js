/**
 * Tests for the chapter-12 helper models: race-condition simulation, element
 * colouring, face-buffer gather (lib/gpu/races.js) and the cost / memory-transaction
 * model (lib/gpu/cost.js).
 */
import { test, assert, assertClose } from '../harness.js';
import { lockstepScatter, colorElements, isValidColoring, gridFaces, faceBufferGather } from '../../lib/gpu/races.js';
import { kernelCosts, stageTotals, derivativeFlops, roofline, ridgePoint, transactions, warpAddresses, simtStats, liftProgram } from '../../lib/gpu/cost.js';
import { makeKernelSources, PARAMS_LAYOUT, PARAMS_N_U32, PARAMS_BYTES, packParams, f32Literal, WORKGROUP_SIZE } from '../../lib/gpu/kernels.js';
import { createGpuSolver, dispatchGroups, maxAbsDiff } from '../../lib/gpu/dgadvgpu.js';
import { makeDGAdv2D } from '../../lib/gpu/cputwin.js';
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

test('roofline ridge point quoted in chapter 12: 30 TFLOP/s, 800 GB/s → I* = 37.5 FLOP/byte; every kernel is far below it', () => {
  assertClose(ridgePoint(30e12, 800e9), 37.5, 1e-12);
  for (let p = 1; p <= 8; p++) for (const k of kernelCosts(p)) assert(k.ai < ridgePoint(30e12, 800e9) / 5, `${k.name} p=${p}`);
  assertClose(roofline(ridgePoint(30e12, 800e9), 30e12, 800e9), 30e12, 1); // the two roofs meet there
});

test('cost model: finite and positive over the whole range of the widgets (p = 1..8, f32/f64)', () => {
  for (let p = 1; p <= 8; p++) for (const b of [4, 8]) {
    for (const k of kernelCosts(p, { bytesPerValue: b })) assert(k.flops > 0 && k.bytes > 0 && Number.isFinite(k.ai) && k.ai > 0, `${k.name} p=${p}`);
    const t = stageTotals(p, { bytesPerValue: b }), f = kernelCosts(p, { bytesPerValue: b }).find((k) => k.name === 'fused');
    assert(f.bytes < t.bytes && f.flops > t.flops, 'fusion: fewer bytes, slightly more FLOPs');
  }
  // layout widget: every combination of its controls gives between 1 and W transactions
  for (const layout of ['aos', 'soa']) for (let nFields = 1; nFields <= 4; nFields++) for (const stride of [1, 4, 9, 16]) for (let first = 0; first <= 8; first++) {
    const nItems = stride === 1 ? 32 : Math.ceil((first + 7 * stride + 1) / 8) * 8;
    const a = warpAddresses({ layout, nFields, nItems, field: 0, warp: 8, stride, first }), tx = transactions(a, 8);
    assert(tx >= 1 && tx <= 8 && a.every((x) => Number.isInteger(x) && x >= 0 && x < nItems * nFields), `${layout} ${nFields} ${stride} ${first}: ${tx}`);
  }
});

test('SIMT model: lane-slot efficiency of the volume and lift kernels (numbers quoted in chapter 12)', () => {
  const all = () => true;
  // no branches, full warps: 100 %
  assertClose(simtStats([{ cost: 3, act: all }], 64, 8).efficiency, 1, 0);
  // partially filled last warp: 9 threads in warps of 8 → 2 warps issued, 9 useful of 16 lanes
  const s9 = simtStats([{ cost: 1, act: all }], 9, 8);
  assertClose(s9.nWorkgroups, 1, 0); assertClose(s9.launched, 16, 0); assertClose(s9.nWarps, 2, 0); assertClose(s9.efficiency, 9 / 16, 1e-15);
  // one thread per element, 9 elements, W = 32: the second warp of the workgroup is empty and issues nothing
  assertClose(simtStats([{ cost: 1, act: all }], 9, 32).efficiency, 9 / 32, 1e-15);
  // a branch taken by half of every warp: the line is issued for all lanes, half of them idle
  assertClose(simtStats([{ cost: 1, act: (t) => t % 2 === 0 }], 64, 8).efficiency, 0.5, 1e-15);
  // a branch on which whole warps agree costs nothing (warps 0, 2, … take it; the others skip it)
  assertClose(simtStats([{ cost: 1, act: (t) => Math.floor(t / 8) % 2 === 0 }], 64, 8).efficiency, 1, 0);
  // the lift kernel on the 3×3 mesh of the widget, toy warps W = 8 ("Try this" box): 64 %, 58 %, 67 %
  const lift = (p, W, nElem = 9) => simtStats(liftProgram(p), nElem * (p + 1) ** 2, W).efficiency;
  assertClose(lift(1, 8), 9 / 14, 1e-12);        // 64.3 %: every DOF takes 2 of the 4 branches, the last warp is half empty
  assertClose(lift(2, 8), 0.585, 5e-4);
  assertClose(lift(3, 8), 2 / 3, 1e-12);         // 66.7 %: a warp = two rows of an element; rows 1–2 skip the top/bottom ifs
  assert(lift(1, 8) < lift(3, 8), 'p = 1 is NOT divergence-free: it is slightly worse than p = 3 for W = 8');
  // real warps on a large mesh: efficiency drops with p (fewer boundary nodes per warp)
  assertClose(lift(1, 32, 4096), 5 / 7, 1e-12); assertClose(lift(3, 32, 4096), 4 / 7, 1e-12);
  // boundary-node counts used in the prose: p = 3 → 12 of 16 DOFs have face work, p = 1 → all 4
  for (const [p, nb] of [[1, 4], [2, 8], [3, 12]]) {
    const prog = liftProgram(p).filter((l) => l.face);
    let c = 0; for (let g = 0; g < (p + 1) ** 2; g++) if (prog.some((l) => l.act(g))) c++;
    assertClose(c, nb, 0);
  }
});

test('GPU host code: Params uniform layout — WGSL struct, PARAMS_LAYOUT and packParams agree', () => {
  assertClose(PARAMS_BYTES, 48, 0); assert(PARAMS_BYTES % 16 === 0, 'uniform struct size must be a multiple of 16');
  const d = makeDGAdv2D({ N: 2, p: 2 }), src = makeKernelSources(2, d.D);
  for (const k of ['volume', 'surface', 'lift', 'rkStage']) {
    const body = /struct Params \{([^}]*)\}/.exec(src[k])[1];
    const fields = [...body.matchAll(/^\s*(\w+)\s*:\s*(u32|f32),/gm)].map((m) => [m[1], m[2]]);
    assertClose(fields.length, PARAMS_LAYOUT.length, 0, 0, k);
    fields.forEach(([name, type], i) => { assert(name === PARAMS_LAYOUT[i], `${k}: field ${i} is ${name}`); assert(type === (i < PARAMS_N_U32 ? 'u32' : 'f32'), `${k}: type of ${name}`); });
    assert(src[k].includes(`@workgroup_size(WG)`) && src[k].includes(`const WG  : u32 = ${WORKGROUP_SIZE}u;`), 'workgroup size');
  }
  const ab = packParams({ N: 7, nElem: 49, nDof: 441, nFace: 294, ax: 1, ay: -0.5, rx: 14, ry: 14, lift: 3, dt: 0.01 });
  const U = new Uint32Array(ab), F = new Float32Array(ab);
  assert(U[0] === 7 && U[1] === 49 && U[2] === 441 && U[3] === 294, 'u32 fields');
  assert(F[4] === 1 && F[5] === -0.5 && F[6] === 14 && F[7] === 14 && F[8] === 3 && F[9] === Math.fround(0.01) && F[10] === 0 && F[11] === 0, 'f32 fields');
  // f32 literals: always a float literal, and they round-trip the f32 value
  for (const v of [0, 1, -0.5, 1 / 3, -2, 1e-7, 123456.789, d.D[0]]) {
    const lit = f32Literal(v);
    assert(/[.eE]/.test(lit), `"${lit}" would be an integer literal`);
    assert(Math.fround(parseFloat(lit)) === Math.fround(v), `${lit} does not round-trip ${v}`);
  }
  // the differentiation matrix in the shader is the row-major D of the CPU twin
  const lits = /array<f32, 9>\(([^)]*)\)/.exec(src.volume)[1].split(',').map((x) => Math.fround(parseFloat(x)));
  for (let i = 0; i < 9; i++) assert(lits[i] === Math.fround(d.D[i]), `D[${i}]`);
});

/** A fake GPUDevice that records what the host code does (enough of the API for createGpuSolver). */
function mockDevice({ failCompile = false, maxGroups = 65535 } = {}) {
  const log = { created: 0, destroyed: 0, dispatches: [], submits: 0, writes: [] };
  if (typeof globalThis.GPUBufferUsage === 'undefined') globalThis.GPUBufferUsage = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
  if (typeof globalThis.GPUMapMode === 'undefined') globalThis.GPUMapMode = { READ: 1 };
  const pass = { pipe: null, setPipeline(p) { this.pipe = p; }, setBindGroup() {}, dispatchWorkgroups(n) { log.dispatches.push([this.pipe.label, n]); }, end() {} };
  const device = {
    limits: { maxComputeWorkgroupsPerDimension: maxGroups },
    createBuffer: (d) => { log.created++; return { label: d.label, size: d.size, destroy() { log.destroyed++; } }; },
    queue: { writeBuffer: (b, off, data) => log.writes.push([b.label, data.byteLength]), submit: () => { log.submits++; }, onSubmittedWorkDone: () => Promise.resolve() },
    createShaderModule: () => ({ getCompilationInfo: () => Promise.resolve({ messages: failCompile ? [{ type: 'error', lineNum: 3, message: 'boom' }] : [] }) }),
    createComputePipelineAsync: (d) => Promise.resolve({ label: d.label, getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({ beginComputePass: () => pass, finish: () => ({}) }),
  };
  return { device, log };
}

test('GPU host code: one LSRK4 step = 5 stages × (volume, surface, lift, rkStage) = 20 dispatches with ⌈items/64⌉ workgroups', async () => {
  const d = makeDGAdv2D({ N: 13, p: 2 }), { device, log } = mockDevice();
  const solver = await createGpuSolver(device, d, d.interpolate((x, y) => x + y), { dt: d.stableDt() });
  assertClose(log.created, 8, 0, 0, 'u0, u1, rhs, res, flux, params, stage, staging');
  await solver.run(70); // more than one command buffer (64 steps each)
  assertClose(log.dispatches.length, 70 * 20, 0); assertClose(log.submits, 2, 0);
  assertClose(solver.steps, 70, 0); assertClose(solver.t, 70 * d.stableDt(), 1e-12);
  const gDof = Math.ceil(d.nDof / 64), gFace = Math.ceil(d.nFace / 64);
  assert(gDof * 64 > d.nDof, 'this mesh has a partially filled last workgroup');
  for (let k = 0; k < log.dispatches.length; k++) {
    const [name, n] = log.dispatches[k], want = ['volume', 'surface', 'lift', 'rkStage'][k % 4];
    assert(name === want && n === (want === 'surface' ? gFace : gDof), `dispatch ${k}: ${name} × ${n}`);
  }
  solver.destroy();
  assertClose(log.destroyed, 8, 0, 0, 'destroy() frees every buffer');
});

test('GPU host code: a failing shader compilation or an over-size mesh leaks no buffers and names the offending count', async () => {
  const d = makeDGAdv2D({ N: 4, p: 3 });
  const { device, log } = mockDevice({ failCompile: true });
  let msg = '';
  try { await createGpuSolver(device, d, d.interpolate(() => 1), { dt: 1e-3 }); } catch (e) { msg = e.message; }
  assert(/WGSL .*line 3: boom/.test(msg), `expected the compile error, got "${msg}"`);
  assert(log.created === 8 && log.destroyed === 8, `created ${log.created} buffers, destroyed ${log.destroyed}`);
  // dispatch limit: the message reports the count that actually exceeds the limit (the DOFs, not the face nodes)
  assertClose(dispatchGroups(1048576, 65535, 'x'), 16384, 0); // the largest mesh of the widget (N = 256, p = 3) fits
  const small = mockDevice({ maxGroups: 3 });
  msg = '';
  try { await createGpuSolver(small.device, d, d.interpolate(() => 1), { dt: 1e-3 }); } catch (e) { msg = e.message; }
  assert(msg.includes(`${d.nDof} DOFs needs ${Math.ceil(d.nDof / 64)} workgroups`) && msg.includes('allows 3'), msg);
  assertClose(small.log.created, 0, 0, 0, 'limit is checked before anything is allocated');
});

test('maxAbsDiff: maximum difference, NaN is not hidden', () => {
  assertClose(maxAbsDiff([1, 2, 3], [1, 2.5, 2]), 1, 0);
  assert(Number.isNaN(maxAbsDiff([1, NaN], [1, 1])), 'NaN must propagate');
});
