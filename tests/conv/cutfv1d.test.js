/**
 * Chapter 11 flow example: explicit upwind finite volumes for u_t + a u_x = 0 on a periodic 1D grid
 * with one small ("cut") cell of width α h — instability at the full-cell CFL number, and the two cures
 * cell merging and state redistribution (lib/core/immersed/cutfv1d.js).
 */
import { test, assert, assertClose, assertRate } from '../harness.js';
import { cutGrid1D, cellAverages, mass, upwindStep, cutFvScheme, cutFvStep, cutFvRun, smallCellFactor } from '../../lib/core/immersed/cutfv1d.js';
import { fitRate } from '../../lib/core/verify/rates.js';

const square = (x) => (x > 0.1 && x < 0.3 ? 1 : 0);
const bump = (x) => Math.exp(-200 * (x - 0.25) ** 2);
const sine = (x) => Math.sin(2 * Math.PI * x);
const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

test('cut grid: N full cells + one cell of width αh fill [0,1]; exact cell averages; α = 1 is the uniform grid', () => {
  const g = cutGrid1D(20, 0.25, 7);
  assert(g.n === 21 && g.small === 7, 'layout');
  assertClose(g.w.reduce((s, v) => s + v, 0), 1, 1e-14);
  assertClose(g.w[7], 0.25 * g.h, 1e-16); assertClose(g.h, 1 / 20.25, 1e-16);
  // averages of sin(2πx): (cos(2πa) − cos(2πb)) / (2π (b − a)); mass of a full period = 0
  const U = cellAverages(g, sine);
  for (let i = 0; i < g.n; i++) assertClose(U[i], (Math.cos(2 * Math.PI * g.x[i]) - Math.cos(2 * Math.PI * g.x[i + 1])) / (2 * Math.PI * g.w[i]), 1e-13);
  assertClose(mass(g, U), 0, 1e-14);
  // shifted averages = averages of the shifted function
  assertClose(maxDiff(cellAverages(g, sine, 0.3), cellAverages(g, (x) => sine(x - 0.3))), 0, 1e-14);
  // α = 1: all three schemes are plain upwind on a uniform grid
  const ref = cutFvRun({ N: 30, alpha: 1, nu: 0.8, method: 'none', steps: 40, u0: bump });
  for (const method of ['merge', 'srd']) assert(maxDiff(ref.U, cutFvRun({ N: 30, alpha: 1, nu: 0.8, method, steps: 40, u0: bump }).U) < 1e-15, method);
});

test('all three schemes conserve mass to round-off', () => {
  for (const method of ['none', 'merge', 'srd']) for (const alpha of [1e-6, 0.01, 0.3, 0.7]) {
    const nu = method === 'none' ? 0.9 * Math.min(1, alpha) : 0.9;
    const r = cutFvRun({ N: 50, alpha, nu, method, steps: 500, u0: bump });
    assert(r.massDrift < 1e-13, `${method} α=${alpha}: mass drift ${r.massDrift}`);
  }
});

test('no stabilisation: monotone iff ν ≤ α, bounded for ν < 2α, blow-up with factor |1 − ν/α| per step for ν > 2α', () => {
  // ν ≤ α: discrete maximum principle
  for (const [alpha, nu] of [[0.3, 0.3], [0.5, 0.45], [0.05, 0.05]]) {
    const r = cutFvRun({ N: 40, alpha, nu, method: 'none', steps: 600, u0: square });
    assert(r.min >= 0 && r.max <= 1 + 1e-13, `monotone α=${alpha} ν=${nu}: [${r.min}, ${r.max}]`);
    assert(smallCellFactor(alpha, nu).monotone, 'flag');
  }
  // α < ν < 2α: one step from a spike in the small cell gives the negative value 1 − ν/α, but the run stays bounded
  {
    const alpha = 0.3, nu = 0.5, g = cutGrid1D(40, alpha), U = new Float64Array(g.n);
    U[g.small] = 1;
    assertClose(upwindStep(g, U, nu)[g.small], 1 - nu / alpha, 1e-14);
    const r = cutFvRun({ N: 40, alpha, nu, method: 'none', steps: 2000, u0: square });
    assert(r.max < 1.5 && r.min > -0.5, `bounded: [${r.min}, ${r.max}]`);
    const f = smallCellFactor(alpha, nu);
    assert(!f.monotone && f.bounded, 'flags');
  }
  // ν > 2α: exponential growth, ratio of successive maxima → |1 − ν/α|
  for (const [alpha, nu] of [[0.1, 0.8], [0.2, 0.9], [0.01, 0.5]]) {
    const g = cutGrid1D(40, alpha), sch = cutFvScheme(g, 'none');
    let U = cellAverages(g, square);
    const m = [];
    for (let k = 0; k < 40; k++) { U = cutFvStep(sch, U, nu); m.push(Math.max(...U.map(Math.abs))); }
    assert(m[39] > 1e6, `blow-up α=${alpha}`);
    assertClose(m[39] / m[38], Math.abs(1 - nu / alpha), 1e-6, 0, `growth factor α=${alpha} ν=${nu}`);
    assert(!smallCellFactor(alpha, nu).bounded, 'flag');
  }
});

test('cell merging = upwind on the grid with the merged cell; monotone for every α and ν ≤ 1', () => {
  // reference: plain upwind on a grid where cells s−1 and s are ONE cell of width (1 + α)h
  const N = 30, alpha = 0.02, nu = 0.9, g = cutGrid1D(N, alpha), s = g.small, sch = cutFvScheme(g, 'merge');
  let U = cellAverages(g, bump);
  // start from merged data (the scheme keeps both members equal from the first step on)
  const m0 = (g.w[s - 1] * U[s - 1] + g.w[s] * U[s]) / (g.w[s - 1] + g.w[s]);
  U[s - 1] = U[s] = m0;
  const wM = [], idx = [];
  for (let i = 0; i < g.n; i++) if (i !== s) { wM.push(i === s - 1 ? g.w[s - 1] + g.w[s] : g.w[i]); idx.push(i); }
  let V = Float64Array.from(idx, (i) => U[i]);
  for (let k = 0; k < 100; k++) {
    U = cutFvStep(sch, U, nu);
    const Vn = new Float64Array(V.length);
    for (let i = 0; i < V.length; i++) Vn[i] = V[i] - (nu * g.h / wM[i]) * (V[i] - V[(i + V.length - 1) % V.length]);
    V = Vn;
  }
  assert(maxDiff(Float64Array.from(idx, (i) => U[i]), V) < 1e-13, 'merge = merged-grid upwind');
  assertClose(U[s], U[s - 1], 1e-15);
  for (const a of [1e-6, 1e-3, 0.1, 0.5, 0.75, 0.999]) for (const v of [0.3, 0.9, 1]) {
    const r = cutFvRun({ N: 40, alpha: a, nu: v, method: 'merge', steps: 1500, u0: square });
    assert(r.min >= -1e-12 && r.max <= 1 + 1e-12, `merge monotone α=${a} ν=${v}: [${r.min}, ${r.max}]`);
  }
});

test('state redistribution: weights of Berger & Giuliani; conservative, bounded for α < ½ and ν ≤ 1, but not monotone', () => {
  // neighbourhoods: the small cell merges with both neighbours, which are then overlapped twice
  const g = cutGrid1D(20, 0.1), s = g.small, sch = cutFvScheme(g, 'srd');
  assert(sch.count[s] === 1 && sch.count[s - 1] === 2 && sch.count[s + 1] === 2 && sch.count[s + 2] === 1, 'overlap counts');
  assert(sch.nbrPtr[s + 1] - sch.nbrPtr[s] === 3, 'M_s = {s−1, s, s+1}');
  // spike in the small cell: U_s^{n+1} = (α − ν/2)/(1 + α) < 0 for ν > 2α (a negative weight ⇒ not monotone)
  for (const [alpha, nu] of [[0.01, 0.8], [0.1, 0.5], [0.3, 0.4]]) {
    const gg = cutGrid1D(20, alpha), U = new Float64Array(gg.n);
    U[gg.small] = 1;
    assertClose(cutFvStep(cutFvScheme(gg, 'srd'), U, nu)[gg.small], (alpha - nu / 2) / (1 + alpha), 1e-14, 0, `SRD weight α=${alpha}`);
  }
  // …but the scheme stays bounded: long runs from a spike and from a square pulse
  for (const alpha of [1e-6, 1e-3, 0.03, 0.2, 0.49]) for (const nu of [0.2, 0.6, 0.9, 1]) {
    const gg = cutGrid1D(40, alpha), sc = cutFvScheme(gg, 'srd');
    let U = new Float64Array(gg.n), amp = 0;
    U[gg.small] = 1;
    for (let k = 0; k < 1500; k++) { U = cutFvStep(sc, U, nu); for (const v of U) amp = Math.max(amp, Math.abs(v)); }
    assert(amp <= 1, `SRD spike bounded α=${alpha} ν=${nu}: ${amp}`);
    const r = cutFvRun({ N: 40, alpha, nu, method: 'srd', steps: 1500, u0: square });
    assert(r.min > -1e-9 && r.max < 1 + 1e-9 && r.massDrift < 1e-13, `SRD square α=${alpha} ν=${nu}: [${r.min}, ${r.max}]`);
  }
  // cells with α ≥ ½ are left alone (threshold of the paper): identical to the base scheme
  const a = cutFvRun({ N: 30, alpha: 0.6, nu: 0.7, method: 'srd', steps: 50, u0: bump }), b = cutFvRun({ N: 30, alpha: 0.6, nu: 0.7, method: 'none', steps: 50, u0: bump });
  assert(maxDiff(a.U, b.U) < 1e-14, 'α ≥ ½ untouched');
});

test('stabilised schemes are first-order accurate at the full-cell CFL number (ν = 0.8, α = 10⁻³ and 0.3)', () => {
  for (const method of ['merge', 'srd']) for (const alpha of [1e-3, 0.3]) {
    const hs = [], e = [];
    for (const N of [160, 320, 640]) {
      const nu = 0.8, steps = Math.round(0.5 * (N + alpha) / nu);
      const r = cutFvRun({ N, alpha, nu, method, steps, u0: sine });
      hs.push(r.grid.h); e.push(r.l1);
    }
    assertRate(fitRate(hs, e), 1, `${method} α=${alpha} L1`, 0.1, 0.1);
  }
});
