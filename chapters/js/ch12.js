/**
 * @file Chapter 12 — Parallelisation on GPUs.
 *
 * Widgets:
 *  - w-threads  : thread → DOF / element mapping on a 3×3 mesh, warps, lockstep, divergence (SIMT efficiency)
 *  - w-roofline : roofline model with the DG kernels' arithmetic intensities (lib/gpu/cost.js) + measured point
 *  - w-layout   : AoS vs SoA vs per-element mapping: addresses read by one warp, memory transactions
 *  - w-race     : scatter-add race (wrong result), colouring (4 launches), DG face buffer + gather
 *  - w-live     : live WebGPU DG-SEM advection (lib/gpu/dgadvgpu.js) vs the f64 CPU twin, with CPU fallback
 * Static content filled by script: the generated WGSL volume kernel (#wgsl-volume) and the
 * sum-factorisation cost table (#sumfact-table).
 */
import { initChapter, mount, selfCheck, reportError } from '../../lib/ui/chapter.js';
import { widgetLayout, slider, select, segmented, button, buttonRow, readout, fmt, h, playControls } from '../../lib/ui/controls.js';
import { createCanvas, View2D, theme, seriesColors } from '../../lib/viz/canvas.js';
import { drawQuadField } from '../../lib/viz/field2d.js';
import { colorbar } from '../../lib/viz/colormap.js';
import { Plot } from '../../lib/viz/plot1d.js';
import { Animator } from '../../lib/viz/anim.js';
import { gaussLobatto } from '../../lib/core/quad/gauss1d.js';
import { baryWeights } from '../../lib/core/basis/lagrange.js';
import { quadGrid } from '../../lib/core/mesh/structured.js';
import { makeDGAdv2D, makeCpuSolver, INITIAL } from '../../lib/gpu/cputwin.js';
import { makeKernelSources } from '../../lib/gpu/kernels.js';
import { kernelCosts, stageTotals, derivativeFlops, roofline, ridgePoint, transactions, warpAddresses, simtStats, liftProgram } from '../../lib/gpu/cost.js';
import { lockstepScatter, colorElements, isValidColoring, gridFaces, faceBufferGather } from '../../lib/gpu/races.js';
import { getGPU } from '../../lib/gpu/device.js';
import { createGpuSolver, maxAbsDiff } from '../../lib/gpu/dgadvgpu.js';

/** Categorical colours for workgroups / colour classes (readable in light and dark mode). */
const CAT = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#b07aa1', '#76b7b2', '#edc948', '#ff9da7', '#9c755f'];

/** Mix a #rrggbb colour with white (f > 0) or black (f < 0). */
function shade(hex, f) {
  const c = [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16));
  const t = f > 0 ? 255 : 0, a = Math.abs(f);
  return `rgb(${c.map((v) => Math.round(v * (1 - a) + t * a)).join(',')})`;
}

/** Parse '#rrggbb' or 'rgb(r,g,b)' into [r, g, b] (0..255). */
function rgbOf(c) {
  if (c[0] === '#') return [1, 3, 5].map((k) => parseInt(c.slice(k, k + 2), 16));
  return c.match(/\d+(\.\d+)?/g).slice(0, 3).map(Number);
}

/**
 * Black or white, whichever has the higher WCAG contrast on the background colour `c`
 * (optionally `c` drawn with opacity `alpha` over `under`).
 */
function textOn(c, alpha = 1, under = '#000000') {
  const a = rgbOf(c), b = rgbOf(under);
  const L = a.map((v, k) => { const x = (alpha * v + (1 - alpha) * b[k]) / 255; return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4; });
  return 0.2126 * L[0] + 0.7152 * L[1] + 0.0722 * L[2] > 0.179 ? '#111' : '#fff';
}

/* ------------------------------------------------------------------ */
/* W1: thread mapping, warps, divergence                                */
/* ------------------------------------------------------------------ */
/**
 * Kernel "programs" as lists of lines; each line has an issue cost (instruction slots)
 * and a predicate telling which threads execute it.
 */
function kernelProgram(kernel, mapping, p) {
  const n = p + 1, Np = n * n;
  const all = () => true;
  if (mapping === 'node') {
    if (kernel === 'volume') {
      return [
        { text: 'g = gid.x   // global_invocation_id', cost: 1, act: all },
        { text: 'if (g >= nDof) { return; }', cost: 1, act: all },
        { text: 'e = g / NP;  i = (g % NP) % NP1;  j = …', cost: 1, act: all },
        { text: `for k in 0..${p}: dx += D[i,k]·u[k,j]; dy += …`, cost: n, act: all },
        { text: 'rhs[g] = −(ax·rx·dx + ay·ry·dy)', cost: 1, act: all },
      ];
    }
    // the lift kernel: costs and predicates come from the tested model (lib/gpu/cost.js), the text is display only
    const text = ['g = gid.x; if (g >= nDof) { return; }', 'uc = u[g];  r = rhs[g]', `if (i == ${p}) { r −= c·(F_right − ax·uc) }`,
      'if (i == 0) { r += c·(F_left − ax·uc) }', `if (j == ${p}) { r −= c·(F_top − ay·uc) }`, 'if (j == 0) { r += c·(F_bottom − ay·uc) }', 'rhs[g] = r'];
    return liftProgram(p).map((ln, k) => ({ ...ln, text: text[k] }));
  }
  if (kernel === 'volume') {
    return [
      { text: 'e = gid.x; if (e >= nElem) { return; }', cost: 1, act: all },
      { text: `for l in 0..${Np - 1}:  // my DOFs, serially`, cost: Np, act: all, serial: Np },
      { text: `  for k in 0..${p}: dx += …; dy += …`, cost: Np * n, act: all, serial: Np },
    ];
  }
  return [
    { text: 'e = gid.x; if (e >= nElem) { return; }', cost: 1, act: all },
    { text: `for 4 faces, for m in 0..${p}:  // serially`, cost: 4 * n, act: all, serial: 4 * n },
    { text: '  r[node] += ±c·(flux − a·n·u)', cost: 4 * n, act: all, serial: 4 * n },
  ];
}

function threadsWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('3×3 mesh: DOFs coloured by workgroup (two shades = two warps)');
  p1.style.flex = '1 1 260px';
  const s1 = createCanvas(p1, { aspect: 1 });
  const p2 = L.panel('kernel source (current line) and lanes of every warp');
  p2.style.flex = '1.3 1 320px';
  const s2 = createCanvas(p2, { aspect: 0.8 });
  const st = { p: 2, mapping: 'node', kernel: 'lift', W: 8, line: 0, sub: 0, last: 0 };
  segmented(L.controls, { label: 'one thread per', value: st.mapping, options: [{ value: 'node', label: 'DOF' }, { value: 'element', label: 'element' }], onChange: (v) => { st.mapping = v; st.line = 0; st.sub = 0; draw(); } });
  segmented(L.controls, { label: 'kernel', value: st.kernel, options: [{ value: 'volume', label: 'volume' }, { value: 'lift', label: 'lift (branches)' }], onChange: (v) => { st.kernel = v; st.line = 0; st.sub = 0; draw(); } });
  slider(L.controls, { label: 'degree $p$', min: 1, max: 3, step: 1, value: st.p, onInput: (v) => { st.p = v; st.line = 0; draw(); } });
  segmented(L.controls, { label: 'warp width $W$ (workgroup = 2W)', value: String(st.W), options: [{ value: '8', label: '8 (toy)' }, { value: '32', label: '32 (real)' }], onChange: (v) => { st.W = +v; draw(); } });
  const anim = new Animator(fig, {
    step: () => {
      const now = performance.now();
      if (now - st.last < 650) return;
      st.last = now;
      const prog = kernelProgram(st.kernel, st.mapping, st.p), ln = prog[st.line];
      if (ln.serial && st.sub < Math.min(ln.serial, 6) - 1) { st.sub++; return; } // show a few serial iterations
      st.sub = 0; st.line = (st.line + 1) % prog.length;
    },
    draw: () => draw(),
  });
  playControls(L.controls, anim, () => { st.line = 0; st.sub = 0; draw(); });
  const out = readout(L.controls);

  function stats() {
    const n = st.p + 1, Np = n * n, nE = 9, nDof = nE * Np;
    const T = st.mapping === 'node' ? nDof : nE, WG = 2 * st.W;
    const prog = kernelProgram(st.kernel, st.mapping, st.p);
    const S = simtStats(prog, T, st.W, WG); // issued vs useful lane-slots (lib/gpu/cost.js, unit-tested)
    const nWG = S.nWorkgroups, nWarps = S.nWarps, slotsPerWarp = S.slotsPerWarp;
    const bnd = []; // boundary nodes in face order (right, left, top, bottom) — visited by the per-element lift
    for (let j = 0; j < n; j++) bnd.push(j * n + n - 1);
    for (let j = 0; j < n; j++) bnd.push(j * n);
    for (let i = 0; i < n; i++) bnd.push((n - 1) * n + i);
    for (let i = 0; i < n; i++) bnd.push(i);
    return { n, Np, nDof, T, WG, nWG, nWarps, prog, eff: S.efficiency, slotsPerWarp, bnd };
  }

  function draw() {
    const S = stats(), Tm = theme(), ctx = s1.ctx;
    const ln = S.prog[st.line];
    // ---- mesh panel
    ctx.fillStyle = Tm.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    const view = new View2D(s1, [0, 3, 0, 3], { equal: true, pad: 10 });
    const xi = gaussLobatto(S.n).x, inset = 0.2;
    ctx.strokeStyle = Tm.faint; ctx.lineWidth = 1;
    for (let k = 0; k <= 3; k++) {
      ctx.beginPath(); ctx.moveTo(view.X(k), view.Y(0)); ctx.lineTo(view.X(k), view.Y(3)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(view.X(0), view.Y(k)); ctx.lineTo(view.X(3), view.Y(k)); ctx.stroke();
    }
    const thrColor = (t) => { const wg = Math.floor(t / S.WG), warp = Math.floor(t / st.W) % 2; return shade(CAT[wg % CAT.length], warp ? -0.25 : 0.15); };
    const rad = Math.max(3, Math.min(11, view.sx * 0.32 / S.n));
    ctx.font = `${Math.max(8, Math.min(11, rad * 1.1))}px ${Tm.mono}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let e = 0; e < 9; e++) {
      const ex = e % 3, ey = Math.floor(e / 3);
      for (let l = 0; l < S.Np; l++) {
        const i = l % S.n, j = Math.floor(l / S.n), g = e * S.Np + l;
        const x = ex + inset + (1 - 2 * inset) * (xi[i] + 1) / 2, y = ey + inset + (1 - 2 * inset) * (xi[j] + 1) / 2;
        let col, active;
        if (st.mapping === 'node') { col = thrColor(g); active = ln.act(g); }
        else { // one thread per element: it visits its nodes one after the other
          const cur = st.kernel === 'volume' ? st.sub % S.Np : S.bnd[st.sub % S.bnd.length];
          active = !!ln.serial && l === cur;
          col = active ? thrColor(e) : Tm.faint;
        }
        ctx.beginPath(); ctx.arc(view.X(x), view.Y(y), rad, 0, 2 * Math.PI);
        ctx.fillStyle = active ? col : Tm.rule; ctx.fill();
        if (!active && st.mapping === 'node') { ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke(); }
        if (S.n <= 3 && st.mapping === 'node') { ctx.fillStyle = active ? textOn(col) : Tm.soft; ctx.fillText(String(g), view.X(x), view.Y(y)); }
      }
      if (st.mapping === 'element') {
        ctx.fillStyle = thrColor(e); ctx.globalAlpha = 0.18;
        ctx.fillRect(view.X(ex) + 2, view.Y(ey + 1) + 2, view.sx - 4, view.sy - 4); ctx.globalAlpha = 1;
        ctx.fillStyle = Tm.ink; ctx.font = `600 11px ${Tm.ui}`; ctx.fillText(`thread ${e}`, view.X(ex + 0.5), view.Y(ey + 0.09));
        ctx.font = `${Math.max(8, Math.min(11, rad * 1.1))}px ${Tm.mono}`;
      }
    }
    // ---- code + lanes panel
    const c2 = s2.ctx, W2 = s2.w, H2 = s2.h;
    c2.fillStyle = Tm.bg; c2.fillRect(0, 0, W2, H2);
    const lh = 17, fs = Math.max(9, Math.min(12, W2 / 40));
    c2.font = `${fs}px ${Tm.mono}`; c2.textAlign = 'left'; c2.textBaseline = 'middle';
    S.prog.forEach((L2, k) => {
      const y = 12 + k * lh;
      if (k === st.line) { c2.fillStyle = Tm.accent; c2.globalAlpha = 0.18; c2.fillRect(2, y - lh / 2, W2 - 4, lh); c2.globalAlpha = 1; }
      c2.fillStyle = k === st.line ? Tm.ink : Tm.soft;
      c2.fillText((k === st.line ? '▶ ' : '  ') + L2.text, 6, y);
    });
    const top = 12 + S.prog.length * lh + 8;
    c2.font = `600 11.5px ${Tm.ui}`; c2.fillStyle = Tm.soft;
    c2.fillText(`lanes for the highlighted line (${S.nWarps} warp${S.nWarps > 1 ? 's' : ''} × ${st.W} lanes)`, 6, top);
    const gx0 = 52, gy0 = top + 12, cw = Math.min(26, (W2 - gx0 - 8) / st.W), chh = Math.min(22, Math.max(6, (H2 - gy0 - 8) / S.nWarps));
    for (let w = 0; w < S.nWarps; w++) {
      const y = gy0 + w * chh;
      c2.fillStyle = Tm.soft; c2.font = `${Math.min(11, chh * 0.8)}px ${Tm.ui}`; c2.textAlign = 'right';
      c2.fillText(`warp ${w}`, gx0 - 6, y + chh / 2);
      for (let l = 0; l < st.W; l++) {
        const t = w * st.W + l, x = gx0 + l * cw;
        if (t >= S.T) { c2.strokeStyle = Tm.rule; c2.lineWidth = 1; c2.strokeRect(x + 1.5, y + 1.5, cw - 3, chh - 3); continue; }
        if (ln.act(t)) { c2.fillStyle = thrColor(t); c2.fillRect(x + 1, y + 1, cw - 2, chh - 2); }
        else {
          c2.fillStyle = Tm.rule; c2.fillRect(x + 1, y + 1, cw - 2, chh - 2);
          c2.strokeStyle = Tm.faint; c2.lineWidth = 1; c2.beginPath();
          c2.moveTo(x + 4, y + 4); c2.lineTo(x + cw - 4, y + chh - 4); c2.moveTo(x + cw - 4, y + 4); c2.lineTo(x + 4, y + chh - 4); c2.stroke();
        }
      }
    }
    c2.textAlign = 'left';
    out.set(`threads used     ${S.T}  (launched ${S.nWG * S.WG})\nworkgroups       ${S.nWG} × ${S.WG}\nwarps            ${S.nWarps}\n`
      + `issue slots/warp ${S.slotsPerWarp}\n<b>SIMT efficiency  ${(100 * S.eff).toFixed(1)} %</b>`);
    selfCheck('threads: SIMT efficiency in (0,1]', S.eff > 0 && S.eff <= 1);
  }
  s1.onResize(draw); s2.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W2: roofline                                                         */
/* ------------------------------------------------------------------ */
const DEVICES = {
  igpu: { label: 'laptop / integrated GPU (generic)', peak: 4, bw: 100 },
  dgpu: { label: 'desktop GPU (generic)', peak: 30, bw: 800 },
  hpc: { label: 'data-centre GPU (generic)', peak: 60, bw: 3000 },
  cpu: { label: 'CPU socket (generic)', peak: 2, bw: 200 },
};
let measured = null; // {dofPerSec, p, info} published by the live widget

function rooflineWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('attainable performance $\\min(P_\\text{peak}, B\\cdot I)$, log–log');
  p1.style.flex = '1 1 420px';
  const s1 = createCanvas(p1, { aspect: 0.66 });
  const st = { dev: 'dgpu', peak: 30, bw: 800, p: 3 };
  const sl = {}; // sliders, filled below (the preset select must come first in the UI)
  select(L.controls, { label: 'device preset', value: st.dev, options: Object.entries(DEVICES).map(([k, v]) => ({ value: k, label: v.label })),
    onChange: (v) => { st.dev = v; st.peak = DEVICES[v].peak; st.bw = DEVICES[v].bw; sl.peak.set(st.peak); sl.bw.set(st.bw); draw(); } });
  sl.peak = slider(L.controls, { label: 'peak f32 rate [TFLOP/s]', min: 0.5, max: 100, log: true, step: 0.001, value: st.peak, format: (v) => fmt(v, 2), onInput: (v) => { st.peak = v; draw(); } });
  sl.bw = slider(L.controls, { label: 'memory bandwidth [GB/s]', min: 20, max: 5000, log: true, step: 0.001, value: st.bw, format: (v) => fmt(v, 2), onInput: (v) => { st.bw = v; draw(); } });
  slider(L.controls, { label: 'degree $p$', min: 1, max: 8, step: 1, value: st.p, onInput: (v) => { st.p = v; draw(); } });
  const out = readout(L.controls);
  globalThis.addEventListener('ch12-measured', () => draw());

  function draw() {
    const Tm = theme(), C = seriesColors();
    const P = st.peak * 1e12, B = st.bw * 1e9, G = 1e9;
    const Pl = new Plot(s1, { xlim: [0.03, 300], ylim: [1, 2e5], xlog: true, ylog: true, xlabel: 'arithmetic intensity I [FLOP/byte]', ylabel: 'GFLOP/s' });
    Pl.frame();
    const xs = [], ys = [];
    for (let k = 0; k <= 200; k++) { const I = 0.03 * (1e4) ** (k / 200); xs.push(I); ys.push(roofline(I, P, B) / G); }
    Pl.line(xs, ys, { color: Tm.ink, width: 2.6 });
    const ridge = ridgePoint(P, B);
    Pl.vline(ridge, { color: Tm.faint });
    // labels are placed on the side with room, so nothing is clipped on narrow (phone) canvases
    const rightRoom = Pl.px1 - Pl.X(ridge) > 110;
    Pl.text(ridge, 1.6, rightRoom ? ` ridge I* = ${fmt(ridge)}` : `ridge I* = ${fmt(ridge)} `, { color: Tm.soft, font: `11px ${Tm.ui}`, align: rightRoom ? 'left' : 'right' });
    Pl.text(0.035, 3e4, 'memory-bound: P = B·I', { color: Tm.soft, font: `11px ${Tm.ui}`, baseline: 'top' });
    if (ridge < 300) Pl.text(300, Math.min(P / G, 1.9e5), 'compute-bound ', { color: Tm.soft, font: `11px ${Tm.ui}`, align: 'right', baseline: P / G > 6e4 ? 'top' : 'bottom', dy: P / G > 6e4 ? 4 : -4 });
    const names = ['volume', 'surface', 'lift', 'rkStage', 'fused'];
    const col = { volume: C[0], surface: C[1], lift: C[2], rkStage: C[3], fused: C[4] };
    // trails p = 1..8
    for (const nm of names) {
      const tx = [], ty = [];
      for (let q = 1; q <= 8; q++) { const k = kernelCosts(q).find((x) => x.name === nm); tx.push(k.ai); ty.push(roofline(k.ai, P, B) / G); }
      Pl.line(tx, ty, { color: col[nm], width: 5, alpha: 0.22 });
    }
    const ks = kernelCosts(st.p);
    for (const k of ks) {
      Pl.points([k.ai], [roofline(k.ai, P, B) / G], { color: col[k.name], r: 5.5 });
      const below = k.name === 'rkStage' || k.name === 'surface' || k.name === 'fused';
      Pl.text(k.ai, roofline(k.ai, P, B) / G, k.name, { color: col[k.name], dx: below ? 8 : -8, dy: below ? 10 : -10, align: below ? 'left' : 'right', font: `600 11px ${Tm.ui}` });
    }
    // model rates for one RK step (5 stages) — DOF-updates per second
    const tStage = ks.filter((k) => k.name !== 'fused').reduce((s, k) => s + Math.max(k.flops / P, k.bytes / B), 0);
    const fz = ks.find((k) => k.name === 'fused'), tFused = Math.max(fz.flops / P, fz.bytes / B);
    // (no legend box: every dot carries its name, and the read-out lists the kernels — a legend would hide dots on small screens)
    let mtxt = '';
    if (measured) {
      const tot = stageTotals(measured.p), gfl = measured.dofPerSec * 5 * tot.flops / G, I = tot.flops / tot.bytes;
      Pl.text(I, gfl, '★', { color: '#e0457b', align: 'center', font: `22px ${Tm.ui}` });
      Pl.text(I, gfl, `measured (p=${measured.p})`, { color: '#e0457b', align: 'left', dx: 12, font: `600 11px ${Tm.ui}` });
      mtxt = `\nmeasured (your GPU, p=${measured.p}):\n  ${fmt(measured.dofPerSec / 1e9)} G DOF-updates/s\n  ≈ ${fmt(gfl)} GFLOP/s`;
    }
    const rows = ks.map((k) => `${k.name.padEnd(8)}${fmt(k.flops).padStart(6)}${fmt(k.bytes).padStart(6)}${fmt(k.ai).padStart(7)}`).join('\n');
    out.set(`per DOF & stage (f32):\nkernel   FLOP  byte  FLOP/B\n${rows}\n\nmodel DOF-updates/s:\n  4 kernels ${fmt(1 / (5 * tStage) / 1e9)} G\n  fused     ${fmt(1 / (5 * tFused) / 1e9)} G${mtxt}`);
    selfCheck('roofline: all DG kernels memory-bound on the generic desktop GPU', ks.every((k) => k.ai < ridgePoint(30e12, 800e9)));
  }
  s1.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W3: memory layout and coalescing                                     */
/* ------------------------------------------------------------------ */
const FIELD_NAMES = ['ρ', 'ρu', 'ρv', 'E'];
function layoutWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('global memory: one cell = one f32 value (number = item), boxes = aligned 32-byte segments');
  p1.style.flex = '1 1 440px';
  const s1 = createCanvas(p1, { aspect: 0.38 });
  const st = { layout: 'aos', nFields: 4, map: 'node', p: 2, first: 0 };
  segmented(L.controls, { label: 'layout', value: st.layout, options: [{ value: 'aos', label: 'AoS' }, { value: 'soa', label: 'SoA' }], onChange: (v) => { st.layout = v; draw(); } });
  slider(L.controls, { label: 'fields per node', min: 1, max: 4, step: 1, value: st.nFields, onInput: (v) => { st.nFields = v; draw(); } });
  segmented(L.controls, { label: 'thread t reads', value: st.map, options: [{ value: 'node', label: 'node t' }, { value: 'element', label: 'node 0 of element t' }], onChange: (v) => { st.map = v; draw(); } });
  slider(L.controls, { label: 'degree $p$ (element size $N_p$)', min: 1, max: 3, step: 1, value: st.p, onInput: (v) => { st.p = v; draw(); } });
  slider(L.controls, { label: 'first item of the warp (alignment)', min: 0, max: 8, step: 1, value: st.first, onInput: (v) => { st.first = v; draw(); } });
  const out = readout(L.controls);
  function draw() {
    const Tm = theme(), ctx = s1.ctx, W = 8, seg = 8;
    const Np = (st.p + 1) ** 2, stride = st.map === 'node' ? 1 : Np;
    const nItems = st.map === 'node' ? 32 : Math.ceil((st.first + (W - 1) * stride + 1) / 8) * 8;
    const o = { layout: st.layout, nFields: st.nFields, nItems, field: 0, warp: W, stride, first: st.first };
    const addrs = warpAddresses(o), tx = transactions(addrs, seg);
    const total = nItems * st.nFields, perRow = 32, rows = Math.ceil(total / perRow);
    // rows to draw: all of them if few, otherwise only rows containing a read address ("…" marks skipped rows)
    let show = [...Array(rows).keys()];
    if (rows > 8) show = [...new Set(addrs.map((a) => Math.floor(a / perRow)))].sort((x, y) => x - y);
    ctx.fillStyle = Tm.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    const left = 44, cw = (s1.w - left - 6) / perRow, ch = Math.min(cw * 1.4, (s1.h - 34) / 8);
    const fieldOf = (a) => (st.layout === 'aos' ? a % st.nFields : Math.floor(a / nItems));
    const itemOf = (a) => (st.layout === 'aos' ? Math.floor(a / st.nFields) : a % nItems);
    const read = new Map(addrs.map((a, t) => [a, t]));
    const touched = new Set(addrs.map((a) => Math.floor(a / seg)));
    ctx.font = `${Math.max(8, Math.min(11, cw * 0.55))}px ${Tm.mono}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    show.forEach((r, k) => {
      const y = 6 + k * ch;
      for (let c = 0; c < perRow; c++) {
        const a = r * perRow + c, x = left + c * cw;
        if (a >= total) break;
        const f = fieldOf(a);
        ctx.fillStyle = shade(CAT[f], 0.55); ctx.globalAlpha = 0.6;
        ctx.fillRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
        ctx.globalAlpha = 1;
        if (read.has(a)) {
          ctx.fillStyle = shade(CAT[f], -0.35); ctx.fillRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
          ctx.fillStyle = textOn(shade(CAT[f], -0.35)); ctx.fillText(`t${read.get(a)}`, x + cw / 2, y + ch / 2);
        } else if (cw > 15 && (itemOf(a) < 100 || cw > 24)) { ctx.fillStyle = textOn(shade(CAT[f], 0.55), 0.6, Tm.bg); ctx.fillText(String(itemOf(a)), x + cw / 2, y + ch / 2); }
      }
      for (let q = 0; q < perRow / seg; q++) { // segment boxes
        const sId = (r * perRow) / seg + q, x = left + q * seg * cw;
        if (sId * seg >= total) break;
        ctx.strokeStyle = touched.has(sId) ? Tm.accent2 : Tm.faint; ctx.lineWidth = touched.has(sId) ? 2.5 : 1;
        ctx.strokeRect(x + 0.5, y + 0.5, seg * cw - 1, ch - 1);
      }
      ctx.fillStyle = Tm.soft; ctx.textAlign = 'right'; ctx.font = `10px ${Tm.mono}`;
      ctx.fillText(String(r * perRow), left - 4, y + ch / 2);
      if (k > 0 && show[k - 1] !== r - 1) { ctx.fillText('⋮', left - 30, y); }
      ctx.textAlign = 'center'; ctx.font = `${Math.max(8, Math.min(11, cw * 0.55))}px ${Tm.mono}`;
    });
    const nShown = show.length;
    // legend
    ctx.textAlign = 'left'; ctx.font = `11px ${Tm.ui}`;
    let lx = left;
    const ly = Math.min(s1.h - 10, 6 + nShown * ch + 14);
    for (let f = 0; f < st.nFields; f++) { ctx.fillStyle = shade(CAT[f], 0.55); ctx.fillRect(lx, ly - 6, 12, 12); ctx.fillStyle = Tm.ink; ctx.fillText(FIELD_NAMES[f], lx + 16, ly); lx += 52; }
    ctx.fillStyle = Tm.accent2; ctx.fillText('▭ segment transferred', lx + 6, ly);
    const used = W * 4, moved = tx * seg * 4;
    const real = transactions(warpAddresses({ ...o, warp: 32, nItems: Math.max(nItems, st.first + 31 * stride + 1) }), 32);
    out.set(`warp: 8 threads, segment 32 B\naddresses: ${addrs.join(', ')}\n<b>transactions: ${tx}</b>  (ideal 1)\nbytes moved ${moved}, used ${used}\n<b>efficiency ${(100 * used / moved).toFixed(0)} %</b>\n\nreal GPU (W = 32, 128-B segments):\n  ${real} transaction${real > 1 ? 's' : ''} (ideal 1)`);
    selfCheck('layout: transactions ≥ 1', tx >= 1);
  }
  s1.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W4: race conditions, colouring, face buffer                         */
/* ------------------------------------------------------------------ */
function raceWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('4×4 mesh: values at vertices (CG) or per element (DG)');
  p1.style.flex = '1 1 420px';
  const s1 = createCanvas(p1, { aspect: 0.82 });
  const nx = 4, g = quadGrid(nx, nx, [0, 4, 0, 4]), k = 4;
  const T = Array.from({ length: g.nElem }, (_, e) => Array.from(g.quads.slice(e * k, e * k + k)));
  const ones = T.map((t) => t.map(() => 1));
  const valence = new Float64Array(g.nVert); for (const v of g.quads) valence[v]++;
  const { color, nColors } = colorElements(g.quads, k);
  const colorGroups = Array.from({ length: nColors }, (_, c) => [...color.keys()].filter((e) => color[e] === c));
  // DG demo data: element values u_e, velocity a = (1, 0.5), upwind fluxes on interior faces (walls elsewhere)
  const F = gridFaces(nx, nx), uE = Float64Array.from({ length: g.nElem }, (_, e) => 1 + ((e * 7) % 5));
  const fluxF = Float64Array.from({ length: F.nFace }, (_, f) => (F.vertical[f] ? 1 : 0.5) * uE[F.minus[f]]);
  const R = faceBufferGather(g.nElem, F, fluxF);
  const faceT = Array.from({ length: F.nFace }, (_, f) => [F.minus[f], F.plus[f]]);
  const faceV = Array.from({ length: F.nFace }, (_, f) => [fluxF[f], -fluxF[f]]);
  const allFaces = [Array.from({ length: F.nFace }, (_, f) => f)];

  const st = { mode: 'naive', phase: 0, last: 0 };
  const MODES = {
    naive: { phases: 2, label: 'CG naive parallel' },
    color: { phases: nColors + 1, label: 'CG colouring' },
    dg: { phases: 3, label: 'DG face buffer + gather' },
    dgnaive: { phases: 2, label: 'DG per-face scatter' },
  };
  segmented(L.controls, { label: 'strategy', value: st.mode, options: [{ value: 'naive', label: 'naive' }, { value: 'color', label: 'colouring' }, { value: 'dg', label: 'face buffer' }, { value: 'dgnaive', label: 'DG scatter' }],
    onChange: (v) => { st.mode = v; st.phase = MODES[v].phases - 1; draw(); } });
  const anim = new Animator(fig, {
    step: () => { const now = performance.now(); if (now - st.last < 1100) return; st.last = now; st.phase = (st.phase + 1) % MODES[st.mode].phases; },
    draw: () => draw(),
  });
  playControls(L.controls, anim, () => { st.phase = 0; draw(); });
  const out = readout(L.controls);
  st.phase = MODES[st.mode].phases - 1;

  function state() {
    const m = st.mode, ph = st.phase;
    if (m === 'naive') {
      const r = ph === 0 ? { out: new Float64Array(g.nVert), lost: 0 } : lockstepScatter(g.nVert, T, ones, [T.map((_, e) => e)]);
      return { vert: r.out, lost: r.lost, active: ph === 1 ? T.map((_, e) => e) : [], launches: 1, done: ph === 1,
        msg: ph === 0 ? 'start: all vertex values 0' : 'all 16 threads loaded 0, then all stored 0 + 1' };
    }
    if (m === 'color') {
      const r = lockstepScatter(g.nVert, T, ones, colorGroups.slice(0, ph));
      return { vert: r.out, lost: r.lost, active: ph > 0 ? colorGroups[ph - 1] : [], launches: nColors, done: ph === nColors,
        msg: ph === 0 ? `start: ${nColors} colours, one launch each` : `launch ${ph}/${nColors}: colour ${ph - 1} (${colorGroups[ph - 1].length} elements, no shared vertex)` };
    }
    if (m === 'dg') {
      return { faces: ph >= 1, elem: ph >= 2 ? R : null, lost: 0, launches: 2, done: ph === 2, active: ph === 2 ? T.map((_, e) => e) : [],
        msg: ph === 0 ? 'start' : ph === 1 ? 'pass 1: one thread per face writes F_f into its own slot' : 'pass 2: one thread per element gathers ±F_f of its 4 faces' };
    }
    const r = ph === 0 ? { out: new Float64Array(g.nElem), lost: 0 } : lockstepScatter(g.nElem, faceT, faceV, allFaces);
    return { faces: ph >= 1, elem: r.out, lost: r.lost, launches: 1, done: ph === 1, active: [],
      msg: ph === 0 ? 'start' : 'each face-thread did R[K−] += F, R[K+] −= F concurrently' };
  }

  function draw() {
    const Tm = theme(), ctx = s1.ctx, S = state();
    ctx.fillStyle = Tm.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    const view = new View2D(s1, [-0.2, 4.2, -0.2, 4.2], { equal: true, pad: 8 });
    const isDG = st.mode === 'dg' || st.mode === 'dgnaive';
    for (let e = 0; e < g.nElem; e++) {
      const ex = e % nx, ey = Math.floor(e / nx);
      const act = S.active.includes(e);
      ctx.fillStyle = st.mode === 'color' ? shade(CAT[color[e]], 0.35) : Tm.elev;
      ctx.globalAlpha = st.mode === 'color' ? (act ? 1 : 0.4) : 0.9;
      ctx.fillRect(view.X(ex) + 1, view.Y(ey + 1) + 1, view.sx - 2, view.sy - 2);
      ctx.globalAlpha = 1;
      if (act && st.mode === 'color') { ctx.strokeStyle = Tm.ink; ctx.lineWidth = 2.5; ctx.strokeRect(view.X(ex) + 3, view.Y(ey + 1) + 3, view.sx - 6, view.sy - 6); }
    }
    ctx.strokeStyle = Tm.faint; ctx.lineWidth = 1;
    for (let q = 0; q <= nx; q++) {
      ctx.beginPath(); ctx.moveTo(view.X(q), view.Y(0)); ctx.lineTo(view.X(q), view.Y(4)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(view.X(0), view.Y(q)); ctx.lineTo(view.X(4), view.Y(q)); ctx.stroke();
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const fs = Math.max(10, Math.min(14, view.sx * 0.16));
    if (!isDG) {
      for (let v = 0; v < g.nVert; v++) {
        const x = view.X(g.nodes[2 * v]), y = view.Y(g.nodes[2 * v + 1]), val = S.vert[v];
        const ok = val === valence[v];
        ctx.beginPath(); ctx.arc(x, y, fs * 0.95, 0, 2 * Math.PI);
        ctx.fillStyle = S.done ? (ok ? Tm.accent3 : Tm.accent2) : Tm.elev; ctx.fill();
        ctx.strokeStyle = Tm.ink; ctx.lineWidth = 1; ctx.stroke();
        ctx.fillStyle = S.done ? Tm.onAccent : Tm.ink; ctx.font = `600 ${fs}px ${Tm.ui}`; // onAccent: readable on the accent discs in both schemes
        ctx.fillText(String(val), x, y);
        if (S.done && !ok) { ctx.fillStyle = Tm.accent2; ctx.font = `${fs * 0.8}px ${Tm.ui}`; ctx.fillText(`≠${valence[v]}`, x + fs * 1.5, y - fs * 0.9); }
      }
    } else {
      ctx.font = `${fs * 0.85}px ${Tm.mono}`;
      if (S.faces) for (let f = 0; f < F.nFace; f++) {
        const a = F.minus[f], ax = a % nx, ay = Math.floor(a / nx);
        const [cx, cy] = F.vertical[f] ? [ax + 1, ay + 0.5] : [ax + 0.5, ay + 1];
        const w = fs * 2.4, hh = fs * 1.3;
        ctx.fillStyle = Tm.elev; ctx.fillRect(view.X(cx) - w / 2, view.Y(cy) - hh / 2, w, hh);
        ctx.strokeStyle = Tm.accent4; ctx.lineWidth = 1.5; ctx.strokeRect(view.X(cx) - w / 2, view.Y(cy) - hh / 2, w, hh);
        ctx.fillStyle = Tm.ink; ctx.fillText(fmt(fluxF[f]), view.X(cx), view.Y(cy));
      }
      for (let e = 0; e < g.nElem; e++) {
        const ex = e % nx, ey = Math.floor(e / nx);
        ctx.fillStyle = Tm.soft; ctx.font = `${fs * 0.75}px ${Tm.ui}`;
        ctx.fillText(`u=${uE[e]}`, view.X(ex + 0.5), view.Y(ey + 0.72));
        if (S.elem && (S.done || st.phase > 0)) {
          const ok = Math.abs(S.elem[e] - R[e]) < 1e-12;
          ctx.fillStyle = S.done ? (ok ? Tm.accent3 : Tm.accent2) : Tm.ink; ctx.font = `600 ${fs}px ${Tm.ui}`;
          ctx.fillText(`R=${fmt(S.elem[e])}`, view.X(ex + 0.5), view.Y(ey + 0.38));
        }
      }
    }
    let wrong = 0;
    if (!isDG) { for (let v = 0; v < g.nVert; v++) if (S.vert[v] !== valence[v]) wrong++; } else if (S.elem) { for (let e = 0; e < g.nElem; e++) if (Math.abs(S.elem[e] - R[e]) > 1e-12) wrong++; }
    const sum = S.elem ? S.elem.reduce((a, b) => a + b, 0) : 0;
    out.set(`<b>${MODES[st.mode].label}</b>\nphase ${st.phase + 1}/${MODES[st.mode].phases}: ${S.msg}\n\nkernel launches: ${S.launches}\nlost updates:    ${S.lost}\n`
      + (S.done ? `wrong entries:   ${wrong}${isDG ? `\nΣ_e R_e = ${fmt(sum)} (conservation: 0)` : ''}` : ''));
    if (S.done) {
      if (st.mode === 'color' || st.mode === 'dg') selfCheck(`race: ${st.mode} gives the exact result`, wrong === 0 && S.lost === 0);
      else selfCheck(`race: ${st.mode} demonstrates lost updates`, S.lost > 0 && wrong > 0);
    }
  }
  selfCheck('race: colouring valid', isValidColoring(g.quads, k, color) && nColors === 4);
  s1.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W5: live WebGPU solver                                               */
/* ------------------------------------------------------------------ */
const INIT_RANGE = { twin: [-0.6, 1], blob: [0, 1], sines: [-1, 1] };

/** Fast nodal evaluation on a tensor GLL element: values f[e·Np + j·n + i]. */
function makeEvaluator(p) {
  const n = p + 1, xi = gaussLobatto(n).x, bw = baryWeights(xi), lx = new Float64Array(n), ly = new Float64Array(n);
  const lag = (s, out) => { // barycentric formula (second form)
    let den = 0;
    for (let j = 0; j < n; j++) { const d = s - xi[j]; if (d === 0) { out.fill(0); out[j] = 1; return; } out[j] = bw[j] / d; den += out[j]; }
    for (let j = 0; j < n; j++) out[j] /= den;
  };
  return (f, e, s, t) => {
    lag(s, lx); lag(t, ly);
    const b = e * n * n;
    let v = 0;
    for (let j = 0; j < n; j++) { let r = 0; for (let i = 0; i < n; i++) r += lx[i] * f[b + j * n + i]; v += ly[j] * r; }
    return v;
  };
}

function liveWidget(fig) {
  const L = widgetLayout(fig);
  const status = h('div', 'gpu-status', 'Checking for WebGPU…');
  status.setAttribute('role', 'status');
  fig.insertBefore(status, L.body);
  const p1 = L.panel('solution $u_h$');
  p1.style.flex = '1 1 380px';
  const s1 = createCanvas(p1, { aspect: 1, description: 'Colour plot of the advected field u_h on the unit square; time, step count and throughput are in the read-out.' });
  const st = { N: 64, p: 3, init: 'twin', batch: 20, running: false, visible: true, verify: null, cpuRate: null, gpuRate: null, busy: false };
  // gpu: the solver whose state is DISPLAYED. scratch: a second GPU solver (same compiled pipelines) on which
  // Verify and Benchmark run, so that they never advance the simulation the user is looking at.
  let disc = null, gpu = null, scratch = null, cpu = null, u0 = null, field = null, evalF = null, gpuStatus = null, mass0 = 0;
  let gen = 0;          // rebuild generation: a queued rebuild that has been superseded is dropped
  let gpuLost = false;  // the device was lost / a GPU operation failed: we run on the CPU twin until Reset re-acquires a device
  const ctl = L.controls;
  select(ctl, { label: 'mesh $N\\times N$', value: String(st.N), options: [8, 16, 32, 64, 128, 256].map((v) => ({ value: String(v), label: `${v} × ${v}` })), onChange: (v) => { st.N = +v; rebuild(); } });
  slider(ctl, { label: 'degree $p$', min: 1, max: 3, step: 1, value: st.p, onChange: (v) => { st.p = v; rebuild(); } });
  select(ctl, { label: 'initial condition', value: st.init, options: [{ value: 'twin', label: 'two blobs (±)' }, { value: 'blob', label: 'one blob' }, { value: 'sines', label: 'sin 2πx · sin 2πy' }], onChange: (v) => { st.init = v; rebuild(); } });
  slider(ctl, { label: 'steps per frame (one submission)', min: 1, max: 200, step: 1, value: st.batch, onInput: (v) => { st.batch = v; } });
  const row = buttonRow(ctl);
  const setRunning = (r) => { st.running = r; playB.setLabel(r ? '❚❚ Pause' : '▶ Run'); if (r) loop(); };
  const playB = button(row, { label: '▶ Run', primary: true, onClick: () => setRunning(!st.running) });
  button(row, { label: '↺ Reset', onClick: () => rebuild() });
  const row2 = buttonRow(ctl);
  const verifyB = button(row2, { label: 'Verify vs CPU', onClick: () => enqueue(verify) });
  button(row2, { label: 'Benchmark', onClick: () => enqueue(() => benchmark()) });
  const out = readout(ctl);
  const esc = (t) => String(t).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  // serialise async operations (GPU runs, read-backs, rebuilds) so they never overlap
  let chain = Promise.resolve();
  const enqueue = (fn) => (chain = chain.then(fn).catch(onFailure));

  /** An operation threw. If a GPU solver is involved this is (almost always) a lost device: carry on with the CPU twin. */
  function onFailure(err) {
    console.error(err);
    if (gpu || scratch) {
      toCpu(gpuStatus && gpuStatus.lost ? `the WebGPU device was lost (${gpuStatus.lostReason}).` : `a GPU operation failed (${err.message}).`);
      return 'recovered';
    }
    reportError('widget #w-live', err);
    out.set(`error: ${esc(err.message)}`);
    setRunning(false);
    return undefined;
  }

  /** Give up the GPU solvers and continue from the last displayed state on the f64 CPU twin. */
  function toCpu(reason) {
    const steps = gpu ? gpu.steps : cpu ? cpu.steps : 0;
    for (const s of [gpu, scratch]) { try { s?.destroy(); } catch (_e) { /* buffers of a lost device are already gone */ } }
    gpu = null; scratch = null; gpuLost = true;
    if (disc) { cpu = makeCpuSolver(disc, field); cpu.steps = steps; cpu.t = steps * disc.stableDt(); }
    st.gpuRate = null; st.verify = null;
    showStatus('bad', `<b>GPU not in use:</b> ${esc(reason)} The simulation continues on the CPU twin (the same algorithm in JavaScript, f64) from the current state. “Reset” tries to get the GPU back.`);
    if (disc) { draw(); report(); }
  }

  function showStatus(cls, html) {
    status.className = `gpu-status ${cls}`; status.innerHTML = html;
    const usingGpu = cls === 'ok';
    verifyB.el.hidden = !usingGpu;
    p1.querySelector('.canvas-label').textContent = usingGpu ? 'solution u_h (read back from the GPU once per frame)' : 'solution u_h (CPU twin)';
  }
  function showDetected() {
    if (gpuStatus.ok) showStatus('ok', `<b>WebGPU available</b> — ${esc(gpuStatus.info)}. Kernels run on your GPU in f32; the CPU twin runs in f64.`);
    else showStatus('bad', `<b>WebGPU not available:</b> ${esc(gpuStatus.reason)} The widget runs the CPU twin (the same algorithm in JavaScript, f64) instead; everything else on this page works.`);
  }
  /** React to a device loss even when no operation is in flight (e.g. while paused). */
  function watchLoss(gs) {
    gs.device.lost.then((info) => enqueue(() => { if (gpuStatus === gs && (gpu || scratch)) toCpu(`the WebGPU device was lost (${info.message || info.reason || 'unknown reason'}).`); }));
  }

  new IntersectionObserver((en) => { st.visible = en[0].isIntersecting; if (st.visible && st.running) loop(); }).observe(fig);

  const dofs = () => disc.nDof;
  const verifySteps = () => Math.max(5, Math.min(100, Math.round(3e6 / dofs())));
  const gpuUsable = () => !!(gpuStatus && gpuStatus.ok && !gpuStatus.lost);

  async function rebuildNow(myGen) {
    for (const s of [gpu, scratch]) { try { s?.destroy(); } catch (_e) { /* lost device */ } }
    gpu = null; scratch = null;
    if (gpuLost) { // after a loss, Reset asks for a fresh device (device.js forgets a lost one)
      gpuStatus = await getGPU();
      gpuLost = !gpuUsable();
      if (!gpuLost) { showDetected(); watchLoss(gpuStatus); }
    }
    disc = makeDGAdv2D({ N: st.N, p: st.p, a: [1, 0.5] });
    u0 = disc.interpolate(INITIAL[st.init]);
    mass0 = disc.mass(u0);
    const dt = disc.stableDt();
    cpu = makeCpuSolver(disc, u0);
    evalF = makeEvaluator(st.p);
    field = u0;
    st.verify = null; st.cpuRate = null; st.gpuRate = null;
    if (gpuUsable() && !gpuLost) {
      const gs = gpuStatus;
      try { gpu = await createGpuSolver(gs.device, disc, u0, { dt }); } catch (err) {
        gpu = null;
        await Promise.race([gs.device.lost, new Promise((r) => setTimeout(r, 100))]); // was it a device loss?
        if (gs.lost) toCpu(`the WebGPU device was lost (${gs.lostReason}).`);
        else {
          gpuLost = true; // a real defect (e.g. a shader that does not compile): visible AND reported to the self-test
          reportError('widget #w-live: GPU solver', err);
          showStatus('bad', `<b>GPU solver failed</b> (${esc(err.message)}); using the CPU twin.`);
        }
      }
    }
    draw(); report();
    // Automatic verification + quick benchmark for moderate sizes. Both run on scratch copies:
    // the displayed state stays the initial condition at t = 0. Skipped if another rebuild is already queued.
    if (myGen === gen && dofs() <= 70000) { await verify(); if (myGen === gen) await benchmark(true); }
    report();
  }
  const rebuild = () => { const g = ++gen; return enqueue(() => (g === gen ? rebuildNow(g) : undefined)); };

  /** The GPU solver used for verification and benchmarks (created on first use after a rebuild). */
  async function scratchSolver() {
    if (!scratch) scratch = await createGpuSolver(gpuStatus.device, disc, u0, { dt: disc.stableDt() });
    return scratch;
  }

  async function verify() {
    if (!gpu) { st.verify = null; report(); return; }
    const K = verifySteps(), dt = disc.stableDt();
    const s = await scratchSolver();
    s.setState(u0);
    await s.run(K);
    const ug = await s.read();
    const c = makeCpuSolver(disc, u0); c.run(K, dt);
    let umax = 0; for (const v of c.u()) umax = Math.max(umax, Math.abs(v));
    st.verify = { K, diff: maxAbsDiff(ug, c.u()), umax };
    selfCheck('live: GPU matches CPU twin to f32 accuracy', st.verify.diff < 1e-4 * Math.max(1, umax));
    report();
  }

  async function benchmark(quick = false) {
    const dt = disc.stableDt();
    // CPU twin: as many steps as fit in the time budget (at least one), on a scratch copy
    const c = makeCpuSolver(disc, field);
    let n = 0; const t0 = performance.now();
    // The step cap matters: if the clock does not advance (headless Chrome with a paused virtual clock) a loop
    // bounded only by performance.now() would never end.
    do { c.step(dt); n++; } while (n < 5000 && performance.now() - t0 < (quick ? 60 : 150));
    const elapsed = performance.now() - t0;
    st.cpuRate = elapsed > 0 ? n / (elapsed / 1000) : NaN;
    if (gpu) {
      const s = await scratchSolver();
      await s.run(2); // warm-up
      let k = 8, ms = await s.run(k);
      while (ms < (quick ? 60 : 200) && k < 4096) { k *= 2; ms = await s.run(k); }
      st.gpuRate = k / (ms / 1000);
      publish();
    }
    report();
  }

  function publish() {
    if (!st.gpuRate) return;
    measured = { dofPerSec: st.gpuRate * dofs(), p: st.p, info: gpuStatus.info };
    globalThis.dispatchEvent(new CustomEvent('ch12-measured'));
  }

  function loop() {
    if (st.busy) return;
    st.busy = true;
    const frame = () => enqueue(async () => {
      if (!st.running || !st.visible) return false;
      const dt = disc.stableDt();
      if (gpu) {
        const ms = await gpu.run(st.batch);
        field = await gpu.read();
        const r = st.batch / (ms / 1000);
        st.gpuRate = st.gpuRate ? 0.8 * st.gpuRate + 0.2 * r : r; // (only Benchmark publishes to the roofline)
      } else {
        const t0 = performance.now(); let n = 0;
        do { cpu.step(dt); n++; } while (n < st.batch && performance.now() - t0 < 30);
        { const el = performance.now() - t0; if (el > 0) st.cpuRate = n / (el / 1000); }
        field = cpu.u();
      }
      let finite = true; for (let g = 0; g < field.length; g += 97) if (!Number.isFinite(field[g])) { finite = false; break; }
      if (!finite) { setRunning(false); out.set('<b class="status-bad">The solution is no longer finite — stopped. Press Reset.</b>'); draw(); return false; }
      draw(); report();
      return true;
    }).then((go) => { if (go === true || (go === 'recovered' && st.running)) requestAnimationFrame(frame); else st.busy = false; });
    frame();
  }

  function report() {
    if (!disc) return;
    const steps = gpu ? gpu.steps : cpu.steps, t = steps * disc.stableDt();
    const tot = stageTotals(st.p), bytesPerStep = 5 * tot.bytes * dofs();
    const big = (v) => (v >= 1e9 ? `${fmt(v / 1e9)} G` : v >= 1e6 ? `${fmt(v / 1e6)} M` : `${fmt(v / 1e3)} k`);
    let m = 0; for (let e = 0; e < disc.nElem; e++) for (let j = 0; j <= st.p; j++) for (let i = 0; i <= st.p; i++) m += disc.w[i] * disc.w[j] * field[e * disc.Np + j * (st.p + 1) + i];
    m *= disc.h * disc.h / 4;
    let s = `DOFs ${dofs().toLocaleString('en')}   Δt = ${fmt(disc.stableDt())}\nt = ${t.toFixed(3)}   steps = ${steps}\n`;
    if (gpu) s += `\n<b>GPU</b>  ${st.gpuRate ? `${fmt(st.gpuRate)} steps/s\n     ${big(st.gpuRate * dofs())}DOF-updates/s\n     ≈ ${fmt(st.gpuRate * bytesPerStep / 1e9)} GB/s effective` : '– (press Benchmark)'}`;
    s += `\n<b>CPU twin</b> ${st.cpuRate ? `${fmt(st.cpuRate)} steps/s\n     ${big(st.cpuRate * dofs())}DOF-updates/s` : '– (press Benchmark)'}`;
    if (gpu && st.gpuRate && st.cpuRate) s += `\n<b>speed-up ×${fmt(st.gpuRate / st.cpuRate)}</b>`;
    if (gpu) s += st.verify ? `\n\nmax|GPU−CPU| after ${st.verify.K} steps from u_0\n  = ${st.verify.diff.toExponential(2)}  (max|u| ${fmt(st.verify.umax)})` : '\n\n(press Verify to compare with CPU)';
    s += `\nmass drift |∫u_h − ∫u_0| = ${Math.abs(m - mass0).toExponential(1)}`;
    out.set(s);
  }

  function draw() {
    if (!disc) return;
    const Tm = theme(), ctx = s1.ctx;
    ctx.fillStyle = Tm.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    const view = new View2D(s1, [0, 1, 0, 1], { equal: true, pad: [8, 52, 8, 8] });
    const [lo, hi] = INIT_RANGE[st.init];
    const f = field;
    drawQuadField(s1, view, disc.N, disc.N, [0, 1, 0, 1], (e, s, t) => evalF(f, e, s, t), { cmap: 'viridis', lo, hi });
    if (disc.N <= 32) {
      ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.lineWidth = 1; ctx.beginPath();
      for (let k = 0; k <= disc.N; k++) { const q = k / disc.N; ctx.moveTo(view.X(q), view.Y(0)); ctx.lineTo(view.X(q), view.Y(1)); ctx.moveTo(view.X(0), view.Y(q)); ctx.lineTo(view.X(1), view.Y(q)); }
      ctx.stroke();
    }
    colorbar(ctx, 'viridis', view.X(1) + 10, view.Y(1), 12, view.Y(0) - view.Y(1), lo, hi, { ink: Tm.soft });
    let finite = true; for (let g = 0; g < f.length; g += 97) if (!Number.isFinite(f[g])) { finite = false; break; }
    selfCheck('live: field finite', finite);
  }

  s1.onResize(draw);
  // initialise: detect the GPU, then build
  const ready = (async () => {
    gpuStatus = await getGPU();
    showDetected();
    if (gpuUsable()) watchLoss(gpuStatus);
    await rebuild();
    // the widget must start at t = 0 showing the initial condition, also after the automatic verify + benchmark
    let untouched = (gpu ? gpu.steps : cpu.steps) === 0 && field === u0;
    if (gpu) untouched = untouched && maxAbsDiff(await enqueue(() => gpu.read()), Float32Array.from(u0)) === 0;
    selfCheck('live: starts at t = 0 with the initial condition (verify/benchmark do not advance it)', untouched);
    // CPU sanity checks (also without a GPU): a few steps conserve mass
    const d = makeDGAdv2D({ N: 8, p: 2, a: [1, 0.5] }), c = makeCpuSolver(d, d.interpolate(INITIAL.twin)), m0 = d.mass(c.u());
    c.run(10, d.stableDt());
    selfCheck('live: CPU twin conserves mass', Math.abs(d.mass(c.u()) - m0) < 1e-13);
  })();
  ready.catch((err) => reportError('widget #w-live', err));
  return { ready };
}

/* ------------------------------------------------------------------ */
/* static content: generated WGSL and the sum-factorisation table       */
/* ------------------------------------------------------------------ */
function fillStatic() {
  const pre = document.getElementById('wgsl-volume');
  if (pre) {
    const d = makeDGAdv2D({ N: 2, p: 2 });
    pre.textContent = makeKernelSources(2, d.D).volume.trim();
  }
  const tab = document.getElementById('sumfact-table');
  if (tab) {
    let html = '<tr><th>$p$</th><th>2D: dense $2n^4$</th><th>2D: sum-fact. $2n^3$</th><th>ratio</th><th>3D: dense $2n^6$</th><th>3D: sum-fact. $2n^4$</th><th>ratio</th></tr>';
    for (const p of [1, 2, 3, 4, 6, 8]) {
      const a = derivativeFlops(p, 2), b = derivativeFlops(p, 3);
      html += `<tr><td>${p}</td><td>${a.dense.toLocaleString('en')}</td><td>${a.sumfact.toLocaleString('en')}</td><td>${a.dense / a.sumfact}</td>`
        + `<td>${b.dense.toLocaleString('en')}</td><td>${b.sumfact.toLocaleString('en')}</td><td>${b.dense / b.sumfact}</td></tr>`;
    }
    tab.innerHTML = `<caption style="caption-side: bottom; font-size: 13px; color: var(--ink-soft); padding-top: .3rem">FLOPs per element for one partial derivative, $n = p+1$ (computed by <code>derivativeFlops</code> in <code>lib/gpu/cost.js</code>). The ratio is $n^{d-1}$.</caption>${html}`;
  }
}

initChapter(async () => {
  fillStatic();
  mount('w-threads', threadsWidget);
  mount('w-roofline', rooflineWidget);
  mount('w-layout', layoutWidget);
  mount('w-race', raceWidget);
  const live = mount('w-live', liveWidget);
  // wait (bounded) for the live widget's GPU detection + first build, so self-test sees its checks
  if (live && live.ready) await Promise.race([live.ready, new Promise((r) => setTimeout(r, 10000))]);
});
