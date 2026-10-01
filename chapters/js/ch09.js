/**
 * @file Chapter 9 — Mixed methods & hybridization.
 *
 * Widgets:
 *  - w-rt0      : RT0–P0 mixed solution: P0 colours, flux arrows, edge fluxes, per-triangle balance on click
 *  - w-condense : static condensation animated on spy plots (field order → element order → Schur complement)
 *  - w-hybrid   : hybridized RT0: multipliers λ on the edges, λ vs exact edge means, equivalence with mixed RT0
 *  - w-hdg      : HDG solver (k, N, τ): u_h or u*_h with traces, convergence plot with slopes k+1, k+2
 *  - w-sizes    : global system size of CG vs SIPG vs HDG versus p
 */
import { initChapter, mount, selfCheck } from '../../lib/ui/chapter.js';
import { widgetLayout, slider, select, segmented, checkbox, readout, fmt, debounce, h, button, buttonRow } from '../../lib/ui/controls.js';
import { createCanvas, View2D, theme, seriesColors, onPointer, arrow } from '../../lib/viz/canvas.js';
import { drawTriField } from '../../lib/viz/field2d.js';
import { drawTriMesh, fillTri } from '../../lib/viz/meshdraw.js';
import { colorOf, colorbar } from '../../lib/viz/colormap.js';
import { Plot } from '../../lib/viz/plot1d.js';
import { Animator } from '../../lib/viz/anim.js';
import { triGrid } from '../../lib/core/mesh/structured.js';
import { EDGE_VERTS } from '../../lib/core/basis/simplex.js';
import { POISSON_MMS } from '../../lib/core/verify/mms.js';
import { solveMixedRT0, rt0Errors, rt0ConservationResidual, rt0Local, rt0Eval } from '../../lib/core/elliptic/mixedRT0.js';
import { solveHybridRT0, hybridRT0Monolithic, edgeMeans } from '../../lib/core/elliptic/hybridRT0.js';
import { solveHDG, hdgErrors, hdgMonolithic, edgeMode } from '../../lib/core/elliptic/hdg.js';
import { brokenEvaluator } from '../../lib/core/elliptic/dgtri.js';
import { systemSizes } from '../../lib/core/elliptic/dofcount.js';
import { csrToDense, nnz } from '../../lib/core/la/sparse.js';
import { cholesky, inverse } from '../../lib/core/la/dense.js';
import { pairwiseRates } from '../../lib/core/verify/rates.js';

const PROBLEMS = [
  { value: 'wave', label: 'u = cos(πx)·eʸ' },
  { value: 'sinsin', label: 'u = sin(πx)·sin(πy)' },
  { value: 'peak', label: 'Gaussian peak' },
];
const meshOf = (N) => triGrid(N, N, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.12 });

function exactRange(u) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i <= 60; i++) for (let j = 0; j <= 60; j++) { const v = u(i / 60, j / 60); lo = Math.min(lo, v); hi = Math.max(hi, v); }
  return [lo, hi];
}
const centroid = (mesh, t) => {
  const { nodes, tris } = mesh;
  let x = 0, y = 0;
  for (let k = 0; k < 3; k++) { x += nodes[2 * tris[3 * t + k]] / 3; y += nodes[2 * tris[3 * t + k] + 1] / 3; }
  return [x, y];
};
/** Index of the triangle containing (x,y), or −1. */
function findTri(mesh, x, y) {
  const { nodes, tris } = mesh;
  for (let t = 0; t < tris.length / 3; t++) {
    const P = [0, 1, 2].map((k) => [nodes[2 * tris[3 * t + k]], nodes[2 * tris[3 * t + k] + 1]]);
    const d = (P[1][0] - P[0][0]) * (P[2][1] - P[0][1]) - (P[2][0] - P[0][0]) * (P[1][1] - P[0][1]);
    const l1 = ((x - P[0][0]) * (P[2][1] - P[0][1]) - (P[2][0] - P[0][0]) * (y - P[0][1])) / d;
    const l2 = ((P[1][0] - P[0][0]) * (y - P[0][1]) - (x - P[0][0]) * (P[1][1] - P[0][1])) / d;
    if (l1 >= 0 && l2 >= 0 && l1 + l2 <= 1) return t;
  }
  return -1;
}

/* ------------------------------------------------------------------ */
/* W1: RT0 mixed solution                                               */
/* ------------------------------------------------------------------ */
function rt0Widget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('$u_h\\in P_0$ (colours) and $\\boldsymbol\\sigma_h$ at centroids (arrows); click a triangle', '2 1 420px');
  const s1 = createCanvas(p1, { aspect: 0.86 });
  const st = { N: 4, prob: 'wave', arrows: true, fluxes: true, sel: -1 };
  slider(L.controls, { label: 'mesh $N$ ($2N^2$ triangles)', min: 2, max: 16, step: 1, value: st.N, onInput: (v) => { st.N = v; st.sel = -1; later(); } });
  select(L.controls, { label: 'exact solution', options: PROBLEMS, value: st.prob, onChange: (v) => { st.prob = v; solve(); } });
  checkbox(L.controls, { label: 'flux arrows $\\boldsymbol\\sigma_h$', value: st.arrows, onChange: (v) => { st.arrows = v; draw(); } });
  checkbox(L.controls, { label: 'edge fluxes (N ≤ 4)', value: st.fluxes, onChange: (v) => { st.fluxes = v; draw(); } });
  const out = readout(L.controls);
  let R = null, view = null;
  const later = debounce(() => solve(), 50);
  function solve() {
    const S = POISSON_MMS[st.prob], mesh = meshOf(st.N);
    const r = solveMixedRT0(mesh, { f: S.f, g: S.u });
    const err = rt0Errors(r, S.u, S.grad), res = rt0ConservationResidual(r);
    let cons = 0, fmax = 0;
    for (let t = 0; t < r.nTri; t++) { cons = Math.max(cons, Math.abs(res[t])); fmax = Math.max(fmax, Math.abs(r.F[t])); }
    const locs = Array.from({ length: r.nTri }, (_, t) => rt0Local(mesh, r.topo, t));
    // normal continuity check at edge midpoints from both sides
    let nc = 0;
    for (let e = 0; e < r.topo.nEdge; e++) {
      const tp = r.topo.edgeTris[2 * e + 1];
      if (tp < 0) continue;
      const a = r.topo.edges[2 * e], b = r.topo.edges[2 * e + 1];
      const x = 0.5 * (mesh.nodes[2 * a] + mesh.nodes[2 * b]), y = 0.5 * (mesh.nodes[2 * a + 1] + mesh.nodes[2 * b + 1]);
      const nx = r.topo.normals[2 * e], ny = r.topo.normals[2 * e + 1];
      const sm = rt0Eval(locs[r.topo.edgeTris[2 * e]], r.sigma, x, y), sp = rt0Eval(locs[tp], r.sigma, x, y);
      nc = Math.max(nc, Math.abs((sm[0] - sp[0]) * nx + (sm[1] - sp[1]) * ny));
    }
    R = { S, mesh, r, err, cons, fmax, locs, nc, range: exactRange(S.u) };
    selfCheck('rt0: exact conservation', cons <= 1e-11 * (1 + fmax));
    selfCheck('rt0: normal continuity', nc < 1e-10);
    draw();
  }
  function draw() {
    if (!R) return;
    const T = theme(), { mesh, r, locs } = R, [lo, hi] = R.range, ctx = s1.ctx;
    view = new View2D(s1, [0, 1, 0, 1], { equal: true, pad: [10, 72, 10, 10] });
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    for (let t = 0; t < r.nTri; t++) fillTri(ctx, view, mesh.nodes, mesh.tris, t, colorOf('viridis', r.u[t], lo, hi));
    drawTriMesh(ctx, view, mesh.nodes, mesh.tris, { color: 'rgba(255,255,255,0.45)', width: 0.8 });
    if (st.sel >= 0) {
      ctx.save(); ctx.strokeStyle = T.accent2; ctx.lineWidth = 3; ctx.beginPath();
      for (let k = 0; k <= 3; k++) { const v = mesh.tris[3 * st.sel + (k % 3)]; const X = view.X(mesh.nodes[2 * v]), Y = view.Y(mesh.nodes[2 * v + 1]); if (k) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }
      ctx.stroke(); ctx.restore();
    }
    if (st.arrows) {
      let smax = 0; const cs = [];
      for (let t = 0; t < r.nTri; t++) { const [x, y] = centroid(mesh, t), s = rt0Eval(locs[t], r.sigma, x, y); cs.push([x, y, s]); smax = Math.max(smax, Math.hypot(...s)); }
      const len = 0.75 / st.N / (smax || 1);
      ctx.save(); ctx.strokeStyle = ctx.fillStyle = '#fff'; ctx.lineWidth = 1.4;
      for (const [x, y, s] of cs) arrow(ctx, view.X(x), view.Y(y), view.X(x + len * s[0]), view.Y(y + len * s[1]), 5);
      ctx.restore();
    }
    if (st.fluxes && st.N <= 4) {
      ctx.save(); ctx.font = `11px ${T.ui}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (let e = 0; e < r.topo.nEdge; e++) {
        const a = r.topo.edges[2 * e], b = r.topo.edges[2 * e + 1];
        const x = 0.5 * (mesh.nodes[2 * a] + mesh.nodes[2 * b]), y = 0.5 * (mesh.nodes[2 * a + 1] + mesh.nodes[2 * b + 1]);
        const nx = r.topo.normals[2 * e], ny = r.topo.normals[2 * e + 1], X = view.X(x), Y = view.Y(y);
        ctx.strokeStyle = T.accent2; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(X + 7 * nx, Y - 7 * ny); ctx.stroke();
        const txt = (r.sigma[e] * r.topo.lengths[e]).toFixed(2);
        const w = ctx.measureText(txt).width + 4;
        ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(X - w / 2, Y - 7, w, 14);
        ctx.fillStyle = '#fff'; ctx.fillText(txt, X, Y);
      }
      ctx.restore();
    }
    colorbar(ctx, 'viridis', s1.w - 62, 14, 12, s1.h - 28, lo, hi, { ink: T.soft });
    let sel = '';
    if (st.sel >= 0) {
      const t = st.sel, loc = locs[t];
      let sum = 0; const parts = [];
      for (let k = 0; k < 3; k++) { const e = loc.edges[k], fl = loc.sign[k] * r.topo.lengths[e] * r.sigma[e]; sum += fl; parts.push(fmt(fl, 4)); }
      sel = `\n<b>triangle ${t}</b>, outward fluxes:\n  ${parts.join('\n  ')}\n  sum   = ${sum.toPrecision(8)}\n  ∫_K f = ${r.F[t].toPrecision(8)}`;
    }
    out.set(`unknowns: ${r.nEdge} σ + ${r.nTri} u\nsaddle matrix ${r.A.n}², indefinite\n`
      + `‖u − u_h‖   = ${fmt(R.err.L2u)}\n‖Π₀u − u_h‖ = ${fmt(R.err.L2u0)}\n‖σ − σ_h‖   = ${fmt(R.err.L2s)}\n`
      + `max |∫∂K σ_h·n − ∫K f|\n  = ${fmt(R.cons, 2)}\nmax normal jump = ${fmt(R.nc, 2)}${sel}`);
  }
  onPointer(s1.canvas, { down: (x, y) => { if (!view || !R) return false; st.sel = findTri(R.mesh, view.invX(x), view.invY(y)); draw(); return false; } });
  s1.onResize(draw);
  solve();
  st.sel = 5; draw();
}

/* ------------------------------------------------------------------ */
/* W2: static condensation animator                                      */
/* ------------------------------------------------------------------ */
function condenseWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('matrix sparsity pattern', '2 1 420px');
  const s1 = createCanvas(p1, { aspect: 1 });
  const st = { method: 'rt0', s: 0, target: 0 };
  segmented(L.controls, { label: 'method', options: [{ value: 'rt0', label: 'hybrid RT0' }, { value: 'hdg', label: 'HDG k = 1' }], value: st.method, onChange: (v) => { st.method = v; build(); } });
  const stageSeg = segmented(L.controls, { label: 'stage', options: [{ value: '0', label: '0 fields' }, { value: '1', label: '1 elements' }, { value: '2', label: '2 condensed' }], value: '0', onChange: (v) => { st.target = +v; anim.play(); } });
  const anim = new Animator(fig, {
    step: () => {
      const d = st.target - st.s;
      if (Math.abs(d) < 0.02) { st.s = st.target; return false; }
      st.s += Math.sign(d) * 0.02;
    },
    draw: () => draw(),
  });
  const row = buttonRow(L.controls);
  const play = button(row, { label: '▶ Play 0 → 2', primary: true, onClick: () => { st.s = 0; st.target = 2; stageSeg.set('2'); anim.play(); } });
  button(row, { label: '↺ Reset', onClick: () => { anim.pause(); st.s = 0; st.target = 0; stageSeg.set('0'); draw(); } });
  anim.o.onState = (r) => play.setLabel(r ? '❚❚ running' : '▶ Play 0 → 2');
  const out = readout(L.controls);
  let M = null;
  function build() {
    const S = POISSON_MMS.wave, mesh = triGrid(2, 2, [0, 1, 0, 1], { diag: 'alt' });
    const sys = st.method === 'rt0' ? hybridRT0Monolithic(mesh, { f: S.f, g: S.u }) : hdgMonolithic(mesh, { k: 1, tau: 1, f: S.f, g: S.u });
    const { A, nTri, mLoc, nLam, fields } = sys, n = A.n, nLoc = nTri * mLoc;
    // stage-0 position of every unknown: field-major ordering
    const pos0 = new Float64Array(n);
    let off = 0, fo = 0;
    for (const f of fields) {
      for (let t = 0; t < nTri; t++) for (let i = 0; i < f.size; i++) pos0[t * mLoc + fo + i] = off + t * f.size + i;
      off += nTri * f.size; fo += f.size;
    }
    for (let j = 0; j < nLam; j++) pos0[nLoc + j] = nLoc + j;
    // Schur complement on the skeleton: S = A_λλ − A_λx D⁻¹ A_xλ  (D block diagonal ⇒ block-wise inverse)
    const Dn = csrToDense(A), Sc = new Float64Array(nLam * nLam);
    for (let i = 0; i < nLam; i++) for (let j = 0; j < nLam; j++) Sc[i * nLam + j] = Dn[(nLoc + i) * n + nLoc + j];
    for (let t = 0; t < nTri; t++) {
      const blk = new Float64Array(mLoc * mLoc);
      for (let a = 0; a < mLoc; a++) for (let b = 0; b < mLoc; b++) blk[a * mLoc + b] = Dn[(t * mLoc + a) * n + t * mLoc + b];
      const Bi = inverse(blk, mLoc);
      for (let i = 0; i < nLam; i++) for (let j = 0; j < nLam; j++) {
        let s = 0;
        for (let a = 0; a < mLoc; a++) {
          const lia = Dn[(nLoc + i) * n + t * mLoc + a];
          if (lia === 0) continue;
          for (let b = 0; b < mLoc; b++) s += lia * Bi[a * mLoc + b] * Dn[(t * mLoc + b) * n + nLoc + j];
        }
        Sc[i * nLam + j] -= s;
      }
    }
    let smax = 0; for (const v of Sc) smax = Math.max(smax, Math.abs(v));
    const entries = [];
    for (let i = 0; i < n; i++) for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) if (A.vals[k] !== 0) entries.push([i, A.colIdx[k]]);
    const fill = [], keep = [];
    for (let i = 0; i < nLam; i++) for (let j = 0; j < nLam; j++) {
      if (Math.abs(Sc[i * nLam + j]) <= 1e-12 * smax) continue;
      (Dn[(nLoc + i) * n + nLoc + j] !== 0 ? keep : fill).push([i, j]);
    }
    let symErr = 0; for (let i = 0; i < nLam; i++) for (let j = 0; j < nLam; j++) symErr = Math.max(symErr, Math.abs(Sc[i * nLam + j] - Sc[j * nLam + i]));
    // the hybrid RT0 / HDG Schur complement is −H (negative definite); check −S is SPD
    const negS = Sc.map((v) => -v);
    const spd = cholesky(negS, nLam) !== null;
    selfCheck('condense: −(Schur complement) symmetric positive definite', spd && symErr <= 1e-12 * smax);
    M = { n, nLoc, nLam, nTri, mLoc, fields, pos0, entries, fill, keep, spd, nnzA: entries.length };
    draw();
  }
  const ease = (x) => x * x * (3 - 2 * x);
  function draw() {
    if (!M) return;
    const T = theme(), C = seriesColors(), ctx = s1.ctx, { n, nLoc, nLam } = M;
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    const size = Math.min(s1.w, s1.h) - 20, ox = (s1.w - size) / 2, oy = 10;
    const s = st.s, a = ease(Math.min(1, s)), b = ease(Math.max(0, s - 1));
    // index → screen: window [b·nLoc, n] mapped to the square
    const w0 = b * nLoc, span = n - w0, cell = size / span;
    const P = (c) => (c - w0) / span * size;
    ctx.save();
    ctx.beginPath(); ctx.rect(ox, oy, size, size); ctx.clip();
    ctx.fillStyle = T.elev; ctx.fillRect(ox, oy, size, size);
    const col = (i, j) => (i < nLoc && j < nLoc ? C[0] : i >= nLoc && j >= nLoc ? C[1] : C[2]);
    for (const [i, j] of M.entries) {
      const local = i < nLoc || j < nLoc;
      ctx.globalAlpha = local ? 1 - b : 1;
      if (ctx.globalAlpha <= 0.01) continue;
      const pi = M.pos0[i] + (i - M.pos0[i]) * a, pj = M.pos0[j] + (j - M.pos0[j]) * a;
      ctx.fillStyle = col(i, j);
      ctx.fillRect(ox + P(pj), oy + P(pi), Math.max(cell * 0.92, 1), Math.max(cell * 0.92, 1));
    }
    ctx.globalAlpha = b;
    ctx.fillStyle = C[5];
    for (const [i, j] of M.fill) ctx.fillRect(ox + P(nLoc + j), oy + P(nLoc + i), cell * 0.92, cell * 0.92);
    ctx.globalAlpha = 1;
    // separators: element blocks (stage 1) and the local/skeleton split
    ctx.strokeStyle = T.accent2; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ox + P(nLoc), oy); ctx.lineTo(ox + P(nLoc), oy + size); ctx.moveTo(ox, oy + P(nLoc)); ctx.lineTo(ox + size, oy + P(nLoc)); ctx.stroke();
    if (s > 0.5 && s < 1.5) {
      ctx.globalAlpha = 1 - 2 * Math.abs(s - 1); ctx.strokeStyle = T.faint;
      ctx.beginPath();
      for (let t = 1; t < M.nTri; t++) { const c = P(t * M.mLoc); ctx.moveTo(ox + c, oy); ctx.lineTo(ox + c, oy + P(nLoc)); ctx.moveTo(ox, oy + c); ctx.lineTo(ox + P(nLoc), oy + c); }
      ctx.stroke();
    }
    ctx.restore();
    ctx.strokeStyle = T.faint; ctx.strokeRect(ox + 0.5, oy + 0.5, size, size);
    const stage = s < 0.5 ? 0 : s < 1.5 ? 1 : 2;
    const fieldTxt = M.fields.map((f) => `${f.name}: ${f.size}/triangle`).join(', ');
    const txt = [
      `stage 0 — by field:\n  ${fieldTxt}, then λ`,
      `stage 1 — by triangle:\n  ${M.nTri} blocks ${M.mLoc}×${M.mLoc}, then λ`,
      `stage 2 — Schur complement\n  on the skeleton: ${nLam}×${nLam}`,
    ][stage];
    const sq = (c) => `<span style="color:${c}">■</span>`;
    out.set(`${txt}\n\nmonolithic: ${n} unknowns\n  ${M.nnzA} non-zeros\ncondensed:  ${nLam} unknowns\n  ${M.keep.length + M.fill.length} non-zeros (${M.fill.length} fill-in)\n`
      + `−S is SPD: ${M.spd ? '<b class="status-good">yes</b>' : '<b class="status-bad">no</b>'}\n\n`
      + `${sq(C[0])} element–element\n${sq(C[2])} element–λ\n${sq(C[1])} λ–λ\n${sq(C[5])} fill-in`);
  }
  s1.onResize(draw);
  build();
}

/* ------------------------------------------------------------------ */
/* W3: hybridized RT0 multipliers                                       */
/* ------------------------------------------------------------------ */
function hybridWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('multipliers $\\lambda_E$ drawn on the edges');
  const s1 = createCanvas(p1, { aspect: 0.9 });
  const p2 = L.panel('$\\lambda_E$ vs exact edge mean $\\frac{1}{|E|}\\int_E u$');
  const s2 = createCanvas(p2, { aspect: 0.9 });
  const st = { N: 6, prob: 'wave' };
  slider(L.controls, { label: 'mesh $N$ ($2N^2$ triangles)', min: 2, max: 16, step: 1, value: st.N, onInput: (v) => { st.N = v; later(); } });
  select(L.controls, { label: 'exact solution', options: PROBLEMS, value: st.prob, onChange: (v) => { st.prob = v; solve(); } });
  const out = readout(L.controls);
  let R = null;
  const later = debounce(() => solve(), 50);
  function solve() {
    const S = POISSON_MMS[st.prob], mesh = meshOf(st.N);
    const hy = solveHybridRT0(mesh, { f: S.f, g: S.u }), mx = solveMixedRT0(mesh, { f: S.f, g: S.u });
    let ds = 0, du = 0, sm = 0;
    for (let e = 0; e < mx.nEdge; e++) { ds = Math.max(ds, Math.abs(mx.sigma[e] - hy.sigmaEdge[e])); sm = Math.max(sm, Math.abs(mx.sigma[e])); }
    for (let t = 0; t < mx.nTri; t++) du = Math.max(du, Math.abs(mx.u[t] - hy.u[t]));
    const em = edgeMeans(mesh, hy.topo, S.u);
    let el = 0; for (let e = 0; e < hy.topo.nEdge; e++) el = Math.max(el, Math.abs(em[e] - hy.lambda[e]));
    const spd = hy.nLam <= 900 ? cholesky(csrToDense(hy.H), hy.nLam) !== null : null;
    R = { S, mesh, hy, ds: ds / (sm || 1), du, em, el, spd, range: exactRange(S.u) };
    selfCheck('hybrid = mixed RT0', ds <= 1e-9 * (sm || 1) && du < 1e-9);
    if (spd !== null) selfCheck('hybrid H SPD', spd);
    draw();
  }
  function draw() {
    if (!R) return;
    const T = theme(), C = seriesColors(), { mesh, hy } = R, [lo, hi] = R.range, ctx = s1.ctx;
    const view = new View2D(s1, [0, 1, 0, 1], { equal: true, pad: [10, 72, 10, 10] });
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    for (let t = 0; t < hy.nTri; t++) fillTri(ctx, view, mesh.nodes, mesh.tris, t, colorOf('viridis', hy.u[t], lo, hi), 0.28);
    ctx.save(); ctx.lineCap = 'round'; ctx.lineWidth = Math.max(2.5, 9 - st.N * 0.45);
    for (let e = 0; e < hy.topo.nEdge; e++) {
      const a = hy.topo.edges[2 * e], b = hy.topo.edges[2 * e + 1];
      const x0 = mesh.nodes[2 * a], y0 = mesh.nodes[2 * a + 1], x1 = mesh.nodes[2 * b], y1 = mesh.nodes[2 * b + 1];
      const sh = 0.12; // shorten so that the edges stay distinguishable at vertices
      ctx.strokeStyle = colorOf('viridis', hy.lambda[e], lo, hi);
      ctx.beginPath();
      ctx.moveTo(view.X(x0 + sh * (x1 - x0)), view.Y(y0 + sh * (y1 - y0)));
      ctx.lineTo(view.X(x1 - sh * (x1 - x0)), view.Y(y1 - sh * (y1 - y0))); ctx.stroke();
    }
    ctx.restore();
    colorbar(ctx, 'viridis', s1.w - 62, 14, 12, s1.h - 28, lo, hi, { ink: T.soft });
    const P = new Plot(s2, { xlim: [lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)], ylim: [lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)], xlabel: 'exact edge mean of u', ylabel: 'λ_E', equal: true });
    P.frame();
    P.line([lo, hi], [lo, hi], { color: T.faint, dash: [5, 4], width: 1.2 });
    const xs = [], ys = [], xb = [], yb = [];
    for (let e = 0; e < hy.topo.nEdge; e++) (hy.topo.isBoundary[e] ? (xb.push(R.em[e]), yb) : (xs.push(R.em[e]), ys)).push(hy.lambda[e]);
    P.points(xs, ys, { color: C[0], r: 2.6 });
    P.points(xb, yb, { color: C[1], r: 2.6, hollow: true });
    P.legend([{ label: 'interior edges (unknowns)', color: C[0], marker: true }, { label: 'boundary edges (mean of g)', color: C[1], marker: true }], 'tl');
    out.set(`H: ${hy.nLam}×${hy.nLam}, ${nnz(hy.H)} non-zeros\n(one unknown per interior edge)\n`
      + `H SPD (Cholesky): ${R.spd === null ? '— (too large)' : R.spd ? '<b class="status-good">yes</b>' : '<b class="status-bad">no</b>'}\n`
      + `hybrid vs mixed RT0:\n  max|Δσ|/max|σ| = ${fmt(R.ds, 2)}\n  max|Δu|        = ${fmt(R.du, 2)}\n`
      + `max |λ_E − mean_E(u)|\n  = ${fmt(R.el)}`);
  }
  s1.onResize(draw); s2.onResize(draw);
  solve();
}

/* ------------------------------------------------------------------ */
/* W4: HDG solver                                                       */
/* ------------------------------------------------------------------ */
function hdgWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('solution and traces $\\lambda_h$ on the edges');
  const s1 = createCanvas(p1, { aspect: 0.9 });
  const p2 = L.panel('$L^2$ errors vs $h = 1/N$');
  const s2 = createCanvas(p2, { aspect: 0.9 });
  const st = { k: 1, N: 4, tau: 1, prob: 'wave', view: 'u', traces: true };
  segmented(L.controls, { label: 'degree $k$', options: [1, 2, 3].map((v) => ({ value: String(v), label: String(v) })), value: '1', onChange: (v) => { st.k = +v; solve(); conv(); } });
  slider(L.controls, { label: 'mesh $N$', min: 2, max: 16, step: 1, value: st.N, onInput: (v) => { st.N = v; later(); } });
  slider(L.controls, { label: 'stabilisation $\\tau$', min: 0.01, max: 100, log: true, value: st.tau, format: (v) => fmt(v, 2), onInput: (v) => { st.tau = v; later(); laterConv(); } });
  select(L.controls, { label: 'exact solution', options: PROBLEMS.slice(0, 2), value: st.prob, onChange: (v) => { st.prob = v; solve(); conv(); } });
  segmented(L.controls, { label: 'show', options: [{ value: 'u', label: 'u_h' }, { value: 'ustar', label: 'u*_h' }], value: st.view, onChange: (v) => { st.view = v; draw(); } });
  checkbox(L.controls, { label: 'traces $\\lambda_h$ on edges', value: st.traces, onChange: (v) => { st.traces = v; draw(); } });
  const out = readout(L.controls);
  let R = null, CV = null;
  const later = debounce(() => solve(), 60), laterConv = debounce(() => conv(), 250);
  function solve() {
    const S = POISSON_MMS[st.prob], mesh = meshOf(st.N), t0 = performance.now();
    const H = solveHDG(mesh, { k: st.k, tau: st.tau, f: S.f, g: S.u });
    const e = hdgErrors(H, S.u, S.grad);
    R = { S, mesh, H, e, ms: performance.now() - t0, range: exactRange(S.u) };
    selfCheck('hdg: finite', H.u.every(Number.isFinite) && H.lambda.every(Number.isFinite));
    draw();
  }
  let job = 0;
  function conv() {
    const my = ++job, S = POISSON_MMS[st.prob], Ns = [2, 4, 8, 16], rows = [];
    const key = { k: st.k, tau: st.tau };
    let i = 0;
    const step = () => {
      if (my !== job) return;
      const N = Ns[i++];
      const e = hdgErrors(solveHDG(meshOf(N), { k: key.k, tau: key.tau, f: S.f, g: S.u }), S.u, S.grad);
      rows.push({ h: 1 / N, ...e });
      CV = { ...key, rows };
      draw();
      if (i < Ns.length) setTimeout(step, 0);
      else if (Math.abs(key.tau - 1) < 1e-9) {
        const r = pairwiseRates(rows.map((x) => x.h), rows.map((x) => x.ustar));
        selfCheck('hdg: u* superconvergence (tau = 1)', r[r.length - 1] > key.k + 1.6);
      }
    };
    setTimeout(step, 0);
  }
  function draw() {
    if (!R) return;
    const T = theme(), C = seriesColors(), { mesh, H } = R, [lo, hi] = R.range, ctx = s1.ctx;
    const view = new View2D(s1, [0, 1, 0, 1], { equal: true, pad: [10, 72, 10, 10] });
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    const deg = st.view === 'u' ? H.k : H.k + 1, ev = brokenEvaluator(deg, st.view === 'u' ? H.u : H.ustar);
    drawTriField(s1, view, mesh.nodes, mesh.tris, (t, l0, l1, l2) => ev(t, l1, l2), { cmap: 'viridis', lo, hi });
    if (st.traces) {
      const { topo } = H, ne = H.k + 1;
      ctx.save(); ctx.lineCap = 'butt';
      // dark casing first, so that the trace strips stand out from the field underneath
      drawTriMesh(ctx, view, mesh.nodes, mesh.tris, { color: 'rgba(0,0,0,0.55)', width: 5.5 });
      ctx.lineWidth = 3.5;
      for (let e = 0; e < topo.nEdge; e++) {
        // global edge parameter s runs along K⁻'s local edge (dgtri.js convention)
        const t = topo.edgeTris[2 * e], kk = topo.edgeLocal[2 * e];
        const va = mesh.tris[3 * t + EDGE_VERTS[kk][0]], vb = mesh.tris[3 * t + EDGE_VERTS[kk][1]];
        const xa = mesh.nodes[2 * va], ya = mesh.nodes[2 * va + 1], xb = mesh.nodes[2 * vb], yb = mesh.nodes[2 * vb + 1];
        const nseg = 8;
        for (let j = 0; j < nseg; j++) {
          const s0 = j / nseg, s1v = (j + 1) / nseg, sm = (s0 + s1v) / 2;
          let lam = 0; for (let a = 0; a < ne; a++) lam += H.lambda[e * ne + a] * edgeMode(a, sm);
          ctx.strokeStyle = colorOf('viridis', lam, lo, hi);
          ctx.beginPath(); ctx.moveTo(view.X(xa + s0 * (xb - xa)), view.Y(ya + s0 * (yb - ya)));
          ctx.lineTo(view.X(xa + s1v * (xb - xa)), view.Y(ya + s1v * (yb - ya))); ctx.stroke();
        }
      }
      ctx.restore();
    }
    colorbar(ctx, 'viridis', s1.w - 62, 14, 12, s1.h - 28, lo, hi, { ink: T.soft });
    // convergence plot
    const rows = CV && CV.k === st.k ? CV.rows : [];
    const all = rows.flatMap((r) => [r.u, r.q, r.ustar]).filter((v) => v > 0);
    const P = new Plot(s2, { xlim: [0.045, 0.7], ylim: all.length ? [Math.min(...all) / 4, Math.max(...all) * 3] : [1e-8, 1], xlog: true, ylog: true, xlabel: 'h = 1/N' });
    P.frame();
    const series = [['u', 'u_h', C[0]], ['q', 'q_h', C[2]], ['ustar', 'u*_h', C[1]]];
    for (const [key, , col] of series) {
      P.line(rows.map((r) => r.h), rows.map((r) => r[key]), { color: col, width: 2 });
      P.points(rows.map((r) => r.h), rows.map((r) => r[key]), { color: col, r: 3 });
    }
    if (rows.length >= 3) {
      P.slopeTriangle(rows[1].h, rows[1].u * 0.25, st.k + 1, 0.5, { color: C[0] });
      P.slopeTriangle(rows[1].h, rows[1].ustar * 0.2, st.k + 2, 0.5, { color: C[1] });
    }
    P.legend(series.map(([, lab, col]) => ({ label: lab, color: col })), 'br');
    let rates = '';
    if (rows.length >= 2) {
      const f = (key) => { const r = pairwiseRates(rows.map((x) => x.h), rows.map((x) => x[key])); return r[r.length - 1].toFixed(2); };
      rates = `\nlast observed rates (τ = ${fmt(CV.tau, 2)}):\n  u_h ${f('u')}, q_h ${f('q')}, u*_h ${f('ustar')}\n  expected ${st.k + 1}, ${st.k + 1}, ${st.k + 2} (τ = O(1))`;
    }
    out.set(`global trace system: ${H.nTrace} unknowns\n(local unknowns condensed: ${H.nLocal})\n`
      + `‖u − u_h‖  = ${fmt(R.e.u)}\n‖q − q_h‖  = ${fmt(R.e.q)}\n‖u − u*_h‖ = ${fmt(R.e.ustar)}\nsolve: ${R.ms.toFixed(0)} ms${rates}`);
  }
  s1.onResize(draw); s2.onResize(draw);
  solve(); conv();
}

/* ------------------------------------------------------------------ */
/* W5: global system sizes                                              */
/* ------------------------------------------------------------------ */
function sizesWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('size of the global system vs polynomial degree $p$ (log scale)');
  const s1 = createCanvas(p1, { aspect: 0.55 });
  const p2 = L.panel('', '1 1 100%');
  const tbl = h('div', ''); tbl.style.overflowX = 'auto'; p2.appendChild(tbl);
  const st = { N: 16, metric: 'n' };
  slider(L.controls, { label: 'mesh $N$ ($2N^2$ triangles)', min: 4, max: 16, step: 1, value: st.N, onInput: (v) => { st.N = v; later(); } });
  segmented(L.controls, { label: 'count', options: [{ value: 'n', label: 'unknowns' }, { value: 'nnz', label: 'non-zeros' }], value: st.metric, onChange: (v) => { st.metric = v; draw(); } });
  const out = readout(L.controls);
  let data = null, rt0 = null;
  function compute() {
    const m = triGrid(st.N, st.N);
    data = [1, 2, 3, 4, 5, 6].map((p) => ({ p, ...systemSizes(m, p) }));
    const s1v = data[0];
    // hybridized RT0 = HDG pattern with one unknown per interior edge (p = 0)
    rt0 = { n: s1v.nIntEdge, nnz: s1v.hdg.nnz / 4 };
    const hy = solveHybridRT0(m, { f: () => 1 });
    selfCheck('sizes: hybrid RT0 unknowns', hy.nLam === rt0.n);
    selfCheck('sizes: HDG < SIPG for all p', data.every((d) => d.hdg.n < d.dg.n));
    draw();
  }
  const later = debounce(compute, 60);
  function draw() {
    if (!data) return;
    const C = seriesColors(), T = theme(), k = st.metric;
    const vals = data.flatMap((d) => [d.cg[k], d.dg[k], d.hdg[k]]).concat([rt0[k]]);
    const P = new Plot(s1, { xlim: [-0.3, 6.3], ylim: [Math.min(...vals) / 2, Math.max(...vals) * 2], ylog: true, xlabel: 'polynomial degree p' });
    P.frame();
    const ser = [['cg', 'CG (continuous P_p)', C[2]], ['dg', 'SIPG', C[0]], ['hdg', 'HDG (condensed traces)', C[1]]];
    for (const [key, , col] of ser) {
      P.line(data.map((d) => d.p), data.map((d) => d[key][k]), { color: col, width: 2.2 });
      P.points(data.map((d) => d.p), data.map((d) => d[key][k]), { color: col });
    }
    P.points([0], [rt0[k]], { color: T.ink, r: 4.5 });
    P.text(0, rt0[k], '  hybrid RT0', { color: T.ink, baseline: 'bottom', dy: -4 });
    P.legend(ser.map(([, lab, col]) => ({ label: lab, color: col })), 'br');
    let html = '<table class="data"><tr><th>p</th><th>CG</th><th>SIPG</th><th>HDG</th><th>HDG/CG</th><th>HDG/SIPG</th></tr>';
    for (const d of data) html += `<tr><td>${d.p}</td><td>${d.cg[k]}</td><td>${d.dg[k]}</td><td>${d.hdg[k]}</td><td class="${d.hdg[k] < d.cg[k] ? 'good' : ''}">${(d.hdg[k] / d.cg[k]).toFixed(2)}</td><td class="good">${(d.hdg[k] / d.dg[k]).toFixed(2)}</td></tr>`;
    tbl.innerHTML = `${html}</table>`;
    out.set(`${k === 'n' ? 'unknowns' : 'non-zeros'} on ${2 * st.N * st.N} triangles\ninterior edges: ${data[0].nIntEdge}\nHDG: (p+1)·#interior edges\nhybrid RT0: ${rt0.n} unknowns`);
  }
  s1.onResize(draw);
  compute();
}

initChapter(() => {
  mount('w-rt0', rt0Widget);
  mount('w-condense', condenseWidget);
  mount('w-hybrid', hybridWidget);
  mount('w-hdg', hdgWidget);
  mount('w-sizes', sizesWidget);
});
