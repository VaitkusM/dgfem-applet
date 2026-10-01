/**
 * @file Chapter 11 — Immersed boundary methods.
 *
 * Widgets:
 *  - w-geometry : level set on a background triangle grid: inside / cut / outside elements, small cuts,
 *                 linear-cut interface segments and sub-triangles (drag the shape)
 *  - w-cutfem   : CutFEM P1 (Nitsche + ghost penalty): solution / error, and κ(A)·h² versus a domain shift
 *                 scanned live with and without ghost penalty
 *  - w-sbm      : Shifted Boundary Method: surrogate domain and boundary, distance vectors, shift on/off, error
 *  - w-web      : WEB-splines: inner/outer B-splines, extension stencils (click), weight function, solution
 *  - w-conv     : convergence of the three methods side by side
 */
import { initChapter, mount, selfCheck } from '../../lib/ui/chapter.js';
import { widgetLayout, slider, select, checkbox, segmented, readout, fmt } from '../../lib/ui/controls.js';
import { createCanvas, View2D, theme, onPointer, arrow, seriesColors } from '../../lib/viz/canvas.js';
import { drawFunction, drawTriField } from '../../lib/viz/field2d.js';
import { colorbar } from '../../lib/viz/colormap.js';
import { drawContours } from '../../lib/viz/contour.js';
import { drawTriMesh } from '../../lib/viz/meshdraw.js';
import { Plot } from '../../lib/viz/plot1d.js';
import { makeShape } from '../../lib/core/mesh/levelset.js';
import { triGrid } from '../../lib/core/mesh/structured.js';
import { classifyMesh, INSIDE, CUT, OUTSIDE } from '../../lib/core/quad/cut.js';
import { solveCutFEM, errorsCutFEM } from '../../lib/core/immersed/cutfem.js';
import { solveSBM, errorsSBM } from '../../lib/core/immersed/sbm.js';
import {
  solveWEB, errorsWEB, evalWEB, webSpace, weightFunction, blendedExtension, diagScaled, cellBsplines,
  BS_INNER, BS_OUTER, CELL_IN, CELL_CUT,
} from '../../lib/core/immersed/webspline.js';
import { conditionEstimate } from '../../lib/core/la/eig.js';
import { csrSymmetryError } from '../../lib/core/la/sparse.js';
import { POISSON_MMS } from '../../lib/core/verify/mms.js';
import { fitRate } from '../../lib/core/verify/rates.js';

const MMS = POISSON_MMS.wave; // u = cos(πx) e^y, defined everywhere
const BOX = [0, 1, 0, 1];
const SHAPES = [{ value: 'flower', label: 'flower r < R(1 + a cos kθ)' }, { value: 'circle', label: 'circle (exact SDF)' }];

/** Shape from widget state: {kind, cx, cy, R?, rot?}. Flower: R = 0.3, a = 0.2, k = 5. */
function shapeOf(st) {
  return st.kind === 'flower'
    ? makeShape({ kind: 'flower', cx: st.cx, cy: st.cy, R: 0.3, a: 0.2, k: 5, rot: st.rot ?? 0 })
    : makeShape({ kind: 'circle', cx: st.cx, cy: st.cy, R: st.R ?? 0.33 });
}

/** Stroke the true boundary Γ. */
function drawGamma(ctx, view, ls, o = {}) {
  const P = ls.polyline(400);
  ctx.save();
  ctx.strokeStyle = o.color || theme().ink; ctx.lineWidth = o.width ?? 1.5; ctx.setLineDash(o.dash || []);
  ctx.beginPath();
  for (let i = 0; i < P.length / 2; i++) { const X = view.X(P[2 * i]), Y = view.Y(P[2 * i + 1]); if (i) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }
  ctx.stroke();
  ctx.restore();
}

/** Fill / stroke a polygon given in world coordinates. */
function poly(ctx, view, pts, fill, stroke, width = 1) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(view.X(x), view.Y(y)) : ctx.moveTo(view.X(x), view.Y(y))));
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.stroke(); }
}

const triPts = (mesh, t) => [0, 1, 2].map((k) => { const v = mesh.tris[3 * t + k]; return [mesh.nodes[2 * v], mesh.nodes[2 * v + 1]]; });
const alpha = (hexOrRgb, a) => {
  // theme colours are #rrggbb
  if (hexOrRgb.startsWith('#') && hexOrRgb.length === 7) {
    const r = parseInt(hexOrRgb.slice(1, 3), 16), g = parseInt(hexOrRgb.slice(3, 5), 16), b = parseInt(hexOrRgb.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${a})`;
  }
  return hexOrRgb;
};

/** Drag the domain centre on a canvas (clamped so that Ω stays inside the box). */
function dragCentre(surf, getView, st, redraw) {
  let start = null;
  onPointer(surf.canvas, {
    down: (x, y) => { const v = getView(); start = { x: v.invX(x), y: v.invY(y), cx: st.cx, cy: st.cy }; },
    move: (x, y) => {
      if (!start) return;
      const v = getView();
      st.cx = Math.max(0.36, Math.min(0.64, start.cx + v.invX(x) - start.x));
      st.cy = Math.max(0.36, Math.min(0.64, start.cy + v.invY(y) - start.y));
      redraw();
    },
    up: () => { start = null; },
  });
}

/* ------------------------------------------------------------------ */
/* W1: geometry, classification, small cuts, cut quadrature             */
/* ------------------------------------------------------------------ */
function geometryWidget(fig) {
  const L = widgetLayout(fig);
  const p = L.panel('background mesh — drag to move the domain');
  const s = createCanvas(p, { aspect: 1 });
  const st = { kind: 'flower', cx: 0.5, cy: 0.5, R: 0.32, rot: 0, N: 12, thr: 0.05, levels: 0, phi: false };
  select(L.controls, { label: 'domain $\\Omega = \\{\\phi < 0\\}$', value: st.kind, options: SHAPES, onChange: (v) => { st.kind = v; draw(); } });
  slider(L.controls, { label: 'cells per side $1/h$', min: 4, max: 32, step: 1, value: st.N, onInput: (v) => { st.N = v; draw(); } });
  slider(L.controls, { label: 'circle radius $R$', min: 0.15, max: 0.4, step: 0.005, value: st.R, onInput: (v) => { st.R = v; draw(); } });
  slider(L.controls, { label: 'flower rotation $\\rho$', min: 0, max: 1.26, step: 0.01, value: st.rot, onInput: (v) => { st.rot = v; draw(); } });
  slider(L.controls, { label: 'small-cut threshold on $\\alpha_K$', min: 1e-4, max: 0.3, log: true, value: st.thr, format: (v) => fmt(v, 2), onInput: (v) => { st.thr = v; draw(); } });
  slider(L.controls, { label: 'cut refinement levels', min: 0, max: 3, step: 1, value: 0, onInput: (v) => { st.levels = v; draw(); } });
  checkbox(L.controls, { label: 'show $\\phi$ (colour + contours)', value: st.phi, onChange: (v) => { st.phi = v; draw(); } });
  const out = readout(L.controls);
  let view;
  function draw() {
    const T = theme(), ctx = s.ctx, ls = shapeOf(st), N = st.N, h = 1 / N;
    view = new View2D(s, BOX, { equal: true, pad: 6 });
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s.w, s.h);
    if (st.phi) {
      drawFunction(s, view, (x, y) => (x < 0 || x > 1 || y < 0 || y > 1 ? NaN : ls.phi(x, y)), { cmap: 'rdbu', lo: -0.35, hi: 0.35, step: 3 });
      drawContours(ctx, view, ls.phi, [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4], { n: 90, color: 'rgba(0,0,0,0.35)', width: 0.8 });
    }
    const mesh = triGrid(N, N), C = classifyMesh(mesh, ls.phi, { levels: st.levels, lip: ls.isSDF ? 1 : 2 });
    let nIn = 0, nCut = 0, nSmall = 0, amin = 1, area = 0, len = 0;
    for (let t = 0; t < mesh.nTri; t++) {
      const P = triPts(mesh, t);
      if (C.cls[t] === INSIDE) { nIn++; area += 0.5 * h * h; if (!st.phi) poly(ctx, view, P, alpha(T.accent3, 0.28)); continue; }
      if (C.cls[t] !== CUT) continue;
      nCut++; const cut = C.cuts[t];
      area += cut.area; len += cut.length; amin = Math.min(amin, C.frac[t]);
      const small = C.frac[t] < st.thr;
      if (small) nSmall++;
      poly(ctx, view, P, alpha(small ? T.accent2 : T.accent, small ? 0.55 : 0.18));
      for (let q = 0; q < cut.tris.length / 6; q++) {
        const c = cut.tris.slice(6 * q, 6 * q + 6);
        poly(ctx, view, [[c[0], c[1]], [c[2], c[3]], [c[4], c[5]]], alpha(T.accent, 0.3), alpha(T.accent, 0.75), 0.6);
      }
    }
    drawTriMesh(ctx, view, mesh.nodes, mesh.tris, { color: T.faint, width: 0.6, alpha: 0.7 });
    drawGamma(ctx, view, ls, { color: T.accent4, width: 1.6, dash: [5, 4] });
    // Γ_h segments (linear cut) on top
    ctx.save(); ctx.strokeStyle = T.ink; ctx.lineWidth = 2.4; ctx.beginPath();
    for (let t = 0; t < mesh.nTri; t++) if (C.cls[t] === CUT) {
      const g = C.cuts[t].segs;
      for (let q = 0; q < g.length / 6; q++) { ctx.moveTo(view.X(g[6 * q]), view.Y(g[6 * q + 1])); ctx.lineTo(view.X(g[6 * q + 2]), view.Y(g[6 * q + 3])); }
    }
    ctx.stroke(); ctx.restore();
    ctx.fillStyle = T.ink; ctx.beginPath(); ctx.arc(view.X(st.cx), view.Y(st.cy), 4, 0, 7); ctx.fill();
    const eA = (area - ls.area) / ls.area, eL = (len - ls.perimeter) / ls.perimeter;
    out.set(`elements: <b>${nIn}</b> inside, <b>${nCut}</b> cut\nsmall cuts (α_K < ${fmt(st.thr, 2)}): <b>${nSmall}</b>\nsmallest α_K = ${fmt(amin)}\n`
      + `|Ω_h|/|Ω| − 1 = ${fmt(eA)}\n|Γ_h|/|Γ| − 1 = ${fmt(eL)}\n`
      + `explicit FV on the smallest cut cell:\nΔt ≈ α_min·Δt_full = ${fmt(amin)}·Δt_full`);
    selfCheck('geometry: |Ω_h| close to |Ω|', Math.abs(eA) < 0.06 * (12 / N) ** 2 + 1e-12);
  }
  dragCentre(s, () => view, st, draw);
  s.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W2: CutFEM with/without ghost penalty, κ scan over domain shifts     */
/* ------------------------------------------------------------------ */
function cutfemWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('$u_h$ on the active mesh (ghost-penalty faces in purple)');
  const s1 = createCanvas(p1, { aspect: 1 });
  const p2 = L.panel('$\\kappa(A)\\,h^2$ while the domain slides by one cell');
  const s2 = createCanvas(p2, { aspect: 1 });
  const st = { kind: 'circle', N: 12, ghost: true, gammaG: 0.1, s: 0.3, view: 'sol' };
  select(L.controls, { label: 'domain', value: st.kind, options: SHAPES, onChange: (v) => { st.kind = v; restartScan(); draw(); } });
  slider(L.controls, { label: 'cells per side $1/h$', min: 6, max: 24, step: 1, value: st.N, onChange: (v) => { st.N = v; restartScan(); }, onInput: (v) => { st.N = v; draw(); } });
  slider(L.controls, { label: 'domain shift $s$ (cells)', min: 0, max: 1, step: 0.005, value: st.s, onInput: (v) => { st.s = v; draw(); } });
  checkbox(L.controls, { label: 'ghost penalty on', value: st.ghost, onChange: (v) => { st.ghost = v; draw(); } });
  slider(L.controls, { label: 'ghost penalty $\\gamma_g$', min: 1e-3, max: 1, log: true, value: st.gammaG, format: (v) => fmt(v, 2), onChange: (v) => { st.gammaG = v; restartScan(); }, onInput: (v) => { st.gammaG = v; draw(); } });
  segmented(L.controls, { label: 'colour shows', value: st.view, options: [{ value: 'sol', label: '$u_h$' }, { value: 'err', label: '$|u - u_h|$' }], onChange: (v) => { st.view = v; draw(); } });
  const out = readout(L.controls);
  const centre = (sv, N) => ({ cx: 0.47 + sv / N, cy: 0.48 + 0.6 * sv / N });
  const scan = { key: '', s: [], on: [], off: [], timer: 0 };
  function restartScan() {
    clearTimeout(scan.timer);
    scan.key = `${st.kind}|${st.N}|${st.gammaG}`; scan.s = []; scan.on = []; scan.off = [];
    const M = 60;
    const step = () => {
      const t0 = performance.now();
      while (scan.s.length <= M && performance.now() - t0 < 30) {
        const sv = scan.s.length / M, ls = shapeOf({ kind: st.kind, ...centre(sv, st.N), rot: 0.4 });
        const o = { ls, N: st.N, f: MMS.f, g: MMS.u, gammaG: st.gammaG, solve: false };
        const kOn = conditionEstimate(solveCutFEM(o).A, { k: 40 }).kappa, kOff = conditionEstimate(solveCutFEM({ ...o, ghost: false }).A, { k: 40 }).kappa;
        scan.s.push(sv); scan.on.push(kOn / st.N ** 2); scan.off.push(kOff / st.N ** 2);
      }
      drawScan();
      if (scan.s.length <= M) scan.timer = setTimeout(step, 0);
    };
    scan.timer = setTimeout(step, 50);
  }
  let cur = null;
  function drawScan() {
    const T = theme(), C = seriesColors();
    // values above CAP (incl. exactly singular matrices, κ = ∞) are drawn at the top edge
    const CAP = 1e12, clip = (v) => (Number.isFinite(v) && v > 0 ? Math.min(v, CAP) : CAP);
    const on = scan.on.map(clip), off = scan.off.map(clip);
    const all = [...on, ...off, cur ? clip(cur.kh2) : 1];
    const hi = Math.min(CAP * 3, Math.max(10, ...all) * 3), lo = Math.min(0.05, ...all) / 2;
    const P = new Plot(s2, { xlim: [0, 1], ylim: [lo, hi], ylog: true, xlabel: 'shift s (in cells)', ylabel: 'κ(A) h²' });
    P.frame();
    P.line(scan.s, off, { color: C[1], width: 1.8 });
    P.line(scan.s, on, { color: C[2], width: 2.2 });
    P.points(scan.s, off, { color: C[1], r: 2 });
    P.vline(st.s, { color: T.ink, dash: [3, 3] });
    if (cur) P.points([st.s], [clip(cur.kh2)], { color: st.ghost ? C[2] : C[1], r: 5 });
    P.legend([{ label: 'with ghost penalty', color: C[2] }, { label: 'without', color: C[1] }], 'tr');
    if (scan.s.length > 60) {
      const mOn = Math.max(...scan.on), mOff = Math.max(...scan.off);
      selfCheck('cutfem scan: ghost penalty bounds κh²', mOn < 10 && mOff > mOn);
    }
  }
  function draw() {
    const T = theme(), ctx = s1.ctx;
    const ls = shapeOf({ kind: st.kind, ...centre(st.s, st.N), rot: 0.4 });
    const sol = solveCutFEM({ ls, N: st.N, f: MMS.f, g: MMS.u, ghost: st.ghost, gammaG: st.gammaG });
    const e = errorsCutFEM(sol, MMS.u, MMS.grad), kap = conditionEstimate(sol.A, { k: 40 }).kappa;
    cur = { kh2: kap / st.N ** 2 };
    const view = new View2D(s1, [0.1, 0.92, 0.1, 0.92], { equal: true, pad: [6, 64, 6, 6] });
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    const { mesh, cut } = sol;
    const nodal = (t, l0, l1, l2) => {
      const d = [0, 1, 2].map((k) => sol.U[sol.dofOf[mesh.tris[3 * t + k]]]);
      return l0 * d[0] + l1 * d[1] + l2 * d[2];
    };
    let lo, hi, cmap;
    if (st.view === 'sol') { lo = -2.8; hi = 2.8; cmap = 'rdbu'; }
    else {
      cmap = 'magma'; lo = 0; hi = 0;
      for (let d = 0; d < sol.nDof; d++) { const v = sol.dofVert[d]; hi = Math.max(hi, Math.abs(sol.U[d] - MMS.u(mesh.nodes[2 * v], mesh.nodes[2 * v + 1]))); }
      hi = hi || 1e-12;
    }
    drawTriField(s1, view, mesh.nodes, mesh.tris, (t, l0, l1, l2, x, y) => {
      const v = nodal(t, l0, l1, l2);
      if (st.view === 'sol') return v;
      return ls.phi(x, y) < 0 ? Math.abs(v - MMS.u(x, y)) : NaN;
    }, { cmap, lo, hi, skip: (t) => cut.cls[t] === OUTSIDE });
    drawTriMesh(ctx, view, mesh.nodes, mesh.tris, { color: T.faint, width: 0.5, alpha: 0.5, skip: (t) => cut.cls[t] === OUTSIDE });
    if (st.ghost) {
      ctx.save(); ctx.strokeStyle = T.accent4; ctx.lineWidth = 2.2; ctx.beginPath();
      for (const ed of sol.ghostFaces) {
        const a = sol.topo.edges[2 * ed], b = sol.topo.edges[2 * ed + 1];
        ctx.moveTo(view.X(mesh.nodes[2 * a]), view.Y(mesh.nodes[2 * a + 1])); ctx.lineTo(view.X(mesh.nodes[2 * b]), view.Y(mesh.nodes[2 * b + 1]));
      }
      ctx.stroke(); ctx.restore();
    }
    drawGamma(ctx, view, ls, { color: T.ink, width: 2 });
    colorbar(ctx, cmap, s1.w - 52, 12, 12, s1.h - 24, lo, hi, { ink: T.soft });
    let amin = 1;
    for (let t = 0; t < mesh.nTri; t++) if (cut.cls[t] === CUT) amin = Math.min(amin, cut.frac[t]);
    out.set(`DOFs (active vertices): ${sol.nDof}\nsmallest α_K = ${fmt(amin)}\n‖u − u_h‖_L² = ${fmt(e.L2)}\n|u − u_h|_H¹ = ${fmt(e.H1)}\n`
      + `κ(A) = ${fmt(kap)}\nκ(A)·h² = <b>${fmt(kap / st.N ** 2)}</b>`);
    selfCheck('cutfem solution finite', Number.isFinite(e.L2));
    drawScan();
  }
  s1.onResize(draw); s2.onResize(drawScan);
  restartScan();
  draw();
}

/* ------------------------------------------------------------------ */
/* W3: Shifted Boundary Method                                          */
/* ------------------------------------------------------------------ */
function sbmWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('surrogate domain $\\tilde\\Omega_h$, boundary $\\tilde\\Gamma_h$, distance vectors $\\mathbf d$ — drag');
  const s1 = createCanvas(p1, { aspect: 1 });
  const p2 = L.panel('error $|u - u_h|$ on $\\tilde\\Omega_h$');
  const s2 = createCanvas(p2, { aspect: 1 });
  const st = { kind: 'flower', cx: 0.5, cy: 0.5, rot: 0.3, N: 14, shift: true, alpha: 10 };
  select(L.controls, { label: 'domain', value: st.kind, options: SHAPES, onChange: (v) => { st.kind = v; draw(); } });
  slider(L.controls, { label: 'cells per side $1/h$', min: 6, max: 40, step: 1, value: st.N, onInput: (v) => { st.N = v; draw(); } });
  checkbox(L.controls, { label: 'Taylor shift $u + \\nabla u\\cdot\\mathbf d$', value: st.shift, onChange: (v) => { st.shift = v; draw(); } });
  slider(L.controls, { label: 'penalty $\\alpha$', min: 1, max: 100, log: true, value: st.alpha, format: (v) => fmt(v, 2), onInput: (v) => { st.alpha = v; draw(); } });
  const out = readout(L.controls);
  let view;
  function draw() {
    const T = theme(), ctx = s1.ctx, ls = shapeOf(st);
    const sol = solveSBM({ ls, N: st.N, f: MMS.f, g: MMS.u, shift: st.shift, alpha: st.alpha });
    const e = errorsSBM(sol, MMS.u, MMS.grad), { mesh } = sol;
    view = new View2D(s1, BOX, { equal: true, pad: 6 });
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    for (let t = 0; t < mesh.nTri; t++) if (sol.inSur[t]) poly(ctx, view, triPts(mesh, t), alpha(T.accent3, 0.3));
    drawTriMesh(ctx, view, mesh.nodes, mesh.tris, { color: T.faint, width: 0.5, alpha: 0.6 });
    drawGamma(ctx, view, ls, { color: T.ink, width: 1.8 });
    // surrogate boundary Γ̃_h + one distance vector d per edge (from the edge midpoint x̃ to M(x̃) on Γ)
    let dmax = 0;
    ctx.save(); ctx.lineCap = 'round'; ctx.strokeStyle = T.accent; ctx.lineWidth = 3; ctx.beginPath();
    for (const ed of sol.bEdges) {
      const a = sol.topo.edges[2 * ed], b = sol.topo.edges[2 * ed + 1];
      ctx.moveTo(view.X(mesh.nodes[2 * a]), view.Y(mesh.nodes[2 * a + 1])); ctx.lineTo(view.X(mesh.nodes[2 * b]), view.Y(mesh.nodes[2 * b + 1]));
    }
    ctx.stroke();
    ctx.strokeStyle = ctx.fillStyle = T.accent2; ctx.lineWidth = 1.3;
    for (let k = 0; k < sol.bEdges.length; k++) {
      // 3 Gauss points per edge are stored consecutively; the middle one is the edge midpoint
      const q = 3 * k + 1, x = sol.qPts[2 * q], y = sol.qPts[2 * q + 1], px = sol.qCP[2 * q], py = sol.qCP[2 * q + 1];
      dmax = Math.max(dmax, Math.hypot(px - x, py - y));
      arrow(ctx, view.X(x), view.Y(y), view.X(px), view.Y(py), 5);
    }
    ctx.restore();
    // error field
    const v2 = new View2D(s2, BOX, { equal: true, pad: [6, 64, 6, 6] }), c2 = s2.ctx;
    c2.fillStyle = T.bg; c2.fillRect(0, 0, s2.w, s2.h);
    let hi = 0;
    for (let v = 0; v < mesh.nVert; v++) if (sol.dofOf[v] >= 0) hi = Math.max(hi, Math.abs(sol.U[sol.dofOf[v]] - MMS.u(mesh.nodes[2 * v], mesh.nodes[2 * v + 1])));
    hi = hi || 1e-12;
    drawTriField(s2, v2, mesh.nodes, mesh.tris, (t, l0, l1, l2, x, y) => {
      const d = [0, 1, 2].map((k) => sol.U[sol.dofOf[mesh.tris[3 * t + k]]]);
      return Math.abs(l0 * d[0] + l1 * d[1] + l2 * d[2] - MMS.u(x, y));
    }, { cmap: 'magma', lo: 0, hi, skip: (t) => !sol.inSur[t] });
    drawGamma(c2, v2, ls, { color: T.ink, width: 1.2, dash: [4, 3] });
    colorbar(c2, 'magma', s2.w - 52, 12, 12, s2.h - 24, 0, hi, { ink: T.soft });
    const sym = csrSymmetryError(sol.A);
    out.set(`DOFs: ${sol.nDof}   surrogate edges: ${sol.bEdges.length}\nmax |d| / h = ${fmt(dmax * st.N)}\n‖u − u_h‖_L²(Ω̃) = ${fmt(e.L2)}\n|u − u_h|_H¹(Ω̃) = ${fmt(e.H1)}\n`
      + `matrix ${sym < 1e-12 ? 'symmetric' : `non-symmetric (rel. asym. ${fmt(sym, 2)})`}`);
    selfCheck('sbm solution finite', Number.isFinite(e.L2) && !sol.singular);
  }
  dragCentre(s1, () => view, st, draw);
  s1.onResize(draw); s2.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W4: WEB-splines                                                      */
/* ------------------------------------------------------------------ */
function webWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('B-splines (dot = centre of the support): inner ● / outer ○ — click one');
  const s1 = createCanvas(p1, { aspect: 1 });
  const p2 = L.panel('field');
  const s2 = createCanvas(p2, { aspect: 1 });
  const st = { kind: 'circle', cx: 0.5, cy: 0.5, rot: 0.3, n: 2, N: 10, view: 'sol', extend: true, sel: -1 };
  select(L.controls, { label: 'domain', value: st.kind, options: SHAPES, onChange: (v) => { st.kind = v; st.sel = -1; rebuild(); } });
  segmented(L.controls, { label: 'degree $n$', value: '2', options: [{ value: '1', label: '1' }, { value: '2', label: '2' }, { value: '3', label: '3' }], onChange: (v) => { st.n = +v; st.sel = -1; rebuild(); } });
  slider(L.controls, { label: 'cells per side $1/h$', min: 6, max: 20, step: 1, value: st.N, onInput: (v) => { st.N = v; st.sel = -1; rebuild(); } });
  select(L.controls, {
    label: 'right panel shows', value: st.view,
    options: [{ value: 'sol', label: 'solution u_h' }, { value: 'err', label: 'error |u − u_h|' }, { value: 'w', label: 'weight function w' }, { value: 'basis', label: 'selected WEB-spline B_i' }],
    onChange: (v) => { st.view = v; drawField(); },
  });
  checkbox(L.controls, { label: 'extension (off: keep outer B-splines as unknowns)', value: st.extend, onChange: (v) => { st.extend = v; rebuild(); } });
  const out = readout(L.controls);
  let sp, sol, err, kap, view, ls;
  function rebuild() {
    ls = shapeOf(st);
    sp = webSpace(ls, { n: st.n, N: st.N });
    const W = weightFunction(ls);
    sol = solveWEB(ls, { n: st.n, N: st.N, f: MMS.f, gt: blendedExtension(W, MMS.u, MMS.grad), extend: st.extend });
    err = errorsWEB(sol, MMS.u, MMS.grad);
    kap = conditionEstimate(diagScaled(sol.A), { k: 50 }).kappa;
    selfCheck('web solution finite', Number.isFinite(err.L2) && !sol.singular);
    draw(); drawField();
  }
  const centreOf = (idx) => { const [k1, k2] = sp.grid.kOf(idx), h = sp.grid.h; return [sp.grid.x0 + (k1 + (st.n + 1) / 2) * h, sp.grid.y0 + (k2 + (st.n + 1) / 2) * h]; };
  function draw() {
    const T = theme(), ctx = s1.ctx, N = st.N, h = 1 / N;
    view = new View2D(s1, BOX, { equal: true, pad: 6 });
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    for (let c = 0; c < N * N; c++) {
      const l1 = c % N, l2 = Math.floor(c / N);
      if (sp.cellCls[c] === CELL_IN || sp.cellCls[c] === CELL_CUT) {
        poly(ctx, view, [[l1 * h, l2 * h], [(l1 + 1) * h, l2 * h], [(l1 + 1) * h, (l2 + 1) * h], [l1 * h, (l2 + 1) * h]],
          sp.cellCls[c] === CELL_IN ? alpha(T.accent3, 0.22) : alpha(T.accent2, 0.13));
      }
    }
    ctx.save(); ctx.strokeStyle = T.faint; ctx.globalAlpha = 0.6; ctx.lineWidth = 0.6; ctx.beginPath();
    for (let i = 0; i <= N; i++) { ctx.moveTo(view.X(i * h), view.Y(0)); ctx.lineTo(view.X(i * h), view.Y(1)); ctx.moveTo(view.X(0), view.Y(i * h)); ctx.lineTo(view.X(1), view.Y(i * h)); }
    ctx.stroke(); ctx.restore();
    drawGamma(ctx, view, ls, { color: T.ink, width: 1.8 });
    // selection: support rectangle + stencil
    const selTxt = [];
    const hl = new Map();
    if (st.sel >= 0) {
      const [k1, k2] = sp.grid.kOf(st.sel);
      ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = T.accent4; ctx.lineWidth = 1.5;
      ctx.strokeRect(view.X(k1 * h), view.Y((k2 + st.n + 1) * h), (st.n + 1) * h * view.sx, (st.n + 1) * h * view.sy);
      ctx.restore();
      if (sp.bsCls[st.sel] === BS_OUTER) {
        const E = sp.ext.get(st.sel);
        let sum = 0;
        E.ids.forEach((id, q) => { hl.set(id, E.e[q]); sum += E.e[q]; });
        selTxt.push(`outer j = (${k1}, ${k2})`, `array ℓ = (${E.ell[0]}, ${E.ell[1]}) + {0..${st.n}}²`, `Σ_i e_ij = ${fmt(sum, 6)} (constants reproduced)`);
        selfCheck('web: extension weights sum to 1', Math.abs(sum - 1) < 1e-9);
      } else if (sp.bsCls[st.sel] === BS_INNER) {
        for (const [j, E] of sp.ext) { const q = Array.from(E.ids).indexOf(st.sel); if (q >= 0) hl.set(j, E.e[q]); }
        selTxt.push(`inner i = (${k1}, ${k2})`, `extended by ${hl.size} outer B-spline(s) J(i)`, `w(x_i) = ${fmt(sp.wx[sp.innerId[st.sel]])}`);
      }
    }
    // markers
    const fs = Math.max(8, Math.min(11, view.sx * h * 0.42));
    for (const r of sp.relevant) {
      const [x, y] = centreOf(r), X = view.X(x), Y = view.Y(y);
      const inner = sp.bsCls[r] === BS_INNER;
      ctx.beginPath(); ctx.arc(X, Y, r === st.sel ? 6 : 3.6, 0, 7);
      if (inner) { ctx.fillStyle = T.accent3; ctx.fill(); } else { ctx.strokeStyle = T.accent2; ctx.lineWidth = 1.8; ctx.stroke(); }
    }
    if (st.sel >= 0) {
      const [sx, sy] = centreOf(st.sel);
      ctx.save(); ctx.font = `600 ${fs}px ${T.ui}`; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      for (const [id, e] of hl) {
        const [x, y] = centreOf(id);
        ctx.strokeStyle = alpha(T.accent4, 0.7); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(view.X(sx), view.Y(sy)); ctx.lineTo(view.X(x), view.Y(y)); ctx.stroke();
        ctx.fillStyle = T.accent4; ctx.beginPath(); ctx.arc(view.X(x), view.Y(y), 4.5, 0, 7); ctx.fill();
        ctx.fillStyle = T.ink; ctx.fillText(fmt(e, 2), view.X(x), view.Y(y) - 5);
      }
      ctx.restore();
    }
    const nOut = sp.relevant.length - sp.inner.length;
    out.set(`inner B-splines (unknowns): <b>${sp.inner.length}</b>\nouter B-splines: <b>${nOut}</b>${st.extend ? ' (extended)' : ' (kept as unknowns!)'}\n`
      + `‖u − u_h‖_L² = ${fmt(err.L2)}\n|u − u_h|_H¹ = ${fmt(err.H1)}\nκ(Jacobi-scaled A)·h² = ${fmt(kap / st.N ** 2)}`
      + (selTxt.length ? `\n\n${selTxt.join('\n')}` : '\n\nclick a dot on the left'));
  }
  function drawField() {
    const T = theme(), ctx = s2.ctx, v2 = new View2D(s2, BOX, { equal: true, pad: [6, 64, 6, 6] });
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s2.w, s2.h);
    const W = sp.weight;
    let f, lo, hi, cmap = 'viridis';
    if (st.view === 'w') { f = (x, y) => W.w(x, y); lo = 0; hi = 0; for (let i = 0; i <= 40; i++) for (let j = 0; j <= 40; j++) hi = Math.max(hi, W.w(i / 40, j / 40)); }
    else if (st.view === 'sol') { f = (x, y) => evalWEB(sol, x, y).u; lo = -2.8; hi = 2.8; cmap = 'rdbu'; }
    else if (st.view === 'err') {
      f = (x, y) => Math.abs(evalWEB(sol, x, y).u - MMS.u(x, y)); cmap = 'magma'; lo = 0; hi = 0;
      for (let i = 0; i <= 50; i++) for (let j = 0; j <= 50; j++) { const x = i / 50, y = j / 50; if (ls.phi(x, y) < 0) hi = Math.max(hi, f(x, y)); }
      hi = hi || 1e-12;
    } else {
      // selected WEB-spline B_i = (w / w(x_i)) (b_i + Σ_j e_ij b_j)
      const i = st.sel >= 0 && sp.bsCls[st.sel] === BS_INNER ? st.sel : sp.inner[Math.floor(sp.inner.length / 2)];
      const coef = new Map([[i, 1]]);
      for (const [j, E] of sp.ext) { const q = Array.from(E.ids).indexOf(i); if (q >= 0) coef.set(j, E.e[q]); }
      const wi = sp.wx[sp.innerId[i]];
      f = (x, y) => {
        const N = st.N, c = Math.min(N - 1, Math.floor(x * N)) + N * Math.min(N - 1, Math.floor(y * N));
        const B = cellBsplines(sp, c, x, y);
        let v = 0;
        for (let a = 0; a < B.idx.length; a++) v += (coef.get(B.idx[a]) || 0) * B.v[a];
        return W.w(x, y) / wi * v;
      };
      cmap = 'rdbu'; hi = 0;
      for (let a = 0; a <= 60; a++) for (let b = 0; b <= 60; b++) { const x = a / 60, y = b / 60; if (ls.phi(x, y) < 0) hi = Math.max(hi, Math.abs(f(x, y))); }
      hi = hi || 1; lo = -hi;
    }
    drawFunction(s2, v2, (x, y) => (x < 0 || x > 1 || y < 0 || y > 1 || ls.phi(x, y) >= 0 ? NaN : f(x, y)), { cmap, lo, hi, step: 2 });
    drawGamma(ctx, v2, ls, { color: T.ink, width: 1.5 });
    colorbar(ctx, cmap, s2.w - 52, 12, 12, s2.h - 24, lo, hi, { ink: T.soft });
    const lab = { w: 'weight function w', sol: 'WEB-spline solution u_h', err: 'error |u − u_h|', basis: 'WEB-spline B_i (click an inner dot)' }[st.view];
    p2.querySelector('.canvas-label').textContent = lab;
  }
  onPointer(s1.canvas, {
    down: (x, y) => {
      let best = -1, bd = 14;
      for (const r of sp.relevant) { const [cx, cy] = centreOf(r), d = Math.hypot(view.X(cx) - x, view.Y(cy) - y); if (d < bd) { bd = d; best = r; } }
      st.sel = best; draw(); if (st.view === 'basis') drawField();
      return false;
    },
  });
  s1.onResize(draw); s2.onResize(drawField);
  rebuild();
  // pre-select an outer B-spline so the stencil is visible at once
  const firstOuter = sp.relevant.find((r) => sp.bsCls[r] === BS_OUTER);
  if (firstOuter !== undefined) { st.sel = firstOuter; draw(); }
}

/* ------------------------------------------------------------------ */
/* W5: convergence of the three methods                                 */
/* ------------------------------------------------------------------ */
function convWidget(fig) {
  const L = widgetLayout(fig);
  const p = L.panel('error versus mesh size (log–log)');
  const s = createCanvas(p, { aspect: 0.62 });
  const st = { kind: 'flower', norm: 'L2' };
  select(L.controls, { label: 'domain', value: st.kind, options: SHAPES, onChange: (v) => { st.kind = v; start(); } });
  segmented(L.controls, { label: 'norm', value: st.norm, options: [{ value: 'L2', label: '$L^2$' }, { value: 'H1', label: '$H^1$' }], onChange: (v) => { st.norm = v; draw(); } });
  const out = readout(L.controls);
  const C = seriesColors();
  const SERIES = [
    { id: 'cut', label: 'CutFEM P1', Ns: [8, 16, 32, 64] },
    { id: 'sbm', label: 'SBM P1 (shift)', Ns: [8, 16, 32, 64] },
    { id: 'sbm0', label: 'SBM P1 (no shift)', Ns: [8, 16, 32, 64] },
    { id: 'web1', label: 'WEB n=1', Ns: [8, 16, 32, 64] },
    { id: 'web2', label: 'WEB n=2', Ns: [8, 16, 32, 64] },
    { id: 'web3', label: 'WEB n=3', Ns: [8, 16, 32] },
  ];
  const res = {};
  let timer = 0;
  function run(id, N, ls) {
    if (id === 'cut') return errorsCutFEM(solveCutFEM({ ls, N, f: MMS.f, g: MMS.u }), MMS.u, MMS.grad);
    if (id === 'sbm' || id === 'sbm0') return errorsSBM(solveSBM({ ls, N, f: MMS.f, g: MMS.u, shift: id === 'sbm' }), MMS.u, MMS.grad);
    const n = +id.slice(3), W = weightFunction(ls);
    return errorsWEB(solveWEB(ls, { n, N, f: MMS.f, gt: blendedExtension(W, MMS.u, MMS.grad) }), MMS.u, MMS.grad);
  }
  function start() {
    clearTimeout(timer);
    const ls = st.kind === 'flower' ? makeShape({ kind: 'flower', cx: 0.51, cy: 0.47, R: 0.3, a: 0.2, k: 5, rot: 0.3 }) : makeShape({ kind: 'circle', cx: 0.513, cy: 0.478, R: 0.33 });
    const jobs = [];
    for (const S of SERIES) { res[S.id] = { h: [], L2: [], H1: [] }; for (const N of S.Ns) jobs.push([S.id, N]); }
    jobs.sort((a, b) => a[1] - b[1]); // coarse first: the plot fills in progressively
    const step = () => {
      const t0 = performance.now();
      while (jobs.length && performance.now() - t0 < 40) {
        const [id, N] = jobs.shift(), e = run(id, N, ls);
        res[id].h.push(1 / N); res[id].L2.push(e.L2); res[id].H1.push(e.H1);
      }
      draw();
      if (jobs.length) timer = setTimeout(step, 0);
      else {
        selfCheck('conv: CutFEM L2 rate ≈ 2', Math.abs(fitRate(res.cut.h, res.cut.L2) - 2) < 0.3);
        selfCheck('conv: SBM L2 rate ≈ 2', Math.abs(fitRate(res.sbm.h, res.sbm.L2) - 2) < 0.4);
      }
    };
    timer = setTimeout(step, 200);
  }
  function draw() {
    const T = theme(), key = st.norm;
    const P = new Plot(s, { xlim: [1 / 90, 1 / 6], ylim: key === 'L2' ? [1e-9, 0.3] : [1e-7, 3], xlog: true, ylog: true, xlabel: 'h', ylabel: key === 'L2' ? '‖u − u_h‖_L²' : '|u − u_h|_H¹' });
    P.frame();
    const lines = [];
    SERIES.forEach((S, i) => {
      const r = res[S.id];
      if (!r || !r.h.length) return;
      const col = C[i % C.length];
      P.line(r.h, r[key], { color: col, width: 2, dash: S.id === 'sbm0' ? [5, 4] : [] });
      P.points(r.h, r[key], { color: col, r: 3 });
      const rate = r.h.length >= 2 ? fitRate(r.h, r[key], 2) : NaN;
      lines.push(`${S.label.padEnd(18)} ${Number.isFinite(rate) ? `rate ${rate.toFixed(2)}` : '…'}`);
    });
    P.legend(SERIES.map((S, i) => ({ label: S.label, color: C[i % C.length], dash: S.id === 'sbm0' ? [5, 4] : undefined })), 'br');
    out.set(`${key} rates (last two points):\n${lines.join('\n')}`);
  }
  s.onResize(draw);
  start();
  draw();
}

initChapter(() => {
  mount('w-geometry', geometryWidget);
  mount('w-cutfem', cutfemWidget);
  mount('w-sbm', sbmWidget);
  mount('w-web', webWidget);
  mount('w-conv', convWidget);
});
