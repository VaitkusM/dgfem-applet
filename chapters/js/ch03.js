/**
 * @file Chapter 3 — Continuous finite elements for the Poisson problem.
 *
 * Widgets:
 *  - w-basis    : P1/P2 basis functions on a small mesh (click a node)
 *  - w-refmap   : the affine map from the reference triangle; Jacobian, det J, gradients
 *  - w-assembly : element-by-element assembly of the global stiffness matrix (cotangent formula)
 *  - w-poisson  : full 2D Poisson solver: solution/error, sparsity (with RCM), CG iterations, convergence
 */
import { initChapter, mount, selfCheck } from '../../lib/ui/chapter.js';
import { widgetLayout, slider, select, segmented, button, buttonRow, readout, fmt, debounce, h } from '../../lib/ui/controls.js';
import { createCanvas, View2D, theme, onPointer, seriesColors, arrow } from '../../lib/viz/canvas.js';
import { drawTriField } from '../../lib/viz/field2d.js';
import { drawTriMesh, fillTri } from '../../lib/viz/meshdraw.js';
import { colorbar, range } from '../../lib/viz/colormap.js';
import { drawSpy } from '../../lib/viz/spy.js';
import { Plot } from '../../lib/viz/plot1d.js';
import { Animator } from '../../lib/viz/anim.js';
import { triGrid, triAffine } from '../../lib/core/mesh/structured.js';
import { cgSpace, assembleVolume, solvePoissonCG, errorsCG, evalCG, refBasis } from '../../lib/core/elliptic/cg.js';
import { POISSON_MMS } from '../../lib/core/verify/mms.js';
import { csrExtract, csrPermute, csrMatVec, nnz } from '../../lib/core/la/sparse.js';
import { rcm } from '../../lib/core/la/direct.js';
import { cg as cgSolve, jacobiPrecond } from '../../lib/core/la/krylov.js';
import { fitRate } from '../../lib/core/verify/rates.js';

/* ------------------------------------------------------------------ */
/* W1: basis functions                                                  */
/* ------------------------------------------------------------------ */
function basisWidget(fig) {
  const L = widgetLayout(fig);
  const panel = L.panel('click a node to see its basis function $\\varphi_j$');
  const surf = createCanvas(panel, { aspect: 0.8 });
  const mesh = triGrid(4, 4, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.25, seed: 3 });
  const st = { p: 1, dof: 12 };
  segmented(L.controls, { label: 'polynomial degree', value: '1', options: [{ value: '1', label: 'P1 (linear)' }, { value: '2', label: 'P2 (quadratic)' }], onChange: (v) => { st.p = +v; st.dof = 12; draw(); } });
  const out = readout(L.controls);
  let space, view;
  function draw() {
    const T = theme(), C = seriesColors(), ctx = surf.ctx;
    space = cgSpace(mesh, st.p);
    view = new View2D(surf, [0, 1, 0, 1], { equal: true, pad: [10, 60, 10, 10] });
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, surf.w, surf.h);
    const n = space.nLoc;
    // value of basis function `dof` on triangle t at barycentric (l0,l1,l2): reference coords (l1, l2)
    const support = new Set();
    for (let t = 0; t < mesh.nTri; t++) for (let k = 0; k < n; k++) if (space.elemDofs[n * t + k] === st.dof) support.add(t);
    drawTriField(surf, view, mesh.nodes, mesh.tris, (t, l0, l1, l2) => {
      for (let k = 0; k < n; k++) if (space.elemDofs[n * t + k] === st.dof) return refBasis(st.p, l1, l2).v[k];
      return 0;
    }, { cmap: 'rdbu', lo: -1, hi: 1 });
    drawTriMesh(ctx, view, mesh.nodes, mesh.tris, { color: T.ink, alpha: 0.5 });
    for (const t of support) {
      ctx.save(); ctx.strokeStyle = T.accent3; ctx.lineWidth = 2;
      ctx.beginPath();
      for (let k = 0; k < 3; k++) { const v = mesh.tris[3 * t + k]; const X = view.X(mesh.nodes[2 * v]), Y = view.Y(mesh.nodes[2 * v + 1]); if (k) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }
      ctx.closePath(); ctx.stroke(); ctx.restore();
    }
    for (let d = 0; d < space.nDof; d++) {
      const x = view.X(space.dofXY[2 * d]), y = view.Y(space.dofXY[2 * d + 1]);
      ctx.fillStyle = d === st.dof ? C[1] : d < mesh.nVert ? T.ink : T.soft;
      ctx.beginPath(); ctx.arc(x, y, d === st.dof ? 6 : 3.5, 0, 7); ctx.fill();
    }
    colorbar(ctx, 'rdbu', surf.w - 48, 10, 12, surf.h - 20, -1, 1, { ink: T.soft });
    const isVert = st.dof < mesh.nVert;
    out.set(`DOF ${st.dof}: ${isVert ? 'vertex' : 'edge midpoint'} node\nsupport: ${support.size} triangles\ntotal DOFs: ${space.nDof}\n${st.p === 2 ? (isVert ? 'vertex functions dip below 0' : 'edge "bubble" 4λ_aλ_b') : 'P1 = barycentric coordinate'}`);
  }
  onPointer(surf.canvas, {
    down: (x, y) => {
      let best = -1, bd = 14;
      for (let d = 0; d < space.nDof; d++) { const dd = Math.hypot(view.X(space.dofXY[2 * d]) - x, view.Y(space.dofXY[2 * d + 1]) - y); if (dd < bd) { bd = dd; best = d; } }
      if (best >= 0) { st.dof = best; draw(); }
      return false;
    },
  });
  surf.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W2: reference map                                                    */
/* ------------------------------------------------------------------ */
function refMapWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('reference triangle $\\hat K$ in $(\\hat x, \\hat y)$');
  const s1 = createCanvas(p1, { aspect: 0.9 });
  const p2 = L.panel('physical triangle $K$ — drag its vertices');
  const s2 = createCanvas(p2, { aspect: 0.9 });
  const st = { P: [[0.6, 0.5], [2.4, 0.9], [1.1, 2.2]], drag: -1 };
  const out = readout(L.controls);
  let v2;
  const lines = (draw) => { // grid lines in the reference triangle: x̂ = const, ŷ = const, x̂+ŷ = const
    for (let k = 1; k < 8; k++) { const c = k / 8; draw([c, 0], [c, 1 - c], 0); draw([0, c], [1 - c, c], 1); draw([c, 0], [0, c], 2); }
  };
  function draw() {
    const T = theme(), C = seriesColors();
    // reference
    const v1 = new View2D(s1, [-0.15, 1.15, -0.15, 1.15], { equal: true, pad: 10 });
    let ctx = s1.ctx;
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    const tri = (ctx, v, A, B, Cc, fill) => { ctx.beginPath(); ctx.moveTo(v.X(A[0]), v.Y(A[1])); ctx.lineTo(v.X(B[0]), v.Y(B[1])); ctx.lineTo(v.X(Cc[0]), v.Y(Cc[1])); ctx.closePath(); if (fill) { ctx.fillStyle = fill; ctx.fill(); } ctx.stroke(); };
    ctx.strokeStyle = T.ink; ctx.lineWidth = 2; tri(ctx, v1, [0, 0], [1, 0], [0, 1], T.elev);
    ctx.lineWidth = 1;
    lines((a, b, fam) => { ctx.strokeStyle = C[fam]; ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.moveTo(v1.X(a[0]), v1.Y(a[1])); ctx.lineTo(v1.X(b[0]), v1.Y(b[1])); ctx.stroke(); ctx.globalAlpha = 1; });
    const labels = [['v₀ (0,0)', 0, 0], ['v₁ (1,0)', 1, 0], ['v₂ (0,1)', 0, 1]];
    ctx.fillStyle = T.ink; ctx.font = `12px ${T.ui}`;
    for (const [s, x, y] of labels) { ctx.beginPath(); ctx.arc(v1.X(x), v1.Y(y), 4, 0, 7); ctx.fill(); ctx.fillText(s, v1.X(x) + 6, v1.Y(y) - 6); }
    // physical
    v2 = new View2D(s2, [0, 3, 0, 3], { equal: true, pad: 10 });
    ctx = s2.ctx;
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s2.w, s2.h);
    const [A, B, Cc] = st.P;
    const J = [B[0] - A[0], Cc[0] - A[0], B[1] - A[1], Cc[1] - A[1]];
    const det = J[0] * J[3] - J[1] * J[2];
    const map = ([x, y]) => [A[0] + J[0] * x + J[1] * y, A[1] + J[2] * x + J[3] * y];
    ctx.strokeStyle = det > 0 ? T.ink : T.accent2; ctx.lineWidth = 2; tri(ctx, v2, A, B, Cc, det > 0 ? T.elev : 'rgba(220,80,40,0.25)');
    ctx.lineWidth = 1;
    lines((a, b, fam) => { const pa = map(a), pb = map(b); ctx.strokeStyle = C[fam]; ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.moveTo(v2.X(pa[0]), v2.Y(pa[1])); ctx.lineTo(v2.X(pb[0]), v2.Y(pb[1])); ctx.stroke(); ctx.globalAlpha = 1; });
    st.P.forEach((p, i) => { ctx.fillStyle = '#fff'; ctx.strokeStyle = T.ink; ctx.beginPath(); ctx.arc(v2.X(p[0]), v2.Y(p[1]), 6, 0, 7); ctx.fill(); ctx.stroke(); ctx.fillStyle = T.ink; ctx.fillText(`v${'₀₁₂'[i]}`, v2.X(p[0]) + 8, v2.Y(p[1]) - 8); });
    // gradient of φ₁ (= λ₁): reference gradient (1,0) mapped by J^{−T}
    const Jinv = [J[3] / det, -J[1] / det, -J[2] / det, J[0] / det];
    const g1 = [Jinv[0] * 1 + Jinv[2] * 0, Jinv[1] * 1 + Jinv[3] * 0];
    const cx = (A[0] + B[0] + Cc[0]) / 3, cy = (A[1] + B[1] + Cc[1]) / 3;
    ctx.strokeStyle = ctx.fillStyle = T.accent2; ctx.lineWidth = 2;
    const gl = Math.hypot(g1[0], g1[1]) || 1, sc = 0.6 / gl;
    if (Number.isFinite(gl)) arrow(ctx, v2.X(cx), v2.Y(cy), v2.X(cx + g1[0] * sc), v2.Y(cy + g1[1] * sc), 8);
    out.set(`J = [ ${fmt(J[0])}  ${fmt(J[1])} ]\n    [ ${fmt(J[2])}  ${fmt(J[3])} ]\ndet J = ${fmt(det)}  (= 2·area${det < 0 ? ', NEGATIVE: inverted!' : ''})\n∇φ₁ = J⁻ᵀ(1,0) = (${fmt(g1[0])}, ${fmt(g1[1])})\n(orange arrow, ⟂ to edge v₀v₂)`);
    selfCheck('refmap det finite', Number.isFinite(det));
  }
  onPointer(s2.canvas, {
    down: (x, y) => { st.drag = st.P.findIndex((p) => Math.hypot(v2.X(p[0]) - x, v2.Y(p[1]) - y) < 12); return st.drag >= 0 ? undefined : false; },
    move: (x, y) => { if (st.drag >= 0) { st.P[st.drag] = [Math.max(0, Math.min(3, v2.invX(x))), Math.max(0, Math.min(3, v2.invY(y)))]; draw(); } },
    up: () => { st.drag = -1; },
  });
  s1.onResize(draw); s2.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W3: assembly                                                         */
/* ------------------------------------------------------------------ */
function assemblyWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('mesh: current element and its 3 global vertex numbers');
  const s1 = createCanvas(p1, { aspect: 0.9 });
  const p2 = L.panel('global matrix $A$ (16 × 16, all vertices): accumulated entries');
  const s2 = createCanvas(p2, { aspect: 0.9 });
  const mesh = triGrid(3, 3, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.22, seed: 11 });
  const nV = mesh.nVert;
  const st = { k: 0, A: new Float64Array(nV * nV), local: null };
  const out = readout(L.controls);
  const row = buttonRow(L.controls);
  const stepFn = () => {
    if (st.k >= mesh.nTri) return false;
    const t = st.k;
    // local P1 stiffness: K_ij = |T| ∇λ_i·∇λ_j  (constant gradients)
    const aff = triAffine(mesh.nodes, mesh.tris, t);
    const grads = [[-1, -1], [1, 0], [0, 1]].map(([gx, gy]) => [aff.Jinv[0] * gx + aff.Jinv[2] * gy, aff.Jinv[1] * gx + aff.Jinv[3] * gy]);
    const area = Math.abs(aff.det) / 2, K = [];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) K.push(area * (grads[i][0] * grads[j][0] + grads[i][1] * grads[j][1]));
    const vs = [0, 1, 2].map((i) => mesh.tris[3 * t + i]);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) st.A[vs[i] * nV + vs[j]] += K[3 * i + j];
    // cotangent formula check: K_01 = −½ cot(angle at vertex 2), etc.
    const P = vs.map((v) => [mesh.nodes[2 * v], mesh.nodes[2 * v + 1]]);
    const cot = (k) => { const a = P[(k + 1) % 3], b = P[(k + 2) % 3], c = P[k]; const ux = a[0] - c[0], uy = a[1] - c[1], vx = b[0] - c[0], vy = b[1] - c[1]; return (ux * vx + uy * vy) / Math.abs(ux * vy - uy * vx); };
    st.local = { t, K, vs, cotErr: Math.max(Math.abs(K[1] + 0.5 * cot(2)), Math.abs(K[5] + 0.5 * cot(0)), Math.abs(K[2] + 0.5 * cot(1))) };
    st.k++;
    return true;
  };
  const anim = new Animator(fig, { step: () => { const r = stepFn(); return r; }, draw: () => draw(), stepsPerFrame: 1 });
  let last = 0;
  anim.o.step = () => { const now = performance.now(); if (now - last < 450) return true; last = now; return stepFn(); };
  const play = button(row, { label: '▶ Play', primary: true, onClick: () => anim.toggle() });
  anim.o.onState = (r) => play.setLabel(r ? '❚❚ Pause' : '▶ Play');
  button(row, { label: 'Step', onClick: () => { stepFn(); draw(); } });
  button(row, { label: '↺ Reset', onClick: () => { anim.pause(); st.k = 0; st.A.fill(0); st.local = null; draw(); } });
  function draw() {
    const T = theme(), C = seriesColors();
    const v = new View2D(s1, [0, 1, 0, 1], { equal: true, pad: 16 });
    let ctx = s1.ctx;
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    for (let t = 0; t < st.k; t++) fillTri(ctx, v, mesh.nodes, mesh.tris, t, T.accent, 0.12);
    if (st.local) fillTri(ctx, v, mesh.nodes, mesh.tris, st.local.t, C[1], 0.55);
    drawTriMesh(ctx, v, mesh.nodes, mesh.tris, { color: T.soft, width: 1 });
    ctx.font = `11px ${T.mono}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let i = 0; i < nV; i++) {
      const x = v.X(mesh.nodes[2 * i]), y = v.Y(mesh.nodes[2 * i + 1]);
      const hl = st.local && st.local.vs.includes(i);
      ctx.fillStyle = hl ? C[1] : T.elev; ctx.beginPath(); ctx.arc(x, y, 9, 0, 7); ctx.fill();
      ctx.strokeStyle = T.ink; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = hl ? '#fff' : T.ink; ctx.fillText(String(i), x, y);
    }
    // global matrix heat grid
    ctx = s2.ctx;
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s2.w, s2.h);
    const cell = Math.min(s2.w, s2.h) / (nV + 1), ox = (s2.w - cell * nV) / 2, oy = (s2.h - cell * nV) / 2;
    let mx = 1e-9;
    for (let i = 0; i < nV * nV; i++) mx = Math.max(mx, Math.abs(st.A[i]));
    for (let i = 0; i < nV; i++) for (let j = 0; j < nV; j++) {
      const a = st.A[i * nV + j];
      ctx.fillStyle = a === 0 ? T.elev : a > 0 ? T.accent2 : T.accent;
      ctx.globalAlpha = a === 0 ? 1 : 0.3 + 0.7 * Math.min(1, Math.abs(a) / mx);
      ctx.fillRect(ox + j * cell + 0.5, oy + i * cell + 0.5, cell - 1, cell - 1);
      ctx.globalAlpha = 1;
    }
    if (st.local) {
      ctx.strokeStyle = C[1]; ctx.lineWidth = 2;
      for (const i of st.local.vs) for (const j of st.local.vs) ctx.strokeRect(ox + j * cell + 1, oy + i * cell + 1, cell - 2, cell - 2);
    }
    // row sums of the full Neumann matrix are zero (constants are in the kernel)
    let rs = 0;
    for (let i = 0; i < nV; i++) { let s = 0; for (let j = 0; j < nV; j++) s += st.A[i * nV + j]; rs = Math.max(rs, Math.abs(s)); }
    if (st.local) {
      const K = st.local.K.map((x) => x.toFixed(3).padStart(7));
      out.set(`element ${st.local.t} → global rows/cols ${st.local.vs.join(', ')}\nlocal K =\n${K.slice(0, 3).join('')}\n${K.slice(3, 6).join('')}\n${K.slice(6).join('')}\ncotangent formula error: ${st.local.cotErr.toExponential(1)}\nmax |row sum| of A = ${rs.toExponential(1)}`);
      selfCheck('cotangent formula', st.local.cotErr < 1e-12);
    } else out.set(`${mesh.nTri} elements, ${nV} vertices.\nPress Play or Step.`);
  }
  s1.onResize(draw); s2.onResize(draw);
  stepFn(); draw();
}

/* ------------------------------------------------------------------ */
/* W4: the full Poisson solver                                          */
/* ------------------------------------------------------------------ */
function poissonWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('');
  const s1 = createCanvas(p1, { aspect: 0.85 });
  const p2 = L.panel('');
  const tabs = h('div');
  p2.appendChild(tabs);
  const s2 = createCanvas(p2, { aspect: 0.85 });
  const st = { p: 1, N: 12, sol: 'wave', jiggle: 0.15, view: 'u', right: 'spy', reorder: 'natural', res: null, conv: null };
  segmented(L.controls, { label: 'element', value: '1', options: [{ value: '1', label: 'P1' }, { value: '2', label: 'P2' }], onChange: (v) => { st.p = +v; st.conv = null; solveD(); } });
  slider(L.controls, { label: 'cells per side $N$', min: 2, max: 40, step: 1, value: st.N, onInput: debounce((v) => { st.N = v; solve(); }, 120) });
  select(L.controls, { label: 'exact solution', value: st.sol, options: Object.entries(POISSON_MMS).map(([k, v]) => ({ value: k, label: v.name })), onChange: (v) => { st.sol = v; st.conv = null; solve(); } });
  slider(L.controls, { label: 'mesh distortion', min: 0, max: 0.3, step: 0.01, value: st.jiggle, onInput: debounce((v) => { st.jiggle = v; st.conv = null; solve(); }, 120) });
  segmented(L.controls, { label: 'left panel', value: 'u', options: [{ value: 'u', label: '$u_h$' }, { value: 'err', label: 'error' }, { value: 'mesh', label: 'mesh' }], onChange: (v) => { st.view = v; draw(); } });
  segmented(tabs, { value: 'spy', options: [{ value: 'spy', label: 'sparsity' }, { value: 'cg', label: 'CG iterations' }, { value: 'conv', label: 'convergence' }], onChange: (v) => { st.right = v; draw(); } });
  select(L.controls, { label: 'numbering of the unknowns (sparsity tab)', value: 'natural', options: [
    { value: 'natural', label: 'natural (row by row)' }, { value: 'random', label: 'random (as from a mesh generator)' }, { value: 'rcm', label: 'reverse Cuthill–McKee' }],
  onChange: (v) => { st.reorder = v; draw(); } });
  const out = readout(L.controls);
  const solveD = debounce(() => solve(), 50);
  function solve() {
    const S = POISSON_MMS[st.sol];
    const mesh = triGrid(st.N, st.N, [0, 1, 0, 1], { diag: 'alt', jiggle: st.jiggle });
    const t0 = performance.now();
    const r = solvePoissonCG(mesh, { p: st.p, f: S.f, g: S.u, bc: 'strong' });
    const tDirect = performance.now() - t0;
    const e = errorsCG(r.space, r.U, S.u, S.grad);
    // iterative solve of the same reduced system for the CG-history panel
    const it = cgSolve(r.A, r.b, { tol: 1e-10, maxIter: 5000, precond: jacobiPrecond(r.A) });
    const itPlain = cgSolve(r.A, r.b, { tol: 1e-10, maxIter: 5000 });
    st.res = { mesh, r, e, tDirect, it, itPlain, S };
    draw();
    selfCheck('poisson errors finite', Number.isFinite(e.L2));
  }
  function convergence() {
    const S = POISSON_MMS[st.sol], Ns = st.p === 1 ? [4, 8, 16, 32] : [4, 8, 16, 24];
    const rows = Ns.map((N) => {
      const mesh = triGrid(N, N, [0, 1, 0, 1], { diag: 'alt', jiggle: st.jiggle });
      const r = solvePoissonCG(mesh, { p: st.p, f: S.f, g: S.u });
      return { h: 1 / N, ...errorsCG(r.space, r.U, S.u, S.grad) };
    });
    st.conv = { rows, rL2: fitRate(rows.map((x) => x.h), rows.map((x) => x.L2)), rH1: fitRate(rows.map((x) => x.h), rows.map((x) => x.H1)) };
  }
  function draw() {
    if (!st.res) return;
    const T = theme(), C = seriesColors(), { mesh, r, e, S } = st.res;
    const ctx = s1.ctx;
    const view = new View2D(s1, [0, 1, 0, 1], { equal: true, pad: [8, 64, 8, 8] });
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    if (st.view === 'mesh') {
      drawTriMesh(ctx, view, mesh.nodes, mesh.tris, { color: T.ink, width: 1 });
    } else {
      // value of u_h (or of the error u_h − u) at reference point (x̂,ŷ) = (λ1, λ2) of triangle t
      const fn = (t, l0, l1, l2, x, y) => {
        const v = evalCG(r.space, r.U, t, l1, l2).u;
        return st.view === 'u' ? v : v - S.u(x, y);
      };
      // colour range from a few sample points per triangle
      const vals = [];
      for (let t = 0; t < mesh.nTri; t++) {
        const af = triAffine(mesh.nodes, mesh.tris, t);
        for (const [a, b] of [[1 / 3, 1 / 3], [0.05, 0.05], [0.9, 0.05], [0.05, 0.9], [0.45, 0.1], [0.1, 0.45]]) {
          vals.push(fn(t, 1 - a - b, a, b, af.a[0] + af.J[0] * a + af.J[1] * b, af.a[1] + af.J[2] * a + af.J[3] * b));
        }
      }
      let [lo, hi] = range(vals);
      let cmap = 'viridis';
      if (st.view === 'err') { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; cmap = 'rdbu'; }
      drawTriField(s1, view, mesh.nodes, mesh.tris, fn, { cmap, lo, hi });
      if (st.N <= 24) drawTriMesh(ctx, view, mesh.nodes, mesh.tris, { color: 'rgba(255,255,255,0.35)', width: 0.6 });
      colorbar(ctx, cmap, s1.w - 52, 10, 12, s1.h - 20, lo, hi, { ink: T.soft });
    }
    // right panel
    if (st.right === 'spy') {
      // a fixed pseudo-random permutation mimics the arbitrary numbering of an unstructured mesh generator
      const n = r.A.n, perm = Int32Array.from({ length: n }, (_, i) => i);
      let seed = 42;
      for (let i = n - 1; i > 0; i--) { seed = (seed * 16807) % 2147483647; const j = seed % (i + 1); [perm[i], perm[j]] = [perm[j], perm[i]]; }
      const Ar = st.reorder === 'random' ? csrPermute(r.A, perm) : r.A;
      const A = st.reorder === 'rcm' ? csrPermute(Ar, rcm(Ar)) : Ar;
      drawSpy(s2, A, { color: T.accent });
    } else if (st.right === 'cg') {
      const hist = st.res.it.history, hp = st.res.itPlain.history;
      const n = Math.max(hist.length, hp.length);
      const P = new Plot(s2, { xlim: [0, Math.max(5, n)], ylim: [1e-11, 10], ylog: true, xlabel: 'CG iteration', ylabel: 'relative residual' });
      P.frame();
      P.line(hp.map((_, i) => i), hp, { color: C[1] });
      P.line(hist.map((_, i) => i), hist, { color: C[0] });
      P.legend([{ label: `plain CG: ${st.res.itPlain.iters} its`, color: C[1] }, { label: `Jacobi-PCG: ${st.res.it.iters} its`, color: C[0] }]);
    } else {
      if (!st.conv) convergence();
      const rows = st.conv.rows;
      const all = rows.flatMap((x) => [x.L2, x.H1]);
      const P = new Plot(s2, { xlim: [0.02, 0.4], ylim: [10 ** Math.floor(Math.log10(Math.min(...all))), 10 ** Math.ceil(Math.log10(Math.max(...all)))], xlog: true, ylog: true, xlabel: 'h', ylabel: 'error' });
      P.frame();
      P.line(rows.map((x) => x.h), rows.map((x) => x.L2), { color: C[0] }); P.points(rows.map((x) => x.h), rows.map((x) => x.L2), { color: C[0] });
      P.line(rows.map((x) => x.h), rows.map((x) => x.H1), { color: C[1] }); P.points(rows.map((x) => x.h), rows.map((x) => x.H1), { color: C[1] });
      P.legend([{ label: `L² error, slope ${st.conv.rL2.toFixed(2)} (expect ${st.p + 1})`, color: C[0] }, { label: `H¹ error, slope ${st.conv.rH1.toFixed(2)} (expect ${st.p})`, color: C[1] }], 'br');
    }
    out.set(`unknowns: ${r.A.n}   non-zeros: ${nnz(r.A)}\n(${(nnz(r.A) / r.A.n).toFixed(1)} per row)\ndirect solve: ${st.res.tDirect.toFixed(0)} ms\nL² error = ${e.L2.toExponential(2)}\nH¹ error = ${e.H1.toExponential(2)}`);
  }
  s1.onResize(draw); s2.onResize(draw);
  solve();
}

initChapter(() => {
  mount('w-basis', basisWidget);
  mount('w-refmap', refMapWidget);
  mount('w-assembly', assemblyWidget);
  mount('w-poisson', poissonWidget);
});

export { csrExtract, csrMatVec, assembleVolume };
