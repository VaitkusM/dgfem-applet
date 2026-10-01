/**
 * @file Chapter 8 — DG for elliptic problems (interior penalty methods).
 *
 * Widgets:
 *  - w-sipg     : IPDG solver (p, N, C_IP, variant, problem): u_h heat map with edge jumps, a cut line,
 *                 λ_min of the symmetric part of the matrix, errors
 *  - w-conv     : live convergence study of SIPG / NIPG / IIPG (L² and energy errors, rate table)
 *  - w-coercive : number of negative eigenvalues vs C_IP, the most dangerous eigenvector, the 1D
 *                 trace-extremal polynomial, and the critical C_IP for three penalty scalings
 *  - w-dofs     : unknowns / non-zeros of CG vs DG as functions of p
 */
import { initChapter, mount, selfCheck } from '../../lib/ui/chapter.js';
import { widgetLayout, slider, select, segmented, checkbox, readout, fmt, debounce, h } from '../../lib/ui/controls.js';
import { createCanvas, View2D, theme, seriesColors } from '../../lib/viz/canvas.js';
import { drawTriField } from '../../lib/viz/field2d.js';
import { drawTriMesh } from '../../lib/viz/meshdraw.js';
import { colorOf, colorbar } from '../../lib/viz/colormap.js';
import { Plot } from '../../lib/viz/plot1d.js';
import { triGrid } from '../../lib/core/mesh/structured.js';
import { POISSON_MMS } from '../../lib/core/verify/mms.js';
import { pairwiseRates } from '../../lib/core/verify/rates.js';
import { solveIP, assembleIP, ipErrors, edgeJumps, smallestEigenvalue } from '../../lib/core/elliptic/sipg.js';
import { brokenEvaluator, triGeometry } from '../../lib/core/elliptic/dgtri.js';
import { systemSizes } from '../../lib/core/elliptic/dofcount.js';
import { csrAdd, csrTranspose, csrToDense, nnz } from '../../lib/core/la/sparse.js';
import { symEig } from '../../lib/core/la/dense.js';
import { inertia } from '../../lib/core/la/inertia.js';
import { gaussLegendre } from '../../lib/core/quad/gauss1d.js';
import { legendreP } from '../../lib/core/basis/legendre.js';

const PROBLEMS = [
  { value: 'wave', label: 'u = cos(πx)·eʸ' },
  { value: 'sinsin', label: 'u = sin(πx)·sin(πy)' },
  { value: 'peak', label: 'Gaussian peak' },
];
const VARIANTS = [{ value: 'sipg', label: 'SIPG' }, { value: 'nipg', label: 'NIPG' }, { value: 'iipg', label: 'IIPG' }];
const meshOf = (N) => triGrid(N, N, [0, 1, 0, 1], { diag: 'alt', jiggle: N > 1 ? 0.12 : 0 });

/** Min/max of the exact solution on a grid (fixed colour range per problem). */
function exactRange(u) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i <= 60; i++) for (let j = 0; j <= 60; j++) { const v = u(i / 60, j / 60); lo = Math.min(lo, v); hi = Math.max(hi, v); }
  return [lo, hi];
}

/** Rasterise a broken Dubiner field of degree p (coefficients U[t*n+m]). */
function drawBroken(surf, view, mesh, p, U, o) {
  const ev = brokenEvaluator(p, U);
  drawTriField(surf, view, mesh.nodes, mesh.tris, (t, l0, l1, l2) => ev(t, l1, l2), o);
}

/**
 * Pieces of the broken function along the horizontal line y = y0:
 * one polyline per triangle crossed (so jumps show as gaps).
 */
function cutPieces(mesh, p, U, y0) {
  const geo = triGeometry(mesh), ev = brokenEvaluator(p, U), { nodes, tris } = mesh, out = [];
  for (let t = 0; t < geo.nTri; t++) {
    const xs = [];
    for (let k = 0; k < 3; k++) {
      const a = tris[3 * t + k], b = tris[3 * t + (k + 1) % 3];
      const ya = nodes[2 * a + 1], yb = nodes[2 * b + 1];
      if ((ya - y0) * (yb - y0) < 0) { const s = (y0 - ya) / (yb - ya); xs.push(nodes[2 * a] + s * (nodes[2 * b] - nodes[2 * a])); }
    }
    if (xs.length < 2) continue;
    const xa = Math.min(...xs), xb = Math.max(...xs), px = [], py = [];
    const o = 4 * t, ji = geo.Jinv;
    for (let i = 0; i <= 12; i++) {
      const x = xa + (xb - xa) * i / 12, dx = x - geo.a[2 * t], dy = y0 - geo.a[2 * t + 1];
      const xr = ji[o] * dx + ji[o + 1] * dy, yr = ji[o + 2] * dx + ji[o + 3] * dy;
      px.push(x); py.push(ev(t, xr, yr));
    }
    out.push([px, py]);
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* W1: the interior penalty solver                                      */
/* ------------------------------------------------------------------ */
function sipgWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('DG solution $u_h$; edges coloured by the jump $|[\\![u_h]\\!]|$');
  const s1 = createCanvas(p1, { aspect: 0.86 });
  const p2 = L.panel('cut along $y = 0.47$: one curve per triangle; dashed = exact $u$');
  const s2 = createCanvas(p2, { aspect: 0.86 });
  const st = { p: 2, N: 6, Cip: 2, variant: 'sipg', prob: 'wave', jumps: true, y0: 0.47 };
  segmented(L.controls, { label: 'variant', options: VARIANTS, value: st.variant, onChange: (v) => { st.variant = v; solve(); } });
  segmented(L.controls, { label: 'degree $p$', options: [1, 2, 3].map((v) => ({ value: String(v), label: String(v) })), value: '2', onChange: (v) => { st.p = +v; solve(); } });
  slider(L.controls, { label: 'mesh $N$ ($2N^2$ triangles)', min: 2, max: 16, step: 1, value: st.N, onInput: (v) => { st.N = v; later(); } });
  slider(L.controls, { label: 'penalty $C_{IP}$', min: 0.01, max: 100, log: true, value: st.Cip, format: (v) => fmt(v, 2), onInput: (v) => { st.Cip = v; later(); } });
  select(L.controls, { label: 'exact solution', options: PROBLEMS, value: st.prob, onChange: (v) => { st.prob = v; solve(); } });
  checkbox(L.controls, { label: 'colour edges by jump', value: st.jumps, onChange: (v) => { st.jumps = v; draw(); } });
  const out = readout(L.controls);
  let R = null;
  const later = debounce(() => solve(), 60);
  function solve() {
    const S = POISSON_MMS[st.prob], mesh = meshOf(st.N), t0 = performance.now();
    const sol = solveIP(mesh, { p: st.p, f: S.f, g: S.u, variant: st.variant, Cip: st.Cip });
    const err = ipErrors(sol, S.u, S.grad), J = edgeJumps(sol, S.u);
    const tSolve = performance.now() - t0;
    let lmin = NaN;
    if (sol.nDof <= 1500) {
      const As = st.variant === 'sipg' ? sol.A : csrAdd(sol.A, csrTranspose(sol.A), 0.5, 0.5);
      lmin = smallestEigenvalue(As, { k: 60 });
    }
    R = { S, mesh, sol, err, J, tSolve, lmin, range: exactRange(S.u) };
    selfCheck('sipg widget: finite solution', sol.U.every(Number.isFinite));
    if (st.variant === 'sipg' && st.Cip >= 1) selfCheck('sipg widget: SPD for C_IP >= 1', !(lmin <= 0));
    draw();
  }
  function draw() {
    if (!R) return;
    const T = theme(), C = seriesColors(), { mesh, sol, J } = R, [lo, hi] = R.range;
    const view = new View2D(s1, [0, 1, 0, 1], { equal: true, pad: [8, 64, 8, 8] });
    const ctx = s1.ctx;
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    drawBroken(s1, view, mesh, st.p, sol.U, { cmap: 'viridis', lo, hi });
    const { topo } = sol;
    let jmax = 0; for (const v of J.rms) jmax = Math.max(jmax, v);
    const urange = hi - lo || 1;
    ctx.save(); ctx.lineCap = 'round';
    for (let e = 0; e < topo.nEdge; e++) {
      const a = topo.edges[2 * e], b = topo.edges[2 * e + 1];
      if (st.jumps) {
        // jump relative to the range of u, on a log scale from 1e-5 to 1e-1
        const r = Math.log10(Math.max(J.rms[e], 1e-16) / urange);
        ctx.strokeStyle = colorOf('magma', r, -5, -1); ctx.lineWidth = 0.8 + 2.4 * Math.min(1, Math.max(0, (r + 5) / 4));
      } else { ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 0.8; }
      ctx.beginPath(); ctx.moveTo(view.X(mesh.nodes[2 * a]), view.Y(mesh.nodes[2 * a + 1]));
      ctx.lineTo(view.X(mesh.nodes[2 * b]), view.Y(mesh.nodes[2 * b + 1])); ctx.stroke();
    }
    ctx.setLineDash([5, 4]); ctx.strokeStyle = T.accent2; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(view.X(0), view.Y(st.y0)); ctx.lineTo(view.X(1), view.Y(st.y0)); ctx.stroke();
    ctx.restore();
    colorbar(ctx, 'viridis', s1.w - 52, 12, 12, s1.h * 0.38, lo, hi, { ink: T.soft });
    if (st.jumps) {
      colorbar(ctx, 'magma', s1.w - 52, s1.h * 0.56, 12, s1.h * 0.38, -5, -1, { ink: T.soft, fmt: (v) => `1e${v}` });
      ctx.save(); ctx.fillStyle = T.soft; ctx.font = `10px ${T.ui}`; ctx.fillText('jump/range', s1.w - 62, s1.h * 0.56 - 5); ctx.restore();
    }
    // cut line
    const pieces = cutPieces(mesh, st.p, sol.U, st.y0);
    const xs = [], ys = [];
    for (let i = 0; i <= 300; i++) { xs.push(i / 300); ys.push(R.S.u(i / 300, st.y0)); }
    let ylo = Math.min(...ys), yhi = Math.max(...ys);
    for (const [, py] of pieces) for (const v of py) { ylo = Math.min(ylo, v); yhi = Math.max(yhi, v); }
    const pad = 0.08 * (yhi - ylo || 1);
    const P = new Plot(s2, { xlim: [0, 1], ylim: [ylo - pad, yhi + pad], xlabel: 'x' });
    P.frame();
    P.line(xs, ys, { color: T.faint, dash: [5, 4], width: 1.5 });
    pieces.forEach(([px, py], i) => P.line(px, py, { color: i % 2 ? C[0] : C[2], width: 2 }));
    P.legend([{ label: 'exact u', color: T.faint, dash: [5, 4] }, { label: 'u_h, one colour per triangle', color: C[0] }], 'tr');
    const lm = Number.isFinite(R.lmin)
      ? `<b class="${R.lmin > 0 ? 'status-good' : 'status-bad'}">${fmt(R.lmin)}</b> (${R.lmin > 0 ? 'coercive' : 'NOT coercive'})`
      : '— (reduce N or p)';
    out.set(`unknowns  = ${sol.nDof}\nnon-zeros = ${nnz(sol.A)}\n`
      + `λ_min(½(A+Aᵀ)):\n  ${lm}\n`
      + `‖u − u_h‖_L2   = ${fmt(R.err.L2)}\nenergy error   = ${fmt(R.err.energy)}\n`
      + `max edge jump  = ${fmt(jmax)}\nsolve: ${R.tSolve.toFixed(0)} ms`);
  }
  s1.onResize(draw); s2.onResize(draw);
  solve();
}

/* ------------------------------------------------------------------ */
/* W2: live convergence study                                           */
/* ------------------------------------------------------------------ */
function convWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('$\\|u - u_h\\|_{L^2}$ versus $h = 1/N$');
  const s1 = createCanvas(p1, { aspect: 0.75 });
  const p2 = L.panel('DG energy error $|||u-u_h|||$ versus $h$');
  const s2 = createCanvas(p2, { aspect: 0.75 });
  const p3 = L.panel('', '1 1 100%');
  const tableWrap = h('div', '', ''); tableWrap.style.overflowX = 'auto'; p3.appendChild(tableWrap);
  const st = { p: 2, Cip: 1, prob: 'wave' };
  segmented(L.controls, { label: 'degree $p$', options: [1, 2, 3].map((v) => ({ value: String(v), label: String(v) })), value: '2', onChange: (v) => { st.p = +v; run(); } });
  slider(L.controls, { label: 'penalty $C_{IP}$', min: 0.1, max: 20, log: true, value: st.Cip, format: (v) => fmt(v, 2), onChange: (v) => { st.Cip = v; run(); } });
  select(L.controls, { label: 'exact solution', options: PROBLEMS.slice(0, 2), value: st.prob, onChange: (v) => { st.prob = v; run(); } });
  const out = readout(L.controls);
  let job = 0;
  const res = { sipg: [], nipg: [], iipg: [] };
  let Ns = [];
  function run() {
    const my = ++job;
    Ns = st.p <= 2 ? [2, 4, 8, 16, 32] : [2, 4, 8, 16];
    for (const k in res) res[k] = [];
    const tasks = [];
    for (const N of Ns) for (const v of ['sipg', 'nipg', 'iipg']) tasks.push([N, v]);
    let i = 0;
    out.set('computing …');
    const step = () => {
      if (my !== job) return;
      const [N, v] = tasks[i++], S = POISSON_MMS[st.prob];
      const sol = solveIP(meshOf(N), { p: st.p, f: S.f, g: S.u, variant: v, Cip: st.Cip });
      const e = ipErrors(sol, S.u, S.grad);
      res[v].push({ h: 1 / N, L2: e.L2, En: e.energy });
      draw();
      if (i < tasks.length) setTimeout(step, 0);
      else {
        const r = pairwiseRates(res.sipg.map((x) => x.h), res.sipg.map((x) => x.L2));
        out.set(`done: ${tasks.length} solves\nlast SIPG L² rate = ${fmt(r[r.length - 1])} (expected ${st.p + 1})`);
        selfCheck('conv widget: SIPG optimal L2 rate', Math.abs(r[r.length - 1] - (st.p + 1)) < 0.35);
      }
    };
    setTimeout(step, 0);
  }
  function draw() {
    const C = seriesColors(), cols = { sipg: C[0], nipg: C[1], iipg: C[2] };
    for (const [surf, key, slope] of [[s1, 'L2', st.p + 1], [s2, 'En', st.p]]) {
      const all = [].concat(...Object.values(res).map((a) => a.map((x) => x[key]))).filter((v) => v > 0);
      const ylo = all.length ? Math.min(...all) / 3 : 1e-6, yhi = all.length ? Math.max(...all) * 3 : 1;
      const P = new Plot(surf, { xlim: [1 / (Ns[Ns.length - 1] * 1.4), 0.7], ylim: [ylo, yhi], xlog: true, ylog: true, xlabel: 'h = 1/N' });
      P.frame();
      for (const v of ['sipg', 'nipg', 'iipg']) {
        const a = res[v];
        P.line(a.map((x) => x.h), a.map((x) => x[key]), { color: cols[v], width: 2, dash: v === 'iipg' ? [6, 4] : [] });
        P.points(a.map((x) => x.h), a.map((x) => x[key]), { color: cols[v], r: 3 });
      }
      if (res.sipg.length >= 3) { const a = res.sipg, j = a.length - 3; P.slopeTriangle(a[j].h, a[j][key] * 0.3, slope, 0.5); }
      P.legend(VARIANTS.map((v) => ({ label: v.label, color: cols[v.value], dash: v.value === 'iipg' ? [6, 4] : undefined })), 'br');
    }
    // table
    let html = '<table class="data"><tr><th>N</th>';
    for (const v of VARIANTS) html += `<th>${v.label} L² error</th><th>rate</th>`;
    html += '<th>SIPG energy</th><th>rate</th></tr>';
    const rate = (a, k, key) => (k > 0 && a[k] ? Math.log(a[k][key] / a[k - 1][key]) / Math.log(a[k].h / a[k - 1].h) : NaN);
    Ns.forEach((N, k) => {
      html += `<tr><td>${N}</td>`;
      for (const v of ['sipg', 'nipg', 'iipg']) {
        const a = res[v][k];
        const r = rate(res[v], k, 'L2');
        const good = Number.isFinite(r) ? (Math.abs(r - (st.p + 1)) < 0.35 ? 'good' : (r < st.p + 0.6 ? 'bad' : '')) : '';
        html += `<td>${a ? a.L2.toExponential(2) : '…'}</td><td class="${good}">${Number.isFinite(r) ? r.toFixed(2) : ''}</td>`;
      }
      const a = res.sipg[k], r = rate(res.sipg, k, 'En');
      html += `<td>${a ? a.En.toExponential(2) : '…'}</td><td>${Number.isFinite(r) ? r.toFixed(2) : ''}</td></tr>`;
    });
    tableWrap.innerHTML = `${html}</table>`;
  }
  s1.onResize(draw); s2.onResize(draw);
  draw();
  run();
}

/* ------------------------------------------------------------------ */
/* W3: coercivity and the penalty scaling                               */
/* ------------------------------------------------------------------ */
const SCALINGS = {
  ours: { label: 'C·(p+1)(p+2)/h_F', fn: undefined },
  nop: { label: 'C/h_F (no p factor)', fn: (hF, p, C) => C / hF },
  noh: { label: 'C·(p+1)(p+2) (no 1/h)', fn: (hF, p, C) => C * (p + 1) * (p + 2) },
};
function coerciveWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('number of negative eigenvalues of $A$ vs $C_{IP}$ (8 triangles)');
  const s1 = createCanvas(p1, { aspect: 0.62 });
  const p2 = L.panel('eigenvector of $\\lambda_{\\min}$ at the current $C_{IP}$');
  const s2 = createCanvas(p2, { aspect: 0.62 });
  const p3 = L.panel('1D: the polynomial of degree $p$ with the largest $w(1)^2/\\|w\\|^2$');
  const s3 = createCanvas(p3, { aspect: 0.62 });
  const p4 = L.panel('critical $C^\\star$ (SPD for $C_{IP} > C^\\star$) for three penalty scalings');
  const tbl = h('div', ''); tbl.style.overflowX = 'auto'; p4.appendChild(tbl);
  const st = { p: 1, Cip: 0.2, scaling: 'ours' };
  segmented(L.controls, { label: 'degree $p$', options: [1, 2, 3].map((v) => ({ value: String(v), label: String(v) })), value: '1', onChange: (v) => { st.p = +v; sweep(); table(); } });
  select(L.controls, { label: 'penalty $\\sigma_F$ =', options: Object.entries(SCALINGS).map(([k, v]) => ({ value: k, label: v.label })), value: st.scaling, onChange: (v) => { st.scaling = v; sweep(); } });
  slider(L.controls, { label: 'current $C_{IP}$', min: 0.01, max: 100, log: true, value: st.Cip, format: (v) => fmt(v, 2), onInput: (v) => { st.Cip = v; later(); } });
  const out = readout(L.controls);
  const mesh = triGrid(2, 2, [0, 1, 0, 1], { diag: 'alt' });
  const dense = (C, m = mesh, scaling = st.scaling, p = st.p) => {
    const A = assembleIP(m, { p, f: () => 0, Cip: C, sigmaFn: SCALINGS[scaling].fn }).A;
    return { D: csrToDense(A), n: A.n };
  };
  const critical = (m, scaling, p) => {
    let lo = 1e-3, hi = 1e3;
    for (let i = 0; i < 34; i++) { const c = Math.sqrt(lo * hi), { D, n } = dense(c, m, scaling, p); if (inertia(D, n).neg === 0) hi = c; else lo = c; }
    return hi;
  };
  let sw = null;
  function sweep() {
    const Cs = [], neg = [];
    for (let i = 0; i <= 60; i++) { const c = 10 ** (-2 + 4 * i / 60); const { D, n } = dense(c); Cs.push(c); neg.push(inertia(D, n).neg); }
    sw = { Cs, neg, Cstar: critical(mesh, st.scaling, st.p) };
    current();
  }
  let cur = null;
  function current() {
    const { D, n } = dense(st.Cip), E = symEig(D, n);
    const vec = new Float64Array(n);
    for (let i = 0; i < n; i++) vec[i] = E.vectors[i * n];
    cur = { lmin: E.values[0], lmax: E.values[n - 1], vec, nNeg: inertia(D, n).neg, nEig: E.values.filter((v) => v < 0).length };
    selfCheck('coercive widget: inertia = eigenvalue count', cur.nNeg === cur.nEig);
    draw();
  }
  const later = debounce(() => current(), 40);
  function draw() {
    if (!sw || !cur) return;
    const T = theme(), C = seriesColors();
    const nmax = Math.max(1, ...sw.neg);
    const P = new Plot(s1, { xlim: [0.01, 100], ylim: [0, nmax * 1.15], xlog: true, xlabel: 'C_IP', ylabel: '# negative eigenvalues' });
    P.frame();
    const xs = [], ys = [];
    sw.Cs.forEach((c, i) => { if (i > 0) { xs.push(c); ys.push(sw.neg[i - 1]); } xs.push(c); ys.push(sw.neg[i]); });
    P.line(xs, ys, { color: C[1], width: 2.2 });
    P.vline(sw.Cstar, { color: T.accent2, width: 1.5 });
    P.text(sw.Cstar, nmax * 1.05, ` C* = ${fmt(sw.Cstar)}`, { color: T.accent2 });
    P.vline(st.Cip, { color: T.ink, width: 1.2, dash: [] });
    // eigenvector
    const view = new View2D(s2, [0, 1, 0, 1], { equal: true, pad: 8 });
    s2.ctx.fillStyle = T.bg; s2.ctx.fillRect(0, 0, s2.w, s2.h);
    const ev = brokenEvaluator(st.p, cur.vec);
    let vm = 0;
    for (let t = 0; t < mesh.tris.length / 3; t++) for (const [x, y] of [[0, 0], [1, 0], [0, 1], [1 / 3, 1 / 3], [0.5, 0], [0.5, 0.5], [0, 0.5]]) vm = Math.max(vm, Math.abs(ev(t, x, y)));
    drawTriField(s2, view, mesh.nodes, mesh.tris, (t, l0, l1, l2) => ev(t, l1, l2), { cmap: 'rdbu', lo: -vm, hi: vm });
    drawTriMesh(s2.ctx, view, mesh.nodes, mesh.tris, { color: T.ink, width: 1, alpha: 0.5 });
    // 1D extremal polynomial w = Σ L_j(1) L_j, normalised in L²(−1,1)
    const p = st.p, G = gaussLegendre(p + 2);
    const w = (x) => { let s = 0; for (let j = 0; j <= p; j++) s += (2 * j + 1) / 2 * legendreP(j, x); return s; };
    let nrm = 0; for (let q = 0; q < G.x.length; q++) nrm += G.w[q] * w(G.x[q]) ** 2;
    nrm = Math.sqrt(nrm);
    const ratio = (w(1) / nrm) ** 2;
    selfCheck('1D trace constant (p+1)^2/2', Math.abs(ratio - (p + 1) ** 2 / 2) < 1e-10);
    const X = [], Y = [];
    for (let i = 0; i <= 200; i++) { const x = -1 + 2 * i / 200; X.push(x); Y.push(w(x) / nrm); }
    const Q = new Plot(s3, { xlim: [-1, 1], ylim: [Math.min(...Y) - 0.2, Math.max(...Y) + 0.3], xlabel: 'x' });
    Q.frame(); Q.hline(0);
    Q.line(X, Y, { color: C[0], width: 2.2 });
    Q.points([1], [w(1) / nrm], { color: C[1], r: 4 });
    Q.text(0.95, w(1) / nrm, `w(1)² = ${fmt(ratio)} = (p+1)²/2`, { align: 'right', color: C[1], dy: -2 });
    const spd = cur.nNeg === 0;
    out.set(`p = ${st.p}, ${mesh.tris.length / 3} triangles, ${cur.vec.length} unknowns\n`
      + `C* (this mesh) = ${fmt(sw.Cstar)}\n`
      + `at C_IP = ${fmt(st.Cip, 2)}:\n  λ_min = <b class="${spd ? 'status-good' : 'status-bad'}">${fmt(cur.lmin)}</b>, λ_max = ${fmt(cur.lmax)}\n`
      + `  # negative eigenvalues = ${cur.nNeg}\n  ${spd ? `κ = λ_max/λ_min = ${fmt(cur.lmax / cur.lmin)}` : 'indefinite: CG not applicable'}`);
  }
  let tjob = 0;
  function table() {
    const my = ++tjob, rows = [];
    const keys = Object.keys(SCALINGS), Ns = [2, 3, 4];
    const render = () => {
      let html = `<table class="data"><tr><th>$\\sigma_F$ (p = ${st.p})</th>${Ns.map((N) => `<th>N = ${N}</th>`).join('')}</tr>`;
      keys.forEach((k, i) => { html += `<tr><td>${SCALINGS[k].label}</td>${Ns.map((N, j) => `<td>${rows[i] && rows[i][j] !== undefined ? fmt(rows[i][j]) : '…'}</td>`).join('')}</tr>`; });
      tbl.innerHTML = `${html}</table>`;
      tbl.querySelectorAll('th').forEach((e) => { e.textContent = e.textContent.replace('$\\sigma_F$', 'σ_F'); });
    };
    const tasks = [];
    keys.forEach((k, i) => Ns.forEach((N, j) => tasks.push([i, j, k, N])));
    let t = 0;
    render();
    const step = () => {
      if (my !== tjob) return;
      const [i, j, k, N] = tasks[t++];
      (rows[i] = rows[i] || [])[j] = critical(triGrid(N, N, [0, 1, 0, 1], { diag: 'alt' }), k, st.p);
      render();
      if (t < tasks.length) setTimeout(step, 0);
    };
    setTimeout(step, 30);
  }
  s1.onResize(draw); s2.onResize(draw); s3.onResize(draw);
  sweep(); table();
}

/* ------------------------------------------------------------------ */
/* W4: unknowns and non-zeros, CG vs DG                                 */
/* ------------------------------------------------------------------ */
function dofsWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('size of the global system vs polynomial degree $p$ (log scale)');
  const s1 = createCanvas(p1, { aspect: 0.55 });
  const p2 = L.panel('', '1 1 100%');
  const tbl = h('div', ''); tbl.style.overflowX = 'auto'; p2.appendChild(tbl);
  const st = { N: 8, metric: 'n' };
  slider(L.controls, { label: 'mesh $N$ ($2N^2$ triangles)', min: 2, max: 16, step: 1, value: st.N, onInput: (v) => { st.N = v; later(); } });
  segmented(L.controls, { label: 'count', options: [{ value: 'n', label: 'unknowns' }, { value: 'nnz', label: 'non-zeros' }], value: st.metric, onChange: (v) => { st.metric = v; draw(); } });
  const out = readout(L.controls);
  let data = null;
  function compute() {
    const m = triGrid(st.N, st.N), ps = [1, 2, 3, 4, 5, 6];
    data = ps.map((p) => ({ p, ...systemSizes(m, p) }));
    selfCheck('dofs widget: DG unknowns formula', data.every((d) => d.dg.n === st.N * st.N * (d.p + 1) * (d.p + 2)));
    selfCheck('dofs widget: CG unknowns formula', data.every((d) => d.cg.n === (d.p * st.N - 1) ** 2));
    draw();
  }
  const later = debounce(compute, 60);
  function draw() {
    if (!data) return;
    const C = seriesColors(), k = st.metric;
    const vals = data.flatMap((d) => [d.cg[k], d.dg[k]]).filter((v) => v > 0);
    const P = new Plot(s1, { xlim: [0.7, 6.3], ylim: [Math.min(...vals) / 2, Math.max(...vals) * 2], ylog: true, xlabel: 'polynomial degree p' });
    P.frame();
    P.line(data.map((d) => d.p), data.map((d) => d.cg[k]), { color: C[2], width: 2.2 });
    P.points(data.map((d) => d.p), data.map((d) => d.cg[k]), { color: C[2] });
    P.line(data.map((d) => d.p), data.map((d) => d.dg[k]), { color: C[0], width: 2.2 });
    P.points(data.map((d) => d.p), data.map((d) => d.dg[k]), { color: C[0] });
    P.legend([{ label: 'CG (continuous P_p)', color: C[2] }, { label: 'DG (SIPG)', color: C[0] }], 'br');
    let html = '<table class="data"><tr><th>p</th><th>CG unknowns</th><th>DG unknowns</th><th>ratio</th><th>CG non-zeros</th><th>DG non-zeros</th><th>ratio</th></tr>';
    for (const d of data) html += `<tr><td>${d.p}</td><td>${d.cg.n}</td><td>${d.dg.n}</td><td>${(d.dg.n / d.cg.n).toFixed(2)}</td><td>${d.cg.nnz}</td><td>${d.dg.nnz}</td><td>${(d.dg.nnz / d.cg.nnz).toFixed(2)}</td></tr>`;
    tbl.innerHTML = `${html}</table>`;
    out.set(`N = ${st.N}: ${2 * st.N * st.N} triangles\nCG: (pN−1)² interior unknowns\nDG: N²(p+1)(p+2) unknowns`);
  }
  s1.onResize(draw);
  compute();
}

initChapter(() => {
  mount('w-sipg', sipgWidget);
  mount('w-conv', convWidget);
  mount('w-coercive', coerciveWidget);
  mount('w-dofs', dofsWidget);
});
