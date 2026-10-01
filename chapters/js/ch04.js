/**
 * @file Chapter 4 — Strong vs weak boundary conditions.
 *
 * Widgets:
 *  - w-bc2d  : Poisson with strong / Nitsche (sym, non-sym) / penalty BCs: solution, boundary mismatch, eigenvalues, rates
 *  - w-gamma : sweep of the penalty parameter γ: error and coercivity (smallest eigenvalue) for each method
 *  - w-hyper : 1D advection: inflow/outflow, strong vs weak (through the numerical flux) imposition
 */
import { initChapter, mount, selfCheck } from '../../lib/ui/chapter.js';
import { widgetLayout, slider, select, segmented, button, buttonRow, readout, fmt, debounce, h } from '../../lib/ui/controls.js';
import { createCanvas, View2D, theme, seriesColors } from '../../lib/viz/canvas.js';
import { drawTriField } from '../../lib/viz/field2d.js';
import { drawTriMesh } from '../../lib/viz/meshdraw.js';
import { colorbar, range } from '../../lib/viz/colormap.js';
import { Plot } from '../../lib/viz/plot1d.js';
import { Animator } from '../../lib/viz/anim.js';
import { triGrid } from '../../lib/core/mesh/structured.js';
import { solvePoissonCG, errorsCG, evalCG, physBasis } from '../../lib/core/elliptic/cg.js';
import { refEdgePoint } from '../../lib/core/mesh/topology.js';
import { triAffine } from '../../lib/core/mesh/structured.js';
import { POISSON_MMS } from '../../lib/core/verify/mms.js';
import { csrToDense, csrSymmetryError } from '../../lib/core/la/sparse.js';
import { symEig } from '../../lib/core/la/dense.js';
import { fitRate } from '../../lib/core/verify/rates.js';
import { makeAdvBC1D } from '../../lib/core/fv/advbc1d.js';
import { makeStepper } from '../../lib/core/time/rk.js';

const METHODS = [
  { value: 'strong', label: 'strong (eliminate boundary DOFs)' },
  { value: 'nitsche', label: 'Nitsche, symmetric' },
  { value: 'nitsche-nonsym', label: 'Nitsche, non-symmetric' },
  { value: 'penalty', label: 'penalty only' },
];

/** Smallest eigenvalue of the symmetric part (A + Aᵀ)/2 (dense; small systems only). */
function lambdaMinSym(A) {
  const n = A.n, D = csrToDense(A), S = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) S[i * n + j] = 0.5 * (D[i * n + j] + D[j * n + i]);
  const ev = symEig(S, n).values;
  return { lmin: ev[0], lmax: ev[n - 1] };
}

/** Perimeter coordinate s ∈ [0,4) of a point on the boundary of the unit square (counter-clockwise from (0,0)). */
function perimeterS(x, y) {
  const e = 1e-9;
  if (y < e) return x;
  if (x > 1 - e) return 1 + y;
  if (y > 1 - e) return 2 + (1 - x);
  return 3 + (1 - y);
}

/* ------------------------------------------------------------------ */
/* W1: Poisson with different BC treatments                             */
/* ------------------------------------------------------------------ */
function bc2dWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('solution $u_h$');
  const s1 = createCanvas(p1, { aspect: 0.85 });
  const p2 = L.panel('boundary mismatch $u_h - g$ along $\\partial\\Omega$ (unrolled, counter-clockwise from (0,0))');
  const s2 = createCanvas(p2, { aspect: 0.85 });
  const st = { bc: 'nitsche', p: 1, N: 8, gamma: 10, conv: null };
  const S = POISSON_MMS.wave;
  select(L.controls, { label: 'boundary condition', value: st.bc, options: METHODS, onChange: (v) => { st.bc = v; st.conv = null; run(); } });
  segmented(L.controls, { label: 'element', value: '1', options: [{ value: '1', label: 'P1' }, { value: '2', label: 'P2' }], onChange: (v) => { st.p = +v; st.conv = null; run(); } });
  slider(L.controls, { label: 'cells per side $N$', min: 2, max: 32, step: 1, value: st.N, onInput: debounce((v) => { st.N = v; run(); }, 100) });
  slider(L.controls, { label: 'penalty $\\gamma$', min: 0.1, max: 1e4, log: true, step: 0.02, value: st.gamma, onInput: debounce((v) => { st.gamma = v; st.conv = null; run(); }, 100) });
  const row = buttonRow(L.controls);
  button(row, { label: 'Convergence study', onClick: () => { convergence(); show(); } });
  const out = readout(L.controls);
  const tbl = h('div');
  L.controls.appendChild(tbl);
  let res;
  function solveOn(N) {
    const mesh = triGrid(N, N, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.1 });
    const r = solvePoissonCG(mesh, { p: st.p, f: S.f, g: S.u, bc: st.bc, gamma: st.gamma });
    return { mesh, r, e: errorsCG(r.space, r.U, S.u, S.grad) };
  }
  function convergence() {
    const Ns = st.p === 1 ? [4, 8, 16, 32] : [4, 8, 16, 24];
    const rows = Ns.map((N) => ({ N, ...solveOn(N).e }));
    const hs = Ns.map((N) => 1 / N);
    st.conv = { rows, rL2: fitRate(hs, rows.map((r) => r.L2)), rH1: fitRate(hs, rows.map((r) => r.H1)) };
  }
  function run() { res = solveOn(st.N); show(); }
  function show() {
    const T = theme(), C = seriesColors(), { mesh, r, e } = res;
    const ctx = s1.ctx, view = new View2D(s1, [0, 1, 0, 1], { equal: true, pad: [8, 60, 8, 8] });
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    const [lo, hi] = [-Math.E, Math.E];
    drawTriField(s1, view, mesh.nodes, mesh.tris, (t, l0, l1, l2) => evalCG(r.space, r.U, t, l1, l2).u, { cmap: 'viridis', lo, hi });
    drawTriMesh(ctx, view, mesh.nodes, mesh.tris, { color: 'rgba(255,255,255,0.3)', width: 0.6 });
    colorbar(ctx, 'viridis', s1.w - 48, 10, 12, s1.h - 20, lo, hi, { ink: T.soft });
    // boundary trace
    const pts = [], { topo } = r.space;
    for (let ed = 0; ed < topo.nEdge; ed++) {
      if (!topo.isBoundary[ed]) continue;
      const t = topo.edgeTris[2 * ed], k = topo.edgeLocal[2 * ed], aff = triAffine(mesh.nodes, mesh.tris, t);
      for (let q = 0; q <= 12; q++) {
        const [xr, yr] = refEdgePoint(k, q / 12), P = physBasis(r.space, t, xr, yr, aff);
        pts.push([perimeterS(P.x, P.y), evalCG(r.space, r.U, t, xr, yr, aff).u - S.u(P.x, P.y)]);
      }
    }
    pts.sort((a, b) => a[0] - b[0]);
    const m = Math.max(1e-6, ...pts.map((p) => Math.abs(p[1]))) * 1.2;
    const P = new Plot(s2, { xlim: [0, 4], ylim: [-m, m], xlabel: 'perimeter coordinate s' });
    P.frame(); P.hline(0);
    for (const s of [1, 2, 3]) P.vline(s);
    P.line(pts.map((p) => p[0]), pts.map((p) => p[1]), { color: C[0], width: 1.6 });
    let mism = 0; for (const p of pts) mism = Math.max(mism, Math.abs(p[1]));
    // eigenvalues (small systems only)
    let eig = '';
    if (r.A.n <= 400) {
      const { lmin } = lambdaMinSym(r.A);
      eig = `\nλ_min of (A+Aᵀ)/2 = ${fmt(lmin)} ${lmin > 0 ? '(coercive)' : '<b class="status-bad">(NOT positive!)</b>'}`;
    }
    out.set(`L² error = ${e.L2.toExponential(2)}\nH¹ error = ${e.H1.toExponential(2)}\nmax |u_h − g| on ∂Ω = ${mism.toExponential(2)}\nsymmetric matrix: ${csrSymmetryError(r.A) < 1e-12 ? 'yes' : 'no'}${eig}`);
    if (st.conv) {
      tbl.innerHTML = `<table class="data"><tr><th>N</th><th>L² error</th><th>H¹ error</th></tr>${st.conv.rows.map((x) => `<tr><td>${x.N}</td><td>${x.L2.toExponential(2)}</td><td>${x.H1.toExponential(2)}</td></tr>`).join('')}
        <tr><td>rate</td><td class="${Math.abs(st.conv.rL2 - (st.p + 1)) < 0.3 ? 'good' : 'bad'}">${st.conv.rL2.toFixed(2)}</td><td class="${Math.abs(st.conv.rH1 - st.p) < 0.3 ? 'good' : 'bad'}">${st.conv.rH1.toFixed(2)}</td></tr></table>`;
    } else tbl.innerHTML = '';
    selfCheck('bc2d errors finite', Number.isFinite(e.L2));
  }
  s1.onResize(show); s2.onResize(show);
  run();
}

/* ------------------------------------------------------------------ */
/* W2: sweep over γ                                                     */
/* ------------------------------------------------------------------ */
function gammaWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('$L^2$ error vs penalty $\\gamma$');
  const s1 = createCanvas(p1, { aspect: 0.75 });
  const p2 = L.panel('smallest eigenvalue of $(A+A^T)/2$ (coercivity) vs $\\gamma$');
  const s2 = createCanvas(p2, { aspect: 0.75 });
  const st = { p: 1, data: null };
  segmented(L.controls, { label: 'element (mesh: 81 unknowns)', value: '1', options: [{ value: '1', label: 'P1, N = 8' }, { value: '2', label: 'P2, N = 4' }], onChange: (v) => { st.p = +v; compute(); } });
  const out = readout(L.controls);
  const S = POISSON_MMS.wave;
  const gammas = Array.from({ length: 31 }, (_, k) => 10 ** (-1 + 5 * k / 30));
  function compute() {
    const N = st.p === 1 ? 8 : 4, mesh = triGrid(N, N, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.1 });
    const data = {};
    for (const m of ['nitsche', 'nitsche-nonsym', 'penalty']) {
      data[m] = gammas.map((g) => {
        const r = solvePoissonCG(mesh, { p: st.p, f: S.f, g: S.u, bc: m, gamma: g });
        const e = errorsCG(r.space, r.U, S.u, S.grad);
        return { g, L2: e.L2, lmin: lambdaMinSym(r.A).lmin };
      });
    }
    const rs = solvePoissonCG(mesh, { p: st.p, f: S.f, g: S.u, bc: 'strong' });
    data.strong = errorsCG(rs.space, rs.U, S.u, S.grad).L2;
    st.data = data;
    draw();
  }
  function draw() {
    if (!st.data) return;
    const C = seriesColors(), D = st.data, T = theme();
    const cols = { nitsche: C[0], 'nitsche-nonsym': C[2], penalty: C[1] };
    const all = [D.strong, ...['nitsche', 'nitsche-nonsym', 'penalty'].flatMap((m) => D[m].map((x) => x.L2))].filter((v) => Number.isFinite(v) && v > 0);
    const P = new Plot(s1, { xlim: [0.1, 1e4], ylim: [10 ** Math.floor(Math.log10(Math.min(...all))), Math.min(1e3, 10 ** Math.ceil(Math.log10(Math.max(...all))))], xlog: true, ylog: true, xlabel: 'γ', ylabel: 'L² error' });
    P.frame();
    P.hline(D.strong, { color: T.ink, dash: [6, 4], width: 1.5 });
    for (const m of ['nitsche', 'nitsche-nonsym', 'penalty']) P.line(D[m].map((x) => x.g), D[m].map((x) => x.L2), { color: cols[m], width: 2 });
    P.legend([{ label: 'strong', color: T.ink, dash: [6, 4] }, { label: 'Nitsche sym.', color: C[0] }, { label: 'Nitsche non-sym.', color: C[2] }, { label: 'penalty only', color: C[1] }], 'tr');
    const lm = ['nitsche', 'nitsche-nonsym', 'penalty'].flatMap((m) => D[m].map((x) => x.lmin));
    const lo = Math.min(...lm), hi = Math.max(...lm);
    const Q = new Plot(s2, { xlim: [0.1, 1e4], ylim: [lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)], xlog: true, xlabel: 'γ' });
    Q.frame(); Q.hline(0, { color: T.accent2, dash: [], width: 1.2 });
    for (const m of ['nitsche', 'nitsche-nonsym', 'penalty']) Q.line(D[m].map((x) => x.g), D[m].map((x) => x.lmin), { color: cols[m], width: 2 });
    const crit = D.nitsche.find((x) => x.lmin > 0);
    out.set(`symmetric Nitsche becomes coercive\n(λ_min > 0) for γ ≳ ${crit ? fmt(crit.g) : '—'}\nstrong BC L² error: ${D.strong.toExponential(2)}`);
    selfCheck('gamma sweep finite', Number.isFinite(D.strong));
  }
  s1.onResize(draw); s2.onResize(draw);
  compute();
}

/* ------------------------------------------------------------------ */
/* W3: hyperbolic boundary conditions                                   */
/* ------------------------------------------------------------------ */
function hyperWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('$u(x,t)$ for $u_t + u_x = 0$: inflow at $x=0$, outflow at $x=1$');
  p1.style.flex = '1 1 420px';
  const s1 = createCanvas(p1, { aspect: 0.5 });
  const st = { flux: 'upwind', inflow: 'weak', outflow: 'extrapolate', gOut: 0.8, t: 0 };
  const N = 100;
  const u0 = (x) => 0.6 * Math.exp(-150 * (x - 0.35) ** 2);
  const gIn = (t) => (t < 0.5 ? 0.5 * Math.sin(2 * Math.PI * t) ** 2 : 0);
  const exact = (x, t) => (x < t ? gIn(t - x) : u0(x - t));
  let d, u, step;
  const reset = () => {
    d = makeAdvBC1D({ N, a: 1, flux: st.flux, inflow: st.inflow, outflow: st.outflow, gIn, gOut: () => st.gOut });
    u = Float64Array.from(d.xc, u0);
    st.t = 0;
    step = makeStepper('ssprk3', N, (v) => d.enforce(v, st.t));
    draw();
  };
  select(L.controls, { label: 'numerical flux', value: st.flux, options: [{ value: 'upwind', label: 'upwind' }, { value: 'central', label: 'central' }], onChange: (v) => { st.flux = v; reset(); } });
  select(L.controls, { label: 'inflow boundary ($x=0$)', value: st.inflow, options: [{ value: 'weak', label: 'weak: ghost state g in the flux' }, { value: 'strong', label: 'strong: overwrite first cell' }], onChange: (v) => { st.inflow = v; reset(); } });
  select(L.controls, { label: 'outflow boundary ($x=1$)', value: st.outflow, options: [{ value: 'extrapolate', label: 'no data (extrapolate)' }, { value: 'weak', label: 'weak: offer value g_out in the flux' }, { value: 'strong', label: 'strong: overwrite last cell with g_out' }], onChange: (v) => { st.outflow = v; reset(); } });
  slider(L.controls, { label: 'outflow value $g_{\\text{out}}$', min: -1, max: 1, step: 0.05, value: st.gOut, onInput: (v) => { st.gOut = v; } });
  const anim = new Animator(fig, {
    step: () => {
      const dt = (st.flux === 'central' ? 0.2 : 0.5) * d.h;
      step(u, st.t, dt, d.rhs); st.t += dt; d.enforce(u, st.t);
      if (st.t > 2) return false;
    },
    draw: () => draw(), stepsPerFrame: 3,
  });
  const row = buttonRow(L.controls);
  const play = button(row, { label: '▶ Play', primary: true, onClick: () => anim.toggle() });
  anim.o.onState = (r) => play.setLabel(r ? '❚❚ Pause' : '▶ Play');
  button(row, { label: '↺ Reset', onClick: () => { anim.pause(); reset(); } });
  const out = readout(L.controls);
  function draw() {
    const T = theme(), C = seriesColors();
    const P = new Plot(s1, { xlim: [0, 1], ylim: [-1.05, 1.05], xlabel: 'x' });
    P.frame(); P.hline(0);
    const xs = Array.from({ length: 401 }, (_, i) => i / 400);
    P.line(xs, xs.map((x) => exact(x, st.t)), { color: T.faint, width: 4 });
    const edges = Array.from({ length: N + 1 }, (_, i) => i / N);
    P.bars(edges, u, { color: C[0], alpha: 0.15, width: 1.8 });
    P.arrow(0.02, 0.85, 0.12, 0.85, { color: C[2] }); P.text(0.13, 0.85, 'inflow', { color: C[2] });
    P.arrow(0.86, 0.85, 0.96, 0.85, { color: C[1] }); P.text(0.85, 0.85, 'outflow', { color: C[1], align: 'right' });
    let err = 0; for (let i = 0; i < N; i++) err += Math.abs(u[i] - exact(d.xc[i], st.t)) / N;
    out.set(`t = ${st.t.toFixed(2)}\nL¹ error = ${err.toExponential(2)}`);
    selfCheck('hyper finite', Number.isFinite(err));
  }
  s1.onResize(draw);
  reset();
}

initChapter(() => {
  mount('w-bc2d', bc2dWidget);
  mount('w-gamma', gammaWidget);
  mount('w-hyper', hyperWidget);
});
