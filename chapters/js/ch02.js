/**
 * @file Chapter 2 — Strong, weak, mixed & flux forms.
 *
 * Widgets:
 *  - w-ibp      : step-by-step derivation of the weak form (steps are in the HTML)
 *  - w-energy   : drag a piecewise-linear function; the Galerkin solution minimises the energy J(v)
 *  - w-galerkin : 1D Galerkin with hat functions: basis, matrix, solution, nodal exactness
 *  - w-forms    : the same 1D problem in strong (FD), weak (CG) and mixed form; solutions and fluxes
 */
import { initChapter, mount, selfCheck } from '../../lib/ui/chapter.js';
import { widgetLayout, slider, select, button, buttonRow, readout, fmt, h } from '../../lib/ui/controls.js';
import { createCanvas, theme, onPointer, seriesColors } from '../../lib/viz/canvas.js';
import { Plot } from '../../lib/viz/plot1d.js';
import { solvePoisson1D, errors1D } from '../../lib/core/elliptic/poisson1d.js';
import { csrToDense } from '../../lib/core/la/sparse.js';
import { gaussLegendre } from '../../lib/core/quad/gauss1d.js';
import { fitRate } from '../../lib/core/verify/rates.js';

/** Right-hand sides with exact solutions for −u'' = f on (0,1). */
const PROBLEMS = {
  sine: { label: 'u = sin(πx)  (f = π² sin πx)', u: (x) => Math.sin(Math.PI * x), du: (x) => Math.PI * Math.cos(Math.PI * x), f: (x) => Math.PI ** 2 * Math.sin(Math.PI * x) },
  mixed: { label: 'u = sin(3x) + x²', u: (x) => Math.sin(3 * x) + x * x, du: (x) => 3 * Math.cos(3 * x) + 2 * x, f: (x) => 9 * Math.sin(3 * x) - 2 },
  bump: {
    label: 'heat source near x = 0.3',
    // u = x(1−x)·exp(−30(x−0.3)²) ;  f = −u'' (computed symbolically below)
    u: (x) => x * (1 - x) * Math.exp(-30 * (x - 0.3) ** 2),
    du: (x) => { const e = Math.exp(-30 * (x - 0.3) ** 2); return (1 - 2 * x) * e + x * (1 - x) * (-60 * (x - 0.3)) * e; },
    f: (x) => {
      const e = Math.exp(-30 * (x - 0.3) ** 2), p = x * (1 - x), dp = 1 - 2 * x, s = -60 * (x - 0.3);
      // u = p e,  u' = (dp + p s) e,  u'' = (−2 + 2 dp s + p s' + p s²) e  with s' = −60
      return -(-2 + 2 * dp * s - 60 * p + p * s * s) * e;
    },
  },
};

/* ------------------------------------------------------------------ */
/* W1: step-through derivation                                         */
/* ------------------------------------------------------------------ */
function stepsWidget(fig) {
  const steps = [...fig.querySelectorAll('.step')];
  let k = 0;
  const row = buttonRow(fig);
  row.style.marginTop = '.6rem';
  const prev = button(row, { label: '← Back', onClick: () => show(k - 1) });
  const next = button(row, { label: 'Next step →', primary: true, onClick: () => show(k + 1) });
  const all = button(row, { label: 'Show all', onClick: () => { steps.forEach((s) => s.classList.add('shown')); k = steps.length - 1; upd(); } });
  const counter = h('span', 'readout');
  counter.style.marginLeft = '.5rem';
  row.appendChild(counter);
  function upd() {
    counter.textContent = `step ${k + 1} / ${steps.length}`;
    prev.el.disabled = k === 0; next.el.disabled = k === steps.length - 1;
  }
  function show(i) {
    k = Math.max(0, Math.min(steps.length - 1, i));
    steps.forEach((s, j) => s.classList.toggle('shown', j <= k));
    upd();
  }
  show(0);
  void all;
}

/* ------------------------------------------------------------------ */
/* W2: energy minimisation (Dirichlet principle)                        */
/* ------------------------------------------------------------------ */
function energyWidget(fig) {
  const L = widgetLayout(fig);
  const panel = L.panel('drag the white points: $v$ is piecewise linear with $v(0)=v(1)=0$');
  const surf = createCanvas(panel, { aspect: 0.5 });
  const N = 8, hh = 1 / N;
  const prob = PROBLEMS.sine;
  const sol = solvePoisson1D('cg', N, prob.f, 0, 0);
  const st = { v: Float64Array.from({ length: N + 1 }, (_, i) => (i === 0 || i === N ? 0 : 0.3 * Math.sin(2 * Math.PI * i / N) + 0.5)), drag: -1 };
  const G = gaussLegendre(8);
  /** J(v) = ½∫ v'² − ∫ f v for piecewise-linear v (exact derivative, Gauss for the load) */
  const J = (v) => {
    let s = 0;
    for (let i = 0; i < N; i++) {
      const a = i * hh, d = (v[i + 1] - v[i]) / hh;
      s += 0.5 * d * d * hh;
      for (let q = 0; q < G.x.length; q++) {
        const t = (G.x[q] + 1) / 2, x = a + t * hh;
        s -= G.w[q] / 2 * hh * prob.f(x) * (v[i] * (1 - t) + v[i + 1] * t);
      }
    }
    return s;
  };
  const Jmin = J(sol.U);
  const row = buttonRow(L.controls);
  button(row, { label: 'Snap to minimiser', primary: true, onClick: () => { st.v = Float64Array.from(sol.U); draw(); } });
  button(row, { label: 'Scramble', onClick: () => { for (let i = 1; i < N; i++) st.v[i] = Math.random() * 1.2 - 0.1; draw(); } });
  const out = readout(L.controls);
  let P;
  function draw() {
    const T = theme(), C = seriesColors();
    P = new Plot(surf, { xlim: [0, 1], ylim: [-0.2, 1.3], xlabel: 'x' });
    P.frame();
    const xs = [], ys = [];
    for (let k = 0; k <= 300; k++) { xs.push(k / 300); ys.push(prob.u(k / 300)); }
    P.line(xs, ys, { color: T.faint, width: 3 });
    const xn = Array.from({ length: N + 1 }, (_, i) => i * hh);
    P.line(xn, sol.U, { color: C[2], width: 1.5, dash: [5, 4] });
    P.line(xn, st.v, { color: C[0], width: 2.4 });
    P.points(xn.slice(1, N), st.v.slice(1, N), { color: '#ffffff', r: 6 });
    P.points(xn.slice(1, N), st.v.slice(1, N), { color: C[0], r: 6, hollow: true });
    P.legend([{ label: 'exact u', color: T.faint }, { label: 'Galerkin u_h (minimiser)', color: C[2], dash: [5, 4] }, { label: 'your v', color: C[0] }], 'tr');
    const Jv = J(st.v);
    out.set(`J(v)   = ${Jv.toFixed(6)}\nJ(u_h) = ${Jmin.toFixed(6)}\nJ(v) − J(u_h) = ${(Jv - Jmin).toExponential(3)}\n(always ≥ 0)`);
    selfCheck('energy minimiser', Jv - Jmin >= -1e-12);
  }
  onPointer(surf.canvas, {
    down: (x, y) => { st.drag = -1; for (let i = 1; i < N; i++) if (Math.hypot(P.X(i * hh) - x, P.Y(st.v[i]) - y) < 12) st.drag = i; return st.drag > 0 ? undefined : false; },
    move: (x, y) => { if (st.drag > 0) { st.v[st.drag] = Math.max(-0.2, Math.min(1.3, P.invY(y))); draw(); } },
    up: () => { st.drag = -1; },
  });
  surf.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W3: 1D Galerkin with hat functions                                   */
/* ------------------------------------------------------------------ */
function galerkinWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('basis functions $\\varphi_j$ scaled by their coefficients $U_j$, and their sum $u_h$');
  const s1 = createCanvas(p1, { aspect: 0.5 });
  const p2 = L.panel('the stiffness matrix $A_{ij} = \\int \\varphi_j\'\\varphi_i\'\\,dx$ (times $h$)');
  const s2 = createCanvas(p2, { aspect: 0.5 });
  const st = { N: 6, prob: 'sine' };
  select(L.controls, { label: 'problem', value: st.prob, options: Object.entries(PROBLEMS).map(([k, v]) => ({ value: k, label: v.label })), onChange: (v) => { st.prob = v; draw(); } });
  slider(L.controls, { label: 'number of cells $N$', min: 2, max: 24, step: 1, value: st.N, onInput: (v) => { st.N = v; draw(); } });
  const out = readout(L.controls);
  function draw() {
    const T = theme(), C = seriesColors(), prob = PROBLEMS[st.prob], N = st.N, hh = 1 / N;
    const sol = solvePoisson1D('cg', N, prob.f, prob.u(0), prob.u(1));
    const xs = [], ys = [];
    for (let k = 0; k <= 400; k++) { xs.push(k / 400); ys.push(prob.u(k / 400)); }
    const ymax = Math.max(...ys, ...sol.U) * 1.15 + 0.05, ymin = Math.min(0, ...ys, ...sol.U) - 0.05;
    const P = new Plot(s1, { xlim: [0, 1], ylim: [ymin, ymax], xlabel: 'x' });
    P.frame();
    for (let j = 1; j < N; j++) {
      const c = C[j % 4];
      P.line([(j - 1) * hh, j * hh, (j + 1) * hh], [0, sol.U[j], 0], { color: c, width: 1.3, alpha: 0.85 });
    }
    P.line(xs, ys, { color: T.faint, width: 4 });
    P.line(Array.from({ length: N + 1 }, (_, i) => i * hh), sol.U, { color: T.ink, width: 2 });
    P.points(Array.from({ length: N + 1 }, (_, i) => i * hh), sol.U, { color: T.ink, r: 3 });
    // matrix as a heat grid with numbers
    const A = csrToDense(sol.A), n = N - 1, ctx = s2.ctx;
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s2.w, s2.h);
    const cell = Math.min((s2.w - 20) / n, (s2.h - 20) / n), ox = (s2.w - cell * n) / 2, oy = (s2.h - cell * n) / 2;
    ctx.font = `${Math.max(8, Math.min(13, cell * 0.38))}px ${T.mono}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = A[i * n + j] * hh; // scale so entries are 2, −1
      ctx.fillStyle = v > 0 ? T.accent2 : v < 0 ? T.accent : T.elev;
      ctx.globalAlpha = v === 0 ? 1 : 0.85;
      ctx.fillRect(ox + j * cell + 1, oy + i * cell + 1, cell - 2, cell - 2);
      ctx.globalAlpha = 1;
      if (cell > 16 && v !== 0) { ctx.fillStyle = '#fff'; ctx.fillText(String(Math.round(v)), ox + (j + 0.5) * cell, oy + (i + 0.5) * cell); }
    }
    let nodal = 0;
    for (let i = 0; i <= N; i++) nodal = Math.max(nodal, Math.abs(sol.U[i] - prob.u(i * hh)));
    const e = errors1D('cg', sol, prob.u, prob.du);
    out.set(`unknowns: ${N - 1} (interior nodes)\nnon-zeros per row ≤ 3 (tridiagonal)\nmax nodal error = ${nodal.toExponential(2)}\nL² error        = ${e.uL2.toExponential(2)}`);
    selfCheck('galerkin nodal exactness', nodal < 1e-10);
  }
  s1.onResize(draw); s2.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W4: one problem, three forms                                         */
/* ------------------------------------------------------------------ */
function formsWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('solution $u$');
  const s1 = createCanvas(p1, { aspect: 0.55 });
  const p2 = L.panel('flux $\\sigma = -u\'$');
  const s2 = createCanvas(p2, { aspect: 0.55 });
  const st = { N: 6, prob: 'bump' };
  select(L.controls, { label: 'problem', value: st.prob, options: Object.entries(PROBLEMS).map(([k, v]) => ({ value: k, label: v.label })), onChange: (v) => { st.prob = v; draw(); } });
  slider(L.controls, { label: 'number of cells $N$', min: 2, max: 40, step: 1, value: st.N, onInput: (v) => { st.N = v; draw(); } });
  const out = readout(L.controls);
  const tableHost = h('div');
  L.controls.appendChild(tableHost);
  function draw() {
    const T = theme(), C = seriesColors(), prob = PROBLEMS[st.prob], N = st.N, hh = 1 / N;
    const fd = solvePoisson1D('fd', N, prob.f, prob.u(0), prob.u(1));
    const cg = solvePoisson1D('cg', N, prob.f, prob.u(0), prob.u(1));
    const mx = solvePoisson1D('mixed', N, prob.f, prob.u(0), prob.u(1));
    const xs = [], ys = [], sg = [];
    for (let k = 0; k <= 400; k++) { const x = k / 400; xs.push(x); ys.push(prob.u(x)); sg.push(-prob.du(x)); }
    const lo = Math.min(...ys), hi = Math.max(...ys), pad = 0.15 * (hi - lo || 1);
    const P = new Plot(s1, { xlim: [0, 1], ylim: [lo - pad, hi + pad], xlabel: 'x' });
    P.frame();
    P.line(xs, ys, { color: T.faint, width: 4 });
    P.line(cg.x, cg.U, { color: C[0], width: 2 });
    P.points(fd.x, fd.U, { color: C[1], r: 4.5, hollow: true });
    const edges = Array.from({ length: N + 1 }, (_, i) => i * hh);
    P.bars(edges, mx.uCell, { color: C[2], alpha: 0.12, width: 2 });
    P.legend([{ label: 'exact', color: T.faint }, { label: 'FD (strong form): point values', color: C[1], marker: true }, { label: 'CG (weak form): continuous P1', color: C[0] }, { label: 'mixed: piecewise-constant u_h', color: C[2] }], 'tl');
    const slo = Math.min(...sg), shi = Math.max(...sg), sp = 0.15 * (shi - slo || 1);
    const Q = new Plot(s2, { xlim: [0, 1], ylim: [slo - sp, shi + sp], xlabel: 'x' });
    Q.frame();
    Q.line(xs, sg, { color: T.faint, width: 4 });
    const fl = new Float64Array(N);
    for (let i = 0; i < N; i++) fl[i] = -(cg.U[i + 1] - cg.U[i]) / hh;
    Q.bars(edges, fl, { color: C[0], alpha: 0.08, width: 2 });
    Q.line(mx.x, mx.sigma, { color: C[2], width: 2.2 });
    Q.legend([{ label: 'exact σ = −u′', color: T.faint }, { label: 'CG: −u_h′ (jumps at nodes)', color: C[0] }, { label: 'mixed: σ_h (continuous)', color: C[2] }], 'tr');
    // rate table over N = 8..64
    const Ns = [8, 16, 32, 64], hs = Ns.map((n) => 1 / n);
    const rate = (form, key) => fitRate(hs, Ns.map((n) => errors1D(form, solvePoisson1D(form, n, prob.f, prob.u(0), prob.u(1)), prob.u, prob.du)[key]));
    tableHost.innerHTML = `<table class="data"><tr><th>form</th><th>rate u</th><th>rate σ</th></tr>
      <tr><td>FD</td><td>${rate('fd', 'uL2').toFixed(2)}</td><td>${rate('fd', 'sigmaL2').toFixed(2)}</td></tr>
      <tr><td>CG</td><td>${rate('cg', 'uL2').toFixed(2)}</td><td>${rate('cg', 'sigmaL2').toFixed(2)}</td></tr>
      <tr><td>mixed</td><td>${rate('mixed', 'uL2').toFixed(2)}</td><td>${rate('mixed', 'sigmaL2').toFixed(2)}</td></tr></table>`;
    const eF = errors1D('fd', fd, prob.u, prob.du), eC = errors1D('cg', cg, prob.u, prob.du), eM = errors1D('mixed', mx, prob.u, prob.du);
    out.set(`L² errors at N = ${N}:\n      u          σ\nFD    ${eF.uL2.toExponential(1)}   ${eF.sigmaL2.toExponential(1)}\nCG    ${eC.uL2.toExponential(1)}   ${eC.sigmaL2.toExponential(1)}\nmixed ${eM.uL2.toExponential(1)}   ${eM.sigmaL2.toExponential(1)}\n(observed rates over N = 8…64:)`);
    selfCheck('forms finite', [eF, eC, eM].every((e) => Number.isFinite(e.uL2)));
  }
  s1.onResize(draw); s2.onResize(draw);
  draw();
}

initChapter(() => {
  mount('w-ibp', stepsWidget);
  mount('w-energy', energyWidget);
  mount('w-galerkin', galerkinWidget);
  mount('w-forms', formsWidget);
});

export { fmt };
