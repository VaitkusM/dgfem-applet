/**
 * @file Chapter 10 — Domain decomposition.
 *
 * Widgets:
 *  - w-schwarz1d : Schwarz's alternating method for −u'' = f on two overlapping intervals (exact local solves)
 *  - w-schwarz2d : 2D Poisson on Kx×Ky overlapping subdomains: multiplicative / ASM-PCG / RAS-GMRES, ± coarse space
 *  - w-scaling   : iterations vs number of subdomains, with and without coarse space (computed live)
 *  - w-schur     : non-overlapping splitting, reordered (arrow) matrix, Schur complement, κ(S) vs κ(A)
 *  - w-dn        : Dirichlet–Neumann iteration with relaxation θ on two subdomains
 */
import { initChapter, mount, selfCheck } from '../../lib/ui/chapter.js';
import { widgetLayout, slider, select, segmented, checkbox, button, buttonRow, readout, fmt, debounce, h } from '../../lib/ui/controls.js';
import { createCanvas, View2D, theme, seriesColors } from '../../lib/viz/canvas.js';
import { Plot } from '../../lib/viz/plot1d.js';
import { Animator } from '../../lib/viz/anim.js';
import { drawTriField } from '../../lib/viz/field2d.js';
import { colorbar } from '../../lib/viz/colormap.js';
import { drawSpy } from '../../lib/viz/spy.js';
import { ddPoissonProblem, toVertices } from '../../lib/core/dd/problem.js';
import { boxPartition } from '../../lib/core/dd/partition.js';
import { schwarzSetup, solveSchwarz, alternatingSchwarz1D } from '../../lib/core/dd/schwarz.js';
import { schurSetup, denseSchur, denseCondition, localSchur, dirichletNeumannSetup, dnSpectrum } from '../../lib/core/dd/schur.js';
import { sparseSolve } from '../../lib/core/la/direct.js';
import { cg } from '../../lib/core/la/krylov.js';
import { conditionEstimate } from '../../lib/core/la/eig.js';
import { csrPermute, denseToCsr } from '../../lib/core/la/sparse.js';
import { fitRate } from '../../lib/core/verify/rates.js';

/** Play/pause + step + reset buttons wired to an Animator. */
function playControls(parent, anim, onStep, onReset) {
  const row = buttonRow(parent);
  const play = button(row, { label: '▶ Play', primary: true, onClick: () => anim.toggle() });
  button(row, { label: 'Step', onClick: () => { anim.pause(); onStep(); } });
  button(row, { label: '↺ Reset', onClick: () => { anim.pause(); onReset(); } });
  anim.o.onState = (r) => play.setLabel(r ? '❚❚ Pause' : '▶ Play');
  return play;
}

/** Semi-transparent version of a CSS colour (#rgb / #rrggbb / rgb()). */
function withAlpha(c, a) {
  c = c.trim();
  if (c.startsWith('#')) {
    const s = c.length === 4 ? c.slice(1).split('').map((x) => x + x).join('') : c.slice(1, 7);
    return `rgba(${parseInt(s.slice(0, 2), 16)},${parseInt(s.slice(2, 4), 16)},${parseInt(s.slice(4, 6), 16)},${a})`;
  }
  const m = c.match(/rgba?\(([^)]+)\)/);
  if (m) { const p = m[1].split(',').slice(0, 3).join(','); return `rgba(${p},${a})`; }
  return c;
}

/** Categorical colour for subdomain k. */
const subColor = (k) => {
  const C = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#b07aa1', '#76b7b2', '#edc948', '#ff9da7', '#9c755f'];
  return C[k % C.length];
};

/* ------------------------------------------------------------------ */
/* W1: Schwarz's alternating method in 1D                              */
/* ------------------------------------------------------------------ */
function schwarz1dWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('iterates: $u_1^n$ on $\\Omega_1=(0,\\beta)$, $u_2^n$ on $\\Omega_2=(\\alpha,1)$; dashed = exact');
  const s1 = createCanvas(p1, { aspect: 0.42 });
  p1.style.flex = '1 1 100%';
  const p2 = L.panel('interface error $|u(\\beta) - u_2^n(\\beta)|$');
  const s2 = createCanvas(p2, { aspect: 0.45 });
  const st = { c: 0.5, w: 0.2, rhs: 'lin', n: 4, frame: 0 };
  const MAXIT = 30;
  select(L.controls, {
    label: 'problem', value: st.rhs,
    options: [{ value: 'lin', label: "−u'' = 0, u(0)=0, u(1)=1" }, { value: 'par', label: "−u'' = 8, u(0)=u(1)=0" }],
    onChange: (v) => { st.rhs = v; reset(); },
  });
  slider(L.controls, { label: 'overlap width $\\beta-\\alpha$', min: 0.02, max: 0.6, step: 0.01, value: st.w, onInput: (v) => { st.w = v; draw(); } });
  slider(L.controls, { label: 'overlap centre $(\\alpha+\\beta)/2$', min: 0.35, max: 0.65, step: 0.01, value: st.c, onInput: (v) => { st.c = v; draw(); } });
  const anim = new Animator(fig, {
    step: () => { if (++st.frame % 24 === 0) { st.n++; if (st.n >= MAXIT) { st.n = MAXIT; return false; } } },
    draw: () => draw(),
  });
  playControls(L.controls, anim, () => { st.n = Math.min(MAXIT, st.n + 1); draw(); }, () => reset());
  const out = readout(L.controls);
  function reset() { st.n = 0; st.frame = 0; draw(); }
  function draw() {
    const T = theme(), C = seriesColors();
    const alpha = st.c - st.w / 2, beta = st.c + st.w / 2;
    const prob = st.rhs === 'lin' ? { f0: 0, uL: 0, uR: 1 } : { f0: 8, uL: 0, uR: 0 };
    const r = alternatingSchwarz1D({ alpha, beta, ...prob, g0: 0, iters: MAXIT });
    const ev = (c, x) => c[0] * x * x + c[1] * x + c[2];
    const P = new Plot(s1, { xlim: [0, 1], ylim: [-0.05, 1.12], xlabel: 'x' });
    P.frame();
    // overlap region
    const ctx = s1.ctx;
    ctx.save(); ctx.fillStyle = withAlpha(T.accent4, 0.13);
    ctx.fillRect(P.X(alpha), P.py0, P.X(beta) - P.X(alpha), P.py1 - P.py0); ctx.restore();
    P.vline(alpha, { color: T.soft }); P.vline(beta, { color: T.soft });
    P.text(alpha, 0.02, 'α', { align: 'right', dx: -4, baseline: 'bottom', color: T.soft }); P.text(beta, 0.02, 'β', { align: 'left', dx: 4, baseline: 'bottom', color: T.soft });
    const xs1 = [], xs2 = [];
    for (let i = 0; i <= 60; i++) { xs1.push(beta * i / 60); xs2.push(alpha + (1 - alpha) * i / 60); }
    const xe = Array.from({ length: 101 }, (_, i) => i / 100);
    P.line(xe, xe.map((x) => ev(r.exact, x)), { color: T.ink, dash: [5, 4], width: 1.4 });
    for (let k = 0; k < st.n; k++) {
      const a = 0.25 + 0.75 * ((k + 1) / st.n) ** 2;
      P.line(xs1, xs1.map((x) => ev(r.u1[k], x)), { color: C[0], alpha: a, width: k === st.n - 1 ? 2.4 : 1.3 });
      P.line(xs2, xs2.map((x) => ev(r.u2[k], x)), { color: C[1], alpha: a, width: k === st.n - 1 ? 2.4 : 1.3 });
    }
    // interface values: u_1 is given u(β) = previous u_2(β), u_2 is given u(α) = u_1(α)
    if (st.n > 0) {
      const a1 = r.u1[st.n - 1];
      P.points([beta], [r.gammaB[st.n - 1]], { color: C[0], r: 4 });
      P.points([alpha], [ev(a1, alpha)], { color: C[1], r: 4 });
    }
    P.legend([{ label: 'u₁ⁿ on (0, β)', color: C[0] }, { label: 'u₂ⁿ on (α, 1)', color: C[1] }, { label: 'exact u', color: T.ink, dash: [5, 4] }], 'tl');
    // error history
    const ue = ev(r.exact, beta), err = r.gammaB.map((g) => Math.abs(g - ue));
    const e0 = Math.max(err[0], 1e-300);
    const Q = new Plot(s2, { xlim: [0, MAXIT], ylim: [1e-12, 2], ylog: true, xlabel: 'iteration n' });
    Q.frame();
    const ns = err.map((_, k) => k);
    Q.line(ns, ns.map((k) => err[0] * r.rho ** k), { color: T.faint, dash: [4, 4], width: 1.2 });
    Q.line(ns.slice(0, st.n + 1), err.slice(0, st.n + 1), { color: C[0], width: 2 });
    Q.points(ns.slice(0, st.n + 1), err.slice(0, st.n + 1), { color: C[0], r: 2.5 });
    Q.legend([{ label: 'observed', color: C[0] }, { label: 'ρⁿ · error₀', color: T.faint, dash: [4, 4] }], 'tr');
    const obs = st.n >= 1 && err[st.n - 1] > 1e-14 ? err[st.n] / err[st.n - 1] : NaN;
    out.set(`α = ${alpha.toFixed(2)}, β = ${beta.toFixed(2)}\nρ = α(1−β)/(β(1−α)) = ${fmt(r.rho, 4)}\nn = ${st.n}   observed ratio = ${Number.isFinite(obs) ? fmt(obs, 4) : '—'}\nerror at β: ${fmt(err[st.n])}`);
    selfCheck('1D Schwarz contraction = ρ', st.n < 1 || !Number.isFinite(obs) || Math.abs(obs - r.rho) < 1e-6 || err[st.n] < 1e-12 * e0);
  }
  s1.onResize(draw); s2.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W2: overlapping Schwarz in 2D                                        */
/* ------------------------------------------------------------------ */
const RHS = {
  one: { label: 'f = 1', f: () => 1 },
  corner: { label: 'source near (0.15, 0.15)', f: (x, y) => 60 * Math.exp(-200 * ((x - 0.15) ** 2 + (y - 0.15) ** 2)) },
};

/** Draw the overlapping partition: translucent boxes Ω'_k and core outlines. */
function drawPartition(surf, view, part, o = {}) {
  const { ctx } = surf, T = theme();
  const h = 1 / part.N;
  if (!o.noFill) {
    ctx.save();
    part.subs.forEach((s, k) => {
      const [x0, x1, y0, y1] = s.box;
      ctx.fillStyle = withAlpha(subColor(k), 0.28);
      ctx.fillRect(view.X(x0), view.Y(y1), view.X(x1) - view.X(x0), view.Y(y0) - view.Y(y1));
    });
    ctx.restore();
  }
  ctx.save();
  // non-overlapping cut lines
  ctx.strokeStyle = o.cutColor || T.ink; ctx.lineWidth = 1.6;
  ctx.beginPath();
  for (const c of part.cutsX) { ctx.moveTo(view.X(c * h), view.Y(0)); ctx.lineTo(view.X(c * h), view.Y(1)); }
  for (const c of part.cutsY) { ctx.moveTo(view.X(0), view.Y(c * h)); ctx.lineTo(view.X(1), view.Y(c * h)); }
  ctx.stroke();
  if (o.boxes) {
    part.subs.forEach((s, k) => {
      const [x0, x1, y0, y1] = s.box;
      ctx.strokeStyle = subColor(k); ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]);
      ctx.strokeRect(view.X(x0) + 1, view.Y(y1) + 1, view.X(x1) - view.X(x0) - 2, view.Y(y0) - view.Y(y1) - 2);
    });
  }
  ctx.restore();
}

function schwarz2dWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('partition: cut lines and overlapping subdomains $\\Omega_k\'$ (dashed)');
  const s1 = createCanvas(p1, { aspect: 0.86 });
  const p2 = L.panel('');
  const p2label = h('div', 'canvas-label');
  p2.appendChild(p2label);
  const fieldLabel = () => {
    p2label.textContent = { ulog: 'iterate u⁽ⁿ⁾ after iteration n (log scale; black = zero or tiny)', log: 'error u_h − u⁽ⁿ⁾ (log scale)', lin: 'error u_h − u⁽ⁿ⁾ (signed)' }[st.scale];
  };
  const s2 = createCanvas(p2, { aspect: 0.86 });
  const p3 = L.panel('relative residual $\\|b-Au^{(n)}\\|/\\|b\\|$');
  const s3 = createCanvas(p3, { aspect: 0.32 });
  p3.style.flex = '1 1 100%';
  L.body.appendChild(p3); // below the controls
  const N = 32;
  const st = { Kx: 3, Ky: 3, ell: 1, method: 'asm-cg', coarse: false, rhs: 'corner', n: 2, frame: 0, res: null, scale: 'ulog' };
  const probs = {};
  const getProb = (key) => (probs[key] ||= ddPoissonProblem(N, RHS[key].f));
  segmented(L.controls, {
    label: 'method', value: st.method,
    options: [{ value: 'multiplicative', label: 'mult.' }, { value: 'asm-cg', label: 'ASM+CG' }, { value: 'ras-gmres', label: 'RAS+GMRES' }],
    onChange: (v) => { st.method = v; recompute(); },
  });
  const ck = checkbox(L.controls, { label: 'coarse space (Q1 on subdomain corners)', value: st.coarse, onChange: (v) => { st.coarse = v; recompute(); } });
  const rc = debounce(() => recompute(), 80);
  slider(L.controls, { label: 'subdomains in x, $K_x$', min: 1, max: 6, step: 1, value: st.Kx, onInput: (v) => { st.Kx = v; rc(); } });
  slider(L.controls, { label: 'subdomains in y, $K_y$', min: 1, max: 6, step: 1, value: st.Ky, onInput: (v) => { st.Ky = v; rc(); } });
  slider(L.controls, { label: 'overlap layers $\\ell$ ($\\delta = 2\\ell h$)', min: 1, max: 4, step: 1, value: st.ell, onInput: (v) => { st.ell = v; rc(); } });
  select(L.controls, {
    label: 'show', value: st.scale,
    options: [{ value: 'ulog', label: 'iterate: log₁₀|u⁽ⁿ⁾|' }, { value: 'log', label: 'error: log₁₀|u_h − u⁽ⁿ⁾|' }, { value: 'lin', label: 'error u_h − u⁽ⁿ⁾ (signed)' }],
    onChange: (v) => { st.scale = v; fieldLabel(); drawField(); },
  });
  select(L.controls, { label: 'right-hand side', value: st.rhs, options: Object.entries(RHS).map(([value, r]) => ({ value, label: r.label })), onChange: (v) => { st.rhs = v; recompute(); } });
  const anim = new Animator(fig, {
    step: () => { if (++st.frame % 12 === 0) { st.n++; if (st.n >= st.res.iterates.length - 1) { st.n = st.res.iterates.length - 1; return false; } } },
    draw: () => drawField(),
  });
  playControls(L.controls, anim, () => { st.n = Math.min(st.res.iterates.length - 1, st.n + 1); drawField(); }, () => { st.n = 0; st.frame = 0; drawField(); });
  const out = readout(L.controls);
  ck.el.title = 'R₀ᵀ A₀⁻¹ R₀ with bilinear coarse functions';

  function recompute() {
    const prob = getProb(st.rhs);
    const part = boxPartition(N, st.Kx, st.Ky, st.ell);
    const dd = schwarzSetup(prob, part, { coarse: st.coarse ? 'q1' : 'none' });
    const maxIter = st.method === 'multiplicative' ? 150 : 80;
    const res = solveSchwarz(dd, prob.b, { method: st.method, tol: 1e-8, maxIter, keepIterates: true });
    const uh = prob.uh || (prob.uh = sparseSolve(prob.A, prob.b));
    let m = 0; for (const v of uh) m = Math.max(m, Math.abs(v));
    let dmax = 0; for (let i = 0; i < uh.length; i++) dmax = Math.max(dmax, Math.abs(res.x[i] - uh[i]));
    st.res = { ...res, part, prob, uh, umax: m, nc: dd.coarse ? dd.coarse.nc : 0, dmax };
    st.n = Math.min(st.n, res.iterates.length - 1);
    if (anim.running) st.n = 0;
    selfCheck('2D Schwarz converges to the monolithic solution', !res.converged || dmax < 1e-6 * m);
    drawStatic(); drawField();
  }
  function drawStatic() {
    const T = theme(), { part } = st.res;
    const view = new View2D(s1, [0, 1, 0, 1], { equal: true, pad: 6 });
    s1.ctx.fillStyle = T.bg; s1.ctx.fillRect(0, 0, s1.w, s1.h);
    // fine grid (every cell)
    const ctx = s1.ctx;
    ctx.save(); ctx.strokeStyle = T.rule; ctx.lineWidth = 0.6; ctx.beginPath();
    for (let i = 0; i <= N; i++) { ctx.moveTo(view.X(i / N), view.Y(0)); ctx.lineTo(view.X(i / N), view.Y(1)); ctx.moveTo(view.X(0), view.Y(i / N)); ctx.lineTo(view.X(1), view.Y(i / N)); }
    ctx.stroke(); ctx.restore();
    drawPartition(s1, view, part, { boxes: true });
    if (st.coarse) { // coarse nodes
      ctx.save(); ctx.fillStyle = T.ink;
      for (let my = 1; my < part.Ky; my++) for (let mx = 1; mx < part.Kx; mx++) {
        ctx.beginPath(); ctx.arc(view.X(part.cutsX[mx] / N), view.Y(part.cutsY[my] / N), 4, 0, 7); ctx.fill();
      }
      ctx.restore();
    }
  }
  function drawField() {
    const T = theme(), C = seriesColors(), R = st.res;
    const view = new View2D(s2, [0, 1, 0, 1], { equal: true, pad: [6, 62, 6, 6] });
    s2.ctx.fillStyle = T.bg; s2.ctx.fillRect(0, 0, s2.w, s2.h);
    const x = R.iterates[st.n];
    const E = new Float64Array(R.prob.n);
    for (let i = 0; i < E.length; i++) E[i] = R.uh[i] - x[i];
    const U = toVertices(R.prob, st.scale === 'ulog' ? x : E), tris = R.prob.mesh.tris;
    const m = R.umax, lg = st.scale !== 'lin';
    const lhi = Math.ceil(Math.log10(m)), llo = lhi - 8; // 8 decades below max|u_h|
    const ev = (t, l0, l1, l2) => l0 * U[tris[3 * t]] + l1 * U[tris[3 * t + 1]] + l2 * U[tris[3 * t + 2]];
    drawTriField(s2, view, R.prob.mesh.nodes, tris, lg ? (t, l0, l1, l2) => Math.log10(Math.max(Math.abs(ev(t, l0, l1, l2)), 1e-300)) : ev,
      lg ? { cmap: 'magma', lo: llo, hi: lhi } : { cmap: 'rdbu', lo: -m, hi: m });
    drawPartition(s2, view, R.part, { noFill: true, cutColor: withAlpha(lg ? '#ffffff' : T.ink, 0.55) });
    colorbar(s2.ctx, lg ? 'magma' : 'rdbu', view.X(1) + 8, view.Y(1), 10, view.Y(0) - view.Y(1), lg ? llo : -m, lg ? lhi : m,
      { ink: T.soft, fmt: lg ? (v) => `1e${v}` : undefined });
    // residual history
    const H = R.history, nMax = Math.max(10, H.length - 1);
    const P = new Plot(s3, { xlim: [0, nMax], ylim: [1e-9, 2], ylog: true, xlabel: 'iteration n' });
    P.frame();
    P.hline(1e-8, { color: T.faint });
    const ns = H.map((_, k) => k);
    P.line(ns, H, { color: C[0], width: 2 });
    P.points([st.n], [H[st.n]], { color: C[1], r: 5 });
    const methodName = { multiplicative: 'multiplicative Schwarz (stationary)', 'asm-cg': 'additive Schwarz + CG', 'ras-gmres': 'restricted additive Schwarz + GMRES' }[st.method];
    P.legend([{ label: `${methodName}${st.coarse ? ', two-level' : ', one-level'}`, color: C[0] }], 'tr');
    let emax = 0; for (const v of E) emax = Math.max(emax, Math.abs(v));
    out.set(`${R.part.subs.length} subdomains, H ≈ ${fmt(1 / Math.max(st.Kx, st.Ky))}, h = 1/${N}, δ = ${2 * st.ell}h\n` +
      `coarse functions n_c = ${R.nc}\n` +
      `iterations to 1e-8: ${R.converged ? R.iters : `> ${R.iters}`}\n` +
      `n = ${st.n}: rel. residual ${fmt(H[st.n])}\n         max |error| ${fmt(emax)}`);
    selfCheck('2D Schwarz iterate finite', Number.isFinite(emax));
  }
  s1.onResize(() => st.res && drawStatic()); s2.onResize(() => st.res && drawField()); s3.onResize(() => st.res && drawField());
  fieldLabel();
  recompute();
}

/* ------------------------------------------------------------------ */
/* W3: scalability experiment                                          */
/* ------------------------------------------------------------------ */
function scalingWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('iterations to reach $\\|b-Au\\|/\\|b\\| \\le 10^{-8}$ with $K\\times K$ subdomains');
  const s1 = createCanvas(p1, { aspect: 0.55 });
  const st = { mode: 'weak', ell: 1, data: [], running: false, token: 0 };
  select(L.controls, {
    label: 'experiment', value: st.mode,
    options: [{ value: 'weak', label: 'weak scaling: H/h = 8 (N = 8K)' }, { value: 'strong', label: 'strong scaling: N = 48 fixed' }],
    onChange: (v) => { st.mode = v; run(); },
  });
  slider(L.controls, { label: 'overlap layers $\\ell$ ($\\delta = 2\\ell h$)', min: 1, max: 3, step: 1, value: st.ell, onChange: (v) => { st.ell = v; run(); } });
  const row = buttonRow(L.controls);
  button(row, { label: '▶ Run experiment', primary: true, onClick: () => run() });
  const out = readout(L.controls);
  const tableDiv = h('div', 'wide');
  fig.insertBefore(tableDiv, fig.querySelector('figcaption'));
  const SERIES = [
    { key: 'cg', label: 'CG, no preconditioner' },
    { key: 'asm1', label: 'ASM + CG, one-level' },
    { key: 'asm2', label: 'ASM + CG, two-level' },
    { key: 'ras1', label: 'RAS + GMRES, one-level' },
    { key: 'ras2', label: 'RAS + GMRES, two-level' },
  ];
  function run() {
    const token = ++st.token;
    st.data = [];
    const Ks = st.mode === 'weak' ? [1, 2, 3, 4, 5, 6, 7, 8] : [1, 2, 3, 4, 6, 8];
    let idx = 0;
    const t0 = performance.now();
    const next = () => {
      if (token !== st.token) return; // superseded
      if (idx >= Ks.length) { out.set(out.el.textContent.replace('running…', 'done') + `\ntime ${fmt((performance.now() - t0) / 1000, 2)} s`); finish(); return; }
      const K = Ks[idx++], N = st.mode === 'weak' ? 8 * K : 48;
      const prob = ddPoissonProblem(N), part = boxPartition(N, K, K, st.ell);
      const row = { K, N };
      row.cg = cg(prob.A, prob.b, { tol: 1e-8 }).iters;
      for (const [lvl, coarse] of [[1, 'none'], [2, 'q1']]) {
        const dd = schwarzSetup(prob, part, { coarse });
        row[`asm${lvl}`] = solveSchwarz(dd, prob.b, { method: 'asm-cg', tol: 1e-8, maxIter: 400 }).iters;
        row[`ras${lvl}`] = solveSchwarz(dd, prob.b, { method: 'ras-gmres', tol: 1e-8, maxIter: 400 }).iters;
      }
      st.data.push(row);
      out.set(`running… K = ${K} (N = ${N}, ${prob.n} unknowns)`);
      draw();
      setTimeout(next, 0);
    };
    setTimeout(next, 0);
  }
  function finish() {
    const d = st.data;
    tableDiv.innerHTML = `<table class="data"><tr><th>K×K</th><th>N</th>${SERIES.map((s) => `<th>${s.label}</th>`).join('')}</tr>` +
      d.map((r) => `<tr><td>${r.K}×${r.K}</td><td>${r.N}</td>${SERIES.map((s) => `<td>${r[s.key]}</td>`).join('')}</tr>`).join('') + '</table>';
    if (d.length >= 4) {
      const last = d[d.length - 1], second = d[1];
      selfCheck('scaling: one-level grows', last.asm1 > second.asm1);
      selfCheck('scaling: two-level better at many subdomains', last.asm2 < last.asm1);
    }
  }
  function draw() {
    const T = theme(), C = seriesColors();
    const d = st.data;
    const ymax = Math.max(20, ...d.flatMap((r) => SERIES.map((s) => r[s.key]))) * 1.08;
    const P = new Plot(s1, { xlim: [0, 8.5], ylim: [0, ymax], xlabel: 'subdomains per direction K  (K² subdomains)', ylabel: 'iterations' });
    P.frame();
    const xs = d.map((r) => r.K);
    SERIES.forEach((s, i) => {
      const ys = d.map((r) => r[s.key]);
      const dash = s.key.startsWith('ras') ? [5, 3] : s.key === 'cg' ? [2, 3] : [];
      const col = s.key === 'cg' ? T.soft : s.key.endsWith('1') ? C[1] : C[0];
      P.line(xs, ys, { color: col, width: 2, dash });
      P.points(xs, ys, { color: col, r: 3 });
    });
    P.legend(SERIES.map((s) => ({ label: s.label, color: s.key === 'cg' ? T.soft : s.key.endsWith('1') ? C[1] : C[0], dash: s.key.startsWith('ras') ? [5, 3] : s.key === 'cg' ? [2, 3] : [] })), 'tl');
  }
  s1.onResize(draw);
  draw();
  run();
}

/* ------------------------------------------------------------------ */
/* W4: Schur complement                                                */
/* ------------------------------------------------------------------ */
function schurWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('interiors $I_k$ (coloured) and interface $\\Gamma$ (rings)');
  const s1 = createCanvas(p1, { aspect: 1 });
  const p2 = L.panel('reordered matrix $[I_1,\\dots,I_K,\\Gamma]$');
  const s2 = createCanvas(p2, { aspect: 1 });
  const p3 = L.panel('$S$ (dense!)');
  const s3 = createCanvas(p3, { aspect: 1 });
  for (const p of [p1, p2, p3]) p.style.flex = '1 1 190px';
  const p4 = L.panel('condition numbers versus mesh size');
  const s4 = createCanvas(p4, { aspect: 0.4 });
  p4.style.flex = '1 1 100%';
  L.body.appendChild(p4);
  const st = { part: '2x2', N: 12, kappa: {} };
  select(L.controls, {
    label: 'subdomains', value: st.part,
    options: [{ value: '2x1', label: '2 × 1' }, { value: '2x2', label: '2 × 2' }, { value: '3x3', label: '3 × 3 (centre floats)' }],
    onChange: (v) => { st.part = v; drawTop(); computeKappa(); },
  });
  slider(L.controls, { label: 'cells per side $N$ (pictures)', min: 6, max: 24, step: 6, value: st.N, onInput: (v) => { st.N = v; drawTop(); } });
  const out = readout(L.controls);
  const out2 = readout(L.controls);
  const dims = () => st.part.split('x').map(Number);

  function drawTop() {
    const T = theme(), [Kx, Ky] = dims(), N = st.N;
    const prob = ddPoissonProblem(N), su = schurSetup(prob, Kx, Ky);
    // (1) vertices
    const view = new View2D(s1, [0, 1, 0, 1], { equal: true, pad: 10 });
    const ctx = s1.ctx;
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    ctx.save(); ctx.strokeStyle = T.rule; ctx.lineWidth = 0.7; ctx.beginPath();
    for (let c = 0; c < N; c++) for (let r = 0; r < N; r++) { // triangulated cells
      ctx.moveTo(view.X(c / N), view.Y(r / N)); ctx.lineTo(view.X((c + 1) / N), view.Y((r + 1) / N));
    }
    for (let i = 0; i <= N; i++) { ctx.moveTo(view.X(i / N), view.Y(0)); ctx.lineTo(view.X(i / N), view.Y(1)); ctx.moveTo(view.X(0), view.Y(i / N)); ctx.lineTo(view.X(1), view.Y(i / N)); }
    ctx.stroke(); ctx.restore();
    const rad = Math.max(2.5, Math.min(6, 0.22 * view.sx / N));
    const xy = (f) => [(f % (N - 1)) + 1, Math.floor(f / (N - 1)) + 1];
    su.interiors.forEach((I, k) => {
      ctx.fillStyle = subColor(k);
      for (const f of I) { const [i, j] = xy(f); ctx.beginPath(); ctx.arc(view.X(i / N), view.Y(j / N), rad, 0, 7); ctx.fill(); }
    });
    ctx.save(); ctx.strokeStyle = T.ink; ctx.lineWidth = 2; ctx.fillStyle = T.bg;
    for (const f of su.gamma) { const [i, j] = xy(f); ctx.beginPath(); ctx.arc(view.X(i / N), view.Y(j / N), rad + 0.5, 0, 7); ctx.fill(); ctx.stroke(); }
    ctx.restore();
    // (2) spy of P A Pᵀ
    const B = csrPermute(prob.A, su.perm);
    drawSpy(s2, B, { blocks: su.blocks, magnitude: true });
    // (3) dense S
    const S = denseSchur(su), nG = su.gamma.length;
    drawSpy(s3, denseToCsr(S.map((v) => (Math.abs(v) < 1e-12 ? 0 : v)), nG), { magnitude: true, color: T.accent2 });
    let nnzS = 0; for (const v of S) if (Math.abs(v) > 1e-12) nnzS++;
    let msg = `N = ${N}: n = ${prob.n}, |Γ| = ${nG}\n|I_k| = ${su.interiors.map((I) => I.length).join(', ')}\nS: ${nG}×${nG}, ${fmt(100 * nnzS / (nG * nG), 3)} % non-zero`;
    if (Kx === 3 && Ky === 3) {
      const loc = localSchur(su, 4);
      const m = loc.gammaLocal.length;
      let rs = 0; for (let i = 0; i < m; i++) { let s = 0; for (let j = 0; j < m; j++) s += loc.S[i * m + j]; rs = Math.max(rs, Math.abs(s)); }
      msg += `\ncentre subdomain: max|S⁽⁵⁾·1| = ${fmt(rs, 2)}\n→ singular (floating subdomain)`;
      selfCheck('floating subdomain local Schur complement kills constants', rs < 1e-10);
    }
    out.set(msg);
  }

  let kToken = 0;
  function computeKappa() {
    const token = ++kToken, key = st.part, [Kx, Ky] = dims();
    const Ns = [8, 16, 32, 64];
    st.kappa[key] ||= [];
    const rows = st.kappa[key];
    let idx = rows.length;
    const step = () => {
      if (token !== kToken) return;
      if (idx >= Ns.length) { drawKappa(); return; }
      const N = Ns[idx++], prob = ddPoissonProblem(N);
      const su = schurSetup(prob, Kx, Ky);
      const cS = denseCondition(denseSchur(su), su.gamma.length);
      const cA = conditionEstimate(prob.A, { k: 80 });
      const exact = 1 / Math.tan(Math.PI / (2 * N)) ** 2;
      selfCheck('Lanczos κ(A) matches cot²(πh/2)', Math.abs(cA.kappa - exact) < 3e-3 * exact);
      rows.push({ N, kA: cA.kappa, kS: cS.kappa, lminS: cS.lmin, lmaxS: cS.lmax });
      drawKappa();
      setTimeout(step, 0);
    };
    drawKappa();
    setTimeout(step, 0);
  }
  function drawKappa() {
    const T = theme(), C = seriesColors(), rows = st.kappa[st.part] || [];
    const P = new Plot(s4, { xlim: [1 / 90, 1 / 6], ylim: [1, 5000], xlog: true, ylog: true, xlabel: 'mesh size h = 1/N', ylabel: 'κ' });
    P.frame();
    const hs = rows.map((r) => 1 / r.N);
    P.line(hs, rows.map((r) => r.kA), { color: C[1], width: 2 }); P.points(hs, rows.map((r) => r.kA), { color: C[1], r: 3.5 });
    P.line(hs, rows.map((r) => r.kS), { color: C[0], width: 2 }); P.points(hs, rows.map((r) => r.kS), { color: C[0], r: 3.5 });
    P.legend([{ label: 'κ(A)  (whole stiffness matrix)', color: C[1] }, { label: 'κ(S)  (interface Schur complement)', color: C[0] }], 'tr');
    if (rows.length >= 1) { // reference slopes through the first points
      const hh = [1 / 8, 1 / 80];
      P.line(hh, hh.map((x) => 0.5 * rows[0].kA * (8 * x) ** -2), { color: T.faint, dash: [4, 4], width: 1.2 });
      P.line(hh, hh.map((x) => 0.5 * rows[0].kS * (8 * x) ** -1), { color: T.faint, dash: [4, 4], width: 1.2 });
      P.text(1 / 40, 0.5 * rows[0].kA * (8 / 40) ** -2, 'slope −2', { color: T.soft, dy: 12, align: 'center' });
      P.text(1 / 40, 0.5 * rows[0].kS * (8 / 40) ** -1, 'slope −1', { color: T.soft, dy: 12, align: 'center' });
    }
    if (rows.length >= 2) {
      const rA = -fitRate(hs, rows.map((r) => r.kA)), rS = -fitRate(hs, rows.map((r) => r.kS));
      const pad = (v, n) => String(v).padStart(n);
      out2.set(`fitted: κ(A) ~ h^-${fmt(rA, 3)}\n        κ(S) ~ h^-${fmt(rS, 3)}\n N    κ(A)   κ(S)  λmin(S)\n` +
        rows.map((r) => `${pad(r.N, 2)} ${pad(fmt(r.kA, 4), 7)} ${pad(fmt(r.kS, 3), 6)}  ${fmt(r.lminS, 3)}`).join('\n') +
        `\nλmax(S) ∈ [${fmt(Math.min(...rows.map((r) => r.lmaxS)), 3)}, ${fmt(Math.max(...rows.map((r) => r.lmaxS)), 3)}]`);
      if (rows.length === 4) { selfCheck('κ(A) ~ h^-2', Math.abs(rA - 2) < 0.1); selfCheck('κ(S) ~ h^-1', Math.abs(rS - 1) < 0.15); }
    } else out2.set('computing κ(A) (Lanczos) and κ(S) (dense eigenvalues)…');
  }
  const redrawTop = debounce(drawTop, 30);
  s1.onResize(redrawTop); s2.onResize(redrawTop); s3.onResize(redrawTop); s4.onResize(drawKappa);
  drawTop();
  computeKappa();
}

/* ------------------------------------------------------------------ */
/* W5: Dirichlet–Neumann                                              */
/* ------------------------------------------------------------------ */
function dnWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('$u^{(n)}$: Dirichlet solve in $\\Omega_1$ (left), Neumann solve in $\\Omega_2$ (right)');
  const s1 = createCanvas(p1, { aspect: 1 });
  const p2 = L.panel('interface values $\\lambda^n(y)$ on $\\Gamma$; dashed = exact');
  const s2 = createCanvas(p2, { aspect: 1 });
  const p3 = L.panel('interface error $\\max_\\Gamma|\\lambda^n - u_h|$');
  const s3 = createCanvas(p3, { aspect: 0.32 });
  p3.style.flex = '1 1 100%';
  L.body.appendChild(p3);
  const N = 32, MAXIT = 30;
  const f = (x, y) => 1 + 4 * Math.exp(-40 * ((x - 0.2) ** 2 + (y - 0.3) ** 2));
  const prob = ddPoissonProblem(N, f), uh = sparseSolve(prob.A, prob.b);
  const st = { s: 8, theta: 0.5, n: 3, frame: 0, run: null };
  const cache = {};
  const rc = debounce(() => recompute(), 60);
  slider(L.controls, { label: 'interface position $x_\\Gamma$', min: 4 / N, max: 28 / N, step: 1 / N, value: st.s / N, format: (v) => fmt(v, 3), onInput: (v) => { st.s = Math.round(v * N); rc(); } });
  const thS = slider(L.controls, { label: 'relaxation $\\theta$', min: 0.05, max: 1.2, step: 0.01, value: st.theta, onInput: (v) => { st.theta = v; rc(); } });
  const row = buttonRow(L.controls);
  button(row, { label: 'use θ_opt', onClick: () => { const sp = cache[st.s].sp; st.theta = Math.round(sp.thetaOpt * 100) / 100; thS.set(st.theta); recompute(); } });
  const anim = new Animator(fig, {
    step: () => { if (++st.frame % 16 === 0) { st.n++; if (st.n >= st.run.lams.length - 1) { st.n = st.run.lams.length - 1; return false; } } },
    draw: () => draw(),
  });
  playControls(L.controls, anim, () => { st.n = Math.min(st.run.lams.length - 1, st.n + 1); draw(); }, () => { st.n = 0; st.frame = 0; draw(); });
  const out = readout(L.controls);
  function recompute() {
    if (!cache[st.s]) {
      const dn = dirichletNeumannSetup(prob, st.s);
      cache[st.s] = { dn, sp: dnSpectrum(dn.S1, dn.S2, dn.G.length) };
    }
    const { dn, sp } = cache[st.s];
    const uG = Float64Array.from(dn.G, (g) => uh[g]);
    let lam = new Float64Array(dn.G.length);
    const lams = [lam], xs = [new Float64Array(prob.n)], err = [Math.max(...uG.map(Math.abs))];
    for (let it = 0; it < MAXIT; it++) {
      const r = dn.step(lam, st.theta);
      lam = r.lambda; lams.push(lam); xs.push(r.x);
      let e = 0; for (let i = 0; i < lam.length; i++) e = Math.max(e, Math.abs(lam[i] - uG[i]));
      err.push(e);
      if (!(e < 1e3)) break;
    }
    st.run = { dn, sp, uG, lams, xs, err };
    st.n = Math.min(st.n, lams.length - 1);
    draw();
  }
  function draw() {
    const T = theme(), C = seriesColors(), R = st.run, xG = st.s / N;
    const umax = Math.max(...uh);
    // (1) field
    const view = new View2D(s1, [0, 1, 0, 1], { equal: true, pad: [6, 56, 6, 6] });
    s1.ctx.fillStyle = T.bg; s1.ctx.fillRect(0, 0, s1.w, s1.h);
    const U = toVertices(prob, R.xs[st.n]), tris = prob.mesh.tris;
    drawTriField(s1, view, prob.mesh.nodes, tris, (t, l0, l1, l2) => l0 * U[tris[3 * t]] + l1 * U[tris[3 * t + 1]] + l2 * U[tris[3 * t + 2]], { cmap: 'viridis', lo: 0, hi: umax });
    const ctx = s1.ctx;
    ctx.save(); ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2.5; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(view.X(xG), view.Y(0)); ctx.lineTo(view.X(xG), view.Y(1)); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = '#ffffff'; ctx.font = `600 13px ${T.ui}`; ctx.textAlign = 'center';
    ctx.fillText('Ω₁ (D)', view.X(xG / 2), view.Y(0.93)); ctx.fillText('Ω₂ (N)', view.X((1 + xG) / 2), view.Y(0.93));
    ctx.restore();
    colorbar(s1.ctx, 'viridis', view.X(1) + 8, view.Y(1), 10, view.Y(0) - view.Y(1), 0, umax, { ink: T.soft });
    // (2) interface traces
    const ys = Array.from(R.dn.G, (_, k) => (k + 1) / N);
    const lmax = Math.max(umax * 1.6, 1e-9);
    const P = new Plot(s2, { xlim: [0, 1], ylim: [-0.4 * lmax, lmax], xlabel: 'y along Γ' });
    P.frame(); P.hline(0);
    P.line([0, ...ys, 1], [0, ...R.uG, 0], { color: T.ink, dash: [5, 4], width: 1.6 });
    for (let k = 0; k <= st.n; k++) {
      const a = k === st.n ? 1 : 0.2 + 0.5 * (k / Math.max(1, st.n));
      P.line([0, ...ys, 1], [0, ...R.lams[k], 0], { color: C[0], alpha: a, width: k === st.n ? 2.4 : 1.1 });
    }
    P.legend([{ label: `λⁿ, n = ${st.n}`, color: C[0] }, { label: 'u_h on Γ', color: T.ink, dash: [5, 4] }], 'tr');
    // (3) error history
    const Q = new Plot(s3, { xlim: [0, MAXIT], ylim: [1e-14, 1e3], ylog: true, xlabel: 'iteration n' });
    Q.frame();
    const ns = R.err.map((_, k) => k), rho = R.sp.rho(st.theta);
    Q.line(ns, ns.map((k) => R.err[0] * rho ** k), { color: T.faint, dash: [4, 4], width: 1.2 });
    Q.line(ns, R.err, { color: C[0], width: 1.4, alpha: 0.4 });
    Q.line(ns.slice(0, st.n + 1), R.err.slice(0, st.n + 1), { color: C[0], width: 2.2 });
    Q.points([st.n], [R.err[st.n]], { color: C[1], r: 4.5 });
    Q.legend([{ label: 'observed', color: C[0] }, { label: 'ρ(θ)ⁿ · error₀', color: T.faint, dash: [4, 4] }], 'tr');
    const sp = R.sp;
    out.set(`eig(S₂⁻¹S₁) ∈ [${fmt(sp.muMin, 4)}, ${fmt(sp.muMax, 4)}]\nρ(θ) = max|1 − θ(1+μ)| = ${fmt(rho, 4)}${rho >= 1 ? '  ⇒ no convergence' : ''}\nθ_opt = 2/(2+μ_min+μ_max) = ${fmt(sp.thetaOpt, 4)}\nn = ${st.n}: error ${fmt(R.err[st.n])}`);
    selfCheck('DN iterate finite', R.xs.every((x) => Number.isFinite(x[0])));
  }
  s1.onResize(draw); s2.onResize(draw); s3.onResize(draw);
  recompute();
  // the default setting must converge (θ = ½, interface at x = 1/4)
  selfCheck('DN θ = ½ converges', st.run.err[st.run.err.length - 1] < 1e-8 * st.run.err[0]);
}

initChapter(() => {
  mount('w-schwarz1d', schwarz1dWidget);
  mount('w-schwarz2d', schwarz2dWidget);
  mount('w-scaling', scalingWidget);
  mount('w-schur', schurWidget);
  mount('w-dn', dnWidget);
});
