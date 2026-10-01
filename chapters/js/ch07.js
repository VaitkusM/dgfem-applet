/**
 * @file Chapter 7 — Flux reconstruction.
 *
 * Widgets:
 *  - w-recon : discontinuous flux, interface fluxes and the corrected continuous flux (whole mesh + one element)
 *  - w-corr  : correction functions g_DG, g_2, g_Ga, VCJH(c) and their derivatives
 *  - w-frdg  : FR vs DG run side by side, pointwise difference on a log axis
 *  - w-cflc  : maximal stable CFL number vs the VCJH parameter c, spectrum at the selected c
 *  - w-accc  : L² error vs c for two meshes (observed order)
 */
import { initChapter, mount, selfCheck } from '../../lib/ui/chapter.js';
import { widgetLayout, slider, select, checkbox, button, buttonRow, readout, fmt } from '../../lib/ui/controls.js';
import { createCanvas, theme, seriesColors } from '../../lib/viz/canvas.js';
import { Plot } from '../../lib/viz/plot1d.js';
import { Animator } from '../../lib/viz/anim.js';
import { drawPiecewise, elementColors, shadeStabilityRegion } from '../../lib/viz/dgviz.js';
import { makeFR1D, frBlochEigs } from '../../lib/core/fr/fr1d.js';
import { correctionCoeffs, evalCorrection, cSD, cHU, vcjhEta } from '../../lib/core/fr/corrections.js';
import { makeDG1D, integrate } from '../../lib/core/dg/dg1d.js';
import { maxStableCFL } from '../../lib/core/dg/spectrum.js';
import { advection, burgers } from '../../lib/core/models/scalar.js';
import { makeStepper, stabilityAmp, RK_LABEL } from '../../lib/core/time/rk.js';
import { lagrangeValues } from '../../lib/core/basis/lagrange.js';
import { gaussLegendre } from '../../lib/core/quad/gauss1d.js';

const TWO_PI = 2 * Math.PI;
const wrap = (x) => x - Math.floor(x);

/** Play/pause + reset buttons wired to an Animator. */
function playControls(parent, anim, onReset) {
  const row = buttonRow(parent);
  const play = button(row, { label: '▶ Play', primary: true, onClick: () => anim.toggle() });
  button(row, { label: '↺ Reset', onClick: onReset });
  anim.o.onState = (r) => play.setLabel(r ? '❚❚ Pause' : '▶ Play');
  return play;
}

/** Run an array of small jobs asynchronously (keeps the page responsive); calls onDone when all finished. */
function runChunked(jobs, onProgress, onDone, token) {
  let i = 0;
  const tick = () => {
    if (token.cancelled) return;
    const t0 = performance.now();
    while (i < jobs.length && performance.now() - t0 < 25) { jobs[i](); i++; }
    onProgress(i / jobs.length);
    if (i < jobs.length) setTimeout(tick, 0); else onDone();
  };
  setTimeout(tick, 0);
}

const CORR_OPTS = [
  { value: 'dg', label: 'g_DG (Radau → DG)' },
  { value: 'g2', label: 'g_2 (Huynh)' },
  { value: 'ga', label: 'g_Ga (zeros at Gauss points)' },
];

/* ------------------------------------------------------------------ */
/* W1: reconstruction of a continuous flux                              */
/* ------------------------------------------------------------------ */
function reconWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('whole mesh: discontinuous flux $F^D$ (thin), interface fluxes $\\hat F$ (dots), corrected flux $F$ (thick)');
  p1.style.flex = '1 1 100%';
  const s1 = createCanvas(p1, { aspect: 0.3 });
  const p2 = L.panel('selected element: $F^D$ + left correction + right correction = $F$');
  p2.style.flex = '1 1 100%';
  const s2 = createCanvas(p2, { aspect: 0.32 });
  L.controls.classList.add('wide');
  const DATA = {
    sine: { label: 'sine wave', f: (x) => Math.sin(TWO_PI * x) },
    bump: { label: 'Gaussian bump', f: (x) => Math.exp(-30 * (x - 0.45) ** 2) },
    step: { label: 'step', f: (x) => (x > 0.3 && x < 0.65 ? 1 : 0) },
  };
  const st = { p: 2, N: 5, k: 2, data: 'step', eq: 'advection', corr: 'dg' };
  select(L.controls, { label: 'data $u_h$ (L² projection of)', value: st.data, options: Object.entries(DATA).map(([k, v]) => ({ value: k, label: v.label })), onChange: (v) => { st.data = v; draw(); } });
  select(L.controls, { label: 'flux $f(u)$', value: st.eq, options: [{ value: 'advection', label: 'advection f = u' }, { value: 'burgers', label: 'Burgers f = u²/2' }], onChange: (v) => { st.eq = v; draw(); } });
  select(L.controls, { label: 'correction function', value: st.corr, options: CORR_OPTS, onChange: (v) => { st.corr = v; draw(); } });
  slider(L.controls, { label: 'degree $p$', min: 1, max: 5, step: 1, value: st.p, onInput: (v) => { st.p = v; draw(); } });
  slider(L.controls, { label: 'element $k$', min: 1, max: 5, step: 1, value: st.k + 1, onInput: (v) => { st.k = v - 1; draw(); } });
  const out = readout(L.controls);
  function draw() {
    const T = theme(), C = seriesColors(), { p, N, k } = st;
    const m = st.eq === 'burgers' ? burgers : advection(1);
    const fr = makeFR1D({ p, N, model: m, flux: st.eq === 'burgers' ? 'rusanov' : 'upwind', points: 'gauss', correction: st.corr });
    const u = fr.project(DATA[st.data].f), du = new Float64Array(fr.n);
    fr.rhs(u, 0, du); // fills fr.faceFlux
    const np = p + 1, lv = new Float64Array(np);
    // per-element pieces as functions of ξ
    const FD = (e, xi) => { lagrangeValues(fr.r, xi, lv); let s = 0; for (let i = 0; i < np; i++) s += lv[i] * m.f(u[e * np + i]); return s; };
    const corr = (xi) => evalCorrection(fr.coef, xi);
    const jumps = (e) => [fr.faceFlux[e] - FD(e, -1), fr.faceFlux[e + 1] - FD(e, 1)];
    const Fc = (e, xi) => { const [jl, jr] = jumps(e), g = corr(xi); return FD(e, xi) + jl * g.gL + jr * g.gR; };
    const X = (e, xi) => fr.xf[e] + (xi + 1) * fr.h / 2;
    // top: whole mesh
    let lo = Infinity, hi = -Infinity;
    const segD = [], segF = [];
    for (let e = 0; e < N; e++) {
      const xs = [], yd = [], yf = [];
      for (let s = 0; s <= 40; s++) { const xi = -1 + 2 * s / 40; xs.push(X(e, xi)); yd.push(FD(e, xi)); yf.push(Fc(e, xi)); }
      segD.push([xs, yd]); segF.push([xs, yf]);
      for (const v of [...yd, ...yf]) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    }
    const pad = 0.12 * (hi - lo + 1e-9);
    const P = new Plot(s1, { xlim: [0, 1], ylim: [lo - pad, hi + pad], xlabel: 'x  (periodic)' });
    P.frame();
    const ctx = s1.ctx;
    ctx.save(); ctx.fillStyle = T.accent; ctx.globalAlpha = 0.08;
    ctx.fillRect(P.X(fr.xf[k]), P.py0, P.X(fr.xf[k + 1]) - P.X(fr.xf[k]), P.py1 - P.py0); ctx.restore();
    for (let f = 0; f <= N; f++) P.vline(fr.xf[f], { color: T.faint });
    const ec = elementColors();
    segD.forEach(([xs, ys], e) => P.line(xs, ys, { color: ec(e), width: 1.3, dash: [4, 3] }));
    segF.forEach(([xs, ys], e) => P.line(xs, ys, { color: ec(e), width: 2.8 }));
    P.points(Array.from(fr.xf), Array.from(fr.faceFlux), { color: T.ink, r: 4 });
    P.legend([{ label: 'F^D (discontinuous)', color: T.accent, dash: [4, 3] }, { label: 'F (corrected, continuous)', color: T.accent }, { label: 'F̂ at faces', color: T.ink, marker: true }], 'tr');
    // bottom: element k
    const [jl, jr] = jumps(k);
    const xs = [], yd = [], yl = [], yr = [], yf = [];
    for (let s = 0; s <= 80; s++) {
      const xi = -1 + 2 * s / 80, g = corr(xi);
      xs.push(X(k, xi)); yd.push(FD(k, xi)); yl.push(jl * g.gL); yr.push(jr * g.gR); yf.push(FD(k, xi) + jl * g.gL + jr * g.gR);
    }
    const all = [...yd, ...yl, ...yr, ...yf, fr.faceFlux[k], fr.faceFlux[k + 1]];
    const l2 = Math.min(...all), h2 = Math.max(...all), pd = 0.12 * (h2 - l2 + 1e-9);
    const Q = new Plot(s2, { xlim: [fr.xf[k], fr.xf[k + 1]], ylim: [l2 - pd, h2 + pd], xlabel: `x in element k = ${k + 1}` });
    Q.frame(); Q.hline(0);
    Q.line(xs, yd, { color: T.soft, width: 1.8, dash: [5, 4] });
    Q.line(xs, yl, { color: C[2], width: 1.8 });
    Q.line(xs, yr, { color: C[1], width: 1.8 });
    Q.line(xs, yf, { color: C[0], width: 3 });
    // mismatch bars at the two ends
    Q.line([fr.xf[k], fr.xf[k]], [FD(k, -1), fr.faceFlux[k]], { color: C[2], width: 5, alpha: 0.6 });
    Q.line([fr.xf[k + 1], fr.xf[k + 1]], [FD(k, 1), fr.faceFlux[k + 1]], { color: C[1], width: 5, alpha: 0.6 });
    Q.points([fr.xf[k], fr.xf[k + 1]], [fr.faceFlux[k], fr.faceFlux[k + 1]], { color: T.ink, r: 4 });
    const sx = [], sy = [];
    for (let i = 0; i < np; i++) { sx.push(fr.x[k * np + i]); sy.push(Fc(k, fr.r[i])); }
    Q.points(sx, sy, { color: C[0], r: 3.5, hollow: true });
    Q.legend([{ label: 'F^D', color: T.soft, dash: [5, 4] }, { label: '(F̂_L − F^D(−1)) g_L', color: C[2] }, { label: '(F̂_R − F^D(1)) g_R', color: C[1] }, { label: 'F = sum', color: C[0] }], 'tr');
    // continuity check over all faces (periodic)
    let cont = 0;
    for (let e = 0; e < N; e++) cont = Math.max(cont, Math.abs(Fc(e, 1) - Fc((e + 1) % N, -1)));
    const dts = Array.from({ length: np }, (_, i) => fmt(du[k * np + i], 4)).join(', ');
    out.set(`mismatch left  F̂ − F^D(−1) = ${fmt(jl, 4)}\nmismatch right F̂ − F^D(1)  = ${fmt(jr, 4)}\n` +
      `max |F_k(1) − F_{k+1}(−1)| = ${fmt(cont, 2)}  (continuous)\n−(2/h) F'(ξ_i) = [${dts}]`);
    selfCheck('FR corrected flux is continuous', cont < 1e-12);
  }
  s1.onResize(draw); s2.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W2: correction function plotter                                      */
/* ------------------------------------------------------------------ */
function corrWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('left correction functions $g_L(\\xi)$', '1 1 320px');
  const s1 = createCanvas(p1, { aspect: 0.8 });
  const p2 = L.panel('derivatives $g_L\'(\\xi)$ (dots at the Gauss points)', '1 1 320px');
  const s2 = createCanvas(p2, { aspect: 0.8 });
  const st = { p: 3, s: 0.5, mirror: false };
  slider(L.controls, { label: 'degree $p$', min: 1, max: 6, step: 1, value: st.p, onInput: (v) => { st.p = v; draw(); } });
  slider(L.controls, { label: 'VCJH: $\\log_{10}(c/c_{HU})$', min: -3, max: 3, step: 0.05, value: st.s, onInput: (v) => { st.s = v; draw(); } });
  checkbox(L.controls, { label: 'show $g_R(\\xi) = g_L(-\\xi)$', value: st.mirror, onChange: (v) => { st.mirror = v; draw(); } });
  const out = readout(L.controls);
  function draw() {
    const T = theme(), C = seriesColors(), p = st.p, c = cHU(p) * 10 ** st.s;
    const fams = [['dg', 0, 'g_DG', C[0]], ['g2', 0, 'g_2', C[1]], ['ga', 0, 'g_Ga', C[2]], ['vcjh', c, `VCJH c = ${fmt(c, 3)}`, C[3]]];
    const xs = []; for (let i = 0; i <= 300; i++) xs.push(-1 + 2 * i / 300);
    const curves = fams.map(([t, cc, lab, col]) => {
      const co = correctionCoeffs(t, p, cc);
      return { lab, col, co, g: xs.map((x) => evalCorrection(co, x).gL), d: xs.map((x) => evalCorrection(co, x).dgL), r: xs.map((x) => evalCorrection(co, x).gR) };
    });
    let lo = -0.3, hi = 1.1, dlo = 0, dhi = 0;
    for (const cv of curves) { for (const v of cv.g) { lo = Math.min(lo, v); hi = Math.max(hi, v); } for (const v of cv.d) { dlo = Math.min(dlo, v); dhi = Math.max(dhi, v); } }
    const P = new Plot(s1, { xlim: [-1, 1], ylim: [lo - 0.05, hi + 0.05], xlabel: 'ξ' });
    P.frame(); P.hline(0); P.hline(1, { color: T.rule });
    for (const cv of curves) {
      P.line(xs, cv.g, { color: cv.col, width: cv.lab.startsWith('VCJH') ? 2.8 : 1.8 });
      if (st.mirror) P.line(xs, cv.r, { color: cv.col, width: 1.2, dash: [4, 3] });
    }
    P.points([-1, 1], [1, 0], { color: T.ink, r: 4 });
    P.legend(curves.map((cv) => ({ label: cv.lab, color: cv.col })), 'tr');
    const Q = new Plot(s2, { xlim: [-1, 1], ylim: [dlo * 1.08 - 0.1, dhi * 1.08 + 0.1], xlabel: 'ξ' });
    Q.frame(); Q.hline(0);
    const G = gaussLegendre(p + 1);
    for (const cv of curves) {
      Q.line(xs, cv.d, { color: cv.col, width: cv.lab.startsWith('VCJH') ? 2.8 : 1.8 });
      Q.points(Array.from(G.x), Array.from(G.x, (x) => evalCorrection(cv.co, x).dgL), { color: cv.col, r: 3 });
    }
    const vc = curves[3].co;
    out.set(`c_SD = ${fmt(cSD(p), 4)}\nc_HU = ${fmt(cHU(p), 4)}\nc    = ${fmt(c, 4)}\nη_p(c) = ${fmt(vcjhEta(p, c), 4)}\n` +
      `g_L(−1) = ${fmt(evalCorrection(vc, -1).gL, 6)}, g_L(1) = ${fmt(Math.abs(evalCorrection(vc, 1).gL) < 1e-14 ? 0 : evalCorrection(vc, 1).gL, 3)}`);
    for (const cv of curves) selfCheck('correction endpoint values', Math.abs(evalCorrection(cv.co, -1).gL - 1) < 1e-12 && Math.abs(evalCorrection(cv.co, 1).gL) < 1e-12);
  }
  s1.onResize(draw); s2.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W3: FR vs DG side by side                                           */
/* ------------------------------------------------------------------ */
function frdgWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('solutions: FR (thick) and DG (thin, on top)');
  p1.style.flex = '1 1 100%';
  const s1 = createCanvas(p1, { aspect: 0.3 });
  const p2 = L.panel('pointwise difference $|u_{FR} - u_{DG}|$ (log scale)');
  p2.style.flex = '1 1 100%';
  const s2 = createCanvas(p2, { aspect: 0.22 });
  L.controls.classList.add('wide');
  const INITS = {
    sine: { label: 'sine wave', f: (x) => 0.3 + Math.sin(TWO_PI * x) },
    square: { label: 'square wave', f: (x) => (wrap(x) > 0.25 && wrap(x) < 0.6 ? 1 : 0) },
  };
  const DGV = {
    strong: { label: 'DG strong form, exact mass', o: { form: 'strong', quad: 'exact' } },
    weak: { label: 'DG weak form, over-integrated', o: { form: 'weak', quad: 'exact' } },
    sem: { label: 'DG-SEM (GLL, lumped mass)', o: { form: 'strong', quad: 'collocated', nodes: 'gll' } },
  };
  const st = { p: 3, N: 10, corr: 'dg', pts: 'gauss', dgv: 'strong', eq: 'advection', init: 'sine', t: 0 };
  let fr, dg, uF, uD, stepF, stepD, maxDiff = 0;
  select(L.controls, { label: 'FR correction', value: st.corr, options: CORR_OPTS, onChange: (v) => { st.corr = v; build(); } });
  select(L.controls, { label: 'solution points', value: st.pts, options: [{ value: 'gauss', label: 'Gauss' }, { value: 'gll', label: 'Gauss–Lobatto (GLL)' }], onChange: (v) => { st.pts = v; build(); } });
  select(L.controls, { label: 'DG variant', value: st.dgv, options: Object.entries(DGV).map(([k, v]) => ({ value: k, label: v.label })), onChange: (v) => { st.dgv = v; build(); } });
  select(L.controls, { label: 'equation', value: st.eq, options: [{ value: 'advection', label: 'advection (a = 1)' }, { value: 'burgers', label: 'Burgers (Rusanov flux)' }], onChange: (v) => { st.eq = v; build(); } });
  select(L.controls, { label: 'initial data', value: st.init, options: Object.entries(INITS).map(([k, v]) => ({ value: k, label: v.label })), onChange: (v) => { st.init = v; build(); } });
  slider(L.controls, { label: 'degree $p$', min: 1, max: 5, step: 1, value: st.p, onInput: (v) => { st.p = v; build(); } });
  slider(L.controls, { label: 'elements $N$', min: 4, max: 32, step: 1, value: st.N, onInput: (v) => { st.N = v; build(); } });
  const anim = new Animator(fig, { step: () => advance(), draw: () => draw(), stepsPerFrame: 3 });
  playControls(L.controls, anim, () => build());
  const out = readout(L.controls);
  function build() {
    const m = st.eq === 'burgers' ? burgers : advection(1), flux = st.eq === 'burgers' ? 'rusanov' : 'upwind';
    fr = makeFR1D({ p: st.p, N: st.N, model: m, flux, points: st.pts, correction: st.corr });
    dg = makeDG1D({ p: st.p, N: st.N, model: m, flux, nodes: st.pts, ...DGV[st.dgv].o });
    uF = fr.project(INITS[st.init].f);
    uD = dg.project(INITS[st.init].f);
    stepF = makeStepper('ssprk3', fr.n); stepD = makeStepper('ssprk3', dg.n);
    st.t = 0; maxDiff = 0;
    draw();
  }
  function advance() {
    if (st.t >= 2 - 1e-12) return false;
    const s = Math.max(fr.maxSpeed(uF), 1e-3);
    const dt = Math.min(0.5 * fr.h / (s * (2 * st.p + 1)), 2 - st.t); // below the DG (c = 0) CFL limit for every p
    stepF(uF, st.t, dt, fr.rhs); stepD(uD, st.t, dt, dg.rhs);
    st.t += dt;
    return true;
  }
  function draw() {
    const T = theme(), C = seriesColors();
    const P = new Plot(s1, { xlim: [0, 1], ylim: [-1, 1.6], xlabel: 'x' });
    P.frame();
    drawPiecewise(P, fr, uF, { color: C[0], width: 4.5, alpha: 0.55 });
    drawPiecewise(P, dg, uD, { color: C[1], width: 1.5 });
    P.legend([{ label: 'FR', color: C[0] }, { label: 'DG', color: C[1] }], 'tr');
    const xs = [], ds = [];
    let mx = 0;
    for (let i = 0; i <= 600; i++) {
      const x = (i + 0.5) / 601;
      const d = Math.abs(fr.evalAt(uF, x) - dg.evalAt(uD, x));
      xs.push(x); ds.push(Math.max(d, 1e-17)); mx = Math.max(mx, d);
    }
    maxDiff = Math.max(maxDiff, mx);
    const Q = new Plot(s2, { xlim: [0, 1], ylim: [1e-17, 10], ylog: true, xlabel: 'x' });
    Q.frame();
    Q.line(xs, ds, { color: C[3], width: 1.6 });
    Q.hline(1e-14, { color: T.faint });
    const match = (st.corr === 'dg' && st.dgv === 'strong') || (st.corr === 'g2' && st.pts === 'gll' && st.dgv === 'sem') ||
      (st.corr === 'dg' && st.dgv === 'weak' && st.eq === 'advection');
    out.set(`t = ${st.t.toFixed(3)}\nmax |u_FR − u_DG| now   = ${fmt(mx, 2)}\nmax over the run so far = ${fmt(maxDiff, 2)}\n` +
      (match ? 'expected: identical schemes (round-off)' : 'expected: different schemes'));
    if (match) selfCheck('FR = DG for matching pair', mx < 1e-11);
  }
  s1.onResize(draw); s2.onResize(draw);
  build();
}

/* ------------------------------------------------------------------ */
/* W4: max stable CFL vs c                                             */
/* ------------------------------------------------------------------ */
function cflcWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('largest stable CFL $\\nu_{\\max}$ vs $c$ (log axis; leftmost point: $c = 0$, DG)', '1 1 360px');
  const s1 = createCanvas(p1, { aspect: 0.75 });
  const p2 = L.panel('$\\nu_{\\max}\\lambda$ for the selected $c$ and the stability region', '1 1 300px');
  const s2 = createCanvas(p2, { aspect: 1 });
  const st = { p: 3, rk: 'rk4', s: 0, curve: null, busy: 0 };
  let token = { cancelled: false };
  slider(L.controls, { label: 'degree $p$', min: 1, max: 5, step: 1, value: st.p, onChange: (v) => { st.p = v; compute(); } });
  select(L.controls, { label: 'time integrator', value: st.rk, options: ['ssprk3', 'rk4', 'lsrk4'].map((k) => ({ value: k, label: RK_LABEL[k] })), onChange: (v) => { st.rk = v; compute(); } });
  slider(L.controls, { label: 'selected $\\log_{10}(c/c_{HU})$', min: -3, max: 3, step: 0.05, value: st.s, onInput: (v) => { st.s = v; drawSpec(); drawCurve(); } });
  const out = readout(L.controls);
  const S = [];
  for (let s = -3; s <= 3 + 1e-9; s += 0.2) S.push(s);
  function compute() {
    token.cancelled = true; token = { cancelled: false };
    const p = st.p, rk = st.rk, ch = cHU(p);
    const res = { p, rk, c0: 0, nu: new Float64Array(S.length) };
    st.curve = null; st.busy = 0;
    const jobs = [() => { res.c0 = maxStableCFL(frBlochEigs(p, { c: 0 }, 48), rk); }]
      .concat(S.map((s, i) => () => { res.nu[i] = maxStableCFL(frBlochEigs(p, { c: ch * 10 ** s }, 48), rk); }));
    runChunked(jobs, (f) => { st.busy = f; drawCurve(); }, () => { st.curve = res; drawCurve(); drawSpec(); }, token);
    drawCurve(); drawSpec();
  }
  function cPlus() {
    const r = st.curve; if (!r) return null;
    let k = 0; for (let i = 1; i < S.length; i++) if (r.nu[i] > r.nu[k]) k = i;
    return { s: S[k], nu: r.nu[k] };
  }
  function drawCurve() {
    const T = theme(), C = seriesColors(), p = st.p, ch = cHU(p);
    const r = st.curve;
    const ymax = r ? Math.max(...r.nu, r.c0) * 1.15 : 1;
    const P = new Plot(s1, { xlim: [ch * 10 ** -3.6, ch * 1e3], ylim: [0, ymax], xlog: true, xlabel: 'c' });
    P.frame();
    if (!r) { P.text(ch * 1e-2, ymax / 2, `computing … ${Math.round(100 * st.busy)} %`, { color: T.soft }); out.set('computing spectra …'); return; }
    const cs = S.map((s) => ch * 10 ** s);
    P.line(cs, Array.from(r.nu), { color: C[0], width: 2.4 });
    P.points(cs, Array.from(r.nu), { color: C[0], r: 2.2 });
    P.points([ch * 10 ** -3.4], [r.c0], { color: C[0], r: 4, hollow: true });
    const cp = cPlus();
    [[cSD(p), 'c_SD', C[2]], [ch, 'c_HU', C[1]], [ch * 10 ** cp.s, 'c_+', C[3]]].forEach(([cv, lab, col], i) => {
      P.vline(cv, { color: col, dash: [5, 4] });
      P.text(cv, ymax * (0.06 + 0.08 * i), lab, { color: col, align: 'left', dx: 3 });
    });
    P.vline(ch * 10 ** st.s, { color: T.ink, dash: [] , width: 1.2 });
    const nuSel = interp(st.s);
    out.set(`p = ${p}, ${RK_LABEL[st.rk]}\nν_max(c = 0, DG) = ${r.c0.toFixed(4)}\nν_max(c_SD)      ≈ ${interpC(cSD(p)).toFixed(4)}\n` +
      `ν_max(c_HU)      ≈ ${interpC(ch).toFixed(4)}\nc_+ ≈ ${fmt(ch * 10 ** cp.s, 3)} (grid), ν_max = ${cp.nu.toFixed(4)}\nselected c = ${fmt(ch * 10 ** st.s, 3)}: ν_max ≈ ${nuSel.toFixed(4)}`);
    selfCheck('CFL(c+) > CFL(DG)', cp.nu > r.c0);
  }
  function interp(s) {
    const r = st.curve; if (!r) return 0;
    const i = Math.max(0, Math.min(S.length - 2, Math.floor((s + 3) / 0.2)));
    const f = (s - S[i]) / (S[i + 1] - S[i]);
    return r.nu[i] * (1 - f) + r.nu[i + 1] * f;
  }
  const interpC = (c) => interp(Math.log10(c / cHU(st.p)));
  function drawSpec() {
    const T = theme();
    const c = cHU(st.p) * 10 ** st.s;
    const ev = frBlochEigs(st.p, { c }, 64);
    const nu = maxStableCFL(ev, st.rk);
    const P = new Plot(s2, { xlim: [-3.2, 0.6], ylim: [-3.1, 3.1], xlabel: 'Re z', ylabel: 'Im z', equal: true });
    P.frame();
    shadeStabilityRegion(P, st.rk);
    P.hline(0); P.vline(0);
    const xs = [], ys = [];
    for (let k = 0; k < ev.re.length; k++) { xs.push(nu * ev.re[k]); ys.push(nu * ev.im[k]); }
    P.points(xs, ys, { color: T.accent, r: 1.8 });
    P.text(-3.1, 2.8, `ν_max = ${nu.toFixed(3)}`, { color: T.ink });
    let worst = 0; for (let k = 0; k < xs.length; k++) worst = Math.max(worst, stabilityAmp(st.rk, xs[k], ys[k]));
    selfCheck('FR spectrum at nu_max inside region', worst <= 1 + 1e-6);
  }
  s1.onResize(drawCurve); s2.onResize(drawSpec);
  compute();
}

/* ------------------------------------------------------------------ */
/* W5: accuracy vs c                                                   */
/* ------------------------------------------------------------------ */
function acccWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('$L^2$ error at $t = 1$ vs $c$ (log–log); leftmost points: $c = 0$ (DG)', '1 1 420px');
  const s1 = createCanvas(p1, { aspect: 0.62 });
  const st = { p: 3, res: null, busy: 0 };
  let token = { cancelled: false };
  slider(L.controls, { label: 'degree $p$', min: 1, max: 4, step: 1, value: st.p, onChange: (v) => { st.p = v; compute(); } });
  const out = readout(L.controls);
  const S = [];
  for (let s = -2; s <= 3 + 1e-9; s += 0.25) S.push(s);
  const u0 = (x) => Math.sin(TWO_PI * x);
  function err(p, N, c) {
    const fr = makeFR1D({ p, N, model: advection(1), flux: 'upwind', correction: 'vcjh', c });
    const u = fr.project(u0);
    integrate(makeStepper('lsrk4', fr.n), u, fr.rhs, 1, 0.05 * fr.h / (2 * p + 1));
    return fr.l2Error(u, u0);
  }
  function compute() {
    token.cancelled = true; token = { cancelled: false };
    const p = st.p, ch = cHU(p);
    const res = { p, e8: new Float64Array(S.length), e16: new Float64Array(S.length), d8: 0, d16: 0 };
    st.res = null;
    const jobs = [() => { res.d8 = err(p, 8, 0); }, () => { res.d16 = err(p, 16, 0); }];
    S.forEach((s, i) => { jobs.push(() => { res.e8[i] = err(p, 8, ch * 10 ** s); }); jobs.push(() => { res.e16[i] = err(p, 16, ch * 10 ** s); }); });
    runChunked(jobs, (f) => { st.busy = f; draw(); }, () => { st.res = res; draw(); }, token);
    draw();
  }
  function draw() {
    const T = theme(), C = seriesColors(), p = st.p, ch = cHU(p), r = st.res;
    const x0 = ch * 10 ** -2.4, xl = ch * 10 ** -2.7;
    if (!r) {
      const P = new Plot(s1, { xlim: [xl, ch * 1e3], ylim: [1e-10, 1], xlog: true, ylog: true, xlabel: 'c' });
      P.frame(); P.text(ch * 1e-1, 1e-5, `computing … ${Math.round(100 * st.busy)} %`, { color: T.soft });
      out.set(`running ${2 * S.length + 2} simulations …`);
      return;
    }
    const all = [...r.e8, ...r.e16, r.d8, r.d16].filter((v) => v > 0);
    const P = new Plot(s1, { xlim: [xl, ch * 1e3], ylim: [Math.min(...all) / 3, Math.max(...all) * 3], xlog: true, ylog: true, xlabel: 'c' });
    P.frame();
    const cs = S.map((s) => ch * 10 ** s);
    P.line(cs, Array.from(r.e8), { color: C[1], width: 2 }); P.points(cs, Array.from(r.e8), { color: C[1], r: 2.4 });
    P.line(cs, Array.from(r.e16), { color: C[0], width: 2 }); P.points(cs, Array.from(r.e16), { color: C[0], r: 2.4 });
    P.points([x0, x0], [r.d8, r.d16], { color: T.ink, r: 4, hollow: true });
    [[cSD(p), 'c_SD', C[2]], [ch, 'c_HU', C[3]]].forEach(([cv, lab, col], i) => { P.vline(cv, { color: col, dash: [5, 4] }); P.text(cv, Math.max(...all) * (2 - 1.2 * i), lab, { color: col, align: 'left', dx: 3 }); });
    P.legend([{ label: 'N = 8', color: C[1] }, { label: 'N = 16', color: C[0] }, { label: 'DG (c = 0)', color: T.ink, marker: true }], 'br');
    const rate = (a, b) => Math.log2(a / b);
    const idx = (c) => Math.round((Math.log10(c / ch) + 2) / 0.25);
    const iHU = idx(ch), iBig = S.length - 1;
    out.set(`p = ${p}\nDG:    e₁₆ = ${fmt(r.d16, 3)}, order ${rate(r.d8, r.d16).toFixed(2)}\n` +
      `c_HU:  e₁₆ = ${fmt(r.e16[iHU], 3)}, order ${rate(r.e8[iHU], r.e16[iHU]).toFixed(2)}\n` +
      `c = 10³c_HU: e₁₆ = ${fmt(r.e16[iBig], 3)}, order ${rate(r.e8[iBig], r.e16[iBig]).toFixed(2)}`);
    selfCheck('accuracy: errors finite', all.every(Number.isFinite));
  }
  s1.onResize(draw);
  compute();
}

initChapter(() => {
  mount('w-recon', reconWidget);
  mount('w-corr', corrWidget);
  mount('w-frdg', frdgWidget);
  mount('w-cflc', cflcWidget);
  mount('w-accc', acccWidget);
});
