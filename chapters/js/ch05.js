/**
 * @file Chapter 5 — Finite volume methods.
 *
 * Widgets:
 *  - w-rea        : Godunov's reconstruct–evolve–average step for linear advection
 *  - w-playground : 1D scalar FV playground (advection / Burgers / Buckley–Leverett, fluxes, MUSCL, CFL)
 *  - w-modeq      : upwind vs the exact solution of its modified (advection–diffusion) equation
 *  - w-sweby      : Sweby's TVD diagram + the selected limiter on a test profile
 *  - w-rates      : measured L1 convergence rates (smooth vs discontinuous data)
 *  - w-sod        : Euler shock tubes: FV vs exact Riemann solution, x–t wave diagram
 *  - w-zalesak    : 2D solid-body rotation of Zalesak's slotted disk, upwind vs MUSCL
 */
import { initChapter, mount, selfCheck } from '../../lib/ui/chapter.js';
import { widgetLayout, slider, select, segmented, button, buttonRow, readout, fmt, debounce } from '../../lib/ui/controls.js';
import { createCanvas, View2D, theme, seriesColors } from '../../lib/viz/canvas.js';
import { drawFunction, drawQuadField } from '../../lib/viz/field2d.js';
import { colorbar } from '../../lib/viz/colormap.js';
import { Plot } from '../../lib/viz/plot1d.js';
import { Animator } from '../../lib/viz/anim.js';
import { advection, burgers, buckley, burgersRiemann, scalarRiemann, SCALAR_FLUXES } from '../../lib/core/models/scalar.js';
import {
  makeFV1D, cellAverages, LIMITERS, LIMITER_PHI, advanceFV1D, totalVariation,
  upwindNumericalDiffusion, diffusedSquare,
} from '../../lib/core/fv/fv1d.js';
import { makeStepper } from '../../lib/core/time/rk.js';
import { fitRate } from '../../lib/core/verify/rates.js';
import { EULER_FLUXES, primToCons, physFlux } from '../../lib/core/models/euler.js';
import { exactRiemann } from '../../lib/core/models/eulerRiemann.js';
import { makeEuler1D, eulerCellAverages, eulerStats, SHOCK_TUBES } from '../../lib/core/fv/euler1d.js';
import { makeFV2D, cellAverages2D, zalesakDisk, rotationVelocity, rotated } from '../../lib/core/fv/fv2d.js';

/** Play/pause + reset buttons wired to an Animator (returns the button row for extra buttons). */
function playControls(parent, anim, onReset) {
  const row = buttonRow(parent);
  const play = button(row, { label: '▶ Play', primary: true, onClick: () => anim.toggle() });
  button(row, { label: '↺ Reset', onClick: () => { anim.pause(); onReset(); } });
  anim.o.onState = (r) => play.setLabel(r ? '❚❚ Pause' : '▶ Play');
  return row;
}

const LIMITER_OPTIONS = ['minmod', 'vanleer', 'mc', 'superbee', 'none'].map((k) => ({ value: k, label: LIMITERS[k].label }));

/* ------------------------------------------------------------------ */
/* W1: Godunov's reconstruct–evolve–average step                        */
/* ------------------------------------------------------------------ */
function reaWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('cells $C_i$ (width $h = 1$), periodic; wind blows to the right ($a > 0$)');
  const s1 = createCanvas(p1, { aspect: 0.5 });
  const M = 10;
  const init = [0.25, 0.3, 0.45, 0.95, 1.0, 0.7, 0.35, 0.3, 0.6, 0.35];
  const st = { u: init.slice(), nu: 0.4, recon: 'const', stage: 'rec', tau: 1, steps: 0, sel: 4 };
  // limited slope σ_i (difference per cell) of the current averages
  const slopes = () => st.u.map((v, i) => {
    if (st.recon === 'const') return 0;
    const a = v - st.u[(i - 1 + M) % M], b = st.u[(i + 1) % M] - v;
    return st.recon === 'minmod' ? LIMITERS.minmod.fn(a, b) : 0.5 * (a + b);
  });
  /** reconstruction at x (periodic in [0, M)) */
  const recon = (x, sig) => { const y = ((x % M) + M) % M, i = Math.min(M - 1, Math.floor(y)); return st.u[i] + sig[i] * (y - i - 0.5); };
  /** new averages: exact cell integrals of the shifted reconstruction (shift s = ν), split at the kink x = i + s */
  const average = (sig) => {
    const s = st.nu, g = [-1 / Math.sqrt(3), 1 / Math.sqrt(3)], out = [];
    for (let i = 0; i < M; i++) {
      let v = 0;
      for (const [a, b] of [[i, i + s], [i + s, i + 1]]) for (const q of g) {
        const x = 0.5 * (a + b) + 0.5 * (b - a) * q;
        v += 0.5 * (b - a) * recon(x - s, sig);
      }
      out.push(v);
    }
    return out;
  };
  /** closed-form update ū_i − ν(ū_i − ū_{i−1}) − ν(1−ν)/2 (σ_i − σ_{i−1}) */
  const formula = (sig) => st.u.map((v, i) => {
    const im = (i - 1 + M) % M;
    return v - st.nu * (v - st.u[im]) - 0.5 * st.nu * (1 - st.nu) * (sig[i] - sig[im]);
  });
  segmented(L.controls, {
    label: 'reconstruction', value: st.recon,
    options: [{ value: 'const', label: 'constant' }, { value: 'minmod', label: 'linear, minmod' }, { value: 'central', label: 'linear, central' }],
    onChange: (v) => { st.recon = v; draw(); },
  });
  slider(L.controls, { label: 'CFL number $\\nu = a\\Delta t/h$', min: 0, max: 1, step: 0.05, value: st.nu, onInput: (v) => { st.nu = v; draw(); } });
  const stageCtl = segmented(L.controls, {
    label: 'stage', value: st.stage,
    options: [{ value: 'rec', label: '1 reconstruct' }, { value: 'evo', label: '2 evolve' }, { value: 'avg', label: '3 average' }],
    onChange: (v) => { st.stage = v; st.tau = 1; draw(); },
  });
  const anim = new Animator(fig, {
    step: () => {
      if (st.stage === 'rec') { st.stage = 'evo'; st.tau = 0; }
      else if (st.stage === 'evo' && st.tau < 1) st.tau = Math.min(1, st.tau + 0.025);
      else if (st.stage === 'evo') { st.stage = 'avg'; }
      else { commit(); st.stage = 'rec'; }
      stageCtl.set(st.stage);
      if (st.stage === 'avg' || st.stage === 'rec') return false; // pause at the interesting moments
    },
    draw: () => draw(),
  });
  const row = playControls(L.controls, anim, () => { st.u = init.slice(); st.steps = 0; st.stage = 'rec'; st.tau = 1; stageCtl.set('rec'); draw(); });
  button(row, { label: 'Next step ⇥', onClick: () => { commit(); st.stage = 'rec'; stageCtl.set('rec'); draw(); } });
  const out = readout(L.controls);
  function commit() { st.u = average(slopes()); st.steps++; st.tau = 1; }
  function draw() {
    const T = theme(), C = seriesColors(), sig = slopes();
    const P = new Plot(s1, { xlim: [0, M], ylim: [0, 1.25], xlabel: 'x' });
    P.frame();
    const edges = Array.from({ length: M + 1 }, (_, i) => i);
    for (let i = 1; i < M; i++) P.vline(i, { color: T.rule, dash: [] });
    const shift = st.stage === 'rec' ? 0 : st.nu * (st.stage === 'evo' ? st.tau : 1);
    // the (possibly shifted) reconstruction as segments, broken at the shifted cell edges
    const segs = [];
    for (let j = -1; j <= M; j++) {
      const a = j + shift, b = j + 1 + shift, xs = [Math.max(0, a), Math.min(M, b)];
      if (xs[1] <= xs[0]) continue;
      const jj = ((j % M) + M) % M; // cell whose (shifted) piece this is
      segs.push([xs, xs.map((x) => st.u[jj] + sig[jj] * (x - shift - j - 0.5))]);
    }
    if (st.stage !== 'rec') {
      // hatched slabs: what crossed each face so far
      const { ctx } = s1; P.clip();
      ctx.fillStyle = C[1]; ctx.globalAlpha = 0.35;
      for (let f = 0; f <= M; f++) {
        if (shift <= 0) break;
        const xs = []; for (let k = 0; k <= 12; k++) xs.push(f + shift * k / 12);
        ctx.beginPath(); ctx.moveTo(P.X(f), P.Y(0));
        const jj = (f - 1 + M) % M; // the slab crossing face f comes from cell f − 1
        for (const x of xs) ctx.lineTo(P.X(x), P.Y(st.u[jj] + sig[jj] * (x - shift - f + 0.5)));
        ctx.lineTo(P.X(f + shift), P.Y(0)); ctx.closePath(); ctx.fill();
      }
      ctx.restore();
    }
    if (st.stage === 'avg') {
      P.bars(edges, st.u, { color: T.faint, alpha: 0.12, width: 1 });
      P.segments(segs, { color: C[0], width: 1.4, alpha: 0.6 });
      P.bars(edges, average(sig), { color: C[2], alpha: 0.35, width: 2.6 });
      P.legend([{ label: 'old averages ūⁿ', color: T.faint }, { label: 'shifted reconstruction', color: C[0] }, { label: 'new averages ūⁿ⁺¹', color: C[2] }], 'tl');
    } else {
      P.bars(edges, st.u, { color: T.faint, alpha: 0.18, width: 1.2 });
      P.segments(segs, { color: C[0], width: 2.6 });
      const items = [{ label: 'cell averages ūⁿ', color: T.faint }, { label: st.stage === 'rec' ? 'reconstruction' : `evolved: shifted by aτ = ${fmt(shift)} h`, color: C[0] }];
      if (st.stage === 'evo') items.push({ label: 'slabs crossing the faces', color: C[1] });
      P.legend(items, 'tl');
    }
    const num = average(sig), cf = formula(sig), i = st.sel, im = (i - 1 + M) % M;
    let maxd = 0; for (let k = 0; k < M; k++) maxd = Math.max(maxd, Math.abs(num[k] - cf[k]));
    selfCheck('REA average equals closed-form update', maxd < 1e-12);
    const inflow = st.nu * (st.u[im] + 0.5 * (1 - st.nu) * sig[im]), outflow = st.nu * (st.u[i] + 0.5 * (1 - st.nu) * sig[i]);
    out.set(`step ${st.steps}, cell i = ${i} (x ∈ [${i}, ${i + 1}])\n` +
      `ūᵢ = ${st.u[i].toFixed(4)}\n+ in  (left face)  ${inflow.toFixed(4)}\n− out (right face) ${outflow.toFixed(4)}\n` +
      `= ${num[i].toFixed(4)}  (integrated)\n  ${cf[i].toFixed(4)}  (formula)\nmax diff over cells: ${maxd.toExponential(1)}\n` +
      `mass Σūᵢ = ${st.u.reduce((a, b) => a + b, 0).toFixed(12)}`);
  }
  s1.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W2: 1D scalar FV playground                                          */
/* ------------------------------------------------------------------ */
const MODELS = {
  adv: { label: 'linear advection, a = 1', model: advection(1) },
  burgers: { label: 'Burgers, f = u²/2', model: burgers },
  bl: { label: 'Buckley–Leverett (M = ½)', model: buckley(0.5) },
};
const INITS = {
  square: { label: 'square pulse (periodic)', periodic: true, u0: (x) => (x > 0.25 && x < 0.6 ? 1 : 0) },
  sine: { label: 'sine wave (periodic)', periodic: true, u0: (x) => 0.5 + 0.5 * Math.sin(2 * Math.PI * x) },
  rA: { label: 'Riemann A: u_L = 1, u_R = 0', periodic: false, states: () => [1, 0] },
  rB: { label: 'Riemann B: u_L < u_R', periodic: false, states: (m) => (m === 'burgers' ? [-1, 1] : [0, 1]) },
};

function playgroundWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('cell averages $\\bar u_i$ (bars) and exact / reference solution (line)');
  const s1 = createCanvas(p1, { aspect: 0.66 });
  p1.style.flex = '1 1 420px';
  const st = { model: 'adv', init: 'square', flux: 'upwind', recon: 'none', limiter: 'minmod', rk: 'fe', cfl: 0.8, N: 100 };
  let sim = null;
  const fluxOpts = Object.keys(SCALAR_FLUXES).map((k) => ({ value: k, label: SCALAR_FLUXES[k].label }));
  select(L.controls, { label: 'PDE', value: st.model, options: Object.keys(MODELS).map((k) => ({ value: k, label: MODELS[k].label })), onChange: (v) => { st.model = v; reset(); } });
  select(L.controls, { label: 'initial data', value: st.init, options: Object.keys(INITS).map((k) => ({ value: k, label: INITS[k].label })), onChange: (v) => { st.init = v; reset(); } });
  select(L.controls, { label: 'numerical flux', value: st.flux, options: fluxOpts, onChange: (v) => { st.flux = v; reset(); } });
  const rkCtl = { set: () => {} };
  segmented(L.controls, {
    label: 'reconstruction', value: st.recon, options: [{ value: 'none', label: '1st order' }, { value: 'muscl', label: 'MUSCL' }],
    onChange: (v) => { st.recon = v; if (v === 'muscl' && st.rk === 'fe') { st.rk = 'ssprk2'; rkCtl.set('ssprk2'); } reset(); },
  });
  select(L.controls, { label: 'limiter (MUSCL)', value: st.limiter, options: LIMITER_OPTIONS, onChange: (v) => { st.limiter = v; reset(); } });
  Object.assign(rkCtl, segmented(L.controls, {
    label: 'time stepping', value: st.rk, options: [{ value: 'fe', label: 'Euler' }, { value: 'ssprk2', label: 'SSP-RK2' }, { value: 'ssprk3', label: 'SSP-RK3' }],
    onChange: (v) => { st.rk = v; reset(); },
  }));
  slider(L.controls, { label: 'CFL number $\\nu$', min: 0.1, max: 1.5, step: 0.05, value: st.cfl, onInput: (v) => { st.cfl = v; reset(); } });
  slider(L.controls, { label: 'cells $N$', min: 20, max: 400, step: 10, value: st.N, onInput: (v) => { st.N = v; reset(); } });
  const anim = new Animator(fig, { step: () => advance(), draw: () => draw() });
  const row = playControls(L.controls, anim, () => reset());
  button(row, { label: 'Step', onClick: () => { advance(); draw(); } });
  const out = readout(L.controls);

  function reset() {
    anim.pause();
    const m = MODELS[st.model].model, I = INITS[st.init];
    let u0, exact = null, a = 0, b = 1, uL = 0, uR = 0;
    if (I.periodic) {
      u0 = I.u0;
      if (st.model === 'adv') exact = (x, t) => I.u0((((x - t) % 1) + 1) % 1);
    } else {
      [uL, uR] = I.states(st.model); a = -1; b = 1;
      u0 = (x) => (x < 0 ? uL : uR);
      const R = st.model === 'burgers' ? (xi) => burgersRiemann(uL, uR, xi) : scalarRiemann(m, uL, uR, 4000);
      exact = (x, t) => (t <= 0 ? u0(x) : R(x / t));
    }
    const fv = makeFV1D({ model: m, N: st.N, a, b, flux: st.flux, recon: st.recon, limiter: st.limiter, bc: I.periodic ? 'periodic' : 'outflow' });
    const u = cellAverages(u0, fv.xf);
    // reference for problems without a closed-form solution: 2000 cells, MUSCL–MC, SSP-RK3
    let ref = null;
    if (!exact) {
      const rf = makeFV1D({ model: m, N: 2000, flux: st.model === 'bl' ? 'rusanov' : 'godunov', recon: 'muscl', limiter: 'mc' });
      ref = { fv: rf, u: cellAverages(u0, rf.xf), t: 0, step: makeStepper('ssprk3', 2000) };
    }
    // fastest characteristic speed over the range of the data (for the stopping time of Riemann problems)
    let smax = 0; for (let k = 0; k <= 100; k++) smax = Math.max(smax, Math.abs(m.df(k / 100 * (Math.max(1, uL, uR) - Math.min(0, uL, uR)) + Math.min(0, uL, uR))));
    const tEnd = I.periodic ? 20 : 0.95 / smax;
    const lo = Math.min(0, uL, uR), hi = Math.max(1, uL, uR), pad = 0.25 * (hi - lo);
    sim = { m, fv, u, t: 0, steps: 0, exact, ref, tEnd, uL, uR, periodic: I.periodic, mass0: u.reduce((s, v) => s + v, 0) * fv.h,
      step: makeStepper(st.rk, st.N), blown: false, ylim: [lo - pad, hi + pad] };
    anim.stepsPerFrame = Math.max(1, Math.min(20, Math.round(st.N / (st.cfl * 150))));
    draw();
  }
  function advance() {
    const S = sim;
    if (S.blown || S.t >= S.tEnd - 1e-12) return false;
    const dt = Math.min(st.cfl * S.fv.h / Math.max(S.fv.maxSpeed(S.u), 1e-12), S.tEnd - S.t);
    S.step(S.u, S.t, dt, S.fv.rhs); S.t += dt; S.steps++;
    let big = 0; for (let i = 0; i < S.u.length; i++) big = Math.max(big, Math.abs(S.u[i]));
    if (!(big < 1e6)) S.blown = true;
    if (S.ref) {
      const R = S.ref;
      while (R.t < S.t - 1e-13) { const d = Math.min(0.4 * R.fv.h / Math.max(R.fv.maxSpeed(R.u), 1e-12), S.t - R.t); R.step(R.u, R.t, d, R.fv.rhs); R.t += d; }
    }
    if (S.blown) return false;
  }
  function draw() {
    const T = theme(), C = seriesColors(), S = sim, fv = S.fv;
    const P = new Plot(s1, { xlim: [fv.xf[0], fv.xf[fv.N]], ylim: S.ylim, xlabel: 'x' });
    P.frame(); P.hline(0);
    const disp = Array.from(S.u, (v) => Math.max(S.ylim[0] - 10, Math.min(S.ylim[1] + 10, v)));
    P.bars(fv.xf, disp, { color: C[0], alpha: 0.28, width: st.N > 200 ? 1.2 : 1.8 });
    if (S.exact) {
      const xs = [], ys = [];
      for (let k = 0; k <= 1000; k++) { const x = fv.xf[0] + (fv.xf[fv.N] - fv.xf[0]) * k / 1000; xs.push(x); ys.push(S.exact(x, S.t)); }
      P.line(xs, ys, { color: T.ink, width: 1.5 });
    } else P.line(S.ref.fv.xc, S.ref.u, { color: T.ink, width: 1.5 });
    P.legend([{ label: `FV: ${SCALAR_FLUXES[st.flux].label}${st.recon === 'muscl' ? ' + MUSCL' : ''}`, color: C[0] },
      { label: S.exact ? 'exact solution' : 'reference (2000 cells)', color: T.ink }], 'tr');
    // diagnostics
    let mass = 0, mn = Infinity, mx = -Infinity;
    for (let i = 0; i < fv.N; i++) { mass += S.u[i] * fv.h; mn = Math.min(mn, S.u[i]); mx = Math.max(mx, S.u[i]); }
    let l1 = NaN;
    if (S.exact) { const ex = cellAverages((x) => S.exact(x, S.t), fv.xf); l1 = 0; for (let i = 0; i < fv.N; i++) l1 += Math.abs(S.u[i] - ex[i]) * fv.h; }
    else { l1 = 0; const r = 2000 / fv.N; for (let i = 0; i < fv.N; i++) { let a = 0; for (let k = 0; k < r; k++) a += S.ref.u[i * r + k]; l1 += Math.abs(S.u[i] - a / r) * fv.h; } }
    let cons;
    if (S.periodic) cons = `mass Σhūᵢ = ${mass.toFixed(12)}\n  change ${(mass - S.mass0).toExponential(1)} (periodic: 0)`;
    else {
      const expect = (S.m.f(S.uL) - S.m.f(S.uR)) * S.t;
      cons = `mass change ${(mass - S.mass0).toFixed(10)}\n(f(u_L)−f(u_R))·t = ${expect.toFixed(10)}`;
      if (!S.blown && S.steps > 0) selfCheck('playground boundary-flux balance', Math.abs(mass - S.mass0 - expect) < 1e-9);
    }
    if (S.periodic && !S.blown && S.steps > 0) selfCheck('playground mass conservation', Math.abs(mass - S.mass0) < 1e-10);
    const status = S.blown ? `<span class="status-bad">BLOW-UP after ${S.steps} steps</span>`
      : S.t >= S.tEnd - 1e-12 ? 'final time reached' : '';
    out.set(`t = ${S.t.toFixed(3)}   steps = ${S.steps}\nΔt = ν h / max|f'| (ν = ${st.cfl})\nmin ū = ${fmt(mn, 4)}   max ū = ${fmt(mx, 4)}\nTV = ${fmt(totalVariation(S.u, S.periodic), 4)}\nL¹ error = ${fmt(l1, 3)}\n${cons}\n${status}`);
  }
  s1.onResize(draw);
  reset();
  // self-test: default configuration (upwind, ν = 0.8) for a few steps
  for (let k = 0; k < 20; k++) advance();
  draw();
}

/* ------------------------------------------------------------------ */
/* W3: the modified equation of upwind                                  */
/* ------------------------------------------------------------------ */
function modEqWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('square pulse after one period: upwind vs advection–diffusion');
  const s1 = createCanvas(p1, { aspect: 0.45 });
  p1.style.flex = '1 1 420px';
  const st = { nu: 0.5, N: 100 };
  slider(L.controls, { label: 'CFL number $\\nu$', min: 0.05, max: 1, step: 0.05, value: st.nu, onInput: (v) => { st.nu = v; draw(); } });
  slider(L.controls, { label: 'cells $N$', min: 25, max: 400, step: 25, value: st.N, onInput: (v) => { st.N = v; draw(); } });
  const out = readout(L.controls);
  const sq = (x) => (x > 0.25 && x < 0.6 ? 1 : 0);
  function draw() {
    const T = theme(), C = seriesColors();
    const fv = makeFV1D({ model: advection(1), N: st.N, flux: 'upwind' });
    const u = cellAverages(sq, fv.xf), step = makeStepper('fe', st.N);
    const n = Math.round(1 / (st.nu * fv.h)), dt = 1 / n, nu = dt / fv.h; // integer number of steps to t = 1
    for (let k = 0; k < n; k++) step(u, k * dt, dt, fv.rhs);
    const D = upwindNumericalDiffusion(1, fv.h, nu), ad = diffusedSquare(0.25, 0.6, 1, D, 1);
    const P = new Plot(s1, { xlim: [0, 1], ylim: [-0.1, 1.15], xlabel: 'x' });
    P.frame();
    P.bars(fv.xf, u, { color: C[0], alpha: 0.3, width: 1.6 });
    const xs = [], ye = [], ya = [];
    for (let k = 0; k <= 1000; k++) { const x = k / 1000; xs.push(x); ye.push(sq(x)); ya.push(ad(x)); }
    P.line(xs, ye, { color: T.faint, width: 1.4 });
    P.line(xs, ya, { color: C[1], width: 2.2, dash: [6, 3] });
    P.legend([{ label: 'upwind + forward Euler', color: C[0] }, { label: 'exact (advection)', color: T.faint }, { label: 'modified equation u_t + u_x = D u_xx', color: C[1], dash: [6, 3] }], 'tr');
    const ex = cellAverages(sq, fv.xf), adc = cellAverages(ad, fv.xf);
    let e1 = 0, e2 = 0; for (let i = 0; i < st.N; i++) { e1 += Math.abs(u[i] - ex[i]) * fv.h; e2 += Math.abs(u[i] - adc[i]) * fv.h; }
    out.set(`h = ${fmt(fv.h)}, ν = ${fmt(nu, 4)}, ${n} steps\nD = (a h/2)(1 − ν) = ${fmt(D, 4)}\n\nL¹ distance to\n  exact solution   ${fmt(e1, 3)}\n  modified eq.     ${fmt(e2, 3)}`);
    selfCheck('modified equation predicts upwind', nu === 1 || e2 < 0.01 * e1);
  }
  s1.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W4: Sweby diagram + limiter test                                     */
/* ------------------------------------------------------------------ */
const SWEBY_PROFILE = (x) => {
  if (x > 0.35 && x < 0.55) return 1;
  if (x > 0.7 && x < 0.9) return 1 - Math.abs(x - 0.8) / 0.1;
  return Math.exp(-300 * (x - 0.15) ** 2);
};
function swebyWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('limiter functions $\\varphi(r)$ and the TVD region');
  const s1 = createCanvas(p1, { aspect: 0.75 });
  const p2 = L.panel('one period of advection with the selected limiter (100 cells)');
  const s2 = createCanvas(p2, { aspect: 0.75 });
  const st = { lim: 'mc' };
  const names = { minmod: 'minmod', vanleer: 'van Leer', mc: 'MC', superbee: 'superbee', none: 'none (Fromm)' };
  segmented(L.controls, {
    label: 'limiter', value: st.lim, options: Object.keys(names).map((k) => ({ value: k, label: names[k] })),
    onChange: (v) => { st.lim = v; draw(); },
  });
  const out = readout(L.controls);
  const cache = {};
  function solve(lim) {
    if (cache[lim]) return cache[lim];
    const fv = makeFV1D({ model: advection(1), N: 100, flux: 'upwind', recon: 'muscl', limiter: lim });
    const u0 = cellAverages(SWEBY_PROFILE, fv.xf), u = Float64Array.from(u0), step = makeStepper('ssprk3', 100);
    const n = Math.round(1 / (0.45 * fv.h)), dt = 1 / n;
    let tvMax = totalVariation(u), tvInc = 0;
    for (let k = 0; k < n; k++) { step(u, k * dt, dt, fv.rhs); const tv = totalVariation(u); tvInc = Math.max(tvInc, tv - tvMax); tvMax = Math.max(tvMax, tv); }
    let e = 0; for (let i = 0; i < 100; i++) e += Math.abs(u[i] - u0[i]) * fv.h;
    return (cache[lim] = { fv, u, u0, e, tvInc });
  }
  function draw() {
    const T = theme(), C = seriesColors(), ctx = s1.ctx;
    const P = new Plot(s1, { xlim: [-0.5, 3.5], ylim: [0, 2.6], xlabel: 'r = Δ₋ / Δ₊', ylabel: 'φ(r)' });
    P.frame();
    const rs = []; for (let k = 0; k <= 400; k++) rs.push(-0.5 + 4 * k / 400);
    // TVD region: 0 ≤ φ ≤ min(2r, 2) for r > 0
    P.clip();
    ctx.fillStyle = C[2]; ctx.globalAlpha = 0.14;
    ctx.beginPath(); ctx.moveTo(P.X(0), P.Y(0));
    for (const r of rs) if (r >= 0) ctx.lineTo(P.X(r), P.Y(Math.min(2 * r, 2)));
    ctx.lineTo(P.X(3.5), P.Y(0)); ctx.closePath(); ctx.fill();
    // second-order TVD region: between minmod and superbee
    ctx.globalAlpha = 0.22;
    ctx.beginPath(); ctx.moveTo(P.X(0), P.Y(0));
    for (const r of rs) if (r >= 0) ctx.lineTo(P.X(r), P.Y(LIMITER_PHI.superbee(r)));
    for (let k = rs.length - 1; k >= 0; k--) if (rs[k] >= 0) ctx.lineTo(P.X(rs[k]), P.Y(LIMITER_PHI.minmod(rs[k])));
    ctx.closePath(); ctx.fill();
    ctx.restore();
    // linear schemes
    P.line([-0.5, 3.5], [1, 1], { color: T.faint, dash: [5, 4], width: 1.2 });
    P.line([0, 2.6], [0, 2.6], { color: T.faint, dash: [5, 4], width: 1.2 });
    P.line([-0.5, 3.5], [0.25, 2.25], { color: T.faint, dash: [2, 3], width: 1.2 });
    P.text(3.45, 1, 'Lax–Wendroff φ = 1', { align: 'right', baseline: 'top', dy: 3, color: T.soft, font: `11px ${T.ui}` });
    P.text(2.45, 2.5, 'Beam–Warming φ = r', { align: 'right', color: T.soft, font: `11px ${T.ui}` });
    P.text(3.45, 2.25, 'Fromm', { align: 'right', baseline: 'bottom', dy: -3, color: T.soft, font: `11px ${T.ui}` });
    const cols = { minmod: C[0], vanleer: C[4], mc: C[1], superbee: C[5], none: C[6] };
    for (const k of Object.keys(names)) {
      if (k === 'none') continue;
      P.line(rs, rs.map(LIMITER_PHI[k]), { color: cols[k], width: k === st.lim ? 3.2 : 1.3, alpha: k === st.lim ? 1 : 0.7 });
    }
    P.legend(['minmod', 'vanleer', 'mc', 'superbee'].map((k) => ({ label: names[k], color: cols[k] })), 'tl');
    P.text(2.2, 0.35, 'TVD region', { color: C[2], font: `600 12px ${T.ui}` });
    // right: test run
    const R = solve(st.lim);
    const Q = new Plot(s2, { xlim: [0, 1], ylim: [-0.15, 1.2], xlabel: 'x' });
    Q.frame(); Q.hline(0);
    const xs = [], ys = []; for (let k = 0; k <= 1000; k++) { xs.push(k / 1000); ys.push(SWEBY_PROFILE(k / 1000)); }
    Q.line(xs, ys, { color: T.faint, width: 1.4 });
    Q.line(R.fv.xc, R.u, { color: cols[st.lim], width: 1.6 });
    Q.points(R.fv.xc, R.u, { color: cols[st.lim], r: 2.2 });
    Q.legend([{ label: 'exact', color: T.faint }, { label: `MUSCL, ${names[st.lim]}`, color: cols[st.lim] }], 'tr');
    let mn = Infinity, mx = -Infinity; for (const v of R.u) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
    out.set(`TV(initial) = ${fmt(totalVariation(R.u0), 4)}\nTV(final)   = ${fmt(totalVariation(R.u), 4)}\nlargest TV increase in a step:\n  ${R.tvInc > 1e-12 ? R.tvInc.toExponential(2) : 'none (TVD)'}\nmin ū = ${fmt(mn, 3)}, max ū = ${fmt(mx, 3)}\nL¹ error = ${fmt(R.e, 3)}`);
    if (st.lim !== 'none') selfCheck(`${st.lim} is TVD`, R.tvInc <= 1e-12);
  }
  s1.onResize(draw); s2.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W5: measured convergence rates                                       */
/* ------------------------------------------------------------------ */
function ratesWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('$L^1$ error after one period vs cell size $h$ (log–log)');
  const s1 = createCanvas(p1, { aspect: 0.55 });
  p1.style.flex = '1 1 420px';
  const st = { data: 'smooth' };
  const DATA = { smooth: (x) => Math.sin(2 * Math.PI * x), square: (x) => (x > 0.25 && x < 0.6 ? 1 : 0) };
  const METHODS = [
    { key: 'upwind', label: 'upwind (1st order)', recon: 'none', lim: 'minmod', rk: 'fe' },
    { key: 'none', label: 'MUSCL, no limiter', recon: 'muscl', lim: 'none', rk: 'ssprk3' },
    { key: 'minmod', label: 'MUSCL, minmod', recon: 'muscl', lim: 'minmod', rk: 'ssprk3' },
    { key: 'vanleer', label: 'MUSCL, van Leer', recon: 'muscl', lim: 'vanleer', rk: 'ssprk3' },
    { key: 'mc', label: 'MUSCL, MC', recon: 'muscl', lim: 'mc', rk: 'ssprk3' },
    { key: 'superbee', label: 'MUSCL, superbee', recon: 'muscl', lim: 'superbee', rk: 'ssprk3' },
  ];
  const Ns = [25, 50, 100, 200, 400];
  const results = { smooth: {}, square: {} };
  segmented(L.controls, { label: 'initial data', value: st.data, options: [{ value: 'smooth', label: 'smooth sine' }, { value: 'square', label: 'square pulse' }], onChange: (v) => { st.data = v; compute(); } });
  const out = readout(L.controls);
  let busy = false;
  /** compute one method per timeout so the page stays responsive */
  function compute() {
    draw();
    if (busy) return;
    const R = results[st.data], todo = METHODS.filter((m) => !R[m.key]);
    if (!todo.length) return;
    busy = true;
    const m = todo[0], data = st.data, u0 = DATA[data], hs = [], es = [];
    setTimeout(() => {
      for (const N of Ns) {
        const fv = makeFV1D({ model: advection(1), N, flux: 'upwind', recon: m.recon, limiter: m.lim });
        const u = cellAverages(u0, fv.xf);
        advanceFV1D(fv, u, 1, 0.45, m.rk);
        const ex = cellAverages(u0, fv.xf);
        let e = 0; for (let i = 0; i < N; i++) e += Math.abs(u[i] - ex[i]) * fv.h;
        hs.push(fv.h); es.push(e);
      }
      results[data][m.key] = { hs, es, rate: fitRate(hs, es) };
      busy = false;
      compute();
    }, 0);
  }
  function draw() {
    const T = theme(), C = seriesColors(), R = results[st.data];
    const P = new Plot(s1, { xlim: [1 / 600, 1 / 15], ylim: st.data === 'smooth' ? [1e-6, 1] : [1e-3, 1], xlog: true, ylog: true, xlabel: 'h', ylabel: 'L¹ error' });
    P.frame();
    const items = [];
    METHODS.forEach((m, k) => {
      const r = R[m.key];
      if (!r) return;
      P.line(r.hs, r.es, { color: C[k], width: 1.8 });
      P.points(r.hs, r.es, { color: C[k], r: 3 });
      items.push({ label: `${m.label}: slope ${r.rate.toFixed(2)}`, color: C[k] });
    });
    if (st.data === 'smooth') { P.slopeTriangle(0.008, 0.25, 1); P.slopeTriangle(0.006, 4e-5, 2); }
    else { P.slopeTriangle(0.008, 0.45, 0.5); P.slopeTriangle(0.006, 3e-3, 1); }
    if (items.length) P.legend(items, 'br');
    const done = Object.keys(R).length;
    out.set(done < METHODS.length ? `computing… ${done}/${METHODS.length}` : `fitted slopes (last 3 grids):\n${METHODS.map((m) => `${m.label.padEnd(20)} ${R[m.key].rate.toFixed(2)}`).join('\n')}`);
    if (done === METHODS.length) {
      if (st.data === 'smooth') selfCheck('rates: smooth MUSCL ≈ 2', R.none.rate > 1.8 && R.upwind.rate > 0.8);
      else selfCheck('rates: square upwind ≈ ½', Math.abs(R.upwind.rate - 0.5) < 0.1);
    }
  }
  s1.onResize(draw);
  compute();
}

/* ------------------------------------------------------------------ */
/* W6: Euler shock tubes                                                */
/* ------------------------------------------------------------------ */
function sodWidget(fig) {
  const L = widgetLayout(fig);
  const panels = ['density $\\rho$', 'velocity $u$', 'pressure $p$', 'exact solution in the $x$–$t$ plane (colour = $\\rho$)'].map((t) => L.panel(t));
  const surfs = panels.map((p) => createCanvas(p, { aspect: 0.62 }));
  const st = { tube: 'sod', flux: 'hllc', recon: 'none', limiter: 'minmod', N: 200, cfl: 0.5 };
  select(L.controls, { label: 'problem', value: st.tube, options: Object.keys(SHOCK_TUBES).map((k) => ({ value: k, label: SHOCK_TUBES[k].label })), onChange: (v) => { st.tube = v; runToEnd(); } });
  select(L.controls, { label: 'numerical flux', value: st.flux, options: Object.keys(EULER_FLUXES).map((k) => ({ value: k, label: EULER_FLUXES[k].label })), onChange: (v) => { st.flux = v; runToEnd(); } });
  segmented(L.controls, { label: 'reconstruction', value: st.recon, options: [{ value: 'none', label: '1st order' }, { value: 'muscl', label: 'MUSCL' }], onChange: (v) => { st.recon = v; runToEnd(); } });
  select(L.controls, { label: 'limiter (MUSCL, on ρ, u, p)', value: st.limiter, options: LIMITER_OPTIONS.filter((o) => o.value !== 'none'), onChange: (v) => { st.limiter = v; runToEnd(); } });
  slider(L.controls, { label: 'cells $N$', min: 50, max: 800, step: 50, value: st.N, onInput: debounce((v) => { st.N = v; runToEnd(); }, 120) });
  slider(L.controls, { label: 'CFL number', min: 0.1, max: 0.9, step: 0.05, value: st.cfl, onInput: debounce((v) => { st.cfl = v; runToEnd(); }, 120) });
  const anim = new Animator(fig, { step: () => advance(), draw: () => draw() });
  const row = playControls(L.controls, anim, () => { reset(); draw(); });
  button(row, { label: '⏭ End', onClick: () => runToEnd() });
  const out = readout(L.controls);
  let S = null;
  function reset() {
    const P = SHOCK_TUBES[st.tube];
    const fv = makeEuler1D({ N: st.N, flux: st.flux, recon: st.recon, limiter: st.limiter });
    const U = eulerCellAverages((x) => (x < P.x0 ? P.WL : P.WR), fv.xf);
    const ex = exactRiemann(P.WL, P.WR);
    // ranges for fixed axes: from the exact solution at the final time
    const rng = [[Infinity, -Infinity], [Infinity, -Infinity], [Infinity, -Infinity]];
    for (let k = 0; k <= 400; k++) { const W = ex.sample((k / 400 - P.x0) / P.T); for (let c = 0; c < 3; c++) { rng[c][0] = Math.min(rng[c][0], W[c]); rng[c][1] = Math.max(rng[c][1], W[c]); } }
    const lims = rng.map(([a, b]) => { const pd = 0.12 * (b - a || 1); return [a - pd, b + pd]; });
    const stats0 = eulerStats(U, fv.h);
    const UL = primToCons(...P.WL), UR = primToCons(...P.WR), FL = physFlux(UL, [0, 0, 0]), FR = physFlux(UR, [0, 0, 0]);
    S = { P, fv, U, ex, t: 0, steps: 0, failed: false, lims, stats0, FL, FR, step: makeStepper(st.recon === 'muscl' ? 'ssprk2' : 'fe', 3 * st.N) };
    const smax = fv.maxSpeed(U) * 1.5; // rough estimate of the number of steps for the animation speed
    anim.stepsPerFrame = Math.max(1, Math.ceil(P.T * smax / (st.cfl * fv.h) / 120));
  }
  function advance() {
    if (S.failed || S.t >= S.P.T - 1e-14) return false;
    const dt = Math.min(st.cfl * S.fv.h / S.fv.maxSpeed(S.U), S.P.T - S.t);
    S.step(S.U, S.t, dt, S.fv.rhs); S.t += dt; S.steps++;
    const s = eulerStats(S.U, S.fv.h);
    if (!s.finite || s.minRho <= 0 || s.minP <= 0) { S.failed = true; return false; }
    if (S.t >= S.P.T - 1e-14) return false;
  }
  function runToEnd() {
    anim.pause(); reset();
    const t0 = performance.now();
    while (advance() !== false) if (performance.now() - t0 > 3000) break;
    draw();
  }
  function draw() {
    const T = theme(), C = seriesColors(), { P, fv, U, ex } = S;
    const xs = [], W = [[], [], []];
    for (let k = 0; k <= 800; k++) {
      const x = k / 800; xs.push(x);
      const w = S.t > 0 ? ex.sample((x - P.x0) / S.t) : (x < P.x0 ? P.WL : P.WR);
      for (let c = 0; c < 3; c++) W[c].push(w[c]);
    }
    const num = [[], [], []];
    for (let i = 0; i < fv.N; i++) {
      const r = U[3 * i], u = U[3 * i + 1] / r, p = 0.4 * (U[3 * i + 2] - 0.5 * r * u * u);
      num[0].push(r); num[1].push(u); num[2].push(p);
    }
    for (let c = 0; c < 3; c++) {
      const Q = new Plot(surfs[c], { xlim: [0, 1], ylim: S.lims[c], xlabel: 'x' });
      Q.frame();
      Q.line(xs, W[c], { color: T.ink, width: 1.6 });
      if (fv.N <= 200) Q.points(fv.xc, num[c], { color: C[0], r: 2.3 }); else Q.line(fv.xc, num[c], { color: C[0], width: 1.8 });
      if (c === 0) Q.legend([{ label: 'exact', color: T.ink }, { label: `FV (${st.recon === 'muscl' ? 'MUSCL' : '1st order'})`, color: C[0], marker: fv.N <= 200 }], 'tr');
    }
    // x–t diagram of the exact solution
    const s4 = surfs[3];
    const Q = new Plot(s4, { xlim: [0, 1], ylim: [0, P.T], xlabel: 'x', ylabel: 't' });
    Q.frame();
    const view = { invX: (px) => Q.invX(px), invY: (py) => Q.invY(py), X: (x) => Q.X(x), Y: (y) => Q.Y(y) };
    const [rlo, rhi] = S.lims[0];
    drawFunction(s4, view, (x, t) => (x < 0 || x > 1 || t < 0 || t > P.T ? NaN : t <= 0 ? (x < P.x0 ? P.WL[0] : P.WR[0]) : ex.sample((x - P.x0) / t)[0]), { cmap: 'viridis', lo: rlo, hi: rhi, step: 2 });
    const ray = (s, o) => Q.line([P.x0, P.x0 + s * P.T], [0, P.T], o);
    const wave = (type, speeds) => {
      if (type === 'shock') ray(speeds[0], { color: '#fff', width: 3 });
      else for (let k = 0; k <= 6; k++) ray(speeds[0] + (speeds[1] - speeds[0]) * k / 6, { color: 'rgba(255,255,255,0.8)', width: 1 });
    };
    // skip acoustic waves of zero strength (p* = p_K), e.g. for an isolated contact
    const weak = (pK) => Math.abs(ex.pStar - pK) <= 1e-10 * pK;
    if (!weak(P.WL[2])) wave(ex.left, ex.leftSpeeds);
    if (!weak(P.WR[2])) wave(ex.right, ex.rightSpeeds);
    ray(ex.uStar, { color: '#fff', width: 1.8, dash: [6, 4] });
    Q.line([0, 1], [S.t, S.t], { color: C[1], width: 2 });
    colorbar(s4.ctx, 'viridis', Q.px1 - 44, Q.py0 + 6, 8, 60, rlo, rhi, { ink: '#fff' });
    // diagnostics
    const sN = eulerStats(U, fv.h);
    const R = eulerCellAverages((x) => (S.t > 0 ? ex.sample((x - P.x0) / S.t) : (x < P.x0 ? P.WL : P.WR)), fv.xf);
    let l1 = 0; for (let i = 0; i < fv.N; i++) l1 += Math.abs(U[3 * i] - R[3 * i]) * fv.h;
    // with constant boundary states the totals change by (F_L − F_R) t exactly
    const dM = sN.mass - S.stats0.mass - (S.FL[0] - S.FR[0]) * S.t, dE = sN.energy - S.stats0.energy - (S.FL[2] - S.FR[2]) * S.t;
    const status = S.failed ? '<span class="status-bad">FAILED: negative density/pressure or NaN</span>' : S.t >= P.T - 1e-14 ? 'final time reached' : '';
    out.set(`t = ${S.t.toFixed(4)} / ${P.T}   steps = ${S.steps}\np* = ${fmt(ex.pStar, 5)}, u* = ${fmt(ex.uStar, 5)}\nwaves: ${Math.abs(ex.pStar - P.WL[2]) <= 1e-10 * P.WL[2] ? '—' : ex.left} | contact | ${Math.abs(ex.pStar - P.WR[2]) <= 1e-10 * P.WR[2] ? '—' : ex.right}\nL¹ error of ρ = ${fmt(l1, 3)}\nmin ρ = ${fmt(sN.minRho, 3)}, min p = ${fmt(sN.minP, 3)}\nmass   − expected: ${S.failed ? '—' : dM.toExponential(1)}\nenergy − expected: ${S.failed ? '—' : dE.toExponential(1)}\n${status}`);
    if (!S.failed) selfCheck('Euler FV conservation', Math.abs(dM) < 1e-11 && Math.abs(dE) < 1e-9 * Math.max(1, Math.abs(sN.energy)));
  }
  surfs.forEach((s) => s.onResize(() => S && draw()));
  runToEnd();
  selfCheck('Sod default run positive', !S.failed);
}

/* ------------------------------------------------------------------ */
/* W7: Zalesak's slotted disk                                           */
/* ------------------------------------------------------------------ */
function zalesakWidget(fig) {
  const L = widgetLayout(fig);
  const pa = L.panel('first-order upwind');
  const sa = createCanvas(pa, { aspect: 1 });
  const pb = L.panel('MUSCL + limiter');
  const sb = createCanvas(pb, { aspect: 1 });
  const st = { n: 100, lim: 'mc' };
  select(L.controls, { label: 'grid', value: String(st.n), options: [50, 100, 150].map((n) => ({ value: String(n), label: `${n} × ${n}` })), onChange: (v) => { st.n = +v; reset(); } });
  select(L.controls, { label: 'limiter (right)', value: st.lim, options: LIMITER_OPTIONS.filter((o) => o.value !== 'none'), onChange: (v) => { st.lim = v; reset(); } });
  const omega = 2 * Math.PI, vel = rotationVelocity(omega);
  let S = null;
  const anim = new Animator(fig, { step: () => advance(), draw: () => draw() });
  const row = playControls(L.controls, anim, () => reset());
  button(row, { label: '⏭ One revolution', onClick: () => { anim.stepsPerFrame = 1000; anim.play(); } });
  const out = readout(L.controls);
  function reset() {
    anim.pause();
    const n = st.n, box = [0, 1, 0, 1];
    const mk = (recon) => makeFV2D({ nx: n, ny: n, box, vel, recon, limiter: st.lim, bc: 'inflow' });
    const fa = mk('none'), fb = mk('muscl');
    const u0 = cellAverages2D(zalesakDisk, n, n, box, 6);
    const nSteps = Math.ceil(1 / fa.stableDt(0.45));
    // wrap the right-hand sides to record the boundary outflow of every RK stage; SSP-RK3 combines the
    // stages as u + Δt (L₀/6 + L₁/6 + 2L₂/3), so the mass that left the box in one step is Δt (o₀/6 + o₁/6 + 2o₂/3)
    const wrap = (fv, rec) => (u, t, o) => { fv.rhs(u, t, o); rec.push(fv.netOutflow()); };
    const ra = [], rb = [];
    S = { fa, fb, ra, rb, rhsA: wrap(fa, ra), rhsB: wrap(fb, rb), outA: 0, outB: 0,
      ua: Float64Array.from(u0), ub: Float64Array.from(u0), step: makeStepper('ssprk3', n * n), k: 0, nSteps, dt: 1 / nSteps,
      mass0: u0.reduce((a, b) => a + b, 0) / (n * n) };
    anim.stepsPerFrame = Math.max(1, Math.round(nSteps / 240));
    draw();
  }
  function advance() {
    if (S.k >= S.nSteps) return false;
    const t = S.k * S.dt;
    const w = (r) => (r[0] / 6 + r[1] / 6 + 2 * r[2] / 3) * S.dt;
    S.ra.length = 0; S.rb.length = 0;
    S.step(S.ua, t, S.dt, S.rhsA); S.step(S.ub, t, S.dt, S.rhsB); S.k++;
    S.outA += w(S.ra); S.outB += w(S.rb);
    if (S.k >= S.nSteps) return false;
  }
  /** outline of the exact slotted disk rotated by angle θ about (½, ½) */
  function outline(P, theta) {
    const ctx = P.ctx, c = Math.cos(theta), s = Math.sin(theta);
    const pts = [], yb = -Math.sqrt(0.15 ** 2 - 0.025 ** 2), a0 = Math.atan2(yb, 0.025);
    for (let k = 0; k <= 80; k++) { const a = a0 + (2 * Math.PI - 2 * (Math.PI / 2 + a0)) * k / 80; pts.push([0.5 + 0.15 * Math.cos(a), 0.75 + 0.15 * Math.sin(a)]); }
    pts.push([0.475, 0.85], [0.525, 0.85]);
    ctx.save(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]); ctx.beginPath();
    pts.forEach(([x, y], k) => {
      const dx = x - 0.5, dy = y - 0.5, X = P.view.X(0.5 + c * dx - s * dy), Y = P.view.Y(0.5 + s * dx + c * dy);
      if (k) ctx.lineTo(X, Y); else ctx.moveTo(X, Y);
    });
    ctx.closePath(); ctx.stroke(); ctx.restore();
  }
  function panel(surf, u) {
    const T = theme(), n = st.n;
    const view = new View2D(surf, [0, 1, 0, 1], { equal: true, pad: [6, 40, 6, 6] });
    surf.ctx.fillStyle = T.bg; surf.ctx.fillRect(0, 0, surf.w, surf.h);
    drawQuadField(surf, view, n, n, [0, 1, 0, 1], (e) => u[e], { cmap: 'viridis', lo: 0, hi: 1 });
    outline({ ctx: surf.ctx, view }, omega * S.k * S.dt);
    colorbar(surf.ctx, 'viridis', view.X(1) + 8, view.Y(1), 10, view.Y(0) - view.Y(1), 0, 1, { ink: T.soft });
  }
  function draw() {
    panel(sa, S.ua); panel(sb, S.ub);
    const t = S.k * S.dt, n = st.n;
    const ex = cellAverages2D(rotated(zalesakDisk, omega, t), n, n, [0, 1, 0, 1], 4);
    const info = (u) => {
      let e = 0, lo = Infinity, hi = -Infinity, m = 0;
      for (let c = 0; c < u.length; c++) { e += Math.abs(u[c] - ex[c]); lo = Math.min(lo, u[c]); hi = Math.max(hi, u[c]); m += u[c]; }
      return { e: e / (n * n), lo, hi, m: m / (n * n) };
    };
    const A = info(S.ua), B = info(S.ub);
    out.set(`t = ${t.toFixed(3)} (${(t * 100).toFixed(0)}% of a revolution)\n${S.k} / ${S.nSteps} steps\n\n            upwind    MUSCL\nL¹ error   ${A.e.toFixed(4)}    ${B.e.toFixed(4)}\nmin        ${fmt(A.lo, 2).padEnd(9)} ${fmt(B.lo, 2)}\nmax        ${A.hi.toFixed(4)}    ${B.hi.toFixed(4)}\nmass       ${A.m.toFixed(6)}  ${B.m.toFixed(6)}\n+ outflow  ${S.outA.toFixed(6)}  ${S.outB.toFixed(6)}\n= ${(A.m + S.outA).toFixed(12)}\n  ${(B.m + S.outB).toFixed(12)}\n(initial mass ${S.mass0.toFixed(12)})`);
    selfCheck('zalesak finite & mass balance', Number.isFinite(A.e + B.e) && Math.abs(A.m + S.outA - S.mass0) < 1e-12 && Math.abs(B.m + S.outB - S.mass0) < 1e-12);
  }
  sa.onResize(() => S && draw()); sb.onResize(() => S && draw());
  reset();
  for (let k = 0; k < 5; k++) advance();
  draw();
}

initChapter(() => {
  mount('w-rea', reaWidget);
  mount('w-playground', playgroundWidget);
  mount('w-modeq', modEqWidget);
  mount('w-sweby', swebyWidget);
  mount('w-rates', ratesWidget);
  mount('w-sod', sodWidget);
  mount('w-zalesak', zalesakWidget);
});
