/**
 * @file Chapter 1 — Fluxes & conservation laws.
 *
 * Widgets:
 *  - w-advdiff : 1D advection–diffusion, exact Gaussian solution, advective vs diffusive flux
 *  - w-control : 2D control volume in a rotating flow; dM/dt = −(net outflow) checked live
 *  - w-charlin : characteristics of linear advection in the x–t plane
 *  - w-burgers : Burgers' equation: crossing characteristics, breaking time, shock (FV reference)
 *  - w-riemann : the Burgers Riemann problem: shock vs rarefaction, Rankine–Hugoniot speed
 */
import { initChapter, mount, selfCheck } from '../../lib/ui/chapter.js';
import { widgetLayout, slider, select, button, buttonRow, readout, fmt } from '../../lib/ui/controls.js';
import { createCanvas, View2D, theme, onPointer, arrow, seriesColors } from '../../lib/viz/canvas.js';
import { drawFunction } from '../../lib/viz/field2d.js';
import { colorOf } from '../../lib/viz/colormap.js';
import { Plot } from '../../lib/viz/plot1d.js';
import { Animator } from '../../lib/viz/anim.js';
import { gaussLegendre } from '../../lib/core/quad/gauss1d.js';
import { burgers, burgersRiemann } from '../../lib/core/models/scalar.js';
import { makeFV1D, cellAverages } from '../../lib/core/fv/fv1d.js';
import { makeStepper } from '../../lib/core/time/rk.js';

/** Play/pause + reset buttons wired to an Animator. */
function playControls(parent, anim, onReset) {
  const row = buttonRow(parent);
  const play = button(row, { label: '▶ Play', primary: true, onClick: () => anim.toggle() });
  button(row, { label: '↺ Reset', onClick: onReset });
  anim.o.onState = (r) => play.setLabel(r ? '❚❚ Pause' : '▶ Play');
  return play;
}

/* ------------------------------------------------------------------ */
/* W1: advection–diffusion flux decomposition                           */
/* ------------------------------------------------------------------ */
function advDiffWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('concentration $u(x,t)$');
  const s1 = createCanvas(p1, { aspect: 0.38 });
  const p2 = L.panel('fluxes: advective $a u$, diffusive $-k\\,u_x$, total $F$');
  const s2 = createCanvas(p2, { aspect: 0.38 });
  p1.style.flex = p2.style.flex = '1 1 100%';
  const st = { a: 1, k: 0.02, t: 0 };
  const sig = 0.25, x0 = 1;
  const U = (x, t) => { const s2v = sig * sig + 4 * st.k * t; return sig / Math.sqrt(s2v) * Math.exp(-((x - x0 - st.a * t) ** 2) / s2v); };
  const Ux = (x, t) => { const s2v = sig * sig + 4 * st.k * t; return -2 * (x - x0 - st.a * t) / s2v * U(x, t); };
  slider(L.controls, { label: 'velocity $a$', min: -1, max: 2, step: 0.05, value: st.a, onInput: (v) => { st.a = v; draw(); } });
  slider(L.controls, { label: 'diffusivity $k$', min: 0, max: 0.1, step: 0.002, value: st.k, onInput: (v) => { st.k = v; draw(); } });
  const tS = slider(L.controls, { label: 'time $t$', min: 0, max: 1.5, step: 0.01, value: 0, onInput: (v) => { st.t = v; draw(); } });
  const anim = new Animator(fig, { step: () => { st.t += 0.01; if (st.t > 1.5) { st.t = 1.5; return false; } tS.set(st.t); }, draw: () => draw() });
  playControls(L.controls, anim, () => { st.t = 0; tS.set(0); draw(); });
  const out = readout(L.controls);
  function draw() {
    const T = theme(), C = seriesColors();
    const xs = [], u = [], fa = [], fd = [], ft = [];
    for (let i = 0; i <= 600; i++) {
      const x = -0.5 + 4.5 * i / 600; xs.push(x);
      const v = U(x, st.t), vx = Ux(x, st.t);
      u.push(v); fa.push(st.a * v); fd.push(-st.k * vx); ft.push(st.a * v - st.k * vx);
    }
    const P = new Plot(s1, { xlim: [-0.5, 4], ylim: [-0.05, 1.1], xlabel: 'x' });
    P.frame();
    P.line(xs, xs.map((x) => U(x, 0)), { color: T.faint, dash: [4, 4], width: 1.2 });
    P.line(xs, u, { color: C[0], width: 2.2 });
    P.legend([{ label: 'u at t = 0', color: T.faint, dash: [4, 4] }, { label: `u at t = ${st.t.toFixed(2)}`, color: C[0] }]);
    const fm = Math.max(0.3, Math.max(...ft.map(Math.abs), ...fd.map(Math.abs), ...fa.map(Math.abs))) * 1.1;
    const Q = new Plot(s2, { xlim: [-0.5, 4], ylim: [-fm, fm], xlabel: 'x' });
    Q.frame(); Q.hline(0);
    Q.line(xs, fa, { color: C[1], width: 1.6 });
    Q.line(xs, fd, { color: C[2], width: 1.6 });
    Q.line(xs, ft, { color: C[0], width: 2.4 });
    Q.legend([{ label: 'advective a·u', color: C[1] }, { label: 'diffusive −k·uₓ', color: C[2] }, { label: 'total F', color: C[0] }]);
    // mass (exact: σ√π, independent of t)
    const G = gaussLegendre(20); let mass = 0;
    for (let e = 0; e < 60; e++) for (let q = 0; q < 20; q++) { const x = -5 + 15 * (e + (G.x[q] + 1) / 2) / 60; mass += G.w[q] / 2 * 15 / 60 * U(x, st.t); }
    out.set(`peak at x = ${fmt(x0 + st.a * st.t)}\nmax u = ${fmt(U(x0 + st.a * st.t, st.t))}\n∫u dx = ${mass.toFixed(6)}  (conserved)`);
    selfCheck('advdiff mass conserved', Math.abs(mass - sig * Math.sqrt(Math.PI)) < 1e-8);
  }
  s1.onResize(draw); s2.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W2: control volume in a rotating flow                                */
/* ------------------------------------------------------------------ */
function controlWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('concentration $u$ and velocity $\\mathbf a$; drag the box or its corners');
  const s1 = createCanvas(p1, { aspect: 0.9 });
  const p2 = L.panel('rate of change of mass in the box vs. net inflow through its sides');
  const s2 = createCanvas(p2, { aspect: 0.9 });
  const st = { t: 0, box: [0.55, 0.85, 0.3, 0.6], hist: [], drag: null };
  const omega = 2 * Math.PI / 3; // one revolution in 3 time units
  const c = [0.5, 0.5], b0 = [0.75, 0.5], s = 0.08;
  // exact solution: rotate the initial blob rigidly around c
  const u = (x, y, t) => {
    const ct = Math.cos(-omega * t), sn = Math.sin(-omega * t), dx = x - c[0], dy = y - c[1];
    const X = c[0] + ct * dx - sn * dy, Y = c[1] + sn * dx + ct * dy;
    return Math.exp(-((X - b0[0]) ** 2 + (Y - b0[1]) ** 2) / (s * s));
  };
  const vel = (x, y) => [-omega * (y - c[1]), omega * (x - c[0])];
  const G = gaussLegendre(5);
  /** mass in the box at time t: tensor Gauss on a 24×24 subdivision */
  const mass = (t) => {
    const [x0, x1, y0, y1] = st.box, n = 24, hx = (x1 - x0) / n, hy = (y1 - y0) / n;
    let m = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++)
      for (let p = 0; p < 5; p++) for (let q = 0; q < 5; q++)
        m += G.w[p] * G.w[q] / 4 * hx * hy * u(x0 + (i + (G.x[p] + 1) / 2) * hx, y0 + (j + (G.x[q] + 1) / 2) * hy, t);
    return m;
  };
  /** net outflow ∮ u a·n ds at time t */
  const outflow = (t) => {
    const [x0, x1, y0, y1] = st.box, n = 48;
    const sides = [[[x0, y0], [x1, y0], [0, -1]], [[x1, y0], [x1, y1], [1, 0]], [[x1, y1], [x0, y1], [0, 1]], [[x0, y1], [x0, y0], [-1, 0]]];
    let f = 0;
    for (const [A, B, nrm] of sides) {
      const len = Math.hypot(B[0] - A[0], B[1] - A[1]);
      for (let k = 0; k < n; k++) for (let q = 0; q < 5; q++) {
        const tt = (k + (G.x[q] + 1) / 2) / n, x = A[0] + tt * (B[0] - A[0]), y = A[1] + tt * (B[1] - A[1]);
        const [ax, ay] = vel(x, y);
        f += G.w[q] / 2 * len / n * u(x, y, t) * (ax * nrm[0] + ay * nrm[1]);
      }
    }
    return f;
  };
  /** both sides of the balance law at time t */
  const sample = (t) => {
    const d = 1e-4, dM = (mass(t + d) - mass(t - d)) / (2 * d);
    return { t, dM, inflow: -outflow(t) };
  };
  /** precompute both curves over the whole period for the current box */
  const record = () => { st.hist = []; for (let k = 0; k <= 120; k++) st.hist.push(sample(6 * k / 120)); };
  const anim = new Animator(fig, {
    step: () => { st.t += 0.02; if (st.t > 6) { st.t = 6; return false; } },
    draw: () => draw(),
  });
  playControls(L.controls, anim, () => { st.t = 0; draw(); });
  const out = readout(L.controls);
  let view;
  function draw() {
    const T = theme(), C = seriesColors(), ctx = s1.ctx;
    view = new View2D(s1, [0, 1, 0, 1], { equal: true, pad: 6 });
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    drawFunction(s1, view, (x, y) => (x < 0 || x > 1 || y < 0 || y > 1 ? NaN : u(x, y, st.t)), { cmap: 'magma', lo: 0, hi: 1, step: 2 });
    ctx.save(); ctx.strokeStyle = ctx.fillStyle = 'rgba(255,255,255,0.7)'; ctx.lineWidth = 1.2;
    for (let i = 1; i < 10; i++) for (let j = 1; j < 10; j++) {
      const x = i / 10, y = j / 10, [ax, ay] = vel(x, y);
      arrow(ctx, view.X(x), view.Y(y), view.X(x + ax * 0.045), view.Y(y + ay * 0.045), 5);
    }
    const [x0, x1, y0, y1] = st.box;
    ctx.strokeStyle = '#7dd3fc'; ctx.lineWidth = 2.5;
    ctx.strokeRect(view.X(x0), view.Y(y1), view.X(x1) - view.X(x0), view.Y(y0) - view.Y(y1));
    ctx.fillStyle = '#7dd3fc';
    for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) { ctx.beginPath(); ctx.arc(view.X(x), view.Y(y), 5, 0, 7); ctx.fill(); }
    ctx.restore();
    const H = st.hist;
    const ym = Math.max(0.05, ...H.map((h) => Math.max(Math.abs(h.dM), Math.abs(h.inflow)))) * 1.15;
    const P = new Plot(s2, { xlim: [0, 6], ylim: [-ym, ym], xlabel: 'time t' });
    P.frame(); P.hline(0);
    if (H.length) {
      P.line(H.map((h) => h.t), H.map((h) => h.inflow), { color: C[1], width: 5, alpha: 0.45 });
      P.line(H.map((h) => h.t), H.map((h) => h.dM), { color: C[0], width: 1.6 });
    }
    P.vline(st.t, { color: T.ink, dash: [], width: 1 });
    P.legend([{ label: 'dM/dt (mass change)', color: C[0] }, { label: '−∮ u a·n ds (net inflow)', color: C[1] }], 'tr');
    const now = sample(st.t);
    out.set(`t = ${st.t.toFixed(2)}\nM(t)      = ${mass(st.t).toFixed(6)}\ndM/dt     = ${now.dM.toFixed(6)}\nnet inflow= ${now.inflow.toFixed(6)}`);
    selfCheck('control volume balance', Math.abs(now.dM - now.inflow) < 1e-5);
  }
  onPointer(s1.canvas, {
    down: (x, y) => {
      const [x0, x1, y0, y1] = st.box;
      const corners = [[x0, y0, 'll'], [x1, y0, 'lr'], [x1, y1, 'ur'], [x0, y1, 'ul']];
      const hit = corners.find(([cx, cy]) => Math.hypot(view.X(cx) - x, view.Y(cy) - y) < 10);
      st.drag = hit ? hit[2] : { from: [view.invX(x), view.invY(y)], box: [...st.box] };
    },
    move: (x, y) => {
      const wx = Math.max(0, Math.min(1, view.invX(x))), wy = Math.max(0, Math.min(1, view.invY(y)));
      const b = st.box;
      if (typeof st.drag === 'string') {
        if (st.drag[1] === 'l') b[0] = Math.min(wx, b[1] - 0.05); else b[1] = Math.max(wx, b[0] + 0.05);
        if (st.drag[0] === 'l') b[2] = Math.min(wy, b[3] - 0.05); else b[3] = Math.max(wy, b[2] + 0.05);
      } else if (st.drag) {
        const dx = wx - st.drag.from[0], dy = wy - st.drag.from[1], o = st.drag.box;
        const sx = Math.max(-o[0], Math.min(1 - o[1], dx)), sy = Math.max(-o[2], Math.min(1 - o[3], dy));
        st.box = [o[0] + sx, o[1] + sx, o[2] + sy, o[3] + sy];
      }
      st.hist = []; draw();
    },
    up: () => { st.drag = null; record(); draw(); },
  });
  s1.onResize(draw); s2.onResize(draw);
  record(); draw();
}

/* ------------------------------------------------------------------ */
/* W3: characteristics of linear advection                              */
/* ------------------------------------------------------------------ */
const PROFILES = {
  bump: (x) => Math.exp(-80 * (x - 0.3) ** 2),
  square: (x) => (x > 0.15 && x < 0.45 ? 1 : 0),
  wave: (x) => 0.5 + 0.5 * Math.sin(2 * Math.PI * x) * Math.exp(-20 * (x - 0.3) ** 2),
};
function charLinWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('space–time diagram: colour = $u(x,t)$, lines = characteristics $x - a t = $ const');
  const s1 = createCanvas(p1, { aspect: 0.62 });
  const p2 = L.panel('profile $u(x, t)$ at the selected time');
  const s2 = createCanvas(p2, { aspect: 0.62 });
  const st = { a: 0.6, t: 0.5, prof: 'bump' };
  select(L.controls, { label: 'initial profile $u_0(x)$', value: st.prof, options: [{ value: 'bump', label: 'smooth bump' }, { value: 'square', label: 'square pulse' }, { value: 'wave', label: 'wave packet' }], onChange: (v) => { st.prof = v; draw(); } });
  slider(L.controls, { label: 'velocity $a$', min: -0.5, max: 1, step: 0.05, value: st.a, onInput: (v) => { st.a = v; draw(); } });
  slider(L.controls, { label: 'time $t$', min: 0, max: 1, step: 0.01, value: st.t, onInput: (v) => { st.t = v; draw(); } });
  const out = readout(L.controls);
  function draw() {
    const T = theme(), C = seriesColors(), u0 = PROFILES[st.prof];
    const P = new Plot(s1, { xlim: [-0.5, 1.5], ylim: [0, 1], xlabel: 'x', ylabel: 't' });
    P.frame();
    const view = { invX: (px) => P.invX(px), invY: (py) => P.invY(py), X: (x) => P.X(x), Y: (y) => P.Y(y) };
    drawFunction(s1, view, (x, t) => (x < -0.5 || x > 1.5 || t < 0 || t > 1 ? NaN : u0(x - st.a * t)), { cmap: 'viridis', lo: 0, hi: 1, step: 2 });
    for (let k = -10; k <= 20; k++) { const x0 = k * 0.1; P.line([x0, x0 + st.a], [0, 1], { color: 'rgba(255,255,255,0.6)', width: 1 }); }
    P.line([-0.5, 1.5], [st.t, st.t], { color: C[1], width: 2 });
    const xs = [], ys = [], y0 = [];
    for (let i = 0; i <= 600; i++) { const x = -0.5 + 2 * i / 600; xs.push(x); ys.push(u0(x - st.a * st.t)); y0.push(u0(x)); }
    const Q = new Plot(s2, { xlim: [-0.5, 1.5], ylim: [-0.1, 1.15], xlabel: 'x' });
    Q.frame();
    Q.line(xs, y0, { color: T.faint, dash: [4, 4], width: 1.2 });
    Q.line(xs, ys, { color: C[1], width: 2.2 });
    Q.legend([{ label: 'u₀(x)', color: T.faint, dash: [4, 4] }, { label: `u(x, ${st.t.toFixed(2)}) = u₀(x − a t)`, color: C[1] }]);
    out.set(`shift a·t = ${fmt(st.a * st.t)}\ncharacteristic slope dx/dt = a = ${fmt(st.a)}`);
  }
  s1.onResize(draw); s2.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W4: Burgers — crossing characteristics and shock formation           */
/* ------------------------------------------------------------------ */
function burgersWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('characteristics $x = x_0 + u_0(x_0)\\,t$ (coloured by $u_0$)');
  const s1 = createCanvas(p1, { aspect: 0.62 });
  const p2 = L.panel('solution at time $t$');
  const s2 = createCanvas(p2, { aspect: 0.62 });
  const u0 = (x) => 0.25 + 0.5 * Math.sin(2 * Math.PI * x);
  const tStar = 1 / Math.PI; // breaking time: −1 / min u0' = 1/π
  const Tmax = 0.8;
  // reference solution: Godunov + MUSCL(MC) on 800 cells, snapshots every 0.01
  const N = 800, fv = makeFV1D({ model: burgers, N, flux: 'godunov', recon: 'muscl', limiter: 'mc' });
  const snaps = [];
  {
    const u = cellAverages(u0, fv.xf), step = makeStepper('ssprk3', N);
    let t = 0;
    snaps.push(Float64Array.from(u));
    for (let k = 1; k <= Math.round(Tmax / 0.01); k++) {
      const tEnd = k * 0.01;
      while (t < tEnd - 1e-12) { const dt = Math.min(0.4 * fv.h / fv.maxSpeed(u), tEnd - t); step(u, t, dt, fv.rhs); t += dt; }
      snaps.push(Float64Array.from(u));
    }
  }
  const st = { t: 0.2, showChar: true };
  const tS = slider(L.controls, { label: 'time $t$', min: 0, max: Tmax, step: 0.01, value: st.t, onInput: (v) => { st.t = v; draw(); } });
  const anim = new Animator(fig, { step: () => { st.t = Math.round((st.t + 0.01) * 100) / 100; if (st.t > Tmax) { st.t = Tmax; return false; } tS.set(st.t); }, draw: () => draw() });
  playControls(L.controls, anim, () => { st.t = 0; tS.set(0); draw(); });
  const out = readout(L.controls);
  function draw() {
    const T = theme(), C = seriesColors();
    const P = new Plot(s1, { xlim: [0, 1], ylim: [0, Tmax], xlabel: 'x  (periodic)', ylabel: 't' });
    P.frame();
    for (let k = 0; k < 60; k++) {
      const x0 = k / 60, v = u0(x0), xs = [], ts = [];
      // follow the characteristic, wrapping periodically (break the polyline at wraps)
      for (let j = 0; j <= 40; j++) { const t = Tmax * j / 40; xs.push(x0 + v * t); ts.push(t); }
      const segs = []; let cx = [], ct = [];
      for (let j = 0; j < xs.length; j++) {
        const xw = ((xs[j] % 1) + 1) % 1;
        if (cx.length && Math.abs(xw - cx[cx.length - 1]) > 0.5) { segs.push([cx, ct]); cx = []; ct = []; }
        cx.push(xw); ct.push(ts[j]);
      }
      segs.push([cx, ct]);
      P.segments(segs, { color: colorOf('viridis', v, -0.25, 0.75), width: 1.3, alpha: 0.9 });
    }
    P.line([0, 1], [tStar, tStar], { color: T.accent2, width: 1.2, dash: [6, 4] });
    P.text(0.02, tStar, 'breaking time t* = 1/π', { color: T.accent2, baseline: 'bottom', dy: -2 });
    P.line([0, 1], [st.t, st.t], { color: C[1], width: 2 });
    // profile
    const Q = new Plot(s2, { xlim: [0, 1], ylim: [-0.4, 0.9], xlabel: 'x' });
    Q.frame();
    // multivalued "solution" by characteristics: parametric curve (x0 + u0 t, u0)
    const cx = [], cy = [];
    for (let k = 0; k <= 800; k++) { const x0 = k / 800, v = u0(x0); cx.push(((x0 + v * st.t) % 1 + 1) % 1); cy.push(v); }
    const segs = []; let ax = [], ay = [];
    for (let k = 0; k < cx.length; k++) {
      if (ax.length && Math.abs(cx[k] - ax[ax.length - 1]) > 0.5) { segs.push([ax, ay]); ax = []; ay = []; }
      ax.push(cx[k]); ay.push(cy[k]);
    }
    segs.push([ax, ay]);
    Q.segments(segs, { color: T.accent2, width: 1.5, dash: [5, 4] });
    const snap = snaps[Math.round(st.t / 0.01)];
    Q.line(fv.xc, snap, { color: C[0], width: 2.2 });
    Q.legend([{ label: 'characteristics (multi-valued after t*)', color: T.accent2, dash: [5, 4] }, { label: 'entropy solution (fine FV)', color: C[0] }], 'bl');
    let mass = 0; for (let i = 0; i < N; i++) mass += snap[i] * fv.h;
    out.set(`t = ${st.t.toFixed(2)}   (t* ≈ ${tStar.toFixed(3)})\n${st.t < tStar ? 'smooth: characteristics have not crossed' : 'SHOCK: characteristics have crossed'}\n∫u dx = ${mass.toFixed(6)}`);
    selfCheck('burgers mass', Math.abs(mass - 0.25) < 1e-10);
  }
  s1.onResize(draw); s2.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W5: the Burgers Riemann problem                                      */
/* ------------------------------------------------------------------ */
function riemannWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('x–t plane: characteristics from the left (blue) and right (orange) states');
  const s1 = createCanvas(p1, { aspect: 0.62 });
  const p2 = L.panel('solution $u(x, t=1)$');
  const s2 = createCanvas(p2, { aspect: 0.62 });
  const st = { uL: 1, uR: -0.4 };
  slider(L.controls, { label: 'left state $u_L$', min: -1, max: 1.2, step: 0.05, value: st.uL, onInput: (v) => { st.uL = v; draw(); } });
  slider(L.controls, { label: 'right state $u_R$', min: -1, max: 1.2, step: 0.05, value: st.uR, onInput: (v) => { st.uR = v; draw(); } });
  const out = readout(L.controls);
  function draw() {
    const T = theme(), C = seriesColors(), { uL, uR } = st;
    const P = new Plot(s1, { xlim: [-1.5, 1.5], ylim: [0, 1], xlabel: 'x', ylabel: 't' });
    P.frame();
    const shock = uL > uR, s = 0.5 * (uL + uR);
    // left characteristics end at the shock (or the fan edge)
    for (let k = 0; k < 16; k++) {
      const x0 = -1.5 + k * 0.1;
      let tEnd = 1;
      if (shock) { const tc = x0 / (s - uL); if (tc > 0 && tc < 1) tEnd = tc; }
      P.line([x0, x0 + uL * tEnd], [0, tEnd], { color: C[0], width: 1.3 });
    }
    for (let k = 1; k <= 16; k++) {
      const x0 = k * 0.1;
      let tEnd = 1;
      if (shock) { const tc = x0 / (s - uR); if (tc > 0 && tc < 1) tEnd = tc; }
      P.line([x0, x0 + uR * tEnd], [0, tEnd], { color: C[1], width: 1.3 });
    }
    if (shock) P.line([0, s], [0, 1], { color: T.ink, width: 3 });
    else for (let k = 0; k <= 8; k++) { const v = uL + (uR - uL) * k / 8; P.line([0, v], [0, 1], { color: C[2], width: 1.3 }); }
    const xs = [], ys = [];
    for (let i = 0; i <= 600; i++) { const x = -1.5 + 3 * i / 600; xs.push(x); ys.push(burgersRiemann(uL, uR, x / 1)); }
    const lo = Math.min(uL, uR), hi = Math.max(uL, uR), pad = 0.2 * (hi - lo) + 0.1;
    const Q = new Plot(s2, { xlim: [-1.5, 1.5], ylim: [lo - pad, hi + pad], xlabel: 'x' });
    Q.frame();
    Q.line(xs, ys, { color: shock ? T.ink : C[2], width: 2.4 });
    out.set(shock
      ? `SHOCK (u_L > u_R)\nRankine–Hugoniot speed\ns = [f]/[u] = (u_L+u_R)/2 = ${fmt(s)}`
      : `RAREFACTION (u_L < u_R)\nfan: u = x/t for ${fmt(uL)} ≤ x/t ≤ ${fmt(uR)}`);
  }
  s1.onResize(draw); s2.onResize(draw);
  draw();
}

initChapter(() => {
  mount('w-advdiff', advDiffWidget);
  mount('w-control', controlWidget);
  mount('w-charlin', charLinWidget);
  mount('w-burgers', burgersWidget);
  mount('w-riemann', riemannWidget);
});
