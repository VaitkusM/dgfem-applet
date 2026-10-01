/**
 * @file Chapter 6 — Discontinuous Galerkin for conservation laws.
 *
 * Widgets:
 *  - w-dg1d     : DG playground (p, N, flux, initial data, quadrature, limiter; advection or Burgers)
 *  - w-basis    : modal vs nodal bases on [−1,1] with mass / lumped mass / differentiation matrices
 *  - w-anatomy  : one element's update split into volume and surface contributions
 *  - w-spectrum : eigenvalues of the DG operator vs the RK stability region, CFL slider
 *  - w-dg2d     : DG-SEM on periodic quads (rotating blob, diagonal advection, 2D Burgers)
 */
import { initChapter, mount, selfCheck, renderMath } from '../../lib/ui/chapter.js';
import { widgetLayout, slider, select, checkbox, button, buttonRow, readout, fmt, h as el } from '../../lib/ui/controls.js';
import { createCanvas, View2D, theme, seriesColors } from '../../lib/viz/canvas.js';
import { Plot } from '../../lib/viz/plot1d.js';
import { Animator } from '../../lib/viz/anim.js';
import { colorbar } from '../../lib/viz/colormap.js';
import { drawGrid } from '../../lib/viz/meshdraw.js';
import { drawPiecewise, elementColors, shadeStabilityRegion, drawQpField } from '../../lib/viz/dgviz.js';
import { makeDG1D, refNodes, refMatrices } from '../../lib/core/dg/dg1d.js';
import { makeSlopeLimiter } from '../../lib/core/dg/limiters.js';
import { makeDG2D } from '../../lib/core/dg/dg2d.js';
import { operatorEigs, maxStableCFL, spectralRadius } from '../../lib/core/dg/spectrum.js';
import { advection, burgers } from '../../lib/core/models/scalar.js';
import { makeFV1D, cellAverages } from '../../lib/core/fv/fv1d.js';
import { makeStepper, stabilityAmp, RK_LABEL } from '../../lib/core/time/rk.js';
import { legendreAll } from '../../lib/core/basis/legendre.js';
import { lagrangeValues, equispaced } from '../../lib/core/basis/lagrange.js';
import { gaussLegendre } from '../../lib/core/quad/gauss1d.js';
import { symEig } from '../../lib/core/la/dense.js';

const TWO_PI = 2 * Math.PI;
/** periodic wrap to [0,1) */
const wrap = (x) => x - Math.floor(x);

/** Play/pause + reset buttons wired to an Animator. */
function playControls(parent, anim, onReset) {
  const row = buttonRow(parent);
  const play = button(row, { label: '▶ Play', primary: true, onClick: () => anim.toggle() });
  button(row, { label: '↺ Reset', onClick: onReset });
  anim.o.onState = (r) => play.setLabel(r ? '❚❚ Pause' : '▶ Play');
  return play;
}

/** Initial data on the periodic unit interval. */
const INIT = {
  sine: { label: 'sine wave', f: (x) => Math.sin(TWO_PI * x) },
  gauss: { label: 'narrow Gaussian', f: (x) => Math.exp(-200 * (wrap(x) - 0.35) ** 2) },
  square: { label: 'square wave', f: (x) => (wrap(x) > 0.25 && wrap(x) < 0.6 ? 1 : 0) },
  combo: { label: 'Gaussian + square', f: (x) => { const y = wrap(x); return Math.exp(-300 * (y - 0.2) ** 2) + (y > 0.5 && y < 0.75 ? 1 : 0); } },
};

/* ------------------------------------------------------------------ */
/* W1: DG playground                                                   */
/* ------------------------------------------------------------------ */
function dg1dWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('$u_h(x,t)$: one polynomial per element (alternating colours), dots = nodes, dashed = exact / reference');
  p1.style.flex = '1 1 100%';
  L.controls.classList.add('wide');
  const s1 = createCanvas(p1, { aspect: 0.42 });
  const st = { p: 2, N: 16, eq: 'advection', init: 'sine', flux: 'upwind', quad: 'exact', lim: 'none', M: 20, cfl: 0.8, t: 0, speed: 2 };
  let dg, u, step, lim, m0, ref = null, troubled = 0;

  select(L.controls, { label: 'equation', value: st.eq, options: [{ value: 'advection', label: 'linear advection, a = 1' }, { value: 'burgers', label: 'Burgers  u_t + (u²/2)_x = 0' }], onChange: (v) => { st.eq = v; build(); } });
  select(L.controls, { label: 'initial data', value: st.init, options: Object.entries(INIT).map(([k, v]) => ({ value: k, label: v.label })), onChange: (v) => { st.init = v; build(); } });
  slider(L.controls, { label: 'degree $p$', min: 0, max: 5, step: 1, value: st.p, onInput: (v) => { st.p = v; build(); } });
  slider(L.controls, { label: 'elements $N$', min: 4, max: 64, step: 1, value: st.N, onInput: (v) => { st.N = v; build(); } });
  select(L.controls, { label: 'numerical flux', value: st.flux, options: [{ value: 'upwind', label: 'upwind' }, { value: 'rusanov', label: 'Rusanov (local Lax–Friedrichs)' }, { value: 'godunov', label: 'Godunov' }, { value: 'central', label: 'central' }], onChange: (v) => { st.flux = v; build(); } });
  select(L.controls, { label: 'quadrature', value: st.quad, options: [{ value: 'exact', label: 'exact (Gauss, full mass matrix)' }, { value: 'collocated', label: 'GLL collocation (DG-SEM, lumped)' }], onChange: (v) => { st.quad = v; build(); } });
  select(L.controls, { label: 'limiter', value: st.lim, options: [{ value: 'none', label: 'none' }, { value: 'minmod', label: 'minmod (M = 0)' }, { value: 'tvb', label: 'TVB minmod (M below)' }], onChange: (v) => { st.lim = v; build(); } });
  slider(L.controls, { label: 'TVB constant $M$', min: 0, max: 400, step: 5, value: st.M, onInput: (v) => { st.M = v; if (st.lim === 'tvb') build(); } });
  slider(L.controls, { label: 'CFL (× 1/(2p+1))', min: 0.1, max: 1.3, step: 0.05, value: st.cfl, onInput: (v) => { st.cfl = v; } });
  const anim = new Animator(fig, { step: () => advance(), draw: () => draw(), stepsPerFrame: 3 });
  playControls(L.controls, anim, () => { build(); });
  const out = readout(L.controls);

  function model() { return st.eq === 'burgers' ? burgers : advection(1); }
  function build() {
    const m = model();
    dg = makeDG1D({ p: st.p, N: st.N, model: m, flux: st.flux, nodes: 'gll', quad: st.quad });
    u = dg.project(INIT[st.init].f);
    lim = st.lim === 'none' ? null : makeSlopeLimiter(dg, { M: st.lim === 'tvb' ? st.M : 0 });
    if (lim) troubled = lim.apply(u); else troubled = 0;
    step = makeStepper('ssprk3', dg.n, lim ? (v) => { troubled = lim.apply(v); } : undefined);
    m0 = dg.mass(u);
    st.t = 0;
    // Burgers reference: fine FV (MUSCL + MC limiter) advanced in lockstep
    if (st.eq === 'burgers') {
      const fv = makeFV1D({ model: burgers, N: 800, flux: 'godunov', recon: 'muscl', limiter: 'mc' });
      ref = { fv, u: cellAverages(INIT[st.init].f, fv.xf), t: 0, step: makeStepper('ssprk3', 800) };
    } else ref = null;
    draw();
  }
  function dtNow() {
    const s = Math.max(dg.maxSpeed(u), 1e-3);
    return st.cfl * dg.h / (s * (2 * st.p + 1));
  }
  function advance() {
    if (st.t >= 4 - 1e-12) return false;
    const dt = Math.min(dtNow(), 4 - st.t);
    step(u, st.t, dt, dg.rhs);
    st.t += dt;
    if (ref) while (ref.t < st.t - 1e-14) {
      const d = Math.min(0.4 * ref.fv.h / Math.max(ref.fv.maxSpeed(ref.u), 1e-3), st.t - ref.t);
      ref.step(ref.u, ref.t, d, ref.fv.rhs); ref.t += d;
    }
    for (let i = 0; i < dg.n; i += 7) if (!Number.isFinite(u[i]) || Math.abs(u[i]) > 1e6) return false; // blown up
    return true;
  }
  function draw() {
    const T = theme(), C = seriesColors();
    const lo = st.init === 'sine' ? -1.5 : -0.5, hi = 1.5;
    const P = new Plot(s1, { xlim: [0, 1], ylim: [lo, hi], xlabel: 'x  (periodic)' });
    P.frame();
    for (let k = 0; k <= dg.N; k++) P.vline(dg.xf[k], { color: T.rule, dash: [2, 3] });
    const f0 = INIT[st.init].f;
    if (st.eq === 'advection') {
      const xs = [], ys = [];
      for (let i = 0; i <= 800; i++) { const x = i / 800; xs.push(x); ys.push(f0(x - st.t)); }
      P.line(xs, ys, { color: T.ink, width: 1.2, dash: [5, 4], alpha: 0.8 });
    } else if (ref) P.line(ref.fv.xc, ref.u, { color: T.ink, width: 1.2, dash: [5, 4], alpha: 0.8 });
    drawPiecewise(P, dg, u, { color: elementColors(), width: 2.2, nodes: st.N * (st.p + 1) <= 200 });
    if (lim) {
      // mark troubled cells along the bottom
      for (let k = 0; k < dg.N; k++) if (lim.troubled[k]) P.line([dg.xf[k], dg.xf[k + 1]], [lo + 0.05, lo + 0.05], { color: C[1], width: 5 });
    }
    P.legend([{ label: st.eq === 'advection' ? 'exact u₀(x − t)' : 'reference (fine FV)', color: T.ink, dash: [5, 4] }, { label: `DG, p = ${st.p}`, color: T.accent }]
      .concat(lim ? [{ label: 'troubled (limited) cells', color: C[1] }] : []), 'tr');
    const mass = dg.mass(u), E = dg.energy(u);
    let umin = Infinity, umax = -Infinity;
    for (let i = 0; i < dg.n; i++) { umin = Math.min(umin, u[i]); umax = Math.max(umax, u[i]); }
    const finite = Number.isFinite(mass) && Number.isFinite(E);
    let txt = `t = ${st.t.toFixed(3)}   Δt = ${fmt(dtNow())}\n` +
      `mass ∫u_h dx = ${mass.toFixed(12)}\n  drift      = ${fmt(mass - m0, 2)}\n` +
      `energy ½‖u_h‖² = ${fmt(E, 6)}\nnodal min / max = ${fmt(umin)} / ${fmt(umax)}`;
    if (st.eq === 'advection') txt += `\nL² error = ${fmt(dg.l2Error(u, (x) => f0(x - st.t)), 3)}`;
    if (lim) txt += `\nlimited cells (last stage) = ${troubled}`;
    if (!finite || Math.abs(umax) > 1e5) txt += '\n⚠ the solution blew up — lower the CFL number';
    out.set(txt);
    if (finite && Math.abs(umax) < 1e5) selfCheck('dg1d mass conserved', Math.abs(mass - m0) < 1e-10);
  }
  s1.onResize(draw);
  build();
}

/* ------------------------------------------------------------------ */
/* W2: basis explorer                                                  */
/* ------------------------------------------------------------------ */
function basisWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('basis functions $\\phi_0,\\dots,\\phi_p$ on the reference element $[-1,1]$', '1 1 420px');
  const s1 = createCanvas(p1, { aspect: 0.75 });
  const p2 = L.panel('reference matrices', '1 1 300px');
  const mats = el('div', 'mat-wrap');
  p2.appendChild(mats);
  const st = { kind: 'gll', p: 3 };
  select(L.controls, { label: 'basis', value: st.kind, options: [
    { value: 'legendre', label: 'modal: Legendre P_j' },
    { value: 'gll', label: 'nodal: Lagrange at GLL nodes' },
    { value: 'gauss', label: 'nodal: Lagrange at Gauss nodes' },
    { value: 'equi', label: 'nodal: Lagrange at equispaced nodes' }], onChange: (v) => { st.kind = v; draw(); } });
  slider(L.controls, { label: 'degree $p$', min: 1, max: 8, step: 1, value: st.p, onInput: (v) => { st.p = v; draw(); } });
  const out = readout(L.controls);

  /** HTML table of an n×n row-major matrix */
  const table = (title, A, n) => {
    let s = `<div class="mat-title">${title}</div><table class="mat">`;
    for (let i = 0; i < n; i++) {
      s += '<tr>';
      for (let j = 0; j < n; j++) {
        const v = A[i * n + j], z = Math.abs(v) < 1e-13;
        s += `<td class="${z ? 'zero' : ''}">${z ? '0' : v.toFixed(3)}</td>`;
      }
      s += '</tr>';
    }
    return `${s}</table>`;
  };
  function nodesOf() {
    if (st.kind === 'gll') return refNodes(st.p, 'gll').x;
    if (st.kind === 'gauss') return refNodes(st.p, 'gauss').x;
    if (st.kind === 'equi') return equispaced(st.p);
    return null;
  }
  function draw() {
    const T = theme(), C = seriesColors(), p = st.p, np = p + 1;
    const r = nodesOf();
    const phi = (j, x) => (r ? lagrangeValues(r, x)[j] : legendreAll(p, x).P[j]);
    // exact mass and stiffness by Gauss quadrature
    const G = gaussLegendre(np + 2);
    const M = new Float64Array(np * np), S = new Float64Array(np * np);
    const dphi = (j, x) => {
      if (!r) return legendreAll(p, x).dP[j];
      const e = 1e-6; return (phi(j, x + e) - phi(j, x - e)) / (2 * e);
    };
    let Dmat = null;
    if (r) {
      const ref = refMatrices(r, G);
      M.set(ref.M); S.set(ref.S); Dmat = ref.D;
    } else {
      for (let i = 0; i < np; i++) for (let j = 0; j < np; j++) {
        let m = 0, s = 0;
        for (let q = 0; q < G.x.length; q++) { m += G.w[q] * phi(i, G.x[q]) * phi(j, G.x[q]); s += G.w[q] * phi(i, G.x[q]) * dphi(j, G.x[q]); }
        M[i * np + j] = m; S[i * np + j] = s;
      }
    }
    // plot basis functions
    let ymax = 1.1, ymin = -1.1;
    const xs = [], Y = [];
    for (let k = 0; k <= 300; k++) xs.push(-1 + 2 * k / 300);
    for (let j = 0; j < np; j++) { const ys = xs.map((x) => phi(j, x)); Y.push(ys); for (const v of ys) { ymax = Math.max(ymax, v); ymin = Math.min(ymin, v); } }
    const pad = 0.08 * (ymax - ymin);
    const P = new Plot(s1, { xlim: [-1, 1], ylim: [ymin - pad, ymax + pad], xlabel: 'ξ' });
    P.frame(); P.hline(0); P.hline(1, { color: T.rule });
    Y.forEach((ys, j) => P.line(xs, ys, { color: C[j % 6], width: 2 }));
    if (r) P.points(Array.from(r), Array.from(r, () => 0), { color: T.ink, r: 3.5 });
    // condition number of M (symmetric positive definite)
    const ev = symEig(M, np).values, cond = ev[np - 1] / ev[0];
    let html = table('exact mass matrix $M_{ij} = \\int \\phi_i\\phi_j\\,d\\xi$', M, np);
    if (st.kind === 'gll') {
      const lump = refMatrices(r, refNodes(p, 'gll')).M;
      html += table('lumped mass matrix (GLL quadrature) = diag of weights', lump, np);
    }
    if (Dmat) html += table('differentiation matrix $D_{ij} = \\ell_j\'(\\xi_i)$', Dmat, np);
    else html += table('stiffness matrix $S_{ij} = \\int \\phi_i\\,\\phi_j\'\\,d\\xi$', S, np);
    mats.innerHTML = html;
    renderMath(mats);
    let maxAbs = 0; for (const ys of Y) for (const v of ys) maxAbs = Math.max(maxAbs, Math.abs(v));
    out.set(`${np} basis functions\ncond(M) = ${fmt(cond, 3)}\nmax |φ_j| on [−1,1] = ${fmt(maxAbs, 3)}` +
      (st.kind === 'legendre' ? '\nM is diagonal: 2/(2j+1)' : ''));
    // sanity: rows of M sum to ∫φ_i (nodal: = ∫ℓ_i, sum of all = 2)
    let tot = 0; for (let i = 0; i < np * np; i++) tot += M[i];
    if (r) selfCheck('basis: Σ M_ij = 2 for a nodal basis', Math.abs(tot - 2) < 1e-10);
  }
  s1.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W3: anatomy of one element's update                                 */
/* ------------------------------------------------------------------ */
function anatomyWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('$u_h$ near element $k$, face values and numerical fluxes $\\hat F$ (arrows: flux to the right)');
  p1.style.flex = '1 1 100%';
  const s1 = createCanvas(p1, { aspect: 0.32 });
  const p2 = L.panel('$\\partial_t u_h$ on element $k$: volume part + surface part = total');
  p2.style.flex = '1 1 100%';
  const s2 = createCanvas(p2, { aspect: 0.32 });
  L.controls.classList.add('wide');
  const ANAT = {
    sine: { label: 'sine wave', f: (x) => Math.sin(TWO_PI * x), df: (x) => TWO_PI * Math.cos(TWO_PI * x) },
    bump: { label: 'Gaussian bump', f: (x) => Math.exp(-40 * (x - 0.45) ** 2), df: (x) => -80 * (x - 0.45) * Math.exp(-40 * (x - 0.45) ** 2) },
    square: { label: 'square wave', f: INIT.square.f, df: null },
  };
  const st = { p: 2, N: 8, k: 3, data: 'sine' };
  select(L.controls, { label: 'data $u_h$ = L² projection of', value: st.data, options: Object.entries(ANAT).map(([k, v]) => ({ value: k, label: v.label })), onChange: (v) => { st.data = v; draw(); } });
  slider(L.controls, { label: 'degree $p$', min: 0, max: 5, step: 1, value: st.p, onInput: (v) => { st.p = v; draw(); } });
  slider(L.controls, { label: 'element $k$', min: 1, max: 8, step: 1, value: st.k + 1, onInput: (v) => { st.k = v - 1; draw(); } });
  const out = readout(L.controls);
  function draw() {
    const T = theme(), C = seriesColors(), { p, N, k } = st, D = ANAT[st.data];
    const dg = makeDG1D({ p, N, model: advection(1), flux: 'upwind', nodes: 'gll', quad: 'exact' });
    const u = dg.project(D.f), du = new Float64Array(dg.n);
    dg.rhs(u, 0, du);
    const { Minv, Iq, Dq, nq, wq, lL, lR, np } = dg.ref;
    const FL = dg.faceFlux[k], FR = dg.faceFlux[k + 1], h = dg.h;
    // volume and surface parts (nodal values on element k)
    const vol = new Float64Array(np), sur = new Float64Array(np), rv = new Float64Array(np), rs = new Float64Array(np);
    for (let i = 0; i < np; i++) {
      let v = 0;
      for (let q = 0; q < nq; q++) { let uq = 0; for (let j = 0; j < np; j++) uq += Iq[q * np + j] * u[k * np + j]; v += wq[q] * uq * Dq[q * np + i]; }
      rv[i] = v; rs[i] = -(lR[i] * FR - lL[i] * FL);
    }
    for (let i = 0; i < np; i++) {
      let a = 0, b = 0;
      for (let j = 0; j < np; j++) { a += Minv[i * np + j] * rv[j]; b += Minv[i * np + j] * rs[j]; }
      vol[i] = 2 / h * a; sur[i] = 2 / h * b;
    }
    // top plot: elements k−1..k+1
    const xa = dg.xf[Math.max(0, k - 1)], xb = dg.xf[Math.min(N, k + 2)];
    const P = new Plot(s1, { xlim: [xa, xb], ylim: [-1.4, 1.4], xlabel: 'x' });
    P.frame();
    const ctx = s1.ctx;
    ctx.save(); ctx.fillStyle = T.accent; ctx.globalAlpha = 0.08;
    ctx.fillRect(P.X(dg.xf[k]), P.py0, P.X(dg.xf[k + 1]) - P.X(dg.xf[k]), P.py1 - P.py0); ctx.restore();
    for (let j = 0; j <= N; j++) P.vline(dg.xf[j], { color: T.faint });
    drawPiecewise(P, dg, u, { color: (j) => (j === k ? T.accent : T.soft), width: 2.2, nodes: true });
    // face values and fluxes
    for (const [f, F] of [[k, FL], [k + 1, FR]]) {
      const x = dg.xf[f], um = dg.traces(u).uMinus[f], up = dg.traces(u).uPlus[f];
      P.points([x, x], [um, up], { color: C[1], r: 4 });
      P.arrow(x, 1.12, x + 0.4 * h * F, 1.12, { color: C[1], width: 2 });
      P.text(x, 1.3, `F̂ = ${fmt(F)}`, { align: 'center', color: C[1], font: `11px ${T.ui}` });
    }
    // bottom plot: du/dt on element k
    const xs = [], yv = [], ys = [], yt = [], ye = [];
    const lv = new Float64Array(np);
    for (let s = 0; s <= 60; s++) {
      const xi = -1 + 2 * s / 60, x = dg.xf[k] + (xi + 1) * h / 2;
      lagrangeValues(dg.r, xi, lv);
      let a = 0, b = 0; for (let i = 0; i < np; i++) { a += lv[i] * vol[i]; b += lv[i] * sur[i]; }
      xs.push(x); yv.push(a); ys.push(b); yt.push(a + b); ye.push(D.df ? -D.df(x) : NaN);
    }
    const all = [...yv, ...ys, ...yt, ...ye.filter(Number.isFinite)];
    const m = Math.max(1, ...all.map(Math.abs)) * 1.1;
    const Q = new Plot(s2, { xlim: [dg.xf[k], dg.xf[k + 1]], ylim: [-m, m], xlabel: `x in element k = ${k + 1}` });
    Q.frame(); Q.hline(0);
    Q.line(xs, yv, { color: C[2], width: 2 });
    Q.line(xs, ys, { color: C[1], width: 2 });
    Q.line(xs, yt, { color: C[0], width: 3 });
    if (D.df) Q.line(xs, ye, { color: T.ink, width: 1.3, dash: [5, 4] });
    Q.legend([{ label: 'volume part', color: C[2] }, { label: 'surface part', color: C[1] }, { label: 'total ∂u_h/∂t', color: C[0] }]
      .concat(D.df ? [{ label: 'exact −a uₓ', color: T.ink, dash: [5, 4] }] : []), 'tr');
    // checks: mean of volume part = 0, mass rate = −(F̂R − F̂L)
    const G = gaussLegendre(np + 1);
    let mv = 0, mt = 0;
    for (let q = 0; q < G.x.length; q++) {
      lagrangeValues(dg.r, G.x[q], lv);
      let a = 0, b = 0; for (let i = 0; i < np; i++) { a += lv[i] * vol[i]; b += lv[i] * (vol[i] + sur[i]); }
      mv += G.w[q] * a * h / 2; mt += G.w[q] * b * h / 2;
    }
    out.set(`element k = ${k + 1}, h = ${fmt(h)}\nF̂(k−½) = ${fmt(FL, 5)}\nF̂(k+½) = ${fmt(FR, 5)}\n` +
      `∫ volume part dx = ${fmt(mv, 2)}  (≈ 0)\nd/dt ∫ u_h dx  = ${fmt(mt, 6)}\n−(F̂(k+½) − F̂(k−½)) = ${fmt(-(FR - FL), 6)}`);
    selfCheck('anatomy: volume part has zero mean', Math.abs(mv) < 1e-12);
    selfCheck('anatomy: mass rate = flux difference', Math.abs(mt + (FR - FL)) < 1e-12);
  }
  s1.onResize(draw); s2.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* W4: spectrum vs stability region                                    */
/* ------------------------------------------------------------------ */
function spectrumWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('complex plane: $z = \\lambda\\,\\Delta t$ (dots) and stability region $|R(z)| \\le 1$ (shaded)');
  p1.style.flex = '1 1 62%';
  const s1 = createCanvas(p1, { aspect: 0.78 });
  const st = { p: 2, N: 12, flux: 'upwind', quad: 'exact', rk: 'ssprk3', nu: 0.2 };
  let ev = null, numax = 0;
  slider(L.controls, { label: 'degree $p$', min: 0, max: 5, step: 1, value: st.p, onInput: (v) => { st.p = v; recompute(); } });
  slider(L.controls, { label: 'elements $N$', min: 3, max: 24, step: 1, value: st.N, onInput: (v) => { st.N = v; recompute(); } });
  select(L.controls, { label: 'flux', value: st.flux, options: [{ value: 'upwind', label: 'upwind' }, { value: 'central', label: 'central' }], onChange: (v) => { st.flux = v; recompute(); } });
  select(L.controls, { label: 'mass matrix', value: st.quad, options: [{ value: 'exact', label: 'exact' }, { value: 'collocated', label: 'lumped (GLL collocation)' }], onChange: (v) => { st.quad = v; recompute(); } });
  select(L.controls, { label: 'time integrator', value: st.rk, options: ['fe', 'ssprk2', 'ssprk3', 'rk4'].map((k) => ({ value: k, label: RK_LABEL[k] })), onChange: (v) => { st.rk = v; recompute(); } });
  slider(L.controls, { label: 'CFL $\\nu = |a|\\Delta t/h$', min: 0.005, max: 1.5, step: 0.005, value: st.nu, onInput: (v) => { st.nu = v; draw(); } });
  const out = readout(L.controls);
  function recompute() {
    // a = 1, h = 1 (domain [0, N]) so that z = ν λ
    const dg = makeDG1D({ p: st.p, N: st.N, a: 0, b: st.N, model: advection(1), flux: st.flux, nodes: 'gll', quad: st.quad });
    ev = operatorEigs(dg.rhs, dg.n);
    numax = maxStableCFL(ev, st.rk);
    draw();
  }
  function draw() {
    const T = theme();
    const P = new Plot(s1, { xlim: [-3.4, 0.9], ylim: [-3.1, 3.1], xlabel: 'Re z', ylabel: 'Im z', equal: true });
    P.frame();
    shadeStabilityRegion(P, st.rk);
    P.hline(0, { color: T.faint }); P.vline(0, { color: T.faint });
    const xs = [], ys = [], bx = [], by = [];
    let bad = 0;
    for (let k = 0; k < ev.re.length; k++) {
      const zr = st.nu * ev.re[k], zi = st.nu * ev.im[k];
      if (stabilityAmp(st.rk, zr, zi) > 1 + 1e-9) { bx.push(zr); by.push(zi); bad++; } else { xs.push(zr); ys.push(zi); }
    }
    P.points(xs, ys, { color: T.accent, r: 3 });
    P.points(bx, by, { color: '#e0457b', r: 4 });
    P.legend([{ label: 'stability region', color: T.accent2 }, { label: 'λΔt inside', color: T.accent, marker: true }, { label: 'λΔt outside (unstable)', color: '#e0457b', marker: true }], 'tl');
    let maxRe = -Infinity; for (let k = 0; k < ev.re.length; k++) maxRe = Math.max(maxRe, ev.re[k]);
    out.set(`${ev.re.length} eigenvalues (h = 1, a = 1)\nmax Re λ = ${fmt(maxRe, 2)}\nspectral radius = ${fmt(spectralRadius(ev), 4)}\n` +
      `largest stable ν (this N) = ${numax > 0.001 ? numax.toFixed(4) : '≈ 0 (unstable)'}\n(2p+1)·ν_max = ${fmt((2 * st.p + 1) * numax, 3)}\n` +
      `ν = ${st.nu.toFixed(3)}: ${bad ? `UNSTABLE (${bad} outside)` : 'stable'}`);
    selfCheck('spectrum: upwind eigenvalues in left half plane', st.flux !== 'upwind' || maxRe < 1e-9);
  }
  s1.onResize(draw);
  recompute();
}

/* ------------------------------------------------------------------ */
/* W5: 2D DG on quads                                                  */
/* ------------------------------------------------------------------ */
function dg2dWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('$u_h(x,y,t)$ on the periodic unit square (true polynomial per pixel; lines = element edges)');
  p1.style.flex = '1 1 60%';
  const s1 = createCanvas(p1, { aspect: 0.9 });
  const st = { p: 3, N: 8, prob: 'rotate', t: 0, edges: true };
  const blob = (x, y) => Math.exp(-((x - 0.72) ** 2 + (y - 0.5) ** 2) / 0.008);
  const PROBS = {
    rotate: { label: 'rotating Gaussian blob', o: { velocity: (x, y) => [-TWO_PI * (y - 0.5), TWO_PI * (x - 0.5)] },
      u0: blob, exact: (x, y, t) => { const c = Math.cos(TWO_PI * t), s = Math.sin(TWO_PI * t), dx = x - 0.5, dy = y - 0.5; return blob(0.5 + c * dx + s * dy, 0.5 - s * dx + c * dy); }, lo: -0.15, hi: 1.05, T: 1 },
    diag: { label: 'square pulse, velocity (1, 1)', o: { velocity: () => [1, 1] },
      u0: (x, y) => (Math.abs(wrap(x) - 0.5) < 0.2 && Math.abs(wrap(y) - 0.5) < 0.2 ? 1 : 0), exact: (x, y, t) => (Math.abs(wrap(x - t) - 0.5) < 0.2 && Math.abs(wrap(y - t) - 0.5) < 0.2 ? 1 : 0), lo: -0.2, hi: 1.2, T: 1 },
    burgers: { label: 'Burgers (shock forms at t ≈ 0.11)', o: { problem: 'burgers' },
      u0: (x, y) => 0.5 + 0.7 * Math.sin(TWO_PI * (x + y)), exact: null, lo: -0.35, hi: 1.35, T: 0.4 },
  };
  let dg, u, step, m0;
  select(L.controls, { label: 'problem', value: st.prob, options: Object.entries(PROBS).map(([k, v]) => ({ value: k, label: v.label })), onChange: (v) => { st.prob = v; build(); } });
  slider(L.controls, { label: 'degree $p$', min: 1, max: 5, step: 1, value: st.p, onInput: (v) => { st.p = v; build(); } });
  slider(L.controls, { label: 'elements per side', min: 3, max: 16, step: 1, value: st.N, onInput: (v) => { st.N = v; build(); } });
  checkbox(L.controls, { label: 'show element edges', value: st.edges, onChange: (v) => { st.edges = v; draw(); } });
  const anim = new Animator(fig, { step: () => advance(), draw: () => draw(), stepsPerFrame: 4 });
  playControls(L.controls, anim, () => build());
  const out = readout(L.controls);
  function build() {
    const P = PROBS[st.prob];
    dg = makeDG2D({ p: st.p, nx: st.N, ny: st.N, flux: 'rusanov', ...P.o });
    u = dg.project(P.u0);
    step = makeStepper('ssprk3', dg.n);
    m0 = dg.mass(u); st.t = 0;
    draw();
  }
  function advance() {
    const P = PROBS[st.prob];
    if (st.t >= P.T - 1e-12) return false;
    const dt = Math.min(dg.dtStable(u, 0.5), P.T - st.t);
    step(u, st.t, dt, dg.rhs); st.t += dt;
    return true;
  }
  function draw() {
    const T = theme(), P = PROBS[st.prob];
    const ctx = s1.ctx;
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, s1.w, s1.h);
    const view = new View2D(s1, [0, 1, 0, 1], { equal: true, pad: [8, 60, 8, 8] });
    drawQpField(s1, view, dg, u, { cmap: 'viridis', lo: P.lo, hi: P.hi });
    if (st.edges) drawGrid(ctx, view, st.N, st.N, [0, 1, 0, 1], { color: 'rgba(255,255,255,0.45)', width: 0.8 });
    colorbar(ctx, 'viridis', view.X(1) + 12, view.Y(1), 12, view.Y(0) - view.Y(1), P.lo, P.hi, { ink: T.soft });
    let umin = Infinity, umax = -Infinity;
    for (let i = 0; i < dg.n; i++) { umin = Math.min(umin, u[i]); umax = Math.max(umax, u[i]); }
    const mass = dg.mass(u);
    let txt = `t = ${st.t.toFixed(3)} / ${P.T}\nunknowns = ${dg.n} (${(st.p + 1) ** 2} per element)\nmass = ${mass.toFixed(12)}\ndrift = ${fmt(mass - m0, 2)}\nmin / max = ${fmt(umin)} / ${fmt(umax)}`;
    if (P.exact) txt += `\nL² error = ${fmt(dg.l2Error(u, (x, y) => P.exact(x, y, st.t)), 3)}`;
    out.set(txt);
    if (Number.isFinite(mass)) selfCheck('dg2d mass conserved', Math.abs(mass - m0) < 1e-10);
  }
  s1.onResize(() => { s1._qp = null; draw(); });
  build();
}

initChapter(() => {
  mount('w-dg1d', dg1dWidget);
  mount('w-basis', basisWidget);
  mount('w-anatomy', anatomyWidget);
  mount('w-spectrum', spectrumWidget);
  mount('w-dg2d', dg2dWidget);
});
