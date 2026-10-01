/**
 * @file Chapter 0 — Primer: scalar/vector fields, gradient, divergence,
 *       divergence theorem, convergence plots.
 *
 * Widgets:
 *  - w-gradient : scalar field heat-map + level sets (marching squares) + gradient arrows
 *  - w-divthm   : draggable polygon in a vector field; flux through edges vs ∫ div F
 *  - w-conv     : piecewise polynomial interpolation and its log–log error plot
 */
import { initChapter, mount, selfCheck } from '../../lib/ui/chapter.js';
import { widgetLayout, slider, select, checkbox, readout, fmt } from '../../lib/ui/controls.js';
import { createCanvas, View2D, theme, onPointer, arrow, seriesColors } from '../../lib/viz/canvas.js';
import { drawFunction } from '../../lib/viz/field2d.js';
import { colorbar } from '../../lib/viz/colormap.js';
import { drawContours } from '../../lib/viz/contour.js';
import { Plot } from '../../lib/viz/plot1d.js';
import { gaussLegendre } from '../../lib/core/quad/gauss1d.js';
import { triangleRule } from '../../lib/core/quad/simplex.js';
import { interpError, interpolantAt, interpNodes } from '../../lib/core/verify/interp1d.js';
import { fitRate } from '../../lib/core/verify/rates.js';

/* ------------------------------------------------------------------ */
/* Widget 1: scalar field, level sets, gradient                         */
/* ------------------------------------------------------------------ */

const SCALARS = {
  bump: {
    label: 'bump  u = exp(−(x²+y²))',
    u: (x, y) => Math.exp(-(x * x + y * y)),
    g: (x, y) => { const u = Math.exp(-(x * x + y * y)); return [-2 * x * u, -2 * y * u]; },
    lap: (x, y) => { const r2 = x * x + y * y; return (4 * r2 - 4) * Math.exp(-r2); },
  },
  saddle: {
    label: 'saddle  u = (x² − y²)/2',
    u: (x, y) => 0.5 * (x * x - y * y),
    g: (x, y) => [x, -y],
    lap: () => 0,
  },
  ripple: {
    label: 'ripple  u = sin(2x) cos(1.5y)',
    u: (x, y) => Math.sin(2 * x) * Math.cos(1.5 * y),
    g: (x, y) => [2 * Math.cos(2 * x) * Math.cos(1.5 * y), -1.5 * Math.sin(2 * x) * Math.sin(1.5 * y)],
    lap: (x, y) => -6.25 * Math.sin(2 * x) * Math.cos(1.5 * y),
  },
  plane: {
    label: 'plane  u = 0.6x + 0.3y',
    u: (x, y) => 0.6 * x + 0.3 * y,
    g: () => [0.6, 0.3],
    lap: () => 0,
  },
};

function gradientWidget(fig) {
  const L = widgetLayout(fig);
  const panel = L.panel();
  const surf = createCanvas(panel, { aspect: 0.82 });
  const st = { field: 'bump', show: 'u', arrows: true, contours: true, probe: [0.7, 0.4] };
  select(L.controls, {
    label: 'Scalar field $u(x,y)$', value: st.field,
    options: Object.entries(SCALARS).map(([k, v]) => ({ value: k, label: v.label })),
    onChange: (v) => { st.field = v; draw(); },
  });
  select(L.controls, {
    label: 'Colour shows', value: st.show,
    options: [{ value: 'u', label: 'u itself' }, { value: 'lap', label: 'Laplacian Δu' }],
    onChange: (v) => { st.show = v; draw(); },
  });
  checkbox(L.controls, { label: 'level sets (contours)', value: true, onChange: (v) => { st.contours = v; draw(); } });
  checkbox(L.controls, { label: 'gradient arrows $\\nabla u$', value: true, onChange: (v) => { st.arrows = v; draw(); } });
  const out = readout(L.controls);

  function draw() {
    const F = SCALARS[st.field], T = theme();
    const view = new View2D(surf, [-2, 2, -2, 2], { equal: true, pad: [8, 72, 8, 8] });
    surf.ctx.fillStyle = T.bg; surf.ctx.fillRect(0, 0, surf.w, surf.h);
    const fn = st.show === 'u' ? F.u : F.lap;
    // colour range from samples
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i <= 40; i++) for (let j = 0; j <= 40; j++) {
      const v = fn(-2 + i * 0.1, -2 + j * 0.1); lo = Math.min(lo, v); hi = Math.max(hi, v);
    }
    let cmap = 'viridis';
    if (st.show === 'lap') { const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-9); lo = -m; hi = m; cmap = 'rdbu'; }
    if (hi - lo < 1e-12) { lo -= 1; hi += 1; }
    drawFunction(surf, view, (x, y) => (x < -2 || x > 2 || y < -2 || y > 2 ? NaN : fn(x, y)), { cmap, lo, hi, step: 2 });
    if (st.contours) {
      let ulo = Infinity, uhi = -Infinity;
      for (let i = 0; i <= 40; i++) for (let j = 0; j <= 40; j++) { const v = F.u(-2 + i * 0.1, -2 + j * 0.1); ulo = Math.min(ulo, v); uhi = Math.max(uhi, v); }
      const levels = [];
      for (let k = 1; k < 12; k++) levels.push(ulo + (uhi - ulo) * k / 12);
      drawContours(surf.ctx, view, F.u, levels, { color: 'rgba(255,255,255,0.75)', width: 1 });
    }
    if (st.arrows) {
      const ctx = surf.ctx;
      ctx.save(); ctx.strokeStyle = ctx.fillStyle = 'rgba(20,20,20,0.85)'; ctx.lineWidth = 1.2;
      let gmax = 1e-9;
      for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) { const g = F.g(-1.8 + i * 0.45, -1.8 + j * 0.45); gmax = Math.max(gmax, Math.hypot(g[0], g[1])); }
      for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) {
        const x = -1.8 + i * 0.45, y = -1.8 + j * 0.45, g = F.g(x, y), s = 0.38 / gmax;
        arrow(ctx, view.X(x), view.Y(y), view.X(x + g[0] * s), view.Y(y + g[1] * s), 5);
      }
      ctx.restore();
    }
    // probe
    const [px, py] = st.probe, g = F.g(px, py), ctx = surf.ctx;
    ctx.save();
    ctx.strokeStyle = ctx.fillStyle = T.accent2; ctx.lineWidth = 2.2;
    const gl = Math.hypot(g[0], g[1]) || 1, s = 0.6 / Math.max(gl, 0.3);
    arrow(ctx, view.X(px), view.Y(py), view.X(px + g[0] * s), view.Y(py + g[1] * s), 8);
    ctx.beginPath(); ctx.arc(view.X(px), view.Y(py), 4, 0, 7); ctx.fill();
    ctx.restore();
    colorbar(surf.ctx, cmap, surf.w - 64, 10, 12, surf.h - 20, lo, hi, { ink: T.soft });
    out.set(`probe (${fmt(px)}, ${fmt(py)})\nu   = ${fmt(F.u(px, py))}\n∇u  = (${fmt(g[0])}, ${fmt(g[1])})\n|∇u| = ${fmt(gl)}\nΔu  = ${fmt(F.lap(px, py))}`);
    selfCheck('gradient widget finite', Number.isFinite(F.u(px, py)));
    surf._view = view;
  }
  onPointer(surf.canvas, {
    down: (x, y) => { const v = surf._view; st.probe = [v.invX(x), v.invY(y)].map((t) => Math.max(-2, Math.min(2, t))); draw(); },
    move: (x, y) => { const v = surf._view; st.probe = [v.invX(x), v.invY(y)].map((t) => Math.max(-2, Math.min(2, t))); draw(); },
  });
  surf.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* Widget 2: divergence theorem playground                              */
/* ------------------------------------------------------------------ */

const VFIELDS = {
  source: { label: 'source  F = (x, y)/2', F: (x, y) => [x / 2, y / 2], div: () => 1 },
  vortex: { label: 'vortex  F = (−y, x)', F: (x, y) => [-y, x], div: () => 0 },
  wind: { label: 'uniform wind  F = (1, 0.4)', F: () => [1, 0.4], div: () => 0 },
  gauss: {
    label: 'localised source  F = (x, y)·exp(−r²/2)',
    F: (x, y) => { const e = Math.exp(-(x * x + y * y) / 2); return [x * e, y * e]; },
    div: (x, y) => { const r2 = x * x + y * y; return (2 - r2) * Math.exp(-r2 / 2); },
  },
  waves: { label: 'compressions  F = (sin 2x, sin 2y)/2', F: (x, y) => [Math.sin(2 * x) / 2, Math.sin(2 * y) / 2], div: (x, y) => Math.cos(2 * x) + Math.cos(2 * y) },
};

/**
 * Outward flux ∮ F·n ds of each polygon edge (Gauss quadrature on edges).
 * Works for either orientation: we flip normals for clockwise polygons.
 */
function edgeFluxes(poly, F) {
  const n = poly.length, G = gaussLegendre(8), SUB = 8; // 8 sub-segments × 8 Gauss points per edge
  let area2 = 0;
  for (let i = 0; i < n; i++) { const [ax, ay] = poly[i], [bx, by] = poly[(i + 1) % n]; area2 += ax * by - bx * ay; }
  const orient = area2 >= 0 ? 1 : -1; // +1 = counter-clockwise
  const fl = [];
  for (let i = 0; i < n; i++) {
    const [ax, ay] = poly[i], [bx, by] = poly[(i + 1) % n];
    const len = Math.hypot(bx - ax, by - ay);
    const nx = orient * (by - ay) / len, ny = -orient * (bx - ax) / len; // outward normal
    let s = 0;
    for (let k = 0; k < SUB; k++)
      for (let q = 0; q < G.x.length; q++) {
        const t = (k + (G.x[q] + 1) / 2) / SUB, [fx, fy] = F(ax + t * (bx - ax), ay + t * (by - ay));
        s += G.w[q] / 2 / SUB * len * (fx * nx + fy * ny);
      }
    fl.push({ flux: s, nx, ny, mx: (ax + bx) / 2, my: (ay + by) / 2 });
  }
  return { fl, orient };
}

/**
 * ∫_P div F dA for a simple polygon P via a signed fan triangulation from
 * vertex 0: triangles with negative orientation subtract. Correct for any
 * simple (also non-convex) polygon.
 */
function areaIntegral(poly, f) {
  const R = triangleRule(12), m = 8; // each fan triangle is split into m² sub-triangles
  let total = 0, area2 = 0;
  const [ox, oy] = poly[0];
  for (let i = 1; i + 1 < poly.length; i++) {
    const [ax, ay] = poly[i], [bx, by] = poly[i + 1];
    const det = (ax - ox) * (by - oy) - (bx - ox) * (ay - oy); // signed 2×area
    const P = (s, t) => [ox + (ax - ox) * s + (bx - ox) * t, oy + (ay - oy) * s + (by - oy) * t];
    let s = 0;
    // uniform refinement of the reference triangle: "upright" and "flipped" sub-triangles
    for (let I = 0; I < m; I++) for (let J = 0; I + J < m; J++) {
      const subs = [[[I, J], [I + 1, J], [I, J + 1]]];
      if (I + J < m - 1) subs.push([[I + 1, J], [I + 1, J + 1], [I, J + 1]]);
      for (const [a, b, c] of subs)
        for (let q = 0; q < R.n; q++) {
          const sx = (a[0] + (b[0] - a[0]) * R.x[q] + (c[0] - a[0]) * R.y[q]) / m;
          const sy = (a[1] + (b[1] - a[1]) * R.x[q] + (c[1] - a[1]) * R.y[q]) / m;
          s += R.w[q] / (m * m) * f(...P(sx, sy));
        }
    }
    total += s * det; area2 += det;
  }
  return Math.sign(area2 || 1) * total;
}

function divThmWidget(fig) {
  const L = widgetLayout(fig);
  const panel = L.panel();
  const surf = createCanvas(panel, { aspect: 0.82 });
  const st = { field: 'gauss', poly: [[-1.2, -0.8], [0.6, -1.3], [1.4, 0.2], [0.5, 1.2], [-0.9, 0.9]], showDiv: true, drag: -1, grab: null };
  select(L.controls, {
    label: 'Vector field $\\mathbf F$', value: st.field,
    options: Object.entries(VFIELDS).map(([k, v]) => ({ value: k, label: v.label })),
    onChange: (v) => { st.field = v; draw(); },
  });
  checkbox(L.controls, { label: 'colour = divergence $\\nabla\\cdot\\mathbf F$', value: true, onChange: (v) => { st.showDiv = v; draw(); } });
  const out = readout(L.controls);
  let view;
  function draw() {
    const V = VFIELDS[st.field], T = theme(), ctx = surf.ctx;
    view = new View2D(surf, [-2.2, 2.2, -2.2, 2.2], { equal: true, pad: [8, 72, 8, 8] });
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, surf.w, surf.h);
    const m = 2.2;
    if (st.showDiv) drawFunction(surf, view, (x, y) => (Math.abs(x) > m || Math.abs(y) > m ? NaN : V.div(x, y)), { cmap: 'rdbu', lo: -2, hi: 2, step: 3 });
    // arrows
    ctx.save(); ctx.strokeStyle = ctx.fillStyle = st.showDiv ? 'rgba(25,25,30,0.65)' : T.soft; ctx.lineWidth = 1;
    let fmax = 1e-9;
    for (let i = 0; i < 11; i++) for (let j = 0; j < 11; j++) { const f = V.F(-2 + i * 0.4, -2 + j * 0.4); fmax = Math.max(fmax, Math.hypot(f[0], f[1])); }
    for (let i = 0; i < 11; i++) for (let j = 0; j < 11; j++) {
      const x = -2 + i * 0.4, y = -2 + j * 0.4, f = V.F(x, y), s = 0.34 / fmax;
      arrow(ctx, view.X(x), view.Y(y), view.X(x + f[0] * s), view.Y(y + f[1] * s), 4);
    }
    ctx.restore();
    // polygon
    const { fl } = edgeFluxes(st.poly, V.F);
    ctx.save();
    ctx.fillStyle = 'rgba(255,255,255,0.22)'; ctx.strokeStyle = '#15171c'; ctx.lineWidth = 2;
    ctx.beginPath();
    st.poly.forEach(([x, y], i) => (i ? ctx.lineTo(view.X(x), view.Y(y)) : ctx.moveTo(view.X(x), view.Y(y))));
    ctx.closePath(); ctx.fill(); ctx.stroke();
    // normals coloured by sign of the flux
    const maxF = Math.max(...fl.map((e) => Math.abs(e.flux)), 1e-9);
    for (const e of fl) {
      const c = e.flux >= 0 ? T.accent2 : T.accent;
      ctx.strokeStyle = ctx.fillStyle = c; ctx.lineWidth = 2.5;
      const len = 0.15 + 0.45 * Math.abs(e.flux) / maxF;
      arrow(ctx, view.X(e.mx), view.Y(e.my), view.X(e.mx + e.nx * len), view.Y(e.my + e.ny * len), 8);
    }
    for (const [x, y] of st.poly) {
      ctx.fillStyle = '#ffffff'; ctx.strokeStyle = '#15171c'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(view.X(x), view.Y(y), 6, 0, 7); ctx.fill(); ctx.stroke();
    }
    ctx.restore();
    colorbar(ctx, 'rdbu', surf.w - 64, 10, 12, surf.h - 20, -2, 2, { ink: T.soft });
    const total = fl.reduce((s, e) => s + e.flux, 0), vol = areaIntegral(st.poly, V.div);
    out.set(fl.map((e, i) => `edge ${i + 1}: ∫F·n ds = ${e.flux >= 0 ? ' ' : ''}${e.flux.toFixed(4)}`).join('\n')
      + `\n<b>Σ edges   = ${total.toFixed(6)}</b>\n<b>∫ div F dA = ${vol.toFixed(6)}</b>`);
    selfCheck('divergence theorem holds', Math.abs(total - vol) < 1e-6);
  }
  onPointer(surf.canvas, {
    down: (x, y) => {
      st.drag = st.poly.findIndex(([px, py]) => Math.hypot(view.X(px) - x, view.Y(py) - y) < 12);
      st.grab = st.drag < 0 ? [view.invX(x), view.invY(y)] : null;
    },
    move: (x, y) => {
      const wx = Math.max(-2.1, Math.min(2.1, view.invX(x))), wy = Math.max(-2.1, Math.min(2.1, view.invY(y)));
      if (st.drag >= 0) st.poly[st.drag] = [wx, wy];
      else if (st.grab) {
        const dx = wx - st.grab[0], dy = wy - st.grab[1];
        st.poly = st.poly.map(([px, py]) => [px + dx, py + dy]);
        st.grab = [wx, wy];
      }
      draw();
    },
    up: () => { st.drag = -1; st.grab = null; },
  });
  surf.onResize(draw);
  draw();
}

/* ------------------------------------------------------------------ */
/* Widget 3: convergence explorer                                       */
/* ------------------------------------------------------------------ */

const FUNCS = {
  smooth: { label: 'smooth: sin(2πx) + x²', f: (x) => Math.sin(2 * Math.PI * x) + x * x },
  runge: { label: 'steep but smooth: 1/(1+100(x−½)²)', f: (x) => 1 / (1 + 100 * (x - 0.5) ** 2) },
  kink: { label: 'kink: |x − ⅓|', f: (x) => Math.abs(x - 1 / 3) },
  jump: { label: 'jump at x = ⅓', f: (x) => (x < 1 / 3 ? 0 : 1) },
};

function convWidget(fig) {
  const L = widgetLayout(fig);
  const p1 = L.panel('the function (grey) and its piecewise interpolant (colour)');
  const s1 = createCanvas(p1, { aspect: 0.45 });
  const p2 = L.panel('error vs mesh size h on log–log axes');
  const s2 = createCanvas(p2, { aspect: 0.62 });
  const st = { fn: 'smooth', p: 1, N: 8, norm: 'L2' };
  select(L.controls, { label: 'Function', value: st.fn, options: Object.entries(FUNCS).map(([k, v]) => ({ value: k, label: v.label })), onChange: (v) => { st.fn = v; update(); } });
  slider(L.controls, { label: 'polynomial degree $p$', min: 0, max: 4, step: 1, value: st.p, onInput: (v) => { st.p = v; update(); } });
  slider(L.controls, { label: 'number of cells $N$ (h = 1/N)', min: 2, max: 64, step: 1, value: st.N, onInput: (v) => { st.N = v; update(); } });
  select(L.controls, { label: 'error norm', value: st.norm, options: [{ value: 'L2', label: 'L² (root mean square)' }, { value: 'Linf', label: 'max norm' }], onChange: (v) => { st.norm = v; update(); } });
  const out = readout(L.controls);
  let curve = null;
  function update() {
    const f = FUNCS[st.fn].f;
    const Ns = [2, 4, 8, 16, 32, 64, 128, 256];
    curve = Ns.map((N) => ({ N, h: 1 / N, e: interpError(f, N, st.p)[st.norm] }));
    draw();
  }
  function draw() {
    const T = theme(), C = seriesColors(), f = FUNCS[st.fn].f;
    // top: function + interpolant
    const xs = [], ys = [], ysI = [];
    for (let k = 0; k <= 800; k++) { const x = k / 800; xs.push(x); ys.push(f(x)); }
    let lo = Math.min(...ys), hi = Math.max(...ys);
    const pad = 0.15 * (hi - lo || 1);
    const P = new Plot(s1, { xlim: [0, 1], ylim: [lo - pad, hi + pad], xlabel: 'x' });
    P.frame();
    P.line(xs, ys, { color: T.faint, width: 3 });
    const h = 1 / st.N, segs = [];
    for (let e = 0; e < st.N; e++) {
      const sx = [], sy = [];
      for (let k = 0; k <= 30; k++) { const x = e * h + h * k / 30 * 0.999999 + (k === 0 ? 1e-9 : 0); sx.push(x); sy.push(interpolantAt(f, st.N, st.p, x)); }
      segs.push([sx, sy]);
      ysI.push(...sy);
    }
    P.segments(segs, { color: C[0], width: 2 });
    const nodes = interpNodes(st.p), nx = [], ny = [];
    for (let e = 0; e < st.N; e++) for (const t of nodes) { const x = e * h + (t + 1) * h / 2; nx.push(x); ny.push(f(x)); }
    P.points(nx, ny, { color: C[1], r: 2.8 });
    for (let e = 0; e <= st.N; e++) P.vline(e * h, { alpha: 0.5 });
    // bottom: log-log
    const es = curve.map((c) => c.e).filter((e) => e > 0);
    const emin = Math.max(1e-16, Math.min(...es)), emax = Math.max(...es);
    const Q = new Plot(s2, { xlim: [1 / 400, 1], ylim: [10 ** Math.floor(Math.log10(emin)), 10 ** Math.ceil(Math.log10(emax))], xlog: true, ylog: true, xlabel: 'mesh size h', ylabel: `${st.norm === 'L2' ? 'L²' : 'max'} error` });
    Q.frame();
    Q.line(curve.map((c) => c.h), curve.map((c) => c.e), { color: C[0] });
    Q.points(curve.map((c) => c.h), curve.map((c) => c.e), { color: C[0] });
    const cur = interpError(f, st.N, st.p)[st.norm];
    Q.points([1 / st.N], [cur], { color: C[1], r: 6, hollow: true });
    const smooth = st.fn === 'smooth' || st.fn === 'runge';
    const expect = st.norm === 'L2' ? (smooth ? st.p + 1 : st.fn === 'kink' ? Math.min(st.p + 1, 1.5) : 0.5) : (smooth ? st.p + 1 : st.fn === 'kink' ? 1 : 0);
    const k = curve.findIndex((c) => c.N === 16);
    if (expect > 0 && curve[k].e > 1e-14) Q.slopeTriangle(curve[k].h * 0.8, curve[k].e * 0.4, expect, 0.25);
    const valid = curve.filter((c) => c.e > 1e-13);
    const r = valid.length >= 3 ? fitRate(valid.map((c) => c.h), valid.map((c) => c.e), 3) : NaN;
    out.set(`error at N=${st.N}: ${fmt(cur)}\nobserved slope (last 3 points): ${Number.isFinite(r) ? r.toFixed(2) : 'n/a (round-off)'}\nexpected: ${expect}`);
    selfCheck('convergence errors finite', curve.every((c) => Number.isFinite(c.e)));
  }
  s1.onResize(draw); s2.onResize(draw);
  update();
}

initChapter(() => {
  mount('w-gradient', gradientWidget);
  mount('w-divthm', divThmWidget);
  mount('w-conv', convWidget);
});
