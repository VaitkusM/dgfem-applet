/**
 * @file 2D line/scatter plots with axes (linear or logarithmic), used for
 *       1D solutions, convergence plots, residual histories, spectra, …
 *
 * Usage:
 *   const P = new Plot(surface, { xlim: [0, 1], ylim: [-1, 1], xlabel: 'x' });
 *   P.frame();                       // clears + draws axes, ticks, grid
 *   P.line(xs, ys, { color });       // polyline in data coordinates
 *   P.legend([{ label: 'exact', color }]);
 */
import { theme, arrow } from './canvas.js';

/** "Nice" tick values covering [a,b] (≈ n ticks). */
export function niceTicks(a, b, n = 5) {
  const span = b - a;
  if (!(span > 0)) return [a];
  const step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0)), r = step0 / mag;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
  const t = [];
  for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + step * 1e-9; v += step) t.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return t;
}

const fmtTick = (v) => {
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e4)) return v.toExponential(0).replace('e+', 'e');
  return (+v.toPrecision(4)).toString();
};

export class Plot {
  /**
   * @param {{ctx: CanvasRenderingContext2D, w: number, h: number}} surf
   * @param {{xlim: number[], ylim: number[], xlog?: boolean, ylog?: boolean, xlabel?: string, ylabel?: string,
   *          margin?: number[], grid?: boolean, equal?: boolean, title?: string}} o margin = [left, right, top, bottom]
   */
  constructor(surf, o) {
    this.surf = surf;
    this.o = { grid: true, ...o };
    this.margin = o.margin || [o.ylabel ? 58 : 46, 12, o.title ? 22 : 10, o.xlabel ? 34 : 22];
    this.setLimits(o.xlim, o.ylim);
  }
  /** Change the data window. */
  setLimits(xlim, ylim) {
    this.xlim = xlim; this.ylim = ylim;
    const { w, h } = this.surf, [ml, mr, mt, mb] = this.margin;
    this.px0 = ml; this.px1 = w - mr; this.py0 = mt; this.py1 = h - mb;
    if (this.o.equal) {
      const sx = (this.px1 - this.px0) / (this.tx(xlim[1]) - this.tx(xlim[0]));
      const sy = (this.py1 - this.py0) / (this.ty(ylim[1]) - this.ty(ylim[0]));
      const s = Math.min(sx, sy);
      const cx = (this.px0 + this.px1) / 2, cy = (this.py0 + this.py1) / 2;
      const hw = s * (this.tx(xlim[1]) - this.tx(xlim[0])) / 2, hh = s * (this.ty(ylim[1]) - this.ty(ylim[0])) / 2;
      this.px0 = cx - hw; this.px1 = cx + hw; this.py0 = cy - hh; this.py1 = cy + hh;
    }
  }
  tx(x) { return this.o.xlog ? Math.log10(x) : x; }
  ty(y) { return this.o.ylog ? Math.log10(y) : y; }
  /** data x → screen x */
  X(x) { const a = this.tx(this.xlim[0]), b = this.tx(this.xlim[1]); return this.px0 + (this.tx(x) - a) / (b - a) * (this.px1 - this.px0); }
  /** data y → screen y */
  Y(y) { const a = this.ty(this.ylim[0]), b = this.ty(this.ylim[1]); return this.py1 - (this.ty(y) - a) / (b - a) * (this.py1 - this.py0); }
  /** screen → data */
  invX(px) { const a = this.tx(this.xlim[0]), b = this.tx(this.xlim[1]); const t = a + (px - this.px0) / (this.px1 - this.px0) * (b - a); return this.o.xlog ? 10 ** t : t; }
  invY(py) { const a = this.ty(this.ylim[0]), b = this.ty(this.ylim[1]); const t = a + (this.py1 - py) / (this.py1 - this.py0) * (b - a); return this.o.ylog ? 10 ** t : t; }

  _ticks(lim, log) {
    if (!log) return niceTicks(lim[0], lim[1], 5).map((v) => ({ v, major: true }));
    const t = [], a = Math.floor(Math.log10(lim[0])), b = Math.ceil(Math.log10(lim[1]));
    const every = Math.max(1, Math.ceil((b - a) / 6));
    for (let e = a; e <= b; e++) {
      if (10 ** e >= lim[0] * 0.999 && 10 ** e <= lim[1] * 1.001) t.push({ v: 10 ** e, major: (e - a) % every === 0 });
      if (b - a <= 3) for (const m of [2, 5]) { const v = m * 10 ** e; if (v > lim[0] && v < lim[1]) t.push({ v, major: false, minorLabel: b - a <= 2 }); }
    }
    return t;
  }

  /** Clear and draw background, grid, axes, ticks and labels. */
  frame() {
    const { ctx, w, h } = this.surf, T = theme();
    ctx.save();
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, w, h);
    ctx.font = `11px ${T.ui}`;
    ctx.lineWidth = 1;
    const xt = this._ticks(this.xlim, this.o.xlog), yt = this._ticks(this.ylim, this.o.ylog);
    ctx.strokeStyle = T.rule;
    if (this.o.grid) {
      ctx.beginPath();
      for (const t of xt) { const x = Math.round(this.X(t.v)) + 0.5; ctx.moveTo(x, this.py0); ctx.lineTo(x, this.py1); }
      for (const t of yt) { const y = Math.round(this.Y(t.v)) + 0.5; ctx.moveTo(this.px0, y); ctx.lineTo(this.px1, y); }
      ctx.stroke();
    }
    ctx.strokeStyle = T.faint;
    ctx.strokeRect(Math.round(this.px0) + 0.5, Math.round(this.py0) + 0.5, Math.round(this.px1 - this.px0), Math.round(this.py1 - this.py0));
    ctx.fillStyle = T.soft;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const t of xt) if (t.major || t.minorLabel) ctx.fillText(fmtTick(t.v), this.X(t.v), this.py1 + 4);
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const t of yt) if (t.major || t.minorLabel) ctx.fillText(fmtTick(t.v), this.px0 - 4, this.Y(t.v));
    ctx.font = `600 11.5px ${T.ui}`;
    if (this.o.xlabel) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(this.o.xlabel, (this.px0 + this.px1) / 2, h - 1); }
    if (this.o.ylabel) {
      ctx.save(); ctx.translate(11, (this.py0 + this.py1) / 2); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(this.o.ylabel, 0, 0); ctx.restore();
    }
    if (this.o.title) { ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = T.ink; ctx.fillText(this.o.title, this.px0, 4); }
    ctx.restore();
  }

  /** Clip subsequent drawing to the plot area; call ctx.restore() after. */
  clip() {
    const { ctx } = this.surf;
    ctx.save();
    ctx.beginPath(); ctx.rect(this.px0, this.py0, this.px1 - this.px0, this.py1 - this.py0); ctx.clip();
  }

  /**
   * Polyline (skips non-finite points, breaking the line there).
   * @param {ArrayLike<number>} xs
   * @param {ArrayLike<number>} ys
   * @param {{color?: string, width?: number, dash?: number[], alpha?: number}} [o]
   */
  line(xs, ys, o = {}) {
    const { ctx } = this.surf;
    this.clip();
    ctx.strokeStyle = o.color || theme().accent; ctx.lineWidth = o.width ?? 1.8;
    ctx.globalAlpha = o.alpha ?? 1; ctx.setLineDash(o.dash || []); ctx.lineJoin = 'round';
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < xs.length; i++) {
      const ok = Number.isFinite(xs[i]) && Number.isFinite(ys[i]) && (!this.o.ylog || ys[i] > 0) && (!this.o.xlog || xs[i] > 0);
      if (!ok) { pen = false; continue; }
      const X = this.X(xs[i]), Y = this.Y(ys[i]);
      if (pen) ctx.lineTo(X, Y); else { ctx.moveTo(X, Y); pen = true; }
    }
    ctx.stroke();
    ctx.restore();
  }

  /** Several disconnected polylines, e.g. a discontinuous piecewise polynomial. */
  segments(list, o = {}) { for (const [xs, ys] of list) this.line(xs, ys, o); }

  /** Markers. */
  points(xs, ys, o = {}) {
    const { ctx } = this.surf, r = o.r ?? 3;
    this.clip();
    ctx.fillStyle = o.color || theme().accent; ctx.strokeStyle = o.stroke || o.color || theme().accent;
    ctx.globalAlpha = o.alpha ?? 1; ctx.lineWidth = 1.2;
    for (let i = 0; i < xs.length; i++) {
      if (!Number.isFinite(ys[i]) || (this.o.ylog && ys[i] <= 0)) continue;
      ctx.beginPath(); ctx.arc(this.X(xs[i]), this.Y(ys[i]), r, 0, 2 * Math.PI);
      if (o.hollow) ctx.stroke(); else ctx.fill();
    }
    ctx.restore();
  }

  /**
   * Piecewise-constant bars: value vals[i] on [edges[i], edges[i+1]].
   * @param {ArrayLike<number>} edges length n+1
   * @param {ArrayLike<number>} vals length n
   */
  bars(edges, vals, o = {}) {
    const { ctx } = this.surf, T = theme();
    this.clip();
    const base = this.Y(o.base ?? Math.max(this.ylim[0], Math.min(0, this.ylim[1])));
    ctx.globalAlpha = o.alpha ?? 0.35; ctx.fillStyle = o.color || T.accent;
    for (let i = 0; i < vals.length; i++) {
      const x0 = this.X(edges[i]), x1 = this.X(edges[i + 1]), y = this.Y(vals[i]);
      ctx.fillRect(x0, Math.min(y, base), x1 - x0, Math.abs(base - y));
    }
    ctx.globalAlpha = 1; ctx.strokeStyle = o.color || T.accent; ctx.lineWidth = o.width ?? 1.6;
    ctx.beginPath();
    for (let i = 0; i < vals.length; i++) {
      const y = this.Y(vals[i]);
      ctx.moveTo(this.X(edges[i]), y); ctx.lineTo(this.X(edges[i + 1]), y);
    }
    ctx.stroke();
    ctx.restore();
  }

  /** Horizontal / vertical reference lines. */
  hline(y, o = {}) { this.line([this.xlim[0], this.xlim[1]], [y, y], { color: theme().faint, width: 1, dash: [4, 4], ...o }); }
  vline(x, o = {}) { this.line([x, x], [this.ylim[0], this.ylim[1]], { color: theme().faint, width: 1, dash: [4, 4], ...o }); }

  /** Text at data coordinates. */
  text(x, y, s, o = {}) {
    const { ctx } = this.surf, T = theme();
    ctx.save();
    ctx.fillStyle = o.color || T.ink; ctx.font = o.font || `12px ${T.ui}`;
    ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.baseline || 'middle';
    ctx.fillText(s, this.X(x) + (o.dx || 0), this.Y(y) + (o.dy || 0));
    ctx.restore();
  }

  /** Arrow between data points. */
  arrow(x0, y0, x1, y1, o = {}) {
    const { ctx } = this.surf;
    ctx.save();
    ctx.strokeStyle = ctx.fillStyle = o.color || theme().ink; ctx.lineWidth = o.width ?? 1.5;
    arrow(ctx, this.X(x0), this.Y(y0), this.X(x1), this.Y(y1), o.head ?? 7);
    ctx.restore();
  }

  /**
   * Legend box in a corner.
   * @param {Array<{label: string, color: string, dash?: number[], marker?: boolean}>} items
   * @param {'tl'|'tr'|'bl'|'br'} [corner='tr']
   */
  legend(items, corner = 'tr') {
    const { ctx } = this.surf, T = theme();
    ctx.save();
    ctx.font = `11.5px ${T.ui}`;
    const lw = Math.max(...items.map((i) => ctx.measureText(i.label).width)) + 34, lh = 16, H = items.length * lh + 8;
    const x = corner[1] === 'r' ? this.px1 - lw - 6 : this.px0 + 6;
    const y = corner[0] === 't' ? this.py0 + 6 : this.py1 - H - 6;
    ctx.globalAlpha = 0.88; ctx.fillStyle = T.elev; ctx.fillRect(x, y, lw, H);
    ctx.globalAlpha = 1; ctx.strokeStyle = T.rule; ctx.strokeRect(x + 0.5, y + 0.5, lw, H);
    items.forEach((it, i) => {
      const yy = y + 4 + lh * i + lh / 2;
      ctx.strokeStyle = it.color; ctx.fillStyle = it.color; ctx.lineWidth = 2; ctx.setLineDash(it.dash || []);
      if (it.marker) { ctx.beginPath(); ctx.arc(x + 15, yy, 3.5, 0, 2 * Math.PI); ctx.fill(); }
      else { ctx.beginPath(); ctx.moveTo(x + 6, yy); ctx.lineTo(x + 24, yy); ctx.stroke(); }
      ctx.setLineDash([]); ctx.fillStyle = T.ink; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      ctx.fillText(it.label, x + 29, yy);
    });
    ctx.restore();
  }

  /**
   * Slope triangle for log–log convergence plots: a right triangle with
   * horizontal leg from h to h·f (f<1) and vertical leg for slope `s`.
   * @param {number} h starting abscissa (data)
   * @param {number} e ordinate at that point (data)
   * @param {number} s slope
   * @param {number} [f=0.5]
   */
  slopeTriangle(h, e, s, f = 0.5, o = {}) {
    const h2 = h * f, e2 = e * f ** s;
    this.line([h, h2, h, h], [e, e2, e2, e], { color: o.color || theme().soft, width: 1.2 });
    this.text(h * Math.sqrt(f) * 1.0, e2, `slope ${s}`, { baseline: 'top', align: 'center', dy: 3, font: `11px ${theme().ui}`, color: o.color || theme().soft });
  }
}
