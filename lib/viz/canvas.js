/**
 * @file Canvas helpers: HiDPI-aware canvases that track their CSS size,
 *       a world→screen 2D view transform, theme colours and pointer input.
 */

/**
 * Read the current theme colours from CSS custom properties (so canvases
 * follow light/dark mode).
 * @returns {Record<string, string>}
 */
export function theme() {
  const cs = getComputedStyle(document.documentElement);
  const g = (n) => cs.getPropertyValue(n).trim();
  return {
    bg: g('--bg'), elev: g('--bg-elev'), sunk: g('--bg-sunk'), ink: g('--ink'), soft: g('--ink-soft'), faint: g('--ink-faint'),
    rule: g('--rule'), accent: g('--accent'), accent2: g('--accent-2'), accent3: g('--accent-3'), accent4: g('--accent-4'),
    mono: g('--font-mono') || 'monospace', ui: g('--font-ui') || 'sans-serif',
  };
}

/** Categorical series colours (theme-aware). */
export function seriesColors() {
  const t = theme();
  return [t.accent, t.accent2, t.accent3, t.accent4, '#d4a017', '#e0457b', t.soft];
}

/**
 * A canvas that keeps its backing store at devicePixelRatio and re-draws on resize.
 * @param {HTMLElement} parent
 * @param {{aspect?: number, height?: number, onResize?: () => void}} [o]
 *   aspect = height/width (default 0.6) or fixed CSS height
 * @returns {{canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D, w: number, h: number, dpr: number, onResize: (cb: () => void) => void}}
 */
export function createCanvas(parent, o = {}) {
  const wrap = document.createElement('div');
  wrap.className = 'canvas-wrap';
  const canvas = document.createElement('canvas');
  wrap.appendChild(canvas);
  parent.appendChild(wrap);
  const ctx = canvas.getContext('2d');
  const surf = { canvas, ctx, w: 300, h: 180, dpr: 1, cbs: [] };
  surf.onResize = (cb) => surf.cbs.push(cb);
  if (o.onResize) surf.cbs.push(o.onResize);
  const fit = () => {
    const w = Math.max(50, wrap.clientWidth || 300);
    const h = o.height ?? Math.round(w * (o.aspect ?? 0.6));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (w === surf.w && h === surf.h && dpr === surf.dpr && canvas.width) return false;
    surf.w = w; surf.h = h; surf.dpr = dpr;
    canvas.style.height = `${h}px`;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return true;
  };
  fit();
  let raf = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => { if (fit()) surf.cbs.forEach((cb) => cb()); });
  }).observe(wrap);
  return surf;
}

/**
 * World-to-screen transform for a rectangle [x0,x1]×[y0,y1] (y up), with
 * padding, optionally preserving aspect ratio (equal scales in x and y).
 */
export class View2D {
  /**
   * @param {{w: number, h: number}} surf
   * @param {number[]} box [x0, x1, y0, y1]
   * @param {{pad?: number|number[], equal?: boolean}} [o] pad = [left, right, top, bottom]
   */
  constructor(surf, box, o = {}) {
    const p = Array.isArray(o.pad) ? o.pad : [o.pad ?? 8, o.pad ?? 8, o.pad ?? 8, o.pad ?? 8];
    const [x0, x1, y0, y1] = box;
    let sx = (surf.w - p[0] - p[1]) / (x1 - x0), sy = (surf.h - p[2] - p[3]) / (y1 - y0);
    let ox = p[0], oy = p[2];
    if (o.equal) {
      const s = Math.min(sx, sy);
      ox += ((surf.w - p[0] - p[1]) - s * (x1 - x0)) / 2;
      oy += ((surf.h - p[2] - p[3]) - s * (y1 - y0)) / 2;
      sx = sy = s;
    }
    Object.assign(this, { x0, x1, y0, y1, sx, sy, ox, oy });
  }
  /** screen x of world x */ X(x) { return this.ox + (x - this.x0) * this.sx; }
  /** screen y of world y (y axis points up) */ Y(y) { return this.oy + (this.y1 - y) * this.sy; }
  /** world x of screen x */ invX(px) { return this.x0 + (px - this.ox) / this.sx; }
  /** world y of screen y */ invY(py) { return this.y1 - (py - this.oy) / this.sy; }
}

/**
 * Pointer interaction helper: reports drag events in canvas CSS pixels.
 * @param {HTMLCanvasElement} canvas
 * @param {{down?: (x:number,y:number,e:PointerEvent)=>boolean|void, move?: (x:number,y:number,e:PointerEvent)=>void, up?: (e:PointerEvent)=>void, hover?: (x:number,y:number)=>void}} h
 */
export function onPointer(canvas, h) {
  let dragging = false;
  const pos = (e) => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  canvas.addEventListener('pointerdown', (e) => {
    const [x, y] = pos(e);
    if (h.down && h.down(x, y, e) === false) return;
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    const [x, y] = pos(e);
    if (dragging) h.move?.(x, y, e); else h.hover?.(x, y);
  });
  const end = (e) => { if (dragging) { dragging = false; h.up?.(e); } };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
}

/** Draw an arrow from (x0,y0) to (x1,y1) in screen coordinates. */
export function arrow(ctx, x0, y0, x1, y1, head = 6) {
  const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy);
  if (L < 1e-9) return;
  const ux = dx / L, uy = dy / L, hh = Math.min(head, 0.45 * L);
  ctx.beginPath();
  ctx.moveTo(x0, y0); ctx.lineTo(x1 - ux * hh * 0.6, y1 - uy * hh * 0.6);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - ux * hh - uy * hh * 0.5, y1 - uy * hh + ux * hh * 0.5);
  ctx.lineTo(x1 - ux * hh + uy * hh * 0.5, y1 - uy * hh - ux * hh * 0.5);
  ctx.closePath();
  ctx.fill();
}

/** Clear a surface with the theme background. */
export function clear(surf, color) {
  const { ctx, w, h } = surf;
  ctx.save();
  ctx.fillStyle = color || theme().bg;
  ctx.fillRect(0, 0, w, h);
  ctx.restore();
}
