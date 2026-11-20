/**
 * @file Canvas helpers: HiDPI-aware canvases that track their CSS size,
 *       a world→screen 2D view transform, theme colours and pointer input.
 *
 * Accessibility built in here (so that every widget gets it for free):
 *  - createCanvas gives the <canvas> role="img" and an accessible name taken from the panel
 *    label above it (or `o.label`), points aria-describedby at the figure caption and puts a
 *    one-sentence fallback text inside the element;
 *  - onPointer makes the canvas focusable and adds a keyboard "virtual pointer": arrow keys move
 *    a cursor, Space/Enter presses or releases it — the widget's own down/move/up/hover
 *    callbacks are called exactly as for a mouse, so no widget code needs to know about it;
 *  - canvases repaint (their onResize callbacks run) when the OS colour scheme changes.
 */

/** All surfaces created on this page (for repainting on a colour-scheme change). */
const surfaces = [];
let schemeWatch = false;
let uid = 0;

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
    onAccent: g('--on-accent') || '#fff', // text colour that is readable ON the accent colours (≥ 4.5:1)
    mono: g('--font-mono') || 'monospace', ui: g('--font-ui') || 'sans-serif',
  };
}

/** Categorical series colours (theme-aware). @returns {string[]} */
export function seriesColors() {
  const t = theme();
  return [t.accent, t.accent2, t.accent3, t.accent4, '#d4a017', '#e0457b', t.soft];
}

/**
 * Give a canvas an accessible name and description (role="img").
 * Name: `o.label`, else the `.canvas-label` element of its panel (aria-labelledby, so rendered
 * math is read correctly), else the bold lead-in of the figure caption. Description: the
 * figure caption (aria-describedby) and a fallback sentence inside the element (`o.description`).
 * @param {HTMLCanvasElement} canvas
 * @param {HTMLElement} parent the panel the canvas lives in
 * @param {{label?: string, description?: string}} [o]
 */
function describeCanvas(canvas, parent, o = {}) {
  canvas.setAttribute('role', 'img');
  const lab = parent.querySelector?.(':scope > .canvas-label');
  const fig = parent.closest?.('figure');
  const cap = fig?.querySelector('figcaption');
  if (o.label) canvas.setAttribute('aria-label', o.label);
  else if (lab) { if (!lab.id) lab.id = `canvas-label-${++uid}`; canvas.setAttribute('aria-labelledby', lab.id); }
  else canvas.setAttribute('aria-label', cap?.querySelector('b')?.textContent?.trim() || 'Figure');
  if (cap) { if (!cap.id) cap.id = `figcaption-${++uid}`; canvas.setAttribute('aria-describedby', cap.id); }
  canvas.textContent = o.description || 'Graphical figure drawn by script; what it shows is explained in the caption and in the read-out next to it.';
}

/**
 * A canvas that keeps its backing store at devicePixelRatio and re-draws on resize
 * (and when the OS switches between light and dark mode).
 * @param {HTMLElement} parent
 * @param {{aspect?: number, height?: number, onResize?: () => void, label?: string, description?: string}} [o]
 *   aspect = height/width (default 0.6) or fixed CSS height; label / description: accessible
 *   name and fallback text (default: the panel's `.canvas-label` and a generic sentence)
 * @returns {{canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D, w: number, h: number, dpr: number, onResize: (cb: () => void) => void}}
 */
export function createCanvas(parent, o = {}) {
  const wrap = document.createElement('div');
  wrap.className = 'canvas-wrap';
  const canvas = document.createElement('canvas');
  wrap.appendChild(canvas);
  parent.appendChild(wrap);
  describeCanvas(canvas, parent, o);
  const ctx = canvas.getContext('2d');
  const surf = { canvas, ctx, w: 300, h: 180, dpr: 1, cbs: [] };
  surf.onResize = (cb) => surf.cbs.push(cb);
  if (o.onResize) surf.cbs.push(o.onResize);
  const fit = () => {
    const w = Math.max(50, wrap.clientWidth || 300);
    const h = o.height ?? Math.round(w * (o.aspect ?? 0.6));
    const dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
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
  surfaces.push(surf);
  if (!schemeWatch && globalThis.matchMedia) {
    schemeWatch = true;
    // theme() is read at draw time, so repainting is all it takes to follow a light/dark switch
    globalThis.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
      for (const s of surfaces) for (const cb of s.cbs) { try { cb(); } catch (err) { console.error(err); } }
    });
  }
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
 *
 * Besides mouse/touch/pen it provides a KEYBOARD alternative with the same callbacks: the canvas
 * becomes focusable, and
 *   arrow keys      move a virtual pointer (drawn as a ring) by 8 px, with Shift by 1 px;
 *                   while "pressed" this calls move(x, y), otherwise hover(x, y);
 *   Space / Enter   press: down(x, y) — if it does not return false the pointer stays pressed
 *                   (a drag) until Space/Enter/Escape is hit again, which calls up();
 *   n / p           (only if `handles` is given) jump to the next / previous handle.
 * The virtual pointer starts at the last real pointer position, else at the first handle, else at
 * the centre. Callbacks receive a plain object {pointerType: 'keyboard', key, shiftKey, …} in place of the PointerEvent.
 *
 * Touch: only canvases that use this helper disable browser panning (`touch-action: none`);
 * all other canvases let a swipe scroll the page.
 * @param {HTMLCanvasElement} canvas
 * @param {{down?: (x:number,y:number,e:PointerEvent)=>boolean|void, move?: (x:number,y:number,e:PointerEvent)=>void, up?: (e:PointerEvent)=>void, hover?: (x:number,y:number)=>void,
 *          handles?: () => Array<[number, number]>, keyboard?: boolean}} h
 *   handles: optional list of grabbable points in canvas CSS pixels (keyboard users can jump to them);
 *   keyboard: false disables the keyboard alternative
 * @returns {{position: () => [number, number]}} position(): current virtual-pointer position (CSS px)
 */
export function onPointer(canvas, h) {
  let dragging = false;
  let last = null; // last pointer position (real or virtual), CSS px
  const pos = (e) => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  canvas.style.touchAction = 'none'; // we handle drags ourselves; other canvases keep normal page scrolling
  canvas.addEventListener('pointerdown', (e) => {
    const [x, y] = pos(e);
    last = [x, y]; hideCursor();
    if (h.down && h.down(x, y, e) === false) return;
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    const [x, y] = pos(e);
    if (dragging) { last = [x, y]; h.move?.(x, y, e); } else h.hover?.(x, y);
  });
  const end = (e) => { if (dragging) { dragging = false; h.up?.(e); } };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);

  // ---- keyboard alternative: a virtual pointer driven through the same callbacks ----
  let pressed = false, ring = null, hint = null, hi = -1;
  function hideCursor() { if (ring) ring.hidden = true; if (hint) hint.hidden = true; }
  const api = { position: () => (last ? [last[0], last[1]] : [canvas.clientWidth / 2, canvas.clientHeight / 2]) };
  if (h.keyboard === false) return api;
  const HELP = `Keyboard: arrow keys move the pointer (Shift = fine), Space or Enter presses / releases${h.handles ? ', N / P jump between handles' : ''}.`;
  canvas.tabIndex = 0;
  canvas.setAttribute('role', 'application');
  canvas.setAttribute('aria-roledescription', 'interactive figure');
  canvas.textContent = `${canvas.textContent ? `${canvas.textContent} ` : ''}${HELP}`;
  const fake = (e) => {
    const r = canvas.getBoundingClientRect();
    return { type: 'keyboard', pointerType: 'keyboard', pointerId: -1, button: 0, buttons: pressed ? 1 : 0, key: e?.key,
      shiftKey: !!e?.shiftKey, ctrlKey: !!e?.ctrlKey, altKey: !!e?.altKey, metaKey: !!e?.metaKey,
      clientX: r.left + last[0], clientY: r.top + last[1], target: canvas, preventDefault() {}, stopPropagation() {} };
  };
  const showCursor = () => {
    const wrap = canvas.parentElement;
    if (!wrap) return;
    if (!ring) {
      ring = document.createElement('div'); ring.className = 'kbd-cursor'; ring.setAttribute('aria-hidden', 'true');
      hint = document.createElement('div'); hint.className = 'kbd-hint'; hint.setAttribute('aria-hidden', 'true'); hint.textContent = HELP;
      if (getComputedStyle(wrap).position === 'static') wrap.style.position = 'relative';
      wrap.append(ring, hint);
    }
    ring.hidden = false; hint.hidden = false;
    ring.classList.toggle('pressed', pressed);
    ring.style.left = `${canvas.offsetLeft + last[0]}px`; ring.style.top = `${canvas.offsetTop + last[1]}px`;
  };
  const release = (e) => { if (pressed) { pressed = false; h.up?.(fake(e)); } };
  const moveTo = (x, y, e) => {
    last = [Math.max(0, Math.min(canvas.clientWidth, x)), Math.max(0, Math.min(canvas.clientHeight, y))];
    if (pressed) h.move?.(last[0], last[1], fake(e)); else h.hover?.(last[0], last[1]);
  };
  canvas.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const step = e.shiftKey ? 1 : 8;
    const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
    const pts = h.handles ? h.handles() : null;
    if (!last) last = pts && pts.length ? [pts[0][0], pts[0][1]] : api.position();
    if (d) moveTo(last[0] + d[0], last[1] + d[1], e);
    else if (e.key === ' ' || e.key === 'Enter') {
      if (pressed) release(e);
      else if (!(h.down && h.down(last[0], last[1], fake(e)) === false)) { pressed = true; if (!h.move) release(e); }
    } else if (e.key === 'Escape' && pressed) release(e);
    else if (pts && pts.length && (e.key === 'n' || e.key === 'p' || e.key === 'N' || e.key === 'P')) {
      release(e);
      hi = (hi + (e.key.toLowerCase() === 'n' ? 1 : pts.length - 1)) % pts.length;
      if (hi < 0) hi = 0;
      moveTo(pts[hi][0], pts[hi][1], e);
    } else return;
    e.preventDefault(); // arrows / space must not scroll the page while the figure has focus
    showCursor();
  });
  canvas.addEventListener('blur', () => { release(null); hideCursor(); });
  return api;
}

/**
 * Draw an arrow from (x0,y0) to (x1,y1) in screen coordinates (uses the current stroke/fill style).
 * @param {CanvasRenderingContext2D} ctx @param {number} x0 @param {number} y0 @param {number} x1 @param {number} y1
 * @param {number} [head=6] head length in px
 */
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

/**
 * Clear a surface with the theme background.
 * @param {{ctx: CanvasRenderingContext2D, w: number, h: number}} surf @param {string} [color]
 */
export function clear(surf, color) {
  const { ctx, w, h } = surf;
  ctx.save();
  ctx.fillStyle = color || theme().bg;
  ctx.fillRect(0, 0, w, h);
  ctx.restore();
}
