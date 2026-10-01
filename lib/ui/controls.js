/**
 * @file Small UI control factories (slider, select, checkbox, segmented
 *       buttons, buttons, readouts) used by all widgets.
 *
 * Every factory appends its element to `parent` and returns a handle
 * { el, get value(), set(v) }. Labels may contain TeX ($…$); it is rendered
 * when the chapter calls renderMath at the end of initialisation (or call
 * renderMath(el) yourself for controls created later).
 */
import { renderMath } from './chapter.js';

/** Create an element with class and optional HTML. */
export function h(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}

/** Debounce a function (ms); trailing call. */
export function debounce(fn, ms = 60) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

/** Format a number compactly for readouts. */
export function fmt(v, digits = 3) {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(digits - 1).replace('e+', 'e');
  return Number(v.toPrecision(digits)).toString();
}

/**
 * Slider. With `log: true` the slider moves linearly in log10(value).
 * @param {HTMLElement} parent
 * @param {{label: string, min: number, max: number, step?: number, value: number, log?: boolean,
 *          format?: (v:number)=>string, onInput?: (v:number)=>void, onChange?: (v:number)=>void}} o
 */
export function slider(parent, o) {
  const wrap = h('div', 'ctl');
  const head = h('div', 'ctl-head');
  const lab = h('span', 'ctl-label', o.label);
  const val = h('span', 'ctl-value');
  head.append(lab, val);
  const inp = document.createElement('input');
  inp.type = 'range';
  const toS = (v) => (o.log ? Math.log10(v) : v), fromS = (s) => (o.log ? 10 ** s : s);
  inp.min = String(toS(o.min)); inp.max = String(toS(o.max));
  inp.step = String(o.log ? (o.step ?? 0.01) : (o.step ?? (o.max - o.min) / 100));
  inp.value = String(toS(o.value));
  inp.setAttribute('aria-label', o.label.replace(/\$/g, ''));
  const format = o.format || ((v) => fmt(v));
  const read = () => {
    let v = fromS(parseFloat(inp.value));
    if (!o.log && o.step && Number.isInteger(o.step)) v = Math.round(v);
    return v;
  };
  const show = () => { val.textContent = format(read()); };
  inp.addEventListener('input', () => { show(); o.onInput?.(read()); });
  inp.addEventListener('change', () => o.onChange?.(read()));
  wrap.append(head, inp);
  parent.appendChild(wrap);
  show();
  return {
    el: wrap, input: inp,
    get value() { return read(); },
    set(v, fire = false) { inp.value = String(toS(v)); show(); if (fire) o.onInput?.(read()); },
  };
}

/**
 * Drop-down select.
 * @param {HTMLElement} parent
 * @param {{label: string, options: Array<{value: string, label: string}>, value: string, onChange?: (v:string)=>void}} o
 */
export function select(parent, o) {
  const wrap = h('div', 'ctl');
  wrap.appendChild(h('span', 'ctl-label', o.label));
  const s = document.createElement('select');
  for (const opt of o.options) {
    const e = document.createElement('option');
    e.value = opt.value; e.textContent = opt.label;
    s.appendChild(e);
  }
  s.value = o.value;
  s.setAttribute('aria-label', o.label.replace(/\$/g, ''));
  s.addEventListener('change', () => o.onChange?.(s.value));
  wrap.appendChild(s);
  parent.appendChild(wrap);
  return { el: wrap, get value() { return s.value; }, set(v, fire = false) { s.value = v; if (fire) o.onChange?.(v); } };
}

/**
 * Checkbox.
 * @param {HTMLElement} parent
 * @param {{label: string, value: boolean, onChange?: (v:boolean)=>void}} o
 */
export function checkbox(parent, o) {
  const wrap = h('label', 'ctl check');
  const c = document.createElement('input');
  c.type = 'checkbox'; c.checked = !!o.value;
  c.addEventListener('change', () => o.onChange?.(c.checked));
  wrap.append(c, h('span', 'ctl-label', o.label));
  parent.appendChild(wrap);
  return { el: wrap, get value() { return c.checked; }, set(v, fire = false) { c.checked = v; if (fire) o.onChange?.(v); } };
}

/**
 * Segmented button group (radio-like).
 * @param {HTMLElement} parent
 * @param {{label?: string, options: Array<{value: string, label: string}>, value: string, onChange?: (v:string)=>void}} o
 */
export function segmented(parent, o) {
  const wrap = h('div', 'ctl');
  if (o.label) wrap.appendChild(h('span', 'ctl-label', o.label));
  const seg = h('div', 'seg');
  let cur = o.value;
  const btns = o.options.map((opt) => {
    const b = h('button', opt.value === cur ? 'on' : '', opt.label);
    b.type = 'button';
    b.addEventListener('click', () => { set(opt.value, true); });
    seg.appendChild(b);
    return b;
  });
  function set(v, fire = false) {
    cur = v;
    btns.forEach((b, i) => b.classList.toggle('on', o.options[i].value === v));
    if (fire) o.onChange?.(v);
  }
  wrap.appendChild(seg);
  parent.appendChild(wrap);
  return { el: wrap, get value() { return cur; }, set };
}

/**
 * Button.
 * @param {HTMLElement} parent
 * @param {{label: string, primary?: boolean, onClick: () => void}} o
 */
export function button(parent, o) {
  const b = h('button', `btn${o.primary ? ' primary' : ''}`, o.label);
  b.type = 'button';
  b.addEventListener('click', o.onClick);
  parent.appendChild(b);
  return { el: b, setLabel(t) { b.innerHTML = t; } };
}

/** A row container for buttons. */
export function buttonRow(parent) {
  const r = h('div', 'btn-row');
  parent.appendChild(r);
  return r;
}

/**
 * Text readout (monospace). set(html) replaces its content and renders TeX.
 * @param {HTMLElement} parent
 */
export function readout(parent) {
  const r = h('div', 'readout');
  parent.appendChild(r);
  return { el: r, set(html) { r.innerHTML = html; if (html.includes('$')) renderMath(r); } };
}

/**
 * Standard widget skeleton inside a <figure class="widget">:
 *   title, body = [panels..., controls], caption kept from the HTML.
 * @param {HTMLElement} fig
 * @param {{title?: string}} [o]
 * @returns {{fig: HTMLElement, body: HTMLElement, controls: HTMLElement, panel: (label?: string) => HTMLElement}}
 */
export function widgetLayout(fig, o = {}) {
  const cap = fig.querySelector('figcaption');
  if (o.title) fig.insertBefore(h('div', 'widget-title', o.title), fig.firstChild);
  const body = h('div', 'widget-body');
  fig.insertBefore(body, cap || null);
  const controls = h('div', 'controls');
  const panel = (label, flex) => {
    const p = h('div', 'panel');
    if (flex) p.style.flex = flex;
    if (label) p.appendChild(h('div', 'canvas-label', label));
    body.insertBefore(p, controls);
    return p;
  };
  body.appendChild(controls);
  return { fig, body, controls, panel };
}
