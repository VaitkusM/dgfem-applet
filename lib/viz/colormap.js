/**
 * @file Colormaps (scalar → RGB) for field plots, plus a colour bar.
 *
 *  - viridis, magma: perceptually uniform sequential maps (matplotlib anchors)
 *  - rdbu: diverging blue–white–red (ColorBrewer RdBu, reversed so that
 *    low = blue, high = red) — use it with a symmetric range for signed data
 *    such as errors.
 * Maps are piecewise-linear interpolations between anchor colours, sampled
 * into 256-entry lookup tables.
 */

const ANCHORS = {
  viridis: ['#440154', '#472d7b', '#3b528b', '#2c728e', '#21918c', '#28ae80', '#5ec962', '#addc30', '#fde725'],
  magma: ['#000004', '#1c1044', '#4f127b', '#812581', '#b5367a', '#e55964', '#fb8761', '#fec287', '#fcfdbf'],
  rdbu: ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'],
  greys: ['#ffffff', '#000000'],
};

const hex = (s) => [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16), parseInt(s.slice(5, 7), 16)];
const LUTS = {};

/**
 * 256×3 Uint8 lookup table for a named map.
 * @param {string} name
 * @returns {Uint8Array}
 */
export function lut(name) {
  if (LUTS[name]) return LUTS[name];
  const a = ANCHORS[name].map(hex), n = a.length - 1, L = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * n, k = Math.min(n - 1, Math.floor(t)), f = t - k;
    for (let c = 0; c < 3; c++) L[3 * i + c] = Math.round(a[k][c] * (1 - f) + a[k + 1][c] * f);
  }
  LUTS[name] = L;
  return L;
}

/**
 * Colour of value v in [lo, hi] as an "rgb(...)" string.
 * @param {string} name
 * @param {number} v
 * @param {number} lo
 * @param {number} hi
 */
export function colorOf(name, v, lo, hi) {
  const L = lut(name);
  let t = (v - lo) / (hi - lo || 1);
  t = Math.max(0, Math.min(1, t));
  const i = Math.round(t * 255);
  return `rgb(${L[3 * i]},${L[3 * i + 1]},${L[3 * i + 2]})`;
}

/**
 * Write the colour of v into a pixel buffer at byte offset o.
 * NaN → transparent.
 */
export function putColor(data, o, L, v, lo, hi) {
  if (!Number.isFinite(v)) { data[o + 3] = 0; return; }
  let t = (v - lo) / (hi - lo || 1);
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const i = 3 * Math.round(t * 255);
  data[o] = L[i]; data[o + 1] = L[i + 1]; data[o + 2] = L[i + 2]; data[o + 3] = 255;
}

/**
 * Draw a vertical colour bar with min/max labels.
 * @param {CanvasRenderingContext2D} ctx
 * @param {string} name
 * @param {number} x left (CSS px)
 * @param {number} y top
 * @param {number} w
 * @param {number} h
 * @param {number} lo
 * @param {number} hi
 * @param {{ink?: string, font?: string, fmt?: (v:number)=>string}} [o]
 */
export function colorbar(ctx, name, x, y, w, h, lo, hi, o = {}) {
  const L = lut(name);
  for (let i = 0; i < h; i++) {
    const k = 3 * Math.round((1 - i / (h - 1)) * 255);
    ctx.fillStyle = `rgb(${L[k]},${L[k + 1]},${L[k + 2]})`;
    ctx.fillRect(x, y + i, w, 1.5);
  }
  ctx.strokeStyle = o.ink || '#888';
  ctx.lineWidth = 0.5;
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = o.ink || '#444';
  ctx.font = o.font || '11px system-ui, sans-serif';
  ctx.textAlign = 'left';
  const f = o.fmt || ((v) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(1) : (+v.toPrecision(3)).toString()));
  ctx.textBaseline = 'top'; ctx.fillText(f(hi), x + w + 4, y);
  ctx.textBaseline = 'bottom'; ctx.fillText(f(lo), x + w + 4, y + h);
}

/**
 * Min and max of finite entries.
 * @param {ArrayLike<number>} a
 * @returns {[number, number]}
 */
export function range(a) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < a.length; i++) { const v = a[i]; if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
  if (lo === Infinity) return [0, 1];
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  return [lo, hi];
}
