/**
 * @file Drawing helpers for DG / FR widgets (chapters 6, 7):
 *  - piecewise (discontinuous) polynomials on a 1D mesh,
 *  - RK stability regions in the complex plane,
 *  - fast rasterisation of tensor-product (Q_p) fields on structured quad grids.
 */
import { theme } from './canvas.js';
import { stabilityAmp } from '../core/time/rk.js';
import { lagrangeValues } from '../core/basis/lagrange.js';
import { lut, putColor } from './colormap.js';

/**
 * Draw a discontinuous piecewise polynomial: one polyline per element.
 * @param {import('./plot1d.js').Plot} P
 * @param {{N: number, xf: ArrayLike<number>, np: number, x: ArrayLike<number>,
 *          evalRef: (u: Float64Array, k: number, xi: number) => number}} d discretisation (dg1d / fr1d)
 * @param {Float64Array} u nodal values
 * @param {{color?: string|((k:number)=>string), width?: number, samples?: number, nodes?: boolean, nodeR?: number, alpha?: number}} [o]
 */
export function drawPiecewise(P, d, u, o = {}) {
  const ns = o.samples ?? 16;
  const col = typeof o.color === 'function' ? o.color : () => o.color || theme().accent;
  for (let k = 0; k < d.N; k++) {
    const xs = [], ys = [];
    for (let s = 0; s <= ns; s++) {
      const xi = -1 + 2 * s / ns;
      xs.push(d.xf[k] + (xi + 1) * (d.xf[k + 1] - d.xf[k]) / 2);
      ys.push(d.evalRef(u, k, xi));
    }
    P.line(xs, ys, { color: col(k), width: o.width ?? 2, alpha: o.alpha ?? 1 });
    if (o.nodes) {
      const nx = [], ny = [];
      for (let i = 0; i < d.np; i++) { nx.push(d.x[k * d.np + i]); ny.push(u[k * d.np + i]); }
      P.points(nx, ny, { color: col(k), r: o.nodeR ?? 2.6 });
    }
  }
}

/** Two alternating colours for neighbouring elements (so the pieces are distinguishable). */
export function elementColors() {
  const T = theme();
  return (k) => (k % 2 === 0 ? T.accent : T.accent3);
}

/**
 * Shade the stability region {z : |R(z)| ≤ 1} of an RK method inside a Plot
 * (x axis = Re z, y axis = Im z) and outline its boundary.
 * @param {import('./plot1d.js').Plot} P
 * @param {string} method RK method name
 * @param {{step?: number, color?: string, alpha?: number}} [o] step = pixel block size
 */
export function shadeStabilityRegion(P, method, o = {}) {
  const { ctx } = P.surf, s = o.step ?? 3;
  ctx.save();
  P.clip();
  ctx.fillStyle = o.color || theme().accent2;
  ctx.globalAlpha = o.alpha ?? 0.16;
  const x0 = Math.floor(P.px0), x1 = Math.ceil(P.px1), y0 = Math.floor(P.py0), y1 = Math.ceil(P.py1);
  const nx = Math.ceil((x1 - x0) / s), ny = Math.ceil((y1 - y0) / s);
  const inside = new Uint8Array((nx + 1) * (ny + 1));
  for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
    const zr = P.invX(x0 + i * s), zi = P.invY(y0 + j * s);
    inside[j * (nx + 1) + i] = stabilityAmp(method, zr, zi) <= 1 ? 1 : 0;
  }
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (inside[j * (nx + 1) + i]) ctx.fillRect(x0 + i * s, y0 + j * s, s, s);
  // boundary: pixels whose neighbour differs
  ctx.globalAlpha = 0.9; ctx.fillStyle = o.color || theme().accent2;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const a = inside[j * (nx + 1) + i];
    if (a !== inside[j * (nx + 1) + i + 1] || a !== inside[(j + 1) * (nx + 1) + i]) ctx.fillRect(x0 + i * s + s / 2 - 1, y0 + j * s + s / 2 - 1, 2, 2);
  }
  ctx.restore(); // clip
  ctx.restore();
}

/**
 * Rasterise a Q_p field stored at tensor nodes on a structured grid (layout of dg2d.js:
 * u[e*np*np + j*np + i], e = ey*nx + ex) by evaluating the true polynomial at every pixel.
 * Lagrange weights are cached per pixel column and row (the grid is a tensor product too).
 * @param {{ctx: CanvasRenderingContext2D, w: number, h: number}} surf
 * @param {import('./canvas.js').View2D} view
 * @param {{nx: number, ny: number, np: number, box: number[], r: ArrayLike<number>}} d
 * @param {Float64Array} u
 * @param {{cmap?: string, lo: number, hi: number}} o
 */
export function drawQpField(surf, view, d, u, o) {
  const w = Math.max(1, Math.round(surf.w)), h = Math.max(1, Math.round(surf.h));
  if (!surf._qp || surf._qp.w !== w || surf._qp.h !== h || surf._qp.key !== `${d.nx},${d.ny},${d.np},${d.box}`) {
    const off = document.createElement('canvas');
    off.width = w; off.height = h;
    const octx = off.getContext('2d');
    const np = d.np, [x0, x1, y0, y1] = d.box, hx = (x1 - x0) / d.nx, hy = (y1 - y0) / d.ny;
    const colE = new Int32Array(w).fill(-1), colW = new Float64Array(w * np);
    const rowE = new Int32Array(h).fill(-1), rowW = new Float64Array(h * np), tmp = new Float64Array(np);
    for (let px = 0; px < w; px++) {
      const x = view.invX(px + 0.5);
      if (x < x0 || x > x1) continue;
      const i = Math.min(d.nx - 1, Math.floor((x - x0) / hx));
      colE[px] = i; lagrangeValues(d.r, 2 * ((x - x0) / hx - i) - 1, tmp); colW.set(tmp, px * np);
    }
    for (let py = 0; py < h; py++) {
      const y = view.invY(py + 0.5);
      if (y < y0 || y > y1) continue;
      const j = Math.min(d.ny - 1, Math.floor((y - y0) / hy));
      rowE[py] = j; lagrangeValues(d.r, 2 * ((y - y0) / hy - j) - 1, tmp); rowW.set(tmp, py * np);
    }
    surf._qp = { w, h, key: `${d.nx},${d.ny},${d.np},${d.box}`, off, octx, img: octx.createImageData(w, h), colE, colW, rowE, rowW };
  }
  const C = surf._qp, L = lut(o.cmap || 'viridis'), dat = C.img.data, np = d.np, nn = np * np;
  dat.fill(0);
  for (let py = 0; py < h; py++) {
    const j = C.rowE[py];
    if (j < 0) continue;
    for (let px = 0; px < w; px++) {
      const i = C.colE[px];
      if (i < 0) continue;
      const base = (j * d.nx + i) * nn;
      // value = Σ_b ly_b Σ_a lx_a u[b, a]
      let s = 0;
      for (let b = 0; b < np; b++) {
        let r = 0;
        for (let a = 0; a < np; a++) r += C.colW[px * np + a] * u[base + b * np + a];
        s += C.rowW[py * np + b] * r;
      }
      putColor(dat, 4 * (py * w + px), L, s, o.lo, o.hi);
    }
  }
  C.octx.putImageData(C.img, 0, 0);
  surf.ctx.save();
  surf.ctx.imageSmoothingEnabled = false;
  surf.ctx.drawImage(C.off, 0, 0, surf.w, surf.h);
  surf.ctx.restore();
}
