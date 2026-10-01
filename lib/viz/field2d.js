/**
 * @file Rendering of scalar fields on 2D meshes by per-pixel evaluation —
 *       a tiny software rasteriser.
 *
 * Because DG/FEM solutions are polynomials per element (possibly
 * discontinuous across edges), we do not interpolate vertex colours like a
 * GPU would; we evaluate the true polynomial at every pixel centre:
 *  - triangles: loop over each triangle's bounding box, compute barycentric
 *    coordinates (λ0, λ1, λ2) of the pixel centre, and if all are ≥ 0 call
 *    evalFn(t, λ0, λ1, λ2). Note the reference coordinates are x̂ = λ1, ŷ = λ2.
 *  - structured quads: every pixel finds its cell (i,j) and local coordinates
 *    (ξ, η) ∈ [−1,1]² directly.
 * Values are mapped through a colormap into an ImageData buffer at CSS-pixel
 * resolution and blitted with drawImage.
 */
import { lut, putColor } from './colormap.js';

/** Off-screen buffer sized like the surface (CSS px). */
function makeBuffer(surf) {
  const w = Math.max(1, Math.round(surf.w)), h = Math.max(1, Math.round(surf.h));
  if (!surf._off || surf._off.width !== w || surf._off.height !== h) {
    surf._off = document.createElement('canvas');
    surf._off.width = w; surf._off.height = h;
    surf._offCtx = surf._off.getContext('2d');
    surf._img = surf._offCtx.createImageData(w, h);
  }
  surf._img.data.fill(0);
  return { w, h, img: surf._img };
}

function blit(surf) {
  surf._offCtx.putImageData(surf._img, 0, 0);
  surf.ctx.save();
  surf.ctx.imageSmoothingEnabled = false;
  surf.ctx.drawImage(surf._off, 0, 0, surf.w, surf.h);
  surf.ctx.restore();
}

/**
 * Rasterise a scalar field defined per triangle.
 * @param {{ctx: CanvasRenderingContext2D, w: number, h: number}} surf
 * @param {import('./canvas.js').View2D} view world→screen transform
 * @param {Float64Array} nodes
 * @param {Int32Array} tris
 * @param {(t: number, l0: number, l1: number, l2: number, x: number, y: number) => number} evalFn
 * @param {{cmap?: string, lo: number, hi: number, skip?: (t: number) => boolean}} o
 */
export function drawTriField(surf, view, nodes, tris, evalFn, o) {
  const { w, h, img } = makeBuffer(surf);
  const L = lut(o.cmap || 'viridis'), d = img.data;
  const nT = tris.length / 3;
  for (let t = 0; t < nT; t++) {
    if (o.skip && o.skip(t)) continue;
    const a = tris[3 * t], b = tris[3 * t + 1], c = tris[3 * t + 2];
    const ax = view.X(nodes[2 * a]), ay = view.Y(nodes[2 * a + 1]);
    const bx = view.X(nodes[2 * b]), by = view.Y(nodes[2 * b + 1]);
    const cx = view.X(nodes[2 * c]), cy = view.Y(nodes[2 * c + 1]);
    const den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    if (Math.abs(den) < 1e-12) continue;
    const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx))), x1 = Math.min(w - 1, Math.ceil(Math.max(ax, bx, cx)));
    const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy))), y1 = Math.min(h - 1, Math.ceil(Math.max(ay, by, cy)));
    const eps = -1e-9;
    for (let py = y0; py <= y1; py++) {
      const Y = py + 0.5;
      for (let px = x0; px <= x1; px++) {
        const X = px + 0.5;
        const l0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den;
        const l1 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / den;
        const l2 = 1 - l0 - l1;
        if (l0 < eps || l1 < eps || l2 < eps) continue;
        const wx = view.invX(X), wy = view.invY(Y);
        putColor(d, 4 * (py * w + px), L, evalFn(t, l0, l1, l2, wx, wy), o.lo, o.hi);
      }
    }
  }
  blit(surf);
}

/**
 * Rasterise a field on a structured nx×ny quad grid over `box`.
 * evalFn(e, ξ, η, x, y) with element e = j*nx + i and ξ, η ∈ [−1, 1].
 * @param {{ctx: CanvasRenderingContext2D, w: number, h: number}} surf
 * @param {import('./canvas.js').View2D} view
 * @param {number} nx
 * @param {number} ny
 * @param {number[]} box
 * @param {(e: number, xi: number, eta: number, x: number, y: number) => number} evalFn
 * @param {{cmap?: string, lo: number, hi: number}} o
 */
export function drawQuadField(surf, view, nx, ny, box, evalFn, o) {
  const { w, h, img } = makeBuffer(surf);
  const L = lut(o.cmap || 'viridis'), d = img.data;
  const [x0, x1, y0, y1] = box, hx = (x1 - x0) / nx, hy = (y1 - y0) / ny;
  const pxa = Math.max(0, Math.floor(view.X(x0))), pxb = Math.min(w - 1, Math.ceil(view.X(x1)));
  const pya = Math.max(0, Math.floor(view.Y(y1))), pyb = Math.min(h - 1, Math.ceil(view.Y(y0)));
  for (let py = pya; py <= pyb; py++) {
    const y = view.invY(py + 0.5);
    if (y < y0 || y > y1) continue;
    const j = Math.min(ny - 1, Math.floor((y - y0) / hy));
    const eta = 2 * ((y - y0) / hy - j) - 1;
    for (let px = pxa; px <= pxb; px++) {
      const x = view.invX(px + 0.5);
      if (x < x0 || x > x1) continue;
      const i = Math.min(nx - 1, Math.floor((x - x0) / hx));
      const xi = 2 * ((x - x0) / hx - i) - 1;
      putColor(d, 4 * (py * w + px), L, evalFn(j * nx + i, xi, eta, x, y), o.lo, o.hi);
    }
  }
  blit(surf);
}

/**
 * Rasterise an arbitrary function f(x,y) over the view's world box.
 * @param {{ctx: CanvasRenderingContext2D, w: number, h: number}} surf
 * @param {import('./canvas.js').View2D} view
 * @param {(x: number, y: number) => number} f NaN = transparent
 * @param {{cmap?: string, lo: number, hi: number, step?: number}} o step = pixel block size (speed)
 */
export function drawFunction(surf, view, f, o) {
  const { w, h, img } = makeBuffer(surf);
  const L = lut(o.cmap || 'viridis'), d = img.data, s = o.step || 1;
  for (let py = 0; py < h; py += s) {
    for (let px = 0; px < w; px += s) {
      const v = f(view.invX(px + s / 2), view.invY(py + s / 2));
      for (let qy = py; qy < Math.min(h, py + s); qy++)
        for (let qx = px; qx < Math.min(w, px + s); qx++) putColor(d, 4 * (qy * w + qx), L, v, o.lo, o.hi);
    }
  }
  blit(surf);
}
