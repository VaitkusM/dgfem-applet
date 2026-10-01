/**
 * @file Quadrature on triangles cut by an implicit boundary Γ = {φ = 0}.
 *
 * Unfitted methods integrate over K ∩ Ω (the part of a background triangle
 * K inside the domain Ω = {φ < 0}) and along Γ ∩ K. We approximate both by
 * the classic "linear cut":
 *
 *  1. Replace φ on a triangle by its linear interpolant φ_h through the three
 *     vertex values (exactly what a rasteriser does with vertex attributes).
 *  2. The zero set of a linear function is a straight line, so Γ_h ∩ K is a
 *     SEGMENT whose end points lie on the edges where φ changes sign, at
 *         P = A + t (B − A),   t = φ_A / (φ_A − φ_B).
 *     (Marching triangles — the triangle version of marching squares.)
 *  3. The inside part {φ_h < 0} ∩ K is a triangle (one vertex inside) or a
 *     quadrilateral (two vertices inside), which we split into two triangles.
 *     Standard triangle rules on these sub-triangles integrate polynomials
 *     EXACTLY over Ω_h ∩ K; Gauss rules on the segment integrate over Γ_h.
 *  4. The interface normal is n_h = ∇φ_h / |∇φ_h| (constant on the segment);
 *     it points from inside to outside, i.e. it is the OUTWARD normal of Ω_h.
 *
 * Geometric error: Γ_h is the piecewise-linear interpolant of a smooth curve,
 * so the area error is O(h²) per unit boundary length (a chord deviates from
 * an arc of curvature κ by ≈ κ h²/8) and the length error is O(h²) as well.
 *
 * Recursive refinement (`levels` > 0): before the linear cut, a triangle that
 * might be cut is split into 4 congruent children (edge midpoints, true φ
 * evaluated at the new vertices), recursively `levels` times; the linear cut
 * is applied on the finest children. This reduces the geometric error by
 * ≈ 4^{−levels} without changing the background mesh. A triangle is declared
 * uncut (fully inside or outside) without subdivision when all vertex values
 * have the same sign and min |φ| > lip · diam(T) — for an SDF (lip = 1) this
 * is a proof that Γ does not touch the triangle.
 *
 * Curved (high-order) variant (`curved: shape`): when the shape provides a
 * parametrisation X(t) of Γ and the parameter of a boundary point (both
 * shapes of levelset.js do: t = polar angle), the leaf triangles are cut
 * along the TRUE curve instead of the chord:
 *   - the edge crossings P*, Q* are the exact roots of φ on the edges
 *     (safeguarded secant iteration), with parameters t_P, t_Q;
 *   - the inside part is decomposed into straight triangles and "curved
 *     triangles" {A + r (C(τ) − A) : r, τ ∈ [0,1]} with apex A at an inside
 *     vertex and curved side C(τ) = X(t_P + τ (t_Q − t_P));
 *   - such a piece is integrated by Gauss rules in r and τ with the signed
 *     Jacobian r (C(τ) − A) × C'(τ) (sign fixed by the orientation of the
 *     piece), which is exact up to quadrature error — the O(h²) geometric
 *     error of the chord disappears.
 * It is used by the WEB-spline solver, whose higher-order rates would
 * otherwise be destroyed by the O(h²) error of the linear cut.
 *
 * Zero vertex values are treated as "outside" (φ ≥ 0), which makes the
 * classification of every point unique.
 */
import { triangleRule } from './simplex.js';
import { gaussForDegree01 } from './gauss1d.js';

/**
 * @typedef {Object} CutData
 * @property {number[]} tris  inside sub-triangles, 6 numbers each (ax, ay, bx, by, cx, cy), CCW
 * @property {number[]} segs  interface segments, 6 numbers each (px, py, qx, qy, nx, ny),
 *                            (nx, ny) = outward unit normal of Ω_h (curved mode: the chords)
 * @property {number[]} curved curved pieces (curved mode only), 4 numbers each (ax, ay, t0, t1):
 *                            apex A and parameter range of the curved side
 * @property {number[]} arcs  arcs of Γ (curved mode only), 2 numbers each (t0, t1)
 * @property {number} area    |K ∩ Ω_h|
 * @property {number} length  |Γ_h ∩ K|
 */

const sub = (o, a, b, s) => [a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])];
const triAreaPts = (a, b, c) => 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]));

/** Push triangle (a,b,c) with CCW orientation. Degenerate triangles are dropped. */
function pushTri(out, a, b, c) {
  const A = triAreaPts(a, b, c);
  if (A === 0) return;
  if (A > 0) out.tris.push(a[0], a[1], b[0], b[1], c[0], c[1]);
  else out.tris.push(a[0], a[1], c[0], c[1], b[0], b[1]);
  out.area += Math.abs(A);
}

/**
 * Cut of one triangle V = [A, B, C] with vertex values f = [fa, fb, fc].
 * Without `shape`: linear cut of φ_h. With `shape` (curved mode): exact edge
 * crossings and curved pieces along Γ. Appends to `out` (see CutData).
 * @param {number[][]} V
 * @param {number[]} f
 * @param {CutData} out
 * @param {object} [shape] level-set shape with point/tangent/param (curved mode)
 */
export function linearCut(V, f, out, shape) {
  const ins = f.map((v) => v < 0);
  const nIn = ins[0] + ins[1] + ins[2];
  if (nIn === 3) { pushTri(out, V[0], V[1], V[2]); return; }
  if (nIn === 0) return;
  // rotate so that vertex 0 is the "odd one out" (alone on its side)
  const lone = nIn === 1 ? ins.indexOf(true) : ins.indexOf(false);
  const i0 = lone, i1 = (lone + 1) % 3, i2 = (lone + 2) % 3;
  const A = V[i0], B = V[i1], C = V[i2], fa = f[i0], fb = f[i1], fc = f[i2];
  let P = sub(null, A, B, fa / (fa - fb)); // on edge A–B
  let Q = sub(null, A, C, fa / (fa - fc)); // on edge A–C
  if (shape) {
    P = edgeRoot(shape.phi, A, B, fa, fb); Q = edgeRoot(shape.phi, A, C, fa, fc);
    const tP = shape.param(P[0], P[1]);
    let dt = shape.param(Q[0], Q[1]) - tP;
    dt -= 2 * Math.PI * Math.round(dt / (2 * Math.PI)); // shortest arc
    if (nIn === 1) pushCurved(out, A, tP, tP + dt, shape);           // curved triangle A, P*→Q*
    else { pushTri(out, B, C, Q); pushCurved(out, B, tP + dt, tP, shape); } // B C Q* + curved B, Q*→P*
    out.arcs.push(tP, tP + dt);
  } else if (nIn === 1) pushTri(out, A, P, Q);       // inside = corner triangle at A
  else { pushTri(out, P, B, C); pushTri(out, P, C, Q); } // inside = quad P B C Q
  // ∇φ_h of the linear interpolant on (A,B,C): solve J^T g = (fb − fa, fc − fa)
  const e1x = B[0] - A[0], e1y = B[1] - A[1], e2x = C[0] - A[0], e2y = C[1] - A[1];
  const det = e1x * e2y - e1y * e2x;
  const db = fb - fa, dc = fc - fa;
  let gx = (db * e2y - dc * e1y) / det, gy = (dc * e1x - db * e2x) / det;
  const gn = Math.hypot(gx, gy);
  const len = Math.hypot(Q[0] - P[0], Q[1] - P[1]);
  if (len === 0 || gn === 0) return;
  gx /= gn; gy /= gn;
  out.segs.push(P[0], P[1], Q[0], Q[1], gx, gy);
  out.length += len;
}

/** Root of φ on the segment A→B (φ(A) = fa < 0 ≤ fb = φ(B)): Illinois (safeguarded secant) iteration. */
function edgeRoot(phi, A, B, fa, fb) {
  let a = 0, b = 1, ga = fa, gb = fb, side = 0;
  if (gb === 0) return [B[0], B[1]];
  let s = ga / (ga - gb);
  for (let it = 0; it < 60; it++) {
    s = (a * gb - b * ga) / (gb - ga);
    const gs = phi(A[0] + s * (B[0] - A[0]), A[1] + s * (B[1] - A[1]));
    if (gs === 0 || b - a < 1e-15) break;
    if ((gs < 0) === (ga < 0)) { a = s; ga = gs; if (side === -1) gb /= 2; side = -1; }
    else { b = s; gb = gs; if (side === 1) ga /= 2; side = 1; }
    if (Math.abs(gs) < 1e-15) break;
  }
  return [A[0] + s * (B[0] - A[0]), A[1] + s * (B[1] - A[1])];
}

/** Gauss rule (r, τ) for a curved triangle {A + r (C(τ) − A)}; returns points and weights. */
function curvedRule(ax, ay, t0, t1, shape, deg) {
  const Gr = gaussForDegree01(deg + 1), Gt = gaussForDegree01(deg + 6);
  const n = Gr.x.length * Gt.x.length;
  // orientation of the loop A → C(0) → C(1) → A; the SIGNED Jacobian times this sign makes the
  // rule exact (up to quadrature error) even if the piece is not star-shaped w.r.t. A
  const [p0x, p0y] = shape.point(t0), [p1x, p1y] = shape.point(t1);
  const sgn = Math.sign((p0x - ax) * (p1y - ay) - (p0y - ay) * (p1x - ax)) || 1;
  const x = new Float64Array(n), y = new Float64Array(n), w = new Float64Array(n);
  let q = 0;
  for (let j = 0; j < Gt.x.length; j++) {
    const t = t0 + Gt.x[j] * (t1 - t0), [cx, cy] = shape.point(t), [tx, ty] = shape.tangent(t);
    const cr = sgn * ((cx - ax) * ty - (cy - ay) * tx) * (t1 - t0); // ±(C − A) × C'(τ)
    for (let i = 0; i < Gr.x.length; i++, q++) {
      const r = Gr.x[i];
      x[q] = ax + r * (cx - ax); y[q] = ay + r * (cy - ay);
      w[q] = Gr.w[i] * Gt.w[j] * r * cr;
    }
  }
  return { x, y, w, n };
}

function pushCurved(out, A, t0, t1, shape) {
  out.curved.push(A[0], A[1], t0, t1);
  const R = curvedRule(A[0], A[1], t0, t1, shape, 2);
  for (let q = 0; q < R.n; q++) out.area += R.w[q];
}

/**
 * Cut a triangle (vertices a, b, c, CCW) by the zero level set of phi.
 * @param {(x:number, y:number) => number} phi
 * @param {number[]} a [x, y]
 * @param {number[]} b
 * @param {number[]} c
 * @param {{levels?: number, lip?: number, values?: number[], curved?: object}} [o]
 *   levels: recursive refinement depth (0 = plain linear cut); lip: Lipschitz bound of φ used by
 *   the "certainly uncut" test; values: precomputed φ at a, b, c; curved: shape (with point, tangent,
 *   param) for the curved cut at the leaves
 * @returns {CutData}
 */
export function cutTriangle(phi, a, b, c, o = {}) {
  const levels = o.levels ?? 0, lip = o.lip ?? 1, shape = o.curved || null;
  const out = { tris: [], segs: [], curved: [], arcs: [], area: 0, length: 0, shape };
  const f0 = o.values || [phi(a[0], a[1]), phi(b[0], b[1]), phi(c[0], c[1])];
  const rec = (V, f, lev) => {
    if (lev >= levels) { linearCut(V, f, out, shape); return; }
    const same = (f[0] < 0) === (f[1] < 0) && (f[1] < 0) === (f[2] < 0);
    if (same) {
      const diam = Math.max(Math.hypot(V[1][0] - V[0][0], V[1][1] - V[0][1]),
        Math.hypot(V[2][0] - V[1][0], V[2][1] - V[1][1]), Math.hypot(V[0][0] - V[2][0], V[0][1] - V[2][1]));
      if (Math.min(Math.abs(f[0]), Math.abs(f[1]), Math.abs(f[2])) > lip * diam) {
        if (f[0] < 0) pushTri(out, V[0], V[1], V[2]);
        return;
      }
    }
    // 4 children through the edge midpoints m01, m12, m20
    const m = [[0, 1], [1, 2], [2, 0]].map(([i, j]) => [(V[i][0] + V[j][0]) / 2, (V[i][1] + V[j][1]) / 2]);
    const fm = m.map((p) => phi(p[0], p[1]));
    rec([V[0], m[0], m[2]], [f[0], fm[0], fm[2]], lev + 1);
    rec([m[0], V[1], m[1]], [fm[0], f[1], fm[1]], lev + 1);
    rec([m[2], m[1], V[2]], [fm[2], fm[1], f[2]], lev + 1);
    rec([m[0], m[1], m[2]], [fm[0], fm[1], fm[2]], lev + 1);
  };
  rec([a, b, c], f0, 0);
  return out;
}

/**
 * Volume quadrature on K ∩ Ω_h: a triangle rule of degree `deg` mapped to every
 * inside sub-triangle (plus the Gauss rules of the curved pieces in curved mode).
 * @param {CutData} cut
 * @param {number} deg polynomial degree integrated exactly (on Ω_h)
 * @returns {{x: Float64Array, y: Float64Array, w: Float64Array, n: number}}
 */
export function cutVolumeRule(cut, deg) {
  const R = triangleRule(deg), nt = cut.tris.length / 6, nc = (cut.curved || []).length / 4;
  const CR = [];
  for (let c = 0; c < nc; c++) CR.push(curvedRule(...cut.curved.slice(4 * c, 4 * c + 4), cut.shape, deg));
  const n = nt * R.n + CR.reduce((s, r) => s + r.n, 0);
  const x = new Float64Array(n), y = new Float64Array(n), w = new Float64Array(n);
  let q = 0;
  for (const r of CR) { x.set(r.x, q); y.set(r.y, q); w.set(r.w, q); q += r.n; }
  for (let t = 0; t < nt; t++) {
    const [ax, ay, bx, by, cx, cy] = cut.tris.slice(6 * t, 6 * t + 6);
    const e1x = bx - ax, e1y = by - ay, e2x = cx - ax, e2y = cy - ay;
    const det = Math.abs(e1x * e2y - e1y * e2x);
    for (let k = 0; k < R.n; k++, q++) {
      x[q] = ax + e1x * R.x[k] + e2x * R.y[k];
      y[q] = ay + e1y * R.x[k] + e2y * R.y[k];
      w[q] = R.w[k] * det;
    }
  }
  return { x, y, w, n };
}

/**
 * Interface quadrature on Γ_h ∩ K: Gauss rule exact for degree `deg` on every segment.
 * @param {CutData} cut
 * @param {number} deg
 * @returns {{x: Float64Array, y: Float64Array, w: Float64Array, nx: Float64Array, ny: Float64Array, n: number}}
 */
export function cutInterfaceRule(cut, deg) {
  const G = gaussForDegree01(deg), ns = cut.segs.length / 6, n = ns * G.x.length;
  const x = new Float64Array(n), y = new Float64Array(n), w = new Float64Array(n);
  const nx = new Float64Array(n), ny = new Float64Array(n);
  let q = 0;
  for (let s = 0; s < ns; s++) {
    const [px, py, qx, qy, mx, my] = cut.segs.slice(6 * s, 6 * s + 6);
    const L = Math.hypot(qx - px, qy - py);
    for (let k = 0; k < G.x.length; k++, q++) {
      x[q] = px + G.x[k] * (qx - px); y[q] = py + G.x[k] * (qy - py);
      w[q] = G.w[k] * L; nx[q] = mx; ny[q] = my;
    }
  }
  return { x, y, w, nx, ny, n };
}

/**
 * Quadrature on the true arcs of Γ ∩ K (curved mode): Gauss in the curve
 * parameter with weight |X'(t)| dt; normals = outward normals of Γ.
 * @param {CutData} cut
 * @param {number} n Gauss points per arc
 * @returns {{x: Float64Array, y: Float64Array, w: Float64Array, nx: Float64Array, ny: Float64Array, n: number}}
 */
export function cutArcRule(cut, n = 8) {
  const G = gaussForDegree01(2 * n - 1), na = cut.arcs.length / 2, N = na * G.x.length;
  const x = new Float64Array(N), y = new Float64Array(N), w = new Float64Array(N), nx = new Float64Array(N), ny = new Float64Array(N);
  let q = 0;
  for (let a = 0; a < na; a++) {
    const t0 = cut.arcs[2 * a], t1 = cut.arcs[2 * a + 1];
    for (let k = 0; k < G.x.length; k++, q++) {
      const t = t0 + G.x[k] * (t1 - t0), [px, py] = cut.shape.point(t), [tx, ty] = cut.shape.tangent(t), L = Math.hypot(tx, ty);
      x[q] = px; y[q] = py; w[q] = G.w[k] * L * Math.abs(t1 - t0); nx[q] = ty / L; ny[q] = -tx / L;
    }
  }
  return { x, y, w, nx, ny, n: N };
}

/** Element classes produced by classifyMesh. */
export const OUTSIDE = 0, INSIDE = 1, CUT = 2;

/**
 * Cut every triangle of a mesh and classify it.
 *   INSIDE : K ⊂ Ω_h (no interface inside K)
 *   OUTSIDE: K ∩ Ω_h = ∅
 *   CUT    : otherwise (some interface in K); `frac` = |K ∩ Ω_h| / |K| is the
 *            volume fraction — "small cuts" are cut elements with tiny frac.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {(x:number,y:number)=>number} phi
 * @param {{levels?: number, lip?: number, curved?: object}} [o] see cutTriangle
 * @returns {{cls: Uint8Array, frac: Float64Array, cuts: (CutData|null)[], phiNodes: Float64Array}}
 *   cuts[t] is filled for CUT elements only
 */
export function classifyMesh(mesh, phi, o = {}) {
  const { nodes, tris } = mesh, nT = tris.length / 3, nV = nodes.length / 2;
  const phiNodes = new Float64Array(nV);
  for (let v = 0; v < nV; v++) phiNodes[v] = phi(nodes[2 * v], nodes[2 * v + 1]);
  const cls = new Uint8Array(nT), frac = new Float64Array(nT), cuts = new Array(nT).fill(null);
  for (let t = 0; t < nT; t++) {
    const V = [0, 1, 2].map((k) => { const v = tris[3 * t + k]; return [nodes[2 * v], nodes[2 * v + 1]]; });
    const f = [0, 1, 2].map((k) => phiNodes[tris[3 * t + k]]);
    const area = Math.abs(triAreaPts(V[0], V[1], V[2]));
    // fast path for the plain linear cut: equal signs ⇒ uncut
    if ((o.levels ?? 0) === 0 && !o.curved && (f[0] < 0) === (f[1] < 0) && (f[1] < 0) === (f[2] < 0)) {
      cls[t] = f[0] < 0 ? INSIDE : OUTSIDE; frac[t] = f[0] < 0 ? 1 : 0; continue;
    }
    const cut = cutTriangle(phi, V[0], V[1], V[2], { levels: o.levels, lip: o.lip, values: f, curved: o.curved });
    if (cut.segs.length === 0) {
      const full = cut.area > 0.5 * area;
      cls[t] = full ? INSIDE : OUTSIDE; frac[t] = full ? 1 : 0;
    } else { cls[t] = CUT; frac[t] = cut.area / area; cuts[t] = cut; }
  }
  return { cls, frac, cuts, phiNodes };
}
