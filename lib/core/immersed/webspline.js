/**
 * @file WEB-splines (weighted extended B-splines) of Höllig, Reif & Wipper,
 *       "Weighted extended B-spline approximation of Dirichlet problems",
 *       SIAM J. Numer. Anal. 39 (2001), for −Δu = f in Ω, u = g on Γ.
 *
 * Ingredients (all on a uniform grid of width h covering Ω, tensor B-splines
 * b_k of degree n from lib/core/basis/bspline.js):
 *
 * 1. WEIGHT FUNCTION w: smooth, w > 0 in Ω, w = 0 on Γ, growing linearly
 *    near Γ (|∇w| ≠ 0 there). Multiplying by w enforces the homogeneous
 *    Dirichlet condition EXACTLY (strongly) without fitting the mesh:
 *      - circle (centre c, radius R): w = (R² − |x − c|²)/(2R)  (≈ R − r near Γ);
 *      - general φ (used for the flower): w = (δ/3)(1 − (1 − t)³) for
 *        t = −φ/δ ∈ [0,1], w = δ/3 for t ≥ 1 (deep inside). This is Höllig's
 *        regularised-distance construction; it is C² and w ≈ −φ near Γ.
 *        δ is chosen below min R(θ) so that the non-smooth point of φ (the
 *        flower centre) lies where w is constant.
 *
 * 2. CLASSIFICATION of grid cells (inside: Q ⊂ Ω; cut; outside) and of
 *    B-splines: b_k is RELEVANT if its support meets Ω; INNER if its support
 *    contains at least one grid cell entirely inside Ω; OUTER = relevant
 *    but not inner. Outer B-splines have only a small part of their support
 *    in Ω — keeping them as unknowns would give a badly conditioned system.
 *
 * 3. EXTENSION: every outer index j is coupled to the (n+1)×(n+1) array
 *    I(j) = ℓ + {0,…,n}² of INNER indices closest to j (we choose the array
 *    whose centre ℓ + (n/2, n/2) has the smallest Euclidean distance to j in
 *    index space; ties broken by the scan order ℓ2, then ℓ1 increasing), with
 *        e_{i,j} = Π_{ν=1}^{2} Π_{μ=0, μ ≠ i_ν − ℓ_ν}^{n} (j_ν − ℓ_ν − μ) / (i_ν − ℓ_ν − μ),  i ∈ I(j).
 *    These are the values at j of the tensor Lagrange polynomials for the
 *    nodes I(j): Σ_i e_{i,j} p(i) = p(j) for every polynomial p of
 *    coordinate degree ≤ n. Since the B-spline coefficients of a
 *    polynomial are polynomials in the index (Marsden), polynomials of
 *    degree ≤ n stay in the span of the EXTENDED B-splines
 *        B^e_i = b_i + Σ_{j ∈ J(i)} e_{i,j} b_j,   J(i) = {outer j : i ∈ I(j)}.
 *
 * 4. WEB-splines: B_i = (w / w(x_i)) B^e_i for inner i, where x_i is the
 *    centre of an inside grid cell of supp b_i (we take the one with the
 *    largest w). The scaling 1/w(x_i) only improves conditioning.
 *
 * 5. NON-HOMOGENEOUS DIRICHLET DATA: choose any smooth extension g̃ of g
 *    (g̃ = g on Γ) and seek u_h = g̃ + Σ_i c_i B_i. Since every B_i vanishes
 *    on Γ, u_h = g on Γ exactly. Galerkin: for all inner i,
 *        Σ_k c_k ∫_Ω ∇B_k·∇B_i = ∫_Ω f B_i − ∫_Ω ∇g̃·∇B_i.
 *    (Remark: if g̃ happened to be the exact solution the remainder would be
 *    zero; the tests use g̃ = (1 − w)·G with G a smooth function equal to g
 *    on Γ, so that the remainder (u − g̃)/w is non-trivial.)
 *
 * Integrals over Ω ∩ Q are computed with tensor Gauss rules on inside cells
 * and with the curved cut quadrature of lib/core/quad/cut.js on cut cells
 * (each cell split into two triangles, one refinement level, then the cut
 * along the true curve). With the plain linear cut the O(h²) geometric error
 * would cap every WEB-spline at second order.
 */
import { bsplineGrid, bsplineCellValues } from '../basis/bspline.js';
import { cutTriangle, cutVolumeRule } from '../quad/cut.js';
import { gaussLegendre } from '../quad/gauss1d.js';
import { SparseBuilder } from '../la/sparse.js';
import { sparseLU } from '../la/direct.js';

/**
 * Weight function for a shape from levelset.js.
 * @param {object} ls shape (circle or flower)
 * @param {{delta?: number}} [o] δ for the regularised-distance weight (default 0.4 min R(θ))
 * @returns {{w: (x:number,y:number)=>number, grad: (x:number,y:number)=>[number,number], tex: string}}
 */
export function weightFunction(ls, o = {}) {
  if (ls.kind === 'circle') {
    const { cx, cy, R } = ls;
    return {
      w: (x, y) => (R * R - (x - cx) ** 2 - (y - cy) ** 2) / (2 * R),
      grad: (x, y) => [-(x - cx) / R, -(y - cy) / R],
      tex: 'w = (R^2 - |\\mathbf x - \\mathbf c|^2)/(2R)',
    };
  }
  const delta = o.delta ?? 0.4 * ls.R * (1 - (ls.a ?? 0));
  return {
    w(x, y) {
      const t = -ls.phi(x, y) / delta;
      return t >= 1 ? delta / 3 : (delta / 3) * (1 - (1 - t) ** 3);
    },
    grad(x, y) {
      const t = -ls.phi(x, y) / delta;
      if (t >= 1) return [0, 0];
      const [gx, gy] = ls.grad(x, y), c = (1 - t) ** 2; // ∇w = −(1 − t)² ∇φ
      return [-c * gx, -c * gy];
    },
    delta,
    tex: 'w = \\tfrac{\\delta}{3}\\big(1 - (1 - t)_+^3\\big),\\ t = -\\phi/\\delta',
  };
}

/** Cell classes. */
export const CELL_OUT = 0, CELL_IN = 1, CELL_CUT = 2;
/** B-spline classes. */
export const BS_IRRELEVANT = 0, BS_INNER = 1, BS_OUTER = 2;

/**
 * Build the WEB-spline space: cell + B-spline classification, extension
 * coefficients, normalisation points.
 * @param {object} ls shape (levelset.js)
 * @param {{n: number, N: number, box?: number[], levels?: number, curved?: boolean, weight?: object, extend?: boolean}} o
 *   extend: false keeps ALL relevant B-splines as unknowns (no extension; to demonstrate the instability)
 *   levels: refinement depth of the cut quadrature (default 1); curved: integrate cut cells along the true
 *   curve (default true when the shape has a parametrisation), see lib/core/quad/cut.js
 * @returns {object} space with fields:
 *   grid (bsplineGrid), cellCls (Uint8Array N²), cellCut (cut data per cut cell or null),
 *   bsCls (Uint8Array nB), inner (Int32Array of grid indices), innerId (Int32Array nB → compact id or −1),
 *   relevant (Int32Array), relId (Int32Array nB → compact id or −1),
 *   ext: Map outer index → {ell: [l1,l2], ids: Int32Array (grid indices of I(j)), e: Float64Array},
 *   wx: Float64Array (w(x_i) per inner compact id), xi: Float64Array (2 per inner: x_i), weight, ls
 */
export function webSpace(ls, o) {
  const { n, N } = o, box = o.box || [0, 1, 0, 1], levels = o.levels ?? 1;
  const curved = o.curved === false || !ls.param ? null : ls;
  const grid = bsplineGrid(n, N, box), { h, x0, y0, M } = grid;
  const weight = o.weight || weightFunction(ls);
  const lip = ls.isSDF ? 1 : 2;
  // --- cells
  const cellCls = new Uint8Array(N * N), cellCut = new Array(N * N).fill(null);
  for (let l2 = 0; l2 < N; l2++) for (let l1 = 0; l1 < N; l1++) {
    const xa = x0 + l1 * h, ya = y0 + l2 * h, xb = xa + h, yb = ya + h;
    const P = [[xa, ya], [xb, ya], [xb, yb], [xa, yb]];
    const c1 = cutTriangle(ls.phi, P[0], P[1], P[2], { levels, lip, curved });
    const c2 = cutTriangle(ls.phi, P[0], P[2], P[3], { levels, lip, curved });
    const area = c1.area + c2.area, c = l1 + N * l2;
    if (c1.segs.length + c2.segs.length === 0) cellCls[c] = area > 0.5 * h * h ? CELL_IN : CELL_OUT;
    else if (area > 0) {
      cellCls[c] = CELL_CUT;
      cellCut[c] = { tris: [...c1.tris, ...c2.tris], segs: [...c1.segs, ...c2.segs], curved: [...c1.curved, ...c2.curved],
        arcs: [...c1.arcs, ...c2.arcs], area, length: c1.length + c2.length, shape: c1.shape };
    }
  }
  // --- B-splines: support cells k … k+n (clipped to the grid)
  const bsCls = new Uint8Array(M * M);
  for (let k2 = -n; k2 < N; k2++) for (let k1 = -n; k1 < N; k1++) {
    let rel = false, inn = false;
    for (let a = Math.max(0, k2); a <= Math.min(N - 1, k2 + n); a++)
      for (let b = Math.max(0, k1); b <= Math.min(N - 1, k1 + n); b++) {
        const c = cellCls[b + N * a];
        if (c !== CELL_OUT) rel = true;
        if (c === CELL_IN) inn = true;
      }
    // extend === false: no extension, every relevant B-spline becomes an unknown (for comparison only)
    bsCls[grid.index(k1, k2)] = inn || (rel && o.extend === false) ? BS_INNER : rel ? BS_OUTER : BS_IRRELEVANT;
  }
  const inner = [], relevant = [];
  const innerId = new Int32Array(M * M).fill(-1), relId = new Int32Array(M * M).fill(-1);
  for (let idx = 0; idx < M * M; idx++) {
    if (bsCls[idx] === BS_INNER) { innerId[idx] = inner.length; inner.push(idx); }
    if (bsCls[idx] !== BS_IRRELEVANT) { relId[idx] = relevant.length; relevant.push(idx); }
  }
  const isInner = (k1, k2) => k1 >= -n && k2 >= -n && k1 < N && k2 < N && bsCls[grid.index(k1, k2)] === BS_INNER;
  // --- extension: closest (n+1)² array of inner indices for every outer j
  const ext = new Map(), W = 3 * (n + 1) + 3;
  for (const j of relevant) {
    if (bsCls[j] !== BS_OUTER) continue;
    const [j1, j2] = grid.kOf(j);
    let best = null, bestD = Infinity;
    for (let l2 = j2 - W; l2 <= j2 + W; l2++) for (let l1 = j1 - W; l1 <= j1 + W; l1++) {
      const d = (l1 + n / 2 - j1) ** 2 + (l2 + n / 2 - j2) ** 2;
      if (d >= bestD) continue;
      let ok = true;
      for (let a = 0; a <= n && ok; a++) for (let b = 0; b <= n && ok; b++) ok = isInner(l1 + b, l2 + a);
      if (ok) { best = [l1, l2]; bestD = d; }
    }
    if (!best) throw new Error('webSpace: no inner array found (grid too coarse)');
    const ids = new Int32Array((n + 1) ** 2), e = new Float64Array((n + 1) ** 2);
    let q = 0;
    for (let a = 0; a <= n; a++) for (let b = 0; b <= n; b++, q++) {
      const i1 = best[0] + b, i2 = best[1] + a;
      ids[q] = grid.index(i1, i2);
      e[q] = lagrangeIndexWeight(n, best[0], i1, j1) * lagrangeIndexWeight(n, best[1], i2, j2);
    }
    ext.set(j, { ell: best, ids, e });
  }
  // --- normalisation points x_i (inside cell of the support with the largest w)
  const wx = new Float64Array(inner.length), xi = new Float64Array(2 * inner.length);
  inner.forEach((idx, c) => {
    const [k1, k2] = grid.kOf(idx);
    let bw = -Infinity;
    wx[c] = 1; // fallback (only reached with extend === false for B-splines without inside cell)
    for (let a = Math.max(0, k2); a <= Math.min(N - 1, k2 + n); a++)
      for (let b = Math.max(0, k1); b <= Math.min(N - 1, k1 + n); b++) {
        if (cellCls[b + N * a] !== CELL_IN) continue;
        const x = x0 + (b + 0.5) * h, y = y0 + (a + 0.5) * h, v = weight.w(x, y);
        if (v > bw) { bw = v; wx[c] = v; xi[2 * c] = x; xi[2 * c + 1] = y; }
      }
  });
  return {
    ls, n, N, box, levels, grid, weight, cellCls, cellCut, bsCls,
    inner: Int32Array.from(inner), innerId, relevant: Int32Array.from(relevant), relId, ext, wx, xi,
  };
}

/**
 * 1D Lagrange weight in index space: Π_{μ=0, μ ≠ i−ℓ}^{n} (j − ℓ − μ) / (i − ℓ − μ).
 * The Lagrange basis polynomial of node i for the nodes ℓ, …, ℓ+n, evaluated at j.
 */
export function lagrangeIndexWeight(n, ell, i, j) {
  let p = 1;
  for (let mu = 0; mu <= n; mu++) {
    if (ell + mu === i) continue;
    p *= (j - ell - mu) / (i - ell - mu);
  }
  return p;
}

/**
 * Coefficient map E (relevant × inner, CSR-like lists): relevant B-spline r
 * receives Σ_i E[r][i] c_i. For an inner r: E[r][r] = 1; for an outer r:
 * E[r][i] = e_{i,r}, i ∈ I(r). (Without the 1/w(x_i) scaling.)
 * @returns {{cols: Int32Array[], vals: Float64Array[]}} per relevant compact id: inner compact ids and values
 */
export function extensionMatrix(sp) {
  const cols = [], vals = [];
  for (const r of sp.relevant) {
    if (sp.bsCls[r] === BS_INNER) { cols.push(Int32Array.of(sp.innerId[r])); vals.push(Float64Array.of(1)); }
    else {
      const E = sp.ext.get(r);
      cols.push(Int32Array.from(E.ids, (i) => sp.innerId[i])); vals.push(Float64Array.from(E.e));
    }
  }
  return { cols, vals };
}

/**
 * Quadrature points of Ω ∩ cell c (tensor Gauss for inside cells, cut rule for cut cells).
 * @returns {{x: Float64Array, y: Float64Array, w: Float64Array, n: number}}
 */
export function cellRule(sp, c, deg) {
  const { N, grid } = sp, h = grid.h, l1 = c % N, l2 = Math.floor(c / N);
  if (sp.cellCls[c] === CELL_CUT) return cutVolumeRule(sp.cellCut[c], deg);
  const G = gaussLegendre(Math.ceil((deg + 1) / 2)), m = G.x.length;
  const x = new Float64Array(m * m), y = new Float64Array(m * m), w = new Float64Array(m * m);
  for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) {
    const q = a * m + b;
    x[q] = grid.x0 + (l1 + (G.x[b] + 1) / 2) * h; y[q] = grid.y0 + (l2 + (G.x[a] + 1) / 2) * h;
    w[q] = G.w[a] * G.w[b] * h * h / 4;
  }
  return { x, y, w, n: m * m };
}

/**
 * Evaluate the (n+1)² B-splines non-zero on cell c at (x, y).
 * @returns {{idx: Int32Array, v: Float64Array, gx: Float64Array, gy: Float64Array}} grid indices, values, gradients
 */
export function cellBsplines(sp, c, x, y, out) {
  const { n, N, grid } = sp, h = grid.h, l1 = c % N, l2 = Math.floor(c / N), m = n + 1;
  out = out || { idx: new Int32Array(m * m), v: new Float64Array(m * m), gx: new Float64Array(m * m), gy: new Float64Array(m * m),
    vx: new Float64Array(m), dx: new Float64Array(m), vy: new Float64Array(m), dy: new Float64Array(m) };
  bsplineCellValues(n, (x - grid.x0) / h - l1, out.vx, out.dx);
  bsplineCellValues(n, (y - grid.y0) / h - l2, out.vy, out.dy);
  for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) {
    const q = a * m + b;
    out.idx[q] = grid.index(l1 - b, l2 - a);
    out.v[q] = out.vx[b] * out.vy[a];
    out.gx[q] = out.dx[b] * out.vy[a] / h;
    out.gy[q] = out.vx[b] * out.dy[a] / h;
  }
  return out;
}

/**
 * Assemble and solve the WEB-spline Galerkin system.
 * @param {object} ls shape
 * @param {{n: number, N: number, box?: number[], levels?: number, f: (x:number,y:number)=>number,
 *          gt?: {val: (x:number,y:number)=>number, grad: (x:number,y:number)=>[number,number]}, weight?: object}} o
 *   gt: extension g̃ of the Dirichlet data (default 0)
 * @returns {{sp: object, A: object, b: Float64Array, c: Float64Array, C: Float64Array, nDof: number, singular: boolean}}
 *   c: coefficients of the WEB-splines B_i (inner compact ids); C: resulting coefficients of w·b_r
 *   for every relevant r, so that u_h = g̃ + w Σ_r C_r b_r
 */
export function solveWEB(ls, o) {
  const sp = webSpace(ls, o);
  const { n, N } = sp, nIn = sp.inner.length, nRel = sp.relevant.length, m = (n + 1) ** 2;
  const deg = 2 * n + 4;
  const Gb = new SparseBuilder(nRel, nRel, N * N * m * m);
  const Frel = new Float64Array(nRel);
  const loc = new Float64Array(m * m), floc = new Float64Array(m), rid = new Int32Array(m);
  const phx = new Float64Array(m), phy = new Float64Array(m), ph = new Float64Array(m);
  let buf;
  for (let c = 0; c < N * N; c++) {
    if (sp.cellCls[c] === CELL_OUT) continue;
    const R = cellRule(sp, c, deg);
    loc.fill(0); floc.fill(0);
    for (let q = 0; q < R.n; q++) {
      const x = R.x[q], y = R.y[q], wq = R.w[q];
      buf = cellBsplines(sp, c, x, y, buf);
      const wv = sp.weight.w(x, y), [wgx, wgy] = sp.weight.grad(x, y);
      // φ_r = w b_r,  ∇φ_r = b_r ∇w + w ∇b_r
      for (let a = 0; a < m; a++) { ph[a] = wv * buf.v[a]; phx[a] = buf.v[a] * wgx + wv * buf.gx[a]; phy[a] = buf.v[a] * wgy + wv * buf.gy[a]; }
      const fq = o.f(x, y);
      const [ggx, ggy] = o.gt ? o.gt.grad(x, y) : [0, 0];
      for (let a = 0; a < m; a++) {
        floc[a] += wq * (fq * ph[a] - (ggx * phx[a] + ggy * phy[a]));
        for (let b = 0; b < m; b++) loc[a * m + b] += wq * (phx[a] * phx[b] + phy[a] * phy[b]);
      }
    }
    for (let a = 0; a < m; a++) { rid[a] = sp.relId[buf.idx[a]]; Frel[rid[a]] += floc[a]; }
    Gb.addBlock(rid, rid, loc);
  }
  const G = Gb.toCSR();
  // E with the normalisation 1/w(x_i): E[r][i] = (δ_ri or e_{i,r}) / w(x_i)
  const E = extensionMatrix(sp);
  for (let r = 0; r < nRel; r++) for (let k = 0; k < E.cols[r].length; k++) E.vals[r][k] /= sp.wx[E.cols[r][k]];
  // A = Eᵀ G E,  b = Eᵀ F   (row r of G·E, then scatter with Eᵀ)
  const Ab = new SparseBuilder(nIn, nIn, 4 * G.vals.length);
  const b = new Float64Array(nIn);
  const rowGE = new Map();
  for (let r = 0; r < nRel; r++) {
    rowGE.clear();
    for (let p = G.rowPtr[r]; p < G.rowPtr[r + 1]; p++) {
      const s = G.colIdx[p], g = G.vals[p];
      for (let k = 0; k < E.cols[s].length; k++) { const j = E.cols[s][k]; rowGE.set(j, (rowGE.get(j) || 0) + g * E.vals[s][k]); }
    }
    for (let k = 0; k < E.cols[r].length; k++) {
      const i = E.cols[r][k], e = E.vals[r][k];
      b[i] += e * Frel[r];
      for (const [j, v] of rowGE) Ab.add(i, j, e * v);
    }
  }
  const A = Ab.toCSR();
  const F = sparseLU(A);
  const cIn = F.solve(b);
  const C = new Float64Array(nRel);
  for (let r = 0; r < nRel; r++) for (let k = 0; k < E.cols[r].length; k++) C[r] += E.vals[r][k] * cIn[E.cols[r][k]];
  return { sp, A, b, c: cIn, C, nDof: nIn, singular: F.singular, gt: o.gt || null };
}

/**
 * Evaluate a WEB-spline solution at (x, y) inside the grid box.
 * @returns {{u: number, ux: number, uy: number}}
 */
export function evalWEB(sol, x, y) {
  const sp = sol.sp, { N, grid } = sp;
  const l1 = Math.min(N - 1, Math.max(0, Math.floor((x - grid.x0) / grid.h)));
  const l2 = Math.min(N - 1, Math.max(0, Math.floor((y - grid.y0) / grid.h)));
  const c = l1 + N * l2;
  const B = cellBsplines(sp, c, x, y);
  let s = 0, sx = 0, sy = 0;
  for (let a = 0; a < B.idx.length; a++) {
    const r = sp.relId[B.idx[a]];
    if (r < 0) continue;
    s += sol.C[r] * B.v[a]; sx += sol.C[r] * B.gx[a]; sy += sol.C[r] * B.gy[a];
  }
  const wv = sp.weight.w(x, y), [wgx, wgy] = sp.weight.grad(x, y);
  let u = wv * s, ux = wgx * s + wv * sx, uy = wgy * s + wv * sy;
  if (sol.gt) { u += sol.gt.val(x, y); const [gx, gy] = sol.gt.grad(x, y); ux += gx; uy += gy; }
  return { u, ux, uy };
}

/**
 * Errors ‖u − u_h‖_{L²(Ω)}, |u − u_h|_{H¹(Ω)} over the (refined-cut) domain.
 * @returns {{L2: number, H1: number}}
 */
export function errorsWEB(sol, u, grad) {
  const sp = sol.sp, deg = 2 * sp.n + 6;
  let e0 = 0, e1 = 0;
  for (let c = 0; c < sp.N * sp.N; c++) {
    if (sp.cellCls[c] === CELL_OUT) continue;
    const R = cellRule(sp, c, deg);
    for (let q = 0; q < R.n; q++) {
      const v = evalWEB(sol, R.x[q], R.y[q]), [gx, gy] = grad(R.x[q], R.y[q]);
      e0 += R.w[q] * (u(R.x[q], R.y[q]) - v.u) ** 2;
      e1 += R.w[q] * ((gx - v.ux) ** 2 + (gy - v.uy) ** 2);
    }
  }
  return { L2: Math.sqrt(e0), H1: Math.sqrt(e1) };
}

/**
 * The extension g̃ = (1 − w) G of Dirichlet data given by a smooth function G
 * (with gradient): g̃ = G on Γ because w = 0 there.
 * @param {{w: Function, grad: Function}} weight
 * @param {(x:number,y:number)=>number} G
 * @param {(x:number,y:number)=>[number,number]} gradG
 */
export function blendedExtension(weight, G, gradG) {
  return {
    val: (x, y) => (1 - weight.w(x, y)) * G(x, y),
    grad: (x, y) => {
      const w = weight.w(x, y), [wx, wy] = weight.grad(x, y), g = G(x, y), [gx, gy] = gradG(x, y);
      return [-wx * g + (1 - w) * gx, -wy * g + (1 - w) * gy];
    },
  };
}

/**
 * Symmetric diagonal (Jacobi) scaling D^{−1/2} A D^{−1/2} of a CSR matrix with
 * positive diagonal (unit diagonal afterwards). Used to compare condition
 * numbers independently of how the basis functions are normalised.
 * @param {import('../la/sparse.js').CSR} A
 * @returns {import('../la/sparse.js').CSR}
 */
export function diagScaled(A) {
  const d = new Float64Array(A.n);
  for (let i = 0; i < A.n; i++) for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) if (A.colIdx[k] === i) d[i] = 1 / Math.sqrt(A.vals[k]);
  const vals = A.vals.slice();
  for (let i = 0; i < A.n; i++) for (let k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++) vals[k] *= d[i] * d[A.colIdx[k]];
  return { n: A.n, m: A.m, rowPtr: A.rowPtr, colIdx: A.colIdx, vals };
}
