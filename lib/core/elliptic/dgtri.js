/**
 * @file Broken (element-wise) polynomial spaces on triangle meshes — the
 *       shared toolbox of the DG/HDG elliptic solvers (sipg.js, hdg.js).
 *
 * A "broken" space  V_h^p = { v : v|_K ∈ P_p(K) for every triangle K }  has no
 * continuity constraint between elements, so every triangle owns its own
 * nLoc = dimP(p) = (p+1)(p+2)/2 coefficients. We use the orthonormal Dubiner
 * basis ψ_m of lib/core/basis/simplex.js on the reference triangle T̂ and the
 * affine map  X = a + J x̂  of each triangle. Global DOF numbering:
 *     dof(t, m) = t·nLoc + m        (element-major, "block" ordering).
 * Because ∫_T̂ ψ_m ψ_n = δ_mn and the map is affine, the physical mass matrix
 * of triangle K is  M_K = |det J| · I = 2|K| · I  (diagonal!).
 *
 * Faces (edges): from lib/core/mesh/topology.js. Edge e has a "minus"
 * triangle K⁻ = edgeTris[2e] and (if interior) a "plus" triangle K⁺; its unit
 * normal n points out of K⁻. The edge parameter s ∈ [0,1] runs along K⁻'s
 * local edge from local vertex EDGE_VERTS[k⁻][0] to EDGE_VERTS[k⁻][1]. Both
 * triangles are counter-clockwise, so K⁺ traverses the same edge in the
 * OPPOSITE direction: the point with parameter s in K⁻ has parameter 1 − s
 * in K⁺. The tables below therefore store, for every local edge k, the basis
 * at s (orientation 0, used by K⁻) and at 1 − s (orientation 1, used by K⁺),
 * so that quadrature point q denotes the same physical point on both sides.
 */
import { dubinerBasis, dimP } from '../basis/simplex.js';
import { triangleRule } from '../quad/simplex.js';
import { gaussForDegree01 } from '../quad/gauss1d.js';
import { refEdgePoint } from '../mesh/topology.js';
import { inverse } from '../la/dense.js';

/**
 * Affine geometry of every triangle.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh (CCW triangles)
 * @returns {{nTri: number, area: Float64Array, det: Float64Array, J: Float64Array, Jinv: Float64Array, a: Float64Array}}
 *   per triangle t: J[4t..4t+3] = row-major [b−a | c−a], Jinv likewise, det = det J = 2|K| (> 0),
 *   a[2t], a[2t+1] = coordinates of local vertex 0.
 */
export function triGeometry(mesh) {
  const { nodes, tris } = mesh, nTri = tris.length / 3;
  const area = new Float64Array(nTri), det = new Float64Array(nTri);
  const J = new Float64Array(4 * nTri), Jinv = new Float64Array(4 * nTri), a = new Float64Array(2 * nTri);
  for (let t = 0; t < nTri; t++) {
    const i0 = tris[3 * t], i1 = tris[3 * t + 1], i2 = tris[3 * t + 2];
    const ax = nodes[2 * i0], ay = nodes[2 * i0 + 1];
    const j00 = nodes[2 * i1] - ax, j01 = nodes[2 * i2] - ax, j10 = nodes[2 * i1 + 1] - ay, j11 = nodes[2 * i2 + 1] - ay;
    const d = j00 * j11 - j01 * j10;
    det[t] = d; area[t] = 0.5 * d;
    J.set([j00, j01, j10, j11], 4 * t);
    Jinv.set([j11 / d, -j01 / d, -j10 / d, j00 / d], 4 * t);
    a[2 * t] = ax; a[2 * t + 1] = ay;
  }
  return { nTri, area, det, J, Jinv, a };
}

/** Physical point of reference point (x̂,ŷ) in triangle t. */
export function mapPoint(geo, t, xr, yr) {
  const J = geo.J, o = 4 * t;
  return [geo.a[2 * t] + J[o] * xr + J[o + 1] * yr, geo.a[2 * t + 1] + J[o + 2] * xr + J[o + 3] * yr];
}

/**
 * Tabulated Dubiner basis values and reference gradients at quadrature points.
 * @param {number} p polynomial degree
 * @param {number} volDeg exactness degree of the triangle rule
 * @param {number} edgeDeg exactness degree of the Gauss rule on edges
 * @returns {{p: number, n: number,
 *   vol: {rule: import('../quad/simplex.js').Rule2D, V: Float64Array, Dx: Float64Array, Dy: Float64Array},
 *   edge: {s: Float64Array, w: Float64Array, nq: number, V: Float64Array[][], Dx: Float64Array[][], Dy: Float64Array[][]}}}
 *   vol.V[q*n + m] = ψ_m at volume point q; edge.V[k][o][q*n + m] = ψ_m on local edge k at
 *   parameter s_q (o = 0) or 1 − s_q (o = 1); s, w = Gauss points/weights on [0,1] (weights sum to 1).
 */
export function basisTables(p, volDeg, edgeDeg) {
  const n = dimP(p), rule = triangleRule(volDeg);
  const tab = (pts) => {
    const V = new Float64Array(pts.length * n), Dx = new Float64Array(pts.length * n), Dy = new Float64Array(pts.length * n);
    pts.forEach(([x, y], q) => {
      const B = dubinerBasis(p, x, y);
      V.set(B.v, q * n); Dx.set(B.dx, q * n); Dy.set(B.dy, q * n);
    });
    return { V, Dx, Dy };
  };
  const vpts = []; for (let q = 0; q < rule.n; q++) vpts.push([rule.x[q], rule.y[q]]);
  const vol = { rule, ...tab(vpts) };
  const G = gaussForDegree01(edgeDeg), nq = G.x.length;
  const V = [], Dx = [], Dy = [];
  for (let k = 0; k < 3; k++) {
    V.push([]); Dx.push([]); Dy.push([]);
    for (let o = 0; o < 2; o++) {
      const pts = []; for (let q = 0; q < nq; q++) pts.push(refEdgePoint(k, o === 0 ? G.x[q] : 1 - G.x[q]));
      const T = tab(pts);
      V[k].push(T.V); Dx[k].push(T.Dx); Dy[k].push(T.Dy);
    }
  }
  return { p, n, vol, edge: { s: G.x, w: G.w, nq, V, Dx, Dy } };
}

/**
 * Physical gradients of all n basis functions at one tabulated point:
 * ∇ψ = J^{−T} ∇̂ψ̂.
 * @param {Float64Array} Jinv geometry Jinv array
 * @param {number} t triangle
 * @param {Float64Array} Dx reference x̂-derivatives table
 * @param {Float64Array} Dy reference ŷ-derivatives table
 * @param {number} q point index
 * @param {number} n basis size
 * @param {Float64Array} gx output (n)
 * @param {Float64Array} gy output (n)
 */
export function physGrads(Jinv, t, Dx, Dy, q, n, gx, gy) {
  const o = 4 * t, i00 = Jinv[o], i01 = Jinv[o + 1], i10 = Jinv[o + 2], i11 = Jinv[o + 3];
  for (let m = 0; m < n; m++) {
    const dx = Dx[q * n + m], dy = Dy[q * n + m];
    gx[m] = i00 * dx + i10 * dy;
    gy[m] = i01 * dx + i11 * dy;
  }
}

/** Monomial exponents (i, j) of x̂^i ŷ^j with i + j ≤ p, ordered by total degree. */
function monomials(p) {
  const r = [];
  for (let d = 0; d <= p; d++) for (let i = d; i >= 0; i--) r.push([i, d - i]);
  return r;
}

const monoCache = new Map();
/**
 * Change of basis Dubiner → monomials on T̂ (for fast per-pixel rendering):
 * if c are Dubiner coefficients, then T·c are the coefficients of x̂^i ŷ^j
 * (ordered as monomials(p)). Built by interpolation at the equispaced lattice
 * points (i/p, j/p), which are unisolvent for P_p.
 * @param {number} p
 * @returns {{T: Float64Array, exps: Array<[number, number]>, n: number}}
 */
export function dubinerToMonomial(p) {
  if (monoCache.has(p)) return monoCache.get(p);
  const n = dimP(p), exps = monomials(p);
  const pts = [];
  if (p === 0) pts.push([1 / 3, 1 / 3]);
  else for (let j = 0; j <= p; j++) for (let i = 0; i + j <= p; i++) pts.push([i / p, j / p]);
  const Vm = new Float64Array(n * n), Vd = new Float64Array(n * n);
  pts.forEach(([x, y], r) => {
    exps.forEach(([i, j], c) => { Vm[r * n + c] = x ** i * y ** j; });
    Vd.set(dubinerBasis(p, x, y).v, r * n);
  });
  const Vi = inverse(Vm, n), T = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let s = 0; for (let k = 0; k < n; k++) s += Vi[i * n + k] * Vd[k * n + j];
    T[i * n + j] = s;
  }
  const r = { T, exps, n };
  monoCache.set(p, r);
  return r;
}

/**
 * Convert a whole broken Dubiner field (element-major coefficients) to
 * monomial coefficients, and return a fast evaluator f(t, x̂, ŷ).
 * @param {number} p
 * @param {Float64Array} U coefficients, U[t*n + m]
 * @returns {(t: number, xr: number, yr: number) => number}
 */
export function brokenEvaluator(p, U) {
  const { T, exps, n } = dubinerToMonomial(p);
  const nT = U.length / n, C = new Float64Array(U.length);
  for (let t = 0; t < nT; t++)
    for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k < n; k++) s += T[i * n + k] * U[t * n + k]; C[t * n + i] = s; }
  const ei = exps.map((e) => e[0]), ej = exps.map((e) => e[1]);
  return (t, xr, yr) => {
    let s = 0;
    for (let m = 0; m < n; m++) s += C[t * n + m] * xr ** ei[m] * yr ** ej[m];
    return s;
  };
}

/**
 * Element-wise L² projection of a function onto the broken space P_p
 * (trivial thanks to orthonormality: c_m = (1/det J) ∫_K f ψ_m).
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {number} p
 * @param {(x:number,y:number)=>number} f
 * @returns {Float64Array} coefficients U[t*n + m]
 */
export function l2Project(mesh, p, f) {
  const geo = triGeometry(mesh), T = basisTables(p, 2 * p + 6, 1), n = T.n, R = T.vol.rule;
  const U = new Float64Array(geo.nTri * n);
  for (let t = 0; t < geo.nTri; t++)
    for (let q = 0; q < R.n; q++) {
      const [x, y] = mapPoint(geo, t, R.x[q], R.y[q]), fq = f(x, y) * R.w[q];
      for (let m = 0; m < n; m++) U[t * n + m] += fq * T.vol.V[q * n + m];
    }
  return U;
}

/**
 * Broken-space errors ‖u − u_h‖_{L²(Ω)} and the broken H¹ seminorm
 * (Σ_K ‖∇(u − u_h)‖²_K)^{1/2}, by a triangle rule of degree 2p + 6.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {number} p
 * @param {Float64Array} U coefficients U[t*n + m]
 * @param {(x:number,y:number)=>number} u exact solution
 * @param {(x:number,y:number)=>[number,number]} [grad] exact gradient (omit to skip H¹)
 * @returns {{L2: number, H1: number}}
 */
export function brokenErrors(mesh, p, U, u, grad) {
  const geo = triGeometry(mesh), T = basisTables(p, 2 * p + 6, 1), n = T.n, R = T.vol.rule;
  const gx = new Float64Array(n), gy = new Float64Array(n);
  let e0 = 0, e1 = 0;
  for (let t = 0; t < geo.nTri; t++)
    for (let q = 0; q < R.n; q++) {
      const [x, y] = mapPoint(geo, t, R.x[q], R.y[q]), w = R.w[q] * geo.det[t];
      let uh = 0, ux = 0, uy = 0;
      if (grad) physGrads(geo.Jinv, t, T.vol.Dx, T.vol.Dy, q, n, gx, gy);
      for (let m = 0; m < n; m++) {
        const c = U[t * n + m];
        uh += c * T.vol.V[q * n + m];
        if (grad) { ux += c * gx[m]; uy += c * gy[m]; }
      }
      e0 += w * (u(x, y) - uh) ** 2;
      if (grad) { const [Gx, Gy] = grad(x, y); e1 += w * ((Gx - ux) ** 2 + (Gy - uy) ** 2); }
    }
  return { L2: Math.sqrt(e0), H1: Math.sqrt(e1) };
}
