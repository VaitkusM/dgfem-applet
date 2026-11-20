/**
 * @file Lowest-order Raviart–Thomas mixed method (RT0 × P0) for
 *       −Δu = f in Ω, u = g on ∂Ω, written as a first-order system
 *           σ = −∇u   (flux, AGENTS.md convention),   ∇·σ = f.
 *
 * Weak (mixed) form: find σ_h ∈ RT0, u_h ∈ P0 such that
 *     (σ_h, τ) − (u_h, ∇·τ) = −⟨g, τ·n⟩_{∂Ω}     for all τ ∈ RT0,
 *     (∇·σ_h, v)            = (f, v)             for all v ∈ P0.
 * (First line: multiply σ + ∇u = 0 by τ and integrate (∇u, τ) by parts;
 *  the Dirichlet datum enters NATURALLY through the boundary term.)
 *
 * RT0 basis. Every edge E gets a global unit normal n_E (topology normal: out
 * of K⁻). On a triangle K containing E with opposite vertex x_E^K,
 *     φ_E|_K (x) = s_{K,E} · |E| / (2|K|) · (x − x_E^K),
 * with s_{K,E} = +1 if n_E points out of K (K = K⁻) and −1 otherwise (K = K⁺).
 *  - φ_E·n_E = 1 on E (the height of K over E is 2|K|/|E|) and φ_E·n = 0 on the
 *    two other edges of K (x − x_E^K is tangent there): the coefficient of φ_E
 *    is the NORMAL COMPONENT of σ_h on E, and σ_h·n_E is single-valued across
 *    E — normal continuity, i.e. σ_h ∈ H(div).
 *  - ∇·φ_E|_K = s_{K,E} |E| / |K| (constant), so ∫_K ∇·φ_E = s_{K,E} |E|.
 *  - The flux of σ_h through E (in direction n_E) is  Σ_E-coefficient × |E|.
 *
 * Discrete system (unknowns ordered [σ_E for all edges | u_K for all triangles]):
 *     [ M   −Bᵀ ] [σ]   [ −G ]          M_EE' = Σ_K ∫_K φ_E·φ_E',
 *     [ −B   0  ] [u] = [ −F ]          B_KE = ∫_K ∇·φ_E = s_{K,E}|E|,
 *                                       G_E = ∫_E g ds (boundary edges),  F_K = ∫_K f.
 * (The second row was multiplied by −1 to make the matrix symmetric.) It is a
 * symmetric INDEFINITE saddle-point matrix; we solve it with RCM + banded LU
 * with partial pivoting (sparseLU), not CG.
 */
import { SparseBuilder } from '../la/sparse.js';
import { sparseLU } from '../la/direct.js';
import { buildTopology } from '../mesh/topology.js';
import { triangleRule } from '../quad/simplex.js';
import { gaussForDegree01 } from '../quad/gauss1d.js';
import { triGeometry, mapPoint } from './dgtri.js';

/**
 * Local RT0 data of triangle t. Local edge k joins local vertices k and k+1,
 * so its opposite vertex is (k+2) mod 3.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh (CCW triangles)
 * @param {import('../mesh/topology.js').Topology} topo
 * @param {number} t triangle index
 * @param {boolean} [outward=false] false: the global functions φ_E restricted to K, with sign
 *   s_{K,E} = ±1 (mixed method); true: all three functions oriented by the OUTWARD normal of K
 *   (sign +1: the "broken" local functions of the hybridized method)
 * @returns {{area: number, edges: Int32Array, sign: Float64Array, c: Float64Array, opp: Float64Array, verts: Float64Array}}
 *   edges[k] = global edge of local edge k; sign[k] = s_{K,E_k}; c[k] = sign[k]·|E_k|/(2|K|);
 *   verts = [x0,y0,x1,y1,x2,y2]; opp[2k], opp[2k+1] = vertex opposite to local edge k.
 *   Local function k is  c[k]·(x − opp_k).
 */
export function rt0Local(mesh, topo, t, outward = false) {
  const { nodes, tris } = mesh;
  const verts = new Float64Array(6), edges = new Int32Array(3), sign = new Float64Array(3), c = new Float64Array(3), opp = new Float64Array(6);
  for (let k = 0; k < 3; k++) { verts[2 * k] = nodes[2 * tris[3 * t + k]]; verts[2 * k + 1] = nodes[2 * tris[3 * t + k] + 1]; }
  const area = 0.5 * ((verts[2] - verts[0]) * (verts[5] - verts[1]) - (verts[4] - verts[0]) * (verts[3] - verts[1]));
  for (let k = 0; k < 3; k++) {
    const e = topo.triEdges[3 * t + k], o = (k + 2) % 3;
    edges[k] = e;
    sign[k] = outward || topo.edgeTris[2 * e] === t ? 1 : -1;
    c[k] = sign[k] * topo.lengths[e] / (2 * area);
    opp[2 * k] = verts[2 * o]; opp[2 * k + 1] = verts[2 * o + 1];
  }
  return { area, edges, sign, c, opp, verts };
}

/**
 * Evaluate σ_h = Σ_E σ_E φ_E at a physical point (x,y) inside triangle t.
 * @param {ReturnType<typeof rt0Local>} loc local data of the triangle
 * @param {Float64Array} sig edge coefficients (normal components σ_h·n_E), indexed by global edge
 * @param {number} x
 * @param {number} y
 * @returns {[number, number]} (σ_x, σ_y)
 */
export function rt0Eval(loc, sig, x, y) {
  let sx = 0, sy = 0;
  for (let k = 0; k < 3; k++) {
    const a = sig[loc.edges[k]] * loc.c[k];
    sx += a * (x - loc.opp[2 * k]); sy += a * (y - loc.opp[2 * k + 1]);
  }
  return [sx, sy];
}

/**
 * Physical point of the reference point (x̂,ŷ) in a triangle given by its vertices.
 * @param {Float64Array} v [x0,y0,x1,y1,x2,y2]
 * @param {number} xr
 * @param {number} yr
 * @returns {[number, number]}
 */
const physPoint = (v, xr, yr) => [v[0] + (v[2] - v[0]) * xr + (v[4] - v[0]) * yr, v[1] + (v[3] - v[1]) * xr + (v[5] - v[1]) * yr];

/**
 * Local RT0 mass matrix M_kl = ∫_K φ_k·φ_l of the three local functions of
 * rt0Local (with their signs), by a degree-2 triangle rule (exact: the
 * integrand is quadratic).
 * @param {ReturnType<typeof rt0Local>} loc
 * @returns {Float64Array} 3×3, row-major
 */
export function rt0LocalMass(loc) {
  const R = triangleRule(2), M = new Float64Array(9), det = 2 * loc.area;
  for (let q = 0; q < R.n; q++) {
    const [x, y] = physPoint(loc.verts, R.x[q], R.y[q]), w = R.w[q] * det;
    for (let k = 0; k < 3; k++) for (let l = 0; l < 3; l++)
      M[3 * k + l] += w * loc.c[k] * loc.c[l] * ((x - loc.opp[2 * k]) * (x - loc.opp[2 * l]) + (y - loc.opp[2 * k + 1]) * (y - loc.opp[2 * l + 1]));
  }
  return M;
}

/**
 * ∫_K f over every triangle.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {(x:number,y:number)=>number} f
 * @param {number} [deg=8] exactness degree of the triangle rule
 * @returns {Float64Array} one integral per triangle
 */
export function cellIntegrals(mesh, f, deg = 8) {
  const geo = triGeometry(mesh), R = triangleRule(deg), F = new Float64Array(geo.nTri);
  for (let t = 0; t < geo.nTri; t++)
    for (let q = 0; q < R.n; q++) { const [x, y] = mapPoint(geo, t, R.x[q], R.y[q]); F[t] += R.w[q] * geo.det[t] * f(x, y); }
  return F;
}

/**
 * ∫_E g ds over the edges of the mesh (Gauss rule).
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {import('../mesh/topology.js').Topology} topo
 * @param {(x:number,y:number)=>number} g
 * @param {number} [deg=8] exactness degree of the Gauss rule
 * @param {boolean} [boundaryOnly=true] true: boundary edges only (0 stored for interior edges); false: every edge
 * @returns {Float64Array} one integral per edge
 */
export function edgeIntegrals(mesh, topo, g, deg = 8, boundaryOnly = true) {
  const G = gaussForDegree01(deg), out = new Float64Array(topo.nEdge), { nodes } = mesh;
  for (let e = 0; e < topo.nEdge; e++) {
    if (boundaryOnly && !topo.isBoundary[e]) continue;
    const a = topo.edges[2 * e], b = topo.edges[2 * e + 1];
    for (let q = 0; q < G.x.length; q++) {
      const s = G.x[q];
      out[e] += G.w[q] * topo.lengths[e] * g(nodes[2 * a] + s * (nodes[2 * b] - nodes[2 * a]), nodes[2 * a + 1] + s * (nodes[2 * b + 1] - nodes[2 * a + 1]));
    }
  }
  return out;
}

/**
 * Assemble and solve the RT0–P0 saddle-point system.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {{f: (x:number,y:number)=>number, g?: (x:number,y:number)=>number}} o
 * @returns {{sigma: Float64Array, u: Float64Array, A: import('../la/sparse.js').CSR, rhs: Float64Array,
 *            topo: import('../mesh/topology.js').Topology, mesh: object, nEdge: number, nTri: number, F: Float64Array}}
 *   sigma[E] = normal component σ_h·n_E on edge E; u[K] = value of u_h on triangle K; F[K] = ∫_K f.
 */
export function solveMixedRT0(mesh, o) {
  const topo = buildTopology(mesh.nodes, mesh.tris);
  const nE = topo.nEdge, nT = mesh.tris.length / 3, n = nE + nT;
  const g = o.g ?? (() => 0);
  const B = new SparseBuilder(n, n, nT * 15);
  const rhs = new Float64Array(n);
  const F = cellIntegrals(mesh, o.f);
  const Gb = edgeIntegrals(mesh, topo, g);
  for (let t = 0; t < nT; t++) {
    const loc = rt0Local(mesh, topo, t), M = rt0LocalMass(loc);
    B.addBlock(loc.edges, loc.edges, M);
    for (let k = 0; k < 3; k++) {
      const div = loc.sign[k] * topo.lengths[loc.edges[k]]; // ∫_K ∇·φ_E
      B.add(loc.edges[k], nE + t, -div);
      B.add(nE + t, loc.edges[k], -div);
    }
    rhs[nE + t] = -F[t];
  }
  // boundary edges: n_E is outward (K⁻ is the only triangle), φ_E·n = 1 ⇒ ⟨g, φ_E·n⟩ = ∫_E g
  for (let e = 0; e < nE; e++) if (topo.isBoundary[e]) rhs[e] = -Gb[e];
  const A = B.toCSR();
  const x = sparseLU(A).solve(rhs);
  return { sigma: x.slice(0, nE), u: x.slice(nE), A, rhs, topo, mesh, nEdge: nE, nTri: nT, F };
}

/**
 * Errors of an RT0 solution against the exact (u, σ = −∇u):
 *  - L2u  = ‖u − u_h‖_{L²}
 *  - L2u0 = ‖Π₀u − u_h‖_{L²}  (Π₀ = element means; superconvergent)
 *  - L2s  = ‖σ − σ_h‖_{L²}
 * @param {ReturnType<typeof solveMixedRT0>} S
 * @param {(x:number,y:number)=>number} u
 * @param {(x:number,y:number)=>[number,number]} grad
 * @returns {{L2u: number, L2u0: number, L2s: number}}
 */
export function rt0Errors(S, u, grad) {
  const { mesh, topo } = S, R = triangleRule(8);
  let eu = 0, eu0 = 0, es = 0;
  for (let t = 0; t < S.nTri; t++) {
    const loc = rt0Local(mesh, topo, t), det = 2 * loc.area;
    let mean = 0;
    for (let q = 0; q < R.n; q++) { const [x, y] = physPoint(loc.verts, R.x[q], R.y[q]); mean += R.w[q] * det * u(x, y); }
    mean /= loc.area;
    for (let q = 0; q < R.n; q++) {
      const [x, y] = physPoint(loc.verts, R.x[q], R.y[q]), w = R.w[q] * det;
      const [sx, sy] = rt0Eval(loc, S.sigma, x, y), [gx, gy] = grad(x, y);
      eu += w * (u(x, y) - S.u[t]) ** 2;
      es += w * ((-gx - sx) ** 2 + (-gy - sy) ** 2);
    }
    eu0 += loc.area * (mean - S.u[t]) ** 2;
  }
  return { L2u: Math.sqrt(eu), L2u0: Math.sqrt(eu0), L2s: Math.sqrt(es) };
}

/**
 * Discrete conservation check: for every triangle the net outflow
 * ∫_{∂K} σ_h·n = Σ_E s_{K,E} |E| σ_E  minus ∫_K f.
 * @param {ReturnType<typeof solveMixedRT0>} S
 * @returns {Float64Array} residual per triangle (≈ round-off)
 */
export function rt0ConservationResidual(S) {
  const { topo } = S, r = new Float64Array(S.nTri);
  for (let t = 0; t < S.nTri; t++) {
    let out = 0;
    for (let k = 0; k < 3; k++) {
      const e = topo.triEdges[3 * t + k];
      out += (topo.edgeTris[2 * e] === t ? 1 : -1) * topo.lengths[e] * S.sigma[e];
    }
    r[t] = out - S.F[t];
  }
  return r;
}
