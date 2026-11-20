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

/**
 * Local RT0 data of triangle t: opposite vertices, scaling c_k = s_k |E_k|/(2|K|), edges, signs.
 * Local edge k joins local vertices k and k+1, so its opposite vertex is (k+2) mod 3.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {import('../mesh/topology.js').Topology} topo
 * @param {number} t
 * @returns {{area: number, edges: number[], sign: number[], c: number[], opp: number[][], verts: number[][]}}
 */
export function rt0Local(mesh, topo, t) {
  const { nodes, tris } = mesh;
  const verts = [0, 1, 2].map((k) => [nodes[2 * tris[3 * t + k]], nodes[2 * tris[3 * t + k] + 1]]);
  const area = 0.5 * ((verts[1][0] - verts[0][0]) * (verts[2][1] - verts[0][1]) - (verts[2][0] - verts[0][0]) * (verts[1][1] - verts[0][1]));
  const edges = [], sign = [], c = [], opp = [];
  for (let k = 0; k < 3; k++) {
    const e = topo.triEdges[3 * t + k];
    const s = topo.edgeTris[2 * e] === t ? 1 : -1;
    edges.push(e); sign.push(s);
    c.push(s * topo.lengths[e] / (2 * area));
    opp.push(verts[(k + 2) % 3]);
  }
  return { area, edges, sign, c, opp, verts };
}

/**
 * Evaluate σ_h = Σ_E σ_E φ_E at a physical point (x,y) inside triangle t.
 * @param {ReturnType<typeof rt0Local>} loc
 * @param {Float64Array} sig edge coefficients (normal components)
 * @returns {[number, number]}
 */
export function rt0Eval(loc, sig, x, y) {
  let sx = 0, sy = 0;
  for (let k = 0; k < 3; k++) {
    const a = sig[loc.edges[k]] * loc.c[k];
    sx += a * (x - loc.opp[k][0]); sy += a * (y - loc.opp[k][1]);
  }
  return [sx, sy];
}

/**
 * Local RT0 mass matrix (3×3, row-major) M_kl = ∫_K φ_k·φ_l, computed with an
 * exact degree-2 triangle rule. Uses the signed local functions of rt0Local.
 */
export function rt0LocalMass(loc) {
  const R = triangleRule(2), M = new Float64Array(9);
  const [a, b, c] = loc.verts, det = 2 * loc.area;
  for (let q = 0; q < R.n; q++) {
    const x = a[0] + (b[0] - a[0]) * R.x[q] + (c[0] - a[0]) * R.y[q];
    const y = a[1] + (b[1] - a[1]) * R.x[q] + (c[1] - a[1]) * R.y[q];
    const w = R.w[q] * det;
    for (let k = 0; k < 3; k++) for (let l = 0; l < 3; l++)
      M[3 * k + l] += w * loc.c[k] * loc.c[l] * ((x - loc.opp[k][0]) * (x - loc.opp[l][0]) + (y - loc.opp[k][1]) * (y - loc.opp[l][1]));
  }
  return M;
}

/** ∫_K f over every triangle (degree-`deg` rule). @returns {Float64Array} */
export function cellIntegrals(mesh, f, deg = 8) {
  const { nodes, tris } = mesh, nT = tris.length / 3, R = triangleRule(deg), F = new Float64Array(nT);
  for (let t = 0; t < nT; t++) {
    const a = 2 * tris[3 * t], b = 2 * tris[3 * t + 1], c = 2 * tris[3 * t + 2];
    const e1x = nodes[b] - nodes[a], e1y = nodes[b + 1] - nodes[a + 1], e2x = nodes[c] - nodes[a], e2y = nodes[c + 1] - nodes[a + 1];
    const det = e1x * e2y - e1y * e2x;
    for (let q = 0; q < R.n; q++) F[t] += R.w[q] * det * f(nodes[a] + e1x * R.x[q] + e2x * R.y[q], nodes[a + 1] + e1y * R.x[q] + e2y * R.y[q]);
  }
  return F;
}

/** ∫_E g ds over every boundary edge (0 for interior edges). @returns {Float64Array} */
export function boundaryEdgeIntegrals(mesh, topo, g, deg = 8) {
  const G = gaussForDegree01(deg), out = new Float64Array(topo.nEdge), { nodes } = mesh;
  for (let e = 0; e < topo.nEdge; e++) {
    if (!topo.isBoundary[e]) continue;
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
  const Gb = boundaryEdgeIntegrals(mesh, topo, g);
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
    const loc = rt0Local(mesh, topo, t), [a, b, c] = loc.verts, det = 2 * loc.area;
    let mean = 0;
    const pts = [];
    for (let q = 0; q < R.n; q++) {
      const x = a[0] + (b[0] - a[0]) * R.x[q] + (c[0] - a[0]) * R.y[q];
      const y = a[1] + (b[1] - a[1]) * R.x[q] + (c[1] - a[1]) * R.y[q];
      pts.push([x, y]); mean += R.w[q] * det * u(x, y);
    }
    mean /= loc.area;
    for (let q = 0; q < R.n; q++) {
      const [x, y] = pts[q], w = R.w[q] * det;
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
