/**
 * @file Continuous Galerkin (CG) finite elements of degree p ∈ {1, 2} on
 *       triangle meshes for the Poisson problem  −Δu = f  in Ω,  u = g on ∂Ω.
 *
 * Weak form (chapter 2): find u_h ∈ V_h with
 *     a(u_h, v) = ∫_Ω ∇u_h · ∇v dx = ∫_Ω f v dx   for all test functions v,
 * plus a way of enforcing u = g on the boundary:
 *  - 'strong'  : boundary DOFs are set to g (nodal interpolation) and
 *                eliminated; their known contributions move to the RHS.
 *  - 'nitsche' : weak imposition (chapter 4). On each boundary edge E add
 *        −∫_E ∂_n u v − θ ∫_E ∂_n v u + (γ/h_E) ∫_E u v
 *     to the bilinear form and  −θ ∫_E ∂_n v g + (γ/h_E) ∫_E g v  to the RHS;
 *     θ = 1 symmetric Nitsche, θ = −1 non-symmetric Nitsche.
 *  - 'penalty' : only (γ/h_E) ∫_E (u − g) v — simple but inconsistent.
 *
 * Degrees of freedom (DOFs): P1 → one per vertex; P2 → vertices first, then
 * one per edge (midpoint). Element-local order = order of lib/core/basis/simplex.js.
 */
import { p1Basis, p2Basis, EDGE_VERTS } from '../basis/simplex.js';
import { triangleRule } from '../quad/simplex.js';
import { gaussForDegree01 } from '../quad/gauss1d.js';
import { buildTopology, refEdgePoint } from '../mesh/topology.js';
import { triAffine, refToPhysGrad } from '../mesh/structured.js';
import { SparseBuilder, csrExtract, csrMatVec } from '../la/sparse.js';
import { sparseLU } from '../la/direct.js';

/**
 * @typedef {Object} CGSpace
 * @property {number} p degree
 * @property {number} nDof
 * @property {number} nLoc local DOFs per triangle (3 or 6)
 * @property {Int32Array} elemDofs nLoc global DOF ids per triangle
 * @property {Float64Array} dofXY coordinates of the DOF nodes (2 per DOF)
 * @property {Uint8Array} onBoundary 1 for DOFs on ∂Ω
 * @property {Object} mesh {nodes, tris}
 * @property {import('../mesh/topology.js').Topology} topo
 */

/** Reference basis for degree p. */
export const refBasis = (p, x, y) => (p === 1 ? p1Basis(x, y) : p2Basis(x, y));

/**
 * Build the DOF map of the P_p Lagrange space on a triangle mesh.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {1|2} p
 * @returns {CGSpace}
 */
export function cgSpace(mesh, p) {
  const { nodes, tris } = mesh;
  const nTri = tris.length / 3, nVert = nodes.length / 2;
  const topo = buildTopology(nodes, tris);
  const nLoc = p === 1 ? 3 : 6;
  const nDof = p === 1 ? nVert : nVert + topo.nEdge;
  const elemDofs = new Int32Array(nLoc * nTri);
  for (let t = 0; t < nTri; t++) {
    for (let k = 0; k < 3; k++) elemDofs[nLoc * t + k] = tris[3 * t + k];
    if (p === 2) for (let k = 0; k < 3; k++) elemDofs[nLoc * t + 3 + k] = nVert + topo.triEdges[3 * t + k];
  }
  const dofXY = new Float64Array(2 * nDof);
  dofXY.set(nodes);
  const onBoundary = new Uint8Array(nDof);
  for (let e = 0; e < topo.nEdge; e++) {
    const a = topo.edges[2 * e], b = topo.edges[2 * e + 1];
    if (p === 2) {
      dofXY[2 * (nVert + e)] = 0.5 * (nodes[2 * a] + nodes[2 * b]);
      dofXY[2 * (nVert + e) + 1] = 0.5 * (nodes[2 * a + 1] + nodes[2 * b + 1]);
    }
    if (topo.isBoundary[e]) {
      onBoundary[a] = onBoundary[b] = 1;
      if (p === 2) onBoundary[nVert + e] = 1;
    }
  }
  return { p, nDof, nLoc, elemDofs, dofXY, onBoundary, mesh, topo };
}

/**
 * Physical basis values/gradients of triangle t at reference point (x̂,ŷ).
 * @returns {{v: Float64Array, gx: Float64Array, gy: Float64Array, x: number, y: number, det: number}}
 */
export function physBasis(space, t, xr, yr, aff) {
  aff = aff || triAffine(space.mesh.nodes, space.mesh.tris, t);
  const B = refBasis(space.p, xr, yr), n = space.nLoc;
  const gx = new Float64Array(n), gy = new Float64Array(n);
  for (let i = 0; i < n; i++) { const g = refToPhysGrad(aff.Jinv, B.dx[i], B.dy[i]); gx[i] = g[0]; gy[i] = g[1]; }
  const x = aff.a[0] + aff.J[0] * xr + aff.J[1] * yr, y = aff.a[1] + aff.J[2] * xr + aff.J[3] * yr;
  return { v: B.v, gx, gy, x, y, det: aff.det };
}

/**
 * Assemble the stiffness matrix A_ij = ∫ ∇φ_j·∇φ_i (optionally with a
 * coefficient κ(x,y)) and load vector b_i = ∫ f φ_i over the whole mesh.
 * @param {CGSpace} space
 * @param {(x:number,y:number)=>number} f
 * @param {{kappa?: (x:number,y:number)=>number, mass?: boolean}} [opts] mass: also return mass matrix
 * @returns {{builder: SparseBuilder, b: Float64Array, M?: import('../la/sparse.js').CSR}}
 */
export function assembleVolume(space, f, opts = {}) {
  const { nDof, nLoc, elemDofs, mesh } = space;
  const nTri = mesh.tris.length / 3;
  const R = triangleRule(2 * space.p + 3);
  const builder = new SparseBuilder(nDof, nDof, nTri * nLoc * nLoc);
  const Mb = opts.mass ? new SparseBuilder(nDof, nDof, nTri * nLoc * nLoc) : null;
  const b = new Float64Array(nDof);
  const K = new Float64Array(nLoc * nLoc), Mloc = new Float64Array(nLoc * nLoc), dofs = new Int32Array(nLoc);
  for (let t = 0; t < nTri; t++) {
    const aff = triAffine(mesh.nodes, mesh.tris, t);
    K.fill(0); Mloc.fill(0);
    for (let k = 0; k < nLoc; k++) dofs[k] = elemDofs[nLoc * t + k];
    for (let q = 0; q < R.n; q++) {
      const P = physBasis(space, t, R.x[q], R.y[q], aff);
      const w = R.w[q] * Math.abs(aff.det);
      const kap = opts.kappa ? opts.kappa(P.x, P.y) : 1;
      const fq = f(P.x, P.y);
      for (let i = 0; i < nLoc; i++) {
        b[dofs[i]] += w * fq * P.v[i];
        for (let j = 0; j < nLoc; j++) {
          K[i * nLoc + j] += w * kap * (P.gx[i] * P.gx[j] + P.gy[i] * P.gy[j]);
          if (Mb) Mloc[i * nLoc + j] += w * P.v[i] * P.v[j];
        }
      }
    }
    builder.addBlock(dofs, dofs, K);
    if (Mb) Mb.addBlock(dofs, dofs, Mloc);
  }
  return { builder, b, M: Mb ? Mb.toCSR() : undefined };
}

/**
 * Add Nitsche / penalty boundary terms on all boundary edges.
 * @param {CGSpace} space
 * @param {SparseBuilder} builder
 * @param {Float64Array} b
 * @param {(x:number,y:number)=>number} g Dirichlet data
 * @param {{gamma: number, theta?: number, penaltyOnly?: boolean}} o
 */
export function addNitsche(space, builder, b, g, o) {
  const { topo, mesh, nLoc, elemDofs } = space;
  const theta = o.theta ?? 1, gamma = o.gamma;
  const G = gaussForDegree01(2 * space.p + 3);
  const K = new Float64Array(nLoc * nLoc), dofs = new Int32Array(nLoc);
  for (let e = 0; e < topo.nEdge; e++) {
    if (!topo.isBoundary[e]) continue;
    const t = topo.edgeTris[2 * e], k = topo.edgeLocal[2 * e];
    const aff = triAffine(mesh.nodes, mesh.tris, t);
    const nx = topo.normals[2 * e], ny = topo.normals[2 * e + 1], len = topo.lengths[e];
    const pen = gamma / len;
    K.fill(0);
    for (let i = 0; i < nLoc; i++) dofs[i] = elemDofs[nLoc * t + i];
    for (let q = 0; q < G.x.length; q++) {
      const [xr, yr] = refEdgePoint(k, G.x[q]);
      const P = physBasis(space, t, xr, yr, aff);
      const w = G.w[q] * len, gq = g(P.x, P.y);
      for (let i = 0; i < nLoc; i++) {
        const dni = P.gx[i] * nx + P.gy[i] * ny;
        if (!o.penaltyOnly) b[dofs[i]] += w * (-theta * dni * gq);
        b[dofs[i]] += w * pen * gq * P.v[i];
        for (let j = 0; j < nLoc; j++) {
          const dnj = P.gx[j] * nx + P.gy[j] * ny;
          let v = pen * P.v[j] * P.v[i];
          if (!o.penaltyOnly) v += -dnj * P.v[i] - theta * dni * P.v[j];
          K[i * nLoc + j] += w * v;
        }
      }
    }
    builder.addBlock(dofs, dofs, K);
  }
}

/**
 * Solve the Poisson problem with CG elements.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {{p: 1|2, f: (x:number,y:number)=>number, g: (x:number,y:number)=>number,
 *          bc?: 'strong'|'nitsche'|'nitsche-nonsym'|'penalty', gamma?: number, kappa?: (x:number,y:number)=>number}} o
 * @returns {{space: CGSpace, U: Float64Array, A: import('../la/sparse.js').CSR, b: Float64Array, free: Int32Array|null, singular: boolean}}
 *   A, b = the system actually solved (reduced to free DOFs for 'strong')
 */
export function solvePoissonCG(mesh, o) {
  const space = cgSpace(mesh, o.p);
  const { builder, b } = assembleVolume(space, o.f, { kappa: o.kappa });
  const bc = o.bc || 'strong';
  if (bc === 'strong') {
    const Afull = builder.toCSR();
    const free = [], fixed = [];
    for (let i = 0; i < space.nDof; i++) (space.onBoundary[i] ? fixed : free).push(i);
    const U = new Float64Array(space.nDof);
    for (const i of fixed) U[i] = o.g(space.dofXY[2 * i], space.dofXY[2 * i + 1]);
    // b_f − A_fd g_d
    const Afd = csrExtract(Afull, free, fixed);
    const gd = Float64Array.from(fixed, (i) => U[i]);
    const corr = csrMatVec(Afd, gd);
    const bf = Float64Array.from(free, (i, k) => b[i] - corr[k]);
    const Aff = csrExtract(Afull, free, free);
    const F = sparseLU(Aff);
    const Uf = F.solve(bf);
    free.forEach((i, k) => { U[i] = Uf[k]; });
    return { space, U, A: Aff, b: bf, free: Int32Array.from(free), singular: F.singular };
  }
  const theta = bc === 'nitsche-nonsym' ? -1 : 1;
  addNitsche(space, builder, b, o.g, { gamma: o.gamma ?? 10 * o.p * o.p, theta, penaltyOnly: bc === 'penalty' });
  const A = builder.toCSR();
  const F = sparseLU(A);
  const U = F.solve(b);
  return { space, U, A, b, free: null, singular: F.singular };
}

/**
 * Value (and gradient) of the CG function U on triangle t at reference point.
 * @returns {{u: number, ux: number, uy: number}}
 */
export function evalCG(space, U, t, xr, yr, aff) {
  const P = physBasis(space, t, xr, yr, aff);
  let u = 0, ux = 0, uy = 0;
  for (let i = 0; i < space.nLoc; i++) {
    const c = U[space.elemDofs[space.nLoc * t + i]];
    u += c * P.v[i]; ux += c * P.gx[i]; uy += c * P.gy[i];
  }
  return { u, ux, uy };
}

/**
 * Errors ‖u − u_h‖_{L²} and |u − u_h|_{H¹} (high-order quadrature).
 * @param {CGSpace} space
 * @param {Float64Array} U
 * @param {(x:number,y:number)=>number} u exact solution
 * @param {(x:number,y:number)=>[number,number]} grad exact gradient
 * @returns {{L2: number, H1: number}}
 */
export function errorsCG(space, U, u, grad) {
  const R = triangleRule(2 * space.p + 6), nTri = space.mesh.tris.length / 3;
  let e0 = 0, e1 = 0;
  for (let t = 0; t < nTri; t++) {
    const aff = triAffine(space.mesh.nodes, space.mesh.tris, t);
    for (let q = 0; q < R.n; q++) {
      const P = physBasis(space, t, R.x[q], R.y[q], aff);
      let uh = 0, ux = 0, uy = 0;
      for (let i = 0; i < space.nLoc; i++) {
        const c = U[space.elemDofs[space.nLoc * t + i]];
        uh += c * P.v[i]; ux += c * P.gx[i]; uy += c * P.gy[i];
      }
      const w = R.w[q] * Math.abs(aff.det), [gx, gy] = grad(P.x, P.y);
      e0 += w * (u(P.x, P.y) - uh) ** 2;
      e1 += w * ((gx - ux) ** 2 + (gy - uy) ** 2);
    }
  }
  return { L2: Math.sqrt(e0), H1: Math.sqrt(e1) };
}

/**
 * Maximum of |u_h − g| over boundary edges (sampled at Gauss points):
 * how well the Dirichlet condition is satisfied.
 */
export function boundaryError(space, U, g) {
  const { topo, mesh } = space;
  const G = gaussForDegree01(6);
  let mx = 0;
  for (let e = 0; e < topo.nEdge; e++) {
    if (!topo.isBoundary[e]) continue;
    const t = topo.edgeTris[2 * e], k = topo.edgeLocal[2 * e];
    const aff = triAffine(mesh.nodes, mesh.tris, t);
    for (let q = 0; q < G.x.length; q++) {
      const [xr, yr] = refEdgePoint(k, G.x[q]);
      const P = physBasis(space, t, xr, yr, aff);
      const { u } = evalCG(space, U, t, xr, yr, aff);
      mx = Math.max(mx, Math.abs(u - g(P.x, P.y)));
    }
  }
  return mx;
}

export { EDGE_VERTS };
