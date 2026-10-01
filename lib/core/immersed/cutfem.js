/**
 * @file CutFEM: unfitted P1 finite elements with Nitsche boundary conditions
 *       and ghost-penalty stabilisation, for  −Δu = f in Ω,  u = g on Γ = ∂Ω.
 *
 * Setting (chapter 11):
 *  - A background triangulation T_h of a box ⊃ Ω (triGrid), mesh size h.
 *  - Ω = {φ < 0} is approximated by Ω_h = {φ_h < 0} (linear cut, optionally
 *    refined — see lib/core/quad/cut.js); Γ_h = ∂Ω_h with normal n.
 *  - Active mesh T_Γ∪inside = {K ∈ T_h : K ∩ Ω_h ≠ ∅}; the unknowns are the P1
 *    nodal values at all vertices of active elements (also those outside Ω!).
 *
 * Discrete problem: find u_h ∈ V_h with A_h(u_h, v) = L_h(v) for all v ∈ V_h,
 *   A_h(u, v) = ∫_{Ω_h} ∇u·∇v
 *             − ∫_{Γ_h} (∂_n u) v − θ ∫_{Γ_h} (∂_n v) u + (γ/h) ∫_{Γ_h} u v      (Nitsche, ch. 4)
 *             + g_h(u, v)                                                          (ghost penalty)
 *   L_h(v)    = ∫_{Ω_h} f v − θ ∫_{Γ_h} (∂_n v) g + (γ/h) ∫_{Γ_h} g v
 * with exactly the sign conventions of lib/core/elliptic/cg.js (addNitsche);
 * θ = 1 gives a symmetric matrix. h is the background mesh size (cell width)
 * — NOT the size of the cut part, which can be arbitrarily small.
 *
 * Ghost penalty (Burman 2010; Burman & Hansbo 2012), P1 version:
 *   g_h(u, v) = γ_g Σ_{F ∈ F_G} h ∫_F [∂_n u][∂_n v],
 * F_G = interior faces of the active mesh that belong to at least one CUT
 * element, [∂_n u] = (∇u|_{K⁻} − ∇u|_{K⁺})·n_F the jump of the normal
 * derivative (constant for P1). For P1, [∂_n u] = 0 on F means the two
 * linear pieces are the same polynomial, so g_h measures how far u_h on a
 * cut element deviates from the extension of its neighbour: this transfers
 * control of ∇u_h from the physical part to the whole active elements and
 * makes the condition number O(h⁻²) independently of how Γ cuts the mesh.
 */
import { triGrid, triAffine } from '../mesh/structured.js';
import { buildTopology } from '../mesh/topology.js';
import { classifyMesh, cutVolumeRule, cutInterfaceRule, INSIDE, CUT, OUTSIDE } from '../quad/cut.js';
import { triangleRule, mapTriangleRule } from '../quad/simplex.js';
import { SparseBuilder } from '../la/sparse.js';
import { sparseLU } from '../la/direct.js';

/** Reference P1 gradients (constant): ∇̂λ0 = (−1,−1), ∇̂λ1 = (1,0), ∇̂λ2 = (0,1). */
const REF_G = [[-1, -1], [1, 0], [0, 1]];

/**
 * Physical P1 data of triangle t: gradients of the 3 barycentric functions and
 * a function returning the barycentrics of a point.
 * @returns {{gx: number[], gy: number[], bary: (x:number,y:number)=>number[], area: number, verts: number[][]}}
 */
export function p1Element(mesh, t) {
  const aff = triAffine(mesh.nodes, mesh.tris, t);
  const Ji = aff.Jinv; // row-major inverse Jacobian; ∇φ = J^{-T} ∇̂φ̂
  const gx = REF_G.map(([a, b]) => Ji[0] * a + Ji[2] * b);
  const gy = REF_G.map(([a, b]) => Ji[1] * a + Ji[3] * b);
  const bary = (x, y) => {
    const dx = x - aff.a[0], dy = y - aff.a[1];
    const xr = Ji[0] * dx + Ji[1] * dy, yr = Ji[2] * dx + Ji[3] * dy;
    return [1 - xr - yr, xr, yr];
  };
  return { gx, gy, bary, area: 0.5 * Math.abs(aff.det), verts: [aff.a, aff.b, aff.c] };
}

/**
 * Assemble and solve CutFEM for −Δu = f, u = g on Γ.
 * @param {{ls: {phi: (x:number,y:number)=>number, isSDF?: boolean}, N: number, box?: number[],
 *          f: (x:number,y:number)=>number, g: (x:number,y:number)=>number,
 *          gamma?: number, gammaG?: number, ghost?: boolean, theta?: number, levels?: number,
 *          diag?: 'right'|'left'|'alt', solve?: boolean}} o
 *   gamma: Nitsche penalty γ (default 10); gammaG: ghost-penalty γ_g (default 0.1);
 *   ghost: switch the ghost penalty on/off; levels: cut refinement depth; solve: false = assemble only
 * @returns {{mesh: object, cut: object, topo: object, h: number, dofOf: Int32Array, nDof: number, dofVert: Int32Array,
 *            ghostFaces: Int32Array, A: import('../la/sparse.js').CSR, b: Float64Array, U: Float64Array|null, singular: boolean}}
 *   dofOf[v] = DOF index of vertex v or −1; U[dof] = nodal value
 */
export function solveCutFEM(o) {
  const box = o.box || [0, 1, 0, 1];
  const mesh = triGrid(o.N, o.N, box, { diag: o.diag || 'right' });
  const h = mesh.h, gamma = o.gamma ?? 10, gammaG = o.gammaG ?? 0.1, theta = o.theta ?? 1;
  const cut = classifyMesh(mesh, o.ls.phi, { levels: o.levels ?? 0, lip: o.ls.isSDF ? 1 : 2 });
  const nT = mesh.nTri, nV = mesh.nVert;
  // DOFs: vertices of active elements
  const dofOf = new Int32Array(nV).fill(-1);
  let nDof = 0;
  for (let t = 0; t < nT; t++) if (cut.cls[t] !== OUTSIDE)
    for (let k = 0; k < 3; k++) { const v = mesh.tris[3 * t + k]; if (dofOf[v] < 0) dofOf[v] = nDof++; }
  const dofVert = new Int32Array(nDof);
  for (let v = 0; v < nV; v++) if (dofOf[v] >= 0) dofVert[dofOf[v]] = v;

  const B = new SparseBuilder(nDof, nDof, 16 * nT);
  const b = new Float64Array(nDof);
  const Rfull = triangleRule(4);
  const K = new Float64Array(9);
  for (let t = 0; t < nT; t++) {
    if (cut.cls[t] === OUTSIDE) continue;
    const E = p1Element(mesh, t);
    const dofs = [0, 1, 2].map((k) => dofOf[mesh.tris[3 * t + k]]);
    K.fill(0);
    // volume part: ∫_{K∩Ω_h} ∇φ_j·∇φ_i = |K∩Ω_h| ∇φ_j·∇φ_i (P1 gradients are constant)
    const isCut = cut.cls[t] === CUT;
    const vol = isCut ? cut.cuts[t].area : E.area;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) K[3 * i + j] += vol * (E.gx[i] * E.gx[j] + E.gy[i] * E.gy[j]);
    const Rv = isCut ? cutVolumeRule(cut.cuts[t], 4) : mapTriangleRule(Rfull, E.verts[0], E.verts[1], E.verts[2]);
    for (let q = 0; q < Rv.n; q++) {
      const lam = E.bary(Rv.x[q], Rv.y[q]), fq = o.f(Rv.x[q], Rv.y[q]) * Rv.w[q];
      for (let i = 0; i < 3; i++) b[dofs[i]] += fq * lam[i];
    }
    if (isCut) {
      // Nitsche terms on Γ_h ∩ K
      const Ri = cutInterfaceRule(cut.cuts[t], 4), pen = gamma / h;
      for (let q = 0; q < Ri.n; q++) {
        const lam = E.bary(Ri.x[q], Ri.y[q]), w = Ri.w[q], gq = o.g(Ri.x[q], Ri.y[q]);
        const dn = [0, 1, 2].map((i) => E.gx[i] * Ri.nx[q] + E.gy[i] * Ri.ny[q]);
        for (let i = 0; i < 3; i++) {
          b[dofs[i]] += w * (-theta * dn[i] * gq + pen * gq * lam[i]);
          for (let j = 0; j < 3; j++) K[3 * i + j] += w * (-dn[j] * lam[i] - theta * dn[i] * lam[j] + pen * lam[i] * lam[j]);
        }
      }
    }
    B.addBlock(dofs, dofs, K);
  }
  // ghost penalty on F_G
  const topo = buildTopology(mesh.nodes, mesh.tris);
  const ghost = [];
  for (let e = 0; e < topo.nEdge; e++) {
    const t1 = topo.edgeTris[2 * e], t2 = topo.edgeTris[2 * e + 1];
    if (t2 < 0 || cut.cls[t1] === OUTSIDE || cut.cls[t2] === OUTSIDE) continue;
    if (cut.cls[t1] !== CUT && cut.cls[t2] !== CUT) continue;
    ghost.push(e);
    if (o.ghost === false) continue;
    const nx = topo.normals[2 * e], ny = topo.normals[2 * e + 1], len = topo.lengths[e];
    const E1 = p1Element(mesh, t1), E2 = p1Element(mesh, t2);
    // jump of ∂_n of every local basis function: +∂_nφ on K⁻, −∂_nφ on K⁺ (shared vertices add up)
    const map = new Map();
    for (let k = 0; k < 3; k++) {
      const d1 = dofOf[mesh.tris[3 * t1 + k]], d2 = dofOf[mesh.tris[3 * t2 + k]];
      map.set(d1, (map.get(d1) || 0) + E1.gx[k] * nx + E1.gy[k] * ny);
      map.set(d2, (map.get(d2) || 0) - (E2.gx[k] * nx + E2.gy[k] * ny));
    }
    const ids = [...map.keys()], J = ids.map((d) => map.get(d)), c = gammaG * h * len;
    const G = new Float64Array(ids.length * ids.length);
    for (let i = 0; i < ids.length; i++) for (let j = 0; j < ids.length; j++) G[i * ids.length + j] = c * J[i] * J[j];
    B.addBlock(ids, ids, G);
  }
  const A = B.toCSR();
  let U = null, singular = false;
  if (o.solve !== false) { const F = sparseLU(A); U = F.solve(b); singular = F.singular; }
  return { mesh, cut, topo, h, dofOf, dofVert, nDof, ghostFaces: Int32Array.from(ghost), A, b, U, singular };
}

/**
 * Value of the CutFEM solution on (active) triangle t at a point.
 * @returns {number}
 */
export function evalCutFEM(sol, t, x, y) {
  const E = p1Element(sol.mesh, t), lam = E.bary(x, y);
  let u = 0;
  for (let k = 0; k < 3; k++) u += lam[k] * sol.U[sol.dofOf[sol.mesh.tris[3 * t + k]]];
  return u;
}

/**
 * Errors ‖u − u_h‖_{L²(Ω_h)} and |u − u_h|_{H¹(Ω_h)} with degree-6 quadrature on the
 * cut sub-triangles (u, ∇u exact, defined everywhere).
 * @returns {{L2: number, H1: number}}
 */
export function errorsCutFEM(sol, u, grad) {
  const { mesh, cut } = sol, R = triangleRule(6);
  let e0 = 0, e1 = 0;
  for (let t = 0; t < mesh.nTri; t++) {
    if (cut.cls[t] === OUTSIDE) continue;
    const E = p1Element(mesh, t);
    const dofs = [0, 1, 2].map((k) => sol.dofOf[mesh.tris[3 * t + k]]);
    let ux = 0, uy = 0;
    for (let k = 0; k < 3; k++) { ux += sol.U[dofs[k]] * E.gx[k]; uy += sol.U[dofs[k]] * E.gy[k]; }
    const Rv = cut.cls[t] === CUT ? cutVolumeRule(cut.cuts[t], 6) : mapTriangleRule(R, E.verts[0], E.verts[1], E.verts[2]);
    for (let q = 0; q < Rv.n; q++) {
      const lam = E.bary(Rv.x[q], Rv.y[q]);
      let uh = 0;
      for (let k = 0; k < 3; k++) uh += lam[k] * sol.U[dofs[k]];
      const [gx, gy] = grad(Rv.x[q], Rv.y[q]);
      e0 += Rv.w[q] * (u(Rv.x[q], Rv.y[q]) - uh) ** 2;
      e1 += Rv.w[q] * ((gx - ux) ** 2 + (gy - uy) ** 2);
    }
  }
  return { L2: Math.sqrt(e0), H1: Math.sqrt(e1) };
}

export { INSIDE, CUT, OUTSIDE };
