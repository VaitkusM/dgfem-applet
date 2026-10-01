/**
 * @file The Shifted Boundary Method (SBM) of Main & Scovazzi (2018),
 *       "The shifted boundary method for embedded domain computations.
 *       Part I: Poisson and Stokes problems", J. Comput. Phys. 372, 972–995,
 *       for  −Δu = f in Ω,  u = g on Γ, with P1 elements on a background grid.
 *
 * Idea: no cut cells at all. Solve on a SURROGATE domain Ω̃_h made of whole
 * background triangles, and move ("shift") the boundary condition from the
 * true boundary Γ to the surrogate boundary Γ̃_h = ∂Ω̃_h by a Taylor expansion.
 *
 *  - Surrogate domain (what we implement): Ω̃_h = union of the background
 *    triangles whose three vertices all satisfy φ < 0. (For a convex Ω this
 *    is exactly the set of triangles K ⊂ Ω; for non-convex parts Γ may bulge
 *    slightly into such a triangle. Other selections are used in the
 *    literature, e.g. adding cut elements whose larger part lies inside to
 *    bring Γ̃ closer to Γ; the formulation below does not depend on the
 *    choice, only the distance |d| = O(h) matters.)
 *  - Γ̃_h = edges of Ω̃_h shared with no other surrogate triangle; ñ = outward
 *    unit normal of Ω̃_h on such an edge.
 *  - Distance vector d(x̃) = M(x̃) − x̃, where M(x̃) is the closest point of x̃
 *    on Γ (we use the exact closest-point projection of levelset.js).
 *  - Shifted Dirichlet condition: from the Taylor expansion
 *        u(x̃ + d) = u(x̃) + ∇u(x̃)·d + O(|d|²)
 *    and u(x̃ + d) = g(M(x̃)) =: ḡ(x̃), the condition imposed on Γ̃_h is
 *        u + ∇u·d = ḡ   on Γ̃_h      (first-order shift operator S_h u = u + ∇u·d).
 *
 * Weak form (Main & Scovazzi 2018, Part I, Poisson; their (w, u) ↔ our (v, u)):
 *   find u_h ∈ V_h(Ω̃_h) such that for all v ∈ V_h(Ω̃_h)
 *     (∇v, ∇u_h)_{Ω̃}  −  ⟨v, ∇u_h·ñ⟩_{Γ̃}  −  ⟨∇v·ñ, u_h + ∇u_h·d⟩_{Γ̃}
 *        + ⟨(α/h) (v + ∇v·d), u_h + ∇u_h·d⟩_{Γ̃}
 *     = (v, f)_{Ω̃}  −  ⟨∇v·ñ, ḡ⟩_{Γ̃}  +  ⟨(α/h) (v + ∇v·d), ḡ⟩_{Γ̃}.
 * Term by term: the 2nd is the boundary term of integration by parts
 * (consistency), the 3rd the Nitsche "adjoint" term applied to the shifted
 * residual (u_h + ∇u_h·d − ḡ), the 4th the penalty on that shifted residual,
 * tested with the shifted test function. The matrix is NOT symmetric
 * (∇u·d appears in the 3rd term, ∇v·d does not appear with ∇u·ñ).
 * With d = 0 everything reduces to the symmetric Nitsche method of chapter 4
 * (cg.js addNitsche, θ = 1) posed on Ω̃_h.
 *
 * Variant "no shift" (shift: false): d = 0 in the operator but the data is
 * still transferred, ḡ(x̃) = g(M(x̃)) — i.e. we pretend that Γ̃_h is the true
 * boundary. This commits an O(h) error in the boundary condition, so the
 * L² error is only first order; the shift restores second order for P1.
 */
import { triGrid } from '../mesh/structured.js';
import { buildTopology } from '../mesh/topology.js';
import { triangleRule, mapTriangleRule } from '../quad/simplex.js';
import { gaussForDegree01 } from '../quad/gauss1d.js';
import { SparseBuilder } from '../la/sparse.js';
import { sparseLU } from '../la/direct.js';
import { p1Element } from './cutfem.js';

/**
 * Assemble and solve the SBM.
 * @param {{ls: {phi: Function, closest: Function}, N: number, box?: number[], f: (x:number,y:number)=>number,
 *          g: (x:number,y:number)=>number, alpha?: number, shift?: boolean, diag?: 'right'|'left'|'alt'}} o
 *   alpha: penalty α (default 10); shift: use the Taylor shift (default true)
 * @returns {{mesh: object, topo: object, h: number, inSur: Uint8Array, dofOf: Int32Array, nDof: number,
 *            bEdges: Int32Array, bTri: Int32Array, bNormal: Float64Array,
 *            qPts: Float64Array, qCP: Float64Array, A: object, b: Float64Array, U: Float64Array, singular: boolean}}
 *   inSur[t] = 1 for surrogate triangles; bEdges: surrogate-boundary edge ids, bTri: their surrogate triangle,
 *   bNormal: ñ (2 per edge); qPts / qCP: boundary quadrature points x̃ and their closest points M(x̃) (2 each)
 */
export function solveSBM(o) {
  const box = o.box || [0, 1, 0, 1];
  const mesh = triGrid(o.N, o.N, box, { diag: o.diag || 'right' });
  const h = mesh.h, alpha = o.alpha ?? 10, shift = o.shift !== false;
  const nT = mesh.nTri, nV = mesh.nVert;
  const phiN = new Float64Array(nV);
  for (let v = 0; v < nV; v++) phiN[v] = o.ls.phi(mesh.nodes[2 * v], mesh.nodes[2 * v + 1]);
  const inSur = new Uint8Array(nT);
  for (let t = 0; t < nT; t++) inSur[t] = phiN[mesh.tris[3 * t]] < 0 && phiN[mesh.tris[3 * t + 1]] < 0 && phiN[mesh.tris[3 * t + 2]] < 0 ? 1 : 0;
  const dofOf = new Int32Array(nV).fill(-1);
  let nDof = 0;
  for (let t = 0; t < nT; t++) if (inSur[t]) for (let k = 0; k < 3; k++) { const v = mesh.tris[3 * t + k]; if (dofOf[v] < 0) dofOf[v] = nDof++; }

  const B = new SparseBuilder(nDof, nDof, 16 * nT);
  const b = new Float64Array(nDof);
  const R = triangleRule(4), K = new Float64Array(9);
  for (let t = 0; t < nT; t++) {
    if (!inSur[t]) continue;
    const E = p1Element(mesh, t), dofs = [0, 1, 2].map((k) => dofOf[mesh.tris[3 * t + k]]);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) K[3 * i + j] = E.area * (E.gx[i] * E.gx[j] + E.gy[i] * E.gy[j]);
    B.addBlock(dofs, dofs, K);
    const Rv = mapTriangleRule(R, E.verts[0], E.verts[1], E.verts[2]);
    for (let q = 0; q < Rv.n; q++) {
      const lam = E.bary(Rv.x[q], Rv.y[q]), fq = o.f(Rv.x[q], Rv.y[q]) * Rv.w[q];
      for (let i = 0; i < 3; i++) b[dofs[i]] += fq * lam[i];
    }
  }
  // surrogate boundary edges
  const topo = buildTopology(mesh.nodes, mesh.tris);
  const bEdges = [], bTri = [], bNormal = [], qPts = [], qCP = [];
  const G = gaussForDegree01(4), pen = alpha / h;
  for (let e = 0; e < topo.nEdge; e++) {
    const t1 = topo.edgeTris[2 * e], t2 = topo.edgeTris[2 * e + 1];
    const s1 = inSur[t1] === 1, s2 = t2 >= 0 && inSur[t2] === 1;
    if (s1 === s2) continue;
    const t = s1 ? t1 : t2, sgn = s1 ? 1 : -1; // topo normal points out of K⁻ = t1
    const nx = sgn * topo.normals[2 * e], ny = sgn * topo.normals[2 * e + 1], len = topo.lengths[e];
    bEdges.push(e); bTri.push(t); bNormal.push(nx, ny);
    const E = p1Element(mesh, t), dofs = [0, 1, 2].map((k) => dofOf[mesh.tris[3 * t + k]]);
    const va = topo.edges[2 * e], vb = topo.edges[2 * e + 1];
    const ax = mesh.nodes[2 * va], ay = mesh.nodes[2 * va + 1], bx = mesh.nodes[2 * vb], by = mesh.nodes[2 * vb + 1];
    const dn = [0, 1, 2].map((i) => E.gx[i] * nx + E.gy[i] * ny);
    K.fill(0);
    for (let q = 0; q < G.x.length; q++) {
      const x = ax + G.x[q] * (bx - ax), y = ay + G.x[q] * (by - ay), w = G.w[q] * len;
      const cp = o.ls.closest(x, y);
      qPts.push(x, y); qCP.push(cp.x, cp.y);
      const dx = shift ? cp.x - x : 0, dy = shift ? cp.y - y : 0;
      const gbar = o.g(cp.x, cp.y);
      const lam = E.bary(x, y);
      const S = [0, 1, 2].map((i) => lam[i] + E.gx[i] * dx + E.gy[i] * dy); // shifted basis φ + ∇φ·d
      for (let i = 0; i < 3; i++) {
        b[dofs[i]] += w * (-dn[i] * gbar + pen * S[i] * gbar);
        for (let j = 0; j < 3; j++) K[3 * i + j] += w * (-lam[i] * dn[j] - dn[i] * S[j] + pen * S[i] * S[j]);
      }
    }
    B.addBlock(dofs, dofs, K);
  }
  const A = B.toCSR();
  const F = sparseLU(A);
  const U = F.solve(b);
  return {
    mesh, topo, h, inSur, dofOf, nDof, A, b, U, singular: F.singular,
    bEdges: Int32Array.from(bEdges), bTri: Int32Array.from(bTri), bNormal: Float64Array.from(bNormal),
    qPts: Float64Array.from(qPts), qCP: Float64Array.from(qCP),
  };
}

/**
 * Errors on the surrogate domain Ω̃_h: ‖u − u_h‖_{L²(Ω̃)}, |u − u_h|_{H¹(Ω̃)} (degree-6 quadrature).
 * @returns {{L2: number, H1: number, area: number}}
 */
export function errorsSBM(sol, u, grad) {
  const { mesh } = sol, R = triangleRule(6);
  let e0 = 0, e1 = 0, area = 0;
  for (let t = 0; t < mesh.nTri; t++) {
    if (!sol.inSur[t]) continue;
    const E = p1Element(mesh, t), dofs = [0, 1, 2].map((k) => sol.dofOf[mesh.tris[3 * t + k]]);
    let ux = 0, uy = 0;
    for (let k = 0; k < 3; k++) { ux += sol.U[dofs[k]] * E.gx[k]; uy += sol.U[dofs[k]] * E.gy[k]; }
    const Rv = mapTriangleRule(R, E.verts[0], E.verts[1], E.verts[2]);
    area += E.area;
    for (let q = 0; q < Rv.n; q++) {
      const lam = E.bary(Rv.x[q], Rv.y[q]);
      let uh = 0;
      for (let k = 0; k < 3; k++) uh += lam[k] * sol.U[dofs[k]];
      const [gx, gy] = grad(Rv.x[q], Rv.y[q]);
      e0 += Rv.w[q] * (u(Rv.x[q], Rv.y[q]) - uh) ** 2;
      e1 += Rv.w[q] * ((gx - ux) ** 2 + (gy - uy) ** 2);
    }
  }
  return { L2: Math.sqrt(e0), H1: Math.sqrt(e1), area };
}
