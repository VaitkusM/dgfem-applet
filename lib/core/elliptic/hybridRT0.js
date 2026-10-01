/**
 * @file Hybridized RT0 mixed method (Arnold–Brezzi 1985) for
 *       −Δu = f, u = g on ∂Ω, with σ = −∇u, ∇·σ = f.
 *
 * Idea: take the RT0 space of mixedRT0.js but BREAK its normal continuity:
 * every triangle K owns three flux unknowns σ_{K,k} (the normal component on
 * its local edge k w.r.t. its OWN outward normal n_K). Normal continuity is
 * then re-imposed weakly by a Lagrange multiplier λ ∈ P0 on every interior edge:
 *
 *   on each K:  (σ_h, τ)_K − (u_h, ∇·τ)_K + ⟨λ_h, τ·n_K⟩_{∂K} = 0     ∀ τ ∈ RT0(K)
 *               (∇·σ_h, v)_K = (f, v)_K                               ∀ v ∈ P0(K)
 *   on each interior edge E:  Σ_{K∋E} ⟨σ_h·n_K, μ⟩_E = 0                ∀ μ ∈ P0(E)
 *   on boundary edges λ_h is known: λ_E = (1/|E|) ∫_E g.
 *
 * The last equation says σ_{K⁻}·n_E = σ_{K⁺}·n_E (the outward normals are
 * opposite), so the solution is EXACTLY the mixed RT0 solution; and comparing
 * with the integration by parts (σ,τ)_K − (u,∇·τ)_K + ⟨u, τ·n⟩ = 0 shows that λ
 * plays the role of u on the edges ("λ approximates the trace of u").
 *
 * Local matrices on K (local RT0 functions ψ_k = |E_k|/(2|K|)(x − x_opp,k)):
 *   A_K = (ψ_k, ψ_l)_K (3×3, SPD),  B_K = (∫_K ∇·ψ_k) = (|E_k|) (1×3),  C_K = diag(|E_k|)
 * Local system with D_K = [[A_K, −B_Kᵀ],[−B_K, 0]] (4×4, symmetric, invertible):
 *   D_K x_K = r_K − Ĉ_K λ_K,   x_K = (σ_K, u_K),  r_K = (0, −∫_K f),  Ĉ_K = [C_K; 0]
 * Static condensation (eliminate x_K element by element):
 *   x_K = D_K⁻¹ (r_K − Ĉ_K λ_K)          (local solver)
 *   Σ_K Ĉ_Kᵀ x_K = 0  ⇒  H λ = Σ_K Ĉ_Kᵀ D_K⁻¹ r_K,   H = Σ_K Ĉ_Kᵀ D_K⁻¹ Ĉ_K
 * H is symmetric positive definite (after removing the known boundary λ's):
 * locally Ĉᵀ D⁻¹ Ĉ = C (A⁻¹ − A⁻¹Bᵀ(BA⁻¹Bᵀ)⁻¹BA⁻¹) C is positive semidefinite
 * with kernel = constant λ_K; the Dirichlet edges remove the constants.
 */
import { SparseBuilder, csrMatVec } from '../la/sparse.js';
import { sparseLU } from '../la/direct.js';
import { buildTopology } from '../mesh/topology.js';
import { inverse } from '../la/dense.js';
import { rt0LocalMass, cellIntegrals, boundaryEdgeIntegrals } from './mixedRT0.js';

/** Unsigned local RT0 data (all three functions oriented by the outward normal of K). */
function localData(mesh, topo, t) {
  const { nodes, tris } = mesh;
  const verts = [0, 1, 2].map((k) => [nodes[2 * tris[3 * t + k]], nodes[2 * tris[3 * t + k] + 1]]);
  const area = 0.5 * ((verts[1][0] - verts[0][0]) * (verts[2][1] - verts[0][1]) - (verts[2][0] - verts[0][0]) * (verts[1][1] - verts[0][1]));
  const edges = [], len = [], c = [], opp = [];
  for (let k = 0; k < 3; k++) {
    const e = topo.triEdges[3 * t + k];
    edges.push(e); len.push(topo.lengths[e]); c.push(topo.lengths[e] / (2 * area)); opp.push(verts[(k + 2) % 3]);
  }
  return { area, edges, len, c, opp, verts, sign: [1, 1, 1] };
}

/**
 * Build the hybridized RT0 method: local matrices, condensed system, and solve.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {{f: (x:number,y:number)=>number, g?: (x:number,y:number)=>number}} o
 * @returns {{
 *   lambda: Float64Array, sigmaLocal: Float64Array, u: Float64Array, sigmaEdge: Float64Array,
 *   H: import('../la/sparse.js').CSR, rhsH: Float64Array, lamIndex: Int32Array, nLam: number,
 *   topo: import('../mesh/topology.js').Topology, nTri: number, F: Float64Array}}
 *   lambda[E] = multiplier on every edge (boundary edges: mean of g);
 *   sigmaLocal[3t+k] = σ_h·n_K on local edge k of triangle t (outward normal of t);
 *   sigmaEdge[E] = σ_h·n_E (global normal, taken from K⁻) — directly comparable with mixed RT0;
 *   lamIndex[E] = row of H for interior edge E (−1 for boundary edges).
 */
export function solveHybridRT0(mesh, o) {
  const topo = buildTopology(mesh.nodes, mesh.tris), nT = mesh.tris.length / 3, nE = topo.nEdge;
  const g = o.g ?? (() => 0);
  const F = cellIntegrals(mesh, o.f), Gb = boundaryEdgeIntegrals(mesh, topo, g);
  const lamIndex = new Int32Array(nE).fill(-1);
  let nLam = 0;
  for (let e = 0; e < nE; e++) if (!topo.isBoundary[e]) lamIndex[e] = nLam++;
  const lambda = new Float64Array(nE);
  for (let e = 0; e < nE; e++) if (topo.isBoundary[e]) lambda[e] = Gb[e] / topo.lengths[e];

  const Hb = new SparseBuilder(nLam, nLam, nT * 9), rhsH = new Float64Array(nLam);
  const Dinv = new Float64Array(16 * nT);
  for (let t = 0; t < nT; t++) {
    const L = localData(mesh, topo, t), A = rt0LocalMass(L);
    const D = new Float64Array(16);
    for (let k = 0; k < 3; k++) {
      for (let l = 0; l < 3; l++) D[4 * k + l] = A[3 * k + l];
      D[4 * k + 3] = -L.len[k]; D[12 + k] = -L.len[k];
    }
    const Di = inverse(D, 4);
    Dinv.set(Di, 16 * t);
    // r_K with the known boundary multipliers moved to the right: r = (−C_b λ_b, −F)
    const r = [0, 0, 0, -F[t]];
    for (let k = 0; k < 3; k++) if (topo.isBoundary[L.edges[k]]) r[k] -= L.len[k] * lambda[L.edges[k]];
    // H_K[k][l] = C_k (D⁻¹)_{kl} C_l;   rhs_k = C_k (D⁻¹ r)_k   (interior edges only)
    for (let k = 0; k < 3; k++) {
      const I = lamIndex[L.edges[k]];
      if (I < 0) continue;
      let s = 0; for (let j = 0; j < 4; j++) s += Di[4 * k + j] * r[j];
      rhsH[I] += L.len[k] * s;
      for (let l = 0; l < 3; l++) {
        const J = lamIndex[L.edges[l]];
        if (J >= 0) Hb.add(I, J, L.len[k] * Di[4 * k + l] * L.len[l]);
      }
    }
  }
  const H = Hb.toCSR();
  const lamI = nLam > 0 ? sparseLU(H).solve(rhsH) : new Float64Array(0);
  for (let e = 0; e < nE; e++) if (lamIndex[e] >= 0) lambda[e] = lamI[lamIndex[e]];

  // local recovery x_K = D_K⁻¹ (r_K − Ĉ_K λ_K)
  const sigmaLocal = new Float64Array(3 * nT), u = new Float64Array(nT);
  for (let t = 0; t < nT; t++) {
    const r = [0, 0, 0, -F[t]];
    for (let k = 0; k < 3; k++) { const e = topo.triEdges[3 * t + k]; r[k] -= topo.lengths[e] * lambda[e]; }
    const x = [0, 0, 0, 0];
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) x[i] += Dinv[16 * t + 4 * i + j] * r[j];
    sigmaLocal.set(x.slice(0, 3), 3 * t); u[t] = x[3];
  }
  const sigmaEdge = new Float64Array(nE);
  for (let e = 0; e < nE; e++) sigmaEdge[e] = sigmaLocal[3 * topo.edgeTris[2 * e] + topo.edgeLocal[2 * e]];
  return { lambda, sigmaLocal, u, sigmaEdge, H, rhsH, lamIndex, nLam, topo, nTri: nT, F };
}

/**
 * The un-condensed ("monolithic") hybridized system, for illustration and
 * verification. Unknown layout (element-major, then multipliers):
 *   [σ_{0,0..2}, u_0, σ_{1,0..2}, u_1, …, σ_{nT−1,·}, u_{nT−1} | λ_interior]
 * Matrix (symmetric, indefinite):
 *   [ D (block diagonal, 4×4 blocks)   Ĉ ] [x]   [r]
 *   [ Ĉᵀ                               0 ] [λ] = [0]
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {{f: (x:number,y:number)=>number, g?: (x:number,y:number)=>number}} o
 * @returns {{A: import('../la/sparse.js').CSR, rhs: Float64Array, nTri: number, mLoc: number, nLam: number,
 *            fields: Array<{name: string, size: number}>, lamIndex: Int32Array}}
 */
export function hybridRT0Monolithic(mesh, o) {
  const topo = buildTopology(mesh.nodes, mesh.tris), nT = mesh.tris.length / 3, nE = topo.nEdge;
  const g = o.g ?? (() => 0);
  const F = cellIntegrals(mesh, o.f), Gb = boundaryEdgeIntegrals(mesh, topo, g);
  const lamIndex = new Int32Array(nE).fill(-1);
  let nLam = 0;
  for (let e = 0; e < nE; e++) if (!topo.isBoundary[e]) lamIndex[e] = nLam++;
  const n = 4 * nT + nLam, B = new SparseBuilder(n, n, nT * 30), rhs = new Float64Array(n);
  for (let t = 0; t < nT; t++) {
    const L = localData(mesh, topo, t), A = rt0LocalMass(L), o4 = 4 * t;
    for (let k = 0; k < 3; k++) {
      for (let l = 0; l < 3; l++) B.add(o4 + k, o4 + l, A[3 * k + l]);
      B.add(o4 + k, o4 + 3, -L.len[k]); B.add(o4 + 3, o4 + k, -L.len[k]);
      const I = lamIndex[L.edges[k]];
      if (I >= 0) { B.add(o4 + k, 4 * nT + I, L.len[k]); B.add(4 * nT + I, o4 + k, L.len[k]); }
      else rhs[o4 + k] -= Gb[L.edges[k]]; // C λ_b = |E| · (mean g) = ∫_E g
    }
    rhs[o4 + 3] = -F[t];
  }
  return { A: B.toCSR(), rhs, nTri: nT, mLoc: 4, nLam, fields: [{ name: 'σ', size: 3 }, { name: 'u', size: 1 }], lamIndex };
}

/**
 * Solve the monolithic hybridized system directly (for the equivalence test).
 * @returns {{lambda: Float64Array, u: Float64Array, sigmaLocal: Float64Array}} lambda on interior edges only (H ordering)
 */
export function solveHybridRT0Monolithic(mesh, o) {
  const S = hybridRT0Monolithic(mesh, o);
  const x = sparseLU(S.A).solve(S.rhs);
  const u = new Float64Array(S.nTri), sigmaLocal = new Float64Array(3 * S.nTri);
  for (let t = 0; t < S.nTri; t++) { u[t] = x[4 * t + 3]; for (let k = 0; k < 3; k++) sigmaLocal[3 * t + k] = x[4 * t + k]; }
  const r = csrMatVec(S.A, x);
  let res = 0; for (let i = 0; i < S.A.n; i++) res = Math.max(res, Math.abs(r[i] - S.rhs[i]));
  return { lambda: x.slice(4 * S.nTri), u, sigmaLocal, residual: res };
}

/** Mean of a function over every edge, (1/|E|)∫_E u ds (Gauss, degree 8). */
export function edgeMeans(mesh, topo, u) {
  const I = boundaryEdgeIntegralsAll(mesh, topo, u);
  for (let e = 0; e < topo.nEdge; e++) I[e] /= topo.lengths[e];
  return I;
}

function boundaryEdgeIntegralsAll(mesh, topo, g) {
  // same as boundaryEdgeIntegrals but on all edges
  const fake = { ...topo, isBoundary: new Uint8Array(topo.nEdge).fill(1) };
  return boundaryEdgeIntegrals(mesh, fake, g);
}
