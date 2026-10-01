/**
 * @file Hybridizable discontinuous Galerkin (HDG, "LDG-H") method of degree k
 *       for −Δu = f in Ω, u = g on ∂Ω, on triangle meshes
 *       (Cockburn, Gopalakrishnan & Lazarov 2009).
 *
 * First-order form (AGENTS.md convention):  q = −∇u  (flux),  ∇·q = f.
 * Unknowns:  q_h ∈ P_k(K)², u_h ∈ P_k(K) on every triangle (broken, Dubiner basis),
 *            λ_h ∈ P_k(F) on every edge (the "numerical trace" û = λ).
 * Numerical flux (n = outward normal of K, τ > 0 the stabilisation parameter):
 *            q̂·n = q_h·n + τ (u_h − λ_h)     on ∂K.
 * Local equations on each K, for all r ∈ P_k(K)², w ∈ P_k(K):
 *   (E1)  (q_h, r)_K − (u_h, ∇·r)_K + ⟨λ_h, r·n⟩_{∂K} = 0
 *   (E2)  −(q_h, ∇w)_K + ⟨q_h·n + τ(u_h − λ_h), w⟩_{∂K} = (f, w)_K
 * Global (transmission) equation on each interior edge F, for all μ ∈ P_k(F):
 *   (E3)  Σ_{K∋F} ⟨q_h·n_K + τ(u_h − λ_h), μ⟩_F = 0   (the numerical flux is single-valued)
 * Boundary edges: λ_h = P_k g (L² projection onto P_k(F)), moved to the right-hand side.
 *
 * Implementation: (E2) is multiplied by −1, which makes the local matrix
 *     A_K = [ M   0   −G_x ]        M_ij = (ψ_j, ψ_i)_K,  (G_x)_ij = (ψ_j, ∂_xψ_i)_K,
 *           [ 0   M   −G_y ]        integration by parts: G_x + G_xᵀ = N_x = ⟨n_x ψ_j, ψ_i⟩_{∂K}
 *           [ −G_xᵀ −G_yᵀ −τE ]     E_ij = ⟨ψ_j, ψ_i⟩_{∂K}
 * symmetric, and the λ-coupling of (E1),(E2) is exactly the transpose of the
 * x-part of (E3):  C_K (rows: local unknowns, columns: local trace modes),
 *     (C_K)[q_x,i ; F,a] = ⟨μ_a, ψ_i n_x⟩_F,  (C_K)[u,i ; F,a] = τ ⟨μ_a, ψ_i⟩_F,
 * (E3) reads Σ_K ( C_Kᵀ x_K − τ ⟨λ, μ⟩_F ) = 0.  Local solver ("static condensation"):
 *     x_K = A_K⁻¹ ( b_K − C_K λ_K ),     b_K = (0, 0, −(f, ψ_i)_K)
 * Substituting into (E3) gives the global trace system, symmetric positive definite:
 *     𝕂 λ = r,   𝕂 = Σ_K ( C_Kᵀ A_K⁻¹ C_K + τ ⟨μ_b, μ_a⟩_{∂K} ),   r = Σ_K C_Kᵀ A_K⁻¹ b_K
 * (entries for known boundary modes moved to r).
 *
 * Trace basis on edge F: orthonormal Legendre polynomials in the edge parameter s ∈ [0,1],
 * μ_a(s) = √(2a+1) P_a(2s − 1), a = 0..k, with s oriented as in K⁻ (see dgtri.js), so
 * ⟨μ_a, μ_b⟩_F = |F| δ_ab. Global trace unknown numbering: lamIndex[F]·(k+1) + a.
 *
 * Post-processing (Stenberg-type, Cockburn et al.): on each K find u*_h ∈ P_{k+1}(K) with
 *     (∇u*_h, ∇w)_K = −(q_h, ∇w)_K   ∀ w ∈ P_{k+1}(K),    (u*_h, 1)_K = (u_h, 1)_K.
 * (The minus sign because q = −∇u.) For τ = O(1), u_h, q_h converge with order k+1
 * and u*_h with order k+2 in L².
 */
import { SparseBuilder, csrMatVec } from '../la/sparse.js';
import { sparseLU } from '../la/direct.js';
import { inverse, solve } from '../la/dense.js';
import { buildTopology, refEdgePoint } from '../mesh/topology.js';
import { legendreP } from '../basis/legendre.js';
import { dimP } from '../basis/simplex.js';
import { triGeometry, basisTables, physGrads, mapPoint, brokenErrors } from './dgtri.js';

/** Orthonormal Legendre value μ_a(s) on [0,1]. */
export const edgeMode = (a, s) => Math.sqrt(2 * a + 1) * legendreP(a, 2 * s - 1);

/**
 * Local HDG matrices of every element (A_K⁻¹, C_K, b_K) and bookkeeping.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {{k: number, tau?: number, f: (x:number,y:number)=>number, g?: (x:number,y:number)=>number}} o
 */
function setup(mesh, o) {
  const k = o.k, tau = o.tau ?? 1, g = o.g ?? (() => 0);
  const geo = triGeometry(mesh), topo = buildTopology(mesh.nodes, mesh.tris);
  const T = basisTables(k, 2 * k + 4, 2 * k + 4), n = T.n, E = T.edge, nq = E.nq, nT = geo.nTri;
  const m = 3 * n, ne = k + 1, mc = 3 * ne; // local unknowns, trace modes per edge, per element
  // edge mode table mu[q*ne + a]
  const mu = new Float64Array(nq * ne);
  for (let q = 0; q < nq; q++) for (let a = 0; a < ne; a++) mu[q * ne + a] = edgeMode(a, E.s[q]);
  // interior-edge numbering and known boundary traces λ = P_k g
  const lamIndex = new Int32Array(topo.nEdge).fill(-1);
  let nLamEdges = 0;
  for (let e = 0; e < topo.nEdge; e++) if (!topo.isBoundary[e]) lamIndex[e] = nLamEdges++;
  const lamAll = new Float64Array(topo.nEdge * ne); // trace coefficients on every edge
  for (let e = 0; e < topo.nEdge; e++) {
    if (!topo.isBoundary[e]) continue;
    const t = topo.edgeTris[2 * e], kk = topo.edgeLocal[2 * e];
    for (let q = 0; q < nq; q++) {
      const [x, y] = mapPoint(geo, t, ...refEdgePoint(kk, E.s[q])), gq = g(x, y);
      for (let a = 0; a < ne; a++) lamAll[e * ne + a] += E.w[q] * gq * mu[q * ne + a];
    }
  }
  const Ainv = new Float64Array(nT * m * m), C = new Float64Array(nT * m * mc), bK = new Float64Array(nT * m);
  const Aloc = new Float64Array(m * m);
  const gx = new Float64Array(n), gy = new Float64Array(n);
  const R = T.vol.rule;
  for (let t = 0; t < nT; t++) {
    Aloc.fill(0);
    const det = geo.det[t];
    // volume: M (orthonormal ⇒ det·I), −G_x, −G_y and their transposes, load
    for (let i = 0; i < n; i++) { Aloc[i * m + i] = det; Aloc[(n + i) * m + n + i] = det; }
    for (let q = 0; q < R.n; q++) {
      physGrads(geo.Jinv, t, T.vol.Dx, T.vol.Dy, q, n, gx, gy);
      const w = R.w[q] * det, [x, y] = mapPoint(geo, t, R.x[q], R.y[q]), fq = o.f(x, y);
      for (let i = 0; i < n; i++) {
        bK[t * m + 2 * n + i] -= w * fq * T.vol.V[q * n + i];
        for (let j = 0; j < n; j++) {
          const vj = T.vol.V[q * n + j];
          // (G_x)_ij = (ψ_j, ∂xψ_i): row q_x,i / col u_j gets −G_x; row u_j / col q_x,i gets −G_x (transpose)
          Aloc[i * m + 2 * n + j] -= w * vj * gx[i];
          Aloc[(n + i) * m + 2 * n + j] -= w * vj * gy[i];
          Aloc[(2 * n + j) * m + i] -= w * vj * gx[i];
          Aloc[(2 * n + j) * m + n + i] -= w * vj * gy[i];
        }
      }
    }
    // boundary of K: −τE and the trace coupling C
    for (let kk = 0; kk < 3; kk++) {
      const e = topo.triEdges[3 * t + kk], minus = topo.edgeTris[2 * e] === t;
      const sgn = minus ? 1 : -1, nx = sgn * topo.normals[2 * e], ny = sgn * topo.normals[2 * e + 1], len = topo.lengths[e];
      const V = E.V[kk][minus ? 0 : 1];
      for (let q = 0; q < nq; q++) {
        const w = E.w[q] * len;
        for (let i = 0; i < n; i++) {
          const vi = V[q * n + i];
          for (let j = 0; j < n; j++) Aloc[(2 * n + i) * m + 2 * n + j] -= w * tau * vi * V[q * n + j];
          for (let a = 0; a < ne; a++) {
            const c = w * vi * mu[q * ne + a], col = kk * ne + a;
            C[t * m * mc + i * mc + col] += c * nx;
            C[t * m * mc + (n + i) * mc + col] += c * ny;
            C[t * m * mc + (2 * n + i) * mc + col] += c * tau;
          }
        }
      }
    }
    Ainv.set(inverse(Aloc, m), t * m * m);
  }
  return { k, tau, geo, topo, T, n, m, ne, mc, nT, lamIndex, nLamEdges, lamAll, Ainv, C, bK, mu };
}

/** Global trace DOF of local trace column (edge kk, mode a) of triangle t, or −1 for boundary edges. */
const traceDof = (S, t, kk, a) => {
  const I = S.lamIndex[S.topo.triEdges[3 * t + kk]];
  return I < 0 ? -1 : I * S.ne + a;
};

/**
 * Assemble the condensed trace system 𝕂 λ = r (SPD).
 * @returns {{K: import('../la/sparse.js').CSR, r: Float64Array}}
 */
function condense(S) {
  const { m, mc, nT, ne, tau } = S, nL = S.nLamEdges * ne;
  const Kb = new SparseBuilder(nL, nL, nT * mc * mc), r = new Float64Array(nL);
  const AC = new Float64Array(m * mc), Ab = new Float64Array(m);
  for (let t = 0; t < nT; t++) {
    const Ai = S.Ainv.subarray(t * m * m, (t + 1) * m * m), C = S.C.subarray(t * m * mc, (t + 1) * m * mc);
    // AC = A⁻¹ C,  Ab = A⁻¹ b
    AC.fill(0); Ab.fill(0);
    for (let i = 0; i < m; i++) for (let l = 0; l < m; l++) {
      const a = Ai[i * m + l];
      if (a === 0) continue;
      Ab[i] += a * S.bK[t * m + l];
      for (let c = 0; c < mc; c++) AC[i * mc + c] += a * C[l * mc + c];
    }
    // known boundary traces: contribution of λ_b to the local solution
    const lamLoc = new Float64Array(mc);
    for (let kk = 0; kk < 3; kk++) {
      const e = S.topo.triEdges[3 * t + kk];
      if (S.lamIndex[e] < 0) for (let a = 0; a < ne; a++) lamLoc[kk * ne + a] = S.lamAll[e * ne + a];
    }
    for (let c1 = 0; c1 < mc; c1++) {
      const I = traceDof(S, t, Math.floor(c1 / ne), c1 % ne);
      if (I < 0) continue;
      // row of 𝕂: Cᵀ A⁻¹ C + τ ⟨μ,μ⟩ (diagonal per edge),  r = Cᵀ A⁻¹ b − (that row)·λ_b
      let rr = 0;
      for (let i = 0; i < m; i++) rr += C[i * mc + c1] * Ab[i];
      for (let c2 = 0; c2 < mc; c2++) {
        let v = 0;
        for (let i = 0; i < m; i++) v += C[i * mc + c1] * AC[i * mc + c2];
        if (c2 === c1) v += tau * S.topo.lengths[S.topo.triEdges[3 * t + Math.floor(c1 / ne)]];
        const J = traceDof(S, t, Math.floor(c2 / ne), c2 % ne);
        if (J >= 0) Kb.add(I, J, v); else rr -= v * lamLoc[c2];
      }
      r[I] += rr;
    }
  }
  return { K: Kb.toCSR(), r };
}

/**
 * Local solver: x_K = A_K⁻¹ (b_K − C_K λ_K) for every element.
 * @returns {{q: Float64Array, u: Float64Array}} qx, qy, u coefficients: q[2(t*n+i)] = q_x, q[2(t*n+i)+1] = q_y; u[t*n+i]
 */
function recover(S, lamAll) {
  const { m, mc, nT, n, ne } = S;
  const q = new Float64Array(2 * nT * n), u = new Float64Array(nT * n), rhs = new Float64Array(m);
  for (let t = 0; t < nT; t++) {
    const Ai = S.Ainv.subarray(t * m * m, (t + 1) * m * m), C = S.C.subarray(t * m * mc, (t + 1) * m * mc);
    for (let i = 0; i < m; i++) {
      let s = S.bK[t * m + i];
      for (let c = 0; c < mc; c++) s -= C[i * mc + c] * lamAll[S.topo.triEdges[3 * t + Math.floor(c / ne)] * ne + (c % ne)];
      rhs[i] = s;
    }
    for (let i = 0; i < m; i++) {
      let s = 0; for (let l = 0; l < m; l++) s += Ai[i * m + l] * rhs[l];
      if (i < n) q[2 * (t * n + i)] = s; else if (i < 2 * n) q[2 * (t * n + i - n) + 1] = s; else u[t * n + i - 2 * n] = s;
    }
  }
  return { q, u };
}

/**
 * Solve the HDG discretisation by static condensation.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {{k: number, tau?: number, f: (x:number,y:number)=>number, g?: (x:number,y:number)=>number}} o
 *   k ≥ 0 polynomial degree, tau > 0 (default 1)
 * @returns {{k: number, tau: number, u: Float64Array, q: Float64Array, lambda: Float64Array, ustar: Float64Array,
 *            K: import('../la/sparse.js').CSR, r: Float64Array, nTrace: number, nLocal: number,
 *            mesh: object, topo: import('../mesh/topology.js').Topology}}
 *   u[t*n+i] Dubiner coefficients of u_h (n = dimP(k)); q[2(t*n+i)+c] of component c of q_h;
 *   lambda[e*(k+1)+a] trace coefficients on EVERY edge (boundary: P_k g);
 *   ustar[t*n'+i] coefficients of u*_h in P_{k+1} (n' = dimP(k+1)).
 */
export function solveHDG(mesh, o) {
  const S = setup(mesh, o);
  const { K, r } = condense(S);
  const lamI = K.n > 0 ? sparseLU(K).solve(r) : new Float64Array(0);
  const lamAll = Float64Array.from(S.lamAll);
  for (let e = 0; e < S.topo.nEdge; e++) {
    const I = S.lamIndex[e];
    if (I >= 0) for (let a = 0; a < S.ne; a++) lamAll[e * S.ne + a] = lamI[I * S.ne + a];
  }
  const { q, u } = recover(S, lamAll);
  const ustar = postProcess(mesh, S.k, q, u);
  return { k: S.k, tau: S.tau, u, q, lambda: lamAll, ustar, K, r, nTrace: K.n, nLocal: S.nT * S.m, mesh, topo: S.topo };
}

/**
 * The monolithic (un-condensed) HDG system, symmetric:
 *     [ A (block diag)   C ] [x]   [b]
 *     [ Cᵀ              −τ⟨μ,μ⟩ ] [λ] = [0]   (+ known boundary traces moved right)
 * Unknown layout: element-major local unknowns [q_x(n), q_y(n), u(n)] per triangle, then interior traces.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {{k: number, tau?: number, f: (x:number,y:number)=>number, g?: (x:number,y:number)=>number}} o
 * @returns {{A: import('../la/sparse.js').CSR, rhs: Float64Array, nTri: number, mLoc: number, nLam: number,
 *            fields: Array<{name: string, size: number}>}}
 */
export function hdgMonolithic(mesh, o) {
  const S = setup(mesh, o);
  const { m, mc, nT, ne, n, tau } = S, nL = S.nLamEdges * ne, N = nT * m + nL;
  const B = new SparseBuilder(N, N, nT * (m * m + 2 * m * mc)), rhs = new Float64Array(N);
  for (let t = 0; t < nT; t++) {
    const A = inverse(S.Ainv.subarray(t * m * m, (t + 1) * m * m), m); // recover A_K (small)
    for (let i = 0; i < m; i++) {
      rhs[t * m + i] = S.bK[t * m + i];
      for (let j = 0; j < m; j++) { const v = A[i * m + j]; if (Math.abs(v) > 1e-14 * S.geo.det[t]) B.add(t * m + i, t * m + j, v); }
    }
    for (let c = 0; c < mc; c++) {
      const kk = Math.floor(c / ne), a = c % ne, e = S.topo.triEdges[3 * t + kk];
      const J = traceDof(S, t, kk, a);
      for (let i = 0; i < m; i++) {
        const v = S.C[t * m * mc + i * mc + c];
        if (v === 0) continue;
        if (J >= 0) { B.add(t * m + i, nT * m + J, v); B.add(nT * m + J, t * m + i, v); }
        else rhs[t * m + i] -= v * S.lamAll[e * ne + a];
      }
      if (J >= 0) B.add(nT * m + J, nT * m + J, -tau * S.topo.lengths[e]);
    }
  }
  return { A: B.toCSR(), rhs, nTri: nT, mLoc: m, nLam: nL, fields: [{ name: 'q_x', size: n }, { name: 'q_y', size: n }, { name: 'u', size: n }] };
}

/**
 * Solve the monolithic system directly (verification of the condensation).
 * @returns {{u: Float64Array, lambda: Float64Array, residual: number}} u as in solveHDG; lambda = interior traces
 */
export function solveHDGMonolithic(mesh, o) {
  const M = hdgMonolithic(mesh, o);
  const x = sparseLU(M.A).solve(M.rhs);
  const n = M.mLoc / 3, u = new Float64Array(M.nTri * n);
  for (let t = 0; t < M.nTri; t++) for (let i = 0; i < n; i++) u[t * n + i] = x[t * M.mLoc + 2 * n + i];
  const y = csrMatVec(M.A, x);
  let res = 0; for (let i = 0; i < M.A.n; i++) res = Math.max(res, Math.abs(y[i] - M.rhs[i]));
  return { u, lambda: x.slice(M.nTri * M.mLoc), residual: res };
}

/**
 * Element-wise post-processing u*_h ∈ P_{k+1}(K):
 *   (∇u*, ∇w)_K = −(q_h, ∇w)_K  for w ∈ P_{k+1}, w ⊥ 1;   mean(u*) = mean(u_h).
 * With the orthonormal Dubiner basis, mode 0 is the constant √2 and modes m ≥ 1 have zero mean,
 * so the mean condition fixes coefficient 0 (equal to u_h's) and the Neumann problem the others.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {number} k
 * @param {Float64Array} q coefficients of q_h (layout of solveHDG)
 * @param {Float64Array} u coefficients of u_h
 * @returns {Float64Array} coefficients of u*_h, layout t*dimP(k+1) + i
 */
export function postProcess(mesh, k, q, u) {
  const geo = triGeometry(mesh), n = dimP(k), n1 = dimP(k + 1);
  const T = basisTables(k + 1, 2 * k + 2, 1), R = T.vol.rule;
  const out = new Float64Array(geo.nTri * n1);
  const gx = new Float64Array(n1), gy = new Float64Array(n1);
  const K = new Float64Array((n1 - 1) * (n1 - 1)), b = new Float64Array(n1 - 1);
  for (let t = 0; t < geo.nTri; t++) {
    K.fill(0); b.fill(0);
    for (let qq = 0; qq < R.n; qq++) {
      physGrads(geo.Jinv, t, T.vol.Dx, T.vol.Dy, qq, n1, gx, gy);
      const w = R.w[qq] * geo.det[t];
      // q_h at this point (P_k Dubiner modes = first n modes of the P_{k+1} table)
      let qx = 0, qy = 0;
      for (let i = 0; i < n; i++) { const v = T.vol.V[qq * n1 + i]; qx += q[2 * (t * n + i)] * v; qy += q[2 * (t * n + i) + 1] * v; }
      for (let i = 1; i < n1; i++) {
        b[i - 1] -= w * (qx * gx[i] + qy * gy[i]);
        for (let j = 1; j < n1; j++) K[(i - 1) * (n1 - 1) + j - 1] += w * (gx[i] * gx[j] + gy[i] * gy[j]);
      }
    }
    const c = solve(K, n1 - 1, b);
    out[t * n1] = u[t * n];
    for (let i = 1; i < n1; i++) out[t * n1 + i] = c[i - 1];
  }
  return out;
}

/**
 * L² errors of an HDG solution against exact u (and q = −∇u).
 * @param {ReturnType<typeof solveHDG>} H
 * @param {(x:number,y:number)=>number} u
 * @param {(x:number,y:number)=>[number,number]} grad
 * @returns {{u: number, q: number, ustar: number}}
 */
export function hdgErrors(H, u, grad) {
  const n = dimP(H.k), nT = H.u.length / n;
  const qx = new Float64Array(nT * n), qy = new Float64Array(nT * n);
  for (let i = 0; i < nT * n; i++) { qx[i] = H.q[2 * i]; qy[i] = H.q[2 * i + 1]; }
  const eu = brokenErrors(H.mesh, H.k, H.u, u).L2;
  const ex = brokenErrors(H.mesh, H.k, qx, (x, y) => -grad(x, y)[0]).L2;
  const ey = brokenErrors(H.mesh, H.k, qy, (x, y) => -grad(x, y)[1]).L2;
  const es = brokenErrors(H.mesh, H.k + 1, H.ustar, u).L2;
  return { u: eu, q: Math.hypot(ex, ey), ustar: es };
}
