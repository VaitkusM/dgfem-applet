/**
 * @file Interior penalty discontinuous Galerkin (IPDG) methods for the
 *       Poisson problem  −Δu = f in Ω,  u = g on ∂Ω,  on triangle meshes.
 *
 * Discrete space: broken polynomials of degree p (Dubiner basis, see dgtri.js).
 * Conventions (AGENTS.md): on a face F with normal n pointing out of K⁻,
 *     ⟦v⟧ = v⁻ − v⁺,     {{w}} = ½ (w⁻ + w⁺).
 * On a boundary face: ⟦v⟧ = v⁻, {{w}} = w⁻, n = outward normal.
 *
 * Bilinear form and right-hand side (θ ∈ {1, −1, 0}):
 *   a(u,v) = Σ_K ∫_K ∇u·∇v
 *          − Σ_F ∫_F {{∇u}}·n ⟦v⟧              (consistency: comes from integrating by parts)
 *          − θ Σ_F ∫_F {{∇v}}·n ⟦u⟧            (symmetrisation; θ = 1 SIPG, −1 NIPG, 0 IIPG)
 *          + Σ_F σ_F ∫_F ⟦u⟧ ⟦v⟧               (penalty)
 *   ℓ(v)   = ∫_Ω f v + Σ_{F⊂∂Ω} ∫_F ( −θ ∇v·n + σ_F v ) g     (Nitsche on boundary faces)
 *
 * Penalty parameter (documented choice):
 *     σ_F = C_IP · (p+1)(p+2)/2 · |F| / min(|K⁻|, |K⁺|)  =  C_IP · (p+1)(p+2) / h_F ,
 *     h_F = 2 min(|K⁻|,|K⁺|) / |F|   (the smaller triangle height over F; only K⁻ on ∂Ω).
 * The factor (p+1)(p+2)/2 · |F|/|K| is the sharp constant of the polynomial trace
 * inequality ‖v‖²_F ≤ (p+1)(p+2)/2 · |F|/|K| · ‖v‖²_K on triangles (Warburton &
 * Hesthaven 2003), so σ_F scales like p²/h as the coercivity argument requires.
 *
 * Unknowns: U[t*n + m] = coefficient of Dubiner mode m on triangle t, n = (p+1)(p+2)/2.
 */
import { SparseBuilder, csrMatVec } from '../la/sparse.js';
import { sparseLU } from '../la/direct.js';
import { lanczosRitz } from '../la/eig.js';
import { asOperator } from '../la/krylov.js';
import { triGeometry, basisTables, physGrads, mapPoint, brokenErrors } from './dgtri.js';
import { buildTopology, refEdgePoint } from '../mesh/topology.js';

/** θ for the three classical variants. */
export const IP_THETA = { sipg: 1, nipg: -1, iipg: 0 };

/**
 * Penalty σ_F of every edge.
 * @param {import('../mesh/topology.js').Topology} topo
 * @param {Float64Array} area triangle areas
 * @param {number} p
 * @param {number} Cip
 * @param {(hF: number, p: number, Cip: number) => number} [sigmaFn] custom penalty as a function of
 *   h_F = 2 min(|K⁻|,|K⁺|)/|F|; default  Cip · (p+1)(p+2) / h_F
 * @returns {Float64Array} σ_F per edge
 */
export function penaltyValues(topo, area, p, Cip, sigmaFn) {
  const s = new Float64Array(topo.nEdge);
  const fn = sigmaFn ?? ((hF) => Cip * (p + 1) * (p + 2) / hF);
  for (let e = 0; e < topo.nEdge; e++) {
    const tm = topo.edgeTris[2 * e], tp = topo.edgeTris[2 * e + 1];
    const amin = tp >= 0 ? Math.min(area[tm], area[tp]) : area[tm];
    s[e] = fn(2 * amin / topo.lengths[e], p, Cip);
  }
  return s;
}

/**
 * Assemble the IPDG matrix and load vector.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {{p: number, f: (x:number,y:number)=>number, g?: (x:number,y:number)=>number,
 *          theta?: number, variant?: 'sipg'|'nipg'|'iipg', Cip?: number,
 *          sigmaFn?: (hF: number, p: number, Cip: number) => number}} o
 *   theta overrides variant; Cip defaults to 2 (see penaltyValues); g defaults to 0
 * @returns {{A: import('../la/sparse.js').CSR, b: Float64Array, n: number, nDof: number, p: number, theta: number,
 *            sigma: Float64Array, topo: import('../mesh/topology.js').Topology, geo: ReturnType<typeof triGeometry>, mesh: object}}
 */
export function assembleIP(mesh, o) {
  const p = o.p, theta = o.theta ?? IP_THETA[o.variant ?? 'sipg'], Cip = o.Cip ?? 2;
  const g = o.g ?? (() => 0);
  const geo = triGeometry(mesh), topo = buildTopology(mesh.nodes, mesh.tris);
  const T = basisTables(p, 2 * p + 4, 2 * p + 4), n = T.n, nT = geo.nTri, nDof = nT * n;
  const sigma = penaltyValues(topo, geo.area, p, Cip, o.sigmaFn);
  const B = new SparseBuilder(nDof, nDof, nT * n * n * 7);
  const b = new Float64Array(nDof);
  const gx = new Float64Array(n), gy = new Float64Array(n);
  const K = new Float64Array(n * n);
  const dofs = (t) => Int32Array.from({ length: n }, (_, m) => t * n + m);

  // --- volume terms: ∫_K ∇u·∇v and ∫_K f v --------------------------------
  const R = T.vol.rule;
  for (let t = 0; t < nT; t++) {
    K.fill(0);
    for (let q = 0; q < R.n; q++) {
      physGrads(geo.Jinv, t, T.vol.Dx, T.vol.Dy, q, n, gx, gy);
      const w = R.w[q] * geo.det[t];
      const [x, y] = mapPoint(geo, t, R.x[q], R.y[q]), fq = o.f(x, y);
      for (let i = 0; i < n; i++) {
        b[t * n + i] += w * fq * T.vol.V[q * n + i];
        for (let j = 0; j < n; j++) K[i * n + j] += w * (gx[i] * gx[j] + gy[i] * gy[j]);
      }
    }
    B.addBlock(dofs(t), dofs(t), K);
  }

  // --- face terms -----------------------------------------------------------
  const E = T.edge, nq = E.nq;
  // side data: values v and normal derivatives dn = ∇v·n (n = face normal out of K⁻)
  const vm = new Float64Array(n), vp = new Float64Array(n), dm = new Float64Array(n), dp = new Float64Array(n);
  const Kmm = new Float64Array(n * n), Kmp = new Float64Array(n * n), Kpm = new Float64Array(n * n), Kpp = new Float64Array(n * n);
  for (let e = 0; e < topo.nEdge; e++) {
    const tm = topo.edgeTris[2 * e], km = topo.edgeLocal[2 * e];
    const tp = topo.edgeTris[2 * e + 1], kp = topo.edgeLocal[2 * e + 1];
    const nx = topo.normals[2 * e], ny = topo.normals[2 * e + 1], len = topo.lengths[e], sg = sigma[e];
    const interior = tp >= 0;
    Kmm.fill(0); Kmp.fill(0); Kpm.fill(0); Kpp.fill(0);
    for (let q = 0; q < nq; q++) {
      const w = E.w[q] * len;
      physGrads(geo.Jinv, tm, E.Dx[km][0], E.Dy[km][0], q, n, gx, gy);
      for (let i = 0; i < n; i++) { vm[i] = E.V[km][0][q * n + i]; dm[i] = gx[i] * nx + gy[i] * ny; }
      if (interior) {
        // K⁺ sees the same physical point at its own parameter 1 − s (orientation 1)
        physGrads(geo.Jinv, tp, E.Dx[kp][1], E.Dy[kp][1], q, n, gx, gy);
        for (let i = 0; i < n; i++) { vp[i] = E.V[kp][1][q * n + i]; dp[i] = gx[i] * nx + gy[i] * ny; }
        // jump of basis function j: +v⁻_j (if on K⁻), −v⁺_j (if on K⁺); average of ∂_n: ½ d⁻_j, ½ d⁺_j
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          // entry = ∫ [ −{∂n φ_j}⟦φ_i⟧ − θ {∂n φ_i}⟦φ_j⟧ + σ ⟦φ_j⟧⟦φ_i⟧ ]
          Kmm[i * n + j] += w * (-0.5 * dm[j] * vm[i] - theta * 0.5 * dm[i] * vm[j] + sg * vm[j] * vm[i]);
          Kmp[i * n + j] += w * (-0.5 * dp[j] * vm[i] + theta * 0.5 * dm[i] * vp[j] - sg * vp[j] * vm[i]);
          Kpm[i * n + j] += w * (+0.5 * dm[j] * vp[i] - theta * 0.5 * dp[i] * vm[j] - sg * vm[j] * vp[i]);
          Kpp[i * n + j] += w * (+0.5 * dp[j] * vp[i] + theta * 0.5 * dp[i] * vp[j] + sg * vp[j] * vp[i]);
        }
      } else {
        // boundary face: Nitsche  −∂n u v − θ ∂n v u + σ u v   and   (−θ ∂n v + σ v) g
        const [x, y] = mapPoint(geo, tm, ...refEdgePoint(km, E.s[q])), gq = g(x, y);
        for (let i = 0; i < n; i++) {
          b[tm * n + i] += w * (-theta * dm[i] + sg * vm[i]) * gq;
          for (let j = 0; j < n; j++) Kmm[i * n + j] += w * (-dm[j] * vm[i] - theta * dm[i] * vm[j] + sg * vm[j] * vm[i]);
        }
      }
    }
    B.addBlock(dofs(tm), dofs(tm), Kmm);
    if (interior) {
      B.addBlock(dofs(tm), dofs(tp), Kmp);
      B.addBlock(dofs(tp), dofs(tm), Kpm);
      B.addBlock(dofs(tp), dofs(tp), Kpp);
    }
  }
  return { A: B.toCSR(), b, n, nDof, p, theta, sigma, topo, geo, mesh };
}

/**
 * Assemble and solve (RCM + banded LU with pivoting, so NIPG/IIPG and
 * indefinite SIPG matrices are handled too).
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {Parameters<typeof assembleIP>[1]} o
 * @returns {ReturnType<typeof assembleIP> & {U: Float64Array, singular: boolean, residual: number}}
 */
export function solveIP(mesh, o) {
  const S = assembleIP(mesh, o);
  const F = sparseLU(S.A);
  const U = F.solve(S.b);
  const r = csrMatVec(S.A, U);
  let res = 0, bn = 0;
  for (let i = 0; i < S.nDof; i++) { res = Math.max(res, Math.abs(r[i] - S.b[i])); bn = Math.max(bn, Math.abs(S.b[i])); }
  return { ...S, U, singular: F.singular, residual: res / (bn || 1) };
}

/**
 * Jumps of u_h across every edge: RMS over the edge of ⟦u_h⟧ = u⁻ − u⁺
 * (interior) or u_h − g (boundary), plus the penalty part of the DG energy norm.
 * @param {ReturnType<typeof solveIP>} S
 * @param {(x:number,y:number)=>number} [g] boundary data (0 if omitted)
 * @returns {{rms: Float64Array, penaltyEnergy: number}}
 *   rms[e] = (|F|⁻¹ ∫_F ⟦u_h⟧²)^{1/2};  penaltyEnergy = Σ_F σ_F ∫_F ⟦u_h⟧²  (with ⟦u_h⟧ = u_h − g on ∂Ω)
 */
export function edgeJumps(S, g) {
  const { topo, geo, p, n, U, sigma } = S;
  const T = basisTables(p, 1, 2 * p + 6), E = T.edge;
  const rms = new Float64Array(topo.nEdge);
  let pen = 0;
  for (let e = 0; e < topo.nEdge; e++) {
    const tm = topo.edgeTris[2 * e], km = topo.edgeLocal[2 * e], tp = topo.edgeTris[2 * e + 1], kp = topo.edgeLocal[2 * e + 1];
    let s2 = 0;
    for (let q = 0; q < E.nq; q++) {
      let um = 0, up = 0;
      for (let m = 0; m < n; m++) um += U[tm * n + m] * E.V[km][0][q * n + m];
      if (tp >= 0) for (let m = 0; m < n; m++) up += U[tp * n + m] * E.V[kp][1][q * n + m];
      else if (g) { const [x, y] = mapPoint(geo, tm, ...refEdgePoint(km, E.s[q])); up = g(x, y); }
      s2 += E.w[q] * (um - up) ** 2;
    }
    rms[e] = Math.sqrt(s2);
    pen += sigma[e] * topo.lengths[e] * s2;
  }
  return { rms, penaltyEnergy: pen };
}

/**
 * Errors of an IPDG solution: L², broken H¹ seminorm, and the DG energy norm
 *   |||u − u_h|||² = Σ_K ‖∇(u − u_h)‖²_K + Σ_F σ_F ‖⟦u − u_h⟧‖²_F
 * (⟦u⟧ = 0 on interior faces for the smooth exact u, and u − u_h = g − u_h on ∂Ω).
 * @param {ReturnType<typeof solveIP>} S
 * @param {(x:number,y:number)=>number} u
 * @param {(x:number,y:number)=>[number,number]} grad
 * @returns {{L2: number, H1: number, energy: number}}
 */
export function ipErrors(S, u, grad) {
  const { L2, H1 } = brokenErrors(S.mesh, S.p, S.U, u, grad);
  const { penaltyEnergy } = edgeJumps(S, u);
  return { L2, H1, energy: Math.sqrt(H1 * H1 + penaltyEnergy) };
}

/**
 * Estimate the smallest eigenvalue λ_min of a symmetric sparse matrix (used to
 * show whether the SIPG matrix is positive definite). Two Lanczos runs:
 *  - on A itself: the smallest Ritz value r₀ ≥ λ_min converges fast when λ_min
 *    is a well-separated negative eigenvalue (the "penalty too small" modes);
 *  - on A⁻¹ (shift-invert at 0, one sparse LU solve per step): gives the
 *    eigenvalue closest to zero, which is λ_min for an SPD matrix.
 * The estimate is min of the two. A negative result is a certificate of
 * indefiniteness (a Ritz value is a Rayleigh quotient vᵀAv/vᵀv).
 * @param {import('../la/sparse.js').CSR} A symmetric
 * @param {{k?: number}} [o] Krylov dimension (default 80)
 * @returns {number}
 */
export function smallestEigenvalue(A, o = {}) {
  const k = o.k ?? 80;
  const r = lanczosRitz(asOperator(A), A.n, k);
  const F = sparseLU(A);
  const ri = lanczosRitz((x, y) => y.set(F.solve(x)), A.n, k);
  const mu = Math.abs(ri[0]) > Math.abs(ri[ri.length - 1]) ? ri[0] : ri[ri.length - 1];
  return Math.min(r[0], 1 / mu);
}
