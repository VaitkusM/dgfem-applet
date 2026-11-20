/**
 * @file Nodal discontinuous Galerkin method for the 1D Euler equations
 *       (a SYSTEM of conservation laws) — the scalar scheme of dg1d.js applied
 *       to every component, coupled only through the flux function.
 *
 *   ∂t U + ∂x F(U) = 0,   U = (ρ, m, E),   F(U) = (m, m²/ρ + p, (E + p) m/ρ)
 *   (models/euler.js; ideal gas, p = (γ − 1)(E − ½ m²/ρ)).
 *
 * Discrete space: every component of U_h is a polynomial of degree p on each
 * element K_k = [x_k, x_{k+1}] (uniform width h), stored by its values at the
 * p+1 GLL nodes ξ_i of the reference interval [−1, 1] (p = 0: the midpoint):
 *
 *     U[3 (k (p+1) + i) + c] = component c of U_h at node i of element k
 *     (element-major, node, then component c = 0: ρ, 1: m = ρu, 2: E).
 *
 * Weak form per element and per component c (same as dg1d.js, WEAK form):
 *
 *   (h/2) Σ_j M_ij dU_j,c/dt = ∫_{-1}^{1} F_c(U_h) ℓ_i'(ξ) dξ − [ ℓ_i(1) F̂_c,k+½ − ℓ_i(−1) F̂_c,k−½ ].
 *
 *  - M is the exact mass matrix of the Lagrange basis.
 *  - The volume integral uses a Gauss rule with ⌈3p/2⌉ + 2 points. F(U) is a
 *    RATIONAL function of U, so no rule is exact; this one is exact for cubic
 *    nonlinearities and is what "over-integration" means for Euler.
 *  - F̂ = F̂(U⁻, U⁺) is an approximate Riemann solver from models/euler.js
 *    (Rusanov, HLL or HLLC): the flux OUT OF the left element K⁻ through the
 *    face with normal +x, as everywhere in this course (AGENTS.md).
 *
 * Boundary conditions: 'periodic', or 'outflow' (transmissive: the exterior
 * state is the interior trace, so F̂ = F(U_interior)).
 *
 * Limiting (makeEulerLimiter) — what it does and what it does NOT guarantee:
 *  1. Slope limiter (optional), applied to the CONSERVED variables component by
 *     component: an element is "troubled" if for ANY component the TVB-minmod
 *     test of limiters.js fails; a troubled element keeps, for every component,
 *     only its mean and a minmod-limited slope (it becomes linear). Limiting
 *     conserved variables is the simplest choice; limiting characteristic
 *     variables is sharper at contacts but needs the eigenvectors of F'(U).
 *     At an outflow boundary the missing one-sided difference is replaced by
 *     the existing one.
 *  2. Positivity fallback (always on, in the spirit of Zhang & Shu, JCP 229, 2010):
 *     if ρ or p is below a floor ε at any node or volume quadrature point, the
 *     polynomial is squeezed towards its cell average, U ← Ū + θ (U − Ū), with
 *     the largest θ ∈ {exact value for ρ, then halved for p} that makes all
 *     check points admissible (θ = 0 is the piecewise-constant, first-order
 *     fallback). Cell averages are never changed, so steps 1 and 2 conserve
 *     mass, momentum and energy exactly.
 *     This is a safeguard, not a proof: it needs positive CELL AVERAGES, which
 *     we do not prove for our time step. If a cell average has ρ ≤ ε or p ≤ ε
 *     the limiter sets `failed` and the caller must stop.
 *
 * All functions are pure and DOM-free.
 */
import { gaussLegendre } from '../quad/gauss1d.js';
import { interpMatrix, lagrangeValues } from '../basis/lagrange.js';
import { inverse } from '../la/dense.js';
import { EULER_FLUXES, GAMMA, physFlux, maxWaveSpeed } from '../models/euler.js';
import { refNodes, refMatrices, vandermonde } from './dg1d.js';
import { minmodTVB } from './limiters.js';

/**
 * Create a 1D DG discretisation of the Euler equations.
 * @param {{p: number, N: number, a?: number, b?: number, flux?: 'rusanov'|'hll'|'hllc'|string,
 *          bc?: 'periodic'|'outflow'}} o
 *   p ≥ 0 polynomial degree, N elements on [a, b] (default [0, 1]); flux = key of EULER_FLUXES (default 'hllc').
 *   The gas is air, γ = GAMMA = 1.4 (the flux registry of models/euler.js is fixed to it).
 * @returns {object} { p, N, np, n (= 3 N (p+1)), h, a, b, xf (faces, N+1), x (node coordinates, N (p+1)),
 *   r (reference nodes), ref (refMatrices), periodic, gamma,
 *   rhs(U, t, out)            — writes dU/dt (layout above),
 *   faceFlux                  — Float64Array 3 (N+1): F̂ of the last rhs call, faceFlux[3 f + c],
 *   project(W)                — L² projection of the conserved variables of a primitive field x ↦ [ρ, u, p],
 *   evalRef(U, k, ξ, out)     — conserved state [ρ, m, E] in element k at reference coordinate ξ,
 *   cellAverages(U)           — Float64Array 3 N, [3 k + c],
 *   totals(U)                 — [∫ρ dx, ∫m dx, ∫E dx],
 *   maxSpeed(U)               — max over nodes of |u| + c,
 *   minima(U)                 — {rho, p, finite}: smallest nodal density and pressure, all values finite?,
 *   l1Error(U, c, f)          — ∫ |U_h,c − f(x)| dx for conserved component c (8-point Gauss per element) }
 */
export function makeDGEuler1D(o) {
  const p = o.p, N = o.N, a = o.a ?? 0, b = o.b ?? 1, h = (b - a) / N, g = GAMMA;
  const np = p + 1, nn = N * np, n = 3 * nn;
  const Ffn = EULER_FLUXES[o.flux || 'hllc'].fn;
  const periodic = (o.bc || 'periodic') === 'periodic';
  const r = refNodes(p, 'gll').x;
  const ref = refMatrices(r, gaussLegendre(Math.ceil(3 * p / 2) + 2));
  const { Minv, lL, lR, Iq, Dq, nq, wq } = ref;

  const xf = new Float64Array(N + 1), x = new Float64Array(nn);
  for (let k = 0; k <= N; k++) xf[k] = a + k * h;
  for (let k = 0; k < N; k++) for (let i = 0; i < np; i++) x[k * np + i] = xf[k] + (r[i] + 1) * h / 2;

  // work arrays: traces U⁻, U⁺ at every face, face fluxes, per-element scratch
  const UM = new Float64Array(3 * (N + 1)), UP = new Float64Array(3 * (N + 1));
  const faceFlux = new Float64Array(3 * (N + 1));
  const sL = new Float64Array(3), sR = new Float64Array(3), fl = new Float64Array(3), uq = new Float64Array(3);
  const Fq = new Float64Array(3 * nq), res = new Float64Array(3 * np);

  /**
   * Semi-discrete right-hand side dU/dt = L(U).
   * @param {Float64Array} U length n
   * @param {number} _t time (unused: the system is autonomous)
   * @param {Float64Array} out length n
   */
  function rhs(U, _t, out) {
    // 1. traces: UP[f] from the element right of face f, UM[f] from the element left of it
    for (let k = 0; k < N; k++) for (let c = 0; c < 3; c++) {
      let l = 0, rr = 0;
      for (let i = 0; i < np; i++) { const v = U[3 * (k * np + i) + c]; l += lL[i] * v; rr += lR[i] * v; }
      UP[3 * k + c] = l; UM[3 * (k + 1) + c] = rr;
    }
    for (let c = 0; c < 3; c++) {
      if (periodic) { UM[c] = UM[3 * N + c]; UP[3 * N + c] = UP[c]; }
      else { UM[c] = UP[c]; UP[3 * N + c] = UM[3 * N + c]; } // transmissive
    }
    // 2. numerical fluxes (approximate Riemann solver, flux out of the left element)
    for (let f = 0; f <= N; f++) {
      for (let c = 0; c < 3; c++) { sL[c] = UM[3 * f + c]; sR[c] = UP[3 * f + c]; }
      Ffn(sL, sR, fl);
      for (let c = 0; c < 3; c++) faceFlux[3 * f + c] = fl[c];
    }
    // 3. volume + surface terms, element by element
    const s = 2 / h;
    for (let k = 0; k < N; k++) {
      const off = 3 * k * np;
      for (let q = 0; q < nq; q++) {
        uq[0] = uq[1] = uq[2] = 0;
        for (let j = 0; j < np; j++) { const w = Iq[q * np + j]; for (let c = 0; c < 3; c++) uq[c] += w * U[off + 3 * j + c]; }
        physFlux(uq, fl, g);
        for (let c = 0; c < 3; c++) Fq[3 * q + c] = wq[q] * fl[c];
      }
      for (let i = 0; i < np; i++) for (let c = 0; c < 3; c++) {
        let v = 0;
        for (let q = 0; q < nq; q++) v += Dq[q * np + i] * Fq[3 * q + c];
        res[3 * i + c] = v - (lR[i] * faceFlux[3 * (k + 1) + c] - lL[i] * faceFlux[3 * k + c]);
      }
      for (let i = 0; i < np; i++) for (let c = 0; c < 3; c++) {
        let v = 0;
        for (let j = 0; j < np; j++) v += Minv[i * np + j] * res[3 * j + c];
        out[off + 3 * i + c] = s * v;
      }
    }
  }

  // exact mass matrix (for projection, averages) and a fine Gauss rule
  const G = gaussLegendre(p + 6), IG = interpMatrix(r, G.x), ng = G.x.length;
  const Mx = new Float64Array(np * np);
  for (let i = 0; i < np; i++) for (let j = 0; j < np; j++) {
    let v = 0; for (let q = 0; q < ng; q++) v += G.w[q] * IG[q * np + i] * IG[q * np + j];
    Mx[i * np + j] = v;
  }
  const Mxi = inverse(Mx, np);
  const lw = new Float64Array(np); // ∫ ℓ_i dξ
  for (let i = 0; i < np; i++) { let v = 0; for (let j = 0; j < np; j++) v += Mx[i * np + j]; lw[i] = v; }

  /** L² projection of the conserved variables of the primitive field W(x) = [ρ, u, p]. */
  function project(W) {
    const U = new Float64Array(n), bv = new Float64Array(3 * np);
    for (let k = 0; k < N; k++) {
      bv.fill(0);
      for (let q = 0; q < ng; q++) {
        const w = W(xf[k] + (G.x[q] + 1) * h / 2);
        const c0 = w[0], c1 = w[0] * w[1], c2 = w[2] / (g - 1) + 0.5 * w[0] * w[1] * w[1];
        for (let i = 0; i < np; i++) {
          const z = G.w[q] * IG[q * np + i];
          bv[3 * i] += z * c0; bv[3 * i + 1] += z * c1; bv[3 * i + 2] += z * c2;
        }
      }
      for (let i = 0; i < np; i++) for (let c = 0; c < 3; c++) {
        let v = 0; for (let j = 0; j < np; j++) v += Mxi[i * np + j] * bv[3 * j + c];
        U[3 * (k * np + i) + c] = v;
      }
    }
    return U;
  }
  const tmp = new Float64Array(np);
  /** Conserved state in element k at reference coordinate ξ. */
  function evalRef(U, k, xi, out = [0, 0, 0]) {
    lagrangeValues(r, xi, tmp);
    out[0] = out[1] = out[2] = 0;
    for (let i = 0; i < np; i++) for (let c = 0; c < 3; c++) out[c] += tmp[i] * U[3 * (k * np + i) + c];
    return out;
  }
  /** Cell averages, [3 k + c]. */
  function cellAverages(U) {
    const A = new Float64Array(3 * N);
    for (let k = 0; k < N; k++) for (let i = 0; i < np; i++) for (let c = 0; c < 3; c++) A[3 * k + c] += 0.5 * lw[i] * U[3 * (k * np + i) + c];
    return A;
  }
  /** [∫ρ, ∫m, ∫E] over the domain (exact for the polynomial). */
  function totals(U) {
    const T = [0, 0, 0];
    for (let k = 0; k < N; k++) for (let i = 0; i < np; i++) for (let c = 0; c < 3; c++) T[c] += lw[i] * U[3 * (k * np + i) + c] * h / 2;
    return T;
  }
  /** max |u| + c over all nodes. */
  function maxSpeed(U) {
    let s = 0;
    for (let i = 0; i < nn; i++) s = Math.max(s, maxWaveSpeed(U, 3 * i, g));
    return s;
  }
  /** Smallest nodal density / pressure and a finiteness flag. */
  function minima(U) {
    let rho = Infinity, pr = Infinity, finite = true;
    for (let i = 0; i < nn; i++) {
      const d = U[3 * i], m = U[3 * i + 1], E = U[3 * i + 2];
      if (!Number.isFinite(d) || !Number.isFinite(m) || !Number.isFinite(E)) { finite = false; continue; }
      rho = Math.min(rho, d); pr = Math.min(pr, (g - 1) * (E - 0.5 * m * m / d));
    }
    return { rho, p: pr, finite };
  }
  const G8 = gaussLegendre(8), I8 = interpMatrix(r, G8.x);
  /** ∫ |U_h,c − f(x)| dx (8 Gauss points per element; only a rough number if f jumps inside an element). */
  function l1Error(U, c, f) {
    let e = 0;
    for (let k = 0; k < N; k++) for (let q = 0; q < 8; q++) {
      let v = 0; for (let j = 0; j < np; j++) v += I8[q * np + j] * U[3 * (k * np + j) + c];
      e += G8.w[q] * h / 2 * Math.abs(v - f(xf[k] + (G8.x[q] + 1) * h / 2));
    }
    return e;
  }

  return { p, N, np, n, h, a, b, xf, x, r, ref, periodic, gamma: g,
    rhs, faceFlux, project, evalRef, cellAverages, totals, maxSpeed, minima, l1Error };
}

/**
 * Slope limiter + positivity fallback for makeDGEuler1D (see the file header for
 * what is and is not guaranteed). Use as the post-stage hook of an SSP-RK stepper.
 * @param {object} dg discretisation from makeDGEuler1D
 * @param {{limit?: boolean, M?: number, eps?: number}} [o]
 *   limit: apply the component-wise TVB-minmod slope limiter (default true);
 *   M: TVB constant (default 0 = pure minmod); eps: floor for ρ and p (default 1e-10).
 * @returns {{apply: (U: Float64Array) => number, troubled: Uint8Array, squeezed: Uint8Array,
 *            stats: {limited: number, squeezed: number, failed: boolean}}}
 *   apply() modifies U in place and returns the number of slope-limited elements;
 *   troubled[k] / squeezed[k] = 1 for elements changed by the slope limiter / positivity fallback
 *   in the last call; stats.failed becomes (and stays) true if a cell AVERAGE is inadmissible.
 */
export function makeEulerLimiter(dg, o = {}) {
  const { N, np, h, r, periodic, gamma: g } = dg;
  const limit = o.limit ?? true, Mh2 = (o.M ?? 0) * h * h, eps = o.eps ?? 1e-10;
  const V = vandermonde(r), Vinv = inverse(V, np), Iq = dg.ref.Iq, nq = dg.ref.nq;
  const avg = new Float64Array(3 * N), slope = new Float64Array(3 * N), eL = new Float64Array(3 * N), eR = new Float64Array(3 * N);
  const troubled = new Uint8Array(N), squeezed = new Uint8Array(N);
  const stats = { limited: 0, squeezed: 0, failed: false };
  const press = (d, m, E) => (g - 1) * (E - 0.5 * m * m / d);

  /** Smallest ρ and p of element k over nodes and volume quadrature points, for U = Ū + θ (U − Ū). */
  function minRhoP(U, k, th, out) {
    const off = 3 * k * np, a0 = avg[3 * k], a1 = avg[3 * k + 1], a2 = avg[3 * k + 2];
    let mr = Infinity, mp = Infinity;
    for (let i = 0; i < np; i++) {
      const d = a0 + th * (U[off + 3 * i] - a0), m = a1 + th * (U[off + 3 * i + 1] - a1), E = a2 + th * (U[off + 3 * i + 2] - a2);
      mr = Math.min(mr, d); mp = Math.min(mp, press(d, m, E));
    }
    for (let q = 0; q < nq; q++) {
      let d = 0, m = 0, E = 0;
      for (let j = 0; j < np; j++) { const w = Iq[q * np + j]; d += w * U[off + 3 * j]; m += w * U[off + 3 * j + 1]; E += w * U[off + 3 * j + 2]; }
      d = a0 + th * (d - a0); m = a1 + th * (m - a1); E = a2 + th * (E - a2);
      mr = Math.min(mr, d); mp = Math.min(mp, press(d, m, E));
    }
    out[0] = mr; out[1] = mp;
  }
  const mm = [0, 0];

  function apply(U) {
    stats.limited = 0; stats.squeezed = 0;
    troubled.fill(0); squeezed.fill(0);
    // modal content per element and component: mean, P_1 coefficient, edge values
    for (let k = 0; k < N; k++) for (let c = 0; c < 3; c++) {
      let a0 = 0, a1 = 0, el = 0, er = 0;
      for (let j = 0; j < np; j++) {
        let m = 0;
        for (let i = 0; i < np; i++) m += Vinv[j * np + i] * U[3 * (k * np + i) + c];
        if (j === 0) a0 = m; else if (j === 1) a1 = m;
        el += (j % 2 ? -m : m); er += m; // P_j(−1) = (−1)^j, P_j(1) = 1
      }
      avg[3 * k + c] = a0; slope[3 * k + c] = a1; eL[3 * k + c] = el; eR[3 * k + c] = er;
    }
    if (limit && np > 1) {
      for (let k = 0; k < N; k++) {
        const km = k > 0 ? k - 1 : (periodic ? N - 1 : -1), kp = k < N - 1 ? k + 1 : (periodic ? 0 : -1);
        let bad = false;
        for (let c = 0; c < 3 && !bad; c++) {
          const A = avg[3 * k + c];
          let dm = km >= 0 ? A - avg[3 * km + c] : NaN, dp = kp >= 0 ? avg[3 * kp + c] - A : NaN;
          if (km < 0) dm = dp; if (kp < 0) dp = dm;
          const tol = 1e-12 * (1 + Math.abs(A));
          const dR = eR[3 * k + c] - A, dL = A - eL[3 * k + c];
          if (Math.abs(minmodTVB(dR, dm, dp, Mh2) - dR) > tol || Math.abs(minmodTVB(dL, dm, dp, Mh2) - dL) > tol) bad = true;
        }
        if (!bad) continue;
        troubled[k] = 1; stats.limited++;
        for (let c = 0; c < 3; c++) {
          const A = avg[3 * k + c];
          let dm = km >= 0 ? A - avg[3 * km + c] : NaN, dp = kp >= 0 ? avg[3 * kp + c] - A : NaN;
          if (km < 0) dm = dp; if (kp < 0) dp = dm;
          const s = minmodTVB(slope[3 * k + c], 0.5 * dm, 0.5 * dp, Mh2);
          for (let i = 0; i < np; i++) U[3 * (k * np + i) + c] = A + s * r[i];
        }
      }
    }
    // positivity fallback
    for (let k = 0; k < N; k++) {
      const a0 = avg[3 * k], pa = press(a0, avg[3 * k + 1], avg[3 * k + 2]);
      if (!(a0 > eps) || !(pa > eps)) { stats.failed = true; continue; }
      if (np === 1) continue;
      minRhoP(U, k, 1, mm);
      if (mm[0] >= eps && mm[1] >= eps) continue;
      let th = mm[0] < eps ? Math.max(0, (a0 - eps) / (a0 - mm[0])) : 1;
      for (let it = 0; it < 40; it++) {
        minRhoP(U, k, th, mm);
        if (mm[0] >= eps && mm[1] >= eps) break;
        th = it < 39 ? 0.5 * th : 0;
      }
      if (!(th > 1e-12)) th = 0;
      const off = 3 * k * np;
      for (let i = 0; i < np; i++) for (let c = 0; c < 3; c++) U[off + 3 * i + c] = avg[3 * k + c] + th * (U[off + 3 * i + c] - avg[3 * k + c]);
      squeezed[k] = 1; stats.squeezed++;
    }
    return stats.limited;
  }
  return { apply, troubled, squeezed, stats };
}

/**
 * Test problems for the Euler DG widget and tests.
 *  - sod:  Sod's shock tube on [0, 1], diaphragm at x0 = 0.5 (put it on an element face: N even),
 *          outflow boundaries, final time T = 0.2. Exact solution: models/eulerriemann.js.
 *  - wave: smooth density wave ρ = 1 + 0.2 sin 2πx, u = 1, p = 1 on the periodic unit interval.
 *          With constant u and p the Euler equations reduce to ρ_t + u ρ_x = 0, so the exact
 *          solution is the initial profile shifted by t (one period at T = 1).
 * Each entry: { label, bc, T, x0?, WL?, WR?, init(x) → [ρ, u, p], exact?(x, t) → [ρ, u, p] }.
 */
export const EULER_DG_PROBLEMS = {
  sod: {
    label: 'Sod shock tube', bc: 'outflow', T: 0.2, x0: 0.5, WL: [1, 0, 1], WR: [0.125, 0, 0.1],
    init: (x) => (x < 0.5 ? [1, 0, 1] : [0.125, 0, 0.1]),
  },
  wave: {
    label: 'smooth density wave', bc: 'periodic', T: 1,
    init: (x) => [1 + 0.2 * Math.sin(2 * Math.PI * x), 1, 1],
    exact: (x, t) => [1 + 0.2 * Math.sin(2 * Math.PI * (x - t)), 1, 1],
  },
};
