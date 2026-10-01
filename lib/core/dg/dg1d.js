/**
 * @file Nodal discontinuous Galerkin (DG) method for scalar conservation laws
 *       u_t + f(u)_x = 0 in 1D.
 *
 * Discrete space (the "broken" space V_h^p): on each element
 * K_k = [x_k, x_{k+1}] (uniform width h) u_h is a polynomial of degree p, with
 * no continuity across element faces. On the reference interval [−1,1]
 * (x = x_k + (ξ+1) h/2, d/dx = (2/h) d/dξ) we store it by its values at p+1
 * nodes ξ_0 < … < ξ_p (a *nodal* Lagrange basis ℓ_0..ℓ_p, ℓ_j(ξ_i) = δ_ij):
 *
 *     u[k*(p+1) + i] = u_h(x_k + (ξ_i+1) h/2)      (element-major layout).
 *
 * Element-wise weak form (multiply by ℓ_i, integrate over K_k, integrate the
 * flux term by parts, replace the face flux by a numerical flux F̂):
 *
 *   (h/2) Σ_j M_ij du_j/dt = ∫_{-1}^{1} f(u_h) ℓ_i'(ξ) dξ − [ ℓ_i(1) F̂_{k+½} − ℓ_i(−1) F̂_{k−½} ]   (WEAK)
 *
 * Integrating by parts once more gives the equivalent STRONG form
 *
 *   (h/2) M du/dt = −∫ ∂_ξ f_h ℓ_i dξ − ℓ_i(1) (F̂_{k+½} − f_h(1)) + ℓ_i(−1) (F̂_{k−½} − f_h(−1)),
 *
 * where f_h is the degree-p interpolant of f(u_h) at the nodes ("collocated
 * flux"). Both are identical when f is linear and the integrals are exact.
 *
 * Integrals are evaluated by one of two quadrature choices (option `quad`):
 *  - 'exact'      Gauss–Legendre with nq points (default nq = ⌈(3p+2)/2⌉ + 1, exact for
 *                 the Burgers volume term u_h² ℓ_i' of degree 3p−1 and for the mass matrix).
 *                 The mass matrix M is the exact (full) Gram matrix of the basis.
 *  - 'collocated' the quadrature points ARE the solution nodes (with their weights).
 *                 With GLL nodes this is the classical DG spectral element method
 *                 (DG-SEM): the mass matrix becomes diagonal ("lumped",
 *                 M = diag(w)) but is NOT exact (GLL with p+1 points is exact only
 *                 to degree 2p−1 < 2p), and nonlinear fluxes are aliased:
 *                 f(u_h) is effectively replaced by its interpolant.
 *                 With Gauss nodes the mass matrix is exact (degree 2p ≤ 2p+1) and
 *                 diagonal.
 * For LINEAR advection all volume integrands have degree ≤ 2p−1, so the only
 * inexact ingredient of GLL collocation is the lumped mass matrix.
 *
 * Face fluxes: F̂(u⁻, u⁺) for the face normal pointing in +x (from the left
 * element K⁻ into the right element K⁺), i.e. the flux OUT OF the left element,
 * matching AGENTS.md. Jumps [[u]] = u⁻ − u⁺ (left minus right in 1D).
 *
 * Boundary conditions:
 *  - 'periodic'
 *  - 'inflow' (weak, through the numerical flux): the exterior state at x = a
 *    and x = b is bcValue(x, t). With an upwind flux, the exterior value is only
 *    used where characteristics enter (inflow); at an outflow end the
 *    flux takes the interior value — exactly the "boundary data only at inflow"
 *    rule of hyperbolic problems.
 *
 * All functions are pure and DOM-free.
 */
import { gaussLegendre, gaussLobatto } from '../quad/gauss1d.js';
import { lagrangeValues, lagrangeDerivs, diffMatrix, interpMatrix } from '../basis/lagrange.js';
import { legendreAll } from '../basis/legendre.js';
import { inverse } from '../la/dense.js';
import { SCALAR_FLUXES } from '../models/scalar.js';

/**
 * Entropy-conservative flux for Burgers (Tadmor):  F̂ = (u⁻² + u⁻u⁺ + u⁺²)/6.
 * With exact integration it makes the semi-discrete energy ½‖u_h‖² exactly
 * conserved (periodic); used to expose aliasing errors of collocation.
 * @param {object} _m model (ignored, Burgers assumed)
 * @param {number} uL
 * @param {number} uR
 */
export function burgersECFlux(_m, uL, uR) { return (uL * uL + uL * uR + uR * uR) / 6; }

/** Fluxes accepted by makeDG1D / makeFR1D (scalar.js registry + Burgers EC flux). */
export const DG_FLUXES = { ...SCALAR_FLUXES, ec: { label: 'Entropy conservative (Burgers)', fn: burgersECFlux } };

/**
 * Reference solution nodes on [−1, 1].
 * @param {number} p degree ≥ 0
 * @param {'gll'|'gauss'} kind GLL needs p ≥ 1; for p = 0 the single Gauss point (ξ = 0) is used.
 * @returns {{x: Float64Array, w: Float64Array}} nodes ascending and their quadrature weights
 */
export function refNodes(p, kind = 'gll') {
  if (kind === 'gll' && p >= 1) return gaussLobatto(p + 1);
  return gaussLegendre(p + 1);
}

/**
 * Reference-element matrices of a nodal basis on [−1,1].
 * @param {ArrayLike<number>} r nodes (p+1)
 * @param {{x: ArrayLike<number>, w: ArrayLike<number>}} Q quadrature rule
 * @returns {{np: number, M: Float64Array, Minv: Float64Array, S: Float64Array, D: Float64Array,
 *            lL: Float64Array, lR: Float64Array, Iq: Float64Array, Dq: Float64Array, nq: number, wq: Float64Array}}
 *   M_ij = Σ_q w_q ℓ_i ℓ_j (mass), S_ij = Σ_q w_q ℓ_i ℓ_j' (stiffness, i.e. ∫ ℓ_i ℓ_j'),
 *   D_ij = ℓ_j'(ξ_i) (differentiation), lL_i = ℓ_i(−1), lR_i = ℓ_i(1),
 *   Iq (nq×np) = ℓ_j(x_q), Dq (nq×np) = ℓ_j'(x_q). All row-major.
 */
export function refMatrices(r, Q) {
  const np = r.length, nq = Q.x.length;
  const Iq = interpMatrix(r, Q.x), Dq = new Float64Array(nq * np), tmp = new Float64Array(np);
  for (let q = 0; q < nq; q++) { lagrangeDerivs(r, Q.x[q], tmp); Dq.set(tmp, q * np); }
  const M = new Float64Array(np * np), S = new Float64Array(np * np);
  for (let i = 0; i < np; i++) for (let j = 0; j < np; j++) {
    let m = 0, s = 0;
    for (let q = 0; q < nq; q++) {
      m += Q.w[q] * Iq[q * np + i] * Iq[q * np + j];
      s += Q.w[q] * Iq[q * np + i] * Dq[q * np + j];
    }
    M[i * np + j] = m; S[i * np + j] = s;
  }
  const D = np > 1 ? diffMatrix(r) : new Float64Array(1);
  const lL = lagrangeValues(r, -1), lR = lagrangeValues(r, 1);
  return { np, M, Minv: inverse(M, np), S, D, lL, lR, Iq, Dq, nq, wq: Float64Array.from(Q.w) };
}

/**
 * Legendre–Vandermonde matrix V_ij = P_j(ξ_i): modal coefficients û (u_h = Σ û_j P_j)
 * map to nodal values by u = V û.
 * @param {ArrayLike<number>} r nodes
 * @returns {Float64Array} (np × np)
 */
export function vandermonde(r) {
  const np = r.length, V = new Float64Array(np * np);
  for (let i = 0; i < np; i++) { const { P } = legendreAll(np - 1, r[i]); for (let j = 0; j < np; j++) V[i * np + j] = P[j]; }
  return V;
}

/**
 * Create a 1D DG discretisation.
 * @param {{p: number, N: number, a?: number, b?: number,
 *          model: import('../models/scalar.js').ScalarModel,
 *          flux?: string|((m: object, uL: number, uR: number) => number),
 *          nodes?: 'gll'|'gauss', quad?: 'exact'|'collocated', nq?: number,
 *          form?: 'weak'|'strong', bc?: 'periodic'|'inflow', bcValue?: (x: number, t: number) => number}} o
 * @returns {object} discretisation handle:
 *   { p, N, np, n, h, a, b, xf (faces, N+1), x (node coordinates, n), r (reference nodes), ref (refMatrices),
 *     rhs(u, t, out)  — writes du/dt,
 *     faceFlux (Float64Array N+1, last F̂ values), traces(u) → {uMinus, uPlus} at faces,
 *     project(f), interpolate(f), evalAt(u, x), l2Error(u, f), mass(u), energy(u), maxSpeed(u) }
 */
export function makeDG1D(o) {
  const p = o.p, N = o.N, a = o.a ?? 0, b = o.b ?? 1, h = (b - a) / N;
  const np = p + 1, n = N * np, m = o.model;
  const Ffn = typeof o.flux === 'function' ? o.flux : DG_FLUXES[o.flux || 'upwind'].fn;
  const nodeKind = p === 0 ? 'gauss' : (o.nodes || 'gll');
  const R = refNodes(p, nodeKind), r = R.x;
  const quad = o.quad || 'exact';
  const nqExact = o.nq ?? Math.ceil((3 * p + 2) / 2) + 1;
  const Q = quad === 'collocated' ? R : gaussLegendre(nqExact);
  const ref = refMatrices(r, Q);
  const strong = (o.form || 'weak') === 'strong';
  const periodic = (o.bc || 'periodic') === 'periodic';
  const bcValue = o.bcValue || (() => 0);
  const { M, Minv, D, lL, lR, Iq, Dq, nq, wq } = ref;

  const xf = new Float64Array(N + 1), x = new Float64Array(n);
  for (let k = 0; k <= N; k++) xf[k] = a + k * h;
  for (let k = 0; k < N; k++) for (let i = 0; i < np; i++) x[k * np + i] = xf[k] + (r[i] + 1) * h / 2;

  // work arrays
  const uM = new Float64Array(N + 1), uP = new Float64Array(N + 1); // u⁻, u⁺ at each face
  const faceFlux = new Float64Array(N + 1);
  const res = new Float64Array(np), fn = new Float64Array(np), uq = new Float64Array(nq);
  const exactM = Float64Array.from(refMatrices(r, gaussLegendre(np + 1)).M);

  /** traces: uM[f] from the element left of face f, uP[f] from the element right of it */
  function traces(u, t = 0) {
    for (let k = 0; k < N; k++) {
      let sl = 0, sr = 0;
      for (let i = 0; i < np; i++) { sl += lL[i] * u[k * np + i]; sr += lR[i] * u[k * np + i]; }
      uP[k] = sl; uM[k + 1] = sr;
    }
    if (periodic) { uM[0] = uM[N]; uP[N] = uP[0]; }
    else { uM[0] = bcValue(a, t); uP[N] = bcValue(b, t); }
    return { uMinus: uM, uPlus: uP };
  }

  /**
   * Semi-discrete right-hand side du/dt = L(u).
   * @param {Float64Array} u length n
   * @param {number} t
   * @param {Float64Array} out length n
   */
  function rhs(u, t, out) {
    traces(u, t);
    for (let f = 0; f <= N; f++) faceFlux[f] = Ffn(m, uM[f], uP[f]);
    const s = 2 / h;
    for (let k = 0; k < N; k++) {
      const off = k * np, FL = faceFlux[k], FR = faceFlux[k + 1];
      if (!strong) {
        // values at quadrature points, then volume term Σ_q w_q f(u_q) ℓ_i'(x_q)
        for (let q = 0; q < nq; q++) { let v = 0; for (let j = 0; j < np; j++) v += Iq[q * np + j] * u[off + j]; uq[q] = m.f(v) * wq[q]; }
        for (let i = 0; i < np; i++) {
          let v = 0;
          for (let q = 0; q < nq; q++) v += Dq[q * np + i] * uq[q];
          res[i] = v - (lR[i] * FR - lL[i] * FL);
        }
      } else {
        // strong form with collocated flux interpolant f_h(ξ) = Σ f(u_j) ℓ_j(ξ)
        let fl = 0, fr = 0;
        for (let j = 0; j < np; j++) { fn[j] = m.f(u[off + j]); fl += lL[j] * fn[j]; fr += lR[j] * fn[j]; }
        for (let i = 0; i < np; i++) {
          // −∫ f_h' ℓ_i  by the chosen quadrature
          let v = 0;
          for (let q = 0; q < nq; q++) {
            let dfq = 0;
            for (let j = 0; j < np; j++) dfq += Dq[q * np + j] * fn[j];
            v += wq[q] * Iq[q * np + i] * dfq;
          }
          res[i] = -v - lR[i] * (FR - fr) + lL[i] * (FL - fl);
        }
      }
      for (let i = 0; i < np; i++) {
        let v = 0;
        for (let j = 0; j < np; j++) v += Minv[i * np + j] * res[j];
        out[off + i] = s * v;
      }
    }
  }

  // high-order Gauss rule for projection / errors (degree ≥ 2p + 8)
  const G = gaussLegendre(p + 6);
  const IG = interpMatrix(r, G.x);
  const exactMinv = inverse(exactM, np);

  /** L2 projection of f onto V_h^p (exact mass matrix, Gauss quadrature with p+6 points). */
  function project(f) {
    const u = new Float64Array(n), rhsv = new Float64Array(np);
    for (let k = 0; k < N; k++) {
      rhsv.fill(0);
      for (let q = 0; q < G.x.length; q++) {
        const fx = f(xf[k] + (G.x[q] + 1) * h / 2);
        for (let i = 0; i < np; i++) rhsv[i] += G.w[q] * fx * IG[q * np + i];
      }
      for (let i = 0; i < np; i++) { let v = 0; for (let j = 0; j < np; j++) v += exactMinv[i * np + j] * rhsv[j]; u[k * np + i] = v; }
    }
    return u;
  }
  /** Nodal interpolation of f. */
  function interpolate(f) { const u = new Float64Array(n); for (let i = 0; i < n; i++) u[i] = f(x[i]); return u; }
  /** Evaluate u_h at x (left-continuous at faces is not guaranteed; picks the element containing x). */
  const tmpL = new Float64Array(np);
  function evalAt(u, xx) {
    let k = Math.floor((xx - a) / h); k = Math.max(0, Math.min(N - 1, k));
    const xi = 2 * (xx - xf[k]) / h - 1;
    lagrangeValues(r, xi, tmpL);
    let v = 0; for (let i = 0; i < np; i++) v += tmpL[i] * u[k * np + i];
    return v;
  }
  /** Evaluate u_h in element k at reference coordinate ξ. */
  function evalRef(u, k, xi) {
    lagrangeValues(r, xi, tmpL);
    let v = 0; for (let i = 0; i < np; i++) v += tmpL[i] * u[k * np + i];
    return v;
  }
  /** ‖u_h − f‖_{L²(a,b)} with (p+6)-point Gauss per element. */
  function l2Error(u, f) {
    let e = 0;
    for (let k = 0; k < N; k++) for (let q = 0; q < G.x.length; q++) {
      let v = 0; for (let j = 0; j < np; j++) v += IG[q * np + j] * u[k * np + j];
      const d = v - f(xf[k] + (G.x[q] + 1) * h / 2);
      e += G.w[q] * h / 2 * d * d;
    }
    return Math.sqrt(e);
  }
  /** Total mass ∫ u_h dx (exact: weights are ∫ ℓ_i). */
  const lw = new Float64Array(np);
  for (let i = 0; i < np; i++) { let s = 0; for (let j = 0; j < np; j++) s += exactM[i * np + j]; lw[i] = s; }
  function mass(u) { let s = 0; for (let k = 0; k < N; k++) for (let i = 0; i < np; i++) s += lw[i] * u[k * np + i]; return s * h / 2; }
  /** Discrete energy ½ uᵀ (h/2 M) u with the scheme's own mass matrix M. */
  function energy(u) {
    let s = 0;
    for (let k = 0; k < N; k++) for (let i = 0; i < np; i++) for (let j = 0; j < np; j++) s += u[k * np + i] * M[i * np + j] * u[k * np + j];
    return 0.5 * s * h / 2;
  }
  const maxSpeed = (u) => { let s = 0; for (let i = 0; i < n; i++) s = Math.max(s, Math.abs(m.df(u[i]))); return s; };

  return { p, N, np, n, h, a, b, xf, x, r, ref, nodeKind, quad, strong, periodic, model: m,
    rhs, traces, faceFlux, project, interpolate, evalAt, evalRef, l2Error, mass, energy, maxSpeed };
}

/**
 * Integrate du/dt = L(u) with a fixed time step to time T (last step shortened).
 * @param {(u: Float64Array, t: number, dt: number, L: Function) => void} step stepper from makeStepper
 * @param {Float64Array} u state (modified in place)
 * @param {(u: Float64Array, t: number, out: Float64Array) => void} L
 * @param {number} T final time
 * @param {number} dt time step
 * @param {number} [t0=0]
 * @returns {number} final time
 */
export function integrate(step, u, L, T, dt, t0 = 0) {
  let t = t0;
  const nSteps = Math.max(1, Math.ceil((T - t0) / dt - 1e-9));
  const d = (T - t0) / nSteps;
  for (let s = 0; s < nSteps; s++) { step(u, t, d, L); t += d; }
  return t;
}
