/**
 * @file DG spectral element method (DG-SEM) for scalar conservation laws
 *       u_t + ∂_x f(u, x, y) + ∂_y g(u, x, y) = 0 on a periodic Cartesian grid of quads.
 *
 * Element e = ey*nx + ex covers [x_ex, x_ex + hx] × [y_ey, y_ey + hy]. On the
 * reference square [−1,1]² the solution is a tensor-product polynomial of
 * degree p in each variable, stored by its values at the (p+1)² tensor GLL
 * nodes (ξ_i, η_j):
 *
 *     u[e*np*np + j*np + i] = u_h(x_ex + (ξ_i+1) hx/2, y_ey + (η_j+1) hy/2),  np = p+1.
 *
 * All integrals use the same GLL nodes as quadrature ("collocation"), so the
 * mass matrix is diagonal, M = diag(w_i w_j) (h_x h_y/4), and the strong-form DG
 * equations decouple into 1D operations along grid lines:
 *
 *   du_ij/dt = −(2/hx) [ Σ_m D_im f_mj + δ_{i0} (f_0j − F̂_left)/w_0 ... ]
 *
 * precisely, per x-line j of element e (and analogously per y-line):
 *   du_ij/dt += −(2/hx) Σ_m D_im f_mj
 *              + (2/hx) δ_{i,0} (F̂_{left,j} − f_0j) / w_0
 *              − (2/hx) δ_{i,p} (F̂_{right,j} − f_pj) / w_p ,
 * where F̂ is the numerical flux in the +x direction (out of the left element)
 * evaluated with the traces at the face node (y-coordinate η_j). This is the
 * 1D strong form of dg1d.js with GLL collocation, applied line by line — a
 * tensor product, like separable filtering in image processing.
 *
 * Numerical flux: local Lax–Friedrichs (Rusanov)
 *     F̂(u⁻, u⁺) = ½ (f(u⁻) + f(u⁺)) − ½ α (u⁺ − u⁻),  α = max(|∂f/∂u (u⁻)|, |∂f/∂u (u⁺)|),
 * which for linear advection f = a_n u is exactly the upwind flux; α = 0 gives
 * the central flux.
 *
 * Problems:
 *  - 'advection': f = a_x(x,y) u, g = a_y(x,y) u with a velocity callback.
 *    The velocity must be continuous across faces (a_x may depend on y only at
 *    vertical faces, etc. — true for constant and solid-body-rotation fields).
 *  - 'burgers':   f = g = u²/2 (2D Burgers along the diagonal).
 */
import { gaussLobatto, gaussLegendre } from '../quad/gauss1d.js';
import { diffMatrix, lagrangeValues, interpMatrix } from '../basis/lagrange.js';
import { inverse } from '../la/dense.js';

/**
 * Create a 2D DG-SEM discretisation on a periodic grid.
 * @param {{p: number, nx: number, ny: number, box?: number[], problem?: 'advection'|'burgers',
 *          velocity?: (x: number, y: number) => number[], flux?: 'rusanov'|'central'}} o
 *   p ≥ 1; box = [x0, x1, y0, y1] (default unit square)
 * @returns {object} { p, np, nx, ny, nE, n, hx, hy, box, r, w, X, Y (node coordinates, length n),
 *   rhs(u,t,out), project(f), interpolate(f), evalRef(u, e, ξ, η), l2Error(u, f), mass(u), maxSpeed(u), dtStable(u, cfl) }
 */
export function makeDG2D(o) {
  const p = Math.max(1, o.p), np = p + 1, nx = o.nx, ny = o.ny;
  const box = o.box || [0, 1, 0, 1];
  const hx = (box[1] - box[0]) / nx, hy = (box[3] - box[2]) / ny;
  const nE = nx * ny, nn = np * np, n = nE * nn;
  const R = gaussLobatto(np), r = R.x, w = R.w, D = diffMatrix(r);
  const burg = o.problem === 'burgers';
  const vel = o.velocity || (() => [1, 0]);
  const central = o.flux === 'central';

  // node coordinates and (for advection) velocities at nodes
  const X = new Float64Array(n), Y = new Float64Array(n), AX = new Float64Array(n), AY = new Float64Array(n);
  for (let ey = 0; ey < ny; ey++) for (let ex = 0; ex < nx; ex++) {
    const e = ey * nx + ex;
    for (let j = 0; j < np; j++) for (let i = 0; i < np; i++) {
      const k = e * nn + j * np + i;
      X[k] = box[0] + ex * hx + (r[i] + 1) * hx / 2;
      Y[k] = box[2] + ey * hy + (r[j] + 1) * hy / 2;
      const v = vel(X[k], Y[k]); AX[k] = v[0]; AY[k] = v[1];
    }
  }
  // flux and its derivative (normal speed) at node k in direction d (0 = x, 1 = y)
  const flux = (u, k, d) => (burg ? 0.5 * u * u : (d === 0 ? AX[k] : AY[k]) * u);
  const speed = (u, k, d) => (burg ? Math.abs(u) : Math.abs(d === 0 ? AX[k] : AY[k]));
  // Rusanov flux in the +d direction between left state uL (node kL) and right state uR (node kR).
  // For advection the velocity is continuous, so AX[kL] = AX[kR] (up to round-off).
  const numFlux = (uL, kL, uR, kR, d) => {
    const fL = flux(uL, kL, d), fR = flux(uR, kR, d);
    const a = central ? 0 : Math.max(speed(uL, kL, d), speed(uR, kR, d));
    return 0.5 * (fL + fR) - 0.5 * a * (uR - uL);
  };

  const fline = new Float64Array(np);
  /**
   * du/dt.
   * @param {Float64Array} u
   * @param {number} t
   * @param {Float64Array} out
   */
  function rhs(u, t, out) {
    out.fill(0);
    const sx = 2 / hx, sy = 2 / hy;
    for (let ey = 0; ey < ny; ey++) for (let ex = 0; ex < nx; ex++) {
      const e = ey * nx + ex, base = e * nn;
      const eL = ey * nx + (ex + nx - 1) % nx, eR = ey * nx + (ex + 1) % nx;
      const eB = ((ey + ny - 1) % ny) * nx + ex, eT = ((ey + 1) % ny) * nx + ex;
      // x-lines
      for (let j = 0; j < np; j++) {
        const row = base + j * np;
        for (let m = 0; m < np; m++) fline[m] = flux(u[row + m], row + m, 0);
        for (let i = 0; i < np; i++) {
          let s = 0;
          for (let m = 0; m < np; m++) s += D[i * np + m] * fline[m];
          out[row + i] -= sx * s;
        }
        const kLn = eL * nn + j * np + p, kRn = eR * nn + j * np; // neighbour face nodes
        const Fl = numFlux(u[kLn], kLn, u[row], row, 0);
        const Fr = numFlux(u[row + p], row + p, u[kRn], kRn, 0);
        out[row] += sx * (Fl - fline[0]) / w[0];
        out[row + p] -= sx * (Fr - fline[p]) / w[p];
      }
      // y-lines
      for (let i = 0; i < np; i++) {
        for (let m = 0; m < np; m++) fline[m] = flux(u[base + m * np + i], base + m * np + i, 1);
        for (let j = 0; j < np; j++) {
          let s = 0;
          for (let m = 0; m < np; m++) s += D[j * np + m] * fline[m];
          out[base + j * np + i] -= sy * s;
        }
        const k0 = base + i, kp = base + p * np + i;
        const kBn = eB * nn + p * np + i, kTn = eT * nn + i;
        const Fb = numFlux(u[kBn], kBn, u[k0], k0, 1);
        const Ft = numFlux(u[kp], kp, u[kTn], kTn, 1);
        out[k0] += sy * (Fb - fline[0]) / w[0];
        out[kp] -= sy * (Ft - fline[p]) / w[p];
      }
    }
  }

  // Gauss rule for projection and errors
  const G = gaussLegendre(p + 4), ng = G.x.length;
  const IG = interpMatrix(r, G.x); // ng × np
  /** L² projection onto Q_p per element (tensor Gauss, exact mass = tensor of 1D masses). */
  function project(f) {
    // 1D exact mass matrix of the GLL Lagrange basis and its inverse (tensor structure)
    const M1 = new Float64Array(np * np);
    for (let a = 0; a < np; a++) for (let b = 0; b < np; b++) {
      let s = 0; for (let q = 0; q < ng; q++) s += G.w[q] * IG[q * np + a] * IG[q * np + b];
      M1[a * np + b] = s;
    }
    const Mi = inverse(M1, np);
    const u = new Float64Array(n), bvec = new Float64Array(nn), tmp = new Float64Array(nn);
    for (let ey = 0; ey < ny; ey++) for (let ex = 0; ex < nx; ex++) {
      const e = ey * nx + ex;
      bvec.fill(0);
      for (let qj = 0; qj < ng; qj++) for (let qi = 0; qi < ng; qi++) {
        const fx = f(box[0] + ex * hx + (G.x[qi] + 1) * hx / 2, box[2] + ey * hy + (G.x[qj] + 1) * hy / 2) * G.w[qi] * G.w[qj];
        for (let j = 0; j < np; j++) for (let i = 0; i < np; i++) bvec[j * np + i] += fx * IG[qi * np + i] * IG[qj * np + j];
      }
      // apply (Mi ⊗ Mi): first along i, then along j
      for (let j = 0; j < np; j++) for (let i = 0; i < np; i++) { let s = 0; for (let m = 0; m < np; m++) s += Mi[i * np + m] * bvec[j * np + m]; tmp[j * np + i] = s; }
      for (let j = 0; j < np; j++) for (let i = 0; i < np; i++) { let s = 0; for (let m = 0; m < np; m++) s += Mi[j * np + m] * tmp[m * np + i]; u[e * nn + j * np + i] = s; }
    }
    return u;
  }
  /** Nodal interpolation. */
  function interpolate(f) { const u = new Float64Array(n); for (let k = 0; k < n; k++) u[k] = f(X[k], Y[k]); return u; }
  const lx = new Float64Array(np), ly = new Float64Array(np);
  /** Value of u_h in element e at reference point (ξ, η). */
  function evalRef(u, e, xi, eta) {
    lagrangeValues(r, xi, lx); lagrangeValues(r, eta, ly);
    let s = 0;
    for (let j = 0; j < np; j++) { let t = 0; for (let i = 0; i < np; i++) t += lx[i] * u[e * nn + j * np + i]; s += ly[j] * t; }
    return s;
  }
  /** ‖u_h − f‖_{L²} with (p+4)² Gauss points per element. */
  function l2Error(u, f) {
    let err = 0;
    for (let ey = 0; ey < ny; ey++) for (let ex = 0; ex < nx; ex++) {
      const e = ey * nx + ex;
      for (let qj = 0; qj < ng; qj++) for (let qi = 0; qi < ng; qi++) {
        let s = 0;
        for (let j = 0; j < np; j++) { let t = 0; for (let i = 0; i < np; i++) t += IG[qi * np + i] * u[e * nn + j * np + i]; s += IG[qj * np + j] * t; }
        const d = s - f(box[0] + ex * hx + (G.x[qi] + 1) * hx / 2, box[2] + ey * hy + (G.x[qj] + 1) * hy / 2);
        err += G.w[qi] * G.w[qj] * d * d;
      }
    }
    return Math.sqrt(err * hx * hy / 4);
  }
  /** Discrete total mass Σ w_i w_j u_ij hx hy/4 — conserved exactly by the scheme (telescoping face fluxes). */
  function mass(u) {
    let s = 0;
    for (let e = 0; e < nE; e++) for (let j = 0; j < np; j++) for (let i = 0; i < np; i++) s += w[i] * w[j] * u[e * nn + j * np + i];
    return s * hx * hy / 4;
  }
  /** max over nodes of |a_x|/hx + |a_y|/hy (advective "inverse time scale"). */
  function maxRate(u) {
    let s = 0;
    for (let k = 0; k < n; k++) s = Math.max(s, burg ? Math.abs(u[k]) * (1 / hx + 1 / hy) : Math.abs(AX[k]) / hx + Math.abs(AY[k]) / hy);
    return s;
  }
  /** Time step Δt = cfl / ((2p+1) · max(|a_x|/hx + |a_y|/hy)). */
  const dtStable = (u, cfl = 0.9) => cfl / ((2 * p + 1) * Math.max(maxRate(u), 1e-12));
  return { p, np, nn, nx, ny, nE, n, hx, hy, box, r, w, X, Y, rhs, project, interpolate, evalRef, l2Error, mass, maxRate, dtStable };
}
