/**
 * @file CPU "twin" of the WebGPU DG advection solver (f64, DOM-free).
 *
 * PROBLEM. Linear advection  u_t + a·∇u = 0  with a constant velocity
 * a = (ax, ay) on the periodic unit square [0, L]², discretised on a uniform
 * N×N grid of square elements of side h = L/N.
 *
 * METHOD. Nodal DG spectral element method (DG-SEM):
 *  - per element a tensor-product Lagrange basis of degree p in each direction
 *    on the (p+1)×(p+1) Gauss–Lobatto–Legendre (GLL) nodes ξ_0 < … < ξ_p of [-1,1];
 *  - quadrature *collocated* with the nodes (GLL quadrature on the same points),
 *    so the mass matrix is diagonal ("lumped"): M = J·diag(w_i w_j), J = h²/4.
 *    This is not the exact mass matrix (GLL with p+1 points is exact only up to
 *    degree 2p−1) but it is the standard DG-SEM choice and makes M⁻¹ free;
 *  - upwind numerical flux  F̂(u⁻, u⁺, n) = (a·n) u⁻ if a·n ≥ 0, else (a·n) u⁺
 *    (= flux out of K⁻ through the face, AGENTS.md convention);
 *  - strong form (equivalent to the weak form for GLL collocation because the
 *    GLL differentiation matrix has the summation-by-parts property):
 *
 *      du_ij/dt = −(2/h)(ax Σ_k D_ik u_kj + ay Σ_k D_jk u_ik)
 *                 − Σ_{faces of K touching node ij} (2/h)(1/w_0) (F̂ − (a·n_K) u⁻)
 *
 *    The surface term follows from  ∫_{∂K}(F̂ − a·n u⁻) v ds  with the face
 *    quadrature weight (h/2)w_m divided by the mass entry (h²/4) w_0 w_m:
 *    (h/2)/(h²/4)/w_0 = (2/h)/w_0, where w_0 = w_p = 2/(p(p+1)) is the GLL
 *    end-point weight. "lift" = 1/w_0.
 *  - time stepping: Carpenter–Kennedy 5-stage 4th-order low-storage RK (LSRK4):
 *      for s = 0..4:  k ← A_s k + dt L(u);  u ← u + B_s k
 *    (two registers per DOF — memory is the scarce resource on a GPU).
 *
 * DATA LAYOUT (identical to the GPU buffers; "structure of arrays", element-major):
 *  - element  e = ey·N + ex  (ex = column, ey = row, both 0..N−1);
 *  - local node  l = j·n + i, n = p+1, i = x-index, j = y-index;
 *  - global DOF  g = e·Np + l,  Np = n²;   u[g] is a Float64Array of length N²·Np.
 *  - face buffer (one value per face node, written by exactly one thread):
 *      flux[0·N²n + e·n + m]  = F̂ on the RIGHT face of element e (normal +x),
 *                               face node m = j (its y-index);
 *      flux[1·N²n + e·n + m]  = F̂ on the TOP face of element e (normal +y),
 *                               face node m = i (its x-index).
 *    Every face is owned by the element on its left/bottom (K⁻), so the
 *    face kernel never writes the same address twice: race-free.
 *
 * KERNELS (each loop below is one GPU dispatch; the loop body is one GPU
 * invocation = "thread"):
 *  1. volume  : one thread per DOF:  rhs[g] = −(ax rx Σ D u + ay ry Σ D u)
 *  2. surface : one thread per face node: flux[f] = upwind F̂
 *  3. lift    : one thread per DOF: gathers the (up to 2) face fluxes of its
 *               element faces and adds the surface term to rhs[g]
 *  4. rkStage : one thread per DOF: k = A k + dt rhs; uOut = uIn + B k
 */
import { gaussLobatto, gaussLegendre } from '../core/quad/gauss1d.js';
import { diffMatrix, interpMatrix, lagrangeValues } from '../core/basis/lagrange.js';

/** Carpenter & Kennedy (1994) 5-stage 4th-order 2N-storage RK coefficients (same as lib/core/time/rk.js 'lsrk4'). */
export const LSRK4 = {
  A: [0, -567301805773 / 1357537059087, -2404267990393 / 2016746695238,
    -3550918686646 / 2091501179385, -1275806237668 / 842570457699],
  B: [1432997174477 / 9575080441755, 5161836677717 / 13612068292357,
    1720146321549 / 2090206949498, 3134564353537 / 4481467310338, 2277821191437 / 14882151754819],
  C: [0, 1432997174477 / 9575080441755, 2526269341429 / 6820363962896,
    2006345519317 / 3224310063776, 2802321613138 / 2924317926251],
};

/**
 * Empirical CFL constant: dt = CFL_SAFE · h / ((|ax|+|ay|)(2p+1)) is stable for
 * LSRK4 + upwind DG-SEM, p = 1..3 (checked by tests/unit/gpu-cputwin.test.js
 * via the spectral radius of the semi-discrete operator).
 */
export const CFL_SAFE = 1.0;

/**
 * Build the discretisation (geometry, reference operators, kernels).
 * @param {{N: number, p: number, a?: number[], L?: number}} o
 *   N elements per direction (N ≥ 2), degree p ≥ 1, velocity a = [ax, ay], domain side L (default 1)
 * @returns {object} disc with fields
 *   N, p, n (= p+1), Np (= n²), nElem (= N²), nDof (= N²Np), nFace (= 2N²n face nodes),
 *   h, ax, ay, rx = 2/h, lift = 1/w_0, xi (GLL nodes), w (GLL weights), D (n×n row-major, D[i*n+k] = ℓ_k'(ξ_i)),
 *   X, Y (Float64Array nDof: physical node coordinates), and the kernel functions documented below.
 */
export function makeDGAdv2D(o) {
  const N = o.N, p = o.p, L = o.L ?? 1;
  const [ax, ay] = o.a ?? [1, 0.5];
  if (!(N >= 2 && p >= 1)) throw new Error('makeDGAdv2D: need N >= 2, p >= 1');
  const n = p + 1, Np = n * n, nElem = N * N, nDof = nElem * Np, nFace = 2 * nElem * n;
  const h = L / N, rx = 2 / h, ry = 2 / h;
  const { x: xi, w } = gaussLobatto(n);
  const D = diffMatrix(xi);
  const lift = 1 / w[0];

  const X = new Float64Array(nDof), Y = new Float64Array(nDof);
  for (let e = 0; e < nElem; e++) {
    const ex = e % N, ey = (e / N) | 0;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const g = e * Np + j * n + i;
      X[g] = (ex + (xi[i] + 1) / 2) * h;
      Y[g] = (ey + (xi[j] + 1) / 2) * h;
    }
  }

  /**
   * Kernel 1 — volume term (sum factorised: O(n) work per DOF, O(p³) per element in 2D).
   * @param {Float64Array} u nDof
   * @param {Float64Array} rhs nDof (overwritten)
   */
  function volume(u, rhs) {
    for (let g = 0; g < nDof; g++) { // ← one GPU thread per g
      const e = (g / Np) | 0, l = g - e * Np, i = l % n, j = (l / n) | 0, base = e * Np;
      let dx = 0, dy = 0;
      for (let k = 0; k < n; k++) {
        dx += D[i * n + k] * u[base + j * n + k]; // ∂/∂ξ along the row j
        dy += D[j * n + k] * u[base + k * n + i]; // ∂/∂η along the column i
      }
      rhs[g] = -(ax * rx * dx + ay * ry * dy);
    }
  }

  /**
   * Kernel 2 — surface: upwind flux at every face node, written to its own slot.
   * @param {Float64Array} u nDof
   * @param {Float64Array} flux nFace (overwritten), layout see file header
   */
  function surface(u, flux) {
    const half = nElem * n;
    for (let f = 0; f < nFace; f++) { // ← one GPU thread per face node
      const dir = f >= half ? 1 : 0, r = f - dir * half, e = (r / n) | 0, m = r - e * n;
      const ex = e % N, ey = (e / N) | 0;
      let uM, uP, an;
      if (dir === 0) { // right face of e: K⁻ = e, K⁺ = right neighbour, n = (+1, 0)
        const eR = ey * N + (ex + 1) % N;
        uM = u[e * Np + m * n + (n - 1)]; uP = u[eR * Np + m * n]; an = ax;
      } else { // top face of e: K⁺ = upper neighbour, n = (0, +1)
        const eT = ((ey + 1) % N) * N + ex;
        uM = u[e * Np + (n - 1) * n + m]; uP = u[eT * Np + m]; an = ay;
      }
      flux[f] = an >= 0 ? an * uM : an * uP; // upwind
    }
  }

  /**
   * Kernel 3 — lift: each DOF on the element boundary gathers the face fluxes of
   * its faces and adds −(2/h)(1/w_0)(F̂_out − (a·n_K) u) for each of them.
   * For the left/bottom face, K is the K⁺ side: the flux out of K is −F̂ and a·n_K = −a_x (−a_y).
   * @param {Float64Array} u nDof
   * @param {Float64Array} flux nFace
   * @param {Float64Array} rhs nDof (updated in place)
   */
  function liftKernel(u, flux, rhs) {
    const half = nElem * n, cx = rx * lift, cy = ry * lift;
    for (let g = 0; g < nDof; g++) { // ← one GPU thread per g
      const e = (g / Np) | 0, l = g - e * Np, i = l % n, j = (l / n) | 0;
      const ex = e % N, ey = (e / N) | 0, uc = u[g];
      let r = rhs[g];
      if (i === n - 1) r -= cx * (flux[e * n + j] - ax * uc);
      if (i === 0) { const eL = ey * N + (ex + N - 1) % N; r += cx * (flux[eL * n + j] - ax * uc); }
      if (j === n - 1) r -= cy * (flux[half + e * n + i] - ay * uc);
      if (j === 0) { const eB = ((ey + N - 1) % N) * N + ex; r += cy * (flux[half + eB * n + i] - ay * uc); }
      rhs[g] = r;
    }
  }

  const fluxBuf = new Float64Array(nFace);
  /**
   * Semi-discrete right-hand side L(u) = volume + surface + lift (three kernels).
   * @param {Float64Array} u nDof
   * @param {Float64Array} out nDof (overwritten with du/dt)
   */
  function rhs(u, out) {
    volume(u, out);
    surface(u, fluxBuf);
    liftKernel(u, fluxBuf, out);
  }

  /**
   * Kernel 4 — one LSRK4 stage update (uOut may be the same array as uIn).
   * @param {Float64Array} uIn @param {Float64Array} uOut @param {Float64Array} res residual register k
   * @param {Float64Array} r rhs  @param {number} A @param {number} B @param {number} dt
   */
  function rkStage(uIn, uOut, res, r, A, B, dt) {
    for (let g = 0; g < nDof; g++) { // ← one GPU thread per g
      const k = A * res[g] + dt * r[g];
      res[g] = k;
      uOut[g] = uIn[g] + B * k;
    }
  }

  /** Nodal interpolant of f(x, y). @returns {Float64Array} nDof */
  function interpolate(f) {
    const u = new Float64Array(nDof);
    for (let g = 0; g < nDof; g++) u[g] = f(X[g], Y[g]);
    return u;
  }

  /** Discrete mass  Σ_g J w_i w_j u_g  (= ∫ u_h with the lumped GLL quadrature; conserved). */
  function mass(u) {
    const J = h * h / 4;
    let s = 0;
    for (let e = 0; e < nElem; e++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) s += J * w[i] * w[j] * u[e * Np + j * n + i];
    return s;
  }

  /** Discrete energy Σ_g J w_i w_j u_g² (non-increasing for upwind DG-SEM). */
  function energy(u) {
    const J = h * h / 4;
    let s = 0;
    for (let e = 0; e < nElem; e++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) s += J * w[i] * w[j] * u[e * Np + j * n + i] ** 2;
    return s;
  }

  /**
   * L2 error ‖u_h − u_ex‖ with a (p+3)-point Gauss rule per direction
   * (exact for degree 2p+5 ≥ 2p+4), u_h evaluated from its nodal values.
   * @param {Float64Array} u @param {(x:number, y:number) => number} exact
   */
  function l2Error(u, exact) {
    const G = gaussLegendre(p + 3), nq = G.x.length, I = interpMatrix(xi, G.x), J = h * h / 4;
    const tmp = new Float64Array(nq * n);
    let s = 0;
    for (let e = 0; e < nElem; e++) {
      const ex = e % N, ey = (e / N) | 0, base = e * Np;
      // interpolate in x first (sum factorisation again): tmp[j][qx]
      for (let j = 0; j < n; j++) for (let q = 0; q < nq; q++) {
        let v = 0; for (let i = 0; i < n; i++) v += I[q * n + i] * u[base + j * n + i];
        tmp[j * nq + q] = v;
      }
      for (let qy = 0; qy < nq; qy++) for (let qx = 0; qx < nq; qx++) {
        let v = 0; for (let j = 0; j < n; j++) v += I[qy * n + j] * tmp[j * nq + qx];
        const x = (ex + (G.x[qx] + 1) / 2) * h, y = (ey + (G.x[qy] + 1) / 2) * h;
        s += J * G.w[qx] * G.w[qy] * (v - exact(x, y)) ** 2;
      }
    }
    return Math.sqrt(s);
  }

  const lx = new Float64Array(n), ly = new Float64Array(n);
  /** Evaluate u_h in element e at reference point (ξ, η) ∈ [-1,1]². */
  function evalAt(u, e, s, t) {
    lagrangeValues(xi, s, lx); lagrangeValues(xi, t, ly);
    let v = 0;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) v += ly[j] * lx[i] * u[e * Np + j * n + i];
    return v;
  }

  /** A stable time step  dt = cfl·h / ((|ax|+|ay|)(2p+1)). */
  const stableDt = (cfl = CFL_SAFE) => cfl * h / ((Math.abs(ax) + Math.abs(ay)) * (2 * p + 1));

  return {
    N, p, n, Np, nElem, nDof, nFace, h, L, ax, ay, rx, ry, lift, xi, w, D, X, Y,
    volume, surface, liftKernel, rhs, rkStage, interpolate, mass, energy, l2Error, evalAt, stableDt,
  };
}

/**
 * CPU solver state mirroring the GPU solver: ping-pong u buffers, residual and rhs registers.
 * @param {ReturnType<typeof makeDGAdv2D>} disc
 * @param {Float64Array} u0 initial nodal values (copied)
 * @returns {{u: () => Float64Array, t: number, steps: number, step: (dt: number) => void, run: (nSteps: number, dt: number) => void}}
 */
export function makeCpuSolver(disc, u0) {
  const bufs = [Float64Array.from(u0), new Float64Array(disc.nDof)];
  const res = new Float64Array(disc.nDof), r = new Float64Array(disc.nDof);
  let cur = 0;
  const S = {
    t: 0, steps: 0,
    /** current solution (a view on the live buffer — copy it if you keep it) */
    u: () => bufs[cur],
    /** one LSRK4 step: 5 × (volume, surface, lift, rkStage), ping-ponging u */
    step(dt) {
      res.fill(0);
      for (let s = 0; s < 5; s++) {
        disc.rhs(bufs[cur], r);
        disc.rkStage(bufs[cur], bufs[1 - cur], res, r, LSRK4.A[s], LSRK4.B[s], dt);
        cur = 1 - cur;
      }
      S.t += dt; S.steps++;
    },
    run(nSteps, dt) { for (let k = 0; k < nSteps; k++) S.step(dt); },
  };
  return S;
}

const pblob = (x, y, x0, y0, k) => Math.exp(k * (Math.cos(2 * Math.PI * (x - x0)) + Math.cos(2 * Math.PI * (y - y0)) - 2));

/** Smooth periodic test functions on the unit square. */
export const INITIAL = {
  /** sin(2πx) sin(2πy) — used for convergence tests */
  sines: (x, y) => Math.sin(2 * Math.PI * x) * Math.sin(2 * Math.PI * y),
  /** a smooth, exactly periodic blob centred at (0.3, 0.35): exp(κ(cos 2π(x−x0) + cos 2π(y−y0) − 2)),
   *  κ = 4 (near the centre ≈ exp(−|x−x0|²/0.0127), a Gaussian of width ≈ 0.11) */
  blob: (x, y) => pblob(x, y, 0.3, 0.35, 4),
  /** a positive and a narrower negative blob — more structure to watch */
  twin: (x, y) => pblob(x, y, 0.3, 0.35, 4) - 0.6 * pblob(x, y, 0.7, 0.65, 8),
};

/**
 * Periodic translation of an initial function: u0(x − ax t, y − ay t) wrapped into [0, L)².
 * @param {(x:number, y:number)=>number} u0 @param {number} ax @param {number} ay @param {number} t @param {number} [L=1]
 */
export function translated(u0, ax, ay, t, L = 1) {
  const wrap = (s) => ((s % L) + L) % L;
  return (x, y) => u0(wrap(x - ax * t), wrap(y - ay * t));
}
