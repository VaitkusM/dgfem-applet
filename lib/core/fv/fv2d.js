/**
 * @file Finite volumes for 2D scalar transport  u_t + ∇·(a u) = 0  on a
 *       Cartesian grid, with a given (time-independent) velocity field a(x, y).
 *
 * Grid: nx × ny cells of size hx × hy on box = [x0, x1] × [y0, y1];
 * cell (i, j) has index c = j·nx + i (row-major, i along x). Unknowns are
 * cell averages ū_c.
 * Integrating the PDE over a cell and using the divergence theorem gives
 * the exact semi-discrete update
 *   dū_ij/dt = −(F_{i+½,j} − F_{i−½,j})/hx − (G_{i,j+½} − G_{i,j−½})/hy,
 * where F, G are the AVERAGE normal fluxes a_x u (resp. a_y u) over the
 * vertical (resp. horizontal) faces. We approximate each face average by
 * the midpoint rule (2nd-order accurate) with the upwind flux
 *   F̂ = a_n · (a_n ≥ 0 ? u⁻ : u⁺),   a_n = normal velocity at the face midpoint.
 * Face values u⁻/u⁺ come from a "dimension-by-dimension" reconstruction:
 *  - first order: the cell averages themselves;
 *  - MUSCL: u = ū_c ± ½ σ^x_c (x-faces) or ± ½ σ^y_c (y-faces) where the
 *    slope (times h) in each direction is limited separately with a 1D
 *    limiter φ(backward difference, forward difference) from fv1d.js.
 * Boundary conditions with 2 ghost layers:
 *  - 'periodic';
 *  - 'inflow': ghost cells hold the constant `inflowValue` (default 0), so
 *    whatever enters through an inflow face carries that value; at outflow
 *    faces the upwind flux uses the interior value automatically.
 * Velocity is sampled once at the face midpoints. If a = (a_x(y), a_y(x))
 * (e.g. solid-body rotation or a constant), these samples are the exact
 * face averages and the discrete divergence of the face velocities is zero.
 */
import { LIMITERS } from './fv1d.js';
import { gaussLegendre } from '../quad/gauss1d.js';

/**
 * Create the 2D FV discretisation.
 * @param {{nx: number, ny: number, box?: number[], vel: (x: number, y: number) => number[],
 *          recon?: 'none'|'muscl', limiter?: string, bc?: 'periodic'|'inflow', inflowValue?: number}} o
 * @returns {{nx: number, ny: number, hx: number, hy: number, box: number[],
 *   rhs: (u: Float64Array, t: number, out: Float64Array) => void,
 *   stableDt: (cfl: number) => number, netOutflow: () => number,
 *   xc: (i: number) => number, yc: (j: number) => number}}
 *   netOutflow(): ∮ F̂·n ds over the domain boundary in the last rhs call, so that
 *   d/dt Σ ū_c hx hy = −netOutflow() (exact discrete conservation).
 */
export function makeFV2D(o) {
  const nx = o.nx, ny = o.ny, box = o.box || [0, 1, 0, 1];
  const hx = (box[1] - box[0]) / nx, hy = (box[3] - box[2]) / ny;
  const muscl = o.recon === 'muscl', lim = LIMITERS[o.limiter || 'minmod'].fn;
  const periodic = (o.bc || 'periodic') === 'periodic', inflowValue = o.inflowValue ?? 0;
  const G = 2, ex = nx + 2 * G, ey = ny + 2 * G;
  // face-normal velocities at face midpoints
  const ax = new Float64Array((nx + 1) * ny), ay = new Float64Array(nx * (ny + 1));
  for (let j = 0; j < ny; j++) for (let i = 0; i <= nx; i++) ax[j * (nx + 1) + i] = o.vel(box[0] + i * hx, box[2] + (j + 0.5) * hy)[0];
  for (let j = 0; j <= ny; j++) for (let i = 0; i < nx; i++) ay[j * nx + i] = o.vel(box[0] + (i + 0.5) * hx, box[2] + j * hy)[1];
  const ue = new Float64Array(ex * ey), sx = new Float64Array(ex * ey), sy = new Float64Array(ex * ey);
  const Fx = new Float64Array((nx + 1) * ny), Fy = new Float64Array(nx * (ny + 1));
  let outflow = 0;

  function fillGhosts(u) {
    for (let J = 0; J < ey; J++) {
      let j = J - G;
      const jin = j >= 0 && j < ny;
      if (periodic) j = (j + ny) % ny;
      for (let I = 0; I < ex; I++) {
        let i = I - G;
        const iin = i >= 0 && i < nx;
        if (periodic) i = (i + nx) % nx;
        ue[J * ex + I] = periodic || (iin && jin) ? u[j * nx + i] : inflowValue;
      }
    }
  }

  function rhs(u, t, out) {
    fillGhosts(u);
    if (muscl) {
      for (let J = 1; J < ey - 1; J++) for (let I = 1; I < ex - 1; I++) {
        const k = J * ex + I, v = ue[k];
        sx[k] = lim(v - ue[k - 1], ue[k + 1] - v);
        sy[k] = lim(v - ue[k - ex], ue[k + ex] - v);
      }
    }
    outflow = 0;
    // x-faces: face i between cells i−1 and i of row j
    for (let j = 0; j < ny; j++) for (let i = 0; i <= nx; i++) {
      const kL = (j + G) * ex + (i - 1 + G), kR = kL + 1, a = ax[j * (nx + 1) + i];
      const uL = muscl ? ue[kL] + 0.5 * sx[kL] : ue[kL], uR = muscl ? ue[kR] - 0.5 * sx[kR] : ue[kR];
      const f = a >= 0 ? a * uL : a * uR;
      Fx[j * (nx + 1) + i] = f;
      if (i === nx) outflow += f * hy; else if (i === 0) outflow -= f * hy;
    }
    // y-faces: face j between cells j−1 and j of column i
    for (let j = 0; j <= ny; j++) for (let i = 0; i < nx; i++) {
      const kB = (j - 1 + G) * ex + (i + G), kT = kB + ex, a = ay[j * nx + i];
      const uB = muscl ? ue[kB] + 0.5 * sy[kB] : ue[kB], uT = muscl ? ue[kT] - 0.5 * sy[kT] : ue[kT];
      const f = a >= 0 ? a * uB : a * uT;
      Fy[j * nx + i] = f;
      if (j === ny) outflow += f * hx; else if (j === 0) outflow -= f * hx;
    }
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++)
      out[j * nx + i] = -(Fx[j * (nx + 1) + i + 1] - Fx[j * (nx + 1) + i]) / hx - (Fy[(j + 1) * nx + i] - Fy[j * nx + i]) / hy;
  }

  /** Largest stable forward-Euler step of the first-order scheme times cfl: dt·max_c(|a_x|/hx + |a_y|/hy) = cfl. */
  function stableDt(cfl) {
    let m = 0;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const vx = Math.max(Math.abs(ax[j * (nx + 1) + i]), Math.abs(ax[j * (nx + 1) + i + 1]));
      const vy = Math.max(Math.abs(ay[j * nx + i]), Math.abs(ay[(j + 1) * nx + i]));
      m = Math.max(m, vx / hx + vy / hy);
    }
    return cfl / Math.max(m, 1e-300);
  }

  return {
    nx, ny, hx, hy, box, rhs, stableDt, netOutflow: () => outflow,
    xc: (i) => box[0] + (i + 0.5) * hx, yc: (j) => box[2] + (j + 0.5) * hy,
  };
}

/**
 * Cell averages of f over the grid (q×q tensor Gauss rule per cell).
 * For a DISCONTINUOUS f (indicator functions such as Zalesak's disk) a Gauss rule on a cell cut by the jump is
 * wrong by O(1) in that cell. With `refine` = r > 0, every cell in which the samples (Gauss points, corners,
 * edge midpoints) are not all equal is
 * bisected recursively (up to r times in each direction), so only sub-cells of size h/2^r that are still cut
 * carry a quadrature error: the error of a cut cell's average drops by about 2^−r (measured for the slotted
 * disk on 100² cells: total mass error 6e-5 with q = 6 and r = 0, 8e-7 with q = 4 and r = 5).
 * @param {(x: number, y: number) => number} f
 * @param {number} nx @param {number} ny @param {number[]} box
 * @param {number} [q=4] Gauss points per direction
 * @param {number} [refine=0] maximal number of recursive bisections of cells with non-constant samples
 *   (0 = plain rule, the right choice for smooth f)
 * @returns {Float64Array} length nx·ny, index j·nx + i
 */
export function cellAverages2D(f, nx, ny, box, q = 4, refine = 0) {
  const { x: gx, w: gw } = gaussLegendre(q);
  const hx = (box[1] - box[0]) / nx, hy = (box[3] - box[2]) / ny, u = new Float64Array(nx * ny);
  // average over the rectangle with lower-left corner (x0, y0) and size wx × wy
  const avg = (x0, y0, wx, wy, level) => {
    let s = 0, lo = Infinity, hi = -Infinity;
    for (let a = 0; a < q; a++) for (let b = 0; b < q; b++) {
      const v = f(x0 + (0.5 + 0.5 * gx[a]) * wx, y0 + (0.5 + 0.5 * gx[b]) * wy);
      s += gw[a] * gw[b] * v; if (v < lo) lo = v; if (v > hi) hi = v;
    }
    if (level <= 0) return 0.25 * s;
    // "cut" test: also look at the corners and edge midpoints, so that a jump clipping only a corner is seen
    for (let a = 0; a <= 2; a++) for (let b = 0; b <= 2; b++) {
      if (a === 1 && b === 1) continue;
      const v = f(x0 + 0.5 * a * wx, y0 + 0.5 * b * wy); if (v < lo) lo = v; if (v > hi) hi = v;
    }
    if (hi === lo) return 0.25 * s;
    const ax = 0.5 * wx, ay = 0.5 * wy;
    return 0.25 * (avg(x0, y0, ax, ay, level - 1) + avg(x0 + ax, y0, ax, ay, level - 1)
      + avg(x0, y0 + ay, ax, ay, level - 1) + avg(x0 + ax, y0 + ay, ax, ay, level - 1));
  };
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) u[j * nx + i] = avg(box[0] + i * hx, box[2] + j * hy, hx, hy, refine);
  return u;
}

/**
 * Exact area (= mass) of Zalesak's slotted disk: disk of radius r minus the part of the slot inside it.
 * The slot |x − ½| < w/2 runs from the bottom of the disk up to the height d above the centre:
 *   area = π r² − ∫_{−w/2}^{w/2} (d + √(r² − s²)) ds.
 * @param {number} [r=0.15] @param {number} [w=0.05] slot width @param {number} [d=0.1] slot top above the centre
 * @returns {number}
 */
export function zalesakArea(r = 0.15, w = 0.05, d = 0.1) {
  const a = 0.5 * w;
  return Math.PI * r * r - (w * d + a * Math.sqrt(r * r - a * a) + r * r * Math.asin(a / r));
}

/**
 * Zalesak's slotted disk (Zalesak 1979) on the unit square: disk of radius 0.15
 * centred at (0.5, 0.75) with a vertical slot of width 0.05 reaching up to
 * y = 0.85 from the bottom of the disk. Value 1 inside, 0 outside.
 */
export function zalesakDisk(x, y) {
  const dx = x - 0.5, dy = y - 0.75;
  if (dx * dx + dy * dy > 0.15 * 0.15) return 0;
  if (Math.abs(dx) < 0.025 && y < 0.85) return 0;
  return 1;
}

/**
 * Solid-body rotation about (0.5, 0.5) with angular velocity ω (one revolution in 2π/ω):
 * a = ω (−(y − ½), x − ½). Divergence-free; a_x depends only on y and a_y only on x.
 * @param {number} [omega=2π]
 */
export const rotationVelocity = (omega = 2 * Math.PI) => (x, y) => [-omega * (y - 0.5), omega * (x - 0.5)];

/**
 * Exact solution of u_t + a·∇u = 0 for solid-body rotation: u0 rotated by angle ω t about (½, ½).
 * @param {(x: number, y: number) => number} u0
 * @param {number} omega @param {number} t
 * @returns {(x: number, y: number) => number}
 */
export function rotated(u0, omega, t) {
  const c = Math.cos(omega * t), s = Math.sin(omega * t);
  return (x, y) => { const dx = x - 0.5, dy = y - 0.5; return u0(0.5 + c * dx + s * dy, 0.5 - s * dx + c * dy); };
}
