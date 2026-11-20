/**
 * @file The 1D Poisson problem  −u'' = f  on (0,1),  u(0) = g0, u(1) = g1,
 *       discretised in three different "forms" on a uniform mesh of N cells
 *       (h = 1/N, nodes x_i = i h):
 *
 *  - 'fd'    strong form, finite differences at the nodes:
 *              (−u_{i−1} + 2u_i − u_{i+1}) / h² = f(x_i)
 *  - 'cg'    weak (primal) form, continuous piecewise-linear "hat" functions φ_i:
 *              Σ_j U_j ∫ φ_j' φ_i' = ∫ f φ_i     (stiffness (1/h)·[−1 2 −1])
 *  - 'mixed' mixed form with the flux σ = −u' as an extra unknown:
 *              ∫ σ τ − ∫ u τ' = −[g τ n]_{∂}   for all continuous P1 τ
 *              ∫ σ' v        = ∫ f v           for all piecewise-constant v
 *            σ_h is continuous piecewise linear, u_h piecewise constant.
 *            The 2nd equation says σ_h(x_{i+1}) − σ_h(x_i) = ∫_{cell i} f:
 *            the discrete flux balances the source in every cell exactly.
 *            The 2nd equation is assembled multiplied by −1, so that the
 *            matrix has the symmetric saddle-point form [M Bᵀ; B 0] with
 *            B_{ij} = −∫ v_i τ_j' (N+1 positive and N negative eigenvalues).
 *
 * All three solve a small (banded) linear system with sparseLU.
 *
 * Load integrals ∫ f φ are computed with a composite Gauss rule (4 × 8 points
 * per cell), so that also sharply peaked sources on very coarse meshes are
 * integrated to round-off — the nodal exactness of 1D linear elements is
 * only visible if the load vector is (numerically) exact.
 */
import { SparseBuilder } from '../la/sparse.js';
import { sparseLU } from '../la/direct.js';
import { gaussLegendre } from '../quad/gauss1d.js';

const G = gaussLegendre(8);
const SUB = 4; // sub-intervals per call of integrate()

/**
 * ∫_a^b fn(x) dx with a composite Gauss rule: SUB = 4 equal sub-intervals with
 * 8 Gauss points each (exact for polynomials of degree ≤ 15 on each piece).
 * @param {number} a
 * @param {number} b
 * @param {(x:number)=>number} fn
 * @returns {number}
 */
export function integrate(a, b, fn) {
  const hs = (b - a) / SUB;
  let s = 0;
  for (let k = 0; k < SUB; k++) {
    const x0 = a + k * hs;
    for (let q = 0; q < G.x.length; q++) s += G.w[q] * fn(x0 + (G.x[q] + 1) * hs / 2);
  }
  return s * hs / 2;
}

/**
 * Energy J(v) = ½ ∫₀¹ v'² dx − ∫₀¹ f v dx of a continuous piecewise-linear
 * function v on the uniform mesh with N = v.length − 1 cells (Dirichlet
 * principle). The first term is exact, the load term uses the same quadrature
 * as the 'cg' right-hand side, so the 'cg' solution is the exact minimiser of
 * this J among all v with the same boundary values.
 * @param {ArrayLike<number>} v nodal values v_0..v_N (length N+1)
 * @param {(x:number)=>number} f
 * @returns {number}
 */
export function energyP1(v, f) {
  const N = v.length - 1, h = 1 / N;
  let s = 0;
  for (let i = 0; i < N; i++) {
    const a = i * h, c = (i + 1) * h, d = (v[i + 1] - v[i]) / h;
    s += 0.5 * d * d * h;
    s -= integrate(a, c, (t) => f(t) * (v[i] * (c - t) + v[i + 1] * (t - a)) / h);
  }
  return s;
}

/**
 * Solve −u'' = f on (0,1) with Dirichlet data.
 * @param {'fd'|'cg'|'mixed'} form
 * @param {number} N number of cells
 * @param {(x:number)=>number} f
 * @param {number} g0 u(0)
 * @param {number} g1 u(1)
 * @returns {{x: Float64Array, U?: Float64Array, uCell?: Float64Array, sigma?: Float64Array, A: import('../la/sparse.js').CSR, b: Float64Array}}
 *   fd/cg: U = nodal values (N+1, boundary included); mixed: uCell (N) and sigma (N+1 nodal values of σ_h)
 */
export function solvePoisson1D(form, N, f, g0, g1) {
  const h = 1 / N, x = new Float64Array(N + 1);
  for (let i = 0; i <= N; i++) x[i] = i * h;
  if (form === 'fd' || form === 'cg') {
    // unknowns: interior nodes 1..N−1 (index i−1)
    const n = N - 1, B = new SparseBuilder(n), b = new Float64Array(n);
    const s = form === 'fd' ? 1 / (h * h) : 1 / h;
    for (let i = 1; i < N; i++) {
      const r = i - 1;
      B.add(r, r, 2 * s);
      if (i > 1) B.add(r, r - 1, -s); else b[r] += s * g0;
      if (i < N - 1) B.add(r, r + 1, -s); else b[r] += s * g1;
      if (form === 'fd') b[r] += f(x[i]);
      else { // ∫ f φ_i over the two neighbouring cells (φ_i = hat function)
        b[r] += integrate(x[i - 1], x[i], (t) => f(t) * (t - x[i - 1]) / h);
        b[r] += integrate(x[i], x[i + 1], (t) => f(t) * (x[i + 1] - t) / h);
      }
    }
    const A = B.toCSR(), Ui = sparseLU(A).solve(b);
    const U = new Float64Array(N + 1);
    U[0] = g0; U[N] = g1; U.set(Ui, 1);
    return { x, U, A, b };
  }
  if (form === 'mixed') {
    // unknowns: σ_0..σ_N (N+1), then u_0..u_{N−1} (N)
    const ns = N + 1, n = ns + N, B = new SparseBuilder(n), b = new Float64Array(n);
    for (let i = 0; i < N; i++) {
      // mass matrix of P1 on cell i: h/6 [2 1; 1 2]
      B.add(i, i, h / 3); B.add(i, i + 1, h / 6); B.add(i + 1, i, h / 6); B.add(i + 1, i + 1, h / 3);
      // −∫ u τ' : τ_i' = −1/h, τ_{i+1}' = +1/h on cell i, u = u_i (constant)
      B.add(i, ns + i, 1); B.add(i + 1, ns + i, -1);
      // −∫ σ' v = −∫ f v (v = indicator of cell i): −(σ_{i+1} − σ_i) = −∫_cell f
      // (sign flipped so that this block is the transpose of the one above)
      B.add(ns + i, i + 1, -1); B.add(ns + i, i, 1);
      b[ns + i] = -integrate(x[i], x[i + 1], f);
    }
    // boundary term −[g τ n]: at x=1 (n=+1) −g1 τ_N(1); at x=0 (n=−1) +g0 τ_0(0)
    b[N] += -g1; b[0] += g0;
    const A = B.toCSR(), sol = sparseLU(A).solve(b);
    return { x, sigma: sol.slice(0, ns), uCell: sol.slice(ns), A, b };
  }
  throw new Error(`unknown form ${form}`);
}

/**
 * L² error of the discrete solution against the exact u, using the natural
 * reconstruction of each form (fd/cg: piecewise linear through nodal values;
 * mixed: piecewise constant u_h), plus the L² error of the flux σ = −u'
 * (fd/cg: −(U_{i+1}−U_i)/h per cell; mixed: σ_h piecewise linear).
 * For the mixed form, `uAvgMax` is in addition the largest difference between
 * u_h on a cell and the exact cell average of u (a "superconvergent" quantity).
 * @param {'fd'|'cg'|'mixed'} form
 * @param {ReturnType<typeof solvePoisson1D>} sol
 * @param {(x:number)=>number} u exact solution
 * @param {(x:number)=>number} du exact derivative u'
 * @returns {{uL2: number, sigmaL2: number, uAvgMax?: number}}
 */
export function errors1D(form, sol, u, du) {
  const { x } = sol, N = x.length - 1, h = 1 / N;
  let eu = 0, es = 0, ea = 0;
  for (let i = 0; i < N; i++) {
    const a = x[i], c = x[i + 1];
    if (form === 'mixed') {
      ea = Math.max(ea, Math.abs(integrate(a, c, u) / h - sol.uCell[i]));
      eu += integrate(a, c, (t) => (u(t) - sol.uCell[i]) ** 2);
      es += integrate(a, c, (t) => (-du(t) - (sol.sigma[i] * (c - t) + sol.sigma[i + 1] * (t - a)) / h) ** 2);
    } else {
      eu += integrate(a, c, (t) => (u(t) - (sol.U[i] * (c - t) + sol.U[i + 1] * (t - a)) / h) ** 2);
      es += integrate(a, c, (t) => (-du(t) + (sol.U[i + 1] - sol.U[i]) / h) ** 2);
    }
  }
  return form === 'mixed' ? { uL2: Math.sqrt(eu), sigmaL2: Math.sqrt(es), uAvgMax: ea } : { uL2: Math.sqrt(eu), sigmaL2: Math.sqrt(es) };
}
