/**
 * @file The model problem of chapter 10 (domain decomposition): P1 continuous
 *       Galerkin for the Poisson problem  −Δu = f  in Ω = (0,1)²,  u = 0 on ∂Ω,
 *       on a structured N×N triangulated grid (lib/core/mesh/structured.js triGrid).
 *
 * Discretisation = exactly what lib/core/elliptic/cg.js does with bc 'strong':
 *  - assemble A_ij = ∫ ∇φ_j·∇φ_i and b_i = ∫ f φ_i over the whole mesh,
 *  - eliminate the boundary vertices (g = 0, so no RHS correction is needed),
 *  - keep the "free" (interior) vertices as unknowns.
 *
 * Unknown numbering (used by every module in lib/core/dd/):
 *   grid vertex (i, j), 0 ≤ i, j ≤ N, has mesh id  v = j (N+1) + i;
 *   the interior vertices 1 ≤ i, j ≤ N−1 are the unknowns, with
 *   free index  k = (j−1)(N−1) + (i−1)   (row-major, i along x).
 * This is the same order as solvePoissonCG's `free` list (ascending vertex id).
 *
 * Remark: on this right-isosceles triangulation the P1 stiffness matrix is the
 * classical 5-point stencil (4 on the diagonal, −1 for the 4 axis neighbours):
 * the diagonal edges carry zero coupling because the angles opposite to them are
 * 90° (cot 90° = 0). Its eigenvalues are known in closed form, which the tests use.
 */
import { triGrid } from '../mesh/structured.js';
import { cgSpace, assembleVolume } from '../elliptic/cg.js';
import { csrExtract } from '../la/sparse.js';

/**
 * @typedef {Object} DDProblem
 * @property {number} N          cells per direction (h = 1/N)
 * @property {number} h
 * @property {number} n          number of unknowns (N−1)²
 * @property {{nodes: Float64Array, tris: Int32Array}} mesh
 * @property {Int32Array} free   free index → mesh vertex id
 * @property {import('../la/sparse.js').CSR} A  n×n stiffness matrix on the free vertices (SPD)
 * @property {Float64Array} b    load vector on the free vertices
 * @property {(f: (x:number,y:number)=>number) => Float64Array} load  re-assemble b for another f
 * @property {(cellFilter: (ci:number, cj:number)=>boolean, f?: (x:number,y:number)=>number) =>
 *            {A: import('../la/sparse.js').CSR, b: Float64Array}} localMatrix
 *   "Neumann" matrix assembled only from the elements of the cells (ci, cj) accepted by
 *   the filter (cell (ci,cj) = square [ci h,(ci+1)h]×[cj h,(cj+1)h], both its triangles),
 *   restricted to the free vertices (rows/cols of untouched vertices are zero).
 */

/** Free index of interior grid vertex (i, j), 1 ≤ i, j ≤ N−1. */
export const freeIndex = (N, i, j) => (j - 1) * (N - 1) + (i - 1);

/**
 * Build the model problem.
 * @param {number} N cells per direction
 * @param {(x:number,y:number)=>number} [f] right-hand side (default f = 1)
 * @returns {DDProblem}
 */
export function ddPoissonProblem(N, f = () => 1) {
  const mesh = triGrid(N, N, [0, 1, 0, 1], { diag: 'right' });
  const space = cgSpace(mesh, 1);
  const free = new Int32Array((N - 1) * (N - 1));
  for (let j = 1; j < N; j++) for (let i = 1; i < N; i++) free[freeIndex(N, i, j)] = j * (N + 1) + i;
  const asm = assembleVolume(space, f);
  const A = csrExtract(asm.builder.toCSR(), free, free);
  const restrict = (bf) => Float64Array.from(free, (v) => bf[v]);
  const b = restrict(asm.b);
  const load = (g) => restrict(assembleVolume(space, g).b);
  // triGrid stores 2 triangles per cell, cell (ci, cj) has triangles 2(cj N + ci) and +1
  const localMatrix = (cellFilter, g = f) => {
    const keep = [];
    for (let cj = 0; cj < N; cj++) for (let ci = 0; ci < N; ci++)
      if (cellFilter(ci, cj)) { const t = 2 * (cj * N + ci); keep.push(t, t + 1); }
    const tris = new Int32Array(3 * keep.length);
    keep.forEach((t, k) => tris.set(mesh.tris.subarray(3 * t, 3 * t + 3), 3 * k));
    const sub = cgSpace({ nodes: mesh.nodes, tris }, 1);
    const a = assembleVolume(sub, g);
    return { A: csrExtract(a.builder.toCSR(), free, free), b: restrict(a.b) };
  };
  return { N, h: 1 / N, n: free.length, mesh, free, A, b, load, localMatrix };
}

/**
 * Scatter a free-vertex vector to all mesh vertices (boundary values 0).
 * @param {DDProblem} prob
 * @param {ArrayLike<number>} x length prob.n
 * @returns {Float64Array} length (N+1)²
 */
export function toVertices(prob, x) {
  const U = new Float64Array((prob.N + 1) ** 2);
  for (let k = 0; k < prob.n; k++) U[prob.free[k]] = x[k];
  return U;
}
