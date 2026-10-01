/**
 * @file Size of the global linear system (unknowns and stored non-zeros) of
 *       several discretisations of the Poisson problem with polynomial degree p
 *       on the same triangle mesh — the "price list" of CG vs DG vs HDG.
 *
 * Counted is the system that is actually solved:
 *  - CG  (continuous P_p, strong Dirichlet BCs): only the FREE (non-boundary) DOFs.
 *        DOFs: 1 per vertex, p−1 per edge, (p−1)(p−2)/2 per triangle interior.
 *  - DG  (SIPG, broken P_p, Nitsche BCs): all (p+1)(p+2)/2 coefficients of every triangle.
 *        A triangle couples to itself and to its (up to 3) face neighbours.
 *  - HDG (degree p, after static condensation): p+1 trace coefficients per INTERIOR
 *        edge. A trace couples to the traces of the (up to 5) edges of its two triangles.
 *  - hybridized RT0 is HDG-like with one multiplier per interior edge (p = 0).
 * The non-zero pattern is "structural": every pair of DOFs that share an element
 * (CG, HDG) or an element/face (DG) counts, whether or not the entry happens to be 0.
 */
import { SparseBuilder, nnz } from '../la/sparse.js';
import { buildTopology } from '../mesh/topology.js';
import { dimP } from '../basis/simplex.js';

/**
 * Pattern size from per-element DOF lists (entries < 0 = eliminated DOF).
 * @param {number} n number of unknowns
 * @param {Int32Array[]} elemDofs list of DOF ids per element
 * @returns {number} structural nnz
 */
function patternNnz(n, elemDofs) {
  let cap = 0; for (const d of elemDofs) cap += d.length * d.length;
  const B = new SparseBuilder(n, n, Math.max(16, cap));
  for (const d of elemDofs) for (const i of d) if (i >= 0) for (const j of d) if (j >= 0) B.add(i, j, 1);
  return nnz(B.toCSR());
}

/**
 * System sizes of CG, DG (SIPG) and HDG of degree p on a triangle mesh.
 * @param {{nodes: Float64Array, tris: Int32Array}} mesh
 * @param {number} p ≥ 1
 * @returns {{cg: {n: number, nnz: number}, dg: {n: number, nnz: number}, hdg: {n: number, nnz: number}, nTri: number, nIntEdge: number}}
 */
export function systemSizes(mesh, p) {
  const topo = buildTopology(mesh.nodes, mesh.tris), nT = mesh.tris.length / 3, nV = mesh.nodes.length / 2;
  // ---- CG: number free DOFs (vertices, edge nodes, interior nodes) ----------
  const onB = new Uint8Array(nV);
  for (let e = 0; e < topo.nEdge; e++) if (topo.isBoundary[e]) { onB[topo.edges[2 * e]] = 1; onB[topo.edges[2 * e + 1]] = 1; }
  let id = 0;
  const vId = new Int32Array(nV), eId = new Int32Array(topo.nEdge);
  for (let v = 0; v < nV; v++) vId[v] = onB[v] ? -1 : (id++);
  for (let e = 0; e < topo.nEdge; e++) { eId[e] = topo.isBoundary[e] ? -1 : id; if (!topo.isBoundary[e]) id += p - 1; }
  const nInt = ((p - 1) * (p - 2)) / 2, tId = id;
  id += nT * nInt;
  const nCG = id;
  const cgElems = [];
  for (let t = 0; t < nT; t++) {
    const d = [];
    for (let k = 0; k < 3; k++) d.push(vId[mesh.tris[3 * t + k]]);
    for (let k = 0; k < 3; k++) { const e = topo.triEdges[3 * t + k]; for (let j = 0; j < p - 1; j++) d.push(eId[e] < 0 ? -1 : eId[e] + j); }
    for (let j = 0; j < nInt; j++) d.push(tId + t * nInt + j);
    cgElems.push(Int32Array.from(d));
  }
  // ---- DG: closed form ------------------------------------------------------
  const n = dimP(p);
  let nIntEdge = 0; for (let e = 0; e < topo.nEdge; e++) if (!topo.isBoundary[e]) nIntEdge++;
  // ---- HDG: traces on interior edges -------------------------------------
  const lam = new Int32Array(topo.nEdge).fill(-1);
  let L = 0; for (let e = 0; e < topo.nEdge; e++) if (!topo.isBoundary[e]) lam[e] = L++;
  const hdgElems = [];
  for (let t = 0; t < nT; t++) {
    const d = [];
    for (let k = 0; k < 3; k++) { const I = lam[topo.triEdges[3 * t + k]]; for (let a = 0; a <= p; a++) d.push(I < 0 ? -1 : I * (p + 1) + a); }
    hdgElems.push(Int32Array.from(d));
  }
  return {
    cg: { n: nCG, nnz: patternNnz(nCG, cgElems) },
    dg: { n: nT * n, nnz: (nT + 2 * nIntEdge) * n * n },
    hdg: { n: L * (p + 1), nnz: patternNnz(L * (p + 1), hdgElems) },
    nTri: nT, nIntEdge,
  };
}
