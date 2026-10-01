/**
 * @file Mesh connectivity ("topology") for triangle meshes: edges, neighbours,
 *       normals, boundary edges.
 *
 * Conventions (see AGENTS.md):
 *  - Local edge k of triangle t joins its local vertices k and (k+1) mod 3.
 *  - Global edge e stores its vertices with v0 < v1 ("canonical orientation").
 *  - edgeTris[2e] = the triangle on the "minus" side K⁻, edgeTris[2e+1] = the
 *    triangle on the "plus" side K⁺, or −1 on the boundary. The edge normal n_e
 *    points OUT of K⁻ (into K⁺).
 *  - The outward normal of a CCW triangle on the directed edge a→b is
 *    (b_y − a_y, −(b_x − a_x)) / |b − a|  (tangent rotated clockwise).
 */
import { EDGE_VERTS } from '../basis/simplex.js';

/**
 * @typedef {Object} Topology
 * @property {number} nEdge
 * @property {Int32Array} edges     2 vertex ids per edge (sorted)
 * @property {Int32Array} triEdges  3 global edge ids per triangle (local edge k)
 * @property {Int32Array} edgeTris  2 triangles per edge (K⁻, K⁺ or −1)
 * @property {Int8Array}  edgeLocal 2 local edge indices per edge (in K⁻, K⁺)
 * @property {Float64Array} normals 2 per edge: unit normal out of K⁻
 * @property {Float64Array} lengths edge lengths
 * @property {Uint8Array} isBoundary
 */

/**
 * Build edge-based connectivity of a triangle mesh.
 * @param {Float64Array} nodes
 * @param {Int32Array} tris
 * @returns {Topology}
 */
export function buildTopology(nodes, tris) {
  const nTri = tris.length / 3;
  const map = new Map();
  const edgesArr = [], edgeTrisArr = [], edgeLocalArr = [];
  const triEdges = new Int32Array(3 * nTri);
  for (let t = 0; t < nTri; t++)
    for (let k = 0; k < 3; k++) {
      const a = tris[3 * t + EDGE_VERTS[k][0]], b = tris[3 * t + EDGE_VERTS[k][1]];
      const lo = Math.min(a, b), hi = Math.max(a, b);
      const key = lo * 4294967296 + hi; // unique numeric key
      let e = map.get(key);
      if (e === undefined) {
        e = edgesArr.length / 2;
        map.set(key, e);
        edgesArr.push(lo, hi);
        edgeTrisArr.push(t, -1);
        edgeLocalArr.push(k, -1);
      } else {
        edgeTrisArr[2 * e + 1] = t;
        edgeLocalArr[2 * e + 1] = k;
      }
      triEdges[3 * t + k] = e;
    }
  const nEdge = edgesArr.length / 2;
  const edges = Int32Array.from(edgesArr), edgeTris = Int32Array.from(edgeTrisArr), edgeLocal = Int8Array.from(edgeLocalArr);
  const normals = new Float64Array(2 * nEdge), lengths = new Float64Array(nEdge), isBoundary = new Uint8Array(nEdge);
  for (let e = 0; e < nEdge; e++) {
    const t = edgeTris[2 * e], k = edgeLocal[2 * e];
    const a = tris[3 * t + EDGE_VERTS[k][0]], b = tris[3 * t + EDGE_VERTS[k][1]];
    const dx = nodes[2 * b] - nodes[2 * a], dy = nodes[2 * b + 1] - nodes[2 * a + 1];
    const L = Math.hypot(dx, dy);
    lengths[e] = L;
    normals[2 * e] = dy / L; normals[2 * e + 1] = -dx / L;
    isBoundary[e] = edgeTris[2 * e + 1] < 0 ? 1 : 0;
  }
  return { nEdge, edges, triEdges, edgeTris, edgeLocal, normals, lengths, isBoundary };
}

/**
 * Reference coordinates (x̂,ŷ) on local edge k of the reference triangle for
 * the edge parameter s ∈ [0,1] running from local vertex EDGE_VERTS[k][0] to
 * EDGE_VERTS[k][1].
 * @returns {[number, number]}
 */
export function refEdgePoint(k, s) {
  const V = [[0, 0], [1, 0], [0, 1]];
  const [a, b] = EDGE_VERTS[k];
  return [V[a][0] + s * (V[b][0] - V[a][0]), V[a][1] + s * (V[b][1] - V[a][1])];
}
