import { test, assert, assertClose } from '../harness.js';
import { triGrid, triArea, quadGrid } from '../../lib/core/mesh/structured.js';
import { buildTopology } from '../../lib/core/mesh/topology.js';

test('triangulated grids: positive areas summing to the box area, Euler formula', () => {
  for (const diag of ['right', 'left', 'alt']) {
    const m = triGrid(5, 4, [0, 2, 0, 1], { diag, jiggle: 0.25 });
    let A = 0;
    for (let t = 0; t < m.nTri; t++) { const a = triArea(m.nodes, m.tris, t); assert(a > 0, 'CW triangle'); A += a; }
    assertClose(A, 2, 1e-12);
    const T = buildTopology(m.nodes, m.tris);
    assert(m.nVert - T.nEdge + m.nTri === 1, 'Euler characteristic V-E+F = 1');
    const nb = T.isBoundary.reduce((a, b) => a + b, 0);
    assert(nb === 2 * (5 + 4), `boundary edges ${nb}`);
  }
});

test('edge normals point from K- to K+ and are outward on the boundary', () => {
  const m = triGrid(3, 3, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.2 });
  const T = buildTopology(m.nodes, m.tris);
  const centroid = (t) => {
    let x = 0, y = 0;
    for (let k = 0; k < 3; k++) { x += m.nodes[2 * m.tris[3 * t + k]] / 3; y += m.nodes[2 * m.tris[3 * t + k] + 1] / 3; }
    return [x, y];
  };
  for (let e = 0; e < T.nEdge; e++) {
    const [cx, cy] = centroid(T.edgeTris[2 * e]);
    const a = T.edges[2 * e];
    const mx = m.nodes[2 * a], my = m.nodes[2 * a + 1];
    // (edge point − centroid of K⁻) · n > 0
    assert((mx - cx) * T.normals[2 * e] + (my - cy) * T.normals[2 * e + 1] > 0, 'normal not outward of K-');
    assertClose(Math.hypot(T.normals[2 * e], T.normals[2 * e + 1]), 1, 1e-14);
  }
});

test('quad grid indexing', () => {
  const g = quadGrid(3, 2, [0, 3, 0, 2]);
  assert(g.nElem === 6 && g.nVert === 12);
  const e = 4; // i=1, j=1
  const v0 = g.quads[4 * e];
  assertClose(g.nodes[2 * v0], 1); assertClose(g.nodes[2 * v0 + 1], 1);
});
