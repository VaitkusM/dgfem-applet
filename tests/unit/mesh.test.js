import { test, assert, assertClose } from '../harness.js';
import { triGrid, triArea, quadGrid, mesh1D, triAffine, refToPhysGrad } from '../../lib/core/mesh/structured.js';
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

test('topology arrays are mutually consistent; K+ lies on the +n side; shared edges have opposite local direction', () => {
  for (const diag of ['right', 'left', 'alt']) {
    const m = triGrid(4, 3, [0, 2, -1, 1], { diag, jiggle: 0.25 });
    const T = buildTopology(m.nodes, m.tris), EV = [[0, 1], [1, 2], [2, 0]];
    const centroid = (t) => { let x = 0, y = 0; for (let k = 0; k < 3; k++) { x += m.nodes[2 * m.tris[3 * t + k]] / 3; y += m.nodes[2 * m.tris[3 * t + k] + 1] / 3; } return [x, y]; };
    let nInterior = 0;
    for (let e = 0; e < T.nEdge; e++) {
      const v0 = T.edges[2 * e], v1 = T.edges[2 * e + 1];
      assert(v0 < v1, 'edge vertices not sorted');
      const mx = (m.nodes[2 * v0] + m.nodes[2 * v1]) / 2, my = (m.nodes[2 * v0 + 1] + m.nodes[2 * v1 + 1]) / 2;
      assertClose(T.lengths[e], Math.hypot(m.nodes[2 * v1] - m.nodes[2 * v0], m.nodes[2 * v1 + 1] - m.nodes[2 * v0 + 1]), 1e-14);
      // normal ⟂ edge
      assertClose((m.nodes[2 * v1] - m.nodes[2 * v0]) * T.normals[2 * e] + (m.nodes[2 * v1 + 1] - m.nodes[2 * v0 + 1]) * T.normals[2 * e + 1], 0, 1e-14);
      for (let s = 0; s < 2; s++) {
        const t = T.edgeTris[2 * e + s], k = T.edgeLocal[2 * e + s];
        if (t < 0) { assert(s === 1 && k === -1 && T.isBoundary[e] === 1, 'boundary bookkeeping'); continue; }
        // triEdges ↔ edgeTris/edgeLocal round trip, and local edge k really joins local vertices k, k+1
        assert(T.triEdges[3 * t + k] === e, 'triEdges/edgeLocal mismatch');
        const a = m.tris[3 * t + EV[k][0]], b = m.tris[3 * t + EV[k][1]];
        assert(Math.min(a, b) === v0 && Math.max(a, b) === v1, 'local edge vertices');
        // K− (s = 0) is behind the normal, K+ (s = 1) in front of it
        const [cx, cy] = centroid(t), side = (cx - mx) * T.normals[2 * e] + (cy - my) * T.normals[2 * e + 1];
        assert(s === 0 ? side < 0 : side > 0, `triangle ${t} on the wrong side of edge ${e}`);
      }
      if (!T.isBoundary[e]) {
        nInterior++;
        // CCW neighbours traverse a shared edge in opposite directions
        const tm = T.edgeTris[2 * e], km = T.edgeLocal[2 * e], tp = T.edgeTris[2 * e + 1], kp = T.edgeLocal[2 * e + 1];
        assert(m.tris[3 * tm + EV[km][0]] === m.tris[3 * tp + EV[kp][1]] && m.tris[3 * tm + EV[km][1]] === m.tris[3 * tp + EV[kp][0]], 'edge direction');
      } else {
        // boundary normals point out of the box
        const out = Math.abs(mx - 0) < 1e-12 ? [-1, 0] : Math.abs(mx - 2) < 1e-12 ? [1, 0] : Math.abs(my + 1) < 1e-12 ? [0, -1] : [0, 1];
        assertClose(T.normals[2 * e], out[0], 1e-13); assertClose(T.normals[2 * e + 1], out[1], 1e-13);
      }
    }
    assert(3 * m.nTri === 2 * nInterior + (T.nEdge - nInterior), 'each interior edge twice, each boundary edge once');
  }
});

test('mesh1D, triAffine and refToPhysGrad: affine map and gradient transformation', () => {
  const m1 = mesh1D(4, -1, 1);
  assert(m1.x.length === 5 && m1.N === 4); assertClose(m1.h, 0.5); assertClose(m1.x[0], -1); assertClose(m1.x[3], 0.5); assertClose(m1.x[4], 1);
  const nodes = Float64Array.of(1, 1, 4, 2, 2, 5), tris = Int32Array.of(0, 1, 2);
  const A = triAffine(nodes, tris, 0);
  assertClose(A.det, 2 * triArea(nodes, tris, 0), 1e-14); assertClose(A.det, 11, 1e-14);
  // J · Jinv = I
  assertClose(A.J[0] * A.Jinv[0] + A.J[1] * A.Jinv[2], 1, 1e-14); assertClose(A.J[0] * A.Jinv[1] + A.J[1] * A.Jinv[3], 0, 1e-14);
  assertClose(A.J[2] * A.Jinv[0] + A.J[3] * A.Jinv[2], 0, 1e-14); assertClose(A.J[2] * A.Jinv[1] + A.J[3] * A.Jinv[3], 1, 1e-14);
  // the linear function u = 3x − 2y has reference gradient (u(b) − u(a), u(c) − u(a)); mapped back it must be (3, −2)
  const u = (p) => 3 * p[0] - 2 * p[1];
  const g = refToPhysGrad(A.Jinv, u(A.b) - u(A.a), u(A.c) - u(A.a));
  assertClose(g[0], 3, 1e-13); assertClose(g[1], -2, 1e-13);
});
