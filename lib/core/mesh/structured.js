/**
 * @file Structured meshes: 1D intervals, Cartesian quad grids, triangulated grids.
 *
 * Mesh data layout (used everywhere in the project):
 *  - nodes: Float64Array [x0, y0, x1, y1, …]          (2 per vertex)
 *  - tris : Int32Array   [a0, b0, c0, a1, b1, c1, …]  (3 vertex ids per triangle,
 *           counter-clockwise, i.e. positive signed area — the same winding
 *           convention as front faces in OpenGL)
 *  - quads: Int32Array   4 ids per quad, CCW starting at the lower-left corner.
 */

/**
 * Uniform 1D mesh of [a,b] with N cells.
 * @returns {{N: number, a: number, b: number, h: number, x: Float64Array}} x = N+1 vertices
 */
export function mesh1D(N, a = 0, b = 1) {
  const x = new Float64Array(N + 1), h = (b - a) / N;
  for (let i = 0; i <= N; i++) x[i] = a + i * h;
  return { N, a, b, h, x };
}

/**
 * Cartesian grid of nx × ny quads on [x0,x1]×[y0,y1]. Cell (i,j) has index
 * e = j*nx + i (row-major, i along x), vertex (i,j) has index j*(nx+1)+i.
 */
export function quadGrid(nx, ny, box = [0, 1, 0, 1]) {
  const [x0, x1, y0, y1] = box;
  const hx = (x1 - x0) / nx, hy = (y1 - y0) / ny;
  const nodes = new Float64Array(2 * (nx + 1) * (ny + 1));
  for (let j = 0; j <= ny; j++)
    for (let i = 0; i <= nx; i++) {
      const v = j * (nx + 1) + i;
      nodes[2 * v] = x0 + i * hx; nodes[2 * v + 1] = y0 + j * hy;
    }
  const quads = new Int32Array(4 * nx * ny);
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const e = j * nx + i, v = j * (nx + 1) + i;
      quads.set([v, v + 1, v + nx + 2, v + nx + 1], 4 * e);
    }
  return { nx, ny, box, hx, hy, nodes, quads, nElem: nx * ny, nVert: (nx + 1) * (ny + 1) };
}

/**
 * Triangulated structured grid: each of the nx × ny squares is split into two
 * triangles along a diagonal.
 * @param {number} nx
 * @param {number} ny
 * @param {number[]} [box=[0,1,0,1]]
 * @param {{diag?: 'right'|'left'|'alt', jiggle?: number, seed?: number}} [opts]
 *   diag: 'right' = "/" diagonals, 'left' = "\", 'alt' = alternating (criss-cross pattern);
 *   jiggle: random interior vertex perturbation as a fraction of h (≤ 0.3 keeps triangles valid)
 * @returns {{nodes: Float64Array, tris: Int32Array, nVert: number, nTri: number, h: number, boundaryVert: Uint8Array}}
 */
export function triGrid(nx, ny, box = [0, 1, 0, 1], opts = {}) {
  const [x0, x1, y0, y1] = box;
  const hx = (x1 - x0) / nx, hy = (y1 - y0) / ny;
  const nVert = (nx + 1) * (ny + 1);
  const nodes = new Float64Array(2 * nVert);
  const boundaryVert = new Uint8Array(nVert);
  let seed = opts.seed ?? 7;
  const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
  const jig = opts.jiggle ?? 0;
  for (let j = 0; j <= ny; j++)
    for (let i = 0; i <= nx; i++) {
      const v = j * (nx + 1) + i;
      const onB = i === 0 || j === 0 || i === nx || j === ny;
      boundaryVert[v] = onB ? 1 : 0;
      const dx = onB ? 0 : jig * hx * rand(), dy = onB ? 0 : jig * hy * rand();
      nodes[2 * v] = x0 + i * hx + dx; nodes[2 * v + 1] = y0 + j * hy + dy;
    }
  const tris = new Int32Array(6 * nx * ny);
  const diag = opts.diag ?? 'right';
  let t = 0;
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const a = j * (nx + 1) + i, b = a + 1, c = a + nx + 2, d = a + nx + 1; // ll, lr, ur, ul
      const right = diag === 'right' || (diag === 'alt' && (i + j) % 2 === 0);
      if (right) { tris.set([a, b, c], t); tris.set([a, c, d], t + 3); }
      else { tris.set([a, b, d], t); tris.set([b, c, d], t + 3); }
      t += 6;
    }
  return { nodes, tris, nVert, nTri: 2 * nx * ny, h: Math.max(hx, hy), hx, hy, nx, ny, box, boundaryVert };
}

/** Signed area of triangle t (positive for CCW). */
export function triArea(nodes, tris, t) {
  const a = tris[3 * t], b = tris[3 * t + 1], c = tris[3 * t + 2];
  const ax = nodes[2 * a], ay = nodes[2 * a + 1];
  return 0.5 * ((nodes[2 * b] - ax) * (nodes[2 * c + 1] - ay) - (nodes[2 * c] - ax) * (nodes[2 * b + 1] - ay));
}

/** Vertex coordinates of triangle t as [[ax,ay],[bx,by],[cx,cy]]. */
export function triVerts(nodes, tris, t) {
  const r = [];
  for (let k = 0; k < 3; k++) { const v = tris[3 * t + k]; r.push([nodes[2 * v], nodes[2 * v + 1]]); }
  return r;
}

/**
 * Affine map data for triangle t: X = a + J x̂ with J = [b−a, c−a] (columns).
 * Returns J, det J and Jinv (row-major 2×2). Reference gradients transform
 * as ∇φ = J^{−T} ∇̂φ̂.
 */
export function triAffine(nodes, tris, t) {
  const [a, b, c] = triVerts(nodes, tris, t);
  const J = [b[0] - a[0], c[0] - a[0], b[1] - a[1], c[1] - a[1]];
  const det = J[0] * J[3] - J[1] * J[2];
  const Jinv = [J[3] / det, -J[1] / det, -J[2] / det, J[0] / det];
  return { a, b, c, J, det, Jinv };
}

/**
 * Physical gradient from reference gradient: ∇φ = J^{−T} ∇̂φ̂.
 * @param {number[]} Jinv row-major inverse Jacobian
 * @returns {[number, number]}
 */
export function refToPhysGrad(Jinv, gx, gy) {
  // J^{-T} = transpose of Jinv
  return [Jinv[0] * gx + Jinv[2] * gy, Jinv[1] * gx + Jinv[3] * gy];
}
