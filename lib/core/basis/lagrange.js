/**
 * @file Lagrange (nodal) polynomial bases in 1D.
 *
 * Given distinct nodes ξ_0..ξ_p, the Lagrange basis ℓ_j is the unique
 * polynomial of degree p with ℓ_j(ξ_i) = δ_ij. A polynomial is then stored by
 * its values at the nodes ("nodal" representation) — exactly like storing a
 * curve by control values that it interpolates.
 *
 * The differentiation matrix D has entries D_ij = ℓ_j'(ξ_i): multiplying the
 * nodal values of u by D gives the nodal values of u' (exact for degree ≤ p).
 */

/**
 * Barycentric weights  λ_j = 1 / Π_{m≠j} (ξ_j − ξ_m).
 * @param {ArrayLike<number>} nodes
 * @returns {Float64Array}
 */
export function baryWeights(nodes) {
  const n = nodes.length, w = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let p = 1;
    for (let m = 0; m < n; m++) if (m !== j) p *= nodes[j] - nodes[m];
    w[j] = 1 / p;
  }
  return w;
}

/**
 * Values of all Lagrange basis functions at x.
 * @param {ArrayLike<number>} nodes
 * @param {number} x
 * @param {Float64Array} [out]
 * @returns {Float64Array} out[j] = ℓ_j(x)
 */
export function lagrangeValues(nodes, x, out) {
  const n = nodes.length;
  out = out || new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let v = 1;
    for (let m = 0; m < n; m++) if (m !== j) v *= (x - nodes[m]) / (nodes[j] - nodes[m]);
    out[j] = v;
  }
  return out;
}

/**
 * Derivatives of all Lagrange basis functions at x (product rule).
 * @param {ArrayLike<number>} nodes
 * @param {number} x
 * @param {Float64Array} [out]
 * @returns {Float64Array} out[j] = ℓ_j'(x)
 */
export function lagrangeDerivs(nodes, x, out) {
  const n = nodes.length;
  out = out || new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let k = 0; k < n; k++) {
      if (k === j) continue;
      let p = 1 / (nodes[j] - nodes[k]);
      for (let m = 0; m < n; m++) {
        if (m === j || m === k) continue;
        p *= (x - nodes[m]) / (nodes[j] - nodes[m]);
      }
      s += p;
    }
    out[j] = s;
  }
  return out;
}

/**
 * Differentiation matrix D (row-major, n×n) with D[i*n+j] = ℓ_j'(ξ_i).
 * Uses the barycentric formula; diagonal by "negative sum trick" so that rows
 * sum to exactly zero (derivative of a constant is zero).
 * @param {ArrayLike<number>} nodes
 * @returns {Float64Array}
 */
export function diffMatrix(nodes) {
  const n = nodes.length, w = baryWeights(nodes), D = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const v = (w[j] / w[i]) / (nodes[i] - nodes[j]);
      D[i * n + j] = v;
      s += v;
    }
    D[i * n + i] = -s;
  }
  return D;
}

/**
 * Interpolation matrix from nodes to target points: I[q*n+j] = ℓ_j(x_q).
 * @param {ArrayLike<number>} nodes
 * @param {ArrayLike<number>} pts
 * @returns {Float64Array} (pts.length × nodes.length)
 */
export function interpMatrix(nodes, pts) {
  const n = nodes.length, m = pts.length, I = new Float64Array(m * n), tmp = new Float64Array(n);
  for (let q = 0; q < m; q++) {
    lagrangeValues(nodes, pts[q], tmp);
    I.set(tmp, q * n);
  }
  return I;
}

/**
 * Equispaced nodes on [-1,1] (used only for plotting/illustration —
 * high-degree interpolation on equispaced nodes is ill-conditioned (Runge)).
 * @param {number} p degree
 */
export function equispaced(p) {
  const x = new Float64Array(p + 1);
  if (p === 0) { x[0] = 0; return x; }
  for (let i = 0; i <= p; i++) x[i] = -1 + 2 * i / p;
  return x;
}
