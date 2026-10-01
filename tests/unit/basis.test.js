import { test, assert, assertClose } from '../harness.js';
import { legendreP, legendreAll, radauRight, jacobiP, jacobiDP } from '../../lib/core/basis/legendre.js';
import { lagrangeValues, lagrangeDerivs, diffMatrix } from '../../lib/core/basis/lagrange.js';
import { p1Basis, p2Basis, P2_NODES, dubinerBasis, dimP } from '../../lib/core/basis/simplex.js';
import { gaussLegendre, gaussLobatto } from '../../lib/core/quad/gauss1d.js';
import { triangleRule } from '../../lib/core/quad/simplex.js';

test('Legendre orthogonality ∫P_m P_n = 2/(2n+1) δ_mn', () => {
  const { x, w } = gaussLegendre(12);
  for (let m = 0; m <= 8; m++) for (let n = 0; n <= 8; n++) {
    let s = 0;
    for (let q = 0; q < 12; q++) s += w[q] * legendreP(m, x[q]) * legendreP(n, x[q]);
    assertClose(s, m === n ? 2 / (2 * n + 1) : 0, 1e-14);
  }
});

test('Legendre derivative matches finite differences, P_n(1)=1', () => {
  for (let n = 0; n <= 8; n++) {
    assertClose(legendreP(n, 1), 1, 1e-14);
    for (const x of [-0.9, -0.3, 0.2, 0.77]) {
      const h = 1e-6, fd = (legendreP(n, x + h) - legendreP(n, x - h)) / (2 * h);
      assertClose(legendreAll(n, x).dP[n], fd, 1e-6);
    }
  }
});

test('Right Radau polynomial: R(-1)=1, R(1)=0, derivative OK', () => {
  for (let k = 1; k <= 7; k++) {
    assertClose(radauRight(k, -1).v, 1, 1e-14);
    assertClose(radauRight(k, 1).v, 0, 1e-14);
    const x = 0.31, h = 1e-6;
    assertClose(radauRight(k, x).d, (radauRight(k, x + h).v - radauRight(k, x - h).v) / (2 * h), 1e-6);
  }
});

test('normalised Jacobi orthonormality and derivative', () => {
  const { x, w } = gaussLegendre(20);
  for (const [a, b] of [[0, 0], [1, 0], [3, 0], [5, 0], [2, 2]]) {
    for (let m = 0; m <= 5; m++) for (let n = 0; n <= 5; n++) {
      let s = 0;
      for (let q = 0; q < 20; q++) s += w[q] * (1 - x[q]) ** a * (1 + x[q]) ** b * jacobiP(x[q], a, b, m) * jacobiP(x[q], a, b, n);
      assertClose(s, m === n ? 1 : 0, 1e-12, 0, `a=${a} b=${b} m=${m} n=${n}`);
    }
    for (let n = 0; n <= 5; n++) {
      const t = 0.3, h = 1e-6;
      assertClose(jacobiDP(t, a, b, n), (jacobiP(t + h, a, b, n) - jacobiP(t - h, a, b, n)) / (2 * h), 1e-6);
    }
  }
});

test('Lagrange basis: cardinal property, partition of unity, derivative', () => {
  for (let p = 1; p <= 6; p++) {
    const nodes = gaussLobatto(p + 1).x;
    for (let i = 0; i <= p; i++) {
      const v = lagrangeValues(nodes, nodes[i]);
      for (let j = 0; j <= p; j++) assertClose(v[j], i === j ? 1 : 0, 1e-13);
    }
    const x = 0.123, v = lagrangeValues(nodes, x), d = lagrangeDerivs(nodes, x);
    assertClose(v.reduce((a, b) => a + b, 0), 1, 1e-13);
    assertClose(d.reduce((a, b) => a + b, 0), 0, 1e-11);
  }
});

test('differentiation matrix is exact for degree ≤ p, rows sum to 0', () => {
  for (let p = 1; p <= 8; p++) {
    const nodes = gaussLobatto(p + 1).x, n = p + 1, D = diffMatrix(nodes);
    for (let k = 0; k <= p; k++) {
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += D[i * n + j] * nodes[j] ** k;
        assertClose(s, k === 0 ? 0 : k * nodes[i] ** (k - 1), 1e-11, 0, `p=${p} k=${k}`);
      }
    }
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += D[i * n + j];
      assertClose(s, 0, 1e-13);
    }
    // consistency with lagrangeDerivs
    const ld = lagrangeDerivs(nodes, nodes[1]);
    for (let j = 0; j < n; j++) assertClose(D[n + j], ld[j], 1e-10);
  }
});

test('P1 and P2 Lagrange: nodal property and gradient by finite differences', () => {
  const V = [[0, 0], [1, 0], [0, 1]];
  for (let i = 0; i < 3; i++) {
    const b = p1Basis(V[i][0], V[i][1]);
    for (let j = 0; j < 3; j++) assertClose(b.v[j], i === j ? 1 : 0);
  }
  for (let i = 0; i < 6; i++) {
    const b = p2Basis(P2_NODES[i][0], P2_NODES[i][1]);
    for (let j = 0; j < 6; j++) assertClose(b.v[j], i === j ? 1 : 0, 1e-14);
  }
  const x = 0.21, y = 0.33, h = 1e-6, b = p2Basis(x, y);
  for (let j = 0; j < 6; j++) {
    assertClose(b.dx[j], (p2Basis(x + h, y).v[j] - p2Basis(x - h, y).v[j]) / (2 * h), 1e-7);
    assertClose(b.dy[j], (p2Basis(x, y + h).v[j] - p2Basis(x, y - h).v[j]) / (2 * h), 1e-7);
  }
});

test('Dubiner basis is L2-orthonormal on the reference triangle; gradients OK', () => {
  for (let p = 0; p <= 5; p++) {
    const N = dimP(p), R = triangleRule(2 * p + 2);
    const G = new Float64Array(N * N);
    for (let q = 0; q < R.n; q++) {
      const b = dubinerBasis(p, R.x[q], R.y[q]);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) G[i * N + j] += R.w[q] * b.v[i] * b.v[j];
    }
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) assertClose(G[i * N + j], i === j ? 1 : 0, 1e-12, 0, `p=${p} (${i},${j})`);
    const x = 0.27, y = 0.41, h = 1e-6, b = dubinerBasis(p, x, y);
    for (let j = 0; j < N; j++) {
      assertClose(b.dx[j], (dubinerBasis(p, x + h, y).v[j] - dubinerBasis(p, x - h, y).v[j]) / (2 * h), 1e-6, 1e-7, `dx p=${p} j=${j}`);
      assertClose(b.dy[j], (dubinerBasis(p, x, y + h).v[j] - dubinerBasis(p, x, y - h).v[j]) / (2 * h), 1e-6, 1e-7, `dy p=${p} j=${j}`);
    }
  }
  // the first mode is the constant 1/sqrt(|T̂|) = sqrt(2)
  assertClose(dubinerBasis(3, 0.2, 0.2).v[0], Math.SQRT2, 1e-14);
  // nested: first dimP(q) modes have degree ≤ q -> check mode count by degree is fine
  assert(dimP(3) === 10);
});
