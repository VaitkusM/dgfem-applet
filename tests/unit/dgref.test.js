/**
 * Reference-element matrices of DG (lib/core/dg/dg1d.js): the formulas quoted in chapter 6.
 */
import { test, assert, assertClose, assertArrayClose } from '../harness.js';
import { refMatrices, refNodes, vandermonde } from '../../lib/core/dg/dg1d.js';
import { gaussLegendre } from '../../lib/core/quad/gauss1d.js';
import { legendreAll } from '../../lib/core/basis/legendre.js';

test('p = 1 on nodes ±1: M = [[2/3,1/3],[1/3,2/3]], S = [[−1/2,1/2],[−1/2,1/2]], D = S, lumped M = I', () => {
  const r = refNodes(1, 'gll').x;
  const ex = refMatrices(r, gaussLegendre(3));
  assertArrayClose(ex.M, [2 / 3, 1 / 3, 1 / 3, 2 / 3], 1e-15);
  assertArrayClose(ex.S, [-0.5, 0.5, -0.5, 0.5], 1e-15);
  assertArrayClose(ex.D, [-0.5, 0.5, -0.5, 0.5], 1e-15);
  assertArrayClose(refMatrices(r, refNodes(1, 'gll')).M, [1, 0, 0, 1], 1e-15);
});

test('Summation by parts: S + Sᵀ = diag(−1, 0, …, 0, 1) for GLL nodes (exact and collocated quadrature)', () => {
  for (let p = 1; p <= 6; p++) {
    const r = refNodes(p, 'gll').x, np = p + 1;
    for (const Q of [gaussLegendre(p + 2), refNodes(p, 'gll')]) {
      const { S } = refMatrices(r, Q);
      for (let i = 0; i < np; i++) for (let j = 0; j < np; j++) {
        const B = i === j ? (i === 0 ? -1 : i === p ? 1 : 0) : 0;
        assertClose(S[i * np + j] + S[j * np + i], B, 1e-12, 0, `p=${p} (${i},${j})`);
      }
    }
  }
});

test('GLL lumped mass matrix: diagonal, same row sums as the exact one, but not equal to it', () => {
  for (let p = 1; p <= 6; p++) {
    const r = refNodes(p, 'gll').x, np = p + 1;
    const ex = refMatrices(r, gaussLegendre(p + 2)).M, lu = refMatrices(r, refNodes(p, 'gll')).M;
    let diff = 0;
    for (let i = 0; i < np; i++) {
      let s1 = 0, s2 = 0;
      for (let j = 0; j < np; j++) { s1 += ex[i * np + j]; s2 += lu[i * np + j]; if (i !== j) assertClose(lu[i * np + j], 0, 1e-14); diff = Math.max(diff, Math.abs(ex[i * np + j] - lu[i * np + j])); }
      assertClose(s1, s2, 1e-13, 0, 'row sums = ∫ ℓ_i');
    }
    assert(diff > 1e-3, `p=${p}: lumped equals exact?`);
  }
});

test('Modal Legendre mass matrix is diagonal with entries 2/(2j+1)', () => {
  for (let p = 0; p <= 6; p++) {
    const G = gaussLegendre(p + 2);
    for (let i = 0; i <= p; i++) for (let j = 0; j <= p; j++) {
      let s = 0;
      for (let q = 0; q < G.x.length; q++) { const { P } = legendreAll(p, G.x[q]); s += G.w[q] * P[i] * P[j]; }
      assertClose(s, i === j ? 2 / (2 * j + 1) : 0, 1e-14);
    }
  }
});

test('Vandermonde maps modal to nodal values', () => {
  const r = refNodes(3, 'gll').x, V = vandermonde(r);
  // u = P_2 ⇒ nodal values = column 2
  for (let i = 0; i < 4; i++) assertClose(V[i * 4 + 2], 1.5 * r[i] * r[i] - 0.5, 1e-15);
});
