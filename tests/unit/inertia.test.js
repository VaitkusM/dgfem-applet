import { test, assert } from '../harness.js';
import { inertia } from '../../lib/core/la/inertia.js';
import { symEig } from '../../lib/core/la/dense.js';

test('inertia (LDLᵀ pivot signs) matches the eigenvalue signs (Sylvester)', () => {
  let seed = 3;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
  for (const n of [3, 8, 20]) for (let rep = 0; rep < 5; rep++) {
    const A = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { const v = rnd(); A[i * n + j] = v; A[j * n + i] = v; }
    const ev = symEig(A, n).values, I = inertia(A, n);
    const neg = ev.filter((x) => x < 0).length;
    assert(I.neg === neg && I.pos === n - neg && I.zero === 0, `n=${n}: ${I.neg} vs ${neg}`);
  }
});

test('inertia with zero pivots: 2×2 pivots instead of miscounting a zero eigenvalue', () => {
  // [0 1; 1 0]: eigenvalues ±1, but the first pivot is 0 (old code reported one ZERO eigenvalue)
  let I = inertia(Float64Array.of(0, 1, 1, 0), 2);
  assert(I.neg === 1 && I.pos === 1 && I.zero === 0, `2x2: ${I.neg}/${I.zero}/${I.pos}`);
  // second leading minor vanishes; eigenvalues −0.80, 0.55, 2.25
  I = inertia(Float64Array.of(1, 1, 0, 1, 1, 1, 0, 1, 0), 3);
  assert(I.neg === 1 && I.pos === 2 && I.zero === 0, `3x3: ${I.neg}/${I.zero}/${I.pos}`);
  // genuinely singular matrices: zero matrix, rank-1 matrix, and a saddle point with a redundant constraint
  I = inertia(new Float64Array(9), 3);
  assert(I.neg === 0 && I.pos === 0 && I.zero === 3, 'zero matrix');
  I = inertia(Float64Array.of(1, 2, 3, 2, 4, 6, 3, 6, 9), 3);
  assert(I.neg === 0 && I.pos === 1 && I.zero === 2, `rank 1: ${I.neg}/${I.zero}/${I.pos}`);
  I = inertia(Float64Array.of(2, 0, 1, 1, 0, 2, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0), 4);
  assert(I.neg === 1 && I.pos === 2 && I.zero === 1, `redundant constraint: ${I.neg}/${I.zero}/${I.pos}`);
});

test('inertia of random matrices with a zero diagonal block (saddle-point pattern) matches eigenvalues', () => {
  let seed = 17;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
  for (const [n, z] of [[4, 2], [9, 4], [16, 16], [21, 7]]) for (let rep = 0; rep < 10; rep++) {
    // leading z×z block is exactly zero ⇒ the unpivoted factorisation breaks down at the first step
    const A = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { const v = i < z && j < z ? 0 : rnd(); A[i * n + j] = v; A[j * n + i] = v; }
    const ev = symEig(A, n).values, I = inertia(A, n);
    const neg = ev.filter((x) => x < -1e-10).length, pos = ev.filter((x) => x > 1e-10).length;
    assert(I.neg === neg && I.pos === pos && I.zero === n - neg - pos, `n=${n} z=${z}: ${I.neg}/${I.zero}/${I.pos} vs ${neg}/${n - neg - pos}/${pos}`);
  }
});
