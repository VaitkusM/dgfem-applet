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
