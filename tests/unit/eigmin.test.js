/**
 * Certified smallest eigenvalue (lib/core/la/eigmin.js): eigenvalue counts by
 * banded LDLᵀ (Sylvester) and shift-invert Lanczos, against the dense eigen-solver.
 */
import { test, assert, assertClose } from '../harness.js';
import { bandSym, countBelow, smallestEigenvalueSym } from '../../lib/core/la/eigmin.js';
import { SparseBuilder, csrToDense, denseToCsr } from '../../lib/core/la/sparse.js';
import { symEig } from '../../lib/core/la/dense.js';
import { triGrid } from '../../lib/core/mesh/structured.js';
import { solveMixedRT0 } from '../../lib/core/elliptic/mixedrt0.js';

/** deterministic pseudo-random numbers in (−0.5, 0.5) */
const rng = (seed) => () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };

/** random symmetric band matrix with half-bandwidth b, diagonal shifted by `shift` */
function randomBand(n, b, shift, seed) {
  const r = rng(seed), B = new SparseBuilder(n, n, n * (2 * b + 1));
  for (let i = 0; i < n; i++) {
    B.add(i, i, 2 * r() + shift);
    for (let j = Math.max(0, i - b); j < i; j++) { const v = r(); B.add(i, j, v); B.add(j, i, v); }
  }
  return B.toCSR();
}

test('countBelow: number of eigenvalues below a shift = dense eigenvalue count (random band matrices, saddle point)', () => {
  const mats = [randomBand(30, 3, 0, 5), randomBand(60, 7, 0.5, 11), randomBand(45, 1, -0.3, 3)];
  mats.push(solveMixedRT0(triGrid(3, 3, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.1 }), { f: () => 1 }).A); // indefinite saddle matrix
  for (const A of mats) {
    const ev = symEig(csrToDense(A), A.n).values, S = bandSym(A);
    for (const s of [-3, -0.7, -0.1, 0, 0.123, 0.9, 2.5]) {
      // a numerically zero pivot (ok = false; happens at s = 0 for the saddle matrix with its zero block) is
      // resolved by moving the shift by a hair, exactly as smallestEigenvalueSym does
      let c = countBelow(S, s);
      if (!c.ok) c = countBelow(S, s - 1e-9);
      assert(c.ok && c.neg === ev.filter((v) => v < s).length, `n=${A.n} shift ${s}: ${c.neg} vs ${ev.filter((v) => v < s).length}`);
    }
    assert(Math.max(Math.abs(ev[0]), Math.abs(ev[A.n - 1])) <= S.norm * (1 + 1e-12), 'norm bounds the spectrum');
  }
});

test('smallestEigenvalueSym: certified λ_min for SPD, indefinite and nearly singular matrices', () => {
  const cases = [randomBand(80, 4, 3, 7), randomBand(80, 4, 0, 7), randomBand(120, 9, 1.1, 21), randomBand(50, 2, -4, 2)];
  cases.push(solveMixedRT0(triGrid(4, 4, [0, 1, 0, 1], { diag: 'alt' }), { f: () => 1 }).A);
  // widely spread spectrum with one tiny negative eigenvalue: diag(−1e−3, 1, 2, …, 1e4)
  const n = 200, d = new Float64Array(n * n);
  for (let i = 0; i < n; i++) d[i * n + i] = i === 137 ? -1e-3 : 1 + (1e4 - 1) * i / (n - 1);
  cases.push(denseToCsr(d, n));
  for (const A of cases) {
    const ex = symEig(csrToDense(A), A.n).values[0], r = smallestEigenvalueSym(A);
    assert(r.certified, `certified (n=${A.n}, λ_min=${ex})`);
    assertClose(r.value, ex, 1e-12, 2e-6, `n=${A.n}`);
    assert(r.lower <= ex + 1e-12 && ex <= r.upper + 1e-12, `bounds [${r.lower}, ${r.upper}] contain ${ex}`);
  }
});

test('smallestEigenvalueSym: zero leading pivot ([[0,1],[1,0]]) and 1×1 matrices', () => {
  const r = smallestEigenvalueSym(denseToCsr(Float64Array.of(0, 1, 1, 0), 2));
  assertClose(r.value, -1, 1e-5); assert(r.certified, 'certified');
  assertClose(smallestEigenvalueSym(denseToCsr(Float64Array.of(-2.5), 1)).value, -2.5, 1e-5);
  assertClose(smallestEigenvalueSym(denseToCsr(Float64Array.of(4), 1)).value, 4, 1e-5);
});
