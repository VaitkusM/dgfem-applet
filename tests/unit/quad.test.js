import { test, assertClose } from '../harness.js';
import { gaussLegendre, gaussLobatto, mapRule } from '../../lib/core/quad/gauss1d.js';
import { triangleRule, mapTriangleRule } from '../../lib/core/quad/simplex.js';

// ∫_{-1}^{1} x^k dx = 2/(k+1) for even k, 0 for odd k
const monoInt = (k) => (k % 2 === 0 ? 2 / (k + 1) : 0);

test('Gauss–Legendre n points is exact for degree 2n-1 and not 2n', () => {
  for (let n = 1; n <= 12; n++) {
    const { x, w } = gaussLegendre(n);
    for (let k = 0; k <= 2 * n - 1; k++) {
      let s = 0;
      for (let q = 0; q < n; q++) s += w[q] * x[q] ** k;
      assertClose(s, monoInt(k), 1e-13, 0, `n=${n} k=${k}`);
    }
    let s = 0;
    for (let q = 0; q < n; q++) s += w[q] * x[q] ** (2 * n);
    if (Math.abs(s - monoInt(2 * n)) < 1e-10) throw new Error(`n=${n} unexpectedly exact at degree 2n`);
    for (let q = 1; q < n; q++) if (!(x[q] > x[q - 1])) throw new Error('nodes not ascending');
  }
});

test('Gauss–Lobatto n points is exact for degree 2n-3, includes ±1', () => {
  for (let n = 2; n <= 12; n++) {
    const { x, w } = gaussLobatto(n);
    assertClose(x[0], -1); assertClose(x[n - 1], 1);
    for (let k = 0; k <= 2 * n - 3; k++) {
      let s = 0;
      for (let q = 0; q < n; q++) s += w[q] * x[q] ** k;
      assertClose(s, monoInt(k), 1e-13, 0, `n=${n} k=${k}`);
    }
    for (let q = 1; q < n; q++) if (!(x[q] > x[q - 1])) throw new Error('nodes not ascending');
  }
});

test('mapped rule integrates on [a,b]', () => {
  const r = mapRule(gaussLegendre(4), 1, 3);
  let s = 0;
  for (let q = 0; q < 4; q++) s += r.w[q] * r.x[q] ** 3;
  assertClose(s, (81 - 1) / 4, 1e-12);
});

// ∫_T̂ x^a y^b = a! b! / (a+b+2)!
const fact = (n) => (n <= 1 ? 1 : n * fact(n - 1));
test('triangle rule exact for total degree d', () => {
  for (let d = 0; d <= 14; d++) {
    const R = triangleRule(d);
    for (let a = 0; a <= d; a++) for (let b = 0; a + b <= d; b++) {
      let s = 0;
      for (let q = 0; q < R.n; q++) s += R.w[q] * R.x[q] ** a * R.y[q] ** b;
      assertClose(s, fact(a) * fact(b) / fact(a + b + 2), 1e-14, 1e-12, `d=${d} a=${a} b=${b}`);
    }
  }
});

test('mapped triangle rule gives area and centroid', () => {
  const R = mapTriangleRule(triangleRule(2), [1, 1], [4, 2], [2, 5]);
  let A = 0, cx = 0, cy = 0;
  for (let q = 0; q < R.n; q++) { A += R.w[q]; cx += R.w[q] * R.x[q]; cy += R.w[q] * R.y[q]; }
  assertClose(A, 0.5 * Math.abs(3 * 4 - 1 * 1), 1e-12);
  assertClose(cx / A, 7 / 3, 1e-12); assertClose(cy / A, 8 / 3, 1e-12);
});
