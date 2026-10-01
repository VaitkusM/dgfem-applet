import { test, assert, assertClose, assertRate } from '../harness.js';
import { cardinalBspline, cardinalBsplineDeriv, marsden, bsplineGrid } from '../../lib/core/basis/bspline.js';
import { circleLS, flowerLS } from '../../lib/core/mesh/levelset.js';
import {
  webSpace, solveWEB, errorsWEB, evalWEB, weightFunction, blendedExtension, lagrangeIndexWeight, cellBsplines,
  BS_INNER, BS_OUTER, CELL_OUT, diagScaled,
} from '../../lib/core/immersed/webspline.js';
import { POISSON_MMS } from '../../lib/core/verify/mms.js';
import { fitRate } from '../../lib/core/verify/rates.js';
import { csrToDense } from '../../lib/core/la/sparse.js';
import { symEig } from '../../lib/core/la/dense.js';

const S = POISSON_MMS.wave;
const FLOWER = { cx: 0.51, cy: 0.47, R: 0.3, a: 0.2, k: 5, rot: 0.3 };

test('cardinal B-splines: partition of unity, Marsden identity, derivative', () => {
  for (const n of [1, 2, 3]) {
    for (const t of [0.13, 0.5, 0.97, 2.4, 3.71]) {
      let s = 0;
      for (let k = -5; k <= 5; k++) s += cardinalBspline(n, t - k);
      assertClose(s, 1, 1e-14, 0, 'partition of unity');
      for (const tau of [-0.7, 0.3, 2.2]) {
        let m = 0;
        for (let k = -6; k <= 6; k++) m += marsden(n, k, tau) * cardinalBspline(n, t - k);
        assertClose(m, (t - tau) ** n, 1e-12, 1e-13, `Marsden n=${n}`);
      }
      const e = 1e-6, fd = (cardinalBspline(n, t + e) - cardinalBspline(n, t - e)) / (2 * e);
      assertClose(cardinalBsplineDeriv(n, t), fd, 1e-7);
    }
  }
});

test('extension coefficients e_ij = tensor Lagrange weights: Σ_i e_ij q(i) = q(j) for coordinate degree ≤ n', () => {
  for (const n of [1, 2, 3]) {
    const sp = webSpace(circleLS({ cx: 0.513, cy: 0.478, R: 0.33 }), { n, N: 16 });
    assert(sp.ext.size > 0, 'there are outer B-splines');
    for (const [j, E] of sp.ext) {
      const [j1, j2] = sp.grid.kOf(j);
      for (const [a, b] of [[0, 0], [n, 0], [0, n], [n, n], [1, n - 1]]) {
        let s = 0;
        for (let q = 0; q < E.ids.length; q++) { const [i1, i2] = sp.grid.kOf(E.ids[q]); s += E.e[q] * i1 ** a * i2 ** b; }
        assertClose(s, j1 ** a * j2 ** b, 1e-8, 1e-11, `n=${n}`);
        for (const id of E.ids) assert(sp.bsCls[id] === BS_INNER, 'array I(j) consists of inner indices');
      }
    }
    assertClose(lagrangeIndexWeight(n, 2, 3, 3), 1, 1e-15);
  }
});

test('extended B-splines reproduce polynomials of coordinate degree ≤ n on Ω (Marsden coefficients)', () => {
  for (const ls of [circleLS({ cx: 0.513, cy: 0.478, R: 0.33 }), flowerLS(FLOWER)]) for (const n of [1, 2, 3]) {
    const sp = webSpace(ls, { n, N: 16 }), { grid } = sp;
    const taus = [[0.3, -1.2], [2.5, 4.1]];
    for (const [t1, t2] of taus) for (const [d1, d2] of [[n, n], [n, 0], [1, 1]]) {
      // p(ξ,η) = (ξ − t1)^{d1} (η − t2)^{d2} in index coordinates. The degree-n B-spline coefficients of
      // (ξ − t)^d, d ≤ n, are its blossom at the knots k+1, …, k+n (Marsden / de Boor–Fix):
      //   c_k = (1 / C(n,d)) Σ_{S ⊂ {1..n}, |S| = d} Π_{μ∈S} (k + μ − t)   (d = n: Marsden's ψ_k(t)).
      const coef1D = (d, k, t) => {
      const idx = [];
        const rec = (start, left, prod) => {
          if (left === 0) { idx.push(prod); return; }
          for (let mu = start; mu <= n; mu++) rec(mu + 1, left - 1, prod * (k + mu - t));
        };
        rec(1, d, 1);
        return idx.reduce((s, v) => s + v, 0) / idx.length;
      };
      const coef = (k) => { const [k1, k2] = grid.kOf(k); return coef1D(d1, k1, t1) * coef1D(d2, k2, t2); };
      // value of Σ_{inner i} p_i (b_i + Σ_j e_ij b_j) = Σ_{relevant r} C_r b_r with C_r = p_r (inner) or Σ e p (outer)
      const C = new Map();
      for (const r of sp.relevant) {
        if (sp.bsCls[r] === BS_INNER) C.set(r, coef(r));
        else { const E = sp.ext.get(r); let s = 0; for (let q = 0; q < E.ids.length; q++) s += E.e[q] * coef(E.ids[q]); C.set(r, s); }
      }
      let maxErr = 0, nPts = 0;
      for (let c = 0; c < sp.N * sp.N; c++) {
        if (sp.cellCls[c] === CELL_OUT) continue;
        const l1 = c % sp.N, l2 = Math.floor(c / sp.N);
        for (const [s1, s2] of [[0.25, 0.5], [0.8, 0.1]]) {
          const x = grid.x0 + (l1 + s1) * grid.h, y = grid.y0 + (l2 + s2) * grid.h;
          if (ls.phi(x, y) >= 0) continue;
          const B = cellBsplines(sp, c, x, y);
          let v = 0;
          for (let a = 0; a < B.idx.length; a++) v += C.get(B.idx[a]) * B.v[a];
          const xi = (x - grid.x0) / grid.h, eta = (y - grid.y0) / grid.h;
          maxErr = Math.max(maxErr, Math.abs(v - (xi - t1) ** d1 * (eta - t2) ** d2) / (1 + Math.abs((xi - t1) ** d1 * (eta - t2) ** d2)));
          nPts++;
        }
      }
      assert(nPts > 50 && maxErr < 1e-9, `${ls.kind} n=${n} reproduction error ${maxErr}`);
    }
  }
});

test('WEB-splines satisfy u_h = g on Γ exactly (weight vanishes on Γ)', () => {
  const ls = flowerLS(FLOWER), W = weightFunction(ls);
  const sol = solveWEB(ls, { n: 2, N: 16, f: S.f, gt: blendedExtension(W, S.u, S.grad) });
  for (let i = 0; i < 40; i++) {
    const [x, y] = ls.point(2 * Math.PI * i / 40);
    assertClose(evalWEB(sol, x, y).u, S.u(x, y), 1e-12);
  }
});

test('WEB-spline Galerkin: L2 rate n+1, H1 rate n (circle, n = 1, 2, 3)', () => {
  const ls = circleLS({ cx: 0.513, cy: 0.478, R: 0.33 }), W = weightFunction(ls), gt = blendedExtension(W, S.u, S.grad);
  for (const n of [1, 2, 3]) {
    const hs = [], L2 = [], H1 = [];
    for (const N of [8, 16, 32]) {
      const e = errorsWEB(solveWEB(ls, { n, N, f: S.f, gt }), S.u, S.grad);
      hs.push(1 / N); L2.push(e.L2); H1.push(e.H1);
    }
    assertRate(fitRate(hs, L2, 2), n + 1, `n=${n} L2`, 0.2, 0.5);
    assertRate(fitRate(hs, H1, 2), n, `n=${n} H1`, 0.2, 0.5);
  }
});

test('WEB-spline Galerkin on the flower: L2 rate at least n+1 (pre-asymptotic, erratic), n = 1, 2', () => {
  const ls = flowerLS(FLOWER), W = weightFunction(ls), gt = blendedExtension(W, S.u, S.grad);
  for (const n of [1, 2]) {
    const hs = [], L2 = [];
    for (const N of [16, 32, 64]) { hs.push(1 / N); L2.push(errorsWEB(solveWEB(ls, { n, N, f: S.f, gt }), S.u, S.grad).L2); }
    const r = fitRate(hs, L2);
    assert(r > n + 1 - 0.2 && r < n + 2.5, `flower n=${n} L2 rate ${r}`);
  }
});

test('extension = stability: on a tiny cut, Jacobi-scaled κ·h² stays O(1) with extension, explodes without (n = 2, 3)', () => {
  const N = 16, h = 1 / N, ls = circleLS({ cx: 0.5, cy: 0.5, R: 5 * h + 1e-3 * h });
  const kap = (n, extend) => {
    const s = solveWEB(ls, { n, N, f: S.f, extend });
    const ev = symEig(csrToDense(diagScaled(s.A)), s.nDof).values;
    return ev[ev.length - 1] / ev[0] * h * h;
  };
  for (const n of [2, 3]) {
    const kw = kap(n, true), ko = kap(n, false);
    assert(kw < 1, `with extension n=${n}: ${kw}`);
    assert(ko > (n === 2 ? 10 : 1000) * kw, `without extension n=${n}: ${ko} vs ${kw}`);
  }
});
