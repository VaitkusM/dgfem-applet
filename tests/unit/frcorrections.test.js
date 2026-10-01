/**
 * Tests for lib/core/fr/corrections.js: defining properties of the FR correction
 * functions (Huynh 2007; Vincent, Castonguay & Jameson 2011).
 */
import { test, assert, assertClose } from '../harness.js';
import { correctionCoeffs, evalCorrection, cSD, cHU, vcjhEta, legendreLeading } from '../../lib/core/fr/corrections.js';
import { radauRight, legendreP } from '../../lib/core/basis/legendre.js';
import { gaussLegendre } from '../../lib/core/quad/gauss1d.js';

const TYPES = [['dg', 0], ['g2', 0], ['ga', 0], ['vcjh', 0.37], ['vcjh', 1e-3], ['vcjh', 25]];

test('Legendre leading coefficients a_p = (2p)!/(2^p p!²)', () => {
  for (const [p, a] of [[0, 1], [1, 1], [2, 1.5], [3, 2.5], [4, 35 / 8]]) assertClose(legendreLeading(p), a, 1e-14);
});

test('g_L(−1) = 1, g_L(1) = 0, g_R(ξ) = g_L(−ξ) for every family and p = 0..7', () => {
  for (let p = 0; p <= 7; p++) for (const [t, c] of TYPES) {
    const co = correctionCoeffs(t, p, c);
    assertClose(evalCorrection(co, -1).gL, 1, 1e-12, 0, `${t} p=${p} gL(-1)`);
    assertClose(evalCorrection(co, 1).gL, 0, 1e-12, 0, `${t} p=${p} gL(1)`);
    assertClose(evalCorrection(co, -1).gR, 0, 1e-12, 0, `${t} p=${p} gR(-1)`);
    assertClose(evalCorrection(co, 1).gR, 1, 1e-12, 0, `${t} p=${p} gR(1)`);
    for (const x of [-0.7, -0.1, 0.33, 0.9]) {
      const A = evalCorrection(co, x), B = evalCorrection(co, -x);
      assertClose(A.gR, B.gL, 1e-13, 0, 'mirror value');
      assertClose(A.dgR, -B.dgL, 1e-12, 0, 'mirror derivative');
    }
  }
});

test('g_DG is the right Radau polynomial R_{R,p+1}', () => {
  for (let p = 1; p <= 6; p++) {
    const co = correctionCoeffs('dg', p);
    for (const x of [-0.9, -0.2, 0.4, 0.8]) assertClose(evalCorrection(co, x).gL, radauRight(p + 1, x).v, 1e-13);
  }
});

test('g_DG lifts the left face: ∫ g_DG\' v dξ = −v(−1) for all v ∈ P_p (so FR(g_DG) = DG)', () => {
  for (let p = 1; p <= 6; p++) {
    const co = correctionCoeffs('dg', p), G = gaussLegendre(p + 2);
    for (let j = 0; j <= p; j++) {
      let s = 0;
      for (let q = 0; q < G.x.length; q++) s += G.w[q] * evalCorrection(co, G.x[q]).dgL * legendreP(j, G.x[q]);
      assertClose(s, -legendreP(j, -1), 1e-12, 0, `p=${p} j=${j}`);
    }
  }
});

test('VCJH: c = 0 → g_DG, c_SD → g_Ga, c_HU → g_2 (identical Legendre coefficients)', () => {
  for (let p = 1; p <= 7; p++) {
    const same = (a, b, msg) => { for (let k = 0; k < a.length; k++) assertClose(a[k], b[k], 1e-12, 0, `${msg} p=${p} k=${k}`); };
    same(correctionCoeffs('vcjh', p, 0), correctionCoeffs('dg', p), 'c=0');
    same(correctionCoeffs('vcjh', p, cSD(p)), correctionCoeffs('ga', p), 'c_SD');
    same(correctionCoeffs('vcjh', p, cHU(p)), correctionCoeffs('g2', p), 'c_HU');
    assertClose(vcjhEta(p, cSD(p)), p / (p + 1), 1e-12);
    assertClose(vcjhEta(p, cHU(p)), (p + 1) / p, 1e-12);
  }
});

test('g_Ga = (−1)^p/2 (1−ξ) P_p vanishes at the p Gauss points', () => {
  for (let p = 1; p <= 7; p++) {
    const co = correctionCoeffs('ga', p), G = gaussLegendre(p);
    for (let q = 0; q < p; q++) assertClose(evalCorrection(co, G.x[q]).gL, 0, 1e-13, 0, `p=${p}`);
    for (const x of [-0.6, 0.15, 0.7]) assertClose(evalCorrection(co, x).gL, (p % 2 ? -0.5 : 0.5) * (1 - x) * legendreP(p, x), 1e-13);
  }
});

test('Correction functions have degree exactly p+1 (finite c) and the c → ∞ limit is R_{R,p}', () => {
  for (let p = 1; p <= 6; p++) {
    for (const [t, c] of TYPES) assert(Math.abs(correctionCoeffs(t, p, c)[p + 1]) > 1e-12, `${t} degree`);
    const big = correctionCoeffs('vcjh', p, 1e12);
    for (const x of [-0.5, 0.25]) assertClose(evalCorrection(big, x).gL, radauRight(p, x).v, 1e-9);
  }
});
