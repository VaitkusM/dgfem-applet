import { test, assert, assertClose, assertRate } from '../harness.js';
import { circleLS, flowerLS } from '../../lib/core/mesh/levelset.js';
import { solveCutFEM, errorsCutFEM, CUT } from '../../lib/core/immersed/cutfem.js';
import { solveSBM, errorsSBM } from '../../lib/core/immersed/sbm.js';
import { POISSON_MMS } from '../../lib/core/verify/mms.js';
import { fitRate } from '../../lib/core/verify/rates.js';
import { conditionEstimate } from '../../lib/core/la/eig.js';
import { csrSymmetryError, csrToDense } from '../../lib/core/la/sparse.js';
import { symEig } from '../../lib/core/la/dense.js';

const S = POISSON_MMS.wave;
const FLOWER = { R: 0.3, a: 0.2, k: 5 };
let seed = 11;
const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };

test('CutFEM P1 + ghost penalty: L2 rate ≈ 2, H1 rate ≈ 1 for random domain offsets (circle and flower)', () => {
  for (let trial = 0; trial < 3; trial++) {
    const cx = 0.45 + 0.1 * rnd(), cy = 0.45 + 0.1 * rnd();
    for (const ls of [circleLS({ cx, cy, R: 0.33 }), flowerLS({ cx, cy, ...FLOWER, rot: 6 * rnd() })]) {
      const hs = [], L2 = [], H1 = [];
      for (const N of [16, 32, 64]) {
        const s = solveCutFEM({ ls, N, f: S.f, g: S.u });
        const e = errorsCutFEM(s, S.u, S.grad);
        hs.push(1 / N); L2.push(e.L2); H1.push(e.H1);
      }
      assertRate(fitRate(hs, L2), 2, `${ls.kind} L2`);
      assertRate(fitRate(hs, H1), 1, `${ls.kind} H1`);
    }
  }
});

test('CutFEM reproduces a linear exact solution (consistency of Nitsche + ghost penalty); symmetric matrix', () => {
  const u = (x, y) => 1 + 2 * x - y;
  const s = solveCutFEM({ ls: flowerLS({ cx: 0.52, cy: 0.49, ...FLOWER }), N: 16, f: () => 0, g: u, gammaG: 0.5 });
  for (let d = 0; d < s.nDof; d++) {
    const v = s.dofVert[d];
    assertClose(s.U[d], u(s.mesh.nodes[2 * v], s.mesh.nodes[2 * v + 1]), 1e-10);
  }
  assert(csrSymmetryError(s.A) < 1e-13, 'symmetric');
});

test('small cuts: κ(A)·h² bounded WITH ghost penalty, huge WITHOUT; without it A can be indefinite', () => {
  // random offsets: κ h² stays O(1) with the ghost penalty
  let worst = 0;
  for (let trial = 0; trial < 6; trial++) {
    const N = trial < 3 ? 16 : 32, h = 1 / N;
    const ls = trial % 2 ? circleLS({ cx: 0.45 + 0.1 * rnd(), cy: 0.45 + 0.1 * rnd(), R: 0.3 + 0.05 * rnd() })
      : flowerLS({ cx: 0.45 + 0.1 * rnd(), cy: 0.45 + 0.1 * rnd(), ...FLOWER, rot: 6 * rnd() });
    const s = solveCutFEM({ ls, N, f: S.f, g: S.u, solve: false });
    worst = Math.max(worst, conditionEstimate(s.A).kappa * h * h);
  }
  assert(worst < 5, `κh² with ghost penalty ${worst}`);
  // deliberately tiny cut: circle through the grid vertex (0.5 + 5h, 0.5) shifted out by 1e-6 h
  const N = 16, h = 1 / N, ls = circleLS({ cx: 0.5, cy: 0.5, R: 5 * h + 1e-6 * h });
  const on = solveCutFEM({ ls, N, f: S.f, g: S.u, solve: false });
  const off = solveCutFEM({ ls, N, f: S.f, g: S.u, ghost: false, solve: false });
  let fmin = 1;
  for (let t = 0; t < on.mesh.nTri; t++) if (on.cut.cls[t] === CUT) fmin = Math.min(fmin, on.cut.frac[t]);
  assert(fmin < 1e-10, `smallest volume fraction ${fmin}`);
  const kOn = conditionEstimate(on.A).kappa * h * h, kOff = conditionEstimate(off.A).kappa * h * h;
  assert(kOn < 5, `with GP ${kOn}`);
  assert(kOff > 1e8, `without GP ${kOff}`);
  // generic configuration without ghost penalty: negative eigenvalue (loss of coercivity)
  const g = solveCutFEM({ ls: circleLS({ cx: 0.513, cy: 0.478, R: 0.33 }), N: 16, f: S.f, g: S.u, ghost: false, solve: false });
  const ev = symEig(csrToDense(g.A), g.nDof).values;
  assert(ev[0] < 0, `expected an indefinite matrix, λmin = ${ev[0]}`);
});

test('SBM: L2 rate ≈ 2 and H1 rate ≈ 1 with the shift; L2 rate ≈ 1 without it', () => {
  for (const ls of [circleLS({ cx: 0.513, cy: 0.478, R: 0.33 }), flowerLS({ cx: 0.51, cy: 0.47, ...FLOWER, rot: 0.3 })]) {
    for (const shift of [true, false]) {
      const hs = [], L2 = [], H1 = [];
      for (const N of [32, 64, 128]) {
        const s = solveSBM({ ls, N, f: S.f, g: S.u, shift });
        const e = errorsSBM(s, S.u, S.grad);
        hs.push(1 / N); L2.push(e.L2); H1.push(e.H1);
      }
      if (shift) { assertRate(fitRate(hs, L2), 2, `${ls.kind} shifted L2`); assertRate(fitRate(hs, H1), 1, `${ls.kind} shifted H1`); }
      else assertRate(fitRate(hs, L2), 1, `${ls.kind} unshifted L2`);
    }
  }
});

test('SBM: exact for linear solutions with the shift (u + ∇u·d = u(x̃ + d)), not without; d reaches Γ', () => {
  const u = (x, y) => 1 + 2 * x - y;
  const ls = flowerLS({ cx: 0.52, cy: 0.49, ...FLOWER, rot: 1 });
  const s = solveSBM({ ls, N: 16, f: () => 0, g: u });
  const nod = s.mesh.nodes;
  let err = 0;
  for (let v = 0; v < nod.length / 2; v++) if (s.dofOf[v] >= 0) err = Math.max(err, Math.abs(s.U[s.dofOf[v]] - u(nod[2 * v], nod[2 * v + 1])));
  assert(err < 1e-10, `shifted SBM error ${err}`);
  const s0 = solveSBM({ ls, N: 16, f: () => 0, g: u, shift: false });
  let err0 = 0;
  for (let v = 0; v < nod.length / 2; v++) if (s0.dofOf[v] >= 0) err0 = Math.max(err0, Math.abs(s0.U[s0.dofOf[v]] - u(nod[2 * v], nod[2 * v + 1])));
  assert(err0 > 1e-3, `unshifted error ${err0}`);
  for (let q = 0; q < s.qCP.length / 2; q++) assert(Math.abs(ls.phi(s.qCP[2 * q], s.qCP[2 * q + 1])) < 1e-12, 'x̃ + d on Γ');
});
