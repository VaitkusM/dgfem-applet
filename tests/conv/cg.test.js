import { test, assert, assertRate } from '../harness.js';
import { triGrid } from '../../lib/core/mesh/structured.js';
import { solvePoissonCG, errorsCG, boundaryError } from '../../lib/core/elliptic/cg.js';
import { POISSON_MMS } from '../../lib/core/verify/mms.js';
import { fitRate } from '../../lib/core/verify/rates.js';
import { csrSymmetryError } from '../../lib/core/la/sparse.js';

const run = (p, bc, Ns, sol = 'wave', extra = {}) => {
  const S = POISSON_MMS[sol], hs = [], L2 = [], H1 = [];
  for (const N of Ns) {
    const mesh = triGrid(N, N, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.15 });
    const r = solvePoissonCG(mesh, { p, f: S.f, g: S.u, bc, ...extra });
    const e = errorsCG(r.space, r.U, S.u, S.grad);
    hs.push(1 / N); L2.push(e.L2); H1.push(e.H1);
  }
  return { L2: fitRate(hs, L2), H1: fitRate(hs, H1), errs: L2 };
};

test('CG P1/P2 with strong Dirichlet BCs: L2 rate p+1, H1 rate p', () => {
  for (const p of [1, 2]) {
    const r = run(p, 'strong', [8, 16, 32]);
    assertRate(r.L2, p + 1, `P${p} L2`); assertRate(r.H1, p, `P${p} H1`);
  }
});

test('CG with symmetric Nitsche BCs: optimal rates; matrix symmetric', () => {
  for (const p of [1, 2]) {
    const r = run(p, 'nitsche', [8, 16, 32], 'wave', { gamma: 10 * p * p });
    assertRate(r.L2, p + 1, `Nitsche P${p} L2`); assertRate(r.H1, p, `Nitsche P${p} H1`);
  }
  const S = POISSON_MMS.wave;
  const res = solvePoissonCG(triGrid(6, 6), { p: 2, f: S.f, g: S.u, bc: 'nitsche', gamma: 40 });
  assert(csrSymmetryError(res.A) < 1e-13, 'symmetric');
});

test('non-symmetric Nitsche converges (H1 rate p); penalty-only loses L2 order for P1', () => {
  for (const p of [1, 2]) {
    const r = run(p, 'nitsche-nonsym', [8, 16, 32], 'wave', { gamma: 10 });
    assertRate(r.H1, p, `nonsym P${p} H1`);
  }
  const pen = run(1, 'penalty', [8, 16, 32, 64], 'wave', { gamma: 10 });
  assert(pen.L2 < 1.4, `penalty L2 rate ${pen.L2} should be about 1`);
  assertRate(pen.L2, 1, 'penalty L2', 0.3, 0.3);
});

test('strong BCs interpolate g exactly at boundary nodes; Nitsche only approximately', () => {
  const S = POISSON_MMS.wave, mesh = triGrid(8, 8);
  const rs = solvePoissonCG(mesh, { p: 1, f: S.f, g: S.u, bc: 'strong' });
  const rn = solvePoissonCG(mesh, { p: 1, f: S.f, g: S.u, bc: 'nitsche', gamma: 10 });
  for (let i = 0; i < rs.space.nDof; i++) if (rs.space.onBoundary[i])
    assert(Math.abs(rs.U[i] - S.u(rs.space.dofXY[2 * i], rs.space.dofXY[2 * i + 1])) < 1e-14);
  const be = boundaryError(rn.space, rn.U, S.u);
  assert(be > 1e-5 && be < 0.05, `Nitsche boundary mismatch ${be}`);
});
