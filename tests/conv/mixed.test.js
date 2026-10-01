/**
 * Mixed RT0, hybridized RT0 and HDG for −Δu = f with σ = −∇u:
 * convergence rates, conservation, normal continuity, equivalences.
 */
import { test, assert, assertRate } from '../harness.js';
import { triGrid } from '../../lib/core/mesh/structured.js';
import { solveMixedRT0, rt0Errors, rt0ConservationResidual, rt0Local, rt0Eval } from '../../lib/core/elliptic/mixedRT0.js';
import { solveHybridRT0, solveHybridRT0Monolithic, edgeMeans } from '../../lib/core/elliptic/hybridRT0.js';
import { solveHDG, solveHDGMonolithic, hdgErrors } from '../../lib/core/elliptic/hdg.js';
import { POISSON_MMS } from '../../lib/core/verify/mms.js';
import { fitRate } from '../../lib/core/verify/rates.js';
import { csrSymmetryError, csrToDense } from '../../lib/core/la/sparse.js';
import { cholesky } from '../../lib/core/la/dense.js';
import { inertia } from '../../lib/core/la/inertia.js';

const S = POISSON_MMS.wave;
const mesh = (N) => triGrid(N, N, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.15 });

test('RT0: ‖u − u_h‖ and ‖σ − σ_h‖ rate 1; ‖Π₀u − u_h‖ superconverges with rate 2', () => {
  const hs = [], eu = [], es = [], e0 = [];
  for (const N of [8, 16, 32]) {
    const r = solveMixedRT0(mesh(N), { f: S.f, g: S.u }), e = rt0Errors(r, S.u, S.grad);
    hs.push(1 / N); eu.push(e.L2u); es.push(e.L2s); e0.push(e.L2u0);
  }
  assertRate(fitRate(hs, eu), 1, 'u'); assertRate(fitRate(hs, es), 1, 'sigma'); assertRate(fitRate(hs, e0), 2, 'Pi0 u');
});

test('RT0: saddle matrix symmetric and indefinite; exact discrete conservation; normal continuity', () => {
  const m = mesh(6), r = solveMixedRT0(m, { f: S.f, g: S.u });
  assert(csrSymmetryError(r.A) < 1e-14, 'symmetric');
  assert(cholesky(csrToDense(r.A), r.A.n) === null, 'not positive definite');
  const res = rt0ConservationResidual(r);
  for (let t = 0; t < r.nTri; t++) assert(Math.abs(res[t]) < 1e-12 * (1 + Math.abs(r.F[t])), `conservation K${t}: ${res[t]}`);
  // σ_h·n_E evaluated from both sides at edge points agrees (and equals the edge coefficient)
  const { topo } = r;
  for (let e = 0; e < topo.nEdge; e++) {
    const a = topo.edges[2 * e], b = topo.edges[2 * e + 1], nx = topo.normals[2 * e], ny = topo.normals[2 * e + 1];
    for (const s of [0.2, 0.5, 0.9]) {
      const x = m.nodes[2 * a] + s * (m.nodes[2 * b] - m.nodes[2 * a]), y = m.nodes[2 * a + 1] + s * (m.nodes[2 * b + 1] - m.nodes[2 * a + 1]);
      for (const side of [0, 1]) {
        const t = topo.edgeTris[2 * e + side];
        if (t < 0) continue;
        const [sx, sy] = rt0Eval(rt0Local(m, topo, t), r.sigma, x, y);
        assert(Math.abs(sx * nx + sy * ny - r.sigma[e]) < 1e-10, `normal continuity e=${e}`);
      }
    }
  }
});

test('RT0 basis: φ_E·n_E = 1 on E, φ_E·n = 0 on the other edges, ∫_K ∇·φ_E = ±|E|', () => {
  const m = mesh(3), r = solveMixedRT0(m, { f: S.f }), { topo } = r;
  for (let t = 0; t < r.nTri; t++) {
    const L = rt0Local(m, topo, t);
    for (let k = 0; k < 3; k++) {
      const sig = new Float64Array(topo.nEdge); sig[L.edges[k]] = 1;
      for (let j = 0; j < 3; j++) {
        const e = L.edges[j], a = L.verts[j], b = L.verts[(j + 1) % 3];
        const [sx, sy] = rt0Eval(L, sig, 0.3 * a[0] + 0.7 * b[0], 0.3 * a[1] + 0.7 * b[1]);
        const fl = sx * topo.normals[2 * e] + sy * topo.normals[2 * e + 1];
        assert(Math.abs(fl - (j === k ? 1 : 0)) < 1e-12, `flux of φ_${k} through edge ${j}`);
      }
      // divergence = 2·c_k (∇·(x − x_opp) = 2), integral = 2 c_k |K| = s |E|
      assert(Math.abs(2 * L.c[k] * L.area - L.sign[k] * topo.lengths[L.edges[k]]) < 1e-14, 'div');
    }
  }
});

test('Hybridized RT0 = mixed RT0 to round-off; condensed = monolithic; H is SPD', () => {
  for (const N of [4, 8]) {
    const m = mesh(N), mx = solveMixedRT0(m, { f: S.f, g: S.u }), hy = solveHybridRT0(m, { f: S.f, g: S.u });
    for (let e = 0; e < mx.nEdge; e++) assert(Math.abs(mx.sigma[e] - hy.sigmaEdge[e]) < 1e-11, `sigma e=${e}`);
    for (let t = 0; t < mx.nTri; t++) assert(Math.abs(mx.u[t] - hy.u[t]) < 1e-12, `u t=${t}`);
    // broken fluxes are continuous: σ_{K⁻}·n_{K⁻} = −σ_{K⁺}·n_{K⁺}
    const { topo } = hy;
    for (let e = 0; e < topo.nEdge; e++) {
      const tp = topo.edgeTris[2 * e + 1];
      if (tp >= 0) assert(Math.abs(hy.sigmaLocal[3 * topo.edgeTris[2 * e] + topo.edgeLocal[2 * e]] + hy.sigmaLocal[3 * tp + topo.edgeLocal[2 * e + 1]]) < 1e-11, 'continuity');
    }
    const mono = solveHybridRT0Monolithic(m, { f: S.f, g: S.u });
    for (let e = 0; e < topo.nEdge; e++) if (hy.lamIndex[e] >= 0) assert(Math.abs(mono.lambda[hy.lamIndex[e]] - hy.lambda[e]) < 1e-12, 'lambda');
    assert(csrSymmetryError(hy.H) < 1e-14, 'H symmetric');
    assert(cholesky(csrToDense(hy.H), hy.nLam) !== null, 'H SPD');
  }
});

test('Hybridized RT0: λ approximates the edge means of u (weighted edge-L² rate ≈ 2)', () => {
  const hs = [], es = [];
  for (const N of [8, 16, 32]) {
    const m = mesh(N), hy = solveHybridRT0(m, { f: S.f, g: S.u }), em = edgeMeans(m, hy.topo, S.u);
    let s = 0; for (let e = 0; e < hy.topo.nEdge; e++) s += hy.topo.lengths[e] / N * (hy.lambda[e] - em[e]) ** 2;
    hs.push(1 / N); es.push(Math.sqrt(s));
  }
  assertRate(fitRate(hs, es), 2, 'lambda', 0.3, 0.3);
});

test('HDG (τ = 1): u_h and q_h rate k+1, post-processed u* rate k+2 (k = 1, 2)', () => {
  for (const k of [1, 2]) {
    const hs = [], eu = [], eq = [], es = [];
    for (const N of k === 1 ? [8, 16, 32] : [4, 8, 16]) {
      const H = solveHDG(mesh(N), { k, tau: 1, f: S.f, g: S.u }), e = hdgErrors(H, S.u, S.grad);
      hs.push(1 / N); eu.push(e.u); eq.push(e.q); es.push(e.ustar);
    }
    assertRate(fitRate(hs, eu), k + 1, `k=${k} u`);
    assertRate(fitRate(hs, eq), k + 1, `k=${k} q`);
    assertRate(fitRate(hs, es), k + 2, `k=${k} u*`);
  }
});

test('HDG: condensed solve = monolithic solve; trace matrix symmetric positive definite', () => {
  for (const k of [0, 1, 2]) for (const tau of [1, 10]) {
    const m = mesh(4), H = solveHDG(m, { k, tau, f: S.f, g: S.u }), M = solveHDGMonolithic(m, { k, tau, f: S.f, g: S.u });
    for (let i = 0; i < H.u.length; i++) assert(Math.abs(H.u[i] - M.u[i]) < 1e-12, `k=${k} u[${i}]`);
    assert(M.residual < 1e-12, 'monolithic residual');
    assert(csrSymmetryError(H.K) < 1e-13, 'symmetric');
    assert(cholesky(csrToDense(H.K), H.K.n) !== null, `k=${k} tau=${tau} SPD`);
  }
});

test('HDG k = 0 with τ = 1 converges with rate 1 (no superconvergence of u* for k = 0)', () => {
  const hs = [], eu = [], es = [];
  for (const N of [8, 16, 32]) {
    const e = hdgErrors(solveHDG(mesh(N), { k: 0, f: S.f, g: S.u }), S.u, S.grad);
    hs.push(1 / N); eu.push(e.u); es.push(e.ustar);
  }
  assertRate(fitRate(hs, eu), 1, 'k=0 u');
  assertRate(fitRate(hs, es), 1, 'k=0 u*');
});

test('RT0 saddle matrix inertia: exactly nEdge positive and nTri negative eigenvalues', () => {
  const r = solveMixedRT0(mesh(5), { f: S.f, g: S.u });
  const I = inertia(csrToDense(r.A), r.A.n);
  assert(I.pos === r.nEdge && I.neg === r.nTri && I.zero === 0, `inertia ${I.pos}/${I.neg}/${I.zero}`);
});

test('HDG τ-dependence (k = 1): small τ inflates the u_h error like 1/τ, u* hardly changes', () => {
  const m = mesh(8), e = (tau) => hdgErrors(solveHDG(m, { k: 1, tau, f: S.f, g: S.u }), S.u, S.grad);
  const a = e(0.01), b = e(0.1);
  assert(Math.abs(a.u / b.u / 10 - 1) < 0.1, `u ratio ${a.u / b.u}`);
  assert(Math.abs(a.ustar / b.ustar - 1) < 0.05, `u* ratio ${a.ustar / b.ustar}`);
  assert(Math.abs(a.q / b.q - 1) < 0.05, `q ratio ${a.q / b.q}`);
});
