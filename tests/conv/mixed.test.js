/**
 * Mixed RT0, hybridized RT0 and HDG for −Δu = f with σ = −∇u:
 * convergence rates, conservation, normal continuity, equivalences.
 */
import { test, assert, assertRate } from '../harness.js';
import { triGrid } from '../../lib/core/mesh/structured.js';
import { solveMixedRT0, rt0Errors, rt0ConservationResidual, rt0Local, rt0Eval } from '../../lib/core/elliptic/mixedrt0.js';
import { solveHybridRT0, solveHybridRT0Monolithic, edgeMeans } from '../../lib/core/elliptic/hybridrt0.js';
import { solveHDG, solveHDGMonolithic, hdgErrors } from '../../lib/core/elliptic/hdg.js';
import { POISSON_MMS } from '../../lib/core/verify/mms.js';
import { fitRate } from '../../lib/core/verify/rates.js';
import { csrSymmetryError, csrToDense, nnz } from '../../lib/core/la/sparse.js';
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
        // verts = [x0,y0,x1,y1,x2,y2] (typed array): end points of local edge j
        const e = L.edges[j], j1 = (j + 1) % 3, a = [L.verts[2 * j], L.verts[2 * j + 1]], b = [L.verts[2 * j1], L.verts[2 * j1 + 1]];
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

// ---------------------------------------------------------------------------------------------
// Claims made in chapter 9 (prose, captions, "try this") — AGENTS.md rule 3
// ---------------------------------------------------------------------------------------------
/** the mesh family of the chapter-9 widgets */
const widgetMesh = (N) => triGrid(N, N, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.12 });
const rates = (hs, es) => es.slice(1).map((e, i) => Math.log(es[i] / e) / Math.log(hs[i] / hs[i + 1]));

test('HDG k = 3 (offered by the widget): rates 4, 4, 5 and SPD trace matrix for τ = 0.01, 1, 100', () => {
  const hs = [], eu = [], eq = [], es = [];
  for (const N of [4, 8, 16]) {
    const e = hdgErrors(solveHDG(widgetMesh(N), { k: 3, tau: 1, f: S.f, g: S.u }), S.u, S.grad);
    hs.push(1 / N); eu.push(e.u); eq.push(e.q); es.push(e.ustar);
  }
  assertRate(fitRate(hs, eu), 4, 'k=3 u'); assertRate(fitRate(hs, eq), 4, 'k=3 q'); assertRate(fitRate(hs, es), 5, 'k=3 u*');
  // SPD over the whole range of the widget's τ slider, for every degree (k = 0…3)
  for (const k of [0, 1, 2, 3]) for (const tau of [0.01, 1, 100]) {
    const H = solveHDG(widgetMesh(3), { k, tau, f: S.f, g: S.u });
    assert(csrSymmetryError(H.K) < 1e-12, 'symmetric');
    assert(cholesky(csrToDense(H.K), H.K.n) !== null, `k=${k} tau=${tau} SPD`);
  }
});

test('RT0 and HDG rates for a second manufactured solution (sin·sin)', () => {
  const P = POISSON_MMS.sinsin, hs = [], eu = [], es = [], hu = [], hq = [], hst = [];
  for (const N of [8, 16, 32]) {
    const e = rt0Errors(solveMixedRT0(mesh(N), { f: P.f, g: P.u }), P.u, P.grad);
    const h = hdgErrors(solveHDG(mesh(N), { k: 1, f: P.f, g: P.u }), P.u, P.grad);
    hs.push(1 / N); eu.push(e.L2u); es.push(e.L2s); hu.push(h.u); hq.push(h.q); hst.push(h.ustar);
  }
  assertRate(fitRate(hs, eu), 1, 'RT0 u'); assertRate(fitRate(hs, es), 1, 'RT0 sigma');
  assertRate(fitRate(hs, hu), 2, 'HDG u'); assertRate(fitRate(hs, hq), 2, 'HDG q'); assertRate(fitRate(hs, hst), 3, 'HDG u*');
});

test('HDG "try this" (k = 1, widget meshes): τ = 0.01 inflates the u_h error ~85×, τ = 100 degrades q_h and the extra order of u*', () => {
  const run = (tau) => {
    const hs = [], E = [];
    for (const N of [2, 4, 8, 16]) { hs.push(1 / N); E.push(hdgErrors(solveHDG(widgetMesh(N), { k: 1, tau, f: S.f, g: S.u }), S.u, S.grad)); }
    return { E, ru: rates(hs, E.map((e) => e.u)), rq: rates(hs, E.map((e) => e.q)), rs: rates(hs, E.map((e) => e.ustar)) };
  };
  const small = run(0.01), one = run(1), big = run(100);
  // N = 4 (index 1): "almost two orders of magnitude" in u_h, q_h and u* within 15 %
  const f = small.E[1].u / one.E[1].u;
  assert(f > 75 && f < 95, `u error factor τ=0.01 vs τ=1: ${f}`);
  assert(Math.abs(small.E[1].q / one.E[1].q - 1) < 0.15 && Math.abs(small.E[1].ustar / one.E[1].ustar - 1) < 0.15, 'q and u* hardly change');
  // τ = 1: last observed rates 2, 2, 3
  assertRate(one.ru[2], 2, 'τ=1 u', 0.15, 0.15); assertRate(one.rq[2], 2, 'τ=1 q', 0.15, 0.15); assertRate(one.rs[2], 3, 'τ=1 u*', 0.15, 0.15);
  // τ = 100: the flux error is an order of magnitude larger and u* has lost its extra order on these meshes
  assert(big.E[3].q > 10 * one.E[3].q, `q error τ=100 vs τ=1 at N=16: ${big.E[3].q / one.E[3].q}`);
  assert(big.rq[2] < 1.7, `q rate at τ=100: ${big.rq[2]}`);
  assert(big.rs[2] < 2.6, `u* rate at τ=100: ${big.rs[2]}`);
});

test('hybridized RT0: H has at most 5 non-zeros per row; constants are reproduced (kernel of the local Schur complements)', () => {
  const m = widgetMesh(6), hy = solveHybridRT0(m, { f: S.f, g: S.u });
  let mx = 0; for (let i = 0; i < hy.H.n; i++) mx = Math.max(mx, hy.H.rowPtr[i + 1] - hy.H.rowPtr[i]);
  assert(mx === 5 && nnz(hy.H) <= 5 * hy.nLam, `max row length ${mx}`);
  // f = 0, g = 1: the solution is u = 1, σ = 0, λ = 1 — only possible if Σ_l (C_KᵀD_K⁻¹C_K)_{kl} = 0 on every triangle
  const c = solveHybridRT0(m, { f: () => 0, g: () => 1 });
  for (let e = 0; e < c.topo.nEdge; e++) assert(Math.abs(c.lambda[e] - 1) < 1e-12 && Math.abs(c.sigmaEdge[e]) < 1e-11, `edge ${e}`);
  for (let t = 0; t < c.nTri; t++) assert(Math.abs(c.u[t] - 1) < 1e-12, `u ${t}`);
});

test('a piecewise-constant gradient (P1 elements) has zero net outflow on every triangle, whatever f; the RT0 flux balances ∫_K f', () => {
  const m = widgetMesh(4), r = solveMixedRT0(m, { f: S.f, g: S.u }), res = rt0ConservationResidual(r), { topo } = r;
  let fmax = 0, rmax = 0, p1max = 0;
  for (let t = 0; t < r.nTri; t++) {
    // gradient of the P1 interpolant of u on triangle t (solve the 2×2 system of the two edge vectors)
    const L = rt0Local(m, topo, t), v = L.verts, u0 = S.u(v[0], v[1]), u1 = S.u(v[2], v[3]), u2 = S.u(v[4], v[5]);
    const ax = v[2] - v[0], ay = v[3] - v[1], bx = v[4] - v[0], by = v[5] - v[1], det = ax * by - ay * bx;
    const gx = ((u1 - u0) * by - (u2 - u0) * ay) / det, gy = (-(u1 - u0) * bx + (u2 - u0) * ax) / det;
    let out = 0; // ∮ −∇u_h·n_K = Σ_k −(∇u_h·n_E) s_{K,E} |E|
    for (let k = 0; k < 3; k++) { const e = L.edges[k]; out -= (gx * topo.normals[2 * e] + gy * topo.normals[2 * e + 1]) * L.sign[k] * topo.lengths[e]; }
    p1max = Math.max(p1max, Math.abs(out)); fmax = Math.max(fmax, Math.abs(r.F[t])); rmax = Math.max(rmax, Math.abs(res[t]));
  }
  assert(p1max < 1e-13, `P1 net outflow ${p1max}`);
  assert(fmax > 1e-2 && rmax < 1e-12 * (1 + fmax), `RT0 residual ${rmax}, max |∫_K f| = ${fmax}`);
});

test('dimension count behind the inf–sup counter-example: 2(N+1)² continuous P1 flux unknowns < 6N² discontinuous P1 unknowns for N ≥ 2', () => {
  for (const N of [1, 2, 3, 8]) {
    const m = triGrid(N, N), nV = m.nodes.length / 2, nT = m.tris.length / 3;
    assert(2 * nV === 2 * (N + 1) ** 2 && 3 * nT === 6 * N * N, 'counts');
    assert((2 * nV < 3 * nT) === (N >= 2), `N=${N}`);
  }
});
