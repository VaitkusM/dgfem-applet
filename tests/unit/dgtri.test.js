/**
 * Broken polynomial spaces on triangles (dgtri.js) and system-size counting (dofcount.js).
 */
import { test, assert, assertClose } from '../harness.js';
import { triGrid } from '../../lib/core/mesh/structured.js';
import { buildTopology, refEdgePoint } from '../../lib/core/mesh/topology.js';
import { dubinerBasis, dimP } from '../../lib/core/basis/simplex.js';
import { triGeometry, mapPoint, brokenEvaluator, l2Project, brokenErrors, basisTables } from '../../lib/core/elliptic/dgtri.js';
import { systemSizes } from '../../lib/core/elliptic/dofcount.js';
import { solvePoissonCG } from '../../lib/core/elliptic/cg.js';
import { assembleIP } from '../../lib/core/elliptic/sipg.js';
import { solveHybridRT0 } from '../../lib/core/elliptic/hybridRT0.js';
import { solveHDG } from '../../lib/core/elliptic/hdg.js';
import { nnz } from '../../lib/core/la/sparse.js';
import { symEig } from '../../lib/core/la/dense.js';
import { gaussLegendre } from '../../lib/core/quad/gauss1d.js';
import { legendreP } from '../../lib/core/basis/legendre.js';

const mesh = triGrid(4, 4, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.15 });

test('edge orientation: parameter s in K⁻ is 1 − s in K⁺ (same physical point)', () => {
  const topo = buildTopology(mesh.nodes, mesh.tris), geo = triGeometry(mesh);
  for (let e = 0; e < topo.nEdge; e++) {
    const tp = topo.edgeTris[2 * e + 1];
    if (tp < 0) continue;
    const tm = topo.edgeTris[2 * e];
    for (const s of [0.1, 0.5, 0.77]) {
      const a = mapPoint(geo, tm, ...refEdgePoint(topo.edgeLocal[2 * e], s));
      const b = mapPoint(geo, tp, ...refEdgePoint(topo.edgeLocal[2 * e + 1], 1 - s));
      assertClose(a[0], b[0], 1e-14); assertClose(a[1], b[1], 1e-14);
    }
  }
});

test('Dubiner → monomial evaluator reproduces the Dubiner expansion', () => {
  for (const p of [0, 1, 2, 3, 4]) {
    const n = dimP(p), U = new Float64Array(2 * n).map((_, i) => Math.sin(1.3 * i + 0.2));
    const ev = brokenEvaluator(p, U);
    for (const [x, y] of [[0.1, 0.2], [0.6, 0.3], [0, 1], [0.25, 0.25]]) {
      const B = dubinerBasis(p, x, y);
      for (const t of [0, 1]) {
        let s = 0; for (let m = 0; m < n; m++) s += U[t * n + m] * B.v[m];
        assertClose(ev(t, x, y), s, 1e-11, 0, `p=${p}`);
      }
    }
  }
});

test('broken L² projection is exact for polynomials of degree ≤ p; mass matrix is det J · I', () => {
  const u = (x, y) => 1 + 2 * x - y + x * y - 3 * y * y, grad = (x, y) => [2 + y, -1 + x - 6 * y];
  const U = l2Project(mesh, 2, u), e = brokenErrors(mesh, 2, U, u, grad);
  assert(e.L2 < 1e-13 && e.H1 < 1e-12, `exact: ${e.L2} ${e.H1}`);
  const T = basisTables(3, 6, 1), R = T.vol.rule, n = T.n;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let s = 0; for (let q = 0; q < R.n; q++) s += R.w[q] * T.vol.V[q * n + i] * T.vol.V[q * n + j];
    assertClose(s, i === j ? 1 : 0, 1e-13);
  }
});

test('system sizes: closed forms on an N×N grid and agreement with assembled matrices', () => {
  for (const N of [3, 5]) for (const p of [1, 2, 3]) {
    const m = triGrid(N, N, [0, 1, 0, 1], { diag: 'alt', jiggle: 0.15 }), s = systemSizes(m, p);
    const nT = 2 * N * N, nIntE = 3 * N * N - 2 * N;
    assert(s.nTri === nT && s.nIntEdge === nIntE, 'counts');
    assert(s.cg.n === (p * N - 1) ** 2, `CG free DOFs ${s.cg.n}`);
    assert(s.dg.n === nT * dimP(p), 'DG DOFs');
    assert(s.hdg.n === nIntE * (p + 1), 'HDG traces');
    // assembled matrices may contain a few exact zeros (dropped by the builder): structural ≥ assembled ≥ 98 %
    const f = () => 1, near = (a, b, msg) => assert(a <= b && a >= 0.98 * b, `${msg}: assembled ${a}, structural ${b}`);
    if (p <= 2) near(nnz(solvePoissonCG(m, { p, f, g: f }).A), s.cg.nnz, `CG nnz p=${p}`);
    near(nnz(assembleIP(m, { p, f }).A), s.dg.nnz, `DG nnz p=${p}`);
    near(nnz(solveHDG(m, { k: p, f }).K), s.hdg.nnz, `HDG nnz p=${p}`);
  }
  const m = triGrid(4, 4, [0, 1, 0, 1], { jiggle: 0.15 });
  const h = solveHybridRT0(m, { f: () => 1 });
  assert(h.nLam === systemSizes(m, 1).nIntEdge, 'hybrid RT0: one multiplier per interior edge');
});

test('system sizes on a 16×16 grid: HDG has fewer unknowns than DG always and than CG for p ≥ 4 (fewer non-zeros for p ≥ 5)', () => {
  const m = triGrid(16, 16);
  for (let p = 1; p <= 6; p++) {
    const s = systemSizes(m, p);
    assert(s.hdg.n < s.dg.n && s.hdg.nnz < s.dg.nnz, `HDG < DG at p=${p}`);
    assert(s.dg.n > s.cg.n && s.dg.nnz > s.cg.nnz, `DG > CG at p=${p}`);
    if (p >= 4) assert(s.hdg.n < s.cg.n, `HDG < CG unknowns at p=${p}`); else assert(s.hdg.n > s.cg.n, `HDG > CG unknowns at p=${p}`);
    if (p >= 5) assert(s.hdg.nnz < s.cg.nnz, `HDG < CG nnz at p=${p}`);
  }
});

test('sharp polynomial trace inequalities: (p+1)²/h in 1D and (p+1)(p+2)/2·|F|/|K| on triangles', () => {
  for (let p = 1; p <= 5; p++) {
    // triangle: largest eigenvalue of the face mass matrix in the L²(K)-orthonormal basis = sup ‖v‖²_F / ‖v‖²_K
    const T = basisTables(p, 1, 2 * p + 2), n = T.n, E = T.edge;
    for (let k = 0; k < 3; k++) {
      const len = [1, Math.SQRT2, 1][k], M = new Float64Array(n * n);
      for (let q = 0; q < E.nq; q++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++)
        M[i * n + j] += E.w[q] * len * E.V[k][0][q * n + i] * E.V[k][0][q * n + j];
      assertClose(symEig(M, n).values[n - 1], (p + 1) * (p + 2) / 2 * len / 0.5, 1e-10, 1e-12, `p=${p} edge ${k}`);
    }
    // 1D on [0,h]: the extremal polynomial is w = Σ_j L_j(1) L_j  (L_j orthonormal Legendre on [−1,1])
    const h = 0.3, G = gaussLegendre(p + 2);
    const w = (xi) => { let s = 0; for (let j = 0; j <= p; j++) s += (2 * j + 1) / 2 * legendreP(j, xi); return s; };
    let l2 = 0; for (let q = 0; q < G.x.length; q++) l2 += G.w[q] * h / 2 * w(G.x[q]) ** 2;
    assertClose(w(1) ** 2 / l2, (p + 1) ** 2 / h, 1e-9, 1e-12, `1D p=${p}`);
  }
});
