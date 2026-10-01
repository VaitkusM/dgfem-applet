/**
 * Tests for lib/gpu/cpuTwin.js — the f64 reference implementation of the
 * WebGPU DG-SEM advection solver (chapter 12).
 */
import { test, assert, assertClose } from '../harness.js';
import { makeDGAdv2D, makeCpuSolver, LSRK4, CFL_SAFE, INITIAL } from '../../lib/gpu/cpuTwin.js';
import { makeStepper, stabilityAmp } from '../../lib/core/time/rk.js';
import { lagrangeValues, lagrangeDerivs } from '../../lib/core/basis/lagrange.js';
import { eigGeneral } from '../../lib/core/la/dense.js';

/**
 * Independent, deliberately naive WEAK-form DG right-hand side:
 *   (J w_i w_j) du_ij/dt = Σ_q J w_q (a·∇φ_ij)(x_q) u(x_q) − Σ_faces Σ_m (h/2) w_m F̂(x_m) φ_ij(x_m)
 * with GLL quadrature, basis derivatives from lagrangeDerivs (not the D matrix),
 * outward normals per face and neighbours found by matching physical coordinates.
 */
function naiveWeakRHS(d, u) {
  const { N, n, Np, h, ax, ay, xi, w } = d, out = new Float64Array(d.nDof), J = h * h / 4;
  const dl = Array.from(xi, (s) => lagrangeDerivs(xi, s)); // dl[k][i] = ℓ_i'(ξ_k)
  const key = (x, y) => `${Math.round(((x % 1) + 1) % 1 * 1e9) % 1e9},${Math.round(((y % 1) + 1) % 1 * 1e9) % 1e9}`;
  for (let e = 0; e < d.nElem; e++) {
    // map of this element's node values by physical position (for neighbour lookup)
    const ex = e % N, ey = (e / N) | 0;
    const faces = [ // [outward normal, neighbour element, which local nodes lie on the face]
      { nx: 1, ny: 0, nb: ey * N + (ex + 1) % N, on: (i, j) => i === n - 1 },
      { nx: -1, ny: 0, nb: ey * N + (ex + N - 1) % N, on: (i, j) => i === 0 },
      { nx: 0, ny: 1, nb: ((ey + 1) % N) * N + ex, on: (i, j) => j === n - 1 },
      { nx: 0, ny: -1, nb: ((ey + N - 1) % N) * N + ex, on: (i, j) => j === 0 },
    ];
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      let r = 0;
      // volume: Σ_{k,l} J w_k w_l (ax ∂xφ_ij + ay ∂yφ_ij)(ξ_k, ξ_l) u_kl
      for (let l = 0; l < n; l++) for (let k = 0; k < n; k++) {
        const phx = (2 / h) * dl[k][i] * (l === j ? 1 : 0), phy = (2 / h) * (k === i ? 1 : 0) * dl[l][j];
        r += J * w[k] * w[l] * (ax * phx + ay * phy) * u[e * Np + l * n + k];
      }
      // faces
      for (const F of faces) {
        if (!F.on(i, j)) continue; // φ_ij vanishes on faces not containing node ij (nodal basis, GLL nodes)
        const g = e * Np + j * n + i, an = ax * F.nx + ay * F.ny;
        // neighbour node at the same physical point
        const kk = key(d.X[g], d.Y[g]);
        let uP = NaN;
        for (let q = 0; q < Np; q++) { const gn = F.nb * Np + q; if (key(d.X[gn], d.Y[gn]) === kk) uP = u[gn]; }
        const Fh = an >= 0 ? an * u[g] : an * uP;
        const m = F.nx !== 0 ? j : i; // face-quadrature index of node ij
        r -= (h / 2) * w[m] * Fh;
      }
      out[e * Np + j * n + i] = r / (J * w[i] * w[j]);
    }
  }
  return out;
}

test('cpuTwin: strong-form kernels = naive weak-form DG (SBP equivalence, upwind, signs)', () => {
  for (const a of [[1, 0.5], [-0.7, 0.4], [0.3, -1]]) for (let p = 1; p <= 3; p++) {
    const d = makeDGAdv2D({ N: 3, p, a });
    // discontinuous random data
    let s = 12345; const rnd = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648) - 0.5;
    const u = new Float64Array(d.nDof).map(rnd);
    const r = new Float64Array(d.nDof); d.rhs(u, r);
    const ref = naiveWeakRHS(d, u);
    let m = 0; for (let g = 0; g < d.nDof; g++) m = Math.max(m, Math.abs(r[g] - ref[g]));
    assert(m < 1e-11, `a=${a} p=${p}: max diff ${m}`);
  }
});

test('cpuTwin: RHS = −a·∇u exactly for polynomials of degree ≤ p (interior elements)', () => {
  for (let p = 1; p <= 3; p++) {
    const d = makeDGAdv2D({ N: 4, p, a: [0.8, -0.6] });
    const f = (x, y) => x ** p * y ** p + 0.3 * x * y - x ** p;
    const fx = (x, y) => p * x ** (p - 1) * y ** p + 0.3 * y - p * x ** (p - 1);
    const fy = (x, y) => p * x ** p * y ** (p - 1) + 0.3 * x;
    const u = d.interpolate(f), r = new Float64Array(d.nDof);
    d.rhs(u, r);
    for (let e = 0; e < d.nElem; e++) {
      const ex = e % 4, ey = (e / 4) | 0;
      if (ex === 0 || ex === 3 || ey === 0 || ey === 3) continue; // these touch the periodic seam where f jumps
      for (let l = 0; l < d.Np; l++) {
        const g = e * d.Np + l, x = d.X[g], y = d.Y[g];
        assertClose(r[g], -(0.8 * fx(x, y) - 0.6 * fy(x, y)), 1e-11, 0, `p=${p} e=${e}`);
      }
    }
  }
});

test('cpuTwin: continuous data ⇒ face terms vanish (lift kernel adds ~0)', () => {
  const d = makeDGAdv2D({ N: 5, p: 3, a: [1, 0.5] });
  const u = d.interpolate(INITIAL.sines), r1 = new Float64Array(d.nDof), r2 = new Float64Array(d.nDof);
  d.volume(u, r1); d.rhs(u, r2);
  let m = 0; for (let g = 0; g < d.nDof; g++) m = Math.max(m, Math.abs(r1[g] - r2[g]));
  assert(m < 1e-12, `lift contribution ${m}`);
});

test('cpuTwin: mass conserved to round-off, energy non-increasing (discontinuous data)', () => {
  for (let p = 1; p <= 3; p++) {
    const d = makeDGAdv2D({ N: 6, p, a: [1, 0.5] });
    const u0 = d.interpolate((x, y) => (x > 0.2 && x < 0.55 && y > 0.3 && y < 0.6 ? 1 : 0) + 0.1 * Math.sin(6 * x));
    const S = makeCpuSolver(d, u0), m0 = d.mass(u0), E0 = d.energy(u0);
    let Eprev = E0;
    for (let k = 0; k < 60; k++) {
      S.step(d.stableDt());
      const E = d.energy(S.u());
      assert(E <= Eprev * (1 + 1e-13), `energy grew at step ${k}`);
      Eprev = E;
    }
    assertClose(d.mass(S.u()), m0, 1e-14, 0, `p=${p} mass`);
    assert(Eprev < E0, 'upwind dissipates energy of discontinuous data');
  }
});

test('cpuTwin: LSRK4 stage kernel (ping-pong) = lib/core/time/rk.js lsrk4', () => {
  const d = makeDGAdv2D({ N: 4, p: 2, a: [0.6, 1] });
  const u0 = d.interpolate(INITIAL.twin), dt = d.stableDt();
  const S = makeCpuSolver(d, u0); S.run(20, dt);
  const v = Float64Array.from(u0), step = makeStepper('lsrk4', d.nDof);
  for (let k = 0; k < 20; k++) step(v, k * dt, dt, (x, t, out) => d.rhs(x, out));
  let m = 0; for (let g = 0; g < d.nDof; g++) m = Math.max(m, Math.abs(S.u()[g] - v[g]));
  assert(m < 1e-14, `diff ${m}`);
  assertClose(LSRK4.B.length, 5, 0);
});

test('cpuTwin: dt and 2dt, dt = CFL_SAFE·h/((|ax|+|ay|)(2p+1)), are inside the LSRK4 stability region (spectrum of L)', () => {
  for (const a of [[1, 0.5], [1, 1], [-0.3, 0.8]]) for (let p = 1; p <= 3; p++) {
    const d = makeDGAdv2D({ N: 4, p, a }), n = d.nDof;
    const A = new Float64Array(n * n), e = new Float64Array(n), r = new Float64Array(n);
    for (let j = 0; j < n; j++) { e.fill(0); e[j] = 1; d.rhs(e, r); for (let i = 0; i < n; i++) A[i * n + j] = r[i]; }
    const { re, im } = eigGeneral(A, n), dt = d.stableDt(CFL_SAFE);
    for (let i = 0; i < n; i++) {
      assert(re[i] < 1e-10, `eigenvalue with positive real part ${re[i]}`);
      // safety margin: even twice the time step is stable (claimed in chapter 12)
      for (const f of [1, 2]) assert(stabilityAmp('lsrk4', re[i] * f * dt, im[i] * f * dt) <= 1 + 1e-12, `unstable mode a=${a} p=${p} factor ${f}`);
    }
  }
});
