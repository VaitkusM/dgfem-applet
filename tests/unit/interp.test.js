import { test, assert, assertClose, assertRate } from '../harness.js';
import { interpError, interpolantAt, interpNodes } from '../../lib/core/verify/interp1d.js';
import { fitRate } from '../../lib/core/verify/rates.js';

test('piecewise interpolation: L2 rate p+1 (smooth), 1.5 (kink), 0.5 (jump)', () => {
  const Ns = [16, 32, 64, 128], hs = Ns.map((N) => 1 / N);
  const smooth = (x) => Math.sin(2 * Math.PI * x) + x * x;
  const kink = (x) => Math.abs(x - 1 / 3);
  const jump = (x) => (x < 1 / 3 ? 0 : 1);
  for (let p = 1; p <= 4; p++) {
    const Ns2 = p >= 4 ? [4, 8, 16, 32] : Ns;
    assertRate(fitRate(Ns2.map((N) => 1 / N), Ns2.map((N) => interpError(smooth, N, p).L2)), p + 1, `smooth p=${p}`);
    assertRate(fitRate(hs, Ns.map((N) => interpError(kink, N, p).L2)), 1.5, `kink p=${p}`, 0.05, 0.05);
    assertRate(fitRate(hs, Ns.map((N) => interpError(jump, N, p).L2)), 0.5, `jump p=${p}`, 0.05, 0.05);
  }
});

// the functions and meshes of the chapter-0 convergence explorer
const FUNCS = {
  smooth: (x) => Math.sin(2 * Math.PI * x) + x * x,
  runge: (x) => 1 / (1 + 100 * (x - 0.5) ** 2),
  kink: (x) => Math.abs(x - 1 / 3),
  jump: (x) => (x < 1 / 3 ? 0 : 1),
};
const WIDGET_NS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

test('chapter 0 rates in both norms incl. p = 0: kink 1.5 (L2) / 1 (max), jump 0.5 (L2) / none (max)', () => {
  const Ns = [16, 32, 64, 128], hs = Ns.map((N) => 1 / N);
  for (let p = 0; p <= 4; p++) {
    const rate = (f, norm) => fitRate(hs, Ns.map((N) => interpError(f, N, p)[norm]));
    assertRate(rate(FUNCS.kink, 'L2'), Math.min(p + 1, 1.5), `kink L2 p=${p}`, 0.05, 0.05);
    assertRate(rate(FUNCS.kink, 'Linf'), 1, `kink max p=${p}`, 0.05, 0.05);
    assertRate(rate(FUNCS.jump, 'L2'), 0.5, `jump L2 p=${p}`, 0.05, 0.05);
    assertRate(rate(FUNCS.jump, 'Linf'), 0, `jump max p=${p}`, 0.05, 0.05);
    // "no convergence at all": the max error stays O(1) on every mesh
    for (const N of WIDGET_NS) assert(interpError(FUNCS.jump, N, p).Linf > 0.4, `jump max error p=${p} N=${N}`);
    if (p <= 3) {
      assertRate(rate(FUNCS.smooth, 'Linf'), p + 1, `smooth max p=${p}`, 0.1, 0.1);
      assertRate(rate(FUNCS.smooth, 'L2'), p + 1, `smooth L2 p=${p}`, 0.1, 0.1);
    }
  }
});

test('chapter 0 widget: every function/degree/mesh is finite; p = 4 reaches the round-off floor only at N = 1024', () => {
  for (const f of Object.values(FUNCS)) for (let p = 0; p <= 4; p++) for (const N of WIDGET_NS) {
    const e = interpError(f, N, p);
    assert(Number.isFinite(e.L2) && Number.isFinite(e.Linf) && e.L2 >= 0, `p=${p} N=${N}`);
  }
  // smooth function, p = 4: clean slope 5 down to N = 256 (the old plot range), i.e. NO floor there …
  const e = (N, norm) => interpError(FUNCS.smooth, N, 4)[norm];
  assertRate(Math.log2(e(128, 'L2') / e(256, 'L2')), 5, 'slope 128→256', 0.05, 0.05);
  assert(e(256, 'L2') > 5e-14 && e(256, 'Linf') > 1e-13, 'N = 256 is still above round-off');
  // … and the last refinement 512→1024 no longer gains the factor 2⁵ = 32: round-off (max norm ≈ 1e-15)
  assert(e(1024, 'Linf') < 1e-14 && e(1024, 'Linf') > 1e-17, `floor value ${e(1024, 'Linf')}`);
  assert(e(512, 'Linf') / e(1024, 'Linf') < 16, `still converging: ratio ${e(512, 'Linf') / e(1024, 'Linf')}`);
  // the widget excludes errors < 1e-14 from its slope fit: that only ever happens for p = 4 on N ≥ 512
  for (const [name, f] of Object.entries(FUNCS)) for (let p = 0; p <= 4; p++) for (const N of WIDGET_NS) for (const norm of ['L2', 'Linf']) {
    const below = interpError(f, N, p)[norm] < 1e-14;
    assert(below === (name === 'smooth' && p === 4 && N >= 512), `${name} p=${p} N=${N} ${norm}`);
  }
});

test('piecewise interpolant reproduces polynomials of degree ≤ p and interpolates at the nodes', () => {
  for (let p = 0; p <= 4; p++) {
    const f = (x) => (1 + x) ** p - 0.3 * x;
    if (p >= 1) for (const x of [0, 0.123, 0.5, 0.77, 1]) assertClose(interpolantAt(f, 5, p, x), f(x), 1e-13);
    const g = (x) => Math.cos(3 * x), N = 4, nodes = interpNodes(p);
    assert(nodes.length === p + 1);
    for (let e = 0; e < N; e++) for (const t of nodes) { const x = (e + (t + 1) / 2) / N; assertClose(interpolantAt(g, N, p, Math.min(x, 1 - 1e-15)), g(x), 1e-12, 0, `p=${p}`); }
  }
});
