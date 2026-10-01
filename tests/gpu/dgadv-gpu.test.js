/**
 * GPU (WebGPU, f32) vs CPU twin (f64) for the chapter-12 DG-SEM advection solver.
 * Browser-only. If WebGPU is unavailable the tests PASS WITH A NOTE (console.warn),
 * because the course must work everywhere; the note says what was skipped.
 */
import { test, assert } from '../harness.js';
import { getGPU } from '../../lib/gpu/device.js';
import { createGpuSolver, maxAbsDiff } from '../../lib/gpu/dgAdvGPU.js';
import { makeDGAdv2D, makeCpuSolver, INITIAL } from '../../lib/gpu/cpuTwin.js';

/** Results of the last run, for scratch pages that want to print details. */
export const GPU_TEST_LOG = [];

async function compare(N, p, a, nSteps, init) {
  const st = await getGPU();
  if (!st.ok) { const msg = `SKIPPED (no WebGPU): ${st.reason}`; console.warn(msg); GPU_TEST_LOG.push(msg); return null; }
  const d = makeDGAdv2D({ N, p, a }), u0 = d.interpolate(init), dt = d.stableDt();
  const gpu = await createGpuSolver(st.device, d, u0, { dt });
  await gpu.run(nSteps);
  const ug = await gpu.read();
  gpu.destroy();
  const cpu = makeCpuSolver(d, u0); cpu.run(nSteps, dt);
  let umax = 0; for (const v of cpu.u()) umax = Math.max(umax, Math.abs(v));
  const diff = maxAbsDiff(ug, cpu.u());
  const msg = `GPU (${st.info}) N=${N} p=${p} a=${a} steps=${nSteps}: max|GPU−CPU| = ${diff.toExponential(2)} (max|u| = ${umax.toFixed(3)})`;
  console.log(msg); GPU_TEST_LOG.push(msg);
  return { diff, umax };
}

test('GPU DG advection matches CPU twin to f32 accuracy (p = 1..3)', async () => {
  for (let p = 1; p <= 3; p++) {
    const r = await compare(8, p, [1, 0.5], 100, INITIAL.twin);
    if (!r) return;
    assert(r.diff < 2e-5 * Math.max(1, r.umax), `p=${p}: max|GPU−CPU| = ${r.diff}`);
  }
}, { browserOnly: true });

test('GPU DG advection: negative velocity components (upwind side switches), non-power-of-two N', async () => {
  const r = await compare(13, 2, [-0.7, -1.1], 80, INITIAL.blob);
  if (!r) return;
  assert(r.diff < 2e-5 * Math.max(1, r.umax), `max|GPU−CPU| = ${r.diff}`);
}, { browserOnly: true });
