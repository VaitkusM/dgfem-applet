/**
 * @file WebGPU solver for 2D DG-SEM linear advection (the GPU half of lib/gpu/cpuTwin.js).
 *
 * Host-side responsibilities (everything that is *not* in the kernels):
 *  - BUFFERS (all f32, layouts identical to cpuTwin.js):
 *      u[0], u[1]  nDof   solution, ping-pong pair: a stage reads u[cur], the RK kernel writes u[1−cur]
 *      rhs         nDof   L(u) of the current stage
 *      res         nDof   low-storage RK register k
 *      flux        nFace  one upwind flux per face node (race-free face buffer)
 *      params      48 B   uniform: N, sizes, velocity, 2/h, 1/w_0, dt
 *      stage       5×256 B uniform slots with the LSRK4 (A_s, B_s) coefficients
 *      staging     nDof   MAP_READ buffer for read-back
 *  - BIND GROUPS: which buffer is attached to which binding of each kernel; built once
 *    for both ping-pong parities (and each RK stage), so encoding a step only switches bind groups.
 *  - ENCODING: one LSRK4 step = 5 stages × 4 dispatches (volume, surface, lift, rkStage).
 *    Many steps are recorded into ONE compute pass / command buffer and submitted at once:
 *    the CPU never waits for the GPU between steps, and no data crosses the PCIe bus.
 *    (WebGPU orders dispatches in a pass and makes storage writes of one dispatch visible to the next.)
 *  - TIMING: wall-clock time from submit to `queue.onSubmittedWorkDone()` over a batch of steps
 *    (includes submission overhead; a batch must be long enough for that to be negligible).
 *  - READ-BACK: copy u[cur] → staging, `mapAsync`, copy out. Slow (a round trip that stalls the
 *    pipeline), so the live widget does it only for display/verification, not every step.
 *
 * Why ping-pong? The RK kernel could update u in place (each thread touches only u[g]), but writing
 * the new stage into the other buffer keeps "inputs of this stage" and "outputs of this stage"
 * strictly separate — the pattern you need as soon as a kernel reads neighbours of what it writes
 * (e.g. a fused surface+update kernel), and it is free here because the parity is known on the host.
 */
import { makeKernelSources, WORKGROUP_SIZE, PARAMS_BYTES } from './kernels.js';
import { compileWGSL } from './device.js';
import { LSRK4 } from './cpuTwin.js';

const STAGE_STRIDE = 256; // minUniformBufferOffsetAlignment is ≤ 256 on every WebGPU implementation

/**
 * Create a GPU solver for a discretisation built by cpuTwin.makeDGAdv2D.
 * @param {GPUDevice} device
 * @param {ReturnType<import('./cpuTwin.js').makeDGAdv2D>} disc
 * @param {ArrayLike<number>} u0 initial nodal values (length disc.nDof)
 * @param {{dt: number}} o time step
 * @returns {Promise<{dt: number, t: number, steps: number, nDispatchPerStep: number,
 *   run: (nSteps: number) => Promise<number>, read: () => Promise<Float32Array>, setState: (u: ArrayLike<number>) => void, destroy: () => void}>}
 *   run resolves to the elapsed wall-clock milliseconds of the batch.
 */
export async function createGpuSolver(device, disc, u0, o) {
  const { nDof, nFace, nElem } = disc;
  const groupsDof = Math.ceil(nDof / WORKGROUP_SIZE), groupsFace = Math.ceil(nFace / WORKGROUP_SIZE);
  const maxGroups = device.limits.maxComputeWorkgroupsPerDimension;
  if (groupsDof > maxGroups || groupsFace > maxGroups) throw new Error(`problem too large for a 1D dispatch (${groupsFace} > ${maxGroups} workgroups)`);
  const bytes = nDof * 4;
  const S = GPUBufferUsage.STORAGE, CS = GPUBufferUsage.COPY_SRC, CD = GPUBufferUsage.COPY_DST;
  const buf = (size, usage, label) => device.createBuffer({ size, usage, label });
  const u = [buf(bytes, S | CS | CD, 'u0'), buf(bytes, S | CS | CD, 'u1')];
  const rhs = buf(bytes, S, 'rhs');
  const res = buf(bytes, S | CD, 'res');
  const flux = buf(nFace * 4, S, 'flux');
  const params = buf(PARAMS_BYTES, GPUBufferUsage.UNIFORM | CD, 'params');
  const stage = buf(5 * STAGE_STRIDE, GPUBufferUsage.UNIFORM | CD, 'stage');
  const staging = buf(bytes, GPUBufferUsage.MAP_READ | CD, 'staging');

  // ---- uniforms -------------------------------------------------------
  const writeParams = (dt) => {
    const ab = new ArrayBuffer(PARAMS_BYTES), U = new Uint32Array(ab), F = new Float32Array(ab);
    U[0] = disc.N; U[1] = nElem; U[2] = nDof; U[3] = nFace;
    F[4] = disc.ax; F[5] = disc.ay; F[6] = disc.rx; F[7] = disc.ry; F[8] = disc.lift; F[9] = dt;
    device.queue.writeBuffer(params, 0, ab);
  };
  {
    const st = new Float32Array(5 * STAGE_STRIDE / 4);
    for (let s = 0; s < 5; s++) { st[s * STAGE_STRIDE / 4] = LSRK4.A[s]; st[s * STAGE_STRIDE / 4 + 1] = LSRK4.B[s]; }
    device.queue.writeBuffer(stage, 0, st);
  }

  // ---- pipelines --------------------------------------------------------
  const src = makeKernelSources(disc.p, disc.D);
  const pipe = {};
  for (const k of ['volume', 'surface', 'lift', 'rkStage']) {
    const module = await compileWGSL(device, src[k], `${k} p=${disc.p}`);
    pipe[k] = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: k });
  }

  // ---- bind groups (for both ping-pong parities) -------------------------
  const bg = (k, entries) => device.createBindGroup({
    layout: pipe[k].getBindGroupLayout(0),
    entries: entries.map(([binding, buffer, offset = 0, size]) => ({ binding, resource: { buffer, offset, size } })),
  });
  const B = [0, 1].map((c) => ({
    volume: bg('volume', [[0, params], [1, u[c]], [2, rhs]]),
    surface: bg('surface', [[0, params], [1, u[c]], [3, flux]]),
    lift: bg('lift', [[0, params], [1, u[c]], [2, rhs], [3, flux]]),
    rk: [0, 1, 2, 3, 4].map((s) => bg('rkStage', [[0, params], [1, u[c]], [2, rhs], [4, res], [5, u[1 - c]], [6, stage, s * STAGE_STRIDE, 16]])),
  }));

  let cur = 0;
  const solver = {
    dt: o.dt, t: 0, steps: 0, nDispatchPerStep: 20,
    /** Upload a new state (Float64 → Float32), reset time and the RK register. */
    setState(v) {
      cur = 0;
      device.queue.writeBuffer(u[0], 0, Float32Array.from(v));
      device.queue.writeBuffer(res, 0, new Float32Array(nDof));
      solver.t = 0; solver.steps = 0;
    },
    /** Encode and submit nSteps LSRK4 steps; resolves (with elapsed ms) when the GPU has finished. */
    async run(nSteps) {
      const t0 = performance.now();
      const CHUNK = 64; // steps per command buffer (keeps command buffers reasonably small)
      for (let done = 0; done < nSteps; done += CHUNK) {
        const m = Math.min(CHUNK, nSteps - done);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        for (let k = 0; k < m; k++) {
          // A_0 = 0, so the first stage overwrites res: no need to clear it between steps.
          for (let s = 0; s < 5; s++) {
            const G = B[cur];
            pass.setPipeline(pipe.volume); pass.setBindGroup(0, G.volume); pass.dispatchWorkgroups(groupsDof);
            pass.setPipeline(pipe.surface); pass.setBindGroup(0, G.surface); pass.dispatchWorkgroups(groupsFace);
            pass.setPipeline(pipe.lift); pass.setBindGroup(0, G.lift); pass.dispatchWorkgroups(groupsDof);
            pass.setPipeline(pipe.rkStage); pass.setBindGroup(0, G.rk[s]); pass.dispatchWorkgroups(groupsDof);
            cur = 1 - cur; // ping-pong
          }
        }
        pass.end();
        device.queue.submit([enc.finish()]);
      }
      await device.queue.onSubmittedWorkDone();
      solver.steps += nSteps; solver.t += nSteps * solver.dt;
      return performance.now() - t0;
    },
    /** Read the current solution back to the CPU (slow: use sparingly). */
    async read() {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(u[cur], 0, staging, 0, bytes);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      return out;
    },
    destroy() { for (const b of [...u, rhs, res, flux, params, stage, staging]) b.destroy(); },
  };
  writeParams(o.dt);
  solver.setState(u0);
  return solver;
}

/**
 * max_g |a[g] − b[g]| (e.g. GPU f32 result vs CPU f64 twin).
 * @param {ArrayLike<number>} a @param {ArrayLike<number>} b
 */
export function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (Number.isNaN(d)) return NaN; // a blown-up or uninitialised value must not hide
    if (d > m) m = d;
  }
  return m;
}
