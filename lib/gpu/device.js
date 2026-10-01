/**
 * @file WebGPU feature detection and device setup with graceful fallback.
 *
 * WebGPU is reached through `navigator.gpu` → `requestAdapter()` (a physical
 * GPU + driver) → `adapter.requestDevice()` (a logical connection with its own
 * queue and resource limits). Each step can fail:
 *  - `navigator.gpu` missing: browser without WebGPU (or a non-secure context —
 *    WebGPU needs https:// or http://localhost);
 *  - `requestAdapter()` returns null: WebGPU exists but no usable GPU (blocklisted
 *    driver, headless browser started with --disable-gpu, remote desktop, …);
 *  - `requestDevice()` rejects: limits or features unavailable.
 * `getGPU()` never throws; it resolves to `{ok: false, reason}` with a
 * human-readable message so that pages can fall back to the CPU.
 */

let cached = null;

/**
 * @typedef {{ok: true, device: GPUDevice, adapter: GPUAdapter, info: string, limits: Record<string, number>}
 *          | {ok: false, reason: string}} GPUStatus
 */

/**
 * Request (once) a WebGPU device. Subsequent calls return the same promise.
 * @param {{powerPreference?: 'high-performance'|'low-power'}} [o]
 * @returns {Promise<GPUStatus>}
 */
export function getGPU(o = {}) {
  if (!cached) cached = init(o);
  return cached;
}

async function init(o) {
  try {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      const secure = typeof isSecureContext === 'undefined' || isSecureContext;
      return { ok: false, reason: secure ? 'This browser does not expose WebGPU (navigator.gpu is undefined).'
        : 'WebGPU requires a secure context (https:// or http://localhost).' };
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: o.powerPreference ?? 'high-performance' });
    if (!adapter) return { ok: false, reason: 'WebGPU is present but no GPU adapter is available (blocked driver, or GPU disabled).' };
    const device = await adapter.requestDevice();
    const ai = adapter.info || {};
    const info = [ai.vendor, ai.architecture, ai.device, ai.description].filter(Boolean).join(' · ') || 'unknown adapter';
    const L = device.limits;
    const limits = {};
    for (const k of ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeWorkgroupsPerDimension',
      'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupStorageSize', 'maxStorageBuffersPerShaderStage']) limits[k] = L[k];
    device.lost.then((d) => { console.warn('WebGPU device lost:', d.message); cached = null; });
    return { ok: true, device, adapter, info, limits };
  } catch (err) {
    return { ok: false, reason: `WebGPU initialisation failed: ${err && err.message ? err.message : err}` };
  }
}

/**
 * Compile a WGSL module and surface compilation errors as an exception with line numbers.
 * @param {GPUDevice} device
 * @param {string} code
 * @param {string} label
 * @returns {Promise<GPUShaderModule>}
 */
export async function compileWGSL(device, code, label) {
  const mod = device.createShaderModule({ code, label });
  if (mod.getCompilationInfo) {
    const info = await mod.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error');
    if (errs.length) throw new Error(`WGSL ${label}: ${errs.map((m) => `line ${m.lineNum}: ${m.message}`).join('; ')}`);
  }
  return mod;
}
