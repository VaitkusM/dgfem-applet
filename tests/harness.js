/**
 * @file Tiny test harness shared by Deno and the browser.
 *
 *  - Under Deno, `test()` forwards to Deno.test, so `deno test` runs everything.
 *  - In the browser, tests are collected in a registry and run by tests/test.html.
 *  - `{browserOnly: true}` marks tests needing DOM/WebGPU (ignored under Deno);
 *    `{denoOnly: true}` marks tests needing Deno APIs (skipped in the browser).
 *
 *  - A test that cannot run in the current environment (e.g. no WebGPU adapter) must call
 *    `skip(reason)` instead of returning early: the browser runners then count it as
 *    SKIPPED, not as passed (Deno has no dynamic skip: there the reason is printed as a warning).
 *  - Runners (tests/runner.js, the appendix page, tests/gpu/gpu-tests.html) execute a registered
 *    test with `runTest(t)`, which classifies the outcome as 'pass' | 'fail' | 'skip'.
 *
 * Test files are named  *.test.js  and must be listed in tests/manifest.js
 * (a Deno test checks that the manifest is complete).
 */

/** @type {{name: string, fn: () => (void|Promise<void>), opts: object}[]} */
export const registry = [];

const isDeno = typeof globalThis.Deno !== 'undefined' && typeof globalThis.Deno.test === 'function';

/**
 * Register a test.
 * @param {string} name
 * @param {() => (void|Promise<void>)} fn
 * @param {{browserOnly?: boolean, denoOnly?: boolean}} [opts]
 */
export function test(name, fn, opts = {}) {
  if (isDeno) {
    const wrapped = async () => {
      try { await fn(); } catch (e) {
        if (!(e instanceof SkipSignal)) throw e;
        console.warn(`SKIPPED ${name}: ${e.message}`);
      }
    };
    globalThis.Deno.test({ name, fn: wrapped, ignore: !!opts.browserOnly, sanitizeOps: false, sanitizeResources: false });
  } else if (!opts.denoOnly) registry.push({ name, fn, opts });
}

/** Thrown by skip(); runners report the test as skipped rather than passed or failed. */
export class SkipSignal extends Error {}

/**
 * Abort the running test and mark it as SKIPPED (not passed), e.g. when WebGPU is unavailable.
 * @param {string} reason shown in the test report
 * @returns {never}
 */
export function skip(reason) {
  throw new SkipSignal(reason);
}

/**
 * Run one registered test and classify the outcome.
 * @param {{name: string, fn: () => (void|Promise<void>)}} t an entry of `registry`
 * @returns {Promise<{status: 'pass'|'fail'|'skip', msg: string, ms: number}>} msg: failure message or skip reason
 */
export async function runTest(t) {
  const t0 = performance.now();
  let status = 'pass', msg = '';
  try { await t.fn(); } catch (e) {
    status = e instanceof SkipSignal ? 'skip' : 'fail';
    msg = e && e.message ? e.message : String(e);
  }
  return { status, msg, ms: performance.now() - t0 };
}

export class AssertionError extends Error {}

/** Assert a condition. @param {unknown} cond @param {string} [msg] */
export function assert(cond, msg = 'assertion failed') {
  if (!cond) throw new AssertionError(msg);
}

/**
 * Assert |a − b| ≤ atol + rtol |b|.
 * @param {number} a actual
 * @param {number} b expected
 * @param {number} [atol=1e-12]
 * @param {number} [rtol=0]
 * @param {string} [msg]
 */
export function assertClose(a, b, atol = 1e-12, rtol = 0, msg = '') {
  if (!(Math.abs(a - b) <= atol + rtol * Math.abs(b)))
    throw new AssertionError(`${msg} expected ${b}, got ${a} (diff ${Math.abs(a - b)})`);
}

/** Element-wise assertClose for arrays. */
export function assertArrayClose(a, b, atol = 1e-12, rtol = 0, msg = '') {
  assert(a.length === b.length, `${msg} length mismatch ${a.length} vs ${b.length}`);
  for (let i = 0; i < a.length; i++) assertClose(a[i], b[i], atol, rtol, `${msg}[${i}]`);
}

/**
 * Assert an observed convergence rate lies in [expected − lo, expected + hi].
 * @param {number} observed
 * @param {number} expected
 * @param {string} [msg]
 * @param {number} [lo=0.2]
 * @param {number} [hi=0.3]
 */
export function assertRate(observed, expected, msg = '', lo = 0.2, hi = 0.3) {
  if (!(observed >= expected - lo && observed <= expected + hi))
    throw new AssertionError(`${msg} rate ${observed.toFixed(3)} not in [${expected - lo}, ${expected + hi}]`);
}
