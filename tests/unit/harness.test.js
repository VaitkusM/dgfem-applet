/**
 * Tests for the test harness itself (tests/harness.js): a test that cannot run must be
 * reported as SKIPPED, never as passed (the GPU tests without WebGPU used to "pass with a note").
 */
import { test, assert, skip, runTest, SkipSignal, AssertionError, assertClose, assertRate } from '../harness.js';

test('harness: runTest classifies pass / fail / skip', async () => {
  const pass = await runTest({ name: 'p', fn: () => {} });
  assert(pass.status === 'pass' && pass.msg === '' && pass.ms >= 0, JSON.stringify(pass));
  const fail = await runTest({ name: 'f', fn: () => assert(false, 'nope') });
  assert(fail.status === 'fail' && fail.msg === 'nope', JSON.stringify(fail));
  const failAsync = await runTest({ name: 'fa', fn: async () => { await null; throw new Error('late'); } });
  assert(failAsync.status === 'fail' && failAsync.msg === 'late', JSON.stringify(failAsync));
  const skipped = await runTest({ name: 's', fn: async () => { await null; skip('no WebGPU: test'); } });
  assert(skipped.status === 'skip' && skipped.msg === 'no WebGPU: test', JSON.stringify(skipped));
  const thrownString = await runTest({ name: 't', fn: () => { throw 'plain string'; } });
  assert(thrownString.status === 'fail' && thrownString.msg === 'plain string', JSON.stringify(thrownString));
});

test('harness: skip() throws a SkipSignal that is not an assertion failure', () => {
  let e = null;
  try { skip('why'); } catch (x) { e = x; }
  assert(e instanceof SkipSignal && !(e instanceof AssertionError) && e.message === 'why', 'SkipSignal');
});

test('harness: assertClose and assertRate reject what they should', () => {
  const throws = (fn) => { try { fn(); return false; } catch (e) { return e instanceof AssertionError; } };
  assert(!throws(() => assertClose(1, 1 + 1e-13)) && throws(() => assertClose(1, 1.1)) && throws(() => assertClose(NaN, 1)), 'assertClose');
  assert(!throws(() => assertRate(2.1, 2)) && throws(() => assertRate(1.7, 2)) && throws(() => assertRate(2.4, 2)) && throws(() => assertRate(NaN, 2)), 'assertRate');
});
