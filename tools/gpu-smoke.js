/**
 * @file GPU smoke test (run by `tools/smoke.sh --gpu`, needs Deno and a machine with a real GPU).
 *
 * Starts headless Chrome with WebGPU enabled, drives it over the Chrome DevTools Protocol
 * (plain WebSocket, no dependencies) and checks, in real time (GPU callbacks are not covered
 * by Chrome's --virtual-time-budget, which the CPU smoke test uses):
 *   1. tests/gpu/gpu-tests.html: every GPU test PASSED — a skipped test (no WebGPU) is a failure here;
 *   2. chapters/12-gpu.html?selftest=1: verdict ok, WebGPU was really used, and the checks
 *      "GPU matches CPU twin" and "starts at t = 0" were recorded;
 *   3. device loss: after destroying the device the live widget continues on the CPU twin,
 *      and Reset gets the GPU back.
 *
 * Usage: deno run --allow-run --allow-net --allow-read --allow-write --allow-env tools/gpu-smoke.js <base-url>
 * Environment: CHROME (browser binary), CHROME_FLAGS (extra flags), CHROME_GPU_FLAGS (override the WebGPU flags).
 * Exit status 0 = all good, 1 = a check failed, 2 = could not run (no Chrome, or no WebGPU adapter so
 * that the GPU tests were skipped) — every non-zero status is a failure of `tools/smoke.sh --gpu`.
 */
const base = (Deno.args[0] ?? 'http://localhost:8765').replace(/\/$/, '');
const CHROME = Deno.env.get('CHROME') ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const GPU_FLAGS = (Deno.env.get('CHROME_GPU_FLAGS') ?? (Deno.build.os === 'darwin'
  ? '--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU --use-angle=metal'
  : '--enable-unsafe-webgpu --enable-features=Vulkan,WebGPU --use-angle=vulkan')).split(/\s+/).filter(Boolean);
const EXTRA = (Deno.env.get('CHROME_FLAGS') ?? '').split(/\s+/).filter(Boolean);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Launch Chrome and return a minimal CDP client for one page. */
async function launch() {
  const profile = await Deno.makeTempDir({ prefix: 'gpu-smoke-' });
  const port = 9400 + Math.floor(Math.random() * 500);
  const proc = new Deno.Command(CHROME, {
    args: ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check', ...GPU_FLAGS, ...EXTRA, 'about:blank'],
    stdout: 'null', stderr: 'null',
  }).spawn();
  let ver = null;
  for (let i = 0; i < 100 && !ver; i++) { try { ver = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json(); } catch (_e) { await sleep(100); } }
  if (!ver) { try { proc.kill(); } catch (_e) { /* not running */ } throw new Error(`Chrome did not start (${CHROME})`); }
  const ws = new WebSocket(ver.webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0;
  const pending = new Map(), events = [];
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pending.has(d.id)) { const { res, rej } = pending.get(d.id); pending.delete(d.id); if (d.error) rej(new Error(JSON.stringify(d.error))); else res(d.result); } else events.push(d);
  };
  const send = (method, params = {}, sessionId) => new Promise((res, rej) => { const i = ++id; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params, sessionId })); });
  const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
  await send('Page.enable', {}, sessionId); await send('Runtime.enable', {}, sessionId);
  const ev = async (expression) => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, sessionId);
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
    return r.result.value;
  };
  return {
    ev,
    goto: (url) => send('Page.navigate', { url }, sessionId),
    async waitFor(expr, timeout = 60000) { const t0 = Date.now(); while (Date.now() - t0 < timeout) { try { if (await ev(expr)) return true; } catch (_e) { /* navigating */ } await sleep(150); } return false; },
    async close() {
      try { await send('Browser.close'); } catch (_e) { /* already gone */ }
      try { ws.close(); } catch (_e) { /* closed */ }
      try { proc.kill(); } catch (_e) { /* exited */ }
      await proc.status;
      try { await Deno.remove(profile, { recursive: true }); } catch (_e) { /* still locked: leave the temp dir */ }
    },
  };
}

let failed = 0, noAdapter = false;
const check = (ok, label, detail = '') => { console.log(`${ok ? 'ok   ' : 'FAIL '} ${label}${detail ? `  ${detail}` : ''}`); if (!ok) failed++; };

let B;
try { B = await launch(); } catch (e) { console.log(`FAIL  ${e.message}`); Deno.exit(2); }
try {
  // ---- 0. is there a WebGPU adapter at all? --------------------------------------
  await B.goto(`${base}/tests/gpu/gpu-tests.html`);
  if (!await B.waitFor(`!document.getElementById('summary').hidden`, 120000)) { check(false, 'tests/gpu/gpu-tests.html', '(no summary within 120 s)'); throw new Error('stop'); }
  let sum = JSON.parse(await B.ev(`document.getElementById('summary').textContent`));
  if (sum.skipped > 0) { // adapter acquisition can fail transiently (e.g. another headless Chrome is just releasing the GPU): retry once
    await sleep(4000);
    await B.goto(`${base}/tests/gpu/gpu-tests.html?retry=1`);
    if (await B.waitFor(`!document.getElementById('summary').hidden && location.search.includes('retry')`, 120000)) sum = JSON.parse(await B.ev(`document.getElementById('summary').textContent`));
  }
  for (const l of sum.log) console.log(`      ${l}`);
  check(sum.failed === 0 && sum.skipped === 0 && sum.passed > 0, 'tests/gpu/gpu-tests.html',
    `passed ${sum.passed}, failed ${sum.failed}, SKIPPED ${sum.skipped} ${[...sum.failures, ...sum.skippedTests].join(' | ')}`);
  if (sum.skipped > 0) { console.log('      (no WebGPU adapter in this Chrome: GPU tests were SKIPPED, which this mode treats as a failure)'); noAdapter = true; throw new Error('stop'); }

  // ---- 1. chapter 12 with a real GPU ------------------------------------------------
  await B.goto(`${base}/chapters/12-gpu.html?selftest=1`);
  if (!await B.waitFor(`!!document.getElementById('selftest-result')`, 60000)) { check(false, 'chapters/12-gpu.html', '(no self-test verdict)'); throw new Error('stop'); }
  await sleep(500);
  const v = JSON.parse(await B.ev(`document.getElementById('selftest-result').textContent`));
  check(v.ok === true, 'chapters/12-gpu.html?selftest=1', JSON.stringify(v));
  const RO = `document.querySelector('#w-live .readout').innerText`, ST = `document.querySelector('#w-live .gpu-status').innerText`;
  const status = await B.ev(ST), ro = await B.ev(RO);
  check(/WebGPU available/.test(status), 'chapter 12 uses WebGPU', status.slice(0, 80));
  check(/max\|GPU−CPU\| after \d+ steps/.test(ro), 'chapter 12 verified the GPU against the CPU twin', (ro.match(/= \S+e-\d+/) ?? [''])[0]);
  check(/t = 0\.000\s+steps = 0\b/.test(ro), 'chapter 12 starts at t = 0 (verify/benchmark did not advance the display)', ro.split('\n')[1]);
  const click = (label) => B.ev(`[...document.querySelectorAll('#w-live button')].find((b) => b.textContent.includes('${label}')).click()`);
  await B.ev(`document.getElementById('w-live').scrollIntoView({ behavior: 'instant', block: 'center' })`); // the widget only runs while visible
  await sleep(300);
  await click('Benchmark'); await sleep(3000);
  check(/t = 0\.000\s+steps = 0\b/.test(await B.ev(RO)), 'Benchmark leaves the displayed state at t = 0');
  await click('Run'); await sleep(1500); await click('Pause'); await sleep(500);
  const steps1 = Number((await B.ev(RO)).match(/steps = (\d+)/)[1]);
  check(steps1 > 0, 'Run advances the GPU solver', `steps = ${steps1}`);

  // ---- 2. device loss → CPU twin → Reset re-acquires the GPU -------------------------
  await B.ev(`(async () => { const m = await import('${base}/lib/gpu/device.js'); (await m.getGPU()).device.destroy(); })()`);
  await sleep(800);
  await click('Run'); await sleep(1500);
  const st2 = await B.ev(ST), ro2 = await B.ev(RO);
  const steps2 = Number((ro2.match(/steps = (\d+)/) ?? [0, -1])[1]);
  check(/CPU twin/.test(st2) && !/WebGPU available/.test(st2), 'device loss: banner announces the CPU twin', st2.slice(0, 90));
  check(steps2 > steps1 && !/error/i.test(ro2), 'device loss: the simulation continues on the CPU from the same state', `steps ${steps1} → ${steps2}`);
  await click('Pause'); await click('Reset'); await sleep(4000);
  const st3 = await B.ev(ST), ro3 = await B.ev(RO);
  check(/WebGPU available/.test(st3) && /t = 0\.000\s+steps = 0\b/.test(ro3) && /max\|GPU−CPU\|/.test(ro3), 'device loss: Reset gets a new device and verifies again', st3.slice(0, 60));
  const v2 = JSON.parse(await B.ev(`document.getElementById('selftest-result').textContent`));
  check(v2.ok === true, 'no errors recorded during loss and recovery', JSON.stringify(v2.errors));
} catch (e) { if (e.message !== 'stop') { console.log(`FAIL  ${e.message}`); failed++; } }
await B.close();
Deno.exit(noAdapter ? 2 : failed ? 1 : 0);
