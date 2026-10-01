/**
 * @file Browser test runner: imports all test files, runs the registered
 * tests sequentially and reports results in a table. Also writes a JSON
 * summary into <pre id="summary"> (used by headless Chrome in CI/smoke tests).
 */
import { TEST_FILES } from './manifest.js';
import { registry } from './harness.js';

const params = new URLSearchParams(location.search);
const grep = params.get('grep');
const tbody = document.querySelector('#results tbody');
const status = document.getElementById('status');
const summary = { passed: 0, failed: 0, failures: [] };

const loadErrors = [];
for (const f of TEST_FILES) {
  try { await import(`./${f}`); } catch (e) { loadErrors.push(`${f}: ${e.message}`); }
}
for (const e of loadErrors) {
  summary.failed++; summary.failures.push(e);
  tbody.insertAdjacentHTML('beforeend', `<tr><td class="name">import</td><td class="fail">FAIL</td><td></td><td class="msg">${e}</td></tr>`);
}
const tests = registry.filter((t) => !grep || t.name.toLowerCase().includes(grep.toLowerCase()));
let i = 0;
for (const t of tests) {
  status.textContent = `Running ${++i}/${tests.length}: ${t.name}`;
  await new Promise((r) => setTimeout(r, 0)); // let the page repaint
  const t0 = performance.now();
  let ok = true, msg = '';
  try { await t.fn(); } catch (e) { ok = false; msg = e && e.message ? e.message : String(e); }
  const ms = (performance.now() - t0).toFixed(0);
  if (ok) summary.passed++; else { summary.failed++; summary.failures.push(`${t.name}: ${msg}`); }
  const tr = document.createElement('tr');
  tr.innerHTML = `<td class="name"></td><td class="${ok ? 'ok' : 'fail'}">${ok ? 'ok' : 'FAIL'}</td><td>${ms}</td><td class="msg"></td>`;
  tr.children[0].textContent = t.name;
  tr.children[3].textContent = msg;
  tbody.appendChild(tr);
}
status.innerHTML = summary.failed === 0
  ? `<b class="status-good">All ${summary.passed} tests passed.</b>`
  : `<b class="status-bad">${summary.failed} failed</b>, ${summary.passed} passed.`;
const pre = document.getElementById('summary');
pre.textContent = JSON.stringify(summary);
pre.hidden = false;
