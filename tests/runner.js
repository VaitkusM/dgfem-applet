/**
 * @file Browser test runner: imports all test files, runs the registered
 * tests sequentially and reports results in a table. Also writes a JSON
 * summary {passed, failed, skipped, failures, skippedTests} into <pre id="summary">
 * (used by headless Chrome in CI/smoke tests). Tests that called skip() — e.g. the GPU
 * tests in a browser without WebGPU — are counted as skipped, never as passed.
 */
import { TEST_FILES } from './manifest.js';
import { registry, runTest } from './harness.js';

const params = new URLSearchParams(location.search);
const grep = params.get('grep');
const tbody = document.querySelector('#results tbody');
const status = document.getElementById('status');
/** passed + failed + skipped = number of tests; a skipped test did NOT verify anything. */
const summary = { passed: 0, failed: 0, skipped: 0, failures: [], skippedTests: [] };

const loadErrors = [];
for (const f of TEST_FILES) {
  try { await import(`./${f}`); } catch (e) { loadErrors.push(`${f}: ${e.message}`); }
}
for (const e of loadErrors) {
  summary.failed++; summary.failures.push(e);
  const tr = document.createElement('tr');
  tr.innerHTML = '<td class="name">import</td><td class="fail">FAIL</td><td></td><td class="msg"></td>';
  tr.children[3].textContent = e;
  tbody.appendChild(tr);
}
const tests = registry.filter((t) => !grep || t.name.toLowerCase().includes(grep.toLowerCase()));
let i = 0;
for (const t of tests) {
  status.textContent = `Running ${++i}/${tests.length}: ${t.name}`;
  await new Promise((r) => setTimeout(r, 0)); // let the page repaint
  const r = await runTest(t);
  if (r.status === 'pass') summary.passed++;
  else if (r.status === 'skip') { summary.skipped++; summary.skippedTests.push(`${t.name}: ${r.msg}`); }
  else { summary.failed++; summary.failures.push(`${t.name}: ${r.msg}`); }
  const tr = document.createElement('tr');
  const label = { pass: 'ok', fail: 'FAIL', skip: 'SKIPPED' }[r.status];
  tr.innerHTML = `<td class="name"></td><td class="${{ pass: 'ok', fail: 'fail', skip: 'skip' }[r.status]}">${label}</td><td>${r.ms.toFixed(0)}</td><td class="msg"></td>`;
  tr.children[0].textContent = t.name;
  tr.children[3].textContent = r.msg;
  tbody.appendChild(tr);
}
const skipNote = summary.skipped ? ` <span class="status-skip">${summary.skipped} skipped (not verified in this browser — see the message column).</span>` : '';
status.innerHTML = (summary.failed === 0
  ? `<b class="status-good">${summary.skipped ? '' : 'All '}${summary.passed} tests passed.</b>`
  : `<b class="status-bad">${summary.failed} failed</b>, ${summary.passed} passed.`) + skipNote;
const pre = document.getElementById('summary');
pre.textContent = JSON.stringify(summary);
pre.hidden = false;
