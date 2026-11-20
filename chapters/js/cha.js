/**
 * @file Appendix — verification & glossary.
 *
 * Widgets:
 *  - w-tests    : runs the project's test suite (tests/manifest.js) in this browser on demand
 *  - w-glossary : the complete glossary (lib/ui/glossary.js), alphabetical
 */
import { initChapter, mount, renderMath, selfCheck } from '../../lib/ui/chapter.js';
import { button, buttonRow, h } from '../../lib/ui/controls.js';
import { GLOSSARY } from '../../lib/ui/glossary.js';
import { TEST_FILES } from '../../tests/manifest.js';
import { registry, runTest } from '../../tests/harness.js';

function testsWidget(fig) {
  const row = buttonRow(fig);
  const status = h('span', 'readout', `${TEST_FILES.length} test files. Running all tests takes roughly half a minute.`);
  const table = h('table', 'data text');
  table.innerHTML = '<tr><th>test</th><th>result</th><th>ms</th></tr>';
  let ran = false;
  const run = async () => {
    if (ran) return;
    ran = true;
    runBtn.el.disabled = true;
    const before = registry.length;
    let importFailed = 0;
    for (const f of TEST_FILES) {
      try { await import(`../../tests/${f}`); } catch (_e) { importFailed++; const tr = document.createElement('tr'); tr.innerHTML = '<td></td><td class="bad">import failed</td><td></td>'; tr.children[0].textContent = f; table.appendChild(tr); }
    }
    const tests = registry.slice(before);
    let pass = 0, fail = importFailed, skipped = 0;
    for (const [i, t] of tests.entries()) {
      status.textContent = `running ${i + 1}/${tests.length}: ${t.name}`;
      await new Promise((r) => setTimeout(r, 0));
      const r = await runTest(t);
      if (r.status === 'pass') pass++; else if (r.status === 'skip') skipped++; else fail++;
      const tr = document.createElement('tr');
      tr.innerHTML = `<td></td><td class="${{ pass: 'good', fail: 'bad', skip: 'skip' }[r.status]}">${{ pass: 'pass', fail: 'FAIL', skip: 'SKIPPED' }[r.status]}</td><td>${r.ms.toFixed(0)}</td>`;
      tr.children[0].textContent = t.name + (r.msg ? ` — ${r.msg}` : '');
      table.appendChild(tr);
    }
    const skipNote = skipped ? ` ${skipped} skipped (could not run in this browser, e.g. the GPU tests without WebGPU — not verified here).` : '';
    status.innerHTML = (fail === 0 ? `<b class="status-good">${skipped ? '' : 'All '}${pass} tests passed in this browser.</b>` : `<b class="status-bad">${fail} failed</b>, ${pass} passed.`) + skipNote;
    selfCheck('tests widget: suite ran without failures', fail === 0);
  };
  const runBtn = button(row, { label: '▶ Run the test suite here', primary: true, onClick: run });
  row.appendChild(status);
  const wrap = h('div');
  wrap.style.overflowX = 'auto';
  wrap.appendChild(table);
  fig.insertBefore(wrap, fig.querySelector('figcaption'));
  fig.insertBefore(row, wrap);
}

function glossaryWidget(el) {
  const keys = Object.keys(GLOSSARY).sort((a, b) => GLOSSARY[a].term.replace(/[^A-Za-z]/g, '').localeCompare(GLOSSARY[b].term.replace(/[^A-Za-z]/g, '')));
  const dl = h('dl', 'glossary-list');
  for (const k of keys) {
    const dt = h('dt', null, GLOSSARY[k].term);
    dt.id = `g-${k}`;
    dl.append(dt, h('dd', null, GLOSSARY[k].def));
  }
  el.appendChild(dl);
  renderMath(el);
  selfCheck('glossary entries', keys.length > 50);
}

initChapter(() => {
  mount('w-tests', testsWidget);
  mount('glossary', glossaryWidget);
});
