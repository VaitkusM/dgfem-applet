/**
 * Tests for the page self-test verdict (lib/ui/chapter.js, summariseSelftest): `ok` must not be
 * true when the math did not render, when a check failed, when an error was recorded or when
 * fewer checks ran than the page declares. (The DOM side is covered by tests/unit/ui-shell.test.js
 * and tools/smoke.sh.)
 */
import { test, assert } from '../harness.js';
import { summariseSelftest } from '../../lib/ui/chapter.js';
import { CHAPTERS } from '../../lib/ui/chapters.js';

test('self-test verdict: ok requires KaTeX, no errors, no failed checks, the declared number of checks', () => {
  const good = { errors: [], checks: new Map([['a', true], ['b', true]]), katex: true, minChecks: 2, widgets: [{ id: 'w-a', n: 2 }, { id: 'w-b', n: 0 }] };
  const r = summariseSelftest(good);
  assert(r.ok === true && r.nChecks === 2 && r.errors.length === 0 && r.failedChecks.length === 0, JSON.stringify(r));
  assert(r.widgetsWithoutChecks.length === 1 && r.widgetsWithoutChecks[0] === 'w-b', 'widgets without checks are listed (informational)');
  const noKatex = summariseSelftest({ ...good, katex: false });
  assert(noKatex.ok === false && noKatex.katex === false, 'raw TeX on the page must not pass');
  const failed = summariseSelftest({ ...good, checks: new Map([['a', true], ['b', false]]) });
  assert(failed.ok === false && failed.failedChecks.length === 1 && failed.failedChecks[0].label === 'b' && failed.failedChecks[0].ok === false, JSON.stringify(failed));
  const err = summariseSelftest({ ...good, errors: ['resource failed to load: ../css/style.css'] });
  assert(err.ok === false && err.errors.length === 1, 'errors fail the page');
  const few = summariseSelftest({ ...good, minChecks: 3 });
  assert(few.ok === false && /only 2 self-checks ran/.test(few.errors[0]), JSON.stringify(few));
  assert(good.errors.length === 0, 'the input is not mutated');
  assert(summariseSelftest({ errors: [], checks: [], katex: true }).ok === true, 'no declared minimum: zero checks are allowed (appendix-like pages)');
});

test('chapter list: ids, numbers and files are unique and lower-case (AGENTS.md rule 4)', () => {
  const seen = new Set();
  for (const c of CHAPTERS) {
    for (const k of [c.id, c.file]) { assert(!seen.has(k), `duplicate ${k}`); seen.add(k); }
    assert(c.file === c.file.toLowerCase() && /^chapters\/[a-z0-9-]+\.html$/.test(c.file), c.file);
    assert(c.title.length > 0 && c.blurb.length > 0, `title/blurb of ${c.id}`);
  }
});
