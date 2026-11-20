/**
 * @file Landing page: chapter cards (generated from lib/ui/chapters.js), math and glossary pop-ups.
 *
 * Shares the shell helpers of chapter pages (lib/ui/chapter.js): KaTeX fast-fail with a visible
 * notice, keyboard-accessible glossary pop-ups, the favicon and the `?selftest=1` verdict
 * (<pre id="selftest-result">, see chapter.js) so that tools/smoke.sh also covers this page.
 */
import { CHAPTERS } from './chapters.js';
import { renderMath, renderMathWhenAvailable, katexReady, katexNotice, installGlossary, installFavicon, selftestSession, selfCheck, reportError } from './chapter.js';

const session = selftestSession({ minChecks: 1 });
installFavicon();
try {
  const cards = document.getElementById('cards');
  cards.innerHTML = CHAPTERS.map((c) => `
  <a class="card" href="${c.file}">
    <div class="num">${c.num === 'A' ? 'APPENDIX' : `CHAPTER ${c.num}`}</div>
    <h3>${c.title}</h3>
    <p>${c.blurb}</p>
  </a>`).join('');
  selfCheck('landing: one card per chapter', cards.querySelectorAll('a.card').length === CHAPTERS.length && CHAPTERS.length > 0);
} catch (err) { reportError('landing', err); }

const ok = await katexReady();
session.setKatex(ok);
if (ok) renderMath(document.body); else { katexNotice(document.querySelector('main.content')); renderMathWhenAvailable(document.querySelector('main.content')); }
installGlossary();
if (session.enabled) {
  await new Promise((r) => setTimeout(r, 300));
  session.publish();
}
