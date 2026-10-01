/**
 * @file Chapter page shell: sidebar navigation, on-page table of contents,
 *       KaTeX rendering, glossary pop-ups, prev/next links and a self-test
 *       mode for automated smoke testing.
 *
 * Usage in a chapter script (chapters/js/chNN.js):
 *
 *   import { initChapter } from '../../lib/ui/chapter.js';
 *   initChapter(async () => {
 *     mountMyWidget(document.getElementById('w-something'));
 *   });
 *
 * The HTML page provides <body data-chapter="03"> and a <main class="content">.
 * `?selftest=1` in the URL makes the page write a JSON verdict into
 * <pre id="selftest-result"> after initialisation (used by headless Chrome).
 */
import { CHAPTERS } from './chapters.js';
import { GLOSSARY } from './glossary.js';

const selftest = { enabled: false, errors: [], checks: [] };

/** Root-relative prefix ('' on the landing page, '../' in chapters/). */
export function siteRoot() {
  return document.body.dataset.root ?? '../';
}

/** Wait until the deferred KaTeX scripts are available (max ~5 s). */
function katexReady() {
  return new Promise((resolve) => {
    let n = 0;
    const poll = () => {
      if (window.katex && window.renderMathInElement) resolve(true);
      else if (n++ > 100) resolve(false);
      else setTimeout(poll, 50);
    };
    poll();
  });
}

const MATH_OPTS = {
  delimiters: [
    { left: '$$', right: '$$', display: true },
    { left: '\\[', right: '\\]', display: true },
    { left: '$', right: '$', display: false },
    { left: '\\(', right: '\\)', display: false },
  ],
  throwOnError: false,
  macros: {
    '\\R': '\\mathbb{R}',
    '\\jump': '[\\![#1]\\!]',
    '\\avg': '\\{\\!\\{#1\\}\\!\\}',
    '\\n': '\\mathbf{n}',
    '\\F': '\\mathbf{F}',
    '\\bu': '\\mathbf{u}',
    '\\bx': '\\mathbf{x}',
    '\\bsigma': '\\boldsymbol{\\sigma}',
    '\\dd': '\\,\\mathrm{d}',
    '\\Th': '\\mathcal{T}_h',
    '\\Eh': '\\mathcal{E}_h',
  },
};

/**
 * Render TeX inside an element (call after inserting dynamic content).
 * @param {HTMLElement} el
 */
export function renderMath(el) {
  if (window.renderMathInElement) window.renderMathInElement(el, MATH_OPTS);
}

/**
 * Render a TeX string to HTML (inline). Falls back to the raw string.
 * @param {string} tex
 * @param {boolean} [display=false]
 */
export function tex(tex, display = false) {
  if (window.katex) return window.katex.renderToString(tex, { throwOnError: false, displayMode: display, macros: { ...MATH_OPTS.macros } });
  return tex;
}

/** Build the sidebar + wrap <main> into the two-column layout. */
function buildShell() {
  const id = document.body.dataset.chapter;
  const root = siteRoot();
  const main = document.querySelector('main.content');
  const layout = document.createElement('div');
  layout.className = 'layout';
  const side = document.createElement('nav');
  side.className = 'sidebar';
  side.setAttribute('aria-label', 'Chapters');
  side.innerHTML = `
    <a class="brand" href="${root}index.html">Fluxes, DG &amp; friends<small>an interactive course</small></a>
    <h4>Chapters</h4>
    <ol class="chapters">${CHAPTERS.map((c) => `<li><a href="${root}${c.file}" class="${c.id === id ? 'current' : ''}"><span class="num">${c.num}</span><span>${c.title}</span></a></li>`).join('')}</ol>
    <h4 class="toc-head">On this page</h4>
    <ul class="toc"></ul>`;
  main.parentNode.insertBefore(layout, main);
  layout.appendChild(side);
  layout.appendChild(main);
  const toggle = document.createElement('button');
  toggle.className = 'menu-toggle';
  toggle.textContent = '☰ Menu';
  toggle.addEventListener('click', () => side.classList.toggle('open'));
  document.body.appendChild(toggle);

  // prev / next
  const idx = CHAPTERS.findIndex((c) => c.id === id);
  if (idx >= 0) {
    const nav = document.createElement('nav');
    nav.className = 'chapnav';
    const prev = CHAPTERS[idx - 1], next = CHAPTERS[idx + 1];
    nav.innerHTML = (prev ? `<a class="prev" href="${root}${prev.file}"><small>← Previous</small>${prev.num}. ${prev.title}</a>` : '<span></span>')
      + (next ? `<a class="next" href="${root}${next.file}"><small>Next →</small>${next.num}. ${next.title}</a>` : '<span></span>');
    main.appendChild(nav);
    const foot = document.createElement('footer');
    foot.className = 'site';
    foot.innerHTML = `Code MIT, text CC BY 4.0 · <a href="https://github.com/VaitkusM/dgfem-applet">source on GitHub</a>`;
    main.appendChild(foot);
  }
}

/** Fill the "On this page" list from h2 headings (adds ids when missing). */
function buildToc() {
  const toc = document.querySelector('.sidebar .toc');
  if (!toc) return;
  const heads = [...document.querySelectorAll('main.content h2')];
  if (!heads.length) { document.querySelector('.sidebar .toc-head')?.remove(); return; }
  for (const h of heads) {
    if (!h.id) h.id = h.textContent.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = `#${h.id}`;
    a.innerHTML = h.innerHTML;
    li.appendChild(a);
    toc.appendChild(li);
  }
  const links = [...toc.querySelectorAll('a')];
  const obs = new IntersectionObserver((entries) => {
    for (const en of entries) if (en.isIntersecting) {
      links.forEach((l) => l.classList.toggle('active', l.getAttribute('href') === `#${en.target.id}`));
    }
  }, { rootMargin: '0px 0px -70% 0px' });
  heads.forEach((h) => obs.observe(h));
}

/** Hover / focus pop-ups for glossary terms. */
function installGlossary() {
  let pop = null;
  const show = (el) => {
    const g = GLOSSARY[el.dataset.g];
    if (!g) return;
    pop?.remove();
    pop = document.createElement('div');
    pop.className = 'glossary-pop';
    pop.innerHTML = `<b>${g.term}</b>${g.def}`;
    document.body.appendChild(pop);
    renderMath(pop);
    const r = el.getBoundingClientRect();
    const w = pop.offsetWidth;
    pop.style.left = `${Math.max(8, Math.min(window.scrollX + r.left, window.scrollX + document.documentElement.clientWidth - w - 8))}px`;
    pop.style.top = `${window.scrollY + r.bottom + 6}px`;
  };
  const hide = () => { pop?.remove(); pop = null; };
  document.querySelectorAll('.term[data-g]').forEach((el) => {
    el.tabIndex = 0;
    el.addEventListener('mouseenter', () => show(el));
    el.addEventListener('focus', () => show(el));
    el.addEventListener('mouseleave', hide);
    el.addEventListener('blur', hide);
    if (!GLOSSARY[el.dataset.g]) selftest.errors.push(`unknown glossary key ${el.dataset.g}`);
  });
}

/**
 * Record a numeric sanity check for self-test mode (e.g. "no NaN in solution").
 * @param {string} label
 * @param {boolean} ok
 */
export function selfCheck(label, ok) {
  selftest.checks.push({ label, ok: !!ok });
}

/** Report an exception from a widget without killing the rest of the page. */
export function reportError(where, err) {
  console.error(where, err);
  selftest.errors.push(`${where}: ${err && err.message ? err.message : err}`);
}

/**
 * Mount a widget safely: errors are shown in place and reported.
 * @param {string} id element id of the <figure class="widget">
 * @param {(el: HTMLElement) => any} factory
 */
export function mount(id, factory) {
  const el = document.getElementById(id);
  if (!el) { reportError('mount', new Error(`missing element #${id}`)); return null; }
  try {
    return factory(el);
  } catch (err) {
    reportError(`widget #${id}`, err);
    const p = document.createElement('p');
    p.className = 'status-bad';
    p.textContent = `This widget failed to load: ${err.message}`;
    el.appendChild(p);
    return null;
  }
}

/**
 * Initialise a chapter page.
 * @param {() => (void|Promise<void>)} [setup] builds the chapter's widgets
 */
export async function initChapter(setup) {
  selftest.enabled = new URLSearchParams(location.search).has('selftest');
  window.addEventListener('error', (e) => selftest.errors.push(`error: ${e.message}`));
  window.addEventListener('unhandledrejection', (e) => selftest.errors.push(`rejection: ${e.reason}`));
  buildShell();
  await katexReady();
  try { await setup?.(); } catch (err) { reportError('setup', err); }
  renderMath(document.querySelector('main.content'));
  buildToc();
  installGlossary();
  if (selftest.enabled) {
    await new Promise((r) => setTimeout(r, 1500)); // let animations run a few frames
    const failed = selftest.checks.filter((c) => !c.ok);
    const pre = document.createElement('pre');
    pre.id = 'selftest-result';
    pre.textContent = JSON.stringify({ ok: selftest.errors.length === 0 && failed.length === 0, katex: !!window.katex, errors: selftest.errors, failedChecks: failed, nChecks: selftest.checks.length });
    document.body.appendChild(pre);
  }
}
