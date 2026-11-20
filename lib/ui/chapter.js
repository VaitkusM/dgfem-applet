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
 *
 * SELF-TEST. `?selftest=1` in the URL makes the page write a JSON verdict into
 * <pre id="selftest-result"> about 1.5 s after initialisation (read by headless
 * Chrome in tools/smoke.sh). Fields:
 *   ok                    true iff there are no errors, no failed checks AND KaTeX rendered the math
 *   katex                 whether KaTeX was available
 *   errors                uncaught exceptions / rejections, widget failures, unknown glossary keys,
 *                         sub-resources that failed to load, fewer checks than the page declared
 *   failedChecks          selfCheck(label, false) results, [{label, ok: false}]
 *   nChecks               number of DISTINCT check labels that were recorded
 *   widgetsWithoutChecks  ids of widgets whose factory recorded no check while mounting
 *                         (informational: checks made later, e.g. in animation frames, are not attributed)
 * A page can demand a minimum number of distinct checks with <body data-min-checks="7"> or
 * initChapter(setup, {minChecks: 7}); fewer is reported as an error (catches widgets whose
 * asynchronous checks silently never ran). Errors that occur after the verdict was written
 * update it in place, so a late failure inside the smoke test's time budget still turns it red.
 *
 * This module touches the DOM only inside functions, so its pure helpers
 * (summariseSelftest) can be unit-tested under Deno.
 */
import { CHAPTERS } from './chapters.js';
import { GLOSSARY } from './glossary.js';

const hasDom = typeof document !== 'undefined';

/**
 * Self-test state. `checks` is keyed by label, so a check repeated in every
 * animation frame occupies one entry (a failure is sticky).
 * @type {{enabled: boolean, errors: string[], checks: Map<string, boolean>, widgets: Array<{id: string, n: number}>, minChecks: number, katex: boolean, onChange: null | (() => void)}}
 */
const selftest = {
  enabled: typeof location !== 'undefined' && new URLSearchParams(location.search).has('selftest'),
  errors: [], checks: new Map(), widgets: [], minChecks: 0, katex: false, onChange: null,
};
let mounting = null; // entry of selftest.widgets whose factory is currently running

function pushError(msg) {
  if (selftest.errors.length < 200) selftest.errors.push(msg);
  selftest.onChange?.();
}

/** Root-relative prefix ('' on the landing page, '../' in chapters/). */
export function siteRoot() {
  return document.body.dataset.root ?? '../';
}

/** Inline SVG favicon (a discontinuous piecewise-linear "DG solution"); avoids a /favicon.ico 404. */
export const FAVICON = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%232c5fb3'/%3E%3Cpath d='M5 22 L15 12 M17 20 L27 8' stroke='%23fff' stroke-width='3.2' stroke-linecap='round'/%3E%3C/svg%3E";

/** Add the favicon <link> unless the page already has one. */
export function installFavicon() {
  if (document.querySelector('link[rel~="icon"]')) return;
  const l = document.createElement('link');
  l.rel = 'icon'; l.type = 'image/svg+xml'; l.href = FAVICON;
  document.head.appendChild(l);
}

/**
 * Wait until the deferred KaTeX scripts are available.
 * Module scripts run after all earlier `defer` scripts have executed or failed, so normally the
 * answer is known immediately; we still poll briefly (≤ 0.5 s; ≤ 5 s while a KaTeX script is
 * demonstrably still being fetched) and stop at once when a KaTeX <script> reports a load error.
 * @returns {Promise<boolean>} true if KaTeX and its auto-render extension are usable
 */
export function katexReady() {
  return new Promise((resolve) => {
    const have = () => !!(globalThis.katex && globalThis.renderMathInElement);
    if (have()) { resolve(true); return; }
    let done = false, n = 0;
    const finish = (v) => { if (!done) { done = true; resolve(v); } };
    const scripts = [...document.querySelectorAll('script[src*="katex"]')];
    scripts.forEach((s) => s.addEventListener('error', () => finish(false)));
    // still being fetched? (a finished fetch — successful or not — has a Resource Timing entry)
    const pending = () => {
      if (document.readyState === 'loading') return true;
      try {
        const seen = new Set(performance.getEntriesByType('resource').map((r) => r.name));
        return scripts.some((s) => !seen.has(s.src));
      } catch { return false; }
    };
    const poll = () => {
      if (done) return;
      if (have()) finish(true);
      else if (n++ >= (pending() ? 100 : 10)) finish(false);
      else setTimeout(poll, 50);
    };
    poll();
  });
}

/**
 * Show a visible notice that formulas could not be rendered (KaTeX CDN unreachable).
 * @param {HTMLElement} main element that receives the notice as its first child
 */
export function katexNotice(main) {
  if (!main || main.querySelector('.katex-missing')) return;
  const d = document.createElement('div');
  d.className = 'box warn katex-missing';
  d.setAttribute('role', 'status');
  d.innerHTML = '<span class="box-title">Formulas not rendered</span><p>The math typesetting library (KaTeX, loaded from cdn.jsdelivr.net) could not be loaded, '
    + 'so formulas appear as raw TeX source between dollar signs. The interactive figures work normally. Check your network connection and reload the page.</p>';
  main.insertBefore(d, main.firstChild);
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
  if (globalThis.renderMathInElement) globalThis.renderMathInElement(el, MATH_OPTS);
}

/**
 * Render a TeX string to HTML (inline). Falls back to the raw string.
 * @param {string} tex
 * @param {boolean} [display=false]
 * @returns {string} HTML
 */
export function tex(tex, display = false) {
  if (globalThis.katex) return globalThis.katex.renderToString(tex, { throwOnError: false, displayMode: display, macros: { ...MATH_OPTS.macros } });
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
  side.id = 'site-sidebar';
  side.setAttribute('aria-label', 'Chapters');
  side.innerHTML = `
    <a class="brand" href="${root}index.html">Fluxes, DG &amp; friends<small>an interactive course</small></a>
    <h4>Chapters</h4>
    <ol class="chapters">${CHAPTERS.map((c) => `<li><a href="${root}${c.file}" class="${c.id === id ? 'current' : ''}"${c.id === id ? ' aria-current="page"' : ''}><span class="num">${c.num}</span><span>${c.title}</span></a></li>`).join('')}</ol>
    <h4 class="toc-head">On this page</h4>
    <ul class="toc"></ul>`;
  main.parentNode.insertBefore(layout, main);
  layout.appendChild(side);
  layout.appendChild(main);

  // Narrow screens: the sidebar is an off-canvas drawer opened by a menu button.
  const toggle = document.createElement('button');
  toggle.className = 'menu-toggle';
  toggle.type = 'button';
  toggle.textContent = '☰ Menu';
  toggle.setAttribute('aria-controls', side.id);
  const narrow = globalThis.matchMedia ? globalThis.matchMedia('(max-width: 900px)') : { matches: false, addEventListener() {} };
  const sync = () => {
    const open = side.classList.contains('open');
    toggle.setAttribute('aria-expanded', String(open));
    side.inert = narrow.matches && !open; // a closed drawer must not be reachable by Tab / screen readers
  };
  const setOpen = (open) => { side.classList.toggle('open', open); sync(); };
  toggle.addEventListener('click', () => setOpen(!side.classList.contains('open')));
  side.addEventListener('click', (e) => { if (e.target.closest('a')) setOpen(false); }); // following a link closes the drawer
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && side.classList.contains('open')) { setOpen(false); toggle.focus(); } });
  narrow.addEventListener('change', sync);
  layout.parentNode.insertBefore(toggle, layout); // before the content in tab order
  sync();

  // prev / next
  const idx = CHAPTERS.findIndex((c) => c.id === id);
  if (idx >= 0) {
    const nav = document.createElement('nav');
    nav.className = 'chapnav';
    nav.setAttribute('aria-label', 'Previous and next chapter');
    const prev = CHAPTERS[idx - 1], next = CHAPTERS[idx + 1];
    nav.innerHTML = (prev ? `<a class="prev" href="${root}${prev.file}"><small>← Previous</small>${prev.num}. ${prev.title}</a>` : '<span></span>')
      + (next ? `<a class="next" href="${root}${next.file}"><small>Next →</small>${next.num}. ${next.title}</a>` : '<span></span>');
    main.appendChild(nav);
    const foot = document.createElement('footer');
    foot.className = 'site';
    foot.innerHTML = `Code <a href="https://github.com/VaitkusM/dgfem-applet/blob/main/LICENSE">MIT</a>, text and figures <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> (credit: “Fluxes, DG &amp; friends”, VaitkusM) · <a href="https://github.com/VaitkusM/dgfem-applet">source on GitHub</a>`;
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
  // Scroll-spy: highlight the last heading that has passed the upper third of the viewport
  // (works in both scroll directions, unlike a one-shot "entered the viewport" trigger).
  const links = [...toc.querySelectorAll('a')];
  let ticking = false;
  const update = () => {
    ticking = false;
    let cur = null;
    for (const h of heads) if (h.getBoundingClientRect().top <= globalThis.innerHeight * 0.3) cur = h;
    links.forEach((l) => {
      const on = !!cur && l.getAttribute('href') === `#${cur.id}`;
      l.classList.toggle('active', on);
      if (on) l.setAttribute('aria-current', 'location'); else l.removeAttribute('aria-current');
    });
  };
  globalThis.addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
  update();
}

/**
 * Hover / focus pop-ups for glossary terms `<span class="term" data-g="key">` inside `root`.
 * Terms become focusable; the pop-up is a tooltip (aria-describedby) that Escape dismisses,
 * stays inside the viewport horizontally and flips above the term near the bottom edge.
 * @param {ParentNode} [root=document]
 */
export function installGlossary(root = document) {
  let pop = null, owner = null;
  const hide = () => { pop?.remove(); pop = null; owner?.removeAttribute('aria-describedby'); owner = null; };
  const show = (el) => {
    const g = GLOSSARY[el.dataset.g];
    if (!g) return;
    hide();
    pop = document.createElement('div');
    pop.className = 'glossary-pop';
    pop.id = 'glossary-pop';
    pop.setAttribute('role', 'tooltip');
    pop.innerHTML = `<b>${g.term}</b>${g.def}`;
    document.body.appendChild(pop);
    renderMath(pop);
    owner = el; el.setAttribute('aria-describedby', pop.id);
    const r = el.getBoundingClientRect();
    const w = pop.offsetWidth, hgt = pop.offsetHeight, vw = document.documentElement.clientWidth, vh = document.documentElement.clientHeight;
    pop.style.left = `${Math.max(8, Math.min(globalThis.scrollX + r.left, globalThis.scrollX + vw - w - 8))}px`;
    const below = r.bottom + 6 + hgt <= vh || r.top - 6 - hgt < 0;
    pop.style.top = `${globalThis.scrollY + (below ? r.bottom + 6 : r.top - 6 - hgt)}px`;
  };
  root.querySelectorAll('.term[data-g]').forEach((el) => {
    el.tabIndex = 0;
    el.addEventListener('mouseenter', () => show(el));
    el.addEventListener('focus', () => show(el));
    el.addEventListener('mouseleave', hide);
    el.addEventListener('blur', hide);
    el.addEventListener('keydown', (e) => { if (e.key === 'Escape') hide(); });
    if (!GLOSSARY[el.dataset.g]) pushError(`unknown glossary key ${el.dataset.g}`);
  });
}

/**
 * Record a numeric sanity check for self-test mode (e.g. "no NaN in solution").
 * Checks are keyed by label: calling it again with the same label (every frame of an
 * animation, say) does not grow any list, and one failure makes the label fail for good.
 * Outside self-test mode this is a no-op.
 * @param {string} label
 * @param {boolean} ok
 */
export function selfCheck(label, ok) {
  if (!selftest.enabled) return;
  if (mounting) mounting.n++;
  const prev = selftest.checks.get(label);
  const now = !!ok && prev !== false;
  if (prev === now) return;
  selftest.checks.set(label, now);
  selftest.onChange?.();
}

/**
 * Report an exception from a widget without killing the rest of the page.
 * @param {string} where short location, e.g. 'widget #w-live'
 * @param {unknown} err
 */
export function reportError(where, err) {
  console.error(where, err);
  pushError(`${where}: ${err && err.message ? err.message : err}`);
}

/**
 * Mount a widget safely: errors are shown in place and reported.
 * @param {string} id element id of the <figure class="widget">
 * @param {(el: HTMLElement) => any} factory
 * @returns {any} whatever the factory returns (null on failure)
 */
export function mount(id, factory) {
  const el = document.getElementById(id);
  if (!el) { reportError('mount', new Error(`missing element #${id}`)); return null; }
  const entry = { id, n: 0 };
  selftest.widgets.push(entry);
  const outer = mounting;
  mounting = entry;
  try {
    return factory(el);
  } catch (err) {
    reportError(`widget #${id}`, err);
    const p = document.createElement('p');
    p.className = 'status-bad';
    p.textContent = `This widget failed to load: ${err.message}`;
    el.appendChild(p);
    return null;
  } finally {
    mounting = outer;
  }
}

/**
 * Pure verdict of a self-test run (unit-tested under Deno).
 * @param {{errors: string[], checks: Map<string, boolean>|Array<[string, boolean]>, katex: boolean, minChecks?: number, widgets?: Array<{id: string, n: number}>}} s
 * @returns {{ok: boolean, katex: boolean, errors: string[], failedChecks: Array<{label: string, ok: boolean}>, nChecks: number, widgetsWithoutChecks: string[]}}
 */
export function summariseSelftest(s) {
  const checks = [...s.checks];
  const failedChecks = checks.filter(([, ok]) => !ok).map(([label]) => ({ label, ok: false }));
  const errors = [...s.errors];
  const min = s.minChecks ?? 0;
  if (checks.length < min) errors.push(`only ${checks.length} self-checks ran, the page declares at least ${min} (a widget's checks never ran?)`);
  return {
    ok: errors.length === 0 && failedChecks.length === 0 && !!s.katex,
    katex: !!s.katex, errors, failedChecks, nChecks: checks.length,
    widgetsWithoutChecks: (s.widgets ?? []).filter((w) => w.n === 0).map((w) => w.id),
  };
}

/** Sub-resources that are present in the DOM but failed to load (stylesheets, images). */
function failedResources() {
  const bad = [];
  document.querySelectorAll('link[rel~="stylesheet"]').forEach((l) => { if (!l.sheet) bad.push(l.getAttribute('href')); });
  document.querySelectorAll('img').forEach((i) => { if (i.complete && i.naturalWidth === 0 && i.getAttribute('src')) bad.push(i.getAttribute('src')); });
  return bad;
}

/**
 * Install the global error listeners of self-test mode and return a function that writes
 * (and from then on keeps up to date) the verdict in <pre id="selftest-result">.
 * Shared by chapter pages and the landing page.
 * @param {{minChecks?: number}} [o]
 * @returns {{enabled: boolean, setKatex: (ok: boolean) => void, publish: () => void}}
 */
export function selftestSession(o = {}) {
  selftest.minChecks = o.minChecks ?? (Number(document.body.dataset.minChecks) || 0);
  globalThis.addEventListener('error', (e) => {
    // capture phase: also sees load failures of <script>/<link>/<img>, which do not bubble
    const t = e.target;
    if (t && t !== globalThis && t.nodeType === 1) pushError(`resource failed to load: <${t.tagName.toLowerCase()}> ${t.getAttribute('src') || t.getAttribute('href') || ''}`);
    else pushError(`error: ${e.message}`);
  }, true);
  globalThis.addEventListener('unhandledrejection', (e) => pushError(`rejection: ${e.reason && e.reason.message ? e.reason.message : e.reason}`));
  let pre = null;
  const reported = new Set();
  const write = () => {
    for (const r of failedResources()) if (!reported.has(r)) { reported.add(r); selftest.errors.push(`resource failed to load: ${r}`); }
    pre.textContent = JSON.stringify(summariseSelftest(selftest));
  };
  return {
    enabled: selftest.enabled,
    setKatex(ok) { selftest.katex = !!ok; },
    publish() {
      if (!selftest.enabled || pre) return;
      pre = document.createElement('pre');
      pre.id = 'selftest-result';
      document.body.appendChild(pre);
      write();
      selftest.onChange = write; // late errors / failed checks update the verdict in place
    },
  };
}

/**
 * Initialise a chapter page.
 * @param {() => (void|Promise<void>)} [setup] builds the chapter's widgets
 * @param {{minChecks?: number}} [o] minChecks: minimum number of distinct selfCheck labels the page must record
 *   (default: <body data-min-checks>, else 0)
 * @returns {Promise<void>}
 */
export async function initChapter(setup, o = {}) {
  const session = selftestSession(o);
  installFavicon();
  buildShell();
  const main = document.querySelector('main.content');
  const haveKatex = await katexReady();
  session.setKatex(haveKatex);
  if (!haveKatex) katexNotice(main);
  try { await setup?.(); } catch (err) { reportError('setup', err); }
  renderMath(main);
  buildToc();
  installGlossary();
  if (session.enabled) {
    await new Promise((r) => setTimeout(r, 1500)); // let animations run a few frames
    session.publish();
  }
}
