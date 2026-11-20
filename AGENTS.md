# AGENTS.md — guide for humans and coding agents

This repository is a **static, no-build** interactive course (HTML + vanilla ES modules).
GitHub Pages serves the repository root as-is: <https://vaitkusm.github.io/dgfem-applet/>.

## Golden rules
1. **No build step, no npm.** Plain ES modules with relative imports ending in `.js`.
   External code only via pinned CDN `<script>` tags with SRI (currently just KaTeX).
2. **`lib/core/` never touches the DOM.** It must run unchanged under Deno (tests) and in the browser.
3. **Every numerical claim made in the prose is backed by a test** in `tests/`
   (exactness, invariants, or convergence rates against manufactured solutions).
4. **Lowercase file names only** for everything the site loads (GitHub Pages is case-sensitive, macOS is not);
   the conventional upper-case repository files (`README.md`, `AGENTS.md`, `LICENSE*`) are the only exceptions.
5. **JSDoc on every exported function**, stating array layouts and sign conventions.
6. Typed arrays (`Float64Array`, `Int32Array`) for all numerical data; no per-DOF objects.

## Layout
```
index.html                   landing page (cards generated from lib/ui/chapters.js by lib/ui/landing.js)
chapters/NN-name.html        one page per chapter: prose + <figure class="widget" id="w-…">
chapters/a-verification.html appendix: verification (runs the test suite in the browser) + glossary
chapters/js/chNN.js          widget wiring for that chapter (imports lib/…); cha.js for the appendix
chapters/js/ellipticcommon.js  manufactured problems shared by the chapter 8 and 9 widgets
css/style.css                all styling; colours are CSS variables (dark mode aware, WCAG AA contrast)
lib/core/                    numerics (pure functions, DOM-free)
  basis/  quad/  la/  mesh/  time/  models/  fv/  dg/  fr/  elliptic/  immersed/  dd/  verify/
lib/viz/                     canvas drawing: canvas.js (HiDPI surfaces, theme(), View2D, pointer + keyboard input),
                             plots, colormaps, field rasteriser, meshes, spy plots, contours, animation
lib/ui/                      chapter.js (page shell: nav, KaTeX, glossary pop-ups, self-test), controls.js,
                             chapters.js (chapter list), glossary.js, landing.js (landing page)
lib/gpu/                     chapter 12. DOM-free and tested under Deno: cputwin.js (f64 reference solver),
                             cost.js (FLOP/byte + SIMT models), races.js (race/colouring models), kernels.js (WGSL
                             sources, uniform layout). Browser/WebGPU only: device.js (device setup, loss handling),
                             dgadvgpu.js (GPU solver; its host logic is unit-tested with a mock device)
tests/                       unit/ conv/ gpu/ *.test.js (listed in tests/manifest.js), harness.js, test.html + runner.js,
                             gpu/gpu-tests.html (GPU tests only)
tools/                       smoke.sh (headless-Chrome smoke test), gpu-smoke.js (its --gpu mode)
.github/workflows/test.yml   CI: deno test, deno lint, smoke test
```

## Conventions (sign conventions are a classic source of bugs — use these everywhere)
| Item | Convention |
|---|---|
| Reference interval | $[-1,1]$, element $[x_L,x_R]$, $x = x_L + (\xi+1)h/2$, so $d/dx = (2/h)\,d/d\xi$ |
| Reference triangle | vertices (0,0), (1,0), (0,1); barycentrics $\lambda_0 = 1-\hat x-\hat y$, $\lambda_1=\hat x$, $\lambda_2=\hat y$ |
| Triangle orientation | counter-clockwise (positive signed area) |
| Local edges | edge $k$ joins local vertices $k$ and $(k+1)\bmod 3$ |
| Face normal | $\mathbf n$ points **out of** $K^-$ (the first element of `edgeTris`) into $K^+$ |
| Jump / average | scalar $[\![u]\!] = u^- - u^+$, vector $[\![u]\!] = u^-\mathbf n^- + u^+\mathbf n^+$, $\{\!\{u\}\!\} = \tfrac12(u^-+u^+)$ |
| Numerical flux | $\hat F(u^-, u^+, \mathbf n)$ = flux **out of** $K^-$ through the face |
| Mixed form | $\boldsymbol\sigma = -\nabla u$ (physical heat flux), $\nabla\cdot\boldsymbol\sigma = f$ |
| Poisson | $-\Delta u = f$ (positive definite operator) |
| Dense matrices | row-major `Float64Array`, `A[i*m + j]` |
| Sparse matrices | CSR `{n, m, rowPtr, colIdx, vals}`, built with `SparseBuilder` (duplicates summed) |
| Mesh arrays | `nodes` = [x0,y0,x1,y1,…], `tris` = 3 ids per triangle, `quads` = 4 ids CCW from lower-left |

## Adding a chapter / widget
1. Add an entry in `lib/ui/chapters.js`.
2. Create `chapters/NN-name.html` (copy an existing chapter; keep the `<head>` KaTeX block identical).
3. Create `chapters/js/chNN.js`:
   ```js
   import { initChapter, mount } from '../../lib/ui/chapter.js';
   initChapter(() => { mount('w-my-widget', createMyWidget); });
   ```
4. Widgets are factories `(figureElement) => handle`. Use `widgetLayout`, controls from `lib/ui/controls.js`,
   and surfaces from `lib/viz/canvas.js`. Call `selfCheck(label, ok)` for sanity checks (e.g. no NaN).
5. Put numerics in `lib/core/…` (chapter-12 models: `lib/gpu/`, see Layout) with tests in `tests/…` and register the test file
   in `tests/manifest.js`. Models behind a widget's read-out belong there too, not in the chapter script.
6. Add the favicon line to the `<head>` (copy it from an existing chapter) and, once the widgets exist, `data-min-checks`
   to `<body>`; check the page at 400 px width (no horizontal page scroll) and in dark mode.

## Shared UI helpers worth knowing
- `createCanvas(parent, {aspect, label?, description?})` gives the canvas `role="img"`, an accessible name (the panel's
  `.canvas-label`, or `label`) and repaints on resize **and** on a light/dark switch — register your draw function with
  `surface.onResize(draw)` and read colours from `theme()` at draw time. Text drawn on an accent-coloured fill uses `theme().onAccent`.
- `onPointer(canvas, {down, move, up, hover, handles?})` is the only place that sets `touch-action: none` (plain canvases
  let a swipe scroll the page) and adds a keyboard alternative for free: arrow keys move a virtual pointer, Space/Enter
  presses/releases it, and your callbacks are called exactly as for a mouse. Return the grabbable points (CSS px) from the
  optional `handles()` so that keyboard users can jump to them with N / P.
- `playControls(parent, animator, onReset)` (lib/ui/controls.js) is the standard Play/Pause + Reset row.
- `segmented`, `slider`, `select` set the ARIA attributes themselves; pass a meaningful `label`.

## Testing
- `deno task test` — the full numerical suite, headless (GitHub Actions, job `deno`). Tests marked `{browserOnly: true}`
  (WebGPU, DOM) are ignored there.
- `deno task lint` — `deno lint` over the repository (job `lint`); it is the only automatic check that parses the chapter scripts
  without a browser, so keep it clean.
- `tools/smoke.sh [base-url]` (job `smoke`) — headless Chrome: `index.html` and every `chapters/*.html` with `?selftest=1`
  must report `"ok":true`, and `tests/test.html` (the same suite in the browser, plus the browser-only tests) must have
  `"failed":0`. `CHROME` selects the browser (autodetected), `CHROME_FLAGS` adds flags (CI: `--no-sandbox`).
- `tools/smoke.sh --gpu` — local only, needs a real GPU and Deno: runs the WebGPU tests and chapter 12 with WebGPU enabled
  (real time, driven over the DevTools protocol by `tools/gpu-smoke.js`) and **fails if a GPU test is skipped**; it also
  exercises device loss and recovery of the live widget. Run it whenever `lib/gpu/` or `chapters/js/ch12.js` changes.
- `python3 -m http.server 8000` then <http://localhost:8000/tests/test.html> — the suite in your own browser.
- **Skipping.** A test that cannot run in the current environment calls `skip(reason)` from `tests/harness.js`; runners count
  it as *skipped* (`{passed, failed, skipped, failures, skippedTests}`), never as passed. Do not `return` early instead.
- **Self-test protocol.** A page opened with `?selftest=1` appends `<pre id="selftest-result">` JSON about 1.5 s after
  initialisation: `ok`, `katex`, `errors`, `failedChecks`, `nChecks` (distinct `selfCheck` labels), `widgetsWithoutChecks`
  (informational). `ok` is true only if there are no errors (exceptions, rejections, widget failures reported with
  `reportError`, unknown glossary keys, sub-resources that failed to load), no failed checks, **and KaTeX rendered the math**.
  Errors arriving later update the verdict in place. `selfCheck(label, ok)` is keyed by label (calling it every frame is
  fine; one failure is sticky). Declare how many checks a page must record with `<body data-min-checks="N">` so that
  checks that silently never ran turn the page red. Anything a widget catches itself must be passed to `reportError`,
  otherwise the self-test cannot see it.
