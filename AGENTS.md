# AGENTS.md — guide for humans and coding agents

This repository is a **static, no-build** interactive course (HTML + vanilla ES modules).
GitHub Pages serves the repository root as-is: <https://vaitkusm.github.io/dgfem-applet/>.

## Golden rules
1. **No build step, no npm.** Plain ES modules with relative imports ending in `.js`.
   External code only via pinned CDN `<script>` tags with SRI (currently just KaTeX).
2. **`lib/core/` never touches the DOM.** It must run unchanged under Deno (tests) and in the browser.
3. **Every numerical claim made in the prose is backed by a test** in `tests/`
   (exactness, invariants, or convergence rates against manufactured solutions).
4. **Lowercase file names only** (GitHub Pages is case-sensitive, macOS is not).
5. **JSDoc on every exported function**, stating array layouts and sign conventions.
6. Typed arrays (`Float64Array`, `Int32Array`) for all numerical data; no per-DOF objects.

## Layout
```
index.html                   landing page (cards generated from lib/ui/chapters.js)
chapters/NN-name.html        one page per chapter: prose + <figure class="widget" id="w-…">
chapters/js/chNN.js          widget wiring for that chapter (imports lib/…)
css/style.css                all styling; colours are CSS variables (dark mode aware)
lib/core/                    numerics (pure functions, DOM-free)
  basis/  quad/  la/  mesh/  time/  models/  fv/  dg/  fr/  elliptic/  immersed/  dd/  verify/
lib/viz/                     canvas drawing: plots, colormaps, field rasteriser, meshes, spy plots, animation
lib/ui/                      page shell (nav, KaTeX, glossary pop-ups, self-test), controls, chapter list, glossary
lib/gpu/                     WebGPU device setup, WGSL kernels, GPU solvers
tests/                       *.test.js (listed in tests/manifest.js), harness.js, test.html + runner.js
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
5. Put numerics in `lib/core/…` with tests in `tests/…` and register the test file in `tests/manifest.js`.

## Testing
- `deno task test` — the full numerical suite, headless (also run by GitHub Actions on every push).
- `python3 -m http.server 8000` then open <http://localhost:8000/tests/test.html> — same suite in the browser (+ GPU tests).
- Chapter smoke test: open any chapter with `?selftest=1`; after ~2 s it appends `<pre id="selftest-result">` JSON
  (`ok`, `errors`, `failedChecks`). `tools/smoke.sh` runs this for every page with headless Chrome.
