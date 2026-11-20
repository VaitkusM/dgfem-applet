# Fluxes, DG & friends — an interactive course

**Live site:** <https://vaitkusm.github.io/dgfem-applet/>

A visual, hands-on introduction to the numerical solution of PDEs for computer-science students with a
geometry/graphics background. Every figure is a live solver running in the browser:

0. Primer: fields, divergence theorem, convergence plots
1. Physical and numerical fluxes, conservation laws, characteristics, shocks
2. Strong, weak, mixed and flux forms of PDEs
3. Continuous finite elements for the Poisson problem
4. Strong vs weak (Nitsche) boundary conditions
5. Finite volume methods, Riemann solvers, limiters
6. Discontinuous Galerkin for conservation laws
7. Flux reconstruction
8. DG for elliptic problems (interior penalty)
9. Mixed methods, hybridization and HDG
10. Domain decomposition
11. Immersed boundary methods: CutFEM, WEB-splines, Shifted Boundary Method and variants
12. Parallelisation on GPUs (live WebGPU solver, with a CPU fallback)

Appendix A: how the site is verified (runs the test suite in your browser) and the glossary.

## Running locally
No build step. Serve the folder with any static server (ES modules do not load from `file://`):
```sh
python3 -m http.server 8000      # then open http://localhost:8000
```

The pages load KaTeX (math typesetting) from a pinned CDN URL; without network access the widgets still work and
a notice explains that formulas are shown as TeX source.

## Tests
```sh
deno task test                   # numerical test suite, headless (Deno ≥ 2)
deno task lint                   # static checks of all scripts
tools/smoke.sh                   # every page in headless Chrome (self-test verdicts) + the browser test page
tools/smoke.sh --gpu             # the same GPU solver tests on a real GPU (WebGPU); fails if they are skipped
```
or open `http://localhost:8000/tests/test.html` (the only place, besides `smoke.sh --gpu`, where the WebGPU tests
run — Deno and CI report them as *skipped*). GitHub Actions runs the first three on every push.
See [AGENTS.md](AGENTS.md) for code conventions, the self-test protocol and how to add a chapter.

## License
Code: MIT ([LICENSE](LICENSE)). Text and figures: CC BY 4.0 ([LICENSE-CONTENT](LICENSE-CONTENT)) — please credit
“Fluxes, DG & friends” by VaitkusM with a link to the site.
