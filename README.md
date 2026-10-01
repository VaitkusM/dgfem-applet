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
12. Parallelisation on GPUs (live WebGPU solver)

## Running locally
No build step. Serve the folder with any static server (ES modules do not load from `file://`):
```sh
python3 -m http.server 8000      # then open http://localhost:8000
```

## Tests
```sh
deno task test                   # headless numerical test suite (Deno ≥ 2)
```
or open `http://localhost:8000/tests/test.html`. See [AGENTS.md](AGENTS.md) for code conventions.

## License
Code: MIT ([LICENSE](LICENSE)). Text and figures: CC BY 4.0 ([LICENSE-CONTENT](LICENSE-CONTENT)).
