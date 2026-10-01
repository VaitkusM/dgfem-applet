/**
 * @file Single source of truth for the chapter list (navigation, landing page).
 * `file` is relative to the site root.
 */

/** @typedef {{id: string, num: string, file: string, title: string, blurb: string}} ChapterInfo */

/** @type {ChapterInfo[]} */
export const CHAPTERS = [
  { id: '00', num: '0', file: 'chapters/00-primer.html', title: 'Primer: fields, divergence & convergence',
    blurb: 'Vector calculus for graphics people: gradient, divergence, the divergence theorem, norms, and how to read a convergence plot.' },
  { id: '01', num: '1', file: 'chapters/01-fluxes.html', title: 'Fluxes & conservation laws',
    blurb: 'What a flux is physically, the integral conservation law, characteristics, shocks and the Rankine–Hugoniot condition.' },
  { id: '02', num: '2', file: 'chapters/02-pde-forms.html', title: 'Strong, weak, mixed & flux forms',
    blurb: 'Integration by parts turns a PDE into a weak form. Test functions, Galerkin, essential vs natural conditions, mixed forms.' },
  { id: '03', num: '3', file: 'chapters/03-cg-fem.html', title: 'Continuous finite elements',
    blurb: 'Reference elements, Jacobians, assembly, sparse matrices and conjugate gradients for the Poisson problem.' },
  { id: '04', num: '4', file: 'chapters/04-boundary-conditions.html', title: 'Strong vs weak boundary conditions',
    blurb: 'Eliminating boundary unknowns vs penalty and Nitsche’s method; inflow/outflow conditions for transport.' },
  { id: '05', num: '5', file: 'chapters/05-finite-volume.html', title: 'Finite volume methods',
    blurb: 'Cell averages, Riemann problems, numerical fluxes (upwind, Rusanov, Roe, HLL), CFL, MUSCL and limiters.' },
  { id: '06', num: '6', file: 'chapters/06-dg.html', title: 'Discontinuous Galerkin for conservation laws',
    blurb: 'High-order polynomials per cell glued by numerical fluxes. Bases, reference matrices, spectra, CFL and limiting.' },
  { id: '07', num: '7', file: 'chapters/07-flux-reconstruction.html', title: 'Flux reconstruction',
    blurb: 'Huynh’s differential view: correct a discontinuous flux with correction functions. Recovers DG and more.' },
  { id: '08', num: '8', file: 'chapters/08-dg-elliptic.html', title: 'DG for elliptic problems',
    blurb: 'Interior penalty methods (SIPG/NIPG): Nitsche’s idea on every face, and why the penalty must be large enough.' },
  { id: '09', num: '9', file: 'chapters/09-hybridization.html', title: 'Mixed methods & hybridization',
    blurb: 'Saddle-point problems, Raviart–Thomas fluxes, Lagrange multipliers on the skeleton, static condensation and HDG.' },
  { id: '10', num: '10', file: 'chapters/10-domain-decomposition.html', title: 'Domain decomposition',
    blurb: 'Divide and conquer: Schwarz methods, coarse spaces, Schur complements and the link to hybridization.' },
  { id: '11', num: '11', file: 'chapters/11-immersed.html', title: 'Immersed boundary methods',
    blurb: 'Solve on a background grid that ignores the geometry: CutFEM, WEB-splines, the Shifted Boundary Method and friends.' },
  { id: '12', num: '12', file: 'chapters/12-gpu.html', title: 'Parallelisation on GPUs',
    blurb: 'Why DG loves GPUs: element-local work, memory layouts, race-free face fluxes and a live WebGPU solver.' },
  { id: 'A', num: 'A', file: 'chapters/a-verification.html', title: 'Appendix: verification & glossary',
    blurb: 'How every claim on this site is tested: manufactured solutions, convergence rates — plus the full glossary.' },
];
