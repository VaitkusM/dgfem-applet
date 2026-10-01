/**
 * @file Manufactured solutions for the Poisson problem  −Δu = f.
 *
 * "Method of manufactured solutions" (MMS): pick a smooth u, compute
 * f = −Δu by hand, and use g = u on the boundary. The discrete solution u_h
 * must then converge to the known u at the theoretical rate — the standard
 * way to verify a PDE code. Each entry provides:
 *   u(x,y), grad(x,y) → [∂u/∂x, ∂u/∂y], f(x,y) = −Δu, and a TeX formula.
 */

const { sin, cos, exp, PI } = Math;

/** @typedef {{name: string, tex: string, u: (x:number,y:number)=>number, grad: (x:number,y:number)=>[number,number], f: (x:number,y:number)=>number}} MMS */

/** @type {Record<string, MMS>} */
export const POISSON_MMS = {
  sinsin: {
    name: 'sin·sin (zero on the unit-square boundary)',
    tex: 'u = \\sin(\\pi x)\\sin(\\pi y)',
    u: (x, y) => sin(PI * x) * sin(PI * y),
    grad: (x, y) => [PI * cos(PI * x) * sin(PI * y), PI * sin(PI * x) * cos(PI * y)],
    f: (x, y) => 2 * PI * PI * sin(PI * x) * sin(PI * y),
  },
  wave: {
    name: 'cos·exp (non-zero boundary values)',
    tex: 'u = \\cos(\\pi x)\\,e^{y}',
    u: (x, y) => cos(PI * x) * exp(y),
    grad: (x, y) => [-PI * sin(PI * x) * exp(y), cos(PI * x) * exp(y)],
    // Δu = (−π² + 1) u  ⇒  f = −Δu = (π² − 1) u
    f: (x, y) => (PI * PI - 1) * cos(PI * x) * exp(y),
  },
  peak: {
    name: 'Gaussian peak at (0.5, 0.5)',
    tex: 'u = e^{-50\\,|x - x_0|^2}',
    u: (x, y) => exp(-50 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)),
    grad: (x, y) => {
      const u = exp(-50 * ((x - 0.5) ** 2 + (y - 0.5) ** 2));
      return [-100 * (x - 0.5) * u, -100 * (y - 0.5) * u];
    },
    // u = e^{−a r²}:  Δu = u (4a² r² − 4a)  (2D)  ⇒  f = u (4a − 4a² r²)
    f: (x, y) => {
      const r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2, a = 50;
      return exp(-a * r2) * (4 * a - 4 * a * a * r2);
    },
  },
};
