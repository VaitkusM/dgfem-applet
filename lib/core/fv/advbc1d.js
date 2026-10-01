/**
 * @file Boundary conditions for 1D linear advection u_t + a u_x = 0 on (0,1),
 *       a > 0 (left boundary = inflow, right boundary = outflow), with a
 *       first-order finite volume method — the simplest setting in which to
 *       compare strong and weak imposition of hyperbolic boundary conditions.
 *
 * Weak imposition: the boundary value enters only through a ghost state in the
 * numerical flux at the boundary face, F̂(u⁻, u⁺) with u⁺ = g (inflow) or the
 * prescribed/extrapolated outflow value.
 * Strong imposition: after every stage, the boundary cell value is overwritten.
 *
 * Exact solution: u(x,t) = g_in(t − x/a) for x < a t, u0(x − a t) otherwise.
 */

/**
 * @param {{N: number, a: number, flux: 'upwind'|'central',
 *          inflow: 'weak'|'strong', outflow: 'extrapolate'|'weak'|'strong',
 *          gIn: (t:number)=>number, gOut: (t:number)=>number}} o
 *   outflow: 'extrapolate' = ghost copies the last cell (zero-gradient),
 *            'weak'  = ghost state = gOut (data offered through the flux),
 *            'strong'= last cell overwritten with gOut (over-specification!)
 * @returns {{N: number, h: number, xc: Float64Array, rhs: (u: Float64Array, t: number, out: Float64Array) => void, enforce: (u: Float64Array, t: number) => void}}
 */
export function makeAdvBC1D(o) {
  const N = o.N, h = 1 / N, a = o.a;
  if (!(a > 0)) throw new Error('makeAdvBC1D assumes a > 0');
  const xc = Float64Array.from({ length: N }, (_, i) => (i + 0.5) * h);
  const F = o.flux === 'central' ? (uL, uR) => 0.5 * a * (uL + uR) : (uL) => a * uL; // upwind for a > 0
  const flux = new Float64Array(N + 1);
  function rhs(u, t, out) {
    const ghostL = o.gIn(t);
    const ghostR = o.outflow === 'weak' ? o.gOut(t) : u[N - 1];
    flux[0] = o.inflow === 'weak' ? F(ghostL, u[0]) : F(u[0], u[0]);
    for (let f = 1; f < N; f++) flux[f] = F(u[f - 1], u[f]);
    flux[N] = F(u[N - 1], ghostR);
    for (let i = 0; i < N; i++) out[i] = -(flux[i + 1] - flux[i]) / h;
    if (o.inflow === 'strong') out[0] = 0;          // value is pinned
    if (o.outflow === 'strong') out[N - 1] = 0;
  }
  function enforce(u, t) {
    if (o.inflow === 'strong') u[0] = o.gIn(t);
    if (o.outflow === 'strong') u[N - 1] = o.gOut(t);
  }
  return { N, h, xc, rhs, enforce };
}
