/**
 * @file Scalar conservation laws  u_t + f(u)_x = 0  in 1D: models, exact
 *       Riemann solutions and numerical fluxes.
 *
 * A *model* is { f(u), df(u) } (flux and characteristic speed f'(u)).
 *  - advection:  f(u) = a u,      f'(u) = a            (linear transport)
 *  - burgers:    f(u) = u²/2,     f'(u) = u            (simplest nonlinear: shocks!)
 *  - buckley:    Buckley–Leverett f(u) = u²/(u² + M(1−u)²) (non-convex: compound waves)
 *
 * A *numerical flux* F̂(uL, uR) approximates the flux through a cell face
 * given the states on its left (uL) and right (uR). Consistency requires
 * F̂(u, u) = f(u). All fluxes below are for the face normal pointing from
 * L to R (+x).
 */

/**
 * @typedef {Object} ScalarModel
 * @property {string} name
 * @property {(u:number)=>number} f flux function
 * @property {(u:number)=>number} df characteristic speed f'(u)
 * @property {boolean} [convex] true if f'' ≥ 0 everywhere (then |f'| is largest at the ends of any interval)
 * @property {boolean} [linear] true for f = a u
 * @property {number} [a] advection speed (linear models only)
 * @property {number[]} [dfCrit] for non-convex f: the points where f'' changes sign (local extrema of f'), sorted;
 *   between two neighbouring entries f' is monotone
 * @property {number[]} [dfCritRange] [lo, hi]: the interval on which `dfCrit` is complete
 */

/**
 * Linear advection with speed a.
 * @param {number} [a=1] advection speed (may be negative)
 * @returns {ScalarModel}
 */
export const advection = (a = 1) => ({ name: 'advection', a, f: (u) => a * u, df: () => a, convex: false, linear: true });

/** Inviscid Burgers. @type {ScalarModel} */
export const burgers = { name: 'burgers', f: (u) => 0.5 * u * u, df: (u) => u, convex: true };

/**
 * Buckley–Leverett (two-phase flow in porous media), mobility ratio M.
 * The flux is S-shaped on [0, 1]: f'(0) = f'(1) = 0 and f' has an interior maximum (M = ½: f' ≈ 2.081 at
 * u ≈ 0.387). The fastest wave of a Riemann problem between 0 and 1 is therefore NOT visible in the two end
 * states — use `maxWaveSpeed` (below) for CFL numbers and for the Rusanov viscosity.
 * @param {number} [M=0.5] mobility ratio (> 0)
 * @returns {ScalarModel}
 */
export const buckley = (M = 0.5) => {
  const df = (u) => {
    const d = u * u + M * (1 - u) * (1 - u);
    return (2 * u * d - u * u * (2 * u - 2 * M * (1 - u))) / (d * d);
  };
  // local extrema of f' (sign changes of f''), located once by a scan of a central difference of f' + bisection
  const lo = -4, hi = 5, n = 9000, e = 1e-6, d2 = (u) => df(u + e) - df(u - e), dfCrit = [];
  for (let k = 0; k < n; k++) {
    let a = lo + (hi - lo) * k / n, b = lo + (hi - lo) * (k + 1) / n;
    if (d2(a) * d2(b) > 0) continue;
    for (let it = 0; it < 60; it++) { const c = 0.5 * (a + b); if (d2(a) * d2(c) <= 0) b = c; else a = c; }
    const c = 0.5 * (a + b);
    if (!dfCrit.length || c - dfCrit[dfCrit.length - 1] > 1e-9) dfCrit.push(c); // a root on a scan point is found twice
  }
  return { name: 'buckley', f: (u) => (u * u) / (u * u + M * (1 - u) * (1 - u)), df, convex: false, dfCrit, dfCritRange: [lo, hi] };
};

/**
 * Largest characteristic speed max |f'(u)| over the closed interval between a and b (in either order).
 *  - linear f: |a|; convex f: f' is monotone, so the maximum sits at one of the two ends;
 *  - non-convex f: the maximum can sit strictly inside (Buckley–Leverett between 0 and 1: f' = 0 at both ends
 *    but ≈ 2.08 in between). We take the ends, the model's `dfCrit` points inside the interval, and — where no
 *    complete `dfCrit` list is available — 64 equally spaced samples.
 * This is the speed to use in a CFL condition and in the Rusanov / local Lax–Friedrichs flux.
 * @param {ScalarModel} m
 * @param {number} a one end of the interval
 * @param {number} b the other end
 * @returns {number} max |f'| ≥ 0
 */
export function maxWaveSpeed(m, a, b) {
  if (m.linear) return Math.abs(m.a);
  let s = Math.max(Math.abs(m.df(a)), Math.abs(m.df(b)));
  if (m.convex) return s;
  const lo = Math.min(a, b), hi = Math.max(a, b);
  if (m.dfCrit) for (const c of m.dfCrit) if (c > lo && c < hi) s = Math.max(s, Math.abs(m.df(c)));
  const complete = m.dfCrit && m.dfCritRange && lo >= m.dfCritRange[0] && hi <= m.dfCritRange[1];
  if (!complete) for (let k = 1; k < 64; k++) s = Math.max(s, Math.abs(m.df(lo + (hi - lo) * k / 64)));
  return s;
}

/**
 * Exact Godunov flux = f(u*(0)), the flux of the exact Riemann solution
 * at the face. For any continuous scalar flux (Osher's formula):
 *   uL ≤ uR:  min_{u ∈ [uL,uR]} f(u)     uL > uR:  max_{u ∈ [uR,uL]} f(u)
 * Exact closed form for advection and Burgers; dense sampling otherwise.
 * @param {ScalarModel} m
 * @param {number} uL state left of the face
 * @param {number} uR state right of the face
 * @returns {number} flux through the face (normal +x)
 */
export function godunovFlux(m, uL, uR) {
  if (m.name === 'burgers') {
    if (uL <= uR) { // rarefaction (or constant): min of u²/2 over [uL,uR]
      if (uL > 0) return 0.5 * uL * uL;
      if (uR < 0) return 0.5 * uR * uR;
      return 0; // sonic point u = 0 lies in the fan
    }
    return Math.max(0.5 * uL * uL, 0.5 * uR * uR); // shock
  }
  if (m.linear) return m.a >= 0 ? m.a * uL : m.a * uR;
  const n = 200;
  let best = uL <= uR ? Infinity : -Infinity;
  for (let k = 0; k <= n; k++) {
    const u = uL + (uR - uL) * k / n, v = m.f(u);
    best = uL <= uR ? Math.min(best, v) : Math.max(best, v);
  }
  return best;
}

/**
 * Upwind flux: take the state the wind comes from.
 * For linear advection this is the exact Godunov flux. For a NONLINEAR f the wind direction is taken from
 * f' at the mean state ½(uL + uR); for Burgers that is the sign of the Rankine–Hugoniot speed, so the flux is
 * then exactly Roe's flux WITHOUT entropy fix (and shares its defect: a transonic rarefaction uL < 0 < uR is kept
 * as a stationary expansion shock). Use `godunovFlux` or `roeFlux` with δ > 0 for nonlinear problems.
 * @param {ScalarModel} m
 * @param {number} uL state left of the face
 * @param {number} uR state right of the face
 * @returns {number} flux through the face (normal +x)
 */
export function upwindFlux(m, uL, uR) {
  const a = m.linear ? m.a : m.df(0.5 * (uL + uR));
  return a >= 0 ? m.f(uL) : m.f(uR);
}

/**
 * Central flux ½(f(uL)+f(uR)) — consistent but unstable with forward Euler.
 * @param {ScalarModel} m
 * @param {number} uL state left of the face
 * @param {number} uR state right of the face
 * @returns {number} flux through the face (normal +x)
 */
export function centralFlux(m, uL, uR) {
  return 0.5 * (m.f(uL) + m.f(uR));
}

/**
 * Local Lax–Friedrichs / Rusanov flux:
 *   F̂ = ½(f(uL) + f(uR)) − ½ α (uR − uL),  α = max |f'(u)| over the interval between uL and uR
 * (`maxWaveSpeed`; for convex f the two end states suffice). With this α the flux is monotone
 * (non-decreasing in uL, non-increasing in uR) and α ≥ |Rankine–Hugoniot speed| by the mean value theorem.
 * @param {ScalarModel} m
 * @param {number} uL state left of the face
 * @param {number} uR state right of the face
 * @returns {number} flux through the face (normal +x)
 */
export function rusanovFlux(m, uL, uR) {
  const alpha = m.linear || m.convex ? Math.max(Math.abs(m.df(uL)), Math.abs(m.df(uR))) : maxWaveSpeed(m, uL, uR);
  return 0.5 * (m.f(uL) + m.f(uR)) - 0.5 * alpha * (uR - uL);
}

/**
 * Roe flux (scalar): upwinding with the Rankine–Hugoniot speed
 * â = (f(uR) − f(uL)) / (uR − uL). Without an entropy fix it admits
 * non-physical "expansion shocks" (e.g. Burgers with uL < 0 < uR).
 * @param {ScalarModel} m
 * @param {number} uL
 * @param {number} uR
 * @param {number} [delta=0] Harten entropy-fix width (0 = no fix)
 * @returns {number} flux through the face (normal +x)
 */
export function roeFlux(m, uL, uR, delta = 0) {
  const du = uR - uL;
  const a = Math.abs(du) > 1e-14 ? (m.f(uR) - m.f(uL)) / du : m.df(uL);
  let aa = Math.abs(a);
  if (delta > 0 && aa < delta) aa = (a * a + delta * delta) / (2 * delta); // Harten's fix
  return 0.5 * (m.f(uL) + m.f(uR)) - 0.5 * aa * du;
}

/** Registry used by the UI. */
export const SCALAR_FLUXES = {
  upwind: { label: 'Upwind', fn: upwindFlux },
  godunov: { label: 'Godunov (exact Riemann)', fn: godunovFlux },
  rusanov: { label: 'Rusanov / local Lax–Friedrichs', fn: rusanovFlux },
  roe: { label: 'Roe (no entropy fix)', fn: (m, a, b) => roeFlux(m, a, b, 0) },
  roefix: { label: 'Roe + Harten entropy fix', fn: (m, a, b) => roeFlux(m, a, b, 0.5) },
  central: { label: 'Central (unstable!)', fn: centralFlux },
};

/**
 * Exact solution of the Burgers Riemann problem, sampled at ξ = x/t.
 *  uL > uR: shock moving with the Rankine–Hugoniot speed s = (uL + uR)/2.
 *  uL < uR: rarefaction fan u = ξ for uL ≤ ξ ≤ uR.
 * @param {number} uL left state
 * @param {number} uR right state
 * @param {number} xi similarity variable ξ = x/t
 * @returns {number} u(ξ)
 */
export function burgersRiemann(uL, uR, xi) {
  if (uL > uR) return xi < 0.5 * (uL + uR) ? uL : uR;
  if (xi <= uL) return uL;
  if (xi >= uR) return uR;
  return xi;
}

/**
 * Exact entropy solution of the scalar Riemann problem for ANY continuous
 * flux f (convex or not), by Oleinik's convex-hull construction:
 *  - uL < uR: take the lower convex envelope of f on [uL, uR];
 *  - uL > uR: take the upper concave envelope of f on [uR, uL].
 * Walking from uL to uR, the envelope slopes increase; straight envelope
 * segments are shocks (speed = chord slope = Rankine–Hugoniot), parts where
 * the envelope touches f are rarefactions (u with f'(u) = ξ).
 * The envelope is computed on n+1 equally spaced samples (monotone chain),
 * so fan values are accurate to about (uR − uL)/n.
 * @param {ScalarModel} m
 * @param {number} uL
 * @param {number} uR
 * @param {number} [n=2000]
 * @returns {((xi: number) => number) & {shocks: number[], breaks: number[]}} ξ = x/t ↦ u; `shocks` = shock speeds,
 *   `breaks` = shock speeds plus the speeds of the first and last wave (jumps and kinks of the solution in ξ)
 */
export function scalarRiemann(m, uL, uR, n = 2000) {
  if (uL === uR) return Object.assign(() => uL, { shocks: [], breaks: [] });
  const us = [], fs = [];
  // chain of points (u, f(u)) along the path uL → uR with non-decreasing slopes
  for (let k = 0; k <= n; k++) {
    const u = uL + (uR - uL) * k / n, f = m.f(u);
    while (us.length >= 2) {
      const a = us.length - 2, b = us.length - 1;
      const s1 = (fs[b] - fs[a]) / (us[b] - us[a]), s2 = (f - fs[b]) / (u - us[b]);
      if (s1 > s2) { us.pop(); fs.pop(); } else break;
    }
    us.push(u); fs.push(f);
  }
  const sl = [];
  for (let k = 0; k + 1 < us.length; k++) sl.push((fs[k + 1] - fs[k]) / (us[k + 1] - us[k]));
  const R = (xi) => {
    // the state at ξ is the chain vertex k with slope_{k−1} ≤ ξ < slope_k
    let lo = 0, hi = sl.length; // first index with sl[k] > xi
    while (lo < hi) { const mid = (lo + hi) >> 1; if (sl[mid] > xi) hi = mid; else lo = mid + 1; }
    return us[lo];
  };
  // speeds of the shocks (envelope segments longer than one sample step) and of the two outermost waves:
  // the break points of the solution in ξ, e.g. for exact cell averages (x = ξ t)
  const du = Math.abs(uR - uL) / n;
  R.shocks = sl.filter((_, k) => Math.abs(us[k + 1] - us[k]) > 1.5 * du);
  R.breaks = [sl[0], ...R.shocks, sl[sl.length - 1]];
  return R;
}
