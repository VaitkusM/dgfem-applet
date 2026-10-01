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

/** @typedef {{name: string, f: (u:number)=>number, df: (u:number)=>number, convex?: boolean}} ScalarModel */

/** Linear advection with speed a. @returns {ScalarModel} */
export const advection = (a = 1) => ({ name: 'advection', a, f: (u) => a * u, df: () => a, convex: false, linear: true });

/** Inviscid Burgers. @type {ScalarModel} */
export const burgers = { name: 'burgers', f: (u) => 0.5 * u * u, df: (u) => u, convex: true };

/** Buckley–Leverett (two-phase flow in porous media), mobility ratio M. @returns {ScalarModel} */
export const buckley = (M = 0.5) => ({
  name: 'buckley',
  f: (u) => (u * u) / (u * u + M * (1 - u) * (1 - u)),
  df: (u) => {
    const d = u * u + M * (1 - u) * (1 - u);
    return (2 * u * d - u * u * (2 * u - 2 * M * (1 - u))) / (d * d);
  },
  convex: false,
});

/**
 * Exact Godunov flux = f(u*(0)), the flux of the exact Riemann solution
 * at the face. For any continuous scalar flux (Osher's formula):
 *   uL ≤ uR:  min_{u ∈ [uL,uR]} f(u)     uL > uR:  max_{u ∈ [uR,uL]} f(u)
 * Exact closed form for advection and Burgers; dense sampling otherwise.
 * @param {ScalarModel} m
 * @param {number} uL
 * @param {number} uR
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

/** Upwind flux for linear advection: take the value the wind comes from. */
export function upwindFlux(m, uL, uR) {
  const a = m.linear ? m.a : m.df(0.5 * (uL + uR));
  return a >= 0 ? m.f(uL) : m.f(uR);
}

/** Central flux ½(f(uL)+f(uR)) — consistent but unstable with forward Euler. */
export function centralFlux(m, uL, uR) {
  return 0.5 * (m.f(uL) + m.f(uR));
}

/**
 * Local Lax–Friedrichs / Rusanov flux:
 *   F̂ = ½(f(uL) + f(uR)) − ½ α (uR − uL),  α = max |f'(u)| over both states
 * (for non-convex f we also sample between the states).
 */
export function rusanovFlux(m, uL, uR) {
  let alpha = Math.max(Math.abs(m.df(uL)), Math.abs(m.df(uR)));
  if (!m.convex && !m.linear) for (let k = 1; k < 8; k++) alpha = Math.max(alpha, Math.abs(m.df(uL + (uR - uL) * k / 8)));
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
 * @returns {(xi: number) => number} ξ = x/t ↦ u
 */
export function scalarRiemann(m, uL, uR, n = 2000) {
  if (uL === uR) return () => uL;
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
  return (xi) => {
    // the state at ξ is the chain vertex k with slope_{k−1} ≤ ξ < slope_k
    let lo = 0, hi = sl.length; // first index with sl[k] > xi
    while (lo < hi) { const mid = (lo + hi) >> 1; if (sl[mid] > xi) hi = mid; else lo = mid + 1; }
    return us[lo];
  };
}
