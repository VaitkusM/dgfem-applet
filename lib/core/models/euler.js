/**
 * @file The 1D Euler equations of gas dynamics and their numerical fluxes.
 *
 *   ∂t U + ∂x F(U) = 0,
 *   U = (ρ, m, E) with momentum m = ρu and total energy density E,
 *   F(U) = (ρu, ρu² + p, u(E + p)),
 *   ideal gas:  p = (γ − 1)(E − ½ρu²),  sound speed c = √(γp/ρ),  γ = 1.4 (air).
 *
 * Conventions
 *  - A single state is a length-3 array [ρ, m, E] ("conservative") or
 *    [ρ, u, p] ("primitive"). Grid data are interleaved Float64Arrays,
 *    U[3i + k] = component k of cell i.
 *  - Numerical fluxes have the signature  flux(UL, UR, out)  and write the
 *    flux through a face with normal +x (from the L state to the R state)
 *    into out[0..2]. Consistency: flux(U, U) = F(U).
 *  - Total enthalpy H = (E + p)/ρ.
 *
 * Reference for all formulas: E. F. Toro, "Riemann Solvers and Numerical
 * Methods for Fluid Dynamics", 3rd ed., Springer 2009 — chapters 3
 * (equations), 10 (HLL/HLLC), 11 (Roe) — cited below as [Toro, eq. …].
 */

import { godunovFluxEuler } from './eulerRiemann.js';

/** Ratio of specific heats for air. */
export const GAMMA = 1.4;

/**
 * Primitive → conservative.
 * @param {number} rho @param {number} u @param {number} p
 * @param {number[]|Float64Array} [out]
 * @param {number} [g=GAMMA]
 * @returns {number[]|Float64Array} [ρ, ρu, E]
 */
export function primToCons(rho, u, p, out = [0, 0, 0], g = GAMMA) {
  out[0] = rho; out[1] = rho * u; out[2] = p / (g - 1) + 0.5 * rho * u * u;
  return out;
}

/**
 * Conservative → primitive.
 * @param {ArrayLike<number>} U [ρ, m, E] (read from offset o)
 * @param {number[]|Float64Array} [out]
 * @param {number} [o=0] offset into U (for interleaved grid arrays)
 * @param {number} [g=GAMMA]
 * @returns {number[]|Float64Array} [ρ, u, p]
 */
export function consToPrim(U, out = [0, 0, 0], o = 0, g = GAMMA) {
  const rho = U[o], u = U[o + 1] / rho;
  out[0] = rho; out[1] = u; out[2] = (g - 1) * (U[o + 2] - 0.5 * rho * u * u);
  return out;
}

/** Pressure of a conservative state (offset o). */
export function pressure(U, o = 0, g = GAMMA) {
  return (g - 1) * (U[o + 2] - 0.5 * U[o + 1] * U[o + 1] / U[o]);
}

/** Sound speed c = √(γ p / ρ). */
export function soundSpeed(rho, p, g = GAMMA) {
  return Math.sqrt(g * p / rho);
}

/**
 * Physical flux F(U) = (m, m²/ρ + p, (E + p) m/ρ).
 * @param {ArrayLike<number>} U @param {number[]|Float64Array} out @param {number} [g=GAMMA]
 */
export function physFlux(U, out, g = GAMMA) {
  const rho = U[0], m = U[1], E = U[2], u = m / rho, p = (g - 1) * (E - 0.5 * m * u);
  out[0] = m; out[1] = m * u + p; out[2] = u * (E + p);
  return out;
}

/** Largest characteristic speed |u| + c of a conservative state at offset o. */
export function maxWaveSpeed(U, o = 0, g = GAMMA) {
  const rho = U[o], u = U[o + 1] / rho, p = (g - 1) * (U[o + 2] - 0.5 * rho * u * u);
  return Math.abs(u) + Math.sqrt(Math.max(0, g * p / rho));
}

// scratch arrays for the flux functions (single-threaded, not re-entrant across fluxes)
const FL = new Float64Array(3), FR = new Float64Array(3);

/**
 * Rusanov / local Lax–Friedrichs flux [Toro, eq. 10.55–10.56]:
 *   F̂ = ½(F(UL) + F(UR)) − ½ S⁺ (UR − UL),   S⁺ = max(|uL| + cL, |uR| + cR).
 * The most diffusive of the family: one wave speed for all three waves.
 */
export function rusanovFlux(UL, UR, out, g = GAMMA) {
  physFlux(UL, FL, g); physFlux(UR, FR, g);
  const S = Math.max(maxWaveSpeed(UL, 0, g), maxWaveSpeed(UR, 0, g));
  for (let k = 0; k < 3; k++) out[k] = 0.5 * (FL[k] + FR[k]) - 0.5 * S * (UR[k] - UL[k]);
  return out;
}

/**
 * Roe averages [Toro, eq. 11.60]: ũ, H̃ weighted by √ρ, c̃² = (γ−1)(H̃ − ½ũ²), ρ̃ = √(ρL ρR).
 * @returns {{u: number, H: number, c: number, rho: number}}
 */
export function roeAverage(UL, UR, g = GAMMA) {
  const rL = UL[0], rR = UR[0], uL = UL[1] / rL, uR = UR[1] / rR;
  const pL = (g - 1) * (UL[2] - 0.5 * rL * uL * uL), pR = (g - 1) * (UR[2] - 0.5 * rR * uR * uR);
  const HL = (UL[2] + pL) / rL, HR = (UR[2] + pR) / rR;
  const sL = Math.sqrt(rL), sR = Math.sqrt(rR);
  const u = (sL * uL + sR * uR) / (sL + sR), H = (sL * HL + sR * HR) / (sL + sR);
  return { u, H, c: Math.sqrt(Math.max(1e-300, (g - 1) * (H - 0.5 * u * u))), rho: sL * sR };
}

/**
 * Wave-speed estimates for HLL/HLLC: Einfeldt's choice [Toro, eq. 10.52]
 *   SL = min(uL − cL, ũ − c̃),  SR = max(uR + cR, ũ + c̃)
 * with Roe averages ũ, c̃ (this makes HLL "HLLE", positively conservative).
 * @returns {number[]} [SL, SR]
 */
export function waveSpeedEstimates(UL, UR, g = GAMMA) {
  const rL = UL[0], rR = UR[0], uL = UL[1] / rL, uR = UR[1] / rR;
  const cL = soundSpeed(rL, pressure(UL, 0, g), g), cR = soundSpeed(rR, pressure(UR, 0, g), g);
  const r = roeAverage(UL, UR, g);
  return [Math.min(uL - cL, r.u - r.c), Math.max(uR + cR, r.u + r.c)];
}

/**
 * HLL flux of Harten, Lax & van Leer (1983) [Toro, eq. 10.21 and 10.26]:
 * the Riemann fan is replaced by ONE constant intermediate state between
 * the fastest left wave SL and the fastest right wave SR:
 *   F̂ = F(UL)                                         if 0 ≤ SL
 *   F̂ = (SR F(UL) − SL F(UR) + SL SR (UR − UL)) / (SR − SL)   if SL ≤ 0 ≤ SR
 *   F̂ = F(UR)                                         if SR ≤ 0
 * The contact wave is ignored, so contacts are smeared like by Rusanov.
 */
export function hllFlux(UL, UR, out, g = GAMMA) {
  physFlux(UL, FL, g); physFlux(UR, FR, g);
  const [SL, SR] = waveSpeedEstimates(UL, UR, g);
  if (SL >= 0) { out[0] = FL[0]; out[1] = FL[1]; out[2] = FL[2]; return out; }
  if (SR <= 0) { out[0] = FR[0]; out[1] = FR[1]; out[2] = FR[2]; return out; }
  for (let k = 0; k < 3; k++) out[k] = (SR * FL[k] - SL * FR[k] + SL * SR * (UR[k] - UL[k])) / (SR - SL);
  return out;
}

/**
 * HLLC flux (Toro, Spruce & Speares 1994) [Toro, eqs. 10.37–10.39]: HLL with
 * the Contact restored — two intermediate states U*L, U*R separated by the
 * contact moving with speed
 *   S* = (pR − pL + ρL uL (SL − uL) − ρR uR (SR − uR)) / (ρL (SL − uL) − ρR (SR − uR)),
 *   U*K = ρK (SK − uK)/(SK − S*) · [1, S*, EK/ρK + (S* − uK)(S* + pK/(ρK (SK − uK)))],
 *   F*K = F(UK) + SK (U*K − UK),   K ∈ {L, R}.
 * Flux = F(UL), F*L, F*R or F(UR) depending on where x/t = 0 lies.
 */
export function hllcFlux(UL, UR, out, g = GAMMA) {
  const [SL, SR] = waveSpeedEstimates(UL, UR, g);
  if (SL >= 0) return physFlux(UL, out, g);
  if (SR <= 0) return physFlux(UR, out, g);
  const rL = UL[0], rR = UR[0], uL = UL[1] / rL, uR = UR[1] / rR;
  const pL = pressure(UL, 0, g), pR = pressure(UR, 0, g);
  const Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (rL * (SL - uL) - rR * (SR - uR));
  const useL = Ss >= 0;
  const U = useL ? UL : UR, S = useL ? SL : SR, r = useL ? rL : rR, u = useL ? uL : uR, p = useL ? pL : pR;
  physFlux(U, out, g);
  const f = r * (S - u) / (S - Ss);
  const Us0 = f, Us1 = f * Ss, Us2 = f * (U[2] / r + (Ss - u) * (Ss + p / (r * (S - u))));
  out[0] += S * (Us0 - U[0]); out[1] += S * (Us1 - U[1]); out[2] += S * (Us2 - U[2]);
  return out;
}

/**
 * Roe flux (Roe 1981) [Toro, eqs. 11.29, 11.58–11.68]: solve the LINEARISED
 * Riemann problem with the Roe matrix Ã(UL, UR) exactly:
 *   F̂ = ½(F(UL) + F(UR)) − ½ Σ_k |λ̃_k| α̃_k r̃_k,
 * eigenvalues λ̃ = (ũ − c̃, ũ, ũ + c̃), eigenvectors
 *   r̃1 = (1, ũ − c̃, H̃ − ũc̃),  r̃2 = (1, ũ, ½ũ²),  r̃3 = (1, ũ + c̃, H̃ + ũc̃),
 * wave strengths (Δq = qR − qL, ρ̃ = √(ρL ρR))
 *   α̃1 = (Δp − ρ̃ c̃ Δu)/(2c̃²),  α̃2 = Δρ − Δp/c̃²,  α̃3 = (Δp + ρ̃ c̃ Δu)/(2c̃²).
 * Entropy fix (Harten 1983, Harten & Hyman 1983 form): in the two acoustic
 * (genuinely nonlinear) fields, |λ| is replaced by (λ² + δ²)/(2δ) whenever
 * |λ| < δ, with δ = ε c̃. Without it (ε = 0) a transonic rarefaction is
 * computed as a stationary, entropy-violating "expansion shock".
 * @param {ArrayLike<number>} UL @param {ArrayLike<number>} UR @param {number[]|Float64Array} out
 * @param {number} [eps=0] entropy-fix width relative to c̃ (0 = no fix)
 */
export function roeFlux(UL, UR, out, eps = 0, g = GAMMA) {
  physFlux(UL, FL, g); physFlux(UR, FR, g);
  const rL = UL[0], rR = UR[0], uL = UL[1] / rL, uR = UR[1] / rR;
  const pL = pressure(UL, 0, g), pR = pressure(UR, 0, g);
  const { u, H, c, rho } = roeAverage(UL, UR, g);
  const dr = rR - rL, du = uR - uL, dp = pR - pL;
  const a1 = (dp - rho * c * du) / (2 * c * c), a2 = dr - dp / (c * c), a3 = (dp + rho * c * du) / (2 * c * c);
  const fix = (l) => {
    const al = Math.abs(l), d = eps * c;
    return d > 0 && al < d ? (l * l + d * d) / (2 * d) : al;
  };
  const l1 = fix(u - c), l2 = Math.abs(u), l3 = fix(u + c);
  out[0] = 0.5 * (FL[0] + FR[0]) - 0.5 * (l1 * a1 + l2 * a2 + l3 * a3);
  out[1] = 0.5 * (FL[1] + FR[1]) - 0.5 * (l1 * a1 * (u - c) + l2 * a2 * u + l3 * a3 * (u + c));
  out[2] = 0.5 * (FL[2] + FR[2]) - 0.5 * (l1 * a1 * (H - u * c) + l2 * a2 * 0.5 * u * u + l3 * a3 * (H + u * c));
  return out;
}

/** Registry used by the UI and the FV solver: key → {label, fn(UL, UR, out)}. */
export const EULER_FLUXES = {
  godunov: { label: 'Godunov (exact Riemann solver)', fn: (a, b, o) => godunovFluxEuler(a, b, o) },
  rusanov: { label: 'Rusanov (local Lax–Friedrichs)', fn: (a, b, o) => rusanovFlux(a, b, o) },
  hll: { label: 'HLL (Einfeldt speeds)', fn: (a, b, o) => hllFlux(a, b, o) },
  hllc: { label: 'HLLC', fn: (a, b, o) => hllcFlux(a, b, o) },
  roe: { label: 'Roe (no entropy fix)', fn: (a, b, o) => roeFlux(a, b, o, 0) },
  roefix: { label: 'Roe + Harten entropy fix', fn: (a, b, o) => roeFlux(a, b, o, 0.2) },
};
