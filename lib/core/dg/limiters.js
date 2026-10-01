/**
 * @file Slope limiter for 1D nodal DG (minmod / TVB, in the form of
 *       Cockburn & Shu and Hesthaven & Warburton, "SlopeLimitN").
 *
 * Near discontinuities high-order polynomials oscillate (Gibbs phenomenon) and
 * the oscillations can grow (overshoots, negative densities, …). A slope
 * limiter post-processes the solution after every RK stage:
 *
 *  1. In element j compute the cell average ū_j and the edge values
 *     u⁺_{j−½} (left end) and u⁻_{j+½} (right end); with Δ⁺ = ū_{j+1} − ū_j and
 *     Δ⁻ = ū_j − ū_{j−1}, the element is TROUBLED unless both edge deviations
 *     already pass the minmod test:
 *        ū_j + m̃(u⁻_{j+½} − ū_j, Δ⁻, Δ⁺) = u⁻_{j+½}   and   ū_j − m̃(ū_j − u⁺_{j−½}, Δ⁻, Δ⁺) = u⁺_{j−½}.
 *  2. In troubled elements keep only the linear part ū_j + s_j ξ (s_j = Legendre
 *     coefficient of P_1 = ξ, so the edge values of the linear part are ū_j ± s_j)
 *     and replace its slope by
 *        s_j ← m̃(s_j, Δ⁻/2, Δ⁺/2),
 *     i.e. in physical units u_x ← minmod(u_x, Δ⁻/h, Δ⁺/h) — the MUSCL minmod slope.
 *  Smooth elements are left untouched (including all higher modes).
 *
 * minmod(a,b,c) = s·min(|a|,|b|,|c|) if a,b,c all have the same sign s, else 0.
 * TVB modification (Shu 1987): m̃(a, b, c) = a if |a| ≤ M h², else minmod(a, b, c).
 * M = 0 gives the TVD(M) minmod limiter, which also clips smooth extrema (accuracy
 * drops to first order there); M > 0 (≈ the size of |u_xx| at extrema) leaves
 * smooth extrema alone.
 *
 * The limiter never changes cell averages, so it preserves conservation.
 */
import { vandermonde } from './dg1d.js';
import { inverse } from '../la/dense.js';
import { legendreAll } from '../basis/legendre.js';

/** minmod of three numbers. */
export function minmod3(a, b, c) {
  if (a > 0 && b > 0 && c > 0) return Math.min(a, b, c);
  if (a < 0 && b < 0 && c < 0) return Math.max(a, b, c);
  return 0;
}

/** TVB-modified minmod m̃(a,b,c) with threshold Mh2 = M h². */
export function minmodTVB(a, b, c, Mh2) {
  return Math.abs(a) <= Mh2 ? a : minmod3(a, b, c);
}

/**
 * Create a slope limiter for a DG discretisation from makeDG1D (or makeFR1D).
 * @param {{p: number, N: number, np: number, h: number, r: Float64Array, periodic: boolean}} dg
 * @param {{M?: number}} [o] TVB constant M ≥ 0 (default 0 = pure minmod)
 * @returns {{apply: (u: Float64Array) => number, troubled: Uint8Array}} apply() limits u in place and
 *   returns the number of troubled (modified) elements; `troubled[k]` = 1 for those of the last call.
 */
export function makeSlopeLimiter(dg, o = {}) {
  const { N, np, h, r } = dg;
  const Mh2 = (o.M ?? 0) * h * h;
  const V = vandermonde(r), Vinv = inverse(V, np);
  const avg = new Float64Array(N), slope = new Float64Array(N), eL = new Float64Array(N), eR = new Float64Array(N);
  const troubled = new Uint8Array(N);
  // Legendre values at the ends: P_j(−1) = (−1)^j, P_j(1) = 1
  const PL = legendreAll(np - 1, -1).P;
  function apply(u) {
    if (np === 1) return 0; // p = 0: piecewise constants need no limiting
    for (let k = 0; k < N; k++) {
      let a0 = 0, a1 = 0, el = 0, er = 0;
      for (let j = 0; j < np; j++) {
        // modal coefficients û = V⁻¹ u; edges via Σ û_j P_j(±1)
        let c = 0;
        for (let i = 0; i < np; i++) c += Vinv[j * np + i] * u[k * np + i];
        if (j === 0) a0 = c; else if (j === 1) a1 = c;
        el += c * PL[j]; er += c;
      }
      avg[k] = a0; slope[k] = a1; eL[k] = el; eR[k] = er;
    }
    let count = 0;
    for (let k = 0; k < N; k++) {
      const km = k > 0 ? k - 1 : (dg.periodic ? N - 1 : k);
      const kp = k < N - 1 ? k + 1 : (dg.periodic ? 0 : k);
      const dm = avg[k] - avg[km], dp = avg[kp] - avg[k];
      const okR = Math.abs(minmodTVB(eR[k] - avg[k], dm, dp, Mh2) - (eR[k] - avg[k])) < 1e-13 * (1 + Math.abs(avg[k]));
      const okL = Math.abs(minmodTVB(avg[k] - eL[k], dm, dp, Mh2) - (avg[k] - eL[k])) < 1e-13 * (1 + Math.abs(avg[k]));
      troubled[k] = okR && okL ? 0 : 1;
      if (!troubled[k]) continue;
      count++;
      const s = minmodTVB(slope[k], 0.5 * dm, 0.5 * dp, Mh2);
      for (let i = 0; i < np; i++) u[k * np + i] = avg[k] + s * r[i];
    }
    return count;
  }
  return { apply, troubled };
}
