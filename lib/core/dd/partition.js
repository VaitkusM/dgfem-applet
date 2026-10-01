/**
 * @file Box partitions of the structured N×N grid into Kx × Ky subdomains,
 *       with and without overlap (chapter 10).
 *
 * Non-overlapping subdomains. The cell columns 0..N−1 are cut at
 *   cutsX[k] = round(k N / Kx),  k = 0..Kx   (same in y),
 * so subdomain (kx, ky) is the box of cells  cutsX[kx] ≤ ci < cutsX[kx+1],
 * cutsY[ky] ≤ cj < cutsY[ky+1]. Its width is H ≈ 1/Kx.
 *
 * Overlapping subdomains Ω'_k. Every box is extended by ℓ ≥ 1 layers of cells
 * on each side (clipped at ∂Ω). Two neighbours then overlap in a strip of width
 *   δ = 2 ℓ h .
 * The unknowns of the local problem on Ω'_k are the free vertices strictly
 * inside Ω'_k (the vertices on ∂Ω'_k carry the Dirichlet data of the local
 * problem — the latest values from the neighbours):
 *   cutsX[kx] − ℓ < i < cutsX[kx+1] + ℓ   (and 1 ≤ i ≤ N−1),  same for j.
 *
 * "Core" (owner) sets. Each free vertex is owned by exactly one subdomain: the
 * one with cutsX[kx] ≤ i < cutsX[kx+1] and cutsY[ky] ≤ j < cutsY[ky+1].
 * The 0/1 indicator of the core is the partition of unity D_k used by
 * restricted additive Schwarz (RAS):  Σ_k R_kᵀ D_k R_k = I.
 */
import { freeIndex } from './problem.js';

/**
 * @typedef {Object} Subdomain
 * @property {number} kx
 * @property {number} ky
 * @property {number[]} cells   [ci0, ci1, cj0, cj1] non-overlapping cell range (half-open)
 * @property {number[]} box     overlapping box [x0, x1, y0, y1] in world coordinates
 * @property {Int32Array} dofs  free indices of the local unknowns (ascending) — the rows of R_k
 * @property {Uint8Array} core  core[l] = 1 if dofs[l] is owned by this subdomain (diagonal of D_k)
 */

/**
 * @typedef {Object} BoxPartition
 * @property {number} N
 * @property {number} Kx
 * @property {number} Ky
 * @property {number} ell      overlap layers ℓ (δ = 2ℓh)
 * @property {Int32Array} cutsX length Kx+1
 * @property {Int32Array} cutsY length Ky+1
 * @property {Subdomain[]} subs subdomain k = ky Kx + kx
 * @property {Int32Array} owner free index → owning subdomain
 */

/** Cut positions round(k N / K), k = 0..K. */
export function cuts(N, K) {
  return Int32Array.from({ length: K + 1 }, (_, k) => Math.round((k * N) / K));
}

/**
 * Overlapping Kx × Ky box partition of the free vertices.
 * @param {number} N cells per direction
 * @param {number} Kx subdomains in x (1 ≤ Kx ≤ N/2)
 * @param {number} Ky subdomains in y
 * @param {number} ell overlap layers ℓ ≥ 1 (neighbours overlap by δ = 2ℓh)
 * @returns {BoxPartition}
 */
export function boxPartition(N, Kx, Ky, ell = 1) {
  if (ell < 1) throw new Error('boxPartition: ℓ must be ≥ 1 (with ℓ = 0 the interface vertices would belong to no subdomain)');
  const cutsX = cuts(N, Kx), cutsY = cuts(N, Ky), h = 1 / N;
  const owner = new Int32Array((N - 1) ** 2);
  const subs = [];
  for (let ky = 0; ky < Ky; ky++)
    for (let kx = 0; kx < Kx; kx++) {
      const k = ky * Kx + kx;
      const ci0 = cutsX[kx], ci1 = cutsX[kx + 1], cj0 = cutsY[ky], cj1 = cutsY[ky + 1];
      const i0 = Math.max(1, ci0 - ell + 1), i1 = Math.min(N - 1, ci1 + ell - 1);
      const j0 = Math.max(1, cj0 - ell + 1), j1 = Math.min(N - 1, cj1 + ell - 1);
      const dofs = [], core = [];
      for (let j = j0; j <= j1; j++)
        for (let i = i0; i <= i1; i++) {
          const f = freeIndex(N, i, j);
          dofs.push(f);
          const own = i >= ci0 && i < ci1 && j >= cj0 && j < cj1;
          core.push(own ? 1 : 0);
          if (own) owner[f] = k;
        }
      const box = [Math.max(0, ci0 - ell) * h, Math.min(N, ci1 + ell) * h, Math.max(0, cj0 - ell) * h, Math.min(N, cj1 + ell) * h];
      subs.push({ kx, ky, cells: [ci0, ci1, cj0, cj1], box, dofs: Int32Array.from(dofs), core: Uint8Array.from(core) });
    }
  return { N, Kx, Ky, ell, cutsX, cutsY, subs, owner };
}

/**
 * Non-overlapping splitting of the free vertices into subdomain interiors and
 * the interface Γ (= free vertices lying on a cut line x = cutsX[k]h or
 * y = cutsY[k]h, 0 < k < K), as used by the Schur complement method.
 * @param {number} N
 * @param {number} Kx
 * @param {number} Ky
 * @returns {{cutsX: Int32Array, cutsY: Int32Array, interiors: Int32Array[], gamma: Int32Array,
 *            isGamma: Uint8Array, cellSub: (ci:number, cj:number) => number}}
 *   interiors[k] = free indices strictly inside box k (ascending), gamma = interface free indices (ascending),
 *   cellSub(ci,cj) = subdomain index ky Kx + kx owning cell (ci,cj)
 */
export function interfacePartition(N, Kx, Ky) {
  const cutsX = cuts(N, Kx), cutsY = cuts(N, Ky);
  const onCutX = new Uint8Array(N + 1), onCutY = new Uint8Array(N + 1);
  for (let k = 1; k < Kx; k++) onCutX[cutsX[k]] = 1;
  for (let k = 1; k < Ky; k++) onCutY[cutsY[k]] = 1;
  const which = (c, K, cs) => { let k = 0; while (k < K - 1 && c >= cs[k + 1]) k++; return k; };
  const interiors = Array.from({ length: Kx * Ky }, () => []);
  const gamma = [], isGamma = new Uint8Array((N - 1) ** 2);
  for (let j = 1; j < N; j++)
    for (let i = 1; i < N; i++) {
      const f = freeIndex(N, i, j);
      if (onCutX[i] || onCutY[j]) { gamma.push(f); isGamma[f] = 1; }
      else interiors[which(j, Ky, cutsY) * Kx + which(i, Kx, cutsX)].push(f);
    }
  const cellSub = (ci, cj) => which(cj, Ky, cutsY) * Kx + which(ci, Kx, cutsX);
  return { cutsX, cutsY, interiors: interiors.map((a) => Int32Array.from(a)), gamma: Int32Array.from(gamma), isGamma, cellSub };
}
