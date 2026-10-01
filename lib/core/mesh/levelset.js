/**
 * @file Implicit geometry for immersed (unfitted) methods: level-set
 *       functions, closest points, normals and distance vectors.
 *
 * A domain is described implicitly by a level-set function φ:
 *     Ω = {x : φ(x) < 0},    Γ = ∂Ω = {x : φ(x) = 0}.
 * Nothing about the computational mesh knows Ω; every method of chapter 11
 * only *queries* φ (sign, value) and, for the Shifted Boundary Method, the
 * closest point on Γ.
 *
 * Two shapes are provided (both star-shaped with respect to a centre c,
 * with optional rotation ρ):
 *  - circle  : φ(x) = |x − c| − R. This is the exact SIGNED DISTANCE
 *              FUNCTION (SDF): |φ(x)| = dist(x, Γ) and |∇φ| = 1.
 *  - flower  : boundary r = R(θ) := R (1 + a cos(k(θ − ρ))) in polar
 *              coordinates (r, θ) around c, and φ(x) = r − R(θ).
 *              This φ has the right SIGN (negative inside, positive
 *              outside, zero exactly on Γ) but it is NOT a signed distance
 *              function: |φ| ≠ dist(x, Γ) and |∇φ| ≠ 1 in general, and φ is
 *              not differentiable at the centre r = 0. The exact distance
 *              is obtained from the closest-point projection below.
 *
 * Every shape object exposes:
 *   phi(x, y)              level-set value
 *   grad(x, y) → [gx, gy]  gradient of φ (not normalised)
 *   closest(x, y) → {x, y, t, dist, nx, ny}
 *        closest point on Γ, its curve parameter t (polar angle for the
 *        parametrisations used here), the unsigned distance and the OUTWARD
 *        unit normal of Γ at that point
 *   distVec(x, y) → [dx, dy]  distance vector d = x_Γ(x) − x (points from x
 *                              to its closest boundary point)
 *   point(t) → [x, y], tangent(t) → [x', y']  parametrisation of Γ (CCW); for both
 *                          shapes the parameter t is the polar angle around the centre
 *   param(x, y) → t        parameter of a point ON Γ (its polar angle)
 *   area, perimeter        exact area; perimeter (exact for the circle,
 *                          spectrally accurate trapezoidal rule for the flower)
 *   polyline(n) → Float64Array of n+1 points (x0,y0,…) closing the curve
 *   isSDF                  true only for the circle
 */

const TWO_PI = 2 * Math.PI;

/**
 * Circle of radius R centred at (cx, cy); φ is the exact signed distance.
 * @param {{cx?: number, cy?: number, R?: number}} [o]
 */
export function circleLS(o = {}) {
  const cx = o.cx ?? 0.5, cy = o.cy ?? 0.5, R = o.R ?? 0.3;
  return {
    kind: 'circle', cx, cy, R, isSDF: true,
    phi: (x, y) => Math.hypot(x - cx, y - cy) - R,
    grad: (x, y) => {
      const r = Math.hypot(x - cx, y - cy) || 1e-300;
      return [(x - cx) / r, (y - cy) / r];
    },
    closest(x, y) {
      let dx = x - cx, dy = y - cy;
      let r = Math.hypot(dx, dy);
      if (r < 1e-300) { dx = 1; dy = 0; r = 1; } // centre: every direction is closest; pick +x
      const nx = dx / r, ny = dy / r;
      const px = cx + R * nx, py = cy + R * ny;
      return { x: px, y: py, t: Math.atan2(ny, nx), dist: Math.hypot(px - x, py - y), nx, ny };
    },
    distVec(x, y) { const c = this.closest(x, y); return [c.x - x, c.y - y]; },
    point: (t) => [cx + R * Math.cos(t), cy + R * Math.sin(t)],
    tangent: (t) => [-R * Math.sin(t), R * Math.cos(t)],
    param: (x, y) => Math.atan2(y - cy, x - cx),
    area: Math.PI * R * R,
    perimeter: TWO_PI * R,
    polyline(n = 256) { return polyline(this, n); },
  };
}

/**
 * "Flower" r < R(θ) = R (1 + a cos(k (θ − rot))) around (cx, cy).
 * Requires 0 ≤ a < 1 so that R(θ) > 0. Area = ½∫₀^{2π} R(θ)² dθ = πR²(1 + a²/2) for k ≥ 1.
 * @param {{cx?: number, cy?: number, R?: number, a?: number, k?: number, rot?: number}} [o]
 */
export function flowerLS(o = {}) {
  const cx = o.cx ?? 0.5, cy = o.cy ?? 0.5, R0 = o.R ?? 0.3, a = o.a ?? 0.2, k = o.k ?? 5, rot = o.rot ?? 0;
  const Rf = (t) => R0 * (1 + a * Math.cos(k * (t - rot)));
  const dRf = (t) => -R0 * a * k * Math.sin(k * (t - rot));
  const d2Rf = (t) => -R0 * a * k * k * Math.cos(k * (t - rot));
  const point = (t) => { const r = Rf(t); return [cx + r * Math.cos(t), cy + r * Math.sin(t)]; };
  // X'(t) = R'(t) (cos t, sin t) + R(t) (−sin t, cos t)
  const tangent = (t) => {
    const r = Rf(t), dr = dRf(t), c = Math.cos(t), s = Math.sin(t);
    return [dr * c - r * s, dr * s + r * c];
  };
  // X''(t) = R'' (cos, sin) + 2R' (−sin, cos) − R (cos, sin)
  const second = (t) => {
    const r = Rf(t), dr = dRf(t), d2 = d2Rf(t), c = Math.cos(t), s = Math.sin(t);
    return [d2 * c - 2 * dr * s - r * c, d2 * s + 2 * dr * c - r * s];
  };
  // perimeter ∫|X'(t)| dt: periodic trapezoidal rule (spectrally accurate)
  let per = 0;
  const M = 4096;
  for (let i = 0; i < M; i++) { const [tx, ty] = tangent(TWO_PI * i / M); per += Math.hypot(tx, ty); }
  per *= TWO_PI / M;
  const nSample = Math.max(64, 16 * k);

  return {
    kind: 'flower', cx, cy, R: R0, a, k, rot, isSDF: false,
    radius: Rf,
    phi(x, y) {
      const dx = x - cx, dy = y - cy;
      return Math.hypot(dx, dy) - Rf(Math.atan2(dy, dx));
    },
    // ∇φ = ∇r − R'(θ) ∇θ with ∇r = (cos θ, sin θ), ∇θ = (−sin θ, cos θ)/r
    grad(x, y) {
      const dx = x - cx, dy = y - cy, r = Math.hypot(dx, dy) || 1e-300;
      const c = dx / r, s = dy / r, dr = dRf(Math.atan2(dy, dx));
      return [c + dr * s / r, s - dr * c / r];
    },
    /**
     * Closest point by minimising D(t) = ½|X(t) − p|² over the curve
     * parameter t: sample D on nSample angles, start a safeguarded Newton
     * iteration for D'(t) = (X − p)·X' = 0 from EVERY sampled local minimum
     * (several initial guesses — the flower's non-convex petals produce
     * several local minima), and keep the best converged point.
     */
    closest(x, y) {
      const D = (t) => { const [X, Y] = point(t); return 0.5 * ((X - x) ** 2 + (Y - y) ** 2); };
      const dt = TWO_PI / nSample;
      const vals = new Float64Array(nSample);
      for (let i = 0; i < nSample; i++) vals[i] = D(i * dt);
      let best = { t: 0, D: Infinity };
      for (let i = 0; i < nSample; i++) {
        const vm = vals[(i - 1 + nSample) % nSample], vp = vals[(i + 1) % nSample];
        if (!(vals[i] <= vm && vals[i] <= vp)) continue; // not a local minimum of the samples
        let t = i * dt;
        for (let it = 0; it < 30; it++) {
          const [X, Y] = point(t), [tx, ty] = tangent(t), [sx, sy] = second(t);
          const g1 = (X - x) * tx + (Y - y) * ty;          // D'(t)
          const g2 = tx * tx + ty * ty + (X - x) * sx + (Y - y) * sy; // D''(t)
          let step = g2 > 1e-14 ? -g1 / g2 : -Math.sign(g1) * 0.25 * dt;
          step = Math.max(-dt, Math.min(dt, step)); // stay near the bracket
          t += step;
          if (Math.abs(step) < 1e-15) break;
        }
        const v = D(t);
        if (v < best.D) best = { t, D: v };
      }
      const t = best.t, [X, Y] = point(t), [tx, ty] = tangent(t), L = Math.hypot(tx, ty);
      // CCW curve: outward normal = tangent rotated clockwise
      return { x: X, y: Y, t, dist: Math.hypot(X - x, Y - y), nx: ty / L, ny: -tx / L };
    },
    distVec(x, y) { const c = this.closest(x, y); return [c.x - x, c.y - y]; },
    point, tangent,
    param: (x, y) => Math.atan2(y - cy, x - cx),
    area: Math.PI * R0 * R0 * (1 + 0.5 * a * a),
    perimeter: per,
    polyline(n = 512) { return polyline(this, n); },
  };
}

/** Closed polyline of n segments along a shape's parametrisation. */
function polyline(shape, n) {
  const P = new Float64Array(2 * (n + 1));
  for (let i = 0; i <= n; i++) { const [x, y] = shape.point(TWO_PI * i / n); P[2 * i] = x; P[2 * i + 1] = y; }
  return P;
}

/**
 * Build a shape from a plain description (used by widgets and tests).
 * @param {{kind: 'circle'|'flower', cx?: number, cy?: number, R?: number, a?: number, k?: number, rot?: number}} o
 */
export function makeShape(o) {
  return o.kind === 'flower' ? flowerLS(o) : circleLS(o);
}
