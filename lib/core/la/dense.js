/**
 * @file Small dense linear algebra on row-major Float64Arrays.
 *
 * A dense n×m matrix is stored as a Float64Array `A` of length n*m with
 * A[i*m + j] = A_ij (row-major, like a C array or an image scanline buffer).
 * Functions take explicit sizes; there is no matrix class on purpose, so the
 * data layout is always obvious.
 *
 * Contents: products, LU with partial pivoting, Cholesky, inverse,
 * symmetric eigenvalues (cyclic Jacobi), and eigenvalues of general real
 * matrices (Hessenberg reduction + shifted QR, after EISPACK/Numerical
 * Recipes `elmhes`/`hqr`) — the latter is needed to plot DG/FR spectra.
 */

/** y = A x for an n×m matrix. */
export function matVec(A, n, m, x, y) {
  y = y || new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const o = i * m;
    for (let j = 0; j < m; j++) s += A[o + j] * x[j];
    y[i] = s;
  }
  return y;
}

/** C = A B with A n×k and B k×m. */
export function matMul(A, B, n, k, m) {
  const C = new Float64Array(n * m);
  for (let i = 0; i < n; i++)
    for (let l = 0; l < k; l++) {
      const a = A[i * k + l];
      if (a === 0) continue;
      for (let j = 0; j < m; j++) C[i * m + j] += a * B[l * m + j];
    }
  return C;
}

/** Transpose of an n×m matrix (returns m×n). */
export function transpose(A, n, m) {
  const T = new Float64Array(n * m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j * n + i] = A[i * m + j];
  return T;
}

/** n×n identity. */
export function eye(n) {
  const I = new Float64Array(n * n);
  for (let i = 0; i < n; i++) I[i * n + i] = 1;
  return I;
}

/**
 * LU factorisation with partial (row) pivoting: P A = L U.
 * @param {Float64Array} A n×n (not modified)
 * @param {number} n
 * @returns {{LU: Float64Array, piv: Int32Array, n: number, singular: boolean}}
 */
export function luFactor(A, n) {
  const LU = Float64Array.from(A), piv = new Int32Array(n);
  let singular = false;
  for (let k = 0; k < n; k++) {
    let p = k, best = Math.abs(LU[k * n + k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(LU[i * n + k]);
      if (v > best) { best = v; p = i; }
    }
    piv[k] = p;
    if (p !== k) {
      for (let j = 0; j < n; j++) {
        const t = LU[k * n + j]; LU[k * n + j] = LU[p * n + j]; LU[p * n + j] = t;
      }
    }
    const d = LU[k * n + k];
    if (d === 0) { singular = true; continue; }
    for (let i = k + 1; i < n; i++) {
      const f = (LU[i * n + k] /= d);
      if (f === 0) continue;
      for (let j = k + 1; j < n; j++) LU[i * n + j] -= f * LU[k * n + j];
    }
  }
  return { LU, piv, n, singular };
}

/**
 * Solve with an LU factorisation (overwrites and returns b's copy).
 * @param {{LU: Float64Array, piv: Int32Array, n: number}} F
 * @param {ArrayLike<number>} b
 * @returns {Float64Array}
 */
export function luSolve(F, b) {
  const { LU, piv, n } = F;
  const x = Float64Array.from(b);
  for (let k = 0; k < n; k++) {
    const p = piv[k];
    if (p !== k) { const t = x[k]; x[k] = x[p]; x[p] = t; }
  }
  for (let i = 0; i < n; i++) {
    let s = x[i];
    for (let j = 0; j < i; j++) s -= LU[i * n + j] * x[j];
    x[i] = s;
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let j = i + 1; j < n; j++) s -= LU[i * n + j] * x[j];
    x[i] = s / LU[i * n + i];
  }
  return x;
}

/** Solve A x = b (dense). */
export function solve(A, n, b) {
  return luSolve(luFactor(A, n), b);
}

/** Inverse of an n×n matrix via LU. */
export function inverse(A, n) {
  const F = luFactor(A, n), Inv = new Float64Array(n * n), e = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    e.fill(0); e[j] = 1;
    const c = luSolve(F, e);
    for (let i = 0; i < n; i++) Inv[i * n + j] = c[i];
  }
  return Inv;
}

/** Determinant via LU. */
export function det(A, n) {
  const { LU, piv } = luFactor(A, n);
  let d = 1;
  for (let k = 0; k < n; k++) { d *= LU[k * n + k]; if (piv[k] !== k) d = -d; }
  return d;
}

/**
 * Cholesky factorisation A = L Lᵀ of an SPD matrix.
 * @returns {Float64Array|null} L (row-major lower triangle) or null if A is not SPD
 */
export function cholesky(A, n) {
  const L = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    let s = A[j * n + j];
    for (let k = 0; k < j; k++) s -= L[j * n + k] * L[j * n + k];
    if (!(s > 0)) return null;
    const d = Math.sqrt(s);
    L[j * n + j] = d;
    for (let i = j + 1; i < n; i++) {
      let t = A[i * n + j];
      for (let k = 0; k < j; k++) t -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = t / d;
    }
  }
  return L;
}

/**
 * All eigenvalues (ascending) and eigenvectors of a symmetric matrix by the
 * cyclic Jacobi rotation method. O(n³) per sweep; fine for n ≲ 300.
 * @param {Float64Array} A symmetric n×n (not modified)
 * @param {number} n
 * @returns {{values: Float64Array, vectors: Float64Array}} vectors column-wise (V[i*n+k] = k-th eigenvector, i-th entry)
 */
export function symEig(A, n) {
  const a = Float64Array.from(A), V = eye(n);
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0, diag = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      if (i !== j) off += a[i * n + j] * a[i * n + j]; else diag += a[i * n + i] * a[i * n + i];
    }
    if (off <= 1e-30 * Math.max(diag, 1e-300)) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p * n + q];
      if (Math.abs(apq) < 1e-300) continue;
      const app = a[p * n + p], aqq = a[q * n + q];
      const theta = (aqq - app) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { // rotate columns p,q
        const akp = a[k * n + p], akq = a[k * n + q];
        a[k * n + p] = c * akp - s * akq;
        a[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) { // rotate rows p,q
        const apk = a[p * n + k], aqk = a[q * n + k];
        a[p * n + k] = c * apk - s * aqk;
        a[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k * n + p], vkq = V[k * n + q];
        V[k * n + p] = c * vkp - s * vkq;
        V[k * n + q] = s * vkp + c * vkq;
      }
    }
  }
  const idx = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i * n + i] - a[j * n + j]);
  const values = new Float64Array(n), vectors = new Float64Array(n * n);
  idx.forEach((k, col) => {
    values[col] = a[k * n + k];
    for (let i = 0; i < n; i++) vectors[i * n + col] = V[i * n + k];
  });
  return { values, vectors };
}

/**
 * Eigenvalues of a general real n×n matrix.
 * Step 1: reduce to upper Hessenberg form by stabilised elementary similarity
 * transforms (elmhes). Step 2: Francis double-shift QR iteration (hqr).
 * @param {Float64Array} A0 (not modified)
 * @param {number} n
 * @returns {{re: Float64Array, im: Float64Array}}
 */
export function eigGeneral(A0, n) {
  const a = Float64Array.from(A0);
  const A = (i, j) => a[i * n + j];
  const S = (i, j, v) => { a[i * n + j] = v; };
  // --- balance-free elmhes (0-based port of Numerical Recipes) ---
  for (let m = 1; m < n - 1; m++) {
    let x = 0, i = m;
    for (let j = m; j < n; j++) {
      if (Math.abs(A(j, m - 1)) > Math.abs(x)) { x = A(j, m - 1); i = j; }
    }
    if (i !== m) {
      for (let j = m - 1; j < n; j++) { const t = A(i, j); S(i, j, A(m, j)); S(m, j, t); }
      for (let j = 0; j < n; j++) { const t = A(j, i); S(j, i, A(j, m)); S(j, m, t); }
    }
    if (x !== 0) {
      for (i = m + 1; i < n; i++) {
        let y = A(i, m - 1);
        if (y !== 0) {
          y /= x;
          S(i, m - 1, y);
          for (let j = m; j < n; j++) S(i, j, A(i, j) - y * A(m, j));
          for (let j = 0; j < n; j++) S(j, m, A(j, m) + y * A(j, i));
        }
      }
    }
  }
  for (let i = 2; i < n; i++) for (let j = 0; j < i - 1; j++) S(i, j, 0);
  // --- hqr (0-based port) ---
  const wr = new Float64Array(n), wi = new Float64Array(n);
  let anorm = 0;
  for (let i = 0; i < n; i++) for (let j = Math.max(i - 1, 0); j < n; j++) anorm += Math.abs(A(i, j));
  let nn = n - 1, t = 0;
  let p = 0, q = 0, r = 0, s = 0, w = 0, x = 0, y = 0, z = 0;
  while (nn >= 0) {
    let its = 0, l;
    do {
      for (l = nn; l >= 1; l--) {
        s = Math.abs(A(l - 1, l - 1)) + Math.abs(A(l, l));
        if (s === 0) s = anorm;
        if (Math.abs(A(l, l - 1)) + s === s) { S(l, l - 1, 0); break; }
      }
      x = A(nn, nn);
      if (l === nn) { wr[nn] = x + t; wi[nn--] = 0; }
      else {
        y = A(nn - 1, nn - 1);
        w = A(nn, nn - 1) * A(nn - 1, nn);
        if (l === nn - 1) {
          p = 0.5 * (y - x);
          q = p * p + w;
          z = Math.sqrt(Math.abs(q));
          x += t;
          if (q >= 0) {
            z = p + (p >= 0 ? Math.abs(z) : -Math.abs(z));
            wr[nn - 1] = wr[nn] = x + z;
            if (z) wr[nn] = x - w / z;
            wi[nn - 1] = wi[nn] = 0;
          } else {
            wr[nn - 1] = wr[nn] = x + p;
            wi[nn - 1] = -(wi[nn] = z);
          }
          nn -= 2;
        } else {
          if (its === 60) throw new Error('eigGeneral: too many iterations');
          if (its === 10 || its === 20) {
            t += x;
            for (let i = 0; i <= nn; i++) S(i, i, A(i, i) - x);
            s = Math.abs(A(nn, nn - 1)) + Math.abs(A(nn - 1, nn - 2));
            y = x = 0.75 * s;
            w = -0.4375 * s * s;
          }
          ++its;
          let m;
          for (m = nn - 2; m >= l; m--) {
            z = A(m, m);
            r = x - z;
            s = y - z;
            p = (r * s - w) / A(m + 1, m) + A(m, m + 1);
            q = A(m + 1, m + 1) - z - r - s;
            r = A(m + 2, m + 1);
            s = Math.abs(p) + Math.abs(q) + Math.abs(r);
            p /= s; q /= s; r /= s;
            if (m === l) break;
            const u = Math.abs(A(m, m - 1)) * (Math.abs(q) + Math.abs(r));
            const v = Math.abs(p) * (Math.abs(A(m - 1, m - 1)) + Math.abs(z) + Math.abs(A(m + 1, m + 1)));
            if (u + v === v) break;
          }
          for (let i = m + 2; i <= nn; i++) {
            S(i, i - 2, 0);
            if (i !== m + 2) S(i, i - 3, 0);
          }
          for (let k = m; k <= nn - 1; k++) {
            if (k !== m) {
              p = A(k, k - 1);
              q = A(k + 1, k - 1);
              r = 0;
              if (k !== nn - 1) r = A(k + 2, k - 1);
              if ((x = Math.abs(p) + Math.abs(q) + Math.abs(r)) !== 0) {
                p /= x; q /= x; r /= x;
              }
            }
            const sq = Math.sqrt(p * p + q * q + r * r);
            if ((s = (p >= 0 ? sq : -sq)) !== 0) {
              if (k === m) {
                if (l !== m) S(k, k - 1, -A(k, k - 1));
              } else S(k, k - 1, -s * x);
              p += s;
              x = p / s; y = q / s; z = r / s; q /= p; r /= p;
              for (let j = k; j <= nn; j++) {
                p = A(k, j) + q * A(k + 1, j);
                if (k !== nn - 1) { p += r * A(k + 2, j); S(k + 2, j, A(k + 2, j) - p * z); }
                S(k + 1, j, A(k + 1, j) - p * y);
                S(k, j, A(k, j) - p * x);
              }
              const mmin = nn < k + 3 ? nn : k + 3;
              for (let i = l; i <= mmin; i++) {
                p = x * A(i, k) + y * A(i, k + 1);
                if (k !== nn - 1) { p += z * A(i, k + 2); S(i, k + 2, A(i, k + 2) - p * r); }
                S(i, k + 1, A(i, k + 1) - p * q);
                S(i, k, A(i, k) - p);
              }
            }
          }
        }
      }
    } while (l < nn - 1);
  }
  return { re: wr, im: wi };
}
