/**
 * @file Write conflicts ("race conditions") in parallel assembly, and the two
 *       classic cures — graph colouring and a face buffer + gather (DOM-free).
 *
 * Model of execution used for the demonstrations: the threads of one *group*
 * run concurrently and we take the worst-case interleaving of their
 * read–modify–write sequences: first ALL threads load the old values they need,
 * then ALL compute, then ALL store. If several threads store to the same address,
 * the last one in thread order wins (on real hardware the winner is unspecified,
 * and with luckier interleavings fewer updates are lost — the result is then
 * merely unpredictable, which is just as bad). Groups run one after the other
 * (separate dispatches, with a barrier in between).
 */

/**
 * Concurrent (worst-case interleaved) scatter-add: each thread k adds vals[k][m] to
 * out[targets[k][m]] for all m — loads of all threads of a group happen before any store.
 * @param {number} nOut size of the output array
 * @param {number[][]} targets targets[k] = output indices thread k adds to (distinct within a thread)
 * @param {number[][]} vals vals[k][m] = amount added by thread k to targets[k][m]
 * @param {number[][]} groups lists of thread ids running concurrently; groups run sequentially
 * @returns {{out: Float64Array, lost: number}} lost = number of additions that were overwritten
 */
export function lockstepScatter(nOut, targets, vals, groups) {
  const out = new Float64Array(nOut);
  let lost = 0;
  for (const G of groups) {
    const loaded = G.map((k) => targets[k].map((t) => out[t]));    // phase 1: every thread loads
    const written = new Set();
    G.forEach((k, a) => {                                            // phase 2: every thread stores
      targets[k].forEach((t, m) => {
        if (written.has(t)) lost++;                                  // overwrites another thread's update
        written.add(t);
        out[t] = loaded[a][m] + vals[k][m];
      });
    });
  }
  return { out, lost };
}

/**
 * Greedy colouring of elements such that two elements sharing a vertex get different colours
 * (then all elements of one colour can scatter to their vertices in parallel without conflicts).
 * @param {Int32Array|number[]} conn element→vertex connectivity, `k` vertices per element
 * @param {number} k vertices per element
 * @returns {{color: Int32Array, nColors: number}}
 */
export function colorElements(conn, k) {
  const nE = conn.length / k;
  const vertElems = new Map();
  for (let e = 0; e < nE; e++) for (let a = 0; a < k; a++) {
    const v = conn[e * k + a];
    if (!vertElems.has(v)) vertElems.set(v, []);
    vertElems.get(v).push(e);
  }
  const color = new Int32Array(nE).fill(-1);
  let nColors = 0;
  for (let e = 0; e < nE; e++) {
    const used = new Set();
    for (let a = 0; a < k; a++) for (const f of vertElems.get(conn[e * k + a])) if (color[f] >= 0) used.add(color[f]);
    let c = 0;
    while (used.has(c)) c++;
    color[e] = c;
    nColors = Math.max(nColors, c + 1);
  }
  return { color, nColors };
}

/**
 * Check a colouring: no two elements of the same colour share a vertex.
 * @param {Int32Array|number[]} conn @param {number} k @param {Int32Array} color
 */
export function isValidColoring(conn, k, color) {
  const seen = new Map();
  for (let e = 0; e < conn.length / k; e++) for (let a = 0; a < k; a++) {
    const key = `${conn[e * k + a]}:${color[e]}`;
    if (seen.has(key) && seen.get(key) !== e) return false;
    seen.set(key, e);
  }
  return true;
}

/**
 * Faces of a structured nx×ny quad grid (non-periodic), for the DG face-buffer demo.
 * Interior faces only; face f has a "minus" element (left/bottom) and a "plus" element.
 * @param {number} nx @param {number} ny
 * @returns {{minus: Int32Array, plus: Int32Array, vertical: Uint8Array, nFace: number}} vertical[f] = 1 for faces x = const
 */
export function gridFaces(nx, ny) {
  const minus = [], plus = [], vertical = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i + 1 < nx; i++) { minus.push(j * nx + i); plus.push(j * nx + i + 1); vertical.push(1); }
  for (let j = 0; j + 1 < ny; j++) for (let i = 0; i < nx; i++) { minus.push(j * nx + i); plus.push((j + 1) * nx + i); vertical.push(0); }
  return { minus: Int32Array.from(minus), plus: Int32Array.from(plus), vertical: Uint8Array.from(vertical), nFace: minus.length };
}

/**
 * Face-buffer assembly of the net outflow per element:  R_e = Σ_{faces} ±F_f
 * (+F_f for the minus element, −F_f for the plus element).
 * Pass 1 (one thread per face) writes F_f into its own slot; pass 2 (one thread per element)
 * gathers its faces. Every address has exactly one writer: no race, deterministic result.
 * @param {number} nElem @param {{minus: Int32Array, plus: Int32Array, nFace: number}} faces @param {Float64Array} F
 * @returns {Float64Array} R
 */
export function faceBufferGather(nElem, faces, F) {
  const slot = new Float64Array(faces.nFace);
  for (let f = 0; f < faces.nFace; f++) slot[f] = F[f];          // pass 1 (parallel over faces)
  const elemFaces = Array.from({ length: nElem }, () => []);       // static incidence (built once)
  for (let f = 0; f < faces.nFace; f++) { elemFaces[faces.minus[f]].push([f, 1]); elemFaces[faces.plus[f]].push([f, -1]); }
  const R = new Float64Array(nElem);
  for (let e = 0; e < nElem; e++) for (const [f, s] of elemFaces[e]) R[e] += s * slot[f]; // pass 2 (parallel over elements)
  return R;
}
