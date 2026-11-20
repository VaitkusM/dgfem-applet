/**
 * @file Back-of-the-envelope performance model for the chapter-12 kernels (DOM-free).
 *
 * Counts are per degree of freedom (DOF) and per RK *stage*, for the kernels in
 * lib/gpu/kernels.js, n = p + 1 nodes per direction, values of `bytesPerValue` bytes
 * (4 for f32). FLOPs are the floating-point additions/multiplications as written in
 * the WGSL source (index arithmetic is integer work and not counted; a compare/select is
 * not a FLOP). Bytes are DRAM traffic under the *ideal-cache* assumption: every array
 * element is loaded from / stored to global memory once per kernel, even if several
 * threads use it (the other uses hit in cache). Real kernels move somewhat more.
 */

/**
 * Cost of each kernel per DOF and per stage.
 * @param {number} p polynomial degree (≥ 1)
 * @param {{bytesPerValue?: number}} [o]
 * @returns {Array<{name: string, flops: number, bytes: number, ai: number}>} ai = flops / bytes (arithmetic intensity)
 */
export function kernelCosts(p, o = {}) {
  const b = o.bytesPerValue ?? 4, n = p + 1;
  const faceNodesPerDof = 2 / n; // 2 faces (right, top) of n nodes are owned per element of n² DOFs
  const list = [
    // volume: 2 directions × n multiply-adds (2n FLOPs each) + 4 multiplications and 1 addition
    { name: 'volume', flops: 4 * n + 5, bytes: 2 * b },                                         // read u, write rhs
    // surface: one multiplication per face node; read u⁻, u⁺, write F̂
    { name: 'surface', flops: 1 * faceNodesPerDof, bytes: 3 * b * faceNodesPerDof },
    // lift: per face term (4n per element) 4 FLOPs (mul, sub, mul, add) + 2 (cx, cy) per DOF;
    // read u, rhs, write rhs + the face buffer once (2n values per element)
    { name: 'lift', flops: (4 * 4 * n) / (n * n) + 2, bytes: 3 * b + b * faceNodesPerDof },
    // rkStage: k = A k + dt r (3), uOut = uIn + B k (2); read res, rhs, uIn, write res, uOut
    { name: 'rkStage', flops: 5, bytes: 5 * b },
  ];
  // fused: one kernel per stage reading u and res, writing uOut and res (neighbour face values are
  // u values, cached); each face flux is computed by both neighbours (2× surface FLOPs)
  const f = list.reduce((s, k) => s + k.flops, 0) + list[1].flops;
  list.push({ name: 'fused', flops: f, bytes: 4 * b });
  for (const k of list) k.ai = k.flops / k.bytes;
  return list;
}

/**
 * Totals of the unfused stage: Σ FLOPs, Σ bytes (per DOF per stage).
 * @param {number} p @param {{bytesPerValue?: number}} [o]
 * @returns {{flops: number, bytes: number}}
 */
export function stageTotals(p, o) {
  const k = kernelCosts(p, o).filter((x) => x.name !== 'fused');
  return { flops: k.reduce((s, x) => s + x.flops, 0), bytes: k.reduce((s, x) => s + x.bytes, 0) };
}

/**
 * FLOPs per element for applying ONE partial derivative (∂/∂ξ_1) to the nodal values
 * on a tensor-product element in d dimensions with n = p+1 nodes per direction.
 *  - dense:  the derivative as an n^d × n^d matrix: 2·n^{2d} FLOPs  → O(p^{2d})
 *  - sum factorised: D acts along one index only (D ⊗ I ⊗ …): n^{d−1} lines × (n×n mat-vec) = 2·n^{d+1} → O(p^{d+1})
 * @param {number} p @param {number} d
 * @returns {{dense: number, sumfact: number}}
 */
export function derivativeFlops(p, d) {
  const n = p + 1;
  return { dense: 2 * n ** (2 * d), sumfact: 2 * n ** (d + 1) };
}

/**
 * Roofline: attainable performance  min(peak, bandwidth × intensity).
 * @param {number} ai arithmetic intensity [FLOP/byte]
 * @param {number} peak peak compute [FLOP/s]
 * @param {number} bw memory bandwidth [byte/s]
 * @returns {number} [FLOP/s]
 */
export const roofline = (ai, peak, bw) => Math.min(peak, bw * ai);

/**
 * Number of memory transactions (aligned segments of `segFloats` values) touched
 * by a set of addresses (in units of values), e.g. the loads of one warp.
 * @param {ArrayLike<number>} addrs @param {number} segFloats
 * @returns {number}
 */
export function transactions(addrs, segFloats) {
  const s = new Set();
  for (const a of addrs) s.add(Math.floor(a / segFloats));
  return s.size;
}

/**
 * Addresses read by the threads t = 0..warp−1 of one warp when they load field `field`
 * of "their" item, for an array of `nItems` items with `nFields` fields each.
 *  - layout 'aos' (array of structures): item k's fields are contiguous → addr = k·nFields + field
 *  - layout 'soa' (structure of arrays): one array per field → addr = field·nItems + k
 * and the item read by thread t is  item = t · stride  (stride 1: thread per node;
 * stride Np: thread per element reading its first node).
 * @param {{layout: 'aos'|'soa', nFields: number, nItems: number, field: number, warp: number, stride: number, first?: number}} o
 * @returns {number[]}
 */
export function warpAddresses(o) {
  const out = [];
  for (let t = 0; t < o.warp; t++) {
    const k = (o.first ?? 0) + t * o.stride;
    out.push(o.layout === 'aos' ? k * o.nFields + o.field : o.field * o.nItems + k);
  }
  return out;
}

/**
 * SIMT execution statistics of a kernel "program" (the model behind the thread-mapping widget).
 *
 * `nThreads` threads are launched in workgroups of `workgroup` threads; consecutive threads form
 * warps of `W` lanes that execute in lockstep. A program is a list of lines, each with an issue
 * cost (instruction slots) and a predicate telling which threads execute it (an `if` body is
 * executed only by the threads whose condition holds). A warp ISSUES a line for all W lanes as
 * soon as at least one of its threads executes it; lanes whose thread does not execute it (masked
 * by the branch) or that have no thread at all (beyond the last item) do no useful work.
 *
 *   efficiency = useful lane-slots / issued lane-slots  ∈ (0, 1]
 *
 * It is 1 only if every warp is full and no line is skipped by part of a warp (no divergence).
 * @param {Array<{cost: number, act: (t: number) => boolean}>} lines
 * @param {number} nThreads number of threads with work (items)
 * @param {number} W warp width (lanes)
 * @param {number} [workgroup=2*W] threads per workgroup (a multiple of W)
 * @returns {{nWorkgroups: number, launched: number, nWarps: number, issued: number, useful: number, efficiency: number, slotsPerWarp: number}}
 *   launched = nWorkgroups·workgroup threads started (≥ nThreads); slotsPerWarp = Σ line costs
 */
export function simtStats(lines, nThreads, W, workgroup = 2 * W) {
  const nWorkgroups = Math.ceil(nThreads / workgroup), launched = nWorkgroups * workgroup, nWarps = launched / W;
  let issued = 0, useful = 0;
  for (const ln of lines) for (let w = 0; w < nWarps; w++) {
    let a = 0;
    for (let l = 0; l < W; l++) { const t = w * W + l; if (t < nThreads && ln.act(t)) a++; }
    if (a > 0) { issued += ln.cost * W; useful += ln.cost * a; }
  }
  return { nWorkgroups, launched, nWarps, issued, useful, efficiency: issued > 0 ? useful / issued : 1, slotsPerWarp: lines.reduce((s2, ln) => s2 + ln.cost, 0) };
}

/**
 * The lift kernel as a SIMT program with one thread per DOF g = e·Np + j·n + i (n = p+1):
 * three unconditional lines and four one-slot `if` bodies (right, left, top, bottom face),
 * each executed only by the DOFs on that face. Used by the widget and by the tests.
 * @param {number} p polynomial degree
 * @returns {Array<{cost: number, act: (g: number) => boolean, face?: string}>}
 */
export function liftProgram(p) {
  const n = p + 1, Np = n * n;
  const i = (g) => (g % Np) % n, j = (g) => Math.floor((g % Np) / n), all = () => true;
  return [
    { cost: 1, act: all }, { cost: 1, act: all },
    { cost: 1, act: (g) => i(g) === p, face: 'right' }, { cost: 1, act: (g) => i(g) === 0, face: 'left' },
    { cost: 1, act: (g) => j(g) === p, face: 'top' }, { cost: 1, act: (g) => j(g) === 0, face: 'bottom' },
    { cost: 1, act: all },
  ];
}

/**
 * Ridge point of the roofline, I* = peak / bandwidth [FLOP/byte]: kernels with a smaller
 * arithmetic intensity are memory-bound.
 * @param {number} peak [FLOP/s] @param {number} bw [byte/s] @returns {number}
 */
export const ridgePoint = (peak, bw) => peak / bw;
