/**
 * @file WGSL compute kernels of the WebGPU DG-SEM advection solver, as JS template strings.
 *
 * They implement *exactly* the algorithm and memory layout of lib/gpu/cputwin.js
 * (read that file's header for the maths), in 32-bit floats:
 *
 *   volume  : one invocation per DOF       rhs[g]  = −(ax·rx·Σ_k D_ik u_kj + ay·ry·Σ_k D_jk u_ik)
 *   surface : one invocation per face node flux[f] = upwind F̂ (each face slot written once → race-free)
 *   lift    : one invocation per DOF       rhs[g] += face terms gathered from flux[]
 *   rkStage : one invocation per DOF       k = A k + dt rhs;  uOut = uIn + B k
 *
 * Why one invocation ("thread") per DOF and not per element?
 *  - parallelism: N²(p+1)² threads instead of N² — a 64×64 mesh with p = 3 gives 65 536
 *    threads, enough to fill a GPU and hide memory latency; one-per-element would give 4 096;
 *  - coalescing: consecutive threads g, g+1, … read consecutive addresses u[g], u[g+1], …;
 *    with one thread per element, thread e would read u[e·Np + l], a stride of Np floats;
 *  - the price: neighbouring threads re-read the same element values (served by caches),
 *    and the lift kernel is *divergent* (only boundary nodes of an element do work).
 *
 * Specialisation: the degree p is baked into the source (`const NP1 = p+1`, the
 * differentiation matrix as a literal array), like C++ templates / shader permutations —
 * the compiler can then fully unroll the k-loops. Run-time constants (N, a, h, dt) are
 * in a uniform buffer.
 *
 * Bindings (group 0) per kernel, see dgadvgpu.js:
 *   volume : 0 = Params (uniform), 1 = u (read), 2 = rhs (read_write)
 *   surface: 0 = Params, 1 = u (read), 3 = flux (read_write)
 *   lift   : 0 = Params, 1 = u (read), 2 = rhs (read_write), 3 = flux (read)
 *   rkStage: 0 = Params, 1 = uIn (read), 2 = rhs (read), 4 = res (read_write), 5 = uOut (read_write), 6 = Stage (uniform)
 */

/** Workgroup size used by all kernels (a multiple of the 32/64-wide SIMD unit of current GPUs). */
export const WORKGROUP_SIZE = 64;

/**
 * Field order of the Params uniform: 12 × 4 bytes = 48 bytes (a multiple of 16, as WGSL
 * requires for a uniform struct). The first PARAMS_N_U32 fields are u32, the rest f32.
 * The WGSL struct in header() is GENERATED from this list and packParams() fills the
 * ArrayBuffer from it, so host and shader cannot get out of step.
 */
export const PARAMS_LAYOUT = ['N', 'nElem', 'nDof', 'nFace', 'ax', 'ay', 'rx', 'ry', 'lift', 'dt', 'pad0', 'pad1'];
/** Number of leading u32 fields in PARAMS_LAYOUT. */
export const PARAMS_N_U32 = 4;
/** Size of the Params uniform buffer in bytes. */
export const PARAMS_BYTES = 4 * PARAMS_LAYOUT.length;

const PARAMS_COMMENT = {
  N: 'elements per direction', nElem: 'N*N', nDof: 'nElem*NP', nFace: '2*nElem*NP1 face nodes', ax: 'velocity',
  rx: '2/h  (d/dx = rx d/dxi)', lift: '1/w_0 (inverse GLL end-point weight)',
};

/**
 * Pack the run-time constants into the byte layout of the WGSL `Params` struct.
 * @param {Record<string, number>} v values by field name (see PARAMS_LAYOUT); missing fields (the pads) are 0
 * @returns {ArrayBuffer} PARAMS_BYTES bytes: u32 fields then f32 fields, 4 bytes each, in PARAMS_LAYOUT order
 */
export function packParams(v) {
  const ab = new ArrayBuffer(PARAMS_BYTES), U = new Uint32Array(ab), F = new Float32Array(ab);
  PARAMS_LAYOUT.forEach((name, i) => { if (i < PARAMS_N_U32) U[i] = v[name] ?? 0; else F[i] = v[name] ?? 0; });
  return ab;
}

/**
 * Format a JS number as a WGSL f32 literal (9 significant digits round-trip an f32).
 * @param {number} v @returns {string}
 */
export const f32Literal = (v) => {
  const s = String(parseFloat(Math.fround(v).toPrecision(9))); // 9 significant digits round-trip an f32
  return /[.eE]/.test(s) ? s : `${s}.0`;
};

/**
 * Common header: constants for degree p, Params struct.
 * @param {number} p polynomial degree
 * @param {Float64Array} D (p+1)×(p+1) GLL differentiation matrix, row-major D[i*n+k] = ℓ_k'(ξ_i)
 * @returns {string} WGSL
 */
function header(p, D) {
  const n = p + 1;
  return /* wgsl */ `
// ---- generated for degree p = ${p} ----
const NP1 : u32 = ${n}u;          // nodes per direction
const NP  : u32 = ${n * n}u;          // nodes per element
const WG  : u32 = ${WORKGROUP_SIZE}u;
// GLL differentiation matrix, row-major: Dm[i*NP1 + k] = l_k'(xi_i)
// (var<private> rather than const so that indexing it with a run-time index is portable)
var<private> Dm : array<f32, ${n * n}> = array<f32, ${n * n}>(${Array.from(D, f32Literal).join(', ')});

struct Params {
${PARAMS_LAYOUT.map((name, i) => `  ${name.padEnd(5)} : ${i < PARAMS_N_U32 ? 'u32' : 'f32'},${PARAMS_COMMENT[name] ? `   // ${PARAMS_COMMENT[name]}` : ''}`).join('\n')}
};
@group(0) @binding(0) var<uniform> P : Params;
`;
}

/**
 * WGSL source of the four kernels for degree p.
 * @param {number} p
 * @param {Float64Array} D differentiation matrix (disc.D of cputwin.js' makeDGAdv2D)
 * @returns {{volume: string, surface: string, lift: string, rkStage: string}}
 */
export function makeKernelSources(p, D) {
  const H = header(p, D);
  const volume = H + /* wgsl */ `
@group(0) @binding(1) var<storage, read> u : array<f32>;
@group(0) @binding(2) var<storage, read_write> rhs : array<f32>;

// One invocation per DOF g = e*NP + j*NP1 + i. Sum factorisation: the 2D derivative
// is two 1D derivatives along the row j and the column i of the element: O(p) work per DOF.
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let g = gid.x;
  if (g >= P.nDof) { return; }          // the last workgroup may be partially empty
  let e = g / NP;
  let l = g % NP;
  let i = l % NP1;
  let j = l / NP1;
  let base = e * NP;
  var dx = 0.0;
  var dy = 0.0;
  for (var k = 0u; k < NP1; k++) {
    dx += Dm[i * NP1 + k] * u[base + j * NP1 + k];
    dy += Dm[j * NP1 + k] * u[base + k * NP1 + i];
  }
  rhs[g] = -(P.ax * P.rx * dx + P.ay * P.ry * dy);
}
`;

  const surface = H + /* wgsl */ `
@group(0) @binding(1) var<storage, read> u : array<f32>;
@group(0) @binding(3) var<storage, read_write> flux : array<f32>;

// One invocation per face node. Face slot f = dir*nElem*NP1 + e*NP1 + m belongs to the
// RIGHT (dir 0) or TOP (dir 1) face of element e: every slot has exactly one writer.
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let f = gid.x;
  if (f >= P.nFace) { return; }
  let half = P.nElem * NP1;
  let dir = f / half;
  let r = f % half;
  let e = r / NP1;
  let m = r % NP1;
  let ex = e % P.N;
  let ey = e / P.N;
  var uM : f32;
  var uP : f32;
  var an : f32;
  if (dir == 0u) {                      // normal (+1, 0): K- = e, K+ = right neighbour
    let eR = ey * P.N + (ex + 1u) % P.N;
    uM = u[e * NP + m * NP1 + (NP1 - 1u)];
    uP = u[eR * NP + m * NP1];
    an = P.ax;
  } else {                              // normal (0, +1): K+ = upper neighbour
    let eT = ((ey + 1u) % P.N) * P.N + ex;
    uM = u[e * NP + (NP1 - 1u) * NP1 + m];
    uP = u[eT * NP + m];
    an = P.ay;
  }
  flux[f] = select(an * uP, an * uM, an >= 0.0);   // upwind: select(falseValue, trueValue, cond)
}
`;

  const lift = H + /* wgsl */ `
@group(0) @binding(1) var<storage, read> u : array<f32>;
@group(0) @binding(2) var<storage, read_write> rhs : array<f32>;
@group(0) @binding(3) var<storage, read> flux : array<f32>;

// One invocation per DOF; only nodes on the element boundary do work (divergence!).
// Each thread READS the face values it needs (gather) and WRITES only its own rhs[g]:
// no two threads write the same address, so no atomics are needed.
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let g = gid.x;
  if (g >= P.nDof) { return; }
  let e = g / NP;
  let l = g % NP;
  let i = l % NP1;
  let j = l / NP1;
  let ex = e % P.N;
  let ey = e / P.N;
  let half = P.nElem * NP1;
  let cx = P.rx * P.lift;
  let cy = P.ry * P.lift;
  let uc = u[g];
  var r = rhs[g];
  if (i == NP1 - 1u) { r -= cx * (flux[e * NP1 + j] - P.ax * uc); }                       // right face (K = K-)
  if (i == 0u) {                                                                            // left face (K = K+)
    let eL = ey * P.N + (ex + P.N - 1u) % P.N;
    r += cx * (flux[eL * NP1 + j] - P.ax * uc);
  }
  if (j == NP1 - 1u) { r -= cy * (flux[half + e * NP1 + i] - P.ay * uc); }                // top face
  if (j == 0u) {                                                                            // bottom face
    let eB = ((ey + P.N - 1u) % P.N) * P.N + ex;
    r += cy * (flux[half + eB * NP1 + i] - P.ay * uc);
  }
  rhs[g] = r;
}
`;

  const rkStage = H + /* wgsl */ `
struct Stage { A : f32, B : f32, pad0 : f32, pad1 : f32 };
@group(0) @binding(1) var<storage, read> uIn : array<f32>;
@group(0) @binding(2) var<storage, read> rhs : array<f32>;
@group(0) @binding(4) var<storage, read_write> res : array<f32>;
@group(0) @binding(5) var<storage, read_write> uOut : array<f32>;
@group(0) @binding(6) var<uniform> S : Stage;

// Low-storage RK stage, purely pointwise: k = A k + dt L(u);  u_out = u_in + B k.
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let g = gid.x;
  if (g >= P.nDof) { return; }
  let k = S.A * res[g] + P.dt * rhs[g];
  res[g] = k;
  uOut[g] = uIn[g] + S.B * k;
}
`;
  return { volume, surface, lift, rkStage };
}
