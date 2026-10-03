// SMI score: rank + Sobel features, one global 4×4 ridge whitening, then per overlap
//   global: n‖C‖²_F                      (C = overlap cross-covariance)
//   exact:  n‖L_A⁻¹ C L_B⁻ᵀ‖²_F          (L = Cholesky of each overlap's own covariance)
// One bind layout for the module. Entry points only touch the slots they need (auto layout).
// Planes are packed plane-major so a pixel's neighborhood stays contiguous in cache.
//
// Stack planes are premultiplied by the pixel weight: [w, w f0, w f1, w f2, w f3].
// Bilinear warping then interpolates weight and signal together (like premultiplied
// alpha), and every moment below is a Σ w_a w_b (·) sum.
//
// Moment / correlation-plane layout (45; the global score uses the first 25):
//   0          n        = Σ w_a w_b
//   1..4       Σ a_i
//   5..8       Σ b_j
//   9..24      Σ a_i b_j          (i*4 + j)
//   25..34     Σ a_i a_j          (upper triangle, see tri_i / tri_j)
//   35..44     Σ b_i b_j

struct U {
    w: u32, h: u32, n: u32, cw: u32,
    ch: u32, plane: u32, plane_b: u32, min_n: u32,
    mean: f32, stdv: f32, scale: f32, ridge: f32,
    gx: u32, gy: u32, theme: u32, n_keys: u32,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> in_a: array<f32>;
@group(0) @binding(2) var<storage, read> in_b: array<f32>;
@group(0) @binding(3) var<storage, read> in_c: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_a: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_b: array<f32>;
@group(0) @binding(6) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> cdf: array<u32>;
@group(0) @binding(9) var<storage, read> in_d: array<f32>;
@group(0) @binding(10) var<storage, read> in_e: array<f32>;
@group(0) @binding(11) var<storage, read> in_f: array<f32>;
@group(0) @binding(12) var<storage, read> w_a: array<f32>;
@group(0) @binding(13) var<storage, read> w_b: array<f32>;
// a second direction's outputs (moments_coefs, gather_sel: both directions of the GPU climb in one dispatch)
@group(0) @binding(14) var<storage, read_write> out_c: array<f32>;
@group(0) @binding(15) var<storage, read_write> out_d: array<f32>;

const N_MOM: u32 = 45u;
// Per-pose gradient coefficients: W (16), μ_B (4), M (16), μ_A (4), then
// g = ∂score/∂(45 moment sums) for the overlap-boundary term, then the 4×4
// Gauss–Newton weight Q, then a flag: 1 = generic score (copula.wgsl scores), whose
// per-sample derivative also carries the linear term g_a (with μ = 0).
const N_COEF: u32 = 102u;
const COEF_FLAG: u32 = 101u;
const G0: u32 = 40u;
const Q0: u32 = 85u;
// Pose scores fade the moving image in over this many pixels inside its valid
// border, so the score is differentiable in the overlap boundary.
const TAPER: f32 = 2.0;

var<workgroup> sh: array<f32, 256>;
var<workgroup> tile: array<f32, 100>;
var<workgroup> shH: array<f32, 18>;
var<workgroup> cpsBase: u32;
var<workgroup> shv: array<vec4f, 256>;

fn hw() -> u32 { return u.w * u.h; }

fn tri_i(k: u32) -> u32 {
    var t = array<u32, 10>(0u, 0u, 0u, 0u, 1u, 1u, 1u, 2u, 2u, 3u);
    return t[k];
}
fn tri_j(k: u32) -> u32 {
    var t = array<u32, 10>(0u, 1u, 2u, 3u, 1u, 2u, 3u, 2u, 3u, 3u);
    return t[k];
}
fn tri_k(i: u32, j: u32) -> u32 {
    let a = min(i, j);
    let b = max(i, j);
    // Row offsets 0, 4, 7, 9 for a = 0..3.
    return a * 4u - (a * (a + 1u)) / 2u + a + (b - a);
}

fn inverse3(m: mat3x3f) -> mat3x3f {
    let a = m[0]; let b = m[1]; let c = m[2];
    let r0 = cross(b, c); let r1 = cross(c, a); let r2 = cross(a, b);
    let d = dot(a, r0);
    let id = 1.0 / (sign(d) * max(abs(d), 1e-12));
    var inv = transpose(mat3x3f(r0 * id, r1 * id, r2 * id));
    // One Newton step: inv ← inv (2I − M inv). Tightens H⁻¹ for the complementary warp.
    let I = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
    inv = inv * (I + I - m * inv);
    return inv;
}

// Row-major 3×3 at in_c[off..off+9]. (One helper per buffer instead of a storage pointer
// parameter: pointer parameters into storage are a Chrome/Tint extension that naga rejects.)
fn loadH_c() -> mat3x3f {
    return loadHAt_c(0u);
}

fn loadHAt_c(off: u32) -> mat3x3f {
    return mat3x3f(
        vec3f(in_c[off + 0u], in_c[off + 3u], in_c[off + 6u]),
        vec3f(in_c[off + 1u], in_c[off + 4u], in_c[off + 7u]),
        vec3f(in_c[off + 2u], in_c[off + 5u], in_c[off + 8u]),
    );
}

fn matTo9(m: mat3x3f, off: u32) {
    shH[off + 0u] = m[0][0]; shH[off + 3u] = m[0][1]; shH[off + 6u] = m[0][2];
    shH[off + 1u] = m[1][0]; shH[off + 4u] = m[1][1]; shH[off + 7u] = m[1][2];
    shH[off + 2u] = m[2][0]; shH[off + 5u] = m[2][1]; shH[off + 8u] = m[2][2];
}

fn matFrom9(off: u32) -> mat3x3f {
    return mat3x3f(
        vec3f(shH[off + 0u], shH[off + 3u], shH[off + 6u]),
        vec3f(shH[off + 1u], shH[off + 4u], shH[off + 7u]),
        vec3f(shH[off + 2u], shH[off + 5u], shH[off + 8u]),
    );
}

fn reduce256(lid: u32) {
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { sh[lid] = sh[lid] + sh[lid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
}

fn reduce256v4(lid: u32) {
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { shv[lid] = shv[lid] + shv[lid + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
}

fn mom4(acc: array<f32, 25>, b: u32) -> vec4f {
    if (b + 3u < 25u) {
        return vec4f(acc[b], acc[b + 1u], acc[b + 2u], acc[b + 3u]);
    }
    return vec4f(acc[24], 0.0, 0.0, 0.0);
}

fn reduceMoments25(acc: array<f32, 25>, lid: u32) -> array<f32, 25> {
    var sums: array<f32, 25>;
    for (var k = 0u; k < 7u; k++) {
        let b = k * 4u;
        shv[lid] = mom4(acc, b);
        workgroupBarrier();
        reduce256v4(lid);
        if (lid == 0u) {
            sums[b] = shv[0].x;
            if (b + 1u < 25u) { sums[b + 1u] = shv[0].y; }
            if (b + 2u < 25u) { sums[b + 2u] = shv[0].z; }
            if (b + 3u < 25u) { sums[b + 3u] = shv[0].w; }
        }
        workgroupBarrier();
    }
    return sums;
}

// 45 moments padded to 48 → 12 vec4 reductions.
fn reduceMoments48(acc: array<f32, 48>, lid: u32) -> array<f32, 48> {
    var sums: array<f32, 48>;
    for (var k = 0u; k < 12u; k++) {
        let b = k * 4u;
        shv[lid] = vec4f(acc[b], acc[b + 1u], acc[b + 2u], acc[b + 3u]);
        workgroupBarrier();
        reduce256v4(lid);
        if (lid == 0u) {
            sums[b] = shv[0].x; sums[b + 1u] = shv[0].y;
            sums[b + 2u] = shv[0].z; sums[b + 3u] = shv[0].w;
        }
        workgroupBarrier();
    }
    return sums;
}

fn tileNS(s: array<f32, 25>, min_n: u32) -> f32 {
    let n = s[0];
    if (n < f32(min_n)) { return 0.0; }
    let den = max(n, 1e-8);
    var acc = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let mai = s[1u + i] / den;
        for (var j = 0u; j < 4u; j++) {
            let mbj = s[5u + j] / den;
            let cij = s[9u + i * 4u + j] / den - mai * mbj;
            acc = acc + cij * cij;
        }
    }
    let score = n * acc;
    return select(0.0, score, score == score);
}

fn cubic_w(t: f32) -> vec4f {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4f(
        (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    );
}

fn clampu(i: i32, n: u32) -> u32 { return u32(clamp(i, 0, i32(n) - 1)); }

// φ is stored in destN units (same as Lie tx/ty). Pixel displacement is φ * (W/2, H/2).
fn dest_half(rw: f32, rh: f32) -> vec2f {
    return vec2f(max(rw, 1.0) * 0.5, max(rh, 1.0) * 0.5);
}

fn ffd_disp(xf: f32, yf: f32, rw: f32, rh: f32) -> vec2f {
    let gx = max(u.gx, 2u);
    let gy = max(u.gy, 2u);
    if (u.gx < 2u || xf < 0.0 || yf < 0.0 || xf > rw - 1.0 || yf > rh - 1.0) {
        return vec2f(0.0);
    }
    let hx = dest_half(rw, rh);
    let spx = max(rw - 1.0, 1.0) / f32(gx - 1u);
    let spy = max(rh - 1.0, 1.0) / f32(gy - 1u);
    let su = xf / spx;
    let sv = yf / spy;
    let iu = i32(floor(su));
    let iv = i32(floor(sv));
    let bu = cubic_w(clamp(su - f32(iu), 0.0, 1.0));
    let bv = cubic_w(clamp(sv - f32(iv), 0.0, 1.0));
    var d = vec2f(0.0);
    for (var jj = 0u; jj < 4u; jj++) {
        let j = clampu(iv - 1 + i32(jj), gy);
        for (var ii = 0u; ii < 4u; ii++) {
            let i = clampu(iu - 1 + i32(ii), gx);
            let w = bu[ii] * bv[jj];
            let o = cpsBase + (j * gx + i) * 2u;
            d = d + w * vec2f(in_f[o] * hx.x, in_f[o + 1u] * hx.y);
        }
    }
    return d;
}

// Clamped fetch / bilinear sample / bilinear value + gradient of plane `plane` of a
// w×h plane-major buffer, one set per buffer (in_a, in_b, w_a).
fn at_plane_a(plane: u32, w: u32, h: u32, x: i32, y: i32) -> f32 {
    let xx = clamp(x, 0, i32(w) - 1);
    let yy = clamp(y, 0, i32(h) - 1);
    return in_a[plane * w * h + u32(yy) * w + u32(xx)];
}

fn bilerp_plane_a(plane: u32, w: u32, h: u32, fx: f32, fy: f32) -> f32 {
    let x0 = i32(floor(fx)); let y0 = i32(floor(fy));
    let ax = fx - f32(x0); let ay = fy - f32(y0);
    return mix(
        mix(at_plane_a(plane, w, h, x0, y0), at_plane_a(plane, w, h, x0 + 1, y0), ax),
        mix(at_plane_a(plane, w, h, x0, y0 + 1), at_plane_a(plane, w, h, x0 + 1, y0 + 1), ax),
        ay,
    );
}

fn bilerp_grad_a(plane: u32, w: u32, h: u32, fx: f32, fy: f32) -> vec3f {
    let x0 = i32(floor(fx)); let y0 = i32(floor(fy));
    let ax = fx - f32(x0); let ay = fy - f32(y0);
    let v00 = at_plane_a(plane, w, h, x0, y0);
    let v10 = at_plane_a(plane, w, h, x0 + 1, y0);
    let v01 = at_plane_a(plane, w, h, x0, y0 + 1);
    let v11 = at_plane_a(plane, w, h, x0 + 1, y0 + 1);
    return vec3f(
        mix(mix(v00, v10, ax), mix(v01, v11, ax), ay),
        mix(v10 - v00, v11 - v01, ay),
        mix(v01 - v00, v11 - v10, ax),
    );
}

fn at_plane_b(plane: u32, w: u32, h: u32, x: i32, y: i32) -> f32 {
    let xx = clamp(x, 0, i32(w) - 1);
    let yy = clamp(y, 0, i32(h) - 1);
    return in_b[plane * w * h + u32(yy) * w + u32(xx)];
}

fn bilerp_plane_b(plane: u32, w: u32, h: u32, fx: f32, fy: f32) -> f32 {
    let x0 = i32(floor(fx)); let y0 = i32(floor(fy));
    let ax = fx - f32(x0); let ay = fy - f32(y0);
    return mix(
        mix(at_plane_b(plane, w, h, x0, y0), at_plane_b(plane, w, h, x0 + 1, y0), ax),
        mix(at_plane_b(plane, w, h, x0, y0 + 1), at_plane_b(plane, w, h, x0 + 1, y0 + 1), ax),
        ay,
    );
}

fn bilerp_grad_b(plane: u32, w: u32, h: u32, fx: f32, fy: f32) -> vec3f {
    let x0 = i32(floor(fx)); let y0 = i32(floor(fy));
    let ax = fx - f32(x0); let ay = fy - f32(y0);
    let v00 = at_plane_b(plane, w, h, x0, y0);
    let v10 = at_plane_b(plane, w, h, x0 + 1, y0);
    let v01 = at_plane_b(plane, w, h, x0, y0 + 1);
    let v11 = at_plane_b(plane, w, h, x0 + 1, y0 + 1);
    return vec3f(
        mix(mix(v00, v10, ax), mix(v01, v11, ax), ay),
        mix(v10 - v00, v11 - v01, ay),
        mix(v01 - v00, v11 - v10, ax),
    );
}

fn at_plane_wa(plane: u32, w: u32, h: u32, x: i32, y: i32) -> f32 {
    let xx = clamp(x, 0, i32(w) - 1);
    let yy = clamp(y, 0, i32(h) - 1);
    return w_a[plane * w * h + u32(yy) * w + u32(xx)];
}

fn bilerp_plane_wa(plane: u32, w: u32, h: u32, fx: f32, fy: f32) -> f32 {
    let x0 = i32(floor(fx)); let y0 = i32(floor(fy));
    let ax = fx - f32(x0); let ay = fy - f32(y0);
    return mix(
        mix(at_plane_wa(plane, w, h, x0, y0), at_plane_wa(plane, w, h, x0 + 1, y0), ax),
        mix(at_plane_wa(plane, w, h, x0, y0 + 1), at_plane_wa(plane, w, h, x0 + 1, y0 + 1), ax),
        ay,
    );
}

fn bilerp_grad_wa(plane: u32, w: u32, h: u32, fx: f32, fy: f32) -> vec3f {
    let x0 = i32(floor(fx)); let y0 = i32(floor(fy));
    let ax = fx - f32(x0); let ay = fy - f32(y0);
    let v00 = at_plane_wa(plane, w, h, x0, y0);
    let v10 = at_plane_wa(plane, w, h, x0 + 1, y0);
    let v01 = at_plane_wa(plane, w, h, x0, y0 + 1);
    let v11 = at_plane_wa(plane, w, h, x0 + 1, y0 + 1);
    return vec3f(
        mix(mix(v00, v10, ax), mix(v01, v11, ax), ay),
        mix(v10 - v00, v11 - v01, ay),
        mix(v01 - v00, v11 - v10, ax),
    );
}

// Border taper of the moving sample at (fx, fy): (w, ∂w/∂fx, ∂w/∂fy).
// Linear ramp from 0 at the valid edge (2 px in) to 1 at TAPER px further in.
fn ramp(v: f32, lo: f32, hi: f32) -> vec2f {
    let a = (v - lo) / TAPER;
    let b = (hi - v) / TAPER;
    let ca = clamp(a, 0.0, 1.0);
    let cb = clamp(b, 0.0, 1.0);
    let da = select(0.0, 1.0 / TAPER, a > 0.0 && a < 1.0);
    let db = select(0.0, -1.0 / TAPER, b > 0.0 && b < 1.0);
    return vec2f(ca * cb, da * cb + ca * db);
}
fn taper(fx: f32, fy: f32) -> vec3f {
    let tx = ramp(fx, 2.0, f32(u.cw) - 3.0);
    let ty = ramp(fy, 2.0, f32(u.ch) - 3.0);
    return vec3f(tx.x * ty.x, tx.y * ty.x, tx.x * ty.y);
}

// ∂score/∂w for one pixel pair: Σ_k g_k φ_k(v_a, v_b) over the 45 moment monomials.
fn boundary_value(e0: u32, va: array<f32, 4>, vb: array<f32, 4>) -> f32 {
    let g = e0 + G0;
    var s = in_e[g];
    for (var i = 0u; i < 4u; i++) {
        s = s + in_e[g + 1u + i] * va[i] + in_e[g + 5u + i] * vb[i];
        for (var j = 0u; j < 4u; j++) { s = s + in_e[g + 9u + i * 4u + j] * va[i] * vb[j]; }
        for (var j = i; j < 4u; j++) {
            let k = tri_k(i, j);
            s = s + in_e[g + 25u + k] * va[i] * va[j] + in_e[g + 35u + k] * vb[i] * vb[j];
        }
    }
    return s;
}

// Pixel weight of a (moving sample, fixed pixel) pair. u.plane_b bit 0: edge weights.
// The border taper is applied separately (taper()); gradients treat edge weights as fixed.
fn pair_weight(fx: f32, fy: f32, t: u32) -> f32 {
    if ((u.plane_b & 1u) == 0u) { return 1.0; }
    return bilerp_plane_wa(0u, u.cw, u.ch, fx, fy) * w_b[t];
}

fn tile_at(lx: u32, ly: u32, dx: i32, dy: i32) -> f32 {
    return tile[u32(i32(ly) + dy) * 10u + u32(i32(lx) + dx)];
}


// A 1-D kernel over more than 65535 workgroups runs on an (x, y) grid (Gpu.flat); this is its
// invocation index (workgroups of 256).
fn flat_index(g: vec3u, nwg: vec3u) -> u32 {
    return g.x + g.y * nwg.x * 256u;
}

@compute @workgroup_size(256)
fn zscore(@builtin(global_invocation_id) gid: vec3u) {
    let n = hw();
    if (gid.x >= n) { return; }
    out_a[gid.x] = (in_a[gid.x] - u.mean) / max(u.stdv, 1e-8);
}

@compute @workgroup_size(256)
fn hist_bins(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    let n = hw();
    if (gid.x >= n) { return; }
    atomicAdd(&hist[u32(clamp((in_a[gid.x] + 4.0) * 32.0, 0.0, 255.0))], 1u);
}

@compute @workgroup_size(1)
fn cdf_scan() {
    if (u.w == 0u) { return; }
    var acc = 0u;
    for (var i = 0u; i < 256u; i++) {
        cdf[i] = acc;
        acc = acc + atomicLoad(&hist[i]);
    }
    cdf[256] = acc;
}

@compute @workgroup_size(256)
fn rank_eq(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    let n = hw();
    if (gid.x >= n) { return; }
    let tot = max(cdf[256], 1u);
    let b = u32(clamp((in_a[gid.x] + 4.0) * 32.0, 0.0, 255.0));
    out_a[gid.x] = (f32(cdf[b]) + 0.5 * f32(atomicLoad(&hist[b]))) / f32(tot) * 2.0 - 1.0;
}

@compute @workgroup_size(8, 8, 1)
fn feats(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u,
         @builtin(global_invocation_id) gid: vec3u) {
    let tid = lid.y * 8u + lid.x;
    let ox = i32(wid.x * 8u);
    let oy = i32(wid.y * 8u);
    let w = i32(u.w); let h = i32(u.h);
    for (var k = tid; k < 100u; k = k + 64u) {
        let tx = i32(k % 10u) - 1;
        let ty = i32(k / 10u) - 1;
        let xx = clamp(ox + tx, 0, w - 1);
        let yy = clamp(oy + ty, 0, h - 1);
        tile[k] = in_a[u32(yy) * u.w + u32(xx)];
    }
    workgroupBarrier();
    if (gid.x >= u.w || gid.y >= u.h) { return; }
    let lx = lid.x + 1u;
    let ly = lid.y + 1u;
    let r = tile[ly * 10u + lx];
    let gx = -tile_at(lx, ly, -1, -1) + tile_at(lx, ly, 1, -1)
           - 2.0 * tile_at(lx, ly, -1, 0) + 2.0 * tile_at(lx, ly, 1, 0)
           - tile_at(lx, ly, -1, 1) + tile_at(lx, ly, 1, 1);
    let gy = -tile_at(lx, ly, -1, -1) - 2.0 * tile_at(lx, ly, 0, -1) - tile_at(lx, ly, 1, -1)
           + tile_at(lx, ly, -1, 1) + 2.0 * tile_at(lx, ly, 0, 1) + tile_at(lx, ly, 1, 1);
    let g = sqrt(gx * gx + gy * gy);
    let i = gid.y * u.w + gid.x;
    let n = hw();
    out_a[i] = r;
    out_a[n + i] = r * r;
    out_a[2u * n + i] = r * r * r;
    out_a[3u * n + i] = g;
    out_b[i] = g;
}

// Φ⁻¹(p), Acklam's rational approximation (relative error ~1e-9 in f64; f32 here).
fn ndtri(p: f32) -> f32 {
    let q = clamp(p, 1e-6, 1.0 - 1e-6);
    if (q < 0.02425 || q > 0.97575) {
        let t = sqrt(-2.0 * log(select(1.0 - q, q, q < 0.5)));
        let v = (((((-7.784894002430293e-03 * t - 3.223964580411365e-01) * t - 2.400758277161838e+00) * t
            - 2.549732539343734e+00) * t + 4.374664141464968e+00) * t + 2.938163982698783e+00)
            / ((((7.784695709041462e-03 * t + 3.224671290700398e-01) * t + 2.445134137142996e+00) * t
            + 3.754408661907416e+00) * t + 1.0);
        return select(-v, v, q < 0.5);
    }
    let r = q - 0.5;
    let t = r * r;
    return (((((-3.969683028665376e+01 * t + 2.209460984245205e+02) * t - 2.759285104469687e+02) * t
        + 1.383577518672690e+02) * t - 3.066479806614716e+01) * t + 2.506628277459239e+00) * r
        / (((((-5.447609879822406e+01 * t + 1.615858368580409e+02) * t - 1.556989798598866e+02) * t
        + 6.680131188771972e+01) * t - 1.328068155288572e+01) * t + 1.0);
}

// Normal-score features for the copula scores: z = Φ⁻¹ of the mid-rank, planes
// [z, z², z³, z⁴] (unwhitened: the scores are written in these raw powers). The
// Sobel magnitude of the rank still goes to out_b for edge weighting.
@compute @workgroup_size(8, 8, 1)
fn feats_normal(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u,
                @builtin(global_invocation_id) gid: vec3u) {
    let tid = lid.y * 8u + lid.x;
    let ox = i32(wid.x * 8u);
    let oy = i32(wid.y * 8u);
    let w = i32(u.w); let h = i32(u.h);
    for (var k = tid; k < 100u; k = k + 64u) {
        let tx = i32(k % 10u) - 1;
        let ty = i32(k / 10u) - 1;
        let xx = clamp(ox + tx, 0, w - 1);
        let yy = clamp(oy + ty, 0, h - 1);
        tile[k] = in_a[u32(yy) * u.w + u32(xx)];
    }
    workgroupBarrier();
    if (gid.x >= u.w || gid.y >= u.h) { return; }
    let lx = lid.x + 1u;
    let ly = lid.y + 1u;
    let r = tile[ly * 10u + lx];
    let gx = -tile_at(lx, ly, -1, -1) + tile_at(lx, ly, 1, -1)
           - 2.0 * tile_at(lx, ly, -1, 0) + 2.0 * tile_at(lx, ly, 1, 0)
           - tile_at(lx, ly, -1, 1) + tile_at(lx, ly, 1, 1);
    let gy = -tile_at(lx, ly, -1, -1) - 2.0 * tile_at(lx, ly, 0, -1) - tile_at(lx, ly, 1, -1)
           + tile_at(lx, ly, -1, 1) + 2.0 * tile_at(lx, ly, 0, 1) + tile_at(lx, ly, 1, 1);
    let z = ndtri(0.5 * (r + 1.0));
    let i = gid.y * u.w + gid.x;
    let n = hw();
    out_a[i] = z;
    out_a[n + i] = z * z;
    out_a[2u * n + i] = z * z * z;
    out_a[3u * n + i] = z * z * z * z;
    out_b[i] = sqrt(gx * gx + gy * gy);
}

@compute @workgroup_size(256)
fn reduce1(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    var acc = 0.0;
    let n = hw();
    for (var t = wid.x * 256u + lid.x; t < n; t = t + 256u * 64u) { acc = acc + in_a[t]; }
    sh[lid.x] = acc;
    workgroupBarrier();
    reduce256(lid.x);
    if (lid.x == 0u) { out_a[wid.x] = sh[0]; }
}

@compute @workgroup_size(256)
fn feat_moments(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let n = hw();
    var acc: array<f32, 21>;
    for (var t = wid.x * 256u + lid.x; t < n; t = t + 256u * 64u) {
        acc[0] = acc[0] + 1.0;
        let x0 = in_a[t]; let x1 = in_a[n + t]; let x2 = in_a[2u * n + t]; let x3 = in_a[3u * n + t];
        acc[1] = acc[1] + x0; acc[2] = acc[2] + x1; acc[3] = acc[3] + x2; acc[4] = acc[4] + x3;
        acc[5] = acc[5] + x0 * x0; acc[6] = acc[6] + x0 * x1; acc[7] = acc[7] + x0 * x2; acc[8] = acc[8] + x0 * x3;
        acc[9] = acc[9] + x1 * x0; acc[10] = acc[10] + x1 * x1; acc[11] = acc[11] + x1 * x2; acc[12] = acc[12] + x1 * x3;
        acc[13] = acc[13] + x2 * x0; acc[14] = acc[14] + x2 * x1; acc[15] = acc[15] + x2 * x2; acc[16] = acc[16] + x2 * x3;
        acc[17] = acc[17] + x3 * x0; acc[18] = acc[18] + x3 * x1; acc[19] = acc[19] + x3 * x2; acc[20] = acc[20] + x3 * x3;
    }
    for (var m = 0u; m < 21u; m++) {
        sh[lid.x] = acc[m];
        workgroupBarrier();
        reduce256(lid.x);
        if (lid.x == 0u) { out_a[wid.x * 21u + m] = sh[0]; }
        workgroupBarrier();
    }
}

@compute @workgroup_size(1)
fn chol() {
    let n = max(in_a[0], 1.0);
    var mu: array<f32, 4>;
    for (var i = 0u; i < 4u; i++) { mu[i] = in_a[1u + i] / n; }
    var G: array<f32, 16>;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) { G[i * 4u + j] = in_a[5u + i * 4u + j] / n - mu[i] * mu[j]; }
    }
    var tr = 0.0;
    for (var i = 0u; i < 4u; i++) { tr = tr + G[i * 4u + i]; }
    let ridge = u.ridge * max(tr * 0.25, 1e-12);
    for (var i = 0u; i < 4u; i++) { G[i * 4u + i] = G[i * 4u + i] + ridge; }
    var L: array<f32, 16>;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j <= i; j++) {
            var s = G[i * 4u + j];
            for (var k = 0u; k < j; k++) { s = s - L[i * 4u + k] * L[j * 4u + k]; }
            if (i == j) { L[i * 4u + i] = sqrt(max(s, 1e-12)); }
            else { L[i * 4u + j] = s / L[j * 4u + j]; }
        }
    }
    var X: array<f32, 16>;
    for (var col = 0u; col < 4u; col++) {
        for (var i = 0u; i < 4u; i++) {
            var s = select(0.0, 1.0, i == col);
            for (var k = 0u; k < i; k++) { s = s - L[i * 4u + k] * X[k * 4u + col]; }
            X[i * 4u + col] = s / L[i * 4u + i];
        }
    }
    for (var i = 0u; i < 16u; i++) { out_a[i] = X[i]; }
    for (var i = 0u; i < 4u; i++) { out_a[16u + i] = mu[i]; }
}

@compute @workgroup_size(256)
fn whiten(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    let n = hw();
    if (gid.x >= n) { return; }
    var x: array<f32, 4>;
    for (var i = 0u; i < 4u; i++) { x[i] = out_a[i * n + gid.x] - in_b[16u + i]; }
    for (var i = 0u; i < 4u; i++) {
        var s = 0.0;
        for (var j = 0u; j < 4u; j++) { s = s + in_b[i * 4u + j] * x[j]; }
        out_a[i * n + gid.x] = s;
    }
}

@compute @workgroup_size(256)
fn weight(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    if (gid.x >= hw()) { return; }
    out_a[gid.x] = in_a[gid.x] / max(u.scale, 1e-8);
}

@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    if (gid.x >= hw()) { return; }
    out_a[gid.x] = u.scale;
}

// [w, w f0, w f1, w f2, w f3]
@compute @workgroup_size(256)
fn stack_planes(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    let n = hw();
    if (gid.x >= n) { return; }
    let w = in_b[gid.x];
    out_a[gid.x] = w;
    for (var i = 0u; i < 4u; i++) { out_a[(i + 1u) * n + gid.x] = w * in_a[i * n + gid.x]; }
}

fn sample_plane(px: f32, py: f32, plane: u32) -> f32 {
    let w = i32(u.w); let h = i32(u.h);
    let x = px - 0.5; let y = py - 0.5;
    let x0 = i32(floor(x)); let y0 = i32(floor(y));
    let fx = x - f32(x0); let fy = y - f32(y0);
    let off = plane * u.w * u.h;
    let xa = u32(clamp(x0, 0, w - 1)); let xb = u32(clamp(x0 + 1, 0, w - 1));
    let ya = u32(clamp(y0, 0, h - 1)); let yb = u32(clamp(y0 + 1, 0, h - 1));
    return mix(
        mix(in_a[off + ya * u.w + xa], in_a[off + ya * u.w + xb], fx),
        mix(in_a[off + yb * u.w + xa], in_a[off + yb * u.w + xb], fx),
        fy,
    );
}

// Real correlation plane r from premultiplied stack samples s = [w, w f0..w f3]:
//   0 → w,  1..4 → w f_i,  5..14 → w f_i f_j  (= s_i s_j / s_0).
fn real_plane(r: u32, s: array<f32, 5>) -> f32 {
    if (r == 0u) { return s[0]; }
    if (r < 5u) { return s[r]; }
    if (s[0] <= 1e-6) { return 0.0; }
    let k = r - 5u;
    return s[1u + tri_i(k)] * s[1u + tri_j(k)] / s[0];
}

// Two real planes per complex FFT input: plane 2z in re, 2z+1 in im.
// u.n_keys = number of real planes (5 global, 15 exact). u.theme == 1: log-polar
// grid, every plane × ρ so correlations integrate with the area element.
// A source wider than the grid is box-filtered over each cell's footprint
// (k×k bilinear taps, k = ⌈scale⌉ ≤ 8) instead of point-sampled, which aliased.
// A map's grid: u.n wide, u.gy high (u.gy 0: square). Shift maps are rectangular (a power of
// two per side); the roto-scale map and the other users pass 0.
fn map_w() -> u32 { return u.n; }
fn map_h() -> u32 { return select(u.n, u.gy, u.gy != 0u); }

// The stack's five planes on the grid (cw × ch cells of the map_w × map_h grid) into out_b,
// plane-major, once; pack_pair then builds every correlation plane from them.
@compute @workgroup_size(8, 8, 1)
fn pack_resample(@builtin(global_invocation_id) gid: vec3u) {
    let n = map_w();
    if (gid.x >= u.cw || gid.y >= u.ch) { return; }
    let fx = f32(u.w) / f32(max(u.cw, 1u));
    let fy = f32(u.h) / f32(max(u.ch, 1u));
    let kx = u32(clamp(ceil(fx - 1e-4), 1.0, 8.0));
    let ky = u32(clamp(ceil(fy - 1e-4), 1.0, 8.0));
    var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0; var s4 = 0.0;
    for (var j = 0u; j < ky; j++) {
        let py = (f32(gid.y) + (f32(j) + 0.5) / f32(ky)) * fy;
        for (var i = 0u; i < kx; i++) {
            let px = (f32(gid.x) + (f32(i) + 0.5) / f32(kx)) * fx;
            s0 = s0 + sample_plane(px, py, 0u);
            s1 = s1 + sample_plane(px, py, 1u);
            s2 = s2 + sample_plane(px, py, 2u);
            s3 = s3 + sample_plane(px, py, 3u);
            s4 = s4 + sample_plane(px, py, 4u);
        }
    }
    let inv = 1.0 / f32(kx * ky);
    let c = gid.y * n + gid.x;
    let nn = n * map_h();
    out_b[c] = s0 * inv;
    out_b[nn + c] = s1 * inv;
    out_b[2u * nn + c] = s2 * inv;
    out_b[3u * nn + c] = s3 * inv;
    out_b[4u * nn + c] = s4 * inv;
}

@compute @workgroup_size(8, 8, 1)
fn pack_pair(@builtin(global_invocation_id) gid: vec3u) {
    let n = map_w();
    let nh = map_h();
    if (gid.x >= n || gid.y >= nh) { return; }
    let nn = n * nh;
    let i = (gid.z * nn + gid.y * n + gid.x) * 2u;
    var re = 0.0;
    var im = 0.0;
    if (gid.x < u.cw && gid.y < u.ch) {
        // the resampled cell (pack_resample)
        let c = gid.y * n + gid.x;
        var s: array<f32, 5>;
        for (var p = 0u; p < 5u; p++) { s[p] = in_c[p * nn + c]; }
        var om = 1.0;
        if (u.theme == 1u) { om = exp(u.mean + (f32(gid.y) + 0.5) / f32(max(u.ch, 1u)) * u.stdv); }
        let r0 = gid.z * 2u;
        re = om * real_plane(r0, s);
        if (r0 + 1u < u.n_keys) { im = om * real_plane(r0 + 1u, s); }
    }
    out_a[i] = re; out_a[i + 1u] = im;
}

// Spectrum of real plane r at flat bin i, given the mirror bin im (−k).
// Z = FFT(x_even + i x_odd):  X_even = (Z + conj Z₋)/2,  X_odd = (Z − conj Z₋)/(2i).
fn unpack_a(r: u32, i: u32, im: u32, nn: u32) -> vec2f {
    let o = (r >> 1u) * nn * 2u;
    let z = vec2f(in_a[o + i * 2u], in_a[o + i * 2u + 1u]);
    let zm = vec2f(in_a[o + im * 2u], in_a[o + im * 2u + 1u]);
    if ((r & 1u) == 0u) { return 0.5 * vec2f(z.x + zm.x, z.y - zm.y); }
    return 0.5 * vec2f(z.y + zm.y, zm.x - z.x);
}
fn unpack_b(r: u32, i: u32, im: u32, nn: u32) -> vec2f {
    let o = (r >> 1u) * nn * 2u;
    let z = vec2f(in_b[o + i * 2u], in_b[o + i * 2u + 1u]);
    let zm = vec2f(in_b[o + im * 2u], in_b[o + im * 2u + 1u]);
    if ((r & 1u) == 0u) { return 0.5 * vec2f(z.x + zm.x, z.y - zm.y); }
    return 0.5 * vec2f(z.y + zm.y, zm.x - z.x);
}

// Correlation spectrum A·conj(B) (so irfft gives Σ_x a(x + t) b(x)), two per output:
// P = X₁ + i X₂, whose inverse FFT is x₁ + i x₂ because both are real.
// in_c holds the pair list, 6 u32 per output: ra₁ rb₁ o₁ ra₂ rb₂ o₂ (o₂ = ~0 if unused).
@compute @workgroup_size(256)
fn cmul_pair(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    let n = map_w();
    let nh = map_h();
    let nn = n * nh;
    if (gid.x >= nn) { return; }
    let e = (u.plane + gid.z) * 6u;
    let y = gid.x / n;
    let x = gid.x - y * n;
    let im = ((nh - y) % nh) * n + ((n - x) % n);
    let a1 = unpack_a(bitcast<u32>(in_c[e]), gid.x, im, nn);
    let b1 = unpack_b(bitcast<u32>(in_c[e + 1u]), gid.x, im, nn);
    var p = vec2f(a1.x * b1.x + a1.y * b1.y, a1.y * b1.x - a1.x * b1.y);
    if (bitcast<u32>(in_c[e + 5u]) != 0xffffffffu) {
        let a2 = unpack_a(bitcast<u32>(in_c[e + 3u]), gid.x, im, nn);
        let b2 = unpack_b(bitcast<u32>(in_c[e + 4u]), gid.x, im, nn);
        let x2 = vec2f(a2.x * b2.x + a2.y * b2.y, a2.y * b2.x - a2.x * b2.y);
        p = vec2f(p.x - x2.y, p.y + x2.x);
    }
    let o = (gid.z * nn + gid.x) * 2u;
    out_a[o] = p.x;
    out_a[o + 1u] = p.y;
}

// Inverse-FFT output z → correlation planes o₁ (re) and o₂ (im).
// u.scale > 1.5: log-polar map, × e^{kΔλ} Δλ Δθ so each pair integrates dest area.
@compute @workgroup_size(256)
fn extract_pair(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    let n = map_w();
    let nh = map_h();
    let nn = n * nh;
    if (gid.x >= nn) { return; }
    let e = (u.plane + gid.z) * 6u;
    var s = 1.0;
    if (u.scale > 1.5) {
        let y = gid.x / n;
        var k = f32(y);
        if (y > nh / 2u) { k = f32(i32(y) - i32(nh)); }
        s = exp(k * u.mean) * u.mean * u.stdv;
    }
    let src = (gid.z * nn + gid.x) * 2u;
    out_a[bitcast<u32>(in_c[e + 2u]) * nn + gid.x] = in_a[src] * s;
    let o2 = bitcast<u32>(in_c[e + 5u]);
    if (o2 != 0xffffffffu) { out_a[o2 * nn + gid.x] = in_a[src + 1u] * s; }
}

// In-place Cholesky of a 4×4 SPD matrix (row-major, lower). False if not PD.
fn chol4(g: ptr<function, array<f32, 16>>) -> bool {
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j <= i; j++) {
            var s = (*g)[i * 4u + j];
            for (var k = 0u; k < j; k++) { s = s - (*g)[i * 4u + k] * (*g)[j * 4u + k]; }
            if (i == j) {
                if (!(s > 0.0)) { return false; }
                (*g)[i * 4u + i] = sqrt(s);
            } else {
                (*g)[i * 4u + j] = s / (*g)[j * 4u + j];
            }
        }
    }
    return true;
}

fn corr_at(p: u32, i: u32, nn: u32) -> f32 { return in_a[p * nn + i]; }

// Wilson–Hilferty normal score of an F(d1, d2) value.
fn f_to_z(f: f32, d1: f32, d2: f32) -> f32 {
    let a = 2.0 / (9.0 * d1);
    let b = 2.0 / (9.0 * d2);
    let c = pow(max(f, 0.0), 1.0 / 3.0);
    return ((1.0 - b) * c - (1.0 - a)) / sqrt(b * c * c + a);
}

// Per-shift score from the correlation planes. u.gx == 1: exact (local whitening).
// u.theme == 1 (exact only): calibrated. n_eff = n / A with A = u.mean the feature
// correlation area in grid cells, and the score is the Wilson–Hilferty z of
// Pillai's F-approximation for V = Σρ² (p = q = 4):
//   F = ((n_eff − 5) / 4) · V / (4 − V),  F ~ F(16, 4(n_eff − 5)) under independence.
// Overlaps with n_eff ≤ 10 score 0; this replaces the 8 % zero-lag floor.
fn combine_cell(t: u32, area: f32) {
    let nn = map_w() * map_h();
    let n = corr_at(0u, t, nn);
    let calib = u.gx == 1u && u.theme == 1u;
    let neff = n / max(area, 1.0);
    var floor_n = max(f32(u.min_n), 0.08 * max(in_a[0], 0.0));
    if (calib) { floor_n = f32(u.min_n); }
    if (n < floor_n || (calib && neff <= 10.0)) { out_a[t] = 0.0; return; }
    let den = max(n, 1e-8);
    // Copula scores (u.gx 2: E4 on the first 25 planes, 3: λmax on all 45): n × MI.
    if (u.gx >= 2u) {
        var mm: array<f32, 45>;
        let cnt = select(25u, 45u, u.gx == 3u);
        for (var k = 1u; k < cnt; k++) { mm[k] = corr_at(k, t, nn) / den; }
        out_a[t] = n * copula_score(&mm, u.gx, u.ridge);
        return;
    }
    var ma: array<f32, 4>;
    var mb: array<f32, 4>;
    for (var i = 0u; i < 4u; i++) {
        ma[i] = corr_at(1u + i, t, nn) / den;
        mb[i] = corr_at(5u + i, t, nn) / den;
    }
    if (u.gx == 0u) {
        var s = 0.0;
        for (var i = 0u; i < 4u; i++) {
            for (var j = 0u; j < 4u; j++) {
                let cij = clamp(corr_at(9u + i * 4u + j, t, nn) / den - ma[i] * mb[j], -1.0, 1.0);
                s = s + cij * cij;
            }
        }
        out_a[t] = n * s;
        return;
    }
    var ga: array<f32, 16>;
    var gb: array<f32, 16>;
    var tra = 0.0;
    var trb = 0.0;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let k = tri_k(i, j);
            ga[i * 4u + j] = corr_at(25u + k, t, nn) / den - ma[i] * ma[j];
            gb[i * 4u + j] = corr_at(35u + k, t, nn) / den - mb[i] * mb[j];
        }
        tra = tra + ga[i * 5u];
        trb = trb + gb[i * 5u];
    }
    for (var i = 0u; i < 4u; i++) {
        ga[i * 5u] = ga[i * 5u] + u.ridge * max(tra * 0.25, 1e-12);
        gb[i * 5u] = gb[i * 5u] + u.ridge * max(trb * 0.25, 1e-12);
    }
    if (!chol4(&ga) || !chol4(&gb)) { out_a[t] = 0.0; return; }
    // X = L_A⁻¹ C (forward substitution down each column of C).
    var x: array<f32, 16>;
    for (var j = 0u; j < 4u; j++) {
        for (var i = 0u; i < 4u; i++) {
            var s = corr_at(9u + i * 4u + j, t, nn) / den - ma[i] * mb[j];
            for (var k = 0u; k < i; k++) { s = s - ga[i * 4u + k] * x[k * 4u + j]; }
            x[i * 4u + j] = s / ga[i * 5u];
        }
    }
    // Y = X L_B⁻ᵀ: each row y solves L_B yᵀ = xᵀ. Accumulate ‖Y‖² as we go.
    var acc = 0.0;
    for (var r = 0u; r < 4u; r++) {
        var y: array<f32, 4>;
        for (var i = 0u; i < 4u; i++) {
            var s = x[r * 4u + i];
            for (var k = 0u; k < i; k++) { s = s - gb[i * 4u + k] * y[k]; }
            y[i] = s / gb[i * 5u];
            acc = acc + y[i] * y[i];
        }
    }
    var score = n * acc;
    if (calib) {
        let v = min(acc, 4.0 - 1e-4);
        let f = (neff - 5.0) * 0.25 * v / (4.0 - v);
        score = max(f_to_z(f, 16.0, 4.0 * (neff - 5.0)), 0.0);
    }
    out_a[t] = select(0.0, score, score == score);
}

@compute @workgroup_size(256)
fn combine(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= map_w() * map_h()) { return; }
    combine_cell(gid.x, u.mean);
}

// combine with the correlation area read on the GPU: area = max(1, in_c[u.plane] · u.scale)
// (in_c[u.plane] = the pair's correlation area in pixels, u.scale = cells per pixel², i.e.
// 1 / g² for a shift map on a g-px grid and 1 for a log-polar map).
@compute @workgroup_size(256)
fn combine_p(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    if (gid.x >= map_w() * map_h()) { return; }
    combine_cell(gid.x, max(1.0, in_c[u.plane] * u.scale));
}

// Autocorrelation of the 4 feature planes (u.w × u.h) at lags |dx|, |dy| ≤ 8,
// one workgroup per lag (17² = 289). out[lag * 4 + i] = Σ_x f_i(x) f_i(x + τ) / count.
@compute @workgroup_size(256)
fn feat_acf(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let dx = i32(wid.x % 17u) - 8;
    let dy = i32(wid.x / 17u) - 8;
    let w = i32(u.w);
    let h = i32(u.h);
    let hw = u.w * u.h;
    var acc = vec4f(0.0);
    var cnt = 0.0;
    for (var t = lid.x; t < hw; t = t + 256u) {
        let x = i32(t % u.w);
        let y = i32(t / u.w);
        let x2 = x + dx;
        let y2 = y + dy;
        if (x2 < 0 || y2 < 0 || x2 >= w || y2 >= h) { continue; }
        let t2 = u32(y2 * w + x2);
        acc = acc + vec4f(in_a[t] * in_a[t2], in_a[hw + t] * in_a[hw + t2],
                          in_a[2u * hw + t] * in_a[2u * hw + t2], in_a[3u * hw + t] * in_a[3u * hw + t2]);
        cnt = cnt + 1.0;
    }
    shv[lid.x] = acc;
    sh[lid.x] = cnt;
    workgroupBarrier();
    var k = 128u;
    loop {
        if (k == 0u) { break; }
        if (lid.x < k) {
            shv[lid.x] = shv[lid.x] + shv[lid.x + k];
            sh[lid.x] = sh[lid.x] + sh[lid.x + k];
        }
        workgroupBarrier();
        k = k / 2u;
    }
    if (lid.x == 0u) {
        let c = max(sh[0], 1.0);
        out_a[wid.x * 4u] = shv[0].x / c;
        out_a[wid.x * 4u + 1u] = shv[0].y / c;
        out_a[wid.x * 4u + 2u] = shv[0].z / c;
        out_a[wid.x * 4u + 3u] = shv[0].w / c;
    }
}

// 45 overlap moments per pose (layout at the top). One workgroup set per pose (wid.z).
// u.plane == 1: pose-specific control points. u.plane_b bit 0: edge weights (w_a, w_b).
@compute @workgroup_size(256)
fn overlap_moments(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u,
    @builtin(num_workgroups) nwg: vec3u) {
    let nPose = max(u.n_keys, 1u);
    if (wid.z >= nPose) { return; }
    let pose = wid.z;
    let dhw = hw();
    if (lid.x == 0u) {
        let H0 = loadHAt_c(pose * 9u);
        matTo9(H0, 0u);
        matTo9(inverse3(H0), 9u);
        cpsBase = select(0u, pose * max(u.gx, 2u) * max(u.gy, 2u) * 2u, u.plane == 1u);
    }
    workgroupBarrier();
    let Hi = matFrom9(9u);
    let sw = f32(u.cw); let shh = f32(u.ch);
    // the 48 moments in vector registers: Σw, Σw a, Σw b, Σw a bᵀ (by column of b's index i:
    // c[i] = Σ w a_i b), Σw a aᵀ and Σw b bᵀ as full 4 × 4 (upper triangles kept); each entry
    // is summed as the array form sums it
    var s0 = 0.0;
    var sa = vec4f(0.0);
    var sb = vec4f(0.0);
    var c0 = vec4f(0.0); var c1 = vec4f(0.0); var c2 = vec4f(0.0); var c3 = vec4f(0.0);
    var p0 = vec4f(0.0); var p1 = vec4f(0.0); var p2 = vec4f(0.0); var p3 = vec4f(0.0);
    var q0 = vec4f(0.0); var q1 = vec4f(0.0); var q2 = vec4f(0.0); var q3 = vec4f(0.0);
    for (var t = wid.x * 256u + lid.x; t < dhw; t = t + 256u * nwg.x) {
        let x = t % u.w; let y = t / u.w;
        var fx: f32;
        var fy: f32;
        if (u.theme == 1u && u.gx >= 2u) {
            let ph = Hi * vec3f(f32(x) + 1.0, f32(y) + 1.0, 1.0);
            if (abs(ph.z) < 1e-8) { continue; }
            let disp = ffd_disp(ph.x / ph.z - 1.0, ph.y / ph.z - 1.0, sw, shh);
            fx = ph.x / ph.z - 1.0 + disp.x;
            fy = ph.y / ph.z - 1.0 + disp.y;
        } else {
            let disp = ffd_disp(f32(x), f32(y), f32(u.w), f32(u.h));
            let ph = Hi * vec3f(f32(x) + 1.0 + disp.x, f32(y) + 1.0 + disp.y, 1.0);
            if (abs(ph.z) < 1e-8) { continue; }
            fx = ph.x / ph.z - 1.0;
            fy = ph.y / ph.z - 1.0;
        }
        if (fx < 2.0 || fy < 2.0 || fx > sw - 3.0 || fy > shh - 3.0) { continue; }
        let wt = pair_weight(fx, fy, t) * taper(fx, fy).x;
        if (wt <= 0.0) { continue; }
        let va = vec4f(bilerp_plane_a(0u, u.cw, u.ch, fx, fy), bilerp_plane_a(1u, u.cw, u.ch, fx, fy),
                       bilerp_plane_a(2u, u.cw, u.ch, fx, fy), bilerp_plane_a(3u, u.cw, u.ch, fx, fy));
        let vb = vec4f(in_b[t], in_b[dhw + t], in_b[2u * dhw + t], in_b[3u * dhw + t]);
        s0 = s0 + wt;
        sa = sa + wt * va;
        sb = sb + wt * vb;
        let wa = wt * va;
        let wb = wt * vb;
        c0 = c0 + wa.x * vb; c1 = c1 + wa.y * vb; c2 = c2 + wa.z * vb; c3 = c3 + wa.w * vb;
        p0 = p0 + wa.x * va; p1 = p1 + wa.y * va; p2 = p2 + wa.z * va; p3 = p3 + wa.w * va;
        q0 = q0 + wb.x * vb; q1 = q1 + wb.y * vb; q2 = q2 + wb.z * vb; q3 = q3 + wb.w * vb;
    }
    // the array layout: 0 Σw, 1–4 Σw a, 5–8 Σw b, 9–24 Σw a_i b_j, 25–34 / 35–44 upper triangles
    var acc: array<f32, 48>;
    acc[0] = s0;
    let c = array<vec4f, 4>(c0, c1, c2, c3);
    let pp = array<vec4f, 4>(p0, p1, p2, p3);
    let qq = array<vec4f, 4>(q0, q1, q2, q3);
    for (var i = 0u; i < 4u; i++) {
        acc[1u + i] = sa[i];
        acc[5u + i] = sb[i];
        for (var j = 0u; j < 4u; j++) { acc[9u + i * 4u + j] = c[i][j]; }
        for (var j = i; j < 4u; j++) {
            let k = tri_k(i, j);
            acc[25u + k] = pp[i][j];
            acc[35u + k] = qq[i][j];
        }
    }
    let sums = reduceMoments48(acc, lid.x);
    if (lid.x == 0u) {
        let base = (pose * nwg.x + wid.x) * N_MOM;
        for (var m = 0u; m < N_MOM; m++) { out_a[base + m] = sums[m]; }
    }
}

// Per-pixel ∂score/∂v_a (4-vector) for dS = Σ 2 dv_aᵀ (W t_b − M t_a), t = v − μ.
// Global: W = C, M = 0. Exact: W = G_A⁻¹ C G_B⁻¹, M = W Cᵀ G_A⁻¹.
fn score_dva(e0: u32, va: array<f32, 4>, vb: array<f32, 4>) -> array<f32, 4> {
    var tb: array<f32, 4>;
    var ta: array<f32, 4>;
    for (var i = 0u; i < 4u; i++) {
        tb[i] = vb[i] - in_e[e0 + 16u + i];
        ta[i] = va[i] - in_e[e0 + 36u + i];
    }
    var r: array<f32, 4>;
    for (var a = 0u; a < 4u; a++) {
        var s = 0.0;
        for (var b = 0u; b < 4u; b++) {
            s = s + in_e[e0 + a * 4u + b] * tb[b] - in_e[e0 + 20u + a * 4u + b] * ta[b];
        }
        r[a] = 2.0 * s;
    }
    // Generic scores: dS/dv_a = g_a + 2 W v_b − 2 M v_a (μ slots are zero).
    if (in_e[e0 + COEF_FLAG] > 0.5) {
        for (var a = 0u; a < 4u; a++) { r[a] = r[a] + in_e[e0 + G0 + 1u + a]; }
    }
    return r;
}

// tangent_grad's per-pose constants (workgroup memory, filled once): the generators pulled
// back through the pose, M_k = −H⁻¹ G_k (so ∂(sample)/∂(tangent k) comes from M_k X), and the
// coefficients as rows: W (0–3), M̂ (4–7), Q (8–11), μ_B (12), μ_A (13), the generic score's
// ∂S/∂v_a (14, zero for SMI).
var<workgroup> tg_M: array<mat3x3f, 8>;
var<workgroup> tg_c: array<vec4f, 15>;

// ∂(sample position)/∂(tangent k) at the pixel X (ph = H⁻¹ X, iz2 = 1 / ph.z²)
fn tgDp(k: u32, X: vec3f, ph: vec3f, iz2: f32) -> vec2f {
    let w = tg_M[k] * X;
    return vec2f(w.x * ph.z - ph.x * w.z, w.y * ph.z - ph.y * w.z) * iz2;
}

// Row r of tg_c from the coefficients at in_e[e0 ..]: W (0–3), M̂ (4–7), Q (8–11), μ_B (12),
// μ_A (13), the generic score's ∂S/∂v_a (14; zero for SMI).
fn tgCoefRow(r: u32, e0: u32) -> vec4f {
    if (r < 12u) {
        let base = e0 + select(select(Q0, 20u, r < 8u), 0u, r < 4u) + (r % 4u) * 4u;
        return vec4f(in_e[base], in_e[base + 1u], in_e[base + 2u], in_e[base + 3u]);
    }
    if (r == 12u) { return vec4f(in_e[e0 + 16u], in_e[e0 + 17u], in_e[e0 + 18u], in_e[e0 + 19u]); }
    if (r == 13u) { return vec4f(in_e[e0 + 36u], in_e[e0 + 37u], in_e[e0 + 38u], in_e[e0 + 39u]); }
    let g = e0 + G0 + 1u;
    return select(vec4f(0.0), vec4f(in_e[g], in_e[g + 1u], in_e[g + 2u], in_e[g + 3u]), in_e[e0 + COEF_FLAG] > 0.5);
}

// one vec4 of a workgroup's outputs: reduced over the workgroup, written at o + m
fn tgOut(lid: u32, o: u32, m: u32, v: vec4f) {
    shv[lid] = v;
    workgroupBarrier();
    reduce256v4(lid);
    if (lid == 0u) {
        out_a[o + m] = shv[0].x;
        out_a[o + m + 1u] = shv[0].y;
        out_a[o + m + 2u] = shv[0].z;
        out_a[o + m + 3u] = shv[0].w;
    }
    workgroupBarrier();
}

// The score's gradient over the group's tangent coordinates (and, u.min_n == 1, the Gauss–Newton
// matrix Σ J_aᵀ Q J_a) at pose wid.z of in_c, its coefficients in_e: per workgroup, 44 partial
// sums (8 gradient, then the matrix's upper triangle row by row) into out_a. Per pixel, the
// gradient needs only the two scalars rx, ry = (∂S/∂v_a) · ∇v_a, times each tangent's
// ∂(sample)/∂(tangent).
@compute @workgroup_size(256)
fn tangent_grad(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u,
    @builtin(num_workgroups) nwg: vec3u) {
    let nPose = max(u.n, 1u);
    if (wid.z >= nPose) { return; }
    let pose = wid.z;
    let dhw = hw();
    let e0 = pose * N_COEF;
    if (lid.x == 0u) {
        let H0 = loadHAt_c(pose * 9u);
        matTo9(H0, 0u);
        matTo9(inverse3(H0), 9u);
        cpsBase = select(0u, pose * max(u.gx, 2u) * max(u.gy, 2u) * 2u, u.plane == 1u);
    }
    if (lid.x >= 32u && lid.x < 47u) { tg_c[lid.x - 32u] = tgCoefRow(lid.x - 32u, e0); }
    workgroupBarrier();
    let Hi = matFrom9(9u);
    if (lid.x < 8u) {
        // the generator (zero past the group's tangents) as a column-major mat3
        let o = lid.x * 9u;
        let G = mat3x3f(
            vec3f(in_d[o], in_d[o + 3u], in_d[o + 6u]),
            vec3f(in_d[o + 1u], in_d[o + 4u], in_d[o + 7u]),
            vec3f(in_d[o + 2u], in_d[o + 5u], in_d[o + 8u]),
        );
        tg_M[lid.x] = (Hi * G) * -1.0;
    }
    workgroupBarrier();
    let W0 = tg_c[0]; let W1 = tg_c[1]; let W2 = tg_c[2]; let W3 = tg_c[3];
    let M0 = tg_c[4]; let M1 = tg_c[5]; let M2 = tg_c[6]; let M3 = tg_c[7];
    let mb = tg_c[12]; let ma = tg_c[13]; let gv = tg_c[14];
    let sw = f32(u.cw); let shh = f32(u.ch);
    let cw = u.cw;
    let ps = u.cw * u.ch;
    // u.min_n == 1: also accumulate the Gauss–Newton matrix Σ J_aᵀ Q J_a (upper triangle).
    let want_h = u.min_n == 1u;
    var accA = vec4f(0.0); var accB = vec4f(0.0);
    // the Gauss–Newton blocks: h0l column l (rows 0–3), h1l column l (rows 4–7), l = column
    var h00 = vec4f(0.0); var h01 = vec4f(0.0); var h02 = vec4f(0.0); var h03 = vec4f(0.0);
    var h04 = vec4f(0.0); var h05 = vec4f(0.0); var h06 = vec4f(0.0); var h07 = vec4f(0.0);
    var h14 = vec4f(0.0); var h15 = vec4f(0.0); var h16 = vec4f(0.0); var h17 = vec4f(0.0);
    for (var t = wid.x * 256u + lid.x; t < dhw; t = t + 256u * nwg.x) {
        let x = t % u.w; let y = t / u.w;
        var fx: f32;
        var fy: f32;
        var X: vec3f;
        if (u.theme == 1u && u.gx >= 2u) {
            X = vec3f(f32(x) + 1.0, f32(y) + 1.0, 1.0);
            let ph0 = Hi * X;
            if (abs(ph0.z) < 1e-8) { continue; }
            let disp = ffd_disp(ph0.x / ph0.z - 1.0, ph0.y / ph0.z - 1.0, sw, shh);
            fx = ph0.x / ph0.z - 1.0 + disp.x;
            fy = ph0.y / ph0.z - 1.0 + disp.y;
        } else {
            let disp = ffd_disp(f32(x), f32(y), f32(u.w), f32(u.h));
            X = vec3f(f32(x) + 1.0 + disp.x, f32(y) + 1.0 + disp.y, 1.0);
            let ph0 = Hi * X;
            if (abs(ph0.z) < 1e-8) { continue; }
            fx = ph0.x / ph0.z - 1.0;
            fy = ph0.y / ph0.z - 1.0;
        }
        let ph = Hi * X;
        if (fx < 2.0 || fy < 2.0 || fx > sw - 3.0 || fy > shh - 3.0) { continue; }
        let we = pair_weight(fx, fy, t);
        let tp = taper(fx, fy);
        let wt = we * tp.x;
        // the moving planes, bilinear with their gradients (the taps are inside: fx, fy ≥ 2)
        let x0 = floor(fx); let y0 = floor(fy);
        let ax = fx - x0; let ay = fy - y0;
        let i00 = u32(y0) * cw + u32(x0);
        let i01 = i00 + cw;
        let v00 = vec4f(in_a[i00], in_a[ps + i00], in_a[2u * ps + i00], in_a[3u * ps + i00]);
        let v10 = vec4f(in_a[i00 + 1u], in_a[ps + i00 + 1u], in_a[2u * ps + i00 + 1u], in_a[3u * ps + i00 + 1u]);
        let v01 = vec4f(in_a[i01], in_a[ps + i01], in_a[2u * ps + i01], in_a[3u * ps + i01]);
        let v11 = vec4f(in_a[i01 + 1u], in_a[ps + i01 + 1u], in_a[2u * ps + i01 + 1u], in_a[3u * ps + i01 + 1u]);
        let va = mix(mix(v00, v10, ax), mix(v01, v11, ax), ay);
        let gxv = mix(v10 - v00, v11 - v01, ay);
        let gyv = mix(v01 - v00, v11 - v10, ax);
        let vb = vec4f(in_b[t], in_b[dhw + t], in_b[2u * dhw + t], in_b[3u * dhw + t]);
        // ∂S/∂v_a (score_dva)
        let tb = vb - mb;
        let ta = va - ma;
        let ra = 2.0 * (vec4f(dot(W0, tb), dot(W1, tb), dot(W2, tb), dot(W3, tb)) - vec4f(dot(M0, ta), dot(M1, ta), dot(M2, ta), dot(M3, ta))) + gv;
        // Boundary term: the taper weight moves with the sample position.
        var bw = 0.0;
        if (tp.y != 0.0 || tp.z != 0.0) {
            bw = we * boundary_value(e0, array<f32, 4>(va.x, va.y, va.z, va.w), array<f32, 4>(vb.x, vb.y, vb.z, vb.w));
        }
        let cx = wt * dot(ra, gxv) + bw * tp.y;
        let cy = wt * dot(ra, gyv) + bw * tp.z;
        let iz2 = 1.0 / max(ph.z * ph.z, 1e-12);
        let d0 = tgDp(0u, X, ph, iz2);
        let d1 = tgDp(1u, X, ph, iz2);
        let d2 = tgDp(2u, X, ph, iz2);
        let d3 = tgDp(3u, X, ph, iz2);
        let d4 = tgDp(4u, X, ph, iz2);
        let d5 = tgDp(5u, X, ph, iz2);
        let d6 = tgDp(6u, X, ph, iz2);
        let d7 = tgDp(7u, X, ph, iz2);
        let pxa = vec4f(d0.x, d1.x, d2.x, d3.x);
        let pxb = vec4f(d4.x, d5.x, d6.x, d7.x);
        let pya = vec4f(d0.y, d1.y, d2.y, d3.y);
        let pyb = vec4f(d4.y, d5.y, d6.y, d7.y);
        accA = accA + cx * pxa + cy * pya;
        accB = accB + cx * pxb + cy * pyb;
        if (want_h) {
            // J_a[a][k] = gx_a·px_k + gy_a·py_k (g: the moving planes' gradients), so
            // J_aᵀ Q J_a (k, l) = px_k (px_l Sxx + py_l Sxy) + py_k (px_l Syx + py_l Syy) with
            // S.. = g.ᵀ Q g.: four scalars per pixel instead of the 4 × 8 products.
            let qgx = vec4f(dot(tg_c[8], gxv), dot(tg_c[9], gxv), dot(tg_c[10], gxv), dot(tg_c[11], gxv));
            let qgy = vec4f(dot(tg_c[8], gyv), dot(tg_c[9], gyv), dot(tg_c[10], gyv), dot(tg_c[11], gyv));
            let sxx = dot(gxv, qgx);
            let sxy = dot(gxv, qgy);
            let syx = dot(gyv, qgx);
            let syy = dot(gyv, qgy);
            // hes = px ⊗ r + py ⊗ s with r = wt (Sxx px + Sxy py), s = wt (Syx px + Syy py): the
            // upper blocks of the 8 × 8 (tangents 0–3, 4–7) as vec4 columns in registers
            let ra2 = wt * (sxx * pxa + sxy * pya);
            let rb2 = wt * (sxx * pxb + sxy * pyb);
            let sa2 = wt * (syx * pxa + syy * pya);
            let sb2 = wt * (syx * pxb + syy * pyb);
            // column l of block (row half, column half): Σ_k-rows p(k) r(l) + q(k) s(l)
            h00 = h00 + pxa * ra2.x + pya * sa2.x; h01 = h01 + pxa * ra2.y + pya * sa2.y;
            h02 = h02 + pxa * ra2.z + pya * sa2.z; h03 = h03 + pxa * ra2.w + pya * sa2.w;
            h04 = h04 + pxa * rb2.x + pya * sb2.x; h05 = h05 + pxa * rb2.y + pya * sb2.y;
            h06 = h06 + pxa * rb2.z + pya * sb2.z; h07 = h07 + pxa * rb2.w + pya * sb2.w;
            h14 = h14 + pxb * rb2.x + pyb * sb2.x; h15 = h15 + pxb * rb2.y + pyb * sb2.y;
            h16 = h16 + pxb * rb2.z + pyb * sb2.z; h17 = h17 + pxb * rb2.w + pyb * sb2.w;
        }
    }
    // the workgroup's partial sums, four at a time (constant indices throughout: no local arrays)
    let o = (pose * nwg.x + wid.x) * 44u;
    tgOut(lid.x, o, 0u, vec4f(accA.x, accA.y, accA.z, accA.w));
    tgOut(lid.x, o, 4u, vec4f(accB.x, accB.y, accB.z, accB.w));
    if (!want_h) { return; }
    tgOut(lid.x, o, 8u, vec4f(h00.x, h01.x, h02.x, h03.x));
    tgOut(lid.x, o, 12u, vec4f(h04.x, h05.x, h06.x, h07.x));
    tgOut(lid.x, o, 16u, vec4f(h01.y, h02.y, h03.y, h04.y));
    tgOut(lid.x, o, 20u, vec4f(h05.y, h06.y, h07.y, h02.z));
    tgOut(lid.x, o, 24u, vec4f(h03.z, h04.z, h05.z, h06.z));
    tgOut(lid.x, o, 28u, vec4f(h07.z, h03.w, h04.w, h05.w));
    tgOut(lid.x, o, 32u, vec4f(h06.w, h07.w, h14.x, h15.x));
    tgOut(lid.x, o, 36u, vec4f(h16.x, h17.x, h15.y, h16.y));
    tgOut(lid.x, o, 40u, vec4f(h17.y, h16.z, h17.z, h17.w));
}

@compute @workgroup_size(256)
fn ffd_grad(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u,
    @builtin(num_workgroups) nwg: vec3u) {
    // Chunk of 16 control points (32 params). u.n_keys is the first CP index.
    // u.n is the pose count. u.plane == 1 selects that pose's control points.
    let nPose = max(u.n, 1u);
    if (wid.z >= nPose) { return; }
    let pose = wid.z;
    let dhw = hw();
    let H = loadHAt_c(pose * 9u);
    let detH = determinant(H);
    var acc: array<f32, 32>;
    for (var z = 0u; z < 32u; z++) { acc[z] = 0.0; }
    if (abs(detH) < 1e-12) {
        for (var m = 0u; m < 32u; m++) {
            sh[lid.x] = 0.0;
            workgroupBarrier();
            reduce256(lid.x);
            if (lid.x == 0u) { out_a[(pose * nwg.x + wid.x) * 32u + m] = 0.0; }
            workgroupBarrier();
        }
        return;
    }
    let gx = max(u.gx, 2u);
    let gy = max(u.gy, 2u);
    if (lid.x == 0u) {
        matTo9(inverse3(H), 9u);
        cpsBase = select(0u, pose * gx * gy * 2u, u.plane == 1u);
    }
    workgroupBarrier();
    let Hi = matFrom9(9u);
    let ncp = gx * gy;
    let cp0 = min(u.n_keys, ncp);
    let ntake = min(16u, ncp - cp0);
    let npar = ntake * 2u;
    let sw = f32(u.cw); let shh = f32(u.ch);
    let rw = f32(u.w); let rh = f32(u.h);
    let src_ffd = u.theme == 1u;
    let lat_w = select(rw, sw, src_ffd);
    let lat_h = select(rh, shh, src_ffd);
    let hx = dest_half(lat_w, lat_h);
    let spx = max(lat_w - 1.0, 1.0) / f32(gx - 1u);
    let spy = max(lat_h - 1.0, 1.0) / f32(gy - 1u);
    let e0 = pose * N_COEF;
    for (var t = wid.x * 256u + lid.x; t < dhw; t = t + 256u * nwg.x) {
        let x = t % u.w; let y = t / u.w;
        let xf = f32(x); let yf = f32(y);
        if (xf < 0.0 || yf < 0.0 || xf > rw - 1.0 || yf > rh - 1.0) { continue; }
        var fx: f32;
        var fy: f32;
        var su: f32;
        var sv: f32;
        var X: vec3f;
        var ph: vec3f;
        if (src_ffd) {
            X = vec3f(xf + 1.0, yf + 1.0, 1.0);
            ph = Hi * X;
            if (abs(ph.z) < 1e-8) { continue; }
            let fx0 = ph.x / ph.z - 1.0;
            let fy0 = ph.y / ph.z - 1.0;
            let disp = ffd_disp(fx0, fy0, sw, shh);
            fx = fx0 + disp.x;
            fy = fy0 + disp.y;
            su = fx0 / spx;
            sv = fy0 / spy;
        } else {
            let disp = ffd_disp(xf, yf, rw, rh);
            X = vec3f(xf + 1.0 + disp.x, yf + 1.0 + disp.y, 1.0);
            ph = Hi * X;
            if (abs(ph.z) < 1e-8) { continue; }
            fx = ph.x / ph.z - 1.0;
            fy = ph.y / ph.z - 1.0;
            su = xf / spx;
            sv = yf / spy;
        }
        if (fx < 2.0 || fy < 2.0 || fx > sw - 3.0 || fy > shh - 3.0) { continue; }
        let we = pair_weight(fx, fy, t);
        let tp = taper(fx, fy);
        let wt = we * tp.x;
        var da: array<vec3f, 4>;
        var va: array<f32, 4>;
        var vb: array<f32, 4>;
        for (var a = 0u; a < 4u; a++) {
            da[a] = bilerp_grad_a(a, u.cw, u.ch, fx, fy);
            va[a] = da[a].x;
            vb[a] = in_b[a * dhw + t];
        }
        let ra = score_dva(e0, va, vb);
        var bw = 0.0;
        if (tp.y != 0.0 || tp.z != 0.0) { bw = we * boundary_value(e0, va, vb); }
        let z = ph.z;
        let z2 = max(z * z, 1e-12);
        let iu = i32(floor(su));
        let iv = i32(floor(sv));
        let bu = cubic_w(clamp(su - f32(iu), 0.0, 1.0));
        let bv = cubic_w(clamp(sv - f32(iv), 0.0, 1.0));
        for (var jj = 0u; jj < 4u; jj++) {
            let j = clampu(iv - 1 + i32(jj), gy);
            for (var ii = 0u; ii < 4u; ii++) {
                let i = clampu(iu - 1 + i32(ii), gx);
                // (bk: the basis weight; bw stays the boundary value of the taper term)
                let bk = bu[ii] * bv[jj];
                let idx = j * gx + i;
                if (idx < cp0 || idx >= cp0 + ntake) { continue; }
                let slot = (idx - cp0) * 2u;
                var dpx_x: f32;
                var dpy_x: f32;
                var dpx_y: f32;
                var dpy_y: f32;
                if (src_ffd) {
                    dpx_x = bk * hx.x; dpy_x = 0.0;
                    dpx_y = 0.0; dpy_y = bk * hx.y;
                } else {
                    let wx = Hi * vec3f(bk * hx.x, 0.0, 0.0);
                    let wy = Hi * vec3f(0.0, bk * hx.y, 0.0);
                    dpx_x = (wx.x * z - ph.x * wx.z) / z2;
                    dpy_x = (wx.y * z - ph.y * wx.z) / z2;
                    dpx_y = (wy.x * z - ph.x * wy.z) / z2;
                    dpy_y = (wy.y * z - ph.y * wy.z) / z2;
                }
                var gxacc = 0.0;
                var gyacc = 0.0;
                for (var a = 0u; a < 4u; a++) {
                    gxacc = gxacc + ra[a] * (da[a].y * dpx_x + da[a].z * dpy_x);
                    gyacc = gyacc + ra[a] * (da[a].y * dpx_y + da[a].z * dpy_y);
                }
                gxacc = wt * gxacc + bw * (tp.y * dpx_x + tp.z * dpy_x);
                gyacc = wt * gyacc + bw * (tp.y * dpx_y + tp.z * dpy_y);
                if (gxacc == gxacc) { acc[slot] = acc[slot] + gxacc; }
                if (gyacc == gyacc) { acc[slot + 1u] = acc[slot + 1u] + gyacc; }
            }
        }
    }
    for (var m = 0u; m < 32u; m++) {
        var v = 0.0;
        if (m < npar) { v = acc[m]; }
        if (v != v) { v = 0.0; }
        sh[lid.x] = v;
        workgroupBarrier();
        reduce256(lid.x);
        if (lid.x == 0u) { out_a[(pose * nwg.x + wid.x) * 32u + m] = sh[0]; }
        workgroupBarrier();
    }
}

fn sample_plane_z(px: f32, py: f32, plane: u32) -> f32 {
    let w = i32(u.w); let h = i32(u.h);
    let x0 = i32(floor(px)); let y0 = i32(floor(py));
    let x1 = x0 + 1; let y1 = y0 + 1;
    if (x0 < 0 || y0 < 0 || x1 >= w || y1 >= h) { return 0.0; }
    let fx = px - f32(x0); let fy = py - f32(y0);
    let off = plane * u.w * u.h;
    let xa = u32(x0); let xb = u32(x1); let ya = u32(y0); let yb = u32(y1);
    return mix(
        mix(in_a[off + ya * u.w + xa], in_a[off + ya * u.w + xb], fx),
        mix(in_a[off + yb * u.w + xa], in_a[off + yb * u.w + xb], fx),
        fy,
    );
}

// Spatial log-polar of the dest-frame stack about (u.scale, u.ridge).
// u.w×u.h source, u.n = n_θ (FFT width), u.cw = n_λ,
// u.mean = log(r0), u.stdv = log(r1/r0). The ρ area weight is applied in pack_pair.
@compute @workgroup_size(8, 8, 1)
fn spatial_logpolar(@builtin(global_invocation_id) gid: vec3u) {
    let n_th = u.n;
    let n_lam = u.cw;
    if (gid.x >= n_th || gid.y >= n_lam) { return; }
    let t = (f32(gid.y) + 0.5) / f32(max(n_lam, 1u));
    let rho = exp(u.mean + t * u.stdv);
    let th = (f32(gid.x) + 0.5) / f32(n_th) * 6.28318530718;
    let xs = u.scale + rho * cos(th);
    let ys = u.ridge + rho * sin(th);
    let pix = gid.y * n_th + gid.x;
    let hw = n_lam * n_th;
    for (var p = 0u; p < 5u; p++) {
        out_a[p * hw + pix] = sample_plane_z(xs, ys, p);
    }
}

// Exchange-symmetric tiled SMI (global whitening).
// One workgroup per (tile x, tile y, pose*2+dir). H/Hi broadcast once per WG.
@compute @workgroup_size(16, 16, 1)
fn tile_moments(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u,
                @builtin(local_invocation_index) lin: u32) {
    let G = max(u.gx, 2u);
    let nPose = max(u.n_keys, 1u);
    let pose = wid.z / 2u;
    let dir = wid.z % 2u;
    if (wid.x >= G || wid.y >= G || pose >= nPose) { return; }
    let dw = u.w; let dh = u.h;
    let sw = u.cw; let shh = u.ch;
    let ownW = select(sw, dw, dir == 0u);
    let ownH = select(shh, dh, dir == 0u);
    let x0 = (wid.x * ownW) / G;
    let x1 = ((wid.x + 1u) * ownW) / G;
    let y0 = (wid.y * ownH) / G;
    let y1 = ((wid.y + 1u) * ownH) / G;
    if (lin == 0u) {
        let H0 = loadHAt_c(pose * 9u);
        matTo9(H0, 0u);
        matTo9(inverse3(H0), 9u);
    }
    workgroupBarrier();
    let H = matFrom9(0u);
    let Hi = matFrom9(9u);
    var acc: array<f32, 25>;
    for (var z = 0u; z < 25u; z++) { acc[z] = 0.0; }
    for (var yy = y0 + lid.y; yy < y1; yy = yy + 16u) {
        for (var xx = x0 + lid.x; xx < x1; xx = xx + 16u) {
            let p = vec3f(f32(xx) + 1.0, f32(yy) + 1.0, 1.0);
            var va: array<f32, 4>;
            var vb: array<f32, 4>;
            if (dir == 0u) {
                let q = Hi * p;
                if (abs(q.z) < 1e-8) { continue; }
                let iz = 1.0 / q.z;
                let fx = q.x * iz - 1.0;
                let fy = q.y * iz - 1.0;
                if (fx < 2.0 || fy < 2.0 || fx > f32(sw) - 3.0 || fy > f32(shh) - 3.0) { continue; }
                let t = yy * dw + xx;
                for (var i = 0u; i < 4u; i++) {
                    va[i] = bilerp_plane_a(i, sw, shh, fx, fy);
                    vb[i] = in_b[i * dw * dh + t];
                }
            } else {
                let q = H * p;
                if (abs(q.z) < 1e-8) { continue; }
                let iz = 1.0 / q.z;
                let fx = q.x * iz - 1.0;
                let fy = q.y * iz - 1.0;
                if (fx < 2.0 || fy < 2.0 || fx > f32(dw) - 3.0 || fy > f32(dh) - 3.0) { continue; }
                let t = yy * sw + xx;
                for (var i = 0u; i < 4u; i++) {
                    va[i] = in_a[i * sw * shh + t];
                    vb[i] = bilerp_plane_b(i, dw, dh, fx, fy);
                }
            }
            acc[0] = acc[0] + 1.0;
            for (var i = 0u; i < 4u; i++) {
                acc[1u + i] = acc[1u + i] + va[i];
                acc[5u + i] = acc[5u + i] + vb[i];
            }
            for (var i = 0u; i < 4u; i++) {
                for (var j = 0u; j < 4u; j++) {
                    acc[9u + i * 4u + j] = acc[9u + i * 4u + j] + va[i] * vb[j];
                }
            }
        }
    }
    let sums = reduceMoments25(acc, lin);
    if (lin == 0u) {
        let nT = G * G;
        let outi = pose * 2u * nT + dir * nT + wid.y * G + wid.x;
        out_a[outi] = tileNS(sums, u.min_n);
    }
}

// ── GPU-resident reductions and pose stacks (zcmir) ─────────────────────────────────────
// The host schedules these instead of reading partial sums back: statistics that later kernels
// need (mean, spread, whitening moments, correlation area, the map peak) stay in small
// storage buffers ("prm") and are read by index. Only final results leave the GPU.

var<workgroup> shi: array<u32, 256>;

// out_a[u.plane] = u.scale · Σ in_a[0..64) (64 partial sums from reduce1).
@compute @workgroup_size(1)
fn finish_sum() {
    var s = 0.0;
    for (var g = 0u; g < 64u; g++) { s = s + in_a[g]; }
    out_a[u.plane] = s * u.scale;
}

// z-score with mean = in_b[u.plane], spread = in_b[u.plane_b] (the prm buffer).
@compute @workgroup_size(256)
fn zscore_p(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    let n = hw();
    if (gid.x >= n) { return; }
    out_a[gid.x] = (in_a[gid.x] - in_b[u.plane]) / max(in_b[u.plane_b], 1e-8);
}

// 64 × 21 partial moment sums (feat_moments) → out_a[0..21) (the layout chol reads).
@compute @workgroup_size(32)
fn finish_moments(@builtin(local_invocation_id) lid: vec3u) {
    let m = lid.x;
    if (m >= 21u) { return; }
    var s = 0.0;
    for (var g = 0u; g < 64u; g++) { s = s + in_a[g * 21u + m]; }
    out_a[m] = s;
}

// out_a[u.plane] = sqrt(Σ x0² / count) from finished moments (in_a[0] = count, in_a[5] = Σ x0²).
@compute @workgroup_size(1)
fn finish_std() {
    out_a[u.plane] = sqrt(max(in_a[5] / max(in_a[0], 1.0), 1e-12));
}

// Autocorrelation (feat_acf, 289 lags × 4) normalized by its zero lag, per plane.
@compute @workgroup_size(256)
fn acf_norm(@builtin(global_invocation_id) gid: vec3u) {
    let k = gid.x;
    if (k >= 289u * 4u) { return; }
    out_a[k] = in_a[k] / max(in_a[144u * 4u + (k & 3u)], 1e-12);
}

// Feature correlation area of a pair (px²), Bartlett: (1/16) Σ_τ Σ_ij ρ_a,i(τ) ρ_b,j(τ), ≥ 1.
// in_a, in_b: the two normalized autocorrelations. → out_a[u.plane].
@compute @workgroup_size(256)
fn corr_area(@builtin(local_invocation_id) lid: vec3u) {
    var s = 0.0;
    for (var k = lid.x; k < 289u; k = k + 256u) {
        var ra = 0.0;
        var rb = 0.0;
        for (var i = 0u; i < 4u; i++) { ra = ra + in_a[k * 4u + i]; rb = rb + in_b[k * 4u + i]; }
        s = s + ra * rb;
    }
    sh[lid.x] = s;
    workgroupBarrier();
    reduce256(lid.x);
    if (lid.x == 0u) { out_a[u.plane] = max(1.0, sh[0] / 16.0); }
}

// Argmax of in_a[0..u.n·u.n), first index on ties. Pass 1: 64 workgroups → out_a[2g] = value,
// out_a[2g+1] = bitcast index. Pass 2 (argmax_finish): out_b[0] = peak, [1] = bitcast index,
// [2] = in_b[0] (zero lag of the map).
fn better(v: f32, i: u32, bv: f32, bi: u32) -> bool {
    return v > bv || (v == bv && i < bi);
}

@compute @workgroup_size(256)
fn argmax_part(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let nn = map_w() * map_h();
    var bv = -3.4e38;
    var bi = 0xffffffffu;
    for (var t = wid.x * 256u + lid.x; t < nn; t = t + 256u * 64u) {
        let v = in_a[t];
        if (better(v, t, bv, bi)) { bv = v; bi = t; }
    }
    sh[lid.x] = bv;
    shi[lid.x] = bi;
    workgroupBarrier();
    var k = 128u;
    loop {
        if (k == 0u) { break; }
        if (lid.x < k && better(sh[lid.x + k], shi[lid.x + k], sh[lid.x], shi[lid.x])) {
            sh[lid.x] = sh[lid.x + k];
            shi[lid.x] = shi[lid.x + k];
        }
        workgroupBarrier();
        k = k / 2u;
    }
    if (lid.x == 0u) {
        out_a[wid.x * 2u] = sh[0];
        out_a[wid.x * 2u + 1u] = bitcast<f32>(shi[0]);
    }
}

@compute @workgroup_size(1)
fn argmax_finish() {
    var bv = -3.4e38;
    var bi = 0xffffffffu;
    for (var g = 0u; g < 64u; g++) {
        let v = in_a[g * 2u];
        let i = bitcast<u32>(in_a[g * 2u + 1u]);
        if (i != 0xffffffffu && better(v, i, bv, bi)) { bv = v; bi = i; }
    }
    out_b[0] = bv;
    out_b[1] = bitcast<f32>(bi);
    out_b[2] = in_b[0];
}

// The inverse used by the image warp (gls_mift.wgsl warp_homography): adjugate / det, no
// refinement step, so pose stacks resample exactly as the per-plane warp did.
fn inverse3_plain(m: mat3x3f) -> mat3x3f {
    let a = m[0]; let b = m[1]; let c = m[2];
    let r0 = cross(b, c); let r1 = cross(c, a); let r2 = cross(a, b);
    let id = 1.0 / dot(a, r0);
    return transpose(mat3x3f(r0 * id, r1 * id, r2 * id));
}

// Moving stack (5 planes, u.cw × u.ch) warped onto the u.w × u.h canvas by H = in_c[0..9]
// (row-major, 1-based pixels, moving → canvas). Bilinear with clamped taps inside the source,
// 0 outside, as warp_homography per plane.
@compute @workgroup_size(8, 8)
fn warp_stack(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= u.w || gid.y >= u.h) { return; }
    let n = u.w * u.h;
    let pix = gid.y * u.w + gid.x;
    let H = loadH_c();
    var ok = abs(determinant(H)) >= 1e-12;
    var fx = 0.0;
    var fy = 0.0;
    if (ok) {
        let q = inverse3_plain(H) * vec3f(f32(gid.x) + 1.0, f32(gid.y) + 1.0, 1.0);
        ok = abs(q.z) >= 1e-8;
        if (ok) {
            fx = q.x / q.z - 1.0;
            fy = q.y / q.z - 1.0;
            ok = !(fx < 0.0 || fy < 0.0 || fx > f32(u.cw - 1u) || fy > f32(u.ch - 1u));
        }
    }
    for (var p = 0u; p < 5u; p++) {
        var v = 0.0;
        if (ok) {
            let x0 = i32(floor(fx));
            let y0 = i32(floor(fy));
            let tx = fx - f32(x0);
            let ty = fy - f32(y0);
            v = mix(mix(at_plane_a(p, u.cw, u.ch, x0, y0), at_plane_a(p, u.cw, u.ch, x0 + 1, y0), tx),
                    mix(at_plane_a(p, u.cw, u.ch, x0, y0 + 1), at_plane_a(p, u.cw, u.ch, x0 + 1, y0 + 1), tx), ty);
        }
        out_a[p * n + pix] = v;
    }
}

// Fixed stack (5 planes, u.cw × u.ch) pasted into the u.w × u.h canvas at offset
// (bitcast<i32>(u.gx), bitcast<i32>(u.gy)); 0 elsewhere.
@compute @workgroup_size(8, 8)
fn paste_stack(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= u.w || gid.y >= u.h) { return; }
    let n = u.w * u.h;
    let pix = gid.y * u.w + gid.x;
    let sx = i32(gid.x) - bitcast<i32>(u.gx);
    let sy = i32(gid.y) - bitcast<i32>(u.gy);
    let inside = sx >= 0 && sy >= 0 && sx < i32(u.cw) && sy < i32(u.ch);
    let sn = u.cw * u.ch;
    for (var p = 0u; p < 5u; p++) {
        var v = 0.0;
        if (inside) { v = in_a[p * sn + u32(sy) * u.cw + u32(sx)]; }
        out_a[p * n + pix] = v;
    }
}

// Roto-scale map post-processing (smi.js rsSymmetric + maskRsScale), per cell t of the n×n map:
// in_a = forward map; u.plane_b == 1: average with in_b (the inverse-frame map) at the mirrored
// lag (−y, −x), or (−y, x) when u.gx == 0 (a reflecting pose); then zero λ rows whose applied
// scale exp(−k Δλ) lies outside [e^lo, e^hi]: u.mean = Δλ, u.stdv = lo, u.scale = hi.
@compute @workgroup_size(256)
fn rs_finish(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    let n = u.n;
    let t = gid.x;
    if (t >= n * n) { return; }
    let y = t / n;
    let x = t - y * n;
    var v = in_a[t];
    if (u.plane_b == 1u) {
        let ym = (n - y) % n;
        let xm = select(x, (n - x) % n, u.gx == 1u);
        v = 0.5 * (v + in_b[ym * n + xm]);
    }
    var k = f32(y);
    if (y > n / 2u) { k = f32(i32(y) - i32(n)); }
    let sg = -k * u.mean;
    if (sg < u.stdv || sg > u.scale) { v = 0.0; }
    out_a[t] = v;
}

// ── the gradient kernels' coefficients on the GPU ───────────────────────────────────────────
// moments.zig scoreFromMoments (SMI) in f32, so a symmetric gradient needs no round trip
// between its moments pass and tangent_grad (f32 here moves Refine's poses by ~0.002 px).

fn mc_chol4(G: array<f32, 16>, ok: ptr<function, bool>) -> array<f32, 16> {
    var L: array<f32, 16>;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j <= i; j++) {
            var s = G[i * 4u + j];
            for (var k = 0u; k < j; k++) { s = s - L[i * 4u + k] * L[j * 4u + k]; }
            if (i == j) {
                if (!(s > 0.0)) { *ok = false; return L; }
                L[i * 5u] = sqrt(s);
            } else {
                L[i * 4u + j] = s / L[j * 5u];
            }
        }
    }
    return L;
}

fn mc_cinv4(L: array<f32, 16>) -> array<f32, 16> {
    var X: array<f32, 16>;
    for (var c = 0u; c < 4u; c++) {
        var y: array<f32, 4>;
        for (var i = 0u; i < 4u; i++) {
            var s = select(0.0, 1.0, i == c);
            for (var k = 0u; k < i; k++) { s = s - L[i * 4u + k] * y[k]; }
            y[i] = s / L[i * 5u];
        }
        for (var r = 0u; r < 4u; r++) {
            let i = 3u - r;
            var s = y[i];
            for (var k = i + 1u; k < 4u; k++) { s = s - L[k * 4u + i] * X[k * 4u + c]; }
            X[i * 4u + c] = s / L[i * 5u];
        }
    }
    return X;
}

fn mc_mm4(A: array<f32, 16>, B: array<f32, 16>) -> array<f32, 16> {
    var o: array<f32, 16>;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            var s = 0.0;
            for (var k = 0u; k < 4u; k++) { s = s + A[i * 4u + k] * B[k * 4u + j]; }
            o[i * 4u + j] = s;
        }
    }
    return o;
}

fn mc_tr4(A: array<f32, 16>) -> array<f32, 16> {
    var o: array<f32, 16>;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) { o[i * 4u + j] = A[j * 4u + i]; }
    }
    return o;
}

// moments_coefs' per-direction inputs and outputs
fn mcPart(lane: u32, i: u32) -> f32 {
    if (lane == 1u) { return in_c[i]; }
    return in_a[i];
}
fn mcCoef(lane: u32, i: u32, v: f32) {
    if (lane == 1u) { out_c[i] = v; } else { out_a[i] = v; }
}
fn mcScore(lane: u32, pose: u32, v: f32) {
    if (lane == 1u) { out_d[pose] = v; } else { out_b[pose] = v; }
}

// One workgroup per pose (wid.x) and direction (wid.y: lane 0 in_a → out_a, out_b; lane 1 in_c →
// out_c, out_d): the pose's partial rows of N_MOM sums (the moments pass's output; u.n rows per
// pose in lane 0, u.n_keys in lane 1, u.cw / u.ch poses) are summed, then its coefficients
// written to out[pose · N_COEF ..] and its score to the score output [pose] (NO_SCORE when
// undefined). u.plane: 1 exact whitening (u.ridge), 0 global.
// Too little overlap or a failed whitening leaves the coefficients zero, as on the host.
// u.gx 2 / 3: the copula scores E4 / λmax instead (moments.zig copulaFromMoments): their
// gradient by central differences in the sums, one sum per thread, as generic coefficients.
@compute @workgroup_size(64)
fn moments_coefs(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let pose = wid.x;
    let lane = wid.y;
    if (pose >= select(u.cw, u.ch, lane == 1u)) { return; }
    if (lid < N_MOM) {
        var s = 0.0;
        let np = select(u.n, u.n_keys, lane == 1u);
        for (var p = 0u; p < np; p++) { s = s + mcPart(lane, (pose * np + p) * N_MOM + lid); }
        sh[lid] = s;
    }
    workgroupBarrier();
    if (u.gx >= 2u) {
        mc_copula(lane, pose, lid);
        return;
    }
    if (lid != 0u) { return; }
    let o = pose * N_COEF;
    for (var k = 0u; k < N_COEF; k++) { mcCoef(lane, o + k, 0.0); }
    mcScore(lane, pose, NO_SCORE);
    let n = sh[0];
    if (!(n >= 12.0)) { return; }
    var ma: array<f32, 4>;
    var mb: array<f32, 4>;
    var C: array<f32, 16>;
    for (var i = 0u; i < 4u; i++) {
        ma[i] = sh[1u + i] / n;
        mb[i] = sh[5u + i] / n;
    }
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) { C[i * 4u + j] = sh[9u + i * 4u + j] / n - ma[i] * mb[j]; }
    }
    var W = C;
    var Mh: array<f32, 16>;
    var Nh: array<f32, 16>;
    var phi = 0.0;
    let exact = u.plane == 1u;
    if (!exact) {
        for (var i = 0u; i < 16u; i++) { phi = phi + C[i] * C[i]; }
    } else {
        var GA: array<f32, 16>;
        var GB: array<f32, 16>;
        for (var i = 0u; i < 4u; i++) {
            for (var j = 0u; j < 4u; j++) {
                let k = tri_k(i, j);
                GA[i * 4u + j] = sh[25u + k] / n - ma[i] * ma[j];
                GB[i * 4u + j] = sh[35u + k] / n - mb[i] * mb[j];
            }
        }
        let ra = u.ridge * max((GA[0] + GA[5] + GA[10] + GA[15]) / 4.0, 1e-12);
        let rb = u.ridge * max((GB[0] + GB[5] + GB[10] + GB[15]) / 4.0, 1e-12);
        for (var i = 0u; i < 4u; i++) {
            GA[i * 5u] = GA[i * 5u] + ra;
            GB[i * 5u] = GB[i * 5u] + rb;
        }
        var ok = true;
        let LA = mc_chol4(GA, &ok);
        let LB = mc_chol4(GB, &ok);
        if (!ok) { return; }
        let GAi = mc_cinv4(LA);
        let GBi = mc_cinv4(LB);
        W = mc_mm4(mc_mm4(GAi, C), GBi);
        for (var i = 0u; i < 16u; i++) { phi = phi + W[i] * C[i]; }
        Mh = mc_mm4(mc_mm4(W, mc_tr4(C)), GAi);
        Nh = mc_mm4(mc_mm4(mc_tr4(W), C), GBi);
        let tm = (Mh[0] + Mh[5] + Mh[10] + Mh[15]) * u.ridge / 4.0;
        let tn = (Nh[0] + Nh[5] + Nh[10] + Nh[15]) * u.ridge / 4.0;
        for (var i = 0u; i < 4u; i++) {
            Mh[i * 5u] = Mh[i * 5u] + tm;
            Nh[i * 5u] = Nh[i * 5u] + tn;
        }
    }
    let score = n * phi;
    if (!(score == score) || abs(score) > 3.0e38) { return; }
    mcScore(lane, pose, score);
    for (var i = 0u; i < 16u; i++) {
        mcCoef(lane, o + i, W[i]);
        mcCoef(lane, o + 20u + i, Mh[i]);
    }
    for (var i = 0u; i < 4u; i++) {
        mcCoef(lane, o + 16u + i, mb[i]);
        mcCoef(lane, o + 36u + i, ma[i]);
    }
    var g: array<f32, 45>;
    for (var i = 0u; i < 4u; i++) {
        var ga = 0.0;
        var gb = 0.0;
        for (var j = 0u; j < 4u; j++) {
            ga = ga - 2.0 * W[i * 4u + j] * mb[j] + 2.0 * Mh[i * 4u + j] * ma[j];
            gb = gb - 2.0 * W[j * 4u + i] * ma[j] + 2.0 * Nh[i * 4u + j] * mb[j];
            g[9u + i * 4u + j] = 2.0 * W[i * 4u + j];
        }
        g[1u + i] = ga;
        g[5u + i] = gb;
        for (var j = i; j < 4u; j++) {
            let k = tri_k(i, j);
            let f = select(2.0, 1.0, i == j);
            g[25u + k] = -f * Mh[i * 4u + j];
            g[35u + k] = -f * Nh[i * 4u + j];
        }
    }
    var rest = 0.0;
    for (var k = 1u; k < N_MOM; k++) { rest = rest + g[k] * sh[k]; }
    g[0] = (score - rest) / n;
    for (var k = 0u; k < N_MOM; k++) { mcCoef(lane, o + G0 + k, g[k]); }
    var Q = Mh;
    if (!exact) { Q = mc_mm4(C, mc_tr4(C)); }
    for (var i = 0u; i < 16u; i++) { mcCoef(lane, o + Q0 + i, Q[i]); }
}

const NO_SCORE: f32 = -3.0e38;

// The chosen pose (index in_d[0], from the GPU climb's climb_select) of each direction, as the
// tangent pass's single pose: its pose (9 floats) at [0, 9) and its gradient coefficients
// (N_COEF) from [SEL_COEF) of the direction's selection buffer. Lane 0: poses in_c, coefficients
// in_a → out_a; lane 1 (u.n == 2): in_f, in_e → out_b.
const SEL_COEF: u32 = 64u;
@compute @workgroup_size(64)
fn gather_sel(@builtin(local_invocation_index) lid: u32) {
    let k = u32(in_d[0]);
    for (var j = lid; j < N_COEF; j = j + 64u) { out_a[SEL_COEF + j] = in_a[k * N_COEF + j]; }
    if (lid < 9u) { out_a[lid] = in_c[k * 9u + lid]; }
    if (u.n < 2u) { return; }
    for (var j = lid; j < N_COEF; j = j + 64u) { out_b[SEL_COEF + j] = in_e[k * N_COEF + j]; }
    if (lid < 9u) { out_b[lid] = in_f[k * 9u + lid]; }
}

// ── the copula scores' coefficients (moments_coefs, u.gx 2 / 3) ──

var<workgroup> mc_g: array<f32, 45>;

// n · I of the sums in `st` (I: copula_score of the sums over n; 0 when undefined)
fn mc_copula_score(st: ptr<function, array<f32, 45>>) -> f32 {
    let n = (*st)[0];
    if (!(n >= 12.0)) { return 0.0; }
    var m: array<f32, 45>;
    for (var k = 1u; k < N_MOM; k++) { m[k] = (*st)[k] / n; }
    let I = copula_score(&m, u.gx, u.ridge);
    return select(0.0, n * I, abs(I) < 1e30);
}

// Thread k < the moments the score uses (25 for E4, 45 for λmax) differentiates in sum k
// (central differences, step 1e-3 of the sum's scale: f32 needs a larger step than the host's
// 1e-5 in f64); thread 0 then writes the generic coefficients (moments.zig copulaFromMoments)
// and, for λmax, the Gauss–Newton weight Q (copula.wgsl copula_lmax_q).
fn mc_copula(lane: u32, pose: u32, lid: u32) {
    let used = select(N_MOM, 25u, u.gx == 2u);
    var st: array<f32, 45>;
    for (var k = 0u; k < N_MOM; k++) { st[k] = sh[k]; }
    let n = st[0];
    let s0 = mc_copula_score(&st);
    var gk = 0.0;
    if (lid < used && n >= 12.0 && s0 != 0.0) {
        let h = 1e-3 * max(max(abs(st[lid]), 1e-3 * n), 1e-6);
        let v = st[lid];
        st[lid] = v + h;
        let up = mc_copula_score(&st);
        st[lid] = v - h;
        let dn = mc_copula_score(&st);
        st[lid] = v;
        gk = (up - dn) / (2.0 * h);
    }
    if (lid < N_MOM) { mc_g[lid] = gk; }
    workgroupBarrier();
    if (lid != 0u) { return; }
    let o = pose * N_COEF;
    for (var k = 0u; k < N_COEF; k++) { mcCoef(lane, o + k, 0.0); }
    mcScore(lane, pose, select(NO_SCORE, s0, n >= 12.0));
    if (!(n >= 12.0) || s0 == 0.0) { return; }
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            mcCoef(lane, o + i * 4u + j, 0.5 * mc_g[9u + i * 4u + j]);
            let kk = tri_k(i, j);
            mcCoef(lane, o + 20u + i * 4u + j, select(-0.5 * mc_g[25u + kk], -mc_g[25u + kk], i == j));
        }
    }
    for (var k = 0u; k < N_MOM; k++) { mcCoef(lane, o + G0 + k, mc_g[k]); }
    mcCoef(lane, o + COEF_FLAG, 1.0);
    // λmax: its Gauss–Newton weight (E4 has none: its climb takes line searches)
    if (u.gx == 3u) {
        var m: array<f32, 45>;
        for (var k = 1u; k < N_MOM; k++) { m[k] = st[k] / n; }
        let Q = copula_lmax_q(&m, u.ridge);
        for (var i = 0u; i < 16u; i++) { mcCoef(lane, o + Q0 + i, Q[i]); }
    }
}

// ── the spline's Gauss–Newton terms per lattice cell (gclimb.zig) ──
// For pose 0 of in_c (H) with its coefficients (in_e), the live spline (in_f) and the lattice of
// u.gx × u.gy control points (u.theme 1: on the moving image, else the fixed), workgroup wid.x
// takes a run of lattice cell (iu, iv)'s pixel tiles (cell wid.x / u.n_keys, (iu, iv) = (cell % gx,
// cell / gx)): the pixels whose spline coordinate falls in it. Each contributes, through its 4 × 4 basis weights bu_i bv_j, the score gradient
// in the displacement (2-vector, as ffd_grad) and the Gauss–Newton 2 × 2 Jdᵀ S Jd (S = ∇vᵀ Q ∇v,
// as tangent_grad's pose matrix). Out (FFD_CELL sums per cell): 300 = 100 symmetric basis-pair
// products (pair i ≤ i2 × pair j ≤ j2, tri_i / tri_j order) × the 2 × 2's xx, xy, yy, then
// 32 = the 16 basis weights (ii + 4 jj) × the gradient's x, y.
const FFD_CELL: u32 = 332u;
// ffd_cells' sums per thread: 100 basis pairs (× 3) and 16 taps (× 2), twice (two halves of a tile)
const FC_TASKS: u32 = 116u;

var<workgroup> fc_bu: array<vec4f, 256>;
var<workgroup> fc_bv: array<vec4f, 256>;
var<workgroup> fc_k: array<vec4f, 256>;   // Kxx, Kxy, Kyy, gx
var<workgroup> fc_gy: array<f32, 256>;

// The positive semi-definite part of a symmetric 4 × 4 (its negative eigenvalues dropped, by
// cyclic Jacobi rotations), so a Gauss–Newton matrix built on it is too.
fn psd4(q0: array<f32, 16>) -> array<f32, 16> {
    var a = q0;
    var v = array<f32, 16>(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    for (var sweep = 0u; sweep < 8u; sweep++) {
        for (var p = 0u; p < 3u; p++) {
            for (var r = p + 1u; r < 4u; r++) {
                let apr = a[p * 4u + r];
                if (abs(apr) < 1e-30) { continue; }
                let th = (a[r * 4u + r] - a[p * 4u + p]) / (2.0 * apr);
                let t = sign(th + select(0.0, 1.0, th == 0.0)) / (abs(th) + sqrt(th * th + 1.0));
                let c = 1.0 / sqrt(t * t + 1.0);
                let sn = t * c;
                for (var k = 0u; k < 4u; k++) {
                    let akp = a[k * 4u + p];
                    let akr = a[k * 4u + r];
                    a[k * 4u + p] = c * akp - sn * akr;
                    a[k * 4u + r] = sn * akp + c * akr;
                }
                for (var k = 0u; k < 4u; k++) {
                    let apk = a[p * 4u + k];
                    let ark = a[r * 4u + k];
                    a[p * 4u + k] = c * apk - sn * ark;
                    a[r * 4u + k] = sn * apk + c * ark;
                }
                for (var k = 0u; k < 4u; k++) {
                    let vkp = v[k * 4u + p];
                    let vkr = v[k * 4u + r];
                    v[k * 4u + p] = c * vkp - sn * vkr;
                    v[k * 4u + r] = sn * vkp + c * vkr;
                }
            }
        }
    }
    var out: array<f32, 16>;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            var x = 0.0;
            for (var k = 0u; k < 4u; k++) { x = x + v[i * 4u + k] * max(a[k * 4u + k], 0.0) * v[j * 4u + k]; }
            out[i * 4u + j] = x;
        }
    }
    return out;
}

@compute @workgroup_size(256)
fn ffd_cells(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let gx = max(u.gx, 2u);
    let gy = max(u.gy, 2u);
    // u.n_keys workgroups per cell, each a run of its pixel tiles
    let nch = max(u.n_keys, 1u);
    let cell = wid.x / nch;
    let chunk = wid.x % nch;
    let iu = i32(cell % gx);
    let iv = i32(cell / gx);
    let H = loadHAt_c(0u);
    if (lid == 0u) {
        matTo9(inverse3(H), 9u);
        cpsBase = 0u;
        // the coefficients' matrix, symmetrized and made positive semi-definite: the spline's
        // banded Cholesky needs a positive definite Gauss–Newton matrix
        var q: array<f32, 16>;
        for (var i = 0u; i < 4u; i++) {
            for (var j = 0u; j < 4u; j++) { q[i * 4u + j] = 0.5 * (in_e[Q0 + i * 4u + j] + in_e[Q0 + j * 4u + i]); }
        }
        let qp = psd4(q);
        for (var i = 0u; i < 4u; i++) { tg_c[8u + i] = vec4f(qp[i * 4u], qp[i * 4u + 1u], qp[i * 4u + 2u], qp[i * 4u + 3u]); }
    }
    if (lid >= 32u && lid < 47u && (lid < 40u || lid >= 44u)) { tg_c[lid - 32u] = tgCoefRow(lid - 32u, 0u); }
    workgroupBarrier();
    let Hi = matFrom9(9u);
    let dhw = hw();
    let sw = f32(u.cw); let shh = f32(u.ch);
    let rw = f32(u.w); let rh = f32(u.h);
    let src_ffd = u.theme == 1u;
    let lat_w = select(rw, sw, src_ffd);
    let lat_h = select(rh, shh, src_ffd);
    let hx = dest_half(lat_w, lat_h);
    let spx = max(lat_w - 1.0, 1.0) / f32(gx - 1u);
    let spy = max(lat_h - 1.0, 1.0) / f32(gy - 1u);
    let W0 = tg_c[0]; let W1 = tg_c[1]; let W2 = tg_c[2]; let W3 = tg_c[3];
    let M0 = tg_c[4]; let M1 = tg_c[5]; let M2 = tg_c[6]; let M3 = tg_c[7];
    let Q0r = tg_c[8]; let Q1r = tg_c[9]; let Q2r = tg_c[10]; let Q3r = tg_c[11];
    let mb = tg_c[12]; let ma = tg_c[13]; let gv = tg_c[14];
    let cw = u.cw;
    let ps = u.cw * u.ch;
    // the fixed pixels to scan: the cell itself (lattice on the fixed image), or the bounding box
    // of the moving cell's corners under H (lattice on the moving image), one pixel wider
    var x0 = 0.0; var x1 = rw - 1.0; var y0 = 0.0; var y1 = rh - 1.0;
    if (!src_ffd) {
        x0 = max(0.0, floor(f32(iu) * spx) - 1.0); x1 = min(rw - 1.0, ceil(f32(iu + 1) * spx) + 1.0);
        y0 = max(0.0, floor(f32(iv) * spy) - 1.0); y1 = min(rh - 1.0, ceil(f32(iv + 1) * spy) + 1.0);
    } else {
        var bx0 = 3.0e38; var bx1 = -3.0e38; var by0 = 3.0e38; var by1 = -3.0e38;
        for (var c = 0u; c < 4u; c++) {
            let mx = f32(iu + i32(c & 1u)) * spx + 1.0;
            let my = f32(iv + i32(c >> 1u)) * spy + 1.0;
            let q = H * vec3f(mx, my, 1.0);
            if (abs(q.z) < 1e-8) { continue; }
            bx0 = min(bx0, q.x / q.z - 1.0); bx1 = max(bx1, q.x / q.z - 1.0);
            by0 = min(by0, q.y / q.z - 1.0); by1 = max(by1, q.y / q.z - 1.0);
        }
        x0 = max(0.0, floor(bx0) - 1.0); x1 = min(rw - 1.0, ceil(bx1) + 1.0);
        y0 = max(0.0, floor(by0) - 1.0); y1 = min(rh - 1.0, ceil(by1) + 1.0);
    }
    let bw_x = u32(max(x1 - x0 + 1.0, 0.0));
    let bh_y = u32(max(y1 - y0 + 1.0, 0.0));
    let total = bw_x * bh_y;
    // the sums this thread owns (task < FC_TASKS, in one of two halves of each tile's pixels):
    // task < 100 the basis pair (task / 10, task % 10) × the 2 × 2's three entries, else tap
    // task − 100 × the gradient's x, y
    let half = lid / FC_TASKS;
    let task = lid % FC_TASKS;
    var bi = 0u; var bi2 = 0u; var bj = 0u; var bj2 = 0u;
    if (task < 100u) {
        bi = tri_i(task / 10u); bi2 = tri_j(task / 10u); bj = tri_i(task % 10u); bj2 = tri_j(task % 10u);
    } else {
        bi = (task - 100u) % 4u; bj = (task - 100u) / 4u;
    }
    var acc = vec3f(0.0);
    let ntile = (total + 255u) / 256u;
    for (var tile = chunk * ntile / nch; tile < (chunk + 1u) * ntile / nch; tile++) {
        let base = tile * 256u;
        // one pixel per thread into the tile
        var ok = false;
        var bu = vec4f(0.0); var bv = vec4f(0.0);
        var kk = vec4f(0.0); var gyv = 0.0;
        let pidx = base + lid;
        if (pidx < total) {
            let xf = x0 + f32(pidx % bw_x);
            let yf = y0 + f32(pidx / bw_x);
            let t = u32(yf) * u.w + u32(xf);
            var fx = 0.0; var fy = 0.0; var su = 0.0; var sv = 0.0;
            var ph = vec3f(0.0, 0.0, 1.0);
            ok = true;
            if (src_ffd) {
                ph = Hi * vec3f(xf + 1.0, yf + 1.0, 1.0);
                if (abs(ph.z) < 1e-8) { ok = false; } else {
                    let fx0 = ph.x / ph.z - 1.0;
                    let fy0 = ph.y / ph.z - 1.0;
                    let disp = ffd_disp(fx0, fy0, sw, shh);
                    fx = fx0 + disp.x; fy = fy0 + disp.y;
                    su = fx0 / spx; sv = fy0 / spy;
                }
            } else {
                let disp = ffd_disp(xf, yf, rw, rh);
                ph = Hi * vec3f(xf + 1.0 + disp.x, yf + 1.0 + disp.y, 1.0);
                if (abs(ph.z) < 1e-8) { ok = false; } else {
                    fx = ph.x / ph.z - 1.0; fy = ph.y / ph.z - 1.0;
                    su = xf / spx; sv = yf / spy;
                }
            }
            // this cell's pixels only (each pixel belongs to exactly one cell)
            if (ok && (i32(floor(su)) != iu || i32(floor(sv)) != iv)) { ok = false; }
            if (ok && (su < 0.0 || sv < 0.0 || su > f32(gx - 1u) || sv > f32(gy - 1u))) { ok = false; }
            if (ok && (fx < 2.0 || fy < 2.0 || fx > sw - 3.0 || fy > shh - 3.0)) { ok = false; }
            if (ok) {
                let we = pair_weight(fx, fy, t);
                let tp = taper(fx, fy);
                let wt = we * tp.x;
                // the moving planes, bilinear with their gradients (the taps are inside: fx, fy ≥ 2)
                let fx0 = floor(fx); let fy0 = floor(fy);
                let ax = fx - fx0; let ay = fy - fy0;
                let i00 = u32(fy0) * cw + u32(fx0);
                let i01 = i00 + cw;
                let v00 = vec4f(in_a[i00], in_a[ps + i00], in_a[2u * ps + i00], in_a[3u * ps + i00]);
                let v10 = vec4f(in_a[i00 + 1u], in_a[ps + i00 + 1u], in_a[2u * ps + i00 + 1u], in_a[3u * ps + i00 + 1u]);
                let v01 = vec4f(in_a[i01], in_a[ps + i01], in_a[2u * ps + i01], in_a[3u * ps + i01]);
                let v11 = vec4f(in_a[i01 + 1u], in_a[ps + i01 + 1u], in_a[2u * ps + i01 + 1u], in_a[3u * ps + i01 + 1u]);
                let va = mix(mix(v00, v10, ax), mix(v01, v11, ax), ay);
                let dgx = mix(v10 - v00, v11 - v01, ay);
                let dgy = mix(v01 - v00, v11 - v10, ax);
                let vb = vec4f(in_b[t], in_b[dhw + t], in_b[2u * dhw + t], in_b[3u * dhw + t]);
                let tb = vb - mb;
                let ta = va - ma;
                let ra = 2.0 * (vec4f(dot(W0, tb), dot(W1, tb), dot(W2, tb), dot(W3, tb)) - vec4f(dot(M0, ta), dot(M1, ta), dot(M2, ta), dot(M3, ta))) + gv;
                var bwb = 0.0;
                if (tp.y != 0.0 || tp.z != 0.0) {
                    bwb = we * boundary_value(0u, array<f32, 4>(va.x, va.y, va.z, va.w), array<f32, 4>(vb.x, vb.y, vb.z, vb.w));
                }
                // Jd: d(sample position) / d(displacement of a unit control point), columns x, y
                var j00 = 0.0; var j10 = 0.0; var j01 = 0.0; var j11 = 0.0;
                if (src_ffd) {
                    j00 = hx.x; j11 = hx.y;
                } else {
                    let z = ph.z;
                    let z2 = max(z * z, 1e-12);
                    let wx = Hi * vec3f(hx.x, 0.0, 0.0);
                    let wy = Hi * vec3f(0.0, hx.y, 0.0);
                    j00 = (wx.x * z - ph.x * wx.z) / z2; j10 = (wx.y * z - ph.y * wx.z) / z2;
                    j01 = (wy.x * z - ph.x * wy.z) / z2; j11 = (wy.y * z - ph.y * wy.z) / z2;
                }
                // the score gradient in the sample position (plus the taper's), then through Jd;
                // S = ∇vᵀ Q ∇v
                let rx = dot(ra, dgx);
                let ry = dot(ra, dgy);
                let qgy = vec4f(dot(Q0r, dgy), dot(Q1r, dgy), dot(Q2r, dgy), dot(Q3r, dgy));
                let qgx = vec4f(dot(Q0r, dgx), dot(Q1r, dgx), dot(Q2r, dgx), dot(Q3r, dgx));
                let sxx = dot(dgx, qgx);
                let sxy = dot(dgx, qgy);
                let syy = dot(dgy, qgy);
                let px = wt * rx + bwb * tp.y;
                let py = wt * ry + bwb * tp.z;
                let gxv = j00 * px + j10 * py;
                gyv = j01 * px + j11 * py;
                // K = wt Jdᵀ S Jd
                let a0 = sxx * j00 + sxy * j10; let a1 = sxy * j00 + syy * j10;
                let b0 = sxx * j01 + sxy * j11; let b1 = sxy * j01 + syy * j11;
                kk = vec4f(wt * (j00 * a0 + j10 * a1), wt * (j00 * b0 + j10 * b1), wt * (j01 * b0 + j11 * b1), gxv);
                bu = cubic_w(clamp(su - floor(su), 0.0, 1.0));
                bv = cubic_w(clamp(sv - floor(sv), 0.0, 1.0));
                if (!(abs(kk.x) + abs(kk.y) + abs(kk.z) + abs(kk.w) + abs(gyv) < 1e30)) { ok = false; }
            }
        }
        if (!ok) { bu = vec4f(0.0); bv = vec4f(0.0); kk = vec4f(0.0); gyv = 0.0; }
        fc_bu[lid] = bu; fc_bv[lid] = bv; fc_k[lid] = kk; fc_gy[lid] = gyv;
        workgroupBarrier();
        let m = min(256u, total - base);
        if (half < 2u) {
            let e1 = min(m, (half + 1u) * 128u);
            if (task < 100u) {
                for (var e = half * 128u; e < e1; e++) {
                    let pr = fc_bu[e][bi] * fc_bu[e][bi2] * fc_bv[e][bj] * fc_bv[e][bj2];
                    acc = acc + pr * fc_k[e].xyz;
                }
            } else {
                for (var e = half * 128u; e < e1; e++) {
                    acc = acc + vec3f(fc_k[e].w, fc_gy[e], 0.0) * (fc_bu[e][bi] * fc_bv[e][bj]);
                }
            }
        }
        workgroupBarrier();
    }
    // the two halves' sums added, in the FFD_CELL layout
    fc_k[lid] = vec4f(acc, 0.0);
    workgroupBarrier();
    if (lid < FC_TASKS) {
        let v = fc_k[lid].xyz + fc_k[lid + FC_TASKS].xyz;
        let o = wid.x * FFD_CELL;
        if (lid < 100u) {
            out_a[o + lid * 3u] = v.x; out_a[o + lid * 3u + 1u] = v.y; out_a[o + lid * 3u + 2u] = v.z;
        } else {
            out_a[o + 300u + (lid - 100u) * 2u] = v.x; out_a[o + 300u + (lid - 100u) * 2u + 1u] = v.y;
        }
    }
}

// ffd_cells' chunks (u.n_keys per cell, in_a) summed into each cell's FFD_CELL sums (out_a).
@compute @workgroup_size(64)
fn ffd_cells_sum(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= u.gx * u.gy * FFD_CELL) { return; }
    let nch = max(u.n_keys, 1u);
    let cell = i / FFD_CELL;
    let s = i % FFD_CELL;
    var acc = 0.0;
    for (var c = 0u; c < nch; c++) { acc = acc + in_a[(cell * nch + c) * FFD_CELL + s]; }
    out_a[i] = acc;
}
