// Cubic B-spline FFD composed with a 3×3 homography.
// Binding 0/1/2/16 match warp_homography (P + src + dst + H).
// φ is destN (Lie tx/ty). Pixel disp = φ * (rw/2, rh/2).
// Lattice lives in full dest pixels (gid + (ox,oy)).

struct Params {
    w: u32, h: u32, src_w: u32, src_h: u32,
    n_sigma: u32, n_angle: u32, n_r: u32, max_points: u32,
    tau: f32, radius: f32, min_contrast: f32, scale: f32,
    nt: f32, inlier2: f32,
    n_query: u32, n_db: u32, seed: u32, n_trials: u32,
    grid: u32, kp_offset: u32,
};

struct F {
    gx: u32, gy: u32, rw: u32, rh: u32,
    ox: f32, oy: f32, _0: f32, _1: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<Texel>;
@group(0) @binding(2) var<storage, read_write> dst: array<Texel>;
@group(0) @binding(4) var<storage, read> cps: array<f32>;
@group(0) @binding(5) var<uniform> u: F;
@group(0) @binding(16) var<storage, read_write> affine: array<f32>;

fn inverse3(m: mat3x3f) -> mat3x3f {
    let a = m[0]; let b = m[1]; let c = m[2];
    let r0 = cross(b, c); let r1 = cross(c, a); let r2 = cross(a, b);
    let det = dot(a, r0);
    let id = 1.0 / det;
    return transpose(mat3x3f(r0 * id, r1 * id, r2 * id));
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

fn ffd(xf: f32, yf: f32) -> vec2f {
    let gx = max(u.gx, 2u);
    let gy = max(u.gy, 2u);
    let rw = f32(max(u.rw, 2u));
    let rh = f32(max(u.rh, 2u));
    if (xf < 0.0 || yf < 0.0 || xf > rw - 1.0 || yf > rh - 1.0) { return vec2f(0.0); }
    let hx = max(rw, 1.0) * 0.5;
    let hy = max(rh, 1.0) * 0.5;
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
            let o = (j * gx + i) * 2u;
            d = d + w * vec2f(cps[o] * hx, cps[o + 1u] * hy);
        }
    }
    return d;
}

fn load_t(v: Texel) -> f32 { return f32(v); }
fn store_t(x: f32) -> Texel { return Texel(x); }

fn at_src(x: i32, y: i32) -> f32 {
    let xx = clamp(x, 0, i32(P.src_w) - 1);
    let yy = clamp(y, 0, i32(P.src_h) - 1);
    return load_t(src[u32(yy) * P.src_w + u32(xx)]);
}

@compute @workgroup_size(8, 8)
fn warp_ffd(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let pix = gid.y * P.w + gid.x;
    let H = mat3x3f(
        vec3f(affine[0], affine[3], affine[6]),
        vec3f(affine[1], affine[4], affine[7]),
        vec3f(affine[2], affine[5], affine[8]),
    );
    let det = determinant(H);
    if (abs(det) < 1e-12) { dst[pix] = store_t(0.0); return; }
    let Hi = inverse3(H);
    var fx: f32;
    var fy: f32;
    if (u._0 > 0.5) {
        let q = Hi * vec3f(f32(gid.x) + 1.0, f32(gid.y) + 1.0, 1.0);
        if (abs(q.z) < 1e-8) { dst[pix] = store_t(0.0); return; }
        let disp = ffd(q.x / q.z - 1.0, q.y / q.z - 1.0);
        fx = q.x / q.z - 1.0 + disp.x;
        fy = q.y / q.z - 1.0 + disp.y;
    } else {
        let disp = ffd(f32(gid.x) + u.ox, f32(gid.y) + u.oy);
        let q = Hi * vec3f(f32(gid.x) + 1.0 + disp.x, f32(gid.y) + 1.0 + disp.y, 1.0);
        if (abs(q.z) < 1e-8) { dst[pix] = store_t(0.0); return; }
        fx = q.x / q.z - 1.0;
        fy = q.y / q.z - 1.0;
    }
    if (fx < 0.0 || fy < 0.0 || fx > f32(P.src_w - 1u) || fy > f32(P.src_h - 1u)) {
        dst[pix] = store_t(0.0);
        return;
    }
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let txp = fx - f32(x0);
    let typ = fy - f32(y0);
    dst[pix] = store_t(mix(mix(at_src(x0, y0), at_src(x0 + 1, y0), txp), mix(at_src(x0, y0 + 1), at_src(x0 + 1, y0 + 1), txp), typ));
}
