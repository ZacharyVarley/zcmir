// SIM(2) sweep of the dense SMI score (js/sweep.js, the app's global Search).
//
// For a batch of (θ, σ) candidates at once:
//   pack_sweep     warp + box-downsample the moving stack for every candidate and
//                  write the real correlation planes, two per complex FFT input
//   (FFT)          batched line FFTs over all candidates' planes
//   cmul_sweep     A·conj(B) for every candidate and every correlation pair
//   (IFFT)
//   combine_sweep  per-shift score read straight from the IFFT output; the same
//                  formula as the app's shift map (smi.wgsl combine, copula.wgsl)
//   peaks_sweep    3×3 local maxima of each candidate's shift map, top P by score
//
// Stacks are interleaved per pixel (8 floats: w, w f0..w f3, 0, 0, 0) so each
// gather reads one cache line. Correlation-plane layout is the app's (smi.wgsl).
//
// Planes are nx × ny and hold a window of the canvas: the fixed planes its footprint
// (cells fx0…, u.win_x × u.win_y), each candidate's planes its own window (from its
// per-candidate cell offset, as wide as the plane allows without wraparound). The
// lags of a linear correlation of the two windows then fill the plane exactly.
struct U {
    nx: u32, win_x: u32, n_real: u32, n_cplx: u32,
    n_out: u32, src_w: u32, src_h: u32, min_n: u32,
    ox: f32, oy: f32, cell: f32, ridge: f32,
    exact: u32, cand0: u32, n_peak: u32, n_corr: u32,
    ny: u32, win_y: u32, fx0: u32, fy0: u32,
    grid: u32, p0: u32, p1: u32, p2: u32,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> in_a: array<f32>;
@group(0) @binding(2) var<storage, read> in_b: array<f32>;
@group(0) @binding(3) var<storage, read> in_c: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_a: array<f32>;
@group(0) @binding(5) var<storage, read> il: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> il_out: array<vec4f>;

var<workgroup> pk_v: array<f32, 256>;
var<workgroup> pk_i: array<u32, 256>;

fn tri_i(k: u32) -> u32 { var t = array<u32, 10>(0u, 0u, 0u, 0u, 1u, 1u, 1u, 2u, 2u, 3u); return t[k]; }
fn tri_j(k: u32) -> u32 { var t = array<u32, 10>(0u, 1u, 2u, 3u, 1u, 2u, 3u, 2u, 3u, 3u); return t[k]; }
fn tri_k(i: u32, j: u32) -> u32 { let a = min(i, j); let b = max(i, j); return a * 4u - (a * (a + 1u)) / 2u + b; }

// Plane-major 5-plane stack (in_a, u.src_w × u.src_h) → interleaved 2 × vec4 per pixel.

// A 1-D kernel over more than 65535 workgroups runs on an (x, y) grid (Gpu.flat); this is its
// invocation index (workgroups of 256).
fn flat_index(g: vec3u, nwg: vec3u) -> u32 {
    return g.x + g.y * nwg.x * 256u;
}

@compute @workgroup_size(256)
fn interleave(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    let hw = u.src_w * u.src_h;
    let i = gid.x;
    if (i >= hw) { return; }
    il_out[2u * i] = vec4f(in_a[i], in_a[hw + i], in_a[2u * hw + i], in_a[3u * hw + i]);
    il_out[2u * i + 1u] = vec4f(in_a[4u * hw + i], 0.0, 0.0, 0.0);
}

// Bilinear sample of the interleaved stack at 0-based (x, y); zero outside.
fn tap(x: f32, y: f32, s0: ptr<function, vec4f>, s1: ptr<function, f32>) {
    let w = i32(u.src_w);
    let h = i32(u.src_h);
    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) { return; }
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let i00 = u32(y0 * w + x0);
    let i10 = i00 + 1u;
    let i01 = i00 + u32(w);
    let i11 = i01 + 1u;
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;
    *s0 = *s0 + w00 * il[2u * i00] + w10 * il[2u * i10] + w01 * il[2u * i01] + w11 * il[2u * i11];
    *s1 = *s1 + w00 * il[2u * i00 + 1u].x + w10 * il[2u * i10 + 1u].x
        + w01 * il[2u * i01 + 1u].x + w11 * il[2u * i11 + 1u].x;
}

fn real_plane(r: u32, s: array<f32, 5>) -> f32 {
    if (r == 0u) { return s[0]; }
    if (r < 5u) { return s[r]; }
    if (s[0] <= 1e-6) { return 0.0; }
    let k = r - 5u;
    return s[1u + tri_i(k)] * s[1u + tri_j(k)] / s[0];
}

// One thread per (plane point, candidate). in_c: per-candidate pose, 12 floats:
// dest → source homography (row-major 3×3, 1-based pixels), taps per side, and the
// window's first canvas cell (x, y). The G×G canvas spans [ox, ox + G·cell) in dest
// pixels (1-based); plane point (x, y) is canvas cell (x0 + x, y0 + y) inside the window.
@compute @workgroup_size(8, 8, 1)
fn pack_sweep(@builtin(global_invocation_id) gid: vec3u) {
    let nx = u.nx;
    if (gid.x >= nx || gid.y >= u.ny) { return; }
    let b = gid.z;
    let nn = nx * u.ny;
    let t = gid.y * nx + gid.x;
    let o = (u.cand0 + b) * 12u;
    let cx = u32(in_c[o + 10u]) + gid.x;
    let cy = u32(in_c[o + 11u]) + gid.y;
    let inside = gid.x < u.win_x && gid.y < u.win_y && cx < u.grid && cy < u.grid;
    var s: array<f32, 5>;
    if (inside) {
        let k = max(u32(in_c[o + 9u]), 1u);
        var a = vec4f(0.0);
        var a4 = 0.0;
        for (var ty = 0u; ty < k; ty++) {
            let py = u.oy + (f32(cy) + (f32(ty) + 0.5) / f32(k)) * u.cell;
            for (var tx = 0u; tx < k; tx++) {
                let px = u.ox + (f32(cx) + (f32(tx) + 0.5) / f32(k)) * u.cell;
                let qx = in_c[o] * px + in_c[o + 1u] * py + in_c[o + 2u];
                let qy = in_c[o + 3u] * px + in_c[o + 4u] * py + in_c[o + 5u];
                let qz = in_c[o + 6u] * px + in_c[o + 7u] * py + in_c[o + 8u];
                if (abs(qz) < 1e-8) { continue; }
                tap(qx / qz - 1.0, qy / qz - 1.0, &a, &a4);
            }
        }
        let inv = 1.0 / f32(k * k);
        s[0] = a.x * inv; s[1] = a.y * inv; s[2] = a.z * inv; s[3] = a.w * inv; s[4] = a4 * inv;
    }
    for (var c = 0u; c < u.n_cplx; c++) {
        let r0 = 2u * c;
        var re = 0.0;
        var im = 0.0;
        if (inside) {
            re = real_plane(r0, s);
            if (r0 + 1u < u.n_real) { im = real_plane(r0 + 1u, s); }
        }
        let i = ((b * u.n_cplx + c) * nn + t) * 2u;
        out_a[i] = re;
        out_a[i + 1u] = im;
    }
}

fn unpack_a(base: u32, r: u32, i: u32, im: u32, nn: u32) -> vec2f {
    let o = (base + (r >> 1u)) * nn * 2u;
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

// gid.z = b · n_out + o. in_a: moving spectra (batch), in_b: fixed spectra, in_c: pair
// list (6 u32 per output: ra₁ rb₁ o₁ ra₂ rb₂ o₂, o₂ = ~0 unused).
@compute @workgroup_size(256)
fn cmul_sweep(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    let nx = u.nx;
    let ny = u.ny;
    let nn = nx * ny;
    if (gid.x >= nn) { return; }
    let b = gid.z / u.n_out;
    let op = gid.z % u.n_out;
    let e = op * 6u;
    let y = gid.x / nx;
    let x = gid.x - y * nx;
    let im = ((ny - y) % ny) * nx + ((nx - x) % nx);
    let base = b * u.n_cplx;
    let a1 = unpack_a(base, bitcast<u32>(in_c[e]), gid.x, im, nn);
    let b1 = unpack_b(bitcast<u32>(in_c[e + 1u]), gid.x, im, nn);
    var p = vec2f(a1.x * b1.x + a1.y * b1.y, a1.y * b1.x - a1.x * b1.y);
    if (bitcast<u32>(in_c[e + 5u]) != 0xffffffffu) {
        let a2 = unpack_a(base, bitcast<u32>(in_c[e + 3u]), gid.x, im, nn);
        let b2 = unpack_b(bitcast<u32>(in_c[e + 4u]), gid.x, im, nn);
        let x2 = vec2f(a2.x * b2.x + a2.y * b2.y, a2.y * b2.x - a2.x * b2.y);
        p = vec2f(p.x - x2.y, p.y + x2.x);
    }
    let oo = (gid.z * nn + gid.x) * 2u;
    out_a[oo] = p.x;
    out_a[oo + 1u] = p.y;
}

// Correlation plane o of candidate b at shift t, straight from the IFFT output:
// pair output o >> 1 holds planes 2p (re) and 2p + 1 (im).
fn corr_at(b: u32, o: u32, t: u32, nn: u32) -> f32 {
    return in_a[((b * u.n_out + (o >> 1u)) * nn + t) * 2u + (o & 1u)];
}

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

// The plane index of the lag that leaves the moving image where the candidate put it
// (canvas displacement 0): lag = fixed origin − window origin, per side, mod the plane.
fn zero_lag(b: u32) -> u32 {
    let o = (u.cand0 + b) * 12u;
    let lx = (i32(u.fx0) - i32(in_c[o + 10u]) + i32(u.nx)) % i32(u.nx);
    let ly = (i32(u.fy0) - i32(in_c[o + 11u]) + i32(u.ny)) % i32(u.ny);
    return u32(ly) * u.nx + u32(lx);
}

// gid.z = b. Score n·‖C‖² (global) or n·‖L_A⁻¹ C L_B⁻ᵀ‖² (exact) per shift → out_a[b·nn + t].
// in_c: the candidates' poses (their window origins).
@compute @workgroup_size(256)
fn combine_sweep(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    let nn = u.nx * u.ny;
    let t = gid.x;
    if (t >= nn) { return; }
    let b = gid.z;
    let o = b * nn + t;
    let n = corr_at(b, 0u, t, nn);
    // the overlap floor: a fraction of the overlap where the candidate placed the image
    let floor_n = max(f32(u.min_n), 0.08 * max(corr_at(b, 0u, zero_lag(b), nn), 0.0));
    if (n < floor_n) { out_a[o] = 0.0; return; }
    let den = max(n, 1e-8);
    // u.exact: 0 global, 1 exact SMI, 2 E4, 3 λmax (copula.wgsl, n × MI).
    if (u.exact >= 2u) {
        var mm: array<f32, 45>;
        let cnt = select(25u, 45u, u.exact == 3u);
        for (var k = 1u; k < cnt; k++) { mm[k] = corr_at(b, k, t, nn) / den; }
        out_a[o] = n * copula_score(&mm, u.exact, u.ridge);
        return;
    }
    var ma: array<f32, 4>;
    var mb: array<f32, 4>;
    for (var i = 0u; i < 4u; i++) {
        ma[i] = corr_at(b, 1u + i, t, nn) / den;
        mb[i] = corr_at(b, 5u + i, t, nn) / den;
    }
    if (u.exact == 0u) {
        var s = 0.0;
        for (var i = 0u; i < 4u; i++) {
            for (var j = 0u; j < 4u; j++) {
                let cij = clamp(corr_at(b, 9u + i * 4u + j, t, nn) / den - ma[i] * mb[j], -1.0, 1.0);
                s = s + cij * cij;
            }
        }
        out_a[o] = n * s;
        return;
    }
    var ga: array<f32, 16>;
    var gb: array<f32, 16>;
    var tra = 0.0;
    var trb = 0.0;
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let k = tri_k(i, j);
            ga[i * 4u + j] = corr_at(b, 25u + k, t, nn) / den - ma[i] * ma[j];
            gb[i * 4u + j] = corr_at(b, 35u + k, t, nn) / den - mb[i] * mb[j];
        }
        tra = tra + ga[i * 5u];
        trb = trb + gb[i * 5u];
    }
    for (var i = 0u; i < 4u; i++) {
        ga[i * 5u] = ga[i * 5u] + u.ridge * max(tra * 0.25, 1e-12);
        gb[i * 5u] = gb[i * 5u] + u.ridge * max(trb * 0.25, 1e-12);
    }
    if (!chol4(&ga) || !chol4(&gb)) { out_a[o] = 0.0; return; }
    var x: array<f32, 16>;
    for (var j = 0u; j < 4u; j++) {
        for (var i = 0u; i < 4u; i++) {
            var s = corr_at(b, 9u + i * 4u + j, t, nn) / den - ma[i] * mb[j];
            for (var k = 0u; k < i; k++) { s = s - ga[i * 4u + k] * x[k * 4u + j]; }
            x[i * 4u + j] = s / ga[i * 5u];
        }
    }
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
    let score = n * acc;
    out_a[o] = select(0.0, score, score == score);
}

// One workgroup per candidate (wid.x = b). Each thread keeps its best 3×3 local
// maximum (periodic shift map); a bitonic sort of the 256 thread bests gives the
// top u.n_peak, written as (score, bitcast index) at candidate u.cand0 + b.
@compute @workgroup_size(256)
fn peaks_sweep(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let nx = u.nx;
    let ny = u.ny;
    let nn = nx * ny;
    let b = wid.x;
    let base = b * nn;
    var bv = 0.0;
    var bi = 0xffffffffu;
    for (var t = lid; t < nn; t = t + 256u) {
        let v = in_a[base + t];
        if (v <= bv) { continue; }
        let y = t / nx;
        let x = t - y * nx;
        var ok = true;
        for (var dy = 0u; dy < 3u && ok; dy++) {
            for (var dx = 0u; dx < 3u; dx++) {
                if (dx == 1u && dy == 1u) { continue; }
                let yy = (y + ny + dy - 1u) % ny;
                let xx = (x + nx + dx - 1u) % nx;
                if (in_a[base + yy * nx + xx] > v) { ok = false; break; }
            }
        }
        if (ok) { bv = v; bi = t; }
    }
    pk_v[lid] = bv;
    pk_i[lid] = bi;
    workgroupBarrier();
    // Bitonic sort, descending by score.
    for (var k = 2u; k <= 256u; k = k * 2u) {
        for (var j = k / 2u; j > 0u; j = j / 2u) {
            let p = lid ^ j;
            if (p > lid) {
                let desc = (lid & k) == 0u;
                let a = pk_v[lid];
                let c = pk_v[p];
                if ((desc && a < c) || (!desc && a > c)) {
                    pk_v[lid] = c; pk_v[p] = a;
                    let ti = pk_i[lid]; pk_i[lid] = pk_i[p]; pk_i[p] = ti;
                }
            }
            workgroupBarrier();
        }
    }
    if (lid < u.n_peak) {
        let o = ((u.cand0 + b) * u.n_peak + lid) * 2u;
        out_a[o] = pk_v[lid];
        out_a[o + 1u] = bitcast<f32>(pk_i[lid]);
    }
}
