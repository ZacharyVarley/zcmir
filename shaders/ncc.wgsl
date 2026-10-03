struct U {
    src_w: u32, src_h: u32, n: u32, cw: u32,
    ch: u32, mode: u32, _p1: u32, _p2: u32,
    mean: f32, _s0: f32, _s1: f32, _s2: f32,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> in_a: array<f32>;
@group(0) @binding(2) var<storage, read> in_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_a: array<f32>;

var<workgroup> sh: array<f32, 256>;
var<workgroup> s0: array<f32, 256>;
var<workgroup> s1: array<f32, 256>;
var<workgroup> s2: array<f32, 256>;
var<workgroup> s3: array<f32, 256>;
var<workgroup> s4: array<f32, 256>;

fn at(x: i32, y: i32) -> f32 {
    let xx = clamp(x, 0, i32(u.src_w) - 1);
    let yy = clamp(y, 0, i32(u.src_h) - 1);
    return in_a[u32(yy) * u.src_w + u32(xx)];
}

fn sample(px: f32, py: f32) -> f32 {
    let x = px - 0.5; let y = py - 0.5;
    let x0 = i32(floor(x)); let y0 = i32(floor(y));
    let fx = x - f32(x0); let fy = y - f32(y0);
    return mix(mix(at(x0, y0), at(x0 + 1, y0), fx), mix(at(x0, y0 + 1), at(x0 + 1, y0 + 1), fx), fy);
}


// A 1-D kernel over more than 65535 workgroups runs on an (x, y) grid (Gpu.flat); this is its
// invocation index (workgroups of 256).
fn flat_index(g: vec3u, nwg: vec3u) -> u32 {
    return g.x + g.y * nwg.x * 256u;
}

@compute @workgroup_size(8, 8, 1)
fn pack(@builtin(global_invocation_id) gid: vec3u) {
    let n = u.n;
    if (gid.x >= n || gid.y >= n) { return; }
    let i = (gid.y * n + gid.x) * 2u;
    var v = 0.0;
    if (gid.x < u.cw && gid.y < u.ch) {
        let px = (f32(gid.x) + 0.5) * f32(u.src_w) / f32(max(u.cw, 1u));
        let py = (f32(gid.y) + 0.5) * f32(u.src_h) / f32(max(u.ch, 1u));
        v = sample(px, py) - u.mean;
    }
    out_a[i] = v; out_a[i + 1u] = 0.0;
}

@compute @workgroup_size(256)
fn reduce_spec(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    var acc = 0.0;
    if (u.mode == 0u) {
        let n = u.cw * u.ch;
        for (var t = wid.x * 256u + lid.x; t < n; t = t + 256u * 64u) {
            let x = t % u.cw; let y = t / u.cw;
            acc = acc + in_a[(y * u.n + x) * 2u];
        }
    } else {
        let n = u.n * u.n;
        for (var t = wid.x * 256u + lid.x; t < n; t = t + 256u * 64u) {
            let re = in_a[t * 2u];
            acc = acc + re * re;
        }
    }
    sh[lid.x] = acc;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid.x < s) { sh[lid.x] = sh[lid.x] + sh[lid.x + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    if (lid.x == 0u) { out_a[wid.x] = sh[0]; }
}

@compute @workgroup_size(256)
fn cmul(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    let n = u.n * u.n;
    if (gid.x >= n) { return; }
    let i = gid.x * 2u;
    let ar = out_a[i]; let ai = out_a[i + 1u];
    let br = in_b[i]; let bi = in_b[i + 1u];
    out_a[i] = ar * br + ai * bi;
    out_a[i + 1u] = ai * br - ar * bi;
}

@compute @workgroup_size(256)
fn moments(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    var sa = 0.0; var sb = 0.0; var saa = 0.0; var sbb = 0.0; var sab = 0.0;
    let n = u.src_w;
    for (var t = wid.x * 256u + lid.x; t < n; t = t + 256u * 64u) {
        let av = in_a[t]; let bv = in_b[t];
        sa = sa + av; sb = sb + bv;
        saa = saa + av * av; sbb = sbb + bv * bv; sab = sab + av * bv;
    }
    s0[lid.x] = sa; s1[lid.x] = sb; s2[lid.x] = saa; s3[lid.x] = sbb; s4[lid.x] = sab;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid.x < s) {
            s0[lid.x] = s0[lid.x] + s0[lid.x + s];
            s1[lid.x] = s1[lid.x] + s1[lid.x + s];
            s2[lid.x] = s2[lid.x] + s2[lid.x + s];
            s3[lid.x] = s3[lid.x] + s3[lid.x + s];
            s4[lid.x] = s4[lid.x] + s4[lid.x + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
    if (lid.x == 0u) {
        let o = wid.x * 5u;
        out_a[o] = s0[0]; out_a[o + 1u] = s1[0]; out_a[o + 2u] = s2[0];
        out_a[o + 3u] = s3[0]; out_a[o + 4u] = s4[0];
    }
}
