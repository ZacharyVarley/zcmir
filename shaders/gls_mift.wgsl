// GLS-MIFT (Fan et al. 2024) detect / describe / match / fit, plus the gray-image utilities.
// Defaults n_sigma=4, n_angle=6, n_r=3, n_tan=12, desc=216. The structure constants below are
// replaced when the shader is compiled (Engine.setStructure); everything sized by them uses them.

const PI: f32 = 3.141592653589793;
const N_SIGMA: u32 = 4u;
const N_ANGLE: u32 = 6u;
const N_R: u32 = 3u;
const N_TAN: u32 = 12u;
const DES_DIM: u32 = 216u; // N_TAN * N_R * N_ANGLE
const NN_VEC: u32 = 54u;   // DES_DIM / 4
const FAST_PAD: i32 = 3;

struct Params {
    w: u32,
    h: u32,
    src_w: u32,
    src_h: u32,
    n_sigma: u32,
    n_angle: u32,
    n_r: u32,
    max_points: u32,
    tau: f32,
    radius: f32,
    min_contrast: f32,
    scale: f32,
    nt: f32,
    inlier2: f32,
    n_query: u32,
    n_db: u32,
    seed: u32,
    n_trials: u32,
    grid: u32,
    kp_offset: u32,
    // Appended so the prefix layout stays what warp_ffd already reads.
    q_offset: u32,
    db_offset: u32,
    flags: u32, // bit 0: emit the second orientation
    off_src: u32, // element offset (paste: signed dx) for the copy kernels
    scale_lo: f32,
    scale_hi: f32,
    off_dst: u32, // element offset (paste: signed dy)
    nn_chunk: u32, // match_nn: database entries per workgroup row (workgroup_id.y)
    border: i32,   // fast_nms: no corners closer than this to the edge (at least FAST_PAD)
    ratio: f32,    // match_nnq_pick_ratio: largest SSD ratio best / second best
    max_ssd: f32,  // match_nnq_pick_ratio: largest SSD of a match
    nn_sub: u32,   // match_nnq: entries per workgroup row within a chunk (0: the whole chunk)
}

struct Kp {
    x: f32,
    y: f32,
    score: f32,
    pad: f32,
}

struct Trial {
    n_inl: u32,
    pad: u32,
    a: f32,
    b: f32,
    tx: f32,
    c: f32,
    d: f32,
    ty: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src_f: array<Texel>;
@group(0) @binding(2) var<storage, read_write> dst_f: array<Texel>;
@group(0) @binding(3) var<storage, read_write> fmap: array<Texel>;
@group(0) @binding(4) var<storage, read_write> sr: array<u32>;
@group(0) @binding(5) var<storage, read_write> scores: array<Texel>;
@group(0) @binding(6) var<storage, read_write> cell_best: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> kps: array<Kp>;
@group(0) @binding(8) var<storage, read_write> des: array<f32>;
@group(0) @binding(9) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> extrema: array<atomic<u32>>;
@group(0) @binding(11) var src_tex: texture_2d<f32>;
@group(0) @binding(12) var<storage, read_write> kps_b: array<Kp>;
@group(0) @binding(13) var<storage, read_write> des_b: array<f32>;
@group(0) @binding(14) var<storage, read_write> match_j: array<u32>;
@group(0) @binding(15) var<storage, read_write> trials: array<Trial>;
@group(0) @binding(16) var<storage, read_write> affine: array<f32>; // 3×3 row-major, affine uses last row 0 0 1
@group(0) @binding(17) var<storage, read_write> cell_kp: array<vec4f>;       // per cell: x, y, score, 0
@group(0) @binding(18) var<storage, read_write> ori2: array<u32>;            // per kp: second orientation
@group(0) @binding(19) var<storage, read_write> cell_pix: array<atomic<u32>>; // per cell: ~(tie-winning pixel)
@group(0) @binding(23) var<storage, read_write> match_rev: array<u32>;
@group(0) @binding(34) var<storage, read_write> kdir: array<u32>;    // per kp: the tangential sector its descriptor starts at
@group(0) @binding(20) var<storage, read_write> desq: array<u32>;   // per keypoint 54 words, 4 u8 each: count · 255 / largest count
@group(0) @binding(21) var<storage, read_write> desq_b: array<u32>;
@group(0) @binding(22) var<storage, read_write> dsc: array<f32>;    // per keypoint: largest count · norm / 255, so desq · dsc ≈ des
@group(0) @binding(31) var<storage, read_write> dsc_b: array<f32>;
// per (query, row): match_nn: best value bits, entry; match_nnq: the two best entries and their
// value bits (entries 0xffffffff: none)
@group(0) @binding(29) var<storage, read_write> nn_part: array<vec4u>;

fn idx2(x: i32, y: i32, w: u32, h: u32) -> u32 {
    let xx = clamp(x, 0, i32(w) - 1);
    let yy = clamp(y, 0, i32(h) - 1);
    return u32(yy) * w + u32(xx);
}

fn load_t(v: Texel) -> f32 { return f32(v); }
fn store_t(x: f32) -> Texel { return Texel(x); }

fn at(x: i32, y: i32) -> f32 {
    return load_t(src_f[idx2(x, y, P.w, P.h)]);
}

fn at_src(x: i32, y: i32) -> f32 {
    return load_t(src_f[idx2(x, y, P.src_w, P.src_h)]);
}

fn mround(x: f32) -> i32 {
    return i32(sign(x) * floor(abs(x) + 0.5));
}

// First-order Gaussian (FOG) derivative responses, separably.
// The 2-D kernel (−o_x / 2πσ⁴) e^{−(o_x²+o_y²)/2σ²} over a 27×27 window factors
// into x and y passes, and clamp-to-edge indexing is per axis, so this equals the
// direct 729-tap sum. One 8×8 workgroup loads a 34×34 tile, runs the horizontal
// pass in shared memory, then each invocation runs its vertical pass.
const FOG_R: i32 = 13;
const FOG_K: u32 = 27u;  // 2 * FOG_R + 1
const FOG_T: u32 = 34u;  // 8 + 2 * FOG_R
var<workgroup> fog_tile: array<f32, 1156>;   // FOG_T²
var<workgroup> fog_h: array<f32, FOG_T * 8u * N_SIGMA * 2u>;      // FOG_T rows × 8 cols × N_SIGMA × (d/dx, smooth)
var<workgroup> fog_w: array<f32, N_SIGMA * FOG_K>;      // N_SIGMA × FOG_K Gaussian weights

struct Fog { rx: array<f32, N_SIGMA>, ry: array<f32, N_SIGMA> }

// Every invocation of the workgroup must call this (it has barriers).
fn fog_responses(wid: vec3u, lid: vec3u) -> Fog {
    let tid = lid.y * 8u + lid.x;
    let ox = i32(wid.x * 8u) - FOG_R;
    let oy = i32(wid.y * 8u) - FOG_R;
    for (var k = tid; k < N_SIGMA * FOG_K; k += 64u) {
        let o = f32(i32(k % FOG_K) - FOG_R);
        let sig = P.tau * f32(k / FOG_K + 1u);
        fog_w[k] = exp(-(o * o) / (2.0 * sig * sig));
    }
    for (var e = tid; e < FOG_T * FOG_T; e += 64u) {
        fog_tile[e] = at(ox + i32(e % FOG_T), oy + i32(e / FOG_T));
    }
    workgroupBarrier();
    for (var e = tid; e < FOG_T * 8u; e += 64u) {
        let r = e / 8u;
        let c = e % 8u;
        for (var sg = 0u; sg < N_SIGMA; sg++) {
            var d = 0.0;
            var g = 0.0;
            for (var k = 0u; k < FOG_K; k++) {
                let v = fog_tile[r * FOG_T + c + k];
                let w = fog_w[sg * FOG_K + k];
                d += f32(FOG_R - i32(k)) * w * v;
                g += w * v;
            }
            let h = ((r * 8u + c) * N_SIGMA + sg) * 2u;
            fog_h[h] = d;
            fog_h[h + 1u] = g;
        }
    }
    workgroupBarrier();
    var out: Fog;
    for (var sg = 0u; sg < N_SIGMA; sg++) {
        let sig = P.tau * f32(sg + 1u);
        let den = 2.0 * PI * sig * sig * sig * sig;
        var rx = 0.0;
        var ry = 0.0;
        for (var k = 0u; k < FOG_K; k++) {
            let w = fog_w[sg * FOG_K + k];
            let h = (((lid.y + k) * 8u + lid.x) * N_SIGMA + sg) * 2u;
            rx += w * fog_h[h];
            ry += f32(FOG_R - i32(k)) * w * fog_h[h + 1u];
        }
        out.rx[sg] = rx / den;
        out.ry[sg] = ry / den;
    }
    return out;
}

fn longest_run16(bits: u32) -> u32 {
    var x = bits | (bits << 16u);
    var run = 0u;
    var best = 0u;
    for (var i = 0u; i < 32u; i++) {
        if ((x & 1u) == 1u) {
            run++;
            best = max(best, run);
        } else {
            run = 0u;
        }
        x = x >> 1u;
    }
    return min(best, 16u);
}

fn inverse3(m: mat3x3f) -> mat3x3f {
    let a = m[0];
    let b = m[1];
    let c = m[2];
    let r0 = cross(b, c);
    let r1 = cross(c, a);
    let r2 = cross(a, b);
    let det = dot(a, r0);
    let id = 1.0 / det;
    // columns of inverse = rows of adjugate / det
    return transpose(mat3x3f(r0 * id, r1 * id, r2 * id));
}

fn apply_aff(a: f32, b: f32, tx: f32, c: f32, d: f32, ty: f32, p: vec2f) -> vec2f {
    return vec2f(a * p.x + b * p.y + tx, c * p.x + d * p.y + ty);
}

fn scale_ok(lin: f32) -> bool {
    if (!(lin > 0.0)) { return false; }
    let sc = sqrt(lin);
    return sc >= P.scale_lo && sc <= P.scale_hi;
}

fn pcg(n: u32) -> u32 {
    var x = n * 747796405u + 2891336453u;
    var w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (w >> 22u) ^ w;
}


// A 1-D kernel over more than 65535 workgroups runs on an (x, y) grid (Gpu.flat); this is its
// invocation index (workgroups of 256).
fn flat_index(g: vec3u, nwg: vec3u) -> u32 {
    return g.x + g.y * nwg.x * 256u;
}

@compute @workgroup_size(8, 8)
fn rgba_to_gray(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let p = textureLoad(src_tex, vec2i(i32(gid.x), i32(gid.y)), 0);
    dst_f[gid.y * P.w + gid.x] = store_t(0.299 * p.r + 0.587 * p.g + 0.114 * p.b);
}

// v / 255 correctly rounded to f32, as textureLoad of an rgba8unorm texture returns it.
// unpack4x8unorm may compute v · (1/255) instead, one ulp off for many v, and CLAHE bins on these
// images are that sensitive.
var<private> UNORM8: array<f32, 256> = array<f32, 256>(
    0.0f, 0.003921569f, 0.007843138f, 0.011764706f, 0.015686275f, 0.019607844f, 0.023529412f, 0.02745098f,
    0.03137255f, 0.03529412f, 0.039215688f, 0.043137256f, 0.047058824f, 0.050980393f, 0.05490196f, 0.05882353f,
    0.0627451f, 0.06666667f, 0.07058824f, 0.07450981f, 0.078431375f, 0.08235294f, 0.08627451f, 0.09019608f,
    0.09411765f, 0.09803922f, 0.101960786f, 0.105882354f, 0.10980392f, 0.11372549f, 0.11764706f, 0.12156863f,
    0.1254902f, 0.12941177f, 0.13333334f, 0.13725491f, 0.14117648f, 0.14509805f, 0.14901961f, 0.15294118f,
    0.15686275f, 0.16078432f, 0.16470589f, 0.16862746f, 0.17254902f, 0.1764706f, 0.18039216f, 0.18431373f,
    0.1882353f, 0.19215687f, 0.19607843f, 0.2f, 0.20392157f, 0.20784314f, 0.21176471f, 0.21568628f,
    0.21960784f, 0.22352941f, 0.22745098f, 0.23137255f, 0.23529412f, 0.23921569f, 0.24313726f, 0.24705882f,
    0.2509804f, 0.25490198f, 0.25882354f, 0.2627451f, 0.26666668f, 0.27058825f, 0.27450982f, 0.2784314f,
    0.28235295f, 0.28627452f, 0.2901961f, 0.29411766f, 0.29803923f, 0.3019608f, 0.30588236f, 0.30980393f,
    0.3137255f, 0.31764707f, 0.32156864f, 0.3254902f, 0.32941177f, 0.33333334f, 0.3372549f, 0.34117648f,
    0.34509805f, 0.34901962f, 0.3529412f, 0.35686275f, 0.36078432f, 0.3647059f, 0.36862746f, 0.37254903f,
    0.3764706f, 0.38039216f, 0.38431373f, 0.3882353f, 0.39215687f, 0.39607844f, 0.4f, 0.40392157f,
    0.40784314f, 0.4117647f, 0.41568628f, 0.41960785f, 0.42352942f, 0.42745098f, 0.43137255f, 0.43529412f,
    0.4392157f, 0.44313726f, 0.44705883f, 0.4509804f, 0.45490196f, 0.45882353f, 0.4627451f, 0.46666667f,
    0.47058824f, 0.4745098f, 0.47843137f, 0.48235294f, 0.4862745f, 0.49019608f, 0.49411765f, 0.49803922f,
    0.5019608f, 0.5058824f, 0.50980395f, 0.5137255f, 0.5176471f, 0.52156866f, 0.5254902f, 0.5294118f,
    0.53333336f, 0.5372549f, 0.5411765f, 0.54509807f, 0.54901963f, 0.5529412f, 0.5568628f, 0.56078434f,
    0.5647059f, 0.5686275f, 0.57254905f, 0.5764706f, 0.5803922f, 0.58431375f, 0.5882353f, 0.5921569f,
    0.59607846f, 0.6f, 0.6039216f, 0.60784316f, 0.6117647f, 0.6156863f, 0.61960787f, 0.62352943f,
    0.627451f, 0.6313726f, 0.63529414f, 0.6392157f, 0.6431373f, 0.64705884f, 0.6509804f, 0.654902f,
    0.65882355f, 0.6627451f, 0.6666667f, 0.67058825f, 0.6745098f, 0.6784314f, 0.68235296f, 0.6862745f,
    0.6901961f, 0.69411767f, 0.69803923f, 0.7019608f, 0.7058824f, 0.70980394f, 0.7137255f, 0.7176471f,
    0.72156864f, 0.7254902f, 0.7294118f, 0.73333335f, 0.7372549f, 0.7411765f, 0.74509805f, 0.7490196f,
    0.7529412f, 0.75686276f, 0.7607843f, 0.7647059f, 0.76862746f, 0.77254903f, 0.7764706f, 0.78039217f,
    0.78431374f, 0.7882353f, 0.7921569f, 0.79607844f, 0.8f, 0.8039216f, 0.80784315f, 0.8117647f,
    0.8156863f, 0.81960785f, 0.8235294f, 0.827451f, 0.83137256f, 0.8352941f, 0.8392157f, 0.84313726f,
    0.84705883f, 0.8509804f, 0.85490197f, 0.85882354f, 0.8627451f, 0.8666667f, 0.87058824f, 0.8745098f,
    0.8784314f, 0.88235295f, 0.8862745f, 0.8901961f, 0.89411765f, 0.8980392f, 0.9019608f, 0.90588236f,
    0.9098039f, 0.9137255f, 0.91764706f, 0.92156863f, 0.9254902f, 0.92941177f, 0.93333334f, 0.9372549f,
    0.9411765f, 0.94509804f, 0.9490196f, 0.9529412f, 0.95686275f, 0.9607843f, 0.9647059f, 0.96862745f,
    0.972549f, 0.9764706f, 0.98039216f, 0.9843137f, 0.9882353f, 0.99215686f, 0.99607843f, 1.0f
);

// rgba_to_gray from packed RGBA8 words in a storage buffer (sr, binding 4) instead of a texture,
// with the texture's channel values.
@compute @workgroup_size(8, 8)
fn rgba_to_gray_buf(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let w = sr[gid.y * P.w + gid.x];
    let r = UNORM8[w & 0xffu];
    let g = UNORM8[(w >> 8u) & 0xffu];
    let b = UNORM8[(w >> 16u) & 0xffu];
    dst_f[gid.y * P.w + gid.x] = store_t(0.299 * r + 0.587 * g + 0.114 * b);
}

@compute @workgroup_size(8, 8)
fn gauss_h(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    // sigma=1, radius=3, separable
    var acc = 0.0;
    var wsum = 0.0;
    for (var k = -3; k <= 3; k++) {
        let ww = exp(-0.5 * f32(k * k));
        acc += at(i32(gid.x) + k, i32(gid.y)) * ww;
        wsum += ww;
    }
    dst_f[gid.y * P.w + gid.x] = store_t(acc / wsum);
}

@compute @workgroup_size(8, 8)
fn gauss_v(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    var acc = 0.0;
    var wsum = 0.0;
    for (var k = -3; k <= 3; k++) {
        let ww = exp(-0.5 * f32(k * k));
        acc += at(i32(gid.x), i32(gid.y) + k) * ww;
        wsum += ww;
    }
    dst_f[gid.y * P.w + gid.x] = store_t(acc / wsum);
}

@compute @workgroup_size(8, 8)
fn resize_bilinear(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let fx = (f32(gid.x) + 0.5) * f32(P.src_w) / f32(P.w) - 0.5;
    let fy = (f32(gid.y) + 0.5) * f32(P.src_h) / f32(P.h) - 0.5;
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let a = at_src(x0, y0);
    let b = at_src(x0 + 1, y0);
    let c = at_src(x0, y0 + 1);
    let d = at_src(x0 + 1, y0 + 1);
    dst_f[gid.y * P.w + gid.x] = store_t(mix(mix(a, b, tx), mix(c, d, tx), ty));
}

@compute @workgroup_size(8, 8)
fn make_fmap(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wg: vec3u,
             @builtin(local_invocation_id) lid: vec3u) {
    let fog = fog_responses(wg, lid);
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let rx = fog.rx;
    let ry = fog.ry;
    var fmap_acc = 0.0;
    for (var th = 0u; th < N_ANGLE; th++) {
        let ang = f32(th) * (PI / f32(N_ANGLE));
        let c = cos(ang);
        let s = sin(ang);
        var s_th = 0.0;
        var peak = 1e-12;
        var fbar_num = 0.0;
        var steered: array<f32, N_SIGMA>;
        for (var si = 0u; si < N_SIGMA; si++) {
            let r = c * rx[si] + s * ry[si];
            steered[si] = r;
            let mag = abs(r);
            s_th += mag;
            peak = max(peak, mag);
            fbar_num += r;
        }
        let fbar = fbar_num / (s_th + 1e-6);
        let wid = (s_th / peak - 1.0) / f32(N_SIGMA - 1u);
        let wgt = 1.0 / (1.0 + exp(0.5 - 10.0 * wid));
        var clamp_acc = 0.0;
        for (var si = 0u; si < N_SIGMA; si++) {
            clamp_acc += steered[si] * fbar;
        }
        fmap_acc += wgt * max(clamp_acc - P.nt, 0.0) / (s_th + 1e-6);
    }
    let pix = gid.y * P.w + gid.x;
    fmap[pix] = store_t(fmap_acc);
    // SR: argmax |R| over theta, per sigma, packed 4 bits per sigma (describe reads one word).
    var srw = 0u;
    for (var si = 0u; si < N_SIGMA; si++) {
        var best = -1.0;
        var bi = 0u;
        for (var th = 0u; th < N_ANGLE; th++) {
            let ang = f32(th) * (PI / f32(N_ANGLE));
            let r = abs(cos(ang) * rx[si] + sin(ang) * ry[si]);
            if (r > best) {
                best = r;
                bi = th;
            }
        }
        srw = srw | (bi << (4u * si));
    }
    sr[pix] = srw;
    atomicMax(&extrema[1], bitcast<u32>(fmap_acc));
    atomicMin(&extrema[0], bitcast<u32>(fmap_acc));
}

@compute @workgroup_size(8, 8)
fn normalize_fmap(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let mn = bitcast<f32>(atomicLoad(&extrema[0]));
    let mx = bitcast<f32>(atomicLoad(&extrema[1]));
    let pix = gid.y * P.w + gid.x;
    fmap[pix] = store_t((load_t(fmap[pix]) - mn) / max(mx - mn, 1e-12));
}

@compute @workgroup_size(8, 8)
fn fast_nms(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let pix = gid.y * P.w + gid.x;
    scores[pix] = store_t(0.0);
    let pad = max(FAST_PAD, P.border);
    if (x < pad || y < pad || x >= i32(P.w) - pad || y >= i32(P.h) - pad) {
        return;
    }
    let c = load_t(fmap[pix]);
    var offs = array<vec2i, 16>(
        vec2i(-3, 0), vec2i(-3, 1), vec2i(-2, 2), vec2i(-1, 3),
        vec2i(0, 3), vec2i(1, 3), vec2i(2, 2), vec2i(3, 1),
        vec2i(3, 0), vec2i(3, -1), vec2i(2, -2), vec2i(1, -3),
        vec2i(0, -3), vec2i(-1, -3), vec2i(-2, -2), vec2i(-3, -1),
    );
    var posb = 0u;
    var negb = 0u;
    var md = 0.0;
    for (var i = 0u; i < 16u; i++) {
        let v = load_t(fmap[idx2(x + offs[i].x, y + offs[i].y, P.w, P.h)]);
        md += abs(v - c);
        if (v >= c + P.min_contrast) { posb |= (1u << (15u - i)); }
        if (v <= c - P.min_contrast) { negb |= (1u << (15u - i)); }
    }
    md = md / 16.0;
    let run = max(longest_run16(posb), longest_run16(negb));
    if (run < 9u) { return; }
    // Corner score is the mean |ring - centre|; nms_and_cells does the 3×3 NMS on it.
    scores[pix] = store_t(md);
}

// ── Deterministic keypoint selection ─────────────────────────────────────
// Every step is order-independent, so an image always yields the same
// keypoints in the same slots: scores are never rewritten during NMS, each grid
// cell keeps its strongest local max (ties go to the lowest pixel index), and one
// workgroup compacts the cells in cell order, keeping the max_points strongest.

fn cell_of(gid: vec3u) -> u32 {
    let gx = min(u32(f32(gid.x) / f32(P.w) * f32(P.grid)), P.grid - 1u);
    let gy = min(u32(f32(gid.y) / f32(P.h) * f32(P.grid)), P.grid - 1u);
    return gy * P.grid + gx;
}

// Corner score at gid if it is a 3×3 local max (ties kept), else 0.
fn local_max(gid: vec3u) -> f32 {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let md = load_t(scores[gid.y * P.w + gid.x]);
    if (md <= 0.0) { return 0.0; }
    for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
            if (load_t(scores[idx2(x + ox, y + oy, P.w, P.h)]) > md) { return 0.0; }
        }
    }
    return md;
}

@compute @workgroup_size(8, 8)
fn nms_and_cells(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let md = local_max(gid);
    if (md <= 0.0) { return; }
    atomicMax(&cell_best[cell_of(gid)], bitcast<u32>(md));
}

// Among the pixels of a cell at its best score, the lowest index wins (max of ~pix).
@compute @workgroup_size(8, 8)
fn cell_tie(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let md = local_max(gid);
    if (md <= 0.0) { return; }
    let cell = cell_of(gid);
    if (atomicLoad(&cell_best[cell]) != bitcast<u32>(md)) { return; }
    atomicMax(&cell_pix[cell], 0xffffffffu - (gid.y * P.w + gid.x));
}

@compute @workgroup_size(8, 8)
fn emit_kps(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let md = local_max(gid);
    if (md <= 0.0) { return; }
    let cell = cell_of(gid);
    if (atomicLoad(&cell_best[cell]) != bitcast<u32>(md)) { return; }
    if (atomicLoad(&cell_pix[cell]) != 0xffffffffu - (gid.y * P.w + gid.x)) { return; }
    cell_kp[cell] = vec4f(f32(gid.x) + 1.0, f32(gid.y) + 1.0, md, 0.0);
}

var<workgroup> scan_a: array<u32, 256>;
var<workgroup> scan_b: array<u32, 256>;
var<workgroup> scan_u: u32;

// Workgroup sum of v, returned uniformly.
fn wg_sum(lid: u32, v: u32) -> u32 {
    workgroupBarrier();
    scan_a[lid] = v;
    workgroupBarrier();
    if (lid == 0u) {
        var t = 0u;
        for (var i = 0u; i < 256u; i++) { t += scan_a[i]; }
        scan_u = t;
    }
    return workgroupUniformLoad(&scan_u);
}

// Exclusive prefix sums of (a, b) over the workgroup, left in scan_a / scan_b.
fn wg_prefix2(lid: u32, a: u32, b: u32) {
    workgroupBarrier();
    scan_a[lid] = a;
    scan_b[lid] = b;
    workgroupBarrier();
    if (lid == 0u) {
        var ta = 0u;
        var tb = 0u;
        for (var i = 0u; i < 256u; i++) {
            let va = scan_a[i];
            let vb = scan_b[i];
            scan_a[i] = ta;
            scan_b[i] = tb;
            ta += va;
            tb += vb;
        }
    }
    workgroupBarrier();
}

// Where this level's keypoints start: P.kp_offset, or with P.kp_offset = 0xffffffff the running
// offset in counters[6] that level_done advances, so a pyramid is detected without reading each
// level's count back.
const KP_RUNNING: u32 = 0xffffffffu;
const LEVEL_REC: u32 = 8u; // counters[8 + 4·level]: offset, keypoints, base, 0
var<workgroup> kp_base: u32;

fn kp_off() -> u32 {
    if (P.kp_offset == KP_RUNNING) { return atomicLoad(&counters[6]); }
    return P.kp_offset;
}

// Close a detection level (P.db_offset: its index): the base count min(counters[0], max_points,
// room) and, with P.n_db = 1, the extra orientations within the room (counters[2]), recorded
// at LEVEL_REC, and the next level's offset.
@compute @workgroup_size(1)
fn level_done() {
    let off = atomicLoad(&counters[6]);
    let room = select(0u, MAX_KP - off, off < MAX_KP);
    let n = min(min(atomicLoad(&counters[0]), P.max_points), room);
    var ex = 0u;
    if (P.n_db == 1u && n > 0u) { ex = min(atomicLoad(&counters[2]), room - n); }
    let r = LEVEL_REC + 4u * P.db_offset;
    atomicStore(&counters[r], off);
    atomicStore(&counters[r + 1u], n + ex);
    atomicStore(&counters[r + 2u], n);
    atomicStore(&counters[6], off + n + ex);
}

// One workgroup. Keeps the cap strongest cells (score bits > T, then == T in cell
// order), writes them to kps[kp_offset ..] in cell order; count goes to counters[0].
@compute @workgroup_size(256)
fn compact_kps(@builtin(local_invocation_index) lid: u32) {
    let nCells = P.grid * P.grid;
    let chunk = (nCells + 255u) / 256u;
    let c0 = min(lid * chunk, nCells);
    let c1 = min(c0 + chunk, nCells);
    // the offset gates barriers below: one uniform read (kp_off may be a storage load)
    if (lid == 0u) { kp_base = kp_off(); }
    let base = workgroupUniformLoad(&kp_base);
    var cap = 0u;
    if (base < MAX_KP) { cap = min(P.max_points, MAX_KP - base); }
    var local = 0u;
    for (var c = c0; c < c1; c++) { if (cell_kp[c].z > 0.0) { local++; } }
    let total = wg_sum(lid, local);
    // T: the largest score bit pattern with count(bits >= T) >= cap. 0 keeps every cell.
    var T = 0u;
    if (total > cap && cap > 0u) {
        var lo = 1u;
        var hi = 0x7f800000u;
        for (var it = 0u; it < 32u; it++) {
            let mid = lo + (hi - lo + 1u) / 2u;
            var cnt = 0u;
            for (var c = c0; c < c1; c++) {
                let z = cell_kp[c].z;
                if (z > 0.0 && bitcast<u32>(z) >= mid) { cnt++; }
            }
            if (wg_sum(lid, cnt) >= cap) { lo = mid; } else { hi = mid - 1u; }
        }
        T = lo;
    }
    var gt = 0u;
    var eq = 0u;
    for (var c = c0; c < c1; c++) {
        let z = cell_kp[c].z;
        if (z <= 0.0) { continue; }
        let b = bitcast<u32>(z);
        if (b > T) { gt++; } else if (b == T) { eq++; }
    }
    let gtTotal = wg_sum(lid, gt);
    wg_prefix2(lid, gt, eq);
    let gtBefore = scan_a[lid];
    let eqBefore = scan_b[lid];
    let budget = select(0u, cap - gtTotal, cap > gtTotal);
    let allow = min(eq, select(0u, budget - eqBefore, budget > eqBefore));
    var rank = gtBefore + min(eqBefore, budget);
    var eqSeen = 0u;
    for (var c = c0; c < c1; c++) {
        let k = cell_kp[c];
        if (k.z <= 0.0) { continue; }
        let b = bitcast<u32>(k.z);
        var keep = b > T;
        if (b == T) {
            keep = eqSeen < allow;
            eqSeen++;
        }
        if (!keep) { continue; }
        kps[base + rank] = Kp(k.x, k.y, k.z, P.scale);
        rank++;
    }
    if (lid == 255u) { atomicStore(&counters[0], rank); }
}

// Descriptors, one workgroup per keypoint (keypoint wid.y · nw.x + wid.x): the invocations
// split the disk's pixels and add into integer histograms in shared memory, so every count,
// and hence the descriptor, is the one a serial pass over the disk produces.
var<workgroup> ds_prim: array<atomic<u32>, N_ANGLE>;
var<workgroup> ds_q: array<atomic<u32>, 4>;
var<workgroup> ds_h: array<atomic<u32>, DES_DIM>;
var<workgroup> ds_u: array<u32, 3>;
var<workgroup> ds_f: f32;

fn ds_zero(lid: u32) {
    for (var k = lid; k < DES_DIM; k += 64u) { atomicStore(&ds_h[k], 0u); }
    if (lid < N_ANGLE) { atomicStore(&ds_prim[lid], 0u); }
    if (lid < 4u) { atomicStore(&ds_q[lid], 0u); }
}

@compute @workgroup_size(64)
fn describe(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u,
            @builtin(local_invocation_index) lid: u32) {
    let i = wid.y * nw.x + wid.x;
    if (lid == 0u) { scan_u = atomicLoad(&counters[0]); }
    let n = workgroupUniformLoad(&scan_u);
    if (i >= n) { return; }
    let slot = kp_off() + i;
    let kp = kps[slot];
    let expand = P.radius / f32(N_R * N_R);
    let rmax = P.radius + expand;
    let lim = i32(ceil(rmax));
    ds_zero(lid);
    workgroupBarrier();
    // Primary direction: votes of every (pixel, sigma) in the disk.
    var pl: array<u32, N_ANGLE>;
    let side = u32(2 * lim + 1);
    for (var e = lid; e < side * side; e += 64u) {
        let ox = i32(e % side) - lim;
        let oy = i32(e / side) - lim;
        if (f32(ox * ox + oy * oy) > rmax * rmax) { continue; }
        let xs = mround(kp.x + f32(ox));
        let ys = mround(kp.y + f32(oy));
        let pxi = idx2(xs - 1, ys - 1, P.w, P.h);
        let srw = sr[pxi];
        for (var si = 0u; si < N_SIGMA; si++) { pl[(srw >> (4u * si)) & 15u] += 1u; }
    }
    for (var t = 0u; t < N_ANGLE; t++) {
        if (pl[t] > 0u) { atomicAdd(&ds_prim[t], pl[t]); }
    }
    workgroupBarrier();
    if (lid == 0u) {
        var prim: array<f32, N_ANGLE>;
        for (var t = 0u; t < N_ANGLE; t++) { prim[t] = f32(atomicLoad(&ds_prim[t])); }
        var top0 = 0u;
        var top1 = 1u;
        if (prim[1] > prim[0]) { top0 = 1u; top1 = 0u; }
        for (var t = 2u; t < N_ANGLE; t++) {
            if (prim[t] > prim[top0]) {
                top1 = top0;
                top0 = t;
            } else if (prim[t] > prim[top1]) {
                top1 = t;
            }
        }
        // Second primary direction when its peak is within 0.8 of the first (Fan et al.).
        // Bit 0 of flags turns this on; compact_extra assigns the slots in keypoint order.
        var flag = 0xffffffffu;
        if ((P.flags & 1u) != 0u && prim[top1] > prim[top0] * 0.8) { flag = top1; }
        ori2[slot] = flag;
        ds_u[0] = top0;
    }
    let pd = workgroupUniformLoad(&ds_u[0]);
    pack_desc(slot, kp, pd, expand, rmax, lim, lid);
}

// One workgroup: the k-th extra descriptor (in keypoint order) goes to slot
// kp_offset + n + k. ori2[slot] becomes (k << 3) | orientation; count to counters[2].
@compute @workgroup_size(256)
fn compact_extra(@builtin(local_invocation_index) lid: u32) {
    let n = atomicLoad(&counters[0]);
    let chunk = (n + 255u) / 256u;
    let i0 = min(lid * chunk, n);
    let i1 = min(i0 + chunk, n);
    var local = 0u;
    for (var i = i0; i < i1; i++) { if (ori2[kp_off() + i] != 0xffffffffu) { local++; } }
    wg_prefix2(lid, local, 0u);
    var k = scan_a[lid];
    for (var i = i0; i < i1; i++) {
        let s = kp_off() + i;
        let v = ori2[s];
        if (v == 0xffffffffu) { continue; }
        ori2[s] = (k << 3u) | v;
        k++;
    }
    if (lid == 255u) { atomicStore(&counters[2], k); }
}

@compute @workgroup_size(64)
fn describe_extra(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u,
                  @builtin(local_invocation_index) lid: u32) {
    let i = wid.y * nw.x + wid.x;
    if (lid == 0u) {
        let n = atomicLoad(&counters[0]);
        var v = 0xffffffffu;
        var s2 = 0xffffffffu;
        if (i < n) {
            v = ori2[kp_off() + i];
            if (v != 0xffffffffu) { s2 = kp_off() + n + (v >> 3u); }
        }
        ds_u[0] = v;
        ds_u[1] = s2;
    }
    let v = workgroupUniformLoad(&ds_u[0]);
    let slot2 = workgroupUniformLoad(&ds_u[1]);
    if (v == 0xffffffffu || slot2 >= MAX_KP) { return; }
    // describe already scaled kps[slot]; scales are 2^o or 1.5·2^o, so this is exact.
    let ks = kps[kp_off() + i];
    let kp = Kp(ks.x / P.scale, ks.y / P.scale, ks.score, ks.pad);
    if (lid == 0u) { kps[slot2] = kp; }
    let expand = P.radius / f32(N_R * N_R);
    let rmax = P.radius + expand;
    ds_zero(lid);
    workgroupBarrier();
    pack_desc(slot2, kp, v & 7u, expand, rmax, i32(ceil(rmax)), lid);
}

// Histogram bin of descriptor element k (output order: tangential bin from the start, ring,
// orientation from the primary one).
fn ds_src(k: u32, start_tan: u32, pd: u32) -> u32 {
    let t = k / (N_R * N_ANGLE);
    let ri = (k / N_ANGLE) % N_R;
    let a = k % N_ANGLE;
    return (((t + start_tan) % N_TAN) * N_R + ri) * N_ANGLE + (a + pd) % N_ANGLE;
}

// Every invocation of the workgroup calls this (barriers); ds_h / ds_q must be zero.
fn pack_desc(slot: u32, kp: Kp, pd: u32, expand: f32, rmax: f32, lim: i32, lid: u32) {
    var ql = array<u32, 4>(0u, 0u, 0u, 0u);
    let ring_w = P.radius / f32(N_R);
    let side = u32(2 * lim + 1);
    for (var e = lid; e < side * side; e += 64u) {
        let ox = i32(e % side) - lim;
        let oy = i32(e / side) - lim;
        let rr = sqrt(f32(ox * ox + oy * oy));
        if (rr > rmax) { continue; }
        let xs = mround(kp.x + f32(ox));
        let ys = mround(kp.y + f32(oy));
        let pxi = idx2(xs - 1, ys - 1, P.w, P.h);
        var ang = atan2(f32(oy), f32(ox));
        if (ang < 0.0) { ang += 2.0 * PI; }
        let tb = min(u32(ang / (2.0 * PI / f32(N_TAN))), N_TAN - 1u);
        var q = 0u;
        if (ox >= 0 && oy >= 0) { q = 0u; }
        else if (ox < 0 && oy >= 0) { q = 1u; }
        else if (ox < 0 && oy < 0) { q = 2u; }
        else { q = 3u; }
        let srw = sr[pxi];
        for (var si = 0u; si < N_SIGMA; si++) {
            let b = (srw >> (4u * si)) & 15u;
            if (b == pd) { ql[q] += 1u; }
            for (var ri = 0u; ri < N_R; ri++) {
                let r0 = max(0.0, f32(ri) * ring_w - expand);
                let r1 = f32(ri + 1u) * ring_w + expand;
                if (rr >= r0 && rr <= r1) {
                    atomicAdd(&ds_h[((tb * N_R) + ri) * N_ANGLE + b], 1u);
                }
            }
        }
    }
    for (var qi = 0u; qi < 4u; qi++) {
        if (ql[qi] > 0u) { atomicAdd(&ds_q[qi], ql[qi]); }
    }
    workgroupBarrier();
    if (lid == 0u) {
        // quadrant argmax >= 2 means flip (same as torch votes.argmax >= 2)
        var qbest = 0u;
        var qvbest = f32(atomicLoad(&ds_q[0]));
        for (var qi = 1u; qi < 4u; qi++) {
            let qv = f32(atomicLoad(&ds_q[qi]));
            if (qv > qvbest) { qvbest = qv; qbest = qi; }
        }
        let do_flip = qbest >= 2u;
        let start_tan = (pd + N_ANGLE * u32(do_flip)) % N_TAN;
        // Norm summed in output order, as the serial pass did.
        var n2 = 0.0;
        for (var t = 0u; t < N_TAN; t++) {
            let ts = (t + start_tan) % N_TAN;
            for (var ri = 0u; ri < N_R; ri++) {
                for (var a = 0u; a < N_ANGLE; a++) {
                    let hv = f32(atomicLoad(&ds_h[((ts * N_R) + ri) * N_ANGLE + (a + pd) % N_ANGLE]));
                    n2 += hv * hv;
                }
            }
        }
        ds_f = 1.0 / sqrt(max(n2, 1e-12));
        ds_u[1] = start_tan;
        kdir[slot] = start_tan;
        var mx = 0u;
        for (var k = 0u; k < DES_DIM; k++) { mx = max(mx, atomicLoad(&ds_h[k])); }
        ds_u[2] = mx;
        dsc[slot] = f32(mx) * ds_f / 255.0;
    }
    let nrm = workgroupUniformLoad(&ds_f);
    let start_tan = ds_u[1];
    let mx = max(ds_u[2], 1u);
    let base = slot * DES_DIM;
    for (var k = lid; k < DES_DIM; k += 64u) {
        des[base + k] = f32(atomicLoad(&ds_h[ds_src(k, start_tan, pd)])) * nrm;
    }
    // The 8-bit copy the matcher screens with (match_nnq): rounded count · 255 / largest count.
    for (var w = lid; w < NN_VEC; w += 64u) {
        var word = 0u;
        for (var e = 0u; e < 4u; e++) {
            let c = atomicLoad(&ds_h[ds_src(w * 4u + e, start_tan, pd)]);
            word = word | (((c * 255u + mx / 2u) / mx) << (8u * e));
        }
        desq[slot * NN_VEC + w] = word;
    }
    if (lid == 0u) {
        kps[slot].x = kp.x * P.scale;
        kps[slot].y = kp.y * P.scale;
        kps[slot].pad = P.scale;
    }
}

// Exact cosine 1-NN, blocked like a matrix product, in two passes. Pass 1: workgroup
// (x, y) takes 64 queries and database entries [y·nn_chunk, (y + 1)·nn_chunk), walked 64 at a
// time (the database split keeps the GPU occupied: a level has only ~100 query blocks); the two 64-descriptor blocks pass through shared
// memory NN_KC vec4s at a time, and each invocation accumulates an 8 × 8 block of dot
// products (queries r·8 + ty, entries c·8 + tx), so every staged vec4 feeds 8 dot
// products instead of 1. Pass 2 (…_merge) keeps the best row per query. Each dot product
// is still summed vec4 by vec4 in index order, and ties still go to the lowest index, so
// the matches are those of the one-query-per-invocation scan.
const NN_B: u32 = 64u;    // queries / entries per block
const NN_KC: u32 = 3u;    // vec4s per staged chunk (54 = 18 × 3)
var<workgroup> nn_a: array<vec4f, NN_KC * NN_B>;  // [k][query], NN_KC × NN_B
var<workgroup> nn_b: array<vec4f, NN_KC * NN_B>;  // [k][entry]
var<workgroup> nn_bv: array<f32, 512>;   // per query, the best of each tx column
var<workgroup> nn_bi: array<u32, 512>;

fn nn_des(from_b: bool, ib: u32) -> vec4f {
    if (from_b) { return vec4f(des_b[ib], des_b[ib + 1u], des_b[ib + 2u], des_b[ib + 3u]); }
    return vec4f(des[ib], des[ib + 1u], des[ib + 2u], des[ib + 3u]);
}

// rev = false: queries des[q_offset ..], entries des_b[db_offset ..].
// rev = true:  queries des_b[db_offset ..], entries des[q_offset ..] (live keypoints only).
// Returns (best value bits, best entry) of query wid·64 + lid over row `row`'s entries
// (entry 0xffffffff if none).
fn nn_block(wid: u32, row_y: u32, lid: u32, rev: bool) -> vec2u {
    let tx = lid % 8u;
    let ty = lid / 8u;
    var qbase = P.q_offset;
    var qn = P.n_query;
    var dbase = P.db_offset;
    var dn = P.n_db;
    if (rev) {
        qbase = P.db_offset;
        qn = P.n_db;
        dbase = P.q_offset;
        dn = P.n_query;
    }
    let q0 = wid * NN_B;
    let t_lo = row_y * P.nn_chunk;
    let t_hi = min(dn, t_lo + P.nn_chunk);
    var bv: array<f32, 8>;
    var bi: array<u32, 8>;
    for (var r = 0u; r < 8u; r++) {
        bv[r] = -1e30;
        bi[r] = 0xffffffffu;
    }
    for (var t0 = t_lo; t0 < t_hi; t0 += NN_B) {
        // acc_r_h: dot products of query row r with entries c = 4h .. 4h + 3 (unrolled so they stay in registers).
        var acc00 = vec4f(0.0);
        var acc01 = vec4f(0.0);
        var acc10 = vec4f(0.0);
        var acc11 = vec4f(0.0);
        var acc20 = vec4f(0.0);
        var acc21 = vec4f(0.0);
        var acc30 = vec4f(0.0);
        var acc31 = vec4f(0.0);
        var acc40 = vec4f(0.0);
        var acc41 = vec4f(0.0);
        var acc50 = vec4f(0.0);
        var acc51 = vec4f(0.0);
        var acc60 = vec4f(0.0);
        var acc61 = vec4f(0.0);
        var acc70 = vec4f(0.0);
        var acc71 = vec4f(0.0);
        for (var k0 = 0u; k0 < NN_VEC; k0 += NN_KC) {
            for (var e = lid; e < NN_KC * NN_B; e += 64u) {
                let row = e / NN_KC;
                let k = e % NN_KC;
                var va = vec4f(0.0);
                if (q0 + row < qn) { va = nn_des(rev, (qbase + q0 + row) * DES_DIM + (k0 + k) * 4u); }
                nn_a[k * NN_B + row] = va;
                var vb = vec4f(0.0);
                if (t0 + row < t_hi) { vb = nn_des(!rev, (dbase + t0 + row) * DES_DIM + (k0 + k) * 4u); }
                nn_b[k * NN_B + row] = vb;
            }
            workgroupBarrier();
            for (var k = 0u; k < NN_KC; k++) {
                let ka = k * NN_B + ty;
                let kb = k * NN_B + tx;
                let b0 = nn_b[kb + 0u];
                let b1 = nn_b[kb + 8u];
                let b2 = nn_b[kb + 16u];
                let b3 = nn_b[kb + 24u];
                let b4 = nn_b[kb + 32u];
                let b5 = nn_b[kb + 40u];
                let b6 = nn_b[kb + 48u];
                let b7 = nn_b[kb + 56u];
                let a0 = nn_a[ka + 0u];
                acc00 += vec4f(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
                acc01 += vec4f(dot(a0, b4), dot(a0, b5), dot(a0, b6), dot(a0, b7));
                let a1 = nn_a[ka + 8u];
                acc10 += vec4f(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
                acc11 += vec4f(dot(a1, b4), dot(a1, b5), dot(a1, b6), dot(a1, b7));
                let a2 = nn_a[ka + 16u];
                acc20 += vec4f(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
                acc21 += vec4f(dot(a2, b4), dot(a2, b5), dot(a2, b6), dot(a2, b7));
                let a3 = nn_a[ka + 24u];
                acc30 += vec4f(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
                acc31 += vec4f(dot(a3, b4), dot(a3, b5), dot(a3, b6), dot(a3, b7));
                let a4 = nn_a[ka + 32u];
                acc40 += vec4f(dot(a4, b0), dot(a4, b1), dot(a4, b2), dot(a4, b3));
                acc41 += vec4f(dot(a4, b4), dot(a4, b5), dot(a4, b6), dot(a4, b7));
                let a5 = nn_a[ka + 40u];
                acc50 += vec4f(dot(a5, b0), dot(a5, b1), dot(a5, b2), dot(a5, b3));
                acc51 += vec4f(dot(a5, b4), dot(a5, b5), dot(a5, b6), dot(a5, b7));
                let a6 = nn_a[ka + 48u];
                acc60 += vec4f(dot(a6, b0), dot(a6, b1), dot(a6, b2), dot(a6, b3));
                acc61 += vec4f(dot(a6, b4), dot(a6, b5), dot(a6, b6), dot(a6, b7));
                let a7 = nn_a[ka + 56u];
                acc70 += vec4f(dot(a7, b0), dot(a7, b1), dot(a7, b2), dot(a7, b3));
                acc71 += vec4f(dot(a7, b4), dot(a7, b5), dot(a7, b6), dot(a7, b7));
            }
            workgroupBarrier();
        }
        var acc: array<f32, 64>;
        acc[0] = acc00[0];
        acc[1] = acc00[1];
        acc[2] = acc00[2];
        acc[3] = acc00[3];
        acc[4] = acc01[0];
        acc[5] = acc01[1];
        acc[6] = acc01[2];
        acc[7] = acc01[3];
        acc[8] = acc10[0];
        acc[9] = acc10[1];
        acc[10] = acc10[2];
        acc[11] = acc10[3];
        acc[12] = acc11[0];
        acc[13] = acc11[1];
        acc[14] = acc11[2];
        acc[15] = acc11[3];
        acc[16] = acc20[0];
        acc[17] = acc20[1];
        acc[18] = acc20[2];
        acc[19] = acc20[3];
        acc[20] = acc21[0];
        acc[21] = acc21[1];
        acc[22] = acc21[2];
        acc[23] = acc21[3];
        acc[24] = acc30[0];
        acc[25] = acc30[1];
        acc[26] = acc30[2];
        acc[27] = acc30[3];
        acc[28] = acc31[0];
        acc[29] = acc31[1];
        acc[30] = acc31[2];
        acc[31] = acc31[3];
        acc[32] = acc40[0];
        acc[33] = acc40[1];
        acc[34] = acc40[2];
        acc[35] = acc40[3];
        acc[36] = acc41[0];
        acc[37] = acc41[1];
        acc[38] = acc41[2];
        acc[39] = acc41[3];
        acc[40] = acc50[0];
        acc[41] = acc50[1];
        acc[42] = acc50[2];
        acc[43] = acc50[3];
        acc[44] = acc51[0];
        acc[45] = acc51[1];
        acc[46] = acc51[2];
        acc[47] = acc51[3];
        acc[48] = acc60[0];
        acc[49] = acc60[1];
        acc[50] = acc60[2];
        acc[51] = acc60[3];
        acc[52] = acc61[0];
        acc[53] = acc61[1];
        acc[54] = acc61[2];
        acc[55] = acc61[3];
        acc[56] = acc70[0];
        acc[57] = acc70[1];
        acc[58] = acc70[2];
        acc[59] = acc70[3];
        acc[60] = acc71[0];
        acc[61] = acc71[1];
        acc[62] = acc71[2];
        acc[63] = acc71[3];
        for (var c = 0u; c < 8u; c++) {
            let ld = t0 + c * 8u + tx;
            if (ld >= t_hi) { continue; }
            let dj = dbase + ld;
            if (rev && kps[dj].score <= 0.0) { continue; }
            for (var r = 0u; r < 8u; r++) {
                if (acc[r * 8u + c] > bv[r]) {
                    bv[r] = acc[r * 8u + c];
                    bi[r] = dj;
                }
            }
        }
    }
    for (var r = 0u; r < 8u; r++) {
        nn_bv[(r * 8u + ty) * 8u + tx] = bv[r];
        nn_bi[(r * 8u + ty) * 8u + tx] = bi[r];
    }
    workgroupBarrier();
    var best = -1e30;
    var bj = 0xffffffffu;
    for (var c = 0u; c < 8u; c++) {
        let v = nn_bv[lid * 8u + c];
        let j = nn_bi[lid * 8u + c];
        if (j == 0xffffffffu) { continue; }
        if (v > best || (v == best && j < bj)) {
            best = v;
            bj = j;
        }
    }
    return vec2u(bitcast<u32>(best), bj);
}

@compute @workgroup_size(64)
fn match_nn(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u,
            @builtin(local_invocation_index) lid: u32) {
    let b = nn_block(wid.x, wid.y, lid, false);
    let lq = wid.x * NN_B + lid;
    if (lq < P.n_query) { nn_part[lq * nw.y + wid.y] = vec4u(b, 0u, 0u); }
}

@compute @workgroup_size(64)
fn match_nn_rev(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u,
                @builtin(local_invocation_index) lid: u32) {
    let b = nn_block(wid.x, wid.y, lid, true);
    let lq = wid.x * NN_B + lid;
    if (lq < P.n_db) { nn_part[lq * nw.y + wid.y] = vec4u(b, 0u, 0u); }
}

// Best of the rows in row order; a row only wins with a strictly larger value, so ties keep
// the lowest entry (rows cover increasing entries).
fn nn_merge(lq: u32, dn: u32) -> u32 {
    let rows = (dn + P.nn_chunk - 1u) / P.nn_chunk;
    var best = -1e30;
    var bj = 0xffffffffu;
    for (var y = 0u; y < rows; y++) {
        let b = nn_part[lq * rows + y].xy;
        if (b.y == 0xffffffffu) { continue; }
        let v = bitcast<f32>(b.x);
        if (v > best) {
            best = v;
            bj = b.y;
        }
    }
    return bj;
}

@compute @workgroup_size(64)
fn match_nn_merge(@builtin(global_invocation_id) gid: vec3u) {
    let lq = gid.x;
    if (lq >= P.n_query) { return; }
    let i = P.q_offset + lq;
    match_j[i] = select(0xffffffffu, nn_merge(lq, P.n_db), kps[i].score > 0.0);
}

@compute @workgroup_size(64)
fn match_nn_rev_merge(@builtin(global_invocation_id) gid: vec3u) {
    let lq = gid.x;
    if (lq >= P.n_db) { return; }
    match_rev[P.db_offset + lq] = nn_merge(lq, P.n_query);
}

// Screened 1-NN (the default; match_nn above is the exhaustive pass). Pass 1 (match_nnq):
// the same blocking over the 8-bit copies, 4 products per packed dot (dot4q), each score
// f32(integer dot) · entry scale; every (query, row) keeps its two best entries. Pass 2
// (match_nnq_pick): exact f32 dot products of those 2·rows candidates, summed as match_nn
// sums them, best value, lowest entry on ties. So the match equals match_nn's unless the
// exact nearest neighbour is outside the top two of its row by the 8-bit score.
const NQ_KC: u32 = 9u;  // words per staged chunk (54 = 6 × 9)
var<workgroup> nq_a: array<u32, NQ_KC * NN_B>;   // [k][query], NQ_KC × NN_B
var<workgroup> nq_b: array<u32, NQ_KC * NN_B>;   // [k][entry]
var<workgroup> nq_bs: array<f32, 64>;   // entry scales of the block
var<workgroup> nq_cv: array<f32, 1024>; // [query][tx][2] candidates
var<workgroup> nq_ci: array<u32, 1024>;

fn nq_better(v: f32, i: u32, bv: f32, bi: u32) -> bool {
    return i != 0xffffffffu && (v > bv || (v == bv && i < bi));
}

fn nq_sub() -> u32 {
    return select(P.nn_chunk, P.nn_sub, P.nn_sub != 0u && P.nn_sub < P.nn_chunk);
}

// The two best entries (8-bit score, then lowest entry) of chunk y over its sub-rows: what one
// workgroup per chunk keeps (the order is total, so the best two of the union are the best two
// of the sub-rows' best two). stride: nn_part entries per query.
fn nq_cands(lq: u32, y: u32, dn: u32) -> vec2u {
    let per = (P.nn_chunk + nq_sub() - 1u) / nq_sub();
    let stride = (dn + P.nn_chunk - 1u) / P.nn_chunk * per;
    var bv = -1.0;
    var bi = 0xffffffffu;
    var cv = -1.0;
    var ci = 0xffffffffu;
    for (var s = 0u; s < per; s++) {
        let c = nn_part[lq * stride + y * per + s];
        for (var h = 0u; h < 2u; h++) {
            let i = c[h];
            let v = bitcast<f32>(c[2u + h]);
            if (nq_better(v, i, bv, bi)) {
                cv = bv;
                ci = bi;
                bv = v;
                bi = i;
            } else if (nq_better(v, i, cv, ci)) {
                cv = v;
                ci = i;
            }
        }
    }
    return vec2u(bi, ci);
}

fn nq_block(wid: u32, row_y: u32, lid: u32, rev: bool) -> vec4u {
    let tx = lid % 8u;
    let ty = lid / 8u;
    var qbase = P.q_offset;
    var qn = P.n_query;
    var dbase = P.db_offset;
    var dn = P.n_db;
    if (rev) {
        qbase = P.db_offset;
        qn = P.n_db;
        dbase = P.q_offset;
        dn = P.n_query;
    }
    let q0 = wid * NN_B;
    // row_y: sub-row s of chunk c (nq_rows), entries [c·nn_chunk + s·sub, …) within the chunk
    let sub = nq_sub();
    let per = (P.nn_chunk + sub - 1u) / sub;
    let c_lo = (row_y / per) * P.nn_chunk;
    let t_lo = c_lo + (row_y % per) * sub;
    let t_hi = min(min(dn, c_lo + P.nn_chunk), t_lo + sub);
    var v1: array<f32, 8>;
    var i1: array<u32, 8>;
    var v2: array<f32, 8>;
    var i2: array<u32, 8>;
    for (var r = 0u; r < 8u; r++) {
        v1[r] = -1.0;
        v2[r] = -1.0;
        i1[r] = 0xffffffffu;
        i2[r] = 0xffffffffu;
    }
    for (var t0 = t_lo; t0 < t_hi; t0 += NN_B) {
        var acc00 = vec4u(0u);
        var acc01 = vec4u(0u);
        var acc10 = vec4u(0u);
        var acc11 = vec4u(0u);
        var acc20 = vec4u(0u);
        var acc21 = vec4u(0u);
        var acc30 = vec4u(0u);
        var acc31 = vec4u(0u);
        var acc40 = vec4u(0u);
        var acc41 = vec4u(0u);
        var acc50 = vec4u(0u);
        var acc51 = vec4u(0u);
        var acc60 = vec4u(0u);
        var acc61 = vec4u(0u);
        var acc70 = vec4u(0u);
        var acc71 = vec4u(0u);
        for (var k0 = 0u; k0 < NN_VEC; k0 += NQ_KC) {
            for (var e = lid; e < NQ_KC * NN_B; e += 64u) {
                let row = e / NQ_KC;
                let k = e % NQ_KC;
                var wa = 0u;
                if (q0 + row < qn) {
                    let o = (qbase + q0 + row) * NN_VEC + k0 + k;
                    wa = select(desq[o], desq_b[o], rev);
                }
                nq_a[k * NN_B + row] = wa;
                var wb = 0u;
                if (t0 + row < t_hi) {
                    let o = (dbase + t0 + row) * NN_VEC + k0 + k;
                    wb = select(desq_b[o], desq[o], rev);
                }
                nq_b[k * NN_B + row] = wb;
            }
            if (k0 == 0u) {
                var sc = 0.0;
                let j = dbase + t0 + lid;
                if (t0 + lid < t_hi) { sc = select(dsc_b[j], dsc[j], rev); }
                nq_bs[lid] = sc;
            }
            workgroupBarrier();
            for (var k = 0u; k < NQ_KC; k++) {
                let ka = k * NN_B + ty;
                let kb = k * NN_B + tx;
                let b0 = nq_b[kb + 0u];
                let b1 = nq_b[kb + 8u];
                let b2 = nq_b[kb + 16u];
                let b3 = nq_b[kb + 24u];
                let b4 = nq_b[kb + 32u];
                let b5 = nq_b[kb + 40u];
                let b6 = nq_b[kb + 48u];
                let b7 = nq_b[kb + 56u];
                let a0 = nq_a[ka + 0u];
                acc00 += vec4u(dot4q(a0, b0), dot4q(a0, b1), dot4q(a0, b2), dot4q(a0, b3));
                acc01 += vec4u(dot4q(a0, b4), dot4q(a0, b5), dot4q(a0, b6), dot4q(a0, b7));
                let a1 = nq_a[ka + 8u];
                acc10 += vec4u(dot4q(a1, b0), dot4q(a1, b1), dot4q(a1, b2), dot4q(a1, b3));
                acc11 += vec4u(dot4q(a1, b4), dot4q(a1, b5), dot4q(a1, b6), dot4q(a1, b7));
                let a2 = nq_a[ka + 16u];
                acc20 += vec4u(dot4q(a2, b0), dot4q(a2, b1), dot4q(a2, b2), dot4q(a2, b3));
                acc21 += vec4u(dot4q(a2, b4), dot4q(a2, b5), dot4q(a2, b6), dot4q(a2, b7));
                let a3 = nq_a[ka + 24u];
                acc30 += vec4u(dot4q(a3, b0), dot4q(a3, b1), dot4q(a3, b2), dot4q(a3, b3));
                acc31 += vec4u(dot4q(a3, b4), dot4q(a3, b5), dot4q(a3, b6), dot4q(a3, b7));
                let a4 = nq_a[ka + 32u];
                acc40 += vec4u(dot4q(a4, b0), dot4q(a4, b1), dot4q(a4, b2), dot4q(a4, b3));
                acc41 += vec4u(dot4q(a4, b4), dot4q(a4, b5), dot4q(a4, b6), dot4q(a4, b7));
                let a5 = nq_a[ka + 40u];
                acc50 += vec4u(dot4q(a5, b0), dot4q(a5, b1), dot4q(a5, b2), dot4q(a5, b3));
                acc51 += vec4u(dot4q(a5, b4), dot4q(a5, b5), dot4q(a5, b6), dot4q(a5, b7));
                let a6 = nq_a[ka + 48u];
                acc60 += vec4u(dot4q(a6, b0), dot4q(a6, b1), dot4q(a6, b2), dot4q(a6, b3));
                acc61 += vec4u(dot4q(a6, b4), dot4q(a6, b5), dot4q(a6, b6), dot4q(a6, b7));
                let a7 = nq_a[ka + 56u];
                acc70 += vec4u(dot4q(a7, b0), dot4q(a7, b1), dot4q(a7, b2), dot4q(a7, b3));
                acc71 += vec4u(dot4q(a7, b4), dot4q(a7, b5), dot4q(a7, b6), dot4q(a7, b7));
            }
            workgroupBarrier();
        }
        var acc: array<u32, 64>;
        acc[0] = acc00[0];
        acc[1] = acc00[1];
        acc[2] = acc00[2];
        acc[3] = acc00[3];
        acc[4] = acc01[0];
        acc[5] = acc01[1];
        acc[6] = acc01[2];
        acc[7] = acc01[3];
        acc[8] = acc10[0];
        acc[9] = acc10[1];
        acc[10] = acc10[2];
        acc[11] = acc10[3];
        acc[12] = acc11[0];
        acc[13] = acc11[1];
        acc[14] = acc11[2];
        acc[15] = acc11[3];
        acc[16] = acc20[0];
        acc[17] = acc20[1];
        acc[18] = acc20[2];
        acc[19] = acc20[3];
        acc[20] = acc21[0];
        acc[21] = acc21[1];
        acc[22] = acc21[2];
        acc[23] = acc21[3];
        acc[24] = acc30[0];
        acc[25] = acc30[1];
        acc[26] = acc30[2];
        acc[27] = acc30[3];
        acc[28] = acc31[0];
        acc[29] = acc31[1];
        acc[30] = acc31[2];
        acc[31] = acc31[3];
        acc[32] = acc40[0];
        acc[33] = acc40[1];
        acc[34] = acc40[2];
        acc[35] = acc40[3];
        acc[36] = acc41[0];
        acc[37] = acc41[1];
        acc[38] = acc41[2];
        acc[39] = acc41[3];
        acc[40] = acc50[0];
        acc[41] = acc50[1];
        acc[42] = acc50[2];
        acc[43] = acc50[3];
        acc[44] = acc51[0];
        acc[45] = acc51[1];
        acc[46] = acc51[2];
        acc[47] = acc51[3];
        acc[48] = acc60[0];
        acc[49] = acc60[1];
        acc[50] = acc60[2];
        acc[51] = acc60[3];
        acc[52] = acc61[0];
        acc[53] = acc61[1];
        acc[54] = acc61[2];
        acc[55] = acc61[3];
        acc[56] = acc70[0];
        acc[57] = acc70[1];
        acc[58] = acc70[2];
        acc[59] = acc70[3];
        acc[60] = acc71[0];
        acc[61] = acc71[1];
        acc[62] = acc71[2];
        acc[63] = acc71[3];
        for (var c = 0u; c < 8u; c++) {
            let ld = t0 + c * 8u + tx;
            if (ld >= t_hi) { continue; }
            let dj = dbase + ld;
            if (rev && kps[dj].score <= 0.0) { continue; }
            let bs = nq_bs[c * 8u + tx];
            for (var r = 0u; r < 8u; r++) {
                let v = f32(acc[r * 8u + c]) * bs;
                if (v > v1[r]) {
                    v2[r] = v1[r];
                    i2[r] = i1[r];
                    v1[r] = v;
                    i1[r] = dj;
                } else if (v > v2[r]) {
                    v2[r] = v;
                    i2[r] = dj;
                }
            }
        }
    }
    for (var r = 0u; r < 8u; r++) {
        let o = ((r * 8u + ty) * 8u + tx) * 2u;
        nq_cv[o] = v1[r];
        nq_ci[o] = i1[r];
        nq_cv[o + 1u] = v2[r];
        nq_ci[o + 1u] = i2[r];
    }
    workgroupBarrier();
    var bv = -1.0;
    var bi = 0xffffffffu;
    var cv = -1.0;
    var ci = 0xffffffffu;
    for (var k = 0u; k < 16u; k++) {
        let v = nq_cv[lid * 16u + k];
        let i = nq_ci[lid * 16u + k];
        if (nq_better(v, i, bv, bi)) {
            cv = bv;
            ci = bi;
            bv = v;
            bi = i;
        } else if (nq_better(v, i, cv, ci)) {
            cv = v;
            ci = i;
        }
    }
    return vec4u(bi, ci, bitcast<u32>(bv), bitcast<u32>(cv));
}

@compute @workgroup_size(64)
fn match_nnq(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u,
             @builtin(local_invocation_index) lid: u32) {
    let b = nq_block(wid.x, wid.y, lid, false);
    let lq = wid.x * NN_B + lid;
    if (lq < P.n_query) { nn_part[lq * nw.y + wid.y] = b; }
}

@compute @workgroup_size(64)
fn match_nnq_rev(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u,
                 @builtin(local_invocation_index) lid: u32) {
    let b = nq_block(wid.x, wid.y, lid, true);
    let lq = wid.x * NN_B + lid;
    if (lq < P.n_db) { nn_part[lq * nw.y + wid.y] = b; }
}

// Exact re-rank of query lq's candidates (rows × 2 in nn_part).
fn nq_pick(lq: u32, rev: bool) -> u32 {
    var qbase = P.q_offset;
    var dn = P.n_db;
    if (rev) {
        qbase = P.db_offset;
        dn = P.n_query;
    }
    let rows = (dn + P.nn_chunk - 1u) / P.nn_chunk;
    var q: array<vec4f, NN_VEC>;
    let iq = (qbase + lq) * DES_DIM;
    for (var kv = 0u; kv < NN_VEC; kv++) { q[kv] = nn_des(rev, iq + kv * 4u); }
    var best = -1e30;
    var bj = 0xffffffffu;
    for (var y = 0u; y < rows; y++) {
        let c = nq_cands(lq, y, dn);
        for (var h = 0u; h < 2u; h++) {
            let j = c[h];
            if (j == 0xffffffffu) { continue; }
            var acc = 0.0;
            let ib = j * DES_DIM;
            for (var kv = 0u; kv < NN_VEC; kv++) { acc += dot(q[kv], nn_des(!rev, ib + kv * 4u)); }
            if (acc > best || (acc == best && j < bj)) {
                best = acc;
                bj = j;
            }
        }
    }
    return bj;
}

@compute @workgroup_size(64)
fn match_nnq_pick(@builtin(global_invocation_id) gid: vec3u) {
    let lq = gid.x;
    if (lq >= P.n_query) { return; }
    let i = P.q_offset + lq;
    if (kps[i].score <= 0.0) {
        match_j[i] = 0xffffffffu;
        return;
    }
    match_j[i] = nq_pick(lq, false);
}

// Exact re-rank with MATLAB matchFeatures' tests (unit descriptors, SSD = 2 − 2·dot): the best
// candidate is kept when its SSD is below max_ssd and below ratio × the second best's SSD
// (POS-GIFT's matcher). Queries with no second candidate keep the threshold test only.
@compute @workgroup_size(64)
fn match_nnq_pick_ratio(@builtin(global_invocation_id) gid: vec3u) {
    let lq = gid.x;
    if (lq >= P.n_query) { return; }
    let i = P.q_offset + lq;
    match_j[i] = 0xffffffffu;
    if (kps[i].score <= 0.0) { return; }
    let rows = (P.n_db + P.nn_chunk - 1u) / P.nn_chunk;
    var q: array<vec4f, NN_VEC>;
    let iq = i * DES_DIM;
    for (var kv = 0u; kv < NN_VEC; kv++) { q[kv] = nn_des(false, iq + kv * 4u); }
    var b1 = -1e30;
    var j1 = 0xffffffffu;
    var b2 = -1e30;
    for (var y = 0u; y < rows; y++) {
        let c = nq_cands(lq, y, P.n_db);
        for (var h = 0u; h < 2u; h++) {
            let j = c[h];
            if (j == 0xffffffffu) { continue; }
            var acc = 0.0;
            let ib = j * DES_DIM;
            for (var kv = 0u; kv < NN_VEC; kv++) { acc += dot(q[kv], nn_des(true, ib + kv * 4u)); }
            if (acc > b1 || (acc == b1 && j < j1)) {
                b2 = b1;
                b1 = acc;
                j1 = j;
            } else if (acc > b2) {
                b2 = acc;
            }
        }
    }
    if (j1 == 0xffffffffu) { return; }
    let d1 = max(2.0 - 2.0 * b1, 0.0);
    if (d1 >= P.max_ssd) { return; }
    if (b2 > -1e29 && d1 >= P.ratio * max(2.0 - 2.0 * b2, 0.0)) { return; }
    match_j[i] = j1;
}

// Append this slice's matches (match_j) to the pooled correspondence list; count in counters[3].
@compute @workgroup_size(64)
fn append_corr(@builtin(global_invocation_id) gid: vec3u) {
    let li = gid.x;
    if (li >= P.n_query) { return; }
    let i = P.q_offset + li;
    if (!match_ok(i)) { return; }
    let s = atomicAdd(&counters[3], 1u);
    if (s < MAX_CORR) { corr[s] = Corr(i, match_j[i]); }
}

@compute @workgroup_size(64)
fn match_nnq_rev_pick(@builtin(global_invocation_id) gid: vec3u) {
    let lq = gid.x;
    if (lq >= P.n_db) { return; }
    match_rev[P.db_offset + lq] = nq_pick(lq, true);
}

@compute @workgroup_size(64)
fn keep_mutual(@builtin(global_invocation_id) gid: vec3u) {
    let li = gid.x;
    if (li >= P.n_query) { return; }
    let i = P.q_offset + li;
    let j = match_j[i];
    if (j == 0xffffffffu || j < P.db_offset || j >= P.db_offset + P.n_db) {
        match_j[i] = 0xffffffffu;
        return;
    }
    if (match_rev[j] != i) {
        match_j[i] = 0xffffffffu;
    }
}

fn match_ok(i: u32) -> bool {
    if (i < P.q_offset || i >= P.q_offset + P.n_query) { return false; }
    if (kps[i].score <= 0.0) { return false; }
    let j = match_j[i];
    if (j == 0xffffffffu || j < P.db_offset || j >= P.db_offset + P.n_db) { return false; }
    if (kps_b[j].score <= 0.0) { return false; }
    return true;
}

@compute @workgroup_size(1)
fn zero_pack() {
    atomicStore(&counters[2], 0u);
    atomicStore(&counters[4], 0u);
}

// One thread per query. counters[2] is the packed length. Order does not matter:
// the trial sampler still draws the original query indices.
@compute @workgroup_size(64)
fn pack_fsc(@builtin(global_invocation_id) gid: vec3u) {
    let li = gid.x;
    if (li >= P.n_query) { return; }
    let i = P.q_offset + li;
    if (!match_ok(i)) {
        fsc_ok[li] = 0xffffffffu;
        return;
    }
    let j = match_j[i];
    fsc_ok[li] = j;
    let s = atomicAdd(&counters[2], 1u);
    fsc_xy[s] = vec4f(kps[i].x, kps[i].y, kps_b[j].x, kps_b[j].y);
}

// Affine through three correspondences, or ok = false (collinear, or scale outside the limits).
struct Hyp { ok: bool, abc: vec3f, defv: vec3f }

fn fsc_solve(p0: vec2f, p1: vec2f, p2: vec2f, q0: vec2f, q1: vec2f, q2: vec2f) -> Hyp {
    var h: Hyp;
    h.ok = false;
    let X = mat3x3f(
        vec3f(p0.x, p1.x, p2.x),
        vec3f(p0.y, p1.y, p2.y),
        vec3f(1.0, 1.0, 1.0),
    );
    let det = dot(X[0], cross(X[1], X[2]));
    if (abs(det) < 1e-8) { return h; }
    let Xi = inverse3(X);
    h.abc = Xi * vec3f(q0.x, q1.x, q2.x);
    h.defv = Xi * vec3f(q0.y, q1.y, q2.y);
    h.ok = scale_ok(h.abc.x * h.defv.y - h.abc.y * h.defv.x);
    return h;
}

fn fsc_hyp(t: u32) -> Hyp {
    var h: Hyp;
    h.ok = false;
    let n = P.n_query;
    let base = P.q_offset;
    if (t >= P.n_trials || n < 3u) { return h; }
    var r = pcg(P.seed + t * 17u + 1u);
    var i0 = 0u;
    var i1 = 1u;
    var i2 = 2u;
    var j0 = 0u;
    var j1 = 0u;
    var j2 = 0u;
    var found = false;
    for (var attempt = 0u; attempt < 24u; attempt++) {
        r = pcg(r);
        i0 = base + (r % n);
        r = pcg(r);
        i1 = base + (r % n);
        r = pcg(r);
        i2 = base + (r % n);
        if (i0 == i1 || i0 == i2 || i1 == i2) { continue; }
        // match_ok per draw, precomputed by pack_fsc
        j0 = fsc_ok[i0 - base];
        j1 = fsc_ok[i1 - base];
        j2 = fsc_ok[i2 - base];
        if (j0 == 0xffffffffu || j1 == 0xffffffffu || j2 == 0xffffffffu) { continue; }
        if (j0 == j1 || j0 == j2 || j1 == j2) { continue; }
        found = true;
        break;
    }
    if (!found) { return h; }
    return fsc_solve(
        vec2f(kps[i0].x, kps[i0].y), vec2f(kps[i1].x, kps[i1].y), vec2f(kps[i2].x, kps[i2].y),
        vec2f(kps_b[j0].x, kps_b[j0].y), vec2f(kps_b[j1].x, kps_b[j1].y), vec2f(kps_b[j2].x, kps_b[j2].y));
}

// Coverage scoring (P.flags bit 2): a hypothesis ranks first by how many cells of an 8 × 8 grid
// over image 1 (P.src_w × P.src_h) its inliers occupy, then by their number; a chance cluster
// of self-consistent matches in one patch no longer outranks a transform supported everywhere.
fn cov_bit(p: vec2f) -> vec2u {
    let cx = u32(clamp(p.x / f32(max(P.src_w, 1u)) * 8.0, 0.0, 7.0));
    let cy = u32(clamp(p.y / f32(max(P.src_h, 1u)) * 8.0, 0.0, 7.0));
    let cell = cy * 8u + cx;
    return vec2u(select(0u, 1u << (cell & 31u), cell < 32u), select(0u, 1u << (cell & 31u), cell >= 32u));
}

fn fsc_score(c: u32, ml: u32, mh: u32) -> u32 {
    if ((P.flags & 4u) == 0u) { return c; }
    return ((countOneBits(ml) + countOneBits(mh)) << 16u) | min(c, 0xffffu);
}

fn fsc_store(t: u32, h: Hyp, ninl: u32) {
    if (t >= P.n_trials) { return; }
    if (!h.ok) {
        trials[t].n_inl = 0u;
        return;
    }
    trials[t].n_inl = ninl;
    trials[t].a = h.abc.x;
    trials[t].b = h.abc.y;
    trials[t].tx = h.abc.z;
    trials[t].c = h.defv.x;
    trials[t].d = h.defv.y;
    trials[t].ty = h.defv.z;
}

// Trials run in two passes. fsc_hyps draws every trial's hypothesis into trials[t] (n_inl 0)
// and lists the valid ones in fsc_list (their count in counters[4]); fsc_trial then counts
// inliers for the listed trials only (the scale gate rejects about half, which would otherwise
// count every point for nothing). Scores land at each trial's own index and ties resolve to
// the lowest index, so the result is the same as scoring every trial.
@group(0) @binding(33) var<storage, read_write> fsc_list: array<u32>;

@compute @workgroup_size(64)
fn fsc_hyps(@builtin(global_invocation_id) gid: vec3u) {
    let t = gid.x;
    if (t >= P.n_trials) { return; }
    let h = fsc_hyp(t);
    fsc_store(t, h, 0u);
    if (h.ok) { fsc_list[atomicAdd(&counters[4], 1u)] = t; }
}

// A trial's best-so-far (score, index): more inliers, then the lower index.
fn fsc_better(v: u32, i: u32, bv: u32, bi: u32) -> bool {
    return v > bv || (v == bv && v > 0u && i < bi);
}

// Inlier counting, blocked: each invocation holds FSC_PER (4, unrolled) listed trials, and the
// workgroup stages the points (fsc_xy, counters[2] of them) through shared memory 256 at a
// time, with their coverage cells, so every point is loaded once per 256 trials.
const FSC_PER: u32 = 4u;
const FSC_WG: u32 = 256u; // listed trials per workgroup = 64 · FSC_PER
var<workgroup> xy_tile: array<vec4f, 256>;
var<workgroup> cov_tile: array<vec2u, 256>;
var<workgroup> fsc_mv: vec2u;

struct Aff { a: f32, b: f32, tx: f32, c: f32, d: f32, ty: f32 }

fn fsc_aff(t: u32) -> Aff {
    if (t == 0xffffffffu) { return Aff(0.0, 0.0, 0.0, 0.0, 0.0, 0.0); }
    let r = trials[t];
    return Aff(r.a, r.b, r.tx, r.c, r.d, r.ty);
}

fn fsc_in(f: Aff, p: vec2f, q: vec2f) -> bool {
    let e = apply_aff(f.a, f.b, f.tx, f.c, f.d, f.ty, p) - q;
    return dot(e, e) < P.inlier2;
}

@compute @workgroup_size(64)
fn fsc_trial(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    if (lid == 0u) { fsc_mv = vec2u(atomicLoad(&counters[2]), atomicLoad(&counters[4])); }
    let mv = workgroupUniformLoad(&fsc_mv);
    let m = mv.x;
    let nv = mv.y;
    let wb = wid.x * FSC_WG;
    if (wb >= nv) { return; }
    let cover = (P.flags & 4u) != 0u;
    var t: array<u32, 4>;
    for (var j = 0u; j < 4u; j++) {
        let k = wb + j * 64u + lid;
        t[j] = select(0xffffffffu, fsc_list[k], k < nv);
    }
    let f0 = fsc_aff(t[0]);
    let f1 = fsc_aff(t[1]);
    let f2 = fsc_aff(t[2]);
    let f3 = fsc_aff(t[3]);
    var c0 = 0u;
    var m0l = 0u;
    var m0h = 0u;
    var c1 = 0u;
    var m1l = 0u;
    var m1h = 0u;
    var c2 = 0u;
    var m2l = 0u;
    var m2h = 0u;
    var c3 = 0u;
    var m3l = 0u;
    var m3h = 0u;
    for (var k0 = 0u; k0 < m; k0 += 256u) {
        for (var e = lid; e < 256u; e += 64u) {
            if (k0 + e < m) {
                let v = fsc_xy[k0 + e];
                xy_tile[e] = v;
                if (cover) { cov_tile[e] = cov_bit(vec2f(v.x, v.y)); }
            }
        }
        workgroupBarrier();
        let kn = min(256u, m - k0);
        if (cover) {
            for (var k = 0u; k < kn; k++) {
                let v = xy_tile[k];
                let p = vec2f(v.x, v.y);
                let q = vec2f(v.z, v.w);
                let cb = cov_tile[k];
                if (fsc_in(f0, p, q)) { c0 += 1u; m0l |= cb.x; m0h |= cb.y; }
                if (fsc_in(f1, p, q)) { c1 += 1u; m1l |= cb.x; m1h |= cb.y; }
                if (fsc_in(f2, p, q)) { c2 += 1u; m2l |= cb.x; m2h |= cb.y; }
                if (fsc_in(f3, p, q)) { c3 += 1u; m3l |= cb.x; m3h |= cb.y; }
            }
        } else {
            for (var k = 0u; k < kn; k++) {
                let v = xy_tile[k];
                let p = vec2f(v.x, v.y);
                let q = vec2f(v.z, v.w);
                c0 += select(0u, 1u, fsc_in(f0, p, q));
                c1 += select(0u, 1u, fsc_in(f1, p, q));
                c2 += select(0u, 1u, fsc_in(f2, p, q));
                c3 += select(0u, 1u, fsc_in(f3, p, q));
            }
        }
        workgroupBarrier();
    }
    var s: array<u32, 4>;
    s[0] = fsc_score(c0, m0l, m0h);
    s[1] = fsc_score(c1, m1l, m1h);
    s[2] = fsc_score(c2, m2l, m2h);
    s[3] = fsc_score(c3, m3l, m3h);
    var bv = 0u;
    var bi = 0xffffffffu;
    for (var j = 0u; j < 4u; j++) {
        if (t[j] == 0xffffffffu) { continue; }
        trials[t[j]].n_inl = s[j];
        if (fsc_better(s[j], t[j], bv, bi)) {
            bv = s[j];
            bi = t[j];
        }
    }
    // this workgroup's best into trials[wb].pad (0xffffffff: none scored)
    scan_a[lid] = bv;
    scan_b[lid] = bi;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st = st >> 1u) {
        if (lid < st && fsc_better(scan_a[lid + st], scan_b[lid + st], scan_a[lid], scan_b[lid])) {
            scan_a[lid] = scan_a[lid + st];
            scan_b[lid] = scan_b[lid + st];
        }
        workgroupBarrier();
    }
    if (lid == 0u) { trials[wb].pad = select(0xffffffffu, scan_b[0], scan_a[0] > 0u); }
}

// One workgroup: the trial with the most inliers, the lowest index among ties (what a serial
// scan keeping the first strict maximum returns), from each counting workgroup's best.
@compute @workgroup_size(256)
fn fsc_reduce(@builtin(local_invocation_index) lid: u32) {
    var tb = 0u;
    var ti = 0xffffffffu;
    let ng = (atomicLoad(&counters[4]) + FSC_WG - 1u) / FSC_WG;
    for (var gi = lid; gi < ng; gi += 256u) {
        let t = trials[gi * FSC_WG].pad;
        if (t == 0xffffffffu) { continue; }
        let v = trials[t].n_inl;
        if (fsc_better(v, t, tb, ti)) {
            tb = v;
            ti = t;
        }
    }
    scan_a[lid] = tb;
    scan_b[lid] = ti;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s && fsc_better(scan_a[lid + s], scan_b[lid + s], scan_a[lid], scan_b[lid])) {
            scan_a[lid] = scan_a[lid + s];
            scan_b[lid] = scan_b[lid + s];
        }
        workgroupBarrier();
    }
    if (lid != 0u) { return; }
    let best = scan_a[0];
    let bi = scan_b[0];
    if (best == 0u) {
        affine[0] = 1.0; affine[1] = 0.0; affine[2] = 0.0;
        affine[3] = 0.0; affine[4] = 1.0; affine[5] = 0.0;
        affine[6] = 0.0; affine[7] = 0.0; affine[8] = 1.0;
        atomicStore(&counters[1], 0u);
        return;
    }
    affine[0] = trials[bi].a;
    affine[1] = trials[bi].b;
    affine[2] = trials[bi].tx;
    affine[3] = trials[bi].c;
    affine[4] = trials[bi].d;
    affine[5] = trials[bi].ty;
    affine[6] = 0.0;
    affine[7] = 0.0;
    affine[8] = 1.0;
    atomicStore(&counters[1], best);
}

fn aff_err2(p: vec2f, q: vec2f) -> f32 {
    let pr = apply_aff(affine[0], affine[1], affine[2], affine[3], affine[4], affine[5], p);
    let e = pr - q;
    return dot(e, e);
}

// Log histogram of |steered response| on the current image. Median of this
// histogram is the automatic feature-map threshold (first octave only).
@compute @workgroup_size(8, 8)
fn mag_hist(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg: vec3u,
    @builtin(local_invocation_id) lxy: vec3u,
) {
    for (var k = lid; k < 256u; k += 64u) { atomicStore(&mag_local[k], 0u); }
    let fog = fog_responses(wg, lxy);
    if (gid.x < P.w && gid.y < P.h) {
        let rx = fog.rx;
        let ry = fog.ry;
        for (var th = 0u; th < N_ANGLE; th++) {
            let ang = f32(th) * (PI / f32(N_ANGLE));
            let c = cos(ang);
            let sn = sin(ang);
            for (var si = 0u; si < N_SIGMA; si++) {
                let a = abs(c * rx[si] + sn * ry[si]);
                var b: i32 = 0;
                if (a > 0.0) {
                    let lg = log2(a);
                    b = i32(floor((lg + 20.0) / (32.0 / 256.0)));
                    b = clamp(b, 0, 255);
                }
                atomicAdd(&mag_local[u32(b)], 1u);
            }
        }
    }
    workgroupBarrier();
    for (var k = lid; k < 256u; k += 64u) {
        let c = atomicLoad(&mag_local[k]);
        if (c > 0u) { atomicAdd(&magb[k], c); }
    }
}

@compute @workgroup_size(64)
fn keep_corr(@builtin(global_invocation_id) gid: vec3u) {
    if (atomicLoad(&counters[1]) == 0u) { return; }
    let li = gid.x;
    if (li >= P.n_query) { return; }
    let i = P.q_offset + li;
    if (!match_ok(i)) { return; }
    let qj = match_j[i];
    let e2 = aff_err2(vec2f(kps[i].x, kps[i].y), vec2f(kps_b[qj].x, kps_b[qj].y));
    if (e2 >= P.inlier2) { return; }
    let s = atomicAdd(&counters[3], 1u);
    if (s < MAX_CORR) { corr[s] = Corr(i, qj); }
}

fn fsc_hyp_corr(t: u32) -> Hyp {
    var h: Hyp;
    h.ok = false;
    let n = P.n_query;
    if (t >= P.n_trials || n < 3u) { return h; }
    var r = pcg(P.seed + t * 17u + 1u);
    var i0 = 0u;
    var i1 = 1u;
    var i2 = 2u;
    var found = false;
    for (var attempt = 0u; attempt < 24u; attempt++) {
        r = pcg(r);
        i0 = r % n;
        r = pcg(r);
        i1 = r % n;
        r = pcg(r);
        i2 = r % n;
        if (i0 == i1 || i0 == i2 || i1 == i2) { continue; }
        if (i0 >= n || i1 >= n || i2 >= n) { continue; }
        let c0 = corr[i0];
        let c1 = corr[i1];
        let c2 = corr[i2];
        if (c0.d == c1.d || c0.d == c2.d || c1.d == c2.d) { continue; }
        found = true;
        break;
    }
    if (!found) { return h; }
    let c0 = corr[i0];
    let c1 = corr[i1];
    let c2 = corr[i2];
    return fsc_solve(
        vec2f(kps[c0.q].x, kps[c0.q].y), vec2f(kps[c1.q].x, kps[c1.q].y), vec2f(kps[c2.q].x, kps[c2.q].y),
        vec2f(kps_b[c0.d].x, kps_b[c0.d].y), vec2f(kps_b[c1.d].x, kps_b[c1.d].y), vec2f(kps_b[c2.d].x, kps_b[c2.d].y));
}

// The fit's correspondences (corr[0..n_query)) as fsc_trial's points; counters[2] = their count.
@compute @workgroup_size(64)
fn pack_corr(@builtin(global_invocation_id) gid: vec3u) {
    let k = gid.x;
    if (k == 0u) { atomicStore(&counters[2], P.n_query); }
    if (k >= P.n_query) { return; }
    let c = corr[k];
    fsc_xy[k] = vec4f(kps[c.q].x, kps[c.q].y, kps_b[c.d].x, kps_b[c.d].y);
}

@compute @workgroup_size(64)
fn fsc_hyps_corr(@builtin(global_invocation_id) gid: vec3u) {
    let t = gid.x;
    if (t >= P.n_trials) { return; }
    let h = fsc_hyp_corr(t);
    fsc_store(t, h, 0u);
    if (h.ok) { fsc_list[atomicAdd(&counters[4], 1u)] = t; }
}

@compute @workgroup_size(8, 8)
fn warp_homography(@builtin(global_invocation_id) gid: vec3u) {
    // H is row-major 3×3 at affine[0..8], mapping moving → fixed (1-based xy).
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let pix = gid.y * P.w + gid.x;
    let H = mat3x3f(
        vec3f(affine[0], affine[3], affine[6]),
        vec3f(affine[1], affine[4], affine[7]),
        vec3f(affine[2], affine[5], affine[8]),
    );
    let det = determinant(H);
    if (abs(det) < 1e-12) {
        dst_f[pix] = store_t(0.0);
        return;
    }
    let Hi = inverse3(H);
    let q = Hi * vec3f(f32(gid.x) + 1.0, f32(gid.y) + 1.0, 1.0);
    if (abs(q.z) < 1e-8) {
        dst_f[pix] = store_t(0.0);
        return;
    }
    let fx = q.x / q.z - 1.0;
    let fy = q.y / q.z - 1.0;
    if (fx < 0.0 || fy < 0.0 || fx > f32(P.src_w - 1u) || fy > f32(P.src_h - 1u)) {
        dst_f[pix] = store_t(0.0);
        return;
    }
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let txp = fx - f32(x0);
    let typ = fy - f32(y0);
    let s00 = at_src(x0, y0);
    let s10 = at_src(x0 + 1, y0);
    let s01 = at_src(x0, y0 + 1);
    let s11 = at_src(x0 + 1, y0 + 1);
    dst_f[pix] = store_t(mix(mix(s00, s10, txp), mix(s01, s11, txp), typ));
}

@compute @workgroup_size(8, 8)
fn paste_rect(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let dx = bitcast<i32>(P.off_src);
    let dy = bitcast<i32>(P.off_dst);
    let sx = i32(gid.x) - dx;
    let sy = i32(gid.y) - dy;
    if (sx < 0 || sy < 0 || sx >= i32(P.src_w) || sy >= i32(P.src_h)) {
        return;
    }
    dst_f[gid.y * P.w + gid.x] = src_f[u32(sy) * P.src_w + u32(sx)];
}

@group(0) @binding(24) var<storage, read_write> clahe_h: array<atomic<u32>>;
@group(0) @binding(25) var<storage, read_write> clahe_c: array<f32>;

fn clahe_reflect(i: i32, n: i32, pad: i32) -> i32 {
    var x = i - pad;
    if (n <= 1) { return 0; }
    if (x < 0) { x = -x; }
    if (x >= n) { x = 2 * n - 2 - x; }
    return clamp(x, 0, n - 1);
}

@compute @workgroup_size(8, 8)
fn clahe_hist(@builtin(global_invocation_id) gid: vec3u) {
    let h_pad = P.n_trials;
    let w_pad = P.grid;
    let ph = P.h + h_pad;
    let pw = P.w + w_pad;
    if (gid.x >= pw || gid.y >= ph) { return; }
    let h_tile = max(P.src_w, 1u);
    let w_tile = max(P.src_h, 1u);
    let n_bins = max(P.seed, 1u);
    let w_grid = max(P.n_db, 1u);
    let ox = clahe_reflect(i32(gid.x), i32(P.w), i32(w_pad));
    let oy = clahe_reflect(i32(gid.y), i32(P.h), i32(h_pad));
    let v = clamp(load_t(src_f[u32(oy) * P.w + u32(ox)]), 0.0, 1.0);
    let bin = u32(clamp(v * f32(n_bins - 1u), 0.0, f32(n_bins - 1u)));
    let ty = gid.y / h_tile;
    let tx = gid.x / w_tile;
    let tile = ty * w_grid + tx;
    atomicAdd(&clahe_h[tile * n_bins + bin], 1u);
}

@compute @workgroup_size(16)
fn clahe_cdf(@builtin(global_invocation_id) gid: vec3u) {
    let n_tiles = P.n_query * P.n_db;
    if (gid.x >= n_tiles) { return; }
    let n_bins = max(P.seed, 1u);
    let voxels = max(P.src_w * P.src_h, 1u);
    let limit = max(P.tau * f32(voxels) / f32(n_bins), 1.0);
    var sumc = 0.0;
    for (var i = 0u; i < n_bins; i++) {
        sumc = sumc + min(f32(atomicLoad(&clahe_h[gid.x * n_bins + i])), limit);
    }
    let clipped = f32(voxels) - sumc;
    let nb = f32(n_bins);
    let residual = i32(clipped - nb * floor(clipped / nb));
    let redist = (clipped - f32(residual)) / nb;
    var acc = 0.0;
    let den = max(f32(voxels), 1.0);
    for (var i = 0u; i < n_bins; i++) {
        var c = min(f32(atomicLoad(&clahe_h[gid.x * n_bins + i])), limit) + redist;
        if (i32(i) < residual) { c = c + 1.0; }
        acc = acc + c;
        clahe_c[gid.x * n_bins + i] = clamp(acc / den, 0.0, 1.0);
    }
}

@compute @workgroup_size(8, 8)
fn clahe_apply(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let h_grid = max(P.n_query, 1u);
    let w_grid = max(P.n_db, 1u);
    let n_bins = max(P.seed, 1u);
    let v = clamp(load_t(src_f[gid.y * P.w + gid.x]), 0.0, 1.0);
    let gy = (f32(gid.y) + 0.5) / f32(P.h) * f32(h_grid - 1u);
    let gx = (f32(gid.x) + 0.5) / f32(P.w) * f32(w_grid - 1u);
    let y0 = u32(floor(gy));
    let x0 = u32(floor(gx));
    let y1 = min(y0 + 1u, h_grid - 1u);
    let x1 = min(x0 + 1u, w_grid - 1u);
    let fy = gy - f32(y0);
    let fx = gx - f32(x0);
    let bf = v * f32(n_bins - 1u);
    let b0 = u32(clamp(floor(bf), 0.0, f32(n_bins - 2u)));
    let fb = bf - f32(b0);
    let b1 = b0 + 1u;
    let stride = n_bins;
    let i00 = (y0 * w_grid + x0) * stride;
    let i10 = (y0 * w_grid + x1) * stride;
    let i01 = (y1 * w_grid + x0) * stride;
    let i11 = (y1 * w_grid + x1) * stride;
    let c00 = mix(clahe_c[i00 + b0], clahe_c[i00 + b1], fb);
    let c10 = mix(clahe_c[i10 + b0], clahe_c[i10 + b1], fb);
    let c01 = mix(clahe_c[i01 + b0], clahe_c[i01 + b1], fb);
    let c11 = mix(clahe_c[i11 + b0], clahe_c[i11 + b1], fb);
    let c0 = mix(c00, c10, fx);
    let c1 = mix(c01, c11, fx);
    dst_f[gid.y * P.w + gid.x] = store_t(mix(c0, c1, fy));
}

@group(0) @binding(26) var<storage, read_write> alt_f: array<f32>;
@group(0) @binding(27) var<storage, read_write> magb: array<atomic<u32>>;
struct Corr { q: u32, d: u32 }
@group(0) @binding(28) var<storage, read_write> corr: array<Corr>;
// Packed (px, py, qx, qy) for the points match_ok already accepts. Built once per pair.
@group(0) @binding(30) var<storage, read_write> fsc_xy: array<vec4<f32>>;
// per query of the slice (pack_fsc): its match when match_ok, else 0xffffffff, so the trial
// sampler reads one small array instead of kps, match_j and kps_b per draw
@group(0) @binding(32) var<storage, read_write> fsc_ok: array<u32>;
const MAX_CORR: u32 = 65536u;
const MAX_KP: u32 = 65536u;
var<workgroup> mag_local: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn texel_to_f32(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    let n = P.w;
    if (gid.x >= n) { return; }
    alt_f[P.off_dst + gid.x] = load_t(src_f[P.off_src + gid.x]);
}

@compute @workgroup_size(256)
fn f32_to_texel(@builtin(global_invocation_id) gid2: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let gid = vec3u(flat_index(gid2, nwg), 0u, gid2.z);
    let n = P.w;
    if (gid.x >= n) { return; }
    dst_f[P.off_dst + gid.x] = store_t(alt_f[P.off_src + gid.x]);
}
