// POS-GIFT (Hou, Liu & Zhang 2024, Information Fusion 102:102027), following the authors'
// released MATLAB (POS_GIFT.p): phasecong3 (4 scales, 6 orientations, minWaveLength 3,
// mult 1.6, sigmaOnf 0.75, k 1, g 3), keypoints on the normalized phase-congruency sum,
// the GIFT descriptor (3 rings × 12 directions + centre, 6 orientation channels = 222,
// stored padded to 224) and the POS guided re-matching. Matching and FSC run in
// gls_mift.wgsl (compiled with DES_DIM 224). Those are the defaults: the structure constants
// NS, NO, NDIR (= 2·NO), NRING, NPT, DP, DQ and KNN are replaced at compile time (PosGift).

const PI: f32 = 3.141592653589793;
const NS: u32 = 4u;        // Log-Gabor scales
const NO: u32 = 6u;        // orientations
const NDIR: u32 = 12u;     // sampling directions
const NRING: u32 = 3u;
const NPT: u32 = 37u;      // NDIR · NRING + centre
const DP: u32 = 224u;      // stored descriptor length (222 + 2 zeros)
const DQ: u32 = 56u;       // 8-bit copy: words per descriptor
// Self-similarity values appended to the descriptor (0, NDIR·NRING: each sampled point vs the
// centre, or 2·NDIR·NRING: also vs its angular neighbour), and the described length.
const SSN: u32 = 0u;
const DLEN: u32 = NPT * NO + SSN;
const HBINS: u32 = 4096u;  // log2 histogram of the scale-1 amplitude (median noise estimate)
const HLO: f32 = -30.0;    // log2 of the lowest bin edge; 64 bins per octave

struct PG {
    w: u32, h: u32, nn: u32, o: u32,
    sw: u32, sh: u32, axis: u32, hsize: u32,
    kp_offset: u32, n_kp: u32, ring: u32, mode: u32,
    ma: u32, pad0: u32, n_b: u32, maxd: u32,
    scale: f32, rscale: f32, mult: f32, sigma_onf: f32,
    min_wl: f32, k: f32, cutoff: f32, g: f32,
    gsig: f32, thr2: f32, cth: f32, sth: f32,
    ssw: f32, dflags: u32, dpow: f32, pad4: u32,
}

struct Kp { x: f32, y: f32, score: f32, pad: f32 }

@group(0) @binding(0) var<uniform> P: PG;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> spec: array<f32>;
@group(0) @binding(4) var<storage, read_write> planes: array<f32>;
@group(0) @binding(5) var<storage, read_write> cs: array<f32>;      // NO × w × h: Σ_s |EO|
@group(0) @binding(6) var<storage, read_write> en: array<f32>;      // NO × w × h: energy before the noise threshold
@group(0) @binding(7) var<storage, read_write> wt: array<f32>;      // NO × w × h: weight / Σ_s |EO|
@group(0) @binding(8) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> tv: array<f32>;      // NO noise thresholds
@group(0) @binding(10) var<storage, read_write> mm: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> vecs: array<f32>;
@group(0) @binding(12) var<storage, read_write> sig: array<f32>;    // NO spectral norms
@group(0) @binding(13) var<storage, read_write> gm: array<f32>;     // NRING × NO × w × h ring maps
@group(0) @binding(14) var<storage, read> offs: array<i32>;         // NRING × NDIR × (dx, dy)
@group(0) @binding(15) var<storage, read_write> kps: array<Kp>;
@group(0) @binding(16) var<storage, read_write> des: array<f32>;
@group(0) @binding(17) var<storage, read_write> ori: array<u32>;
@group(0) @binding(18) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(19) var<storage, read_write> desq: array<u32>;
@group(0) @binding(20) var<storage, read_write> fmap_t: array<Texel>;
@group(0) @binding(21) var<storage, read_write> dsc: array<f32>;
@group(0) @binding(22) var<storage, read> kpb: array<vec4f>;
@group(0) @binding(23) var<storage, read> desb: array<f32>;
@group(0) @binding(24) var<storage, read> kpp: array<vec4f>;
@group(0) @binding(25) var<storage, read> desp: array<f32>;
@group(0) @binding(26) var<storage, read_write> res: array<vec4f>;
@group(0) @binding(28) var<storage, read_write> pcm: array<f32>;      // phase-congruency sums (pg_pcacc)

var<workgroup> red: array<f32, 256>;
var<workgroup> redu: u32;

// MATLAB round: halves away from zero.
fn mround(x: f32) -> i32 { return i32(sign(x) * floor(abs(x) + 0.5)); }

// ── Pyramid: MATLAB imresize(·, scale, 'bilinear') with antialiasing, one axis per pass ──
// axis 0 resamples rows (length sw → w, h rows); axis 1 columns (sh → h, w columns).
// Out-of-range taps mirror (MATLAB's [1:L, L:-1:1] index extension).
@compute @workgroup_size(8, 8)
fn pg_resize(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    var L = P.sw;
    var xo = gid.x;
    if (P.axis == 1u) { L = P.sh; xo = gid.y; }
    let sc = P.rscale;
    let kw = 2.0 / sc;
    let u = (f32(xo) + 1.0) / sc + 0.5 * (1.0 - 1.0 / sc);
    let left = i32(floor(u - kw / 2.0));
    let ntap = i32(ceil(kw)) + 2;
    var acc = 0.0;
    var wsum = 0.0;
    for (var t = 0; t < ntap; t++) {
        let idx = left + t;                       // 1-based
        let d = sc * (u - f32(idx));
        let wgt = sc * max(0.0, 1.0 - abs(d));
        if (wgt == 0.0) { continue; }
        var m = (idx - 1) % (2 * i32(L));
        if (m < 0) { m += 2 * i32(L); }
        if (m >= i32(L)) { m = 2 * i32(L) - 1 - m; }
        var v = 0.0;
        if (P.axis == 0u) { v = src[gid.y * P.sw + u32(m)]; } else { v = src[u32(m) * P.sw + gid.x]; }
        acc += wgt * v;
        wsum += wgt;
    }
    dst[gid.y * P.w + gid.x] = acc / wsum;
}

// ── Phase congruency (Kovesi phasecong3, noiseMethod −1) ──
// The w × h level sits in the top-left of an nn × nn transform, extended by half-sample
// symmetry (nn = w = h needs no extension: the MATLAB grid exactly). Intensities × 255
// (phasecong3 ran on the uint8 values).
@compute @workgroup_size(8, 8)
fn pg_pad(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.nn || gid.y >= P.nn) { return; }
    var sx = gid.x % (2u * P.w);
    if (sx >= P.w) { sx = 2u * P.w - 1u - sx; }
    var sy = gid.y % (2u * P.h);
    if (sy >= P.h) { sy = 2u * P.h - 1u - sy; }
    let o = (gid.y * P.nn + gid.x) * 2u;
    spec[o] = 255.0 * src[sy * P.w + sx];
    spec[o + 1u] = 0.0;
}

// Normalized frequency of FFT index k (MATLAB: ifftshift of the centred range, / (n − 1) for odd n).
fn freq(k: u32, n: u32) -> f32 {
    var kk = f32(k);
    if (k >= (n + 1u) / 2u) { kk = f32(k) - f32(n); }
    return kk / f32(select(n, n - 1u, (n & 1u) == 1u));
}

// planes[s] = spectrum × logGabor_s × spread_o, for the four scales (z).
@compute @workgroup_size(8, 8)
fn pg_filter(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.nn || gid.y >= P.nn) { return; }
    let s = gid.z;
    let fx = freq(gid.x, P.nn);
    let fy = freq(gid.y, P.nn);
    var radius = sqrt(fx * fx + fy * fy);
    let dc = gid.x == 0u && gid.y == 0u;
    if (dc) { radius = 1.0; }
    let theta = atan2(-fy, fx);
    let lp = 1.0 / (1.0 + pow(radius / 0.45, 30.0));
    let fo = 1.0 / (P.min_wl * pow(P.mult, f32(s)));
    let ls = log(P.sigma_onf);
    let lr = log(radius / fo);
    var lg = exp(-(lr * lr) / (2.0 * ls * ls)) * lp;
    if (dc) { lg = 0.0; }
    let angl = f32(P.o) * PI / f32(NO);
    let st = sin(theta);
    let ct = cos(theta);
    let dsn = st * cos(angl) - ct * sin(angl);
    let dcs = ct * cos(angl) + st * sin(angl);
    let dtheta = min(abs(atan2(dsn, dcs)) * f32(NO) / 2.0, PI);
    let spread = (cos(dtheta) + 1.0) / 2.0;
    let f = lg * spread;
    let i = (gid.y * P.nn + gid.x) * 2u;
    let o = s * P.nn * P.nn * 2u + i;
    planes[o] = spec[i] * f;
    planes[o + 1u] = spec[i + 1u] * f;
}

// One orientation from its four inverse transforms: Σ|EO| (the descriptor's LG feature),
// the energy before the noise threshold, weight / Σ|EO|, and the scale-1 amplitude histogram.
@compute @workgroup_size(8, 8)
fn pg_orient(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let eps = 0.0001;
    let q = (gid.y * P.nn + gid.x) * 2u;
    let plane = P.nn * P.nn * 2u;
    var E: array<f32, NS>;
    var O: array<f32, NS>;
    var sumAn = 0.0;
    var sumE = 0.0;
    var sumO = 0.0;
    var maxAn = 0.0;
    var an1 = 0.0;
    for (var s = 0u; s < NS; s++) {
        E[s] = planes[s * plane + q];
        O[s] = planes[s * plane + q + 1u];
        let an = sqrt(E[s] * E[s] + O[s] * O[s]);
        sumAn += an;
        sumE += E[s];
        sumO += O[s];
        if (s == 0u) { maxAn = an; an1 = an; } else { maxAn = max(maxAn, an); }
    }
    let xe = sqrt(sumE * sumE + sumO * sumO) + eps;
    let me = sumE / xe;
    let mo = sumO / xe;
    var energy = 0.0;
    for (var s = 0u; s < NS; s++) {
        energy += E[s] * me + O[s] * mo - abs(E[s] * mo - O[s] * me);
    }
    let width = (sumAn / (maxAn + eps) - 1.0) / f32(NS - 1u);
    let weight = 1.0 / (1.0 + exp((P.cutoff - width) * P.g));
    let p1 = gid.y * P.w + gid.x;
    cs[P.o * P.w * P.h + p1] = sumAn;
    // this orientation's energy and weight (one plane each; pg_pcacc folds them in)
    en[p1] = energy;
    wt[p1] = select(0.0, weight / sumAn, sumAn > 0.0);
    var b = 0u;
    if (an1 > 0.0) { b = u32(clamp((log2(an1) - HLO) * 64.0, 0.0, f32(HBINS - 1u))); }
    atomicAdd(&hist[P.o * HBINS + b], 1u);
}

// Noise threshold per orientation (workgroup o): the median of the scale-1 amplitude
// (interpolated within its 1/64-octave bin), tau = median / √ln 4, T = mean + k·σ of the
// Rayleigh noise energy.
//
// The scan is the serial one (f32 running sum, first bin reaching half), split over 256
// invocations of 16 bins each: below 2^24 pixels every partial sum is an integer the f32 sum
// holds exactly, so each invocation starts from its exact prefix and finds the same bin.
var<workgroup> tau_bin: atomic<u32>;
var<workgroup> tau_med: f32;

@compute @workgroup_size(256)
fn pg_tau(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let o = wid.x + P.o;
    let half = f32(P.w * P.h) * 0.5;
    if (lid == 0u) {
        atomicStore(&tau_bin, 0xffffffffu);
        tau_med = 0.0;
    }
    if (P.w * P.h >= 16777216u) {
        // large images: the serial scan (its f32 sum rounds)
        if (lid == 0u) {
            var acc = 0.0;
            for (var b = 0u; b < HBINS; b++) {
                let c = f32(atomicLoad(&hist[o * HBINS + b]));
                if (acc + c >= half && c > 0.0) {
                    tau_med = exp2(HLO + (f32(b) + (half - acc) / c) / 64.0);
                    break;
                }
                acc += c;
            }
        }
    } else {
        const PER = HBINS / 256u;
        var own = 0u;
        for (var k = 0u; k < PER; k++) { own += atomicLoad(&hist[o * HBINS + lid * PER + k]); }
        scan[lid] = own;
        workgroupBarrier();
        // inclusive Hillis–Steele scan of the invocation sums
        for (var d = 1u; d < 256u; d = d << 1u) {
            var v = scan[lid];
            if (lid >= d) { v += scan[lid - d]; }
            workgroupBarrier();
            scan[lid] = v;
            workgroupBarrier();
        }
        var acc = f32(scan[lid] - own);
        var hit = 0xffffffffu;
        var hacc = 0.0;
        for (var k = 0u; k < PER; k++) {
            let b = lid * PER + k;
            let c = f32(atomicLoad(&hist[o * HBINS + b]));
            if (acc + c >= half && c > 0.0) {
                hit = b;
                hacc = acc;
                break;
            }
            acc += c;
        }
        atomicMin(&tau_bin, hit);
        workgroupBarrier();
        if (hit != 0xffffffffu && hit == atomicLoad(&tau_bin)) {
            let c = f32(atomicLoad(&hist[o * HBINS + hit]));
            tau_med = exp2(HLO + (f32(hit) + (half - hacc) / c) / 64.0);
        }
    }
    workgroupBarrier();
    if (lid != 0u) { return; }
    let med = tau_med;
    let tau = med / sqrt(log(4.0));
    let total_tau = tau * (1.0 - pow(1.0 / P.mult, f32(NS))) / (1.0 - 1.0 / P.mult);
    tv[o] = total_tau * sqrt(PI / 2.0) + P.k * total_tau * sqrt((4.0 - PI) / 2.0);
}

// Phase-congruency sum over orientations, accumulated as each orientation is done (its
// energy, weight and noise threshold) into pcm: the sum, and with a moment detection map
// Σ cx², Σ cy², Σ cx·cy (planes 1–3), in orientation order as one pass over them would.
@compute @workgroup_size(8, 8)
fn pg_pcacc(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let p = gid.y * P.w + gid.x;
    let n = P.w * P.h;
    let o = P.o;
    let po = wt[p] * max(en[p] - tv[o], 0.0);
    let first = o == 0u;
    pcm[p] = select(pcm[p] + po, 0.0 + po, first);
    if (((P.dflags >> 1u) & 3u) == 0u) { return; }
    // cos, sin of the orientation's angle f32(o)·π/NO, correctly rounded (host), as a loop over
    // the constant orientations folds them
    let cx = po * P.cth;
    let cy = po * P.sth;
    pcm[n + p] = select(pcm[n + p], 0.0, first) + cx * cx;
    pcm[2u * n + p] = select(pcm[2u * n + p], 0.0, first) + cy * cy;
    pcm[3u * n + p] = select(pcm[3u * n + p], 0.0, first) + cx * cy;
}

// The phase-congruency map from the accumulated sums, with its range (non-negative, so the
// bit patterns order like the values).
@compute @workgroup_size(8, 8)
fn pg_pcsum(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let p = gid.y * P.w + gid.x;
    let n = P.w * P.h;
    var pc = pcm[p];
    var cx2 = 0.0;
    var cy2 = 0.0;
    var cxy = 0.0;
    if (((P.dflags >> 1u) & 3u) != 0u) {
        cx2 = pcm[n + p];
        cy2 = pcm[2u * n + p];
        cxy = pcm[3u * n + p];
    }
    // Detection map (P.dflags bits 1–2): 0 the phase-congruency sum; 1 the minimum moment of
    // phase congruency (corners); 2 the maximum moment (edges and corners). phasecong3's moments.
    let dm = (P.dflags >> 1u) & 3u;
    if (dm != 0u) {
        cx2 = cx2 / (f32(NO) / 2.0);
        cy2 = cy2 / (f32(NO) / 2.0);
        cxy = 4.0 * cxy / f32(NO);
        let den = sqrt(cxy * cxy + (cx2 - cy2) * (cx2 - cy2)) + 0.0001;
        pc = select((cy2 + cx2 + den) / 2.0, max((cy2 + cx2 - den) / 2.0, 0.0), dm == 1u);
    }
    dst[p] = pc;
    atomicMin(&mm[0], bitcast<u32>(pc));
    atomicMax(&mm[1], bitcast<u32>(pc));
}

// Min-max normalized phase congruency into the detector's feature map.
@compute @workgroup_size(8, 8)
fn pg_fmap(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let p = gid.y * P.w + gid.x;
    let mn = bitcast<f32>(atomicLoad(&mm[0]));
    let mx = bitcast<f32>(atomicLoad(&mm[1]));
    fmap_t[p] = Texel((src[p] - mn) / max(mx - mn, 1e-30));
}

// ── Spectral norm of each Σ|EO| map (MATLAB norm(CS(:,:,j)), mode 2), by power iteration ──
// vecs per orientation: v (maxd), u (maxd), v' (maxd).
@compute @workgroup_size(64)
fn pg_pw_init(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w) { return; }
    vecs[gid.y * 3u * P.maxd + gid.x] = 1.0 / sqrt(f32(P.w));
}

// u = A v: workgroup (row y, orientation z).
@compute @workgroup_size(256)
fn pg_pw_rows(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let y = wid.x;
    let o = wid.y;
    let base = o * 3u * P.maxd;
    var a = 0.0;
    for (var x = lid; x < P.w; x += 256u) { a += cs[o * P.w * P.h + y * P.w + x] * vecs[base + x]; }
    red[lid] = a;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { red[lid] += red[lid + s]; }
        workgroupBarrier();
    }
    if (lid == 0u) { vecs[base + P.maxd + y] = red[0]; }
}

// v' = Aᵀ u: invocation (column x, orientation y).
@compute @workgroup_size(64)
fn pg_pw_cols(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w) { return; }
    let o = gid.y;
    let base = o * 3u * P.maxd;
    var a = 0.0;
    let c0 = o * P.w * P.h + gid.x;
    let u0 = base + P.maxd;
    var y = 0u;
    // loads issued eight rows ahead, products added in row order (the serial sum)
    for (; y + 8u <= P.h; y += 8u) {
        let p0 = cs[c0 + y * P.w] * vecs[u0 + y];
        let p1 = cs[c0 + (y + 1u) * P.w] * vecs[u0 + y + 1u];
        let p2 = cs[c0 + (y + 2u) * P.w] * vecs[u0 + y + 2u];
        let p3 = cs[c0 + (y + 3u) * P.w] * vecs[u0 + y + 3u];
        let p4 = cs[c0 + (y + 4u) * P.w] * vecs[u0 + y + 4u];
        let p5 = cs[c0 + (y + 5u) * P.w] * vecs[u0 + y + 5u];
        let p6 = cs[c0 + (y + 6u) * P.w] * vecs[u0 + y + 6u];
        let p7 = cs[c0 + (y + 7u) * P.w] * vecs[u0 + y + 7u];
        a += p0;
        a += p1;
        a += p2;
        a += p3;
        a += p4;
        a += p5;
        a += p6;
        a += p7;
    }
    for (; y < P.h; y++) { a += cs[c0 + y * P.w] * vecs[u0 + y]; }
    vecs[base + 2u * P.maxd + gid.x] = a;
}

// v = v' / |v'|, σ = √|v'| (|AᵀA v| for a unit v converges to σ²): workgroup o.
@compute @workgroup_size(256)
fn pg_pw_norm(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let base = wid.x * 3u * P.maxd;
    var a = 0.0;
    for (var x = lid; x < P.w; x += 256u) {
        let v = vecs[base + 2u * P.maxd + x];
        a += v * v;
    }
    red[lid] = a;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { red[lid] += red[lid + s]; }
        workgroupBarrier();
    }
    let nrm = sqrt(red[0]);
    for (var x = lid; x < P.w; x += 256u) { vecs[base + x] = vecs[base + 2u * P.maxd + x] / max(nrm, 1e-30); }
    if (lid == 0u) { sig[wid.x] = sqrt(nrm); }
}

// ── Ring maps: imfilter(CS, fspecial('gaussian', hsize, gsig)), zero boundary ──
// An even hsize puts the kernel's centre at element floor((hsize + 1) / 2), as imfilter does.
fn gauss_w(k: u32) -> f32 {
    let x = f32(k) - f32(P.hsize - 1u) / 2.0;
    return exp(-(x * x) / (2.0 * P.gsig * P.gsig));
}

// The kernel's weights (gauss_w) and their serial sum, computed once per workgroup.
const GW_MAX: u32 = 512u;
var<workgroup> gw: array<f32, GW_MAX>;
var<workgroup> gw_sum: f32;

fn gauss_table(lid: u32) {
    for (var k = lid; k < min(P.hsize, GW_MAX); k += 64u) { gw[k] = gauss_w(k); }
    workgroupBarrier();
    if (lid == 0u) {
        var s = 0.0;
        for (var k = 0u; k < P.hsize; k++) { s += gw_at(k); }
        gw_sum = s;
    }
    workgroupBarrier();
}

fn gw_at(k: u32) -> f32 {
    if (k < GW_MAX) { return gw[k]; }
    return gauss_w(k);
}

// Horizontal pass of all NO channels (z); mode 2 divides channel o by its spectral norm.
@compute @workgroup_size(8, 8)
fn pg_gauss_h(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
    gauss_table(lid);
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let o = P.o;
    let c = i32((P.hsize + 1u) / 2u) - 1;   // 0-based centre
    var acc = 0.0;
    for (var k = 0u; k < P.hsize; k++) {
        let x = i32(gid.x) + i32(k) - c;
        if (x < 0 || x >= i32(P.w)) { continue; }
        acc += gw_at(k) * cs[o * P.w * P.h + gid.y * P.w + u32(x)];
    }
    var sc = 1.0 / gw_sum;
    if (P.mode == 2u) { sc = sc / sig[o]; }
    dst[gid.y * P.w + gid.x] = acc * sc;
}

@compute @workgroup_size(8, 8)
fn pg_gauss_v(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
    gauss_table(lid);
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let o = P.o;
    let c = i32((P.hsize + 1u) / 2u) - 1;
    var acc = 0.0;
    for (var k = 0u; k < P.hsize; k++) {
        let y = i32(gid.y) + i32(k) - c;
        if (y < 0 || y >= i32(P.h)) { continue; }
        acc += gw_at(k) * src[u32(y) * P.w + gid.x];
    }
    gm[(P.ring * NO + o) * P.w * P.h + gid.y * P.w + gid.x] = acc / gw_sum;
}

// ── Descriptor ──
// Sample (ring r, direction a, channel o) of the keypoint at 1-based (x, y); a = NDIR: centre.
fn samp(x: i32, y: i32, r: u32, a: u32, o: u32) -> f32 {
    var px = x;
    var py = y;
    if (a < NDIR) {
        px += offs[(r * NDIR + a) * 2u];
        py += offs[(r * NDIR + a) * 2u + 1u];
    }
    return gm[(r * NO + o) * P.w * P.h + u32(py - 1) * P.w + u32(px - 1)];
}

// Staged descriptors (P.dflags bit 4; modes 0, 1, 4): the ring maps are made one ring at a time
// into gm's first ring slot, and after each, pg_sample stores every keypoint's points on that
// ring (ring 0: also the centre) in its descriptor slot, per point unit length in mode 4, as
// desc_core would sample them; pg_describe finishes from the slot. gm then holds one ring.
const STAGED: u32 = 16u;

fn staged() -> bool {
    return (P.dflags & STAGED) != 0u;
}

@compute @workgroup_size(64)
fn pg_sample(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= n_kp()) { return; }
    let slot = kp_off() + i;
    let kp = kps[slot];
    let x = mround(kp.x);
    let y = mround(kp.y);
    if (!inside(x, y)) { return; }
    let r = P.ring;
    let per_point = P.mode == 4u;
    for (var a = 0u; a <= NDIR; a++) {
        if (a == NDIR && r != 0u) { continue; }
        let pt = select(NDIR * NRING, a * NRING + r, a < NDIR);
        var px = x;
        var py = y;
        if (a < NDIR) {
            px += offs[(r * NDIR + a) * 2u];
            py += offs[(r * NDIR + a) * 2u + 1u];
        }
        let at = u32(py - 1) * P.w + u32(px - 1);
        var v: array<f32, NO>;
        var n2 = 0.0;
        for (var o = 0u; o < NO; o++) {
            v[o] = gm[o * P.w * P.h + at];
            n2 += v[o] * v[o];
        }
        if (per_point && n2 > 0.0) {
            let s = 1.0 / sqrt(n2);
            for (var o = 0u; o < NO; o++) { v[o] *= s; }
        }
        for (var o = 0u; o < NO; o++) { des[slot * DP + pt * NO + o] = v[o]; }
    }
}

// Outermost ring inside the image (MATLAB skips keypoints whose outer ring leaves it).
fn inside(x: i32, y: i32) -> bool {
    let r3 = offs[((NRING - 1u) * NDIR) * 2u];   // direction 0 of the outer ring: dx = −round(R)
    let R = -r3;
    return x - R >= 1 && x + R <= i32(P.w) && y - R >= 1 && y + R <= i32(P.h);
}

// P.kp_offset, or with 0xffffffff the running offset of the level in cnt[6] (gls_mift.wgsl kp_off).
fn kp_off() -> u32 {
    if (P.kp_offset == 0xffffffffu) { return atomicLoad(&cnt[6]); }
    return P.kp_offset;
}


// Modes: 0 plain GIFT, 4 plain GIFT with unit sampled points, and 2 paper orientation (detected
// keypoints, count in cnt[0]); 1 POS re-description of a given list (P.n_kp keypoints,
// full-resolution coordinates).
fn n_kp() -> u32 {
    if (P.mode == 1u) { return P.n_kp; }
    return atomicLoad(&cnt[0]);
}

// Mode 2, pass A: primary direction(s) from the GFP norms (paper Eq. 9): the direction whose
// three sampled points have the largest summed norm, and the runner-up when it reaches 0.8 of
// it (MATLAB maxk). ori = a1 | a2 << 4 | second << 8, or 0xffffffff for a dropped keypoint.
@compute @workgroup_size(64)
fn pg_orient_kp(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= n_kp()) { return; }
    let slot = kp_off() + i;
    let kp = kps[slot];
    let x = mround(kp.x);
    let y = mround(kp.y);
    if (!inside(x, y)) {
        ori[slot] = 0xffffffffu;
        return;
    }
    var v1 = -1.0;
    var v2 = -1.0;
    var a1 = 0u;
    var a2 = 0u;
    for (var a = 0u; a < NDIR; a++) {
        var e = 0.0;
        for (var r = 0u; r < NRING; r++) {
            var n2 = 0.0;
            for (var o = 0u; o < NO; o++) {
                let v = samp(x, y, r, a, o);
                n2 += v * v;
            }
            e += sqrt(n2);
        }
        if (e > v1) {
            v2 = v1;
            a2 = a1;
            v1 = e;
            a1 = a;
        } else if (e > v2) {
            v2 = e;
            a2 = a;
        }
    }
    ori[slot] = a1 | (a2 << 4u) | (select(0u, 1u, v1 * 0.8 < v2) << 8u);
}

// One workgroup: the k-th keypoint with a second direction (in keypoint order) gets extra
// slot kp_offset + n + k (k stored in ori bits 9+); count to cnt[2].
var<workgroup> scan: array<u32, 256>;

@compute @workgroup_size(256)
fn pg_compact(@builtin(local_invocation_index) lid: u32) {
    let n = atomicLoad(&cnt[0]);
    let chunk = (n + 255u) / 256u;
    let i0 = min(lid * chunk, n);
    let i1 = min(i0 + chunk, n);
    var local = 0u;
    for (var i = i0; i < i1; i++) {
        let v = ori[kp_off() + i];
        if (v != 0xffffffffu && (v & 256u) != 0u) { local++; }
    }
    scan[lid] = local;
    workgroupBarrier();
    if (lid == 0u) {
        var t = 0u;
        for (var k = 0u; k < 256u; k++) {
            let c = scan[k];
            scan[k] = t;
            t += c;
        }
        atomicStore(&cnt[2], t);
    }
    workgroupBarrier();
    var k = scan[lid];
    for (var i = i0; i < i1; i++) {
        let s = kp_off() + i;
        let v = ori[s];
        if (v == 0xffffffffu || (v & 256u) == 0u) { continue; }
        ori[s] = (v & 511u) | (k << 9u);
        k++;
    }
}

// Descriptor of the keypoint at (x, y) for primary direction ma, into slot. Element
// (direction a', ring r, channel c) = D[(c − ma) mod 6][(a' + ma) mod 12][r], the centre last,
// column-major like MATLAB's RIFT_des(:). Mode 2 first scales every sampled point's 6-vector
// to unit length; all modes then scale the whole to unit length. Also the 8-bit copy.
fn cos6(D: ptr<function, DescD>, a: u32, b: u32) -> f32 {
    var ab = 0.0;
    var aa = 0.0;
    var bb = 0.0;
    for (var o = 0u; o < NO; o++) {
        let x = (*D)[a * NO + o];
        let y = (*D)[b * NO + o];
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    return select(0.0, ab / sqrt(aa * bb), aa > 0.0 && bb > 0.0);
}

// Element e of the self-similarity block for direction a, ring r (block 0: vs the centre,
// block 1: vs the next direction on the same ring).
fn ss_val(D: ptr<function, DescD>, blk: u32, a: u32, r: u32) -> f32 {
    let pt = a * NRING + r;
    if (blk == 0u) { return cos6(D, pt, NDIR * NRING); }
    return cos6(D, pt, ((a + 1u) % NDIR) * NRING + r);
}

// P.dpow ≠ 1: element-wise power of the orientation part (0.5: Hellinger; above 1 sharpens
// each sampled point's dominant orientations), then unit length. With SSN > 0 the self-similarity block (unit length) is appended with weight
// √ssw against √(1 − ssw) for the orientation part.
struct DescScale { sc: f32, ssc: f32 }
alias DescD = array<f32, NPT * NO>;
alias DescS = array<f32, max(SSN, 1u)>;

// The sampled vectors of (x, y) before orientation: D (per point unit length when per_point,
// then the power), S (self-similarity) and their scales. The orientation only re-indexes them
// (desc_at), so both descriptors of a keypoint share one sampling pass.
fn desc_core(x: i32, y: i32, per_point: bool, sslot: u32, D: ptr<function, DescD>, S: ptr<function, DescS>) -> DescScale {
    if (staged() && sslot != 0xffffffffu) {
        // sampled (and per point normalized) by pg_sample into the keypoint's slot
        for (var k = 0u; k < NPT * NO; k++) { (*D)[k] = des[sslot * DP + k]; }
    }
    for (var pt = 0u; pt < NPT; pt++) {
        if (staged() && sslot != 0xffffffffu) { break; }
        var a = NDIR;
        var r = 0u;
        if (pt < NDIR * NRING) {
            a = pt / NRING;
            r = pt % NRING;
        }
        var n2 = 0.0;
        for (var o = 0u; o < NO; o++) {
            let v = samp(x, y, r, a, o);
            (*D)[pt * NO + o] = v;
            n2 += v * v;
        }
        if (per_point && n2 > 0.0) {
            let s = 1.0 / sqrt(n2);
            for (var o = 0u; o < NO; o++) { (*D)[pt * NO + o] *= s; }
        }
    }
    // Self-similarity, from the vectors before any square root; its norm.
    var ssn2 = 0.0;
    for (var b = 0u; b < SSN / (NDIR * NRING); b++) {
        for (var q = 0u; q < NDIR * NRING; q++) {
            let v = ss_val(D, b, q / NRING, q % NRING);
            (*S)[b * NDIR * NRING + q] = v;
            ssn2 += v * v;
        }
    }
    if (P.dpow != 1.0) {
        for (var k = 0u; k < NPT * NO; k++) { (*D)[k] = pow(max((*D)[k], 0.0), P.dpow); }
    }
    var tot = 0.0;
    for (var k = 0u; k < NPT * NO; k++) { tot += (*D)[k] * (*D)[k]; }
    var sc = select(1.0, 1.0 / sqrt(tot), tot > 0.0);
    var ssc = 0.0;
    if (SSN > 0u) {
        sc *= sqrt(1.0 - P.ssw);
        ssc = select(0.0, sqrt(P.ssw) / sqrt(ssn2), ssn2 > 0.0);
    }
    return DescScale(sc, ssc);
}

// Element k of the descriptor turned by ma directions (0 past DLEN).
fn desc_at(k: u32, ma: u32, D: ptr<function, DescD>, S: ptr<function, DescS>, ds: DescScale) -> f32 {
    if (k < NPT * NO) {
        let pt = k / NO;
        let c = k % NO;
        var spt = NDIR * NRING;
        if (pt < NDIR * NRING) { spt = (((pt / NRING) + ma) % NDIR) * NRING + pt % NRING; }
        return (*D)[spt * NO + (c + NO - ma % NO) % NO] * ds.sc;
    }
    if (k < DLEN) {
        let bq = k - NPT * NO;
        let bb = bq / (NDIR * NRING);
        let q = bq % (NDIR * NRING);
        return (*S)[bb * NDIR * NRING + (((q / NRING) + ma) % NDIR) * NRING + q % NRING] * ds.ssc;
    }
    return 0.0;
}

// The descriptor of (x, y) in direction ma, with the SSN block (each ring point's dominant
// orientations), unit length.
fn build_desc(x: i32, y: i32, ma: u32, per_point: bool) -> array<f32, DP> {
    var D: DescD;
    var S: DescS;
    let ds = desc_core(x, y, per_point, 0xffffffffu, &D, &S);
    var out: array<f32, DP>;
    for (var k = 0u; k < DP; k++) { out[k] = desc_at(k, ma, &D, &S, ds); }
    return out;
}

// Descriptors of (x, y) in directions mas[j] into slots[j] (j < count), with their 8-bit copies.
// Each slot's direction goes to ori[DIR_REC + slot], except for POS's re-described copies (mode 1).
fn write_descs(slots: vec2u, mas: vec2u, count: u32, x: i32, y: i32, per_point: bool) {
    var D: DescD;
    var S: DescS;
    let ds = desc_core(x, y, per_point, slots[0], &D, &S);
    for (var j = 0u; j < count; j++) {
        let slot = slots[j];
        let ma = mas[j];
        let base = slot * DP;
        var mx = 0.0;
        for (var k = 0u; k < DP; k++) {
            let v = desc_at(k, ma, &D, &S, ds);
            des[base + k] = v;
            mx = max(mx, v);
        }
        dsc[slot] = mx / 255.0;
        if (P.mode != 1u) { ori[DIR_REC + slot] = ma; }
        let inv = select(0.0, 255.0 / mx, mx > 0.0);
        for (var w = 0u; w < DQ; w++) {
            var word = 0u;
            for (var e = 0u; e < 4u; e++) {
                let q = u32(round(desc_at(w * 4u + e, ma, &D, &S, ds) * inv));
                word = word | (min(q, 255u) << (8u * e));
            }
            desq[slot * DQ + w] = word;
        }
    }
}

// Descriptor of (x, y) into slot, and its 8-bit copy.
fn write_desc(slot: u32, x: i32, y: i32, ma: u32, per_point: bool) {
    write_descs(vec2u(slot, 0u), vec2u(ma, 0u), 1u, x, y, per_point);
}

// Pass B. Mode 2: the primary descriptor in the keypoint's slot, the second in its extra
// slot. Modes 0 and 1: direction P.ma for every keypoint. Score 0 marks a keypoint whose
// outer ring leaves the image.
// Coordinates become full resolution (× scale) here, as in describe (gls_mift.wgsl).
@compute @workgroup_size(64)
fn pg_describe(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let n = n_kp();
    if (i >= n) { return; }
    let slot = kp_off() + i;
    let kp = kps[slot];
    let x = mround(kp.x);
    let y = mround(kp.y);
    if (P.mode != 2u) {
        if (!inside(x, y)) {
            kps[slot].score = 0.0;
            return;
        }
        write_desc(slot, x, y, P.ma, P.mode == 4u);
        kps[slot] = Kp(kp.x * P.scale, kp.y * P.scale, kp.score, P.scale);
        return;
    }
    let v = ori[slot];
    if (v == 0xffffffffu) {
        kps[slot].score = 0.0;
        return;
    }
    let full = Kp(kp.x * P.scale, kp.y * P.scale, kp.score, P.scale);
    let two = (v & 256u) != 0u;
    let s2 = kp_off() + n + (v >> 9u);
    write_descs(vec2u(slot, s2), vec2u(v & 15u, (v >> 4u) & 15u), select(1u, 2u, two), x, y, true);
    kps[slot] = full;
    if (two) { kps[s2] = full; }
}

// ── POS guided re-matching ──
// Per keypoint i of image 1 (kps, des): its predicted point kpp[i] (rounded H·p) and that
// point's descriptor (desp); the KNN nearest keypoints of image 2 (kpb, desb) by position. The nearest must lie within √thr2. The match is whichever of the K neighbours
// (nearest first) and the predicted point has the smallest descriptor distance, the first
// on ties (MATLAB min). res[i] = (x2, y2, 1, choice) or 0.
const KNN: u32 = 20u;

fn ssd_ab(ia: u32, ib: u32) -> f32 {
    var s = 0.0;
    for (var k = 0u; k < DLEN; k++) {
        let d = des[ia * DP + k] - desb[ib * DP + k];
        s += d * d;
    }
    return s;
}

@compute @workgroup_size(64)
fn pg_pos_pick(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= P.n_kp) { return; }
    res[i] = vec4f(0.0);
    let pp = kpp[i];
    if (kps[i].score <= 0.0 || pp.z <= 0.0) { return; }
    var nd: array<f32, KNN>;
    var ni: array<u32, KNN>;
    for (var k = 0u; k < KNN; k++) {
        nd[k] = 3.4e38;
        ni[k] = 0xffffffffu;
    }
    for (var j = 0u; j < P.n_b; j++) {
        let b = kpb[j];
        if (b.z <= 0.0) { continue; }
        let dx = b.x - pp.x;
        let dy = b.y - pp.y;
        let d = dx * dx + dy * dy;
        if (d >= nd[KNN - 1u]) { continue; }
        var k = KNN - 1u;
        loop {
            if (k == 0u || nd[k - 1u] <= d) { break; }
            nd[k] = nd[k - 1u];
            ni[k] = ni[k - 1u];
            k--;
        }
        nd[k] = d;
        ni[k] = j;
    }
    if (ni[0] == 0xffffffffu || nd[0] >= P.thr2) { return; }
    var best = 3.4e38;
    var pick = 0xffffffffu;
    for (var k = 0u; k < KNN; k++) {
        if (ni[k] == 0xffffffffu) { break; }
        let s = ssd_ab(i, ni[k]);
        if (s < best) {
            best = s;
            pick = k;
        }
    }
    var sp = 0.0;
    for (var k = 0u; k < DLEN; k++) {
        let d = des[i * DP + k] - desp[i * DP + k];
        sp += d * d;
    }
    if (sp < best) {
        res[i] = vec4f(pp.x, pp.y, 1.0, f32(KNN));
        return;
    }
    let b = kpb[ni[pick]];
    res[i] = vec4f(b.x, b.y, 1.0, f32(pick));
}

// ── Rotation mode 1 of the released code ("robust", MatchDemo's default) ──
// Per-pixel range normalization of the spectrally normalized channels: MATLAB
// normalize(CS, 3, "range"). Written into en (free once the phase congruency is summed).
@compute @workgroup_size(8, 8)
fn pg_rangenorm(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let n = P.w * P.h;
    let p = gid.y * P.w + gid.x;
    var v: array<f32, NO>;
    var mn = 3.4e38;
    var mx = -3.4e38;
    for (var o = 0u; o < NO; o++) {
        v[o] = cs[o * n + p] / sig[o];
        mn = min(mn, v[o]);
        mx = max(mx, v[o]);
    }
    for (var o = 0u; o < NO; o++) { en[o * n + p] = select(0.0, (v[o] - mn) / (mx - mn), mx > mn); }
}

// Sobel gradient of the normalized phase congruency (replicate edges): magnitude into wt,
// orientation bin into ori2b: round(doubled angle · 12 / 360) mod 12 (kptsOrientation).
@group(0) @binding(27) var<storage, read_write> ori2b: array<u32>;

fn pcn(x: i32, y: i32) -> f32 {
    let xx = clamp(x, 0, i32(P.w) - 1);
    let yy = clamp(y, 0, i32(P.h) - 1);
    let mn = bitcast<f32>(atomicLoad(&mm[0]));
    let mx = bitcast<f32>(atomicLoad(&mm[1]));
    return (src[u32(yy) * P.w + u32(xx)] - mn) / max(mx - mn, 1e-30);
}

@compute @workgroup_size(8, 8)
fn pg_sobel(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let gx = -pcn(x - 1, y - 1) + pcn(x + 1, y - 1) - 2.0 * pcn(x - 1, y) + 2.0 * pcn(x + 1, y) - pcn(x - 1, y + 1) + pcn(x + 1, y + 1);
    let gy = -pcn(x - 1, y - 1) - 2.0 * pcn(x, y - 1) - pcn(x + 1, y - 1) + pcn(x - 1, y + 1) + 2.0 * pcn(x, y + 1) + pcn(x + 1, y + 1);
    let p = gid.y * P.w + gid.x;
    wt[p] = sqrt(gx * gx + gy * gy);
    var ang = atan2(gy, gx) * 360.0 / PI;
    if (ang < 0.0) { ang += 360.0; }
    var b = u32(floor(ang * 12.0 / 360.0 + 0.5));
    if (b >= 12u) { b -= 12u; }
    ori2b[p] = b;
}

// Orientation histogram of a keypoint (workgroup per keypoint): 129 × 129 window, disk of
// radius 64, Gaussian σ = 128 / 6, magnitude-weighted; bins 1..11 only (the released
// code's loop never counts bin 0), [1 4 6 4 1] / 16 circular smoothing, peaks above 0.8 of
// the maximum with parabolic interpolation, angle = round(3 · bin) (its units), kept mod 12.
// ori = count | angle_k << (3 + 4k); count 0 drops the keypoint.
var<workgroup> oh: array<f32, 768>;   // 64 invocations × 12 bins

@compute @workgroup_size(64)
fn pg_orient_robust(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nw: vec3u,
                    @builtin(local_invocation_index) lid: u32) {
    let i = wid.y * nw.x + wid.x;
    if (lid == 0u) { redu = atomicLoad(&cnt[0]); }
    let n = workgroupUniformLoad(&redu);
    if (i >= n) { return; }
    let slot = kp_off() + i;
    let kp = kps[slot];
    let x = mround(kp.x) - 1;
    let y = mround(kp.y) - 1;
    var th: array<f32, 12>;
    let sigma = 128.0 / 6.0;
    for (var e = lid; e < 129u * 129u; e += 64u) {
        let dx = i32(e % 129u) - 64;
        let dy = i32(e / 129u) - 64;
        let r2 = f32(dx * dx + dy * dy);
        if (r2 > 64.0 * 64.0) { continue; }
        let px = x + dx;
        let py = y + dy;
        if (px < 0 || py < 0 || px >= i32(P.w) || py >= i32(P.h)) { continue; }
        let p = u32(py) * P.w + u32(px);
        th[ori2b[p]] += wt[p] * exp(-r2 / (2.0 * sigma * sigma));
    }
    for (var b = 0u; b < 12u; b++) { oh[lid * 12u + b] = th[b]; }
    workgroupBarrier();
    if (lid != 0u) { return; }
    var t: array<f32, 12>;   // t[k - 1] = temp_hist(k), k = 1..12: gradient bin k (12 never occurs)
    for (var k = 1u; k <= 12u; k++) {
        var s = 0.0;
        if (k < 12u) { for (var l = 0u; l < 64u; l++) { s += oh[l * 12u + k]; } }
        t[k - 1u] = s;
    }
    var hh: array<f32, 12>;
    var mxh = 0.0;
    for (var k = 0u; k < 12u; k++) {
        hh[k] = (t[(k + 10u) % 12u] + t[(k + 2u) % 12u]) / 16.0 + 4.0 * (t[(k + 11u) % 12u] + t[(k + 1u) % 12u]) / 16.0 + t[k] * 6.0 / 16.0;
        mxh = max(mxh, hh[k]);
    }
    var count = 0u;
    var word = 0u;
    for (var k = 0u; k < 12u; k++) {
        let h1 = hh[(k + 11u) % 12u];
        let h2 = hh[(k + 1u) % 12u];
        if (hh[k] > h1 && hh[k] > h2 && hh[k] > mxh * 0.8) {
            var bin = f32(k + 1u) + 0.5 * (h1 - h2) / (h1 + h2 - 2.0 * hh[k]);
            if (bin < 0.0) { bin += 12.0; } else if (bin >= 12.0) { bin -= 12.0; }
            let ang = u32(mround(3.0 * bin)) % 12u;
            word = word | (ang << (3u + 4u * count));
            count++;
        }
    }
    ori[slot] = word | count;
}

// Extra slots for keypoints with more than one direction (count − 1 each), in keypoint
// order; the prefix goes to ori[MAX_KP_PG + slot], the total to cnt[2].
const MAX_KP_PG: u32 = 65536u;
const DIR_REC: u32 = 2u * MAX_KP_PG;

@compute @workgroup_size(256)
fn pg_compact_robust(@builtin(local_invocation_index) lid: u32) {
    let n = atomicLoad(&cnt[0]);
    let chunk = (n + 255u) / 256u;
    let i0 = min(lid * chunk, n);
    let i1 = min(i0 + chunk, n);
    var local = 0u;
    for (var i = i0; i < i1; i++) { local += max(ori[kp_off() + i] & 7u, 1u) - 1u; }
    scan[lid] = local;
    workgroupBarrier();
    if (lid == 0u) {
        var t = 0u;
        for (var k = 0u; k < 256u; k++) {
            let c = scan[k];
            scan[k] = t;
            t += c;
        }
        atomicStore(&cnt[2], t);
    }
    workgroupBarrier();
    var k = scan[lid];
    for (var i = i0; i < i1; i++) {
        let s = kp_off() + i;
        ori[MAX_KP_PG + s] = k;
        k += max(ori[s] & 7u, 1u) - 1u;
    }
}

@compute @workgroup_size(64)
fn pg_describe_robust(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let n = atomicLoad(&cnt[0]);
    if (i >= n) { return; }
    let slot = kp_off() + i;
    let kp = kps[slot];
    let x = mround(kp.x);
    let y = mround(kp.y);
    let v = ori[slot];
    let count = v & 7u;
    let full = Kp(kp.x * P.scale, kp.y * P.scale, kp.score, P.scale);
    if (count == 0u || !inside(x, y)) {
        for (var k = 0u; k < DP; k++) { des[slot * DP + k] = 0.0; }
        for (var w = 0u; w < DQ; w++) { desq[slot * DQ + w] = 0u; }
        dsc[slot] = 0.0;
        kps[slot] = Kp(full.x, full.y, 0.0, P.scale);
        return;
    }
    let base2 = kp_off() + n + ori[MAX_KP_PG + slot];
    for (var j = 0u; j < count; j++) {
        var s = slot;
        if (j > 0u) { s = base2 + j - 1u; }
        write_desc(s, x, y, (v >> (3u + 4u * j)) & 15u, true);
        kps[s] = full;
    }
}

// ── Global rotation search ──
// Upright descriptors (desb) turned by P.ma directions (30° each) by pure re-indexing, the
// same permutation write_desc applies for a primary direction: element (a', r, c) of the
// result = element ((a' + ma) mod 12, r, (c − ma) mod 6) of the upright one. Norms and the
// largest element are unchanged; the 8-bit copy is rebuilt.
@compute @workgroup_size(64)
fn pg_shift(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= P.n_kp) { return; }
    let k = P.ma;
    let base = i * DP;
    var mx = 0.0;
    for (var pt = 0u; pt < NPT; pt++) {
        var spt = NDIR * NRING;
        if (pt < NDIR * NRING) { spt = (((pt / NRING) + k) % NDIR) * NRING + pt % NRING; }
        for (var c = 0u; c < NO; c++) {
            let v = desb[base + spt * NO + (c + NO - k % NO) % NO];
            des[base + pt * NO + c] = v;
            mx = max(mx, v);
        }
    }
    for (var b = 0u; b < SSN / (NDIR * NRING); b++) {
        for (var q = 0u; q < NDIR * NRING; q++) {
            let o = base + NPT * NO + b * NDIR * NRING;
            let v = desb[o + (((q / NRING) + k) % NDIR) * NRING + q % NRING];
            des[o + q] = v;
            mx = max(mx, v);
        }
    }
    for (var kk = DLEN; kk < DP; kk++) { des[base + kk] = 0.0; }
    dsc[i] = mx / 255.0;
    let inv = select(0.0, 255.0 / mx, mx > 0.0);
    for (var w = 0u; w < DQ; w++) {
        var word = 0u;
        for (var e = 0u; e < 4u; e++) {
            word = word | (min(u32(round(des[base + w * 4u + e] * inv)), 255u) << (8u * e));
        }
        desq[i * DQ + w] = word;
    }
}

// Image turned about its centre by the angle (cos cth, sin sth): dst(x) = src(R⁻¹ (x − c) + c),
// bilinear, zero outside (rotation search, the half-step frame).
@compute @workgroup_size(8, 8)
fn pg_rotate(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= P.w || gid.y >= P.h) { return; }
    let cx = (f32(P.w) - 1.0) / 2.0;
    let cy = (f32(P.h) - 1.0) / 2.0;
    let dx = f32(gid.x) - cx;
    let dy = f32(gid.y) - cy;
    let sx = P.cth * dx + P.sth * dy + cx;
    let sy = -P.sth * dx + P.cth * dy + cy;
    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let fx = sx - f32(x0);
    let fy = sy - f32(y0);
    var acc = 0.0;
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let x = x0 + i;
            let y = y0 + j;
            if (x < 0 || y < 0 || x >= i32(P.w) || y >= i32(P.h)) { continue; }
            acc += select(1.0 - fx, fx, i == 1) * select(1.0 - fy, fy, j == 1) * src[u32(y) * P.w + u32(x)];
        }
    }
    dst[gid.y * P.w + gid.x] = acc;
}

// ── Dense POS (a modification): instead of choosing among the KNN keypoints near the predicted
// point, describe image 2 on a grid around it (±P.hsize px, step 2, then step 1 around the best)
// and keep the position of the smallest descriptor distance, refined to a sub-pixel by a
// parabola through its neighbours. gm holds image 2's ring maps; des the image-1 descriptors
// (POS re-description, direction P.ma for image 2). res[i] = (x2, y2, 1, SSD) or 0.
fn desc_ssd(x: i32, y: i32, ia: u32) -> f32 {
    if (!inside(x, y)) { return 3.4e38; }
    let d = build_desc(x, y, P.ma, false);
    var s = 0.0;
    for (var k = 0u; k < DLEN; k++) {
        let e = des[ia * DP + k] - d[k];
        s += e * e;
    }
    return s;
}

fn parab(a: f32, b: f32, c: f32) -> f32 {
    let den = a - 2.0 * b + c;
    if (!(den > 0.0) || a >= 3.0e38 || c >= 3.0e38) { return 0.0; }
    return clamp(0.5 * (a - c) / den, -0.5, 0.5);
}

@compute @workgroup_size(64)
fn pg_pos_dense(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= P.n_kp) { return; }
    res[i] = vec4f(0.0);
    let pp = kpp[i];
    if (kps[i].score <= 0.0) { return; }
    let px = i32(pp.x);
    let py = i32(pp.y);
    let W = i32(P.hsize);
    var best = 3.4e38;
    var bx = px;
    var by = py;
    for (var dy = -W; dy <= W; dy += 2) {
        for (var dx = -W; dx <= W; dx += 2) {
            let s = desc_ssd(px + dx, py + dy, i);
            if (s < best) { best = s; bx = px + dx; by = py + dy; }
        }
    }
    if (best >= 3.0e38) { return; }
    let cx = bx;
    let cy = by;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let s = desc_ssd(cx + dx, cy + dy, i);
            if (s < best) { best = s; bx = cx + dx; by = cy + dy; }
        }
    }
    let ox = parab(desc_ssd(bx - 1, by, i), best, desc_ssd(bx + 1, by, i));
    let oy = parab(desc_ssd(bx, by - 1, i), best, desc_ssd(bx, by + 1, i));
    res[i] = vec4f(f32(bx) + ox, f32(by) + oy, 1.0, best);
}
