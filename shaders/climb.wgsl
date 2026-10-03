// Refine on the GPU (gclimb.zig): the climb's decisions, in f32, between the score kernels of
// smi.wgsl. Every step is a stage: candidate poses plus the current one are scored (overlap
// moments, moments_coefs), the best that beats the current score is kept (climb_select), and the
// gradient with the Gauss–Newton matrix there follows (gather_sel, tangent_grad,
// climb_combine). The state buffer carries the pose, its score, the gradient and the
// iteration's bookkeeping from stage to stage; the host reads it once per batch.
//
// Poses are row-major 3×3 (9 floats), moving → fixed pixels, 1-based, as lie.zig. The pose
// algebra below is lie.zig's in f32.

struct U {
    n: u32, nk: u32, homog: u32, sym: u32,
    hop: u32, kind: u32, n_iters: u32, np: u32,
    stride: u32, mx: u32, my: u32, cw: u32,
    ch: u32, fw: u32, fh: u32, iter0: u32,
    rw: f32, rh: f32, lw: f32, lh: f32,
    ox: f32, oy: f32, p0: f32, p1: f32,
    np1: u32, gate: u32, pad1: u32, pad2: u32,
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> st: array<f32>;
@group(0) @binding(2) var<storage, read_write> hs0: array<f32>;
@group(0) @binding(3) var<storage, read_write> hs1: array<f32>;
@group(0) @binding(4) var<storage, read> sc0: array<f32>;
@group(0) @binding(5) var<storage, read> sc1: array<f32>;
@group(0) @binding(6) var<storage, read> g0: array<f32>;
@group(0) @binding(7) var<storage, read> g1: array<f32>;
@group(0) @binding(8) var<storage, read_write> pick: array<f32>;
@group(0) @binding(9) var<storage, read> aux_in: array<f32>;
@group(0) @binding(10) var<storage, read_write> aux_out: array<f32>;
@group(0) @binding(11) var<storage, read_write> aux_out2: array<f32>;
@group(0) @binding(12) var<storage, read> aux_in2: array<f32>;
@group(0) @binding(13) var<storage, read_write> aux_out3: array<f32>;

// ── the state buffer (gclimb.zig State) ──
const S_DONE: u32 = 0u;     // > 0.5: the climb has stopped (gates every later dispatch)
const S_CUR: u32 = 1u;      // the current pose's score
const S_MOVED: u32 = 2u;    // a stage moved the pose in this iteration
const S_STALL: u32 = 3u;    // iterations in a row without a move
const S_ITER: u32 = 4u;     // iterations done
const S_GNSTOP: u32 = 5u;   // no Gauss–Newton step beat the score in this iteration
const S_NREC: u32 = 6u;     // trail records written
const S_FWD: u32 = 7u;      // the current pose's direction scores
const S_INV: u32 = 8u;
const S_START: u32 = 9u;    // the start pose's score (the climb's first stage)
const S_H: u32 = 16u;       // the current pose, forward (9) and inverse (9)
const S_HI: u32 = 25u;
const S_G: u32 = 40u;       // gradient (8)
const S_A: u32 = 48u;       // Gauss–Newton matrix (8 × 8)
const S_MU: u32 = 112u;     // step-length factor of each candidate (8)
const S_REC: u32 = 128u;    // trail records (REC floats each)
const REC: u32 = 16u;       // [kind, iteration, mean, fwd, inv, factor, 0, H (9)]
const MAX_REC: u32 = 256u;
const S_AMPS: u32 = 4224u;  // the gradient ladder's step lengths (up to 16)
const S_SAMPS: u32 = 4240u; // the second line search's (up to 16)
const NO_SCORE: f32 = -3.0e38;
const NO_FLAG: u32 = 0xffffffffu;

// ── self-gating ──
// The climb's kernels are dispatched directly inside a gated batch (gclimb.zig k1) and skip their
// work themselves, as a gated dispatch would be skipped: once the climb is done, or the flag
// u.gate (NO_FLAG: none) is set. Kernels without uniforms check the done flag only.
fn gated() -> bool {
    return st[S_DONE] > 0.5 || (u.gate != NO_FLAG && st[u.gate] > 0.5);
}
var<workgroup> w_off: u32;
// gated() for a workgroup with barriers (a uniform answer)
fn gateOff(lid: u32) -> bool {
    if (lid == 0u) { w_off = select(0u, 1u, gated()); }
    return workgroupUniformLoad(&w_off) == 1u;
}

// ── pose algebra (lie.zig, f32) ──
fn mul3(a: array<f32, 9>, b: array<f32, 9>) -> array<f32, 9> {
    var o: array<f32, 9>;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            o[i * 3u + j] = a[i * 3u] * b[j] + a[i * 3u + 1u] * b[3u + j] + a[i * 3u + 2u] * b[6u + j];
        }
    }
    return o;
}


// tangent coordinates [tx ty th sg al ga px py] → algebra element (lie.zig hat)
fn hat(x: array<f32, 8>) -> array<f32, 9> {
    var iso = x[3];
    var z = 0.0;
    var px = 0.0;
    var py = 0.0;
    if (u.homog == 1u) {
        iso = x[3] / 3.0;
        z = -2.0 * iso;
        px = x[6];
        py = x[7];
    }
    return array<f32, 9>(iso + x[4], x[5] - x[2], x[0], x[5] + x[2], iso - x[4], x[1], px, py, z);
}

fn vee(A0: array<f32, 9>) -> array<f32, 8> {
    var a = A0[0];
    var e = A0[4];
    if (u.homog == 1u) {
        let tr = (a + e + A0[8]) / 3.0;
        a = a - tr;
        e = e - tr;
        let iso = (a + e) / 2.0;
        return array<f32, 8>(A0[2], A0[5], (A0[3] - A0[1]) / 2.0, iso * 3.0, (a - e) / 2.0, (A0[3] + A0[1]) / 2.0, A0[6], A0[7]);
    }
    return array<f32, 8>(A0[2], A0[5], (A0[3] - A0[1]) / 2.0, (a + e) / 2.0, (a - e) / 2.0, (A0[3] + A0[1]) / 2.0, 0.0, 0.0);
}

// scaling and squaring with a 13-term Taylor series (lie.zig expm3)
// The matrix exponential by scaling and squaring of a 13-term Taylor series, on WGSL's native
// 3 × 3 type (Apple's paravirtual GPU compiler failed on the same loops over 9-float arrays).
// The series uses only powers of A, so working on Aᵀ (the column-major view of the row-major
// array) and reading the result back the same way gives exp(A).
fn expm3(A: array<f32, 9>) -> array<f32, 9> {
    let At = mat3x3f(vec3f(A[0], A[1], A[2]), vec3f(A[3], A[4], A[5]), vec3f(A[6], A[7], A[8]));
    var nrm = 0.0;
    for (var c = 0u; c < 3u; c++) { nrm = max(nrm, max(abs(At[c].x), max(abs(At[c].y), abs(At[c].z)))); }
    var s = 0u;
    if (nrm > 0.5) { s = u32(min(10.0, max(0.0, ceil(log2(nrm / 0.5))))); }
    let B = At * exp2(-f32(s));
    let I = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
    var T = I;
    var P = I;
    for (var k = 1u; k < 14u; k++) {
        P = (P * B) * (1.0 / f32(k));
        T = T + P;
    }
    for (var i = 0u; i < s; i++) { T = T * T; }
    return array<f32, 9>(T[0].x, T[0].y, T[0].z, T[1].x, T[1].y, T[1].z, T[2].x, T[2].y, T[2].z);
}

fn destN(w0: f32, h0: f32) -> array<f32, 9> {
    let w = max(w0, 1.0);
    let h = max(h0, 1.0);
    return array<f32, 9>(2.0 / w, 0.0, -(w + 1.0) / w, 0.0, 2.0 / h, -(h + 1.0) / h, 0.0, 0.0, 1.0);
}
// destN's inverse
fn destNInv(w0: f32, h0: f32) -> array<f32, 9> {
    let w = max(w0, 1.0);
    let h = max(h0, 1.0);
    return array<f32, 9>(w / 2.0, 0.0, (w + 1.0) / 2.0, 0.0, h / 2.0, (h + 1.0) / 2.0, 0.0, 0.0, 1.0);
}

fn projectGroup(H: array<f32, 9>) -> array<f32, 9> {
    let s = select(1.0, H[8], abs(H[8]) > 1e-12);
    var o: array<f32, 9>;
    for (var i = 0u; i < 9u; i++) { o[i] = H[i] / s; }
    if (u.homog != 1u) {
        o[6] = 0.0;
        o[7] = 0.0;
        o[8] = 1.0;
    }
    return o;
}

// exp(ξ̂) applied in normalized fixed coordinates (lie.zig composeN)
// Row-major 9-float poses as WGSL's native 3 × 3 matrices and back (products and inverses on
// the native type: Apple's paravirtual GPU compiler failed on chains of them over arrays).
fn toM(a: array<f32, 9>) -> mat3x3f {
    return mat3x3f(vec3f(a[0], a[3], a[6]), vec3f(a[1], a[4], a[7]), vec3f(a[2], a[5], a[8]));
}
fn fromM(m: mat3x3f) -> array<f32, 9> {
    return array<f32, 9>(m[0].x, m[1].x, m[2].x, m[0].y, m[1].y, m[2].y, m[0].z, m[1].z, m[2].z);
}
// The inverse by the adjugate (rows: cross products of the columns), one Newton step refining it.
fn invM(m: mat3x3f) -> mat3x3f {
    let r0 = cross(m[1], m[2]);
    let r1 = cross(m[2], m[0]);
    let r2 = cross(m[0], m[1]);
    let det = dot(m[0], r0);
    let X = transpose(mat3x3f(r0, r1, r2)) * (1.0 / (sign(det) * max(abs(det), 1e-30)));
    let I = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
    return X * (2.0 * I - m * X);
}
fn composeN(H0: array<f32, 9>, x: array<f32, 8>) -> array<f32, 9> {
    let H = toM(destNInv(u.rw, u.rh)) * toM(expm3(hat(x))) * toM(destN(u.rw, u.rh)) * toM(H0);
    return projectGroup(fromM(H));
}

fn translateL(dx: f32, dy: f32, H: array<f32, 9>) -> array<f32, 9> {
    return mul3(array<f32, 9>(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0), H);
}

fn loadState9(o: u32) -> array<f32, 9> {
    var m: array<f32, 9>;
    for (var i = 0u; i < 9u; i++) { m[i] = st[o + i]; }
    return m;
}

// candidate k of a stage: the forward pose and its inverse
fn putCand(k: u32, H: array<f32, 9>) {
    let Hi = fromM(invM(toM(H)));
    for (var i = 0u; i < 9u; i++) {
        hs0[k * 9u + i] = H[i];
        hs1[k * 9u + i] = Hi[i];
    }
}

// the current pose as candidate k (index u.n of every stage: its fresh score is the baseline)
fn putCurrent(k: u32) {
    for (var i = 0u; i < 9u; i++) {
        hs0[k * 9u + i] = st[S_H + i];
        hs1[k * 9u + i] = st[S_HI + i];
    }
}

// ── stage generators ──

// A stage with no candidates: the current pose only (the climb's first gradient).
@compute @workgroup_size(1)
fn cands_current() {
    if (st[S_DONE] > 0.5) { return; }
    putCurrent(0u);
}

var<workgroup> w_step: array<f32, 8>;
var<workgroup> w_sn: f32;
var<workgroup> w_ok: u32;

// Gauss–Newton step (climb.zig localStep): solve (A + 1e-4 diag) δ = ½ g, then the step at
// five lengths (×0.5 … ×8, each capped at a tangent norm of 0.25) as candidates 0–4, the current
// pose as 5. Thread 0 solves; threads 0–4 compose one candidate each. After a step that beat
// nothing in this iteration (or a failed solve) every candidate is the current pose.
@compute @workgroup_size(8)
fn gn_cands(@builtin(local_invocation_index) lid: u32) {
    if (gateOff(lid)) { return; }
    let nk = u.nk;
    if (lid == 0u) {
        var ok = st[S_GNSTOP] < 0.5;
        var M: array<f32, 72>;   // nk × 9: the matrix, the right-hand side in column 8
        if (ok) {
            for (var i = 0u; i < nk; i++) {
                for (var j = 0u; j < nk; j++) { M[i * 9u + j] = st[S_A + i * nk + j]; }
                M[i * 9u + i] = M[i * 9u + i] + 1e-4 * max(M[i * 9u + i], 1e-12);
                M[i * 9u + 8u] = 0.5 * st[S_G + i];
            }
            // Gauss–Jordan with partial pivoting (lie.zig solveLinear)
            for (var i = 0u; i < nk && ok; i++) {
                var piv = i;
                for (var r = i + 1u; r < nk; r++) { if (abs(M[r * 9u + i]) > abs(M[piv * 9u + i])) { piv = r; } }
                if (piv != i) {
                    for (var j = 0u; j < 9u; j++) {
                        let t = M[i * 9u + j];
                        M[i * 9u + j] = M[piv * 9u + j];
                        M[piv * 9u + j] = t;
                    }
                }
                let d = M[i * 9u + i];
                if (!(abs(d) > 1e-30)) { ok = false; break; }
                for (var r = 0u; r < nk; r++) {
                    if (r == i) { continue; }
                    let f = M[r * 9u + i] / d;
                    if (f == 0.0) { continue; }
                    for (var j = i; j < 9u; j++) {
                        if (j < nk || j == 8u) { M[r * 9u + j] = M[r * 9u + j] - f * M[i * 9u + j]; }
                    }
                }
            }
        }
        var sn = 0.0;
        for (var i = 0u; i < 8u; i++) {
            var v = 0.0;
            if (ok && i < nk) { v = M[i * 9u + 8u] / M[i * 9u + i]; }
            if (!(abs(v) < 1e30)) { ok = false; }
            w_step[i] = v;
            sn = sn + v * v;
        }
        w_sn = sqrt(sn);
        w_ok = select(0u, 1u, ok);
    }
    workgroupBarrier();
    if (lid == 5u) { putCurrent(5u); }
    if (lid >= 5u) { return; }
    if (w_ok == 0u) {
        putCurrent(lid);
        st[S_MU + lid] = 0.0;
        return;
    }
    let mus = array<f32, 5>(0.5, 1.0, 2.0, 4.0, 8.0);
    let mu = mus[lid];
    let nrm = w_sn * mu;
    let sc = mu * select(1.0, 0.25 / nrm, nrm > 0.25);
    var x: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) { x[i] = sc * w_step[i]; }
    putCand(lid, composeN(loadState9(S_H), x));
    st[S_MU + lid] = mu;
}

// Gradient line search (climb.zig search, for the copula scores, which have no Gauss–Newton
// matrix): the unit ascent direction at u.n step lengths (the ladder u.kind 0: S_AMPS,
// 1: S_SAMPS) as candidates, the current pose last.
@compute @workgroup_size(16)
fn ls_cands(@builtin(local_invocation_index) lid: u32) {
    if (gateOff(lid)) { return; }
    let nk = u.nk;
    if (lid == u.n) { putCurrent(u.n); }
    if (lid >= u.n) { return; }
    var gn = 0.0;
    for (var i = 0u; i < nk; i++) { gn = gn + st[S_G + i] * st[S_G + i]; }
    gn = sqrt(gn);
    if (st[S_GNSTOP] > 0.5 || !(gn > 1e-12) || !(gn < 1e30)) {
        putCurrent(lid);
        st[S_MU + lid] = 0.0;
        return;
    }
    let a = select(st[S_AMPS + lid], st[S_SAMPS + lid], u.kind == 1u);
    var x: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) { x[i] = select(0.0, a * st[S_G + i] / gn, i < nk); }
    putCand(lid, composeN(loadState9(S_H), x));
    st[S_MU + lid] = a;
}

// Shift hop (climb.zig transHop): the shift map's peak (aux_in = smi.res: score, flat index)
// as a translation of the current pose, candidate 0; the current pose 1.
@compute @workgroup_size(1)
fn hop_cand() {
    if (gated()) { return; }
    let idx = bitcast<u32>(aux_in[1]);
    let H = loadState9(S_H);
    if (idx >= u.mx * u.my) {
        putCurrent(0u);
    } else {
        let py = idx / u.mx;
        let px = idx % u.mx;
        let lx = select(i32(px) - i32(u.mx), i32(px), px <= u.mx / 2u);
        let ly = select(i32(py) - i32(u.my), i32(py), py <= u.my / 2u);
        let dx = f32(lx) * (f32(u.fw) / f32(u.cw));
        let dy = f32(ly) * (f32(u.fh) / f32(u.ch));
        putCand(0u, translateL(-dx, -dy, H));
    }
    st[S_MU] = 0.0;
    putCurrent(1u);
}

// Roto-scale hop (climb.zig fmHop): the roto-scale map's peak (aux_in = smi.res; u.mx angles
// by u.mx scales on the map, u.my of them angles, u.p0 the log-scale step) as a turn and
// rescale of the current pose about (u.ox, u.oy) (fixed pixels), candidate 0; the current pose 1.
// A peak within 2e-4 of no change is no candidate.
@compute @workgroup_size(1)
fn rs_cand() {
    if (gated()) { return; }
    let idx = bitcast<u32>(aux_in[1]);
    let n = u.mx;
    var ok = idx < n * n;
    var dth = 0.0;
    var dsg = 0.0;
    if (ok) {
        let py = idx / n;
        let px = idx % n;
        let lx = select(i32(px) - i32(n), i32(px), px <= n / 2u);
        let ly = select(i32(py) - i32(n), i32(py), py <= n / 2u);
        dth = -(f32(lx) / f32(max(u.my, 1u))) * 6.283185307179586;
        dsg = -f32(ly) * u.p0;
        ok = !(abs(dth) < 2e-4 && abs(dsg) < 2e-4);
    }
    if (ok) {
        // X ↦ c + e^σ R_θ (X − c) (lie.zig simAbout), then the current pose
        let s = exp(dsg);
        let a = s * cos(dth);
        let b = -s * sin(dth);
        let d = s * sin(dth);
        let e = s * cos(dth);
        let S = array<f32, 9>(a, b, u.ox - a * u.ox - b * u.oy, d, e, u.oy - d * u.ox - e * u.oy, 0.0, 0.0, 1.0);
        putCand(0u, projectGroup(mul3(S, loadState9(S_H))));
    } else {
        putCurrent(0u);
    }
    st[S_MU] = 0.0;
    putCurrent(1u);
}

// The second Gauss–Newton block of an iteration (after the roto-scale hop) starts afresh.
@compute @workgroup_size(1)
fn gn_reset() {
    if (st[S_DONE] > 0.5) { return; }
    st[S_GNSTOP] = 0.0;
}

// The current pose in canvas pixels (lie.zig canvasH: translate(−ox, −oy) · H) into aux_out,
// the pose the map's warp_stack reads.
@compute @workgroup_size(1)
fn hc_canvas() {
    if (gated()) { return; }
    let Hc = translateL(-u.ox, -u.oy, loadState9(S_H));
    for (var i = 0u; i < 9u; i++) { aux_out[i] = Hc[i]; }
}

// ── the pick ──

fn meanScore(k: u32) -> f32 {
    let f = sc0[k];
    if (u.sym == 0u) { return f; }
    let r = sc1[k];
    return select(NO_SCORE, 0.5 * (f + r), f > NO_SCORE && r > NO_SCORE);
}

// what a stage ascends: the score, less a spline candidate's bending penalty (aux_in)
fn objective(k: u32) -> f32 {
    let s = meanScore(k);
    if (u.kind != 3u || s <= NO_SCORE) { return s; }
    return s - aux_in[k];
}

// Of u.n candidates and the current pose (index u.n), the highest score that beats the
// current pose's by 1e-6 of it (above the f32 sums' round-off) becomes the current pose,
// recorded on the trail with its kind (u.kind: 1 shift hop, 2 Gauss–Newton, 3 spline,
// 4 roto-scale hop, 5 gradient line search); else the current pose stays (and, for a
// Gauss–Newton or line-search stage, the rest of the block is off).
// pick[0]: the chosen index, for gather_sel.
@compute @workgroup_size(1)
fn climb_select() {
    if (gated()) { return; }
    let n = u.n;
    let cur = objective(n);
    var best = n;
    var best_s = cur;
    for (var i = 0u; i < n; i++) {
        let sc = objective(i);
        if (sc <= NO_SCORE) { continue; }
        if (sc > cur + max(1e-6 * abs(cur), 1e-12) && (best == n || sc > best_s)) {
            best = i;
            best_s = sc;
        }
    }
    pick[0] = f32(best);
    if (u.kind == 0u && st[S_ITER] == 0.0) { st[S_START] = cur; }
    // the score (a spline's penalty is not part of what the trail shows)
    best_s = meanScore(best);
    st[S_CUR] = best_s;
    st[S_FWD] = sc0[best];
    st[S_INV] = select(0.0, sc1[best], u.sym == 1u);
    if (best == n) {
        if (u.kind == 2u || u.kind == 5u) { st[S_GNSTOP] = 1.0; }
        return;
    }
    for (var i = 0u; i < 9u; i++) {
        st[S_H + i] = hs0[best * 9u + i];
        st[S_HI + i] = hs1[best * 9u + i];
    }
    st[S_MOVED] = 1.0;
    let r = u32(st[S_NREC]);
    if (r < MAX_REC) {
        let o = S_REC + r * REC;
        st[o] = f32(u.kind);
        st[o + 1u] = st[S_ITER] + 1.0;
        st[o + 2u] = best_s;
        st[o + 3u] = sc0[best];
        st[o + 4u] = select(0.0, sc1[best], u.sym == 1u);
        st[o + 5u] = select(0.0, st[S_MU + best], u.kind == 2u || u.kind == 5u);
        st[o + 6u] = 0.0;
        for (var i = 0u; i < 9u; i++) { st[o + 7u + i] = st[S_H + i]; }
        st[S_NREC] = f32(r + 1u);
    }
}

// ── the gradient at the pick ──

// Both directions' tangent rows summed (g0: the forward lane's u.np partial rows of GRAD_STRIDE,
// g1: the inverse lane's u.np1), into w_gs (forward at 0, inverse at GRAD_STRIDE): each of the
// 2 · GRAD_STRIDE sums in SUM_SLICES slices of rows, one per thread, then the slices added.
const GRAD_STRIDE: u32 = 44u;
const SUM_SLICES: u32 = 2u;
var<workgroup> w_gs: array<f32, 88>;
var<workgroup> w_gsl: array<f32, 176>;
fn sumRows(lid: u32) {
    for (var task = lid; task < 2u * GRAD_STRIDE * SUM_SLICES; task = task + 256u) {
        let j = task % (2u * GRAD_STRIDE);
        let sl = task / (2u * GRAD_STRIDE);
        let which = j / GRAD_STRIDE;
        let c = j % GRAD_STRIDE;
        let np = select(u.np, select(0u, u.np1, u.sym == 1u), which == 1u);
        let r1 = min(np, (sl + 1u) * ((np + SUM_SLICES - 1u) / SUM_SLICES));
        var s = 0.0;
        for (var r = sl * ((np + SUM_SLICES - 1u) / SUM_SLICES); r < r1; r++) {
            s = s + select(g0[r * GRAD_STRIDE + c], g1[r * GRAD_STRIDE + c], which == 1u);
        }
        w_gsl[task] = s;
    }
    workgroupBarrier();
    for (var j = lid; j < 2u * GRAD_STRIDE; j = j + 256u) {
        var s = 0.0;
        for (var sl = 0u; sl < SUM_SLICES; sl++) { s = s + w_gsl[sl * 2u * GRAD_STRIDE + j]; }
        w_gs[j] = s;
    }
    workgroupBarrier();
}
fn gsum(which: u32, c: u32) -> f32 { return w_gs[which * GRAD_STRIDE + c]; }

// a direction's gradient and Gauss–Newton matrix from its summed tangent row (pose.zig
// gradFrom: grad[0..nk], then the 8 × 8 upper triangle from index 8)
fn unpackGrad(row: u32, which: u32, g: ptr<function, array<f32, 8>>, A: ptr<function, array<f32, 64>>) {
    let nk = u.nk;
    for (var q = 0u; q < nk; q++) { (*g)[q] = gsum(which, row + q); }
    var h = 8u;
    for (var q = 0u; q < 8u; q++) {
        for (var m = q; m < 8u; m++) {
            if (q < nk && m < nk) {
                let v = gsum(which, row + h);
                (*A)[q * nk + m] = v;
                (*A)[m * nk + q] = v;
            }
            h = h + 1u;
        }
    }
}

var<workgroup> w_P: array<f32, 64>;
var<workgroup> w_Ab: array<f32, 64>;
var<workgroup> w_B: array<f32, 64>;

// The symmetric gradient at the current pose (pair.zig sym): g = ½ (g_fwd + P g_inv),
// A = ½ (A_fwd + P A_inv Pᵀ), P the inverse direction's pullback (lie.zig adjointGrad). Threads
// i < nk build row i of P; the products then take one entry per thread. (A direction with too
// little overlap has zero coefficients, so its gradient is zero, as on the host.)
@compute @workgroup_size(256)
fn climb_combine(@builtin(local_invocation_index) lid: u32) {
    if (gateOff(lid)) { return; }
    sumRows(lid);
    let nk = u.nk;
    let nn = nk * nk;
    if (u.sym == 0u) {
        if (lid == 0u) {
            var gf: array<f32, 8>;
            var Af: array<f32, 64>;
            unpackGrad(0u, 0u, &gf, &Af);
            for (var i = 0u; i < 8u; i++) { st[S_G + i] = gf[i]; }
            for (var i = 0u; i < 64u; i++) { st[S_A + i] = Af[i]; }
        }
        return;
    }
    if (lid == 0u) {
        var gb: array<f32, 8>;
        var Ab: array<f32, 64>;
        unpackGrad(0u, 1u, &gb, &Ab);
        for (var i = 0u; i < 64u; i++) { w_Ab[i] = Ab[i]; }
        for (var i = 0u; i < 8u; i++) { w_B[i] = gb[i]; }   // g_inv, until the products below
    }
    if (lid < nk) {
        // row lid of P: the pullback of each unit vector's component lid
        // M and its inverse from the pose and its inverse (the state keeps both) with the
        // normalizations' closed-form inverses: no general 3 × 3 inverse here (Apple's
        // paravirtual GPU compiler failed on this kernel)
        let M = mul3(mul3(destN(u.lw, u.lh), loadState9(S_HI)), destNInv(u.rw, u.rh));
        let Mi = mul3(mul3(destN(u.rw, u.rh), loadState9(S_H)), destNInv(u.lw, u.lh));
        let x = array<f32, 8>(select(0.0, 1.0, lid == 0u), select(0.0, 1.0, lid == 1u), select(0.0, 1.0, lid == 2u), select(0.0, 1.0, lid == 3u),
                              select(0.0, 1.0, lid == 4u), select(0.0, 1.0, lid == 5u), select(0.0, 1.0, lid == 6u), select(0.0, 1.0, lid == 7u));
        // the generator in plain (unnormalized) coordinates, as lie.generators(group, null)
        let cc = vee(mul3(mul3(M, hat(x)), Mi));
        for (var j = 0u; j < nk; j++) { w_P[lid * nk + j] = -cc[j]; }
    }
    workgroupBarrier();
    if (lid < nk) {
        var sgr = 0.0;
        for (var j = 0u; j < nk; j++) { sgr = sgr + w_P[lid * nk + j] * w_B[j]; }
        st[S_G + lid] = 0.5 * (gsum(0u, lid) + sgr);
    }
    workgroupBarrier();
    // B = P A_inv
    var b = 0.0;
    if (lid < nn) {
        let i = lid / nk;
        let j = lid % nk;
        for (var x = 0u; x < nk; x++) { b = b + w_P[i * nk + x] * w_Ab[x * nk + j]; }
    }
    workgroupBarrier();
    if (lid < nn) { w_B[lid] = b; }
    workgroupBarrier();
    // A = ½ (A_fwd + B Pᵀ)
    if (lid < nn) {
        let i = lid / nk;
        let j = lid % nk;
        var c = 0.0;
        for (var y = 0u; y < nk; y++) { c = c + w_B[i * nk + y] * w_P[j * nk + y]; }
        st[S_A + lid] = 0.5 * (fwdHess(i, j) + c);
    }
}

// entry (i, j) of the forward direction's Gauss–Newton matrix from its summed tangent row
fn fwdHess(i: u32, j: u32) -> f32 {
    let q = min(i, j);
    let m = max(i, j);
    // upper-triangle index of (q, m) in the 8 × 8 layout from index 8
    return gsum(0u, 8u + q * 8u - (q * (q - 1u)) / 2u + (m - q));
}

// ── the spline's Gauss–Newton step ──
// Each lattice cell's sums (smi.wgsl ffd_cells, both directions: g0 forward, g1 inverse) are
// assembled into the gradient and the Gauss–Newton matrix over the control points (banded:
// points interact within ±3 lattice steps), the step solves (A + λR + damping) δ = ½ g − λ R c
// with R the bending energy (thin plate on the lattice, gclimb.zig bendingEnergy) and λ fixed at
// the climb's first spline step: λ = u.p1 · tr A (u.p1 from the stiffness, gclimb.zig). Candidates
// along δ are picked by score − λ cᵀ R c, the objective the step ascends.

const FFD_CELL: u32 = 332u;
const S_LAM: u32 = 4256u;   // λ (< 0: not set yet)
const S_FOK: u32 = 4257u;   // the spline step solved
const S_FD: u32 = 4272u;    // the spline step δ (up to 512)
const FFD_MUS: u32 = 6u;    // step lengths along δ

fn tri4(a: u32, b: u32) -> u32 {
    let i = min(a, b);
    let j = max(a, b);
    return i * 4u - (i * (i + 1u)) / 2u + j;
}

fn clampi(i: i32, n: u32) -> u32 { return u32(clamp(i, 0, i32(n) - 1)); }

fn cellSum(which: u32, idx: u32) -> f32 {
    return select(g0[idx], g1[idx], which == 1u);
}

// One thread per pair of control points (k, l): their 2 × 2 block of the matrix (and, k = l, the
// gradient), summed over the cells whose 4 × 4 taps reach both (edge taps clamp, as ffd_disp).
// u.mx × u.my control points, u.np = 2 · that parameters. Out (aux_out): the gradient (np), then
// the matrix (np × np, row-major).
@compute @workgroup_size(64)
fn ffd_assemble(@builtin(global_invocation_id) gid: vec3u) {
    let gx = u.mx;
    let gy = u.my;
    let ncp = gx * gy;
    let pair = gid.x;
    if (pair >= ncp * ncp) { return; }
    let k = pair / ncp;
    let l = pair % ncp;
    let ik = i32(k % gx); let jk = i32(k / gx);
    let il = i32(l % gx); let jl = i32(l / gx);
    let np = u.np;
    var bxx = 0.0; var bxy = 0.0; var byy = 0.0;
    var gxk = 0.0; var gyk = 0.0;
    let nd = select(1u, 2u, u.sym == 1u);
    if (abs(ik - il) <= 3 && abs(jk - jl) <= 3) {
        for (var w = 0u; w < nd; w++) {
            var axx = 0.0; var axy = 0.0; var ayy = 0.0; var ax = 0.0; var ay = 0.0;
            for (var iv = max(max(jk, jl) - 2, 0); iv <= min(min(jk, jl) + 1, i32(gy) - 1); iv++) {
                for (var iu = max(max(ik, il) - 2, 0); iu <= min(min(ik, il) + 1, i32(gx) - 1); iu++) {
                    let cell = u32(iv) * gx + u32(iu);
                    let o = cell * FFD_CELL;
                    for (var jj = 0u; jj < 4u; jj++) {
                        if (clampi(iv - 1 + i32(jj), gy) != u32(jk)) { continue; }
                        for (var ii = 0u; ii < 4u; ii++) {
                            if (clampi(iu - 1 + i32(ii), gx) != u32(ik)) { continue; }
                            if (k == l) {
                                let t = o + 300u + (ii + 4u * jj) * 2u;
                                ax = ax + cellSum(w, t);
                                ay = ay + cellSum(w, t + 1u);
                            }
                            for (var jj2 = 0u; jj2 < 4u; jj2++) {
                                if (clampi(iv - 1 + i32(jj2), gy) != u32(jl)) { continue; }
                                for (var ii2 = 0u; ii2 < 4u; ii2++) {
                                    if (clampi(iu - 1 + i32(ii2), gx) != u32(il)) { continue; }
                                    let q = (tri4(ii, ii2) * 10u + tri4(jj, jj2)) * 3u;
                                    axx = axx + cellSum(w, o + q);
                                    axy = axy + cellSum(w, o + q + 1u);
                                    ayy = ayy + cellSum(w, o + q + 2u);
                                }
                            }
                        }
                    }
                }
            }
            // symmetric: ½ (A_fwd + A_inv), ½ (g_fwd − g_inv) (the inverse spline is the negated one)
            let f = select(1.0, 0.5, nd == 2u);
            let sg = select(1.0, -1.0, w == 1u);
            bxx = bxx + f * axx; bxy = bxy + f * axy; byy = byy + f * ayy;
            gxk = gxk + f * sg * ax; gyk = gyk + f * sg * ay;
        }
    }
    let r = 2u * k;
    let c = 2u * l;
    aux_out[np + r * np + c] = bxx;
    aux_out[np + r * np + c + 1u] = bxy;
    aux_out[np + (r + 1u) * np + c] = bxy;
    aux_out[np + (r + 1u) * np + c + 1u] = byy;
    if (k == l) {
        aux_out[r] = select(0.0, gxk, abs(gxk) < 1e30);
        aux_out[r + 1u] = select(0.0, gyk, abs(gyk) < 1e30);
    }
}

var<workgroup> w_d: f32;
var<workgroup> w_lam: f32;

// The spline step (one workgroup): M = A + λR + damping in aux_out (np × np), the right-hand
// side ½ g − λ R c (c: the live spline, g0), a banded Cholesky (half-bandwidth hb parameters)
// in place, then the two triangular solves; δ into the state. aux_in: the assembled gradient and
// matrix, aux_in2: R.
@compute @workgroup_size(256)
fn ffd_solve(@builtin(local_invocation_index) lid: u32) {
    if (gateOff(lid)) { return; }
    let n = u.np;
    let hb = 2u * (3u * u.mx + 3u) + 1u;
    if (lid == 0u) {
        var lam = st[S_LAM];
        if (lam < 0.0) {
            var ta = 0.0;
            for (var i = 0u; i < n; i++) { ta = ta + aux_in[n + i * n + i]; }
            lam = select(0.0, u.p1 * ta, ta > 0.0);
            st[S_LAM] = lam;
        }
        w_lam = lam;
    }
    workgroupBarrier();
    let lam = w_lam;
    // M (the band only) and the right-hand side (kept in the state's δ slots)
    for (var e = lid; e < n * n; e = e + 256u) {
        let i = e / n;
        let j = e % n;
        var v = 0.0;
        if (max(i, j) - min(i, j) <= hb) { v = aux_in[n + e] + lam * aux_in2[e]; }
        aux_out[e] = v;
    }
    for (var i = lid; i < n; i = i + 256u) {
        var rc = 0.0;
        for (var j = select(0u, i - hb, i > hb); j < min(n, i + hb + 1u); j++) { rc = rc + aux_in2[i * n + j] * g0[j]; }
        st[S_FD + i] = 0.5 * aux_in[i] - lam * rc;
    }
    workgroupBarrier();
    // damping: 0.01 of each diagonal entry, and a floor of 1e-3 of their mean (Levenberg–Marquardt:
    // points few pixels reach would otherwise take huge, meaningless steps)
    if (lid == 0u) {
        var tm = 0.0;
        for (var i = 0u; i < n; i++) { tm = tm + abs(aux_out[i * n + i]); }
        let fl = 1e-3 * tm / f32(max(n, 1u)) + 1e-20;
        for (var i = 0u; i < n; i++) { aux_out[i * n + i] = aux_out[i * n + i] * (1.0 + 0.01) + fl; }
        st[S_FOK] = 1.0;
    }
    workgroupBarrier();
    // Cholesky, column by column: M = L Lᵀ (L in the lower triangle)
    for (var j = 0u; j < n; j++) {
        if (lid == 0u) {
            let d = aux_out[j * n + j];
            if (!(d > 0.0)) { st[S_FOK] = 0.0; }
            w_d = sqrt(max(d, 1e-30));
            aux_out[j * n + j] = w_d;
        }
        workgroupBarrier();
        let i1 = min(n, j + hb + 1u);
        for (var i = j + 1u + lid; i < i1; i = i + 256u) { aux_out[i * n + j] = aux_out[i * n + j] / w_d; }
        workgroupBarrier();
        // the trailing band: rows i, columns k ≤ i, both in (j, i1)
        let m = i1 - j - 1u;
        for (var e = lid; e < m * m; e = e + 256u) {
            let i = j + 1u + e / m;
            let k = j + 1u + e % m;
            if (k <= i) { aux_out[i * n + k] = aux_out[i * n + k] - aux_out[i * n + j] * aux_out[k * n + j]; }
        }
        workgroupBarrier();
    }
    // L y = b, then Lᵀ δ = y (banded, in the state's δ slots)
    if (lid == 0u) {
        for (var i = 0u; i < n; i++) {
            var s = st[S_FD + i];
            for (var k = select(0u, i - hb, i > hb); k < i; k++) { s = s - aux_out[i * n + k] * st[S_FD + k]; }
            st[S_FD + i] = s / aux_out[i * n + i];
        }
        for (var r = 0u; r < n; r++) {
            let i = n - 1u - r;
            var s = st[S_FD + i];
            for (var k = i + 1u; k < min(n, i + hb + 1u); k++) { s = s - aux_out[k * n + i] * st[S_FD + k]; }
            let v = s / aux_out[i * n + i];
            st[S_FD + i] = v;
            if (!(abs(v) < 1e30)) { st[S_FOK] = 0.0; }
        }
    }
}

var<workgroup> w_pen: array<f32, 64>;
var<workgroup> w_scale: f32;

// The spline candidates: the live points c (aux_in) moved by μ δ for μ = ½ … 16 (a wide ladder: far
// from the optimum the score is much flatter than its local Gauss–Newton model; the ladder
// shifted down so its top moves no point more than 0.2 lattice spacings: u.p0 in parameter
// units), then c
// itself (index FFD_MUS), as forward packs (aux_out) and negated (aux_out2); each one's bending
// penalty λ pᵀ R p into aux_out3 (R: aux_in2). Every pose of the stage is the current pose.
@compute @workgroup_size(64)
fn ffd_gn_cands(@builtin(local_invocation_index) lid: u32) {
    if (gateOff(lid)) { return; }
    let n = u.np;
    let hb = 2u * (3u * u.mx + 3u) + 1u;
    let ok = st[S_FOK] > 0.5;
    if (lid == 0u) {
        var mx = 0.0;
        for (var i = 0u; i < n; i++) { mx = max(mx, abs(st[S_FD + i])); }
        w_scale = select(1.0, u.p0 / (16.0 * mx), mx * 16.0 > u.p0);
    }
    workgroupBarrier();
    let mus = array<f32, 6>(0.5, 1.0, 2.0, 4.0, 8.0, 16.0);
    for (var t = 0u; t <= FFD_MUS; t++) {
        var a = 0.0;
        if (t < FFD_MUS && ok) { a = mus[t] * w_scale; }
        for (var i = lid; i < n; i = i + 64u) {
            let v = aux_in[i] + a * st[S_FD + i];
            aux_out[t * n + i] = v;
            aux_out2[t * n + i] = -v;
        }
        workgroupBarrier();
        // λ pᵀ R p
        var s = 0.0;
        for (var i = lid; i < n; i = i + 64u) {
            var rp = 0.0;
            for (var j = select(0u, i - hb, i > hb); j < min(n, i + hb + 1u); j++) { rp = rp + aux_in2[i * n + j] * aux_out[t * n + j]; }
            s = s + aux_out[t * n + i] * rp;
        }
        w_pen[lid] = s;
        workgroupBarrier();
        for (var k = 32u; k > 0u; k = k >> 1u) {
            if (lid < k) { w_pen[lid] = w_pen[lid] + w_pen[lid + k]; }
            workgroupBarrier();
        }
        if (lid == 0u) {
            aux_out3[t] = max(st[S_LAM], 0.0) * w_pen[0];
            st[S_MU + t] = a;
        }
        workgroupBarrier();
    }
    if (lid == 0u) { for (var t = 0u; t <= FFD_MUS; t++) { putCurrent(t); } }
}

// The pick's control points (pick[0] < u.n: a step won) become the live spline: forward
// into aux_out, negated into aux_out2, from the forward packs (aux_in).
@compute @workgroup_size(64)
fn ffd_apply(@builtin(local_invocation_index) lid: u32) {
    let k = u32(pick[0]);
    if (k >= u.n) { return; }
    for (var i = lid; i < u.np; i = i + 64u) {
        let v = aux_in[k * u.np + i];
        aux_out[i] = v;
        aux_out2[i] = -v;
    }
}

// ── the iteration ──

@compute @workgroup_size(1)
fn iter_begin() {
    if (st[S_DONE] > 0.5) { return; }
    st[S_MOVED] = 0.0;
    st[S_GNSTOP] = 0.0;
}

// climb.zig run: a move resets the stall count; with the shift hop on, two iterations in a row
// without a move end the climb (one without it), as does the iteration budget.
@compute @workgroup_size(1)
fn iter_end() {
    if (gated()) { return; }
    let moved = st[S_MOVED] > 0.5;
    st[S_STALL] = select(st[S_STALL] + 1.0, 0.0, moved);
    st[S_ITER] = st[S_ITER] + 1.0;
    let stop = (!moved && (u.hop == 0u || st[S_STALL] >= 2.0)) || st[S_ITER] >= f32(u.n_iters);
    if (stop) { st[S_DONE] = 1.0; }
}
