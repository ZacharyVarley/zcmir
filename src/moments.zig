//! Scores from the 45 overlap moment sums (host, f64): SMI (smi.js scoreFromMoments) and the
//! Shannon-calibrated copula scores E4 and λmax (copula.js). Moment layout (smi.wgsl):
//!   0 n = Σ w_a w_b · 1..4 Σ a_i · 5..8 Σ b_j · 9..24 Σ a_i b_j (i*4+j)
//!   25..34 Σ a_i a_j (upper triangle) · 35..44 Σ b_i b_j
//! Coefficients (per pose, for the gradient kernels): W (16), μ_B (4), M̂ (16), μ_A (4),
//! g = ∂S/∂sums (45) at G0, the Gauss–Newton weight Q (16) at Q0, a generic-score flag at FLAG.
const std = @import("std");

pub const N_MOM = 45;
pub const N_COEF = 102;
pub const G0 = 40;
pub const Q0 = 85;
pub const FLAG = 101;
pub const MIN_N: f64 = 12;
pub const RIDGE: f64 = 1e-4;

pub const Family = enum(u32) { smi = 0, e4 = 1, lmax = 2 };

pub const Stats = struct {
    score: f64,
    n: f64,
    coef: [N_COEF]f32,
};

pub fn triK(i: usize, j: usize) usize {
    const a = @min(i, j);
    const b = @max(i, j);
    return a * 4 - (a * (a + 1)) / 2 + b;
}

const M4 = [16]f64;

fn chol4(G: M4) ?M4 {
    var L: M4 = @splat(0);
    for (0..4) |i| for (0..i + 1) |j| {
        var s = G[i * 4 + j];
        for (0..j) |k| s -= L[i * 4 + k] * L[j * 4 + k];
        if (i == j) {
            if (!(s > 0)) return null;
            L[i * 5] = @sqrt(s);
        } else L[i * 4 + j] = s / L[j * 5];
    };
    return L;
}

fn cholInv4(L: M4) M4 {
    var X: M4 = @splat(0);
    for (0..4) |c| {
        var y: [4]f64 = @splat(0);
        for (0..4) |i| {
            var s: f64 = if (i == c) 1 else 0;
            for (0..i) |k| s -= L[i * 4 + k] * y[k];
            y[i] = s / L[i * 5];
        }
        var i: usize = 4;
        while (i > 0) {
            i -= 1;
            var s = y[i];
            for (i + 1..4) |k| s -= L[k * 4 + i] * X[k * 4 + c];
            X[i * 4 + c] = s / L[i * 5];
        }
    }
    return X;
}

fn mm4(A: M4, B: M4) M4 {
    var o: M4 = @splat(0);
    for (0..4) |i| for (0..4) |j| {
        var s: f64 = 0;
        for (0..4) |k| s += A[i * 4 + k] * B[k * 4 + j];
        o[i * 4 + j] = s;
    };
    return o;
}

fn tr4(A: M4) M4 {
    var o: M4 = undefined;
    for (0..4) |i| for (0..4) |j| {
        o[i * 4 + j] = A[j * 4 + i];
    };
    return o;
}

/// SMI score n·Φ and its gradient coefficients (smi.js scoreFromMoments).
///   global: Φ = ‖C‖²;  exact: Φ = tr(G_A⁻¹ C G_B⁻¹ Cᵀ) with ridge-regularized covariances.
pub fn scoreFromMoments(st: *const [N_MOM]f64, exact: bool, ridge: f64) Stats {
    const n = st[0];
    var out: Stats = .{ .score = 0, .n = n, .coef = @splat(0) };
    if (!(n >= MIN_N)) return out;
    var ma: [4]f64 = undefined;
    var mb: [4]f64 = undefined;
    var C: M4 = undefined;
    for (0..4) |i| {
        ma[i] = st[1 + i] / n;
        mb[i] = st[5 + i] / n;
    }
    for (0..4) |i| for (0..4) |j| {
        C[i * 4 + j] = st[9 + i * 4 + j] / n - ma[i] * mb[j];
    };
    var W = C;
    var Mh: M4 = @splat(0);
    var Nh: M4 = @splat(0);
    var phi: f64 = 0;
    if (!exact) {
        for (C) |v| phi += v * v;
    } else {
        var GA: M4 = undefined;
        var GB: M4 = undefined;
        for (0..4) |i| for (0..4) |j| {
            const k = triK(i, j);
            GA[i * 4 + j] = st[25 + k] / n - ma[i] * ma[j];
            GB[i * 4 + j] = st[35 + k] / n - mb[i] * mb[j];
        };
        const ra = ridge * @max((GA[0] + GA[5] + GA[10] + GA[15]) / 4, 1e-12);
        const rb = ridge * @max((GB[0] + GB[5] + GB[10] + GB[15]) / 4, 1e-12);
        for (0..4) |i| {
            GA[i * 5] += ra;
            GB[i * 5] += rb;
        }
        const LA = chol4(GA) orelse return out;
        const LB = chol4(GB) orelse return out;
        const GAi = cholInv4(LA);
        const GBi = cholInv4(LB);
        W = mm4(mm4(GAi, C), GBi);
        for (0..16) |i| phi += W[i] * C[i];
        Mh = mm4(mm4(W, tr4(C)), GAi);
        Nh = mm4(mm4(tr4(W), C), GBi);
        const tm = (Mh[0] + Mh[5] + Mh[10] + Mh[15]) * ridge / 4;
        const tn = (Nh[0] + Nh[5] + Nh[10] + Nh[15]) * ridge / 4;
        for (0..4) |i| {
            Mh[i * 5] += tm;
            Nh[i * 5] += tn;
        }
    }
    const score = n * phi;
    if (!std.math.isFinite(score)) return out;
    out.score = score;
    for (0..16) |i| out.coef[i] = @floatCast(W[i]);
    for (0..4) |i| out.coef[16 + i] = @floatCast(mb[i]);
    for (0..16) |i| out.coef[20 + i] = @floatCast(Mh[i]);
    for (0..4) |i| out.coef[36 + i] = @floatCast(ma[i]);
    var g: [N_MOM]f64 = @splat(0);
    for (0..4) |i| {
        var ga: f64 = 0;
        var gb: f64 = 0;
        for (0..4) |j| {
            ga += -2 * W[i * 4 + j] * mb[j] + 2 * Mh[i * 4 + j] * ma[j];
            gb += -2 * W[j * 4 + i] * ma[j] + 2 * Nh[i * 4 + j] * mb[j];
            g[9 + i * 4 + j] = 2 * W[i * 4 + j];
        }
        g[1 + i] = ga;
        g[5 + i] = gb;
        for (i..4) |j| {
            const k = triK(i, j);
            const f: f64 = if (i == j) 1 else 2;
            g[25 + k] = -f * Mh[i * 4 + j];
            g[35 + k] = -f * Nh[i * 4 + j];
        }
    }
    var rest: f64 = 0;
    for (1..N_MOM) |k| rest += g[k] * st[k];
    g[0] = (score - rest) / n;
    for (0..N_MOM) |k| out.coef[G0 + k] = @floatCast(g[k]);
    const Q = if (exact) Mh else mm4(C, tr4(C));
    for (0..16) |i| out.coef[Q0 + i] = @floatCast(Q[i]);
    return out;
}

// ── copula scores (copula.js) ────────────────────────────────────────────────────────────
fn jointNegentropy(w30: f64, w21: f64, w12: f64, w03: f64, w40: f64, w31: f64, w22: f64, w13: f64, w04: f64) f64 {
    const s = struct {
        fn f(x: f64) f64 {
            return x * x;
        }
    }.f;
    return 0.145833333333333 * s(s(w03)) - 0.125 * s(w03) * w04 + 0.875 * s(w03) * s(w12) + 0.125 * s(w03) * s(w21) +
        s(w03) / 12 + 1.5 * w03 * s(w12) * w21 - 0.5 * w03 * w12 * w13 + 0.25 * w03 * w12 * w21 * w30 +
        w03 * w21 * s(w21) / 3 - 0.25 * w03 * w21 * w22 + s(w04) / 48 - 0.125 * w04 * s(w12) + 0.5625 * s(s(w12)) +
        w12 * s(w12) * w30 / 3 + 2 * s(w12) * s(w21) - 0.5 * s(w12) * w22 + 0.125 * s(w12) * s(w30) + 0.25 * s(w12) -
        0.5 * w12 * w13 * w21 + 1.5 * w12 * s(w21) * w30 - 0.5 * w12 * w21 * w31 - 0.25 * w12 * w22 * w30 +
        s(w13) / 12 + 0.5625 * s(s(w21)) - 0.5 * s(w21) * w22 + 0.875 * s(w21) * s(w30) - 0.125 * s(w21) * w40 +
        0.25 * s(w21) - 0.5 * w21 * w30 * w31 + 0.125 * s(w22) + 0.145833333333333 * s(s(w30)) -
        0.125 * s(w30) * w40 + s(w30) / 12 + s(w31) / 12 + s(w40) / 48;
}

fn marginalNegentropy(l3: f64, l4: f64) f64 {
    return (l3 * l3) / 12 + (l4 * l4) / 48 - (l3 * l3 * l4) / 8 + (7 * std.math.pow(f64, l3, 4)) / 48;
}

const BIN = [5][5]f64{ .{ 1, 0, 0, 0, 0 }, .{ 1, 1, 0, 0, 0 }, .{ 1, 2, 1, 0, 0 }, .{ 1, 3, 3, 1, 0 }, .{ 1, 4, 6, 4, 1 } };

fn ipow(x: f64, p: usize) f64 {
    var r: f64 = 1;
    for (0..p) |_| r *= x;
    return r;
}

/// E4 MI (nats) from the sums; NaN when undefined.
pub fn e4FromSums(st: *const [N_MOM]f64) f64 {
    const n = st[0];
    if (!(n >= MIN_N)) return std.math.nan(f64);
    const R = struct {
        fn f(s: *const [N_MOM]f64, nn: f64, p: usize, q: usize) f64 {
            if (p == 0 and q == 0) return 1;
            if (q == 0) return s[p] / nn;
            if (p == 0) return s[4 + q] / nn;
            return s[9 + (p - 1) * 4 + (q - 1)] / nn;
        }
    }.f;
    const mx = R(st, n, 1, 0);
    const my = R(st, n, 0, 1);
    const mu = struct {
        fn f(s: *const [N_MOM]f64, nn: f64, mxx: f64, myy: f64, p: usize, q: usize) f64 {
            var v: f64 = 0;
            for (0..p + 1) |i| for (0..q + 1) |j| {
                v += BIN[p][i] * BIN[q][j] * R(s, nn, i, j) * ipow(-mxx, p - i) * ipow(-myy, q - j);
            };
            return v;
        }
    }.f;
    const k20 = mu(st, n, mx, my, 2, 0);
    const k02 = mu(st, n, mx, my, 0, 2);
    const k11 = mu(st, n, mx, my, 1, 1);
    if (!(k20 > 1e-12 and k02 > 1e-12)) return std.math.nan(f64);
    const sx = @sqrt(k20);
    const sy = @sqrt(k02);
    const rho = k11 / (sx * sy);
    const det = 1 - rho * rho;
    if (!(det > 1e-6)) return std.math.nan(f64);
    // cumulants k[p][q] for p + q ∈ {3, 4}
    var k: [5][5]f64 = @splat(@splat(0));
    k[3][0] = mu(st, n, mx, my, 3, 0);
    k[2][1] = mu(st, n, mx, my, 2, 1);
    k[1][2] = mu(st, n, mx, my, 1, 2);
    k[0][3] = mu(st, n, mx, my, 0, 3);
    k[4][0] = mu(st, n, mx, my, 4, 0) - 3 * k20 * k20;
    k[3][1] = mu(st, n, mx, my, 3, 1) - 3 * k20 * k11;
    k[2][2] = mu(st, n, mx, my, 2, 2) - k20 * k02 - 2 * k11 * k11;
    k[1][3] = mu(st, n, mx, my, 1, 3) - 3 * k02 * k11;
    k[0][4] = mu(st, n, mx, my, 0, 4) - 3 * k02 * k02;
    var gg: [5][5]f64 = undefined;
    for (0..5) |p| for (0..5) |q| {
        gg[p][q] = k[p][q] / (ipow(sx, p) * ipow(sy, q));
    };
    const d = @sqrt(det);
    const w = struct {
        fn f(g: *const [5][5]f64, r: f64, dd: f64, p: usize, q: usize) f64 {
            var v: f64 = 0;
            for (0..q + 1) |jj| v += BIN[q][jj] * ipow(-r, q - jj) * g[p + q - jj][jj];
            return v / ipow(dd, q);
        }
    }.f;
    const jj = jointNegentropy(w(&gg, rho, d, 3, 0), w(&gg, rho, d, 2, 1), w(&gg, rho, d, 1, 2), w(&gg, rho, d, 0, 3), w(&gg, rho, d, 4, 0), w(&gg, rho, d, 3, 1), w(&gg, rho, d, 2, 2), w(&gg, rho, d, 1, 3), w(&gg, rho, d, 0, 4));
    return -0.5 * @log(det) + jj - marginalNegentropy(gg[3][0], gg[4][0]) - marginalNegentropy(gg[0][3], gg[0][4]);
}

fn chol3(G: [9]f64) ?[9]f64 {
    var L: [9]f64 = @splat(0);
    for (0..3) |i| for (0..i + 1) |j| {
        var s = G[i * 3 + j];
        for (0..j) |kk| s -= L[i * 3 + kk] * L[j * 3 + kk];
        if (i == j) {
            if (!(s > 0)) return null;
            L[i * 3 + i] = @sqrt(s);
        } else L[i * 3 + j] = s / L[j * 3 + j];
    };
    return L;
}

/// Largest eigenvalue of a symmetric 3×3 (closed form).
pub fn topEig3(S: [9]f64) f64 {
    const p1 = S[1] * S[1] + S[2] * S[2] + S[5] * S[5];
    const q = (S[0] + S[4] + S[8]) / 3;
    const p2 = (S[0] - q) * (S[0] - q) + (S[4] - q) * (S[4] - q) + (S[8] - q) * (S[8] - q) + 2 * p1;
    if (p2 < 1e-30) return q;
    const p = @sqrt(p2 / 6);
    var B: [9]f64 = undefined;
    for (0..9) |i| B[i] = (S[i] - (if (i % 4 == 0) q else 0)) / p;
    const detB = B[0] * (B[4] * B[8] - B[5] * B[7]) - B[1] * (B[3] * B[8] - B[5] * B[6]) + B[2] * (B[3] * B[7] - B[4] * B[6]);
    const phi = std.math.acos(@max(-1, @min(1, detB / 2))) / 3;
    return q + 2 * p * @cos(phi);
}

/// λ²max's parts from the sums: the moving side's covariance factor L_A and S = M Mᵀ, M =
/// L_A⁻¹ C L_B⁻ᵀ the whitened cross-covariance (degree-3 features); null where undefined.
const LmaxParts = struct { LA: [9]f64, S: [9]f64 };
fn lmaxParts(st: *const [N_MOM]f64, ridge: f64) ?LmaxParts {
    const n = st[0];
    if (!(n >= MIN_N)) return null;
    const ma = [3]f64{ st[1] / n, st[2] / n, st[3] / n };
    const mb = [3]f64{ st[5] / n, st[6] / n, st[7] / n };
    var GA: [9]f64 = undefined;
    var GB: [9]f64 = undefined;
    var Cr: [9]f64 = undefined;
    for (0..3) |i| for (0..3) |j| {
        GA[i * 3 + j] = st[25 + triK(i, j)] / n - ma[i] * ma[j];
        GB[i * 3 + j] = st[35 + triK(i, j)] / n - mb[i] * mb[j];
        Cr[i * 3 + j] = st[9 + i * 4 + j] / n - ma[i] * mb[j];
    };
    for (0..3) |i| {
        GA[i * 4] *= 1 + ridge;
        GB[i * 4] *= 1 + ridge;
    }
    const LA = chol3(GA) orelse return null;
    const LB = chol3(GB) orelse return null;
    var X: [9]f64 = @splat(0);
    var M: [9]f64 = @splat(0);
    for (0..3) |j| for (0..3) |i| {
        var s = Cr[i * 3 + j];
        for (0..i) |kk| s -= LA[i * 3 + kk] * X[kk * 3 + j];
        X[i * 3 + j] = s / LA[i * 4];
    };
    for (0..3) |r| for (0..3) |i| {
        var s = X[r * 3 + i];
        for (0..i) |kk| s -= LB[i * 3 + kk] * M[r * 3 + kk];
        M[r * 3 + i] = s / LB[i * 4];
    };
    var S: [9]f64 = undefined;
    for (0..3) |i| for (0..3) |j| {
        var s: f64 = 0;
        for (0..3) |kk| s += M[i * 3 + kk] * M[j * 3 + kk];
        S[i * 3 + j] = s;
    };
    return .{ .LA = LA, .S = S };
}

/// λmax MI (nats) from the sums; NaN when undefined.
pub fn lmaxFromSums(st: *const [N_MOM]f64, ridge: f64) f64 {
    const p = lmaxParts(st, ridge) orelse return std.math.nan(f64);
    const l2 = @min(@max(topEig3(p.S), 0), 1 - 1e-6);
    return -0.5 * @log(1 - l2);
}

/// λmax's Gauss–Newton weight on the moving features (Q0, 4 × 4; copula.wgsl copula_lmax_q):
/// λ² a aᵀ / (2 (1 − λ²)), a = L_A⁻ᵀ u the moving side's top canonical direction.
pub fn lmaxQ(st: *const [N_MOM]f64, ridge: f64) [16]f64 {
    var Q: [16]f64 = @splat(0);
    const p = lmaxParts(st, ridge) orelse return Q;
    const S = p.S;
    const l2 = @min(@max(topEig3(S), 0), 1 - 1e-6);
    const rows = [3][3]f64{ .{ S[0] - l2, S[1], S[2] }, .{ S[3], S[4] - l2, S[5] }, .{ S[6], S[7], S[8] - l2 } };
    const cross = struct {
        fn f(a: [3]f64, b: [3]f64) [3]f64 {
            return .{ a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
        }
    }.f;
    const dot = struct {
        fn f(a: [3]f64, b: [3]f64) f64 {
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        }
    }.f;
    var u = cross(rows[0], rows[1]);
    for ([_][3]f64{ cross(rows[0], rows[2]), cross(rows[1], rows[2]) }) |c| {
        if (dot(c, c) > dot(u, u)) u = c;
    }
    const sc = @max(@abs(S[0]) + @abs(S[4]) + @abs(S[8]), 1e-300);
    if (!(dot(u, u) > 1e-20 * sc * sc * sc * sc)) {
        // a repeated top eigenvalue: power iteration (any vector of its space will do)
        u = .{ 1, 1, 1 };
        for (0..64) |_| {
            const v = [3]f64{ S[0] * u[0] + S[1] * u[1] + S[2] * u[2], S[3] * u[0] + S[4] * u[1] + S[5] * u[2], S[6] * u[0] + S[7] * u[1] + S[8] * u[2] };
            const nv = @max(@sqrt(dot(v, v)), 1e-300);
            u = .{ v[0] / nv, v[1] / nv, v[2] / nv };
        }
    }
    const nu = @sqrt(dot(u, u));
    u = .{ u[0] / nu, u[1] / nu, u[2] / nu };
    var a: [3]f64 = undefined;
    var r: usize = 0;
    while (r < 3) : (r += 1) {
        const i = 2 - r;
        var s = u[i];
        for (i + 1..3) |k| s -= p.LA[k * 3 + i] * a[k];
        a[i] = s / p.LA[i * 4];
    }
    const c = l2 / (2 * (1 - l2));
    for (0..3) |i| for (0..3) |j| {
        Q[i * 4 + j] = c * a[i] * a[j];
    };
    for (Q) |q| if (!std.math.isFinite(q)) return @splat(0);
    return Q;
}

pub fn copulaScore(st: *const [N_MOM]f64, fam: Family) f64 {
    const I = if (fam == .e4) e4FromSums(st) else lmaxFromSums(st, RIDGE);
    return if (std.math.isFinite(I)) st[0] * I else 0;
}

/// Copula score and generic gradient coefficients (central differences in the sums); for
/// λmax also its Gauss–Newton weight (lmaxQ).
pub fn copulaFromMoments(st: *const [N_MOM]f64, fam: Family, with_grad: bool) Stats {
    const n = st[0];
    var out: Stats = .{ .score = copulaScore(st, fam), .n = n, .coef = @splat(0) };
    if (!with_grad or !(n >= MIN_N) or out.score == 0) return out;
    var g: [N_MOM]f64 = @splat(0);
    var x = st.*;
    const used: usize = if (fam == .e4) 25 else N_MOM;
    for (0..used) |k| {
        const h = 1e-5 * @max(@max(@abs(st[k]), 1e-3 * n), 1e-6);
        x[k] = st[k] + h;
        const up = copulaScore(&x, fam);
        x[k] = st[k] - h;
        const dn = copulaScore(&x, fam);
        x[k] = st[k];
        g[k] = (up - dn) / (2 * h);
    }
    for (0..4) |i| for (0..4) |j| {
        out.coef[i * 4 + j] = @floatCast(0.5 * g[9 + i * 4 + j]);
        const kk = triK(i, j);
        out.coef[20 + i * 4 + j] = @floatCast(if (i == j) -g[25 + kk] else -0.5 * g[25 + kk]);
    };
    for (0..N_MOM) |k| out.coef[G0 + k] = @floatCast(g[k]);
    out.coef[FLAG] = 1;
    if (fam == .lmax) {
        const Q = lmaxQ(st, RIDGE);
        for (0..16) |k| out.coef[Q0 + k] = @floatCast(Q[k]);
    }
    return out;
}

/// Σ over `groups` rows of `stride` partial sums (f32 from the GPU) into f64.
pub fn sumPart(raw: []const f32, groups: usize, stride: usize, out: []f64) void {
    @memset(out[0..stride], 0);
    for (0..groups) |g| for (0..stride) |i| {
        out[i] += raw[g * stride + i];
    };
}
