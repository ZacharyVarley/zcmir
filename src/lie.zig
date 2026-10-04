//! Pose algebra (port of lie.js). Conventions:
//!   H: row-major 3×3, MOVING pixel → FIXED pixel, 1-based (centre of the top-left pixel = (1, 1)).
//!   Tangent coordinates ξ = [tx, ty, th, sg, al, ga, px, py]: translation, rotation, log-scale,
//!   anisotropic stretch, shear, perspective, as left increments in normalized fixed coordinates
//!   (destN). Affine uses the first 6; homography all 8.
const std = @import("std");

pub const Mat3 = [9]f64;
pub const Group = enum(u32) { affine = 0, homography = 1 };

pub const TX = 0;
pub const TY = 1;
pub const TH = 2;
pub const SG = 3;
pub const AL = 4;
pub const GA = 5;
pub const PX = 6;
pub const PY = 7;
/// Tangent coordinates, always 8 wide (affine ignores px, py).
pub const Coords = [8]f64;

pub fn nKeys(g: Group) usize {
    return if (g == .homography) 8 else 6;
}

pub const identity: Mat3 = .{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };

pub fn destN(w0: f64, h0: f64) Mat3 {
    const w = @max(w0, 1);
    const h = @max(h0, 1);
    const sx = 0.5 * w;
    const sy = 0.5 * h;
    const cx = 0.5 * (w + 1);
    const cy = 0.5 * (h + 1);
    return .{ 1 / sx, 0, -cx / sx, 0, 1 / sy, -cy / sy, 0, 0, 1 };
}

/// JavaScript's Math.hypot of two numbers (V8: scaled by the larger, Kahan-summed squares), so
/// host code ported from the app rounds the same way.
pub fn jsHypot(a: f64, b: f64) f64 {
    return jsHypotN(&.{ a, b });
}

/// Math.hypot(...xs).
pub fn jsHypotN(xs: []const f64) f64 {
    var mx: f64 = 0;
    var nan = false;
    for (xs) |x| {
        if (std.math.isNan(x)) {
            nan = true;
        } else mx = @max(mx, @abs(x));
    }
    if (std.math.isInf(mx)) return mx;
    if (nan) return std.math.nan(f64);
    if (mx == 0) return 0;
    var sum: f64 = 0;
    var comp: f64 = 0;
    for (xs) |x| {
        const n = @abs(x) / mx;
        const summand = n * n - comp;
        const pre = sum + summand;
        comp = (pre - sum) - summand;
        sum = pre;
    }
    return @sqrt(sum) * mx;
}

/// JavaScript's Math.round: halves toward +∞.
pub fn jsRound(x: f64) f64 {
    const r = @floor(x);
    return if (x - r >= 0.5) r + 1 else r;
}

/// MATLAB round: halves away from zero.
pub fn mround(x: f64) f64 {
    const s: f64 = if (x > 0) 1 else if (x < 0) -1 else 0;
    return s * @floor(@abs(x) + 0.5);
}

pub fn mul3(a: Mat3, b: Mat3) Mat3 {
    var o: Mat3 = undefined;
    for (0..3) |r| for (0..3) |c| {
        o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
    };
    return o;
}

fn sign(x: f64) f64 {
    return if (x > 0) 1 else if (x < 0) -1 else 0;
}

/// Inverse with one Newton refinement step (as lie.js inv3).
pub fn inv3(m: Mat3) Mat3 {
    const a = m[0];
    const b = m[1];
    const c = m[2];
    const d = m[3];
    const e = m[4];
    const f = m[5];
    const g = m[6];
    const h = m[7];
    const i = m[8];
    const A = e * i - f * h;
    const B = f * g - d * i;
    const C = d * h - e * g;
    const D = c * h - b * i;
    const E = a * i - c * g;
    const F = b * g - a * h;
    const G = b * f - c * e;
    const Hh = c * d - a * f;
    const I = a * e - b * d;
    const det = a * A + b * B + c * C;
    const s = 1 / (sign(det) * @max(@abs(det), 1e-18));
    const inv: Mat3 = .{ A * s, D * s, G * s, B * s, E * s, Hh * s, C * s, F * s, I * s };
    const p = mul3(m, inv);
    const corr: Mat3 = .{ 2 - p[0], -p[1], -p[2], -p[3], 2 - p[4], -p[5], -p[6], -p[7], 2 - p[8] };
    return mul3(inv, corr);
}

pub fn hat(x: Coords, group: Group) Mat3 {
    const tx = x[TX];
    const ty = x[TY];
    const th = x[TH];
    const sg = x[SG];
    const al = x[AL];
    const ga = x[GA];
    var px = x[PX];
    var py = x[PY];
    var iso: f64 = undefined;
    var z: f64 = undefined;
    if (group != .homography) {
        px = 0;
        py = 0;
        z = 0;
        iso = sg;
    } else {
        iso = sg / 3;
        z = -2 * iso;
    }
    return .{ iso + al, ga - th, tx, ga + th, iso - al, ty, px, py, z };
}

pub fn expm3(A: Mat3) Mat3 {
    var nrm: f64 = 0;
    for (A) |v| nrm = @max(nrm, @abs(v));
    var s: u32 = 0;
    if (nrm > 0.5) s = @intFromFloat(@min(10, @max(0, @ceil(std.math.log2(nrm / 0.5)))));
    const sc = std.math.pow(f64, 0.5, @floatFromInt(s));
    var B: Mat3 = undefined;
    for (0..9) |i| B[i] = A[i] * sc;
    var T = identity;
    var P = identity;
    var k: u32 = 1;
    while (k < 14) : (k += 1) {
        P = mul3(P, B);
        const kf: f64 = @floatFromInt(k);
        for (&P) |*v| v.* /= kf;
        for (0..9) |i| T[i] += P[i];
    }
    var i: u32 = 0;
    while (i < s) : (i += 1) T = mul3(T, T);
    return T;
}

pub fn projectGroup(H: Mat3, group: Group) Mat3 {
    const s = if (@abs(H[8]) > 1e-12) H[8] else 1;
    var o: Mat3 = undefined;
    for (0..9) |i| o[i] = H[i] / s;
    if (group != .homography) {
        o[6] = 0;
        o[7] = 0;
        o[8] = 1;
    }
    return o;
}

pub fn compose(H0: Mat3, x: Coords, group: Group) Mat3 {
    return projectGroup(mul3(expm3(hat(x, group)), H0), group);
}

/// exp(ξ̂) applied in normalized fixed coordinates of a dw×dh image.
pub fn composeN(H0: Mat3, x: Coords, group: Group, dw: f64, dh: f64) Mat3 {
    const N = destN(dw, dh);
    return projectGroup(mul3(mul3(mul3(inv3(N), expm3(hat(x, group))), N), H0), group);
}

pub fn vee(A0: Mat3, group: Group) Coords {
    var a = A0[0];
    const b = A0[1];
    const c = A0[2];
    const d = A0[3];
    var e = A0[4];
    const f = A0[5];
    const g = A0[6];
    const h = A0[7];
    const i = A0[8];
    if (group == .homography) {
        const tr = (a + e + i) / 3;
        a -= tr;
        e -= tr;
        const iso = (a + e) / 2;
        return .{ c, f, (d - b) / 2, iso * 3, (a - e) / 2, (d + b) / 2, g, h };
    }
    return .{ c, f, (d - b) / 2, (a + e) / 2, (a - e) / 2, (d + b) / 2, 0, 0 };
}

/// Generators of the tangent space (one per key), optionally in pixel coordinates of a
/// dw×dh destination (N⁻¹ G N).
pub fn generators(group: Group, dest: ?[2]f64) [8]Mat3 {
    var out: [8]Mat3 = undefined;
    const nk = nKeys(group);
    for (0..nk) |k| {
        var x: Coords = @splat(0);
        x[k] = 1;
        out[k] = hat(x, group);
    }
    for (nk..8) |k| out[k] = @splat(0);
    if (dest) |wh| {
        const N = destN(wh[0], wh[1]);
        const Ni = inv3(N);
        for (0..nk) |k| out[k] = mul3(mul3(Ni, out[k]), N);
    }
    return out;
}

/// dS/dξ on H, given dS/dη on H⁻¹ (η: left increment in the moving frame).
pub fn adjointGrad(gEta: []const f64, H: Mat3, group: Group, movW: f64, movH: f64, fixW: f64, fixH: f64) Coords {
    const Hi = inv3(H);
    const Nm = destN(movW, movH);
    const Nf = destN(fixW, fixH);
    const M = mul3(mul3(Nm, Hi), inv3(Nf));
    const Mi = inv3(M);
    const gens = generators(group, null);
    const nk = nKeys(group);
    var out: Coords = @splat(0);
    for (0..nk) |i| {
        const cc = vee(mul3(mul3(M, gens[i]), Mi), group);
        for (0..nk) |j| out[i] -= gEta[j] * cc[j];
    }
    return out;
}

pub fn unitAscent(g: []const f64, group: Group) Coords {
    const nk = nKeys(group);
    var u: Coords = @splat(0);
    var n: f64 = 0;
    for (0..nk) |i| {
        u[i] = g[i];
        n += u[i] * u[i];
    }
    n = @sqrt(n);
    if (n < 1e-18) return u;
    for (0..nk) |i| u[i] /= n;
    return u;
}

/// Jacobi eigendecomposition of a symmetric n×n (row-major); vec[:, j] is eigenvector j.
pub fn symEig(A: []const f64, n: usize, val: []f64, vec: []f64) void {
    var M: [64]f64 = undefined;
    @memcpy(M[0 .. n * n], A[0 .. n * n]);
    @memset(vec[0 .. n * n], 0);
    for (0..n) |i| vec[i * n + i] = 1;
    var it: u32 = 0;
    while (it < 64) : (it += 1) {
        var p: usize = 0;
        var q: usize = 1;
        var best: f64 = 0;
        for (0..n) |i| for (i + 1..n) |j| {
            const a = @abs(M[i * n + j]);
            if (a > best) {
                best = a;
                p = i;
                q = j;
            }
        };
        if (best < 1e-14) break;
        const app = M[p * n + p];
        const aqq = M[q * n + q];
        const apq = M[p * n + q];
        const tau = (aqq - app) / (2 * apq);
        const t = (if (tau >= 0) @as(f64, 1) else -1) / (@abs(tau) + @sqrt(1 + tau * tau));
        const c = 1 / @sqrt(1 + t * t);
        const s = t * c;
        for (0..n) |k| {
            if (k == p or k == q) continue;
            const mkp = M[k * n + p];
            const mkq = M[k * n + q];
            M[k * n + p] = c * mkp - s * mkq;
            M[p * n + k] = M[k * n + p];
            M[k * n + q] = s * mkp + c * mkq;
            M[q * n + k] = M[k * n + q];
        }
        M[p * n + p] = app - t * apq;
        M[q * n + q] = aqq + t * apq;
        M[p * n + q] = 0;
        M[q * n + p] = 0;
        for (0..n) |k| {
            const vkp = vec[k * n + p];
            const vkq = vec[k * n + q];
            vec[k * n + p] = c * vkp - s * vkq;
            vec[k * n + q] = s * vkp + c * vkq;
        }
    }
    for (0..n) |i| val[i] = M[i * n + i];
}

pub fn translate(dx: f64, dy: f64) Mat3 {
    return .{ 1, 0, dx, 0, 1, dy, 0, 0, 1 };
}

pub fn mapPt(H: Mat3, x: f64, y: f64) [2]f64 {
    const X = H[0] * x + H[1] * y + H[2];
    const Y = H[3] * x + H[4] * y + H[5];
    const Z = H[6] * x + H[7] * y + H[8];
    const z = if (@abs(Z) < 1e-8) 1e-8 else Z;
    return .{ X / z, Y / z };
}

pub fn movingCentroid(H: Mat3, lw: f64, lh: f64) [2]f64 {
    const cs = [4][2]f64{ .{ 1, 1 }, .{ lw, 1 }, .{ lw, lh }, .{ 1, lh } };
    var sx: f64 = 0;
    var sy: f64 = 0;
    for (cs) |p| {
        const q = mapPt(H, p[0], p[1]);
        sx += q[0];
        sy += q[1];
    }
    return .{ sx * 0.25, sy * 0.25 };
}

/// Destination-pixel similarity about c: X ↦ c + e^σ R_θ (X − c).
pub fn simAbout(c: [2]f64, th: f64, sg: f64) Mat3 {
    const s = @exp(sg);
    const co = @cos(th);
    const si = @sin(th);
    const a = s * co;
    const b = -s * si;
    const d = s * si;
    const e = s * co;
    return .{ a, b, c[0] - a * c[0] - b * c[1], d, e, c[1] - d * c[0] - e * c[1], 0, 0, 1 };
}

pub fn composeSimC(H0: Mat3, c: [2]f64, th: f64, sg: f64, group: Group) Mat3 {
    return projectGroup(mul3(simAbout(c, th, sg), H0), group);
}

/// Solve A x = b (n×n row-major, partial pivoting); null if singular or non-finite.
pub fn solveLinear(A: []const f64, b: []const f64, n: usize, x: []f64) bool {
    var M: [64]f64 = undefined;
    @memcpy(M[0 .. n * n], A[0 .. n * n]);
    @memcpy(x[0..n], b[0..n]);
    for (0..n) |i| {
        var piv = i;
        for (i + 1..n) |r| if (@abs(M[r * n + i]) > @abs(M[piv * n + i])) {
            piv = r;
        };
        if (piv != i) {
            for (0..n) |j| std.mem.swap(f64, &M[i * n + j], &M[piv * n + j]);
            std.mem.swap(f64, &x[i], &x[piv]);
        }
        const d = M[i * n + i];
        if (!std.math.isFinite(d) or @abs(d) < 1e-300) return false;
        for (0..n) |r| if (r != i) {
            const f = M[r * n + i] / d;
            if (f == 0) continue;
            for (i..n) |j| M[r * n + j] -= f * M[i * n + j];
            x[r] -= f * x[i];
        };
    }
    for (0..n) |i| {
        x[i] /= M[i * n + i];
        if (!std.math.isFinite(x[i])) return false;
    }
    return true;
}

/// Solve with Gauss–Jordan as smi.js solveNk (normalizes each pivot row); false if singular.
pub fn solveNk(A: []const f64, b: []const f64, n: usize, x: []f64) bool {
    var M: [64]f64 = undefined;
    @memcpy(M[0 .. n * n], A[0 .. n * n]);
    @memcpy(x[0..n], b[0..n]);
    for (0..n) |i| {
        var piv = i;
        for (i + 1..n) |r| if (@abs(M[r * n + i]) > @abs(M[piv * n + i])) {
            piv = r;
        };
        if (piv != i) {
            for (0..n) |j| std.mem.swap(f64, &M[i * n + j], &M[piv * n + j]);
            std.mem.swap(f64, &x[i], &x[piv]);
        }
        const d = M[i * n + i];
        if (!std.math.isFinite(d) or @abs(d) < 1e-18) return false;
        for (i..n) |j| M[i * n + j] /= d;
        x[i] /= d;
        for (0..n) |r| if (r != i) {
            const f = M[r * n + i];
            for (i..n) |j| M[r * n + j] -= f * M[i * n + j];
            x[r] -= f * x[i];
        };
    }
    for (0..n) |i| if (!std.math.isFinite(x[i])) return false;
    return true;
}

/// Weighted least-squares affine H from point pairs (normal equations, as lie.js).
pub fn HAffineFromPts(src: []const [2]f64, dst: []const [2]f64, wts: ?[]const f64) Mat3 {
    var ata: [36]f64 = @splat(0);
    var atb: [6]f64 = @splat(0);
    var rows: usize = 0;
    for (src, dst, 0..) |s, d, i| {
        const wt = if (wts) |w| @sqrt(@max(w[i], 0)) else 1;
        if (wt <= 0) continue;
        const r1 = [6]f64{ wt * s[0], wt * s[1], wt, 0, 0, 0 };
        const r2 = [6]f64{ 0, 0, 0, wt * s[0], wt * s[1], wt };
        for (0..6) |j| {
            atb[j] += r1[j] * wt * d[0] + r2[j] * wt * d[1];
            for (0..6) |k| ata[j * 6 + k] += r1[j] * r1[k] + r2[j] * r2[k];
        }
        rows += 2;
    }
    if (rows < 6) return identity;
    // Gauss–Jordan with d = pivot || 1e-12 (lie.js solve)
    for (0..6) |i| {
        var piv = i;
        for (i + 1..6) |r| if (@abs(ata[r * 6 + i]) > @abs(ata[piv * 6 + i])) {
            piv = r;
        };
        if (piv != i) {
            for (0..6) |j| std.mem.swap(f64, &ata[i * 6 + j], &ata[piv * 6 + j]);
            std.mem.swap(f64, &atb[i], &atb[piv]);
        }
        const d = if (ata[i * 6 + i] != 0) ata[i * 6 + i] else 1e-12;
        for (i..6) |j| ata[i * 6 + j] /= d;
        atb[i] /= d;
        for (0..6) |r| if (r != i) {
            const f = ata[r * 6 + i];
            for (i..6) |j| ata[r * 6 + j] -= f * ata[i * 6 + j];
            atb[r] -= f * atb[i];
        };
    }
    return .{ atb[0], atb[1], atb[2], atb[3], atb[4], atb[5], 0, 0, 1 };
}

/// Hartley normalization of weighted points: (cx, cy, s) with s·(p − c) of mean length √2.
fn hartley(pts: []const [2]f64, wts: ?[]const f64) [3]f64 {
    var sw: f64 = 0;
    var cx: f64 = 0;
    var cy: f64 = 0;
    for (pts, 0..) |p, i| {
        const wt = if (wts) |w| w[i] else 1;
        if (wt <= 0) continue;
        sw += wt;
        cx += wt * p[0];
        cy += wt * p[1];
    }
    if (sw <= 0) return .{ 0, 0, 1 };
    cx /= sw;
    cy /= sw;
    var d: f64 = 0;
    for (pts, 0..) |p, i| {
        const wt = if (wts) |w| w[i] else 1;
        if (wt > 0) d += wt * std.math.hypot(p[0] - cx, p[1] - cy);
    }
    return .{ cx, cy, std.math.sqrt2 / @max(d / sw, 1e-12) };
}

/// Homography from point pairs: the DLT on Hartley-normalized coordinates (in pixels its normal
/// matrix is too ill-conditioned to solve), smallest eigenvector by inverse iteration (lie.js
/// HHomographyFromPts).
pub fn HHomographyFromPts(src: []const [2]f64, dst: []const [2]f64, wts: ?[]const f64) Mat3 {
    const na = hartley(src, wts);
    const nb = hartley(dst, wts);
    var ata: [81]f64 = @splat(0);
    for (src, dst, 0..) |s, d, i| {
        const wt = if (wts) |w| w[i] else 1;
        if (wt <= 0) continue;
        const x = na[2] * (s[0] - na[0]);
        const y = na[2] * (s[1] - na[1]);
        const u = nb[2] * (d[0] - nb[0]);
        const v = nb[2] * (d[1] - nb[1]);
        const rows = [2][9]f64{
            .{ -x, -y, -1, 0, 0, 0, u * x, u * y, u },
            .{ 0, 0, 0, -x, -y, -1, v * x, v * y, v },
        };
        for (rows) |row| for (0..9) |j| for (0..9) |k| {
            ata[j * 9 + k] += wt * row[j] * row[k];
        };
    }
    for (0..9) |i| ata[i * 9 + i] += 1e-12;
    var v: [9]f64 = @splat(0);
    v[8] = 1;
    var it: u32 = 0;
    while (it < 48) : (it += 1) {
        var x: [9]f64 = undefined;
        solveSquare9(&ata, &v, &x);
        var nrm: f64 = 0;
        for (x) |e| nrm += e * e;
        nrm = @sqrt(nrm);
        if (nrm == 0) nrm = 1;
        for (0..9) |j| v[j] = x[j] / nrm;
    }
    // back to pixels: Tb⁻¹ · Hn · Ta
    const Ta: Mat3 = .{ na[2], 0, -na[2] * na[0], 0, na[2], -na[2] * na[1], 0, 0, 1 };
    const Tbi: Mat3 = .{ 1 / nb[2], 0, nb[0], 0, 1 / nb[2], nb[1], 0, 0, 1 };
    return projectGroup(mul3(Tbi, mul3(v, Ta)), .homography);
}

fn solveSquare9(A0: *const [81]f64, b0: *const [9]f64, out: *[9]f64) void {
    var A = A0.*;
    var b = b0.*;
    for (0..9) |i| {
        var piv = i;
        for (i + 1..9) |r| if (@abs(A[r * 9 + i]) > @abs(A[piv * 9 + i])) {
            piv = r;
        };
        if (piv != i) {
            for (0..9) |j| std.mem.swap(f64, &A[i * 9 + j], &A[piv * 9 + j]);
            std.mem.swap(f64, &b[i], &b[piv]);
        }
        const d = if (A[i * 9 + i] != 0) A[i * 9 + i] else 1e-18;
        for (i..9) |j| A[i * 9 + j] /= d;
        b[i] /= d;
        for (0..9) |r| if (r != i) {
            const f = A[r * 9 + i];
            for (i..9) |j| A[r * 9 + j] -= f * A[i * 9 + j];
            b[r] -= f * b[i];
        };
    }
    out.* = b;
}

// ── canvas geometry ──────────────────────────────────────────────────────────────────────
pub const Frame = struct { ox: i32, oy: i32, w: u32, h: u32 };

/// Canvas of a pose: the fixed image plus the moving image's warped corners, clipped to
/// ±2·max(size).
pub fn regFrame(H: Mat3, lw: u32, lh: u32, rw: u32, rh: u32) Frame {
    const fl = struct {
        fn f(v: u32) f64 {
            return @floatFromInt(v);
        }
    }.f;
    const corners = [4][2]f64{ .{ 1, 1 }, .{ fl(lw), 1 }, .{ fl(lw), fl(lh) }, .{ 1, fl(lh) } };
    var xs_min: f64 = 0;
    var ys_min: f64 = 0;
    var xs_max: f64 = fl(rw);
    var ys_max: f64 = fl(rh);
    const lim = @max(@max(fl(lw), fl(rw)), @max(@max(fl(lh), fl(rh)), 8)) * 2;
    for (corners) |p| {
        const x = H[0] * p[0] + H[1] * p[1] + H[2];
        const y = H[3] * p[0] + H[4] * p[1] + H[5];
        const z = H[6] * p[0] + H[7] * p[1] + H[8];
        const zz = if (@abs(z) < 1e-8) 1e-8 else z;
        xs_min = @min(xs_min, x / zz - 1);
        xs_max = @max(xs_max, x / zz - 1);
        ys_min = @min(ys_min, y / zz - 1);
        ys_max = @max(ys_max, y / zz - 1);
    }
    const clip = struct {
        fn f(v: f64, l: f64) f64 {
            return @min(l, @max(-l, v));
        }
    }.f;
    const ox = @min(@floor(clip(xs_min, lim)), 0);
    const oy = @min(@floor(clip(ys_min, lim)), 0);
    const x1 = @max(@ceil(clip(xs_max, lim)), fl(rw));
    const y1 = @max(@ceil(clip(ys_max, lim)), fl(rh));
    return .{
        .ox = @intFromFloat(ox),
        .oy = @intFromFloat(oy),
        .w = @intFromFloat(@max(8, x1 - ox)),
        .h = @intFromFloat(@max(8, y1 - oy)),
    };
}

/// H in canvas pixels: moving → canvas.
pub fn canvasH(H: Mat3, ox: i32, oy: i32) Mat3 {
    return mul3(translate(-@as(f64, @floatFromInt(ox)), -@as(f64, @floatFromInt(oy))), H);
}

test "inv3 · H = I" {
    const H: Mat3 = .{ 1.01, 0.012, -3.06, -0.006, 1.008, 4.86, 1e-5, 2e-5, 1 };
    const P = mul3(inv3(H), H);
    for (0..9) |i| try std.testing.expectApproxEqAbs(identity[i], P[i], 1e-12);
}

test "compose / vee round trip" {
    const x: Coords = .{ 0.1, -0.2, 0.03, 0.02, 0.01, -0.01, 1e-4, -2e-4 };
    const A = hat(x, .homography);
    const y = vee(A, .homography);
    for (0..8) |i| try std.testing.expectApproxEqAbs(x[i], y[i], 1e-12);
}
