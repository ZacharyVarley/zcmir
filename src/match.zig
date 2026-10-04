//! Robust homography / affine fits from putative matches on the host (match.js): PROSAC with
//! geometric sample predicates, and a MAGSAC++-style soft-weighted variant. (Lo-FSC runs on the
//! GPU, gls.zig.) The random stream is match.js rng32 bit for bit, so seeds reproduce.
const std = @import("std");
const lie = @import("lie.zig");

const Mat3 = lie.Mat3;

pub const Method = enum { prosac, magsac };

pub const Fit = struct { H: Mat3, ninl: u32, quality: f64, trials: u32 };

const Rng = struct {
    s: u32,
    fn init(seed: u32) Rng {
        return .{ .s = if (seed == 0) 1 else seed };
    }
    fn next(self: *Rng) f64 {
        self.s = (self.s ^ (self.s >> 16)) *% 0x7feb352d;
        self.s = (self.s ^ (self.s >> 15)) *% 0x846ca68b;
        return @as(f64, @floatFromInt(self.s ^ (self.s >> 16))) / 4294967296.0;
    }
};

fn orient(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) f64 {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

fn sign(x: f64) f64 {
    return if (x > 0) 1 else if (x < 0) -1 else 0;
}

/// Same cyclic orientation on both sides, no bow-tie, no 3 collinear (match.js geometricOk).
pub fn geometricOk(px: []const f64, py: []const f64, qx: []const f64, qy: []const f64, idx: []const usize) bool {
    const m = idx.len;
    if (m < 3) return false;
    const eps = 2.0;
    for (0..m) |i| for (i + 1..m) |j| for (j + 1..m) |k| {
        const a = idx[i];
        const b = idx[j];
        const c = idx[k];
        const op = orient(px[a], py[a], px[b], py[b], px[c], py[c]);
        const oq = orient(qx[a], qy[a], qx[b], qy[b], qx[c], qy[c]);
        if (@abs(op) < eps or @abs(oq) < eps) return false;
        if (sign(op) != sign(oq)) return false;
    };
    if (m == 4) {
        for (0..4) |hide| {
            var tri: [3]usize = undefined;
            var n: usize = 0;
            for (0..4) |t| if (t != hide) {
                tri[n] = idx[t];
                n += 1;
            };
            const h = idx[hide];
            inline for (.{ .{ px, py }, .{ qx, qy } }) |xy| {
                const x = xy[0];
                const y = xy[1];
                const o1 = orient(x[tri[0]], y[tri[0]], x[tri[1]], y[tri[1]], x[h], y[h]);
                const o2 = orient(x[tri[1]], y[tri[1]], x[tri[2]], y[tri[2]], x[h], y[h]);
                const o3 = orient(x[tri[2]], y[tri[2]], x[tri[0]], y[tri[0]], x[h], y[h]);
                if ((o1 >= 0 and o2 >= 0 and o3 >= 0) or (o1 <= 0 and o2 <= 0 and o3 <= 0)) return false;
            }
        }
    }
    return true;
}

fn cheiralityOk(H: Mat3, px: []const f64, py: []const f64, idx: []const usize) bool {
    var s0: f64 = 0;
    for (idx) |i| {
        const z = H[6] * px[i] + H[7] * py[i] + H[8];
        if (@abs(z) < 1e-10) return false;
        const s: f64 = if (z > 0) 1 else -1;
        if (s0 == 0) s0 = s else if (s != s0) return false;
    }
    return true;
}

fn scaleOk(H: Mat3, lo: f64, hi: f64) bool {
    const z = if (@abs(H[8]) > 1e-12) H[8] else 1;
    const det = (H[0] / z) * (H[4] / z) - (H[1] / z) * (H[3] / z);
    if (!(det > 0)) return false;
    const sc = @sqrt(det);
    return sc >= lo and sc <= hi;
}

/// What a fitted pose may do to the moving image (settings scale_lo, scale_hi, max_aniso,
/// max_persp), and the image it is checked on (w × h, 1-based pixels).
pub const Limits = struct {
    w: f64,
    h: f64,
    scale_lo: f64 = 0.2,
    scale_hi: f64 = 5,
    /// largest stretch: the ratio of the pose's two local scales at the image centre
    max_aniso: f64 = 5,
    /// largest perspective: the ratio of the local scale between the image's corners
    max_persp: f64 = 3,
};

pub const Verdict = enum(u32) {
    ok,
    /// a coefficient is not a finite number
    not_finite,
    /// the horizon (points sent to infinity) crosses the moving image: it would fold (a bow-tie)
    horizon,
    /// the image is mirrored
    mirrored,
    scale,
    stretch,
    perspective,

    pub fn text(v: Verdict) []const u8 {
        return switch (v) {
            .ok => "ok",
            .not_finite => "not a finite pose",
            .horizon => "its horizon crosses the moving image (the image folds)",
            .mirrored => "it mirrors the image",
            .scale => "its scale is outside the scale limits",
            .stretch => "it stretches the image beyond the stretch limit",
            .perspective => "its perspective is beyond the perspective limit",
        };
    }
};

/// The pose's two local scales (singular values of its Jacobian) at (x, y), larger first.
fn localScales(H: Mat3, x: f64, y: f64) [2]f64 {
    const z = H[6] * x + H[7] * y + H[8];
    const X = (H[0] * x + H[1] * y + H[2]) / z;
    const Y = (H[3] * x + H[4] * y + H[5]) / z;
    const a = (H[0] - X * H[6]) / z;
    const b = (H[1] - X * H[7]) / z;
    const c = (H[3] - Y * H[6]) / z;
    const d = (H[4] - Y * H[7]) / z;
    const q = a * a + b * b + c * c + d * d;
    const det = @abs(a * d - b * c);
    const disc = @sqrt(@max(q * q - 4 * det * det, 0));
    return .{ @sqrt((q + disc) / 2), @sqrt(@max((q - disc) / 2, 0)) };
}

/// Whether H is a plausible pose of the moving image: finite, its horizon off the image (so the
/// image's outline stays a convex quadrilateral), not mirrored, and within the limits.
pub fn poseCheck(H: Mat3, l: Limits) Verdict {
    for (H) |v| if (!std.math.isFinite(v)) return .not_finite;
    const cs = [4][2]f64{ .{ 1, 1 }, .{ l.w, 1 }, .{ l.w, l.h }, .{ 1, l.h } };
    var zmin = std.math.inf(f64);
    var zmax = -std.math.inf(f64);
    for (cs) |c| {
        const z = H[6] * c[0] + H[7] * c[1] + H[8];
        zmin = @min(zmin, z);
        zmax = @max(zmax, z);
    }
    if (!(zmin * zmax > 0) or @min(@abs(zmin), @abs(zmax)) < 1e-9 * @max(@abs(zmin), @abs(zmax))) return .horizon;
    // the Jacobian's determinant is det(H) / z³: one sign over the image once z has one
    const det = H[0] * (H[4] * H[8] - H[5] * H[7]) - H[1] * (H[3] * H[8] - H[5] * H[6]) + H[2] * (H[3] * H[7] - H[4] * H[6]);
    if (!(det * zmin > 0)) return .mirrored;
    const mid = localScales(H, (1 + l.w) / 2, (1 + l.h) / 2);
    const sc = @sqrt(mid[0] * mid[1]);
    if (!(sc >= l.scale_lo and sc <= l.scale_hi)) return .scale;
    if (!(mid[0] <= l.max_aniso * mid[1])) return .stretch;
    var lo = std.math.inf(f64);
    var hi: f64 = 0;
    for (cs) |c| {
        const s = localScales(H, c[0], c[1]);
        const g = @sqrt(s[0] * s[1]);
        lo = @min(lo, g);
        hi = @max(hi, g);
    }
    if (!(hi <= l.max_persp * lo)) return .perspective;
    return .ok;
}

fn modelOk(H: Mat3, px: []const f64, py: []const f64, idx: []const usize, o: Options) bool {
    if (!std.math.isFinite(H[0])) return false;
    if (o.limits) |l| if (poseCheck(H, l) != .ok) return false;
    if (idx.len == 0) return scaleOk(H, o.scale_lo, o.scale_hi);
    return cheiralityOk(H, px, py, idx) and scaleOk(H, o.scale_lo, o.scale_hi);
}

pub fn reproj2(H: Mat3, px: []const f64, py: []const f64, qx: []const f64, qy: []const f64, out: []f64) void {
    for (px, py, qx, qy, 0..) |x, y, u, v, i| {
        const X = H[0] * x + H[1] * y + H[2];
        const Y = H[3] * x + H[4] * y + H[5];
        const Z = H[6] * x + H[7] * y + H[8];
        const z = if (@abs(Z) < 1e-12) 1e-12 else Z;
        const dx = X / z - u;
        const dy = Y / z - v;
        out[i] = dx * dx + dy * dy;
    }
}

fn magWeight(e2: f64, tau2: f64) f64 {
    if (e2 >= tau2) return 0;
    const x = 1 - e2 / tau2;
    return x * x;
}

const Pts = struct {
    px: []f64,
    py: []f64,
    qx: []f64,
    qy: []f64,
    fn n(self: Pts) usize {
        return self.px.len;
    }
};

fn fitPts(gpa: std.mem.Allocator, p: Pts, idx: []const usize, w: ?[]const f64, homog: bool) !Mat3 {
    const src = try gpa.alloc([2]f64, idx.len);
    defer gpa.free(src);
    const dst = try gpa.alloc([2]f64, idx.len);
    defer gpa.free(dst);
    for (idx, 0..) |i, k| {
        src[k] = .{ p.px[i], p.py[i] };
        dst[k] = .{ p.qx[i], p.qy[i] };
    }
    return if (homog) lie.HHomographyFromPts(src, dst, w) else lie.HAffineFromPts(src, dst, w);
}

fn loFit(gpa: std.mem.Allocator, p: Pts, err: []const f64, tau2: f64, homog: bool, weighted: bool, o: Options) !?Mat3 {
    var idx: std.ArrayList(usize) = .empty;
    defer idx.deinit(gpa);
    var w: std.ArrayList(f64) = .empty;
    defer w.deinit(gpa);
    for (0..p.n()) |i| {
        const wt = if (weighted) magWeight(err[i], tau2) else if (err[i] < tau2) @as(f64, 1) else 0;
        if (wt <= 0) continue;
        try idx.append(gpa, i);
        try w.append(gpa, wt);
    }
    if (idx.items.len < @as(usize, if (homog) 4 else 3)) return null;
    const H = try fitPts(gpa, p, idx.items, w.items, homog);
    return if (modelOk(H, p.px, p.py, idx.items, o)) H else null;
}

fn scoreH(H: Mat3, p: Pts, err: []f64, tau2: f64, mag: bool) struct { q: f64, ninl: u32 } {
    reproj2(H, p.px, p.py, p.qx, p.qy, err);
    var q: f64 = 0;
    var ninl: u32 = 0;
    for (err) |e| {
        if (mag) {
            const w = magWeight(e, tau2);
            q += w;
            if (w > 0) ninl += 1;
        } else if (e < tau2) {
            ninl += 1;
            q += 1;
        }
    }
    return .{ .q = q, .ninl = ninl };
}

pub const Options = struct {
    method: Method = .prosac,
    homography: bool = true,
    inlier_px: f64 = 3,
    n_trials: u32 = 2048,
    seed: u32 = 1,
    scale_lo: f64 = 0.2,
    scale_hi: f64 = 5,
    /// every candidate and refit must also pass poseCheck on the moving image
    limits: ?Limits = null,
};

/// Robust fit of q ≈ H(p); `scores` (optional, higher is better) orders PROSAC's sampling (uniform
/// sampling without them).
pub fn fitRobust(gpa: std.mem.Allocator, px0: []const f64, py0: []const f64, qx0: []const f64, qy0: []const f64, scores: ?[]const f64, o: Options) !Fit {
    const n = px0.len;
    const homog = o.homography;
    const m: usize = if (homog) 4 else 3;
    const tau2 = o.inlier_px * o.inlier_px;
    if (n < m) return .{ .H = lie.identity, .ninl = 0, .quality = 0, .trials = 0 };
    const order = try gpa.alloc(usize, n);
    defer gpa.free(order);
    for (0..n) |i| order[i] = i;
    if (scores) |s| if (s.len >= n) {
        const Ctx = struct {
            s: []const f64,
            fn lt(ctx: @This(), a: usize, b: usize) bool {
                if (ctx.s[b] != ctx.s[a]) return ctx.s[b] < ctx.s[a];
                return a < b;
            }
        };
        std.sort.pdq(usize, order, Ctx{ .s = s }, Ctx.lt);
    };
    const buf = try gpa.alloc(f64, 5 * n);
    defer gpa.free(buf);
    const p: Pts = .{ .px = buf[0..n], .py = buf[n .. 2 * n], .qx = buf[2 * n .. 3 * n], .qy = buf[3 * n .. 4 * n] };
    const err = buf[4 * n .. 5 * n];
    for (order, 0..) |k, i| {
        p.px[i] = px0[k];
        p.py[i] = py0[k];
        p.qx[i] = qx0[k];
        p.qy[i] = qy0[k];
    }
    var rng = Rng.init(o.seed);
    // PROSAC grows its sampling pool down a quality ranking; without one (the pooled per-octave
    // matches come in keypoint order, neighbours first) every sample draws from all the matches
    var pool: usize = if (scores != null) m else n;
    var Tn: f64 = 1;
    const mag = o.method == .magsac;
    var best_q: f64 = -1;
    var best_h = lie.identity;
    var best_n: u32 = 0;
    var used: u32 = 0;
    var t: u32 = 1;
    while (t <= o.n_trials) : (t += 1) {
        if (@as(f64, @floatFromInt(t)) > Tn and pool < n) {
            pool += 1;
            Tn = Tn * @as(f64, @floatFromInt(pool)) / @as(f64, @floatFromInt(@max(pool - m, 1)));
        }
        // sampleIdx(rand, m, pool, avoidLast = pool > m)
        // (the newest pool member joins every sample only while the pool grows down a ranking)
        const avoid_last = scores != null and pool > m;
        var samp: [4]usize = undefined;
        var ns: usize = 0;
        var guard: u32 = 0;
        while (ns < m and guard < 80) : (guard += 1) {
            const i: i64 = if (avoid_last and ns == m - 1)
                @as(i64, @intCast(pool)) - 1
            else
                @intFromFloat(@floor(rng.next() * @as(f64, @floatFromInt(if (avoid_last) pool - 1 else pool))));
            if (i < 0 or i >= pool) continue;
            const iu: usize = @intCast(i);
            if (std.mem.indexOfScalar(usize, samp[0..ns], iu) != null) continue;
            samp[ns] = iu;
            ns += 1;
        }
        if (ns != m) continue;
        if (!geometricOk(p.px, p.py, p.qx, p.qy, samp[0..m])) continue;
        const H = try fitPts(gpa, p, samp[0..m], null, homog);
        if (!modelOk(H, p.px, p.py, samp[0..m], o)) continue;
        used += 1;
        var cur = scoreH(H, p, err, tau2, mag);
        var Hcur = H;
        if (try loFit(gpa, p, err, tau2, homog, mag, o)) |Hlo| {
            const s2 = scoreH(Hlo, p, err, tau2, mag);
            if (s2.q >= cur.q) {
                cur = s2;
                Hcur = Hlo;
            }
        }
        if (mag) {
            reproj2(Hcur, p.px, p.py, p.qx, p.qy, err);
            if (try loFit(gpa, p, err, tau2, homog, true, o)) |H2| {
                const s3 = scoreH(H2, p, err, tau2, mag);
                if (s3.q >= cur.q) {
                    cur = s3;
                    Hcur = H2;
                }
            }
        }
        if (cur.q <= best_q) continue;
        best_q = cur.q;
        best_h = Hcur;
        best_n = cur.ninl;
    }
    if (best_n > 0) {
        reproj2(best_h, p.px, p.py, p.qx, p.qy, err);
        if (try loFit(gpa, p, err, tau2, homog, mag, o)) |Hf| best_h = Hf;
        best_n = scoreH(best_h, p, err, tau2, false).ninl;
    }
    return .{ .H = best_h, .ninl = best_n, .quality = best_q, .trials = used };
}

/// Descriptor dot products of each query's match (−1e9 where unmatched), as f32 like match.js.
pub fn matchDots(desA: []const f32, desB: []const f32, mj: []const u32, n1: usize, dim: usize, out: []f32) void {
    for (0..n1) |i| {
        const j = mj[i];
        if (j == 0xffffffff) {
            out[i] = -1e9;
            continue;
        }
        var d: f64 = 0;
        for (0..dim) |k| d += desA[i * dim + k] * desB[j * dim + k];
        out[i] = @floatCast(d);
    }
}
