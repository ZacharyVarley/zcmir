//! Square-loss mutual information (SMI) on the GPU: features per image, and the dense shift map.
//!
//! The statistics between kernels (image mean, spread, whitening moments, feature
//! autocorrelation, the map peak) are finished on the GPU into small parameter buffers, so preparing an image or computing a map is one submit with no
//! synchronization, and only the requested results are read.
const std = @import("std");
const mom = @import("moments.zig");
const gpu_mod = @import("gpu/gpu.zig");
const shaders = @import("shaders");
const fft = @import("fft.zig");
const lie = @import("lie.zig");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Bind = gpu_mod.Bind;

pub const MIN_N: u32 = 12;
pub const RIDGE: f32 = 1e-4;
const N_BINS = 256;
const IFFT_CHUNK: u32 = 8;
/// Map FFT-length cap (the browser app's default map resolution).
pub const MAP_CAP: u32 = 1024;

/// Automatic map resolution (MapOptions.cap 0, the default). A score peak is a few percent of
/// the image wide (its basin, not a delta), so a grid of a few hundred cells across locates it;
/// the climb refines within the cell. The canvas is resampled by the smallest power-of-two
/// factor that fits its longer side in a linear-correlation FFT of MAP_AUTO_MAX, and the FFT is
/// the next power of two (at least MAP_AUTO_MIN). The roto-scale map takes MAP_AUTO_MIN–MAX angles.
pub const MAP_AUTO_MIN: u32 = 128;
pub const MAP_AUTO_MAX: u32 = 512;

/// (nx, ny, cw, ch) of the automatic shift map for a w × h canvas: one grid factor for both
/// sides (square cells), and each side's FFT the next power of two of its own linear
/// correlation (MAP_AUTO_MIN–MAP_AUTO_MAX), so a long thin canvas no longer pays for a square.
pub fn autoShiftPlan(w: u32, h: u32) [4]u32 {
    const d = @max(w, h, 1);
    var g: u32 = 1;
    while (2 * ((d + g - 1) / g) - 1 > MAP_AUTO_MAX) g *= 2;
    const cw = @max(1, (w + g - 1) / g);
    const ch = @max(1, (h + g - 1) / g);
    const pow2 = struct {
        fn f(need: u32) u32 {
            return std.math.clamp(std.math.ceilPowerOfTwoAssert(u32, @max(need, 1)), MAP_AUTO_MIN, MAP_AUTO_MAX);
        }
    }.f;
    return .{ pow2(2 * cw - 1), pow2(2 * ch - 1), cw, ch };
}

/// smi.wgsl `U` (64 bytes), with the defaults of smi.js packU.
pub const U = extern struct {
    w: u32,
    h: u32,
    n: u32 = 1,
    cw: u32 = 1,
    ch: u32 = 1,
    plane: u32 = 0,
    plane_b: u32 = 0,
    min_n: u32 = MIN_N,
    mean: f32 = 0,
    stdv: f32 = 1,
    scale: f32 = 1,
    ridge: f32 = RIDGE,
    gx: u32 = 0,
    gy: u32 = 0,
    theme: u32 = 0,
    n_keys: u32 = 0,
};
comptime {
    std.debug.assert(@sizeOf(U) == 64);
}

/// Real-plane correlations a map needs (smi.js corrPlan): 25 (global whitening) or 45 (exact).
const CorrPlan = struct {
    n_real: u32,
    n_corr: u32,
    n_out: u32,
    pairs: [23 * 6]u32,

    fn make(exact: bool) CorrPlan {
        var list: [45][2]u32 = undefined;
        var len: usize = 0;
        list[len] = .{ 0, 0 };
        len += 1;
        for (0..4) |i| {
            list[len] = .{ @intCast(1 + i), 0 };
            len += 1;
        }
        for (0..4) |j| {
            list[len] = .{ 0, @intCast(1 + j) };
            len += 1;
        }
        for (0..4) |i| for (0..4) |j| {
            list[len] = .{ @intCast(1 + i), @intCast(1 + j) };
            len += 1;
        };
        if (exact) {
            for (0..10) |k| {
                list[len] = .{ @intCast(5 + k), 0 };
                len += 1;
            }
            for (0..10) |k| {
                list[len] = .{ 0, @intCast(5 + k) };
                len += 1;
            }
        }
        var p: CorrPlan = .{ .n_real = if (exact) 15 else 5, .n_corr = @intCast(len), .n_out = @intCast((len + 1) / 2), .pairs = undefined };
        for (0..p.n_out) |o| {
            const o1 = 2 * o;
            const o2 = o1 + 1;
            p.pairs[o * 6 + 0] = list[o1][0];
            p.pairs[o * 6 + 1] = list[o1][1];
            p.pairs[o * 6 + 2] = @intCast(o1);
            if (o2 < len) {
                p.pairs[o * 6 + 3] = list[o2][0];
                p.pairs[o * 6 + 4] = list[o2][1];
                p.pairs[o * 6 + 5] = @intCast(o2);
            } else {
                p.pairs[o * 6 + 3] = 0;
                p.pairs[o * 6 + 4] = 0;
                p.pairs[o * 6 + 5] = 0xffffffff;
            }
        }
        return p;
    }
};

/// One image's SMI features, resident on the GPU.
pub const Features = struct {
    w: u32 = 0,
    h: u32 = 0,
    /// normal-score features [z, z², z³, z⁴] (copula scores E4 / λmax) instead of whitened ranks
    normal: bool = false,
    /// 4 whitened planes [r, r², r³, |∇r|]
    feat: Buf = .{},
    /// Sobel magnitude of the rank (edge weight)
    mag: Buf = .{},
    /// normalized autocorrelation, 289 lags × 4 planes
    acf: Buf = .{},
    /// premultiplied stack [w, w f0..w f3], w = 1
    stack: Buf = .{},
    /// the same with w = the Sobel magnitude ("SMI (edge)"), built on first use
    stack_edge: Buf = .{},

    pub fn deinit(self: *Features, g: *Gpu) void {
        g.release(&self.feat);
        g.release(&self.mag);
        g.release(&self.acf);
        g.release(&self.stack);
        g.release(&self.stack_edge);
    }
};

pub const MapOptions = struct {
    family: mom.Family = .smi,
    /// exact per-overlap whitening (n Σρ²) vs one global whitening (SMI)
    exact: bool = true,
    /// Pillai / Wilson–Hilferty z instead of n Σρ² (exact SMI only)
    calibrated: bool = true,
    /// FFT-length cap (map resolution); 0: automatic (autoShiftPlan)
    cap: u32 = 0,
    /// shift maps: lags whose overlap is below this fraction of the grid score 0
    min_overlap: f64 = 0,
    /// edge-weighted stacks ("SMI (edge)")
    edge: bool = false,

    /// the correlation plan's exactness (E4: 25 moments, λmax: 45)
    pub fn planExact(o: MapOptions) bool {
        return switch (o.family) {
            .e4 => false,
            .lmax => true,
            .smi => o.exact,
        };
    }
    /// combine mode: 0 global SMI, 1 exact SMI, 2 E4, 3 λmax
    pub fn mode(o: MapOptions) u32 {
        return switch (o.family) {
            .e4 => 2,
            .lmax => 3,
            .smi => @intFromBool(o.exact),
        };
    }
    pub fn calib(o: MapOptions) bool {
        return o.family == .smi and o.exact and o.calibrated;
    }
};

/// A shift map's grid: n (width) × ny (height) cells, lag 0 at index 0.
pub const ShiftPlan = struct {
    n: u32,
    ny: u32,
    cw: u32,
    ch: u32,
    frame: lie.Frame,
    exact: bool,
    calibrated: bool,
};

/// A roto-scale map's grid: θ over the FFT width n (nTh = n), log ρ over nLam rows from
/// log0 over logSpan (Δλ = dlam), about (cx, cy) in canvas pixels of `frame`.
pub const RsPlan = struct {
    n: u32,
    n_th: u32,
    n_lam: u32,
    log0: f64,
    log_span: f64,
    dlam: f64,
    r0: f64,
    r1: f64,
    cx: f64,
    cy: f64,
    frame: lie.Frame,
    calibrated: bool,
};

pub const Smi = struct {
    g: *Gpu,
    hw_max_n: u32,
    plans: [2]CorrPlan,
    pairs: [2]Buf,
    /// parameters finished on the GPU: [0] mean, [1] 1.0, [2] spread
    prm: Buf,
    /// [0] the pair's feature correlation area (px²)
    pair_prm: Buf,
    part: Buf,
    stats: Buf,
    linv: Buf,
    hist: Buf,
    cdf: Buf,
    arg: Buf,
    /// [0] peak, [1] bitcast peak index, [2] zero-lag score
    res: Buf,
    z: Buf = .{},
    rank: Buf = .{},
    wgt: Buf = .{},
    cstack: Buf = .{},
    cstack_b: Buf = .{},
    spec_a: Buf = .{},
    spec_b: Buf = .{},
    work: Buf = .{},
    corr: Buf = .{},
    ns: Buf = .{},
    /// roto-scale: log-polar stacks, the inverse-frame map, the finished map
    lp_a: Buf = .{},
    lp_b: Buf = .{},
    ns_inv: Buf = .{},
    ns_rs: Buf = .{},
    /// a stack resampled onto the map grid (pack_resample): five nx × ny planes
    smp: Buf = .{},
    /// the maps' 2-D FFTs by (nx << 32 | ny)
    ffts: std.AutoHashMapUnmanaged(u64, fft.BatchFft) = .empty,

    pub fn init(g: *Gpu) !Smi {
        var s: Smi = .{
            .g = g,
            .hw_max_n = fft.maxFftN(g.lim.max_workgroup_storage),
            .plans = .{ CorrPlan.make(false), CorrPlan.make(true) },
            .pairs = undefined,
            .prm = g.storage(16 * 4),
            .pair_prm = g.storage(16),
            .part = g.storage(64 * 96 * 4),
            .stats = g.storage(24 * 4),
            .linv = g.storage(24 * 4),
            .hist = g.storage((N_BINS + 1) * 4),
            .cdf = g.storage((N_BINS + 2) * 4),
            .arg = g.storage(64 * 2 * 4),
            .res = g.storage(4 * 4),
        };
        for (0..2) |k| {
            const p = &s.plans[k];
            s.pairs[k] = g.storage(p.n_out * 6 * 4);
            g.writeSlice(s.pairs[k], 0, u32, p.pairs[0 .. p.n_out * 6]);
        }
        g.writeSlice(s.prm, 0, f32, &.{ 0, 1, 1, 0 });
        return s;
    }

    pub fn deinit(self: *Smi) void {
        const g = self.g;
        for (&self.pairs) |*b| g.release(b);
        inline for (.{ "prm", "pair_prm", "part", "stats", "linv", "hist", "cdf", "arg", "res", "z", "rank", "wgt", "cstack", "cstack_b", "spec_a", "spec_b", "work", "corr", "ns", "lp_a", "lp_b", "ns_inv", "ns_rs", "smp" }) |f|
            g.release(&@field(self, f));
        var it = self.ffts.valueIterator();
        while (it.next()) |f| f.deinit(g);
        self.ffts.deinit(g.gpa);
    }

    pub fn pipe(self: *Smi, comptime entry: []const u8) u32 {
        return self.g.pipeline("smi/" ++ entry, shaders.smi, entry) catch |e| {
            std.log.err("smi.wgsl {s}: {s}", .{ entry, @errorName(e) });
            return gpu_mod.NO_PIPE;
        };
    }

    pub fn un(self: *Smi, u: U) Bind {
        return self.g.uniform(0, std.mem.asBytes(&u));
    }

    /// A flat_index kernel over `groups` workgroups (folded past 65535), in z layers.
    pub fn run1(self: *Smi, comptime entry: []const u8, groups: u32, z: u32, binds: []const Bind) void {
        self.g.dispatchFlat(self.pipe(entry), groups, z, binds);
    }

    pub fn run(self: *Smi, comptime entry: []const u8, x: u32, y: u32, z: u32, binds: []const Bind) void {
        self.g.dispatch(self.pipe(entry), x, y, z, binds);
    }

    /// Rank-equalize, build [r, r², r³, |∇r|], whiten them globally, and precompute the
    /// autocorrelation and the weight-1 stack. `gray`: w×h f32. Recorded, not submitted.
    pub fn prepare(self: *Smi, gray: Buf, w: u32, h: u32, out: *Features, normal: bool) void {
        const g = self.g;
        const n = w * h;
        const n256 = (n + 255) / 256;
        out.deinit(g);
        out.* = .{
            .w = w,
            .h = h,
            .normal = normal,
            .feat = g.storage(@as(u64, n) * 16),
            .mag = g.storage(@as(u64, n) * 4),
            .acf = g.storage(289 * 4 * 4),
            .stack = g.storage(@as(u64, n) * 20),
        };
        for ([_]*Buf{ &self.z, &self.rank, &self.wgt }) |b| g.ensure(b, @as(u64, n) * 4);
        const ub: U = .{ .w = w, .h = h };

        // mean → prm[0]
        self.run("reduce1", 64, 1, 1, &.{ self.un(ub), gray.at(1), self.part.at(4) });
        self.run("finish_sum", 1, 1, 1, &.{ self.un(.{ .w = w, .h = h, .plane = 0, .scale = 1.0 / @as(f32, @floatFromInt(@max(n, 1))) }), self.part.at(1), self.prm.at(4) });
        // spread of the centered image → prm[2] (moments of plane 0 of a zeroed feature stack)
        self.run1("zscore_p", n256, 1, &.{ self.un(.{ .w = w, .h = h, .plane = 0, .plane_b = 1 }), gray.at(1), self.prm.at(2), out.mag.at(4) });
        g.clear(out.feat);
        g.copy(out.mag, 0, out.feat, 0, @as(u64, n) * 4);
        self.run("feat_moments", 64, 1, 1, &.{ self.un(ub), out.feat.at(1), self.part.at(4) });
        self.run("finish_moments", 1, 1, 1, &.{ self.part.at(1), self.stats.at(4) });
        self.run("finish_std", 1, 1, 1, &.{ self.un(.{ .w = w, .h = h, .plane = 2 }), self.stats.at(1), self.prm.at(4) });
        // z-score, histogram rank, features
        self.run1("zscore_p", n256, 1, &.{ self.un(.{ .w = w, .h = h, .plane = 0, .plane_b = 2 }), gray.at(1), self.prm.at(2), self.z.at(4) });
        g.clear(self.hist);
        self.run1("hist_bins", n256, 1, &.{ self.un(ub), self.z.at(1), self.hist.at(6) });
        self.run("cdf_scan", 1, 1, 1, &.{ self.un(ub), self.hist.at(6), self.cdf.at(7) });
        self.run1("rank_eq", n256, 1, &.{ self.un(ub), self.z.at(1), self.rank.at(4), self.hist.at(6), self.cdf.at(7) });
        if (normal) {
            // normal-score powers, left unwhitened (the copula scores are written in them)
            self.run("feats_normal", (w + 7) / 8, (h + 7) / 8, 1, &.{ self.un(ub), self.rank.at(1), out.feat.at(4), out.mag.at(5) });
        } else {
            self.run("feats", (w + 7) / 8, (h + 7) / 8, 1, &.{ self.un(ub), self.rank.at(1), out.feat.at(4), out.mag.at(5) });
            // global ridge whitening
            self.run("feat_moments", 64, 1, 1, &.{ self.un(ub), out.feat.at(1), self.part.at(4) });
            self.run("finish_moments", 1, 1, 1, &.{ self.part.at(1), self.stats.at(4) });
            self.run("chol", 1, 1, 1, &.{ self.un(.{ .w = w, .h = h }), self.stats.at(1), self.linv.at(4) });
            self.run1("whiten", n256, 1, &.{ self.un(ub), self.linv.at(2), out.feat.at(4) });
        }
        // autocorrelation (for the correlation area) and the weight-1 stack
        self.run("feat_acf", 289, 1, 1, &.{ self.un(ub), out.feat.at(1), self.part.at(4) });
        self.run("acf_norm", (289 * 4 + 255) / 256, 1, 1, &.{ self.part.at(1), out.acf.at(4) });
        self.run1("fill", n256, 1, &.{ self.un(.{ .w = w, .h = h, .scale = 1 }), self.wgt.at(4) });
        self.run1("stack_planes", n256, 1, &.{ self.un(ub), out.feat.at(1), self.wgt.at(2), out.stack.at(4) });
    }

    /// Feature correlation area of a pair (px²) into pair_prm[0].
    pub fn pairArea(self: *Smi, a: *const Features, b: *const Features) void {
        self.run("corr_area", 1, 1, 1, &.{ self.un(.{ .w = 1, .h = 1, .plane = 0 }), a.acf.at(1), b.acf.at(2), self.pair_prm.at(4) });
    }

    fn capN(self: *Smi, plan: *const CorrPlan, cap0: u32) u32 {
        const bind = self.g.lim.max_storage_binding;
        var cap: u32 = @min(cap0, self.hw_max_n);
        while (cap > 64) : (cap >>= 1) {
            const c2: u64 = @as(u64, cap) * cap;
            if (@max(c2 * 4 * plan.n_corr, c2 * 8 * IFFT_CHUNK) <= bind) break;
        }
        return fft.largestSmooth(cap, self.g.fft_sizes);
    }

    /// The batched 2-D FFT of nx × ny planes (maps are at most hw_max_n per side, which it
    /// always fits).
    fn fft2(self: *Smi, nx: u32, ny: u32) !*fft.BatchFft {
        const key = (@as(u64, nx) << 32) | ny;
        const gop = try self.ffts.getOrPut(self.g.gpa, key);
        if (!gop.found_existing) {
            gop.value_ptr.* = fft.BatchFft.initRect(self.g, nx, ny) catch |e| {
                _ = self.ffts.remove(key);
                return e;
            };
        }
        return gop.value_ptr;
    }

    /// The premultiplied stack of `f`: weight 1, or the Sobel magnitude for edge weighting.
    pub fn stackOf(self: *Smi, f: *Features, edge: bool) Buf {
        if (!edge) return f.stack;
        if (f.stack_edge.id == 0) {
            const g = self.g;
            const n = f.w * f.h;
            g.ensure(&self.wgt, @as(u64, n) * 4);
            f.stack_edge = g.storage(@as(u64, n) * 20);
            self.run1("weight", (n + 255) / 256, 1, &.{ self.un(.{ .w = f.w, .h = f.h, .scale = 1 }), f.mag.at(1), self.wgt.at(4) });
            self.run1("stack_planes", (n + 255) / 256, 1, &.{ self.un(.{ .w = f.w, .h = f.h }), f.feat.at(1), self.wgt.at(2), f.stack_edge.at(4) });
        }
        return f.stack_edge;
    }

    /// Pose stacks on the canvas of H: the moving stack warped by the canvas H into `cstack`, the
    /// fixed stack pasted at (−ox, −oy) into `cstack_b` (engine.js poseStacks).
    fn poseStacks(self: *Smi, a: *Features, b: *Features, H: lie.Mat3, edge: bool) lie.Frame {
        const fr = lie.regFrame(H, a.w, a.h, b.w, b.h);
        const Hc = lie.canvasH(H, fr.ox, fr.oy);
        var h32: [9]f32 = undefined;
        for (0..9) |k| h32[k] = @floatCast(Hc[k]);
        self.poseStacksOn(a, b, fr, self.g.stage(3, std.mem.asBytes(&h32)), edge);
        return fr;
    }

    /// The canvas stacks of frame `fr` with the moving image warped by the canvas pose bound as
    /// `hc` (9 floats at slot 3: a staged host pose, or one a GPU kernel wrote).
    fn poseStacksOn(self: *Smi, a: *Features, b: *Features, fr: lie.Frame, hc: Bind, edge: bool) void {
        const g = self.g;
        const cn: u64 = @as(u64, fr.w) * fr.h;
        g.ensure(&self.cstack, cn * 20);
        g.ensure(&self.cstack_b, cn * 20);
        const sa = self.stackOf(a, edge);
        const sb = self.stackOf(b, edge);
        const cgx = (fr.w + 7) / 8;
        const cgy = (fr.h + 7) / 8;
        self.run("warp_stack", cgx, cgy, 1, &.{ self.un(.{ .w = fr.w, .h = fr.h, .cw = a.w, .ch = a.h }), sa.at(1), hc, self.cstack.at(4) });
        self.run("paste_stack", cgx, cgy, 1, &.{ self.un(.{ .w = fr.w, .h = fr.h, .cw = b.w, .ch = b.h, .gx = @bitCast(-fr.ox), .gy = @bitCast(-fr.oy) }), sb.at(1), self.cstack_b.at(4) });
    }

    const LogPolar = struct { log0: f64, span: f64, dlam: f64, dth: f64 };

    /// Resample a stack onto the nx × ny grid, build its real correlation planes and
    /// forward-FFT them, two per complex plane. lp: log-polar grid (every plane × ρ, the area
    /// element).
    fn packFft(self: *Smi, stack: Buf, w: u32, h: u32, nx: u32, ny: u32, cw: u32, ch: u32, dst: Buf, plan: *const CorrPlan, f: *fft.BatchFft, lp: ?LogPolar) void {
        const n_cplx = (plan.n_real + 1) / 2;
        const ub: U = .{
            .w = w, .h = h, .n = nx, .gy = ny, .cw = cw, .ch = ch, .n_keys = plan.n_real, .theme = @intFromBool(lp != null),
            .mean = if (lp) |l| @floatCast(l.log0) else 0, .stdv = if (lp) |l| @floatCast(l.span) else 1,
        };
        // the stack on the grid once, then every correlation plane from it
        self.g.ensure(&self.smp, @as(u64, nx) * ny * 5 * 4);
        self.run("pack_resample", (cw + 7) / 8, (ch + 7) / 8, 1, &.{ self.un(ub), stack.at(1), self.smp.at(5) });
        self.run("pack_pair", (nx + 7) / 8, (ny + 7) / 8, n_cplx, &.{ self.un(ub), self.smp.at(3), dst.at(4) });
        f.run(self.g, dst, false, n_cplx);
    }

    /// spec_a ⋆ spec_b → correlation planes → per-lag score into `dst` (smi.js correlate).
    /// area_scale: cells per px² for the correlation area (1 / g² on a shift grid, 1 log-polar).
    fn correlate(self: *Smi, nx: u32, ny: u32, plan: *const CorrPlan, opt: MapOptions, min_n: u32, lp: ?LogPolar, area_scale: f64, dst: Buf, f: *fft.BatchFft) void {
        const g = self.g;
        const nn: u64 = @as(u64, nx) * ny;
        const ncell: u32 = @intCast((nn + 255) / 256);
        const pairs = self.pairs[@intFromBool(opt.planExact())];
        var c0: u32 = 0;
        while (c0 < plan.n_out) : (c0 += IFFT_CHUNK) {
            const k = @min(IFFT_CHUNK, plan.n_out - c0);
            self.run1("cmul_pair", ncell, k, &.{ self.un(.{ .w = nx, .h = ny, .n = nx, .gy = ny, .plane = c0 }), self.spec_a.at(1), self.spec_b.at(2), pairs.at(3), self.work.at(4) });
            f.run(g, self.work, true, k);
            self.run1("extract_pair", ncell, k, &.{ self.un(.{
                .w = nx, .h = ny, .n = nx, .gy = ny, .plane = c0, .scale = if (lp != null) 2 else 0,
                .mean = if (lp) |l| @floatCast(l.dlam) else 0, .stdv = if (lp) |l| @floatCast(l.dth) else 1,
            }), self.work.at(1), pairs.at(3), self.corr.at(4) });
        }
        self.run1("combine_p", ncell, 1, &.{
            self.un(.{ .w = nx, .h = ny, .n = nx, .gy = ny, .min_n = min_n, .gx = opt.mode(), .ridge = RIDGE, .theme = @intFromBool(opt.calib()), .plane = 0, .scale = @floatCast(area_scale) }),
            self.corr.at(1),
            self.pair_prm.at(3),
            dst.at(4),
        });
    }

    fn ensureCorr(self: *Smi, nx: u32, ny: u32, plan: *const CorrPlan) void {
        const g = self.g;
        const nn: u64 = @as(u64, nx) * ny;
        const n_cplx: u64 = (plan.n_real + 1) / 2;
        g.ensure(&self.spec_a, nn * 8 * n_cplx);
        g.ensure(&self.spec_b, nn * 8 * n_cplx);
        g.ensure(&self.work, nn * 8 * IFFT_CHUNK);
        g.ensure(&self.corr, nn * 4 * plan.n_corr);
    }

    /// Argmax of `map` (nx × ny) into res: [peak, bitcast index, map[0]].
    pub fn argmax(self: *Smi, map: Buf, nx: u32, ny: u32) void {
        self.run("argmax_part", 64, 1, 1, &.{ self.un(.{ .w = nx, .h = ny, .n = nx, .gy = ny }), map.at(1), self.arg.at(4) });
        self.run("argmax_finish", 1, 1, 1, &.{ self.arg.at(1), map.at(2), self.res.at(5) });
    }

    /// Record the dense shift map of pose H (moving → fixed, 1-based): every translation of the
    /// moving image about H, scored over the overlap, into `ns` (nx wide × ny high, lag 0 at
    /// index 0, negative lags wrapped) and its argmax into `res`.
    pub fn shiftMap(self: *Smi, a: *Features, b: *Features, H: lie.Mat3, opt: MapOptions) !ShiftPlan {
        const fr = self.poseStacks(a, b, H, opt.edge);
        return self.shiftMapFrom(a, b, fr, opt);
    }

    /// The shift map on frame `fr` with the moving image warped by the canvas pose in `hc`
    /// (9 floats a GPU kernel wrote: the GPU-resident climb), its peak into `res`.
    pub fn shiftMapAt(self: *Smi, a: *Features, b: *Features, fr: lie.Frame, hc: Buf, opt: MapOptions) !ShiftPlan {
        self.poseStacksOn(a, b, fr, hc.at(3), opt.edge);
        return self.shiftMapFrom(a, b, fr, opt);
    }

    fn shiftMapFrom(self: *Smi, a: *Features, b: *Features, fr: lie.Frame, opt: MapOptions) !ShiftPlan {
        _ = a;
        _ = b;
        const g = self.g;
        const plan = &self.plans[@intFromBool(opt.planExact())];
        const nmc = if (opt.cap == 0) autoShiftPlan(fr.w, fr.h) else fft.linearCorrPlanRect(fr.w, fr.h, self.capN(plan, opt.cap), self.g.fft_sizes);
        const nx = nmc[0];
        const ny = nmc[1];
        const cw = nmc[2];
        const ch = nmc[3];
        const f = try self.fft2(nx, ny);
        self.ensureCorr(nx, ny, plan);
        g.ensure(&self.ns, @as(u64, nx) * ny * 4);
        self.packFft(self.cstack_b, fr.w, fr.h, nx, ny, cw, ch, self.spec_b, plan, f, null);
        self.packFft(self.cstack, fr.w, fr.h, nx, ny, cw, ch, self.spec_a, plan, f, null);
        const floor_n: u32 = if (opt.min_overlap > 0) @intFromFloat(@floor(opt.min_overlap * @as(f64, @floatFromInt(cw * ch)))) else 0;
        const gcell = @as(f64, @floatFromInt(fr.w)) / @as(f64, @floatFromInt(cw));
        self.correlate(nx, ny, plan, opt, @max(MIN_N, floor_n), null, 1.0 / (gcell * gcell), self.ns, f);
        self.argmax(self.ns, nx, ny);
        return .{ .n = nx, .ny = ny, .cw = cw, .ch = ch, .frame = fr, .exact = opt.exact, .calibrated = opt.calib() };
    }

    /// Record the roto-scale map of pose H about c (fixed-image pixels, 1-based) into `dst`
    /// (engine.js fmMapPose + smi.js fmMapFromPlanes, before masking). r0 / r1 pin the radii, so
    /// the inverse-frame map shares the forward map's grid.
    pub fn rsMap(self: *Smi, a: *Features, b: *Features, H: lie.Mat3, c: [2]f64, r0_opt: ?f64, r1_opt: ?f64, opt: MapOptions, dst: *Buf) !RsPlan {
        const fr = self.poseStacks(a, b, H, opt.edge);
        return self.rsMapFrom(fr, c, r0_opt, r1_opt, opt, dst);
    }

    /// The roto-scale map on frame `fr` about c with the moving image warped by the canvas pose
    /// in `hc` (9 floats a GPU kernel wrote: the GPU-resident climb).
    pub fn rsMapAt(self: *Smi, a: *Features, b: *Features, fr: lie.Frame, hc: Buf, c: [2]f64, opt: MapOptions, dst: *Buf) !RsPlan {
        self.poseStacksOn(a, b, fr, hc.at(3), opt.edge);
        return self.rsMapFrom(fr, c, null, null, opt, dst);
    }

    fn rsMapFrom(self: *Smi, fr: lie.Frame, c: [2]f64, r0_opt: ?f64, r1_opt: ?f64, opt: MapOptions, dst: *Buf) !RsPlan {
        const g = self.g;
        const plan = &self.plans[@intFromBool(opt.planExact())];
        const cx = c[0] - 1 - @as(f64, @floatFromInt(fr.ox));
        const cy = c[1] - 1 - @as(f64, @floatFromInt(fr.oy));
        var r1: f64 = 0;
        if (r1_opt) |v| r1 = v;
        if (r1 == 0) {
            const W: f64 = @floatFromInt(fr.w);
            const Hh: f64 = @floatFromInt(fr.h);
            for ([4][2]f64{ .{ 0, 0 }, .{ W - 1, 0 }, .{ W - 1, Hh - 1 }, .{ 0, Hh - 1 } }) |p|
                r1 = @max(r1, std.math.hypot(p[0] - cx, p[1] - cy));
        }
        const r0 = r0_opt orelse @max(2, @min(r1 / 16, 8));
        const r_out = @max(r1, 2);
        const r_in = @min(@max(r0, 1e-3), r_out / 8);
        const n_ring: u32 = @max(64, @as(u32, @intFromFloat(@ceil(2 * std.math.pi * r_out))));
        var n: u32 = undefined;
        if (opt.cap == 0) {
            // auto: a power of two, MAP_AUTO_MIN–MAP_AUTO_MAX angles (0.7° at 512)
            n = std.math.clamp(std.math.ceilPowerOfTwoAssert(u32, n_ring), MAP_AUTO_MIN, MAP_AUTO_MAX);
        } else {
            n = fft.planSize(n_ring, 1, self.g.fft_sizes);
            const cap = self.capN(plan, opt.cap);
            if (n > cap) n = cap;
        }
        const n_th = n;
        const n_lam = @max(16, n >> 1);
        const f = try self.fft2(n, n);
        self.ensureCorr(n, n, plan);
        const cells: u64 = @as(u64, n_th) * n_lam * 5 * 4;
        g.ensure(&self.lp_a, cells);
        g.ensure(&self.lp_b, cells);
        g.ensure(dst, @as(u64, n) * n * 4);
        const log0 = @log(r_in);
        const span = @log(r_out) - log0;
        const lp: LogPolar = .{ .log0 = log0, .span = span, .dlam = span / @as(f64, @floatFromInt(n_lam)), .dth = 2 * std.math.pi / @as(f64, @floatFromInt(n_th)) };
        const gx = (n_th + 7) / 8;
        const gy = (n_lam + 7) / 8;
        const ul: U = .{ .w = fr.w, .h = fr.h, .n = n_th, .cw = n_lam, .mean = @floatCast(log0), .stdv = @floatCast(span), .scale = @floatCast(cx), .ridge = @floatCast(cy) };
        self.run("spatial_logpolar", gx, gy, 1, &.{ self.un(ul), self.cstack.at(1), self.lp_a.at(4) });
        self.run("spatial_logpolar", gx, gy, 1, &.{ self.un(ul), self.cstack_b.at(1), self.lp_b.at(4) });
        self.packFft(self.lp_a, n_th, n_lam, n, n, n_th, n_lam, self.spec_a, plan, f, lp);
        self.packFft(self.lp_b, n_th, n_lam, n, n, n_th, n_lam, self.spec_b, plan, f, lp);
        self.correlate(n, n, plan, opt, MIN_N, lp, 1, dst.*, f);
        return .{
            .n = n, .n_th = n_th, .n_lam = n_lam, .log0 = log0, .log_span = span, .dlam = lp.dlam,
            .r0 = r_in, .r1 = r_out, .cx = cx, .cy = cy, .frame = fr, .calibrated = opt.calib(),
        };
    }

    /// Mask (and symmetrize) roto-scale maps into ns_rs and take its argmax.
    pub fn rsFinish(self: *Smi, fwd: Buf, inv: ?Buf, mirror_theta: bool, p: RsPlan, smin: f64, smax: f64) void {
        const n = p.n;
        self.g.ensure(&self.ns_rs, @as(u64, n) * n * 4);
        const lo = @log(@max(if (smin > 0) smin else 0.2, 1e-6));
        const hi = @log(@max(if (smax > 0) smax else 5, 1.000001));
        self.run1("rs_finish", @intCast((@as(u64, n) * n + 255) / 256), 1, &.{
            self.un(.{ .w = n, .h = n, .n = n, .plane_b = @intFromBool(inv != null), .gx = @intFromBool(mirror_theta), .mean = @floatCast(p.dlam), .stdv = @floatCast(lo), .scale = @floatCast(hi) }),
            fwd.at(1), (inv orelse fwd).at(2), self.ns_rs.at(4),
        });
        self.argmax(self.ns_rs, n, n);
    }
};
