//! The registration engine both front ends drive. Operations are written as ordinary blocking
//! code: record the GPU schedule, queue the reads of its results, `wait`, compute the result.
//! Natively `wait` polls the device; in the browser the module suspends there (JS Promise
//! Integration) while the adapter awaits the reads. Same code, same schedule.
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const smi_mod = @import("smi.zig");
const pose_mod = @import("pose.zig");
const pair_mod = @import("pair.zig");
const lie = @import("lie.zig");
const mom = @import("moments.zig");
const settings_mod = @import("settings.zig");
const events_mod = @import("events.zig");
const climb_mod = @import("climb.zig");
const gclimb_mod = @import("gclimb.zig");
const gls_mod = @import("gls.zig");
const pg_mod = @import("posgift.zig");
const match_mod = @import("match.zig");
const search_mod = @import("search.zig");
const overlay_mod = @import("overlay.zig");
const export_mod = @import("export.zig");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
pub const Settings = settings_mod.Settings;

pub const version = "0.1.3";

/// Result of a shift map. Lags are in fixed-image pixels: translating the moving image by
/// (−dx, −dy) after H moves it onto the peak (the browser app's "shift hop"), i.e. the corrected
/// pose is translate(−dx, −dy) · H.
pub const ShiftResult = extern struct {
    /// FFT width; the map is n_y rows of n with lag (0, 0) at index 0 and negative lags wrapped
    n: u32 = 0,
    /// correlation grid (cells) and the canvas it covers (px)
    cw: u32 = 0,
    ch: u32 = 0,
    canvas_w: u32 = 0,
    canvas_h: u32 = 0,
    ox: i32 = 0,
    oy: i32 = 0,
    peak_index: u32 = 0,
    dx: f64 = 0,
    dy: f64 = 0,
    peak: f64 = 0,
    zero: f64 = 0,
    corr_area_px: f64 = 0,
    exact: u32 = 0,
    calibrated: u32 = 0,
    /// FFT height (the map's rows; n_y = n for a square map)
    n_y: u32 = 0,
    _pad: u32 = 0,
};

/// Result of a roto-scale map: the peak's correction about the centre c (fixed pixels,
/// 1-based) is a rotation dth (radians) and a log-scale dsg, applied to the moving image:
/// corrected pose = simAbout(c, dth, dsg) · H. The map is n×n (θ across, log ρ down, lag 0 at
/// index 0, negative lags wrapped), rows outside [rs_smin, rs_smax] zeroed.
pub const RsResult = extern struct {
    n: u32 = 0,
    n_th: u32 = 0,
    n_lam: u32 = 0,
    peak_index: u32 = 0,
    dlam: f64 = 0,
    r0: f64 = 0,
    r1: f64 = 0,
    cx: f64 = 0,
    cy: f64 = 0,
    peak: f64 = 0,
    zero: f64 = 0,
    dth: f64 = 0,
    dsg: f64 = 0,
    symmetric: u32 = 0,
    calibrated: u32 = 0,
};

/// Pose score and gradient (tangent coordinates [tx ty th sg al ga px py], 6 used for affine).
pub const GradResult = extern struct {
    score: f64 = 0,
    fwd: f64 = 0,
    inv: f64 = 0,
    n: f64 = 0,
    grad: [8]f64 = @splat(0),
    hess: [64]f64 = @splat(0),
    has_hess: u32 = 0,
    nk: u32 = 0,
};

/// One image: the gray upload, its preprocessed work texels (detection), and those as f32
/// (the SMI features' input).
const Side = struct {
    img: gls_mod.Gls.Gray = .{},
    work: Buf = .{},
    work32: Buf = .{},
    w: u32 = 0,
    h: u32 = 0,
    /// bake per the preprocessing settings; false: the upload is already the work image
    preprocess: bool = true,
    baked: ?BakeKey = null,
    has: bool = false,
};

const BakeKey = struct {
    clahe: bool,
    grid: u32,
    bins: u32,
    band: bool,
    fine: f64,
    coarse: f64,
    invert: bool,
};

pub const DetectResult = extern struct { n1: u32 = 0, n2: u32 = 0, levels1: u32 = 0, levels2: u32 = 0 };
/// ninl: inliers of the pose (POS-GIFT: of its POS fit when that set the pose); n_corr: the
/// correspondences fitted. POS-GIFT also: the pooled affine's inliers, POS candidates and inliers,
/// and the rotation found by the search (has_rot).
pub const MatchResult = extern struct {
    H: [9]f64 = lie.identity,
    ninl: u32 = 0,
    n_corr: u32 = 0,
    ninl_aff: u32 = 0,
    pos_n: u32 = 0,
    pos_inl: u32 = 0,
    has_rot: u32 = 0,
    rot_deg: f64 = 0,
    /// POS-GIFT: H is POS's fit (else the pooled affine fit)
    pos_kept: u32 = 0,
    _pad: u32 = 0,
};

pub const Flags = struct {
    pub const exact: u32 = 1;
    pub const calibrated: u32 = 2;
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    dev: *gpu_mod.Backend.Device,
    g: Gpu,
    smi: smi_mod.Smi,
    pose: pose_mod.PoseGpu,
    pr: pair_mod.Pair,
    set: Settings = .{},
    ffd: pair_mod.Ffd = .{},
    /// the current pose (moving → fixed, 1-based)
    H: lie.Mat3 = lie.identity,
    /// roto-scale centre override (fixed pixels, 1-based); default: the moving centroid under H
    rs_center: ?[2]f64 = null,
    gls: gls_mod.Gls,
    sides: [2]Side = .{ .{}, .{} },
    feats: [2]smi_mod.Features = .{ .{}, .{} },
    kp_n: [2]u32 = .{ 0, 0 },
    levels: [2]std.ArrayList(gls_mod.Level) = .{ .empty, .empty },
    matched: bool = false,
    /// POS-GIFT: the module (created on first use), its detection record, whether the last match was its
    pg: ?pg_mod.PosGift = null,
    pg_det: ?pg_mod.Detection = null,
    match_pos: bool = false,
    /// the global sweep (created on first use)
    sw: ?search_mod.Sweep = null,
    /// overlay, NCC and tile-heat buffers
    ov: overlay_mod.State = .{},
    /// set by zc_cancel (the page's Stop, while an operation is suspended on the GPU); climbs
    /// and searches end early with their best so far
    cancel: bool = false,
    /// settings.half changed: texel buffers are rebuilt before the next GPU work (configure
    /// never waits, so it stays a plain call on the web)
    texels_stale: bool = false,
    res_host: [4]f32 = @splat(0),
    area_host: [4]f32 = @splat(0),
    err_buf: [1024]u8 = undefined,
    err_len: usize = 0,
    events: events_mod.Sink,
    /// the last maps computed (by a call or a climb's hops)
    last_shift: ?ShiftResult = null,
    last_rs: ?RsResult = null,

    pub fn create(gpa: std.mem.Allocator, dev: *gpu_mod.Backend.Device) !*Engine {
        const self = try gpa.create(Engine);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .dev = dev, .g = undefined, .smi = undefined, .pose = undefined, .pr = undefined, .gls = undefined, .events = .{ .gpa = gpa } };
        self.g = try Gpu.init(gpa, dev);
        self.smi = try smi_mod.Smi.init(&self.g);
        self.gls = try gls_mod.Gls.init(&self.g, self.set.half);
        self.pose = pose_mod.PoseGpu.init(&self.smi);
        self.pr = pair_mod.Pair.init(&self.pose, &self.feats[0], &self.feats[1], .{}, &self.ffd);
        self.syncSettings();
        return self;
    }

    pub fn destroy(self: *Engine) void {
        for (0..2) |s| {
            self.feats[s].deinit(&self.g);
            self.releaseSide(s);
            self.levels[s].deinit(self.gpa);
        }
        self.dropPosGift();
        if (self.pg) |*p| p.deinit();
        if (self.sw) |*w| w.deinit();
        self.ov.deinit(&self.g);
        self.gls.deinit();
        self.events.deinit();
        self.pr.deinit();
        self.pose.deinit();
        self.smi.deinit();
        self.g.submit();
        self.g.deinit();
        self.gpa.destroy(self);
    }

    pub fn fail(self: *Engine, comptime fmt: []const u8, args: anytype) void {
        const m = std.fmt.bufPrint(&self.err_buf, fmt, args) catch self.err_buf[0..0];
        self.err_len = m.len;
    }

    /// The first engine or GPU error since the last call (copied into `buf`), or 0.
    pub fn takeError(self: *Engine, buf: []u8) usize {
        if (self.g.oversize) |o| {
            // a kernel asked for more workgroups than a dimension allows (images too large)
            const tail = std.fmt.bufPrint(self.err_buf[self.err_len..], " ({s}: {d}×{d}×{d} workgroups, limit {d} per dimension)", .{
                self.g.pipeName(o.pipe), o.n[0], o.n[1], o.n[2], gpu_mod.MAX_GROUPS,
            }) catch "";
            self.err_len += tail.len;
            self.g.oversize = null;
        }
        var m: []const u8 = self.err_buf[0..self.err_len];
        if (m.len == 0) m = self.dev.takeError() orelse "";
        const n = @min(m.len, buf.len);
        @memcpy(buf[0..n], m[0..n]);
        self.err_len = 0;
        return n;
    }

    // ── settings and state ──
    fn syncSettings(self: *Engine) void {
        const s = &self.set;
        self.g.fft_sizes = s.fft_sizes;
        self.pr.set = .{ .family = s.family(), .exact = s.exact, .edge = s.metric == .smi_edge, .symmetric = s.symmetric };
        const g = std.math.clamp(s.ffd_grid, 2, pose_mod.FFD_MAX);
        if (g != self.ffd.g) {
            self.ffd.g = g;
            @memset(&self.ffd.cps, 0);
        }
        self.ffd.on = s.ffd;
        self.ffd.source = s.ffd_source;
    }

    pub fn configure(self: *Engine, json: []const u8) !void {
        const half0 = self.set.half;
        try self.set.apply(self.gpa, json);
        self.syncSettings();
        if (self.set.half != half0) self.texels_stale = true;
    }

    /// Half precision changed: new texel buffers throughout (images re-uploaded from f32 copies).
    fn rebuildTexels(self: *Engine) !void {
        self.texels_stale = false;
        const gpa = self.gpa;
        var keep: [2]?[]f32 = .{ null, null };
        defer for (keep) |k| if (k) |b| gpa.free(b);
        for (0..2) |s| if (self.sides[s].has) {
            const sd = &self.sides[s];
            const n = @as(usize, sd.w) * sd.h;
            const buf = try gpa.alloc(f32, n);
            keep[s] = buf;
            // the raw image as f32 (raw32 when half, else raw)
            const src = if (sd.img.raw32.id != 0) sd.img.raw32 else sd.img.raw;
            self.g.read(src, 0, std.mem.sliceAsBytes(buf));
        };
        try self.g.wait();
        self.gls.deinit();
        self.gls = try gls_mod.Gls.init(&self.g, self.set.half);
        for (0..2) |s| if (keep[s]) |buf| {
            const sd = self.sides[s];
            try self.setSide(s, self.gls.uploadGray(buf, sd.w, sd.h), sd.preprocess);
        };
    }

    /// Spline control points (2·g·g floats; g = ffd_grid).
    pub fn setFfdCps(self: *Engine, cps: []const f32) !void {
        const n = self.ffd.n();
        if (cps.len != n) return error.BadControlPoints;
        @memcpy(self.ffd.cps[0..n], cps);
    }

    pub fn mapOpts(self: *const Engine) smi_mod.MapOptions {
        const s = &self.set;
        return .{
            .family = s.family(), .exact = s.exact, .calibrated = s.calibrated, .cap = s.map_res,
            .min_overlap = s.min_overlap, .edge = s.metric == .smi_edge,
        };
    }

    pub fn hasImages(self: *const Engine) bool {
        return self.sides[0].has and self.sides[1].has;
    }

    pub fn needImages(self: *Engine) !void {
        if (!self.hasImages()) return error.NoImages;
        if (self.texels_stale) try self.rebuildTexels();
        try self.ensureBaked();
        self.syncFeatures();
    }

    fn releaseSide(self: *Engine, s: usize) void {
        const sd = &self.sides[s];
        self.g.release(&sd.img.raw);
        self.g.release(&sd.img.raw32);
        self.g.release(&sd.work);
        self.g.release(&sd.work32);
        sd.* = .{};
    }

    fn bakeKey(self: *const Engine) BakeKey {
        const s = &self.set;
        return .{ .clahe = s.clahe, .grid = s.clahe_grid, .bins = s.clahe_bins, .band = s.band, .fine = s.bp_fine, .coarse = s.bp_coarse, .invert = s.invert };
    }

    /// Preprocess side s per the settings into its work texels (app bakeSide), then its f32 copy
    /// and SMI features. Keypoints of that side become stale.
    fn bake(self: *Engine, s: usize) !void {
        const sd = &self.sides[s];
        const n: u64 = @as(u64, sd.w) * sd.h;
        const set = &self.set;
        if (sd.work.id == 0) sd.work = self.gls.storageTexels(n);
        if (sd.preprocess and set.clahe) {
            self.gls.clahe(if (sd.img.raw32.id != 0) sd.img.raw32 else sd.img.raw, sd.work, sd.w, sd.h, 40, set.clahe_bins, set.clahe_grid);
        } else self.g.copy(sd.img.raw, 0, sd.work, 0, n * (if (self.gls.half) @as(u64, 2) else 4));
        if (sd.preprocess and set.band) try self.gls.bandpass(sd.work, sd.w, sd.h, set.bp_fine, set.bp_coarse);
        if (sd.preprocess and set.invert) try self.gls.invert(sd.work, sd.w, sd.h);
        self.g.ensure(&sd.work32, n * 4);
        self.gls.promoteTexels(sd.work, sd.work32, @intCast(n), 0, 0);
        sd.baked = if (sd.preprocess) self.bakeKey() else null;
        self.smi.prepare(sd.work32, sd.w, sd.h, &self.feats[s], self.set.family() != .smi);
        self.kp_n[s] = 0;
        self.matched = false;
        self.dropPosGift();
        if (self.hasImages()) self.smi.pairArea(&self.feats[0], &self.feats[1]);
        self.g.submit();
    }

    /// Images up to date with the preprocessing and precision settings.
    pub fn refreshImages(self: *Engine) !void {
        if (self.texels_stale) try self.rebuildTexels();
        try self.ensureBaked();
    }

    /// Exchange the moving and fixed images with everything computed from them (features,
    /// GLS-MIFT keypoints and descriptors); POS-GIFT detections and matches are dropped.
    pub fn swap(self: *Engine) void {
        std.mem.swap(Side, &self.sides[0], &self.sides[1]);
        std.mem.swap(smi_mod.Features, &self.feats[0], &self.feats[1]);
        std.mem.swap(u32, &self.kp_n[0], &self.kp_n[1]);
        std.mem.swap(std.ArrayList(gls_mod.Level), &self.levels[0], &self.levels[1]);
        const gl = &self.gls;
        std.mem.swap(Buf, &gl.kps, &gl.kps_b);
        std.mem.swap(Buf, &gl.des, &gl.des_b);
        std.mem.swap(Buf, &gl.desq, &gl.desq_b);
        std.mem.swap(Buf, &gl.dsc, &gl.dsc_b);
        std.mem.swap(Buf, &gl.kdir, &gl.kdir_b);
        if (self.pg_det != null) self.kp_n = .{ 0, 0 };
        self.dropPosGift();
        self.matched = false;
        self.rs_center = null;
        if (self.hasImages()) self.smi.pairArea(&self.feats[0], &self.feats[1]);
    }

    /// Descriptor inner product of each moving keypoint with its match (−1e9: none), after match.
    pub fn matchScores(self: *Engine, out: []f32) !u32 {
        if (!self.matched) return 0;
        const n1 = @min(self.kp_n[0], @as(u32, @intCast(out.len)));
        if (n1 == 0) return 0;
        const gpa = self.gpa;
        var des_a = self.gls.des;
        var des_b = self.gls.des_b;
        var dim: u32 = self.gls.des_dim;
        var n2 = self.kp_n[1];
        if (self.match_pos) {
            const pg = &self.pg.?;
            des_a = pg.sides[@intFromEnum(pg_mod.SideId.a)].des;
            des_b = pg.sides[@intFromEnum(pg_mod.SideId.b)].des;
            dim = pg.S.dp;
            n2 = pg.last.n2;
        }
        if (n2 == 0) return 0;
        const mj = try gpa.alloc(u32, n1);
        defer gpa.free(mj);
        const da = try gpa.alloc(f32, @as(usize, n1) * dim);
        defer gpa.free(da);
        const db = try gpa.alloc(f32, @as(usize, n2) * dim);
        defer gpa.free(db);
        self.g.read(self.gls.match_j, 0, std.mem.sliceAsBytes(mj));
        self.g.read(des_a, 0, std.mem.sliceAsBytes(da));
        self.g.read(des_b, 0, std.mem.sliceAsBytes(db));
        try self.g.wait();
        for (mj) |*j| if (j.* != 0xffffffff and j.* >= n2) {
            j.* = 0xffffffff;
        };
        match_mod.matchDots(da, db, mj, n1, dim, out[0..n1]);
        return n1;
    }

    fn ensureBaked(self: *Engine) !void {
        const key = self.bakeKey();
        for (0..2) |s| {
            const sd = &self.sides[s];
            if (sd.has and sd.preprocess and !std.meta.eql(sd.baked orelse continue, key)) try self.bake(s);
        }
    }

    fn setSide(self: *Engine, side: usize, img: gls_mod.Gls.Gray, preprocess: bool) !void {
        std.debug.assert(!self.texels_stale);
        self.releaseSide(side);
        self.sides[side] = .{ .img = img, .w = img.w, .h = img.h, .preprocess = preprocess, .has = true };
        self.rs_center = null;
        try self.bake(side);
    }

    /// An RGBA8 image (w·h·4 bytes) as the moving (0) or fixed (1) image, preprocessed per the
    /// settings (CLAHE, band pass, invert) as the app does.
    pub fn setImageRgba(self: *Engine, side: usize, rgba: []const u8, w: u32, h: u32) !void {
        if (side > 1 or w < 8 or h < 8 or rgba.len < @as(usize, w) * h * 4) return error.BadImage;
        if (self.texels_stale) try self.rebuildTexels();
        try self.setSide(side, try self.gls.uploadRgba(rgba, w, h), true);
    }

    /// A gray image (w×h f32 in [0, 1], row-major). preprocess: bake per the settings; false:
    /// use it as the work image as is.
    pub fn setImageGray(self: *Engine, side: usize, data: []const f32, w: u32, h: u32, preprocess: bool) !void {
        if (side > 1 or w < 8 or h < 8 or data.len < @as(usize, w) * h) return error.BadImage;
        if (self.texels_stale) try self.rebuildTexels();
        try self.setSide(side, self.gls.uploadGray(data, w, h), preprocess);
    }

    /// A preprocessed gray work image (the app's work buffers; tests).
    pub fn setImage(self: *Engine, side: usize, data: []const f32, w: u32, h: u32) !void {
        return self.setImageGray(side, data, w, h, false);
    }

    /// Features follow the score family: whitened ranks for SMI, normal scores for E4 / λmax
    /// (engine.js feats caches per mode). Rebuilt from the work images when it changes.
    fn syncFeatures(self: *Engine) void {
        const normal = self.set.family() != .smi;
        var changed = false;
        for (0..2) |s| if (self.sides[s].has and self.feats[s].normal != normal) {
            self.smi.prepare(self.sides[s].work32, self.sides[s].w, self.sides[s].h, &self.feats[s], normal);
            changed = true;
        };
        if (changed and self.hasImages()) self.smi.pairArea(&self.feats[0], &self.feats[1]);
    }

    // ── detection and matching ──
    fn glsSettings(self: *const Engine) gls_mod.Settings {
        const s = &self.set;
        return .{
            .n_octaves = s.n_octaves, .max_points = s.max_points, .min_contrast = s.min_contrast, .radius = s.radius,
            .tau = s.tau, .nt = s.nt, .auto_nt = s.auto_nt, .second_ori = s.second_ori,
            .structure = .{ .n_sigma = s.n_sigma, .n_angle = s.n_angle, .n_r = s.n_r },
            .n_trials = s.n_trials, .inlier_px = s.inlier_px, .scale_lo = s.scale_lo, .scale_hi = s.scale_hi, .seed = s.seed,
        };
    }

    /// What a fitted pose may do to the moving image (settings scale_lo … max_persp).
    fn fitLimits(self: *const Engine) match_mod.Limits {
        const s = &self.set;
        return .{
            .w = @floatFromInt(@max(self.sides[0].w, 2)), .h = @floatFromInt(@max(self.sides[0].h, 2)),
            .scale_lo = s.scale_lo, .scale_hi = s.scale_hi, .max_aniso = @max(1, s.max_aniso), .max_persp = @max(1, s.max_persp),
        };
    }

    /// The sanity check of a pose fitted to matches (match.zig poseCheck).
    pub fn fitCheck(self: *const Engine, H: lie.Mat3) match_mod.Verdict {
        return match_mod.poseCheck(H, self.fitLimits());
    }

    fn warnFit(self: *Engine) void {
        const v = self.fitCheck(self.H);
        if (v != .ok) self.events.log("warning: the matching fit fails the sanity check: {s}", .{v.text()});
    }

    fn dropPosGift(self: *Engine) void {
        if (self.pg_det) |*d| d.deinit(self.gpa);
        self.pg_det = null;
        self.match_pos = false;
    }

    fn posGift(self: *Engine) !*pg_mod.PosGift {
        if (self.pg == null) self.pg = try pg_mod.PosGift.init(&self.g, &self.gls);
        const p = &self.pg.?;
        p.set = self.pgSettings();
        return p;
    }

    /// POS-GIFT settings as the app's posGiftSettings(): its defaults, the exposed fields clamped
    /// to their ranges, the Match tab's trials / seed and model (affine or homography).
    fn pgSettings(self: *const Engine) pg_mod.Settings {
        const s = &self.set;
        const cf = struct {
            fn f(x: f64, lo: f64, hi: f64) f64 {
                return @min(hi, @max(lo, x));
            }
            fn u(x: u32, lo: u32, hi: u32) u32 {
                return @min(hi, @max(lo, x));
            }
        };
        var p: pg_mod.Settings = .{
            .rotation = if (s.pg_search) .search else .upright,
            .pos_affine = s.group == .affine,
            .n_trials = if (s.n_trials > 0) s.n_trials else 100000,
            .seed = if (s.seed != 0) s.seed else 1,
            .n_octaves = cf.u(s.pg_n_octaves, 1, 4),
            .max_points = cf.u(s.pg_max_points, 64, 16000),
            .min_contrast = cf.f(s.pg_min_contrast, 0.001, 0.2),
            .p1 = cf.f(s.pg_p1, 4, 32),
            .desc_pow = cf.f(s.pg_desc_pow, 0.5, 4),
            .n_orient = @max(4, @min(8, 2 * ((cf.u(s.pg_n_orient, 4, 8) + 1) / 2))),
            .n_rings = cf.u(s.pg_n_rings, 2, 4),
            .n_scales = cf.u(s.pg_n_scales, 2, 6),
            .min_wl = cf.f(s.pg_min_wl, 2, 12),
            .mult = cf.f(s.pg_mult, 1.2, 3),
            .sigma_onf = cf.f(s.pg_sigma_onf, 0.3, 0.95),
            .pc_k = cf.f(s.pg_pc_k, 0, 5),
            .cutoff = cf.f(s.pg_cutoff, 0, 1),
            .pc_g = cf.f(s.pg_pc_g, 1, 20),
            .gsig = cf.f(s.pg_gsig, 1, 10),
            .pos_p1 = cf.f(s.pg_pos_p1, 4, 40),
            .pos_k = cf.u(s.pg_pos_k, 4, 64),
            .ratio = cf.f(s.pg_ratio, 0.3, 1),
            .match = if (s.per_octave) .octave else .nn,
            .inlier_px = cf.f(s.inlier_px, 0.5, 100),
            .fit = .{
                .method = switch (s.match_method) {
                    .lofsc => .lofsc,
                    .prosac => .prosac,
                    .magsac => .magsac,
                },
                .homography = s.group == .homography and s.match_method != .lofsc,
            },
            .pos = s.pg_pos,
            .pos_search_px = cf.f(s.pg_pos_search_px, 1, 100),
            .pos_px = cf.f(s.pg_pos_px, 0.5, 50),
            .det_map = if (s.pg_corners) .min_moment else .pcsum,
        };
        if (s.pg_released) p = p.released();
        return p;
    }

    fn workImage(self: *const Engine, s: usize) pg_mod.Image {
        return .{ .buf = self.sides[s].work, .w = self.sides[s].w, .h = self.sides[s].h };
    }

    /// Keypoints and descriptors on both work images (settings.detector).
    pub fn detect(self: *Engine) !DetectResult {
        try self.needImages();
        self.dropPosGift();
        self.matched = false;
        if (self.set.detector == .pos_gift) return self.detectPosGift();
        const gs = self.glsSettings();
        self.gls.applySettings(gs);
        if (try self.gls.setStructure(gs.structure))
            self.events.log("GLS-MIFT structure: {d} scales, {d} orientations ({d} sectors), {d} rings → {d}-float descriptor", .{ gs.structure.n_sigma, gs.structure.n_angle, 2 * gs.structure.n_angle, gs.structure.n_r, self.gls.des_dim });
        self.gls.nt_used.clearRetainingCapacity();
        var out: DetectResult = .{};
        for (0..2) |s| {
            const sd = &self.sides[s];
            const n = try self.gls.detect(sd.work, sd.w, sd.h, @intCast(s));
            self.kp_n[s] = n;
            self.levels[s].clearRetainingCapacity();
            try self.levels[s].appendSlice(self.gpa, self.gls.last_levels.items);
        }
        out.n1 = self.kp_n[0];
        out.n2 = self.kp_n[1];
        out.levels1 = @intCast(self.levels[0].items.len);
        out.levels2 = @intCast(self.levels[1].items.len);
        self.matched = false;
        self.events.log("kps {d}/{d}  threshold {d:.5} / {d:.5}", .{ out.n1, out.n2, if (self.gls.nt_used.items.len > 0) self.gls.nt_used.items[0] else 0, if (self.gls.nt_used.items.len > 1) self.gls.nt_used.items[1] else 0 });
        return out;
    }

    fn detectPosGift(self: *Engine) !DetectResult {
        const pg = try self.posGift();
        const st = pg.set;
        self.events.log("detect POS-GIFT{s}", .{if (st.rotation == .search) " with rotation search" else " upright"});
        for (&self.levels) |*l| l.clearRetainingCapacity();
        self.pg_det = try pg.detectPair(self.workImage(0), self.workImage(1));
        const d = &self.pg_det.?;
        self.kp_n = .{ d.dl.n, d.dr.n };
        if (d.n_frames > 1) {
            self.events.log("kps {d}/{d}  levels {d}/{d}  + {d}° frame", .{ d.dl.n, d.dr.n, d.dl.levels.items.len, d.dr.levels.items.len, d.frames[1].deg });
        } else self.events.log("kps {d}/{d}  levels {d}/{d}", .{ d.dl.n, d.dr.n, d.dl.levels.items.len, d.dr.levels.items.len });
        return .{ .n1 = d.dl.n, .n2 = d.dr.n, .levels1 = @intCast(d.dl.levels.items.len), .levels2 = @intCast(d.dr.levels.items.len) };
    }

    fn matchPosGift(self: *Engine) !MatchResult {
        const d = &(self.pg_det orelse return error.DetectFirst);
        const pg = try self.posGift();
        const r = try pg.match(self.workImage(0), self.workImage(1), d);
        // Two candidates: the pooled fit and POS's. POS keeps the correspondences within pos_px of
        // one homography: few on a deformed pair, and the fit through them can fall far behind the
        // pooled fit. A candidate that fails the sanity check gives way to one that passes;
        // otherwise the higher pose score stays.
        var pos_kept = r.pos_h;
        var buf3: [160]u8 = undefined;
        var kept: []const u8 = "";
        if (r.pos_h) {
            const v_pos = self.fitCheck(r.H);
            const v_aff = self.fitCheck(r.Haff);
            if ((v_pos == .ok) != (v_aff == .ok)) {
                pos_kept = v_pos == .ok;
                if (!pos_kept) kept = std.fmt.bufPrint(&buf3, "  · the pooled fit stays (POS's: {s})", .{v_pos.text()}) catch "";
            } else {
                const s_pos = (try self.score(r.H)).mean;
                const s_aff = (try self.score(r.Haff)).mean;
                if (!(s_pos >= s_aff)) {
                    pos_kept = false;
                    kept = std.fmt.bufPrint(&buf3, "  · the pooled fit stays (score {e:.3} against {e:.3})", .{ s_aff, s_pos }) catch "";
                }
            }
        }
        self.H = if (pos_kept) r.H else r.Haff;
        self.matched = true;
        self.match_pos = true;
        var buf: [96]u8 = undefined;
        const rot = if (r.rot_deg) |deg| std.fmt.bufPrint(&buf, "  rotation {d}°", .{deg}) catch "" else "";
        var buf2: [96]u8 = undefined;
        const pos = if (r.pos_ran) std.fmt.bufPrint(&buf2, "  POS {d} → {d} inliers", .{ r.pos_n, r.pos_inl }) catch "" else "";
        self.events.log("match POS-GIFT{s}  pooled {d}  fit inliers {d}{s}{s}", .{ rot, r.pooled, r.ninl_aff, pos, kept });
        self.warnFit();
        return .{
            .H = self.H, .ninl = if (pos_kept) r.pos_inl else r.ninl_aff, .n_corr = r.pooled, .ninl_aff = r.ninl_aff,
            .pos_n = r.pos_n, .pos_inl = r.pos_inl, .has_rot = @intFromBool(r.rot_deg != null), .rot_deg = r.rot_deg orelse 0,
            .pos_kept = @intFromBool(pos_kept),
        };
    }

    /// Match the detected keypoints and fit the pose; the fit becomes the engine's pose.
    pub fn match(self: *Engine) !MatchResult {
        try self.needImages();
        if (self.kp_n[0] == 0 or self.kp_n[1] == 0) return error.DetectFirst;
        self.gls.limits = self.fitLimits();
        if (self.set.detector == .pos_gift) return self.matchPosGift();
        if (self.pg_det != null) return error.DetectFirst;
        self.gls.applySettings(self.glsSettings());
        const s = &self.set;
        const r = try self.gls.matchPair(self.kp_n[0], self.kp_n[1], self.sides[0].w, self.sides[0].h, self.sides[1].w, self.sides[1].h, .{
            .method = switch (s.match_method) {
                .lofsc => .lofsc,
                .prosac => .prosac,
                .magsac => .magsac,
            },
            .homography = s.group == .homography,
            .mutual = s.mutual,
            .per_octave = s.per_octave,
        }, self.levels[0].items, self.levels[1].items);
        self.H = r.H;
        self.matched = true;
        self.events.log("match {s} {s}  corr {d}  inliers {d}", .{ @tagName(s.match_method), if (s.per_octave) "per-octave" else "global", self.gls.corr_count, r.ninl });
        self.warnFit();
        return .{ .H = r.H, .ninl = r.ninl, .n_corr = self.gls.corr_count };
    }

    pub fn sweeper(self: *Engine) !*search_mod.Sweep {
        if (self.sw == null) self.sw = search_mod.Sweep.init(&self.g, &self.smi);
        return &self.sw.?;
    }

    /// Global search (settings.search_mode: the SIM(2) sweep or the seed cloud); a better pose
    /// becomes the engine's.
    pub fn search(self: *Engine) !search_mod.Result {
        return search_mod.run(self);
    }

    /// The composed warp at the engine's pose (and spline when on), for export.
    pub fn warpOpts(self: *const Engine) !export_mod.Opts {
        if (!self.hasImages()) return error.NoImages;
        return .{
            .H = self.H, .cps = if (self.ffd.on) self.ffd.cps[0..self.ffd.n()] else null, .g = self.ffd.g, .source = self.ffd.source,
            .lw = self.sides[0].w, .lh = self.sides[0].h, .rw = self.sides[1].w, .rh = self.sides[1].h,
        };
    }

    /// Detect → match → climb (the app's Auto).
    pub fn auto(self: *Engine) !climb_mod.Result {
        _ = try self.detect();
        _ = try self.match();
        return self.climb();
    }

    /// Keypoints of a side (x, y 1-based fixed/moving pixels, score, pad) into out; returns count.
    /// After a POS-GIFT rotation-searched match the fixed image's keypoints are those of the frame
    /// its matches index, mapped back to the image.
    pub fn keypoints(self: *Engine, side: u32, out: []gls_mod.Kp) !u32 {
        if (side > 1) return error.BadSide;
        if (side == 1 and self.match_pos) {
            const last = self.pg.?.last;
            const n = @min(last.n2, @as(u32, @intCast(out.len)));
            if (n > 0) {
                self.g.read(last.kps_b, 0, std.mem.sliceAsBytes(out[0..n]));
                try self.g.wait();
            }
            if (last.back) |M| for (out[0..n]) |*k| {
                const x: f64 = k.x;
                const y: f64 = k.y;
                const z = M[6] * x + M[7] * y + M[8];
                k.x = @floatCast((M[0] * x + M[1] * y + M[2]) / z);
                k.y = @floatCast((M[3] * x + M[4] * y + M[5]) / z);
            };
            return last.n2;
        }
        const n = @min(self.kp_n[side], @as(u32, @intCast(out.len)));
        if (n == 0) return self.kp_n[side];
        try self.gls.readKps(side, n, out);
        return self.kp_n[side];
    }

    /// Each keypoint's descriptor frame (gls.KpFrame) in keypoints() order; returns the count.
    pub fn keypointFrames(self: *Engine, side: u32, out: []gls_mod.KpFrame) !u32 {
        const kps = try self.gpa.alloc(gls_mod.Kp, out.len);
        defer self.gpa.free(kps);
        const total = try self.keypoints(side, kps);
        const n = @min(total, @as(u32, @intCast(out.len)));
        if (self.pg_det != null) {
            try self.pg.?.kpFrames(side, self.match_pos, kps[0..n], out[0..n]);
        } else try self.gls.kpFrames(side, kps[0..n], out[0..n]);
        return total;
    }

    /// Each moving keypoint's matched fixed keypoint index (0xffffffff: none) into out.
    pub fn matches(self: *Engine, out: []u32) !u32 {
        const n = @min(self.kp_n[0], @as(u32, @intCast(out.len)));
        if (!self.matched or n == 0) return 0;
        self.g.read(self.gls.match_j, 0, std.mem.sliceAsBytes(out[0..n]));
        try self.g.wait();
        return n;
    }

    // ── pose ──
    pub fn setPose(self: *Engine, H: lie.Mat3) void {
        self.H = H;
    }

    /// Climb from the current pose on the GPU (settings: hop, hop_fm, g_*, the spline), leaving
    /// the best pose (and spline) in the engine. Progress arrives as events, once per batch.
    pub fn climb(self: *Engine) !climb_mod.Result {
        try self.needImages();
        return gclimb_mod.run(self);
    }

    // ── scores ──
    /// The pose score at H (NCC: the zero-lag NCC of the overlay, as the app's readout).
    pub fn score(self: *Engine, H: lie.Mat3) !pair_mod.Parts {
        try self.needImages();
        if (self.set.metric == .ncc) {
            const s = try overlay_mod.nccScore(self, H);
            return .{ .fwd = s, .inv = null, .mean = s };
        }
        return self.pr.parts(H);
    }

    /// The shift map the settings ask for (NCC for the NCC metric).
    pub fn shiftMapSettings(self: *Engine, H: lie.Mat3, map: ?[]f32) !ShiftResult {
        if (self.set.metric == .ncc) return overlay_mod.nccShiftMap(self, H, map);
        return self.shiftMapOpts(H, self.mapOpts(), map);
    }

    /// Score and gradient over the spline control points at H (2·g·g into grad_out).
    pub fn ffdGradient(self: *Engine, H: lie.Mat3, grad_out: []f64) !f64 {
        try self.needImages();
        if (grad_out.len < self.ffd.n()) return error.BufferTooSmall;
        return self.pr.ffdGrad(H, grad_out);
    }

    pub fn gradient(self: *Engine, H: lie.Mat3, hess: bool) !GradResult {
        try self.needImages();
        const r = try self.pr.grad(H, self.set.group, hess);
        const nk = lie.nKeys(self.set.group);
        var out: GradResult = .{ .score = r.score, .n = r.n, .grad = r.grad, .nk = @intCast(nk) };
        if (r.hess) |h| {
            out.hess = h;
            out.has_hess = 1;
        }
        return out;
    }

    // ── maps ──
    /// The lag of flat index idx in an nx-wide, ny-high map (negative lags wrapped).
    pub fn peakLag(idx: u32, nx: u32, ny: u32) [2]i64 {
        const py = idx / nx;
        const px = idx % nx;
        return .{ if (px <= nx / 2) px else @as(i64, px) - nx, if (py <= ny / 2) py else @as(i64, py) - ny };
    }

    pub fn shiftMapOpts(self: *Engine, H: lie.Mat3, opt: smi_mod.MapOptions, map: ?[]f32) !ShiftResult {
        try self.needImages();
        const p = try self.smi.shiftMap(&self.feats[0], &self.feats[1], H, opt);
        self.g.read(self.smi.res, 0, std.mem.sliceAsBytes(&self.res_host));
        self.g.read(self.smi.pair_prm, 0, std.mem.sliceAsBytes(&self.area_host));
        const nn = @as(usize, p.n) * p.ny;
        if (map) |m| {
            if (m.len < nn) return error.MapTooSmall;
            self.g.read(self.smi.ns, 0, std.mem.sliceAsBytes(m[0..nn]));
        }
        try self.g.wait();
        const idx: u32 = @bitCast(self.res_host[1]);
        var r: ShiftResult = .{
            .n = p.n, .n_y = p.ny, .cw = p.cw, .ch = p.ch, .canvas_w = p.frame.w, .canvas_h = p.frame.h, .ox = p.frame.ox, .oy = p.frame.oy,
            .peak = self.res_host[0], .zero = self.res_host[2], .corr_area_px = self.area_host[0],
            .exact = @intFromBool(p.exact), .calibrated = @intFromBool(p.calibrated),
        };
        if (idx < nn) {
            const lag = peakLag(idx, p.n, p.ny);
            r.peak_index = idx;
            r.dx = @as(f64, @floatFromInt(lag[0])) * (@as(f64, @floatFromInt(p.frame.w)) / @as(f64, @floatFromInt(p.cw)));
            r.dy = @as(f64, @floatFromInt(lag[1])) * (@as(f64, @floatFromInt(p.frame.h)) / @as(f64, @floatFromInt(p.ch)));
        }
        return r;
    }

    /// Shift map with explicit whitening / calibration flags and no overlap floor (tests).
    pub fn shiftMap(self: *Engine, H: lie.Mat3, flags: u32, cap: u32, map: ?[]f32) !ShiftResult {
        var o = self.mapOpts();
        o.exact = flags & Flags.exact != 0;
        o.calibrated = flags & Flags.calibrated != 0;
        o.cap = if (cap == 0) smi_mod.MAP_CAP else cap;
        o.min_overlap = 0;
        return self.shiftMapOpts(H, o, map);
    }

    /// The roto-scale centre for pose H: the override, else the moving image's centroid.
    pub fn rsCenter(self: *const Engine, H: lie.Mat3) [2]f64 {
        return self.rs_center orelse lie.movingCentroid(H, @floatFromInt(self.feats[0].w), @floatFromInt(self.feats[0].h));
    }

    /// Roto-scale map at H about c (null: rsCenter). symmetric: also the inverse-frame map,
    /// averaged at the mirrored lag. Optional outputs: the finished map, the raw forward and
    /// inverse maps.
    pub fn rsMap(self: *Engine, H: lie.Mat3, c_opt: ?[2]f64, symmetric: bool, map: ?[]f32, fwd_map: ?[]f32, inv_map: ?[]f32) !RsResult {
        try self.needImages();
        const opt = self.mapOpts();
        const c = c_opt orelse self.rsCenter(H);
        const p = try self.smi.rsMap(&self.feats[0], &self.feats[1], H, c, null, null, opt, &self.smi.ns);
        var inv: ?Buf = null;
        if (symmetric) {
            const Hi = lie.inv3(H);
            const ci = lie.mapPt(Hi, c[0], c[1]);
            const pi = try self.smi.rsMap(&self.feats[1], &self.feats[0], Hi, ci, p.r0, p.r1, opt, &self.smi.ns_inv);
            if (pi.n != p.n) return error.GridMismatch;
            inv = self.smi.ns_inv;
        }
        const mirror = H[0] * H[4] - H[1] * H[3] > 0;
        self.smi.rsFinish(self.smi.ns, inv, mirror, p, self.set.rs_smin, self.set.rs_smax);
        self.g.read(self.smi.res, 0, std.mem.sliceAsBytes(&self.res_host));
        const nn = @as(usize, p.n) * p.n;
        if (map) |m| self.g.read(self.smi.ns_rs, 0, std.mem.sliceAsBytes(m[0..nn]));
        if (fwd_map) |m| self.g.read(self.smi.ns, 0, std.mem.sliceAsBytes(m[0..nn]));
        if (inv_map) |m| if (symmetric) self.g.read(self.smi.ns_inv, 0, std.mem.sliceAsBytes(m[0..nn]));
        try self.g.wait();
        const idx: u32 = @bitCast(self.res_host[1]);
        var r: RsResult = .{
            .n = p.n, .n_th = p.n_th, .n_lam = p.n_lam, .dlam = p.dlam, .r0 = p.r0, .r1 = p.r1, .cx = c[0], .cy = c[1],
            .peak = self.res_host[0], .zero = self.res_host[2], .symmetric = @intFromBool(symmetric), .calibrated = @intFromBool(p.calibrated),
        };
        if (idx < nn) {
            const lag = peakLag(idx, p.n, p.n);
            r.peak_index = idx;
            r.dth = -(@as(f64, @floatFromInt(lag[0])) / @as(f64, @floatFromInt(@max(p.n_th, 1)))) * 2 * std.math.pi;
            r.dsg = -@as(f64, @floatFromInt(lag[1])) * p.dlam;
        }
        return r;
    }
};
