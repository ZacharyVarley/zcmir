//! GLS-MIFT (Fan et al. 2024) detection, description and matching, and the gray-image
//! utilities that share its shader. The engine keeps one mutable Params block and changes it
//! in dispatch order (ordered writes), so every dispatch sees the values set for it.
//!
//! Images: `raw` (the gray upload) and `work` (after CLAHE / band pass / invert) are texel
//! buffers, f16 when half precision is on (the app's default) else f32.
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const shaders = @import("shaders");
const lie = @import("lie.zig");
const match = @import("match.zig");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Bind = gpu_mod.Bind;

pub const MAX_KP: u32 = 65536;
pub const MAX_TRIALS: u32 = 100000;
pub const MAX_CORR: u32 = 65536;
/// counters: [0..6) per-operation counts, [6] a pyramid's running keypoint offset, from [8] four
/// words per detected level (level_done)
pub const KP_RUNNING: u32 = 0xffffffff;
pub const LEVEL_REC = 8;
pub const MAX_LEVELS = 64;
const COUNTERS_BYTES = (LEVEL_REC + 4 * MAX_LEVELS) * 4;
const CELL_G: u32 = 90; // ceil(sqrt(8000)): cell buffers for up to 8000 points per level

pub const Params = extern struct {
    w: u32 = 1,
    h: u32 = 1,
    src_w: u32 = 1,
    src_h: u32 = 1,
    n_sigma: u32 = 4,
    n_angle: u32 = 6,
    n_r: u32 = 3,
    max_points: u32 = 2000,
    tau: f32 = 0.8,
    radius: f32 = 36,
    min_contrast: f32 = 0.01,
    scale: f32 = 1,
    nt: f32 = 0,
    inlier2: f32 = 25,
    n_query: u32 = 0,
    n_db: u32 = 0,
    seed: u32 = 1,
    n_trials: u32 = 100000,
    grid: u32 = 45,
    kp_offset: u32 = 0,
    q_offset: u32 = 0,
    db_offset: u32 = 0,
    flags: u32 = 1,
    off_src: u32 = 0,
    scale_lo: f32 = 0.2,
    scale_hi: f32 = 5,
    off_dst: u32 = 0,
    nn_chunk: u32 = 0,
    border: i32 = 0,
    ratio: f32 = 0,
    max_ssd: f32 = 0,
    /// match_nnq: entries per workgroup row within a chunk (0: the whole chunk)
    nn_sub: u32 = 0,
};
comptime {
    std.debug.assert(@sizeOf(Params) == 128);
}

pub const Level = struct { offset: u32, count: u32, scale: f64 };
pub const Kp = extern struct { x: f32, y: f32, score: f32, pad: f32 };

/// A keypoint's descriptor frame at full resolution: the direction the descriptor starts from
/// (radians; image axes x right, y down, turning from +x toward +y) and the radius it reads.
pub const KpFrame = extern struct { angle: f32, radius: f32 };

pub const Structure = struct { n_sigma: u32 = 4, n_angle: u32 = 6, n_r: u32 = 3 };

pub const Scratch = struct { a: Buf, b: Buf, tmp: Buf, fmap: Buf, sr: Buf, scores: Buf };

pub const MatchOptions = struct {
    method: enum { lofsc, prosac, magsac } = .lofsc,
    homography: bool = true,
    mutual: bool = false,
    per_octave: bool = true,
};

pub const Settings = struct {
    n_octaves: u32 = 3,
    max_points: u32 = 2000,
    min_contrast: f64 = 0.01,
    radius: f64 = 36,
    tau: f64 = 0.8,
    nt: f64 = 0,
    auto_nt: bool = true,
    second_ori: bool = true,
    structure: Structure = .{},
    n_trials: u32 = 100000,
    inlier_px: f64 = 5,
    scale_lo: f64 = 0.2,
    scale_hi: f64 = 5,
    seed: u32 = 12345,
};

pub const Gls = struct {
    g: *Gpu,
    /// texel precision in use (half = f16)
    half: bool,
    packed_dot: bool,
    structure: Structure = .{},
    des_dim: u32 = 216,
    code32: []u8 = &.{},
    code16: []u8 = &.{},
    key_buf: [64]u8 = undefined,
    p: Params = .{},
    n_octaves: u32 = 3,
    auto_nt: bool = true,
    second_ori: bool = true,
    last_nt: f64 = 0,
    nt_used: std.ArrayList(f64) = .empty,
    last_levels: std.ArrayList(Level) = .empty,
    corr_count: u32 = 0,
    // buffers
    kps: Buf,
    kps_b: Buf,
    match_j: Buf,
    match_rev: Buf,
    mag_hist: Buf,
    corr: Buf,
    fsc_xy: Buf,
    /// the model in `affine` when it was last written from the host (writeH), as the f32 values
    /// the GPU holds, so readH needs no round trip; null once fsc_reduce has written it
    affine_host: ?[9]f32 = null,
    /// per query of the slice: its match when match_ok, else 0xffffffff (pack_fsc)
    fsc_ok: Buf,
    /// trials with a valid hypothesis (fsc_hyps), the ones fsc_trial scores
    fsc_list: Buf,
    trials: Buf,
    affine: Buf,
    counters: Buf,
    extrema: Buf,
    cell_best: Buf,
    cell_pix: Buf,
    cell_kp: Buf,
    /// what a fitted pose may do to the moving image (the engine sets it before a match)
    limits: ?match.Limits = null,
    ori2: Buf,
    /// per keypoint slot of kps / kps_b: the tangential sector (of N_TAN) its descriptor starts at
    kdir: Buf,
    kdir_b: Buf,
    des: Buf = .{},
    des_b: Buf = .{},
    desq: Buf = .{},
    desq_b: Buf = .{},
    dsc: Buf,
    dsc_b: Buf,
    clahe_hist: Buf,
    clahe_cdf: Buf,
    clahe_out: Buf = .{},
    nn_part: Buf = .{},
    rgba: Buf = .{},
    gray_up: Buf = .{},
    ones: Buf = .{},
    dog: [4]Buf = .{ .{}, .{}, .{}, .{} },
    dog_n: u32 = 0,
    scratch: std.AutoHashMapUnmanaged(u64, Scratch) = .empty,
    host_u32: std.ArrayList(u32) = .empty,
    host_f32: std.ArrayList(f32) = .empty,

    pub fn init(g: *Gpu, half: bool) !Gls {
        const half_ok = half and g.lim.has_f16 != 0;
        var s: Gls = .{
            .g = g,
            .half = half_ok,
            .packed_dot = g.lim.packed_dot != 0,
            .kps = g.storage(MAX_KP * 16),
            .kps_b = g.storage(MAX_KP * 16),
            .match_j = g.storage(MAX_KP * 4),
            .match_rev = g.storage(MAX_KP * 4),
            .mag_hist = g.storage(256 * 4),
            .corr = g.storage(MAX_CORR * 8),
            .fsc_xy = g.storage(MAX_KP * 16),
            .fsc_ok = g.storage(MAX_KP * 4),
            .fsc_list = g.storage(MAX_TRIALS * 4),
            .trials = g.storage(MAX_TRIALS * 32),
            .affine = g.storage(48),
            .counters = g.storage(COUNTERS_BYTES),
            .extrema = g.storage(16),
            .cell_best = g.storage(CELL_G * CELL_G * 4),
            .cell_pix = g.storage(CELL_G * CELL_G * 4),
            .cell_kp = g.storage(CELL_G * CELL_G * 16),
            .ori2 = g.storage(MAX_KP * 4),
            .kdir = g.storage(MAX_KP * 4),
            .kdir_b = g.storage(MAX_KP * 4),
            .dsc = g.storage(MAX_KP * 4),
            .dsc_b = g.storage(MAX_KP * 4),
            .clahe_hist = g.storage(8 * 8 * 256 * 4),
            .clahe_cdf = g.storage(8 * 8 * 256 * 4),
        };
        try s.buildCode();
        s.allocDescriptors();
        return s;
    }

    pub fn deinit(self: *Gls) void {
        const g = self.g;
        inline for (.{ "kps", "kps_b", "match_j", "match_rev", "mag_hist", "corr", "fsc_xy", "fsc_ok", "fsc_list", "trials", "affine", "counters", "extrema", "cell_best", "cell_pix", "cell_kp", "ori2", "kdir", "kdir_b", "des", "des_b", "desq", "desq_b", "dsc", "dsc_b", "clahe_hist", "clahe_cdf", "clahe_out", "nn_part", "rgba", "gray_up", "ones" }) |f|
            g.release(&@field(self, f));
        for (&self.dog) |*b| g.release(b);
        var it = self.scratch.valueIterator();
        while (it.next()) |sc| inline for (.{ "a", "b", "tmp", "fmap", "sr", "scores" }) |f| g.release(&@field(sc, f));
        self.scratch.deinit(g.gpa);
        g.gpa.free(self.code32);
        g.gpa.free(self.code16);
        self.nt_used.deinit(g.gpa);
        self.last_levels.deinit(g.gpa);
        self.host_u32.deinit(g.gpa);
        self.host_f32.deinit(g.gpa);
    }

    pub fn texelBytes(self: *const Gls) u64 {
        return if (self.half) 2 else 4;
    }

    pub fn storageTexels(self: *Gls, n: u64) Buf {
        return self.g.storage(n * self.texelBytes());
    }

    // ── shader source per structure and precision ──
    pub fn sub(gpa: std.mem.Allocator, src: []const u8, comptime name: []const u8, v: u32) ![]u8 {
        const pat = "const " ++ name ++ ": u32 = ";
        const i = std.mem.indexOf(u8, src, pat) orelse return error.ShaderConstant;
        const j = std.mem.indexOfPos(u8, src, i + pat.len, "u;") orelse return error.ShaderConstant;
        return std.fmt.allocPrint(gpa, "{s}{s}{d}{s}", .{ src[0..i], pat, v, src[j..] });
    }

    fn buildCode(self: *Gls) !void {
        const gpa = self.g.gpa;
        const st = self.structure;
        const des_dim = 2 * st.n_angle * st.n_angle * st.n_r;
        const vec = des_dim / 4;
        const kc = struct {
            fn f(limit: u32, v: u32) u32 {
                var k = @min(limit, v);
                while (k >= 1) : (k -= 1) if (v % k == 0) return k;
                return 1;
            }
        }.f;
        var c: []u8 = try gpa.dupe(u8, shaders.gls_mift);
        const subs = .{
            .{ "N_SIGMA", st.n_sigma }, .{ "N_ANGLE", st.n_angle }, .{ "N_R", st.n_r }, .{ "N_TAN", 2 * st.n_angle },
            .{ "DES_DIM", des_dim }, .{ "NN_VEC", vec }, .{ "NQ_KC", kc(9, vec) }, .{ "NN_KC", kc(3, vec) },
        };
        inline for (subs) |kv| {
            const next = try sub(gpa, c, kv[0], kv[1]);
            gpa.free(c);
            c = next;
        }
        defer gpa.free(c);
        const dot = self.dot4q();
        gpa.free(self.code32);
        gpa.free(self.code16);
        self.code32 = try std.fmt.allocPrint(gpa, "alias Texel = f32;\n{s}{s}", .{ dot, c });
        self.code16 = try std.fmt.allocPrint(gpa, "enable f16;\nalias Texel = f16;\n{s}{s}", .{ dot, c });
        self.des_dim = des_dim;
    }

    /// dot4q: the 8-bit descriptor screen's packed dot product (native, else shifts).
    pub fn dot4q(self: *const Gls) []const u8 {
        return if (self.packed_dot)
            "fn dot4q(a: u32, b: u32) -> u32 { return dot4U8Packed(a, b); }\n"
        else
            "fn dot4q(a: u32, b: u32) -> u32 {\n    return (a & 0xffu) * (b & 0xffu) + ((a >> 8u) & 0xffu) * ((b >> 8u) & 0xffu)\n        + ((a >> 16u) & 0xffu) * ((b >> 16u) & 0xffu) + (a >> 24u) * (b >> 24u);\n}\n";
    }

    fn allocDescriptors(self: *Gls) void {
        const bytes: u64 = @as(u64, MAX_KP) * self.des_dim * 4;
        if (self.des.id != 0 and self.des.size >= bytes) return;
        const g = self.g;
        g.release(&self.des);
        g.release(&self.des_b);
        g.release(&self.desq);
        g.release(&self.desq_b);
        self.des = g.storage(bytes);
        self.des_b = g.storage(bytes);
        self.desq = g.storage(@as(u64, MAX_KP) * self.des_dim);
        self.desq_b = g.storage(@as(u64, MAX_KP) * self.des_dim);
    }

    /// Change the descriptor structure (recompiles on use). Returns whether it changed.
    pub fn setStructure(self: *Gls, st: Structure) !bool {
        if (std.meta.eql(st, self.structure)) return false;
        self.structure = st;
        try self.buildCode();
        self.allocDescriptors();
        return true;
    }

    pub fn applySettings(self: *Gls, s: Settings) void {
        self.n_octaves = @max(1, s.n_octaves);
        self.p.max_points = @max(16, s.max_points);
        self.p.min_contrast = @floatCast(s.min_contrast);
        self.p.radius = @floatCast(s.radius);
        self.p.tau = @floatCast(s.tau);
        self.p.nt = @floatCast(s.nt);
        self.p.n_sigma = @max(1, s.structure.n_sigma);
        self.p.n_angle = @max(1, s.structure.n_angle);
        self.p.n_r = @max(1, s.structure.n_r);
        self.p.n_trials = @max(32, @min(MAX_TRIALS, s.n_trials));
        self.p.inlier2 = @floatCast(s.inlier_px * s.inlier_px);
        self.p.scale_lo = @floatCast(@max(1e-4, if (s.scale_lo > 0) s.scale_lo else 0.2));
        self.p.scale_hi = @floatCast(@max(@as(f64, self.p.scale_lo), if (s.scale_hi > 0) s.scale_hi else 5));
        self.p.seed = s.seed;
        self.p.grid = @max(1, @as(u32, @intFromFloat(@ceil(@sqrt(@as(f64, @floatFromInt(self.p.max_points)))))));
        self.auto_nt = s.auto_nt;
        self.second_ori = s.second_ori;
    }

    /// Pipeline of `entry`, texel precision f16 unless `f32`.
    pub fn pipe(self: *Gls, entry: []const u8, force32: bool) u32 {
        const use16 = self.half and !force32;
        const st = self.structure;
        const key = std.fmt.bufPrint(&self.key_buf, "gls{d}/{d}.{d}.{d}/{s}", .{ @as(u32, if (use16) 16 else 32), st.n_sigma, st.n_angle, st.n_r, entry }) catch entry;
        return self.g.pipeline(key, if (use16) self.code16 else self.code32, entry) catch |e| {
            std.log.err("gls_mift.wgsl {s}: {s}", .{ entry, @errorName(e) });
            return gpu_mod.NO_PIPE;
        };
    }

    pub fn P(self: *Gls) Bind {
        return self.g.uniform(0, std.mem.asBytes(&self.p));
    }

    pub fn Pwith(self: *Gls, over: Params) Bind {
        return self.g.uniform(0, std.mem.asBytes(&over));
    }

    pub fn run(self: *Gls, entry: []const u8, x: u32, y: u32, binds: []const Bind) void {
        self.g.dispatch(self.pipe(entry, false), x, y, 1, binds);
    }

    pub fn run32(self: *Gls, entry: []const u8, x: u32, y: u32, binds: []const Bind) void {
        self.g.dispatch(self.pipe(entry, true), x, y, 1, binds);
    }

    pub fn ensure(self: *Gls, w: u32, h: u32) !Scratch {
        const key = (@as(u64, w) << 32) | h;
        if (self.scratch.get(key)) |s| return s;
        const n: u64 = @as(u64, w) * h;
        const s: Scratch = .{
            .a = self.storageTexels(n), .b = self.storageTexels(n), .tmp = self.storageTexels(n),
            .fmap = self.storageTexels(n), .sr = self.g.storage(n * 4), .scores = self.storageTexels(n),
        };
        try self.scratch.put(self.g.gpa, key, s);
        return s;
    }

    pub fn readU32(self: *Gls, src: Buf, n: usize) ![]u32 {
        try self.host_u32.resize(self.g.gpa, n);
        self.g.read(src, 0, std.mem.sliceAsBytes(self.host_u32.items));
        try self.g.wait();
        return self.host_u32.items;
    }

    pub fn readF32(self: *Gls, src: Buf, n: usize) ![]f32 {
        try self.host_f32.resize(self.g.gpa, n);
        self.g.read(src, 0, std.mem.sliceAsBytes(self.host_f32.items));
        try self.g.wait();
        return self.host_f32.items;
    }

    /// Texel buffer → f32 values on the host (f16 decoded).
    fn readTexels(self: *Gls, src: Buf, n: usize, out: []f32) !void {
        if (!self.half) {
            self.g.read(src, 0, std.mem.sliceAsBytes(out[0..n]));
            try self.g.wait();
            return;
        }
        const tmp = try self.g.gpa.alloc(f16, n);
        defer self.g.gpa.free(tmp);
        self.g.read(src, 0, std.mem.sliceAsBytes(tmp));
        try self.g.wait();
        for (tmp, 0..) |v, i| out[i] = v;
    }

    // ── texel conversion ──
    pub fn packTexels(self: *Gls, src: Buf, dst: Buf, n: u32, src_off: u32, dst_off: u32) void {
        var q = self.p;
        q.w = n;
        q.h = 1;
        q.off_src = src_off;
        q.off_dst = dst_off;
        self.run("f32_to_texel", Gpu.flat((n + 255) / 256)[0], Gpu.flat((n + 255) / 256)[1], &.{ self.Pwith(q), dst.at(2), src.at(26) });
    }

    /// Texel buffer → f32 buffer (a copy when texels are f32).
    pub fn promoteTexels(self: *Gls, src: Buf, dst: Buf, n: u32, src_off: u32, dst_off: u32) void {
        if (!self.half) {
            self.g.copy(src, @as(u64, src_off) * 4, dst, @as(u64, dst_off) * 4, @as(u64, n) * 4);
            return;
        }
        var q = self.p;
        q.w = n;
        q.h = 1;
        q.off_src = src_off;
        q.off_dst = dst_off;
        self.run("texel_to_f32", Gpu.flat((n + 255) / 256)[0], Gpu.flat((n + 255) / 256)[1], &.{ self.Pwith(q), src.at(1), dst.at(26) });
    }

    // ── gray images ──
    pub const Gray = struct { raw: Buf = .{}, raw32: Buf = .{}, w: u32 = 0, h: u32 = 0 };

    /// RGBA8 image → gray (0.299 R + 0.587 G + 0.114 B), f32 then packed to texels when half.
    pub fn uploadRgba(self: *Gls, rgba: []const u8, w: u32, h: u32) !Gray {
        const g = self.g;
        const n: u64 = @as(u64, w) * h;
        g.ensure(&self.rgba, n * 4);
        g.write(self.rgba, 0, rgba[0..@intCast(n * 4)]);
        self.p.w = w;
        self.p.h = h;
        self.p.src_w = w;
        self.p.src_h = h;
        if (self.half) {
            const gray32 = g.storage(n * 4);
            self.run32("rgba_to_gray_buf", (w + 7) / 8, (h + 7) / 8, &.{ self.P(), gray32.at(2), self.rgba.at(4) });
            const gray = self.storageTexels(n);
            self.packTexels(gray32, gray, @intCast(n), 0, 0);
            return .{ .raw = gray, .raw32 = gray32, .w = w, .h = h };
        }
        const gray = self.storageTexels(n);
        self.run("rgba_to_gray_buf", (w + 7) / 8, (h + 7) / 8, &.{ self.P(), gray.at(2), self.rgba.at(4) });
        return .{ .raw = gray, .w = w, .h = h };
    }

    /// f32 gray (0..1) → raw texels (and the f32 copy when half).
    pub fn uploadGray(self: *Gls, data: []const f32, w: u32, h: u32) Gray {
        const g = self.g;
        const n: u64 = @as(u64, w) * h;
        const gray32 = g.storage(n * 4);
        g.write(gray32, 0, std.mem.sliceAsBytes(data[0..@intCast(n)]));
        if (!self.half) return .{ .raw = gray32, .w = w, .h = h };
        const gray = self.storageTexels(n);
        self.packTexels(gray32, gray, @intCast(n), 0, 0);
        return .{ .raw = gray, .raw32 = gray32, .w = w, .h = h };
    }

    fn gauss(self: *Gls, src: Buf, dst: Buf, w: u32, h: u32) !void {
        const mid = (try self.ensure(w, h)).tmp;
        self.p.w = w;
        self.p.h = h;
        self.run("gauss_h", (w + 7) / 8, (h + 7) / 8, &.{ self.P(), src.at(1), mid.at(2) });
        self.run("gauss_v", (w + 7) / 8, (h + 7) / 8, &.{ self.P(), mid.at(1), dst.at(2) });
    }

    fn resize(self: *Gls, src: Buf, dst: Buf, sw: u32, sh: u32, dw: u32, dh: u32) void {
        self.p.src_w = sw;
        self.p.src_h = sh;
        self.p.w = dw;
        self.p.h = dh;
        self.run("resize_bilinear", (dw + 7) / 8, (dh + 7) / 8, &.{ self.P(), src.at(1), dst.at(2) });
    }

    /// Contrast-limited adaptive histogram equalization (engine.js clahe): src → dst texels.
    /// `src` is f32 when half (the raw f32 copy), texels otherwise.
    pub fn clahe(self: *Gls, src: Buf, dst: Buf, w: u32, h: u32, clip_limit: f64, n_bins0: u32, grid: u32) void {
        const snap = self.p;
        const hg: u32 = @min(8, @max(2, grid));
        const n_bins: u32 = @min(256, @max(8, n_bins0));
        const hp = h + ((hg - h % hg) % hg);
        const wp = w + ((hg - w % hg) % hg);
        self.p.w = w;
        self.p.h = h;
        self.p.src_w = @max(1, hp / hg);
        self.p.src_h = @max(1, wp / hg);
        self.p.n_query = hg;
        self.p.n_db = hg;
        self.p.seed = n_bins;
        self.p.n_trials = (hg - h % hg) % hg;
        self.p.grid = (hg - w % hg) % hg;
        self.p.tau = @floatCast(clip_limit);
        const n: u64 = @as(u64, w) * h;
        var out = dst;
        if (self.half) {
            self.g.ensure(&self.clahe_out, n * 4);
            out = self.clahe_out;
        }
        const f32k = self.half;
        self.g.clear(self.clahe_hist);
        self.g.dispatch(self.pipe("clahe_hist", f32k), (wp + 7) / 8, (hp + 7) / 8, 1, &.{ self.P(), src.at(1), self.clahe_hist.at(24) });
        self.g.dispatch(self.pipe("clahe_cdf", f32k), (hg * hg + 15) / 16, 1, 1, &.{ self.P(), self.clahe_hist.at(24), self.clahe_cdf.at(25) });
        self.g.dispatch(self.pipe("clahe_apply", f32k), (w + 7) / 8, (h + 7) / 8, 1, &.{ self.P(), src.at(1), out.at(2), self.clahe_cdf.at(25) });
        if (self.half) self.packTexels(out, dst, @intCast(n), 0, 0);
        self.p = snap;
    }

    fn ensureDog(self: *Gls, w: u32, h: u32) void {
        const n = w * h;
        if (self.dog_n == n and self.dog[0].id != 0) return;
        for (&self.dog) |*b| {
            self.g.release(b);
            b.* = self.storageTexels(n);
        }
        self.dog_n = n;
    }

    fn blurPasses(self: *Gls, src: Buf, dst: Buf, w: u32, h: u32, passes: u32) !void {
        if (passes == 0) {
            if (src.id != dst.id) self.g.copy(src, 0, dst, 0, @as(u64, w) * h * self.texelBytes());
            return;
        }
        self.ensureDog(w, h);
        var from = src;
        var i: u32 = 0;
        while (i < passes) : (i += 1) {
            const to = if (i == passes - 1) dst else if (i % 2 == 0) self.dog[0] else self.dog[1];
            try self.gauss(from, to, w, h);
            from = to;
        }
    }

    /// Host f32 values → texel buffer.
    fn writeGray(self: *Gls, dst: Buf, data: []const f32) void {
        const g = self.g;
        if (!self.half) {
            g.write(dst, 0, std.mem.sliceAsBytes(data));
            return;
        }
        g.ensure(&self.gray_up, data.len * 4);
        g.write(self.gray_up, 0, std.mem.sliceAsBytes(data));
        self.packTexels(self.gray_up, dst, @intCast(data.len), 0, 0);
    }

    /// Difference of Gaussians, rescaled to [0, 1] (engine.js bandpass).
    pub fn bandpass(self: *Gls, buf: Buf, w: u32, h: u32, fine_px: f64, coarse_px: f64) !void {
        const fine_n: u32 = @intFromFloat(@max(1, @min(16, @round(std.math.pow(f64, if (fine_px > 0) fine_px else 1, 2)))));
        const coarse_n: u32 = @max(fine_n + 1, @as(u32, @intFromFloat(@min(64, @round(std.math.pow(f64, if (coarse_px > 0) coarse_px else 6, 2))))));
        self.ensureDog(w, h);
        try self.blurPasses(buf, self.dog[2], w, h, fine_n);
        try self.blurPasses(self.dog[2], self.dog[3], w, h, coarse_n - fine_n);
        const n: usize = @as(usize, w) * h;
        const gpa = self.g.gpa;
        const fine = try gpa.alloc(f32, n);
        defer gpa.free(fine);
        const coarse = try gpa.alloc(f32, n);
        defer gpa.free(coarse);
        try self.readTexels(self.dog[2], n, fine);
        try self.readTexels(self.dog[3], n, coarse);
        var lo: f32 = std.math.inf(f32);
        var hi: f32 = -std.math.inf(f32);
        for (fine, coarse) |*f, cv| {
            // readGray clamps to [0, 1]
            const v = std.math.clamp(f.*, 0, 1) - std.math.clamp(cv, 0, 1);
            f.* = v;
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        const sc: f32 = if (hi > lo) 1 / (hi - lo) else 0;
        for (fine) |*f| f.* = (f.* - lo) * sc;
        self.writeGray(buf, fine);
    }

    pub fn invert(self: *Gls, buf: Buf, w: u32, h: u32) !void {
        const n: usize = @as(usize, w) * h;
        const a = try self.g.gpa.alloc(f32, n);
        defer self.g.gpa.free(a);
        try self.readTexels(buf, n, a);
        for (a) |*v| v.* = 1 - std.math.clamp(v.*, 0, 1);
        self.writeGray(buf, a);
    }

    // ── detection ──
    fn medianAbsResponse(self: *Gls, gray: Buf, w: u32, h: u32) !f64 {
        self.g.clear(self.mag_hist);
        self.p.w = w;
        self.p.h = h;
        self.p.src_w = w;
        self.p.src_h = h;
        self.run("mag_hist", (w + 7) / 8, (h + 7) / 8, &.{ self.P(), gray.at(1), self.mag_hist.at(27) });
        const hist = try self.readU32(self.mag_hist, 256);
        var total: f64 = 0;
        for (hist) |v| total += @floatFromInt(v);
        if (total == 0) return 0;
        const half = total / 2;
        var acc: f64 = 0;
        var bin: usize = 255;
        for (0..256) |i| {
            acc += @floatFromInt(hist[i]);
            if (acc >= half) {
                bin = i;
                break;
            }
        }
        const hb: f64 = @floatFromInt(hist[bin]);
        const prev = acc - hb;
        const frac = if (hb > 0) (half - prev) / hb else 0.5;
        const log_lo = -20 + (32.0 / 256.0) * @as(f64, @floatFromInt(bin));
        return std.math.pow(f64, 2, log_lo + (32.0 / 256.0) * frac);
    }

    /// One pyramid level at the running offset (counters[6]); its counts land in level record
    /// `level` (levelDone), read once for the whole pyramid.
    fn detectLevel(self: *Gls, gray: Buf, w: u32, h: u32, scale: f64, level: u32, kps: Buf, des: Buf, desq: Buf, dsc: Buf, kdir: Buf) !void {
        const g = self.g;
        const s = try self.ensure(w, h);
        self.p.w = w;
        self.p.h = h;
        self.p.scale = @floatCast(scale);
        self.p.kp_offset = KP_RUNNING;
        self.p.grid = @max(1, @as(u32, @intFromFloat(@ceil(@sqrt(@as(f64, @floatFromInt(self.p.max_points)))))));
        self.p.n_query = self.p.max_points;
        self.p.flags = @intFromBool(self.second_ori);
        const ex = [2]u32{ @bitCast(std.math.inf(f32)), 0 };
        g.writeSlice(self.extrema, 0, u32, &ex);
        const gx = (w + 7) / 8;
        const gy = (h + 7) / 8;
        self.clearCounts();
        g.clear(self.cell_best);
        g.clear(self.cell_pix);
        g.clear(self.cell_kp);
        g.clear(s.scores);
        self.run("make_fmap", gx, gy, &.{ self.P(), gray.at(1), s.fmap.at(3), s.sr.at(4), self.extrema.at(10) });
        self.run("normalize_fmap", gx, gy, &.{ self.P(), s.fmap.at(3), self.extrema.at(10) });
        self.run("fast_nms", gx, gy, &.{ self.P(), s.fmap.at(3), s.scores.at(5) });
        self.run("nms_and_cells", gx, gy, &.{ self.P(), s.scores.at(5), self.cell_best.at(6) });
        self.run("cell_tie", gx, gy, &.{ self.P(), s.scores.at(5), self.cell_best.at(6), self.cell_pix.at(19) });
        self.run("emit_kps", gx, gy, &.{ self.P(), s.scores.at(5), self.cell_best.at(6), self.cell_kp.at(17), self.cell_pix.at(19) });
        self.run("compact_kps", 1, 1, &.{ self.P(), kps.at(7), self.counters.at(9), self.cell_kp.at(17) });
        const nk = self.p.max_points;
        const dx = @min(nk, 32768);
        const dy = (nk + dx - 1) / dx;
        self.run("describe", dx, dy, &.{ self.P(), s.sr.at(4), kps.at(7), des.at(8), self.counters.at(9), self.ori2.at(18), desq.at(20), dsc.at(22), kdir.at(34) });
        self.run("compact_extra", 1, 1, &.{ self.P(), self.counters.at(9), self.ori2.at(18) });
        self.run("describe_extra", dx, dy, &.{ self.P(), s.sr.at(4), kps.at(7), des.at(8), self.counters.at(9), self.ori2.at(18), desq.at(20), dsc.at(22), kdir.at(34) });
        self.levelDone(level, true);
    }

    /// Zero the per-operation counts (counters[0..6)), keeping the running offset and records.
    pub fn clearCounts(self: *Gls) void {
        self.g.clearRange(self.counters, 0, 24);
    }

    /// Start a pyramid at keypoint 0 (counters[6]).
    pub fn beginLevels(self: *Gls) void {
        self.g.clearRange(self.counters, 24, 8);
    }

    /// Record the level's counts (with the extra orientations when `extras`) and advance the
    /// running offset.
    pub fn levelDone(self: *Gls, level: u32, extras: bool) void {
        var q = self.p;
        q.db_offset = level;
        q.n_db = @intFromBool(extras);
        self.run("level_done", 1, 1, &.{ self.Pwith(q), self.counters.at(9) });
    }

    /// The first `n` level records: offset, keypoints (with extras), base, per level.
    pub fn readLevels(self: *Gls, n: u32) ![]const u32 {
        const all = try self.readU32(self.counters, LEVEL_REC + 4 * n);
        return all[LEVEL_REC..];
    }

    /// Detect and describe on the scale pyramid (levels 1, 2/3, 1/2, 1/3, …) of a work image.
    /// side 0 writes kps / des, side 1 kps_b / des_b. Returns the keypoint count; levels in
    /// last_levels.
    pub fn detect(self: *Gls, gray: Buf, w: u32, h: u32, side: u32) !u32 {
        const kps = if (side == 0) self.kps else self.kps_b;
        const des = if (side == 0) self.des else self.des_b;
        const desq = if (side == 0) self.desq else self.desq_b;
        const dsc = if (side == 0) self.dsc else self.dsc_b;
        const kdir = if (side == 0) self.kdir else self.kdir_b;
        const gpa = self.g.gpa;
        self.g.clear(kps);
        self.g.clear(des);
        self.p.flags = @intFromBool(self.second_ori);
        if (self.auto_nt) self.p.nt = @floatCast(try self.medianAbsResponse(gray, w, h));
        self.last_nt = self.p.nt;
        try self.nt_used.append(gpa, self.last_nt);
        self.last_levels.clearRetainingCapacity();
        self.beginLevels();
        // levels run as (scale, whether the loop stopped after it on a full keypoint budget);
        // their counts come back together
        const Lv = struct { scale: f64, octave_end: bool };
        var lvs: std.ArrayList(Lv) = .empty;
        defer lvs.deinit(gpa);
        var cur_a = gray;
        var wa = w;
        var ha = h;
        const sa = try self.ensure(w, h);
        try self.gauss(cur_a, sa.b, wa, ha);
        var wb: u32 = @max(1, @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(w)) * 2.0 / 3.0))));
        var hb: u32 = @max(1, @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(h)) * 2.0 / 3.0))));
        const sb = try self.ensure(wb, hb);
        var cur_b = sb.a;
        self.resize(sa.b, cur_b, wa, ha, wb, hb);
        var o: u32 = 0;
        while (o < self.n_octaves) : (o += 1) {
            if (@min(wa, ha) < 16) break;
            const sc = std.math.pow(f64, 2, @floatFromInt(o));
            try self.detectLevel(cur_a, wa, ha, sc, @intCast(lvs.items.len), kps, des, desq, dsc, kdir);
            try lvs.append(gpa, .{ .scale = sc, .octave_end = false });
            if (@min(wb, hb) >= 16) {
                try self.detectLevel(cur_b, wb, hb, 1.5 * sc, @intCast(lvs.items.len), kps, des, desq, dsc, kdir);
                try lvs.append(gpa, .{ .scale = 1.5 * sc, .octave_end = false });
            }
            lvs.items[lvs.items.len - 1].octave_end = true;
            if (o == self.n_octaves - 1) break;
            const scur = try self.ensure(wa, ha);
            try self.gauss(cur_a, scur.b, wa, ha);
            const wa2 = @max(1, wa >> 1);
            const ha2 = @max(1, ha >> 1);
            const na2 = (try self.ensure(wa2, ha2)).a;
            self.resize(scur.b, na2, wa, ha, wa2, ha2);
            cur_a = na2;
            wa = wa2;
            ha = ha2;
            const sbb = try self.ensure(wb, hb);
            try self.gauss(cur_b, sbb.b, wb, hb);
            const wb2 = @max(1, wb >> 1);
            const hb2 = @max(1, hb >> 1);
            const nb2 = (try self.ensure(wb2, hb2)).a;
            self.resize(sbb.b, nb2, wb, hb, wb2, hb2);
            cur_b = nb2;
            wb = wb2;
            hb = hb2;
        }
        const rec = try self.readLevels(@intCast(lvs.items.len));
        var total: u32 = 0;
        for (lvs.items, 0..) |lv, i| {
            const cnt = rec[4 * i + 1];
            try self.last_levels.append(gpa, .{ .offset = total, .count = cnt, .scale = lv.scale });
            total += cnt;
            // (the budget ends the pyramid after a full octave; later levels found no room)
            if (lv.octave_end and total >= MAX_KP) break;
        }
        return @min(total, MAX_KP);
    }

    // ── matching ──
    /// Screened 1-NN layout: the database in chunks (their best two per query are the
    /// candidates, as before), each walked by `rows / chunks` workgroup rows of `sub` entries,
    /// so a level pair keeps the GPU busy; nn_part holds nq × rows entries (≤ 32 MB).
    pub const NnRows = struct { chunk: u32, sub: u32, rows: u32 };

    pub fn nnRows(nq: u32, nd: u32) NnRows {
        const cap: u32 = @intFromFloat(@floor(2097152.0 / @as(f64, @floatFromInt(nq))));
        const rows0: u32 = @max(1, @min(@min(64, (nd + 511) / 512), cap));
        const chunk = ((nd + rows0 - 1) / rows0 + 63) / 64 * 64;
        const chunks = (nd + chunk - 1) / chunk;
        // sub-rows per chunk: one per 64 entries, within the memory cap
        const per_max = @max(1, @min(chunk / 64, cap / chunks));
        const sub_n = (chunk / 64 + per_max - 1) / per_max * 64;
        const per = (chunk + sub_n - 1) / sub_n;
        return .{ .chunk = chunk, .sub = sub_n, .rows = chunks * per };
    }

    fn matchNN(self: *Gls, rev: bool) void {
        const nq = if (rev) self.p.n_db else self.p.n_query;
        const nd = if (rev) self.p.n_query else self.p.n_db;
        if (nq < 1 or nd < 1) return;
        const nr = nnRows(nq, nd);
        self.g.ensure(&self.nn_part, @as(u64, nq) * nr.rows * 16);
        var q = self.p;
        q.nn_chunk = nr.chunk;
        q.nn_sub = nr.sub;
        const name = if (rev) "match_nnq_rev" else "match_nnq";
        self.run(name, (nq + 63) / 64, nr.rows, &.{ self.Pwith(q), self.kps.at(7), self.desq.at(20), self.desq_b.at(21), self.dsc.at(22), self.dsc_b.at(31), self.nn_part.at(29) });
        if (rev) {
            self.run("match_nnq_rev_pick", (nq + 63) / 64, 1, &.{ self.Pwith(q), self.des.at(8), self.des_b.at(13), self.nn_part.at(29), self.match_rev.at(23) });
        } else {
            self.run("match_nnq_pick", (nq + 63) / 64, 1, &.{ self.Pwith(q), self.des.at(8), self.des_b.at(13), self.nn_part.at(29), self.kps.at(7), self.match_j.at(14) });
        }
    }

    fn mutualSlice(self: *Gls) void {
        self.matchNN(true);
        self.run("keep_mutual", (self.p.n_query + 63) / 64, 1, &.{ self.P(), self.match_j.at(14), self.match_rev.at(23) });
    }

    pub fn fscTrials(self: *Gls, n_query: u32, n_trials: u32) void {
        self.run("zero_pack", 1, 1, &.{self.counters.at(9)});
        self.run("pack_fsc", (n_query + 63) / 64, 1, &.{ self.P(), self.kps.at(7), self.counters.at(9), self.kps_b.at(12), self.match_j.at(14), self.fsc_xy.at(30), self.fsc_ok.at(32) });
        self.run("fsc_hyps", (n_trials + 63) / 64, 1, &.{ self.P(), self.kps.at(7), self.counters.at(9), self.kps_b.at(12), self.trials.at(15), self.fsc_ok.at(32), self.fsc_list.at(33) });
        self.countTrials(n_trials);
    }

    /// Inlier counts of the listed trials (fsc_hyps / fsc_hyps_corr) over fsc_xy.
    fn countTrials(self: *Gls, n_trials: u32) void {
        self.run("fsc_trial", (n_trials + 255) / 256, 1, &.{ self.P(), self.counters.at(9), self.trials.at(15), self.fsc_xy.at(30), self.fsc_list.at(33) });
    }

    pub fn fscReduce(self: *Gls) void {
        self.run("fsc_reduce", 1, 1, &.{ self.counters.at(9), self.trials.at(15), self.affine.at(16) });
        self.affine_host = null;
    }

    fn matchPerOctave(self: *Gls, n1: u32, n2: u32, levels_l: []const Level, levels_r: []const Level, mutual: bool) !void {
        const seed0 = self.p.seed;
        var pair: u32 = 0;
        for (levels_l) |a| for (levels_r) |b| {
            if (a.count < 3 or b.count < 3) continue;
            self.p.n_query = a.count;
            self.p.n_db = b.count;
            self.p.q_offset = a.offset;
            self.p.db_offset = b.offset;
            self.p.seed = seed0 +% pair *% 10007;
            pair += 1;
            self.matchNN(false);
            if (mutual) self.mutualSlice();
            self.fscTrials(a.count, self.p.n_trials);
            self.fscReduce();
            self.run("keep_corr", (a.count + 63) / 64, 1, &.{ self.P(), self.kps.at(7), self.counters.at(9), self.kps_b.at(12), self.match_j.at(14), self.affine.at(16), self.corr.at(28) });
        };
        self.p.seed = seed0;
        self.p.q_offset = 0;
        self.p.db_offset = 0;
        self.p.n_query = n1;
        self.p.n_db = n2;
    }

    pub fn readH(self: *Gls) !lie.Mat3 {
        var H: lie.Mat3 = undefined;
        if (self.affine_host) |a| {
            for (0..9) |i| H[i] = a[i];
            return H;
        }
        const a = try self.readF32(self.affine, 9);
        for (0..9) |i| H[i] = a[i];
        return H;
    }

    pub fn writeH(self: *Gls, H: lie.Mat3) void {
        var h32: [9]f32 = undefined;
        for (0..9) |i| h32[i] = @floatCast(H[i]);
        self.g.writeSlice(self.affine, 0, f32, &h32);
        self.affine_host = h32;
    }

    pub fn readKps(self: *Gls, side: u32, n: usize, out: []Kp) !void {
        const src = if (side == 0) self.kps else self.kps_b;
        self.g.read(src, 0, std.mem.sliceAsBytes(out[0..n]));
        try self.g.wait();
    }

    /// Descriptor frames of a side's first kps.len keypoints (readKps order): the sector each
    /// descriptor starts at, and the outer ring with its overlap (P.radius + P.radius / N_R²).
    pub fn kpFrames(self: *Gls, side: u32, kps: []const Kp, out: []KpFrame) !void {
        if (kps.len == 0) return;
        const dirs = try self.readU32(if (side == 0) self.kdir else self.kdir_b, kps.len);
        const step = 2 * std.math.pi / @as(f32, @floatFromInt(2 * self.structure.n_angle));
        const nr: f32 = @floatFromInt(self.structure.n_r);
        const r = self.p.radius * (1 + 1 / (nr * nr));
        for (kps, dirs, out) |k, d, *o| o.* = .{ .angle = @as(f32, @floatFromInt(d)) * step, .radius = r * k.pad };
    }

    fn resid2(H: lie.Mat3, px: f64, py: f64, qx: f64, qy: f64) f64 {
        const X = H[0] * px + H[1] * py + H[2];
        const Y = H[3] * px + H[4] * py + H[5];
        const Z = H[6] * px + H[7] * py + H[8];
        const z = if (@abs(Z) < 1e-12) 1e-12 else Z;
        const dx = X / z - qx;
        const dy = Y / z - qy;
        return dx * dx + dy * dy;
    }

    const Pair = struct { qi: u32, dj: u32, e2: f64 };

    fn writeMatches(self: *Gls, n1: u32, pairs: []const Pair, H: lie.Mat3) !u32 {
        const gpa = self.g.gpa;
        const mj = try gpa.alloc(u32, n1);
        defer gpa.free(mj);
        @memset(mj, 0xffffffff);
        const fit2: f64 = self.p.inlier2;
        var ninl: u32 = 0;
        for (pairs) |pp| {
            if (pp.e2 < fit2 and mj[pp.qi] == 0xffffffff) {
                mj[pp.qi] = pp.dj;
                ninl += 1;
            }
        }
        self.g.writeSlice(self.match_j, 0, u32, mj);
        self.writeH(H);
        return ninl;
    }

    /// Fit the pooled correspondences (counters[3] of them in corr, appended by keep_corr /
    /// append_corr); their count lands in corr_count. The count, the list and both keypoint sets
    /// come back in one readback.
    pub fn fitCorr(self: *Gls, n1: u32, n2: u32, opt: MatchOptions, n_trials: u32) !u32 {
        const gpa = self.g.gpa;
        var cnt: [4]u32 = undefined;
        const raw = try gpa.alloc(u32, MAX_CORR * 2);
        defer gpa.free(raw);
        const k1 = try gpa.alloc(Kp, @max(n1, 1));
        defer gpa.free(k1);
        const k2 = try gpa.alloc(Kp, @max(n2, 1));
        defer gpa.free(k2);
        self.g.read(self.counters, 0, std.mem.sliceAsBytes(&cnt));
        self.g.read(self.corr, 0, std.mem.sliceAsBytes(raw));
        self.g.read(self.kps, 0, std.mem.sliceAsBytes(k1));
        self.g.read(self.kps_b, 0, std.mem.sliceAsBytes(k2));
        try self.g.wait();
        self.corr_count = @min(MAX_CORR, cnt[3]);
        const ncorr = self.corr_count;
        if (ncorr < 3) {
            self.writeH(lie.identity);
            try self.fillMatchJ(n1);
            return 0;
        }
        // keep_corr appends with atomics: sort (query, match) so the trials see a fixed list
        var idx: std.ArrayList([2]u32) = .empty;
        defer idx.deinit(gpa);
        for (0..ncorr) |i| {
            const qi = raw[i * 2];
            const dj = raw[i * 2 + 1];
            if (qi < n1 and dj < n2) try idx.append(gpa, .{ qi, dj });
        }
        std.sort.pdq([2]u32, idx.items, {}, struct {
            fn lt(_: void, a: [2]u32, b: [2]u32) bool {
                return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1]);
            }
        }.lt);
        const m = idx.items.len;
        const pts = try gpa.alloc(f64, 4 * @max(m, 1));
        defer gpa.free(pts);
        const px = pts[0..m];
        const py = pts[m .. 2 * m];
        const qx = pts[2 * m .. 3 * m];
        const qy = pts[3 * m .. 4 * m];
        for (idx.items, 0..) |pr, i| {
            px[i] = k1[pr[0]].x;
            py[i] = k1[pr[0]].y;
            qx[i] = k2[pr[1]].x;
            qy[i] = k2[pr[1]].y;
        }
        const pk = try gpa.alloc(u32, @max(1, m) * 2);
        defer gpa.free(pk);
        for (idx.items, 0..) |pr, i| {
            pk[i * 2] = pr[0];
            pk[i * 2 + 1] = pr[1];
        }
        self.g.writeSlice(self.corr, 0, u32, pk);
        if (m < 3) {
            self.writeH(lie.identity);
            try self.fillMatchJ(n1);
            return 0;
        }
        var H: lie.Mat3 = undefined;
        if (opt.method == .lofsc) {
            self.p.n_query = @intCast(m);
            self.p.n_db = n2;
            self.p.q_offset = 0;
            self.p.db_offset = 0;
            self.p.n_trials = n_trials;
            self.run("zero_pack", 1, 1, &.{self.counters.at(9)});
            self.run("pack_corr", (@as(u32, @intCast(m)) + 63) / 64, 1, &.{ self.P(), self.kps.at(7), self.counters.at(9), self.kps_b.at(12), self.corr.at(28), self.fsc_xy.at(30) });
            self.run("fsc_hyps_corr", (n_trials + 63) / 64, 1, &.{ self.P(), self.kps.at(7), self.counters.at(9), self.kps_b.at(12), self.trials.at(15), self.corr.at(28), self.fsc_list.at(33) });
            self.countTrials(n_trials);
            self.fscReduce();
            H = try self.readH();
            const fit2: f64 = self.p.inlier2;
            var ip: std.ArrayList([2]f64) = .empty;
            defer ip.deinit(gpa);
            var iq: std.ArrayList([2]f64) = .empty;
            defer iq.deinit(gpa);
            for (0..m) |i| if (resid2(H, px[i], py[i], qx[i], qy[i]) < fit2) {
                try ip.append(gpa, .{ px[i], py[i] });
                try iq.append(gpa, .{ qx[i], qy[i] });
            };
            if (ip.items.len >= 3) H = lie.HAffineFromPts(ip.items, iq.items, null);
        } else {
            const fit = try match.fitRobust(gpa, px, py, qx, qy, null, .{
                .method = if (opt.method == .magsac) .magsac else .prosac, .homography = opt.homography,
                .inlier_px = @sqrt(@as(f64, self.p.inlier2)), .n_trials = @min(n_trials, 8192), .seed = self.p.seed,
                .scale_lo = self.p.scale_lo, .scale_hi = self.p.scale_hi, .limits = self.limits,
            });
            H = fit.H;
        }
        const kept = try gpa.alloc(Pair, m);
        defer gpa.free(kept);
        for (idx.items, 0..) |pr, i| kept[i] = .{ .qi = pr[0], .dj = pr[1], .e2 = resid2(H, px[i], py[i], qx[i], qy[i]) };
        // stable by residual (Array.prototype.sort is stable)
        std.sort.block(Pair, kept, {}, struct {
            fn lt(_: void, a: Pair, b: Pair) bool {
                return a.e2 < b.e2;
            }
        }.lt);
        return self.writeMatches(n1, kept, H);
    }

    pub fn fillMatchJ(self: *Gls, n: u32) !void {
        const gpa = self.g.gpa;
        const mj = try gpa.alloc(u32, @max(n, 1));
        defer gpa.free(mj);
        @memset(mj, 0xffffffff);
        self.g.writeSlice(self.match_j, 0, u32, mj);
    }

    fn refineAffine(self: *Gls, n1: u32, accept_px: f64) !u32 {
        var H = try self.readH();
        if (n1 < 3) return 0;
        const gpa = self.g.gpa;
        const n2 = self.p.n_db;
        const k1 = try gpa.alloc(Kp, n1);
        defer gpa.free(k1);
        const k2 = try gpa.alloc(Kp, @max(n2, 1));
        defer gpa.free(k2);
        try self.readKps(0, n1, k1);
        try self.readKps(1, @max(n2, 1), k2);
        const mj = try gpa.dupe(u32, try self.readU32(self.match_j, n1));
        defer gpa.free(mj);
        var p: std.ArrayList([2]f64) = .empty;
        defer p.deinit(gpa);
        var q: std.ArrayList([2]f64) = .empty;
        defer q.deinit(gpa);
        for (0..n1) |i| if (mj[i] != 0xffffffff and mj[i] < n2) {
            try p.append(gpa, .{ k1[i].x, k1[i].y });
            try q.append(gpa, .{ k2[mj[i]].x, k2[mj[i]].y });
        };
        if (p.items.len < 3) return 0;
        const fit2 = accept_px * accept_px;
        var ip: std.ArrayList([2]f64) = .empty;
        defer ip.deinit(gpa);
        var iq: std.ArrayList([2]f64) = .empty;
        defer iq.deinit(gpa);
        for (p.items, q.items) |a, b| if (resid2(H, a[0], a[1], b[0], b[1]) < fit2) {
            try ip.append(gpa, a);
            try iq.append(gpa, b);
        };
        if (ip.items.len >= 3) H = lie.HAffineFromPts(ip.items, iq.items, null);
        var ninl: u32 = 0;
        for (p.items, q.items) |a, b| {
            if (resid2(H, a[0], a[1], b[0], b[1]) < fit2) ninl += 1;
        }
        self.writeH(H);
        return ninl;
    }

    fn fitMatchesJs(self: *Gls, n1: u32, n2: u32, opt: MatchOptions, n_trials: u32) !struct { H: lie.Mat3, ninl: u32 } {
        const gpa = self.g.gpa;
        const k1 = try gpa.alloc(Kp, n1);
        defer gpa.free(k1);
        const k2 = try gpa.alloc(Kp, n2);
        defer gpa.free(k2);
        try self.readKps(0, n1, k1);
        try self.readKps(1, n2, k2);
        const mj = try gpa.dupe(u32, try self.readU32(self.match_j, n1));
        defer gpa.free(mj);
        if (opt.mutual) {
            const rev = try gpa.alloc(i64, n2);
            defer gpa.free(rev);
            @memset(rev, -1);
            for (0..n1) |i| {
                const j = mj[i];
                if (j != 0xffffffff and j < n2 and rev[j] < 0) rev[j] = @intCast(i);
            }
            for (0..n1) |i| {
                const j = mj[i];
                if (j != 0xffffffff and j < n2 and rev[j] != @as(i64, @intCast(i))) mj[i] = 0xffffffff;
            }
        }
        const dim = self.des_dim;
        const desA = try gpa.dupe(f32, try self.readF32(self.des, @as(usize, n1) * dim));
        defer gpa.free(desA);
        const desB = try gpa.dupe(f32, try self.readF32(self.des_b, @as(usize, n2) * dim));
        defer gpa.free(desB);
        const sc = try gpa.alloc(f32, n1);
        defer gpa.free(sc);
        match.matchDots(desA, desB, mj, n1, dim, sc);
        var cols: [5]std.ArrayList(f64) = .{ .empty, .empty, .empty, .empty, .empty };
        defer for (&cols) |*c| c.deinit(gpa);
        for (0..n1) |i| {
            const j = mj[i];
            if (j == 0xffffffff or j >= n2) continue;
            try cols[0].append(gpa, k1[i].x);
            try cols[1].append(gpa, k1[i].y);
            try cols[2].append(gpa, k2[j].x);
            try cols[3].append(gpa, k2[j].y);
            try cols[4].append(gpa, sc[i]);
        }
        const fit = try match.fitRobust(gpa, cols[0].items, cols[1].items, cols[2].items, cols[3].items, cols[4].items, .{
            .method = if (opt.method == .magsac) .magsac else .prosac, .homography = opt.homography,
            .inlier_px = @sqrt(@as(f64, self.p.inlier2)), .n_trials = @min(n_trials, 8192), .seed = self.p.seed,
            .scale_lo = self.p.scale_lo, .scale_hi = self.p.scale_hi, .limits = self.limits,
        });
        self.writeH(fit.H);
        return .{ .H = fit.H, .ninl = fit.ninl };
    }

    /// Match side 0 to side 1 and fit the pose (engine.js matchAndWarp without the warp).
    pub fn matchPair(self: *Gls, n1: u32, n2: u32, lw: u32, lh: u32, rw: u32, rh: u32, opt: MatchOptions, levels_l: []const Level, levels_r: []const Level) !struct { H: lie.Mat3, ninl: u32 } {
        const gpa = self.g.gpa;
        self.p.n_query = n1;
        self.p.n_db = n2;
        self.p.q_offset = 0;
        self.p.db_offset = 0;
        self.p.w = rw;
        self.p.h = rh;
        self.p.src_w = lw;
        self.p.src_h = lh;
        const n_trials = @min(MAX_TRIALS, self.p.n_trials);
        self.p.n_trials = n_trials;
        self.g.clear(self.counters);
        self.corr_count = 0;
        const unset = try gpa.alloc(u32, MAX_KP);
        defer gpa.free(unset);
        @memset(unset, 0xffffffff);
        self.g.writeSlice(self.match_j, 0, u32, unset);
        const use_octave = opt.per_octave and levels_l.len > 0 and levels_r.len > 0;
        if (use_octave) {
            try self.matchPerOctave(n1, n2, levels_l, levels_r, opt.mutual);
        } else if (n1 >= 1 and n2 >= 1) {
            self.matchNN(false);
            if (opt.mutual) self.mutualSlice();
        }
        var ninl: u32 = 0;
        if (use_octave) {
            ninl = try self.fitCorr(n1, n2, opt, n_trials);
        } else if (opt.method != .lofsc) {
            ninl = (try self.fitMatchesJs(n1, n2, opt, n_trials)).ninl;
        } else if (n1 >= 3 and n2 >= 3) {
            self.fscTrials(n1, n_trials);
            self.fscReduce();
            ninl = try self.refineAffine(n1, @sqrt(@max(@as(f64, self.p.inlier2), 1e-8)));
        } else self.writeH(lie.identity);
        return .{ .H = try self.readH(), .ninl = ninl };
    }
};
