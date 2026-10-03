//! Pose-level scores and gradients (smi.js poseStats / tangentGradBatch / ffdGradBatch):
//! direct sums over the fixed image of every pixel pair at a pose, many poses per submit.
//!
//! One direction: moving features `a` are sampled at H⁻¹(x) for every fixed pixel x of `b`.
//! The symmetric pair score (mean of this and the inverse direction) is built on top (pair.zig).
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const smi_mod = @import("smi.zig");
const lie = @import("lie.zig");
const mom = @import("moments.zig");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Bind = gpu_mod.Bind;
const Features = smi_mod.Features;

pub const FFD_MAX = 16;
const FFD_CHUNK_CPS = 16;
const FFD_CHUNK_PAR = FFD_CHUNK_CPS * 2;
pub const GRAD_STRIDE = 44;
/// where a lane's selection buffer keeps the coefficients (smi.wgsl gather_sel: 256 bytes in, a
/// storage binding offset)
const SEL_COEF = 64;
/// sums per lattice cell of smi.wgsl ffd_cells
pub const FFD_CELL = 332;

/// Workgroups per pose of the per-pixel reductions (moments, gradients): about 32 pixels per
/// invocation, 64 to 1024 (more keeps a large image's
/// pass from running on a few SMs).
/// Pixels each thread sums before its workgroup reduces (the 48-value workgroup reduction costs
/// as much as a few hundred pixels of work, so threads should sum many).
pub var px_per_thread: u64 = 1024;
const MIN_PX = 64;
/// Workgroups a pass keeps at least, over all its poses (so a few poses still fill the GPU).
pub var fill_wg: u64 = 64;

/// Workgroups per pose for the per-pixel sum kernels (each pose's partial sums, one per workgroup).
fn nParts(pixels: u64, poses: u64) u32 {
    const by_work = (pixels + 256 * px_per_thread - 1) / (256 * px_per_thread);
    const by_fill = (fill_wg + @max(poses, 1) - 1) / @max(poses, 1);
    // and never so many that a thread sums fewer than MIN_PX pixels
    const most = @max(1, @min(1024, (pixels + 256 * MIN_PX - 1) / (256 * MIN_PX)));
    return @intCast(std.math.clamp(@max(by_work, by_fill), 1, most));
}

/// The tangent pass's pixels per thread, and the most workgroups it uses over all its poses.
pub var tg_px: u64 = 16;
pub var tg_cap: u64 = 256;

/// Workgroups per pose for tangent_grad: its per-pixel work is light next to its 44-value
/// workgroup reduction, so threads sum about tg_px pixels each, up to tg_cap workgroups in all.
fn nPartsTangent(pixels: u64, poses: u64) u32 {
    const want = (pixels + 256 * tg_px - 1) / (256 * tg_px);
    const cap = @max(1, tg_cap / @max(poses, 1));
    return @intCast(std.math.clamp(want, 1, @min(cap, 1024)));
}

/// The B-spline displacement a pass applies: lattice size (0 = none), the frame its lattice
/// lives in (fixed "target" or moving "source"), and the control points to bind.
pub const FfdBind = struct {
    gx: u32 = 0,
    gy: u32 = 0,
    source: bool = false,
    /// control points (bound at slot 11); per-pose packs when `per_pose`
    cps: Buf = .{},
    per_pose: bool = false,
};

pub const Cfg = struct {
    family: mom.Family = .smi,
    exact: bool = true,
    edge: bool = false,
    ffd: FfdBind = .{},
};

pub const Grad = struct {
    score: f64 = 0,
    n: f64 = 0,
    grad: lie.Coords = @splat(0),
    /// Gauss–Newton matrix (nk×nk row-major) when requested (SMI only)
    hess: ?[64]f64 = null,
};

/// One direction's work buffers: its poses, partial sums, gradient coefficients and tangent
/// generators. A symmetric score records the forward and inverse directions in two lanes and
/// reads both back with one wait.
pub const Lane = struct {
    hs: Buf = .{},
    part: Buf = .{},
    /// the tangent pass's partial sums (read back with the moments')
    tpart: Buf = .{},
    cmb: Buf = .{},
    gens: Buf = .{},
    /// the GPU's f32 score per pose (moments_coefs); the GPU climb's chosen pose and its
    /// coefficients (gather_sel: the pose at 0, the coefficients from SEL_COEF floats)
    sc: Buf = .{},
    selb: Buf = .{},
    /// the GPU climb's spline terms per lattice cell (ffd_cells), and per chunk of a cell
    ffdp: Buf = .{},
    ffdq: Buf = .{},
    raw: std.ArrayList(f32) = .empty,
    raw2: std.ArrayList(f32) = .empty,
    /// workgroups per pose of the lane's last pass (nParts)
    nparts: u32 = 64,
};

/// One direction of a pass: images (a warped onto b), poses and config.
pub const Dir = struct {
    a: *const Features,
    b: *const Features,
    Hs: []const lie.Mat3,
    cfg: Cfg,
};

pub const PoseGpu = struct {
    smi: *smi_mod.Smi,
    g: *Gpu,
    lanes: [2]Lane,
    /// all-zero control points (bound when no spline applies)
    cps0: Buf,
    ffd_parts: Buf = .{},

    pub fn init(smi: *smi_mod.Smi) PoseGpu {
        const g = smi.g;
        var self: PoseGpu = .{ .smi = smi, .g = g, .lanes = .{ .{}, .{} }, .cps0 = g.storage(FFD_MAX * FFD_MAX * 8) };
        for (&self.lanes) |*l| l.gens = g.storage(8 * 9 * 4);
        g.clear(self.cps0);
        return self;
    }

    pub fn deinit(self: *PoseGpu) void {
        for (&self.lanes) |*l| {
            inline for (.{ "hs", "part", "tpart", "cmb", "gens", "sc", "selb", "ffdp", "ffdq" }) |f| self.g.release(&@field(l, f));
            l.raw.deinit(self.g.gpa);
            l.raw2.deinit(self.g.gpa);
        }
        self.g.release(&self.cps0);
        self.g.release(&self.ffd_parts);
    }

    fn writeHs(self: *PoseGpu, lane: usize, Hs: []const lie.Mat3) !void {
        const g = self.g;
        const l = &self.lanes[lane];
        g.ensure(&l.hs, Hs.len * 9 * 4);
        const pk = try g.gpa.alloc(f32, Hs.len * 9);
        defer g.gpa.free(pk);
        for (Hs, 0..) |H, i| for (0..9) |k| {
            pk[i * 9 + k] = @floatCast(H[k]);
        };
        g.writeSlice(l.hs, 0, f32, pk);
    }

    fn cpsBuf(self: *PoseGpu, cfg: Cfg) Buf {
        return if (cfg.ffd.cps.id != 0) cfg.ffd.cps else self.cps0;
    }

    /// Edge-weight flag and the weight buffers (slots 12, 13); feature buffers stand in when off.
    fn weights(cfg: Cfg, a: *const Features, b: *const Features) struct { flag: u32, wa: Buf, wb: Buf } {
        if (cfg.edge) return .{ .flag = 1, .wa = a.mag, .wb = b.mag };
        return .{ .flag = 0, .wa = a.feat, .wb = b.feat };
    }

    /// Queue a read of the first `floats` of `src` into the lane's staging array (valid after
    /// the next wait).
    fn queueRead(self: *PoseGpu, lane: usize, src: Buf, floats: usize) ![]f32 {
        return queueInto(self.g, &self.lanes[lane].raw, src, floats);
    }

    fn queueInto(g: *Gpu, dst: *std.ArrayList(f32), src: Buf, floats: usize) ![]f32 {
        try dst.resize(g.gpa, floats);
        g.read(src, 0, std.mem.sliceAsBytes(dst.items));
        return dst.items;
    }

    fn readRaw(self: *PoseGpu, src: Buf, floats: usize) ![]f32 {
        const r = try self.queueRead(0, src, floats);
        try self.g.wait();
        return r;
    }

    /// The overlap moment sums (moments.zig layout) of a onto b at one pose.
    pub fn momentSums(self: *PoseGpu, a: *const Features, b: *const Features, H: lie.Mat3, cfg: Cfg, out: *[mom.N_MOM]f64) !void {
        try self.momentsPass(0, a, b, &.{H}, cfg);
        const np = self.lanes[0].nparts;
        const raw = try self.readRaw(self.lanes[0].part, @as(usize, np) * mom.N_MOM);
        mom.sumPart(raw, np, mom.N_MOM, out);
    }

    fn momentsPass(self: *PoseGpu, lane: usize, a: *const Features, b: *const Features, Hs: []const lie.Mat3, cfg: Cfg) !void {
        try self.writeHs(lane, Hs);
        const l = &self.lanes[lane];
        l.nparts = self.momentsOn(a, b, l.hs, @intCast(Hs.len), cfg, &l.part);
    }

    /// The moments pass over the n poses in `hs` into `part`; returns its workgroups per pose.
    fn momentsOn(self: *PoseGpu, a: *const Features, b: *const Features, hs: Buf, n: u32, cfg: Cfg, part: *Buf) u32 {
        const np = nParts(@as(u64, b.w) * b.h, n);
        self.g.ensure(part, @as(u64, n) * np * mom.N_MOM * 4);
        const W = weights(cfg, a, b);
        const s = self.smi;
        s.run("overlap_moments", np, 1, n, &.{
            s.un(.{
                .w = b.w, .h = b.h, .cw = a.w, .ch = a.h, .n_keys = n, .plane = @intFromBool(cfg.ffd.per_pose), .plane_b = W.flag,
                .gx = cfg.ffd.gx, .gy = cfg.ffd.gy, .theme = @intFromBool(cfg.ffd.gx >= 2 and cfg.ffd.source),
            }),
            a.feat.at(1), b.feat.at(2), hs.at(3), part.at(4), self.cpsBuf(cfg).at(11), W.wa.at(12), W.wb.at(13),
        });
        return np;
    }

    /// moments_coefs over n poses' partial sums (np rows each): coefficients into the lane's cmb,
    /// f32 scores into its sc.
    fn coefsOn(self: *PoseGpu, lane: usize, part: Buf, n: u32, np: u32, cfg: Cfg) void {
        self.coefsLanes(&.{lane}, &.{part}, n, .{ np, np }, cfg);
    }

    /// moments_coefs for one lane or both (lanes[0], then lanes[1]) in one dispatch: each lane's
    /// n poses' partial sums (np[k] rows each) into its cmb and sc. The kernel binds two lanes'
    /// outputs; a single lane's second ones are the other lane's buffers, left untouched.
    fn coefsLanes(self: *PoseGpu, lanes: []const usize, parts: []const Buf, n: u32, np: [2]u32, cfg: Cfg) void {
        const l0 = &self.lanes[lanes[0]];
        const l1 = &self.lanes[if (lanes.len == 2) lanes[1] else 1 - lanes[0]];
        for ([_]*Lane{ l0, l1 }) |l| {
            self.g.ensure(&l.cmb, @as(u64, @max(n, 1)) * mom.N_COEF * 4);
            self.g.ensure(&l.sc, @as(u64, @max(n, 4)) * 4);
        }
        const two = lanes.len == 2;
        const s = self.smi;
        s.run("moments_coefs", n, @intCast(lanes.len), 1, &.{
            s.un(.{
                .w = 1, .h = 1, .n = np[0], .n_keys = np[1], .cw = n, .ch = if (two) n else 0, .plane = @intFromBool(cfg.exact),
                .ridge = @floatCast(mom.RIDGE), .gx = familyCode(cfg.family),
            }),
            parts[0].at(1), parts[if (two) 1 else 0].at(3), l0.cmb.at(4), l0.sc.at(5), l1.cmb.at(14), l1.sc.at(15),
        });
    }

    /// moments_coefs' score family (u.gx): 0 SMI, 2 E4, 3 λmax (copula.wgsl copula_score modes).
    fn familyCode(f: mom.Family) u32 {
        return switch (f) {
            .smi => 0,
            .e4 => 2,
            .lmax => 3,
        };
    }

    /// tangent_grad over the n poses in `hs` with coefficients `cmb`, into the lane's tpart;
    /// returns its workgroups per pose.
    fn tangentOn(self: *PoseGpu, lane: usize, d: Dir, hs: Bind, cmb: Bind, n: u32, group: lie.Group, hess: bool) u32 {
        const l = &self.lanes[lane];
        self.writeGens(lane, group, d.b.w, d.b.h);
        const np = nPartsTangent(@as(u64, d.b.w) * d.b.h, n);
        self.g.ensure(&l.tpart, @as(u64, n) * np * GRAD_STRIDE * 4);
        const W = weights(d.cfg, d.a, d.b);
        const s = self.smi;
        s.run("tangent_grad", np, 1, n, &.{
            s.un(.{
                .w = d.b.w, .h = d.b.h, .n = n, .cw = d.a.w, .ch = d.a.h, .n_keys = @intCast(lie.nKeys(group)), .plane = @intFromBool(d.cfg.ffd.per_pose), .plane_b = W.flag,
                .min_n = @intFromBool(hess), .gx = d.cfg.ffd.gx, .gy = d.cfg.ffd.gy, .theme = @intFromBool(d.cfg.ffd.gx >= 2 and d.cfg.ffd.source),
            }),
            d.a.feat.at(1), d.b.feat.at(2), hs, l.tpart.at(4), l.gens.at(9), cmb, self.cpsBuf(d.cfg).at(11), W.wa.at(12), W.wb.at(13),
        });
        return np;
    }

    /// A gradient from a pose's moment sums (host f64 score) and its tangent sums.
    fn gradFrom(mraw: []const f32, mnp: u32, traw: []const f32, tnp: u32, cfg: Cfg, nk: usize, hess: bool) Grad {
        var sum: [mom.N_MOM]f64 = undefined;
        mom.sumPart(mraw, mnp, mom.N_MOM, &sum);
        const st = mom.scoreFromMoments(&sum, cfg.exact, mom.RIDGE);
        var r: Grad = .{ .score = st.score, .n = st.n };
        if (hess) r.hess = @splat(0);
        if (st.n >= mom.MIN_N) {
            var acc: [GRAD_STRIDE]f64 = undefined;
            mom.sumPart(traw, tnp, GRAD_STRIDE, &acc);
            for (0..nk) |q| r.grad[q] = acc[q];
            if (hess) {
                var h: usize = 8;
                for (0..8) |q| for (q..8) |m| {
                    if (q < nk and m < nk) {
                        r.hess.?[q * nk + m] = acc[h];
                        r.hess.?[m * nk + q] = acc[h];
                    }
                    h += 1;
                };
            }
        }
        return r;
    }

    /// Benchmarks: `iters` moments passes over the same poses, recorded and waited for once.
    pub fn benchMoments(self: *PoseGpu, a: *const Features, b: *const Features, Hs: []const lie.Mat3, cfg: Cfg, iters: u32) !void {
        for (0..iters) |_| try self.momentsPass(0, a, b, Hs, cfg);
        try self.g.wait();
    }

    fn collectStats(raw: []const f32, np: u32, cfg: Cfg, score_only: bool, out: []mom.Stats) void {
        for (out, 0..) |*o, i| {
            var st: [mom.N_MOM]f64 = undefined;
            mom.sumPart(raw[i * np * mom.N_MOM ..], np, mom.N_MOM, &st);
            o.* = if (cfg.family == .smi) mom.scoreFromMoments(&st, cfg.exact, mom.RIDGE) else mom.copulaFromMoments(&st, cfg.family, !score_only);
        }
    }

    /// Scores (and gradient coefficients unless score_only) at every pose of up to two
    /// directions, recorded together and read back with one wait.
    pub fn stats2(self: *PoseGpu, dirs: []const Dir, score_only: bool, out: []const []mom.Stats) !void {
        var raws: [2][]f32 = undefined;
        var any = false;
        for (dirs, 0..) |d, k| {
            if (d.Hs.len == 0) continue;
            try self.momentsPass(k, d.a, d.b, d.Hs, d.cfg);
            raws[k] = try self.queueRead(k, self.lanes[k].part, d.Hs.len * self.lanes[k].nparts * mom.N_MOM);
            any = true;
        }
        if (!any) return;
        try self.g.wait();
        for (dirs, 0..) |d, k| if (d.Hs.len > 0) collectStats(raws[k], self.lanes[k].nparts, d.cfg, score_only, out[k][0..d.Hs.len]);
    }

    /// Scores (and gradient coefficients unless score_only) at every pose.
    pub fn stats(self: *PoseGpu, a: *const Features, b: *const Features, Hs: []const lie.Mat3, cfg: Cfg, score_only: bool, out: []mom.Stats) !void {
        try self.stats2(&.{.{ .a = a, .b = b, .Hs = Hs, .cfg = cfg }}, score_only, &.{out});
    }

    /// Scores of up to two directions with one wait.
    pub fn scores2(self: *PoseGpu, dirs: []const Dir, out: []const []f64) !void {
        const gpa = self.g.gpa;
        var st: [2][]mom.Stats = .{ &.{}, &.{} };
        defer for (st) |x| gpa.free(x);
        for (dirs, 0..) |d, k| st[k] = try gpa.alloc(mom.Stats, d.Hs.len);
        try self.stats2(dirs, true, st[0..dirs.len]);
        for (dirs, 0..) |d, k| for (0..d.Hs.len) |i| {
            out[k][i] = st[k][i].score;
        };
    }

    pub fn scores(self: *PoseGpu, a: *const Features, b: *const Features, Hs: []const lie.Mat3, cfg: Cfg, out: []f64) !void {
        try self.scores2(&.{.{ .a = a, .b = b, .Hs = Hs, .cfg = cfg }}, &.{out});
    }

    fn writeCoefs(self: *PoseGpu, lane: usize, st: []const mom.Stats) !void {
        const g = self.g;
        const l = &self.lanes[lane];
        g.ensure(&l.cmb, @max(1, st.len) * mom.N_COEF * 4);
        const pk = try g.gpa.alloc(f32, st.len * mom.N_COEF);
        defer g.gpa.free(pk);
        for (st, 0..) |s, i| @memcpy(pk[i * mom.N_COEF ..][0..mom.N_COEF], &s.coef);
        g.writeSlice(l.cmb, 0, f32, pk);
    }

    fn writeGens(self: *PoseGpu, lane: usize, group: lie.Group, wb: u32, hb: u32) void {
        const gens = lie.generators(group, .{ @floatFromInt(wb), @floatFromInt(hb) });
        var pk: [72]f32 = @splat(0);
        for (0..lie.nKeys(group)) |k| for (0..9) |j| {
            pk[k * 9 + j] = @floatCast(gens[k][j]);
        };
        self.g.writeSlice(self.lanes[lane].gens, 0, f32, &pk);
    }

    /// Score, ∇score over the group's tangent coordinates, and (hess) the Gauss–Newton matrix at
    /// every pose of up to two directions: one wait for both directions' moments, one for both
    /// directions' tangent sums.
    pub fn grads2(self: *PoseGpu, dirs: []const Dir, group: lie.Group, hess: bool, out: []const []Grad) !void {
        var smi_only = true;
        for (dirs) |d| smi_only = smi_only and d.cfg.family == .smi;
        if (smi_only) return self.grads2Gpu(dirs, group, hess, out);
        const gpa = self.g.gpa;
        var st: [2][]mom.Stats = .{ &.{}, &.{} };
        defer for (st) |x| gpa.free(x);
        for (dirs, 0..) |d, k| st[k] = try gpa.alloc(mom.Stats, d.Hs.len);
        try self.stats2(dirs, false, st[0..dirs.len]);
        const nk: u32 = @intCast(lie.nKeys(group));
        var raws: [2][]f32 = undefined;
        var nps: [2]u32 = undefined;
        var any = false;
        for (dirs, 0..) |d, k| {
            const n: u32 = @intCast(d.Hs.len);
            if (n == 0) continue;
            self.writeGens(k, group, d.b.w, d.b.h);
            try self.writeCoefs(k, st[k]);
            const l = &self.lanes[k];
            const np = nPartsTangent(@as(u64, d.b.w) * d.b.h, n);
            nps[k] = np;
            self.g.ensure(&l.part, @as(u64, n) * np * GRAD_STRIDE * 4);
            const W = weights(d.cfg, d.a, d.b);
            const s = self.smi;
            s.run("tangent_grad", np, 1, n, &.{
                s.un(.{
                    .w = d.b.w, .h = d.b.h, .n = n, .cw = d.a.w, .ch = d.a.h, .n_keys = nk, .plane = @intFromBool(d.cfg.ffd.per_pose), .plane_b = W.flag,
                    .min_n = @intFromBool(hess), .gx = d.cfg.ffd.gx, .gy = d.cfg.ffd.gy, .theme = @intFromBool(d.cfg.ffd.gx >= 2 and d.cfg.ffd.source),
                }),
                d.a.feat.at(1), d.b.feat.at(2), l.hs.at(3), l.part.at(4), l.gens.at(9), l.cmb.at(10), self.cpsBuf(d.cfg).at(11), W.wa.at(12), W.wb.at(13),
            });
            raws[k] = try self.queueRead(k, l.part, @as(usize, n) * np * GRAD_STRIDE);
            any = true;
        }
        if (!any) return;
        try self.g.wait();
        for (dirs, 0..) |d, k| for (0..d.Hs.len) |i| {
            var r: Grad = .{ .score = st[k][i].score, .n = st[k][i].n };
            const want_h = hess and d.cfg.family != .e4;
            if (want_h) r.hess = @splat(0);
            if (st[k][i].n >= mom.MIN_N) {
                var acc: [GRAD_STRIDE]f64 = undefined;
                mom.sumPart(raws[k][i * nps[k] * GRAD_STRIDE ..], nps[k], GRAD_STRIDE, &acc);
                for (0..nk) |q| r.grad[q] = acc[q];
                if (want_h) {
                    var h: usize = 8;
                    for (0..8) |q| for (q..8) |m| {
                        if (q < nk and m < nk) {
                            r.hess.?[q * nk + m] = acc[h];
                            r.hess.?[m * nk + q] = acc[h];
                        }
                        h += 1;
                    };
                }
            }
            out[k][i] = r;
        };
    }

    /// grads2 for SMI with the coefficients made on the GPU (moments_coefs): every direction's
    /// moments, coefficients and tangent sums in one submission and one wait. The scores and
    /// overlaps come from the moments on the host (f64), as in stats.
    fn grads2Gpu(self: *PoseGpu, dirs: []const Dir, group: lie.Group, hess: bool, out: []const []Grad) !void {
        const g = self.g;
        const nk = lie.nKeys(group);
        var mraw: [2][]f32 = undefined;
        var traw: [2][]f32 = undefined;
        var mnp: [2]u32 = undefined;
        var tnp: [2]u32 = undefined;
        var any = false;
        for (dirs, 0..) |d, k| {
            const n: u32 = @intCast(d.Hs.len);
            if (n == 0) continue;
            try self.momentsPass(k, d.a, d.b, d.Hs, d.cfg);
            const l = &self.lanes[k];
            mnp[k] = l.nparts;
            self.coefsOn(k, l.part, n, mnp[k], d.cfg);
            tnp[k] = self.tangentOn(k, d, l.hs.at(3), l.cmb.at(10), n, group, hess);
            mraw[k] = try queueInto(g, &l.raw, l.part, @as(usize, n) * mnp[k] * mom.N_MOM);
            traw[k] = try queueInto(g, &l.raw2, l.tpart, @as(usize, n) * tnp[k] * GRAD_STRIDE);
            any = true;
        }
        if (!any) return;
        try g.wait();
        for (dirs, 0..) |d, k| for (0..d.Hs.len) |pi| {
            out[k][pi] = gradFrom(mraw[k][pi * mnp[k] * mom.N_MOM ..], mnp[k], traw[k][pi * tnp[k] * GRAD_STRIDE ..], tnp[k], d.cfg, nk, hess);
        };
    }

    // ── the GPU-resident climb's stages (gclimb.zig): poses written by climb kernels ──

    /// Record the scores of the n_all poses already in each direction's lane (hs): overlap
    /// moments, then moments_coefs (f32 scores into the lane's sc, coefficients into its cmb).
    pub fn stageScores(self: *PoseGpu, dirs: []const Dir, n_all: u32) void {
        for (dirs, 0..) |d, k| {
            const l = &self.lanes[k];
            l.nparts = self.momentsOn(d.a, d.b, l.hs, n_all, d.cfg, &l.part);
        }
        const L = &self.lanes;
        if (dirs.len == 2) {
            self.coefsLanes(&.{ 0, 1 }, &.{ L[0].part, L[1].part }, n_all, .{ L[0].nparts, L[1].nparts }, dirs[0].cfg);
        } else self.coefsOn(0, L[0].part, n_all, L[0].nparts, dirs[0].cfg);
    }

    /// A lane's chosen pose and its coefficients (gather_sel) as tangent_grad's / ffd_cells'
    /// pose (slot 3) and coefficient (slot 10) bindings.
    fn selPose(l: *const Lane) Bind {
        return l.selb.range(3, 0, 16 * 4);
    }
    fn selCoef(l: *const Lane) Bind {
        return l.selb.range(10, SEL_COEF * 4, mom.N_COEF * 4);
    }

    /// Record the gradient at the pose `pick` (index buffer, from climb_select) chose in each
    /// lane: its pose and coefficients gathered, then tangent_grad into the lane's tpart.
    /// Returns the workgroups per pose of each lane's tangent sums.
    pub fn stageGrad(self: *PoseGpu, dirs: []const Dir, pick: Buf, group: lie.Group) [2]u32 {
        const s = self.smi;
        const L = &self.lanes;
        for (L) |*l| self.g.ensure(&l.selb, (SEL_COEF + mom.N_COEF) * 4);
        const two = dirs.len == 2;
        const l1 = &L[if (two) 1 else 0];
        s.run("gather_sel", 1, 1, 1, &.{
            s.un(.{ .w = 1, .h = 1, .n = @intCast(dirs.len) }), L[0].cmb.at(1), L[0].hs.at(3), pick.at(9), l1.cmb.at(10), l1.hs.at(11), L[0].selb.at(4), L[1].selb.at(5),
        });
        var tnp: [2]u32 = .{ 0, 0 };
        for (dirs, 0..) |d, k| tnp[k] = self.tangentOn(k, d, selPose(&L[k]), selCoef(&L[k]), 1, group, true);
        return tnp;
    }

    /// Sums per lattice cell (smi.wgsl ffd_cells: FFD_CELL each) of the spline's gradient and
    /// Gauss–Newton terms at the lane's chosen pose (selb: the pose and its coefficients), into
    /// the lane's ffdp.
    pub fn stageFfdCells(self: *PoseGpu, lane: usize, d: Dir) void {
        const l = &self.lanes[lane];
        const gx = d.cfg.ffd.gx;
        const gy = d.cfg.ffd.gy;
        const ncell = gx * gy;
        self.g.ensure(&l.ffdp, @as(u64, ncell) * FFD_CELL * 4);
        // each cell's pixels in chunks of about 8 tiles, so a coarse lattice still fills the GPU
        const nch: u32 = @intCast(std.math.clamp((@as(u64, d.b.w) * d.b.h / ncell + 2047) / 2048, 1, 64));
        if (nch > 1) self.g.ensure(&l.ffdq, @as(u64, ncell) * nch * FFD_CELL * 4);
        const W = weights(d.cfg, d.a, d.b);
        const s = self.smi;
        const un = s.un(.{ .w = d.b.w, .h = d.b.h, .n = 1, .cw = d.a.w, .ch = d.a.h, .gx = gx, .gy = gy, .plane_b = W.flag, .plane = 0, .theme = @intFromBool(d.cfg.ffd.source), .n_keys = nch });
        s.run("ffd_cells", ncell * nch, 1, 1, &.{
            un, d.a.feat.at(1), d.b.feat.at(2), selPose(l), (if (nch > 1) l.ffdq else l.ffdp).at(4), selCoef(l), self.cpsBuf(d.cfg).at(11), W.wa.at(12), W.wb.at(13),
        });
        if (nch > 1) s.run("ffd_cells_sum", (ncell * FFD_CELL + 63) / 64, 1, 1, &.{ un, l.ffdq.at(1), l.ffdp.at(4) });
    }

    /// Score, ∇score over the group's tangent coordinates, and (hess) the Gauss–Newton matrix.
    pub fn grads(self: *PoseGpu, a: *const Features, b: *const Features, Hs: []const lie.Mat3, group: lie.Group, cfg: Cfg, hess: bool, out: []Grad) !void {
        try self.grads2(&.{.{ .a = a, .b = b, .Hs = Hs, .cfg = cfg }}, group, hess, &.{out});
    }

    /// Score and ∇ over the spline control points (2·gx·gy, x then y per point) at every pose.
    pub fn ffdGrads(self: *PoseGpu, a: *const Features, b: *const Features, Hs: []const lie.Mat3, cfg: Cfg, scores_out: []f64, grads_out: []f64) !void {
        const n: u32 = @intCast(Hs.len);
        const gx = cfg.ffd.gx;
        const gy = cfg.ffd.gy;
        const npar = gx * gy * 2;
        @memset(grads_out[0 .. n * npar], 0);
        if (n == 0 or gx < 2) return;
        const gpa = self.g.gpa;
        const st = try gpa.alloc(mom.Stats, n);
        defer gpa.free(st);
        try self.stats(a, b, Hs, cfg, false, st);
        try self.writeCoefs(0, st);
        const l = &self.lanes[0];
        const ncp = gx * gy;
        const n_chunks = (ncp + FFD_CHUNK_CPS - 1) / FFD_CHUNK_CPS;
        const np = nParts(@as(u64, b.w) * b.h, @as(u64, n) * n_chunks);
        const one: usize = @as(usize, n) * np * FFD_CHUNK_PAR;
        self.g.ensure(&l.part, one * 4);
        self.g.ensure(&self.ffd_parts, n_chunks * one * 4);
        const W = weights(cfg, a, b);
        const s = self.smi;
        var cp0: u32 = 0;
        var i: u32 = 0;
        while (cp0 < ncp) : ({
            cp0 += FFD_CHUNK_CPS;
            i += 1;
        }) {
            s.run("ffd_grad", np, 1, n, &.{
                s.un(.{
                    .w = b.w, .h = b.h, .n = n, .cw = a.w, .ch = a.h, .gx = gx, .gy = gy, .n_keys = cp0, .plane_b = W.flag,
                    .plane = @intFromBool(cfg.ffd.per_pose), .theme = @intFromBool(cfg.ffd.source),
                }),
                a.feat.at(1), b.feat.at(2), l.hs.at(3), l.part.at(4), l.cmb.at(10), self.cpsBuf(cfg).at(11), W.wa.at(12), W.wb.at(13),
            });
            self.g.copy(l.part, 0, self.ffd_parts, i * one * 4, one * 4);
        }
        const all = try self.readRaw(self.ffd_parts, n_chunks * one);
        const slot = np * FFD_CHUNK_PAR;
        cp0 = 0;
        i = 0;
        while (cp0 < ncp) : ({
            cp0 += FFD_CHUNK_CPS;
            i += 1;
        }) {
            const take = @min(FFD_CHUNK_PAR, npar - cp0 * 2);
            for (0..n) |p| {
                if (!(st[p].n >= mom.MIN_N)) continue;
                var acc: [FFD_CHUNK_PAR]f64 = undefined;
                mom.sumPart(all[i * one + p * slot ..], np, FFD_CHUNK_PAR, &acc);
                for (0..take) |k| grads_out[p * npar + cp0 * 2 + k] = if (std.math.isFinite(acc[k])) acc[k] else 0;
            }
        }
        for (0..n) |p| scores_out[p] = st[p].score;
    }
};
