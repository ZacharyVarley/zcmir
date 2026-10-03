//! The pair's pose score and gradients, as the app computes them (app.js pairParts, scorePoses,
//! pairGrad, pairGradBatch, symGrad, pairFfdGrad): the forward direction (moving warped onto
//! fixed) and, when symmetric, the inverse direction (fixed warped onto moving by H⁻¹, with the
//! spline's adjoint), averaged. Gradients of the inverse direction are pulled back to the
//! forward tangent space (lie.adjointGrad).
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const smi_mod = @import("smi.zig");
const pose_mod = @import("pose.zig");
const lie = @import("lie.zig");
const mom = @import("moments.zig");

const Buf = gpu_mod.Buf;
const Features = smi_mod.Features;
const PoseGpu = pose_mod.PoseGpu;

/// B-spline free-form deformation on top of the homography (app state.cps / eng.setFfd).
pub const Ffd = struct {
    on: bool = false,
    g: u32 = 4,
    /// lattice in the moving image ("source") instead of the fixed image ("target")
    source: bool = false,
    /// control points, 2·g·g (x, y per point) in destN units
    cps: [pose_mod.FFD_MAX * pose_mod.FFD_MAX * 2]f32 = @splat(0),

    pub fn n(self: *const Ffd) usize {
        return self.g * self.g * 2;
    }
    pub fn slice(self: *Ffd) []f32 {
        return self.cps[0..self.n()];
    }
    pub fn live(self: *const Ffd) bool {
        if (!self.on) return false;
        for (self.cps[0..self.n()]) |v| if (@abs(v) > 1e-12) return true;
        return false;
    }
};

pub const Settings = struct {
    family: mom.Family = .smi,
    exact: bool = true,
    edge: bool = false,
    symmetric: bool = true,
};

pub const Parts = struct { fwd: f64, inv: ?f64, mean: f64 };

/// A score beats another when it is higher by more than 1e-6 of the other's magnitude: above the
/// round-off of the GPU's f32 moment sums, so steps that only gain round-off are not taken (the
/// GPU climb's climb_select applies the same rule).
pub fn beats(new: f64, old: f64) bool {
    return new > old + @max(1e-6 * @abs(old), 1e-12);
}

/// A pose's direction scores, remembered with the context they were computed in.
const Memo = struct { H: lie.Mat3, ctx: u64, fwd: f64, inv: ?f64 };
const MEMO_N = 32;

pub const Pair = struct {
    pg: *PoseGpu,
    a: *const Features,
    b: *const Features,
    set: Settings,
    ffd: *const Ffd,
    cps: Buf = .{},
    cps_adj: Buf = .{},
    cps_pose: Buf = .{},
    cps_pose_adj: Buf = .{},
    /// Recently scored poses (each scoring path records here; parts() looks here first), so a
    /// pose the climb has just scored is not scored again for its trail entry.
    memo: [MEMO_N]Memo = undefined,
    memo_len: usize = 0,
    memo_next: usize = 0,

    pub fn init(pg: *PoseGpu, a: *const Features, b: *const Features, set: Settings, ffd: *const Ffd) Pair {
        return .{ .pg = pg, .a = a, .b = b, .set = set, .ffd = ffd };
    }

    pub fn deinit(self: *Pair) void {
        const g = self.pg.g;
        g.release(&self.cps);
        g.release(&self.cps_adj);
        g.release(&self.cps_pose);
        g.release(&self.cps_pose_adj);
    }

    /// What a score depends on besides the pose: the score settings, both images' features and
    /// the live spline (its lattice and control points).
    fn ctx(self: *const Pair) u64 {
        var h = std.hash.Wyhash.init(0x5eed);
        h.update(std.mem.asBytes(&self.set));
        inline for (.{ self.a, self.b }) |f| h.update(std.mem.asBytes(&[_]u32{ f.feat.id, f.w, f.h }));
        const live = self.ffd.live();
        h.update(std.mem.asBytes(&[_]u32{ @intFromBool(live), self.ffd.g, @intFromBool(self.ffd.source) }));
        if (live) h.update(std.mem.sliceAsBytes(self.ffd.cps[0..self.ffd.n()]));
        return h.final();
    }

    fn remember(self: *Pair, c: u64, H: lie.Mat3, fwd: f64, inv: ?f64) void {
        self.memo[self.memo_next] = .{ .H = H, .ctx = c, .fwd = fwd, .inv = inv };
        self.memo_next = (self.memo_next + 1) % MEMO_N;
        self.memo_len = @min(self.memo_len + 1, MEMO_N);
    }

    fn recall(self: *const Pair, c: u64, H: lie.Mat3) ?Memo {
        for (self.memo[0..self.memo_len]) |m| if (m.ctx == c and std.mem.eql(f64, &m.H, &H)) return m;
        return null;
    }

    fn lw(self: *const Pair) f64 {
        return @floatFromInt(self.a.w);
    }
    fn lh(self: *const Pair) f64 {
        return @floatFromInt(self.a.h);
    }
    fn rw(self: *const Pair) f64 {
        return @floatFromInt(self.b.w);
    }
    fn rh(self: *const Pair) f64 {
        return @floatFromInt(self.b.h);
    }

    fn base(self: *const Pair) pose_mod.Cfg {
        return .{ .family = self.set.family, .exact = self.set.exact, .edge = self.set.edge };
    }

    /// Forward-direction config: the live spline (smi.js ffdLive gating).
    pub fn fwdCfg(self: *Pair) pose_mod.Cfg {
        var c = self.base();
        if (self.ffd.live()) {
            const g = self.pg.g;
            g.ensure(&self.cps, pose_mod.FFD_MAX * pose_mod.FFD_MAX * 8);
            g.writeSlice(self.cps, 0, f32, self.ffd.cps[0..self.ffd.n()]);
            c.ffd = .{ .gx = self.ffd.g, .gy = self.ffd.g, .source = self.ffd.source, .cps = self.cps };
        }
        return c;
    }

    /// Inverse-direction config (engine.withAdjoint): the spline negated in the other frame.
    pub fn invCfg(self: *Pair) pose_mod.Cfg {
        var c = self.base();
        if (self.ffd.live()) {
            const g = self.pg.g;
            const nc = self.ffd.n();
            g.ensure(&self.cps_adj, pose_mod.FFD_MAX * pose_mod.FFD_MAX * 8);
            var neg: [pose_mod.FFD_MAX * pose_mod.FFD_MAX * 2]f32 = undefined;
            for (0..nc) |i| neg[i] = -self.ffd.cps[i];
            g.writeSlice(self.cps_adj, 0, f32, neg[0..nc]);
            c.ffd = .{ .gx = self.ffd.g, .gy = self.ffd.g, .source = !self.ffd.source, .cps = self.cps_adj };
        }
        return c;
    }

    /// Forward-direction score only (smi.js overlapNs).
    pub fn fwdScore(self: *Pair, H: lie.Mat3) !f64 {
        var f: [1]f64 = undefined;
        try self.pg.scores(self.a, self.b, &.{H}, self.fwdCfg(), &f);
        return if (std.math.isFinite(f[0])) f[0] else 0;
    }

    /// Forward, inverse and mean score at H.
    pub fn parts(self: *Pair, H: lie.Mat3) !Parts {
        const c = self.ctx();
        if (self.recall(c, H)) |m| if (m.inv != null or !self.set.symmetric) {
            return .{ .fwd = m.fwd, .inv = m.inv, .mean = if (m.inv) |i| 0.5 * (m.fwd + i) else m.fwd };
        };
        var f: [1]f64 = undefined;
        if (!self.set.symmetric) {
            try self.pg.scores(self.a, self.b, &.{H}, self.fwdCfg(), &f);
            const fwd = if (std.math.isFinite(f[0])) f[0] else 0;
            self.remember(c, H, fwd, null);
            return .{ .fwd = fwd, .inv = null, .mean = fwd };
        }
        // both directions recorded together, one wait
        var r: [1]f64 = undefined;
        try self.pg.scores2(&.{
            .{ .a = self.a, .b = self.b, .Hs = &.{H}, .cfg = self.fwdCfg() },
            .{ .a = self.b, .b = self.a, .Hs = &.{lie.inv3(H)}, .cfg = self.invCfg() },
        }, &.{ &f, &r });
        const fwd = if (std.math.isFinite(f[0])) f[0] else 0;
        const inv = if (std.math.isFinite(r[0])) r[0] else 0;
        self.remember(c, H, fwd, inv);
        return .{ .fwd = fwd, .inv = inv, .mean = 0.5 * (fwd + inv) };
    }

    pub fn score(self: *Pair, H: lie.Mat3) !f64 {
        return (try self.parts(H)).mean;
    }

    /// Scores of many poses, one readback per direction. `cps_packed` (optional): one control
    /// point set per pose (2·g·g each), for the spline line search.
    pub fn scorePoses(self: *Pair, Hs: []const lie.Mat3, cps_packed: ?[]const f32, out: []f64) !void {
        const gpa = self.pg.g.gpa;
        const g = self.pg.g;
        var cfg = self.fwdCfg();
        if (cps_packed) |pk| {
            g.ensure(&self.cps_pose, @max(pk.len, 4) * 4);
            g.writeSlice(self.cps_pose, 0, f32, pk);
            cfg.ffd = .{ .gx = if (self.ffd.on) self.ffd.g else 0, .gy = if (self.ffd.on) self.ffd.g else 0, .source = self.ffd.source, .cps = self.cps_pose, .per_pose = true };
        }
        const c = self.ctx();
        if (!self.set.symmetric) {
            try self.pg.scores(self.a, self.b, Hs, cfg, out);
            if (cps_packed == null) for (Hs, 0..) |H, i| self.remember(c, H, if (std.math.isFinite(out[i])) out[i] else 0, null);
            return;
        }
        const Hi = try gpa.alloc(lie.Mat3, Hs.len);
        defer gpa.free(Hi);
        for (Hs, 0..) |H, i| Hi[i] = lie.inv3(H);
        const inv = try gpa.alloc(f64, Hs.len);
        defer gpa.free(inv);
        var icfg = if (cps_packed == null) self.invCfg() else cfg;
        if (cps_packed) |pk| {
            var live = false;
            for (pk) |v| if (@abs(v) > 1e-12) {
                live = true;
                break;
            };
            if (live) {
                const neg = try gpa.alloc(f32, pk.len);
                defer gpa.free(neg);
                for (pk, 0..) |v, i| neg[i] = -v;
                g.ensure(&self.cps_pose_adj, @max(pk.len, 4) * 4);
                g.writeSlice(self.cps_pose_adj, 0, f32, neg);
                icfg.ffd.source = !self.ffd.source;
                icfg.ffd.cps = self.cps_pose_adj;
            }
        }
        // both directions recorded together, one wait
        try self.pg.scores2(&.{
            .{ .a = self.a, .b = self.b, .Hs = Hs, .cfg = cfg },
            .{ .a = self.b, .b = self.a, .Hs = Hi, .cfg = icfg },
        }, &.{ out[0..Hs.len], inv });
        if (cps_packed == null) for (Hs, 0..) |H, i| {
            self.remember(c, H, if (std.math.isFinite(out[i])) out[i] else 0, if (std.math.isFinite(inv[i])) inv[i] else 0);
        };
        for (out[0..Hs.len], 0..) |*o, i| o.* = 0.5 * ((if (std.math.isFinite(o.*)) o.* else 0) + (if (std.math.isFinite(inv[i])) inv[i] else 0));
    }

    /// Symmetric gradient: g = ½ (g_fwd + Pᵀ g_inv), A = ½ (A_fwd + Pᵀ A_inv P).
    fn sym(self: *const Pair, fwd: pose_mod.Grad, back: pose_mod.Grad, H: lie.Mat3, group: lie.Group) pose_mod.Grad {
        const nk = lie.nKeys(group);
        const pulled = lie.adjointGrad(&back.grad, H, group, self.lw(), self.lh(), self.rw(), self.rh());
        var out = fwd;
        out.score = 0.5 * (fwd.score + back.score);
        for (0..nk) |i| out.grad[i] = 0.5 * (fwd.grad[i] + pulled[i]);
        if (fwd.hess != null and back.hess != null) {
            var PT: [64]f64 = @splat(0);
            for (0..nk) |j| {
                var e: [8]f64 = @splat(0);
                e[j] = 1;
                const col = lie.adjointGrad(&e, H, group, self.lw(), self.lh(), self.rw(), self.rh());
                for (0..nk) |i| PT[i * nk + j] = col[i];
            }
            var h: [64]f64 = @splat(0);
            const bh = back.hess.?;
            const fh = fwd.hess.?;
            for (0..nk) |i| for (0..nk) |j| {
                var s: f64 = 0;
                for (0..nk) |x| for (0..nk) |y| {
                    s += PT[i * nk + x] * bh[x * nk + y] * PT[j * nk + y];
                };
                h[i * nk + j] = 0.5 * (fh[i * nk + j] + s);
            };
            out.hess = h;
        } else out.hess = null;
        return out;
    }

    pub fn grad(self: *Pair, H: lie.Mat3, group: lie.Group, hess: bool) !pose_mod.Grad {
        var out: [1]pose_mod.Grad = undefined;
        try self.gradBatch(&.{H}, group, hess, &out);
        return out[0];
    }

    pub fn gradBatch(self: *Pair, Hs: []const lie.Mat3, group: lie.Group, hess: bool, out: []pose_mod.Grad) !void {
        return self.gradBoth(Hs, group, hess, self.fwdCfg(), if (self.set.symmetric) self.invCfg() else null, out, true);
    }

    /// Forward (and, symmetric, inverse) gradients recorded together and symmetrized.
    fn gradBoth(self: *Pair, Hs: []const lie.Mat3, group: lie.Group, hess: bool, fcfg: pose_mod.Cfg, icfg: ?pose_mod.Cfg, out: []pose_mod.Grad, memo: bool) !void {
        const c = self.ctx();
        const ic = icfg orelse {
            try self.pg.grads(self.a, self.b, Hs, group, fcfg, hess, out);
            if (memo) for (Hs, 0..) |H, i| self.remember(c, H, if (std.math.isFinite(out[i].score)) out[i].score else 0, null);
            return;
        };
        const gpa = self.pg.g.gpa;
        const Hi = try gpa.alloc(lie.Mat3, Hs.len);
        defer gpa.free(Hi);
        for (Hs, 0..) |H, i| Hi[i] = lie.inv3(H);
        const back = try gpa.alloc(pose_mod.Grad, Hs.len);
        defer gpa.free(back);
        try self.pg.grads2(&.{
            .{ .a = self.a, .b = self.b, .Hs = Hs, .cfg = fcfg },
            .{ .a = self.b, .b = self.a, .Hs = Hi, .cfg = ic },
        }, group, hess, &.{ out[0..Hs.len], back });
        if (memo) for (Hs, 0..) |H, i| {
            self.remember(c, H, if (std.math.isFinite(out[i].score)) out[i].score else 0, if (std.math.isFinite(back[i].score)) back[i].score else 0);
        };
        for (0..Hs.len) |i| out[i] = self.sym(out[i], back[i], Hs[i], group);
    }

    /// The GPU climb's configs (gclimb.zig): both directions with the live spline bound whenever
    /// the spline is on (its control points change on the device mid-batch: forward in `cps`,
    /// negated in `cps_adj`), and their per-pose versions bound to the line search's packs
    /// (`cps_pose`, `cps_pose_adj`, room for `packs` sets).
    pub fn climbCfgs(self: *Pair, packs: usize) struct { fwd: pose_mod.Cfg, inv: pose_mod.Cfg, fwd_pk: pose_mod.Cfg, inv_pk: pose_mod.Cfg } {
        var fwd = self.base();
        var inv = self.base();
        if (!self.ffd.on) return .{ .fwd = fwd, .inv = inv, .fwd_pk = fwd, .inv_pk = inv };
        const g = self.pg.g;
        const nc = self.ffd.n();
        g.ensure(&self.cps, pose_mod.FFD_MAX * pose_mod.FFD_MAX * 8);
        g.ensure(&self.cps_adj, pose_mod.FFD_MAX * pose_mod.FFD_MAX * 8);
        g.writeSlice(self.cps, 0, f32, self.ffd.cps[0..nc]);
        var neg: [pose_mod.FFD_MAX * pose_mod.FFD_MAX * 2]f32 = undefined;
        for (0..nc) |i| neg[i] = -self.ffd.cps[i];
        g.writeSlice(self.cps_adj, 0, f32, neg[0..nc]);
        g.ensure(&self.cps_pose, @max(packs * nc, 4) * 4);
        g.ensure(&self.cps_pose_adj, @max(packs * nc, 4) * 4);
        fwd.ffd = .{ .gx = self.ffd.g, .gy = self.ffd.g, .source = self.ffd.source, .cps = self.cps };
        inv.ffd = .{ .gx = self.ffd.g, .gy = self.ffd.g, .source = !self.ffd.source, .cps = self.cps_adj };
        var fwd_pk = fwd;
        fwd_pk.ffd.cps = self.cps_pose;
        fwd_pk.ffd.per_pose = true;
        var inv_pk = inv;
        inv_pk.ffd.cps = self.cps_pose_adj;
        inv_pk.ffd.per_pose = true;
        return .{ .fwd = fwd, .inv = inv, .fwd_pk = fwd_pk, .inv_pk = inv_pk };
    }

    /// Per-pose spline configs (the cloud search's packed control points, 2·g·g per pose): forward
    /// as given, inverse negated in the other frame (app withInverseCps).
    fn poseCfgs(self: *Pair, cps_packed: []const f32) !struct { fwd: pose_mod.Cfg, inv: pose_mod.Cfg } {
        const g = self.pg.g;
        g.ensure(&self.cps_pose, @max(cps_packed.len, 4) * 4);
        g.writeSlice(self.cps_pose, 0, f32, cps_packed);
        var fwd = self.base();
        fwd.ffd = .{ .gx = self.ffd.g, .gy = self.ffd.g, .source = self.ffd.source, .cps = self.cps_pose, .per_pose = true };
        const neg = try g.gpa.alloc(f32, cps_packed.len);
        defer g.gpa.free(neg);
        for (cps_packed, 0..) |v, i| neg[i] = -v;
        g.ensure(&self.cps_pose_adj, @max(cps_packed.len, 4) * 4);
        g.writeSlice(self.cps_pose_adj, 0, f32, neg);
        var inv = fwd;
        inv.ffd.source = !self.ffd.source;
        inv.ffd.cps = self.cps_pose_adj;
        return .{ .fwd = fwd, .inv = inv };
    }

    /// gradBatch with one control point set per pose (null: the engine's spline).
    pub fn gradBatchCps(self: *Pair, Hs: []const lie.Mat3, cps_packed: ?[]const f32, group: lie.Group, hess: bool, out: []pose_mod.Grad) !void {
        const pk = cps_packed orelse return self.gradBatch(Hs, group, hess, out);
        const c = try self.poseCfgs(pk);
        return self.gradBoth(Hs, group, hess, c.fwd, if (self.set.symmetric) c.inv else null, out, false);
    }

    /// Scores and control-point gradients of many poses, each with its own control points
    /// (symmetric: ½ (score_fwd + score_inv), ½ (g_fwd − g_inv)).
    pub fn ffdGradBatch(self: *Pair, Hs: []const lie.Mat3, cps_packed: []const f32, scores: []f64, grads: []f64) !void {
        const npar = self.ffd.n();
        const c = try self.poseCfgs(cps_packed);
        try self.pg.ffdGrads(self.a, self.b, Hs, c.fwd, scores, grads);
        if (!self.set.symmetric) return;
        const gpa = self.pg.g.gpa;
        const Hi = try gpa.alloc(lie.Mat3, Hs.len);
        defer gpa.free(Hi);
        for (Hs, 0..) |H, i| Hi[i] = lie.inv3(H);
        const isc = try gpa.alloc(f64, Hs.len);
        defer gpa.free(isc);
        const ig = try gpa.alloc(f64, Hs.len * npar);
        defer gpa.free(ig);
        try self.pg.ffdGrads(self.b, self.a, Hi, c.inv, isc, ig);
        const z = struct {
            fn f(v: f64) f64 {
                return if (std.math.isNan(v)) 0 else v;
            }
        }.f;
        for (0..Hs.len) |p| {
            for (0..npar) |k| grads[p * npar + k] = 0.5 * (z(grads[p * npar + k]) - z(ig[p * npar + k]));
            scores[p] = 0.5 * (z(scores[p]) + z(isc[p]));
        }
    }

    /// Score and ∇ over the spline's control points (symmetric: ½ (g_fwd − g_inv)).
    pub fn ffdGrad(self: *Pair, H: lie.Mat3, grad_out: []f64) !f64 {
        const npar = self.ffd.n();
        if (!self.ffd.on) {
            @memset(grad_out[0..npar], 0);
            return 0;
        }
        const g = self.pg.g;
        g.ensure(&self.cps, pose_mod.FFD_MAX * pose_mod.FFD_MAX * 8);
        g.writeSlice(self.cps, 0, f32, self.ffd.cps[0..npar]);
        var cfg = self.base();
        cfg.ffd = .{ .gx = self.ffd.g, .gy = self.ffd.g, .source = self.ffd.source, .cps = self.cps };
        var sc: [1]f64 = undefined;
        try self.pg.ffdGrads(self.a, self.b, &.{H}, cfg, &sc, grad_out);
        if (!self.set.symmetric or !self.ffd.live()) return sc[0];
        const gpa = g.gpa;
        const ig = try gpa.alloc(f64, npar);
        defer gpa.free(ig);
        var icfg = self.invCfg();
        var isc: [1]f64 = undefined;
        try self.pg.ffdGrads(self.b, self.a, &.{lie.inv3(H)}, icfg, &isc, ig);
        icfg = undefined;
        for (0..npar) |i| grad_out[i] = 0.5 * (grad_out[i] - ig[i]);
        return 0.5 * (sc[0] + isc[0]);
    }
};
