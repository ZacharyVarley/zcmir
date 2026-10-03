//! Refine on the GPU: the pose climb with every decision made on the device. Each step is a
//! stage (climb.wgsl): candidate poses and the current one scored in both directions, the best
//! that beats the current score kept, and the gradient with the Gauss–Newton matrix there; an
//! iteration is [shift hop] → up to three Gauss–Newton steps [→ roto-scale hop → steps] [→ spline
//! line search]; the copula scores (no Gauss–Newton matrix) take gradient line searches as their
//! steps. Iterations are recorded BATCH at
//! a time into one submission, gated on the state's done flag (Gpu.beginGate), so iterations
//! after convergence launch no work; the host reads the state once per batch to report the
//! steps taken (the trail and the pose) and to honour Stop.
//!
//! Decisions are f32 (an experiment moved final poses by thousandths of a pixel).
const std = @import("std");
const lie = @import("lie.zig");
const shaders = @import("shaders");
const gpu_mod = @import("gpu/gpu.zig");
const pose_mod = @import("pose.zig");
const engine_mod = @import("engine.zig");
const climb_mod = @import("climb.zig");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Bind = gpu_mod.Bind;
const Engine = engine_mod.Engine;
const Mat3 = lie.Mat3;

// the state buffer (climb.wgsl)
const S_DONE = 0;
const S_CUR = 1;
const S_STALL = 3;
const S_ITER = 4;
const S_GNSTOP = 5;
const S_NREC = 6;
const S_FWD = 7;
const S_INV = 8;
const S_START = 9;
const S_H = 16;
const S_HI = 25;
const S_REC = 128;
const REC = 16;
const MAX_REC = 256;
const S_AMPS = 4224;
const S_SAMPS = 4240;
const S_LAM = 4256;
const S_FD = 4272;
const STATE_FLOATS = S_FD + pose_mod.FFD_MAX * pose_mod.FFD_MAX * 2;
/// the spline step's lengths along its Gauss–Newton direction (climb.wgsl FFD_MUS)
const FFD_MUS = 6;

/// iterations recorded per submission (the trail and Stop act between batches)
pub const BATCH = 8;
const GN_STEPS = 3;
const N_GN = 5;

const KIND_HOP = 1;
const KIND_GN = 2;
const KIND_FFD = 3;
const KIND_RS = 4;
const KIND_LS = 5;

/// Uniform block U of climb.wgsl.
const CU = extern struct {
    n: u32 = 0,
    nk: u32 = 0,
    homog: u32 = 0,
    sym: u32 = 0,
    hop: u32 = 0,
    kind: u32 = 0,
    n_iters: u32 = 0,
    np: u32 = 0,
    stride: u32 = 0,
    mx: u32 = 0,
    my: u32 = 0,
    cw: u32 = 0,
    ch: u32 = 0,
    fw: u32 = 0,
    fh: u32 = 0,
    iter0: u32 = 0,
    rw: f32 = 0,
    rh: f32 = 0,
    lw: f32 = 0,
    lh: f32 = 0,
    ox: f32 = 0,
    oy: f32 = 0,
    p0: f32 = 0,
    p1: f32 = 0,
    /// climb_combine: the inverse lane's tangent rows per pose (np: the forward lane's)
    np1: u32 = 0,
    /// the flag word gating the kernel besides the done flag (k1 fills it)
    gate: u32 = gpu_mod.NO_FLAG,
    _pad: [2]u32 = .{ 0, 0 },
};
comptime {
    std.debug.assert(@sizeOf(CU) == 112);
}


const Run = struct {
    e: *Engine,
    g: *Gpu,
    base: CU,
    dirs: [2]pose_mod.Dir,
    /// the spline line search's directions (per-pose control points)
    dirs_pk: [2]pose_mod.Dir,
    n_dirs: usize,
    n_amps: u32 = 0,
    n_samps: u32 = 0,
    npar: u32 = 0,
    /// the spline step's buffers: the assembled gradient and matrix, the factorization's work,
    /// the bending energy R, each candidate's penalty
    gn: Buf = .{},
    work: Buf = .{},
    rmat: Buf = .{},
    pen: Buf = .{},
    state: Buf = .{},
    pick: Buf = .{},
    hc: Buf = .{},

    fn pipe(self: *Run, entry: []const u8) u32 {
        var kb: [48]u8 = undefined;
        const key = std.fmt.bufPrint(&kb, "climb/{s}", .{entry}) catch entry;
        return self.g.pipeline(key, shaders.climb, entry) catch |err| {
            std.log.err("climb.wgsl {s}: {s}", .{ entry, @errorName(err) });
            return gpu_mod.NO_PIPE;
        };
    }

    /// A one-workgroup climb kernel; `u` null for kernels that read no uniforms. The kernels
    /// gate themselves (climb.wgsl gateOff: the done flag, and u.gate): a direct dispatch costs
    /// a fraction of an indirect one in the browser (Dawn validates each indirect dispatch's
    /// counts with a dispatch of its own).
    fn k1(self: *Run, entry: []const u8, u: ?CU, binds: []const Bind) void {
        var all: [12]Bind = undefined;
        var n: usize = 0;
        if (u) |uu| {
            var ug = uu;
            ug.gate = self.g.gateWord();
            all[0] = self.g.uniform(0, std.mem.asBytes(&ug));
            n = 1;
        }
        @memcpy(all[n .. n + binds.len], binds);
        self.g.dispatchUngated(self.pipe(entry), 1, 1, 1, all[0 .. n + binds.len]);
    }

    fn lanes(self: *Run) *[2]pose_mod.Lane {
        return &self.e.pr.pg.lanes;
    }

    /// One stage over the n candidates (plus the current pose) a generator kernel just wrote.
    /// A Gauss–Newton stage's gradient is gated on its own pick: when no step beat the score,
    /// the pose (and so its gradient) is unchanged.
    fn stage(self: *Run, n: u32, kind: u32) void {
        self.stageOn(n, kind, self.dirs[0..self.n_dirs], true);
    }

    fn stageOn(self: *Run, n: u32, kind: u32, dirs: []const pose_mod.Dir, with_grad: bool) void {
        const pg = self.e.pr.pg;
        pg.stageScores(dirs, n + 1);
        const L = self.lanes();
        var u = self.base;
        u.n = n;
        u.kind = kind;
        self.k1("climb_select", u, &.{
            self.state.at(1), L[0].hs.at(2), L[1].hs.at(3), L[0].sc.at(4), (if (self.n_dirs == 2) L[1].sc else L[0].sc).at(5), self.pick.at(8), self.pen.at(9),
        });
        if (!with_grad) return;
        if (kind == KIND_GN) self.g.gateRefresh();
        const tnp = pg.stageGrad(dirs, self.pick, self.e.set.group);
        var uc = self.base;
        uc.np = tnp[0];
        uc.np1 = tnp[1];
        self.k1("climb_combine", uc, &.{ self.state.at(1), L[0].tpart.at(6), L[if (self.n_dirs == 2) 1 else 0].tpart.at(7) });
    }

    /// The local steps: up to three Gauss–Newton steps, those after one that beat nothing
    /// launching no work (E4, without a Gauss–Newton matrix: a gradient line search, then a
    /// finer one, both always).
    fn gnBlock(self: *Run) void {
        if (gaussNewton(self.e.set.family())) {
            self.g.gateOn(S_GNSTOP);
            for (0..GN_STEPS) |_| {
                self.k1("gn_cands", self.base, &self.genBinds());
                self.stage(N_GN, KIND_GN);
            }
        } else {
            for ([_]u32{ self.n_amps, self.n_samps }, 0..) |n, ladder| {
                var u = self.base;
                u.n = n;
                u.kind = @intCast(ladder);
                self.k1("ls_cands", u, &self.genBinds());
                self.stage(n, KIND_LS);
            }
        }
        self.g.gateOn(null);
    }

    /// The candidate generators write both lanes' poses.
    fn genBinds(self: *Run) [3]Bind {
        const L = self.lanes();
        return .{ self.state.at(1), L[0].hs.at(2), L[1].hs.at(3) };
    }

    /// The spline stage: its Gauss–Newton terms per lattice cell in both directions, assembled
    /// into the gradient and the banded matrix, the step (with the bending energy), candidates
    /// along it picked by score less bending penalty, then, as the spline may have moved, the
    /// gradient at the current pose again.
    fn splineStage(self: *Run) void {
        const pg = self.e.pr.pg;
        const L = self.lanes();
        const pr = &self.e.pr;
        const gx = self.e.ffd.g;
        for (0..self.n_dirs) |k| pg.stageFfdCells(k, self.dirs[k]);
        var uf = self.base;
        uf.mx = gx;
        uf.my = gx;
        uf.np = self.npar;
        uf.n = FFD_MUS;
        // λ = c (s L / 2π)⁴ with c the score's curvature per pixel per squared pixel of
        // displacement, L = √(lattice area) and s the stiffness: a wave of wavelength s L then
        // costs as much bending energy as it can gain, shorter ones more. c = tr A / (Σ_k b_k² ·
        // area · hx hy): tr A sums each pixel's curvature times its basis weights' squares (mean
        // (151/315)² for the cubic B-spline) in parameter units (hx, hy pixels each). The shader
        // multiplies this by tr A at the first spline step, so s means the same on any grid.
        const lat = if (self.e.ffd.source) self.e.feats[0] else self.e.feats[1];
        const lw: f64 = @floatFromInt(lat.w);
        const lh: f64 = @floatFromInt(lat.h);
        const area = @max(lw - 1, 1) * @max(lh - 1, 1);
        const bsq = (151.0 / 315.0) * (151.0 / 315.0);
        const s = self.e.set.ffd_stiffness;
        uf.p1 = @floatCast(s * s * s * s * area / (bsq * @max(lw, 1) * 0.5 * @max(lh, 1) * 0.5 * std.math.pow(f64, 2 * std.math.pi, 4)));
        // the step cap: no control point moves more than 0.2 lattice spacings (parameter units:
        // half the lattice frame per unit)
        uf.p0 = 0.2 * 2.0 / @as(f32, @floatFromInt(@max(gx, 2) - 1));
        const ncp = gx * gx;
        self.g.dispatch(self.pipe("ffd_assemble"), (ncp * ncp + 63) / 64, 1, 1, &.{
            self.g.uniform(0, std.mem.asBytes(&uf)), L[0].ffdp.at(6), L[if (self.n_dirs == 2) 1 else 0].ffdp.at(7), self.gn.at(10),
        });
        self.k1("ffd_solve", uf, &.{ self.state.at(1), pr.cps.at(6), self.gn.at(9), self.work.at(10), self.rmat.at(12) });
        self.k1("ffd_gn_cands", uf, &.{
            self.state.at(1), L[0].hs.at(2), L[1].hs.at(3), pr.cps.at(9), pr.cps_pose.at(10), pr.cps_pose_adj.at(11), self.rmat.at(12), self.pen.at(13),
        });
        self.stageOn(FFD_MUS, KIND_FFD, self.dirs_pk[0..self.n_dirs], false);
        // (ffd_apply reads no state: gated by the dispatch)
        self.g.dispatch(self.pipe("ffd_apply"), 1, 1, 1, &.{ self.g.uniform(0, std.mem.asBytes(&uf)), self.pick.at(8), pr.cps_pose.at(9), pr.cps.at(10), pr.cps_adj.at(11) });
        self.k1("cands_current", null, &self.genBinds());
        self.stage(0, 0);
    }

    fn release(self: *Run) void {
        inline for (.{ &self.state, &self.pick, &self.hc, &self.gn, &self.work, &self.rmat, &self.pen }) |b| self.g.release(b);
    }
};

/// The spline's bending energy as a quadratic form cᵀ R c over its parameters (x, y per control
/// point, in the half-frame units the spline stores): the thin-plate energy Σ f_xx² + 2 f_xy² +
/// f_yy² of the displacement in pixels, from second differences on the control lattice (the
/// lattice spacing's powers included, so an anisotropic lattice is weighed in pixels too).
fn bendingEnergy(gpa: std.mem.Allocator, g: u32, source: bool, mov: anytype, fix: anytype) ![]f32 {
    const n: usize = 2 * @as(usize, g) * g;
    const R = try gpa.alloc(f64, n * n);
    defer gpa.free(R);
    @memset(R, 0);
    const lw: f64 = @floatFromInt(if (source) mov.w else fix.w);
    const lh: f64 = @floatFromInt(if (source) mov.h else fix.h);
    const hx = [2]f64{ @max(lw, 1) * 0.5, @max(lh, 1) * 0.5 };
    const gf: f64 = @floatFromInt(@max(g, 2) - 1);
    const sp = [2]f64{ @max(lw - 1, 1) / gf, @max(lh - 1, 1) / gf };
    const area = sp[0] * sp[1];
    const add = struct {
        fn f(Rm: []f64, nn: usize, taps: []const [3]i64, gg: u32, comp: usize, w: f64) void {
            for (taps) |a| for (taps) |b| {
                const ia: usize = @intCast(a[1] * gg + a[0]);
                const ib: usize = @intCast(b[1] * gg + b[0]);
                Rm[(2 * ia + comp) * nn + 2 * ib + comp] += w * @as(f64, @floatFromInt(a[2] * b[2]));
            };
        }
    }.f;
    const gi: i64 = g;
    for (0..2) |comp| {
        const s2 = hx[comp] * hx[comp];
        var j: i64 = 0;
        while (j < gi) : (j += 1) {
            var i: i64 = 0;
            while (i < gi) : (i += 1) {
                if (i >= 1 and i + 1 < gi) add(R, n, &.{ .{ i - 1, j, 1 }, .{ i, j, -2 }, .{ i + 1, j, 1 } }, g, comp, s2 * area / (sp[0] * sp[0] * sp[0] * sp[0]));
                if (j >= 1 and j + 1 < gi) add(R, n, &.{ .{ i, j - 1, 1 }, .{ i, j, -2 }, .{ i, j + 1, 1 } }, g, comp, s2 * area / (sp[1] * sp[1] * sp[1] * sp[1]));
                if (i + 1 < gi and j + 1 < gi) add(R, n, &.{ .{ i, j, 1 }, .{ i + 1, j, -1 }, .{ i, j + 1, -1 }, .{ i + 1, j + 1, 1 } }, g, comp, 2 * s2 * area / (sp[0] * sp[0] * sp[1] * sp[1]));
            }
        }
    }
    const out = try gpa.alloc(f32, n * n);
    for (R, 0..) |v, k| out[k] = @floatCast(v);
    return out;
}

/// Whether the score's steps are Gauss–Newton: SMI, and λmax through its top canonical pair
/// (copula.wgsl copula_lmax_q); E4 takes gradient line searches.
fn gaussNewton(f: @import("moments.zig").Family) bool {
    return f != .e4;
}

fn mat32(H: Mat3) [9]f32 {
    var o: [9]f32 = undefined;
    for (0..9) |i| o[i] = @floatCast(H[i]);
    return o;
}

fn mat64(s: []const f32) Mat3 {
    var o: Mat3 = undefined;
    for (0..9) |i| o[i] = s[i];
    return o;
}

/// Climb from the engine's current pose on the GPU; updates e.H to the best pose found.
pub fn run(e: *Engine) !climb_mod.Result {
    const g = &e.g;
    const set = e.set;
    const ev = &e.events;
    const group = set.group;
    const sym = set.symmetric;
    const n_iters = @max(1, set.g_iters);
    const pr = &e.pr;
    var r: Run = .{
        .e = e,
        .g = g,
        .base = .{
            .nk = @intCast(lie.nKeys(group)), .homog = @intFromBool(group == .homography), .sym = @intFromBool(sym),
            .hop = @intFromBool(set.hop or set.hop_fm), .n_iters = n_iters,
            .rw = @floatFromInt(e.feats[1].w), .rh = @floatFromInt(e.feats[1].h),
            .lw = @floatFromInt(e.feats[0].w), .lh = @floatFromInt(e.feats[0].h),
        },
        .dirs = undefined,
        .dirs_pk = undefined,
        .n_dirs = if (sym) 2 else 1,
    };
    // the step lengths of the spline line search (climb.zig: the gradient ladder)
    var abuf: [12]f64 = undefined;
    const amps = climb_mod.geoAmps(set.g_a0, set.g_decay, set.g_steps, &abuf);
    r.n_amps = @intCast(amps.len);
    // the second line search's ladder: one more, finer step (climb.zig s_amps)
    var sbuf: [13]f64 = undefined;
    @memcpy(sbuf[0..amps.len], amps);
    sbuf[amps.len] = amps[amps.len - 1] * (if (set.g_decay > 0) set.g_decay else 0.5);
    const samps = sbuf[0 .. amps.len + 1];
    r.n_samps = @intCast(samps.len);
    r.npar = @intCast(e.ffd.n());
    const cf = pr.climbCfgs(FFD_MUS + 1);
    r.dirs[0] = .{ .a = pr.a, .b = pr.b, .Hs = &.{}, .cfg = cf.fwd };
    r.dirs[1] = .{ .a = pr.b, .b = pr.a, .Hs = &.{}, .cfg = cf.inv };
    r.dirs_pk[0] = .{ .a = pr.a, .b = pr.b, .Hs = &.{}, .cfg = cf.fwd_pk };
    r.dirs_pk[1] = .{ .a = pr.b, .b = pr.a, .Hs = &.{}, .cfg = cf.inv_pk };
    const spline = e.ffd.on;
    defer r.release();
    r.state = g.storage(STATE_FLOATS * 4);
    r.pick = g.storage(16);
    r.hc = g.storage(16 * 4);
    r.pen = g.storage(16 * 4);
    if (spline) {
        const np: u64 = r.npar;
        r.gn = g.storage((np + np * np) * 4);
        r.work = g.storage(np * np * 4);
        r.rmat = g.storage(np * np * 4);
        const R = try bendingEnergy(e.gpa, e.ffd.g, e.ffd.source, e.feats[0], e.feats[1]);
        defer e.gpa.free(R);
        g.writeSlice(r.rmat, 0, f32, R);
    }
    for (r.lanes()) |*l| g.ensure(&l.hs, 8 * 9 * 4);
    // the state: the start pose
    var host: [STATE_FLOATS]f32 = @splat(0);
    @memcpy(host[S_H .. S_H + 9], &mat32(e.H));
    @memcpy(host[S_HI .. S_HI + 9], &mat32(lie.inv3(e.H)));
    for (amps, 0..) |a, i| host[S_AMPS + i] = @floatCast(a);
    for (samps, 0..) |a, i| host[S_SAMPS + i] = @floatCast(a);
    host[S_LAM] = -1; // λ: set at the first spline step
    g.writeSlice(r.state, 0, f32, &host);
    // the first gradient (and the start score)
    r.k1("cands_current", null, &r.genBinds());
    r.stage(0, 0);
    e.cancel = false;
    var n_rec: usize = 0;
    var iters: u32 = 0;
    var first = true;
    var H = e.H;
    var s_start: f64 = 0;
    while (true) {
        // the maps' canvas (and the roto-scale centre) for this batch, from the pose it starts at
        const fr = lie.regFrame(H, e.feats[0].w, e.feats[0].h, e.feats[1].w, e.feats[1].h);
        const rs_c = e.rsCenter(H);
        const mirror = H[0] * H[4] - H[1] * H[3] > 0;
        g.beginGate(r.state, S_DONE);
        for (0..BATCH) |_| {
            g.gateRefresh();
            r.k1("iter_begin", null, &.{r.state.at(1)});
            // (the flags iter_begin reset gate the stages below)
            g.gateRefresh();
            var uc = r.base;
            uc.ox = @floatFromInt(fr.ox);
            uc.oy = @floatFromInt(fr.oy);
            if (set.hop) {
                r.k1("hc_canvas", uc, &.{ r.state.at(1), r.hc.at(10) });
                // In the browser the map's passes run ungated: an indirect dispatch costs Dawn
                // ~25 µs more than a direct one, more than the map itself once (in the batch's
                // iterations after the climb stopped) hop_cand ignores it.
                const paused = if (gpu_mod.is_web) g.pauseGate() else null;
                const p = try e.smi.shiftMapAt(&e.feats[0], &e.feats[1], fr, r.hc, e.mapOpts());
                if (gpu_mod.is_web) g.resumeGate(paused);
                var uh = r.base;
                uh.mx = p.n;
                uh.my = p.ny;
                uh.cw = p.cw;
                uh.ch = p.ch;
                uh.fw = p.frame.w;
                uh.fh = p.frame.h;
                r.k1("hop_cand", uh, &(r.genBinds() ++ [_]Bind{e.smi.res.at(9)}));
                r.stage(1, KIND_HOP);
            }
            if (set.hop or !set.hop_fm) r.gnBlock();
            if (set.hop_fm) {
                if (set.hop) {
                    r.k1("gn_reset", null, &.{r.state.at(1)});
                    g.gateRefresh();
                }
                r.k1("hc_canvas", uc, &.{ r.state.at(1), r.hc.at(10) });
                const paused = if (gpu_mod.is_web) g.pauseGate() else null;
                const p = try e.smi.rsMapAt(&e.feats[0], &e.feats[1], fr, r.hc, rs_c, e.mapOpts(), &e.smi.ns);
                e.smi.rsFinish(e.smi.ns, null, mirror, p, set.rs_smin, set.rs_smax);
                if (gpu_mod.is_web) g.resumeGate(paused);
                var ur = r.base;
                ur.mx = p.n;
                ur.my = p.n_th;
                ur.p0 = @floatCast(p.dlam);
                ur.ox = @floatCast(rs_c[0]);
                ur.oy = @floatCast(rs_c[1]);
                r.k1("rs_cand", ur, &(r.genBinds() ++ [_]Bind{e.smi.res.at(9)}));
                r.stage(1, KIND_RS);
                r.gnBlock();
            }
            if (spline) r.splineStage();
            r.k1("iter_end", r.base, &.{r.state.at(1)});
        }
        g.endGate();
        g.read(r.state, 0, std.mem.sliceAsBytes(&host));
        // the live spline (the line search moves it on the device)
        if (spline) g.read(pr.cps, 0, std.mem.sliceAsBytes(e.ffd.cps[0..r.npar]));
        try g.wait();
        const cps: ?[]const f32 = if (spline) e.ffd.cps[0..r.npar] else null;
        if (first) {
            first = false;
            ev.log("ascent {s} score {d:.4} {s}{s}{s}{s}{s} (GPU)", .{
                @tagName(group), host[S_START], if (sym) "mean " else "", if (set.hop) "shift hop+" else "", if (set.hop_fm) "roto-scale hop+" else "",
                if (gaussNewton(set.family())) "Gauss–Newton" else "line search", if (spline) "+spline" else "",
            });
            if (!gaussNewton(set.family())) ev.log("line search α₀={d}", .{set.g_a0});
            s_start = host[S_START];
        }
        // the steps taken: trail entries, then the pose
        const nr = @min(MAX_REC, @as(usize, @intFromFloat(host[S_NREC])));
        while (n_rec < nr) : (n_rec += 1) {
            const o = S_REC + n_rec * REC;
            const kind: u32 = @intFromFloat(host[o]);
            var lb: [48]u8 = undefined;
            const label = switch (kind) {
                KIND_HOP => "shift hop",
                KIND_GN => std.fmt.bufPrint(&lb, "GN ×{d}", .{host[o + 5]}) catch "GN",
                KIND_FFD => "spline",
                KIND_RS => "roto-scale hop",
                KIND_LS => std.fmt.bufPrint(&lb, "grad α={d}", .{host[o + 5]}) catch "grad",
                else => "step",
            };
            const Hr = mat64(host[o + 7 .. o + 16]);
            // (each step's own control points are not kept: a spline trail entry shows the latest)
            ev.trail(label, host[o + 2], host[o + 3], if (sym) host[o + 4] else null, Hr, cps);
        }
        H = mat64(host[S_H .. S_H + 9]);
        e.H = H;
        ev.pose(H, cps);
        iters = @intFromFloat(host[S_ITER]);
        ev.log("ascent {d}/{d} score {d:.4}  stall {d}  ({d} steps so far)", .{ iters, n_iters, host[S_CUR], host[S_STALL], n_rec });
        if (host[S_DONE] > 0.5) break;
        if (e.cancel) {
            ev.log("ascent stopped  score {d:.4}", .{host[S_CUR]});
            break;
        }
    }
    e.H = H;
    ev.log("ascent done  score {d:.4}", .{host[S_CUR]});
    _ = S_FWD;
    _ = S_INV;
    return .{ .H = H, .score = host[S_CUR], .iterations = iters, .moved = @intFromBool(n_rec > 0 and host[S_CUR] > s_start) };
}
