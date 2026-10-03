//! Global search (port of sweep.js and the app's runSweep / runHho):
//!   sweep  a sweep of the dense pose score — every (θ, σ), optionally × shear × stretch, on a grid
//!          transforms the moving image about its centre (after a prior pose H0), each candidate's
//!          translation map on a coarse G×G canvas with the app's score (FFTs sized to the
//!          candidate's footprint), all batched on the GPU and reduced to a few peaks each — then
//!          NMS across (θ, σ, shear, stretch, shift), a short batched Gauss–Newton screen and full
//!          climbs of the finalists (converged poses leave the batch). The spline is off during
//!          the search.
//!   cloud  a seed cloud around the pose (or the identity) in the Lie algebra, every seed taking
//!          gradient line-search steps at once until it stalls ("Harris-hawks"-style swarm of the
//!          app, HHO), optionally with per-seed B-spline control points.
//! The current pose is kept unless the search scores higher.
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const shaders = @import("shaders");
const lie = @import("lie.zig");
const fft = @import("fft.zig");
const smi_mod = @import("smi.zig");
const pose_mod = @import("pose.zig");
const pair_mod = @import("pair.zig");
const climb_mod = @import("climb.zig");
const engine = @import("engine.zig");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Bind = gpu_mod.Bind;
const Mat3 = lie.Mat3;
const Engine = engine.Engine;

pub const Result = extern struct {
    H: [9]f64 = lie.identity,
    score: f64 = 0,
    /// the pose score before the search
    prev: f64 = 0,
    /// the search found a better pose (now the engine's)
    improved: u32 = 0,
    /// sweep: peaks after NMS, finalists; cloud: score evaluations, seeds
    count_a: u32 = 0,
    count_b: u32 = 0,
    _pad: u32 = 0,
};

// ── the candidate grid: rotation θ × scale σ × shear k × stretch a ──
pub const Grid = struct {
    thetas: []f64,
    sigmas: []f64,
    /// shear k and stretch a (a single 0 / 1 when not swept)
    shears: []f64,
    anisos: []f64,
    wrap: bool,

    pub fn deinit(self: *Grid, gpa: std.mem.Allocator) void {
        gpa.free(self.thetas);
        gpa.free(self.sigmas);
        gpa.free(self.shears);
        gpa.free(self.anisos);
    }

    pub fn count(self: *const Grid) usize {
        return self.thetas.len * self.sigmas.len * self.shears.len * self.anisos.len;
    }
};

pub const GridOpts = struct {
    /// rotation range in degrees; a span of 360° or more covers the circle once (and wraps in NMS)
    rot_lo: f64 = -180,
    rot_hi: f64 = 180,
    n_theta: u32 = 90,
    s_min: f64 = 0.5,
    s_max: f64 = 2,
    n_sigma: u32 = 20,
    shear_max: f64 = 0,
    n_shear: u32 = 1,
    aniso_max: f64 = 0,
    n_aniso: u32 = 1,
};

fn linspace(gpa: std.mem.Allocator, lo: f64, hi: f64, n: u32) ![]f64 {
    const out = try gpa.alloc(f64, @max(1, n));
    for (out, 0..) |*v, i| {
        const fi: f64 = @floatFromInt(i);
        v.* = if (out.len == 1) 0.5 * (lo + hi) else lo + (fi * (hi - lo)) / @as(f64, @floatFromInt(out.len - 1));
    }
    return out;
}

/// The full circle is sampled without repeating ±180°; a partial range includes both ends.
/// Scales and stretches are log-spaced (stretch symmetric about 1), shears linear about 0.
pub fn sweepGrid(gpa: std.mem.Allocator, o: GridOpts) !Grid {
    const nT = @max(1, o.n_theta);
    const lo_d = @min(o.rot_lo, o.rot_hi);
    const span_d = @abs(o.rot_hi - o.rot_lo);
    const wrap = span_d >= 360 - 1e-9;
    const rad = std.math.pi / 180.0;
    const th = try gpa.alloc(f64, nT);
    errdefer gpa.free(th);
    for (0..nT) |i| {
        const fi: f64 = @floatFromInt(i);
        const fn_: f64 = @floatFromInt(nT);
        const d = if (wrap) -180 + 360 * fi / fn_ else if (nT == 1) lo_d + span_d / 2 else lo_d + span_d * fi / (fn_ - 1);
        th[i] = d * rad;
    }
    const llo = @log(@max(1e-3, @min(o.s_min, o.s_max)));
    const lhi = @log(@max(1e-3, @max(o.s_min, o.s_max)));
    const sg = try linspace(gpa, llo, lhi, o.n_sigma);
    errdefer gpa.free(sg);
    for (sg) |*v| v.* = @exp(v.*);
    const kmax = @abs(o.shear_max);
    const sh = try linspace(gpa, -kmax, kmax, if (kmax > 0) o.n_shear else 1);
    errdefer gpa.free(sh);
    const amax = @log(1 + @abs(o.aniso_max));
    const an = try linspace(gpa, -amax, amax, if (amax > 0) o.n_aniso else 1);
    for (an) |*v| v.* = @exp(v.*);
    return .{ .thetas = th, .sigmas = sg, .shears = sh, .anisos = an, .wrap = wrap };
}

/// The linear part σ R(θ) diag(a, 1/a) [1 k; 0 1] (row-major 2×2): stretch and shear act along the
/// moving image's own axes (after the prior pose), then rotation and scale.
pub fn candLinear(th: f64, s: f64, k: f64, a: f64) [4]f64 {
    const c = s * @cos(th);
    const sn = s * @sin(th);
    // diag(a, 1/a) · [1 k; 0 1] = [a, a k; 0, 1/a]
    const p00 = a;
    const p01 = a * k;
    const p11 = 1 / a;
    return .{ c * p00, c * p01 - sn * p11, sn * p00, sn * p01 + c * p11 };
}

/// Dest-pixel map about c: X ↦ c + L (X − c).
fn aboutC(cc: [2]f64, L: [4]f64) Mat3 {
    return .{ L[0], L[1], cc[0] - L[0] * cc[0] - L[1] * cc[1], L[2], L[3], cc[1] - L[2] * cc[0] - L[3] * cc[1], 0, 0, 1 };
}

/// Singular values of a 2×2 (largest, smallest).
fn sv2(L: [4]f64) [2]f64 {
    const a = L[0] * L[0] + L[2] * L[2];
    const b = L[0] * L[1] + L[2] * L[3];
    const d = L[1] * L[1] + L[3] * L[3];
    const t = 0.5 * (a + d);
    const r = @sqrt(@max(0, 0.25 * (a - d) * (a - d) + b * b));
    return .{ @sqrt(t + r), @sqrt(@max(0, t - r)) };
}

pub const Peak = struct { score: f64, cand: u32, it: u32, is: u32, ik: u32 = 0, ia: u32 = 0, th: f64, s: f64, sx: i32, sy: i32, H: Mat3 };

/// Uniform block U of sweep.wgsl.
const SU = extern struct {
    /// plane width (FFT row length) and the packed window's width
    nx: u32 = 0,
    win_x: u32 = 0,
    n_real: u32 = 0,
    n_cplx: u32 = 0,
    n_out: u32 = 0,
    src_w: u32 = 0,
    src_h: u32 = 0,
    min_n: u32 = smi_mod.MIN_N,
    ox: f32 = 0,
    oy: f32 = 0,
    cell: f32 = 1,
    ridge: f32 = smi_mod.RIDGE,
    mode: u32 = 0,
    cand0: u32 = 0,
    n_peak: u32 = 0,
    n_corr: u32 = 0,
    ny: u32 = 0,
    win_y: u32 = 0,
    /// the fixed footprint's first canvas cell (the fixed planes' origin)
    fx0: u32 = 0,
    fy0: u32 = 0,
    /// canvas cells per side
    grid: u32 = 0,
    _p0: u32 = 0,
    _p1: u32 = 0,
    _p2: u32 = 0,
};
comptime {
    std.debug.assert(@sizeOf(SU) == 96);
}

/// FFT lengths the sweep's planes take (all factor into radices ≤ 13 and divide by 16, so column
/// FFTs batch 4 lines per workgroup); few lengths keep the number of compiled pipelines small.
const LADDER = [_]u32{ 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512 };

fn ladderSize(need: u32) u32 {
    for (LADDER) |L| if (L >= need) return L;
    return fft.planSize(need, 16);
}

pub const SweepOpts = struct {
    H0: Mat3,
    grid_v: *const Grid,
    grid: u32 = 64,
    /// combine mode: 0 / 1 global / exact SMI, 2 E4, 3 λmax
    mode: u32 = 1,
    n_peak: u32 = 4,
    batch: u32 = 64,
    min_overlap: f64 = 0.05,
};

pub const SweepOut = struct {
    peaks: std.ArrayList(Peak) = .empty,
    /// each candidate's map maximum (θ-major, then σ, shear, stretch)
    best: []f32 = &.{},
    n: u32 = 0,
    cell: f64 = 0,
    nc: u32 = 0,

    pub fn deinit(self: *SweepOut, gpa: std.mem.Allocator) void {
        self.peaks.deinit(gpa);
        gpa.free(self.best);
    }
};

pub const Sweep = struct {
    g: *Gpu,
    smi: *smi_mod.Smi,
    il: [2]Buf = .{ .{}, .{} },
    img: [2][2]u32 = .{ .{ 0, 0 }, .{ 0, 0 } },
    poses: Buf = .{},
    pose_fixed: Buf = .{},
    spec_b: Buf = .{},
    spec_a: Buf = .{},
    work: Buf = .{},
    ns: Buf = .{},
    peaks: Buf = .{},
    ffts: std.AutoHashMapUnmanaged(u64, fft.BatchFft) = .empty,

    pub const moving = 0;
    pub const fixed = 1;

    pub fn init(g: *Gpu, smi: *smi_mod.Smi) Sweep {
        return .{ .g = g, .smi = smi };
    }

    pub fn deinit(self: *Sweep) void {
        self.release();
        for (&self.il) |*b| self.g.release(b);
        var it = self.ffts.valueIterator();
        while (it.next()) |f| f.deinit(self.g);
        self.ffts.deinit(self.g.gpa);
    }

    /// Free the per-run work buffers (they can be a few hundred MB).
    pub fn release(self: *Sweep) void {
        for ([_]*Buf{ &self.poses, &self.pose_fixed, &self.spec_b, &self.spec_a, &self.work, &self.ns, &self.peaks }) |b| self.g.release(b);
    }

    fn pipe(self: *Sweep, entry: []const u8) u32 {
        var kb: [48]u8 = undefined;
        const key = std.fmt.bufPrint(&kb, "sweep/{s}", .{entry}) catch entry;
        return self.g.pipeline(key, shaders.sweep, entry) catch |e| {
            std.log.err("sweep.wgsl {s}: {s}", .{ entry, @errorName(e) });
            return gpu_mod.NO_PIPE;
        };
    }

    /// A flat_index kernel over `groups` workgroups (folded past 65535), in z layers.
    fn run1(self: *Sweep, entry: []const u8, groups: u32, z: u32, u: SU, binds: []const Bind) void {
        const f = Gpu.flat(groups);
        self.run(entry, f[0], f[1], z, u, binds);
    }

    fn run(self: *Sweep, entry: []const u8, x: u32, y: u32, z: u32, u: SU, binds: []const Bind) void {
        var all: [8]Bind = undefined;
        all[0] = self.g.uniform(0, std.mem.asBytes(&u));
        @memcpy(all[1 .. binds.len + 1], binds);
        self.g.dispatch(self.pipe(entry), x, y, z, all[0 .. binds.len + 1]);
    }

    /// A plane-major premultiplied 5-plane stack (Smi.stackOf) as the moving or fixed image,
    /// interleaved per pixel.
    pub fn setImage(self: *Sweep, side: usize, stack: Buf, w: u32, h: u32) void {
        self.g.ensure(&self.il[side], @as(u64, w) * h * 32);
        self.run1("interleave", (w * h + 255) / 256, 1, .{ .src_w = w, .src_h = h }, &.{ stack.at(1), self.il[side].at(6) });
        self.img[side] = .{ w, h };
    }

    fn batchFft(self: *Sweep, nx: u32, ny: u32) !*fft.BatchFft {
        const key = (@as(u64, nx) << 32) | ny;
        const r = try self.ffts.getOrPut(self.g.gpa, key);
        if (!r.found_existing) {
            r.value_ptr.* = fft.BatchFft.initRect(self.g, nx, ny) catch |e| {
                _ = self.ffts.remove(key);
                return e;
            };
        }
        return r.value_ptr;
    }

    /// Every candidate's shift map, reduced to its top n_peak local maxima. H0 maps moving →
    /// fixed (1-based) and sets the rotation / scale centre (the moving centre mapped by H0).
    /// Scores carry n in fixed-image pixels.
    ///
    /// Each candidate's planes hold only its footprint on the canvas: the moving image's cells at
    /// that pose, correlated with the fixed image's cells. A linear correlation of an M-cell and
    /// an F-cell window needs M + F − 1 points per side, so each candidate takes the smallest
    /// ladder length that fits (per side) instead of the padded canvas (2G): the same lags without
    /// wraparound, at a fraction of the FFT work for small scales and small fixed images.
    pub fn sweep(self: *Sweep, o: SweepOpts) !SweepOut {
        const g = self.g;
        const gpa = g.gpa;
        const exact = o.mode == 1 or o.mode == 3;
        const plan = self.smi.plans[@intFromBool(exact)];
        const n_cplx = (plan.n_real + 1) / 2;
        const fx = self.img[fixed];
        const mv = self.img[moving];
        const grid = o.grid;
        const fw: f64 = @floatFromInt(fx[0]);
        const fh: f64 = @floatFromInt(fx[1]);
        const mw: f64 = @floatFromInt(mv[0]);
        const mh: f64 = @floatFromInt(mv[1]);
        // Canvas: a square around the fixed image, large enough for the moving image at the
        // largest scale (capped at twice the fixed size to keep the resolution).
        const cm = lie.mapPt(o.H0, (mw + 1) / 2, (mh + 1) / 2);
        const gv = o.grid_v;
        // the largest stretch of any candidate (its larger singular value) sizes the canvas
        var s_max: f64 = -std.math.inf(f64);
        for (gv.sigmas) |s| for (gv.shears) |k| for (gv.anisos) |a| {
            s_max = @max(s_max, sv2(candLinear(0, s, k, a))[0]);
        };
        const D = @min(2 * @max(fw, fh), @max(@max(fw, fh), s_max * lie.jsHypot(mw, mh)));
        const cell = D / @as(f64, @floatFromInt(grid));
        const ox = (fw + 1) / 2 - D / 2;
        const oy = (fh + 1) / 2 - D / 2;
        const G: i64 = grid;
        // canvas cells [lo, hi) covering dest-pixel extent [a, b] (one cell of margin for the taps)
        const cells = struct {
            fn f(a: f64, b: f64, o0: f64, c: f64, gg: i64) [2]u32 {
                const lo = std.math.clamp(@as(i64, @intFromFloat(@floor((a - o0) / c))) - 1, 0, gg);
                const hi = std.math.clamp(@as(i64, @intFromFloat(@ceil((b - o0) / c))) + 1, lo + 1, gg);
                return .{ @intCast(lo), @intCast(hi) };
            }
        }.f;
        const fcx = cells(0.5, fw + 0.5, ox, cell, G);
        const fcy = cells(0.5, fh + 0.5, oy, cell, G);
        const Fx = fcx[1] - fcx[0];
        const Fy = fcy[1] - fcy[0];
        // Candidate poses (θ-major), their footprints and plane sizes.
        const nc: u32 = @intCast(gv.count());
        const Cd = struct { it: u32, is: u32, ik: u32, ia: u32, th: f64, s: f64, H: Mat3, gx0: u32, gy0: u32, nx: u32, ny: u32 };
        const cands = try gpa.alloc(Cd, nc);
        defer gpa.free(cands);
        const s0_raw = @sqrt(@abs(o.H0[0] * o.H0[4] - o.H0[1] * o.H0[3]));
        const s0 = if (s0_raw == 0 or std.math.isNan(s0_raw)) 1 else s0_raw;
        const taps = try gpa.alloc(f32, nc);
        defer gpa.free(taps);
        var ci: usize = 0;
        for (gv.thetas, 0..) |th, i| for (gv.sigmas, 0..) |s, j| for (gv.shears, 0..) |kk, ik| for (gv.anisos, 0..) |a, ia| {
            const L = candLinear(th, s, kk, a);
            const H = lie.mul3(aboutC(cm, L), o.H0);
            var x0: f64 = std.math.inf(f64);
            var x1: f64 = -std.math.inf(f64);
            var y0: f64 = std.math.inf(f64);
            var y1: f64 = -std.math.inf(f64);
            for ([_][2]f64{ .{ 0.5, 0.5 }, .{ mw + 0.5, 0.5 }, .{ mw + 0.5, mh + 0.5 }, .{ 0.5, mh + 0.5 } }) |q| {
                const pq = lie.mapPt(H, q[0], q[1]);
                x0 = @min(x0, pq[0]);
                x1 = @max(x1, pq[0]);
                y0 = @min(y0, pq[1]);
                y1 = @max(y1, pq[1]);
            }
            const ok = std.math.isFinite(x0 + x1 + y0 + y1);
            const mcx: [2]u32 = if (ok) cells(x0, x1, ox, cell, G) else .{ 0, grid };
            const mcy: [2]u32 = if (ok) cells(y0, y1, oy, cell, G) else .{ 0, grid };
            cands[ci] = .{
                .it = @intCast(i), .is = @intCast(j), .ik = @intCast(ik), .ia = @intCast(ia), .th = th, .s = s, .H = H,
                .gx0 = mcx[0], .gy0 = mcy[0], .nx = ladderSize(mcx[1] - mcx[0] + Fx - 1), .ny = ladderSize(mcy[1] - mcy[0] + Fy - 1),
            };
            // box-filter taps per cell side from the candidate's smallest stretch
            taps[ci] = @floatCast(@min(6, @max(1, @ceil(cell / (sv2(L)[1] * s0) - 1e-3))));
            ci += 1;
        };
        // Process candidates grouped by plane size; perm[k] = the candidate in GPU slot k.
        const perm = try gpa.alloc(u32, nc);
        defer gpa.free(perm);
        for (perm, 0..) |*q, k| q.* = @intCast(k);
        std.sort.block(u32, perm, cands, struct {
            fn lt(cs: []const Cd, a: u32, b: u32) bool {
                const ka = (@as(u64, cs[a].nx) << 32) | cs[a].ny;
                const kb = (@as(u64, cs[b].nx) << 32) | cs[b].ny;
                return if (ka != kb) ka < kb else a < b;
            }
        }.lt);
        const poses = try gpa.alloc(f32, @as(usize, nc) * 12);
        defer gpa.free(poses);
        @memset(poses, 0);
        for (perm, 0..) |c, k| {
            const Hi = lie.inv3(cands[c].H);
            for (0..9) |q| poses[k * 12 + q] = @floatCast(Hi[q]);
            poses[k * 12 + 9] = taps[c];
            poses[k * 12 + 10] = @floatFromInt(cands[c].gx0);
            poses[k * 12 + 11] = @floatFromInt(cands[c].gy0);
        }
        g.ensure(&self.poses, poses.len * 4);
        g.writeSlice(self.poses, 0, f32, poses);
        g.ensure(&self.pose_fixed, 48);
        g.writeSlice(self.pose_fixed, 0, f32, &.{ 1, 0, 0, 0, 1, 0, 0, 0, 1, @floatCast(@min(6, @max(1, @ceil(cell - 1e-3)))), @floatFromInt(fcx[0]), @floatFromInt(fcy[0]) });
        // Work buffers from a memory budget (spectra + IFFT work + score maps), sized once for the
        // largest planes.
        const budget: u64 = @min(@min(256 << 20, g.lim.max_storage_binding), g.lim.max_buffer / 2);
        var nn_max: u64 = 0;
        for (cands) |cd| nn_max = @max(nn_max, @as(u64, cd.nx) * cd.ny);
        const per_max: u64 = nn_max * 8 * (n_cplx + plan.n_out) + nn_max * 4;
        const B_max: u64 = @max(1, @min(@as(u64, o.batch), budget / per_max));
        g.ensure(&self.spec_b, nn_max * 8 * n_cplx);
        g.ensure(&self.spec_a, @max(nn_max * 8 * n_cplx * B_max, @min(budget, nn_max * 8 * n_cplx * o.batch)));
        g.ensure(&self.work, @max(nn_max * 8 * plan.n_out * B_max, @min(budget, nn_max * 8 * plan.n_out * o.batch)));
        g.ensure(&self.ns, @max(nn_max * 4 * B_max, @min(budget, nn_max * 4 * o.batch)));
        g.ensure(&self.peaks, @as(u64, nc) * o.n_peak * 8);
        const min_n: u32 = @intFromFloat(@max(@as(f64, smi_mod.MIN_N), lie.jsRound(o.min_overlap * (fw * fh) / (cell * cell))));
        var k0: u32 = 0;
        while (k0 < nc) {
            const c_first = cands[perm[k0]];
            const nx = c_first.nx;
            const ny = c_first.ny;
            var k1 = k0;
            while (k1 < nc and cands[perm[k1]].nx == nx and cands[perm[k1]].ny == ny) k1 += 1;
            const nn: u64 = @as(u64, nx) * ny;
            const bf = try self.batchFft(nx, ny);
            const base: SU = .{
                .nx = nx, .ny = ny, .n_real = plan.n_real, .n_cplx = n_cplx, .n_out = plan.n_out, .n_corr = plan.n_corr,
                .ox = @floatCast(ox), .oy = @floatCast(oy), .cell = @floatCast(cell), .mode = o.mode, .n_peak = o.n_peak,
                .fx0 = fcx[0], .fy0 = fcy[0], .grid = grid,
            };
            // the fixed planes: its footprint at the plane origin
            var uf = base;
            uf.src_w = fx[0];
            uf.src_h = fx[1];
            uf.win_x = Fx;
            uf.win_y = Fy;
            const gx = (nx + 7) / 8;
            const gy = (ny + 7) / 8;
            const gnn: u32 = @intCast((nn + 255) / 256);
            self.run("pack_sweep", gx, gy, 1, uf, &.{ self.il[fixed].at(5), self.pose_fixed.at(3), self.spec_b.at(4) });
            bf.run(g, self.spec_b, false, n_cplx);
            const per_cand: u64 = nn * 8 * (n_cplx + plan.n_out) + nn * 4;
            const B: u32 = @intCast(@max(1, @min(@as(u64, o.batch), budget / per_cand)));
            var c0 = k0;
            while (c0 < k1) : (c0 += B) {
                const b = @min(B, k1 - c0);
                var ub = base;
                ub.src_w = mv[0];
                ub.src_h = mv[1];
                ub.cand0 = c0;
                ub.min_n = min_n;
                // moving windows: as wide as the plane allows without wraparound
                ub.win_x = nx - Fx + 1;
                ub.win_y = ny - Fy + 1;
                self.run("pack_sweep", gx, gy, b, ub, &.{ self.il[moving].at(5), self.poses.at(3), self.spec_a.at(4) });
                bf.run(g, self.spec_a, false, b * n_cplx);
                self.run1("cmul_sweep", gnn, b * plan.n_out, ub, &.{ self.spec_a.at(1), self.spec_b.at(2), self.smi.pairs[@intFromBool(exact)].at(3), self.work.at(4) });
                bf.run(g, self.work, true, b * plan.n_out);
                self.run1("combine_sweep", gnn, b, ub, &.{ self.work.at(1), self.poses.at(3), self.ns.at(4) });
                self.run("peaks_sweep", b, 1, 1, ub, &.{ self.ns.at(1), self.peaks.at(4) });
            }
            k0 = k1;
        }
        const raw = try gpa.alloc(f32, @as(usize, nc) * o.n_peak * 2);
        defer gpa.free(raw);
        g.read(self.peaks, 0, std.mem.sliceAsBytes(raw));
        try g.wait();
        const idx: []const u32 = @ptrCast(raw);
        const c2 = cell * cell;
        var out: SweepOut = .{ .n = 0, .cell = cell, .nc = nc };
        errdefer out.deinit(gpa);
        out.best = try gpa.alloc(f32, nc);
        @memset(out.best, 0);
        for (perm, 0..) |c, slot| for (0..o.n_peak) |k| {
            const v = @as(f64, raw[(slot * o.n_peak + k) * 2]) * c2;
            const t = idx[(slot * o.n_peak + k) * 2 + 1];
            const cd = cands[c];
            if (!(v > 0) or t >= @as(u64, cd.nx) * cd.ny) continue;
            if (k == 0) out.best[c] = @floatCast(v);
            out.n = @max(out.n, @max(cd.nx, cd.ny));
            // lag (moving index − fixed index) in [−(F − 1), window − 1], then the canvas
            // displacement of the moving image: gx0 + lag − fx0 cells
            const py = t / cd.nx;
            const px = t - py * cd.nx;
            const wx = cd.nx - Fx + 1;
            const wy = cd.ny - Fy + 1;
            const lx: i64 = if (px < wx) px else @as(i64, px) - cd.nx;
            const ly: i64 = if (py < wy) py else @as(i64, py) - cd.ny;
            const sx: i32 = @intCast(@as(i64, cd.gx0) + lx - fcx[0]);
            const sy: i32 = @intCast(@as(i64, cd.gy0) + ly - fcy[0]);
            // Map peak = moving displaced by +d; the correction is T(−d).
            const dx = @as(f64, @floatFromInt(sx)) * cell;
            const dy = @as(f64, @floatFromInt(sy)) * cell;
            const H = lie.mul3(.{ 1, 0, -dx, 0, 1, -dy, 0, 0, 1 }, cd.H);
            try out.peaks.append(gpa, .{ .score = v, .cand = c, .it = cd.it, .is = cd.is, .ik = cd.ik, .ia = cd.ia, .th = cd.th, .s = cd.s, .sx = sx, .sy = sy, .H = H });
        };
        return out;
    }
};

/// Non-maximum suppression across (θ, σ, shear, stretch, shift): a peak survives only if no
/// stronger survivor lies within ±2 θ steps (circular when wrap), ±2 σ steps, ±1 shear and
/// stretch step and 4 cells.
pub fn nms4(gpa: std.mem.Allocator, peaks: []const Peak, n_theta: u32, keep: u32, wrap: bool) !std.ArrayList(Peak) {
    const sorted = try gpa.dupe(Peak, peaks);
    defer gpa.free(sorted);
    std.sort.block(Peak, sorted, {}, struct {
        fn gt(_: void, a: Peak, b: Peak) bool {
            return a.score > b.score;
        }
    }.gt);
    var kept: std.ArrayList(Peak) = .empty;
    errdefer kept.deinit(gpa);
    for (sorted) |p| {
        var ok = true;
        for (kept.items) |q| {
            const a: u32 = if (p.it > q.it) p.it - q.it else q.it - p.it;
            const dt = if (wrap) @min(a, n_theta - a) else a;
            const ds: u32 = if (p.is > q.is) p.is - q.is else q.is - p.is;
            const dk: u32 = if (p.ik > q.ik) p.ik - q.ik else q.ik - p.ik;
            const da: u32 = if (p.ia > q.ia) p.ia - q.ia else q.ia - p.ia;
            if (dt <= 2 and ds <= 2 and dk <= 1 and da <= 1 and lie.jsHypot(@floatFromInt(p.sx - q.sx), @floatFromInt(p.sy - q.sy)) <= 4) {
                ok = false;
                break;
            }
        }
        if (ok) {
            try kept.append(gpa, p);
            if (kept.items.len >= keep) break;
        }
    }
    return kept;
}

pub const Cand = struct { H: Mat3, score: f64 };

/// Gauss–Newton on many poses at once: each iteration scores, per pose, 5 lengths along the GN
/// step and the `amps` ladder along the unit ascent direction in one batch, and keeps the best
/// step only when the score rises (sweep.js gnBatch). In input order.
///
/// A pose whose step was refused is done: its next iteration would see the same pose, gradient
/// and tries, so it leaves the batch (the result is the same as climbing every pose to the end).
pub fn gnBatch(pr: *pair_mod.Pair, Hs: []const Mat3, iters: u32, group: lie.Group, rw: f64, rh: f64, amps: []const f64, stop: *const bool) ![]Cand {
    const gpa = pr.pg.g.gpa;
    const nk = lie.nKeys(group);
    const cur = try gpa.alloc(Cand, Hs.len);
    errdefer gpa.free(cur);
    for (Hs, 0..) |H, i| cur[i] = .{ .H = H, .score = -std.math.inf(f64) };
    const mults = [_]f64{ 0.5, 1, 2, 4, 8 };
    const grads = try gpa.alloc(pose_mod.Grad, Hs.len);
    defer gpa.free(grads);
    const hs = try gpa.alloc(Mat3, Hs.len);
    defer gpa.free(hs);
    var tries: std.ArrayList(struct { i: usize, H: Mat3 }) = .empty;
    defer tries.deinit(gpa);
    var th: std.ArrayList(Mat3) = .empty;
    defer th.deinit(gpa);
    var sc: std.ArrayList(f64) = .empty;
    defer sc.deinit(gpa);
    const move = struct {
        fn f(H: Mat3, v: []const f64, a: f64, grp: lie.Group, w: f64, h: f64) Mat3 {
            var x: lie.Coords = @splat(0);
            for (v, 0..) |e, k| x[k] = a * e;
            return lie.composeN(H, x, grp, w, h);
        }
    }.f;
    var act: std.ArrayList(usize) = .empty;
    defer act.deinit(gpa);
    for (0..Hs.len) |i| try act.append(gpa, i);
    const moved_i = try gpa.alloc(bool, Hs.len);
    defer gpa.free(moved_i);
    var it: u32 = 0;
    while (it < iters and !stop.* and act.items.len > 0) : (it += 1) {
        const na = act.items.len;
        for (act.items, 0..) |i, k| hs[k] = cur[i].H;
        try pr.gradBatch(hs[0..na], group, true, grads[0..na]);
        tries.clearRetainingCapacity();
        for (grads[0..na], act.items) |gi, i| {
            cur[i].score = gi.score;
            if (amps.len > 0) {
                const d = lie.unitAscent(gi.grad[0..nk], group);
                if (lie.jsHypotN(d[0..nk]) > 1e-12) for (amps) |a| try tries.append(gpa, .{ .i = i, .H = move(cur[i].H, d[0..nk], a, group, rw, rh) });
            }
            const hess = gi.hess orelse continue;
            var A = hess;
            for (0..nk) |k| A[k * nk + k] += 1e-4 * @max(A[k * nk + k], 1e-12);
            var half: [8]f64 = undefined;
            for (0..nk) |k| half[k] = 0.5 * gi.grad[k];
            var step: [8]f64 = undefined;
            if (!lie.solveLinear(A[0 .. nk * nk], half[0..nk], nk, step[0..nk])) continue;
            const nrm = lie.jsHypotN(step[0..nk]);
            for (mults) |m| {
                const s = m * (if (nrm * m > 0.25) 0.25 / (nrm * m) else 1);
                try tries.append(gpa, .{ .i = i, .H = move(cur[i].H, step[0..nk], s, group, rw, rh) });
            }
        }
        if (tries.items.len == 0) break;
        th.clearRetainingCapacity();
        for (tries.items) |t| try th.append(gpa, t.H);
        try sc.resize(gpa, th.items.len);
        try pr.scorePoses(th.items, null, sc.items);
        @memset(moved_i, false);
        for (tries.items, 0..) |t, k| {
            if (climb_mod.beats(sc.items[k], cur[t.i].score)) {
                cur[t.i] = .{ .H = t.H, .score = sc.items[k] };
                moved_i[t.i] = true;
            }
        }
        // the poses that moved climb on
        var w: usize = 0;
        for (act.items) |i| if (moved_i[i]) {
            act.items[w] = i;
            w += 1;
        };
        act.shrinkRetainingCapacity(w);
    }
    return cur;
}

/// Rotation (rad) and scale of H relative to H0 at moving point c (1-based).
pub fn simOf(H: Mat3, H0: Mat3, c: [2]f64) struct { th: f64, s: f64 } {
    const jac = struct {
        fn f(M: Mat3, p: [2]f64) [4]f64 {
            const w = M[6] * p[0] + M[7] * p[1] + M[8];
            const x = (M[0] * p[0] + M[1] * p[1] + M[2]) / w;
            const y = (M[3] * p[0] + M[4] * p[1] + M[5]) / w;
            return .{ (M[0] - M[6] * x) / w, (M[1] - M[7] * x) / w, (M[3] - M[6] * y) / w, (M[4] - M[7] * y) / w };
        }
    }.f;
    const J = jac(H, c);
    const J0 = jac(H0, c);
    const det_raw = J0[0] * J0[3] - J0[1] * J0[2];
    const det0 = if (det_raw == 0 or std.math.isNan(det_raw)) 1e-12 else det_raw;
    const q00 = J0[3] / det0;
    const q01 = -J0[1] / det0;
    const q10 = -J0[2] / det0;
    const q11 = J0[0] / det0;
    const m00 = J[0] * q00 + J[1] * q10;
    const m01 = J[0] * q01 + J[1] * q11;
    const m10 = J[2] * q00 + J[3] * q10;
    const m11 = J[2] * q01 + J[3] * q11;
    return .{ .th = std.math.atan2(m10 - m01, m00 + m11), .s = @sqrt(@abs(m00 * m11 - m01 * m10)) };
}

fn clampF(x: f64, lo: f64, hi: f64) f64 {
    return @min(hi, @max(lo, x));
}

/// The engine's Search: sweep or cloud per settings.search_mode.
pub fn run(e: *Engine) !Result {
    try e.needImages();
    e.cancel = false;
    return switch (e.set.search_mode) {
        .sweep => runSweep(e),
        .cloud => runCloud(e),
    };
}

fn trail(e: *Engine, label: []const u8, H: Mat3) !void {
    const p = try e.pr.parts(H);
    const cps: ?[]const f32 = if (e.ffd.on) e.ffd.cps[0..e.ffd.n()] else null;
    e.events.trail(label, p.mean, p.fwd, p.inv, H, cps);
    e.events.pose(H, cps);
}

/// The app's runSweep: sweep → NMS → batched Gauss–Newton screen → full climbs of the finalists.
fn runSweep(e: *Engine) !Result {
    const gpa = e.gpa;
    const s = &e.set;
    if (s.metric == .ncc) return error.SweepNeedsMomentScore;
    const group = s.group;
    const lw: f64 = @floatFromInt(e.sides[0].w);
    const lh: f64 = @floatFromInt(e.sides[0].h);
    const rw: f64 = @floatFromInt(e.sides[1].w);
    const rh: f64 = @floatFromInt(e.sides[1].h);
    const grid: u32 = switch (s.sw_grid) {
        32, 64, 128, 256 => s.sw_grid,
        else => 128,
    };
    const rot_deg = clampF(s.sw_rot, 0, 180);
    // a range wider than the circle is the circle; hi < lo runs through ±180° (170 → −170)
    const rot_lo: f64 = if (s.sw_rot_range) clampF(s.sw_rot_lo, -360, 360) else -rot_deg;
    var rot_hi: f64 = if (s.sw_rot_range) clampF(s.sw_rot_hi, -360, 360) else rot_deg;
    if (rot_hi < rot_lo) rot_hi += 360;
    rot_hi = @min(rot_hi, rot_lo + 360);
    const s_min = clampF(s.sw_smin, 0.05, 1);
    const s_max = clampF(s.sw_smax, 1, 20);
    var gridv = try sweepGrid(gpa, .{
        .rot_lo = rot_lo, .rot_hi = rot_hi, .n_theta = std.math.clamp(s.sw_nth, 1, 720),
        .s_min = s_min, .s_max = s_max, .n_sigma = std.math.clamp(s.sw_ns, 1, 129),
        .shear_max = if (s.sw_shear) clampF(s.sw_shear_max, 0, 1) else 0, .n_shear = std.math.clamp(s.sw_nshear, 1, 15),
        .aniso_max = if (s.sw_aniso) clampF(s.sw_aniso_max, 0, 1) else 0, .n_aniso = std.math.clamp(s.sw_naniso, 1, 15),
    });
    defer gridv.deinit(gpa);
    const K: u32 = switch (s.sw_k) {
        8, 16, 32 => s.sw_k,
        else => 32,
    };
    const n_fin = std.math.clamp(s.sw_fin, 1, 16);
    const min_ov = clampF(s.min_overlap, 0, 0.5);
    const Hcur = e.H;
    const H0: Mat3 = if (s.sw_rel) Hcur else .{ 1, 0, (rw - lw) / 2, 0, 1, (rh - lh) / 2, 0, 0, 1 };
    const c_mov = [2]f64{ (lw + 1) / 2, (lh + 1) / 2 };
    var s_cur_a: [1]f64 = undefined;
    try e.pr.scorePoses(&.{Hcur}, null, &s_cur_a);
    const s_cur = s_cur_a[0];
    const ffd_was = e.ffd.on;
    e.ffd.on = false;
    defer e.ffd.on = ffd_was;
    const edge = s.metric == .smi_edge;
    const sw = try e.sweeper();
    sw.setImage(Sweep.moving, e.smi.stackOf(&e.feats[0], edge), e.sides[0].w, e.sides[0].h);
    sw.setImage(Sweep.fixed, e.smi.stackOf(&e.feats[1], edge), e.sides[1].w, e.sides[1].h);
    const mode: u32 = switch (s.family()) {
        .e4 => 2,
        .lmax => 3,
        .smi => @intFromBool(s.exact),
    };
    var res = blk: {
        defer sw.release();
        break :blk try sw.sweep(.{ .H0 = H0, .grid_v = &gridv, .grid = grid, .mode = mode, .n_peak = 4, .min_overlap = min_ov });
    };
    defer res.deinit(gpa);
    // the landscape view is θ × σ: each cell's best over shear and stretch
    const n_ka = gridv.shears.len * gridv.anisos.len;
    const land = try gpa.alloc(f32, gridv.thetas.len * gridv.sigmas.len);
    defer gpa.free(land);
    for (land, 0..) |*v, i| v.* = std.mem.max(f32, res.best[i * n_ka .. (i + 1) * n_ka]);
    var top = try nms4(gpa, res.peaks.items, @intCast(gridv.thetas.len), K, gridv.wrap);
    defer top.deinit(gpa);
    const score_name = switch (s.metric) {
        .smi => if (s.exact) "exact" else "global",
        .smi_edge => if (s.exact) "exact edge" else "global edge",
        .e4 => "E4",
        .lmax => "λmax",
        .ncc => "NCC",
    };
    {
        var xb: [96]u8 = undefined;
        var xw: std.Io.Writer = .fixed(&xb);
        if (gridv.shears.len > 1) xw.print(" × {d} shear ±{d}", .{ gridv.shears.len, gridv.shears[gridv.shears.len - 1] }) catch {};
        if (gridv.anisos.len > 1) xw.print(" × {d} stretch ≤{d:.3}", .{ gridv.anisos.len, gridv.anisos[gridv.anisos.len - 1] }) catch {};
        e.events.log("search  {d}θ {d}°…{d}° × {d}σ {d}–{d}{s}  {d} maps  {d}² grid ({d:.1} px, FFT {d}²)  {s}  {d} peaks", .{ gridv.thetas.len, rot_lo, rot_hi, gridv.sigmas.len, s_min, s_max, xw.buffered(), res.nc, grid, res.cell, res.n, score_name, top.items.len });
    }
    var out: Result = .{ .H = Hcur, .score = s_cur, .prev = s_cur, .count_a = @intCast(top.items.len) };
    if (top.items.len == 0) {
        e.events.log("search  no peak above the overlap floor", .{});
        try emitSweep(e, &gridv, land, top.items, &.{}, null, &[_]struct { H: Mat3, score: f64, peak: usize, th: f64, s: f64 }{}, false);
        return out;
    }
    var amps_buf: [12]f64 = undefined;
    const amps = climb_mod.geoAmps(s.g_a0, s.g_decay, if (s.g_steps > 0) s.g_steps else 7, &amps_buf);
    const starts = try gpa.alloc(Mat3, top.items.len);
    defer gpa.free(starts);
    for (top.items, 0..) |p, i| starts[i] = p.H;
    const screened = try gnBatch(&e.pr, starts, 4, group, rw, rh, amps, &e.cancel);
    defer gpa.free(screened);
    // Finalists: the best after the short climb, plus the best coarse peaks. A true pose a cell
    // off sits in a narrow full-resolution basin and can still be low after 4 steps while wrong
    // poses in broad basins are already at their top.
    const order = try gpa.alloc(usize, screened.len);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.sort.block(usize, order, screened, struct {
        fn gt(sc: []const Cand, a: usize, b: usize) bool {
            return sc[a].score > sc[b].score;
        }
    }.gt);
    var fin_idx: std.ArrayList(usize) = .empty;
    defer fin_idx.deinit(gpa);
    for (order[0..@min(n_fin, order.len)]) |i| if (std.mem.indexOfScalar(usize, fin_idx.items, i) == null) try fin_idx.append(gpa, i);
    for (0..@min(n_fin, top.items.len)) |i| if (std.mem.indexOfScalar(usize, fin_idx.items, i) == null) try fin_idx.append(gpa, i);
    {
        var lb: [160]u8 = undefined;
        var w: std.Io.Writer = .fixed(&lb);
        for (fin_idx.items, 0..) |i, k| w.print("{s}#{d}", .{ if (k > 0) " " else "", i + 1 }) catch {};
        e.events.log("search  screen {d} × 4 steps  best nS {d:.1} (peak #{d})  finalists {s}", .{ top.items.len, screened[order[0]].score, order[0] + 1, w.buffered() });
    }
    const fin_h = try gpa.alloc(Mat3, fin_idx.items.len);
    defer gpa.free(fin_h);
    for (fin_idx.items, 0..) |i, k| fin_h[k] = screened[i].H;
    const finals = try gnBatch(&e.pr, fin_h, 25, group, rw, rh, amps, &e.cancel);
    defer gpa.free(finals);
    var bi: usize = 0;
    for (finals, 0..) |f, i| if (f.score > finals[bi].score) {
        bi = i;
    };
    const best = finals[bi];
    const rs = simOf(best.H, H0, c_mov);
    {
        var lb: [256]u8 = undefined;
        var w: std.Io.Writer = .fixed(&lb);
        for (finals, 0..) |f, k| w.print("{s}{d:.1}", .{ if (k > 0) " / " else "", f.score }) catch {};
        e.events.log("search  climb {d} × ≤25 GN  nS {s}", .{ finals.len, w.buffered() });
    }
    // the climbed finalists, best first (the Search tab's list; a click applies one)
    const Cd = struct { H: Mat3, score: f64, peak: usize, th: f64, s: f64 };
    const cands = try gpa.alloc(Cd, finals.len);
    defer gpa.free(cands);
    for (finals, 0..) |f, i| {
        const so = simOf(f.H, H0, c_mov);
        cands[i] = .{ .H = f.H, .score = f.score, .peak = fin_idx.items[i] + 1, .th = so.th, .s = so.s };
    }
    std.sort.block(Cd, cands, {}, struct {
        fn gt(_: void, a: Cd, b: Cd) bool {
            return a.score > b.score;
        }
    }.gt);
    try emitSweep(e, &gridv, land, top.items, fin_idx.items, .{ rs.th, rs.s }, cands, climb_mod.beats(best.score, s_cur));
    e.ffd.on = ffd_was;
    out.count_b = @intCast(finals.len);
    if (climb_mod.beats(best.score, s_cur)) {
        e.H = best.H;
        if (ffd_was) @memset(&e.ffd.cps, 0);
        try trail(e, "search", best.H);
        out.H = best.H;
        out.score = best.score;
        out.improved = 1;
        e.events.log("search done  nS {d:.1} (was {d:.1})  θ {d:.1}° σ {d:.3}{s}  peak #{d}{s}", .{ best.score, s_cur, rs.th * 180 / std.math.pi, rs.s, if (s.sw_rel) " about H" else "", fin_idx.items[bi] + 1, if (ffd_was) "  spline reset" else "" });
    } else {
        e.events.log("search done  kept the current pose (nS {d:.1} ≥ {d:.1})", .{ s_cur, best.score });
    }
    return out;
}

/// The sweep's landscape for the population view: each (θ, σ) cell's best coarse score, the NMS
/// peaks, the finalists (indices into peaks) and the result's (θ, σ).
fn emitSweep(e: *Engine, gridv: *const Grid, best: []const f32, top: []const Peak, fin: []const usize, result: ?[2]f64, cands: anytype, applied: bool) !void {
    const gpa = e.gpa;
    const pk = try gpa.alloc([2]u32, top.len);
    defer gpa.free(pk);
    for (top, 0..) |p, i| pk[i] = .{ p.it, p.is };
    e.events.emit(.{
        .kind = "sweep", .thetas = gridv.thetas, .sigmas = gridv.sigmas, .wrap = gridv.wrap, .best = best, .top = pk,
        .finalists = fin, .result = result, .cands = cands, .applied = @as(i32, if (applied) 0 else -1),
    });
}

/// The app's runHho: a seed cloud in the Lie algebra (with per-seed spline control points when the
/// spline is on), all seeds stepping along their unit ascent directions over the step ladder at
/// once; a seed stops when it diverges, flattens or stalls.
fn runCloud(e: *Engine) !Result {
    const gpa = e.gpa;
    const s = &e.set;
    const group = s.group;
    const rw: f64 = @floatFromInt(e.sides[1].w);
    const rh: f64 = @floatFromInt(e.sides[1].h);
    const auto = s.hho_auto;
    const g_ffd = @max(2, e.ffd.g);
    const n_seed: u32 = if (auto) 100 else std.math.clamp(if (s.hho_n > 0) s.hho_n else 100, 1, 256);
    const n_step: u32 = if (auto) 20 else std.math.clamp(if (s.hho_t > 0) s.hho_t else 20, 1, 80);
    const edge: f64 = 0.01;
    const orD = struct {
        fn f(v: f64, d: f64) f64 {
            return if (v == 0 or std.math.isNan(v)) d else v;
        }
    }.f;
    const sg = if (auto) edge else @max(1e-6, orD(s.hho_sg, 0.01));
    const shear = if (auto) edge else @max(1e-6, orD(s.hho_sh, 0.01));
    const persp = if (auto) edge * 0.01 else @max(1e-8, orD(s.hho_p, 0.0001));
    const trans = if (auto) edge else @max(1e-6, orD(s.hho_tx, 0.01));
    const th_w = if (auto) edge else @max(1e-6, orD(s.hho_th, 0.01));
    const phi_amp = if (auto) @round(0.05 * 2 / @as(f64, @floatFromInt(@max(g_ffd - 1, 1))) * 1e4) / 1e4 else @max(0, s.hho_phi);
    const seed: u32 = if (s.hho_seed != 0) s.hho_seed else 1;
    const Hbase = if (s.hho_rel) e.H else lie.identity;
    // search space: tx, ty, then the group's keys after translation (search.js searchSpace)
    const nk = lie.nKeys(group);
    var lo: [8]f64 = undefined;
    var hi: [8]f64 = undefined;
    for (0..nk) |k| {
        const w: f64 = switch (k) {
            lie.TX, lie.TY => trans,
            lie.TH => th_w,
            lie.SG => sg,
            lie.AL, lie.GA => shear,
            else => persp,
        };
        lo[k] = -w;
        hi[k] = w;
    }
    var amps_buf: [12]f64 = undefined;
    const amps = climb_mod.geoAmps(if (s.g_a0 > 0) s.g_a0 else 1, s.g_decay, if (s.g_steps > 0) s.g_steps else 4, &amps_buf);
    const use_ffd = s.ffd and e.ffd.on;
    const npar: usize = if (use_ffd) e.ffd.n() else 0;
    var rng: u32 = seed;
    const rnd = struct {
        fn f(st: *u32) f64 {
            st.* +%= 0x6D2B79F5;
            var t = st.*;
            t = (t ^ (t >> 15)) *% (t | 1);
            t ^= t +% ((t ^ (t >> 7)) *% (t | 61));
            return @as(f64, @floatFromInt(t ^ (t >> 14))) / 4294967296.0;
        }
    }.f;
    const Member = struct { H: Mat3, cps: []f32, s: f64, s0: f64, alive: bool, why: enum { none, diverged, flat, stall } };
    const members = try gpa.alloc(Member, n_seed);
    defer gpa.free(members);
    const cps_all = try gpa.alloc(f32, @max(1, n_seed * npar));
    defer gpa.free(cps_all);
    for (members, 0..) |*m, i| {
        var x: lie.Coords = @splat(0);
        if (i > 0) for (0..nk) |k| {
            x[k] = lo[k] + (hi[k] - lo[k]) * rnd(&rng);
        };
        const cps = cps_all[i * npar ..][0..npar];
        if (use_ffd) {
            @memcpy(cps, e.ffd.cps[0..npar]);
            if (i > 0 and phi_amp > 0) for (cps) |*c| {
                c.* = @floatCast(@as(f64, c.*) + (rnd(&rng) * 2 - 1) * phi_amp);
            };
        }
        m.* = .{ .H = lie.composeN(Hbase, x, group, rw, rh), .cps = cps, .s = -std.math.inf(f64), .s0 = 0, .alive = true, .why = .none };
    }
    var best_h = Hbase;
    var best_s = -std.math.inf(f64);
    const best_cps = try gpa.alloc(f32, @max(1, npar));
    defer gpa.free(best_cps);
    if (use_ffd) @memcpy(best_cps[0..npar], e.ffd.cps[0..npar]);
    // scratch for the batched calls
    var hs: std.ArrayList(Mat3) = .empty;
    defer hs.deinit(gpa);
    var pk: std.ArrayList(f32) = .empty;
    defer pk.deinit(gpa);
    var sc: std.ArrayList(f64) = .empty;
    defer sc.deinit(gpa);
    for (members) |m| {
        try hs.append(gpa, m.H);
        try pk.appendSlice(gpa, m.cps);
    }
    try sc.resize(gpa, members.len);
    try e.pr.scorePoses(hs.items, if (use_ffd) pk.items else null, sc.items);
    for (members, 0..) |*m, i| {
        m.s = if (std.math.isNan(sc.items[i])) 0 else sc.items[i];
        m.s0 = m.s;
        if (m.s > best_s) {
            best_s = m.s;
            best_h = m.H;
            if (use_ffd) @memcpy(best_cps[0..npar], m.cps);
        }
    }
    const setBest = struct {
        fn f(en: *Engine, H: Mat3, cps: []const f32, ffd: bool) void {
            en.H = H;
            if (ffd) @memcpy(en.ffd.cps[0..cps.len], cps);
        }
    }.f;
    e.events.log("climb  {d} seeds × {d} steps  {s}{s}{s}  nS {d:.1}", .{ n_seed, n_step, if (s.symmetric) "mean SMI  " else "SMI  ", if (use_ffd) "H+φ  " else "", if (s.hho_rel) "around H₀" else "from I", best_s });
    setBest(e, best_h, best_cps[0..npar], use_ffd);
    try trail(e, "climb 0", best_h);
    try emitSwarm(e, members);
    var best_mark = best_s;
    var evals: u32 = n_seed;
    const Try = struct { m: usize, a: f64, H: Mat3, cps_off: usize };
    var trials: std.ArrayList(Try) = .empty;
    defer trials.deinit(gpa);
    var tcps: std.ArrayList(f32) = .empty;
    defer tcps.deinit(gpa);
    var live: std.ArrayList(usize) = .empty;
    defer live.deinit(gpa);
    var grads: std.ArrayList(pose_mod.Grad) = .empty;
    defer grads.deinit(gpa);
    var phi_s: std.ArrayList(f64) = .empty;
    defer phi_s.deinit(gpa);
    var phi_g: std.ArrayList(f64) = .empty;
    defer phi_g.deinit(gpa);
    var it: u32 = 0;
    while (it < n_step and !e.cancel) : (it += 1) {
        live.clearRetainingCapacity();
        for (members, 0..) |m, i| if (m.alive) try live.append(gpa, i);
        if (live.items.len == 0) break;
        hs.clearRetainingCapacity();
        pk.clearRetainingCapacity();
        for (live.items) |i| {
            try hs.append(gpa, members[i].H);
            try pk.appendSlice(gpa, members[i].cps);
        }
        try grads.resize(gpa, live.items.len);
        try e.pr.gradBatchCps(hs.items, if (use_ffd) pk.items else null, group, false, grads.items);
        if (use_ffd) {
            try phi_s.resize(gpa, live.items.len);
            try phi_g.resize(gpa, live.items.len * npar);
            try e.pr.ffdGradBatch(hs.items, pk.items, phi_s.items, phi_g.items);
        }
        trials.clearRetainingCapacity();
        tcps.clearRetainingCapacity();
        for (live.items, 0..) |mi, j| {
            const m = &members[mi];
            const g = grads.items[j];
            if (!std.math.isFinite(g.score) or g.score < 0.25 * @max(m.s0, 1)) {
                m.alive = false;
                m.why = .diverged;
                continue;
            }
            const d = lie.unitAscent(g.grad[0..nk], group);
            const nrm = lie.jsHypotN(d[0..nk]);
            var n_phi: f64 = 0;
            const gp: []const f64 = if (use_ffd) phi_g.items[j * npar ..][0..npar] else &.{};
            for (gp) |v| n_phi += v * v;
            n_phi = @sqrt(n_phi);
            if (nrm < 1e-12 and n_phi < 1e-12) {
                m.alive = false;
                m.why = .flat;
                continue;
            }
            for (amps) |a| {
                var x: lie.Coords = @splat(0);
                for (0..nk) |k| x[k] = a * d[k] / nrm;
                const Hn = if (nrm >= 1e-12) lie.composeN(m.H, x, group, rw, rh) else m.H;
                const off = tcps.items.len;
                if (use_ffd) {
                    try tcps.appendSlice(gpa, m.cps);
                    if (n_phi >= 1e-12) for (0..npar) |k| {
                        tcps.items[off + k] = @floatCast(@as(f64, tcps.items[off + k]) + a * gp[k] / n_phi);
                    };
                }
                try trials.append(gpa, .{ .m = mi, .a = a, .H = Hn, .cps_off = off });
            }
        }
        evals += @intCast(live.items.len + trials.items.len);
        if (trials.items.len == 0) break;
        hs.clearRetainingCapacity();
        for (trials.items) |t| try hs.append(gpa, t.H);
        try sc.resize(gpa, trials.items.len);
        try e.pr.scorePoses(hs.items, if (use_ffd) tcps.items else null, sc.items);
        // each member's best step (the first of equal scores)
        var moved = false;
        for (live.items) |mi| {
            const m = &members[mi];
            if (!m.alive) continue;
            var hit: ?usize = null;
            for (trials.items, 0..) |t, k| if (t.m == mi and (hit == null or sc.items[k] > sc.items[hit.?])) {
                hit = k;
            };
            if (hit == null or !climb_mod.beats(sc.items[hit.?], m.s)) {
                m.alive = false;
                m.why = .stall;
                continue;
            }
            const t = trials.items[hit.?];
            m.H = t.H;
            if (use_ffd) @memcpy(m.cps, tcps.items[t.cps_off..][0..npar]);
            m.s = sc.items[hit.?];
            moved = true;
            if (climb_mod.beats(m.s, best_s)) {
                best_s = m.s;
                best_h = m.H;
                if (use_ffd) @memcpy(best_cps[0..npar], m.cps);
            }
        }
        var n_live: u32 = 0;
        for (members) |m| n_live += @intFromBool(m.alive);
        if (!moved) break;
        if (climb_mod.beats(best_s, best_mark)) {
            best_mark = best_s;
            setBest(e, best_h, best_cps[0..npar], use_ffd);
            var lb: [24]u8 = undefined;
            try trail(e, std.fmt.bufPrint(&lb, "climb {d}", .{it + 1}) catch "climb", best_h);
        }
        try emitSwarm(e, members);
        e.events.log("climb {d}/{d}  nS {d:.1}  live {d}", .{ it + 1, n_step, best_s, n_live });
    }
    setBest(e, best_h, best_cps[0..npar], use_ffd);
    var why = [_]u32{ 0, 0, 0 };
    for (members) |m| switch (m.why) {
        .diverged => why[0] += 1,
        .flat => why[1] += 1,
        .stall => why[2] += 1,
        .none => {},
    };
    var phi_max: f64 = 0;
    if (use_ffd) for (best_cps[0..npar]) |v| {
        phi_max = @max(phi_max, @abs(v));
    };
    var note: [40]u8 = undefined;
    const phi_note = if (use_ffd) std.fmt.bufPrint(&note, "  φ {e:.2}", .{phi_max}) catch "" else "";
    e.events.log("climb done  nS {d:.4}  {d} evals  {d} diverged, {d} flat, {d} stall{s}", .{ best_s, evals, why[0], why[1], why[2], phi_note });
    return .{ .H = best_h, .score = best_s, .prev = members[0].s0, .improved = @intFromBool(climb_mod.beats(best_s, members[0].s0)), .count_a = evals, .count_b = n_seed };
}

/// The cloud's member scores after a step (the population view's rows).
fn emitSwarm(e: *Engine, members: anytype) !void {
    const gpa = e.gpa;
    const s = try gpa.alloc(?f64, members.len);
    defer gpa.free(s);
    for (members, 0..) |m, i| s[i] = if (std.math.isFinite(m.s)) m.s else null;
    e.events.emit(.{ .kind = "swarm", .scores = s });
}
