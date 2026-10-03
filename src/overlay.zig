//! Images at a pose (port of engine.js warpH / pasteGray / assemblePose / assemblePoseInverse,
//! ncc.js and smi.js tileExchange):
//!   overlay     the moving image warped by H (and the live spline) and the fixed image pasted
//!               into the pose's union canvas, as f32 in [0, 1] — what the app's overlay shows;
//!               inverse: the fixed image backprojected into the moving frame (spline adjoint)
//!   NCC         the normalized cross-correlation shift map and zero-lag score of that pair
//!   tile heat   the exchange-symmetric per-tile score grid (population view)
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const shaders = @import("shaders");
const lie = @import("lie.zig");
const fft = @import("fft.zig");
const smi_mod = @import("smi.zig");
const gls_mod = @import("gls.zig");
const pose_mod = @import("pose.zig");
const engine = @import("engine.zig");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Bind = gpu_mod.Bind;
const Mat3 = lie.Mat3;
const Engine = engine.Engine;

const ffd32 = "alias Texel = f32;\n" ++ shaders.ffd;
const ffd16 = "enable f16;\nalias Texel = f16;\n" ++ shaders.ffd;

/// The union canvas of a pose: offset of its top-left pixel in fixed pixels, its size, and the
/// pose mapping moving pixels onto it.
pub const Canvas = extern struct { ox: i32 = 0, oy: i32 = 0, ow: u32 = 0, oh: u32 = 0, Hc: [9]f64 = lie.identity };

/// Where a warp's spline lattice lives (engine.js warpH frame): the canvas offset and the fixed
/// (rw, rh) and moving (sw, sh) sizes.
pub const Frame = struct { ox: i32 = 0, oy: i32 = 0, rw: u32, rh: u32, sw: u32, sh: u32 };

/// A spline to warp with: 2·g·g control points in destN units, lattice on the moving image
/// ("source") or the fixed image.
pub const Spline = struct { g: u32, cps: []const f32, source: bool };

pub const State = struct {
    m: Buf = .{},
    f: Buf = .{},
    read: Buf = .{},
    a32: Buf = .{},
    b32: Buf = .{},
    // NCC
    part: Buf = .{},
    spec_a: Buf = .{},
    spec_b: Buf = .{},
    n_cap: u32 = 0,
    ffts: std.AutoHashMapUnmanaged(u32, fft.LineFft) = .empty,
    tile_ns: Buf = .{},

    pub fn deinit(self: *State, g: *Gpu) void {
        for ([_]*Buf{ &self.m, &self.f, &self.read, &self.a32, &self.b32, &self.part, &self.spec_a, &self.spec_b, &self.tile_ns }) |b| g.release(b);
        var it = self.ffts.valueIterator();
        while (it.next()) |f| f.deinit(g);
        self.ffts.deinit(g.gpa);
    }
};

fn texelBytes(e: *Engine) u64 {
    return if (e.gls.half) 2 else 4;
}

/// Warp `left` (lw × lh texels) into dst (dw × dh) through H (moving → dst pixels, 1-based),
/// through the spline when given; use32: f32 buffers.
pub fn warpH(e: *Engine, left: Buf, lw: u32, lh: u32, dw: u32, dh: u32, H: Mat3, dst: Buf, frame: ?Frame, use32: bool, sp_opt: ?Spline) void {
    const g = &e.g;
    var h32: [9]f32 = undefined;
    for (0..9) |i| h32[i] = @floatCast(H[i]);
    var q = e.gls.p;
    q.w = dw;
    q.h = dh;
    q.src_w = lw;
    q.src_h = lh;
    const P = e.gls.Pwith(q);
    const Hb = g.stage(16, std.mem.sliceAsBytes(&h32));
    if (sp_opt) |sp| {
        const src = sp.source;
        const fr = frame orelse Frame{ .rw = dw, .rh = dh, .sw = lw, .sh = lh };
        const rw = if (src) fr.sw else fr.rw;
        const rh = if (src) fr.sh else fr.rh;
        const F = extern struct { gx: u32, gy: u32, rw: u32, rh: u32, ox: f32, oy: f32, src: f32, pad: f32 };
        const f: F = .{
            .gx = sp.g, .gy = sp.g, .rw = rw, .rh = rh,
            .ox = if (src) 0 else @floatFromInt(fr.ox), .oy = if (src) 0 else @floatFromInt(fr.oy), .src = if (src) 1 else 0, .pad = 0,
        };
        const half = e.gls.half and !use32;
        const pipe = g.pipeline(if (half) "ffd16/warp_ffd" else "ffd32/warp_ffd", if (half) ffd16 else ffd32, "warp_ffd") catch gpu_mod.NO_PIPE;
        // own buffer: warp_ffd declares H (binding 16) read-write, so the two cannot share the
        // staging arena; released after the next submit
        var cb = g.storage(sp.cps.len * 4);
        g.writeSlice(cb, 0, f32, sp.cps);
        g.dispatch(pipe, (dw + 7) / 8, (dh + 7) / 8, 1, &.{ P, left.at(1), dst.at(2), cb.at(4), g.uniform(5, std.mem.asBytes(&f)), Hb });
        g.release(&cb);
        return;
    }
    g.dispatch(e.gls.pipe("warp_homography", use32), (dw + 7) / 8, (dh + 7) / 8, 1, &.{ P, left.at(1), dst.at(2), Hb });
}

/// Copy src (sw × sh) into dst (dw × dh) at offset (dx, dy); pixels outside are left as they are.
pub fn paste(e: *Engine, src: Buf, sw: u32, sh: u32, dst: Buf, dw: u32, dh: u32, dx: i32, dy: i32, use32: bool) void {
    var q = e.gls.p;
    q.w = dw;
    q.h = dh;
    q.src_w = sw;
    q.src_h = sh;
    q.off_src = @bitCast(dx);
    q.off_dst = @bitCast(dy);
    e.g.dispatch(e.gls.pipe("paste_rect", use32), (dw + 7) / 8, (dh + 7) / 8, 1, &.{ e.gls.Pwith(q), src.at(1), dst.at(2) });
}

/// The engine's spline as a warp takes it (null unless live); adjoint: negated, other frame.
fn spline(e: *Engine, adjoint: bool, buf: []f32) ?Spline {
    if (!e.ffd.live()) return null;
    const n = e.ffd.n();
    for (0..n) |i| buf[i] = if (adjoint) -e.ffd.cps[i] else e.ffd.cps[i];
    return .{ .g = e.ffd.g, .cps = buf[0..n], .source = e.ffd.source != adjoint };
}

/// The pose's union canvas (host only).
pub fn canvas(e: *const Engine, H: Mat3, inverse: bool) Canvas {
    const a = e.sides[if (inverse) 1 else 0];
    const b = e.sides[if (inverse) 0 else 1];
    const Hp = if (inverse) lie.inv3(H) else H;
    const fr = lie.regFrame(Hp, a.w, a.h, b.w, b.h);
    return .{ .ox = fr.ox, .oy = fr.oy, .ow = fr.w, .oh = fr.h, .Hc = lie.canvasH(Hp, fr.ox, fr.oy) };
}

/// Warp and paste the pair onto the canvas (texels in st.m, st.f). inverse: the fixed image
/// warped by H⁻¹ (spline adjoint) with the moving image pasted.
fn assemble(e: *Engine, H: Mat3, inverse: bool) Canvas {
    const c = canvas(e, H, inverse);
    const st = &e.ov;
    const g = &e.g;
    const n: u64 = @as(u64, c.ow) * c.oh;
    g.ensure(&st.m, n * texelBytes(e));
    g.ensure(&st.f, n * texelBytes(e));
    const a = e.sides[if (inverse) 1 else 0];
    const b = e.sides[if (inverse) 0 else 1];
    var cps: [pose_mod.FFD_MAX * pose_mod.FFD_MAX * 2]f32 = undefined;
    warpH(e, a.work, a.w, a.h, c.ow, c.oh, c.Hc, st.m, .{ .ox = c.ox, .oy = c.oy, .rw = b.w, .rh = b.h, .sw = a.w, .sh = a.h }, false, spline(e, inverse, &cps));
    g.clear(st.f);
    paste(e, b.work, b.w, b.h, st.f, c.ow, c.oh, -c.ox, -c.oy, false);
    return c;
}

/// The overlay at H (engine.js assemblePose / assemblePoseInverse): moving and fixed on the
/// union canvas, f32 clamped to [0, 1], into moving_out and fixed_out (ow·oh each).
pub fn overlay(e: *Engine, H: Mat3, inverse: bool, moving_out: []f32, fixed_out: []f32) !Canvas {
    try e.needImages();
    const c = canvas(e, H, inverse);
    const n: usize = @as(usize, c.ow) * c.oh;
    if (moving_out.len < n or fixed_out.len < n) return error.BufferTooSmall;
    _ = assemble(e, H, inverse);
    const st = &e.ov;
    e.g.ensure(&st.read, n * 2 * 4);
    e.gls.promoteTexels(st.m, st.read, @intCast(n), 0, 0);
    e.gls.promoteTexels(st.f, st.read, @intCast(n), 0, @intCast(n));
    e.g.read(st.read, 0, std.mem.sliceAsBytes(moving_out[0..n]));
    e.g.read(st.read, n * 4, std.mem.sliceAsBytes(fixed_out[0..n]));
    try e.g.wait();
    for (moving_out[0..n]) |*v| v.* = std.math.clamp(v.*, 0, 1);
    for (fixed_out[0..n]) |*v| v.* = std.math.clamp(v.*, 0, 1);
    return c;
}

/// The assembled pair as f32 buffers (engine.js asF32).
fn assembledF32(e: *Engine, c: Canvas) void {
    const st = &e.ov;
    const n: u64 = @as(u64, c.ow) * c.oh;
    e.g.ensure(&st.a32, n * 4);
    e.g.ensure(&st.b32, n * 4);
    e.gls.promoteTexels(st.m, st.a32, @intCast(n), 0, 0);
    e.gls.promoteTexels(st.f, st.b32, @intCast(n), 0, 0);
}

// ── NCC (ncc.js NccFft) ──
const NU = extern struct { src_w: u32, src_h: u32, n: u32, cw: u32, ch: u32, mode: u32, p1: u32 = 0, p2: u32 = 0, mean: f32, s0: f32 = 0, s1: f32 = 0, s2: f32 = 0 };

fn nrun(e: *Engine, entry: []const u8, x: u32, y: u32, u: NU, binds: []const Bind) void {
    var kb: [32]u8 = undefined;
    const key = std.fmt.bufPrint(&kb, "ncc/{s}", .{entry}) catch entry;
    const pipe = e.g.pipeline(key, shaders.ncc, entry) catch gpu_mod.NO_PIPE;
    var all: [4]Bind = undefined;
    all[0] = e.g.uniform(0, std.mem.asBytes(&u));
    @memcpy(all[1 .. binds.len + 1], binds);
    e.g.dispatch(pipe, x, y, 1, all[0 .. binds.len + 1]);
}

fn nccReduce(e: *Engine, spec: Buf, n: u32, cw: u32, ch: u32, mode: u32) !f64 {
    const st = &e.ov;
    nrun(e, "reduce_spec", 64, 1, .{ .src_w = 1, .src_h = 1, .n = n, .cw = cw, .ch = ch, .mean = 0, .mode = mode }, &.{ spec.at(1), st.part.at(3) });
    var raw: [64]f32 = undefined;
    e.g.read(st.part, 0, std.mem.sliceAsBytes(&raw));
    try e.g.wait();
    var s: f64 = 0;
    for (raw) |v| s += v;
    return s;
}

fn nccPack(e: *Engine, gray: Buf, sw: u32, sh: u32, n: u32, cw: u32, ch: u32, spec: Buf, mean: f64) void {
    const gx = (n + 7) / 8;
    nrun(e, "pack", gx, gx, .{ .src_w = sw, .src_h = sh, .n = n, .cw = cw, .ch = ch, .mean = @floatCast(mean), .mode = 0 }, &.{ gray.at(1), spec.at(3) });
}

/// NCC shift map of the pair at H (engine.js metricMapPose "NCC"): the n×n map (lag 0 at index
/// 0, clamped to ±1.5) into map, the peak and its lag in canvas pixels.
pub fn nccShiftMap(e: *Engine, H: Mat3, map: ?[]f32) !engine.ShiftResult {
    try e.needImages();
    const c = assemble(e, H, false);
    assembledF32(e, c);
    const st = &e.ov;
    const g = &e.g;
    const wa = c.ow;
    const ha = c.oh;
    const plan = fft.linearCorrPlan(wa, ha, wa, ha, @min(fft.maxFftN(g.lim.max_workgroup_storage), 1024));
    const n = plan[0];
    const cw = plan[1];
    const ch = plan[2];
    if (st.spec_a.id == 0 or n > st.n_cap) {
        g.release(&st.spec_a);
        g.release(&st.spec_b);
        st.spec_a = g.storage(@as(u64, n) * n * 8);
        st.spec_b = g.storage(@as(u64, n) * n * 8);
        st.n_cap = n;
    }
    if (st.part.id == 0) st.part = g.storage(64 * 5 * 4);
    const gop = try st.ffts.getOrPut(g.gpa, n);
    if (!gop.found_existing) gop.value_ptr.* = fft.LineFft.init(g, n) catch |err| {
        _ = st.ffts.remove(n);
        return err;
    };
    const lf = gop.value_ptr;
    nccPack(e, st.a32, wa, ha, n, cw, ch, st.spec_a, 0);
    nccPack(e, st.b32, wa, ha, n, cw, ch, st.spec_b, 0);
    const cells: f64 = @floatFromInt(cw * ch);
    const sa = (try nccReduce(e, st.spec_a, n, cw, ch, 0)) / cells;
    const sb = (try nccReduce(e, st.spec_b, n, cw, ch, 0)) / cells;
    nccPack(e, st.a32, wa, ha, n, cw, ch, st.spec_a, sa);
    nccPack(e, st.b32, wa, ha, n, cw, ch, st.spec_b, sb);
    const nrm_a = @sqrt(@max(try nccReduce(e, st.spec_a, n, cw, ch, 1), 0));
    const nrm_b = @sqrt(@max(try nccReduce(e, st.spec_b, n, cw, ch, 1), 0));
    lf.planes(g, st.spec_a, false, 1);
    lf.planes(g, st.spec_b, false, 1);
    nrun(e, "cmul", Gpu.flat((n * n + 255) / 256)[0], Gpu.flat((n * n + 255) / 256)[1], .{ .src_w = wa, .src_h = ha, .n = n, .cw = cw, .ch = ch, .mean = 0, .mode = 0 }, &.{ st.spec_b.at(2), st.spec_a.at(3) });
    lf.planes(g, st.spec_a, true, 1);
    const nn = @as(usize, n) * n;
    const raw = try g.gpa.alloc(f32, nn * 2);
    defer g.gpa.free(raw);
    g.read(st.spec_a, 0, std.mem.sliceAsBytes(raw));
    try g.wait();
    const ncc = try g.gpa.alloc(f32, nn);
    defer g.gpa.free(ncc);
    const den = @max(nrm_a * nrm_b, 1e-12);
    for (0..nn) |i| ncc[i] = @floatCast(@min(1.5, @max(-1.5, @as(f64, raw[i * 2]) / den)));
    var pk: usize = 0;
    for (1..nn) |i| if (ncc[i] > ncc[pk]) {
        pk = i;
    };
    if (map) |m| {
        if (m.len < nn) return error.MapTooSmall;
        @memcpy(m[0..nn], ncc);
    }
    const lag = Engine.peakLag(@intCast(pk), n, n);
    return .{
        .n = n, .n_y = n, .cw = cw, .ch = ch, .canvas_w = wa, .canvas_h = ha, .ox = c.ox, .oy = c.oy, .peak_index = @intCast(pk),
        .dx = @as(f64, @floatFromInt(lag[0])) * (@as(f64, @floatFromInt(wa)) / @as(f64, @floatFromInt(cw))),
        .dy = @as(f64, @floatFromInt(lag[1])) * (@as(f64, @floatFromInt(ha)) / @as(f64, @floatFromInt(ch))),
        .peak = ncc[pk], .zero = ncc[0],
    };
}

/// Zero-lag NCC of the pair at H over the union canvas (engine.js overlapScore "NCC").
pub fn nccScore(e: *Engine, H: Mat3) !f64 {
    try e.needImages();
    const c = assemble(e, H, false);
    assembledF32(e, c);
    const st = &e.ov;
    if (st.part.id == 0) st.part = e.g.storage(64 * 5 * 4);
    const n = c.ow * c.oh;
    nrun(e, "moments", 64, 1, .{ .src_w = n, .src_h = 1, .n = 1, .cw = 1, .ch = 1, .mean = 0, .mode = 0 }, &.{ st.a32.at(1), st.b32.at(2), st.part.at(3) });
    var p: [64 * 5]f32 = undefined;
    e.g.read(st.part, 0, std.mem.sliceAsBytes(&p));
    try e.g.wait();
    var sa: f64 = 0;
    var sb: f64 = 0;
    var saa: f64 = 0;
    var sbb: f64 = 0;
    var sab: f64 = 0;
    for (0..64) |i| {
        sa += p[i * 5];
        sb += p[i * 5 + 1];
        saa += p[i * 5 + 2];
        sbb += p[i * 5 + 3];
        sab += p[i * 5 + 4];
    }
    const nf: f64 = @floatFromInt(n);
    const cov = sab - sa * sb / nf;
    const va = @max(saa - sa * sa / nf, 0);
    const vb = @max(sbb - sb * sb / nf, 0);
    return @min(1.5, @max(-1.5, cov / @sqrt(@max(va * vb, 1e-24))));
}

// ── tile heat (smi.js tileExchange) ──
pub const TILE_MAX = 16;

/// Exchange-symmetric tile scores at H on a G×G grid: per tile the score in the fixed frame
/// (dest) and the moving frame (src); score = ½ (mean dest + mean src).
pub fn tileHeat(e: *Engine, H: Mat3, G0: u32, min_n: u32, dest: []f32, src: []f32) !f64 {
    try e.needImages();
    const G = std.math.clamp(G0, 2, TILE_MAX);
    const nT = G * G;
    if (dest.len < nT or src.len < nT) return error.BufferTooSmall;
    const g = &e.g;
    const st = &e.ov;
    g.ensure(&st.tile_ns, 2 * TILE_MAX * TILE_MAX * 4);
    var h32: [9]f32 = undefined;
    for (0..9) |i| h32[i] = @floatCast(H[i]);
    const fa = &e.feats[0];
    const fb = &e.feats[1];
    e.smi.run("tile_moments", G, G, 2, &.{
        e.smi.un(.{ .w = fb.w, .h = fb.h, .cw = fa.w, .ch = fa.h, .gx = G, .min_n = min_n, .n_keys = 1 }),
        fa.feat.at(1), fb.feat.at(2), g.stage(3, std.mem.sliceAsBytes(&h32)), st.tile_ns.at(4),
    });
    g.read(st.tile_ns, 0, std.mem.sliceAsBytes(dest[0..nT]));
    g.read(st.tile_ns, nT * 4, std.mem.sliceAsBytes(src[0..nT]));
    try g.wait();
    var sd: f64 = 0;
    var ss: f64 = 0;
    for (0..nT) |i| {
        sd += dest[i];
        ss += src[i];
    }
    const nf: f64 = @floatFromInt(nT);
    const score = 0.5 * (sd / nf + ss / nf);
    return if (std.math.isFinite(score)) score else 0;
}
