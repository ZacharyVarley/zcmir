//! The engine's C ABI, shared by the native library (Python, ctypes) and the browser module
//! (web/zcmir.js). Root-specific functions (creation, allocation) live in native_api.zig and
//! web_api.zig. Every call returns 0 (or a size) on success and a negative code on failure;
//! zc_error says why. Calls that wait for the GPU block natively and suspend in the browser.
const std = @import("std");
const engine = @import("engine.zig");
const lie = @import("lie.zig");
const pose_mod = @import("pose.zig");
const gpu_mod = @import("gpu/gpu.zig");

const Engine = engine.Engine;

fn failed(e: *Engine, what: []const u8, err: anyerror) i32 {
    e.fail("{s}: {s}", .{ what, @errorName(err) });
    return -1;
}

fn optSlice(comptime T: type, p: ?[*]T, len: usize) ?[]T {
    return if (p) |q| q[0..len] else null;
}

export fn zc_adapter(e: *Engine, buf: [*]u8, cap: usize) usize {
    const s = e.dev.info();
    const n = @min(s.len, cap);
    @memcpy(buf[0..n], s[0..n]);
    return n;
}

/// The last error of this engine (or the GPU) into buf; returns its length.
export fn zc_error(e: *Engine, buf: [*]u8, cap: usize) usize {
    return e.takeError(buf[0..cap]);
}

/// Apply settings overrides (a JSON object; see settings.zig).
export fn zc_configure(e: *Engine, json: [*]const u8, len: usize) i32 {
    e.configure(json[0..len]) catch |err| return failed(e, "configure", err);
    return 0;
}

/// The current settings as JSON into buf; returns the full length (retry with a larger buf if
/// it exceeds cap).
export fn zc_settings(e: *Engine, buf: [*]u8, cap: usize) usize {
    const j = e.set.toJson(e.gpa) catch return 0;
    defer e.gpa.free(j);
    @memcpy(buf[0..@min(cap, j.len)], j[0..@min(cap, j.len)]);
    return j.len;
}

/// Set the moving (side 0) or fixed (side 1) image as a preprocessed work image: w×h f32 gray.
export fn zc_set_image(e: *Engine, side: u32, data: [*]const f32, w: u32, h: u32) i32 {
    e.setImage(side, data[0 .. @as(usize, w) * h], w, h) catch |err| return failed(e, "set_image", err);
    return 0;
}

/// Set an image from w×h f32 gray in [0, 1]; preprocess (CLAHE / band / invert per settings) if
/// asked.
export fn zc_set_image_gray(e: *Engine, side: u32, data: [*]const f32, w: u32, h: u32, preprocess: u32) i32 {
    e.setImageGray(side, data[0 .. @as(usize, w) * h], w, h, preprocess != 0) catch |err| return failed(e, "set_image_gray", err);
    return 0;
}

/// Set an image from w×h RGBA8 (preprocessed per the settings, as the app loads images).
export fn zc_set_image_rgba(e: *Engine, side: u32, data: [*]const u8, w: u32, h: u32) i32 {
    e.setImageRgba(side, data[0 .. @as(usize, w) * h * 4], w, h) catch |err| return failed(e, "set_image_rgba", err);
    return 0;
}

// ── FFT benchmark and check (the 2-D FFT the maps and POS-GIFT use) ──
const fft_mod = @import("fft.zig");

fn benchFft(e: *Engine, n: u32, four_step: bool) !fft_mod.LineFft {
    fft_mod.force_four_step = four_step;
    defer fft_mod.force_four_step = false;
    return fft_mod.LineFft.init(&e.g, n);
}

/// One 2-D FFT (inverse: scaled by 1/n²) of `planes` n×n complex planes in place: data holds
/// planes·n·n interleaved (re, im) f32. four_step forces the long-line path.
export fn zc_fft2(e: *Engine, data: [*]f32, n: u32, planes: u32, inverse: u32, four_step: u32) i32 {
    const g = &e.g;
    var f = benchFft(e, n, four_step != 0) catch |err| return failed(e, "fft2", err);
    defer f.deinit(g);
    const len = @as(usize, planes) * n * n * 2;
    var b = g.storage(len * 4);
    defer g.release(&b);
    g.writeSlice(b, 0, f32, data[0..len]);
    f.planes(g, b, inverse != 0, planes);
    g.read(b, 0, std.mem.sliceAsBytes(data[0..len]));
    g.wait() catch |err| return failed(e, "fft2", err);
    return 0;
}

/// Tuning: lines per workgroup of the batched FFT for rows / columns (0: default), for FFTs
/// created from now on.
export fn zc_fft_tune(rows: u32, cols: u32) void {
    fft_mod.BatchFft.tune = .{ rows, cols };
}

/// Benchmarks: workgroup sizes for the batched kernel's rows / columns (0: default), and the
/// balanced (1) or greedy (0) factorization.
export fn zc_fft_tune_wg(rows: u32, cols: u32, balanced: u32) void {
    fft_mod.BatchFft.tune_wg = .{ rows, cols };
    fft_mod.balanced = balanced != 0;
}

/// `iters` passes of the pose-moments kernel (Refine's and the sweep screen's score) over
/// n_poses poses spread about the current pose, recorded in one batch and waited for once;
/// time it with zc_profile.
/// GPU traffic since the last call: dispatches, submits, reads, waits (round trips).
export fn zc_gpu_counts(e: *Engine, out: *[4]u32) void {
    const c = e.g.countsTake();
    out.* = .{ c.dispatches, c.submits, c.reads, c.waits };
}

/// Benchmarks: pixels per thread and the least workgroups of the per-pixel sum kernels (0: keep).
export fn zc_tune_parts(px_per_thread: u32, fill_wg: u32, tg_px: u32, tg_cap: u32) void {
    if (tg_px > 0) pose_mod.tg_px = tg_px;
    if (tg_cap > 0) pose_mod.tg_cap = tg_cap;
    if (px_per_thread > 0) pose_mod.px_per_thread = px_per_thread;
    if (fill_wg > 0) pose_mod.fill_wg = fill_wg;
}

export fn zc_bench_moments(e: *Engine, n_poses: u32, iters: u32) i32 {
    e.needImages() catch |err| return failed(e, "bench_moments", err);
    const gpa = e.gpa;
    const Hs = gpa.alloc(lie.Mat3, @max(1, n_poses)) catch |err| return failed(e, "bench_moments", err);
    defer gpa.free(Hs);
    for (Hs, 0..) |*H, i| {
        const t = @as(f64, @floatFromInt(i)) * 0.37;
        H.* = lie.mul3(.{ @cos(0.01 * @sin(t)), -@sin(0.01 * @sin(t)), 3 * @cos(t), @sin(0.01 * @sin(t)), @cos(0.01 * @sin(t)), 3 * @sin(t), 0, 0, 1 }, e.H);
    }
    e.pr.pg.benchMoments(e.pr.a, e.pr.b, Hs, e.pr.fwdCfg(), iters) catch |err| return failed(e, "bench_moments", err);
    return 0;
}

/// `iters` forward + inverse 2-D FFTs of `planes` n×n complex planes, recorded in one batch
/// and waited for once (after one untimed warm-up); the wall time in ms into out_ms.
export fn zc_bench_fft2(e: *Engine, n: u32, planes: u32, iters: u32, four_step: u32, out_ms: *f64) i32 {
    const g = &e.g;
    // four_step 2: the sweep's batched kernel (lpw neighbouring lines per workgroup)
    // 3 / 4: the batched kernel's rows / columns alone (forward + inverse)
    var bf: ?fft_mod.BatchFft = if (four_step >= 2) (fft_mod.BatchFft.init(g, n) catch |err| return failed(e, "bench_fft2", err)) else null;
    defer if (bf) |*x| x.deinit(g);
    var f = benchFft(e, n, four_step == 1) catch |err| return failed(e, "bench_fft2", err);
    defer f.deinit(g);
    var b = g.storage(@as(u64, planes) * n * n * 8);
    defer g.release(&b);
    g.clear(b);
    const mode = four_step;
    // 5: batched rows + four-step columns batched over neighbouring columns; 6: those columns alone
    var big: ?fft_mod.Big = if (mode >= 5) (fft_mod.Big.init(g, n) catch |err| return failed(e, "bench_fft2", err)) else null;
    defer if (big) |*x| x.deinit(g);
    if (big != null) g.ensure(&g.fft_scratch, @as(u64, planes) * n * n * 8);
    const Run = struct {
        var m: u32 = 0;
        var bg: ?*fft_mod.Big = null;
        fn pair(ff: *fft_mod.LineFft, bb: *?fft_mod.BatchFft, gg: *@TypeOf(e.g), buf: @TypeOf(b), np: u32) void {
            if (bg) |x| {
                for ([_]bool{ false, true }) |inv| {
                    if (m == 5) bb.*.?.axis(gg, buf, inv, np, 0);
                    var q: u32 = 0;
                    while (q < np) : (q += 1) x.cols(gg, buf, gg.fft_scratch, inv, q * x.n1 * x.n2 * x.n1 * x.n2);
                }
                return;
            }
            if (bb.*) |*x| {
                if (m >= 3) {
                    x.axis(gg, buf, false, np, m - 3);
                    x.axis(gg, buf, true, np, m - 3);
                    return;
                }
                x.run(gg, buf, false, np);
                x.run(gg, buf, true, np);
            } else {
                ff.planes(gg, buf, false, np);
                ff.planes(gg, buf, true, np);
            }
        }
    };
    Run.m = mode;
    Run.bg = if (big) |*x| x else null;
    Run.pair(&f, &bf, g, b, planes);
    var one: [4]u8 = undefined;
    g.read(b, 0, &one);
    g.wait() catch |err| return failed(e, "bench_fft2", err);
    const t0 = nowMs();
    for (0..iters) |_| Run.pair(&f, &bf, g, b, planes);
    g.read(b, 0, &one);
    g.wait() catch |err| return failed(e, "bench_fft2", err);
    out_ms.* = nowMs() - t0;
    return 0;
}

fn nowMs() f64 {
    if (@import("builtin").os.tag == .windows) {
        const k = struct {
            extern "kernel32" fn QueryPerformanceCounter(*i64) callconv(.winapi) i32;
            extern "kernel32" fn QueryPerformanceFrequency(*i64) callconv(.winapi) i32;
        };
        var c: i64 = 0;
        var fq: i64 = 1;
        _ = k.QueryPerformanceCounter(&c);
        _ = k.QueryPerformanceFrequency(&fq);
        return @as(f64, @floatFromInt(c)) * 1000.0 / @as(f64, @floatFromInt(fq));
    }
    return 0; // timed by the caller elsewhere
}

/// Time every GPU dispatch (native only): 1 on, 0 off; returns 0 when unavailable.
export fn zc_profile(e: *Engine, on: u32) i32 {
    return @intFromBool(e.g.setProfile(on != 0));
}

/// GPU time per pipeline since the last call ("ms  count  pipeline" lines, slowest first).
export fn zc_profile_report(e: *Engine, buf: [*]u8, cap: usize) usize {
    return e.g.profileReport(buf[0..cap]);
}

/// Ask the running climb or search to stop (it keeps its best so far). Callable while an
/// operation is suspended on the GPU.
export fn zc_cancel(e: *Engine) void {
    e.cancel = true;
}

/// Exchange the moving and fixed images (and their features and GLS-MIFT keypoints).
export fn zc_swap(e: *Engine) void {
    e.swap();
}

/// Descriptor inner product of each moving keypoint with its match (−1e9: none) into out;
/// returns the count (after match).
export fn zc_match_scores(e: *Engine, out: [*]f32, cap: usize) i32 {
    const n = e.matchScores(out[0..cap]) catch |err| return failed(e, "match_scores", err);
    return @intCast(n);
}

/// Size of side 0 (moving) or 1 (fixed): out = [w, h] (0, 0 when not set).
export fn zc_image_size(e: *Engine, side: u32, out: *[2]u32) void {
    out.* = if (side < 2 and e.sides[side].has) .{ e.sides[side].w, e.sides[side].h } else .{ 0, 0 };
}

/// The work image of a side (after preprocessing) as f32 into out (w·h floats).
/// Build a compute pipeline from WGSL source (tests/check_shaders.py): 0 when it builds, else
/// the length of the compiler's message written to msg (at least 1). Native only (-1 on the web).
export fn zc_compile_wgsl(e: *Engine, code: [*]const u8, code_len: usize, entry: [*]const u8, entry_len: usize, msg: [*]u8, cap: usize) i32 {
    if (comptime gpu_mod.is_web) {
        return -1;
    } else {
        _ = e.g.dev.createPipeline(code[0..code_len], entry[0..entry_len]) catch {
            const m = e.g.dev.failMessage();
            const src = if (m.len > 0) m else "failed";
            const n = @min(src.len, cap);
            @memcpy(msg[0..n], src[0..n]);
            _ = e.g.dev.takeError();
            return @intCast(@max(n, 1));
        };
        return 0;
    }
}

// ── inspection (the tutorials) ──

/// The 4 SMI feature planes of side 0 (moving) or 1 (fixed), plane-major (4 × h × w floats).
export fn zc_features(e: *Engine, side: u32, out: [*]f32, len: usize) i32 {
    if (side > 1 or !e.sides[side].has) return failed(e, "features", error.NoImage);
    e.needImages() catch |err| return failed(e, "features", err);
    const f = e.feats[side];
    const n = 4 * @as(usize, f.w) * f.h;
    if (len < n) return failed(e, "features", error.BufferTooSmall);
    e.g.read(f.feat, 0, std.mem.sliceAsBytes(out[0..n]));
    e.g.wait() catch |err| return failed(e, "features", err);
    return 0;
}

/// The 45 overlap moment sums of the moving image onto the fixed at H (moments.zig layout).
export fn zc_moments(e: *Engine, H: *const [9]f64, out: *[45]f64) i32 {
    e.needImages() catch |err| return failed(e, "moments", err);
    e.pr.pg.momentSums(e.pr.a, e.pr.b, H.*, e.pr.fwdCfg(), out) catch |err| return failed(e, "moments", err);
    return 0;
}

/// POS-GIFT's descriptor structure after detect: orientations, directions, rings, descriptor
/// length, first ring radius (pixels).
export fn zc_pg_info(e: *Engine, out: *[5]f64) i32 {
    const pg = &(e.pg orelse return failed(e, "pg_info", error.DetectFirst));
    const S = pg.S;
    out.* = .{ @floatFromInt(S.no), @floatFromInt(S.ndir), @floatFromInt(S.nring), @floatFromInt(S.dp), pg.set.p1 };
    return 0;
}

/// POS-GIFT's oriented phase-congruency energies of a side at full resolution (Σ|EO| per
/// orientation, n_orient × h × w), after detect.
export fn zc_pg_maps(e: *Engine, side: u32, out: [*]f32, len: usize) i32 {
    if (side > 1 or e.pg_det == null) return failed(e, "pg_maps", error.DetectFirst);
    const pg = &e.pg.?;
    const sd = e.sides[side];
    const n = @as(usize, pg.S.no) * sd.w * sd.h;
    if (len < n) return failed(e, "pg_maps", error.BufferTooSmall);
    e.g.read(pg.sides[side].cs0, 0, std.mem.sliceAsBytes(out[0..n]));
    e.g.wait() catch |err| return failed(e, "pg_maps", err);
    return 0;
}

/// POS-GIFT's descriptors of a side's keypoints after detect (keypoints() order, the
/// descriptor length per keypoint); returns the keypoint count.
export fn zc_pg_descriptors(e: *Engine, side: u32, out: [*]f32, len: usize) i32 {
    if (side > 1 or e.pg_det == null) return failed(e, "pg_descriptors", error.DetectFirst);
    const pg = &e.pg.?;
    const d = &e.pg_det.?;
    const nk = if (side == 0) d.dl.n else d.dr.n;
    const n = @as(usize, nk) * pg.S.dp;
    if (len < n) return failed(e, "pg_descriptors", error.BufferTooSmall);
    if (n > 0) {
        e.g.read(pg.sides[side].des, 0, std.mem.sliceAsBytes(out[0..n]));
        e.g.wait() catch |err| return failed(e, "pg_descriptors", err);
    }
    return @intCast(nk);
}

export fn zc_work_image(e: *Engine, side: u32, out: [*]f32, len: usize) i32 {
    if (side > 1 or !e.sides[side].has) return failed(e, "work_image", error.NoImage);
    e.refreshImages() catch |err| return failed(e, "work_image", err);
    const sd = e.sides[side];
    const n = @as(usize, sd.w) * sd.h;
    if (len < n) return failed(e, "work_image", error.BufferTooSmall);
    e.g.read(sd.work32, 0, std.mem.sliceAsBytes(out[0..n]));
    e.g.wait() catch |err| return failed(e, "work_image", err);
    return 0;
}

/// Detect keypoints on both images (settings.detector).
export fn zc_detect(e: *Engine, out: *engine.DetectResult) i32 {
    out.* = e.detect() catch |err| return failed(e, "detect", err);
    return 0;
}

/// Match and fit; the fitted pose becomes the engine's pose.
export fn zc_match(e: *Engine, out: *engine.MatchResult) i32 {
    out.* = e.match() catch |err| return failed(e, "match", err);
    return 0;
}

/// Global search (sweep or cloud per settings); the pose becomes the engine's when it improves.
export fn zc_search(e: *Engine, out: *@import("search.zig").Result) i32 {
    out.* = e.search() catch |err| return failed(e, "search", err);
    return 0;
}

/// Detect → match → climb.
export fn zc_auto(e: *Engine, out: *@import("climb.zig").Result) i32 {
    out.* = e.auto() catch |err| return failed(e, "auto", err);
    return 0;
}

/// Keypoints of a side, 4 floats each (x, y, score, 0; 1-based pixels); returns the count.
export fn zc_keypoints(e: *Engine, side: u32, out: [*]f32, cap_points: usize) i32 {
    const kp: [*]@import("gls.zig").Kp = @ptrCast(@alignCast(out));
    const n = e.keypoints(side, kp[0..cap_points]) catch |err| return failed(e, "keypoints", err);
    return @intCast(n);
}

/// Descriptor frame of each keypoint of a side, in zc_keypoints order: the direction it starts
/// from (radians; x right, y down) and the radius it reads (pixels). Returns the keypoint count.
export fn zc_keypoint_frames(e: *Engine, side: u32, out: [*]f32, cap_points: usize) i32 {
    const fr: [*]@import("gls.zig").KpFrame = @ptrCast(@alignCast(out));
    const n = e.keypointFrames(side, fr[0..cap_points]) catch |err| return failed(e, "keypoint_frames", err);
    return @intCast(n);
}

/// The sanity check of a pose fitted to matches (match.zig Verdict): 0 passes, else why not.
export fn zc_fit_check(e: *Engine, H: *const [9]f64) u32 {
    return @intFromEnum(e.fitCheck(H.*));
}

/// Match of each moving keypoint (fixed keypoint index, 0xffffffff none); returns the count.
export fn zc_matches(e: *Engine, out: [*]u32, cap: usize) i32 {
    const n = e.matches(out[0..cap]) catch |err| return failed(e, "matches", err);
    return @intCast(n);
}

/// Spline control points (2·g·g floats, g = ffd_grid).
export fn zc_set_ffd(e: *Engine, cps: [*]const f32, n: usize) i32 {
    e.setFfdCps(cps[0..n]) catch |err| return failed(e, "set_ffd", err);
    return 0;
}

/// The engine's pose (moving → fixed, 1-based, row-major).
export fn zc_set_pose(e: *Engine, H: *const [9]f64) void {
    e.setPose(H.*);
}

export fn zc_get_pose(e: *Engine, H: *[9]f64) void {
    H.* = e.H;
}

/// Spline control points of the engine (2·g·g); returns the count.
export fn zc_get_ffd(e: *Engine, cps: [*]f32, cap: usize) usize {
    const n = e.ffd.n();
    @memcpy(cps[0..@min(n, cap)], e.ffd.cps[0..@min(n, cap)]);
    return n;
}

/// Climb from the engine's pose with the current settings; the result pose is also the
/// engine's new pose. Progress events: zc_events (native) or the page's event import (web).
export fn zc_climb(e: *Engine, out: *@import("climb.zig").Result) i32 {
    out.* = e.climb() catch |err| return failed(e, "climb", err);
    return 0;
}

/// Queued progress events (native): newline-separated JSON objects, whole lines up to cap
/// bytes; returns the bytes written (0 when none are queued).
export fn zc_events(e: *Engine, buf: [*]u8, cap: usize) usize {
    return e.events.take(buf[0..cap]);
}

/// Pose score at H: out = [mean, forward, inverse] (inverse NaN when not symmetric).
export fn zc_score(e: *Engine, H: *const [9]f64, out: *[3]f64) i32 {
    const p = e.score(H.*) catch |err| return failed(e, "score", err);
    out.* = .{ p.mean, p.fwd, p.inv orelse std.math.nan(f64) };
    return 0;
}

/// Score and tangent gradient at H (hess: also the Gauss–Newton matrix).
export fn zc_gradient(e: *Engine, H: *const [9]f64, hess: u32, out: *engine.GradResult) i32 {
    out.* = e.gradient(H.*, hess != 0) catch |err| return failed(e, "gradient", err);
    return 0;
}

/// Score and gradient over the spline control points at H: grad (2·g·g floats, g = ffd_grid).
export fn zc_ffd_gradient(e: *Engine, H: *const [9]f64, score: *f64, grad: [*]f64, len: usize) i32 {
    score.* = e.ffdGradient(H.*, grad[0..len]) catch |err| return failed(e, "ffd_gradient", err);
    return 0;
}

/// Dense shift map at H with explicit flags (1 exact, 2 calibrated), no overlap floor.
export fn zc_shift_map(e: *Engine, H: *const [9]f64, flags: u32, cap: u32, out: *engine.ShiftResult, map: ?[*]f32, map_len: usize) i32 {
    out.* = e.shiftMap(H.*, flags, cap, optSlice(f32, map, map_len)) catch |err| return failed(e, "shift_map", err);
    return 0;
}

/// Dense shift map at H with the current settings (score, resolution, overlap floor).
export fn zc_shift_map_set(e: *Engine, H: *const [9]f64, out: *engine.ShiftResult, map: ?[*]f32, map_len: usize) i32 {
    out.* = e.shiftMapSettings(H.*, optSlice(f32, map, map_len)) catch |err| return failed(e, "shift_map", err);
    return 0;
}

/// Roto-scale map at H about c (null: the moving centroid), symmetric or forward only.
/// Optional n×n outputs (len floats each): the finished map, the raw forward and inverse maps.
export fn zc_rs_map(e: *Engine, H: *const [9]f64, c: ?*const [2]f64, symmetric: u32, out: *engine.RsResult, map: ?[*]f32, fwd: ?[*]f32, inv: ?[*]f32, len: usize) i32 {
    out.* = e.rsMap(H.*, if (c) |p| p.* else null, symmetric != 0, optSlice(f32, map, len), optSlice(f32, fwd, len), optSlice(f32, inv, len)) catch |err| return failed(e, "rs_map", err);
    return 0;
}

const overlay = @import("overlay.zig");

/// The union canvas of the pose H (inverse: of H⁻¹, fixed onto moving): offset, size and the
/// canvas pose. Host arithmetic only.
export fn zc_canvas(e: *Engine, H: *const [9]f64, inverse: u32, out: *overlay.Canvas) i32 {
    if (!e.hasImages()) return failed(e, "canvas", error.NoImages);
    out.* = overlay.canvas(e, H.*, inverse != 0);
    return 0;
}

/// The overlay at H: moving and fixed images on the union canvas (zc_canvas size), f32 in [0, 1].
export fn zc_overlay(e: *Engine, H: *const [9]f64, inverse: u32, out: *overlay.Canvas, moving: [*]f32, fixed: [*]f32, len: usize) i32 {
    out.* = overlay.overlay(e, H.*, inverse != 0, moving[0..len], fixed[0..len]) catch |err| return failed(e, "overlay", err);
    return 0;
}

/// Exchange-symmetric tile scores (G×G, fixed and moving frames) at H; returns the score in out.
export fn zc_tile_heat(e: *Engine, H: *const [9]f64, grid: u32, min_n: u32, out: *f64, dest: [*]f32, src: [*]f32, len: usize) i32 {
    out.* = overlay.tileHeat(e, H.*, grid, min_n, dest[0..len], src[0..len]) catch |err| return failed(e, "tile_heat", err);
    return 0;
}

const export_warp = @import("export.zig");

/// The warp export (a stored ZIP: {stem}_Warp.nrrd and, for a spline or an affine pose,
/// {stem}.tfm). Returns its size; the bytes are written when cap is large enough.
export fn zc_export_warp(e: *Engine, stem: [*]const u8, stem_len: usize, out: ?[*]u8, cap: usize) i64 {
    const o = e.warpOpts() catch |err| return failed(e, "export_warp", err);
    const zip = export_warp.bundle(e.gpa, &o, stem[0..stem_len]) catch |err| return failed(e, "export_warp", err);
    defer e.gpa.free(zip);
    if (out) |p| if (cap >= zip.len) @memcpy(p[0..zip.len], zip);
    return @intCast(zip.len);
}

/// The displacement field of the pose (fixed w × h × 2, x_moving − x_fixed, 0-based pixels).
export fn zc_displacement(e: *Engine, out: [*]f32, len: usize) i32 {
    const o = e.warpOpts() catch |err| return failed(e, "displacement", err);
    const n = @as(usize, o.rw) * o.rh * 2;
    if (len < n) return failed(e, "displacement", error.BufferTooSmall);
    export_warp.displacementField(&o, out[0..n]);
    return 0;
}

export fn zc_result_sizes(out: *[4]u32) void {
    out.* = .{ @sizeOf(engine.ShiftResult), @sizeOf(engine.RsResult), @sizeOf(engine.GradResult), 0 };
}
