//! Mixed-radix Stockham line FFT: one workgroup per line,
//! the whole line in workgroup memory, radices 13, 11, 8, 7, 5, 4, 3, 2. The WGSL is generated
//! per length and direction, with each radix's DFT unrolled and its constant twiddles folded.
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;

const EMS = [_]u32{ 2, 3, 5, 7, 11, 13 };
const RADICES = [_]u32{ 13, 11, 8, 7, 5, 4, 3, 2 };
const TWE = 1e-12;
const GRID_X: u32 = 32768;

/// Which FFT lengths plans use. ladder: 2^a·3^b only (about five per octave, at most 4/3 apart),
/// so a few dozen shaders cover every image; compact: any length whose prime factors are ≤ 13,
/// the least padding but a shader per length.
pub const Sizes = enum { ladder, compact };

/// Smallest length ≥ target in `sizes` (and divisible by div).
pub fn planSize(target: u32, div: u32, sizes: Sizes) u32 {
    var n = @max(target, 1);
    while (true) : (n += 1) {
        if (n % div == 0 and smooth(n, sizes)) return n;
    }
}

fn smooth(n0: u32, sizes: Sizes) bool {
    var n = n0;
    const primes: []const u32 = if (sizes == .ladder) EMS[0..2] else &EMS;
    for (primes) |p| while (n % p == 0) {
        n /= p;
    };
    return n == 1;
}

/// The largest length ≤ maxN (≥ 32) in `sizes`.
pub fn largestSmooth(maxN: u32, sizes: Sizes) u32 {
    var n = @max(32, maxN);
    while (n >= 32) : (n -= 1) {
        if (smooth(n, sizes)) return n;
    }
    return 32;
}

/// Radices of N in RADICES order; error if N has a factor above 13.
pub fn factorize(N: u32, out: ?*std.ArrayList(u32)) !void {
    var n = N;
    var twos: u32 = 0; // radix-2 stages so far (balanced: an 8 and a 2 become 4 · 4)
    for (RADICES) |R| while (n % R == 0) {
        if (out) |o| o.appendAssumeCapacity(R);
        if (R == 2) twos += 1;
        n /= R;
    };
    if (n != 1) return error.NotFactorizable;
    if (out) |o| if (balanced and twos == 1) {
        // a lone radix-2 stage moves the whole line through workgroup memory for one
        // butterfly per pair; 8 · 2 = 4 · 4 does the same in two equal radix-4 stages
        if (std.mem.indexOfScalar(u32, o.items, 8)) |k8| if (std.mem.indexOfScalar(u32, o.items, 2)) |k2| {
            o.items[k8] = 4;
            o.items[k2] = 4;
        };
    };
}

/// Tuning: balanced factorization (see factorize).
pub var balanced = true;

/// (n, cw, ch): FFT length and the correlation grid for linear (non-wrapping) correlation of a
/// wa×ha against a wb×hb image, as fft.js linearCorrPlan.
pub fn linearCorrPlan(wa: u32, ha: u32, wb: u32, hb: u32, maxN0: u32, sizes: Sizes) [3]u32 {
    const mw = @max(wa, wb);
    const mh = @max(ha, hb);
    const maxN = largestSmooth(maxN0, sizes);
    var s: f64 = @min(1.0, @as(f64, @floatFromInt(@max(8, maxN >> 1))) / @as(f64, @floatFromInt(@max(@max(mw, mh), 1))));
    var n: u32 = 8;
    var cw: u32 = 8;
    var ch: u32 = 8;
    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        cw = @max(8, jsRound(@as(f64, @floatFromInt(wa)) * s));
        ch = @max(8, jsRound(@as(f64, @floatFromInt(ha)) * s));
        const need = @max(@max(cw + cw - 1, ch + ch - 1), 32);
        n = planSize(need, 1, sizes);
        if (n > maxN) n = maxN;
        if (n >= cw + cw - 1 and n >= ch + ch - 1) return .{ n, cw, ch };
        s *= 0.85;
    }
    cw = @max(8, @min(cw, (n + 1) >> 1));
    ch = @max(8, @min(ch, (n + 1) >> 1));
    return .{ n, cw, ch };
}

/// (nx, ny, cw, ch): linearCorrPlan with each side's FFT length of its own (the correlation of a
/// w×h canvas with itself needs 2·cw − 1 by 2·ch − 1 cells; a long thin canvas no longer pays
/// for a square).
pub fn linearCorrPlanRect(w: u32, h: u32, maxN0: u32, sizes: Sizes) [4]u32 {
    const maxN = largestSmooth(maxN0, sizes);
    var s: f64 = @min(1.0, @as(f64, @floatFromInt(@max(8, maxN >> 1))) / @as(f64, @floatFromInt(@max(@max(w, h), 1))));
    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        const cw = @max(8, jsRound(@as(f64, @floatFromInt(w)) * s));
        const ch = @max(8, jsRound(@as(f64, @floatFromInt(h)) * s));
        const nx = @min(planSize(@max(cw + cw - 1, 32), 1, sizes), maxN);
        const ny = @min(planSize(@max(ch + ch - 1, 32), 1, sizes), maxN);
        if (nx >= cw + cw - 1 and ny >= ch + ch - 1) return .{ nx, ny, cw, ch };
        s *= 0.85;
    }
    return .{ maxN, maxN, @max(8, (maxN + 1) >> 1), @max(8, (maxN + 1) >> 1) };
}

/// Math.round (half up), for the grid sizes computed as in JS.
fn jsRound(x: f64) u32 {
    return @intFromFloat(@floor(x + 0.5));
}

/// Largest line length the shared-memory kernel fits (16 B per point), any factorizable length
/// (plans take their own largest length below it).
pub fn maxFftN(max_workgroup_storage: u32) u32 {
    return largestSmooth(@min(2048, max_workgroup_storage / 16), .compact);
}

fn pickWg(N: u32) u32 {
    for ([_]u32{ 256, 128, 64, 32 }) |wg| if (wg <= N) return wg;
    return 32;
}

// ── WGSL generation (same text as fft.js) ─────────────────────────────────────────────────
const W = std.Io.Writer;

const Term = struct { neg: bool, text: []const u8 };

fn coeffTerm(a: std.mem.Allocator, coeff: f64, v: []const u8) !?Term {
    if (@abs(coeff) < TWE) return null;
    if (@abs(coeff - 1) < TWE) return .{ .neg = false, .text = v };
    if (@abs(coeff + 1) < TWE) return .{ .neg = true, .text = v };
    return .{ .neg = coeff < 0, .text = try std.fmt.allocPrint(a, "({f})*{s}", .{ jsNum(@abs(coeff)), v }) };
}

fn sumTerms(a: std.mem.Allocator, terms: []const ?Term) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    var first = true;
    for (terms) |t0| {
        const t = t0 orelse continue;
        if (first) {
            if (t.neg) try out.writer.writeAll("-");
            first = false;
        } else try out.writer.writeAll(if (t.neg) "-" else "+");
        try out.writer.writeAll(t.text);
    }
    if (first) return "0.0";
    return out.written();
}

/// A number printed as JavaScript prints it (shortest round-trip), for identical shader text.
const JsNum = struct {
    v: f64,
    pub fn format(self: JsNum, w: *W) W.Error!void {
        try w.print("{d}", .{self.v});
    }
};
fn jsNum(v: f64) JsNum {
    return .{ .v = v };
}

fn radixDft(a: std.mem.Allocator, w: *W, R: u32, sign: f64) !void {
    const half = (R - 1) >> 1;
    const even = R % 2 == 0;
    const mid = R >> 1;
    var j: u32 = 1;
    while (j <= half) : (j += 1) {
        try w.print("let s{d}r=a{d}r+a{d}r; let s{d}i=a{d}i+a{d}i;\n        ", .{ j, j, R - j, j, j, R - j });
        try w.print("let d{d}r=a{d}r-a{d}r; let d{d}i=a{d}i-a{d}i;\n        ", .{ j, j, R - j, j, j, R - j });
    }
    // o0 = a0 + Σ s_j (+ a_mid)
    try w.writeAll("let o0r=a0r");
    j = 1;
    while (j <= half) : (j += 1) try w.print("+s{d}r", .{j});
    if (even) try w.print("+a{d}r", .{mid});
    try w.writeAll("; let o0i=a0i");
    j = 1;
    while (j <= half) : (j += 1) try w.print("+s{d}i", .{j});
    if (even) try w.print("+a{d}i", .{mid});
    try w.writeAll(";\n        ");
    var k: u32 = 1;
    while (k <= half) : (k += 1) {
        var rr: std.ArrayList(?Term) = .empty;
        var ri: std.ArrayList(?Term) = .empty;
        var qr: std.ArrayList(?Term) = .empty;
        var qi: std.ArrayList(?Term) = .empty;
        try rr.append(a, .{ .neg = false, .text = "a0r" });
        try ri.append(a, .{ .neg = false, .text = "a0i" });
        j = 1;
        while (j <= half) : (j += 1) {
            const ang = 2.0 * std.math.pi * @as(f64, @floatFromInt(k * j)) / @as(f64, @floatFromInt(R));
            const cc = @cos(ang);
            const ss = sign * @sin(ang);
            try rr.append(a, try coeffTerm(a, cc, try std.fmt.allocPrint(a, "s{d}r", .{j})));
            try ri.append(a, try coeffTerm(a, cc, try std.fmt.allocPrint(a, "s{d}i", .{j})));
            try qr.append(a, try coeffTerm(a, ss, try std.fmt.allocPrint(a, "d{d}r", .{j})));
            try qi.append(a, try coeffTerm(a, ss, try std.fmt.allocPrint(a, "d{d}i", .{j})));
        }
        if (even) {
            const sg: f64 = if (k % 2 == 0) 1 else -1;
            try rr.append(a, try coeffTerm(a, sg, try std.fmt.allocPrint(a, "a{d}r", .{mid})));
            try ri.append(a, try coeffTerm(a, sg, try std.fmt.allocPrint(a, "a{d}i", .{mid})));
        }
        try w.print("let rr{d}={s}; let ri{d}={s};\n        ", .{ k, try sumTerms(a, rr.items), k, try sumTerms(a, ri.items) });
        try w.print("let qr{d}={s}; let qi{d}={s};\n        ", .{ k, try sumTerms(a, qr.items), k, try sumTerms(a, qi.items) });
        try w.print("let o{d}r=rr{d}-qi{d}; let o{d}i=ri{d}+qr{d};\n        ", .{ k, k, k, k, k, k });
        try w.print("let o{d}r=rr{d}+qi{d}; let o{d}i=ri{d}-qr{d};", .{ R - k, k, k, R - k, k, k });
        if (k < half or even) try w.writeAll("\n        ");
    }
    if (even) {
        try w.print("let o{d}r=", .{mid});
        var b: u32 = 0;
        while (b < R) : (b += 1) {
            if (b > 0) try w.writeAll(if (b % 2 == 0) "+" else "-");
            try w.print("a{d}r", .{b});
        }
        try w.print("; let o{d}i=", .{mid});
        b = 0;
        while (b < R) : (b += 1) {
            if (b > 0) try w.writeAll(if (b % 2 == 0) "+" else "-");
            try w.print("a{d}i", .{b});
        }
        try w.writeAll(";");
    }
}

/// Stockham stages for one line (lpw > 1: lpw lines per workgroup); returns the name of the
/// buffer holding the result.
fn stockhamBody(a: std.mem.Allocator, w: *W, N: u32, radices: []const u32, wg: u32, sign: f64, lpw: u32) ![]const u8 {
    var Ns: u32 = 1;
    var src: []const u8 = "sa";
    var dst: []const u8 = "sb";
    const off: []const u8 = if (lpw > 1) "so+" else "";
    for (radices, 0..) |R, stage| {
        const Nr = N / R;
        const step = N / (Ns * R);
        if (lpw > 1) {
            try w.print("\n    // stage {d}: radix {d}, Ns={d}\n    for (var idx=tid; idx<{d}u; idx=idx+{d}u) {{\n        let ll=idx/{d}u; let j=idx%{d}u; let so=ll*{d}u;\n        ", .{ stage, R, Ns, lpw * Nr, wg, Nr, Nr, N });
        } else {
            try w.print("\n    // stage {d}: radix {d}, Ns={d}\n    for (var j=tid; j<{d}u; j=j+{d}u) {{\n        ", .{ stage, R, Ns, Nr, wg });
        }
        if (Ns != 1) try w.print("let base=j%{d}u; ", .{Ns});
        try w.print("let od=(j/{d}u)*{d}u{s};\n        ", .{ Ns, Ns * R, if (Ns == 1) "" else "+base" });
        var r: u32 = 0;
        while (r < R) : (r += 1) {
            if (r > 0) try w.writeAll("\n        ");
            if (r == 0 or Ns == 1) {
                try w.print("let a{d}v={s}[{s}j + {d}u]; let a{d}r=a{d}v.x; let a{d}i=a{d}v.y;", .{ r, src, off, r * Nr, r, r, r, r });
            } else {
                try w.print("let z{d}v={s}[{s}j + {d}u]; let z{d}r=z{d}v.x; let z{d}i=z{d}v.y;\n        ", .{ r, src, off, r * Nr, r, r, r, r });
                try w.print("let w{d}v=tw[base*{d}u]; let w{d}r=w{d}v.x; let w{d}i=w{d}v.y;\n        ", .{ r, r * step, r, r, r, r });
                try w.print("let a{d}r=z{d}r*w{d}r-z{d}i*w{d}i; let a{d}i=z{d}r*w{d}i+z{d}i*w{d}r;", .{ r, r, r, r, r, r, r, r, r, r });
            }
        }
        try w.writeAll("\n        ");
        try radixDft(a, w, R, sign);
        try w.writeAll("\n        ");
        r = 0;
        while (r < R) : (r += 1) {
            if (r > 0) try w.writeAll("\n        ");
            try w.print("{s}[{s}od+{d}u]=vec2<f32>(o{d}r, o{d}i);", .{ dst, off, r * Ns, r, r });
        }
        try w.writeAll("\n    }\n    workgroupBarrier();");
        Ns *= R;
        const t = src;
        src = dst;
        dst = t;
    }
    return src;
}

/// The line-FFT kernel for length N (inverse: +i sign and a 1/N scale).
pub fn lineShader(gpa: std.mem.Allocator, N: u32, radices: []const u32, wg: u32, inverse: bool) ![]u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const sign: f64 = if (inverse) 1 else -1;
    var body: std.Io.Writer.Allocating = .init(a);
    const final = try stockhamBody(a, &body.writer, N, radices, wg, sign, 1);
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    const w = &out.writer;
    const scale: f64 = if (inverse) 1.0 / @as(f64, @floatFromInt(N)) else 1.0;
    try w.print(
        \\
        \\struct U {{ n_lines: u32, grid_x: u32, axis: u32, base: u32 }};
        \\@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
        \\@group(0) @binding(1) var<storage, read>       tw:   array<vec2<f32>>;
        \\@group(0) @binding(2) var<uniform>             u:    U;
        \\const N:u32={d}u; const SC:f32={f}; const LPW:u32=1u;
        \\var<workgroup> sa: array<vec2<f32>, {d}>;
        \\var<workgroup> sb: array<vec2<f32>, {d}>;
        \\@compute @workgroup_size({d})
        \\fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
        \\    let block = wid.y * u.grid_x + wid.x;
        \\    let base = block * LPW;
        \\    let tid = lid.x;
        \\    for (var e=tid; e<LPW*N; e=e+{d}u) {{
        \\        let ll=e/N; let p=e%N; let line=base+ll;
        \\        var v = vec2<f32>(0.0, 0.0);
        \\        if (line<u.n_lines) {{
        \\            v = data[u.base / 2u + select(line * N + p, p * N + line, u.axis == 1u)];
        \\        }}
        \\        sa[ll*N+p]=v;
        \\    }}
        \\    workgroupBarrier();
        \\
    , .{ N, jsNum(scale), N, N, wg, wg });
    try w.writeAll(body.written());
    try w.print(
        \\
        \\    for (var e=tid; e<LPW*N; e=e+{d}u) {{
        \\        let ll=e/N; let p=e%N; let line=base+ll;
        \\        if (line<u.n_lines) {{
        \\            data[u.base / 2u + select(line * N + p, p * N + line, u.axis == 1u)]={s}[ll*N+p]*SC;
        \\        }}
        \\    }}
        \\}}
        \\
    , .{ wg, final });
    return out.toOwnedSlice();
}

/// A strided sub-FFT of length N for the four-step long-line FFT (LineFft.Big). Sub-line l
/// (hi = l / l1, lo = l % l1) is the N points src[i_base + hi·i_sl + lo·i_si + p·i_sp] (complex
/// units); output k is optionally multiplied by e^{∓2πi t·k / NB} (tw_on; t = lo, or hi with
/// tw_hi) and written to dst[o_base + hi·o_sl + lo·o_si + k·o_sp]. A workgroup takes LPW
/// consecutive sub-lines; with i_fast / o_fast (their lo stride is 1) consecutive invocations
/// walk the sub-lines, else the points, so memory is read and written in runs. Inverse: 1/N
/// per pass.
pub fn strideShader(gpa: std.mem.Allocator, N: u32, radices: []const u32, wg: u32, inverse: bool, NB: u32, lpw: u32) ![]u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const sign: f64 = if (inverse) 1 else -1;
    var body: std.Io.Writer.Allocating = .init(a);
    const final = try stockhamBody(a, &body.writer, N, radices, wg, sign, lpw);
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    const w = &out.writer;
    const scale: f64 = if (inverse) 1.0 / @as(f64, @floatFromInt(N)) else 1.0;
    try w.print(
        \\
        \\struct U {{ n_lines: u32, grid_x: u32, l1: u32, tw_on: u32,
        \\           i_base: u32, i_sl: u32, i_si: u32, i_sp: u32,
        \\           o_base: u32, o_sl: u32, o_si: u32, o_sp: u32,
        \\           tw_hi: u32, i_fast: u32, o_fast: u32, pad: u32 }};
        \\@group(0) @binding(0) var<storage, read>       src:  array<vec2<f32>>;
        \\@group(0) @binding(1) var<storage, read>       tw:   array<vec2<f32>>;
        \\@group(0) @binding(2) var<uniform>             u:    U;
        \\@group(0) @binding(3) var<storage, read_write> dst:  array<vec2<f32>>;
        \\@group(0) @binding(4) var<storage, read>       twb:  array<vec2<f32>>;
        \\const N:u32={d}u; const NB:u32={d}u; const SC:f32={f}; const LPW:u32={d}u;
        \\var<workgroup> sa: array<vec2<f32>, {d}>;
        \\var<workgroup> sb: array<vec2<f32>, {d}>;
        \\// element e of the workgroup's LPW·N points as (sub-line, point)
        \\fn split(e: u32, fast: u32) -> vec2<u32> {{
        \\    if (fast == 1u) {{ return vec2<u32>(e % LPW, e / LPW); }}
        \\    return vec2<u32>(e / N, e % N);
        \\}}
        \\@compute @workgroup_size({d})
        \\fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
        \\    let l0 = (wid.y * u.grid_x + wid.x) * LPW;
        \\    if (l0 >= u.n_lines) {{ return; }}
        \\    _ = tw[0]; // keeps the binding when a single radix leaves the twiddles unused
        \\    let tid = lid.x;
        \\    for (var e=tid; e<LPW*N; e=e+{d}u) {{
        \\        let s = split(e, u.i_fast);
        \\        let l = l0 + s.x;
        \\        var v = vec2<f32>(0.0, 0.0);
        \\        if (l < u.n_lines) {{
        \\            v = src[u.i_base + (l / u.l1) * u.i_sl + (l % u.l1) * u.i_si + s.y * u.i_sp];
        \\        }}
        \\        sa[s.x*N+s.y]=v;
        \\    }}
        \\    workgroupBarrier();
        \\
    , .{ N, NB, jsNum(scale), lpw, N * lpw, N * lpw, wg, wg });
    try w.writeAll(body.written());
    try w.print(
        \\
        \\    for (var e=tid; e<LPW*N; e=e+{d}u) {{
        \\        let s = split(e, u.o_fast);
        \\        let l = l0 + s.x;
        \\        if (l >= u.n_lines) {{ continue; }}
        \\        let hi = l / u.l1;
        \\        let lo = l % u.l1;
        \\        let v = {s}[s.x*N+s.y]*SC;
        \\        var re = v.x;
        \\        var im = v.y;
        \\        if (u.tw_on == 1u) {{
        \\            let wv = twb[(select(lo, hi, u.tw_hi == 1u) * s.y) % NB];
        \\            let wr = wv.x;
        \\            let wi = wv.y;
        \\            let r2 = re * wr - im * wi;
        \\            im = re * wi + im * wr;
        \\            re = r2;
        \\        }}
        \\        dst[u.o_base + hi * u.o_sl + lo * u.o_si + s.y * u.o_sp]=vec2<f32>(re, im);
        \\    }}
        \\}}
        \\
    , .{ wg, final });
    return out.toOwnedSlice();
}

/// Stockham stages in place in `sa` for lpw lines of N: each invocation reads its butterflies'
/// inputs and computes them into registers, the workgroup waits, then writes them back, so one
/// shared array suffices (half the workgroup memory of stockhamBody, twice the lines per
/// workgroup). The arithmetic is stockhamBody's, term for term.
fn stockhamInPlace(a: std.mem.Allocator, w: *W, N: u32, radices: []const u32, wg: u32, sign: f64, lpw: u32, NP: u32) !void {
    var Ns: u32 = 1;
    for (radices, 0..) |R, stage| {
        const Nr = N / R;
        const step = N / (Ns * R);
        const total = lpw * Nr;
        const K = (total + wg - 1) / wg;
        try w.print("\n    // stage {d}: radix {d}, Ns={d}, in place\n    {{\n", .{ stage, R, Ns });
        var it: u32 = 0;
        while (it < K) : (it += 1) {
            try w.print("    var q{d}: u32;", .{it});
            var r: u32 = 0;
            while (r < R) : (r += 1) try w.print(" var v{d}_{d}r: f32; var v{d}_{d}i: f32;", .{ it, r, it, r });
            try w.print("\n    let x{d} = tid + {d}u;\n    if (x{d} < {d}u) {{\n        let ll=x{d}/{d}u; let j=x{d}%{d}u; let so=ll*{d}u;\n        ", .{ it, it * wg, it, total, it, Nr, it, Nr, NP });
            if (Ns != 1) try w.print("let base=j%{d}u; ", .{Ns});
            try w.print("let od=(j/{d}u)*{d}u{s};\n        ", .{ Ns, Ns * R, if (Ns == 1) "" else "+base" });
            r = 0;
            while (r < R) : (r += 1) {
                if (r > 0) try w.writeAll("\n        ");
                if (r == 0 or Ns == 1) {
                    try w.print("let a{d}v=sa[so+j + {d}u]; let a{d}r=a{d}v.x; let a{d}i=a{d}v.y;", .{ r, r * Nr, r, r, r, r });
                } else {
                    try w.print("let z{d}v=sa[so+j + {d}u]; let z{d}r=z{d}v.x; let z{d}i=z{d}v.y;\n        ", .{ r, r * Nr, r, r, r, r });
                    try w.print("let w{d}v=tw[base*{d}u]; let w{d}r=w{d}v.x; let w{d}i=w{d}v.y;\n        ", .{ r, r * step, r, r, r, r });
                    try w.print("let a{d}r=z{d}r*w{d}r-z{d}i*w{d}i; let a{d}i=z{d}r*w{d}i+z{d}i*w{d}r;", .{ r, r, r, r, r, r, r, r, r, r });
                }
            }
            try w.writeAll("\n        ");
            try radixDft(a, w, R, sign);
            try w.print("\n        q{d}=so+od;", .{it});
            r = 0;
            while (r < R) : (r += 1) try w.print(" v{d}_{d}r=o{d}r; v{d}_{d}i=o{d}i;", .{ it, r, r, it, r, r });
            try w.writeAll("\n    }\n");
        }
        try w.writeAll("    workgroupBarrier();\n");
        it = 0;
        while (it < K) : (it += 1) {
            try w.print("    if (x{d} < {d}u) {{", .{ it, total });
            var r: u32 = 0;
            while (r < R) : (r += 1) try w.print(" sa[q{d}+{d}u]=vec2<f32>(v{d}_{d}r, v{d}_{d}i);", .{ it, r * Ns, it, r, it, r });
            try w.writeAll(" }\n");
        }
        try w.writeAll("    workgroupBarrier();\n    }");
        Ns *= R;
    }
}

/// The batched line-FFT kernel (sweep.js batchFftShader) for one axis of planes M lines by N
/// points (rows: N = width, M = height; columns: N = height, M = width): a workgroup
/// transforms lpw consecutive lines of one plane; columns read lpw contiguous values per row.
/// The stages run in place (stockhamInPlace).
pub fn batchShader(gpa: std.mem.Allocator, N: u32, M: u32, radices: []const u32, wg: u32, inverse: bool, lpw: u32, NP: u32, ax: u32) ![]u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const sign: f64 = if (inverse) 1 else -1;
    var body: std.Io.Writer.Allocating = .init(a);
    try stockhamInPlace(a, &body.writer, N, radices, wg, sign, lpw, NP);
    const final = "sa";
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    const w = &out.writer;
    const scale: f64 = if (inverse) 1.0 / @as(f64, @floatFromInt(N)) else 1.0;
    try w.print(
        \\
        \\struct U {{ n_lines: u32, grid_x: u32, axis: u32, pad: u32 }};
        \\@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
        \\@group(0) @binding(1) var<storage, read> tw: array<vec2<f32>>;
        \\@group(0) @binding(2) var<uniform> u: U;
        \\const N: u32 = {d}u; const M: u32 = {d}u; const NN: u32 = {d}u; const SC: f32 = {f}; const LPW: u32 = {d}u; const NP: u32 = {d}u;
        \\const AX: u32 = {d}u;
        \\var<workgroup> sa: array<vec2<f32>, {d}>;
        \\// point p of line l: rows (AX 0) run along a row, columns (AX 1) down a column
        \\fn gidx(l: u32, p: u32, pbase: u32) -> u32 {{
        \\    if (AX == 0u) {{ return pbase + l * N + p; }}
        \\    return pbase + p * M + l;
        \\}}
        \\fn split(e: u32) -> vec2<u32> {{
        \\    if (AX == 0u) {{ return vec2<u32>(e / N, e % N); }}
        \\    return vec2<u32>(e % LPW, e / LPW);
        \\}}
        \\@compute @workgroup_size({d})
        \\fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
        \\    let line0 = (wid.y * u.grid_x + wid.x) * LPW;
        \\    if (line0 >= u.n_lines) {{ return; }}
        \\    let pbase = (line0 / M) * NN;
        \\    let l0 = line0 % M;
        \\    let tid = lid.x;
        \\    for (var e = tid; e < LPW * N; e = e + {d}u) {{
        \\        let q = split(e);
        \\        sa[q.x * NP + q.y] = data[gidx(l0 + q.x, q.y, pbase)];
        \\    }}
        \\    workgroupBarrier();
        \\
    , .{ N, M, N * M, jsNum(scale), lpw, NP, ax, NP * lpw, wg, wg });
    try w.writeAll(body.written());
    try w.print(
        \\
        \\    for (var e = tid; e < LPW * N; e = e + {d}u) {{
        \\        let q = split(e);
        \\        data[gidx(l0 + q.x, q.y, pbase)] = {s}[q.x * NP + q.y] * SC;
        \\    }}
        \\}}
    , .{ wg, final });
    return out.toOwnedSlice();
}

/// 2-D FFTs of many nx × ny planes at once (sweep.js BatchFft): all planes' rows, then all
/// columns, one dispatch each. init(N) is the square case.
pub const BatchFft = struct {
    /// plane width (row length) and height (column length)
    nx: u32,
    ny: u32,
    /// lines per workgroup, rows / columns
    lpw: [2]u32,
    /// [axis][forward, inverse]
    pipe: [2][2]u32,
    /// twiddles of each axis's line length, [axis][forward, inverse]
    tw: [2][2]Buf,

    /// Tuning (benchmarks): lines per workgroup for rows / columns, 0 = the default.
    pub var tune: [2]u32 = .{ 0, 0 };
    /// Tuning (benchmarks): workgroup size for rows / columns, 0 = the default.
    pub var tune_wg: [2]u32 = .{ 0, 0 };

    pub fn init(g: *Gpu, N: u32) !BatchFft {
        return initRect(g, N, N);
    }

    pub fn initRect(g: *Gpu, nx: u32, ny: u32) !BatchFft {
        const lim = g.lim.max_workgroup_storage;
        if (8 * nx > lim or 8 * ny > lim) return error.FftTooLong;
        var self: BatchFft = .{ .nx = nx, .ny = ny, .lpw = undefined, .pipe = undefined, .tw = undefined };
        var made: usize = 0;
        errdefer for (self.tw[0..made]) |*t| {
            g.release(&t[0]);
            g.release(&t[1]);
        };
        for (0..2) |ax| {
            const L = if (ax == 0) nx else ny; // line length
            const M = if (ax == 0) ny else nx; // lines per plane
            var rad: std.ArrayList(u32) = try .initCapacity(g.gpa, 32);
            defer rad.deinit(g.gpa);
            try factorize(L, &rad);
            // lines per workgroup (8 B per point, in place): rows are contiguous and gain
            // nothing from more; columns want runs that fill memory sectors. Wider batches
            // cost occupancy.
            const want: u32 = if (tune[ax] != 0) tune[ax] else if (ax == 0) 1 else 4;
            var lpw: u32 = 1;
            while (lpw < want and 8 * L * lpw * 2 <= lim and M % (lpw * 2) == 0) lpw *= 2;
            self.lpw[ax] = lpw;
            // measured: short rows and 1024-point columns run best at 128 invocations
            const wg = if (tune_wg[ax] != 0) tune_wg[ax] else if ((ax == 0 and L <= 512) or (ax == 1 and L == 1024)) @min(128, pickWg(L)) else pickWg(L);
            const NP: u32 = L; // (padding the lines apart measured no gain)
            for ([_]bool{ false, true }, 0..) |inv, k| {
                const code = try batchShader(g.gpa, L, M, rad.items, wg, inv, lpw, NP, @intCast(ax));
                defer g.gpa.free(code);
                var key_buf: [80]u8 = undefined;
                const key = try std.fmt.bufPrint(&key_buf, "bfft2/{d}/{d}/{d}/{d}/{d}/{s}/{s}", .{ L, M, ax, lpw, wg, if (inv) "i" else "f", if (balanced) "b" else "g" });
                self.pipe[ax][k] = try g.pipeline(key, code, "main");
            }
            self.tw[ax][0] = try twiddles(g, L, false);
            self.tw[ax][1] = try twiddles(g, L, true);
            made += 1;
        }
        return self;
    }

    pub fn deinit(self: *BatchFft, g: *Gpu) void {
        for (&self.tw) |*t| {
            g.release(&t[0]);
            g.release(&t[1]);
        }
    }

    /// In-place 2-D FFT of n_planes consecutive nx × ny complex planes.
    pub fn run(self: *BatchFft, g: *Gpu, buf: Buf, inverse: bool, n_planes: u32) void {
        self.axis(g, buf, inverse, n_planes, 0);
        self.axis(g, buf, inverse, n_planes, 1);
    }

    /// The 1-D FFTs along rows (axis 0) or columns (axis 1) of n_planes planes.
    pub fn axis(self: *BatchFft, g: *Gpu, buf: Buf, inverse: bool, n_planes: u32, ax: u32) void {
        const k: usize = @intFromBool(inverse);
        const n_lines = n_planes * (if (ax == 0) self.ny else self.nx);
        const n_wg = n_lines / self.lpw[ax];
        const gx = @min(GRID_X, n_wg);
        const gy = (n_wg + gx - 1) / gx;
        const u = [4]u32{ n_lines, gx, ax, 0 };
        g.dispatch(self.pipe[ax][k], gx, gy, 1, &.{ buf.at(0), self.tw[ax][k].at(1), g.uniform(2, std.mem.asBytes(&u)) });
    }
};

/// e^{∓2πi t/N}, t < N (forward: −), as interleaved f32 in a new buffer.
fn twiddles(g: *Gpu, N: u32, inverse: bool) !Buf {
    const tw = try g.gpa.alloc(f32, 2 * N);
    defer g.gpa.free(tw);
    const sgn: f64 = if (inverse) 1 else -1;
    for (0..N) |t| {
        const ang = sgn * 2.0 * std.math.pi * @as(f64, @floatFromInt(t)) / @as(f64, @floatFromInt(N));
        tw[2 * t] = @floatCast(@cos(ang));
        tw[2 * t + 1] = @floatCast(@sin(ang));
    }
    const b = g.storage(@max(16, N * 8));
    g.writeSlice(b, 0, f32, tw);
    return b;
}

/// Longest line one workgroup transforms (16 B of workgroup memory per point).
fn fitsOne(g: *Gpu, N: u32) bool {
    if (force_four_step and N > 64) return false;
    return 16 * N <= g.lim.max_workgroup_storage;
}

/// Tests and benchmarks: take the four-step path for every line longer than 64.
pub var force_four_step = false;

/// Lines too long for one workgroup: N = n1 · n2, both short enough, by the four-step method.
/// A line is read as n2 interleaved sub-lines of n1 points (x[n2·i + j], sub-line j), each is
/// transformed and multiplied by e^{∓2πi j·k1/N}; then its n1 contiguous blocks of n2 points
/// (block k1) are transformed, and point k2 of block k1 is X[k1 + n1·k2]. The first pass writes
/// the scratch buffer at the points it read, the second writes the data in natural order.
pub const Big = struct {
    n1: u32,
    n2: u32,
    /// [sub-length n1, n2][forward, inverse]
    pipe: [2][2]u32,
    /// sub-lines per workgroup, per sub-length
    lpw: [2]u32,
    tw: [2][2]Buf,
    /// e^{∓2πi t/N}, t < N
    twb: [2]Buf,

    pub fn init(g: *Gpu, N: u32) !Big {
        // n2 (pass 1's stride, in lines) as large as possible up to √N while a column's pass-1
        // points stay under 2 MB apart (at 2 MB, a large page each, the GPU's address translation
        // thrashes), both factors short enough
        var n2: u32 = 0;
        var d: u32 = 2;
        while (d * d <= N) : (d += 1) {
            if (N % d != 0 or !fitsOne(g, d) or !fitsOne(g, N / d)) continue;
            if (n2 == 0 or @as(u64, d) * N * 8 < 2 << 20) n2 = d;
        }
        if (n2 == 0) return error.FftTooLong;
        var self: Big = .{ .n1 = N / n2, .n2 = n2, .pipe = undefined, .lpw = undefined, .tw = undefined, .twb = undefined };
        for ([_]u32{ self.n1, self.n2 }, 0..) |M, s| {
            var rad: std.ArrayList(u32) = try .initCapacity(g.gpa, 32);
            defer rad.deinit(g.gpa);
            try factorize(M, &rad);
            // as many sub-lines per workgroup as its memory holds (up to 64), so a workgroup
            // reads and writes runs of neighbouring points
            var lpw: u32 = 1;
            while (lpw < 64 and 16 * M * lpw * 2 <= g.lim.max_workgroup_storage) lpw *= 2;
            self.lpw[s] = lpw;
            const wg = pickWg(M * lpw);
            for ([_]bool{ false, true }, 0..) |inv, k| {
                const code = try strideShader(g.gpa, M, rad.items, wg, inv, N, lpw);
                defer g.gpa.free(code);
                var key_buf: [48]u8 = undefined;
                const key = try std.fmt.bufPrint(&key_buf, "fft4/{d}/{d}/{d}/{s}", .{ N, M, lpw, if (inv) "i" else "f" });
                self.pipe[s][k] = try g.pipeline(key, code, "main");
                self.tw[s][k] = try twiddles(g, M, inv);
            }
        }
        for ([_]bool{ false, true }, 0..) |inv, k| self.twb[k] = try twiddles(g, N, inv);
        return self;
    }

    pub fn deinit(self: *Big, g: *Gpu) void {
        for (&self.tw) |*t| for (t) |*b| g.release(b);
        for (&self.twb) |*b| g.release(b);
    }

    const Map = struct { base: u32, sl: u32, si: u32, sp: u32 };

    fn pass(self: *Big, g: *Gpu, s: usize, inverse: bool, src: Buf, dst: Buf, n_lines: u32, l1: u32, tw: enum { none, lo, hi }, i: Map, o: Map) void {
        const k: usize = @intFromBool(inverse);
        const f = Gpu.flat((n_lines + self.lpw[s] - 1) / self.lpw[s]);
        const u = [16]u32{
            n_lines,                      f[0],   l1,     @intFromBool(tw != .none),
            i.base,                       i.sl,   i.si,   i.sp,
            o.base,                       o.sl,   o.si,   o.sp,
            @intFromBool(tw == .hi), @intFromBool(i.si == 1), @intFromBool(o.si == 1), 0,
        };
        g.dispatch(self.pipe[s][k], f[0], f[1], 1, &.{ src.at(0), self.tw[s][k].at(1), g.uniform(2, std.mem.asBytes(&u)), dst.at(3), self.twb[k].at(4) });
    }

    /// Transform the N columns of one N×N plane at `base` in `a`, with `b` as the intermediate:
    /// a workgroup takes neighbouring columns (the sub-lines' lo), so every read and write is a
    /// run of lpw points along a row.
    pub fn cols(self: *Big, g: *Gpu, a: Buf, b: Buf, inverse: bool, base: u32) void {
        const n1 = self.n1;
        const n2 = self.n2;
        const N = n1 * n2;
        // pass 1: sub-line (j, L) = (hi, lo), point i at L + (n2·i + j)·N; twiddle j·k1
        const m: Map = .{ .base = base, .sl = N, .si = 1, .sp = n2 * N };
        self.pass(g, 0, inverse, a, b, N * n2, N, .hi, m, m);
        // pass 2: block (k1, L) = (hi, lo), point j at L + (n2·k1 + j)·N → X[k1 + n1·k2]
        self.pass(g, 1, inverse, b, a, N * n1, N, .none, .{ .base = base, .sl = n2 * N, .si = 1, .sp = N }, .{ .base = base, .sl = N, .si = 1, .sp = n1 * N });
    }

    /// Transform `count` rows of length N = n1·n2 (row L at L·N) in `a`, with `b` (as large) as
    /// the intermediate. Sub-lines are numbered so that neighbours in memory are neighbours in a
    /// workgroup.
    pub fn rows(self: *Big, g: *Gpu, a: Buf, b: Buf, inverse: bool, count: u32) void {
        const n1 = self.n1;
        const n2 = self.n2;
        const N = n1 * n2;
        // pass 1: sub-line (L, j) = (hi, lo), point i at L·N + n2·i + j; twiddle j·k1
        const m: Map = .{ .base = 0, .sl = N, .si = 1, .sp = n2 };
        self.pass(g, 0, inverse, a, b, count * n2, n2, .lo, m, m);
        // pass 2: block (L, k1) = (hi, lo), point j at L·N + n2·k1 + j → X[k1 + n1·k2]
        self.pass(g, 1, inverse, b, a, count * n1, n1, .none, .{ .base = 0, .sl = N, .si = n2, .sp = 1 }, .{ .base = 0, .sl = N, .si = 1, .sp = n1 });
    }
};

/// 2-D FFTs of n×n complex planes (interleaved re, im) in place, on the GPU.
///
/// Rows: the batched line kernel (several neighbouring rows per workgroup) when a line fits
/// one workgroup's memory, else the four-step passes. Columns: the batched kernel up to
/// COLS_ONE_PASS; longer columns hold one line per workgroup, which reads 8 bytes per row, so
/// they take the four-step passes instead, batched over neighbouring columns (every access a
/// run along a row): two passes, but at memory speed.
pub const LineFft = struct {
    N: u32,
    batch: ?BatchFft = null,
    big: ?Big = null,

    const COLS_ONE_PASS = 2048;

    pub fn init(g: *Gpu, N: u32) !LineFft {
        var self: LineFft = .{ .N = N };
        errdefer self.deinit(g);
        if (!force_four_step) self.batch = BatchFft.init(g, N) catch |err| switch (err) {
            error.FftTooLong => null,
            else => return err,
        };
        if (self.batch == null or N > COLS_ONE_PASS) self.big = try Big.init(g, N);
        return self;
    }

    pub fn deinit(self: *LineFft, g: *Gpu) void {
        if (self.batch) |*b| b.deinit(g);
        if (self.big) |*b| b.deinit(g);
        self.batch = null;
        self.big = null;
    }

    /// Forward or inverse 2-D FFT of `n_planes` consecutive N×N complex planes.
    pub fn planes(self: *LineFft, g: *Gpu, data: Buf, inverse: bool, n_planes: u32) void {
        const N = self.N;
        const plane: u64 = @as(u64, N) * N * 8;
        if (self.batch) |*bf| {
            bf.axis(g, data, inverse, n_planes, 0);
        } else {
            const b = &self.big.?;
            g.ensure(&g.fft_scratch, n_planes * plane);
            b.rows(g, data, g.fft_scratch, inverse, n_planes * N);
        }
        if (self.big) |*b| {
            // the column passes use the scratch at each plane's own offset
            g.ensure(&g.fft_scratch, n_planes * plane);
            var q: u32 = 0;
            while (q < n_planes) : (q += 1) b.cols(g, data, g.fft_scratch, inverse, q * N * N);
        } else self.batch.?.axis(g, data, inverse, n_planes, 1);
    }
};
