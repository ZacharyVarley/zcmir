//! Fitted warps as Insight Transform File V1.0 (.tfm) and an NRRD displacement field (port of
//! export_warp.js): the ITK / 3D Slicer / SimpleITK formats, bundled in a stored ZIP.
//!
//! Displacement convention of itk::ResampleImageFilter / sitk.DisplacementFieldTransform:
//! T(x_fixed) = x_fixed + d(x_fixed) = x_moving, pixel spacing 1, origin 0 (0-based pixel
//! physical coordinates).
const std = @import("std");
const lie = @import("lie.zig");

const Mat3 = lie.Mat3;

pub const Opts = struct {
    /// 1-based moving → fixed homography
    H: Mat3,
    /// B-spline control points (2·g·g, destN units) when the spline is on
    cps: ?[]const f32 = null,
    g: u32 = 4,
    /// lattice on the moving image ("source") instead of the fixed image
    source: bool = false,
    lw: u32,
    lh: u32,
    rw: u32,
    rh: u32,
};

fn cubicW(t: f64) [4]f64 {
    const t2 = t * t;
    const t3 = t2 * t;
    return .{ (1 - t) * (1 - t) * (1 - t) / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6 };
}

fn clampi(i: i64, n: u32) usize {
    return @intCast(@min(@as(i64, n) - 1, @max(0, i)));
}

/// Pixel displacement of the cubic FFD at (x, y) in the lattice image (as wgsl/ffd.wgsl, in f64).
pub fn ffdDisp(x: f64, y: f64, cps: []const f32, gx: u32, gy: u32, w: f64, h: f64) [2]f64 {
    if (gx < 2 or gy < 2) return .{ 0, 0 };
    if (x < 0 or y < 0 or x > w - 1 or y > h - 1) return .{ 0, 0 };
    const hx = w * 0.5;
    const hy = h * 0.5;
    const spx = @max(w - 1, 1) / @as(f64, @floatFromInt(gx - 1));
    const spy = @max(h - 1, 1) / @as(f64, @floatFromInt(gy - 1));
    const su = x / spx;
    const sv = y / spy;
    const iu = @floor(su);
    const iv = @floor(sv);
    const bu = cubicW(@min(1, @max(0, su - iu)));
    const bv = cubicW(@min(1, @max(0, sv - iv)));
    var dx: f64 = 0;
    var dy: f64 = 0;
    for (0..4) |jj| {
        const j = clampi(@as(i64, @intFromFloat(iv)) - 1 + @as(i64, @intCast(jj)), gy);
        for (0..4) |ii| {
            const i = clampi(@as(i64, @intFromFloat(iu)) - 1 + @as(i64, @intCast(ii)), gx);
            const wt = bu[ii] * bv[jj];
            const o = (j * gx + i) * 2;
            dx += wt * cps[o] * hx;
            dy += wt * cps[o + 1] * hy;
        }
    }
    return .{ dx, dy };
}

fn apply3(M: Mat3, x: f64, y: f64) ?[2]f64 {
    const X = M[0] * x + M[1] * y + M[2];
    const Y = M[3] * x + M[4] * y + M[5];
    const Z = M[6] * x + M[7] * y + M[8];
    if (@abs(Z) < 1e-12) return null;
    return .{ X / Z, Y / Z };
}

/// A 0-based fixed pixel mapped to a 0-based moving pixel (the composition of warp_ffd: target
/// lattice: dest + φ then H⁻¹; source lattice: H⁻¹ then φ in the moving frame).
fn destToMoving(o: *const Opts, Hi: Mat3, x: f64, y: f64) ?[2]f64 {
    const ffd = o.cps != null;
    if (ffd and o.source) {
        const q = apply3(Hi, x + 1, y + 1) orelse return null;
        const sx = q[0] - 1;
        const sy = q[1] - 1;
        const d = ffdDisp(sx, sy, o.cps.?, o.g, o.g, @floatFromInt(o.lw), @floatFromInt(o.lh));
        return .{ sx + d[0], sy + d[1] };
    }
    if (ffd) {
        const d = ffdDisp(x, y, o.cps.?, o.g, o.g, @floatFromInt(o.rw), @floatFromInt(o.rh));
        const q = apply3(Hi, x + 1 + d[0], y + 1 + d[1]) orelse return null;
        return .{ q[0] - 1, q[1] - 1 };
    }
    const q = apply3(Hi, x + 1, y + 1) orelse return null;
    return .{ q[0] - 1, q[1] - 1 };
}

/// The displacement field over the fixed image (rw × rh × 2, component fastest) into out.
pub fn displacementField(o: *const Opts, out: []f32) void {
    const Hi = lie.inv3(o.H);
    for (0..o.rh) |y| for (0..o.rw) |x| {
        const fx: f64 = @floatFromInt(x);
        const fy: f64 = @floatFromInt(y);
        const k = (y * o.rw + x) * 2;
        const p = destToMoving(o, Hi, fx, fy);
        const px = if (p) |q| q[0] else std.math.nan(f64);
        const py = if (p) |q| q[1] else std.math.nan(f64);
        out[k] = if (std.math.isFinite(px)) @floatCast(px - fx) else 0;
        out[k + 1] = if (std.math.isFinite(py)) @floatCast(py - fy) else 0;
    };
}

// ── numbers as JavaScript's toPrecision(17) prints them ──
const Big = u1536;

fn pow10(k: u32) Big {
    var r: Big = 1;
    for (0..k) |_| r *= 10;
    return r;
}

/// The 17 significant digits of v > 0, correctly rounded (ties up), and its decimal exponent.
fn digits17(v: f64) struct { n: u64, e: i32 } {
    const bits: u64 = @bitCast(v);
    const bexp: i32 = @intCast((bits >> 52) & 0x7ff);
    const frac = bits & ((@as(u64, 1) << 52) - 1);
    const mant: u64 = if (bexp == 0) frac else frac | (@as(u64, 1) << 52);
    const e2: i32 = if (bexp == 0) -1074 else bexp - 1075;
    var E: i32 = @intFromFloat(@floor(std.math.log10(v)));
    while (true) {
        const k = 16 - E;
        var num: Big = mant;
        var den: Big = 1;
        if (e2 >= 0) num <<= @intCast(e2) else den <<= @intCast(-e2);
        if (k >= 0) num *= pow10(@intCast(k)) else den *= pow10(@intCast(-k));
        var q = num / den;
        const r = num % den;
        if (2 * r >= den) q += 1;
        if (q >= 100_000_000_000_000_000) {
            E += 1;
            continue;
        }
        if (q < 10_000_000_000_000_000) {
            E -= 1;
            continue;
        }
        return .{ .n = @intCast(q), .e = E };
    }
}

/// export_warp.js fmtNum: v.toPrecision(17), "0" for non-finite values.
pub fn fmtNum(w: *std.Io.Writer, v: f64) !void {
    if (!std.math.isFinite(v)) return w.writeAll("0");
    if (v == 0) return w.writeAll("0.0000000000000000");
    const d = digits17(@abs(v));
    var ds: [17]u8 = undefined;
    _ = std.fmt.bufPrint(&ds, "{d}", .{d.n}) catch unreachable;
    if (v < 0) try w.writeAll("-");
    const e = d.e;
    if (e < -6 or e >= 17) {
        try w.print("{c}.{s}e{s}{d}", .{ ds[0], ds[1..], if (e < 0) "-" else "+", @abs(e) });
    } else if (e >= 0) {
        const ip: usize = @intCast(e + 1);
        try w.writeAll(ds[0..ip]);
        if (ip < 17) try w.print(".{s}", .{ds[ip..]});
    } else {
        try w.writeAll("0.");
        for (0..@intCast(-e - 1)) |_| try w.writeAll("0");
        try w.writeAll(&ds);
    }
}

// ── NRRD ──
/// Vector NRRD of the field (component fastest, then x, then y: ITK VectorImage layout).
pub fn nrrd(w: *std.Io.Writer, field: []const f32, fw: u32, fh: u32, extra: []const []const u8) !void {
    try w.print("NRRD0005\n# Complete NRRD file format specification:\n# http://teem.sourceforge.net/nrrd/format.html\ntype: float\ndimension: 3\nspace dimension: 2\nsizes: 2 {d} {d}\nspace directions: none (1,0) (0,1)\nkinds: vector domain domain\nendian: little\nencoding: raw\nspace origin: (0,0)\n", .{ fw, fh });
    for (extra) |l| try w.print("{s}\n", .{l});
    try w.writeAll("\n");
    for (field) |v| try w.writeInt(u32, @bitCast(v), .little);
}

// ── Insight Transform File ──
const SHIFT: Mat3 = .{ 1, 0, 1, 0, 1, 1, 0, 0, 1 };
const SHIFT_INV: Mat3 = .{ 1, 0, -1, 0, 1, -1, 0, 0, 1 };

/// 1-based moving → fixed H to 0-based.
pub fn H0FromH(H: Mat3) Mat3 {
    return lie.mul3(SHIFT_INV, lie.mul3(H, SHIFT));
}

fn isAffineH(H: Mat3) bool {
    return @abs(H[6]) < 1e-8 and @abs(H[7]) < 1e-8 and @abs(H[8] - 1) < 1e-5;
}

pub fn isAffineHomography(H: Mat3) bool {
    return isAffineH(H) and isAffineH(H0FromH(H));
}

fn params(w: *std.Io.Writer, vals: []const f64) !void {
    for (vals, 0..) |v, i| {
        if (i > 0) try w.writeAll(" ");
        try fmtNum(w, v);
    }
}

fn affineTfm(w: *std.Io.Writer, H0inv: Mat3) !void {
    try w.writeAll("Transform: AffineTransform_double_2_2\nParameters: ");
    try params(w, &.{ H0inv[0], H0inv[1], H0inv[3], H0inv[4], H0inv[2], H0inv[5] });
    try w.writeAll("\nFixedParameters: 0 0");
}

/// Our clamped g×g FFD on ITK's coefficient grid (cubic: g + 2 per side, origin −spacing, the
/// first interior point at index 1).
fn bsplineTfm(w: *std.Io.Writer, gpa: std.mem.Allocator, cps: []const f32, g: u32, lw: u32, lh: u32) !void {
    const fw: f64 = @floatFromInt(lw);
    const fh: f64 = @floatFromInt(lh);
    const gf: f64 = @floatFromInt(@max(g - 1, 1));
    const spx = @max(fw - 1, 1) / gf;
    const spy = @max(fh - 1, 1) / gf;
    const hx = fw * 0.5;
    const hy = fh * 0.5;
    const nx = g + 2;
    const n = nx * nx;
    const p = try gpa.alloc(f64, 2 * n);
    defer gpa.free(p);
    for (0..nx) |j| {
        const sj = clampi(@as(i64, @intCast(j)) - 1, g);
        for (0..nx) |i| {
            const si = clampi(@as(i64, @intCast(i)) - 1, g);
            const o = (sj * g + si) * 2;
            p[j * nx + i] = cps[o] * hx;
            p[n + j * nx + i] = cps[o + 1] * hy;
        }
    }
    try w.writeAll("Transform: BSplineTransform_double_2_2\nParameters: ");
    try params(w, p);
    try w.writeAll("\nFixedParameters: ");
    const nxf: f64 = @floatFromInt(nx);
    try params(w, &.{ nxf, nxf, -spx, -spy, spx, spy, 1, 0, 0, 1 });
}

pub fn insightTransformFile(w: *std.Io.Writer, gpa: std.mem.Allocator, o: *const Opts) !void {
    try w.writeAll("#Insight Transform File V1.0\n");
    const H0 = H0FromH(o.H);
    const H0inv = lie.inv3(H0);
    const affine = isAffineH(H0) and isAffineH(o.H);
    const ffd = o.cps != null and o.g >= 2;
    const lat_w = if (o.source) o.lw else o.rw;
    const lat_h = if (o.source) o.lh else o.rh;
    if (ffd and affine) {
        try w.writeAll("#Transform 0\nTransform: CompositeTransform_double_2_2\n");
        if (o.source) {
            try w.writeAll("#Transform 1\n");
            try bsplineTfm(w, gpa, o.cps.?, o.g, lat_w, lat_h);
            try w.writeAll("\n#Transform 2\n");
            try affineTfm(w, H0inv);
        } else {
            try w.writeAll("#Transform 1\n");
            try affineTfm(w, H0inv);
            try w.writeAll("\n#Transform 2\n");
            try bsplineTfm(w, gpa, o.cps.?, o.g, lat_w, lat_h);
        }
    } else if (ffd) {
        try w.writeAll("#Transform 0\n");
        try bsplineTfm(w, gpa, o.cps.?, o.g, lat_w, lat_h);
    } else if (affine) {
        try w.writeAll("#Transform 0\n");
        try affineTfm(w, H0inv);
    } else {
        try w.writeAll("# no AffineTransform_double_2_2: H has perspective. Use the NRRD displacement field.\n#Transform 0\nTransform: IdentityTransform_double_2_2\nParameters:\nFixedParameters:");
    }
    try w.writeAll("\n");
}

// ── ZIP (APPNOTE.TXT, STORE) ──
pub const File = struct { name: []const u8, data: []const u8 };

pub fn zipStore(w: *std.Io.Writer, files: []const File) !void {
    var offsets: [8]u32 = undefined;
    var crcs: [8]u32 = undefined;
    var offset: u32 = 0;
    for (files, 0..) |f, k| {
        const crc = std.hash.Crc32.hash(f.data);
        crcs[k] = crc;
        offsets[k] = offset;
        const len: u32 = @intCast(f.data.len);
        inline for (.{ @as(u32, 0x04034b50), @as(u16, 20), @as(u16, 0), @as(u16, 0), @as(u16, 0), @as(u16, 0) }) |v| try w.writeInt(@TypeOf(v), v, .little);
        try w.writeInt(u32, crc, .little);
        try w.writeInt(u32, len, .little);
        try w.writeInt(u32, len, .little);
        try w.writeInt(u16, @intCast(f.name.len), .little);
        try w.writeInt(u16, 0, .little);
        try w.writeAll(f.name);
        try w.writeAll(f.data);
        offset += 30 + @as(u32, @intCast(f.name.len)) + len;
    }
    var cd_len: u32 = 0;
    for (files, 0..) |f, k| {
        const len: u32 = @intCast(f.data.len);
        inline for (.{ @as(u32, 0x02014b50), @as(u16, 20), @as(u16, 20), @as(u16, 0), @as(u16, 0), @as(u16, 0), @as(u16, 0) }) |v| try w.writeInt(@TypeOf(v), v, .little);
        try w.writeInt(u32, crcs[k], .little);
        try w.writeInt(u32, len, .little);
        try w.writeInt(u32, len, .little);
        try w.writeInt(u16, @intCast(f.name.len), .little);
        inline for (.{ @as(u16, 0), @as(u16, 0), @as(u16, 0), @as(u16, 0), @as(u32, 0) }) |v| try w.writeInt(@TypeOf(v), v, .little);
        try w.writeInt(u32, offsets[k], .little);
        try w.writeAll(f.name);
        cd_len += 46 + @as(u32, @intCast(f.name.len));
    }
    inline for (.{ @as(u32, 0x06054b50), @as(u16, 0), @as(u16, 0) }) |v| try w.writeInt(@TypeOf(v), v, .little);
    try w.writeInt(u16, @intCast(files.len), .little);
    try w.writeInt(u16, @intCast(files.len), .little);
    try w.writeInt(u32, cd_len, .little);
    try w.writeInt(u32, offset, .little);
    try w.writeInt(u16, 0, .little);
}

/// The app's export: {stem}_Warp.nrrd (the composed warp as a displacement field) and, when the
/// spline is on or H is affine, {stem}.tfm — one stored ZIP (caller frees).
pub fn bundle(gpa: std.mem.Allocator, o: *const Opts, stem: []const u8) ![]u8 {
    const field = try gpa.alloc(f32, @as(usize, o.rw) * o.rh * 2);
    defer gpa.free(field);
    displacementField(o, field);
    var hrow: std.Io.Writer.Allocating = .init(gpa);
    defer hrow.deinit();
    try params(&hrow.writer, &o.H);
    const l1 = try std.fmt.allocPrint(gpa, "# 1-based moving→fixed homography (row-major 3×3): {s}", .{hrow.written()});
    defer gpa.free(l1);
    const l2 = if (o.cps != null) try std.fmt.allocPrint(gpa, "# B-spline cubic FFD  {d}×{d}  frame={s}", .{ o.g, o.g, if (o.source) "source" else "target" }) else try gpa.dupe(u8, "# B-spline off");
    defer gpa.free(l2);
    var nr: std.Io.Writer.Allocating = .init(gpa);
    defer nr.deinit();
    try nrrd(&nr.writer, field, o.rw, o.rh, &.{ "# CMIR composed warp: T(x_fixed) = x_moving. 0-based pixels, spacing 1.", l1, l2 });
    var tf: std.Io.Writer.Allocating = .init(gpa);
    defer tf.deinit();
    try insightTransformFile(&tf.writer, gpa, o);
    const n1 = try std.fmt.allocPrint(gpa, "{s}_Warp.nrrd", .{stem});
    defer gpa.free(n1);
    const n2 = try std.fmt.allocPrint(gpa, "{s}.tfm", .{stem});
    defer gpa.free(n2);
    var files: [2]File = .{ .{ .name = n1, .data = nr.written() }, .{ .name = n2, .data = tf.written() } };
    const nfiles: usize = if (o.cps != null or isAffineHomography(o.H)) 2 else 1;
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    try zipStore(&out.writer, files[0..nfiles]);
    return out.toOwnedSlice();
}

test "fmtNum matches toPrecision(17)" {
    const cases = .{
        .{ 1.0, "1.0000000000000000" },               .{ 0.1, "0.10000000000000001" },
        .{ 1.0 / 3.0, "0.33333333333333331" },        .{ 123456.789, "123456.78900000000" },
        .{ 1e-8, "1.0000000000000000e-8" },           .{ 2.5e-7, "2.4999999999999999e-7" },
        .{ 1e21, "1.0000000000000000e+21" },          .{ 1e17, "1.0000000000000000e+17" },
        .{ 1e16, "10000000000000000" },               .{ -0.0001234, "-0.00012339999999999999" },
        .{ 657.0, "657.00000000000000" },             .{ 0.5000000000000001, "0.50000000000000011" },
        .{ 0.0, "0.0000000000000000" },
    };
    inline for (cases) |c| {
        var b: [64]u8 = undefined;
        var w: std.Io.Writer = .fixed(&b);
        try fmtNum(&w, c[0]);
        try std.testing.expectEqualStrings(c[1], w.buffered());
    }
}
