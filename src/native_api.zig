//! Native shared library: the C ABI the Python package calls through ctypes. The shared calls
//! are in exports.zig; creation (which opens wgpu-native) is here.
const std = @import("std");
pub const gpu = @import("gpu/gpu.zig");
pub const shaders = @import("shaders");
pub const engine = @import("engine.zig");
pub const fft = @import("fft.zig");
pub const lie = @import("lie.zig");

comptime {
    _ = @import("exports.zig");
}

const Engine = engine.Engine;
const gpa = std.heap.smp_allocator;

var create_err: [512]u8 = undefined;
var create_err_len: usize = 0;

export fn zc_version() [*:0]const u8 {
    return engine.version;
}

/// Open wgpu-native (`lib_path`: the library file or its directory) and create an engine.
/// Null on failure; zc_create_error says why.
export fn zc_create(lib_path: [*:0]const u8) ?*Engine {
    create_err_len = 0;
    const dev = gpu.Backend.Device.init(gpa, std.mem.span(lib_path)) catch |e| {
        const m = std.fmt.bufPrint(&create_err, "GPU init failed: {s}", .{@errorName(e)}) catch "GPU init failed";
        create_err_len = m.len;
        return null;
    };
    return Engine.create(gpa, dev) catch |e| {
        const m = std.fmt.bufPrint(&create_err, "engine init failed: {s}", .{@errorName(e)}) catch "engine init failed";
        create_err_len = m.len;
        dev.deinit();
        return null;
    };
}

export fn zc_create_error(buf: [*]u8, cap: usize) usize {
    const n = @min(create_err_len, cap);
    @memcpy(buf[0..n], create_err[0..n]);
    return n;
}

export fn zc_destroy(e: ?*Engine) void {
    const self = e orelse return;
    const dev = self.dev;
    self.destroy();
    dev.deinit();
}
