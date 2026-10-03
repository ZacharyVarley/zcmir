//! Browser module (wasm32-freestanding): the engine's exports for web/zcmir.js. GPU calls are
//! imported from the adapter (src/gpu/web.zig). Exports that wait for the GPU suspend through
//! JS Promise Integration; the adapter calls them through WebAssembly.promising. The shared
//! calls are in exports.zig; memory and creation are here.
const std = @import("std");
const gpu = @import("gpu/gpu.zig");
const engine = @import("engine.zig");

comptime {
    _ = @import("exports.zig");
}

const Engine = engine.Engine;
const gpa = std.heap.wasm_allocator;

extern "env" fn zc_log(ptr: [*]const u8, len: usize) void;

pub const std_options: std.Options = .{ .logFn = log };

fn log(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime fmt: []const u8, args: anytype) void {
    _ = scope;
    var buf: [512]u8 = undefined;
    const m = std.fmt.bufPrint(&buf, "[" ++ @tagName(level) ++ "] " ++ fmt, args) catch return;
    zc_log(m.ptr, m.len);
}

pub fn panic(msg: []const u8, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    zc_log(msg.ptr, msg.len);
    @trap();
}

export fn zc_alloc(n: usize) ?[*]u8 {
    const s = gpa.alloc(u8, n) catch return null;
    return s.ptr;
}

export fn zc_free(p: [*]u8, n: usize) void {
    gpa.free(p[0..n]);
}

export fn zc_create() ?*Engine {
    const dev = gpu.Backend.Device.init(gpa) catch return null;
    return Engine.create(gpa, dev) catch null;
}

export fn zc_destroy(e: *Engine) void {
    e.destroy();
}
