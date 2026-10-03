//! Native GPU probe: open wgpu-native, report the adapter and limits, run one kernel.
//!   zig build probe            (the path to wgpu-native's lib/ is passed by build.zig)
const std = @import("std");
const zc = @import("zcmir");
const gpu = zc.gpu;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var args = try init.minimal.args.iterateAllocator(gpa);
    defer args.deinit();
    _ = args.next();
    const lib_path = args.next() orelse ".deps/wgpu-native/win_amd64/lib";

    const dev = try gpu.Backend.Device.init(gpa, lib_path);
    defer dev.deinit();
    var g = try gpu.Gpu.init(gpa, dev);
    defer g.deinit();
    const lim = g.lim;
    std.debug.print("adapter: {s}\nlimits: storage binding {d} MiB, buffer {d} MiB, workgroup storage {d} B, storage/stage {d}, uniform align {d}\n", .{
        dev.info(), lim.max_storage_binding >> 20, lim.max_buffer >> 20, lim.max_workgroup_storage, lim.max_storage_per_stage, lim.uniform_align,
    });

    // smi.wgsl `fill`: out_a[i] = u.scale for i < w·h.
    const n: u32 = 1000;
    const pipe = try g.pipeline("smi/fill", zc.shaders.smi, "fill");
    if (dev.takeError()) |e| std.debug.print("error: {s}\n", .{e});
    var out = g.storage(n * 4);
    defer g.release(&out);
    var u: [64]u8 = @splat(0);
    std.mem.writeInt(u32, u[0..4], n, .little);
    std.mem.writeInt(u32, u[4..8], 1, .little);
    std.mem.writeInt(u32, u[40..44], @bitCast(@as(f32, 3.5)), .little);
    g.dispatch(pipe, (n + 255) / 256, 1, 1, &.{ g.uniform(0, &u), out.at(4) });
    var host: [1000]f32 = undefined;
    g.read(out, 0, std.mem.sliceAsBytes(&host));
    g.submit();
    try dev.wait();
    if (dev.takeError()) |e| std.debug.print("error: {s}\n", .{e});
    std.debug.print("fill: first {d}, last {d} (expect 3.5)\n", .{ host[0], host[n - 1] });
}
