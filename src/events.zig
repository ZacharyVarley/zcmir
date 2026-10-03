//! Progress events from long operations (climbs, searches): one JSON object per event.
//!   {"kind":"log","text":…}
//!   {"kind":"trail","label":…,"score":…,"fwd":…,"inv":…,"H":[9],"cps":[…]}
//!   {"kind":"pose","H":[9]}  (the current best pose changed)
//! In the browser each event goes straight to the page (import env.zc_event) so the UI can update
//! while the module is suspended on the GPU. Natively events queue until zc_events drains them.
const std = @import("std");
const builtin = @import("builtin");

const is_web = builtin.cpu.arch == .wasm32;
extern "env" fn zc_event(ptr: [*]const u8, len: usize) void;

pub const Sink = struct {
    gpa: std.mem.Allocator,
    queued: std.ArrayList(u8) = .empty,
    scratch: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Sink) void {
        self.queued.deinit(self.gpa);
        self.scratch.deinit(self.gpa);
    }

    fn send(self: *Sink, line: []const u8) void {
        if (is_web) {
            zc_event(line.ptr, line.len);
        } else {
            self.queued.appendSlice(self.gpa, line) catch return;
            self.queued.append(self.gpa, '\n') catch return;
        }
    }

    /// Queued events (native), newline separated; `take` copies up to buf.len bytes of whole
    /// lines and removes them. Returns the bytes copied.
    pub fn take(self: *Sink, buf: []u8) usize {
        var end: usize = 0;
        var i: usize = 0;
        while (i < self.queued.items.len) : (i += 1) {
            if (self.queued.items[i] == '\n') {
                if (i + 1 > buf.len) break;
                end = i + 1;
            }
        }
        @memcpy(buf[0..end], self.queued.items[0..end]);
        self.queued.replaceRangeAssumeCapacity(0, end, &.{});
        return end;
    }

    pub fn pending(self: *const Sink) usize {
        return self.queued.items.len;
    }

    pub fn log(self: *Sink, comptime fmt: []const u8, args: anytype) void {
        var text: [768]u8 = undefined;
        const t = std.fmt.bufPrint(&text, fmt, args) catch text[0..];
        self.scratch.clearRetainingCapacity();
        var w: std.Io.Writer.Allocating = .fromArrayList(self.gpa, &self.scratch);
        std.json.Stringify.value(.{ .kind = "log", .text = t }, .{}, &w.writer) catch return;
        self.scratch = w.toArrayList();
        self.send(self.scratch.items);
    }

    pub fn trail(self: *Sink, label: []const u8, score: f64, fwd: f64, inv: ?f64, H: [9]f64, cps: ?[]const f32) void {
        self.scratch.clearRetainingCapacity();
        var w: std.Io.Writer.Allocating = .fromArrayList(self.gpa, &self.scratch);
        std.json.Stringify.value(.{ .kind = "trail", .label = label, .score = score, .fwd = fwd, .inv = inv, .H = H, .cps = cps }, .{}, &w.writer) catch return;
        self.scratch = w.toArrayList();
        self.send(self.scratch.items);
    }

    /// Any event: a struct with a `kind` field, as JSON.
    pub fn emit(self: *Sink, value: anytype) void {
        self.scratch.clearRetainingCapacity();
        var w: std.Io.Writer.Allocating = .fromArrayList(self.gpa, &self.scratch);
        std.json.Stringify.value(value, .{}, &w.writer) catch return;
        self.scratch = w.toArrayList();
        self.send(self.scratch.items);
    }

    pub fn pose(self: *Sink, H: [9]f64, cps: ?[]const f32) void {
        self.scratch.clearRetainingCapacity();
        var w: std.Io.Writer.Allocating = .fromArrayList(self.gpa, &self.scratch);
        std.json.Stringify.value(.{ .kind = "pose", .H = H, .cps = cps }, .{}, &w.writer) catch return;
        self.scratch = w.toArrayList();
        self.send(self.scratch.items);
    }
};
