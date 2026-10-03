//! What Refine (gclimb.zig, on the GPU) and the searches share: the result, the acceptance rule
//! and the step-length ladder.
const std = @import("std");

pub const beats = @import("pair.zig").beats;

pub const Result = extern struct {
    H: [9]f64,
    score: f64,
    iterations: u32,
    moved: u32,
};

/// Step lengths a0, a0·r, a0·r², … (controls.js geoAmps).
pub fn geoAmps(a0: f64, ratio: f64, n: u32, out: []f64) []f64 {
    var a = if (a0 > 0) a0 else 0.08;
    const r = @min(0.95, @max(0.2, if (ratio > 0) ratio else 0.5));
    const k = @max(2, @min(12, n));
    for (0..k) |i| {
        out[i] = a;
        a *= r;
    }
    return out[0..k];
}
