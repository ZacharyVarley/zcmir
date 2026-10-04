//! Every option of the registration pipeline, with the browser app's defaults. Front ends set
//! them with a JSON object (zc_configure): keys are these field names, enums are their names as
//! strings; absent keys keep their current value.
const std = @import("std");
const lie = @import("lie.zig");
const mom = @import("moments.zig");
const fft = @import("fft.zig");

pub const Metric = enum { smi, smi_edge, e4, lmax, ncc };
pub const Detector = enum { pos_gift, gls_mift };
pub const MatchMethod = enum { lofsc, prosac, magsac };
pub const SearchMode = enum { sweep, cloud };

pub const Settings = struct {
    // ── score ──
    metric: Metric = .smi,
    /// per-overlap exact whitening (else one global whitening)
    exact: bool = true,
    /// maps report the Pillai / Wilson–Hilferty z (exact SMI only)
    calibrated: bool = true,
    /// pose score = mean of forward and inverse directions; roto-scale map symmetric
    symmetric: bool = true,
    group: lie.Group = .homography,
    // ── maps ──
    /// map resolution: 0 automatic (a power-of-two grid, FFT 128–512 per side; smi.zig
    /// autoShiftPlan), else the FFT-length cap (256, 512, 1024, 2048: finer maps)
    map_res: u32 = 0,
    /// shift-map floor: shifts whose overlap is below this fraction of the grid score 0
    min_overlap: f64 = 0.05,
    rs_smin: f64 = 0.2,
    rs_smax: f64 = 5,
    // ── B-spline ──
    ffd: bool = false,
    ffd_grid: u32 = 4,
    /// lattice on the moving image ("source") instead of the fixed ("target")
    ffd_source: bool = false,
    /// spline stiffness, a fraction of the lattice's size: a wave of that wavelength costs as much
    /// bending energy (thin plate) as it can gain in score, shorter ones more (0: no bending
    /// penalty)
    ffd_stiffness: f64 = 0.15,
    // ── climb (Pose) ──
    hop: bool = true,
    hop_fm: bool = false,
    g_a0: f64 = 0.08,
    g_decay: f64 = 0.5,
    g_steps: u32 = 7,
    g_iters: u32 = 100,
    // ── preprocessing ──
    clahe: bool = true,
    clahe_grid: u32 = 4,
    clahe_bins: u32 = 64,
    band: bool = false,
    bp_fine: f64 = 1,
    bp_coarse: f64 = 6,
    invert: bool = false,
    half: bool = true,
    /// FFT lengths: ladder (2^a·3^b, a few dozen shaders for any image, slightly more padding) or
    /// compact (the smallest length with prime factors ≤ 13: least work, a new shader per length)
    fft_sizes: fft.Sizes = .ladder,
    // ── detection / matching ──
    detector: Detector = .pos_gift,
    // GLS-MIFT
    n_octaves: u32 = 3,
    max_points: u32 = 2000,
    min_contrast: f64 = 0.01,
    tau: f64 = 0.8,
    radius: f64 = 36,
    nt: f64 = 0,
    auto_nt: bool = true,
    /// GLS-MIFT: also describe each keypoint at its second orientation
    second_ori: bool = true,
    /// inlier distance of the robust fits (pixels)
    inlier_px: f64 = 10,
    /// limits on a fitted pose (match.zig poseCheck): its scale, its stretch (the ratio of its two
    /// local scales) and its perspective (the ratio of its local scale between the moving
    /// image's corners)
    scale_lo: f64 = 0.2,
    scale_hi: f64 = 5,
    max_aniso: f64 = 5,
    max_persp: f64 = 3,
    n_trials: u32 = 100000,
    seed: u32 = 12345,
    n_sigma: u32 = 4,
    n_angle: u32 = 6,
    n_r: u32 = 3,
    match_method: MatchMethod = .lofsc,
    per_octave: bool = true,
    mutual: bool = false,
    // POS-GIFT
    pg_n_octaves: u32 = 3,
    pg_max_points: u32 = 5000,
    pg_min_contrast: f64 = 0.01,
    pg_p1: f64 = 10,
    pg_search: bool = true,
    pg_corners: bool = true,
    pg_desc_pow: f64 = 1.5,
    pg_n_orient: u32 = 6,
    pg_n_rings: u32 = 3,
    pg_n_scales: u32 = 4,
    pg_min_wl: f64 = 3,
    pg_mult: f64 = 1.6,
    pg_sigma_onf: f64 = 0.75,
    pg_pc_k: f64 = 1,
    pg_cutoff: f64 = 0.5,
    pg_pc_g: f64 = 3,
    pg_gsig: f64 = 3,
    pg_pos_p1: f64 = 20,
    pg_pos_k: u32 = 20,
    /// nearest-neighbour ratio test: the best descriptor distance over the second best, at most
    pg_ratio: f64 = 0.6,
    /// POS guided re-matching after the pooled fit; its homography's inlier distance, then the
    /// distance of the correspondences it keeps
    pg_pos: bool = true,
    pg_pos_search_px: f64 = 10,
    pg_pos_px: f64 = 3,
    /// POS-GIFT as the authors' released MATLAB (overrides the POS-GIFT settings above)
    pg_released: bool = false,
    // ── search ──
    search_mode: SearchMode = .sweep,
    sw_grid: u32 = 128,
    sw_nth: u32 = 90,
    /// rotation half-range ± (degrees), or the range [sw_rot_lo, sw_rot_hi] when sw_rot_range
    sw_rot: f64 = 180,
    sw_rot_range: bool = false,
    sw_rot_lo: f64 = -180,
    sw_rot_hi: f64 = 180,
    /// also sweep shear k (±sw_shear_max, sw_nshear values) and stretch a (a and 1/a along the
    /// moving image's axes, up to 1 + sw_aniso_max, sw_naniso values): pose σ R(θ) diag(a, 1/a) [1 k; 0 1]
    sw_shear: bool = false,
    sw_shear_max: f64 = 0.1,
    sw_nshear: u32 = 3,
    sw_aniso: bool = false,
    sw_aniso_max: f64 = 0.1,
    sw_naniso: u32 = 3,
    sw_ns: u32 = 20,
    sw_smin: f64 = 0.5,
    sw_smax: f64 = 2,
    sw_k: u32 = 32,
    sw_fin: u32 = 4,
    sw_rel: bool = false,
    hho_auto: bool = true,
    hho_n: u32 = 100,
    hho_t: u32 = 20,
    hho_sg: f64 = 0.01,
    hho_sh: f64 = 0.01,
    hho_p: f64 = 0.0001,
    /// translation and rotation half-widths of the cloud (fraction of the image, radians)
    hho_tx: f64 = 0.01,
    hho_th: f64 = 0.01,
    hho_phi: f64 = 0.01,
    hho_seed: u32 = 1,
    hho_rel: bool = true,

    pub fn family(self: *const Settings) mom.Family {
        return switch (self.metric) {
            .e4 => .e4,
            .lmax => .lmax,
            else => .smi,
        };
    }

    /// Plan exactness of maps: E4 uses the 25-moment plan, λmax all 45 (smi.js mapExact).
    pub fn mapExact(self: *const Settings) bool {
        return switch (self.metric) {
            .e4 => false,
            .lmax => true,
            else => self.exact,
        };
    }

    /// Apply a JSON object of overrides. Unknown keys are an error (so typos are caught).
    pub fn apply(self: *Settings, gpa: std.mem.Allocator, json: []const u8) !void {
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
        defer parsed.deinit();
        const obj = switch (parsed.value) {
            .object => |o| o,
            else => return error.ExpectedObject,
        };
        var it = obj.iterator();
        outer: while (it.next()) |kv| {
            inline for (@typeInfo(Settings).@"struct".fields) |f| {
                if (std.mem.eql(u8, kv.key_ptr.*, f.name)) {
                    @field(self, f.name) = try std.json.parseFromValueLeaky(f.type, gpa, kv.value_ptr.*, .{});
                    continue :outer;
                }
            }
            return error.UnknownSetting;
        }
    }

    /// The settings as JSON (caller frees).
    pub fn toJson(self: *const Settings, gpa: std.mem.Allocator) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(gpa);
        errdefer out.deinit();
        try std.json.Stringify.value(self.*, .{}, &out.writer);
        return out.toOwnedSlice();
    }
};

test "apply" {
    var s: Settings = .{};
    try s.apply(std.testing.allocator, "{\"metric\":\"e4\",\"g_iters\":5,\"symmetric\":false,\"group\":\"affine\"}");
    try std.testing.expectEqual(Metric.e4, s.metric);
    try std.testing.expectEqual(@as(u32, 5), s.g_iters);
    try std.testing.expect(!s.symmetric);
    try std.testing.expectEqual(lie.Group.affine, s.group);
    try std.testing.expectError(error.UnknownSetting, s.apply(std.testing.allocator, "{\"nope\":1}"));
    const j = try s.toJson(std.testing.allocator);
    defer std.testing.allocator.free(j);
    try std.testing.expect(std.mem.indexOf(u8, j, "\"metric\":\"e4\"") != null);
}
