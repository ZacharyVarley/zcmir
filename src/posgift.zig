//! POS-GIFT (Hou, Liu & Zhang 2024) on the GPU, port of posgift.js (after the authors' released
//! MATLAB): pyramid → phase congruency → keypoints on its normalized sum → GIFT descriptors →
//! nearest neighbours over every level pair → one affine FSC over the pooled matches → POS guided
//! re-matching at full resolution → final fit. Corners, matching and FSC run GLS-MIFT's kernels
//! (gls.zig); the matcher is compiled for POS-GIFT's descriptor length. The engine parameters
//! it shares with GLS-MIFT change in the same order as in the JS module.
const std = @import("std");
const gpu_mod = @import("gpu/gpu.zig");
const shaders = @import("shaders");
const lie = @import("lie.zig");
const fft = @import("fft.zig");
const gls_mod = @import("gls.zig");

const Gpu = gpu_mod.Gpu;
const Buf = gpu_mod.Buf;
const Bind = gpu_mod.Bind;
const Gls = gls_mod.Gls;
const MAX_KP = gls_mod.MAX_KP;
const PW_ITERS = 20;
/// U.dflags: descriptors staged ring by ring (pos_gift.wgsl pg_sample)
const STAGED: u32 = 16;

/// Per-keypoint orientation: search (upright descriptors, one global rotation among 2·NDIR
/// hypotheses), upright (none), released (the released default: range-normalized channels, Sobel
/// histogram), paper (the paper's GFP-norm direction).
pub const Rotation = enum { search, upright, released, paper };
pub const DescNorm = enum { raw, spectral, range };
pub const MatchMode = enum { octave, matlab, nn };
pub const DetMap = enum { pcsum, min_moment, max_moment };
pub const PosMode = enum { pick, dense };

/// POS_GIFT_DEFAULTS of posgift.js.
pub const Settings = struct {
    n_octaves: u32 = 3,
    max_points: u32 = 5000,
    min_contrast: f64 = 0.01,
    rotation: Rotation = .search,
    search_half: bool = true,
    search_max_scale: f64 = 1.5,
    search_trials: u32 = 100000,
    /// rank rotation hypotheses by a global affine each (else the summed per-pair consensus)
    search_global: bool = false,
    search_keep: u32 = 3,
    p1: f64 = 10,
    desc_norm: DescNorm = .range,
    match: MatchMode = .octave,
    ratio: f64 = 0.6,
    max_ssd: f64 = 0.04,
    pos: bool = true,
    pos_p1: f64 = 20,
    pos_k: u32 = 20,
    pos_mode: PosMode = .pick,
    pos_win: u32 = 12,
    n_scales: u32 = 4,
    n_orient: u32 = 6,
    n_rings: u32 = 3,
    desc_ss: u32 = 0,
    ss_w: f64 = 0.3,
    desc_pow: f64 = 1.5,
    det_map: DetMap = .min_moment,
    min_wl: f64 = 3,
    mult: f64 = 1.6,
    sigma_onf: f64 = 0.75,
    pc_k: f64 = 1,
    cutoff: f64 = 0.5,
    pc_g: f64 = 3,
    gsig: f64 = 3,
    /// final least-squares model on the POS inliers: affine, else homography (the engine sets it
    /// from the chosen model)
    pos_affine: bool = false,
    n_trials: u32 = 100000,
    fsc_cover: bool = false,
    inlier_px: f64 = 10,
    seed: u32 = 12345,

    /// POS_GIFT_RELEASED: the authors' released MATLAB (POS_GIFT.p with MatchDemo's settings).
    pub fn released(self: Settings) Settings {
        var s = self;
        s.max_points = 5000;
        s.rotation = .released;
        s.p1 = 16;
        s.match = .matlab;
        s.pos_affine = false;
        s.desc_norm = .raw;
        s.desc_pow = 1;
        s.det_map = .pcsum;
        return s;
    }
};

/// Uniform block PG of pos_gift.wgsl.
const U = extern struct {
    w: u32 = 0,
    h: u32 = 0,
    nn: u32 = 0,
    o: u32 = 0,
    sw: u32 = 0,
    sh: u32 = 0,
    axis: u32 = 0,
    hsize: u32 = 0,
    kp_offset: u32 = 0,
    n_kp: u32 = 0,
    ring: u32 = 0,
    mode: u32 = 0,
    ma: u32 = 0,
    pad0: u32 = 0,
    n_b: u32 = 0,
    maxd: u32 = 0,
    scale: f32 = 0,
    rscale: f32 = 0,
    mult: f32 = 0,
    sigma_onf: f32 = 0,
    min_wl: f32 = 0,
    k: f32 = 0,
    cutoff: f32 = 0,
    g: f32 = 0,
    gsig: f32 = 0,
    thr2: f32 = 0,
    cth: f32 = 0,
    sth: f32 = 0,
    ssw: f32 = 0,
    dflags: u32 = 0,
    dpow: f32 = 0,
    pad4: u32 = 0,
};
comptime {
    std.debug.assert(@sizeOf(U) == 128);
}

/// Descriptor structure (shader constants).
pub const Struct = struct { ns: u32, no: u32, ndir: u32, nring: u32, npt: u32, dp: u32, dq: u32, knn: u32, ssn: u32 };

pub const SideId = enum(u2) { a, b, b0, b15 };
/// ori: orientation words (pos_gift.wgsl), extra-slot prefixes, then from DIR_REC each keypoint
/// slot's descriptor direction (of ndir).
const SideBufs = struct { des: Buf = .{}, desq: Buf = .{}, dsc: Buf = .{}, ori: Buf = .{}, cs0: Buf = .{}, cs0n: u64 = 0 };
pub const DIR_REC: u64 = 2 * @as(u64, MAX_KP);

pub const Level = struct { offset: u32, count: u32, base: u32, scale: f64, w: u32, h: u32 };

/// Keypoints of one image: count and pyramid levels (offsets into the keypoint buffer).
pub const Det = struct {
    n: u32 = 0,
    levels: std.ArrayList(Level) = .empty,
    w: u32 = 0,
    h: u32 = 0,

    fn deinit(self: *Det, gpa: std.mem.Allocator) void {
        self.levels.deinit(gpa);
    }

    fn scale1(self: *const Det) ?Level {
        for (self.levels.items) |l| if (l.scale == 1) return l;
        return null;
    }
};

/// A frame of image 2 for the rotation search: its keypoints (upright, or turned by deg).
const Frame = struct { deg: f64, det: u1, kps: Buf, side: SideId };

/// detectPair's record, which match takes (repeatable).
pub const Detection = struct {
    dl: Det = .{},
    dr: Det = .{},
    dr15: Det = .{},
    rotation: Rotation,
    structure: Struct,
    frames: [2]Frame = undefined,
    n_frames: u32 = 0,

    pub fn deinit(self: *Detection, gpa: std.mem.Allocator) void {
        self.dl.deinit(gpa);
        self.dr.deinit(gpa);
        self.dr15.deinit(gpa);
    }

    fn frameDet(self: *const Detection, f: Frame) *const Det {
        return if (f.det == 0) &self.dr else &self.dr15;
    }
};

pub const Image = struct { buf: Buf, w: u32, h: u32 };

pub const MatchOut = struct {
    H: lie.Mat3,
    Haff: lie.Mat3,
    pooled: u32,
    ninl_aff: u32,
    pos_ran: bool = false,
    /// H is POS's fit (else the pooled affine)
    pos_h: bool = false,
    pos_n: u32 = 0,
    pos_inl: u32 = 0,
    pos_pred: u32 = 0,
    /// rotation search: the chosen rotation and frame (degrees)
    rot_deg: ?f64 = null,
    frame_deg: f64 = 0,
};

/// The last match: which keypoints image 2's matches index, and the map from that frame back to
/// image 2 (rotation search). Image 2's descriptors: the directions recorded in side `dirs`, turned
/// by `turn` more (pg_shift), in its frame turned by `deg`.
pub const Last = struct { n1: u32 = 0, n2: u32 = 0, kps_b: Buf = .{}, back: ?lie.Mat3 = null, dirs: SideId = .b, turn: u32 = 0, deg: f64 = 0 };

const Radii = struct { R: [8]f64, G: [8]u32, n: u32 };

/// Ring radii of get_radius(P1, nring) (R_i = R_{i−1} + P1·(i − 1)) and their Gaussian sizes ceil(R / 2).
fn radii(p1: f64, nring: u32) Radii {
    var r: Radii = .{ .R = @splat(0), .G = @splat(0), .n = nring };
    r.R[0] = p1;
    var i: u32 = 2;
    while (i <= nring) : (i += 1) r.R[i - 1] = r.R[i - 2] + p1 * @as(f64, @floatFromInt(i - 1));
    for (0..nring) |k| r.G[k] = @intFromFloat(@ceil(r.R[k] / 2 - 1e-9));
    return r;
}

pub const PosGift = struct {
    g: *Gpu,
    gls: *Gls,
    set: Settings = .{},
    S: Struct = undefined,
    have_s: bool = false,
    code32: []u8 = &.{},
    code16: []u8 = &.{},
    mcode: []u8 = &.{},
    key_buf: [96]u8 = undefined,
    hist: Buf,
    tv: Buf,
    mm: Buf,
    sig: Buf,
    ffts: std.AutoHashMapUnmanaged(u32, fft.LineFft) = .empty,
    cap: u64 = 0,
    cap_n: u32 = 0,
    spec: Buf = .{},
    planes: Buf = .{},
    cs: Buf = .{},
    en: Buf = .{},
    wt: Buf = .{},
    pc: Buf = .{},
    /// phase-congruency sums (pg_pcacc): the sum, Σ cx², Σ cy², Σ cx·cy
    pcm: Buf = .{},
    tmp: Buf = .{},
    gm: Buf = .{},
    vecs: Buf = .{},
    lv_a: Buf = .{},
    lv_b: Buf = .{},
    gray32: Buf = .{},
    gbin: Buf = .{},
    sides: [4]SideBufs = .{ .{}, .{}, .{}, .{} },
    offs_cache: std.AutoHashMapUnmanaged(u64, Buf) = .empty,
    kps_b15: Buf = .{},
    score_buf: Buf = .{},
    last: Last = .{},

    pub fn init(g: *Gpu, gls: *Gls) !PosGift {
        var self: PosGift = .{
            .g = g,
            .gls = gls,
            .hist = g.storage(8 * 4096 * 4),
            .tv = g.storage(64),
            .mm = g.storage(16),
            .sig = g.storage(64),
        };
        try self.setStructure();
        return self;
    }

    pub fn deinit(self: *PosGift) void {
        const g = self.g;
        const gpa = g.gpa;
        for ([_]*Buf{ &self.hist, &self.tv, &self.mm, &self.sig, &self.kps_b15, &self.score_buf }) |b| g.release(b);
        self.releaseWork();
        self.releaseSides();
        self.releaseOffs();
        self.offs_cache.deinit(gpa);
        var it = self.ffts.valueIterator();
        while (it.next()) |f| f.deinit(g);
        self.ffts.deinit(gpa);
        gpa.free(self.code32);
        gpa.free(self.code16);
        gpa.free(self.mcode);
    }

    fn releaseWork(self: *PosGift) void {
        for ([_]*Buf{ &self.spec, &self.planes, &self.cs, &self.en, &self.wt, &self.pc, &self.pcm, &self.tmp, &self.gm, &self.vecs, &self.lv_a, &self.lv_b, &self.gray32, &self.gbin }) |b| self.g.release(b);
    }

    fn releaseSides(self: *PosGift) void {
        for (&self.sides) |*sb| {
            for ([_]*Buf{ &sb.des, &sb.desq, &sb.dsc, &sb.ori, &sb.cs0 }) |b| self.g.release(b);
            sb.* = .{};
        }
    }

    fn releaseOffs(self: *PosGift) void {
        var it = self.offs_cache.valueIterator();
        while (it.next()) |b| self.g.release(b);
        self.offs_cache.clearRetainingCapacity();
    }

    /// Descriptor structure from the settings (n_scales, n_orient, n_rings, pos_k, desc_ss):
    /// shader constants, the descriptor length, and the matcher compiled for it. Buffers sized by
    /// it are rebuilt on demand.
    pub fn setStructure(self: *PosGift) !void {
        const st = self.set;
        const no = st.n_orient;
        const ndir = 2 * no;
        const nring = st.n_rings;
        const npt = ndir * nring + 1;
        const ssn = st.desc_ss * ndir * nring;
        const dp = (npt * no + ssn + 3) / 4 * 4;
        const S: Struct = .{ .ns = st.n_scales, .no = no, .ndir = ndir, .nring = nring, .npt = npt, .dp = dp, .dq = dp / 4, .knn = st.pos_k, .ssn = ssn };
        if (self.have_s and std.meta.eql(S, self.S)) return;
        const gpa = self.g.gpa;
        var c: []u8 = try gpa.dupe(u8, shaders.pos_gift);
        defer gpa.free(c);
        const subs = .{
            .{ "NS", S.ns }, .{ "NO", S.no }, .{ "NDIR", S.ndir }, .{ "NRING", S.nring }, .{ "NPT", S.npt },
            .{ "DP", S.dp }, .{ "DQ", S.dq }, .{ "KNN", S.knn }, .{ "SSN", S.ssn },
        };
        inline for (subs) |kv| {
            const next = try Gls.sub(gpa, c, kv[0], kv[1]);
            gpa.free(c);
            c = next;
        }
        gpa.free(self.code32);
        gpa.free(self.code16);
        self.code32 = try std.fmt.allocPrint(gpa, "alias Texel = f32;\n{s}", .{c});
        self.code16 = try std.fmt.allocPrint(gpa, "enable f16;\nalias Texel = f16;\n{s}", .{c});
        // gls_mift.wgsl matching / pooling kernels for this descriptor length (f32 texels)
        const vec = dp / 4;
        const kc = struct {
            fn f(limit: u32, v: u32) u32 {
                var k = @min(limit, v);
                while (k >= 1) : (k -= 1) if (v % k == 0) return k;
                return 1;
            }
        }.f;
        var m: []u8 = try gpa.dupe(u8, shaders.gls_mift);
        defer gpa.free(m);
        const msubs = .{ .{ "DES_DIM", dp }, .{ "NN_VEC", vec }, .{ "NQ_KC", kc(9, vec) }, .{ "NN_KC", kc(4, vec) } };
        inline for (msubs) |kv| {
            const next = try Gls.sub(gpa, m, kv[0], kv[1]);
            gpa.free(m);
            m = next;
        }
        gpa.free(self.mcode);
        self.mcode = try std.fmt.allocPrint(gpa, "alias Texel = f32;\n{s}{s}", .{ self.gls.dot4q(), m });
        self.S = S;
        self.have_s = true;
        self.releaseSides();
        self.releaseOffs();
        self.cap = 0;
        self.cap_n = 0;
    }

    // ── plumbing ──
    fn pipe(self: *PosGift, entry: []const u8) u32 {
        const S = self.S;
        const h = self.gls.half;
        const key = std.fmt.bufPrint(&self.key_buf, "pg{d}/{d}.{d}.{d}.{d}.{d}/{s}", .{ @as(u32, if (h) 16 else 32), S.ns, S.no, S.nring, S.knn, S.ssn, entry }) catch entry;
        return self.g.pipeline(key, if (h) self.code16 else self.code32, entry) catch |e| {
            std.log.err("pos_gift.wgsl {s}: {s}", .{ entry, @errorName(e) });
            return gpu_mod.NO_PIPE;
        };
    }

    fn mpipe(self: *PosGift, entry: []const u8) u32 {
        const key = std.fmt.bufPrint(&self.key_buf, "pgm/{d}/{s}", .{ self.S.dp, entry }) catch entry;
        return self.g.pipeline(key, self.mcode, entry) catch |e| {
            std.log.err("gls_mift.wgsl ({d}) {s}: {s}", .{ self.S.dp, entry, @errorName(e) });
            return gpu_mod.NO_PIPE;
        };
    }

    /// The uniform block: the phase-congruency and descriptor settings plus the call's fields.
    fn u(self: *PosGift, p: U) Bind {
        var q = p;
        const s = &self.set;
        q.mult = @floatCast(s.mult);
        q.sigma_onf = @floatCast(s.sigma_onf);
        q.min_wl = @floatCast(s.min_wl);
        q.k = @floatCast(s.pc_k);
        q.cutoff = @floatCast(s.cutoff);
        q.g = @floatCast(s.pc_g);
        q.gsig = @floatCast(s.gsig);
        q.ssw = @floatCast(s.ss_w);
        q.dflags = (@as(u32, @intFromEnum(s.det_map)) << 1) | (p.dflags & STAGED);
        q.dpow = @floatCast(s.desc_pow);
        return self.g.uniform(0, std.mem.asBytes(&q));
    }

    fn run(self: *PosGift, entry: []const u8, x: u32, y: u32, z: u32, p: U, binds: []const Bind) void {
        var all: [16]Bind = undefined;
        all[0] = self.u(p);
        @memcpy(all[1 .. binds.len + 1], binds);
        self.g.dispatch(self.pipe(entry), x, y, z, all[0 .. binds.len + 1]);
    }

    /// A gls_mift.wgsl matcher kernel with the engine parameters `q`.
    fn mrun(self: *PosGift, entry: []const u8, x: u32, y: u32, q: gls_mod.Params, binds: []const Bind) void {
        var all: [16]Bind = undefined;
        all[0] = self.gls.Pwith(q);
        @memcpy(all[1 .. binds.len + 1], binds);
        self.g.dispatch(self.mpipe(entry), x, y, 1, all[0 .. binds.len + 1]);
    }

    fn lineFft(self: *PosGift, N: u32) !*fft.LineFft {
        const r = try self.ffts.getOrPut(self.g.gpa, N);
        if (!r.found_existing) {
            r.value_ptr.* = fft.LineFft.init(self.g, N) catch |e| {
                _ = self.ffts.remove(N);
                return e;
            };
        }
        return r.value_ptr;
    }

    /// Sample offsets (dx, dy) per ring and direction: (−round(R cos θ), −round(R sin θ)), MATLAB rounding.
    fn offs(self: *PosGift, p1: f64) !Buf {
        const key: u64 = @bitCast(p1);
        if (self.offs_cache.get(key)) |b| return b;
        const S = self.S;
        const rr = radii(p1, S.nring);
        const o = try self.g.gpa.alloc(i32, S.nring * S.ndir * 2);
        defer self.g.gpa.free(o);
        for (0..S.nring) |r| for (0..S.ndir) |a| {
            const t = @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(S.ndir)) * 2 * std.math.pi;
            o[(r * S.ndir + a) * 2] = @intFromFloat(-lie.mround(rr.R[r] * @cos(t)));
            o[(r * S.ndir + a) * 2 + 1] = @intFromFloat(-lie.mround(rr.R[r] * @sin(t)));
        };
        const b = self.g.storage(o.len * 4);
        self.g.writeSlice(b, 0, i32, o);
        try self.offs_cache.put(self.g.gpa, key, b);
        return b;
    }

    /// Work buffers for images up to n pixels (FFT side nn).
    fn ensure(self: *PosGift, n: u64, nn: u32) void {
        if (n <= self.cap and nn <= self.cap_n) return;
        const g = self.g;
        self.releaseWork();
        self.cap = @max(n, self.cap);
        self.cap_n = @max(nn, self.cap_n);
        const N2: u64 = @as(u64, self.cap_n) * self.cap_n;
        const S = self.S;
        const cap = self.cap;
        self.spec = g.storage(N2 * 8);
        self.planes = g.storage(4 * N2 * 8);
        self.cs = g.storage(S.no * cap * 4);
        // energy and weight: one orientation at a time (the range normalization, which fills
        // every orientation of en, grows it); pcm: the sum, and the moments' three planes
        self.en = g.storage(cap * 4);
        self.wt = g.storage(cap * 4);
        self.pc = g.storage(cap * 4);
        self.pcm = g.storage(4 * cap * 4);
        self.tmp = g.storage(cap * 4);
        self.gm = g.storage(S.no * cap * 4);
        self.vecs = g.storage(@as(u64, S.no) * 3 * self.cap_n * 4);
        self.lv_a = g.storage(cap * 4);
        self.lv_b = g.storage(cap * 4);
        self.gray32 = g.storage(cap * 4);
        self.gbin = g.storage(cap * 4);
    }

    /// The full-resolution Σ|EO| of a side's image: the upright frame (b0) is image 2's own.
    fn cs0Of(self: *PosGift, id: SideId) Buf {
        return self.sides[@intFromEnum(if (id == .b0) SideId.b else id)].cs0;
    }

    fn side(self: *PosGift, id: SideId) *SideBufs {
        const sb = &self.sides[@intFromEnum(id)];
        if (sb.des.id == 0) {
            const g = self.g;
            sb.des = g.storage(@as(u64, MAX_KP) * self.S.dp * 4);
            sb.desq = g.storage(@as(u64, MAX_KP) * self.S.dq * 4);
            sb.dsc = g.storage(@as(u64, MAX_KP) * 4);
            sb.ori = g.storage(3 * @as(u64, MAX_KP) * 4);
        }
        return sb;
    }

    /// MATLAB imresize(src, scale, 'bilinear') (antialiased), sw × sh → dst; tmp holds the first pass.
    fn resize(self: *PosGift, src: Buf, sw: u32, sh: u32, scale: f64, dst: Buf, tmp: Buf) [2]u32 {
        const w: u32 = @intFromFloat(@ceil(@as(f64, @floatFromInt(sw)) * scale - 1e-9));
        const h: u32 = @intFromFloat(@ceil(@as(f64, @floatFromInt(sh)) * scale - 1e-9));
        const rs: f32 = @floatCast(scale);
        self.run("pg_resize", (w + 7) / 8, (sh + 7) / 8, 1, .{ .w = w, .h = sh, .sw = sw, .sh = sh, .axis = 0, .rscale = rs }, &.{ src.at(1), tmp.at(2) });
        self.run("pg_resize", (w + 7) / 8, (h + 7) / 8, 1, .{ .w = w, .h = h, .sw = w, .sh = sh, .axis = 1, .rscale = rs }, &.{ tmp.at(1), dst.at(2) });
        return .{ w, h };
    }

    /// Phase congruency of a w × h f32 image: Σ|EO| per orientation in cs, the normalized sum into fmap (texels).
    fn phaseCong(self: *PosGift, img: Buf, w: u32, h: u32, fmap: Buf) !void {
        const g = self.g;
        const nn = fft.planSize(@max(w, h), 1);
        const f = try self.lineFft(nn);
        const gn = (nn + 7) / 8;
        const gx = (w + 7) / 8;
        const gy = (h + 7) / 8;
        self.run("pg_pad", gn, gn, 1, .{ .w = w, .h = h, .nn = nn }, &.{ img.at(1), self.spec.at(3) });
        f.planes(g, self.spec, false, 1);
        g.clear(self.hist);
        var o: u32 = 0;
        while (o < self.S.no) : (o += 1) {
            self.run("pg_filter", gn, gn, 4, .{ .nn = nn, .o = o }, &.{ self.spec.at(3), self.planes.at(4) });
            f.planes(g, self.planes, true, 4);
            self.run("pg_orient", gx, gy, 1, .{ .w = w, .h = h, .nn = nn, .o = o }, &.{ self.planes.at(4), self.cs.at(5), self.en.at(6), self.wt.at(7), self.hist.at(8) });
            // this orientation's noise threshold, then its share of the sum
            self.run("pg_tau", 1, 1, 1, .{ .w = w, .h = h, .o = o }, &.{ self.hist.at(8), self.tv.at(9) });
            const angl: f32 = @as(f32, @floatFromInt(o)) * std.math.pi / @as(f32, @floatFromInt(self.S.no));
            const ca: f32 = @floatCast(@cos(@as(f64, angl)));
            const sa: f32 = @floatCast(@sin(@as(f64, angl)));
            self.run("pg_pcacc", gx, gy, 1, .{ .w = w, .h = h, .o = o, .cth = ca, .sth = sa }, &.{ self.en.at(6), self.wt.at(7), self.tv.at(9), self.pcm.at(28) });
        }
        g.writeSlice(self.mm, 0, u32, &.{ 0x7f800000, 0, 0, 0 });
        self.run("pg_pcsum", gx, gy, 1, .{ .w = w, .h = h }, &.{ self.pc.at(2), self.mm.at(10), self.pcm.at(28) });
        self.run("pg_fmap", gx, gy, 1, .{ .w = w, .h = h }, &.{ self.pc.at(1), self.mm.at(10), fmap.at(20) });
    }

    /// Spectral norm of each cs channel (the spectral / range normalizations) into sig.
    fn spectralNorms(self: *PosGift, w: u32, h: u32) void {
        const p: U = .{ .w = w, .h = h, .maxd = self.cap_n };
        const no = self.S.no;
        self.run("pg_pw_init", (w + 63) / 64, no, 1, p, &.{self.vecs.at(11)});
        for (0..PW_ITERS) |_| {
            self.run("pg_pw_rows", h, no, 1, p, &.{ self.cs.at(5), self.vecs.at(11) });
            self.run("pg_pw_cols", (w + 63) / 64, no, 1, p, &.{ self.cs.at(5), self.vecs.at(11) });
            self.run("pg_pw_norm", no, 1, 1, p, &.{ self.vecs.at(11), self.sig.at(12) });
        }
    }

    /// Ring maps of cs (w × h) for ring radii P1 (mode 2 divides by the spectral norms, 1: POS).
    /// With `stage` (descriptor modes 0, 1, 4) each ring's maps go to gm's first ring slot and
    /// pg_sample stores the keypoints' points on that ring (stage.p: the describe call's
    /// parameters, stage.binds: gm, offsets, keypoints, descriptors, counters); gm holds one ring.
    fn ringMaps(self: *PosGift, cs: Buf, w: u32, h: u32, p1: f64, mode: u32, stage: ?Stage) void {
        const rr = radii(p1, self.S.nring);
        if (stage == null) self.g.ensure(&self.gm, @as(u64, self.S.nring) * self.S.no * self.cap * 4);
        var r: u32 = 0;
        while (r < self.S.nring) : (r += 1) {
            // one orientation at a time: the intermediate is one plane
            var o: u32 = 0;
            while (o < self.S.no) : (o += 1) {
                const p: U = .{ .w = w, .h = h, .hsize = rr.G[r], .ring = if (stage != null) 0 else r, .mode = mode, .o = o };
                self.run("pg_gauss_h", (w + 7) / 8, (h + 7) / 8, 1, p, &.{ cs.at(5), self.sig.at(12), self.tmp.at(2) });
                self.run("pg_gauss_v", (w + 7) / 8, (h + 7) / 8, 1, p, &.{ self.tmp.at(1), self.gm.at(13) });
            }
            if (stage) |s| {
                var q = s.p;
                q.ring = r;
                self.run("pg_sample", s.groups, 1, 1, q, s.binds);
            }
        }
    }

    const Stage = struct { p: U, groups: u32, binds: []const Bind };

    const Img = struct { buf: Buf, w: u32, h: u32, scale: f64 };

    /// Keypoints and descriptors of one image (gray: texels w × h) into kps and a side's
    /// descriptor buffers; image turned by rot_deg first (the rotation search's half-step frame).
    fn detect(self: *PosGift, gray: Buf, w: u32, h: u32, kps: Buf, which: SideId, rot_deg: f64) !Det {
        const g = self.g;
        const gpa = g.gpa;
        const e = self.gls;
        const st = self.set;
        const S = self.S;
        const sb = self.side(which);
        const n: u64 = @as(u64, w) * h;
        const nn0 = fft.planSize(@max(w, h), 1);
        self.ensure(n, nn0);
        e.promoteTexels(gray, self.gray32, @intCast(n), 0, 0);
        if (rot_deg != 0) {
            const t = rot_deg * std.math.pi / 180;
            self.run("pg_rotate", (w + 7) / 8, (h + 7) / 8, 1, .{ .w = w, .h = h, .cth = @floatCast(@cos(t)), .sth = @floatCast(@sin(t)) }, &.{ self.gray32.at(1), self.lv_b.at(2) });
            g.copy(self.lv_b, 0, self.gray32, 0, n * 4);
        }
        // Pyramid (MultiScale): chains from the image and from its 2/3 copy, interleaved.
        var made: std.ArrayList(Buf) = .empty;
        defer {
            for (made.items) |*b| g.release(b);
            made.deinit(gpa);
        }
        var imgs: std.ArrayList(Img) = .empty;
        defer imgs.deinit(gpa);
        var a: Img = .{ .buf = self.gray32, .w = w, .h = h, .scale = 1 };
        var b: Img = undefined;
        {
            const buf = g.storage(n * 4);
            try made.append(gpa, buf);
            const wh = self.resize(self.gray32, w, h, 2.0 / 3.0, buf, self.lv_a);
            b = .{ .buf = buf, .w = wh[0], .h = wh[1], .scale = 1 };
        }
        var o: u32 = 0;
        while (o < st.n_octaves) : (o += 1) {
            const s2 = std.math.pow(f64, 2, @floatFromInt(o));
            try imgs.append(gpa, .{ .buf = a.buf, .w = a.w, .h = a.h, .scale = s2 });
            try imgs.append(gpa, .{ .buf = b.buf, .w = b.w, .h = b.h, .scale = 1.5 * s2 });
            if (o == st.n_octaves - 1) break;
            const na = g.storage(@as(u64, (a.w + 1) / 2) * ((a.h + 1) / 2) * 4);
            try made.append(gpa, na);
            const awh = self.resize(a.buf, a.w, a.h, 0.5, na, self.lv_a);
            const nb = g.storage(@as(u64, (b.w + 1) / 2) * ((b.h + 1) / 2) * 4);
            try made.append(gpa, nb);
            const bwh = self.resize(b.buf, b.w, b.h, 0.5, nb, self.lv_a);
            a = .{ .buf = na, .w = awh[0], .h = awh[1], .scale = 0 };
            b = .{ .buf = nb, .w = bwh[0], .h = bwh[1], .scale = 0 };
            if (@min(awh[0], awh[1]) < 16) {
                imgs.shrinkRetainingCapacity(2 * (o + 1));
                break;
            }
            if (@min(bwh[0], bwh[1]) < 16) {
                try imgs.append(gpa, .{ .buf = a.buf, .w = a.w, .h = a.h, .scale = std.math.pow(f64, 2, @floatFromInt(o + 1)) });
                break;
            }
        }
        const rr = radii(st.p1, S.nring);
        const border = rr.R[S.nring - 1] + 1;
        var det: Det = .{ .w = w, .h = h };
        errdefer det.deinit(gpa);
        const offs_b = try self.offs(st.p1);
        // levels run at the GPU's running offset (gls level_done); their counts come back together
        var ran: std.ArrayList(Img) = .empty;
        defer ran.deinit(gpa);
        e.beginLevels();
        const extras = st.rotation == .released or st.rotation == .paper;
        for (imgs.items) |lv| {
            if (@as(f64, @floatFromInt(@min(lv.w, lv.h))) <= 2 * border + 2) continue;
            const s = try e.ensure(lv.w, lv.h);
            try self.phaseCong(lv.buf, lv.w, lv.h, s.fmap);
            // Keep the full-resolution Σ|EO| for the POS re-description.
            if (lv.scale == 1) {
                const need: u64 = @as(u64, S.no) * n;
                if (sb.cs0.id == 0 or sb.cs0n < need) {
                    g.release(&sb.cs0);
                    sb.cs0 = g.storage(need * 4);
                    sb.cs0n = need;
                }
                g.copy(self.cs, 0, sb.cs0, 0, need * 4);
            }
            // Corners: the GLS-MIFT detector (FAST ring test, 3 × 3 maxima, strongest per grid cell).
            e.p.w = lv.w;
            e.p.h = lv.h;
            e.p.scale = @floatCast(lv.scale);
            e.p.kp_offset = gls_mod.KP_RUNNING;
            e.p.max_points = st.max_points;
            e.p.grid = @max(1, @as(u32, @intFromFloat(@ceil(@sqrt(@as(f64, @floatFromInt(st.max_points)))))));
            e.p.min_contrast = @floatCast(st.min_contrast);
            var q = e.p;
            q.border = @intFromFloat(@trunc(border));
            e.clearCounts();
            g.clear(e.cell_best);
            g.clear(e.cell_pix);
            g.clear(e.cell_kp);
            g.clear(s.scores);
            const gx = (lv.w + 7) / 8;
            const gy = (lv.h + 7) / 8;
            e.run("fast_nms", gx, gy, &.{ e.Pwith(q), s.fmap.at(3), s.scores.at(5) });
            e.run("nms_and_cells", gx, gy, &.{ e.Pwith(q), s.scores.at(5), e.cell_best.at(6) });
            e.run("cell_tie", gx, gy, &.{ e.Pwith(q), s.scores.at(5), e.cell_best.at(6), e.cell_pix.at(19) });
            e.run("emit_kps", gx, gy, &.{ e.Pwith(q), s.scores.at(5), e.cell_best.at(6), e.cell_kp.at(17), e.cell_pix.at(19) });
            e.run("compact_kps", 1, 1, &.{ e.Pwith(q), kps.at(7), e.counters.at(9), e.cell_kp.at(17) });
            // Descriptor modes: 0 upright raw, 4 upright with unit sampled points, 2 paper direction,
            // 3 the released default. Channels: released range, paper spectral, else desc_norm.
            const norm: DescNorm = switch (st.rotation) {
                .released => .range,
                .paper => .spectral,
                else => st.desc_norm,
            };
            const mode: u32 = switch (st.rotation) {
                .paper => 2,
                .released => 3,
                else => if (norm == .raw) 0 else 4,
            };
            if (norm != .raw) self.spectralNorms(lv.w, lv.h);
            const nk = (st.max_points + 63) / 64;
            const staged = mode == 0 or mode == 4;
            const p: U = .{ .w = lv.w, .h = lv.h, .kp_offset = gls_mod.KP_RUNNING, .mode = mode, .scale = @floatCast(lv.scale), .ma = 0, .dflags = if (staged) STAGED else 0 };
            const sbinds = [_]Bind{ self.gm.at(13), offs_b.at(14), kps.at(15), sb.des.at(16), e.counters.at(18) };
            const stage: ?Stage = if (staged) .{ .p = p, .groups = nk, .binds = &sbinds } else null;
            if (norm == .range) {
                // the range normalization fills every orientation's plane of en
                g.ensure(&self.en, @as(u64, S.no) * self.cap * 4);
                self.run("pg_rangenorm", gx, gy, 1, .{ .w = lv.w, .h = lv.h }, &.{ self.cs.at(5), self.en.at(6), self.sig.at(12) });
                self.ringMaps(self.en, lv.w, lv.h, st.p1, 0, stage);
            } else {
                self.ringMaps(self.cs, lv.w, lv.h, st.p1, if (norm == .spectral) 2 else 0, stage);
            }
            const res = [_]Bind{ self.gm.at(13), offs_b.at(14), kps.at(15), sb.des.at(16), sb.ori.at(17), e.counters.at(18), sb.desq.at(19), sb.dsc.at(21) };
            if (mode == 3) {
                self.run("pg_sobel", gx, gy, 1, .{ .w = lv.w, .h = lv.h }, &.{ self.pc.at(1), self.wt.at(7), self.mm.at(10), self.gbin.at(27) });
                const dx = @min(st.max_points, 32768);
                const dy = (st.max_points + dx - 1) / dx;
                self.run("pg_orient_robust", dx, dy, 1, p, &.{ self.wt.at(7), kps.at(15), sb.ori.at(17), e.counters.at(18), self.gbin.at(27) });
                self.run("pg_compact_robust", 1, 1, 1, p, &.{ sb.ori.at(17), e.counters.at(18) });
                self.run("pg_describe_robust", nk, 1, 1, p, &res);
            } else if (mode == 2) {
                self.run("pg_orient_kp", nk, 1, 1, p, &.{ self.gm.at(13), offs_b.at(14), kps.at(15), sb.ori.at(17), e.counters.at(18) });
                self.run("pg_compact", 1, 1, 1, p, &.{ sb.ori.at(17), e.counters.at(18) });
            }
            if (mode != 3) self.run("pg_describe", nk, 1, 1, p, &res);
            e.p.max_points = st.max_points;
            e.levelDone(@intCast(ran.items.len), extras);
            try ran.append(gpa, lv);
        }
        const rec = try e.readLevels(@intCast(ran.items.len));
        var total: u32 = 0;
        for (ran.items, 0..) |lv, i| {
            const count = rec[4 * i + 1];
            try det.levels.append(gpa, .{ .offset = total, .count = count, .base = rec[4 * i + 2], .scale = lv.scale, .w = lv.w, .h = lv.h });
            total += count;
            if (total >= MAX_KP) break;
        }
        det.n = total;
        return det;
    }

    /// Ratio-tested 1-NN of every level pair (queries: image 1), pooled in the engine's corr list
    /// (octave: each pair's correspondences within its own affine FSC).
    fn matchPooled(self: *PosGift, L: *const Det, R: *const Det, n_trials: u32, max_scale: f64) !void {
        const g = self.g;
        const e = self.gls;
        const st = self.set;
        const A = self.side(.a);
        const B = self.side(.b);
        g.clear(e.counters);
        const unset = try g.gpa.alloc(u32, MAX_KP);
        defer g.gpa.free(unset);
        @memset(unset, 0xffffffff);
        g.writeSlice(e.match_j, 0, u32, unset);
        const seed0 = e.p.seed;
        var pair: u32 = 0;
        e.p.src_w = L.w; // image 1's size: coverage cells (fsc_score)
        e.p.src_h = L.h;
        e.p.n_trials = n_trials;
        for (L.levels.items) |a| for (R.levels.items) |b| {
            if (a.count < 3 or b.count < 3 or a.scale > max_scale or b.scale > max_scale) continue;
            e.p.n_query = a.count;
            e.p.n_db = b.count;
            e.p.q_offset = a.offset;
            e.p.db_offset = b.offset;
            const nq = a.count;
            const nd = b.count;
            const nr = gls_mod.Gls.nnRows(nq, nd);
            const chunk = nr.chunk;
            const rows = nr.rows;
            g.ensure(&e.nn_part, @as(u64, nq) * rows * 16);
            var q = e.p;
            q.nn_chunk = chunk;
            q.nn_sub = nr.sub;
            q.ratio = @floatCast(st.ratio);
            q.max_ssd = @floatCast(st.max_ssd);
            self.mrun("match_nnq", (nq + 63) / 64, rows, q, &.{ e.kps.at(7), A.desq.at(20), B.desq.at(21), A.dsc.at(22), B.dsc.at(31), e.nn_part.at(29) });
            self.mrun(if (st.match == .matlab) "match_nnq_pick_ratio" else "match_nnq_pick", (nq + 63) / 64, 1, q, &.{ e.kps.at(7), A.des.at(8), B.des.at(13), e.match_j.at(14), e.nn_part.at(29) });
            if (st.match == .octave) {
                // GLS-MIFT's per-pair model: affine FSC at 10 px, keep that pair's correspondences within it.
                const user2 = e.p.inlier2;
                e.p.inlier2 = 100;
                e.p.seed = seed0 +% pair *% 10007;
                pair += 1;
                e.fscTrials(nq, n_trials);
                e.fscReduce();
                e.p.inlier2 = @max(user2, 100);
                e.run("keep_corr", (nq + 63) / 64, 1, &.{ e.P(), e.kps.at(7), e.counters.at(9), e.kps_b.at(12), e.match_j.at(14), e.affine.at(16), e.corr.at(28) });
                e.p.inlier2 = user2;
                continue;
            }
            var q2 = e.p;
            q2.nn_chunk = chunk;
            q2.ratio = @floatCast(st.ratio);
            q2.max_ssd = @floatCast(st.max_ssd);
            self.mrun("append_corr", (nq + 63) / 64, 1, q2, &.{ e.kps.at(7), e.counters.at(9), e.kps_b.at(12), e.match_j.at(14), e.corr.at(28) });
        };
        e.p.seed = seed0;
    }

    const Fit = struct { pooled: u32, ninl_aff: u32, Haff: lie.Mat3 };

    /// One global affine FSC over the pooled correspondences of matchPooled.
    fn fitPooled(self: *PosGift, dl: *const Det, dr: *const Det, n_trials: u32) !Fit {
        const e = self.gls;
        e.p.n_trials = n_trials;
        const ninl = try e.fitCorr(dl.n, dr.n, .{ .method = .lofsc, .homography = false }, n_trials);
        return .{ .pooled = e.corr_count, .ninl_aff = ninl, .Haff = try e.readH() };
    }

    /// Engine parameters POS-GIFT reads (shared with GLS-MIFT) for the duration of a stage.
    fn enterParams(self: *PosGift) gls_mod.Params {
        const saved = self.gls.p;
        const s = self.set;
        self.gls.p.inlier2 = @floatCast(s.inlier_px * s.inlier_px);
        self.gls.p.seed = s.seed;
        self.gls.p.scale_lo = 0.2;
        self.gls.p.scale_hi = 5;
        self.gls.p.flags = if (s.fsc_cover) 4 else 0;
        self.gls.p.nt = 0;
        return saved;
    }

    /// Keypoints and descriptors of both images (engine kps / kps_b and this module's buffers).
    /// With rotation search also image 2 turned by half a step and a copy of its upright
    /// descriptors, so match can be repeated.
    pub fn detectPair(self: *PosGift, L: Image, R: Image) !Detection {
        try self.setStructure();
        const saved = self.enterParams();
        defer self.gls.p = saved;
        const g = self.g;
        const e = self.gls;
        const st = self.set;
        var d: Detection = .{ .rotation = st.rotation, .structure = self.S };
        errdefer d.deinit(g.gpa);
        d.dl = try self.detect(L.buf, L.w, L.h, e.kps, .a, 0);
        d.dr = try self.detect(R.buf, R.w, R.h, e.kps_b, .b, 0);
        if (st.rotation == .search) {
            const b = self.side(.b);
            const b0 = self.side(.b0);
            g.copy(b.des, 0, b0.des, 0, @as(u64, d.dr.n) * self.S.dp * 4);
            // (its Σ|EO| stays side b's: cs0Of)
            d.frames[0] = .{ .deg = 0, .det = 0, .kps = e.kps_b, .side = .b0 };
            d.n_frames = 1;
            if (st.search_half) {
                if (self.kps_b15.id == 0) self.kps_b15 = g.storage(@as(u64, MAX_KP) * 16);
                const half = 180.0 / @as(f64, @floatFromInt(self.S.ndir));
                d.dr15 = try self.detect(R.buf, R.w, R.h, self.kps_b15, .b15, half);
                d.frames[1] = .{ .deg = half, .det = 1, .kps = self.kps_b15, .side = .b15 };
                d.n_frames = 2;
            }
        }
        return d;
    }

    /// Match and fit a detectPair record (repeatable). H maps moving → fixed, 1-based pixels.
    pub fn match(self: *PosGift, L: Image, R: Image, d: *const Detection) !MatchOut {
        if (d.rotation != self.set.rotation) return error.PosGiftRotationChangedSinceDetect;
        try self.setStructure();
        if (!std.meta.eql(d.structure, self.S)) return error.PosGiftStructureChangedSinceDetect;
        const saved = self.enterParams();
        defer self.gls.p = saved;
        const st = self.set;
        self.last = .{ .n1 = d.dl.n, .n2 = d.dr.n, .kps_b = self.gls.kps_b };
        if (st.rotation == .search) return self.registerSearch(L, R, d);
        try self.matchPooled(&d.dl, &d.dr, st.n_trials, std.math.inf(f64));
        const fit = try self.fitPooled(&d.dl, &d.dr, st.n_trials);
        var out: MatchOut = .{ .H = fit.Haff, .Haff = fit.Haff, .pooled = fit.pooled, .ninl_aff = fit.ninl_aff };
        if (st.pos) try self.pos(&d.dl, &d.dr, fit.Haff, L, R, .b, &out, null);
        return out;
    }

    const Hyp = struct { frame: u32, k: u32, deg: f64, coarse: u32, full: ?u32 = null };

    /// Rotation-searched POS-GIFT: image 2's upright descriptors (and those of its half-step turned
    /// copy) turned by k directions by permutation (pg_shift), hypotheses ranked on the finest
    /// levels, the best verified at full resolution, POS in the winning frame, H mapped back.
    fn registerSearch(self: *PosGift, L: Image, R: Image, d: *const Detection) !MatchOut {
        const g = self.g;
        const gpa = g.gpa;
        const e = self.gls;
        const st = self.set;
        const S = self.S;
        const kps_b0 = e.kps_b;
        defer e.kps_b = kps_b0;
        const B = self.side(.b);
        var hyp: std.ArrayList(Hyp) = .empty;
        defer hyp.deinit(gpa);
        const frames = d.frames[0..d.n_frames];
        const step = 360.0 / @as(f64, @floatFromInt(S.ndir));
        if (st.search_global) {
            for (frames, 0..) |f, fi| {
                var k: u32 = 0;
                while (k < S.ndir) : (k += 1) {
                    self.use(f, d, k, B);
                    try self.matchPooled(&d.dl, d.frameDet(f), st.search_trials, st.search_max_scale);
                    const r = try self.fitPooled(&d.dl, d.frameDet(f), st.search_trials);
                    try hyp.append(gpa, .{ .frame = @intCast(fi), .k = k, .deg = f.deg + step * @as(f64, @floatFromInt(k)), .coarse = r.ninl_aff });
                }
            }
        } else {
            // Per-pair consensus summed over the finest level pairs; every hypothesis in one batch,
            // its count copied out of counters[3], one readback.
            const nh: u32 = @intCast(frames.len * S.ndir);
            g.ensure(&self.score_buf, @as(u64, nh) * 16);
            for (frames, 0..) |f, fi| {
                var k: u32 = 0;
                while (k < S.ndir) : (k += 1) {
                    self.use(f, d, k, B);
                    try self.matchPooled(&d.dl, d.frameDet(f), st.search_trials, st.search_max_scale);
                    g.copy(e.counters, 0, self.score_buf, (@as(u64, @intCast(fi)) * S.ndir + k) * 16, 16);
                }
            }
            const sc = try e.readU32(self.score_buf, nh * 4);
            for (frames, 0..) |f, fi| {
                var k: u32 = 0;
                while (k < S.ndir) : (k += 1) try hyp.append(gpa, .{ .frame = @intCast(fi), .k = k, .deg = f.deg + step * @as(f64, @floatFromInt(k)), .coarse = sc[(fi * S.ndir + k) * 4 + 3] });
            }
        }
        std.sort.block(Hyp, hyp.items, {}, struct {
            fn gt(_: void, a: Hyp, b: Hyp) bool {
                return a.coarse > b.coarse;
            }
        }.gt);
        const keep: usize = @min(hyp.items.len, if (hyp.items[0].coarse >= 2 * hyp.items[1].coarse) 1 else st.search_keep);
        var best_i: usize = 0;
        var best: ?Fit = null;
        for (0..keep) |i| {
            const h = &hyp.items[i];
            const f = frames[h.frame];
            self.use(f, d, h.k, B);
            try self.matchPooled(&d.dl, d.frameDet(f), st.n_trials, std.math.inf(f64));
            const r = try self.fitPooled(&d.dl, d.frameDet(f), st.n_trials);
            h.full = r.ninl_aff;
            if (best == null or r.ninl_aff > best.?.ninl_aff) {
                best = r;
                best_i = i;
            }
        }
        var r = best.?;
        const h = hyp.items[best_i];
        const f = frames[h.frame];
        if (best_i != keep - 1) {
            // The engine's match list and model belong to the last hypothesis verified: redo the winner.
            self.use(f, d, h.k, B);
            try self.matchPooled(&d.dl, d.frameDet(f), st.n_trials, std.math.inf(f64));
            r = try self.fitPooled(&d.dl, d.frameDet(f), st.n_trials);
        }
        // Frame turn (1-based pixel coordinates): x' = Rf (x − c) + c; H = Rf⁻¹ · H_frame.
        const t = f.deg * std.math.pi / 180;
        const c = @cos(t);
        const sn = @sin(t);
        const cx = (@as(f64, @floatFromInt(R.w)) - 1) / 2 + 1;
        const cy = (@as(f64, @floatFromInt(R.h)) - 1) / 2 + 1;
        const Rf: lie.Mat3 = .{ c, -sn, cx - c * cx + sn * cy, sn, c, cy - sn * cx - c * cy, 0, 0, 1 };
        const Rinv = lie.inv3(Rf);
        const back = struct {
            fn apply(Ri: lie.Mat3, M: lie.Mat3) lie.Mat3 {
                var o = lie.mul3(Ri, M);
                const z = o[8];
                for (&o) |*v| v.* /= z;
                return o;
            }
        }.apply;
        self.last = .{ .n1 = d.dl.n, .n2 = d.frameDet(f).n, .kps_b = f.kps, .back = if (f.deg != 0) Rinv else null, .dirs = if (f.side == .b0) .b else f.side, .turn = h.k, .deg = f.deg };
        var out: MatchOut = .{ .Haff = back(Rinv, r.Haff), .H = back(Rinv, r.Haff), .pooled = r.pooled, .ninl_aff = r.ninl_aff, .rot_deg = h.deg, .frame_deg = f.deg };
        if (st.pos) try self.pos(&d.dl, d.frameDet(f), r.Haff, L, R, f.side, &out, Rinv);
        return out;
    }

    /// Descriptor frames of a side's keypoints (the engine's keypoints() order, image 2 in the last
    /// match's frame when matched): each slot's recorded direction (direction a samples at
    /// −R (cos t, sin t), t = 2π a / ndir) and the outer ring with half its Gaussian.
    pub fn kpFrames(self: *PosGift, side_i: u32, matched: bool, kps: []const gls_mod.Kp, out: []gls_mod.KpFrame) !void {
        if (kps.len == 0) return;
        const g = self.g;
        var id: SideId = if (side_i == 0) .a else .b;
        var turn: u32 = 0;
        var deg: f64 = 0;
        if (side_i == 1 and matched) {
            id = self.last.dirs;
            turn = self.last.turn;
            deg = self.last.deg;
        }
        const dirs = try g.gpa.alloc(u32, kps.len);
        defer g.gpa.free(dirs);
        g.read(self.side(id).ori, DIR_REC * 4, std.mem.sliceAsBytes(dirs));
        try g.wait();
        const nd: f64 = @floatFromInt(self.S.ndir);
        const rr = radii(self.set.p1, self.S.nring);
        const outer = rr.R[rr.n - 1] + @as(f64, @floatFromInt(rr.G[rr.n - 1])) / 2;
        for (kps, dirs, out) |k, d, *o| {
            const t = @as(f64, @floatFromInt(d + turn)) / nd * 2 * std.math.pi + std.math.pi - deg * std.math.pi / 180;
            o.* = .{ .angle = @floatCast(t), .radius = @floatCast(outer * k.pad) };
        }
    }

    /// Image 2's descriptors of frame f turned by k directions into side b (and its keypoints in use).
    fn use(self: *PosGift, f: Frame, d: *const Detection, k: u32, B: *SideBufs) void {
        self.gls.kps_b = f.kps;
        const n = d.frameDet(f).n;
        const src = self.side(f.side);
        self.run("pg_shift", (n + 63) / 64, 1, 1, .{ .n_kp = n, .ma = k }, &.{ src.des.at(23), B.des.at(16), B.desq.at(19), B.dsc.at(21) });
    }

    /// POS: re-describe full-resolution keypoints with P1 = pos_p1 at the affine's rotation / scale
    /// and re-match near H·p; then a perspective FSC, inliers within 3 px unique on both sides, and
    /// the final least-squares model. Sets out.H (mapped back by `back` when given).
    fn pos(self: *PosGift, dl: *const Det, dr: *const Det, H: lie.Mat3, L: Image, R: Image, side2: SideId, out: *MatchOut, back_m: ?lie.Mat3) !void {
        const g = self.g;
        const gpa = g.gpa;
        const e = self.gls;
        const st = self.set;
        const S = self.S;
        out.pos_ran = true;
        const a = H[0];
        const b = H[3];
        const c = H[1];
        const dd = H[4];
        const rotation = std.math.atan2(b, a) / std.math.pi * 360;
        const sx = lie.jsHypot(a, b);
        const sy = (a * dd - b * c) / lie.jsHypot(a, b);
        var scale = lie.jsRound(@abs(sx) + @abs(sy)) / 2;
        if (scale == 0) scale = 1;
        // rotation is twice the angle in degrees; angle in directions (360 / ndir each)
        const nd: f64 = @floatFromInt(S.ndir);
        var angle = lie.mround(rotation / 720 * nd);
        while (angle < 0) angle += nd;
        while (angle > nd) angle -= nd;
        const s1: f64 = if (scale > 0.25) 1 else 1 / scale;
        const s2: f64 = if (scale > 0.25) scale else 1;
        const lv1 = dl.scale1() orelse return;
        const lv2 = dr.scale1() orelse return;
        const n1 = lv1.base;
        const n2 = lv2.base;
        // both keypoint lists in one readback
        const k1 = try gpa.alloc(f32, @max(1, (lv1.offset + lv1.base) * 4));
        defer gpa.free(k1);
        const k2 = try gpa.alloc(f32, @max(1, (lv2.offset + lv2.base) * 4));
        defer gpa.free(k2);
        g.read(e.kps, 0, std.mem.sliceAsBytes(k1));
        g.read(e.kps_b, 0, std.mem.sliceAsBytes(k2));
        try g.wait();
        const A = try gpa.alloc(f32, @max(1, n1) * 4);
        defer gpa.free(A);
        const Bl = try gpa.alloc(f32, @max(1, n2) * 4);
        defer gpa.free(Bl);
        const Pp = try gpa.alloc(f32, @max(1, n1) * 4);
        defer gpa.free(Pp);
        for (0..n1) |i| {
            const src = (lv1.offset + i) * 4;
            A[i * 4] = k1[src];
            A[i * 4 + 1] = k1[src + 1];
            A[i * 4 + 2] = 1;
            A[i * 4 + 3] = 1;
        }
        for (0..n2) |i| {
            const src = (lv2.offset + i) * 4;
            Bl[i * 4] = k2[src];
            Bl[i * 4 + 1] = k2[src + 1];
            Bl[i * 4 + 2] = 1;
            Bl[i * 4 + 3] = 1;
        }
        for (0..n1) |i| {
            const x: f64 = A[i * 4];
            const y: f64 = A[i * 4 + 1];
            const z = H[6] * x + H[7] * y + H[8];
            Pp[i * 4] = @floatCast(lie.mround((H[0] * x + H[1] * y + H[2]) / z));
            Pp[i * 4 + 1] = @floatCast(lie.mround((H[3] * x + H[4] * y + H[5]) / z));
            Pp[i * 4 + 2] = 1;
            Pp[i * 4 + 3] = 1;
        }
        const mk = struct {
            fn f(gg: *Gpu, arr: []const f32, n: u32) Buf {
                const bf = gg.storage(@as(u64, @max(1, n)) * 16);
                gg.writeSlice(bf, 0, f32, arr[0 .. @as(usize, n) * 4]);
                return bf;
            }
        }.f;
        var bufA = mk(g, A, n1);
        var bufB = mk(g, Bl, n2);
        var bufP = mk(g, Pp, n1);
        var dA = g.storage(@as(u64, @max(1, n1)) * S.dp * 4);
        var dB = g.storage(@as(u64, @max(1, n2)) * S.dp * 4);
        var dP = g.storage(@as(u64, @max(1, n1)) * S.dp * 4);
        var jq = g.storage(@as(u64, @max(@max(n1, n2), 1)) * S.dq * 4);
        var js = g.storage(@as(u64, @max(@max(n1, n2), 1)) * 4);
        var resB = g.storage(@as(u64, @max(1, n1)) * 16);
        defer for ([_]*Buf{ &bufA, &bufB, &bufP, &dA, &dB, &dP, &jq, &js, &resB }) |bf| g.release(bf);
        const sa = self.side(.a);
        const cs0b = self.cs0Of(side2);
        const p1a = st.pos_p1 * s1;
        const p1b = st.pos_p1 * s2;
        const ang: u32 = @intFromFloat(angle);
        const Desc = struct {
            fn run(pg: *PosGift, cs: Buf, w: u32, h: u32, p1: f64, ma: u32, kbuf: Buf, n: u32, dbuf: Buf, ori: Buf, q: Buf, s: Buf) !void {
                const ob = try pg.offs(p1);
                const p: U = .{ .w = w, .h = h, .n_kp = n, .mode = 1, .ma = ma, .scale = 1, .kp_offset = 0, .dflags = STAGED };
                const sb = [_]Bind{ pg.gm.at(13), ob.at(14), kbuf.at(15), dbuf.at(16), pg.gls.counters.at(18) };
                pg.ringMaps(cs, w, h, p1, 1, .{ .p = p, .groups = (n + 63) / 64, .binds = &sb });
                pg.run("pg_describe", (n + 63) / 64, 1, 1, p, &.{ pg.gm.at(13), ob.at(14), kbuf.at(15), dbuf.at(16), ori.at(17), pg.gls.counters.at(18), q.at(19), s.at(21) });
            }
        };
        try Desc.run(self, sa.cs0, L.w, L.h, p1a, 0, bufA, n1, dA, sa.ori, jq, js);
        if (st.pos_mode == .dense) {
            self.ringMaps(cs0b, R.w, R.h, p1b, 1, null);
            const ob = try self.offs(p1b);
            self.run("pg_pos_dense", (n1 + 63) / 64, 1, 1, .{ .w = R.w, .h = R.h, .n_kp = n1, .ma = ang, .hsize = st.pos_win }, &.{ self.gm.at(13), ob.at(14), bufA.at(15), dA.at(16), bufP.at(24), resB.at(26) });
        } else {
            try Desc.run(self, cs0b, R.w, R.h, p1b, ang, bufB, n2, dB, sa.ori, jq, js);
            try Desc.run(self, cs0b, R.w, R.h, p1b, ang, bufP, n1, dP, sa.ori, jq, js);
            const thr = @max(20, @as(f64, @floatFromInt(@min(R.w, R.h))) / 20);
            self.run("pg_pos_pick", (n1 + 63) / 64, 1, 1, .{ .n_kp = n1, .n_b = n2, .thr2 = @floatCast(thr * thr) }, &.{ bufA.at(15), dA.at(16), bufB.at(22), dB.at(23), bufP.at(24), dP.at(25), resB.at(26) });
        }
        const res = try gpa.dupe(f32, try e.readF32(resB, @as(usize, n1) * 4));
        defer gpa.free(res);
        var px: std.ArrayList(f64) = .empty;
        defer px.deinit(gpa);
        var py: std.ArrayList(f64) = .empty;
        defer py.deinit(gpa);
        var qx: std.ArrayList(f64) = .empty;
        defer qx.deinit(gpa);
        var qy: std.ArrayList(f64) = .empty;
        defer qy.deinit(gpa);
        var chose_pred: u32 = 0;
        for (0..n1) |i| {
            if (res[i * 4 + 2] <= 0) continue;
            try px.append(gpa, A[i * 4]);
            try py.append(gpa, A[i * 4 + 1]);
            try qx.append(gpa, res[i * 4]);
            try qy.append(gpa, res[i * 4 + 1]);
            if (res[i * 4 + 3] == @as(f32, @floatFromInt(S.knn))) chose_pred += 1;
        }
        const m = px.items.len;
        out.pos_n = @intCast(m);
        if (m < 4) return;
        // FSC(…, 'perspective', 10), refit on its inliers; then inliers within 3 px, unique on both sides.
        const mask = try gpa.alloc(u8, m);
        defer gpa.free(mask);
        _ = try ransacHomography(gpa, px.items, py.items, qx.items, qy.items, 10, st.n_trials, 1, mask);
        var s10: std.ArrayList([2]f64) = .empty;
        defer s10.deinit(gpa);
        var d10: std.ArrayList([2]f64) = .empty;
        defer d10.deinit(gpa);
        for (0..m) |i| if (mask[i] != 0) {
            try s10.append(gpa, .{ px.items[i], py.items[i] });
            try d10.append(gpa, .{ qx.items[i], qy.items[i] });
        };
        if (s10.items.len < 4) return;
        const Hh = try homographyLsq(gpa, s10.items, d10.items);
        var seen1: std.AutoHashMapUnmanaged([2]u64, void) = .empty;
        defer seen1.deinit(gpa);
        var seen2: std.AutoHashMapUnmanaged([2]u64, void) = .empty;
        defer seen2.deinit(gpa);
        var src: std.ArrayList([2]f64) = .empty;
        defer src.deinit(gpa);
        var dst: std.ArrayList([2]f64) = .empty;
        defer dst.deinit(gpa);
        for (0..m) |i| {
            const x = px.items[i];
            const y = py.items[i];
            const z = Hh[6] * x + Hh[7] * y + Hh[8];
            const ex = (Hh[0] * x + Hh[1] * y + Hh[2]) / z - qx.items[i];
            const ey = (Hh[3] * x + Hh[4] * y + Hh[5]) / z - qy.items[i];
            if (ex * ex + ey * ey >= 9) continue;
            // `${x},${y}` keys: +0 and −0 print alike
            const k1s: [2]u64 = .{ @bitCast(x + 0.0), @bitCast(y + 0.0) };
            const k2s: [2]u64 = .{ @bitCast(qx.items[i] + 0.0), @bitCast(qy.items[i] + 0.0) };
            if (seen1.contains(k1s) or seen2.contains(k2s)) continue;
            try seen1.put(gpa, k1s, {});
            try seen2.put(gpa, k2s, {});
            try src.append(gpa, .{ x, y });
            try dst.append(gpa, .{ qx.items[i], qy.items[i] });
        }
        const Hpos = if (src.items.len < 4) Hh else if (st.pos_affine) lie.HAffineFromPts(src.items, dst.items, null) else try homographyLsq(gpa, src.items, dst.items);
        out.pos_inl = @intCast(src.items.len);
        out.pos_pred = chose_pred;
        out.pos_h = true;
        if (back_m) |Ri| {
            var o = lie.mul3(Ri, Hpos);
            const z = o[8];
            for (&o) |*v| v.* /= z;
            out.H = o;
        } else out.H = Hpos;
    }
};

/// Least-squares homography src → dst on Hartley-normalized coordinates (mean 0, mean distance √2).
fn homographyLsq(gpa: std.mem.Allocator, src: []const [2]f64, dst: []const [2]f64) !lie.Mat3 {
    const Norm = struct {
        T: lie.Mat3,
        p: [][2]f64,
        fn of(al: std.mem.Allocator, pts: []const [2]f64) !@This() {
            var mx: f64 = 0;
            var my: f64 = 0;
            for (pts) |q| {
                mx += q[0];
                my += q[1];
            }
            const n: f64 = @floatFromInt(pts.len);
            mx /= n;
            my /= n;
            var d: f64 = 0;
            for (pts) |q| d += lie.jsHypot(q[0] - mx, q[1] - my);
            const s = std.math.sqrt2 / @max(d / n, 1e-12);
            const p = try al.alloc([2]f64, pts.len);
            for (pts, 0..) |q, i| p[i] = .{ s * (q[0] - mx), s * (q[1] - my) };
            return .{ .T = .{ s, 0, -s * mx, 0, s, -s * my, 0, 0, 1 }, .p = p };
        }
    };
    const a = try Norm.of(gpa, src);
    defer gpa.free(a.p);
    const b = try Norm.of(gpa, dst);
    defer gpa.free(b.p);
    const Hn = lie.HHomographyFromPts(a.p, b.p, null);
    var H = lie.mul3(lie.inv3(b.T), lie.mul3(Hn, a.T));
    const z = H[8];
    for (&H) |*v| v.* /= z;
    return H;
}

/// Homography through four correspondences (8 × 8 elimination), or null if degenerate.
fn h4(px: []const f64, py: []const f64, qx: []const f64, qy: []const f64, id: [4]usize) ?[9]f64 {
    var A: [64]f64 = @splat(0);
    var b: [8]f64 = @splat(0);
    for (0..4) |k| {
        const x = px[id[k]];
        const y = py[id[k]];
        const uu = qx[id[k]];
        const v = qy[id[k]];
        const r0 = [8]f64{ x, y, 1, 0, 0, 0, -uu * x, -uu * y };
        const r1 = [8]f64{ 0, 0, 0, x, y, 1, -v * x, -v * y };
        @memcpy(A[(2 * k) * 8 ..][0..8], &r0);
        @memcpy(A[(2 * k + 1) * 8 ..][0..8], &r1);
        b[2 * k] = uu;
        b[2 * k + 1] = v;
    }
    for (0..8) |i| {
        var piv = i;
        for (i + 1..8) |r| if (@abs(A[r * 8 + i]) > @abs(A[piv * 8 + i])) {
            piv = r;
        };
        if (@abs(A[piv * 8 + i]) < 1e-10) return null;
        if (piv != i) {
            for (0..8) |j| std.mem.swap(f64, &A[i * 8 + j], &A[piv * 8 + j]);
            std.mem.swap(f64, &b[i], &b[piv]);
        }
        for (i + 1..8) |r| {
            const f = A[r * 8 + i] / A[i * 8 + i];
            if (f == 0 or std.math.isNan(f)) continue;
            for (i..8) |j| A[r * 8 + j] -= f * A[i * 8 + j];
            b[r] -= f * b[i];
        }
    }
    var h: [9]f64 = @splat(0);
    var i: usize = 8;
    while (i > 0) {
        i -= 1;
        var t = b[i];
        for (i + 1..8) |j| t -= A[i * 8 + j] * h[j];
        h[i] = t / A[i * 8 + i];
    }
    h[8] = 1;
    return h;
}

/// FSC-style RANSAC for a homography (the released code's FSC(…, 'perspective', 10)): random
/// 4-point models on Hartley-normalized points, inliers within thr px, up to max_trials with the
/// usual adaptive stop (99.9 % confidence). The best model's inlier mask into mask; returns its count.
fn ransacHomography(gpa: std.mem.Allocator, px: []const f64, py: []const f64, qx: []const f64, qy: []const f64, thr: f64, max_trials: u32, seed: u32, mask_out: []u8) !u32 {
    const n = px.len;
    var st: u32 = if (seed == 0) 1 else seed;
    const Nz = struct {
        s: f64,
        x: []f64,
        y: []f64,
        fn of(al: std.mem.Allocator, xs: []const f64, ys: []const f64) !@This() {
            var mx: f64 = 0;
            var my: f64 = 0;
            for (xs, ys) |x, y| {
                mx += x;
                my += y;
            }
            const nf: f64 = @floatFromInt(xs.len);
            mx /= nf;
            my /= nf;
            var d: f64 = 0;
            for (xs, ys) |x, y| d += lie.jsHypot(x - mx, y - my);
            const s = std.math.sqrt2 / @max(d / nf, 1e-12);
            const ox = try al.alloc(f64, xs.len);
            const oy = try al.alloc(f64, xs.len);
            for (xs, ys, 0..) |x, y, i| {
                ox[i] = s * (x - mx);
                oy[i] = s * (y - my);
            }
            return .{ .s = s, .x = ox, .y = oy };
        }
    };
    const a = try Nz.of(gpa, px, py);
    defer {
        gpa.free(a.x);
        gpa.free(a.y);
    }
    const b = try Nz.of(gpa, qx, qy);
    defer {
        gpa.free(b.x);
        gpa.free(b.y);
    }
    const t2 = (thr * b.s) * (thr * b.s);
    var best: i64 = -1;
    @memset(mask_out, 0);
    const mask = try gpa.alloc(u8, n);
    defer gpa.free(mask);
    var trials: u32 = max_trials;
    var id = [4]usize{ 0, 0, 0, 0 };
    var t: u32 = 0;
    while (t < trials) : (t += 1) {
        for (0..4) |k| {
            var r: usize = undefined;
            while (true) {
                // JS: st ^= st << 13; st >>>= 0; st ^= st >> 17 (a signed shift); st ^= st << 5; st >>>= 0
                st ^= st << 13;
                st ^= @bitCast(@as(i32, @bitCast(st)) >> 17);
                st ^= st << 5;
                r = @intFromFloat(@floor(@as(f64, @floatFromInt(st)) / 4294967296.0 * @as(f64, @floatFromInt(n))));
                if (std.mem.indexOfScalar(usize, id[0..k], r) == null) break;
            }
            id[k] = r;
        }
        const H = h4(a.x, a.y, b.x, b.y, id) orelse continue;
        var c: i64 = 0;
        for (0..n) |i| {
            const z = H[6] * a.x[i] + H[7] * a.y[i] + 1;
            const ex = (H[0] * a.x[i] + H[1] * a.y[i] + H[2]) / z - b.x[i];
            const ey = (H[3] * a.x[i] + H[4] * a.y[i] + H[5]) / z - b.y[i];
            const ok = ex * ex + ey * ey < t2;
            mask[i] = @intFromBool(ok);
            c += @intFromBool(ok);
        }
        if (c > best) {
            best = c;
            @memcpy(mask_out, mask);
            const w = @as(f64, @floatFromInt(c)) / @as(f64, @floatFromInt(n));
            const need = @log(1e-3) / @log(@max(1e-12, 1 - std.math.pow(f64, w, 4)));
            trials = @min(max_trials, @as(u32, @intFromFloat(@ceil(need))) + 1);
        }
    }
    return @intCast(@max(best, 0));
}
