"""zcmir: multimodal image registration on the GPU (WebGPU via wgpu-native).

The registration code is Zig + WGSL, shared with the browser app (``zcmir`` / ``python -m zcmir``
serves it); this package passes arrays in and results out.

    import zcmir
    r = zcmir.register(moving, fixed)                  # keypoints → refine (homography)
    r.H, r.score                                       # 3×3 moving → fixed, 1-based pixels
    warped = r.warp(moving)                            # moving resampled into the fixed frame
    r = zcmir.register(moving, fixed, spline=True)     # + a B-spline
    s = zcmir.register_stack(moving_slices, fixed_slices)
    s.save_tiff("registered.tif", moving_slices)

Images: numpy arrays (uint8 gray / RGB / RGBA, uint16 gray, float gray in [0, 1]) or paths.
Every function takes the engine's settings as keywords (``zcmir.settings()`` lists them). For
step-by-step control (detect, match, the climb, maps, progress events) use ``zcmir.Registrar``.

Pose convention: H maps MOVING pixel coordinates to FIXED pixel coordinates, row-major, with
(1, 1) the centre of the top-left pixel. Tangent coordinates are [tx ty th sg al ga px py]
(6 for the affine group).
"""
import ctypes
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _native

__all__ = [
    "register", "refine", "register_stack", "score", "shift_map", "rs_map", "search", "match", "overlay",
    "warp_image", "write_tiff", "settings", "close", "Registration", "StackRegistration", "Matches",
    "Registrar", "Score", "Gradient", "ShiftMap", "RotoScaleMap", "Detection", "Match", "Climb", "Search",
    "Overlay", "TileHeat", "__version__",
]

__version__ = "0.1.0"

_lib = None
_wgpu = None


def _load():
    global _lib, _wgpu
    if _lib is None:
        _lib, _wgpu = _native.load()
    return _lib


def _mat(H):
    a = np.ascontiguousarray(np.asarray(H, dtype=np.float64).reshape(9))
    return a, a.ctypes.data_as(_native.f64p)


def _H(c_mat):
    return np.array(list(c_mat), dtype=np.float64).reshape(3, 3)


def _fp(a):
    return a.ctypes.data_as(_native.f32p)


@dataclass
class Score:
    """Pose score: the mean of the forward and inverse directions when symmetric."""
    mean: float
    fwd: float
    inv: float | None


@dataclass
class Gradient:
    score: float
    fwd: float
    inv: float | None
    n: float
    grad: np.ndarray
    hess: np.ndarray | None


@dataclass
class ShiftMap:
    """Dense score over every translation of the moving image about a pose.

    map: ny rows × n columns (each side's FFT length; a power of two per side by default), lag
    (0, 0) at [0, 0], negative lags wrapped (np.fft.fftshift to centre it); one cell spans
    (canvas_w / cw, canvas_h / ch) pixels. dx, dy: the peak's lag in fixed pixels.
    """
    peak: float
    dx: float
    dy: float
    zero: float
    n: int
    cw: int
    ch: int
    canvas: tuple
    offset: tuple
    corr_area_px: float
    exact: bool
    calibrated: bool
    map: np.ndarray | None
    ny: int = 0

    def corrected(self, H):
        """The pose moved onto the peak: translate(−dx, −dy) · H."""
        T = np.array([[1, 0, -self.dx], [0, 1, -self.dy], [0, 0, 1.0]])
        return T @ np.asarray(H, dtype=np.float64).reshape(3, 3)


@dataclass
class RotoScaleMap:
    """Score over rotations (dth, radians) and log-scales (dsg) about a centre (fixed pixels).

    map: n×n (θ across, log radius down, lag 0 at [0, 0]); fwd / inv: the two directions'
    maps when symmetric.
    """
    peak: float
    dth: float
    dsg: float
    zero: float
    center: tuple
    n: int
    n_th: int
    n_lam: int
    dlam: float
    r0: float
    r1: float
    symmetric: bool
    calibrated: bool
    map: np.ndarray | None
    fwd: np.ndarray | None
    inv: np.ndarray | None

    def corrected(self, H):
        """simAbout(center, dth, dsg) · H."""
        s = math.exp(self.dsg)
        c, sn = s * math.cos(self.dth), s * math.sin(self.dth)
        cx, cy = self.center
        S = np.array([[c, -sn, cx - c * cx + sn * cy], [sn, c, cy - sn * cx - c * cy], [0, 0, 1.0]])
        return S @ np.asarray(H, dtype=np.float64).reshape(3, 3)


@dataclass
class Detection:
    n_moving: int
    n_fixed: int
    levels_moving: int
    levels_fixed: int


@dataclass
class Match:
    H: np.ndarray
    inliers: int
    correspondences: int
    affine_inliers: int = 0
    pos_candidates: int = 0
    pos_inliers: int = 0
    rotation_deg: float | None = None


@dataclass
class Climb:
    H: np.ndarray
    score: float
    iterations: int
    moved: bool


@dataclass
class Search:
    H: np.ndarray
    score: float
    previous: float
    improved: bool
    count_a: int
    count_b: int


@dataclass
class Overlay:
    """Both images on the pose's union canvas (fixed pixel (1, 1) at canvas (1 − ox, 1 − oy))."""
    moving: np.ndarray
    fixed: np.ndarray
    offset: tuple
    Hc: np.ndarray


@dataclass
class TileHeat:
    score: float
    dest: np.ndarray
    src: np.ndarray


class Registrar:
    """Step-by-step control: one GPU device and one image pair, whose features and keypoints stay
    on the GPU between calls. (The functions — zcmir.register etc. — share one of these.)

    Keyword arguments are settings (``configure``). on_event(dict) receives progress events
    (log lines, trail entries, poses, sweep / swarm views) after each call. close() releases the
    device (also on garbage collection; ``with`` works too).
    """

    def __init__(self, on_event=None, **settings):
        lib = _load()
        self._lib = lib
        self._e = lib.zc_create(_wgpu.encode())
        self.on_event = on_event
        self._events = []
        self._sizes = {}
        if not self._e:
            buf = ctypes.create_string_buffer(1024)
            n = lib.zc_create_error(buf, 1024)
            raise RuntimeError(buf.raw[:n].decode(errors="replace") or "zcmir: could not create the engine")
        if settings:
            self.configure(**settings)

    # ── plumbing ──
    def _error(self):
        buf = ctypes.create_string_buffer(1024)
        n = self._lib.zc_error(self._e, buf, 1024)
        return buf.raw[:n].decode(errors="replace")

    def _check(self, rc, what):
        self._drain()
        if rc != 0:
            raise RuntimeError(f"{what}: {self._error() or rc}")

    def _drain(self):
        buf = ctypes.create_string_buffer(1 << 16)
        while True:
            n = self._lib.zc_events(self._e, buf, len(buf))
            if n == 0:
                break
            for line in buf.raw[:n].decode(errors="replace").splitlines():
                if not line:
                    continue
                ev = json.loads(line)
                self._events.append(ev)
                if self.on_event:
                    self.on_event(ev)

    def events(self):
        """Progress events since the last call (dicts with "kind": log, trail, pose, sweep, swarm)."""
        self._drain()
        out, self._events = self._events, []
        return out

    @property
    def adapter(self):
        buf = ctypes.create_string_buffer(256)
        n = self._lib.zc_adapter(self._e, buf, 256)
        return buf.raw[:n].decode(errors="replace")

    # ── settings ──
    def configure(self, **settings):
        """Change settings (names as in ``settings``; enums as strings, e.g. metric="e4")."""
        b = json.dumps(settings).encode()
        self._check(self._lib.zc_configure(self._e, b, len(b)), "configure")

    @property
    def settings(self):
        cap = 8192
        buf = ctypes.create_string_buffer(cap)
        n = self._lib.zc_settings(self._e, buf, cap)
        if n > cap:
            buf = ctypes.create_string_buffer(n)
            n = self._lib.zc_settings(self._e, buf, n)
        return json.loads(buf.raw[:n])

    # ── images ──
    def _set(self, side, img, preprocess):
        a = np.asarray(img)
        if a.dtype == np.uint8:
            if a.ndim == 2:
                a = np.repeat(a[:, :, None], 3, axis=2)
            if a.ndim != 3 or a.shape[2] not in (3, 4):
                raise ValueError("uint8 images: (h, w), (h, w, 3) or (h, w, 4)")
            if a.shape[2] == 3:
                a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], axis=2)
            a = np.ascontiguousarray(a)
            h, w = a.shape[:2]
            if not preprocess:
                raise ValueError("uint8 images are always preprocessed (pass float gray to skip it)")
            rc = self._lib.zc_set_image_rgba(self._e, side, a.ctypes.data_as(_native.u8p), w, h)
            return self._check(rc, "set_image")
        if a.ndim == 3 and a.shape[2] in (3, 4):
            a = a[:, :, :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
        if a.dtype == np.uint16:
            a = a / 65535.0
        a = np.ascontiguousarray(a, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError("expected a gray image (h, w)")
        h, w = a.shape
        rc = self._lib.zc_set_image_gray(self._e, side, _fp(a), w, h, 1 if preprocess else 0)
        self._check(rc, "set_image")

    def set_moving(self, img, preprocess=True):
        """The moving image. preprocess: the settings' CLAHE / band pass / invert, as the app
        does on load; False uses float gray as the work image unchanged."""
        self._set(0, img, preprocess)

    def set_fixed(self, img, preprocess=True):
        self._set(1, img, preprocess)

    def set_images(self, moving, fixed, preprocess=True):
        self._set(0, moving, preprocess)
        self._set(1, fixed, preprocess)

    def image_size(self, side):
        """(w, h) of side 0 (moving) or 1 (fixed); (0, 0) when not set."""
        out = (ctypes.c_uint32 * 2)()
        self._lib.zc_image_size(self._e, side, out)
        return out[0], out[1]

    def work_image(self, side):
        """The preprocessed image of side 0 (moving) or 1 (fixed) as float32 (h, w)."""
        w, h = self.image_size(side)
        out = np.empty((h, w), np.float32)
        self._check(self._lib.zc_work_image(self._e, side, _fp(out), out.size), "work_image")
        return out

    def _compile_wgsl(self, code, entry):
        """Build a compute pipeline from WGSL source: None when it builds, else the GPU
        compiler's message (tests/check_shaders.py)."""
        c, en = code.encode(), entry.encode()
        msg = ctypes.create_string_buffer(4096)
        n = self._lib.zc_compile_wgsl(self._e, c, len(c), en, len(en), msg, len(msg))
        if n == 0:
            return None
        return msg.raw[: max(n, 0)].decode(errors="replace") if n > 0 else "unsupported here"

    # ── inspection ──
    def features(self, side):
        """The SMI feature planes of side 0 (moving) or 1 (fixed), (4, h, w) float32: the
        whitened rank r, r², r³ and the rank's gradient magnitude."""
        w, h = self.image_size(side)
        out = np.empty((4, h, w), np.float32)
        self._check(self._lib.zc_features(self._e, side, _fp(out), out.size), "features")
        return out

    def moments(self, H=None):
        """The 45 overlap moment sums of the moving features onto the fixed ones at H: n, Σa_i,
        Σb_j, Σa_i b_j (4 × 4), then the upper triangles of Σa_i a_j and Σb_i b_j."""
        a, p = self._Hp(H)
        out = np.empty(45, np.float64)
        self._check(self._lib.zc_moments(self._e, p, out.ctypes.data_as(_native.f64p)), "moments")
        return out

    def pos_gift_structure(self):
        """POS-GIFT's descriptor layout after detect(): orientations, directions, rings, ring
        radii (pixels) and each sample point's offset from the keypoint (directions × rings,
        then the centre)."""
        info = np.empty(5, np.float64)
        self._check(self._lib.zc_pg_info(self._e, info.ctypes.data_as(_native.f64p)), "pos_gift_structure")
        no, ndir, nring, dp, p1 = (int(info[0]), int(info[1]), int(info[2]), int(info[3]), float(info[4]))
        radii = [p1]
        for i in range(2, nring + 1):
            radii.append(radii[-1] + p1 * (i - 1))
        mround = lambda v: np.sign(v) * np.floor(np.abs(v) + 0.5)
        offsets = np.zeros((ndir * nring + 1, 2))
        for a in range(ndir):
            t = 2 * np.pi * a / ndir
            for r, R in enumerate(radii):
                offsets[a * nring + r] = (-mround(R * np.cos(t)), -mround(R * np.sin(t)))
        return {"n_orient": no, "n_dir": ndir, "n_ring": nring, "length": dp, "radii": radii, "offsets": offsets}

    def pos_gift_maps(self, side):
        """POS-GIFT's oriented phase-congruency energy of side 0 or 1 at full resolution, after
        detect(): (n_orient, h, w) float32, orientation k at k·180°/n_orient."""
        no = self.pos_gift_structure()["n_orient"]
        w, h = self.image_size(side)
        out = np.empty((no, h, w), np.float32)
        self._check(self._lib.zc_pg_maps(self._e, side, _fp(out), out.size), "pos_gift_maps")
        return out

    def pos_gift_descriptors(self, side):
        """POS-GIFT descriptors of side 0 or 1 after detect() (before match()), in keypoints()
        order: (n, points, n_orient) float32, points as pos_gift_structure()["offsets"]."""
        st = self.pos_gift_structure()
        npt, no = st["n_dir"] * st["n_ring"] + 1, st["n_orient"]
        buf = np.empty(65536 * st["length"], np.float32)
        n = self._lib.zc_pg_descriptors(self._e, side, _fp(buf), buf.size)
        self._check(0 if n >= 0 else -1, "pos_gift_descriptors")
        return buf[: n * st["length"]].reshape(n, st["length"])[:, : npt * no].reshape(n, npt, no).copy()

    def _map_cap(self, resolution=0):
        n = max(int(resolution or 0), int(self.settings.get("map_res", 0)), 1024)
        return n * n

    # ── pose and spline ──
    @property
    def pose(self):
        H = (ctypes.c_double * 9)()
        self._lib.zc_get_pose(self._e, H)
        return _H(H)

    @pose.setter
    def pose(self, H):
        a, p = _mat(H)
        self._lib.zc_set_pose(self._e, p)

    @property
    def ffd(self):
        """The B-spline control points (2·g·g, destN units; g = settings["ffd_grid"])."""
        buf = np.empty(512, np.float32)
        n = self._lib.zc_get_ffd(self._e, _fp(buf), buf.size)
        return buf[:n].copy()

    @ffd.setter
    def ffd(self, cps):
        a = np.ascontiguousarray(cps, dtype=np.float32).ravel()
        self._check(self._lib.zc_set_ffd(self._e, _fp(a), a.size), "set_ffd")

    def _Hp(self, H):
        return _mat(self.pose if H is None else H)

    # ── scores ──
    def score(self, H=None):
        a, p = self._Hp(H)
        out = np.empty(3, np.float64)
        self._check(self._lib.zc_score(self._e, p, out.ctypes.data_as(_native.f64p)), "score")
        return Score(float(out[0]), float(out[1]), None if math.isnan(out[2]) else float(out[2]))

    def gradient(self, H=None, hess=False):
        """Score and tangent gradient (and the Gauss–Newton matrix) at H."""
        a, p = self._Hp(H)
        r = _native.GradResult()
        self._check(self._lib.zc_gradient(self._e, p, 1 if hess else 0, ctypes.byref(r)), "gradient")
        nk = r.nk
        g = np.array(r.grad[:nk])
        Hm = np.array(r.hess[: nk * nk]).reshape(nk, nk) if r.has_hess else None
        return Gradient(r.score, r.fwd, None if math.isnan(r.inv) else r.inv, r.n, g, Hm)

    def ffd_gradient(self, H=None):
        """Score and its gradient over the spline's control points at H."""
        a, p = self._Hp(H)
        s = ctypes.c_double()
        g = np.zeros(512, np.float64)
        self._check(self._lib.zc_ffd_gradient(self._e, p, ctypes.byref(s), g.ctypes.data_as(_native.f64p), g.size), "ffd_gradient")
        n = 2 * self.settings["ffd_grid"] ** 2
        return s.value, g[:n].copy()

    # ── maps ──
    def shift_map(self, H=None, *, exact=None, calibrated=None, resolution=1024, return_map=True):
        """Score every translation of the moving image about pose H. Without exact / calibrated
        the settings decide (score, resolution, overlap floor, NCC for the NCC metric); with
        them those are explicit and there is no overlap floor."""
        a, p = self._Hp(H)
        out = _native.ShiftResult()
        buf = np.empty(self._map_cap(resolution), np.float32) if return_map else None
        mp, ml = (_fp(buf), buf.size) if buf is not None else (None, 0)
        if exact is None and calibrated is None:
            rc = self._lib.zc_shift_map_set(self._e, p, ctypes.byref(out), mp, ml)
        else:
            flags = (1 if (exact is None or exact) else 0) | (2 if (calibrated is None or calibrated) else 0)
            rc = self._lib.zc_shift_map(self._e, p, flags, resolution, ctypes.byref(out), mp, ml)
        self._check(rc, "shift_map")
        ny = out.n_y or out.n
        m = buf[: out.n * ny].reshape(ny, out.n).copy() if buf is not None else None
        return ShiftMap(
            peak=out.peak, dx=out.dx, dy=out.dy, zero=out.zero, n=out.n, cw=out.cw, ch=out.ch,
            canvas=(out.canvas_w, out.canvas_h), offset=(out.ox, out.oy), corr_area_px=out.corr_area_px,
            exact=bool(out.exact), calibrated=bool(out.calibrated), map=m, ny=ny,
        )

    def rs_map(self, H=None, *, center=None, symmetric=True, return_maps=True):
        """Roto-scale map at H about center (fixed pixels, 1-based; default the moving centroid)."""
        a, p = self._Hp(H)
        cap = self._map_cap() if return_maps else 0
        bufs = [np.empty(cap, np.float32) if return_maps else None for _ in range(3)]
        ptr = [_fp(b) if b is not None else None for b in bufs]
        c = (ctypes.c_double * 2)(*center) if center is not None else None
        r = _native.RsResult()
        self._check(self._lib.zc_rs_map(self._e, p, c, 1 if symmetric else 0, ctypes.byref(r), ptr[0], ptr[1], ptr[2] if symmetric else None, cap), "rs_map")
        nn = r.n * r.n
        get = lambda b: b[:nn].reshape(r.n, r.n).copy() if b is not None else None
        return RotoScaleMap(
            peak=r.peak, dth=r.dth, dsg=r.dsg, zero=r.zero, center=(r.cx, r.cy), n=r.n, n_th=r.n_th, n_lam=r.n_lam,
            dlam=r.dlam, r0=r.r0, r1=r.r1, symmetric=bool(r.symmetric), calibrated=bool(r.calibrated),
            map=get(bufs[0]), fwd=get(bufs[1]), inv=get(bufs[2]) if symmetric else None,
        )

    def tile_heat(self, H=None, grid=8, min_n=12):
        a, p = self._Hp(H)
        G = max(2, min(16, int(grid)))
        dest = np.empty(G * G, np.float32)
        src = np.empty(G * G, np.float32)
        s = ctypes.c_double()
        self._check(self._lib.zc_tile_heat(self._e, p, G, min_n, ctypes.byref(s), _fp(dest), _fp(src), G * G), "tile_heat")
        return TileHeat(s.value, dest.reshape(G, G), src.reshape(G, G))

    # ── detection, matching, optimization ──
    def detect(self):
        """Keypoints on both images (settings["detector"]: "gls_mift" or "pos_gift")."""
        r = _native.DetectResult()
        self._check(self._lib.zc_detect(self._e, ctypes.byref(r)), "detect")
        return Detection(r.n1, r.n2, r.levels1, r.levels2)

    def match(self):
        """Match and fit; the fit becomes the pose."""
        r = _native.MatchResult()
        self._check(self._lib.zc_match(self._e, ctypes.byref(r)), "match")
        return Match(_H(r.H), r.ninl, r.n_corr, r.ninl_aff, r.pos_n, r.pos_inl, r.rot_deg if r.has_rot else None)

    def keypoints(self, side):
        """Keypoints of side 0 (moving) or 1 (fixed): (n, 4) x, y (1-based), score, scale."""
        buf = np.empty((65536, 4), np.float32)
        n = self._lib.zc_keypoints(self._e, side, _fp(buf), 65536)
        self._check(0 if n >= 0 else -1, "keypoints")
        return buf[: min(n, 65536)].copy()

    def keypoint_frames(self, side):
        """Descriptor frame of each keypoint of side 0 or 1, in keypoints() order: (n, 2) the
        direction the descriptor starts from (radians; x right, y down) and the radius it reads
        (pixels)."""
        buf = np.empty((65536, 2), np.float32)
        n = self._lib.zc_keypoint_frames(self._e, side, _fp(buf), 65536)
        self._check(0 if n >= 0 else -1, "keypoint_frames")
        return buf[: min(n, 65536)].copy()

    def matches(self):
        """For each moving keypoint the matched fixed keypoint (-1: none), after match()."""
        buf = np.empty(65536, np.uint32)
        n = self._lib.zc_matches(self._e, buf.ctypes.data_as(_native.u32p), 65536)
        self._check(0 if n >= 0 else -1, "matches")
        return np.where(buf[:n] == 0xFFFFFFFF, -1, buf[:n].astype(np.int64))

    def climb(self):
        """Refine from the current pose, on the GPU (shift / roto-scale hops, Gauss–Newton
        steps or, for E4, gradient line searches; with the spline, its Gauss–Newton step
        under the bending penalty settings["ffd_stiffness"]); the result becomes the pose."""
        r = _native.ClimbResult()
        self._check(self._lib.zc_climb(self._e, ctypes.byref(r)), "climb")
        return Climb(_H(r.H), r.score, r.iterations, bool(r.moved))

    def auto(self):
        """Detect → match → climb (the app's Auto)."""
        r = _native.ClimbResult()
        self._check(self._lib.zc_auto(self._e, ctypes.byref(r)), "auto")
        return Climb(_H(r.H), r.score, r.iterations, bool(r.moved))

    def search(self):
        """Global search (settings["search_mode"]: "sweep" SIM(2) sweep, or "cloud" seed cloud);
        the pose changes only if the search scores higher. count_a / count_b: peaks and
        finalists (sweep) or evaluations and seeds (cloud)."""
        r = _native.SearchResult()
        self._check(self._lib.zc_search(self._e, ctypes.byref(r)), "search")
        return Search(_H(r.H), r.score, r.prev, bool(r.improved), r.count_a, r.count_b)

    # ── images at the pose, export ──
    def overlay(self, H=None, inverse=False):
        """Moving (warped, with the spline) and fixed images on the pose's union canvas.
        inverse: the fixed image backprojected into the moving frame."""
        a, p = self._Hp(H)
        c = _native.Canvas()
        self._check(self._lib.zc_canvas(self._e, p, 1 if inverse else 0, ctypes.byref(c)), "canvas")
        mov = np.empty((c.oh, c.ow), np.float32)
        fix = np.empty((c.oh, c.ow), np.float32)
        self._check(self._lib.zc_overlay(self._e, p, 1 if inverse else 0, ctypes.byref(c), _fp(mov), _fp(fix), mov.size), "overlay")
        return Overlay(mov, fix, (c.ox, c.oy), _H(c.Hc))

    def displacement(self):
        """The composed warp over the fixed image as (h, w, 2): x_moving − x_fixed, 0-based
        pixels (ITK / SimpleITK DisplacementFieldTransform convention)."""
        w, h = self.image_size(1)
        out = np.empty((h, w, 2), np.float32)
        self._check(self._lib.zc_displacement(self._e, _fp(out), out.size), "displacement")
        return out

    def export_warp(self, path=None, stem=None):
        """The app's warp export: a ZIP with {stem}_Warp.nrrd (displacement field) and, for a
        spline or an affine pose, {stem}.tfm (Insight Transform File). Written to path when
        given; returns the bytes."""
        if stem is None:
            stem = Path(path).stem if path else "cmir_warp"
        s = stem.encode()
        n = self._lib.zc_export_warp(self._e, s, len(s), None, 0)
        self._check(0 if n >= 0 else -1, "export_warp")
        buf = ctypes.create_string_buffer(n)
        self._lib.zc_export_warp(self._e, s, len(s), ctypes.cast(buf, ctypes.c_void_p), n)
        data = buf.raw[:n]
        if path:
            Path(path).write_bytes(data)
        return data

    # ── lifetime ──
    def close(self):
        if getattr(self, "_e", None):
            self._lib.zc_destroy(self._e)
            self._e = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


from .api import (  # noqa: E402  (the functions use Registrar)
    Matches, Registration, StackRegistration, close, match, overlay, refine, register, register_stack,
    rs_map, score, search, settings, shift_map, warp_image,
)
from .tiff import write_tiff  # noqa: E402
from . import examples  # noqa: E402,F401
