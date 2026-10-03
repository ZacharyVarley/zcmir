"""The functional API: plain functions over one shared GPU engine.

    import zcmir
    r = zcmir.register(moving, fixed)                    # keypoints → refine; r.H, r.score
    r = zcmir.register(moving, fixed, spline=True)       # + a cubic B-spline (4×4 lattice)
    warped = r.warp(moving)                              # moving resampled into the fixed frame
    s = zcmir.register_stack(moving_slices, fixed_slices)

Every function takes the engine's settings as keyword arguments (``zcmir.settings()`` lists them)
and starts from the defaults, so one call never changes another. Images are numpy arrays (uint8
gray / RGB / RGBA, uint16 gray, float gray in [0, 1]) or paths (read with Pillow).

The engine (a WebGPU device, its buffers and pipelines) is made on first use and kept; images are
uploaded again only when they change. Calls are serialized (the engine is one device). For
step-by-step control (detect, match, the climb, maps, events) use ``zcmir.Registrar``.
"""
import difflib
import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import Registrar, RotoScaleMap, ShiftMap  # noqa: F401  (re-exported types)
from .tiff import write_tiff

_LOCK = threading.RLock()
_ENGINE = None
_DEFAULTS = None
_LOADED = {}  # side → (key, array) of the image in the engine


# ── engine and settings ──────────────────────────────────────────────────────

def _engine():
    global _ENGINE, _DEFAULTS
    if _ENGINE is None:
        _ENGINE = Registrar()
        _DEFAULTS = _ENGINE.settings
    return _ENGINE


def close():
    """Release the shared engine (its GPU device and memory). The next call makes a new one."""
    global _ENGINE
    with _LOCK:
        if _ENGINE is not None:
            _ENGINE.close()
            _ENGINE = None
            _LOADED.clear()


def settings():
    """Every setting with its default value (names as in the app; enums as strings)."""
    with _LOCK:
        _engine()
        return dict(_DEFAULTS)


_SPELL = {"model": "group", "spline_lattice": "ffd_grid", "spline_stiffness": "ffd_stiffness"}


def _resolve(model="homography", spline=False, detector="pos_gift", **kw):
    """Keyword arguments → engine settings (defaults + these), with friendly names checked."""
    _engine()
    s = dict(_DEFAULTS)
    if model not in ("affine", "homography"):
        raise ValueError(f'model: "affine" or "homography", not {model!r}')
    if detector not in ("pos_gift", "gls_mift"):
        raise ValueError(f'detector: "pos_gift" or "gls_mift", not {detector!r}')
    s["group"], s["detector"] = model, detector
    lattice = _lattice(spline)
    s["ffd"] = False  # register() turns the spline on for its second stage
    if lattice:
        s["ffd_grid"] = lattice
    for k, v in kw.items():
        k = _SPELL.get(k, k)
        if k not in s:
            near = difflib.get_close_matches(k, s, n=3)
            raise TypeError(f"unknown setting {k!r}" + (f"; did you mean {', '.join(near)}?" if near else " (zcmir.settings() lists them)"))
        s[k] = v
    return s, lattice


def _lattice(spline):
    if spline is True:
        return 4
    if not spline:
        return 0
    g = int(spline)
    if not 2 <= g <= 16:
        raise ValueError("spline: True (a 4×4 lattice) or the lattice size 2–16")
    return g


def _configure(reg, s):
    reg.configure(**s)


# ── images ───────────────────────────────────────────────────────────────────

def _imread(x):
    if isinstance(x, (str, Path)):
        from PIL import Image
        im = Image.open(x)
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            return np.asarray(im).astype(np.uint16)
        return np.asarray(im.convert("RGBA") if im.mode not in ("L", "RGB", "RGBA") else im)
    a = np.asarray(x)
    if a.ndim not in (2, 3):
        raise ValueError(f"an image is (h, w) or (h, w, channels), not {a.shape}")
    return a


def _fingerprint(a):
    """Cheap content key: shape, dtype and a strided sample (catches in-place edits of most kinds)."""
    flat = a.reshape(-1)
    step = max(1, flat.size // 4096)
    return (a.shape, a.dtype.str, hash(flat[::step].tobytes()), float(flat[: min(flat.size, 64)].sum()))


_PREP = ("clahe", "clahe_grid", "clahe_bins", "band", "bp_fine", "bp_coarse", "invert", "half")


def _load(reg, side, img, preprocess, s):
    key = (id(img), _fingerprint(img), preprocess, tuple(s[k] for k in _PREP))
    cur = _LOADED.get(side)
    if cur and cur[0] == key and cur[1] is img:
        return
    (reg.set_moving if side == 0 else reg.set_fixed)(img, preprocess=preprocess)
    _LOADED[side] = (key, img)


def _prepare(moving, fixed, preprocess, kw):
    """The engine configured for this call, with both images in it."""
    reg = _engine()
    s, lattice = _resolve(**kw)
    _configure(reg, s)
    m, f = _imread(moving), _imread(fixed)
    _load(reg, 0, m, preprocess, s)
    _load(reg, 1, f, preprocess, s)
    return reg, s, lattice, m, f


# ── warping ──────────────────────────────────────────────────────────────────

def _field_from_H(H, fixed_shape, rows=None):
    """x_moving − x_fixed (0-based) over the fixed grid for a 3×3 pose H (moving → fixed, 1-based)."""
    h, w = fixed_shape[:2]
    Hi = np.linalg.inv(np.asarray(H, np.float64))
    xs = np.arange(1, w + 1, dtype=np.float64)
    out = np.empty((h, w, 2), np.float32)
    for y0 in range(0, h, 256):
        ys = np.arange(y0 + 1, min(h, y0 + 256) + 1, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        d = Hi[2, 0] * X + Hi[2, 1] * Y + Hi[2, 2]
        mx = (Hi[0, 0] * X + Hi[0, 1] * Y + Hi[0, 2]) / d
        my = (Hi[1, 0] * X + Hi[1, 1] * Y + Hi[1, 2]) / d
        out[y0: y0 + len(ys), :, 0] = mx - X
        out[y0: y0 + len(ys), :, 1] = my - Y
    return out


def warp_image(image, field, fill=0):
    """Resample image (any dtype, gray or with channels) on a displacement field (h, w, 2:
    x_moving − x_fixed, 0-based); bilinear. Returns (warped, mask) — mask: inside the image."""
    img = _imread(image)
    sh, sw = img.shape[:2]
    h, w = field.shape[:2]
    src = img.reshape(sh, sw, -1).astype(np.float32)
    out = np.empty((h, w, src.shape[2]), np.float32)
    mask = np.empty((h, w), bool)
    xs = np.arange(w, dtype=np.float32)
    for y0 in range(0, h, 256):
        y1 = min(h, y0 + 256)
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        mx = xs[None, :] + field[y0:y1, :, 0]
        my = ys + field[y0:y1, :, 1]
        inside = (mx > -0.5) & (my > -0.5) & (mx < sw - 0.5) & (my < sh - 0.5)
        cx = np.clip(mx, 0, sw - 1)
        cy = np.clip(my, 0, sh - 1)
        x0 = np.minimum(np.floor(cx).astype(np.int64), max(sw - 2, 0))
        y0i = np.minimum(np.floor(cy).astype(np.int64), max(sh - 2, 0))
        fx = (cx - x0)[..., None]
        fy = (cy - y0i)[..., None]
        x1 = np.minimum(x0 + 1, sw - 1)
        y1i = np.minimum(y0i + 1, sh - 1)
        v = (src[y0i, x0] * (1 - fx) + src[y0i, x1] * fx) * (1 - fy) + (src[y1i, x0] * (1 - fx) + src[y1i, x1] * fx) * fy
        v[~inside] = fill
        out[y0:y1] = v
        mask[y0:y1] = inside
    out = out.reshape((h, w) + img.shape[2:])
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(np.rint(out), info.min, info.max).astype(img.dtype)
    else:
        out = out.astype(img.dtype)
    return out, mask


# ── results ──────────────────────────────────────────────────────────────────

@dataclass
class Registration:
    """A registered pair. H maps moving pixel coordinates to fixed ones (row-major 3×3, 1-based:
    (1, 1) is the centre of the top-left pixel). With a spline, the full warp is H followed by
    the B-spline: use warp() / displacement(), which include it."""
    H: np.ndarray
    score: float
    model: str
    moving_shape: tuple
    fixed_shape: tuple
    spline: np.ndarray | None = None       # control-point displacements (2·g·g), or None
    lattice: int = 0
    method: str = "keypoints"              # "keypoints", "refine" (from a given pose), "carry" (stacks)
    keypoints: tuple | None = None         # (moving, fixed) counts, when detected
    inliers: int | None = None
    iterations: int = 0
    redetected: bool = False
    _field: np.ndarray | None = field(default=None, repr=False)

    def __repr__(self):
        sp = f", spline {self.lattice}×{self.lattice}" if self.spline is not None else ""
        kp = f", {self.inliers} inliers" if self.inliers is not None else ""
        return f"Registration({self.model}{sp}, score {self.score:.4g}, {self.method}{kp})"

    def map_points(self, points):
        """Moving (x, y) points (1-based, (n, 2)) to fixed coordinates by H (the global model;
        a spline's local correction is not included — fixed_to_moving() includes it)."""
        p = np.asarray(points, np.float64).reshape(-1, 2)
        q = np.c_[p, np.ones(len(p))] @ self.H.T
        return q[:, :2] / q[:, 2:]

    def displacement(self):
        """The full warp over the fixed image, (h, w, 2) float32: x_moving − x_fixed, 0-based
        pixels (the ITK / SimpleITK DisplacementFieldTransform convention)."""
        if self._field is not None:
            return self._field
        return _field_from_H(self.H, self.fixed_shape)

    def fixed_to_moving(self, points):
        """Fixed (x, y) points (1-based) to moving coordinates through the full warp."""
        d = self.displacement()
        p = np.asarray(points, np.float64).reshape(-1, 2) - 1
        h, w = d.shape[:2]
        x = np.clip(p[:, 0], 0, w - 1)
        y = np.clip(p[:, 1], 0, h - 1)
        x0, y0 = np.minimum(np.floor(x).astype(int), w - 2), np.minimum(np.floor(y).astype(int), h - 2)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        v = (d[y0, x0] * (1 - fx) + d[y0, x0 + 1] * fx) * (1 - fy) + (d[y0 + 1, x0] * (1 - fx) + d[y0 + 1, x0 + 1] * fx) * fy
        return p + v + 1

    def warp(self, moving, fill=0, mask=False):
        """The moving image (its own dtype and channels — the original, not the preprocessed
        one) resampled into the fixed frame; bilinear, fill outside. mask=True also returns the
        boolean mask of fixed pixels the moving image covers."""
        out, m = warp_image(moving, self.displacement(), fill)
        return (out, m) if mask else out

    def export_warp(self, path=None, stem=None):
        """The app's warp export: a ZIP with {stem}_Warp.nrrd (the displacement field) and, for
        a spline or an affine pose, {stem}.tfm (Insight Transform File). Returns the bytes."""
        with _LOCK:
            reg = _engine()
            s, _ = _resolve(model=self.model)
            if self.spline is not None:
                s["ffd"], s["ffd_grid"] = True, self.lattice
            _configure(reg, s)
            # the export depends on the pose and the two images' sizes only
            reg.set_moving(np.zeros(self.moving_shape[:2], np.float32), preprocess=False)
            reg.set_fixed(np.zeros(self.fixed_shape[:2], np.float32), preprocess=False)
            _LOADED.clear()
            if self.spline is not None:
                reg.ffd = self.spline
            reg.pose = self.H
            return reg.export_warp(path, stem)

    def to_dict(self):
        return {
            "H": self.H.tolist(), "score": self.score, "model": self.model, "method": self.method,
            "moving_shape": list(self.moving_shape[:2]), "fixed_shape": list(self.fixed_shape[:2]),
            "spline": None if self.spline is None else {"lattice": self.lattice, "coefficients": self.spline.tolist()},
            "inliers": self.inliers, "redetected": self.redetected,
        }


# ── pairs ────────────────────────────────────────────────────────────────────

def register(moving, fixed, *, model="homography", spline=False, detector="pos_gift", init=None,
             preprocess=True, **settings):
    """Register moving onto fixed: keypoints (detect → match → robust fit), then refine the score
    over the model; with spline, then the B-spline together with the model (the app's Auto).

    init: a starting pose (3×3, moving → fixed, 1-based) — skips the keypoints and only refines.
    spline: False, True (4×4 control points) or the lattice size; spline_stiffness (default 0.15)
    is the bending penalty as a wavelength in fractions of the lattice's size (0: none). Other
    keywords are settings (zcmir.settings()), e.g. metric="smi", pg_max_points=5000, clahe=False.
    """
    with _LOCK:
        reg, s, lattice, m, f = _prepare(moving, fixed, preprocess, dict(model=model, spline=spline, detector=detector, **settings))
        kps = inl = None
        if init is None:
            d = reg.detect()
            mt = reg.match()
            kps, inl = (d.n_moving, d.n_fixed), mt.inliers
        else:
            reg.pose = np.asarray(init, np.float64).reshape(3, 3)
        c = reg.climb()
        iters = c.iterations
        cps = fld = None
        if lattice:
            reg.configure(ffd=True, ffd_grid=lattice)
            reg.ffd = np.zeros(2 * lattice * lattice, np.float32)
            c = reg.climb()
            iters += c.iterations
            cps = reg.ffd
            fld = reg.displacement()
        H = reg.pose
        return Registration(
            H=H, score=reg.score(H).mean, model=model, moving_shape=m.shape, fixed_shape=f.shape,
            spline=cps, lattice=lattice, method="keypoints" if init is None else "refine",
            keypoints=kps, inliers=inl, iterations=iters, _field=fld,
        )


def refine(moving, fixed, H, **kw):
    """Refine a pose (no keypoints): register(moving, fixed, init=H, ...)."""
    return register(moving, fixed, init=H, **kw)


def score(moving, fixed, H, *, preprocess=True, **settings):
    """The registration score at pose H (the mean of both directions when symmetric)."""
    with _LOCK:
        reg, *_ = _prepare(moving, fixed, preprocess, settings)
        return reg.score(np.asarray(H, np.float64)).mean


def shift_map(moving, fixed, H=None, *, preprocess=True, **settings):
    """The score at every translation of the moving image about pose H (default: identity)."""
    with _LOCK:
        reg, *_ = _prepare(moving, fixed, preprocess, settings)
        return reg.shift_map(np.eye(3) if H is None else np.asarray(H, np.float64))


def rs_map(moving, fixed, H=None, *, center=None, preprocess=True, **settings):
    """The score at every rotation × scale of the moving image about center (fixed px)."""
    with _LOCK:
        reg, *_ = _prepare(moving, fixed, preprocess, settings)
        return reg.rs_map(np.eye(3) if H is None else np.asarray(H, np.float64), center=center)


def search(moving, fixed, H=None, *, mode="sweep", rotation=None, scale=None, shear=None, stretch=None,
           preprocess=True, **settings):
    """Brute force: "sweep" scores every rotation × scale × shift, "cloud" climbs seeds around H.
    Returns the best pose found as a Registration (H, score).

    Sweep ranges (each also settable through the sw_* settings):
      rotation  degrees: a half-range (30 → ±30°) or a (from, to) range, which may run through
                ±180° ((150, -150) covers 60° about 180°); default the full circle
      scale     (min, max) of the moving image's scale; default (0.5, 2)
      shear     largest shear either way (0.05), or (largest, values); off by default
      stretch   largest stretch either way as a fraction (0.1: x by up to 1.1 and y by 1/1.1, and
                the reverse), or (largest, values); off by default
    Shear and stretch act along the moving image's own axes and multiply the number of maps by
    their number of values (3 by default), so keep them to small ranges and few values.
    """
    if rotation is not None:
        if np.ndim(rotation) == 0:
            settings.update(sw_rot_range=False, sw_rot=float(rotation))
        else:
            lo, hi = rotation
            settings.update(sw_rot_range=True, sw_rot_lo=float(lo), sw_rot_hi=float(hi))
    if scale is not None:
        settings.update(sw_smin=float(scale[0]), sw_smax=float(scale[1]))
    for name, key in (("shear", "shear"), ("stretch", "aniso")):
        v = shear if name == "shear" else stretch
        if v is None:
            continue
        mx, n = (v, None) if np.ndim(v) == 0 else v
        settings[f"sw_{key}"] = float(mx) > 0
        settings[f"sw_{key}_max"] = float(mx)
        if n is not None:
            settings[f"sw_n{key}"] = int(n)
    with _LOCK:
        reg, s, _, m, f = _prepare(moving, fixed, preprocess, dict(search_mode=mode, **settings))
        reg.pose = np.eye(3) if H is None else np.asarray(H, np.float64)
        r = reg.search()
        return Registration(H=reg.pose, score=reg.score().mean, model=s["group"], moving_shape=m.shape,
                            fixed_shape=f.shape, method=f"search ({mode})")


@dataclass
class Matches:
    """Keypoints (x, y 1-based, score, scale) and the robust fit's correspondences."""
    H: np.ndarray
    moving: np.ndarray
    fixed: np.ndarray
    pairs: np.ndarray       # (k, 2): moving index, fixed index
    inliers: np.ndarray     # (k,) bool: within the inlier radius of H


def match(moving, fixed, *, preprocess=True, **settings):
    """Keypoints on both images and their matches under the robust fit (no refinement)."""
    with _LOCK:
        reg, s, *_ = _prepare(moving, fixed, preprocess, settings)
        reg.detect()
        mt = reg.match()
        km, kf = reg.keypoints(0), reg.keypoints(1)
        j = reg.matches()
        i = np.nonzero(j >= 0)[0]
        pairs = np.c_[i, j[i]]
        if len(pairs):
            p = np.c_[km[pairs[:, 0], :2], np.ones(len(pairs))] @ mt.H.T
            err = np.hypot(*(p[:, :2] / p[:, 2:] - kf[pairs[:, 1], :2]).T)
            inl = err <= s["inlier_px"]
        else:
            inl = np.zeros(0, bool)
        return Matches(mt.H, km, kf, pairs, inl)


def overlay(moving, fixed, H, *, preprocess=True, **settings):
    """Both (preprocessed) images on the pose's union canvas (see Overlay)."""
    with _LOCK:
        reg, *_ = _prepare(moving, fixed, preprocess, settings)
        return reg.overlay(np.asarray(H, np.float64))


# ── stacks ───────────────────────────────────────────────────────────────────

@dataclass
class StackRegistration:
    """Per slice pair: its Registration (None if not registered), how far its pose is from its
    neighbours' (off_trend, fixed px on the moving image's corners) and whether that exceeds the
    threshold (flagged)."""
    slices: list
    off_trend: list
    flagged: list
    threshold: float

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        return self.slices[i]

    def __repr__(self):
        n = sum(r is not None for r in self.slices)
        return (f"StackRegistration({n}/{len(self.slices)} registered, {sum(self.flagged)} flagged, "
                f"threshold {self.threshold:.1f} px)")

    @property
    def H(self):
        return [None if r is None else r.H for r in self.slices]

    def warp(self, moving, fill=0):
        """Each registered slice's moving image resampled into its fixed frame (a generator)."""
        for r, m in zip(self.slices, moving):
            if r is not None:
                yield r.warp(m, fill)

    def save_tiff(self, path, moving):
        """The registered moving stack as one multi-page TIFF (Fiji / ImageJ open it as a
        stack), written slice by slice; unregistered slices are left out."""
        write_tiff(path, self.warp(moving))

    def to_json(self, path=None):
        d = json.dumps({
            "convention": "H maps moving pixel coordinates to fixed pixel coordinates, row-major 3x3, 1-based pixel centres",
            "threshold_px": self.threshold,
            "slices": [None if r is None else {**r.to_dict(), "off_trend_px": t, "flagged": bool(fl)}
                       for r, t, fl in zip(self.slices, self.off_trend, self.flagged)],
        }, indent=1)
        if path:
            Path(path).write_text(d)
        return d


def _corners(r):
    h, w = r.moving_shape[:2]
    return np.array([[1, 1], [w, 1], [w, h], [1, h]], np.float64)


def _apply(H, p):
    q = np.c_[p, np.ones(len(p))] @ np.asarray(H).T
    return q[:, :2] / q[:, 2:]


def _pose_gap(r, H, refs):
    """Largest corner distance (fixed px) between pose H and the mean of the reference poses."""
    if not refs:
        return None
    c = _corners(r)
    ref = np.mean([_apply(R, c) for R in refs], axis=0)
    return float(np.max(np.hypot(*(_apply(H, c) - ref).T)))


def _expected(res, k, c):
    """Where slice k's corners c should land given its neighbours: their mean between two, a
    straight-line extrapolation at the ends (so a stack that drifts steadily flags nothing)."""
    get = lambda j: res[j] if 0 <= j < len(res) else None
    a, b = get(k - 1), get(k + 1)
    if a is not None and b is not None:
        return (_apply(a.H, c) + _apply(b.H, c)) / 2
    for near, far in ((a, get(k - 2)), (b, get(k + 2))):
        if near is not None:
            return 2 * _apply(near.H, c) - _apply(far.H, c) if far is not None else _apply(near.H, c)
    return None


def _trend(res, jump):
    e = []
    for k, r in enumerate(res):
        ref = None if r is None else _expected(res, k, _corners(r))
        e.append(None if ref is None else float(np.max(np.hypot(*(_apply(r.H, _corners(r)) - ref).T))))
    vals = sorted(v for v in e if v is not None)
    thr = float(jump) if jump is not None else max(2.0, 4 * vals[len(vals) // 2]) if vals else math.inf
    return e, [v is not None and len(vals) >= 3 and v > thr for v in e], thr


def register_stack(moving, fixed, *, start="previous", drop=0.15, jump=None, polish=False,
                   progress=None, model="homography", spline=False, detector="pos_gift", preprocess=True, **settings):
    """Register a moving stack to a fixed stack, slice pair by slice pair (serial sections, time
    series): the first pair from keypoints, then each from the previous pair's pose (refined),
    as the app's Register stack.

    moving, fixed: sequences of images or paths (paths are read one slice at a time).
    start: "previous" (carry the previous pose) or "keypoints" (every slice from keypoints).
    drop: re-detect from keypoints when a slice's score falls more than this fraction below the
        previous slice's (the better of the two poses is kept).
    jump: the same when the pose jumps more than this many fixed px from the previous slice's
        (None: automatic, 4× the median jump so far, at least 2 px); it also flags slices off the
        trend of their neighbours in the result.
    polish: afterwards refine each slice again from the next slice's pose (last to first) and
        keep it when it scores higher.
    progress: callable(i, n, registration) after each slice.
    """
    if start not in ("previous", "keypoints"):
        raise ValueError('start: "previous" or "keypoints"')
    kw = dict(model=model, spline=spline, detector=detector, preprocess=preprocess, **settings)
    n = min(len(moving), len(fixed))
    res, jumps = [None] * n, []

    def thr_now():
        if jump is not None:
            return float(jump)
        if len(jumps) < 3:
            return math.inf
        v = sorted(jumps)
        return max(2.0, 4 * v[len(v) // 2])

    for i in range(n):
        m, f = _imread(moving[i]), _imread(fixed[i])
        prev = res[i - 1] if i else None
        if start == "keypoints" or prev is None:
            r = register(m, f, **kw)
        else:
            r = register(m, f, init=prev.H, **kw)
            r.method = "carry"
            g = _pose_gap(r, r.H, [prev.H])
            dropped = prev.score > 0 and r.score < prev.score * (1 - drop)
            if dropped or g > thr_now():
                k = register(m, f, **kw)
                r = k if k.score > r.score else r
                r.redetected = True
        if prev is not None:
            jumps.append(_pose_gap(r, r.H, [prev.H]))
        res[i] = r
        if progress:
            progress(i, n, r)
    if polish:
        for i in range(n - 2, -1, -1):
            m, f = _imread(moving[i]), _imread(fixed[i])
            r = register(m, f, init=res[i + 1].H, **kw)
            if r.score > res[i].score * (1 + 1e-4):
                r.method, r.redetected = "carry", res[i].redetected
                res[i] = r
    e, flags, thr = _trend(res, jump)
    return StackRegistration(res, e, flags, thr)
