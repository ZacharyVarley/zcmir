#!/usr/bin/env python3
"""End-to-end check of the Python API, native build.

    python tests/test_api.py                    (after `zig build -Doptimize=ReleaseFast`)
    python tests/test_api.py moving.png fixed.png

Without arguments the pair is synthetic: a textured image and a copy warped by a known
homography, with inverted, nonlinear contrast and noise (a cross-modal pair with ground truth).
Checks the functions (register with each detector and with a spline, refine, score, both maps,
match, overlay, both searches, warping, the warp export), a synthetic stack with a drifting
known pose (register_stack, the TIFF writer), and the step-by-step Registrar.
"""
import io
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
import zcmir  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def rgba(path):
    return np.asarray(Image.open(path).convert("RGBA"))


def smooth_noise(rng, n, power):
    f = np.fft.fftfreq(n)
    r = np.hypot(*np.meshgrid(f, f))
    r[0, 0] = 1
    spec = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / r ** power
    spec[0, 0] = 0
    img = np.fft.ifft2(spec).real
    return (img - img.min()) / np.ptp(img)


def bilinear(img, x, y):
    """img sampled at 0-based (x, y); 0 outside."""
    h, w = img.shape
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    out = np.zeros_like(x)
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            out[ok] += (wx * wy)[ok] * img[yi[ok], xi[ok]]
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    return np.where(inside, out, 0.0), inside


def texture(n, seed):
    rng = np.random.default_rng(seed)
    img = 0.6 * smooth_noise(rng, n, 1.6)
    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(80):  # rotated rectangles: corners for the detectors
        cx, cy = rng.uniform(0, n, 2)
        a, b = rng.uniform(6, 40, 2)
        t = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
        v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
        img[(np.abs(u) < a) & (np.abs(v) < b)] = rng.uniform(0, 1)
    return np.clip(img + 0.02 * rng.standard_normal((n, n)), 0, 1)


def pose(n, deg, s, tx, ty, persp=(2e-5, -1e-5)):
    c = (n + 1) / 2
    th = np.deg2rad(deg)
    C = np.array([[1, 0, c], [0, 1, c], [0, 0, 1.0]])
    R = np.array([[s * np.cos(th), -s * np.sin(th), 0], [s * np.sin(th), s * np.cos(th), 0], [persp[0], persp[1], 1.0]])
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1.0]]) @ C @ R @ np.linalg.inv(C)


def other_modality(img, H, seed):
    """fixed(X) = f(moving(H⁻¹ X)), 1-based X, with f inverted and nonlinear, plus noise."""
    n = img.shape[0]
    rng = np.random.default_rng(seed + 1000)
    yy, xx = np.mgrid[0:n, 0:n]
    X = np.stack([xx + 1.0, yy + 1.0, np.ones_like(xx, dtype=float)], -1) @ np.linalg.inv(H).T
    src, inside = bilinear(img, X[..., 0] / X[..., 2] - 1, X[..., 1] / X[..., 2] - 1)
    fixed = np.where(inside, 1 - src ** 0.6, 0.0)
    return np.clip(fixed + 0.03 * rng.standard_normal((n, n)), 0, 1)


def u8(g):
    return (np.clip(g, 0, 1) * 255 + 0.5).astype(np.uint8)


def synthetic_pair(n=512, seed=7):
    """(moving gray uint8, fixed gray uint8, H true): H maps moving → fixed, 1-based pixels."""
    img = texture(n, seed)
    H = pose(n, 8.0, 1.06, 14.0, -9.0)
    return u8(img), u8(other_modality(img, H, seed)), H


def corner_dist(A, B, w, h):
    d = 0.0
    for x, y in [(1, 1), (w, 1), (w, h), (1, h)]:
        a = A @ [x, y, 1.0]
        b = B @ [x, y, 1.0]
        d = max(d, float(np.hypot(*(a[:2] / a[2] - b[:2] / b[2]))))
    return d


failures = []


def check(name, ok, info=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + info if info else ''}", flush=True)
    if not ok:
        failures.append(name)


if len(sys.argv) == 3:
    mov, fix, H_true = rgba(sys.argv[1]), rgba(sys.argv[2]), None
else:
    mov, fix, H_true = synthetic_pair()
lw, lh = mov.shape[1], mov.shape[0]

# ── the functions ───────────────────────────────────────────────────────────
t0 = time.perf_counter()
r = zcmir.register(mov, fix)
dt = time.perf_counter() - t0
check("register (POS-GIFT)", r.score > 0 and r.inliers and r.inliers > 20, f"{r}  {dt:.2f} s")
if H_true is not None:
    e = corner_dist(r.H, H_true, lw, lh)
    check("register recovers the true pose", e < 1.0, f"{e:.3f} px from the truth (corners)")
rg = zcmir.register(mov, fix, detector="gls_mift")
check("register (GLS-MIFT) reaches the same pose", corner_dist(rg.H, r.H, lw, lh) < 1.0, f"{rg}  {corner_dist(rg.H, r.H, lw, lh):.3f} px apart")
rs = zcmir.register(mov, fix, spline=True)
check("register with a spline", rs.spline is not None and rs.spline.shape == (32,) and rs.score >= 0.999 * r.score
      and rs.displacement().shape == (fix.shape[0], fix.shape[1], 2), f"{rs}")
ref = zcmir.refine(mov, fix, r.H)
check("refine from the registered pose stays", corner_dist(ref.H, r.H, lw, lh) < 0.2 and ref.method == "refine", f"{corner_dist(ref.H, r.H, lw, lh):.3f} px")
s = zcmir.score(mov, fix, r.H)
check("score = register's score", abs(s - r.score) < 1e-3 * r.score, f"{s:.1f}")
check("score changes with the settings only for the call", abs(zcmir.score(mov, fix, r.H, metric="ncc")) <= 1
      and abs(zcmir.score(mov, fix, r.H) - s) < 1e-6 * s)
sm = zcmir.shift_map(mov, fix, r.H)
check("shift map peaks at the pose", abs(sm.dx) < 2 and abs(sm.dy) < 2 and sm.map is not None, f"peak {sm.peak:.2f} at ({sm.dx:.2f}, {sm.dy:.2f})  map {sm.map.shape}")
rm = zcmir.rs_map(mov, fix, r.H)
check("roto-scale map peaks at the pose", abs(rm.dth) < 0.02 and abs(rm.dsg) < 0.02, f"dθ {rm.dth:.4f} dσ {rm.dsg:.4f}")
mt = zcmir.match(mov, fix)
check("match", mt.inliers.sum() > 20 and mt.moving.shape[1] == 4 and len(mt.pairs) == len(mt.inliers), f"{len(mt.moving)}/{len(mt.fixed)} keypoints, {len(mt.pairs)} matches, {mt.inliers.sum()} inliers")
ov = zcmir.overlay(mov, fix, r.H)
check("overlay", ov.moving.shape == ov.fixed.shape and ov.fixed.max() > 0.5, f"canvas {ov.moving.shape[1]}×{ov.moving.shape[0]}")
T = np.array([[np.cos(0.3), -np.sin(0.3), 40], [np.sin(0.3), np.cos(0.3), -30], [0, 0, 1.0]])
t0 = time.perf_counter()
sw = zcmir.search(mov, fix, T @ r.H, mode="sweep")
check("search sweep recovers the pose", corner_dist(sw.H, r.H, lw, lh) < 5 and sw.score > 0.99 * r.score, f"{corner_dist(sw.H, r.H, lw, lh):.2f} px from register  {time.perf_counter() - t0:.2f} s")
# a rotation range around the answer (and one that misses it), shear and stretch in the sweep
J = r.H[:2, :2]
ang = np.degrees(np.arctan2(J[1, 0] - J[0, 1], J[0, 0] + J[1, 1]))
t0 = time.perf_counter()
sr = zcmir.search(mov, fix, rotation=(ang - 40, ang + 40), sw_nth=21, shear=0.05, stretch=(0.05, 3))
check("search sweep with a rotation range, shear and stretch", corner_dist(sr.H, r.H, lw, lh) < 5,
      f"{corner_dist(sr.H, r.H, lw, lh):.2f} px from register  {time.perf_counter() - t0:.2f} s")
sx = zcmir.search(mov, fix, rotation=(ang + 90, ang + 150), sw_nth=11)
check("search sweep keeps to its rotation range", corner_dist(sx.H, r.H, lw, lh) > 5, f"{corner_dist(sx.H, r.H, lw, lh):.0f} px from register")
cl = zcmir.search(mov, fix, T @ r.H, mode="cloud")
check("search cloud improves", cl.score > zcmir.score(mov, fix, T @ r.H), f"{cl.score:.1f}")
w, m = r.warp(mov, mask=True)
check("warp: the moving image in the fixed frame", w.shape == fix.shape and w.dtype == mov.dtype and 0.5 < m.mean() <= 1, f"{w.shape} {w.dtype}, covers {100 * m.mean():.0f}%")
if H_true is not None:
    truth_w, tm = zcmir.warp_image(mov, zcmir.Registration(H=H_true, score=0, model="homography", moving_shape=mov.shape, fixed_shape=fix.shape).displacement())
    both = m & tm
    err = np.abs(w.astype(float) - truth_w.astype(float))[both].mean()
    check("warp matches the truth's warp", err < 4, f"mean |difference| {err:.2f} gray levels")
z = zipfile.ZipFile(io.BytesIO(r.export_warp(stem="pair")))
head, body = z.read("pair_Warp.nrrd").split(b"\n\n", 1)
field = np.frombuffer(body, "<f4").reshape(fix.shape[0], fix.shape[1], 2)
check("warp export = displacement()", np.abs(field - r.displacement()).max() < 1e-3, f"{z.namelist()}")
try:
    zcmir.register(mov, fix, metrc="smi")
    check("unknown settings are rejected", False)
except TypeError as e:
    check("unknown settings are rejected", "metric" in str(e), str(e))

# ── a stack: drifting known poses, the texture changing slowly from slice to slice ──
if H_true is not None:
    n = 5
    base = texture(384, 11)
    rng = np.random.default_rng(3)
    mov_s, fix_s, truth = [], [], []
    for k in range(n):
        img = np.clip(0.85 * base + 0.15 * texture(384, 100 + k) + 0.01 * rng.standard_normal(base.shape), 0, 1)
        Hk = pose(384, -6.0 + 0.4 * k, 1.04 + 0.004 * k, 10.0 + 1.5 * k, -7.0 + k)
        mov_s.append(u8(img))
        fix_s.append(u8(other_modality(img, Hk, 50 + k)))
        truth.append(Hk)
    t0 = time.perf_counter()
    st = zcmir.register_stack(mov_s, fix_s)
    dt = time.perf_counter() - t0
    errs = [corner_dist(sl.H, Hk, 384, 384) for sl, Hk in zip(st.slices, truth)]
    check("register_stack recovers every slice", max(errs) < 1.0, f"{st}  errors {', '.join(f'{e:.2f}' for e in errs)} px  {dt:.2f} s")
    check("register_stack: keypoints, then carried", [sl.method for sl in st.slices] == ["keypoints"] + ["carry"] * (n - 1) and not any(st.flagged),
          f"{[sl.method for sl in st.slices]}")
    tif = Path(tempfile.gettempdir()) / "zcmir_test_stack.tif"
    st.save_tiff(tif, mov_s)
    im = Image.open(tif)
    check("save_tiff: a multi-page TIFF", im.n_frames == n and im.size == (384, 384), f"{im.n_frames} pages {im.size} {im.mode}  {im.tag_v2.get(270, '')!r}"[:120])
    im.close()
    tif.unlink()

# ── step by step ────────────────────────────────────────────────────────────
reg = zcmir.Registrar(group="homography")
print("adapter:", reg.adapter)
reg.set_images(mov, fix)
a = reg.auto()
H = reg.pose
check("Registrar.auto", corner_dist(H, r.H, lw, lh) < 1.0, f"nS {a.score:.1f}")
g = reg.gradient(hess=True)
step = np.linalg.solve(g.hess + 1e-4 * np.diag(np.maximum(np.diag(g.hess), 1e-12)), 0.5 * g.grad)
check("Gauss–Newton step at the optimum is small", np.linalg.norm(step) < 5e-3, f"|step| {np.linalg.norm(step):.2e} (tangent units)")
th = reg.tile_heat()
check("tile heat", th.dest.shape == (8, 8) and np.isfinite(th.score), f"{th.score:.2f}")
# descriptor frames: a match's frames differ by the pose's rotation (GLS-MIFT up to its 180° polarity)
rot = np.arctan2(H[1, 0] - H[0, 1], H[0, 0] + H[1, 1])
for det, period in (("pos_gift", 2 * np.pi), ("gls_mift", np.pi)):
    reg.configure(detector=det)
    reg.detect()
    reg.match()
    kl, kr, mj = reg.keypoints(0), reg.keypoints(1), reg.matches()
    fl, fr = reg.keypoint_frames(0), reg.keypoint_frames(1)
    i = np.flatnonzero(mj >= 0)
    p = np.c_[kl[i, :2], np.ones(len(i))] @ H.T
    i = i[np.hypot(*(p[:, :2] / p[:, 2:] - kr[mj[i], :2]).T) < 3]
    d = (fr[mj[i], 0] - fl[i, 0] - rot + period / 2) % period - period / 2
    ok = np.abs(d) < np.radians(20)
    check(f"keypoint frames ({det}) turn with the pose", len(fl) == len(kl) and len(fr) == len(kr) and np.all(fl[:, 1] > 0)
          and len(i) > 20 and ok.mean() > 0.9, f"{ok.sum()}/{len(i)} matches within 20° of {np.degrees(rot):.1f}°")
reg.configure(detector="pos_gift")
# the sanity check of fitted poses, and every robust fit under it
bow = H.copy()
bow[2, :2] = [-1.5 / lw, 0]  # its horizon crosses the moving image
mir = H @ np.diag([-1.0, 1, 1])
check("fit check", reg.fit_check(H) is None and "horizon" in reg.fit_check(bow) and "mirror" in reg.fit_check(mir)
      and "scale" in reg.fit_check(np.diag([8.0, 8, 1])), f"{reg.fit_check(bow)}; {reg.fit_check(mir)}")
for det in ("pos_gift", "gls_mift"):
    for method in ("lofsc", "prosac", "magsac"):
        reg.configure(detector=det, match_method=method)
        reg.detect()
        m = reg.match()
        e = corner_dist(m.H, H, lw, lh)
        check(f"match {det} / {method}", e < 8 and m.inliers > 20 and reg.fit_check(m.H) is None, f"{e:.2f} px from the registered pose, {m.inliers} inliers")
reg.configure(detector="pos_gift", match_method="lofsc")
reg.configure(search_mode="sweep")
reg.pose = T @ H
reg.search()
ev = reg.events()
kinds = {e["kind"] for e in ev}
check("events", {"log", "trail", "pose", "sweep"} <= kinds, f"{len(ev)} events, kinds {sorted(kinds)}")
reg.close()
zcmir.close()
print("API", "OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
