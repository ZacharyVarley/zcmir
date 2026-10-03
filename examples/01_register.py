# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Register an image pair
#
# Two Landsat 8 views of the Pra River in Ghana: one in the flood season, one in the dry season.
# The moving image is rotated by 30° and scaled by 1.2 relative to the fixed one, and the true
# pose is known, so every result below is checked against it.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np

import zcmir

moving, fixed, truth = zcmir.examples.load("landsat-b")

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(moving, cmap="gray"); ax[0].set_title(f"moving {moving.shape[1]}×{moving.shape[0]}")
ax[1].imshow(fixed, cmap="gray"); ax[1].set_title(f"fixed {fixed.shape[1]}×{fixed.shape[0]}")
for a in ax: a.axis("off")
plt.tight_layout()

# %% [markdown]
# Two small helpers: a checkerboard of the warped moving image over the fixed one, and the
# distance between a pose and the truth at every moving pixel.

# %%
def checkerboard(a, b, mask, tiles=8):
    h, w = b.shape
    y, x = np.mgrid[0:h, 0:w]
    sq = ((x * tiles // w) + (y * tiles // h)) % 2 == 0
    out = np.where(sq & mask, a, b).astype(np.float32)
    return out


def pose_error(H, T, shape):
    """Distance in fixed pixels between H and T at every moving pixel."""
    h, w = shape
    y, x = np.mgrid[1:h + 1, 1:w + 1]
    p = np.stack([x, y, np.ones_like(x)], -1).astype(np.float64)
    a, b = p @ H.T, p @ T.T
    return np.hypot(*(a[..., :2] / a[..., 2:] - b[..., :2] / b[..., 2:]).transpose(2, 0, 1))

# %% [markdown]
# ## One call
#
# `register` finds POS-GIFT keypoints on both images, matches them, fits a homography to the
# matches and then refines it by climbing the SMI similarity score on the GPU.

# %%
zcmir.register(moving, fixed)  # the first call compiles the GPU pipelines

t = time.perf_counter()
r = zcmir.register(moving, fixed)
print(r, f"in {time.perf_counter() - t:.2f} s")
print(np.round(r.H, 4))

# %%
warped, mask = r.warp(moving, mask=True)
err = pose_error(r.H, truth, moving.shape)

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].imshow(checkerboard(warped, fixed, mask), cmap="gray"); ax[0].set_title("checkerboard: warped moving and fixed")
im = ax[1].imshow(err, cmap="viridis"); ax[1].set_title(f"distance to the true pose (px), max {err.max():.2f}")
plt.colorbar(im, ax=ax[1], fraction=0.046)
for a in ax: a.axis("off")
plt.tight_layout()

# %% [markdown]
# ## The matches behind the first fit
#
# `zcmir.match` returns the keypoints and their correspondences. Lines join the matches the
# robust fit kept.

# %%
mt = zcmir.match(moving, fixed)
print(f"{len(mt.moving)} moving and {len(mt.fixed)} fixed keypoints, "
      f"{len(mt.pairs)} matches, {mt.inliers.sum()} kept")

gap = 20
canvas = np.full((max(moving.shape[0], fixed.shape[0]), moving.shape[1] + gap + fixed.shape[1]), 255, np.uint8)
canvas[:moving.shape[0], :moving.shape[1]] = moving
canvas[:fixed.shape[0], moving.shape[1] + gap:] = fixed
keep = mt.pairs[mt.inliers]
pm = mt.moving[keep[:, 0], :2] - 1
pf = mt.fixed[keep[:, 1], :2] - 1 + [moving.shape[1] + gap, 0]

plt.figure(figsize=(12, 5.5))
plt.imshow(canvas, cmap="gray")
sel = np.random.default_rng(0).choice(len(keep), min(150, len(keep)), replace=False)
for i in sel:
    plt.plot([pm[i, 0], pf[i, 0]], [pm[i, 1], pf[i, 1]], lw=0.6, alpha=0.7, c="tab:cyan")
plt.scatter(*pm[sel].T, s=6, c="yellow"); plt.scatter(*pf[sel].T, s=6, c="yellow")
plt.title(f"{len(sel)} of the {len(keep)} kept matches")
plt.axis("off")
plt.tight_layout()

# %% [markdown]
# ## Refine from a rough guess
#
# `refine` starts from any pose and climbs the score. Here the start is the true pose shifted
# by 25 pixels and turned by 4°.

# %%
c = np.array([fixed.shape[1], fixed.shape[0]]) / 2
a = np.deg2rad(4)
nudge = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
shift = lambda dx, dy: np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
H0 = shift(*(c + [20, -15])) @ nudge @ shift(*-c) @ truth

r2 = zcmir.refine(moving, fixed, H0)
w0, m0 = zcmir.Registration(H=H0, score=0, model="homography", moving_shape=moving.shape, fixed_shape=fixed.shape).warp(moving, mask=True)
w1, m1 = r2.warp(moving, mask=True)

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].imshow(checkerboard(w0, fixed, m0), cmap="gray"); ax[0].set_title(f"start, {pose_error(H0, truth, moving.shape).max():.1f} px off")
ax[1].imshow(checkerboard(w1, fixed, m1), cmap="gray"); ax[1].set_title(f"refined in {r2.iterations} iterations, {pose_error(r2.H, truth, moving.shape).max():.2f} px off")
for a in ax: a.axis("off")
plt.tight_layout()

# %% [markdown]
# ## A spline on top
#
# `spline=True` adds a cubic B-spline lattice to the homography for local distortion. To see it
# work, bend the moving image with a smooth 4 pixel wave and register the bent copy.

# %%
from scipy.ndimage import map_coordinates

h, w = moving.shape
yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
wave = lambda x, y: (4 * np.sin(2 * np.pi * y / h + 0.4), 4 * np.cos(2 * np.pi * x / w + 1.1))
dx, dy = wave(xx, yy)
bent = map_coordinates(moving.astype(np.float64), [yy + dy, xx + dx], order=1).astype(np.uint8)

r3 = zcmir.register(bent, fixed, spline=True)
print(r3)

# both fields less the true homography: what the spline adds to it
fh, fw = fixed.shape
fy, fx = np.mgrid[0:fh, 0:fw].astype(np.float64)
q = np.stack([fx + 1, fy + 1, np.ones_like(fx)], -1) @ np.linalg.inv(truth).T
qx, qy = q[..., 0] / q[..., 2] - 1, q[..., 1] / q[..., 2] - 1
local = r3.displacement() - np.stack([qx - fx, qy - fy], -1)
inside = (qx > 8) & (qy > 8) & (qx < w - 9) & (qy < h - 9)
ax_, ay_ = wave(qx, qy)

s = 24
fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
for a, (u, v), title in [(ax[0], (-ax_, -ay_), "the applied wave"), (ax[1], (local[..., 0], local[..., 1]), "the recovered spline")]:
    a.imshow(fixed, cmap="gray", alpha=0.6)
    m = inside[::s, ::s]
    a.quiver(fx[::s, ::s][m], fy[::s, ::s][m], u[::s, ::s][m], v[::s, ::s][m], color="tab:red", angles="xy", scale_units="xy", scale=0.25, width=0.004)
    a.set_title(title); a.axis("off")
plt.tight_layout()
print(f"median difference {np.median(np.hypot(local[..., 0] + ax_, local[..., 1] + ay_)[inside]):.2f} px")

# %%
assert pose_error(r.H, truth, moving.shape).max() < 3
assert pose_error(r2.H, truth, moving.shape).max() < 3
