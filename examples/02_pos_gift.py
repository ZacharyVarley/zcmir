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
# # POS-GIFT: keypoints, descriptors and matching
#
# POS-GIFT matches keypoints between images from different sensors. Everything it measures comes
# from phase congruency, a measure of local structure that responds to an edge the same way
# whether it is light on dark or dark on light. This notebook follows each step on the Landsat
# pair from the first notebook.

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

import zcmir

moving, fixed, truth = zcmir.examples.load("landsat-b")
reg = zcmir.Registrar()
reg.set_images(moving, fixed)
det = reg.detect()
st = reg.pos_gift_structure()
print(det)
print({k: v for k, v in st.items() if k != "offsets"})

# %% [markdown]
# ## Oriented phase congruency
#
# Log-Gabor filters at six orientations measure how strongly each pixel's neighbourhood is
# organized along that orientation. These are the six energy maps of the fixed image.

# %%
E1 = reg.pos_gift_maps(1)
no = st["n_orient"]
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
for k, a in enumerate(ax.flat):
    a.imshow(E1[k], cmap="magma", vmax=np.percentile(E1[k], 99.5))
    a.set_title(f"{k * 180 // no}°")
    a.axis("off")
plt.tight_layout()

# %% [markdown]
# Together they give each pixel a dominant orientation (colour), how clearly one orientation
# dominates (saturation) and the total strength (brightness). Roads, field edges and the
# river bank stand out in both images. Their colours differ by the 30° turn between the images.

# %%
def orientation_rgb(E):
    t = 2 * np.pi * np.arange(len(E)) / len(E)          # orientations are axial: double the angle
    c, s = np.tensordot(np.cos(t), E, 1), np.tensordot(np.sin(t), E, 1)
    total = E.sum(0) + 1e-12
    hue = (np.arctan2(s, c) / (2 * np.pi)) % 1
    sat = np.clip(2 * np.hypot(c, s) / total, 0, 1)
    val = np.clip(total / np.percentile(total, 99.5), 0, 1) ** 0.8
    return hsv_to_rgb(np.stack([hue, sat, val], -1))

E0 = reg.pos_gift_maps(0)
fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
ax[0].imshow(orientation_rgb(E0)); ax[0].set_title("moving")
ax[1].imshow(orientation_rgb(E1)); ax[1].set_title("fixed")
for a in ax: a.axis("off")
plt.tight_layout()

# %% [markdown]
# ## Keypoints at six scales
#
# Corners of the phase congruency map are found on a pyramid: the image, two thirds of it,
# then halvings of both. Colour shows the scale each keypoint came from.

# %%
K0, K1 = reg.keypoints(0), reg.keypoints(1)
scales = np.unique(np.r_[K0[:, 3], K1[:, 3]])
fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
for a, img, K in [(ax[0], moving, K0), (ax[1], fixed, K1)]:
    a.imshow(img, cmap="gray")
    for s, c in zip(scales[::-1], plt.cm.plasma(np.linspace(0.1, 0.9, len(scales)))):
        m = K[:, 3] == s
        a.scatter(K[m, 0] - 1, K[m, 1] - 1, s=4 * s, color=c, label=f"×{s:g}  ({m.sum()})")
    a.set_title(f"{len(K)} keypoints"); a.axis("off")
ax[1].legend(loc="lower left", fontsize=8, markerscale=2)
plt.tight_layout()

# %% [markdown]
# ## One descriptor
#
# Each keypoint is described by the oriented energies, smoothed, at 37 points: its centre and
# 12 directions on 3 rings. Each small star below shows the six energies at one point, one
# coloured bar per orientation.
#
# The two keypoints are the same place on the ground. The truth pose pairs them here.

# %%
D0, D1 = reg.pos_gift_descriptors(0), reg.pos_gift_descriptors(1)
nd, nr = st["n_dir"], st["n_ring"]

def map_points(H, p):
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:]

# true pairs at full resolution: a moving keypoint and the fixed keypoint at its true position
s0, s1 = np.nonzero(K0[:, 3] == 1)[0], np.nonzero(K1[:, 3] == 1)[0]
p = map_points(truth, K0[s0, :2])
d = np.hypot(*(p[:, None] - K1[s1, None, :2].transpose(1, 0, 2)).transpose(2, 0, 1))
near = d.argmin(1)
pairs = np.c_[s0, s1[near]][d.min(1) < 1.5]
print(f"{len(pairs)} true pairs at full resolution")

def roll(D, k):
    """A descriptor as the moving image turned by k direction steps would give it."""
    ring = np.roll(np.roll(D[: nd * nr].reshape(nd, nr, no), k, 0), -k, 2)
    return np.r_[ring.reshape(-1, no), np.roll(D[nd * nr:], -k, 1)]

cx, cy = moving.shape[1] / 2, moving.shape[0] / 2
central = np.hypot(K0[pairs[:, 0], 0] - cx, K0[pairs[:, 0], 1] - cy) < 150
cand = pairs[central]
i, j = cand[np.argmin([np.sum((roll(D0[a], -1) - D1[b]) ** 2) for a, b in cand])]

def draw(ax, img, kp, D, title):
    x, y = kp[0] - 1, kp[1] - 1
    r = 50
    ax.imshow(img, cmap="gray", alpha=0.55)
    ax.set_xlim(x - r, x + r); ax.set_ylim(y + r, y - r)
    for R in st["radii"]:
        ax.add_patch(plt.Circle((x, y), R, fill=False, color="black", lw=0.6, alpha=0.4))
    scale = 6 / D.max()
    colors = hsv_to_rgb(np.stack([np.arange(no) / no, np.full(no, 0.9), np.full(no, 0.85)], -1))
    for (ox, oy), v in zip(st["offsets"], D):
        px, py = x + ox, y + oy
        for o in range(no):
            t = np.pi * o / no
            u, w = np.cos(t) * v[o] * scale, np.sin(t) * v[o] * scale
            ax.plot([px - u, px + u], [py - w, py + w], color=colors[o], lw=2, solid_capstyle="round")
    ax.plot(x, y, "+", ms=10, color="black")
    ax.set_title(title); ax.axis("off")

fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
draw(ax[0], moving, K0[i], D0[i], f"moving keypoint {i}")
draw(ax[1], fixed, K1[j], D1[j], f"fixed keypoint {j}")
plt.tight_layout()

# %% [markdown]
# The same two descriptors as 37 × 6 arrays (points down, orientations across). The fixed one
# matches the moving one turned by one 30° step: the points move one direction around each
# ring and the orientations move one channel.

# %%
fig, ax = plt.subplots(1, 3, figsize=(9, 6))
vmax = max(D0[i].max(), D1[j].max())
for a, D, title in [(ax[0], D0[i], "moving"), (ax[1], roll(D0[i], -1), "moving, turned 30°"), (ax[2], D1[j], "fixed")]:
    a.imshow(D, cmap="viridis", vmax=vmax, aspect="auto"); a.set_title(title)
    a.set_xticks(range(no)); a.set_xticklabels([f"{k * 180 // no}°" for k in range(no)], fontsize=7)
    a.set_yticks([])
ax[0].set_ylabel("sample point")
plt.tight_layout()

# %% [markdown]
# ## One rotation for every keypoint
#
# The images differ by one global rotation, so the descriptors are compared under 12 rotation
# hypotheses and the matcher keeps the best. Over the true pairs, the descriptor distance is
# smallest at 330°, the rotation between the two images.

# %%
dist = [np.mean([np.sum((roll(D0[a], k) - D1[b]) ** 2) for a, b in pairs]) for k in range(nd)]
angles = np.arange(nd) * 360 // nd
plt.figure(figsize=(8, 3.5))
plt.bar(angles, dist, width=20, color=["tab:orange" if k == np.argmin(dist) else "tab:gray" for k in range(nd)])
plt.xticks(angles); plt.xlabel("rotation hypothesis (°)"); plt.ylabel("mean descriptor distance")
plt.tight_layout()

# %% [markdown]
# ## Matches and the fit
#
# Nearest neighbours between descriptors, a robust affine fit over all scales, then POS guided
# re-matching at full resolution and a final homography.

# %%
mt = reg.match()
print(mt)
km, kf, mj = reg.keypoints(0), reg.keypoints(1), reg.matches()
ii = np.nonzero(mj >= 0)[0]
err = np.hypot(*(map_points(mt.H, km[ii, :2]) - kf[mj[ii], :2]).T)
keep = ii[err <= 3]

gap = 20
canvas = np.full((max(moving.shape[0], fixed.shape[0]), moving.shape[1] + gap + fixed.shape[1]), 255, np.uint8)
canvas[:moving.shape[0], :moving.shape[1]] = moving
canvas[:fixed.shape[0], moving.shape[1] + gap:] = fixed
plt.figure(figsize=(12, 5.5))
plt.imshow(canvas, cmap="gray")
for a in keep:
    b = mj[a]
    plt.plot([km[a, 0] - 1, kf[b, 0] - 1 + moving.shape[1] + gap], [km[a, 1] - 1, kf[b, 1] - 1], lw=0.5, alpha=0.6, c="tab:cyan")
plt.title(f"{len(keep)} matches within 3 px of the fit, rotation {mt.rotation_deg:g}°")
plt.axis("off")
plt.tight_layout()

# %%
h, w = moving.shape
corners = np.array([[1, 1], [w, 1], [w, h], [1, h]], np.float64)
corner_err = np.hypot(*(map_points(mt.H, corners) - map_points(truth, corners)).T)
print("corner distance to the true pose (px):", np.round(corner_err, 2))
assert corner_err.max() < 5
assert np.argmin(dist) == 11
