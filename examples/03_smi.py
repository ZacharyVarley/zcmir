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
# # SMI: the score, the maps and the Jacobian
#
# zcmir scores an alignment with square-loss mutual information (SMI). Every quantity in the
# score is a sum over the overlap of the two images, so the same 45 sums give the score at one
# pose, the score at every translation (the shift map), the score at every rotation and scale
# (the roto-scale map), and the gradient that Refine climbs. This notebook builds each of them
# in NumPy from the engine's features and checks it against the engine.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.ndimage import map_coordinates

import zcmir

moving, fixed, truth = zcmir.examples.load("landsat-b")
reg = zcmir.Registrar(symmetric=False)   # the forward direction: moving onto fixed
reg.set_images(moving, fixed)

# %% [markdown]
# ## Four features per pixel
#
# Each image becomes four planes: its rank r (histogram equalized intensity), r², r³ and the
# gradient magnitude of r, whitened so the four are uncorrelated with unit variance.

# %%
A, B = reg.features(0), reg.features(1)
names = ["r", "r²", "r³", "|∇r|"]
fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))
for k in range(4):
    for row, F, side in [(0, A, "moving"), (1, B, "fixed")]:
        ax[row, k].imshow(F[k], cmap="gray", vmin=np.percentile(F[k], 1), vmax=np.percentile(F[k], 99))
        ax[row, k].set_title(f"{side} {names[k]}"); ax[row, k].axis("off")
plt.tight_layout()

# %% [markdown]
# ## The overlap at one pose
#
# At a pose H, every fixed pixel x looks up the moving features at H⁻¹x (bilinear). A weight w
# is 1 where the moving image covers x and ramps to 0 over its last 2 pixels.

# %%
hm, wm = A.shape[1:]
hf, wf = B.shape[1:]
TRI = [(i, j) for i in range(4) for j in range(i, 4)]


def ramp(v, lo, hi):
    """The edge taper (value and slope) over 2 pixels inside [lo, hi]."""
    a, b = (v - lo) / 2, (hi - v) / 2
    ca, cb = np.clip(a, 0, 1), np.clip(b, 0, 1)
    da = np.where((a > 0) & (a < 1), 0.5, 0.0)
    db = np.where((b > 0) & (b < 1), -0.5, 0.0)
    return ca * cb, da * cb + ca * db


def sample(H):
    """The moving features at every fixed pixel under H, their gradients, and the weight."""
    y, x = np.mgrid[0:hf, 0:wf].astype(np.float64)
    X = np.stack([x + 1, y + 1, np.ones_like(x)])          # 1-based pixel coordinates
    ph = np.tensordot(np.linalg.inv(H), X, 1)
    fx, fy = ph[0] / ph[2] - 1, ph[1] / ph[2] - 1
    ok = (fx >= 2) & (fy >= 2) & (fx <= wm - 3) & (fy <= hm - 3)
    tx, dtx = ramp(fx, 2, wm - 3)
    ty, dty = ramp(fy, 2, hm - 3)
    x0 = np.clip(np.floor(fx).astype(int), 0, wm - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, hm - 2)
    ax_, ay_ = fx - x0, fy - y0
    v00, v10, v01, v11 = A[:, y0, x0], A[:, y0, x0 + 1], A[:, y0 + 1, x0], A[:, y0 + 1, x0 + 1]
    return dict(
        va=(v00 * (1 - ax_) + v10 * ax_) * (1 - ay_) + (v01 * (1 - ax_) + v11 * ax_) * ay_,
        gx=(v10 - v00) * (1 - ay_) + (v11 - v01) * ay_,
        gy=(v01 - v00) * (1 - ax_) + (v11 - v10) * ax_,
        w=np.where(ok, tx * ty, 0.0),
        dw=np.where(ok, np.stack([dtx * ty, tx * dty]), 0.0),
        ph=ph, X=X,
    )


S = sample(truth)
fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
ax[0].imshow(S["w"], cmap="gray"); ax[0].set_title("weight w")
ax[1].imshow(S["va"][0] * S["w"], cmap="gray"); ax[1].set_title("moving r at H⁻¹x")
ax[2].imshow(B[0], cmap="gray"); ax[2].set_title("fixed r")
for a in ax: a.axis("off")
plt.tight_layout()

# %% [markdown]
# ## 45 sums
#
# The score needs the overlap n = Σw, the means Σw a_i and Σw b_j, the cross products
# Σw a_i b_j and the second moments of each side: 1 + 4 + 4 + 16 + 10 + 10 = 45 sums.

# %%
def moments(va, vb, w):
    s = [w.sum()] + [(w * va[i]).sum() for i in range(4)] + [(w * vb[j]).sum() for j in range(4)]
    s += [(w * va[i] * vb[j]).sum() for i in range(4) for j in range(4)]
    s += [(w * va[i] * va[j]).sum() for i, j in TRI] + [(w * vb[i] * vb[j]).sum() for i, j in TRI]
    return np.array(s)


m_np = moments(S["va"], B, S["w"])
m_gpu = reg.moments(truth)
print(f"largest difference to the engine: {np.abs(m_np - m_gpu).max() / np.abs(m_gpu).max():.1e} of the largest sum")

# %% [markdown]
# ## The score
#
# With C the cross-covariance of the two sides' features and G_A, G_B each side's covariance
# over the overlap, SMI is the squared norm of the whitened cross-covariance:
#
# $$S = n \, \mathrm{tr}\left(G_A^{-1} C \, G_B^{-1} C^\top\right)$$

# %%
def smi(s, ridge=1e-4):
    """SMI from the 45 sums (s: (45,) or (45, ...) for many at once)."""
    s = np.asarray(s, np.float64)
    n = s[0]
    ok = n > 0.05 * min(hf * wf, hm * wm)
    n = np.where(ok, n, 1.0)
    ma, mb = s[1:5] / n, s[5:9] / n
    C = s[9:25].reshape((4, 4) + n.shape) / n - ma[:, None] * mb[None]
    Ga, Gb = np.zeros((4, 4) + n.shape), np.zeros((4, 4) + n.shape)
    for k, (i, j) in enumerate(TRI):
        Ga[i, j] = Ga[j, i] = s[25 + k] / n - ma[i] * ma[j]
        Gb[i, j] = Gb[j, i] = s[35 + k] / n - mb[i] * mb[j]
    for G in (Ga, Gb):
        tr = np.trace(G) / 4
        for i in range(4):
            G[i, i] += ridge * tr
    Ga, Gb, C = (np.moveaxis(x, (0, 1), (-2, -1)) for x in (Ga, Gb, C))
    W = np.linalg.solve(Ga, C) @ np.linalg.inv(Gb)
    return np.where(ok, n * np.sum(W * C, axis=(-2, -1)), np.nan)


print(f"NumPy {smi(m_np):.2f}   engine {reg.score(truth).fwd:.2f}")

shift = lambda dx, dy: np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
dxs = np.arange(-30, 31, 2)
plt.figure(figsize=(8, 3.5))
plt.plot(dxs, [smi(moments(**{k: v for k, v in sample(shift(d, 0) @ truth).items() if k in ("va", "w")}, vb=B)) for d in dxs], "o-", label="NumPy")
plt.plot(dxs, [reg.score(shift(d, 0) @ truth).fwd for d in dxs], "x", ms=9, label="engine")
plt.xlabel("x shift from the true pose (px)"); plt.ylabel("SMI"); plt.legend()
plt.tight_layout()

# %% [markdown]
# ## The shift map
#
# Shifting the moving image by d changes every sum to a cross-correlation: Σ_x A(x + d) B(x)
# for one of 15 moving planes A (w, w a_i, w a_i a_j) and 15 fixed planes B (1, b_j, b_i b_j).
# FFTs give all 45 at every shift at once, and SMI turns them into a map. Here the start pose is
# the truth moved by (14, −9) pixels.

# %%
H0 = shift(14, -9) @ truth
S0 = sample(H0)
w, va = S0["w"], S0["va"]
planes_a = [w] + [w * va[i] for i in range(4)] + [w * va[i] * va[j] for i, j in TRI]
planes_b = [np.ones((hf, wf))] + [B[j] for j in range(4)] + [B[i] * B[j] for i, j in TRI]
PAIRS = [(0, 0)] + [(1 + i, 0) for i in range(4)] + [(0, 1 + j) for j in range(4)]
PAIRS += [(1 + i, 1 + j) for i in range(4) for j in range(4)]
PAIRS += [(5 + k, 0) for k in range(10)] + [(0, 5 + k) for k in range(10)]

P = 1024
FA = [np.fft.rfft2(a, (P, P)) for a in planes_a]
FB = [np.fft.rfft2(b, (P, P)) for b in planes_b]
sums = np.stack([np.fft.irfft2(FA[a] * np.conj(FB[b]), (P, P)) for a, b in PAIRS])

L = 64                                                      # shifts up to ±64 px
idx = np.r_[P - L:P, 0:L + 1]
sums = sums[:, idx][:, :, idx]
shift_np = smi(sums)
iy, ix = np.unravel_index(np.nanargmax(shift_np), shift_np.shape)
print(f"NumPy peak at ({ix - L}, {iy - L})")

show = [(0, "n"), (9, "Σ a₁b₁"), (14, "Σ a₂b₂"), (24, "Σ a₄b₄")]
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
for a, (k, t) in zip(ax, show):
    a.imshow(sums[k], cmap="coolwarm", extent=(-L, L, L, -L)); a.set_title(t)
plt.suptitle("4 of the 45 correlations")
plt.tight_layout()

# %%
sm = reg.shift_map(H0, exact=True, calibrated=False)
cell = sm.canvas[0] / sm.cw, sm.canvas[1] / sm.ch
eng = np.fft.fftshift(sm.map)
ext = (-sm.n / 2 * cell[0], sm.n / 2 * cell[0], sm.ny / 2 * cell[1], -sm.ny / 2 * cell[1])

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].imshow(shift_np, cmap="magma", extent=(-L, L, L, -L)); ax[0].plot(ix - L, iy - L, "c+", ms=14)
ax[0].set_title(f"NumPy, peak ({ix - L}, {iy - L})")
ax[1].imshow(eng, cmap="magma", extent=ext); ax[1].plot(sm.dx, sm.dy, "c+", ms=14)
ax[1].set_xlim(-L, L); ax[1].set_ylim(L, -L)
ax[1].set_title(f"engine, peak ({sm.dx:.1f}, {sm.dy:.1f})")
for a in ax: a.set_xlabel("x shift (px)")
ax[0].set_ylabel("y shift (px)")
plt.tight_layout()

# %% [markdown]
# ## The roto-scale map
#
# Resampled on log-polar coordinates about a centre, a rotation becomes a shift in angle and a
# scaling becomes a shift in log radius. The same 45 correlations on those planes give the score
# over every rotation and scale. Here the start pose is the truth turned by 6° and scaled by 1.08.
# The engine's map shows a calibrated z-score, so its contrast differs from the raw SMI.

# %%
c = np.array(reg.rs_map(truth, return_maps=False).center)    # the engine's centre (1-based)
a6 = np.deg2rad(6)
turn = np.array([[1.08 * np.cos(a6), -1.08 * np.sin(a6), 0], [1.08 * np.sin(a6), 1.08 * np.cos(a6), 0], [0, 0, 1]])
H1 = shift(*c) @ turn @ shift(*-c) @ truth

n_th, n_rho = 360, 160
th = np.arange(n_th) * 2 * np.pi / n_th
rho = np.linspace(np.log(8), np.log(0.75 * max(hf, wf)), n_rho)
TH, RHO = np.meshgrid(th, rho)
xs, ys = c[0] - 1 + np.exp(RHO) * np.cos(TH), c[1] - 1 + np.exp(RHO) * np.sin(TH)
area = np.exp(2 * RHO)                                      # each cell's share of the image
logpolar = lambda img: map_coordinates(img, [ys, xs], order=1, mode="constant", cval=0.0)

S1 = sample(H1)
w, va = S1["w"], S1["va"]
lp_a = [logpolar(p) * area for p in [w] + [w * va[i] for i in range(4)] + [w * va[i] * va[j] for i, j in TRI]]
inside = logpolar(np.ones((hf, wf)))
lp_b = [logpolar(p) * inside * area for p in planes_b]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow(logpolar(w * va[0]), cmap="gray", aspect="auto", extent=(0, 360, rho[-1], rho[0])); ax[0].set_title("moving r, log-polar")
ax[1].imshow(logpolar(B[0]), cmap="gray", aspect="auto", extent=(0, 360, rho[-1], rho[0])); ax[1].set_title("fixed r, log-polar")
for a in ax: a.set_xlabel("angle (°)")
ax[0].set_ylabel("log radius")
plt.tight_layout()

# %%
PR = 2 * n_rho
FA = [np.fft.fft2(a, (PR, n_th)) for a in lp_a]
FB = [np.fft.fft2(b, (PR, n_th)) for b in lp_b]
sums = np.stack([np.real(np.fft.ifft2(FA[a] * np.conj(FB[b]))) for a, b in PAIRS])
LR = 30                                                     # log-scale lags up to ±30 rows
rows = np.r_[PR - LR:PR, 0:LR + 1]
cols = np.r_[n_th // 2:n_th, 0:n_th // 2]
rs_np = smi(sums[:, rows][:, :, cols])
drho = rho[1] - rho[0]
i, j = np.unravel_index(np.nanargmax(rs_np), rs_np.shape)
peak_np = ((j - n_th // 2) * 360 / n_th, np.exp((i - LR) * drho))

rs = reg.rs_map(H1)
eng = np.fft.fftshift(rs.map)
nr_, nc_ = eng.shape
dl = rs.dlam
peak_eng = (-np.rad2deg(rs.dth), np.exp(-rs.dsg))           # the map's peak: the offset

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].imshow(rs_np, cmap="magma", aspect="auto", extent=(-180, 180, LR * drho, -LR * drho))
ax[0].plot(peak_np[0], np.log(peak_np[1]), "c+", ms=14); ax[0].set_title(f"NumPy, peak {peak_np[0]:.1f}°, ×{peak_np[1]:.3f}")
ax[1].imshow(eng, cmap="magma", aspect="auto", extent=(-180, 180, nr_ / 2 * dl, -nr_ / 2 * dl))
ax[1].plot(peak_eng[0], np.log(peak_eng[1]), "c+", ms=14); ax[1].set_title(f"engine, peak {peak_eng[0]:.1f}°, ×{peak_eng[1]:.3f}")
ax[1].set_ylim(LR * drho, -LR * drho)
for a in ax: a.set_xlabel("rotation (°)"); a.invert_yaxis()
ax[0].set_ylabel("log scale")
plt.tight_layout()

# %% [markdown]
# ## The Jacobian
#
# Refine moves the pose along 8 tangent directions of the homography. Each direction ξ_k moves
# every sample point H⁻¹x by a flow ∂p/∂ξ_k.

# %%
def destN(w, h):
    return np.array([[2 / w, 0, -(w + 1) / w], [0, 2 / h, -(h + 1) / h], [0, 0, 1.0]])


def hat(x):
    tx, ty, th, sg, al, ga, px, py = x
    iso = sg / 3
    return np.array([[iso + al, ga - th, tx], [ga + th, iso - al, ty], [px, py, -2 * iso]])


N = destN(wf, hf)
tangents = ["x shift", "y shift", "rotation", "scale", "stretch", "shear", "x perspective", "y perspective"]
H2 = np.linalg.inv(N) @ expm(hat([0.02, -0.015, 0.02, 0.01, 0, 0, 0, 0])) @ N @ truth   # a pose near the truth
S2 = sample(H2)
Hi, ph, X = np.linalg.inv(H2), S2["ph"], S2["X"]
flows = []
for k in range(8):
    Gk = np.linalg.inv(N) @ hat(np.eye(8)[k]) @ N
    v = -np.tensordot(Hi @ Gk, X, 1)
    flows.append(((v[0] * ph[2] - ph[0] * v[2]) / ph[2] ** 2, (v[1] * ph[2] - ph[1] * v[2]) / ph[2] ** 2))

s = 48
yy, xx = np.mgrid[0:hf:s, 0:wf:s]
fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))
for a, (u, v), t in zip(ax.flat, flows, tangents):
    a.quiver(xx, yy, u[::s, ::s], v[::s, ::s], angles="xy", color="tab:blue")
    a.set_xlim(0, wf); a.set_ylim(hf, 0); a.set_aspect("equal"); a.set_title(t); a.set_xticks([]); a.set_yticks([])
plt.tight_layout()

# %% [markdown]
# The score's derivative in the sample position combines each feature's image gradient with
# ∂S/∂a_i, which comes from the 45 sums. Multiplying by each flow gives one image per tangent:
# where in the overlap the score pulls the pose, and which way. Their sums are the gradient.

# %%
m2 = moments(S2["va"], B, S2["w"])
g = np.zeros(45)                                             # ∂S/∂(each sum)
for k in range(45):
    h = 1e-6 * max(abs(m2[k]), 1e-3 * m2[0])
    e = np.zeros(45); e[k] = h
    g[k] = (smi(m2 + e) - smi(m2 - e)) / (2 * h)

va, vb = S2["va"], B
dS_da = np.zeros_like(va)                                    # ∂S/∂a_i at every pixel
for i in range(4):
    dS_da[i] = g[1 + i] + sum(g[9 + 4 * i + j] * vb[j] for j in range(4))
    for k, (p, q) in enumerate(TRI):
        if p == i:
            dS_da[i] += g[25 + k] * va[q] * (2 if q == i else 1)
        elif q == i:
            dS_da[i] += g[25 + k] * va[p]
dS_dw = g[0] + sum(g[1 + i] * va[i] for i in range(4)) + sum(g[5 + j] * vb[j] for j in range(4))
dS_dw += sum(g[9 + 4 * i + j] * va[i] * vb[j] for i in range(4) for j in range(4))
dS_dw += sum(g[25 + k] * va[p] * va[q] for k, (p, q) in enumerate(TRI))
dS_dw += sum(g[35 + k] * vb[p] * vb[q] for k, (p, q) in enumerate(TRI))
dSx = S2["w"] * (dS_da * S2["gx"]).sum(0) + dS_dw * S2["dw"][0]
dSy = S2["w"] * (dS_da * S2["gy"]).sum(0) + dS_dw * S2["dw"][1]

per_pixel = [dSx * u + dSy * v for u, v in flows]
grad_np = np.array([p.sum() for p in per_pixel])
grad_gpu = reg.gradient(H2).grad

fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))
for a, img, t in zip(ax.flat, per_pixel, tangents):
    lim = np.percentile(np.abs(img), 99)
    a.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim); a.set_title(t); a.axis("off")
plt.tight_layout()

# %%
x = np.arange(8)
plt.figure(figsize=(9, 3.5))
plt.bar(x - 0.2, grad_np / np.abs(grad_gpu).max(), 0.4, label="NumPy")
plt.bar(x + 0.2, grad_gpu / np.abs(grad_gpu).max(), 0.4, label="engine")
plt.xticks(x, tangents, rotation=30); plt.ylabel("gradient (scaled)"); plt.legend()
plt.tight_layout()
print(f"relative difference {np.linalg.norm(grad_np - grad_gpu) / np.linalg.norm(grad_gpu):.1e}")

# %%
assert np.abs(m_np - m_gpu).max() < 1e-4 * np.abs(m_gpu).max()
assert abs(smi(m_np) - reg.score(truth).fwd) < 1e-4 * reg.score(truth).fwd
assert (ix - L, iy - L) == (14, -9) and np.hypot(sm.dx - 14, sm.dy + 9) < 2
assert abs(peak_np[0] - 6) < 2 and abs(peak_eng[0] - 6) < 2
assert np.linalg.norm(grad_np - grad_gpu) < 1e-2 * np.linalg.norm(grad_gpu)
