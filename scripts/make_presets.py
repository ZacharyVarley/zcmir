#!/usr/bin/env python3
"""Build the app's example pairs (web/app/presets/) from openly licensed sources.

    python scripts/make_presets.py SRC_DIR       (numpy, Pillow)

SRC_DIR holds the downloaded sources (see web/app/presets/ATTRIBUTION.md for where each comes from):
  landsat/prariver_oli_2014033_lrg.jpg, landsat/prariver_oli_2017009_lrg.jpg   (NASA Earth Observatory)
  in718/LEROY_0118_BSE_2_XXXwarped.png, in718/ipfz_XXX.png  (NIST mds2-2767; sections 65, 85)
  mida/<set>/A_<name>_T.<ext>, mida/<set>/B_<name>_R.<ext>, mida/<set>/info_test.csv  (Zenodo 5557568)

Each preset is moving.* + fixed.*, listed in presets.json, all grayscale (the engine's own 0.299 R + 0.587 G +
0.114 B, so registration sees the same image as from the colour original) to keep the wheels small; "truth" (where known) is the 3×3 pose
moving → fixed in 1-based pixel centres, the app's convention.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "web" / "app" / "presets"


def homog(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
    h = np.linalg.svd(np.array(A))[2][-1]
    return (h / h[-1]).reshape(3, 3)


def bilinear(img, x, y):
    h, w = img.shape[:2]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    out = np.zeros(x.shape + img.shape[2:], np.float64).reshape(x.shape + (-1,))
    src = img.reshape(h, w, -1).astype(np.float64)
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            xi, yi = np.clip(x0 + dx, 0, w - 1), np.clip(y0 + dy, 0, h - 1)
            out += wx * wy * src[yi, xi]
    return out.reshape(x.shape + img.shape[2:])


def gray(img):
    """RGB → gray with the engine's weights (float, 0–255)."""
    return np.asarray(img, np.float64)[..., :3] @ np.array([0.299, 0.587, 0.114])


def save(img, path, quality=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(np.clip(img + 0.5, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img)
    if quality:
        im.save(path, quality=quality, subsampling=0 if quality >= 90 else 2)
    else:
        im.save(path, optimize=True)
    return path.name


def landsat(src, presets):
    """Dry season (2014) crop = fixed; the flood scene (2017) through a known similarity = moving."""
    dry = np.asarray(Image.open(src / "landsat" / "prariver_oli_2014033_lrg.jpg").convert("RGB"))
    dry = gray(dry)
    wet = gray(Image.open(src / "landsat" / "prariver_oli_2017009_lrg.jpg").convert("RGB"))
    cases = [("a", (700, 520), 0.0, 1.00, (40, -25)), ("b", (1150, 700), -30.0, 1.20, (-30, 20)), ("c", (500, 1000), 75.0, 0.85, (15, 35))]
    for tag, (x0, y0), deg, s, (dx, dy) in cases:
        n = 640
        fixed = dry[y0:y0 + n, x0:x0 + n]
        m = 560
        th = np.radians(deg)
        # moving 1-based centre → scene 1-based: about the moving centre, rotate/scale, then to the fixed crop's centre + (dx, dy)
        cm = (m + 1) / 2
        c, si = np.cos(th) * s, np.sin(th) * s
        G = np.array([[1, 0, x0 + (n + 1) / 2 + dx], [0, 1, y0 + (n + 1) / 2 + dy], [0, 0, 1]]) @ \
            np.array([[c, -si, 0], [si, c, 0], [0, 0, 1]]) @ np.array([[1, 0, -cm], [0, 1, -cm], [0, 0, 1]])
        jj, ii = np.meshgrid(np.arange(1, m + 1, dtype=float), np.arange(1, m + 1, dtype=float))
        P = np.stack([jj, ii, np.ones_like(jj)], -1) @ G.T
        moving = bilinear(wet, P[..., 0] - 1, P[..., 1] - 1)
        truth = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ G
        d = OUT / f"landsat-{tag}"
        presets.append(dict(
            id=f"landsat-{tag}", group="Landsat 8 · flood vs dry season", name=f"Pra River {tag.upper()} (rot {deg:+.0f}°, ×{s:.2f})",
            moving=f"{d.name}/{save(moving, d / 'moving.jpg', 92)}", fixed=f"{d.name}/{save(fixed, d / 'fixed.jpg', 92)}",
            source="nasa-landsat", model="homography", truth=truth.tolist()))


def in718(src, presets):
    for sec in (65, 85):
        a = np.asarray(Image.open(src / "in718" / f"LEROY_0118_BSE_2_{sec:03d}warped.png"), dtype=np.float64)
        v = a[(a > 0) & (a < 65535)]  # saturated pixels (0.3–0.5% at 65535) would set the top
        lo, hi = np.percentile(v, [0.5, 99.5])
        bse = np.clip((a - lo) / (hi - lo) * 255, 0, 255)
        bse[a == 0] = 0
        ebsd = gray(Image.open(src / "in718" / f"ipfz_{sec:03d}.png").convert("RGB"))
        d = OUT / f"in718-{sec}"
        presets.append(dict(
            id=f"in718-{sec}", group="IN718 · EBSD vs BSE (NIST AM Bench)", name=f"section {sec}",
            moving=f"{d.name}/{save(ebsd, d / 'moving.png')}", fixed=f"{d.name}/{save(bse, d / 'fixed.png')}",
            source="nist-ambench", model="homography"))


def mida(src, presets):
    sets = [("aerial", "png", "zh15_01_01", "Aerial · NIR vs RGB (Zurich)", "patch zh15"),
            ("aerial", "png", "zh9_02_02", "Aerial · NIR vs RGB (Zurich)", "patch zh9"),
            ("hist", "tif", "1B_A1", "Histology · SHG vs bright field", "core 1B-A1"),
            ("cyto", "png", "DU145_Fluo_4_f45_02_02", "Cells · fluorescence vs phase (hard)", "DU145")]
    for ds, ext, n, group, name in sets:
        rows = {r["Filename"]: r for r in csv.DictReader(open(src / "mida" / ds / "info_test.csv"))}
        r = rows[n]
        ref = [(float(r[f"X{i}_Ref"]) + .5, float(r[f"Y{i}_Ref"]) + .5) for i in range(1, 5)]
        tr = [(float(r[f"X{i}_Trans"]) + .5, float(r[f"Y{i}_Trans"]) + .5) for i in range(1, 5)]
        mov = gray(Image.open(src / "mida" / ds / f"A_{n}_T.{ext}").convert("RGB"))
        fix = gray(Image.open(src / "mida" / ds / f"B_{n}_R.{ext}").convert("RGB"))
        d = OUT / f"mida-{ds}-{n.split('_')[0].lower()}{n.split('_')[1].lower() if ds == 'aerial' else ''}"
        q = 92 if ds == "hist" else None
        presets.append(dict(
            id=d.name, group=group, name=name,
            moving=f"{d.name}/{save(mov, d / ('moving.jpg' if q else 'moving.png'), q)}",
            fixed=f"{d.name}/{save(fix, d / ('fixed.jpg' if q else 'fixed.png'), q)}",
            source="mida", model="homography", truth=homog(ref, tr).tolist()))


def main():
    src = Path(sys.argv[1])
    presets = []
    in718(src, presets)
    landsat(src, presets)
    mida(src, presets)
    (OUT / "presets.json").write_text(json.dumps({"presets": presets}, indent=1))
    for p in presets:
        print(p["id"], p.get("type", "pair"), p["moving"] if isinstance(p["moving"], str) else f"{len(p['moving'])} slices")


if __name__ == "__main__":
    main()
