"""The example image pairs the web app ships (its presets), as numpy arrays.

    moving, fixed, truth = zcmir.examples.load("landsat-b")

truth is the known pose (moving → fixed, 1-based pixels) where the pair has one, else None.
"""
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def _root():
    # the wheel carries the app as zcmir/web/; a checkout has it in web/app/
    for d in (_HERE / "web" / "presets", _HERE.parent.parent / "web" / "app" / "presets"):
        if (d / "presets.json").exists():
            return d
    raise FileNotFoundError("zcmir: the example images are not in this installation")


def names():
    """The examples' ids."""
    return [p["id"] for p in json.loads((_root() / "presets.json").read_text())["presets"]]


def info(name):
    """An example's description (its group, name, source and, where known, truth)."""
    for p in json.loads((_root() / "presets.json").read_text())["presets"]:
        if p["id"] == name:
            return p
    raise KeyError(f"no example {name!r}; zcmir.examples.names() lists them")


def load(name):
    """(moving, fixed, truth): grayscale uint8 images and the known pose (or None)."""
    from PIL import Image

    p = info(name)
    root = _root()
    img = lambda f: np.asarray(Image.open(root / f).convert("L"))
    truth = np.array(p["truth"], np.float64) if "truth" in p else None
    return img(p["moving"]), img(p["fixed"]), truth
