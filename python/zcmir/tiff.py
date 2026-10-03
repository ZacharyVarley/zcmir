"""A minimal multi-page TIFF writer (no dependencies beyond numpy): 8- or 16-bit, gray or RGB,
uncompressed, one strip per page, written page by page (a stack never has to fit in memory).
Equal pages get an ImageJ description, so Fiji / ImageJ open the file as a z-stack.
"""
import struct
from pathlib import Path

import numpy as np


def _page(a):
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.dtype == np.bool_:
        a = a.astype(np.uint8) * 255
    if a.dtype not in (np.uint8, np.uint16):
        if np.issubdtype(a.dtype, np.floating):
            a = np.clip(np.rint(a * (255 if a.max() <= 1.0 else 1)), 0, 255).astype(np.uint8)
        else:
            a = np.clip(a, 0, 65535).astype(np.uint16)
    if a.ndim == 3 and a.shape[2] != 3:
        raise ValueError(f"a page is gray (h, w) or RGB(A) (h, w, 3|4), not {a.shape}")
    return np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<"), copy=False))


def write_tiff(path, pages, description=None):
    """Write images (an iterable of arrays) as one multi-page TIFF; returns the page count."""
    path = Path(path)
    count = 0
    first = None
    uniform = True
    with open(path, "wb") as f:
        f.write(b"II*\x00" + struct.pack("<I", 0))  # first IFD offset patched below
        prev_next = 4                                # where the previous "next IFD" pointer lives
        for a in pages:
            a = _page(a)
            h, w = a.shape[:2]
            c = 1 if a.ndim == 2 else 3
            bits = a.dtype.itemsize * 8
            sig = (w, h, c, bits)
            first = first or sig
            uniform &= sig == first
            # pixel data, then this page's IFD (its offsets are known once the data is written)
            data_off = f.tell()
            f.write(a.tobytes())
            if f.tell() & 1:
                f.write(b"\x00")
            extra = []
            tags = [(254, 4, 1, 0), (256, 4, 1, w), (257, 4, 1, h)]
            tags.append((258, 3, c, bits) if c == 1 else (258, 3, 3, None))
            tags += [(259, 3, 1, 1), (262, 3, 1, 1 if c == 1 else 2)]
            desc = None
            if count == 0:
                desc = description  # filled in below for ImageJ when the stack turns out uniform
                tags.append((270, 2, 0, None))
            tags += [(273, 4, 1, data_off), (277, 3, 1, c), (278, 4, 1, h), (279, 4, 1, a.nbytes), (284, 3, 1, 1)]
            ifd_off = f.tell()
            ifd_size = 2 + 12 * len(tags) + 4
            ext = ifd_off + ifd_size
            if c == 3:
                extra.append((258, ext, struct.pack("<3H", bits, bits, bits)))
                ext += 6
            if count == 0:
                # room for an ImageJ description (written when the file is complete)
                desc_room = 96 if description is None else len(description.encode()) + 1
                extra.append((270, ext, b"\x00" * desc_room))
                desc_pos = ext
                ext += desc_room
            body = struct.pack("<H", len(tags))
            for tag, typ, n, val in tags:
                if tag == 258 and c == 3:
                    body += struct.pack("<HHII", tag, typ, 3, extra[0][1])
                elif tag == 270:
                    body += struct.pack("<HHII", tag, typ, desc_room, desc_pos)
                elif typ == 3:
                    body += struct.pack("<HHIHH", tag, typ, n, val, 0)
                else:
                    body += struct.pack("<HHII", tag, typ, n, val)
            body += struct.pack("<I", 0)
            f.write(body)
            for _, _, b in extra:
                f.write(b)
            if f.tell() & 1:
                f.write(b"\x00")
            end = f.tell()
            f.seek(prev_next)
            f.write(struct.pack("<I", ifd_off))
            prev_next = ifd_off + 2 + 12 * len(tags)
            if count == 0:
                first_desc_pos = desc_pos
                first_desc_room = desc_room
                first_desc_count = ifd_off + 2 + 12 * [t[0] for t in tags].index(270) + 4  # its count field
            f.seek(end)
            count += 1
        if count:
            text = description if description is not None else (
                f"ImageJ=1.54f\nimages={count}\nslices={count}\nloop=false\n" if uniform and count > 1 else "")
            raw = text.encode()[: first_desc_room - 1] + b"\x00"
            f.seek(first_desc_pos)
            f.write(raw.ljust(first_desc_room, b"\x00"))
            f.seek(first_desc_count)
            f.write(struct.pack("<I", len(raw)))  # the text and its NUL, not the reserved room
    return count
