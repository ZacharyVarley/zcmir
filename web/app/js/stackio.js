// Writers for the stack export: a multi-page TIFF (8-bit gray or RGB, uncompressed; Fiji / ImageJ
// open it as a stack) and a stored ZIP. Both stream: each page goes into a Blob as soon as it is
// made (the browser can keep large Blobs on disk), checksums are updated as the bytes pass and
// combined at the end, so a large stack never has to be one array in memory.

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

/** CRC-32 (ZIP) of bytes, continuing from crc (0 to start). */
export function crc32(bytes, crc = 0) {
  let c = (crc ^ 0xffffffff) >>> 0;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// crc32Combine (zlib): the CRC of A‖B from crc(A), crc(B) and B's length, by squaring the GF(2)
// operator that appends one zero bit.
function gf2Times(mat, vec) {
  let sum = 0;
  for (let i = 0; vec; i++, vec >>>= 1) if (vec & 1) sum ^= mat[i];
  return sum >>> 0;
}
function gf2Square(square, mat) {
  for (let n = 0; n < 32; n++) square[n] = gf2Times(mat, mat[n]);
}
export function crc32Combine(crc1, crc2, len2) {
  if (len2 <= 0) return crc1 >>> 0;
  const even = new Uint32Array(32), odd = new Uint32Array(32);
  odd[0] = 0xedb88320;
  for (let n = 1, row = 1; n < 32; n++, row <<= 1) odd[n] = row;
  gf2Square(even, odd);  // 2 zero bits
  gf2Square(odd, even);  // 4 zero bits
  let c = crc1 >>> 0;
  for (;;) {
    gf2Square(even, odd);
    if (len2 % 2) c = gf2Times(even, c);
    len2 = Math.floor(len2 / 2);
    if (!len2) break;
    gf2Square(odd, even);
    if (len2 % 2) c = gf2Times(odd, c);
    len2 = Math.floor(len2 / 2);
    if (!len2) break;
  }
  return (c ^ crc2) >>> 0;
}

/** RGBA → 1 channel when every pixel is gray (r = g = b), else 3; transparent pixels count as gray. */
export function channelsOf(rgba) {
  for (let i = 0; i < rgba.length; i += 4) {
    if (rgba[i + 3] && (rgba[i] !== rgba[i + 1] || rgba[i] !== rgba[i + 2])) return 3;
  }
  return 1;
}

/** RGBA → packed gray (c = 1) or RGB (c = 3) bytes. */
export function pack(rgba, c) {
  const n = rgba.length >> 2, out = new Uint8Array(n * c);
  if (c === 1) for (let i = 0; i < n; i++) out[i] = rgba[4 * i];
  else for (let i = 0; i < n; i++) { out[3 * i] = rgba[4 * i]; out[3 * i + 1] = rgba[4 * i + 1]; out[3 * i + 2] = rgba[4 * i + 2]; }
  return out;
}

/**
 * A multi-page TIFF built page by page: pixel data first, every page's directory after it (its
 * offsets are only known then), little-endian, one uncompressed strip per page. With equal pages
 * the first directory carries an ImageJ description, so Fiji reads a z-stack, not a series.
 */
export class TiffStream {
  constructor() {
    this.pages = [];
    this.parts = [];
    this.off = 8;          // the header comes first
    this.dataCrc = 0;      // CRC of everything after the header so far
  }

  /** One page: w × h, c = 1 (gray) or 3 (RGB) channels, data = Uint8Array(w·h·c). */
  addPage(w, h, c, data) {
    const pad = data.length & 1 ? new Uint8Array(1) : null;
    this.pages.push({ w, h, c, off: this.off, bytes: data.length });
    this.dataCrc = crc32(data, this.dataCrc);
    this.parts.push(new Blob([data]));
    this.off += data.length;
    if (pad) { this.dataCrc = crc32(pad, this.dataCrc); this.parts.push(pad); this.off += 1; }
  }

  /** { parts, size, crc } of the whole file (header, pixel data, directories). */
  finish() {
    const pages = this.pages;
    const uniform = pages.every((p) => p.w === pages[0].w && p.h === pages[0].h && p.c === pages[0].c);
    const desc = uniform && pages.length > 1
      ? new TextEncoder().encode(`ImageJ=1.54f\nimages=${pages.length}\nslices=${pages.length}\nloop=false\n\0`)
      : null;
    // the directories, each followed by its out-of-line values
    const ifds = [];
    let off = this.off, first = 0, prevNext = null;
    pages.forEach((pg, k) => {
      const c = pg.c, d = k === 0 ? desc : null;
      const tags = [[254, 4, 1, 0], [256, 4, 1, pg.w], [257, 4, 1, pg.h], [258, 3, c, 8], [259, 3, 1, 1], [262, 3, 1, c === 1 ? 1 : 2]];
      if (d) tags.push([270, 2, d.length, 0]);
      tags.push([273, 4, 1, pg.off], [277, 3, 1, c], [278, 4, 1, pg.h], [279, 4, 1, pg.bytes], [284, 3, 1, 1]);
      const ifdOff = off, ifdSize = 2 + 12 * tags.length + 4;
      let extra = ifdOff + ifdSize;
      const bpsOff = c === 3 ? extra : 0;
      if (c === 3) extra += 6;
      const descOff = d ? extra : 0;
      if (d) extra += d.length;
      const end = extra + (extra & 1);
      const buf = new Uint8Array(end - ifdOff);
      const v = new DataView(buf.buffer);
      v.setUint16(0, tags.length, true);
      tags.forEach(([tag, type, count, value], j) => {
        const p = 2 + 12 * j;
        v.setUint16(p, tag, true);
        v.setUint16(p + 2, type, true);
        v.setUint32(p + 4, count, true);
        if (tag === 258 && c === 3) v.setUint32(p + 8, bpsOff, true);
        else if (tag === 270) v.setUint32(p + 8, descOff, true);
        else if (type === 3) v.setUint16(p + 8, value, true);
        else v.setUint32(p + 8, value, true);
      });
      if (c === 3) for (let q = 0; q < 3; q++) v.setUint16(bpsOff - ifdOff + 2 * q, 8, true);
      if (d) buf.set(d, descOff - ifdOff);
      if (prevNext) prevNext.v.setUint32(prevNext.p, ifdOff, true);
      else first = ifdOff;
      prevNext = { v, p: 2 + 12 * tags.length };
      ifds.push(buf);
      off = end;
    });
    if (off > 0xffffffff) throw new Error("the TIFF would exceed 4 GB; export PNG files instead");
    const head = new Uint8Array(8);
    head.set([0x49, 0x49, 42, 0]);
    new DataView(head.buffer).setUint32(4, first, true);
    let tailCrc = 0, tailLen = 0;
    for (const b of ifds) { tailCrc = crc32(b, tailCrc); tailLen += b.length; }
    const crc = crc32Combine(crc32Combine(crc32(head), this.dataCrc, this.off - 8), tailCrc, tailLen);
    return { parts: [head, ...this.parts, ...ifds], size: off, crc };
  }
}

/**
 * A stored (uncompressed) ZIP of files [{ name, parts: (Uint8Array | Blob)[], size?, crc? }] as a
 * Blob; size and crc must be given for Blob parts (Uint8Array parts are checksummed here).
 */
export function zipStore(files) {
  const enc = new TextEncoder();
  const now = new Date();
  const time = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
  const date = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
  const out = [], dir = [];
  let off = 0;
  for (const f of files) {
    const name = enc.encode(f.name);
    let { size, crc } = f;
    if (size == null || crc == null) {
      size = 0;
      crc = 0;
      for (const p of f.parts) {
        if (p instanceof Blob) throw new Error(`${f.name}: Blob parts need size and crc`);
        crc = crc32(p, crc);
        size += p.length;
      }
    }
    const lh = new Uint8Array(30 + name.length);
    const v = new DataView(lh.buffer);
    v.setUint32(0, 0x04034b50, true);
    v.setUint16(4, 20, true);
    v.setUint16(6, 0x0800, true);
    v.setUint16(10, time, true);
    v.setUint16(12, date, true);
    v.setUint32(14, crc, true);
    v.setUint32(18, size, true);
    v.setUint32(22, size, true);
    v.setUint16(26, name.length, true);
    lh.set(name, 30);
    out.push(lh, ...f.parts);
    dir.push({ name, crc, size, off });
    off += lh.length + size;
    if (off > 0xffffffff) throw new Error("the ZIP would exceed 4 GB; export fewer slices at a time");
  }
  const cd = [];
  let cdSize = 0;
  for (const e of dir) {
    const ch = new Uint8Array(46 + e.name.length);
    const v = new DataView(ch.buffer);
    v.setUint32(0, 0x02014b50, true);
    v.setUint16(4, 20, true);
    v.setUint16(6, 20, true);
    v.setUint16(8, 0x0800, true);
    v.setUint16(12, time, true);
    v.setUint16(14, date, true);
    v.setUint32(16, e.crc, true);
    v.setUint32(20, e.size, true);
    v.setUint32(24, e.size, true);
    v.setUint16(28, e.name.length, true);
    v.setUint32(42, e.off, true);
    ch.set(e.name, 46);
    cd.push(ch);
    cdSize += ch.length;
  }
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, dir.length, true);
  ev.setUint16(10, dir.length, true);
  ev.setUint32(12, cdSize, true);
  ev.setUint32(16, off, true);
  return new Blob([...out, ...cd, end], { type: "application/zip" });
}
