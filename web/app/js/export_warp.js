/**
 * Export fitted warps as Insight Transform File V1.0 (.tfm) and an NRRD
 * displacement field. Those are the ITK / 3D Slicer / SimpleITK formats —
 * not a CMIR-specific container.
 *
 * NRRD: http://teem.sourceforge.net/nrrd/format.html
 * .tfm: Insight Transform File V1.0 (itkTxtTransformIO)
 *
 * Displacement convention matches itk::ResampleImageFilter /
 * sitk.DisplacementFieldTransform: T(x_fixed) = x_fixed + d(x_fixed) = x_moving,
 * with pixel spacing 1 and origin 0 (0-based pixel physical coordinates).
 */
import { inv3, mul3 } from "./lie.js";

function cubicW(t) {
  const t2 = t * t, t3 = t2 * t;
  return [
    (1 - t) * (1 - t) * (1 - t) / 6,
    (3 * t3 - 6 * t2 + 4) / 6,
    (-3 * t3 + 3 * t2 + 3 * t + 1) / 6,
    t3 / 6,
  ];
}

function clampi(i, n) {
  return Math.min(n - 1, Math.max(0, i));
}

/** Pixel displacement of the cubic FFD at (x,y) in the lattice image. Matches wgsl/ffd.wgsl. */
export function ffdDisp(x, y, cps, gx, gy, w, h) {
  if (!cps || gx < 2 || gy < 2) return [0, 0];
  if (x < 0 || y < 0 || x > w - 1 || y > h - 1) return [0, 0];
  const hx = w * 0.5, hy = h * 0.5;
  const spx = Math.max(w - 1, 1) / (gx - 1);
  const spy = Math.max(h - 1, 1) / (gy - 1);
  const su = x / spx, sv = y / spy;
  const iu = Math.floor(su), iv = Math.floor(sv);
  const bu = cubicW(Math.min(1, Math.max(0, su - iu)));
  const bv = cubicW(Math.min(1, Math.max(0, sv - iv)));
  let dx = 0, dy = 0;
  for (let jj = 0; jj < 4; jj++) {
    const j = clampi(iv - 1 + jj, gy);
    for (let ii = 0; ii < 4; ii++) {
      const i = clampi(iu - 1 + ii, gx);
      const wt = bu[ii] * bv[jj];
      const o = (j * gx + i) * 2;
      dx += wt * cps[o] * hx;
      dy += wt * cps[o + 1] * hy;
    }
  }
  return [dx, dy];
}

const SHIFT = new Float64Array([1, 0, 1, 0, 1, 1, 0, 0, 1]);
const SHIFT_INV = new Float64Array([1, 0, -1, 0, 1, -1, 0, 0, 1]);

/** 1-based moving→fixed H to 0-based moving→fixed. */
export function H0FromH(H) {
  return mul3(SHIFT_INV, mul3(H, SHIFT));
}

function apply3(M, x, y) {
  const X = M[0] * x + M[1] * y + M[2];
  const Y = M[3] * x + M[4] * y + M[5];
  const Z = M[6] * x + M[7] * y + M[8];
  if (Math.abs(Z) < 1e-12) return [NaN, NaN];
  return [X / Z, Y / Z];
}

function isAffineH(H, eps = 1e-8) {
  return Math.abs(H[6]) < eps && Math.abs(H[7]) < eps && Math.abs(H[8] - 1) < 1e-5;
}

/**
 * Map a 0-based fixed pixel to a 0-based moving pixel.
 * Same composition as warp_ffd: target-frame is dest + φ then H⁻¹;
 * source-frame is H⁻¹ then φ in the moving frame.
 */
export function destToMoving(x, y, { H, cps, gx, gy, frame, lw, lh, rw, rh, ffd }) {
  const Hi = inv3(H);
  if (ffd && frame === "source") {
    const q = apply3(Hi, x + 1, y + 1);
    if (!Number.isFinite(q[0])) return q;
    const sx = q[0] - 1, sy = q[1] - 1;
    const d = ffdDisp(sx, sy, cps, gx, gy, lw, lh);
    return [sx + d[0], sy + d[1]];
  }
  if (ffd) {
    const d = ffdDisp(x, y, cps, gx, gy, rw, rh);
    const q = apply3(Hi, x + 1 + d[0], y + 1 + d[1]);
    if (!Number.isFinite(q[0])) return q;
    return [q[0] - 1, q[1] - 1];
  }
  const q = apply3(Hi, x + 1, y + 1);
  if (!Number.isFinite(q[0])) return q;
  return [q[0] - 1, q[1] - 1];
}

export function displacementField(opts) {
  const w = opts.rw, h = opts.rh;
  const d = new Float32Array(w * h * 2);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = destToMoving(x, y, opts);
      const o = (y * w + x) * 2;
      d[o] = Number.isFinite(p[0]) ? p[0] - x : 0;
      d[o + 1] = Number.isFinite(p[1]) ? p[1] - y : 0;
    }
  }
  return d;
}

function littleF32(arr) {
  const out = new Uint8Array(arr.length * 4);
  const view = new DataView(out.buffer);
  for (let i = 0; i < arr.length; i++) view.setFloat32(i * 4, arr[i], true);
  return out;
}

function nrrdHeader(w, h, extra = []) {
  const lines = [
    "NRRD0005",
    "# Complete NRRD file format specification:",
    "# http://teem.sourceforge.net/nrrd/format.html",
    "type: float",
    "dimension: 3",
    "space dimension: 2",
    `sizes: 2 ${w} ${h}`,
    "space directions: none (1,0) (0,1)",
    "kinds: vector domain domain",
    "endian: little",
    "encoding: raw",
    "space origin: (0,0)",
    ...extra,
    "",
  ];
  return lines.join("\n");
}

/** Vector NRRD: component-fastest, then x, then y. ITK VectorImage layout. */
export function nrrdDisplacement(field, w, h, extra = []) {
  const head = new TextEncoder().encode(nrrdHeader(w, h, extra));
  const body = littleF32(field);
  const out = new Uint8Array(head.length + body.length);
  out.set(head, 0);
  out.set(body, head.length);
  return out;
}

function fmtNum(v) {
  if (!Number.isFinite(v)) return "0";
  const s = v.toPrecision(17);
  return s;
}

function itkAffineTfm(H0inv) {
  const p = [
    H0inv[0], H0inv[1],
    H0inv[3], H0inv[4],
    H0inv[2], H0inv[5],
  ];
  return [
    "Transform: AffineTransform_double_2_2",
    `Parameters: ${p.map(fmtNum).join(" ")}`,
    "FixedParameters: 0 0",
  ].join("\n");
}

/**
 * Pad our clamped gx×gy FFD onto ITK's coefficient grid.
 * Cubic order 3: gridSize = gx+2, origin = −spacing, first interior CP at index 1.
 */
export function itkBSplinePad(cps, gx, gy, w, h) {
  const spx = Math.max(w - 1, 1) / Math.max(gx - 1, 1);
  const spy = Math.max(h - 1, 1) / Math.max(gy - 1, 1);
  const hx = w * 0.5, hy = h * 0.5;
  const nx = gx + 2, ny = gy + 2;
  const n = nx * ny;
  const px = new Float64Array(n), py = new Float64Array(n);
  for (let j = 0; j < ny; j++) {
    const sj = clampi(j - 1, gy);
    for (let i = 0; i < nx; i++) {
      const si = clampi(i - 1, gx);
      const o = (sj * gx + si) * 2;
      const k = j * nx + i;
      px[k] = cps[o] * hx;
      py[k] = cps[o + 1] * hy;
    }
  }
  const params = new Float64Array(n * 2);
  params.set(px, 0);
  params.set(py, n);
  const fixed = [nx, ny, -spx, -spy, spx, spy, 1, 0, 0, 1];
  return { params, fixed, nx, ny, spx, spy };
}

function itkBSplineTfm(cps, gx, gy, w, h) {
  const { params, fixed } = itkBSplinePad(cps, gx, gy, w, h);
  return [
    "Transform: BSplineTransform_double_2_2",
    `Parameters: ${Array.from(params, fmtNum).join(" ")}`,
    `FixedParameters: ${fixed.map(fmtNum).join(" ")}`,
  ].join("\n");
}

export function insightTransformFile(opts) {
  const lines = ["#Insight Transform File V1.0"];
  const H0 = H0FromH(opts.H);
  const H0inv = inv3(H0);
  const affine = isAffineH(H0) && isAffineH(opts.H);
  const ffd = !!opts.ffd && opts.cps && opts.gx >= 2;
  const lw = opts.lw, lh = opts.lh, rw = opts.rw, rh = opts.rh;
  const latW = opts.frame === "source" ? lw : rw;
  const latH = opts.frame === "source" ? lh : rh;

  if (ffd && affine) {
    lines.push("#Transform 0", "Transform: CompositeTransform_double_2_2");
    const a = itkAffineTfm(H0inv);
    const b = itkBSplineTfm(opts.cps, opts.gx, opts.gy, latW, latH);
    if (opts.frame === "source") {
      lines.push("#Transform 1", b, "#Transform 2", a);
    } else {
      lines.push("#Transform 1", a, "#Transform 2", b);
    }
  } else if (ffd) {
    lines.push("#Transform 0", itkBSplineTfm(opts.cps, opts.gx, opts.gy, latW, latH));
  } else if (affine) {
    lines.push("#Transform 0", itkAffineTfm(H0inv));
  } else {
    lines.push("# no AffineTransform_double_2_2: H has perspective. Use the NRRD displacement field.");
    lines.push("#Transform 0", "Transform: IdentityTransform_double_2_2", "Parameters:", "FixedParameters:");
  }
  return lines.join("\n") + "\n";
}

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(u8) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < u8.length; i++) c = CRC_TABLE[(c ^ u8[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

function u16(n) { const b = new Uint8Array(2); new DataView(b.buffer).setUint16(0, n, true); return b; }
function u32(n) { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, n >>> 0, true); return b; }

function cat(parts) {
  let n = 0;
  for (const p of parts) n += p.length;
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

/** Uncompressed ZIP (APPNOTE.TXT STORE method). */
export function zipStore(files) {
  const enc = new TextEncoder();
  const locals = [];
  const centrals = [];
  let offset = 0;
  for (const f of files) {
    const name = enc.encode(f.name);
    const data = f.data instanceof Uint8Array ? f.data : enc.encode(String(f.data));
    const crc = crc32(data);
    const local = cat([
      u32(0x04034b50), u16(20), u16(0), u16(0), u16(0), u16(0),
      u32(crc), u32(data.length), u32(data.length), u16(name.length), u16(0),
      name, data,
    ]);
    const central = cat([
      u32(0x02014b50), u16(20), u16(20), u16(0), u16(0), u16(0), u16(0),
      u32(crc), u32(data.length), u32(data.length), u16(name.length), u16(0), u16(0),
      u16(0), u16(0), u32(0), u32(offset), name,
    ]);
    locals.push(local);
    centrals.push(central);
    offset += local.length;
  }
  const cd = cat(centrals);
  const eocd = cat([
    u32(0x06054b50), u16(0), u16(0), u16(files.length), u16(files.length),
    u32(cd.length), u32(offset), u16(0),
  ]);
  return cat([...locals, cd, eocd]);
}

export function warpExportBundle(opts) {
  const w = opts.rw, h = opts.rh;
  const field = displacementField(opts);
  const Hrow = Array.from(opts.H, fmtNum).join(" ");
  const extra = [
    `# CMIR composed warp: T(x_fixed) = x_moving. 0-based pixels, spacing 1.`,
    `# 1-based moving→fixed homography (row-major 3×3): ${Hrow}`,
    opts.ffd ? `# B-spline cubic FFD  ${opts.gx}×${opts.gy}  frame=${opts.frame}` : "# B-spline off",
  ];
  const nrrd = nrrdDisplacement(field, w, h, extra);
  const tfm = insightTransformFile(opts);
  return { field, nrrd, tfm };
}

export function isAffineHomography(H) {
  const H0 = H0FromH(H);
  return isAffineH(H) && isAffineH(H0);
}
