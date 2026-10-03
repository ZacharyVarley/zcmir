/**
 * Map display helpers (CPU): colormaps, the zero-lag-centred colorized map, and the roto-scale
 * map's scale band and lag conversion. The maps themselves come from the zcmir engine.
 */

export const FFD_MAX = 16;

export const THEMES = [
  ["ice", 0], ["viridis", 1], ["magma", 2], ["turbo", 3], ["gray", 4], ["twilight", 5],
];

export function colormap(t, theme) {
  const x = Math.min(1, Math.max(0, t));
  const c = (v) => Math.min(1, Math.max(0, v));
  if (theme === 1) return [c(0.267 + 0.8 * x - 0.4 * x * x), c(0.005 + 1.15 * x - 0.35 * x * x), c(0.329 + 0.2 * x + 0.3 * (1 - x) * x)];
  if (theme === 2) return [c(1.55 * x - 0.2 * x * x), c(1.3 * x * x * x), c(0.12 + 1.35 * x - 1.4 * x * x)];
  if (theme === 3) return [c(0.14 + 2.15 * x - 2.2 * x * x), c(4 * x * (1 - x)), c(0.92 - 1.75 * x + 1.05 * x * x)];
  if (theme === 4) return [x, x, x];
  if (theme === 5) {
    const s = x * 2 - 1, pos = Math.max(s, 0), neg = Math.max(-s, 0);
    return [0.18 + 0.82 * pos, 0.16 + 0.4 * pos + 0.25 * neg, 0.22 + 0.75 * neg];
  }
  return [0.12 + 0.88 * x * x, 0.08 + 0.55 * x, 0.22 + 0.35 * (1 - x)];
}

const _luts = [];

function themeLut(theme) {
  let lut = _luts[theme];
  if (lut) return lut;
  lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = colormap(i / 255, theme);
    lut[i * 3] = (r * 255) | 0;
    lut[i * 3 + 1] = (g * 255) | 0;
    lut[i * 3 + 2] = (b * 255) | 0;
  }
  _luts[theme] = lut;
  return lut;
}

/** A map n wide × ny high (lag 0 at index 0) as an RGBA image with lag 0 centred. */
export function colorizeShifted(score, n, { theme = 0, signed = false, crop = true, log = false, cropRect = null, ny = n } = {}) {
  const N = n * ny, h = n >> 1, hy = ny >> 1;
  let peakI = 0, mx = 1e-6, mn = Infinity, x0 = n, y0 = ny, x1 = -1, y1 = -1;
  const mag = (v) => (signed ? Math.abs(v) : (v > 0 ? v : 0));
  for (let i = 0; i < N; i++) {
    const v = score[i];
    if (v > score[peakI]) peakI = i;
    const a = mag(v);
    if (a > mx) mx = a;
    if (a > 0 && a < mn) mn = a;
    if (v > 0) {
      const y = (i / n) | 0, x = i - y * n;
      const sx = (x + h) % n, sy = (y + hy) % ny;
      if (sx < x0) x0 = sx;
      if (sy < y0) y0 = sy;
      if (sx > x1) x1 = sx;
      if (sy > y1) y1 = sy;
    }
  }
  if (cropRect) {
    x0 = Math.max(0, cropRect.x0 | 0);
    y0 = Math.max(0, cropRect.y0 | 0);
    x1 = Math.min(n - 1, x0 + Math.max(1, cropRect.w | 0) - 1);
    y1 = Math.min(ny - 1, y0 + Math.max(1, cropRect.h | 0) - 1);
  } else if (!crop) {
    x0 = 0; y0 = 0; x1 = n - 1; y1 = ny - 1;
  } else {
    if (x1 < x0) { x0 = 0; y0 = 0; x1 = n - 1; y1 = ny - 1; }
    const pad = Math.max(4, (Math.max(n, ny) * 0.03) | 0);
    x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad);
    x1 = Math.min(n - 1, x1 + pad); y1 = Math.min(ny - 1, y1 + pad);
  }
  const vw = x1 - x0 + 1, vh = y1 - y0 + 1;
  const ox = x0, oy = y0;
  const rgb = new Uint8ClampedArray(vw * vh * 4);
  const lut = themeLut(theme);
  let mapT;
  if (log) {
    const floor = Math.max(mn < Infinity && mn > 0 ? mn : mx * 1e-4, mx * 1e-4, 1e-12);
    const hi = Math.log(Math.max(mx, floor * 1.0001));
    const lo = Math.log(floor);
    const span = Math.max(hi - lo, 1e-8);
    mapT = (v) => {
      const a = mag(v);
      if (!(a > 0)) return 0;
      return (Math.log(Math.max(a, floor)) - lo) / span;
    };
  } else {
    const inv = 1 / mx;
    mapT = (v) => mag(v) * inv;
  }
  for (let y = 0; y < vh; y++) {
    const sy = (oy + y + ny - hy) % ny;
    const srow = sy * n;
    let d = y * vw * 4;
    for (let x = 0; x < vw; x++, d += 4) {
      const v = score[srow + ((ox + x + n - h) % n)];
      const t = signed ? 128 + 127 * Math.sign(v) * mapT(v) : mapT(v) * 255;
      const i = Math.min(255, Math.max(0, Math.round(t))) * 3;
      rgb[d] = lut[i];
      rgb[d + 1] = lut[i + 1];
      rgb[d + 2] = lut[i + 2];
      rgb[d + 3] = 255;
    }
  }
  return { rgb, vw, vh, crop: { x0: ox, y0: oy, w: vw, h: vh }, peakI, mx };
}

/** FFT lag of moving vs fixed → Sim_c increment that undoes it (same sign rule as shift hop). */

export function rsCorrToSim(dx, dy, nTh, dlam) {
  return { dth: -(dx / Math.max(nTh, 1)) * Math.PI * 2, dsg: -dy * dlam };
}

/** Zero λ-rows whose applied scale exp(−k Δλ) lies outside [sMin, sMax]. */

export function maskRsScale(score, n, dlam, sMin, sMax) {
  const out = score.slice();
  const lo = Math.log(Math.max(+sMin || 0.2, 1e-6));
  const hi = Math.log(Math.max(+sMax || 5, 1.000001));
  for (let py = 0; py < n; py++) {
    const sy = py <= n / 2 ? py : py - n;
    const sg = -sy * dlam;
    if (sg < lo || sg > hi) {
      const row = py * n;
      out.fill(0, row, row + n);
    }
  }
  return out;
}

/** FFT-shifted crop that drops blank scale bands when [sMin, sMax] does not fill the map. */

export function rsScaleCrop(n, dlam, sMin, sMax) {
  const lo = Math.log(Math.max(+sMin || 0.2, 1e-6));
  const hi = Math.log(Math.max(+sMax || 5, 1.000001));
  const h = n >> 1;
  let y0 = n, y1 = -1;
  for (let py = 0; py < n; py++) {
    const sy = py <= n / 2 ? py : py - n;
    const sg = -sy * dlam;
    if (sg < lo || sg > hi) continue;
    const iy = (py + h) % n;
    if (iy < y0) y0 = iy;
    if (iy > y1) y1 = iy;
  }
  if (y1 < y0 || y1 - y0 + 1 >= n - 4) return null;
  const pad = Math.max(2, (n * 0.02) | 0);
  y0 = Math.max(0, y0 - pad);
  y1 = Math.min(n - 1, y1 + pad);
  return { x0: 0, y0, w: n, h: y1 - y0 + 1 };
}
