export const AFFINE_KEYS = ["tx", "ty", "th", "sg", "al", "ga"];
export const HOMO_KEYS = [...AFFINE_KEYS, "px", "py"];
export const TANGENT_KEYS = HOMO_KEYS;
export const TANGENT_SCALE = Object.fromEntries(TANGENT_KEYS.map((k) => [k, 1]));

const eye = () => new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

export const identityH = () => eye();

export function destN(w, h) {
  w = Math.max(+w, 1); h = Math.max(+h, 1);
  const sx = 0.5 * w, sy = 0.5 * h;
  const cx = 0.5 * (w + 1), cy = 0.5 * (h + 1);
  return new Float64Array([1 / sx, 0, -cx / sx, 0, 1 / sy, -cy / sy, 0, 0, 1]);
}

export const destHalf = (w, h) => [0.5 * Math.max(+w, 1), 0.5 * Math.max(+h, 1)];

export function inv3(m) {
  const a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
  const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g;
  const D = c * h - b * i, E = a * i - c * g, F = b * g - a * h;
  const G = b * f - c * e, H = c * d - a * f, I = a * e - b * d;
  const det = a * A + b * B + c * C;
  const s = 1 / (Math.sign(det) * Math.max(Math.abs(det), 1e-18));
  const inv = new Float64Array([A * s, D * s, G * s, B * s, E * s, H * s, C * s, F * s, I * s]);
  const prod = mul3(m, inv);
  const corr = new Float64Array([
    2 - prod[0], -prod[1], -prod[2],
    -prod[3], 2 - prod[4], -prod[5],
    -prod[6], -prod[7], 2 - prod[8],
  ]);
  return mul3(inv, corr);
}

export function mul3(A, B) {
  const o = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
    o[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c];
  return o;
}

export function hat(coords, group = "affine") {
  const tx = +coords.tx || 0, ty = +coords.ty || 0, th = +coords.th || 0;
  const sg = +coords.sg || 0, al = +coords.al || 0, ga = +coords.ga || 0;
  let px = +coords.px || 0, py = +coords.py || 0, iso, z;
  if (group !== "homography") { px = py = 0; z = 0; iso = sg; }
  else { iso = sg / 3; z = -2 * iso; }
  return new Float64Array([iso + al, ga - th, tx, ga + th, iso - al, ty, px, py, z]);
}

export function expm3(A) {
  let nrm = 0;
  for (let i = 0; i < 9; i++) nrm = Math.max(nrm, Math.abs(A[i]));
  let s = 0;
  if (nrm > 0.5) s = Math.min(10, Math.max(0, Math.ceil(Math.log2(nrm / 0.5))));
  const sc = 0.5 ** s;
  const B = new Float64Array(9);
  for (let i = 0; i < 9; i++) B[i] = A[i] * sc;
  let T = eye(), P = eye();
  for (let k = 1; k < 14; k++) {
    P = mul3(P, B);
    for (let i = 0; i < 9; i++) P[i] /= k;
    for (let i = 0; i < 9; i++) T[i] += P[i];
  }
  for (let i = 0; i < s; i++) T = mul3(T, T);
  return T;
}

export const tangentKeys = (group) => (group === "homography" ? HOMO_KEYS : AFFINE_KEYS);

function projectGroup(H, group) {
  const s = Math.abs(H[8]) > 1e-12 ? H[8] : 1;
  const o = new Float64Array(9);
  for (let i = 0; i < 9; i++) o[i] = H[i] / s;
  if (group !== "homography") { o[6] = 0; o[7] = 0; o[8] = 1; }
  return o;
}

export function compose(H0, coords, group = "affine") {
  return projectGroup(mul3(expm3(hat(coords, group)), H0), group);
}

export function composeN(H0, coords, group, dw, dh) {
  const N = destN(dw, dh);
  return projectGroup(mul3(mul3(mul3(inv3(N), expm3(hat(coords, group))), N), H0), group);
}

function vee(A, group) {
  let a = A[0], b = A[1], c = A[2], d = A[3], e = A[4], f = A[5], g = A[6], h = A[7], i = A[8];
  if (group === "homography") {
    const tr = (a + e + i) / 3;
    a -= tr; e -= tr; i -= tr;
    const iso = (a + e) / 2;
    return { tx: c, ty: f, th: (d - b) / 2, sg: iso * 3, al: (a - e) / 2, ga: (d + b) / 2, px: g, py: h };
  }
  return { tx: c, ty: f, th: (d - b) / 2, sg: (a + e) / 2, al: (a - e) / 2, ga: (d + b) / 2, px: 0, py: 0 };
}

// dS/dξ on H, given dS/dη on H⁻¹. η is a left increment in the moving frame;
// ξ is a left increment in the fixed frame. η̂ = Ad(−ξ̂).
export function adjointGrad(gEta, H, group, movW, movH, fixW, fixH) {
  const Hi = inv3(H);
  const Nm = destN(movW, movH);
  const Nf = destN(fixW, fixH);
  const M = mul3(mul3(Nm, Hi), inv3(Nf));
  const Mi = inv3(M);
  const keys = tangentKeys(group);
  const gens = generators(group);
  const out = new Float64Array(keys.length);
  for (let i = 0; i < keys.length; i++) {
    const coords = vee(mul3(mul3(M, gens[i]), Mi), group);
    for (let j = 0; j < keys.length; j++) out[i] -= gEta[j] * (coords[keys[j]] || 0);
  }
  return out;
}

export function unitAscent(g, group = "affine") {
  const keys = tangentKeys(group);
  const u = new Float64Array(keys.length);
  let n = 0;
  for (let i = 0; i < keys.length; i++) {
    u[i] = (g[i] || 0) * TANGENT_SCALE[keys[i]];
    n += u[i] * u[i];
  }
  n = Math.sqrt(n);
  if (n < 1e-18) return u;
  for (let i = 0; i < u.length; i++) u[i] = (u[i] / n) * TANGENT_SCALE[keys[i]];
  return u;
}

/** Jacobi eigendecomposition of a symmetric n×n row-major matrix. vec[:,j] is eigenvector j. */
export function symEig(A, n) {
  const M = Float64Array.from(A);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let it = 0; it < 64; it++) {
    let p = 0, q = 1, best = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const a = Math.abs(M[i * n + j]);
      if (a > best) { best = a; p = i; q = j; }
    }
    if (best < 1e-14) break;
    const app = M[p * n + p], aqq = M[q * n + q], apq = M[p * n + q];
    const tau = (aqq - app) / (2 * apq);
    const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t), s = t * c;
    for (let k = 0; k < n; k++) {
      if (k === p || k === q) continue;
      const mkp = M[k * n + p], mkq = M[k * n + q];
      M[k * n + p] = M[p * n + k] = c * mkp - s * mkq;
      M[k * n + q] = M[q * n + k] = s * mkp + c * mkq;
    }
    M[p * n + p] = app - t * apq;
    M[q * n + q] = aqq + t * apq;
    M[p * n + q] = M[q * n + p] = 0;
    for (let k = 0; k < n; k++) {
      const vkp = V[k * n + p], vkq = V[k * n + q];
      V[k * n + p] = c * vkp - s * vkq;
      V[k * n + q] = s * vkp + c * vkq;
    }
  }
  const val = new Float64Array(n);
  for (let i = 0; i < n; i++) val[i] = M[i * n + i];
  return { val, vec: V };
}

export function destTranslation(H) {
  const z = Math.abs(H[8]) > 1e-12 ? H[8] : 1;
  return [H[2] / z, H[5] / z];
}

/** Linear polar-ish read of a 3×3 H into destN generators (tx/ty left at 0). */
export function approxCoords(H) {
  const z = Math.abs(H[8]) > 1e-12 ? H[8] : 1;
  const a = H[0] / z, b = H[1] / z, c = H[3] / z, d = H[4] / z;
  const det = a * d - b * c;
  return {
    tx: 0, ty: 0,
    th: Math.atan2(c - b, a + d),
    sg: 0.5 * Math.log(Math.max(Math.abs(det), 1e-12)),
    al: 0, ga: 0,
    px: H[6] / z, py: H[7] / z,
  };
}

export function coordsFromVec(g, group = "affine") {
  const keys = tangentKeys(group);
  const o = {};
  keys.forEach((k, i) => { o[k] = +g[i] || 0; });
  return o;
}

export function aff6ToH(h6) {
  const a = [...h6];
  return new Float64Array([a[0], a[1], a[2], a[3], a[4], a[5], 0, 0, 1]);
}

export function HToAff6(H) {
  const s = Math.abs(H[8]) > 1e-12 ? H[8] : 1;
  return new Float64Array([H[0] / s, H[1] / s, H[2] / s, H[3] / s, H[4] / s, H[5] / s]);
}

export function fmtH(H, digits = 4) {
  const n = Math.max(2, Math.min(12, digits | 0));
  const r = (i) => H[i].toFixed(n);
  return `${r(0)} ${r(1)} ${r(2)}\n${r(3)} ${r(4)} ${r(5)}\n${r(6)} ${r(7)} ${r(8)}`;
}

export function poseLine(H) {
  const s = Math.abs(H[8]) > 1e-12 ? H[8] : 1;
  const a = H[0] / s, b = H[1] / s, tx = H[2] / s, c = H[3] / s, d = H[4] / s, ty = H[5] / s;
  const det = a * d - b * c;
  const ang = Math.atan2(c - b, a + d) * (180 / Math.PI);
  const sc = Math.sqrt(Math.max(Math.abs(det), 0));
  const rot = `${ang >= 0 ? "+" : ""}${ang.toFixed(1)}°`;
  const base = `${sc.toFixed(3)}×  ${rot}  ${tx.toFixed(1)}, ${ty.toFixed(1)}`;
  const px = H[6] / s, py = H[7] / s;
  return Math.hypot(px, py) > 1e-7 ? `${base}  p ${px.toExponential(1)}, ${py.toExponential(1)}` : base;
}

export function summary(H) {
  const s = Math.abs(H[8]) > 1e-12 ? H[8] : 1;
  const a = H[0] / s, b = H[1] / s, tx = H[2] / s, c = H[3] / s, d = H[4] / s, ty = H[5] / s;
  const det = a * d - b * c;
  const ang = Math.atan2(c - b, a + d) * (180 / Math.PI);
  const sc = Math.sqrt(Math.max(Math.abs(det), 0));
  const persp = Math.abs(H[6] / s) + Math.abs(H[7] / s);
  const base = `s=${sc.toFixed(3)}  rot=${ang >= 0 ? "+" : ""}${ang.toFixed(1)}deg  det=${det.toFixed(3)}  t=(${tx.toFixed(1)},${ty.toFixed(1)})`;
  return persp > 1e-7 ? `${base}  p=(${(H[6] / s).toExponential(2)},${(H[7] / s).toExponential(2)})` : base;
}

export function generators(group = "affine", destWh = null) {
  const keys = tangentKeys(group);
  const out = [];
  for (const k of keys) {
    const c = Object.fromEntries(TANGENT_KEYS.map((kk) => [kk, 0]));
    c[k] = 1;
    out.push(hat(c, group));
  }
  if (!destWh) return out;
  const N = destN(destWh[0], destWh[1]);
  const Ni = inv3(N);
  return out.map((G) => mul3(mul3(Ni, G), N));
}

export function regFrame(H, lw, lh, rw, rh) {
  const corners = [
    [1, 1, 1], [lw, 1, 1], [lw, lh, 1], [1, lh, 1],
  ];
  const xs = [0, rw], ys = [0, rh];
  for (const p of corners) {
    const x = H[0] * p[0] + H[1] * p[1] + H[2] * p[2];
    const y = H[3] * p[0] + H[4] * p[1] + H[5] * p[2];
    const z = H[6] * p[0] + H[7] * p[1] + H[8] * p[2];
    const zz = Math.abs(z) < 1e-8 ? 1e-8 : z;
    xs.push(x / zz - 1); ys.push(y / zz - 1);
  }
  const lim = Math.max(lw, rw, lh, rh, 8) * 2;
  const clip = (v) => Math.min(lim, Math.max(-lim, v));
  let ox = Math.floor(clip(Math.min(...xs)));
  let oy = Math.floor(clip(Math.min(...ys)));
  let x1 = Math.ceil(clip(Math.max(...xs)));
  let y1 = Math.ceil(clip(Math.max(...ys)));
  ox = Math.min(ox, 0); oy = Math.min(oy, 0);
  x1 = Math.max(x1, rw); y1 = Math.max(y1, rh);
  return [ox, oy, Math.max(8, x1 - ox), Math.max(8, y1 - oy)];
}

export function canvasH(H, ox, oy) {
  const T = new Float64Array([1, 0, -ox, 0, 1, -oy, 0, 0, 1]);
  return mul3(T, H);
}

export function movingSrcCorners(lw, lh) {
  return [[1, 1], [lw, 1], [lw, lh], [1, lh]];
}

export function movingCentroid(H, lw, lh) {
  const p = mapPts(H, movingSrcCorners(lw, lh));
  return [(p[0][0] + p[1][0] + p[2][0] + p[3][0]) * 0.25, (p[0][1] + p[1][1] + p[2][1] + p[3][1]) * 0.25];
}

/** Dest-pixel Euclidean similarity about c: X ↦ c + e^σ R_θ (X − c). */
export function simAbout(c, th, sg) {
  const s = Math.exp(+sg || 0);
  const co = Math.cos(+th || 0), si = Math.sin(+th || 0);
  const a = s * co, b = -s * si, d = s * si, e = s * co;
  const cx = +c[0], cy = +c[1];
  return new Float64Array([a, b, cx - a * cx - b * cy, d, e, cy - d * cx - e * cy, 0, 0, 1]);
}

export function composeSimC(H0, c, th, sg, group = "affine") {
  return projectGroup(mul3(simAbout(c, th, sg), H0), group);
}

export function mapPts(H, pts) {
  return pts.map(([x, y]) => {
    const X = H[0] * x + H[1] * y + H[2];
    const Y = H[3] * x + H[4] * y + H[5];
    const Z = H[6] * x + H[7] * y + H[8];
    const z = Math.abs(Z) < 1e-8 ? 1e-8 : Z;
    return [X / z, Y / z];
  });
}

export function parallelogramMove(dst, i, xy) {
  const d = dst.map((p) => [...p]);
  i = ((i % 4) + 4) % 4;
  d[i] = [...xy];
  const a = (i + 1) % 4, b = (i + 3) % 4;
  d[(i + 2) % 4] = [d[a][0] + d[b][0] - d[i][0], d[a][1] + d[b][1] - d[i][1]];
  return d;
}

function solve(A, b, n, m) {
  /* A is n x m, b is n. Normal equations if n!=m. */
  const ata = Array.from({ length: m }, () => new Float64Array(m));
  const atb = new Float64Array(m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      atb[j] += A[i][j] * b[i];
      for (let k = 0; k < m; k++) ata[j][k] += A[i][j] * A[i][k];
    }
  }
  for (let i = 0; i < m; i++) {
    let piv = i;
    for (let r = i + 1; r < m; r++) if (Math.abs(ata[r][i]) > Math.abs(ata[piv][i])) piv = r;
    [ata[i], ata[piv]] = [ata[piv], ata[i]];
    [atb[i], atb[piv]] = [atb[piv], atb[i]];
    const d = ata[i][i] || 1e-12;
    for (let j = i; j < m; j++) ata[i][j] /= d;
    atb[i] /= d;
    for (let r = 0; r < m; r++) if (r !== i) {
      const f = ata[r][i];
      for (let j = i; j < m; j++) ata[r][j] -= f * ata[i][j];
      atb[r] -= f * atb[i];
    }
  }
  return atb;
}

export function HAffineFromPts(src, dst, wts = null) {
  const n = src.length;
  const A = [], b = [];
  for (let i = 0; i < n; i++) {
    const wt = wts ? Math.sqrt(Math.max(+wts[i] || 0, 0)) : 1;
    if (wt <= 0) continue;
    const [x, y] = src[i], [u, v] = dst[i];
    A.push([wt * x, wt * y, wt, 0, 0, 0]);
    A.push([0, 0, 0, wt * x, wt * y, wt]);
    b.push(wt * u, wt * v);
  }
  if (A.length < 6) return identityH();
  return aff6ToH(solve(A, b, A.length, 6));
}

function solveSquare(A0, b0, n) {
  const A = Array.from({ length: n }, (_, i) => Float64Array.from(A0[i]));
  const b = Float64Array.from(b0);
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r;
    [A[i], A[piv]] = [A[piv], A[i]];
    [b[i], b[piv]] = [b[piv], b[i]];
    const d = A[i][i] || 1e-18;
    for (let j = i; j < n; j++) A[i][j] /= d;
    b[i] /= d;
    for (let r = 0; r < n; r++) if (r !== i) {
      const f = A[r][i];
      for (let j = i; j < n; j++) A[r][j] -= f * A[i][j];
      b[r] -= f * b[i];
    }
  }
  return b;
}

/** Hartley normalization of weighted points: [cx, cy, s] with s·(p − c) of mean length √2. */
function hartley(pts, wts) {
  let sw = 0, cx = 0, cy = 0;
  for (let i = 0; i < pts.length; i++) {
    const wt = wts ? +wts[i] || 0 : 1;
    if (wt <= 0) continue;
    sw += wt; cx += wt * pts[i][0]; cy += wt * pts[i][1];
  }
  if (sw <= 0) return [0, 0, 1];
  cx /= sw; cy /= sw;
  let d = 0;
  for (let i = 0; i < pts.length; i++) {
    const wt = wts ? +wts[i] || 0 : 1;
    if (wt > 0) d += wt * Math.hypot(pts[i][0] - cx, pts[i][1] - cy);
  }
  return [cx, cy, Math.SQRT2 / Math.max(d / sw, 1e-12)];
}

/** The DLT on Hartley-normalized coordinates (in pixels its normal matrix is too ill-conditioned
 *  to solve), as lie.zig HHomographyFromPts. */
export function HHomographyFromPts(src, dst, wts = null) {
  const n = src.length;
  const na = hartley(src, wts), nb = hartley(dst, wts);
  const ATA = Array.from({ length: 9 }, () => new Float64Array(9));
  for (let i = 0; i < n; i++) {
    const x = na[2] * (src[i][0] - na[0]), y = na[2] * (src[i][1] - na[1]);
    const u = nb[2] * (dst[i][0] - nb[0]), v = nb[2] * (dst[i][1] - nb[1]);
    const wt = wts ? +wts[i] || 0 : 1;
    if (wt <= 0) continue;
    const rows = [
      [-x, -y, -1, 0, 0, 0, u * x, u * y, u],
      [0, 0, 0, -x, -y, -1, v * x, v * y, v],
    ];
    for (const row of rows) for (let j = 0; j < 9; j++) for (let k = 0; k < 9; k++) ATA[j][k] += wt * row[j] * row[k];
  }
  for (let i = 0; i < 9; i++) ATA[i][i] += 1e-12;
  let v = new Float64Array(9); v[8] = 1;
  for (let it = 0; it < 48; it++) {
    const x = solveSquare(ATA, v, 9);
    let nrm = 0;
    for (let j = 0; j < 9; j++) nrm += x[j] * x[j];
    nrm = Math.sqrt(nrm) || 1;
    for (let j = 0; j < 9; j++) v[j] = x[j] / nrm;
  }
  // back to pixels: Tb⁻¹ · Hn · Ta
  const Ta = [na[2], 0, -na[2] * na[0], 0, na[2], -na[2] * na[1], 0, 0, 1];
  const Tbi = [1 / nb[2], 0, nb[0], 0, 1 / nb[2], nb[1], 0, 0, 1];
  return projectGroup(mul3(Tbi, mul3(Array.from(v), Ta)), "homography");
}

/** Solve the n×n system A x = b (row-major A), partial pivoting. null if singular. */
export function solveLinear(A, b, n) {
  const M = Array.from({ length: n }, (_, i) => Float64Array.from(A.subarray(i * n, i * n + n)));
  const x = Float64Array.from(b);
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[piv][i])) piv = r;
    [M[i], M[piv]] = [M[piv], M[i]];
    [x[i], x[piv]] = [x[piv], x[i]];
    const d = M[i][i];
    if (!Number.isFinite(d) || Math.abs(d) < 1e-300) return null;
    for (let r = 0; r < n; r++) if (r !== i) {
      const f = M[r][i] / d;
      if (!f) continue;
      for (let j = i; j < n; j++) M[r][j] -= f * M[i][j];
      x[r] -= f * x[i];
    }
  }
  for (let i = 0; i < n; i++) x[i] /= M[i][i];
  return x.every(Number.isFinite) ? x : null;
}

export function affineLsm(p, q) {
  return HAffineFromPts(p, q);
}
