/**
 * B-spline displacement display: geometry for the overlay, in 0-based fixed
 * (dest) pixels. The field is the real cubic B-spline displacement (ffdDisp,
 * the same formula the shaders use), not the control-point coefficients, which
 * a cubic B-spline does not interpolate.
 *
 * target frame: dest x samples the moving image at H⁻¹(x + d(x)); drawn as x + d(x).
 * source frame: moving s is displaced to s + d(s); drawn at H(s + d(s)).
 */
import { ffdDisp } from "./export_warp.js";
import { mapPts } from "./lie.js";

const LINES = 20;        // grid lines across the lattice frame
const SUB = 6;           // polyline samples per grid cell
const ARROW_EVERY = 2;   // update arrows on every other grid node

/**
 * @param o { cps, prev, gx, gy, frame, H, lw, lh, rw, rh, exag }
 *   cps / prev: current and previous coefficients (prev may be null);
 *   exag: "auto" or a number.
 * @returns { lines, cells, arrows, maxD, minDet, k, kUpd, maxUpd, spacing }
 */
export function ffdOverlay(o) {
  const src = o.frame === "source";
  const fw = src ? o.lw : o.rw, fh = src ? o.lh : o.rh;
  const disp = (c, x, y) => ffdDisp(x, y, c, o.gx, o.gy, fw, fh);
  const toDest = src
    ? (x, y) => { const [X, Y] = mapPts(o.H, [[x + 1, y + 1]])[0]; return [X - 1, Y - 1]; }
    : (x, y) => [x, y];
  const spacing = Math.max(fw, fh) / LINES;
  const nx = Math.max(2, Math.round((fw - 1) / spacing)), ny = Math.max(2, Math.round((fh - 1) / spacing));
  const gxs = Array.from({ length: nx + 1 }, (_, i) => (i * (fw - 1)) / nx);
  const gys = Array.from({ length: ny + 1 }, (_, j) => (j * (fh - 1)) / ny);

  // True field on the grid nodes: magnitude and Jacobian determinant of x ↦ x + d(x).
  const node = gys.map((y) => gxs.map((x) => disp(o.cps, x, y)));
  let maxD = 0;
  for (const row of node) for (const [dx, dy] of row) maxD = Math.max(maxD, Math.hypot(dx, dy));
  // Differences stay inside the frame: the field is cut to 0 outside it, and a
  // stencil straddling that edge would report a fake fold.
  const h = 0.5;
  const detAt = (x, y) => {
    const xa = Math.min(fw - 1, x + h), xb = Math.max(0, x - h);
    const ya = Math.min(fh - 1, y + h), yb = Math.max(0, y - h);
    const [ax, ay] = disp(o.cps, xa, y), [bx, by] = disp(o.cps, xb, y);
    const [cx, cy] = disp(o.cps, x, ya), [ex, ey] = disp(o.cps, x, yb);
    const j11 = 1 + (ax - bx) / (xa - xb), j21 = (ay - by) / (xa - xb);
    const j12 = (cx - ex) / (ya - yb), j22 = 1 + (cy - ey) / (ya - yb);
    return j11 * j22 - j12 * j21;
  };

  // Exaggeration: the largest drawn displacement is a quarter of a grid cell.
  let k = typeof o.exag === "number" ? o.exag : 1;
  if (o.exag === "auto") k = maxD > 1e-9 ? Math.min(200, Math.max(1, (0.25 * spacing) / maxD)) : 1;
  const place = (x, y) => {
    const [dx, dy] = disp(o.cps, x, y);
    return toDest(x + k * dx, y + k * dy);
  };

  const lines = [];
  for (const y of gys) {
    const pts = [];
    for (let s = 0; s <= nx * SUB; s++) pts.push(place((s * (fw - 1)) / (nx * SUB), y));
    lines.push(pts);
  }
  for (const x of gxs) {
    const pts = [];
    for (let s = 0; s <= ny * SUB; s++) pts.push(place(x, (s * (fh - 1)) / (ny * SUB)));
    lines.push(pts);
  }

  const cells = [];
  let minDet = Infinity;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const x0 = gxs[i], x1 = gxs[i + 1], y0 = gys[j], y1 = gys[j + 1];
      const det = detAt(0.5 * (x0 + x1), 0.5 * (y0 + y1));
      minDet = Math.min(minDet, det);
      cells.push({ det, poly: [place(x0, y0), place(x1, y0), place(x1, y1), place(x0, y1)] });
    }
  }
  for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) minDet = Math.min(minDet, detAt(gxs[i], gys[j]));

  // Change of the field since the previous coefficients, at every other node.
  const arrows = [];
  let maxUpd = 0;
  if (o.prev && o.prev.length === o.cps.length) {
    for (let j = 0; j <= ny; j += ARROW_EVERY) {
      for (let i = 0; i <= nx; i += ARROW_EVERY) {
        const x = gxs[i], y = gys[j];
        const [ax, ay] = node[j][i];
        const [bx, by] = disp(o.prev, x, y);
        const ux = ax - bx, uy = ay - by;
        maxUpd = Math.max(maxUpd, Math.hypot(ux, uy));
        arrows.push({ x, y, ux, uy });
      }
    }
  }
  const kUpd = maxUpd > 1e-9 ? Math.min(2000, Math.max(1, (0.8 * spacing) / maxUpd)) : 1;
  for (const a of arrows) {
    const [px, py] = place(a.x, a.y);
    const [qx, qy] = src
      ? (() => { const [dx, dy] = disp(o.cps, a.x, a.y); return toDest(a.x + k * dx + kUpd * a.ux, a.y + k * dy + kUpd * a.uy); })()
      : [px + kUpd * a.ux, py + kUpd * a.uy];
    Object.assign(a, { from: [px, py], to: [qx, qy] });
  }
  let maxLog = 0, maxDet = 0;
  for (const c of cells) if (c.det > 0) { maxLog = Math.max(maxLog, Math.abs(Math.log(c.det))); maxDet = Math.max(maxDet, c.det); }
  return { lines, cells, arrows, maxD, minDet, maxDet, maxLog, k, kUpd, maxUpd, spacing };
}

/**
 * Diverging fill for det J: blue compresses, red expands, magenta folds. Scaled to
 * the largest |log det J| on screen (scale), so a gentle field still shows its
 * pattern; the legend carries the true range.
 */
export function detColor(det, scale = Math.log(2)) {
  if (!(det > 0)) return "rgba(255,0,255,0.6)";
  const l = Math.max(-1, Math.min(1, Math.log(det) / Math.max(scale, 1e-6)));
  const a = 0.5 * Math.abs(l);
  return l < 0 ? `rgba(70,130,255,${a.toFixed(3)})` : `rgba(255,90,60,${a.toFixed(3)})`;
}
