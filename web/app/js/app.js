import { Zcmir } from "../zcmir/zcmir.js";
import {
  adjointGrad, compose, composeN, solveLinear, composeSimC, coordsFromVec, destHalf, destTranslation, fmtH, identityH,
  inv3, mapPts, movingCentroid, mul3, movingSrcCorners, parallelogramMove, HAffineFromPts, HHomographyFromPts,
  poseLine, summary, symEig, tangentKeys, unitAscent,
} from "./lie.js";
import { applyNudge, dragNudge, geoAmps, HELP, keyNudge, typingTarget, wheelNudge } from "./controls.js";
import { FFD_MAX, colormap, colorizeShifted, maskRsScale, rsCorrToSim, rsScaleCrop } from "./maps.js";
import { isAffineHomography } from "./export_warp.js";
import { TiffStream, channelsOf, crc32, pack, zipStore } from "./stackio.js";
import { detColor, ffdOverlay } from "./ffdview.js";

// Registration runs in the zcmir engine (Zig + WGSL, zcmir/zcmir.wasm); this page is its UI.
// Detection and matching defaults (src/settings.zig, src/gls.zig).
const N_TRIALS = 100000, MAX_TRIALS = 100000;
const N_OCTAVES = 3, MAX_POINTS = 2000, MIN_CONTRAST = 0.01, RADIUS = 36, TAU = 0.8, INLIER_PX = 5;
// POS-GIFT's exposed settings and their defaults (src/posgift.zig Settings).
const POS_GIFT_DEFAULTS = {
  n_octaves: 3, max_points: 5000, min_contrast: 0.01, p1: 10, desc_pow: 1.5, n_orient: 6, n_rings: 3, n_scales: 4,
  min_wl: 3, mult: 1.6, sigma_onf: 0.75, pc_k: 1, cutoff: 0.5, pc_g: 3, gsig: 3, pos_p1: 20, pos_k: 20,
};
// The score select's labels → the engine's metric names.
const METRIC_KEY = { "SMI": "smi", "SMI (edge)": "smi_edge", "E4 (normal scores)": "e4", "λmax MI (normal scores)": "lmax", "NCC": "ncc" };
/** The engine (created in main). */
let zc = null;
/** Map color theme. */
let theme = 4;

const $ = (id) => document.getElementById(id);
function rsScaleBounds() {
  let lo = +$("rs-smin")?.value || 0.2;
  let hi = +$("rs-smax")?.value || 5;
  lo = Math.min(Math.max(lo, 0.02), 0.99);
  hi = Math.max(hi, 1);
  return [lo, hi];
}
function rsCenterDest(H) {
  if (state.rsCenter) return state.rsCenter;
  if (!state.left) return [1, 1];
  return movingCentroid(H || currentH(), state.left.w, state.left.h);
}
function rsMapOpts(H) {
  const [sMin, sMax] = rsScaleBounds();
  const c = rsCenterDest(H);
  return { cDest: [c[0], c[1]], sMin, sMax };
}
function rsMapOptsInv(H) {
  const [sMin, sMax] = rsScaleBounds();
  const c = mapPts(inv3(H), [rsCenterDest(H)])[0];
  return { cDest: [c[0], c[1]], sMin, sMax };
}
const logEl = $("log");
const log = (m) => {
  logEl.textContent += m + "\n";
  if ($("app").classList.contains("log-open")) logEl.scrollTop = logEl.scrollHeight;
};

const job = { stop: false, kind: null };
/** A stack registration in progress: it runs many jobs, and Stop ends the whole stack. */
const stackRun = { active: false, stop: false };
function canceledError() {
  const e = new Error("canceled");
  e.canceled = true;
  return e;
}
function throwIfCanceled() {
  if (job.stop) throw canceledError();
}
function requestCancel() {
  if (stackRun.active) stackRun.stop = true;
  if (!job.kind) return;
  job.stop = true;
  zc?.cancel();
  const el = $("busy");
  if (el && !el.hidden) el.textContent = `${el.textContent.replace(/ — stopping…$/, "")} — stopping…`;
}
function toggleLog(force) {
  const app = $("app");
  const on = force != null ? force : !app.classList.contains("log-open");
  app.classList.toggle("log-open", on);
  $("btn-log").setAttribute("aria-pressed", String(on));
  $("btn-log").textContent = on ? "Hide log" : "Show log";
  if (on) logEl.scrollTop = logEl.scrollHeight;
  else logEl.textContent = "";
  requestAnimationFrame(() => onStageChange());
}
function toggleRail(force) {
  const app = $("app");
  const on = force != null ? force : !app.classList.contains("rail-hidden");
  app.classList.toggle("rail-hidden", on);
  $("btn-rail").setAttribute("aria-pressed", String(on));
  requestAnimationFrame(() => onStageChange());
}

function bindSplitter(handle, { parent, prop, axis = "x", fromEnd = false, min = 140, minOther = 140, unit = "%", ondrag }) {
  if (!handle || !parent) return;
  handle.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    handle.classList.add("dragging");
    document.body.classList.add(axis === "x" ? "split-drag-v" : "split-drag-h");
    const onMove = (ev) => {
      const rect = parent.getBoundingClientRect();
      const total = axis === "x" ? rect.width : rect.height;
      const pos = axis === "x" ? ev.clientX - rect.left : ev.clientY - rect.top;
      const raw = fromEnd ? total - pos : pos;
      const clamped = Math.max(min, Math.min(total - minOther, raw));
      parent.style.setProperty(prop, unit === "px" ? `${Math.round(clamped)}px` : `${((clamped / total) * 100).toFixed(2)}%`);
      ondrag?.();
    };
    const onUp = () => {
      handle.classList.remove("dragging");
      document.body.classList.remove("split-drag-v", "split-drag-h");
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onUp);
      handle.removeEventListener("pointercancel", onUp);
      ondrag?.();
    };
    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onUp);
    handle.addEventListener("pointercancel", onUp);
  });
}

function setGroup(g) {
  state.group = g;
  $("app").dataset.group = g;
  syncClimbAuto();
  const el = document.querySelector(`input[name="group"][value="${g}"]`);
  if (el) el.checked = true;
  syncModelUi();
}

/** The model row (affine / homography, + B-spline) into the page: what Auto will do, which
 *  spline controls show. */
function syncModelUi() {
  const ffd = !!$("ffd")?.checked;
  const app = $("app");
  if (app) app.dataset.ffd = ffd ? "on" : "off";
  if ($("ffd-g")) $("ffd-g").disabled = !ffd;
  const plan = $("auto-plan");
  if (plan) plan.textContent = `detect → match → refine ${state.group}${ffd ? " + spline" : ""}`;
}

/* ── one workspace: the overlay, and a tabbed side column ── */
let onStageChange = () => {};
let applyStageHalf = () => {};
let syncClimbAuto = () => {};
function dockName() { return $("app")?.dataset.dock || ""; }
function mapDockOn() { return dockName() === "map"; }
function popDockOn() { return dockName() === "pop"; }
const DOCKS = ["kp", "map", "score", "pop", "stack"];
/** Show a side view. While a stack registers, only the user's own clicks switch it (the steps'
 *  own switches would flip the side column back and forth on every slice). */
function setDock(which, user = false) {
  if (stackRun.active && !user) return;
  const app = $("app");
  const next = DOCKS.includes(which) ? which : "map";
  if (app.dataset.dock === next && !$("dock-" + next)?.hidden) return;
  app.dataset.dock = next;
  for (const name of DOCKS) {
    const pane = $("dock-" + name);
    if (pane) pane.hidden = next !== name;
    const btn = document.querySelector(`.dock-tab[data-dock="${name}"]`);
    if (btn) btn.setAttribute("aria-pressed", String(next === name));
  }
  requestAnimationFrame(() => requestAnimationFrame(() => onStageChange()));
}
document.querySelectorAll(".dock-tab").forEach((btn) => {
  btn.onclick = () => setDock(btn.dataset.dock, true);
});
/* The inspector's pipeline cards: a header (status, settings summary, last result, run button)
 * over a body of settings. One card is open at a time; the choice is remembered per browser. */
function openCard(name) {
  document.querySelectorAll(".card[data-card]").forEach((card) => {
    const on = card.dataset.card === name;
    card.classList.toggle("open", on);
    card.querySelector(".card-body").hidden = !on;
    card.querySelector(".card-toggle").setAttribute("aria-expanded", String(on));
  });
  try { localStorage.setItem("zcmir.card", name || ""); } catch { /* storage unavailable */ }
}
document.querySelectorAll(".card[data-card]").forEach((card) => {
  card.querySelector(".card-toggle").onclick = () => openCard(card.classList.contains("open") ? "" : card.dataset.card);
});
{
  let saved = "";
  try { saved = localStorage.getItem("zcmir.card") || ""; } catch { /* storage unavailable */ }
  openCard(saved);
}
const radioValue = (name) => document.querySelector(`input[name="${name}"]:checked`)?.value;
function searchMode() { return radioValue("search-mode") === "cloud" ? "cloud" : "sweep"; }

/** One line per card header: what the step will do with the current settings. */
function syncSummaries() {
  const on = (id) => !!$(id)?.checked;
  const val = (id) => $(id)?.value;
  const set = (id, text) => { const el = $(id); if (el) { el.textContent = text; el.title = text; } };
  const band = on("band") ? `band ${val("bp-fine")}–${val("bp-coarse")} px` : "";
  set("sum-pre", [on("clahe") ? "CLAHE" : "", band, on("invert") ? "inverted" : ""].filter(Boolean).join(" · ") || "none");
  const pos = radioValue("detector") !== "gls";
  set("sum-det", pos ? `POS-GIFT · ${on("pg-search") ? "any rotation" : "upright"}` : `GLS-MIFT · ${val("d-n_octaves") || ""} octaves`);
  const px = val("d-inlier_px");
  const method = { lofsc: "Lo-FSC", prosac: "PROSAC", magsac: "MAGSAC++" }[val("match-method")] || "";
  set("sum-mat", `${pos ? "FSC + POS" : method}${px ? ` · ${px} px` : ""}`);
  const step = /^E4/.test(val("metric") || "") ? "line search" : "Gauss–Newton";
  const metric = val("metric") || "SMI";
  set("sum-pose", `${metric}${metric !== "NCC" && on("sym") ? " · sym" : ""} · ${step}`);
  set("sum-search", searchMode() === "cloud" ? `cloud · ${val("hho-n")} seeds` : `sweep · ${sweepShape().join(" × ")}`);
  syncSweepUi();
}

/** The sweep's grid sizes as the engine will build them (rotations, scales, then shear and
 *  stretch when on). */
function sweepShape() {
  const int = (id, d) => { const v = parseInt($(id)?.value, 10); return Number.isFinite(v) && v > 0 ? v : d; };
  const out = [int("sw-nth", 90), int("sw-ns", 20)];
  if ($("sw-shear")?.checked && +$("sw-shear-max")?.value > 0) out.push(int("sw-nshear", 3));
  if ($("sw-aniso")?.checked && +$("sw-aniso-max")?.value > 0) out.push(int("sw-naniso", 3));
  return out;
}

/** The sweep card: ± or from–to rotation inputs, shear / stretch rows dimmed when off, the map count. */
function syncSweepUi() {
  const range = $("sw-rot-mode")?.value === "range";
  if ($("sw-rot-pm")) $("sw-rot-pm").hidden = range;
  if ($("sw-rot-range")) $("sw-rot-range").hidden = !range;
  for (const [chk, ids] of [["sw-shear", ["sw-shear-max", "sw-nshear"]], ["sw-aniso", ["sw-aniso-max", "sw-naniso"]]]) {
    const on = !!$(chk)?.checked;
    for (const id of ids) {
      const el = $(id);
      if (!el) continue;
      el.disabled = !on;
      el.parentElement.classList.toggle("field-dim", !on);
    }
  }
  const shape = sweepShape();
  const maps = shape.reduce((a, b) => a * b, 1);
  if ($("sw-count")) $("sw-count").textContent = `${maps.toLocaleString()} maps (${shape.join(" × ")})${maps > 20000 ? " — a long sweep" : ""}`;
}
$("rail").addEventListener("change", syncSummaries);
$("rail").addEventListener("input", syncSummaries);
$("btn-rail-show").onclick = () => toggleRail(false);
window.addEventListener("keydown", (ev) => {
  if (typingTarget(ev.target)) return;
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (ev.key === "l" || ev.key === "L") { ev.preventDefault(); toggleLog(); }
  else if ((ev.key === "d" || ev.key === "D") && !tweakOn()) { ev.preventDefault(); $("btn-run").click(); }
  else if (ev.key === "g" || ev.key === "G") { ev.preventDefault(); $("btn-grad").click(); }
  else if (ev.key === "h" || ev.key === "H") { ev.preventDefault(); $("btn-hho-align").click(); }
  else if (ev.key >= "1" && ev.key <= "4") { ev.preventDefault(); setDock(DOCKS[+ev.key - 1], true); }
  else if (ev.key === "5" && $("app").dataset.mode === "stack") { ev.preventDefault(); setDock("stack", true); }
  else if (ev.key === "Escape") { requestCancel(); }
});
$("btn-log").onclick = () => toggleLog();
$("btn-keys").onclick = () => { toggleLog(true); log(HELP); };
$("btn-rail").onclick = () => toggleRail();


const DETECT = [
  ["n_octaves", N_OCTAVES, 1, 6, 1, "Scale pyramid levels", "octaves"],
  ["max_points", MAX_POINTS, 64, 8000, 64, "Max keypoints per image", "max pts"],
  ["min_contrast", MIN_CONTRAST, 0.001, 0.2, 0.001, "FAST corner contrast threshold", "FAST"],
  ["tau", TAU, 0.2, 2, 0.05, "Steerable-filter scale τ", "τ"],
];
const DESCRIBE = [
  ["radius", RADIUS, 8, 80, 1, "Descriptor patch radius in octave pixels. The full-resolution patch is r × scale.", "radius"],
  ["nt", 0, 0, 5, 0.001, "Feature-map soft threshold. Ignored while auto threshold is on.", "threshold"],
];
const MATCH = [
  ["inlier_px", INLIER_PX, 0.5, 80, 0.5, "How far a correspondence may sit from the model and still count. The model is chosen at 10 px; a smaller radius counts a tighter subset of the same model.", "inlier px"],
  ["scale_lo", 0.2, 0.05, 2, 0.05, "Smallest area scale kept. Scale is the square root of the affine area change; 1 preserves area.", "scale min"],
  ["scale_hi", 5, 1, 30, 0.1, "Largest area scale kept.", "scale max"],
  ["n_trials", N_TRIALS, 256, MAX_TRIALS, 256, "Random trials for Lo-FSC / PROSAC / MAGSAC++", "trials"],
  ["seed", 12345, 1, 1e9, 1, "RNG seed for matching", "seed"],
];

// GLS-MIFT descriptor structure (Engine.setStructure; recompiles the shaders).
const GLS_ADV = [
  ["n_sigma", 4, 2, 5, 1, "Steerable-filter scales per pixel (σ = τ, 2τ, …)", "scales"],
  ["n_angle", 6, 4, 8, 2, "Filter orientations over 180°. The descriptor's angular sectors are twice this (so a primary orientation turns both by the same step): 6 → 12 sectors. Descriptor length 2·orientations²·rings.", "orientations"],
  ["n_r", 3, 2, 4, 1, "Descriptor rings", "rings"],
];

// POS-GIFT (js/posgift.js) numeric fields: [id, default, min, max, step, tip, label, settings key].
const PG_DETECT = [
  ["pg_n_octaves", POS_GIFT_DEFAULTS.n_octaves, 1, 4, 1, "Pyramid octaves; each adds a level and its 2/3 copy (levels 1, 2/3, 1/2, 1/3, 1/4, 1/6)", "octaves", "n_octaves"],
  ["pg_max_points", POS_GIFT_DEFAULTS.max_points, 64, 16000, 64, "Keypoints per pyramid level (the strongest in each grid cell)", "max pts", "max_points"],
  ["pg_min_contrast", POS_GIFT_DEFAULTS.min_contrast, 0.001, 0.2, 0.001, "FAST threshold on the normalized phase congruency", "FAST", "min_contrast"],
  ["pg_p1", POS_GIFT_DEFAULTS.p1, 4, 32, 1, "Descriptor ring radii P1, 2·P1, 4·P1, in pixels of each pyramid level", "P1", "p1"],
];
const PG_ADV = [
  ["pg_desc_pow", POS_GIFT_DEFAULTS.desc_pow, 0.5, 4, 0.25, "Sharpening: each sampled point's orientation channels are raised to this power before normalization, emphasizing its dominant orientations (1 = the plain descriptor)", "sharpen", "desc_pow"],
  ["pg_n_orient", POS_GIFT_DEFAULTS.n_orient, 4, 8, 2, "Log-Gabor orientations over 180° (descriptor channels). The angular sectors are twice this (6 → 12 sectors, 30° apart), which also sets the rotation-search step: with 4 (45° steps) the search misses rotations between steps. Recompiles the shaders.", "orientations", "n_orient"],
  ["pg_n_rings", POS_GIFT_DEFAULTS.n_rings, 2, 4, 1, "Descriptor rings; radii P1, 2·P1, 4·P1, 7·P1. Recompiles the shaders.", "rings", "n_rings"],
  ["pg_n_scales", POS_GIFT_DEFAULTS.n_scales, 2, 6, 1, "Log-Gabor scales (wavelengths min · mult^s). Recompiles the shaders.", "scales", "n_scales"],
  ["pg_min_wl", POS_GIFT_DEFAULTS.min_wl, 2, 12, 0.5, "Wavelength of the finest Log-Gabor filter, pixels", "min λ", "min_wl"],
  ["pg_mult", POS_GIFT_DEFAULTS.mult, 1.2, 3, 0.1, "Wavelength ratio between successive scales", "λ ratio", "mult"],
  ["pg_sigma_onf", POS_GIFT_DEFAULTS.sigma_onf, 0.3, 0.95, 0.05, "Log-Gabor bandwidth: σ/f₀ of the radial Gaussian on log frequency (smaller is wider)", "σ/f₀", "sigma_onf"],
  ["pg_pc_k", POS_GIFT_DEFAULTS.pc_k, 0, 5, 0.1, "Phase-congruency noise threshold, in standard deviations of the estimated noise energy", "noise k", "pc_k"],
  ["pg_cutoff", POS_GIFT_DEFAULTS.cutoff, 0, 1, 0.05, "Frequency-spread cutoff below which phase congruency is down-weighted", "cutoff", "cutoff"],
  ["pg_pc_g", POS_GIFT_DEFAULTS.pc_g, 1, 20, 1, "Sharpness of that down-weighting", "g", "pc_g"],
  ["pg_gsig", POS_GIFT_DEFAULTS.gsig, 1, 10, 0.5, "σ of the Gaussian that pools each sampled point's neighbourhood, pixels", "ring σ", "gsig"],
  ["pg_pos_p1", POS_GIFT_DEFAULTS.pos_p1, 4, 40, 1, "POS re-matching: ring radius P1 of the full-resolution descriptors", "POS P1", "pos_p1"],
  ["pg_pos_k", POS_GIFT_DEFAULTS.pos_k, 4, 64, 1, "POS re-matching: keypoints searched around each predicted point. Recompiles the shaders.", "POS K", "pos_k"],
];

// Population dock: exchange-symmetric tile grid and its per-tile overlap floor.
const TILE_GRID = 8, TILE_MIN_N = 12;

const TAN_KEYS = [
  ["tx", -80, 80], ["ty", -80, 80], ["th", -0.5, 0.5], ["sg", -0.4, 0.4],
  ["al", -0.4, 0.4], ["ga", -0.4, 0.4], ["px", -0.02, 0.02], ["py", -0.02, 0.02],
];

function cropSpecimen(rgba, w, h, thresh = 5) {
  let x0 = w, y0 = h, x1 = 0, y1 = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = (y * w + x) * 4;
    const g = (rgba[i] + rgba[i + 1] + rgba[i + 2]) / 3;
    if (g > thresh) { if (x < x0) x0 = x; if (y < y0) y0 = y; if (x >= x1) x1 = x + 1; if (y >= y1) y1 = y + 1; }
  }
  if (x1 <= x0) return { rgba, w, h };
  const nw = x1 - x0, nh = y1 - y0;
  const out = new Uint8ClampedArray(nw * nh * 4);
  for (let y = 0; y < nh; y++) out.set(rgba.subarray(((y0 + y) * w + x0) * 4, ((y0 + y) * w + x1) * 4), y * nw * 4);
  return { rgba: out, w: nw, h: nh };
}


function fitDraw(cv, img, w, h, pad) {
  const ctx = cv.getContext("2d");
  const dpr = devicePixelRatio || 1;
  const rw = cv.clientWidth, rh = cv.clientHeight;
  cv.width = Math.max(1, rw * dpr); cv.height = Math.max(1, rh * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0c0c0c"; ctx.fillRect(0, 0, rw, rh);
  const p = pad || { l: 0, r: 0, t: 0, b: 0 };
  if (!img) {
    ctx.fillStyle = "#5c5c5c";
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("drop image", rw / 2, rh / 2);
    return { s: 1, ox: 0, oy: 0, w: w || 1, h: h || 1 };
  }
  const bytes = img.byteLength != null ? img.byteLength : img.length;
  w = Math.max(1, w | 0); h = Math.max(1, h | 0);
  if (bytes && bytes % 4 === 0 && bytes !== w * h * 4) {
    if (bytes % (w * 4) === 0) h = bytes / (w * 4);
    else if (bytes % (h * 4) === 0) w = bytes / (h * 4);
  }
  const innerW = Math.max(1, rw - (p.l || 0) - (p.r || 0));
  const innerH = Math.max(1, rh - (p.t || 0) - (p.b || 0));
  const s = Math.min(innerW / w, innerH / h);
  const dw = w * s, dh = h * s;
  const ox = (p.l || 0) + (innerW - dw) / 2, oy = (p.t || 0) + (innerH - dh) / 2;
  const tmp = document.createElement("canvas");
  tmp.width = w; tmp.height = h;
  tmp.getContext("2d").putImageData(new ImageData(img instanceof Uint8ClampedArray ? img : new Uint8ClampedArray(img), w, h), 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, ox, oy, dw, dh);
  return { s, ox, oy, w, h };
}

function drawCbar() {
  const cv = $("cv-cbar");
  if (!cv || cv.hidden || !$("map-cbar")?.checked) return;
  const dpr = devicePixelRatio || 1;
  const w = Math.max(1, cv.clientWidth), h = Math.max(1, cv.clientHeight);
  cv.width = Math.max(1, (w * dpr) | 0);
  cv.height = Math.max(1, (h * dpr) | 0);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const theme = +($("theme")?.value || 4);
  for (let y = 0; y < h; y++) {
    const [r, g, b] = colormap(1 - (y + 0.5) / h, theme);
    ctx.fillStyle = `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;
    ctx.fillRect(0, y, w, 1);
  }
  ctx.strokeStyle = "rgba(220,220,220,0.65)";
  ctx.lineWidth = 1;
  const ticks = [0, 0.5, 1];
  for (const t of ticks) {
    const y = Math.min(h - 1, Math.max(0, (1 - t) * (h - 1)));
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(5, y); ctx.stroke();
  }
}

function syncMapBar() {
  const on = !!$("map-cbar")?.checked;
  if ($("cv-cbar")) $("cv-cbar").hidden = !on;
  $("map-body")?.classList.toggle("no-cbar", !on);
  if (on) drawCbar();
}

/* ── Dense maps: shift and rotation × scale, side by side ─────────────────────
 * Results live in state.maps[kind] (each carries the pose H it was computed at).
 * Each pane keeps its own view in state.mapViews[kind] (null = fit), and one
 * transform serves both drawing and picking, so a click lands on the cell drawn
 * under it. Map cell ix covers [ix, ix + 1) in shifted (zero-lag-centred) indices.
 * The rotation × scale map is drawn mirrored in x so the applied Δθ increases to the
 * right; mapCol / mapIx convert between index and drawn column for both maps. */
const mapCol = (kind, c, x) => (kind === "rs" ? c.x0 + c.w - x : x - c.x0);
const mapIx = (kind, c, col) => (kind === "rs" ? c.x0 + c.w - col : col + c.x0);
const MAP_PANES = {
  shift: { cv: "cv-map", meta: "map-meta" },
  rs: { cv: "cv-rs", meta: "rs-meta" },
};
const RS_PAD = { l: 38, r: 8, t: 8, b: 24 };
const NO_PAD = { l: 0, r: 0, t: 0, b: 0 };

/** Colorized map image, cached on the result per theme / log / scale band. */
function mapVis(out) {
  const theme = +($("theme")?.value || 4);
  const log = !!$("map-log")?.checked;
  const rs = out.kind === "rs";
  const n = out.n, ny = out.ny || n;
  const [sMin, sMax] = rsScaleBounds();
  const key = `${theme}|${log}|${rs ? `${sMin}|${sMax}` : ""}`;
  if (out._vis && out._visKey === key) return out._vis;
  let field = out.score || out.ncc;
  let rgb = out.rgb, vw = out.vw || n, vh = out.vh || ny;
  let crop = out.crop || { x0: 0, y0: 0, w: vw, h: vh };
  let cropRect = null;
  if (rs && field && n && out.dlam) {
    field = maskRsScale(field, n, out.dlam, sMin, sMax);
    cropRect = rsScaleCrop(n, out.dlam, sMin, sMax);
  }
  if (field && n) {
    const vis = colorizeShifted(field, n, {
      theme, signed: theme === 5 || !!out.ncc || !!out.signed, crop: !rs, cropRect, log, ny,
    });
    rgb = vis.rgb; vw = vis.vw; vh = vis.vh; crop = vis.crop;
  }
  if (!rgb) return null;
  const img = document.createElement("canvas");
  img.width = vw; img.height = vh;
  const data = new ImageData(rgb instanceof Uint8ClampedArray ? rgb : new Uint8ClampedArray(rgb), vw, vh);
  if (rs) {
    const tmp = document.createElement("canvas");
    tmp.width = vw; tmp.height = vh;
    tmp.getContext("2d").putImageData(data, 0, 0);
    const g = img.getContext("2d");
    g.scale(-1, 1);
    g.drawImage(tmp, -vw, 0);
  } else img.getContext("2d").putImageData(data, 0, 0);
  out._vis = { img, vw, vh, crop: { x0: crop.x0, y0: crop.y0, w: vw, h: vh } };
  out._visKey = key;
  return out._vis;
}

/** Fit view: the shift map keeps square cells; rotation × scale fills its frame. */
function mapFit(W, H, vis, pad, stretch) {
  const iw = Math.max(1, W - pad.l - pad.r), ih = Math.max(1, H - pad.t - pad.b);
  let sx = iw / vis.vw, sy = ih / vis.vh;
  if (!stretch) sx = sy = Math.min(sx, sy);
  return { sx, sy, ox: pad.l + (iw - vis.vw * sx) / 2, oy: pad.t + (ih - vis.vh * sy) / 2 };
}

/** A zoomed map view kept on the map: along each axis the map covers its frame (or, where it is
 *  narrower than the frame, stays centred as in the fit view). */
function clampMapView(v, frame, vis) {
  const axis = (o, size, lo, len) => (size <= len + 0.5 ? lo + (len - size) / 2 : Math.min(lo, Math.max(lo + len - size, o)));
  return { ...v, ox: axis(v.ox, vis.vw * v.sx, frame.x, frame.w), oy: axis(v.oy, vis.vh * v.sy, frame.y, frame.h) };
}

function renderMap(kind) {
  const P = MAP_PANES[kind];
  const cv = $(P.cv);
  if (!cv) return;
  const dpr = devicePixelRatio || 1;
  const W = Math.max(1, cv.clientWidth), H = Math.max(1, cv.clientHeight);
  const tw = Math.max(1, (W * dpr) | 0), th = Math.max(1, (H * dpr) | 0);
  if (cv.width !== tw || cv.height !== th) { cv.width = tw; cv.height = th; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0c0c0c";
  ctx.fillRect(0, 0, W, H);
  const out = state.maps[kind];
  const vis = out && mapVis(out);
  state.mapDraw[kind] = null;
  if (!vis) {
    ctx.fillStyle = "#5c5c5c";
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(kind === "rs" ? "rotation × scale map" : "shift map", W / 2, H / 2);
    return;
  }
  const rs = kind === "rs";
  const pad = rs ? RS_PAD : NO_PAD;
  const fit = mapFit(W, H, vis, pad, rs);
  const v = state.mapViews[kind] || fit;
  const frame = { x: pad.l, y: pad.t, w: Math.max(1, W - pad.l - pad.r), h: Math.max(1, H - pad.t - pad.b) };
  const c = vis.crop, n = out.n, ny = out.ny || n;
  const to = (ix, iy) => [v.ox + mapCol(kind, c, ix) * v.sx, v.oy + (iy - c.y0) * v.sy];
  ctx.save();
  ctx.beginPath();
  ctx.rect(frame.x, frame.y, frame.w, frame.h);
  ctx.clip();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(vis.img, v.ox, v.oy, vis.vw * v.sx, vis.vh * v.sy);
  // Current pose (zero lag): a small cross.
  const [zx, zy] = to((n >> 1) + 0.5, (ny >> 1) + 0.5);
  ctx.strokeStyle = "rgba(220,220,220,0.6)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(zx - 6, zy); ctx.lineTo(zx + 6, zy);
  ctx.moveTo(zx, zy - 6); ctx.lineTo(zx, zy + 6);
  ctx.stroke();
  // Peak: an amber square.
  const wrap = (x, m) => ((x % m) + m) % m;
  const psx = rs ? out.dx : out.dx * (out.cw || n) / (out.aw || n);
  const psy = rs ? out.dy : out.dy * (out.ch || ny) / (out.ah || ny);
  const [px, py] = to(wrap((n >> 1) + Math.round(psx), n) + 0.5, wrap((ny >> 1) + Math.round(psy), ny) + 0.5);
  ctx.strokeStyle = "#c4a35a";
  ctx.strokeRect(px - 4, py - 4, 8, 8);
  // Previewed sample: a gold ring.
  const mk = state.mapMark;
  if (mk && mk.kind === kind) {
    const [mx, my] = to(mk.ix + 0.5, mk.iy + 0.5);
    ctx.strokeStyle = "#000"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(mx, my, 7, 0, 6.2832); ctx.stroke();
    ctx.strokeStyle = "#e6c36a"; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(mx, my, 7, 0, 6.2832); ctx.stroke();
  }
  ctx.restore();
  if (rs) drawRsFrame(ctx, frame, out, v, c);
  state.mapDraw[kind] = { v, fit, crop: c, frame, n };
  if (kind === "shift") syncMapBar();
}

function renderMaps() {
  renderMap("shift");
  renderMap("rs");
}

/** Axes of the rotation × scale map, placed for the current zoom: Δθ applied (degrees)
 * along x, scale factor applied along y (log). */
function drawRsFrame(ctx, f, out, v, c) {
  const n = out.n, nTh = out.nTh || n, dlam = out.dlam || 0.01;
  const ixOf = (x) => mapIx("rs", c, (x - v.ox) / v.sx);
  const iyOf = (y) => (y - v.oy) / v.sy + c.y0;
  const degOf = (ix) => -((ix - n / 2) / nTh) * 360;       // cell ix → applied Δθ
  const logsOf = (iy) => -(iy - n / 2) * dlam;              // cell iy → applied log scale
  const xOfDeg = (d) => v.ox + mapCol("rs", c, n / 2 - (d / 360) * nTh + 0.5) * v.sx;
  const yOfLogs = (l) => v.oy + (n / 2 - l / dlam + 0.5 - c.y0) * v.sy;
  ctx.save();
  ctx.strokeStyle = "rgba(168,168,168,0.5)";
  ctx.fillStyle = "rgba(190,190,190,0.92)";
  ctx.lineWidth = 0.75;
  ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  ctx.beginPath();
  ctx.moveTo(f.x, f.y + f.h + 0.5); ctx.lineTo(f.x + f.w, f.y + f.h + 0.5);
  ctx.moveTo(f.x - 0.5, f.y); ctx.lineTo(f.x - 0.5, f.y + f.h);
  ctx.stroke();
  // Δθ ticks: a step that gives about six labels across the visible span.
  const d0 = degOf(ixOf(f.x + f.w)), d1 = degOf(ixOf(f.x));
  const span = Math.abs(d1 - d0);
  const step = [0.5, 1, 2, 5, 10, 15, 30, 45, 90].find((s) => span / s <= 7) || 90;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let d = Math.ceil(Math.min(d0, d1) / step) * step; d <= Math.max(d0, d1); d += step) {
    if (Math.abs(d) > 180.001) continue;
    const x = xOfDeg(d);
    if (x < f.x - 0.5 || x > f.x + f.w + 0.5) continue;
    ctx.beginPath(); ctx.moveTo(x, f.y + f.h); ctx.lineTo(x, f.y + f.h + 3); ctx.stroke();
    ctx.fillText(`${+d.toFixed(1)}°`, x, f.y + f.h + 4);
  }
  // Scale ticks: nice values inside the visible log range.
  const l0 = logsOf(iyOf(f.y + f.h)), l1 = logsOf(iyOf(f.y));
  const cand = [0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.25, 1.4, 1.5, 1.75, 2, 2.5, 3, 4, 5];
  // ×1 first, then the others at least 16 px from every kept label.
  const inView = cand.filter((s) => Math.log(s) >= Math.min(l0, l1) && Math.log(s) <= Math.max(l0, l1));
  const ticks = [];
  for (const s of [1, ...inView.filter((v) => v !== 1)]) {
    const y = yOfLogs(Math.log(s));
    if (y < f.y - 0.5 || y > f.y + f.h + 0.5) continue;
    if (ticks.every((t) => Math.abs(t.y - y) >= 16)) ticks.push({ s, y });
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const { s, y } of ticks) {
    ctx.beginPath(); ctx.moveTo(f.x, y); ctx.lineTo(f.x - 3, y); ctx.stroke();
    ctx.fillText(`×${s}`, f.x - 4, y);
  }
  ctx.fillStyle = "rgba(150,150,150,0.85)";
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText("Δθ", f.x + f.w, f.y + f.h + 13);
  ctx.restore();
}

function kpInliers(kL, kR, mj, H, px) {
  const t2 = px * px;
  const inl = new Uint8Array(kL.length);
  for (let i = 0; i < kL.length; i++) {
    const j = mj[i];
    if (j === 0xFFFFFFFF || j >= kR.length) continue;
    const p = kL[i], q = kR[j];
    const X = H[0] * p[0] + H[1] * p[1] + H[2];
    const Y = H[3] * p[0] + H[4] * p[1] + H[5];
    const Z = H[6] * p[0] + H[7] * p[1] + H[8];
    const z = Math.abs(Z) < 1e-8 ? 1e-8 : Z;
    const dx = X / z - q[0], dy = Y / z - q[1];
    if (dx * dx + dy * dy <= t2) inl[i] = 1;
  }
  return inl;
}

function hash32(i, seed) {
  let x = Math.imul(i + 1, 0x9E3779B9) ^ Math.imul(seed || 1, 0x85EBCA6B);
  x = Math.imul(x ^ (x >>> 16), 0x7FEB352D);
  x = Math.imul(x ^ (x >>> 15), 0x846CA68B);
  return x >>> 0;
}

function takeHashed(ids, k, seed) {
  if (ids.length <= k) return ids;
  const scored = ids.map((i) => ({ i, h: hash32(i, seed) }));
  scored.sort((a, b) => a.h - b.h || a.i - b.i);
  return scored.slice(0, k).map((o) => o.i);
}

function visMask() {
  const nL = state.kpsL?.length || 0;
  const nR = state.kpsR?.length || 0;
  const L = new Uint8Array(nL), R = new Uint8Array(nR), lines = new Uint8Array(nL);
  const max = Math.max(50, +$("kp-max")?.value || 1000);
  const mode = $("kp-show")?.value || "inliers";
  const seed = (+$("d-seed")?.value || 1) | 0;
  if (!nL) return { L, R, lines, shown: 0, total: 0 };
  const buckets = [[], [], []];
  for (let i = 0; i < nL; i++) {
    const matched = !!(state.matchJ && state.matchJ[i] !== 0xFFFFFFFF);
    const inlier = !!(state.inlier && state.inlier[i]);
    if (mode === "inliers" && !inlier) continue;
    if (mode === "matched" && !matched) continue;
    buckets[inlier ? 0 : matched ? 1 : 2].push(i);
  }
  const pick = [];
  for (const b of buckets) {
    const room = max - pick.length;
    if (room <= 0) break;
    pick.push(...takeHashed(b, room, seed));
  }
  for (const i of pick) {
    L[i] = 1;
    const j = state.matchJ ? state.matchJ[i] : 0xFFFFFFFF;
      if (j !== 0xFFFFFFFF && j < nR) {
        R[j] = 1;
        lines[i] = 1;
      }
  }
  if (mode === "all") {
    const rest = [];
    for (let j = 0; j < nR; j++) if (!R[j]) rest.push(j);
    for (const j of takeHashed(rest, Math.max(0, max - pick.length), seed ^ 0xA5A5)) R[j] = 1;
  }
  return { L, R, lines, shown: pick.length, total: nL };
}

const KP_MAX_TITLE = "Draw at most this many (a random subsample, so correspondences stay readable).";

// Pipeline compiles in the status bar: shown once a burst outlasts 150 ms, so a single quick
// compile does not flash.
let compilePending = false, compileShowTimer = 0;
function showCompiling({ done, total }) {
  const el = $("compiling");
  if (!el) return;
  compilePending = done < total;
  if (!compilePending) {
    clearTimeout(compileShowTimer);
    compileShowTimer = 0;
    el.hidden = true;
    return;
  }
  $("compiling-txt").textContent = `compiling WebGPU shaders ${done} / ${total}`;
  $("compiling-bar").style.setProperty("--p", String(done / total));
  if (el.hidden && !compileShowTimer) {
    compileShowTimer = setTimeout(() => { compileShowTimer = 0; el.hidden = !compilePending; }, 150);
  }
}

function kpScale(kp) {
  const s = kp?.[3];
  return Number.isFinite(s) && s > 0 ? s : 1;
}

function paintKps(cv, view, kps, role, mask) {
  if (!kps?.length || !view) return;
  const dots = $("kp-dots")?.checked !== false;
  const ctx = cv.getContext("2d");
  const dpr = devicePixelRatio || 1;
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const mj = state.matchJ, rev = state.matchRev, inl = state.inlier;
  const sel = state.kpSel || state.kpHover;
  const byScore = $("kp-line-color")?.value === "score";
  for (let i = 0; i < kps.length; i++) {
    if (mask && !mask[i]) continue;
    const x = view.ox + (kps[i][0] - 1) * view.s;
    const y = view.oy + (kps[i][1] - 1) * view.s;
    let matched = false, inlier = false, li = -1;
    if (role === "L") {
      matched = mj && mj[i] !== 0xFFFFFFFF;
      inlier = !!(inl && inl[i]);
      li = i;
    } else {
      li = rev ? rev[i] : -1;
      matched = li >= 0;
      inlier = li >= 0 && inl && inl[li];
    }
    if (dots) {
      const r = Math.max(2.2, 3.2 * kpScale(kps[i]) * view.s);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 6.28);
      ctx.strokeStyle = inlier ? (byScore ? matchStrengthColor(li) : "#7dba6a") : matched ? "#c45a5a" : "#8a8a8a";
      ctx.globalAlpha = inlier ? 0.95 : matched ? 0.85 : 0.4;
      ctx.lineWidth = inlier || matched ? 1.25 : 1;
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + r, y);
      ctx.stroke();
    }
  }
  if (sel) {
    const i = role === sel.side ? sel.i : sel.pair;
    if (i >= 0 && i < kps.length && (!mask || mask[i])) {
      const x = view.ox + (kps[i][0] - 1) * view.s;
      const y = view.oy + (kps[i][1] - 1) * view.s;
      const r = Math.max(2.2, 3.2 * kpScale(kps[i]) * view.s);
      ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.arc(x, y, r + 3, 0, 6.28);
      ctx.strokeStyle = "#dcdcdc"; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }
  ctx.restore();
}

function hitKp(view, kps, mx, my, mask) {
  if (!view || !kps?.length) return -1;
  let best = -1, bd = 14;
  for (let i = 0; i < kps.length; i++) {
    if (mask && !mask[i]) continue;
    const x = view.ox + (kps[i][0] - 1) * view.s;
    const y = view.oy + (kps[i][1] - 1) * view.s;
    const r = Math.max(8, 3.2 * kpScale(kps[i]) * view.s + 4);
    const d = Math.hypot(x - mx, y - my);
    if (d < r && d < bd) { bd = d; best = i; }
  }
  return best;
}

function matchScoreRange() {
  const s = state.matchScore, mj = state.matchJ;
  if (!s || !mj) return null;
  const xs = [];
  for (let i = 0; i < mj.length; i++) {
    if (mj[i] === 0xFFFFFFFF) continue;
    const v = s[i];
    if (Number.isFinite(v)) xs.push(v);
  }
  if (!xs.length) return null;
  xs.sort((a, b) => a - b);
  const lo = xs[(xs.length * 0.05) | 0];
  const hi = xs[Math.min(xs.length - 1, (xs.length * 0.95) | 0)];
  return { lo, hi: hi > lo ? hi : lo + 1e-6 };
}

function matchStrengthColor(i) {
  const rng = state._scoreRange;
  const v = state.matchScore?.[i];
  if (!rng || !Number.isFinite(v)) return "#6a9ec8";
  const t = Math.min(1, Math.max(0, (v - rng.lo) / (rng.hi - rng.lo)));
  const [r, g, b] = colormap(t, 3);
  return `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;
}

function lineColor(i) {
  if (!state.inlier?.[i]) return "#c45a5a";
  if ($("kp-line-color")?.value === "score") return matchStrengthColor(i);
  return "#7dba6a";
}

function drawMatchLines(vis) {
  const cv = $("cv-matches");
  if (!cv) return;
  const dpr = devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = Math.max(1, w * dpr);
  cv.height = Math.max(1, h * dpr);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const showAll = !!$("kp-lines")?.checked;
  const sel = state.kpSel;
  if (!showAll && !sel) return;
  if (!vis || !state.kpsL || !state.kpsR || !state.matchJ || !state._viewL || !state._viewR) return;
  const origin = cv.getBoundingClientRect();
  const to = (canvas, view, kp) => {
    const r = canvas.getBoundingClientRect();
    return [
      r.left - origin.left + view.ox + (kp[0] - 1) * view.s,
      r.top - origin.top + view.oy + (kp[1] - 1) * view.s,
    ];
  };
  ctx.lineCap = "round";
  for (let i = 0; i < state.matchJ.length; i++) {
    const j = state.matchJ[i];
    if (j === 0xFFFFFFFF || j >= state.kpsR.length) continue;
    const picked = sel && (
      (sel.side === "L" && sel.i === i) ||
      (sel.side === "R" && sel.pair === i)
    );
    if (!picked && !(showAll && vis.lines?.[i])) continue;
    const [x0, y0] = to($("cv-left"), state._viewL, state.kpsL[i]);
    const [x1, y1] = to($("cv-right"), state._viewR, state.kpsR[j]);
    ctx.strokeStyle = lineColor(i);
    ctx.globalAlpha = picked || !showAll ? 0.95 : 0.55;
    ctx.lineWidth = picked || !showAll ? 1.6 : 0.9;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  }
}

function putGray(cv, g, w, h) {
  if (!cv || cv.width !== w || cv.height !== h) {
    cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
  }
  const id = new ImageData(w, h);
  const d = id.data, n = w * h;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    const v = g[i] <= 0 ? 0 : g[i] >= 1 ? 255 : (g[i] * 255 + 0.5) | 0;
    d[j] = d[j + 1] = d[j + 2] = v; d[j + 3] = 255;
  }
  cv.getContext("2d").putImageData(id, 0, 0);
  return cv;
}

function checkerTile() {
  return Math.max(2, Math.min(256, (+$("chk-px")?.value || 16) | 0));
}

function putChecker(cv, moving, fixed, w, h, tile, invert) {
  tile = Math.max(1, tile | 0);
  if (!cv || cv.width !== w || cv.height !== h) {
    cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
  }
  const id = new ImageData(w, h);
  const d = id.data;
  const phase = invert ? 1 : 0;
  for (let y = 0, i = 0, j = 0; y < h; y++) {
    const ty = ((y / tile) | 0) + phase;
    for (let x = 0; x < w; x++, i++, j += 4) {
      const g = ((ty + ((x / tile) | 0)) & 1) ? moving[i] : fixed[i];
      const v = g <= 0 ? 0 : g >= 1 ? 255 : (g * 255 + 0.5) | 0;
      d[j] = d[j + 1] = d[j + 2] = v; d[j + 3] = 255;
    }
  }
  cv.getContext("2d").putImageData(id, 0, 0);
  return cv;
}

const poseSpr = { gen: -1, tile: -1, mov: null, fix: null, chk: null, chkInv: null };
// Sprites of the previewed pose (a map click), kept apart from the committed pose's.
const previewSpr = { gen: -1, tile: -1, mov: null, fix: null, chk: null, chkInv: null };
const poseSprInv = { gen: -1, tile: -1, mov: null, fix: null, chk: null, chkInv: null };

function poseSprites(pose, cache = poseSpr) {
  const tile = checkerTile();
  if (cache.gen !== pose.gen) {
    cache.mov = putGray(cache.mov, pose.moving_np, pose.ow, pose.oh);
    cache.fix = putGray(cache.fix, pose.fixed_np, pose.ow, pose.oh);
    cache.gen = pose.gen;
    cache.tile = -1;
  }
  if (cache.tile !== tile) {
    cache.chk = putChecker(cache.chk, pose.moving_np, pose.fixed_np, pose.ow, pose.oh, tile, false);
    cache.chkInv = putChecker(cache.chkInv, pose.moving_np, pose.fixed_np, pose.ow, pose.oh, tile, true);
    cache.tile = tile;
  }
  return cache;
}

function softMix(now, period) {
  return 0.5 - 0.5 * Math.cos((Math.PI * now) / period);
}

function drawArrow(ctx, x0, y0, x1, y1) {
  const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy);
  if (L < 1) return;
  const ah = Math.min(6, L * 0.35);
  const ux = dx / L, uy = dy / L;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - ux * ah - uy * ah * 0.45, y1 - uy * ah + ux * ah * 0.45);
  ctx.lineTo(x1 - ux * ah + uy * ah * 0.45, y1 - uy * ah - ux * ah * 0.45);
  ctx.closePath();
  ctx.fill();
}

function drawLogVecs(ctx, view, pts, vecs, color, pxMax) {
  let mmax = 0;
  for (let i = 0; i < pts.length; i++) mmax = Math.max(mmax, Math.hypot(vecs[i][0], vecs[i][1]));
  if (mmax < 1e-18) return;
  const den = Math.log(100);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.6;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (let i = 0; i < pts.length; i++) {
    const vx = vecs[i][0], vy = vecs[i][1];
    const mag = Math.hypot(vx, vy);
    if (mag < 1e-18) continue;
    const L = Math.max(7, pxMax * Math.log1p(99 * mag / mmax) / den);
    const x0 = view.ox + pts[i][0] * view.s;
    const y0 = view.oy + pts[i][1] * view.s;
    drawArrow(ctx, x0, y0, x0 + (vx / mag) * L, y0 + (vy / mag) * L);
  }
}

const state = {
  left: null, right: null,
  H0: identityH(), tan: Object.fromEntries(TAN_KEYS.map(([k]) => [k, 0])),
  pose: null, group: "homography",
  // Dense maps (see renderMap): results, per-pane views (null = fit), last draw geometry,
  // the previewed map sample, and the previewed pose on the overlay.
  maps: { shift: null, rs: null }, mapViews: { shift: null, rs: null }, mapDraw: { shift: null, rs: null },
  mapMark: null, preview: null,
  undo: [], trail: [], trailSel: -1, _restoring: false,
  kpsL: null, kpsR: null, matchJ: null, matchRev: null, inlier: null, matchScore: null, kpSel: null, kpHover: null,
  n1: 0, n2: 0, poseClimbed: false,
  regView: { z: 1, x: 0, y: 0 },
  trailView: { i0: 0, i1: null },
  rsCenter: null,
};

function tweakOn() { return !!$("tweak")?.checked; }
function zoomOverlapOn() {
  return !!$("zoom-ov")?.checked;
}
function mapLiveOn() { return !!$("map-live")?.checked; }
function overlapBox(pose, bgW, bgH, mapped) {
  if (!zoomOverlapOn() || !pose) return { x0: 0, y0: 0, w: pose.ow, h: pose.oh };
  const { ox, oy, ow, oh } = pose;
  const fx0 = Math.max(0, -ox), fy0 = Math.max(0, -oy);
  const fx1 = Math.min(ow, -ox + bgW);
  const fy1 = Math.min(oh, -oy + bgH);
  const dst = mapped.map(([x, y]) => [x - 1 - ox, y - 1 - oy]);
  let mx0 = Infinity, my0 = Infinity, mx1 = -Infinity, my1 = -Infinity;
  for (const [x, y] of dst) {
    if (x < mx0) mx0 = x;
    if (y < my0) my0 = y;
    if (x > mx1) mx1 = x;
    if (y > my1) my1 = y;
  }
  const x0 = Math.max(0, Math.floor(Math.max(fx0, mx0)));
  const y0 = Math.max(0, Math.floor(Math.max(fy0, my0)));
  const x1 = Math.min(ow - 1, Math.ceil(Math.min(fx1, mx1) - 1e-6));
  const y1 = Math.min(oh - 1, Math.ceil(Math.min(fy1, my1) - 1e-6));
  if (x1 - x0 < 8 || y1 - y0 < 8) return { x0: 0, y0: 0, w: ow, h: oh };
  // A margin of 10% of the overlap per side, so its edges and the image corners that stick out of
  // it stay in view (never past the whole canvas).
  const pad = 0.1 * Math.max(x1 - x0, y1 - y0);
  const bx0 = Math.max(0, x0 - pad), by0 = Math.max(0, y0 - pad);
  const bx1 = Math.min(ow - 1, x1 + pad), by1 = Math.min(oh - 1, y1 + pad);
  return { x0: bx0, y0: by0, w: bx1 - bx0 + 1, h: by1 - by0 + 1 };
}
function overlayBox(pose, H) {
  if (!state.left || !state.right || !pose) return { x0: 0, y0: 0, w: pose?.ow || 1, h: pose?.oh || 1 };
  return overlapBox(pose, state.right.w, state.right.h, mapPts(H || currentH(), movingSrcCorners(state.left.w, state.left.h)));
}
function overlayBoxInv(pose, H) {
  if (!state.left || !state.right || !pose) return { x0: 0, y0: 0, w: pose?.ow || 1, h: pose?.oh || 1 };
  const Hi = inv3(H || currentH());
  const { w, h } = state.right;
  const corners = [[1, 1], [w, 1], [w, h], [1, h]];
  return overlapBox(pose, state.left.w, state.left.h, mapPts(Hi, corners));
}
function overlayView(cv, pose, H, box) {
  const rw = cv.clientWidth, rh = cv.clientHeight;
  const rv = state.regView || { z: 1, x: 0, y: 0 };
  box = box || overlayBox(pose, H);
  const sFit = Math.min(rw / Math.max(box.w, 1), rh / Math.max(box.h, 1));
  const s = sFit * (rv.z || 1);
  const dw = box.w * s, dh = box.h * s;
  const ox = (rw - dw) / 2 + (rv.x || 0) - box.x0 * s;
  const oy = (rh - dh) / 2 + (rv.y || 0) - box.y0 * s;
  return { s, ox, oy, w: pose.ow, h: pose.oh, box, sFit };
}
function ffdSource() { return $("ffd-frame")?.value === "source"; }
function syncTweakUi() {
  $("cv-reg")?.classList.toggle("tweak-on", tweakOn());
  const hint = $("keys-hint");
  if (hint) hint.textContent = tweakOn()
    ? "Edit pose is on: drag, wheel and keys move the moving image. ? lists the keys."
    : "Turn on edit pose (overlay bar) to move the image by hand. ? lists the keys.";
}

function settings() {
  const s = {};
  for (const [k] of DETECT) s[k] = +$(`d-${k}`).value;
  for (const [k] of DESCRIBE) s[k] = +$(`d-${k}`).value;
  for (const [k] of MATCH) s[k] = +$(`d-${k}`).value;
  for (const [k, v, lo, hi, st] of GLS_ADV) s[k] = Math.min(hi, Math.max(lo, Math.round(+$(`d-${k}`).value / st) * st || v));
  return s;
}

function currentH() {
  return compose(state.H0, state.tan, state.group);
}

function hDigits() {
  return $("h-digits")?.checked ? 8 : 4;
}

function writeH(H) {
  H = H || currentH();
  if ($("Htxt")) $("Htxt").textContent = fmtH(H, hDigits());
  if ($("h-summary")) $("h-summary").textContent = poseLine(H);
}

function syncTan() {
  for (const [k] of TAN_KEYS) {
    const el = $(`t-${k}`);
    if (el) el.value = (+state.tan[k]).toFixed(4);
  }
}

function zeroTan() {
  for (const [k] of TAN_KEYS) state.tan[k] = 0;
  syncTan();
}

function snapshotPose() {
  return {
    H0: Float64Array.from(state.H0),
    tan: { ...state.tan },
    cps: state.cps ? Float32Array.from(state.cps) : null,
    cpsGx: state.cpsGx || 0,
    ffd: !!$("ffd")?.checked,
    ffdFrame: $("ffd-frame")?.value || "target",
    gx: +($("ffd-g")?.value || 4),
    group: state.group,
  };
}

function pushUndo() {
  state.undo.push(snapshotPose());
  if (state.undo.length > 40) state.undo.shift();
}

function restorePose(s) {
  state.H0 = Float64Array.from(s.H0);
  Object.assign(state.tan, s.tan);
  state.cps = s.cps ? Float32Array.from(s.cps) : null;
  state.cpsGx = s.cpsGx || 0;
  if (s.group) setGroup(s.group);
  if ($("ffd") && s.ffd != null) $("ffd").checked = !!s.ffd;
  if ($("ffd-g") && s.gx) $("ffd-g").value = s.gx;
  if ($("ffd-frame") && s.ffdFrame) $("ffd-frame").value = s.ffdFrame;
  syncTan();
  $("app").dataset.group = state.group;
  syncModelUi();
}

function fmtAxis(v) {
  const a = Math.abs(v);
  if (!Number.isFinite(v)) return "—";
  if (a >= 10000 || (a > 0 && a < 0.01)) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  return v.toFixed(2);
}

function trailWindow() {
  const n = state.trail.length;
  const tv = state.trailView || { i0: 0, i1: null };
  if (!n) return { i0: 0, i1: 0 };
  let i0 = Math.max(0, Math.min(n - 1, tv.i0 | 0));
  let i1 = tv.i1 == null ? n - 1 : tv.i1;
  i1 = Math.max(i0, Math.min(n - 1, i1 | 0));
  return { i0, i1 };
}

function resetTrailView() {
  state.trailView = { i0: 0, i1: null };
  drawTrail();
}

function trailCaption(p, extra = "") {
  if (!p) return "";
  const bit = (name, v) => (Number.isFinite(v) ? `${name} ${v.toFixed(4)}` : "");
  const parts = [bit(p.metric || "score", p.score), bit(p.metric2 || "inverse", p.score2)].filter(Boolean);
  return `${p.label}  ${parts.join("  ·  ")}${extra}`;
}

function trailRange(vals) {
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (!(hi > lo)) { hi += 1; lo -= 1; }
  const pad = 0.04 * (hi - lo);
  return { lo: lo - pad, hi: hi + pad };
}

function trailLayout() {
  const cv = $("cv-flex");
  const w = Math.max(1, cv.clientWidth), h = Math.max(1, cv.clientHeight);
  const n = state.trail.length;
  const { i0, i1 } = trailWindow();
  const vis = n ? state.trail.slice(i0, i1 + 1) : [];
  const pad = { l: 52, r: 16, t: 18, b: 24 };
  const vals = [];
  for (const p of vis) for (const k of ["score", "score2"]) if (Number.isFinite(p[k])) vals.push(p[k]);
  let lo = 0, hi = 1;
  if (vals.length) ({ lo, hi } = trailRange(vals));
  const span = Math.max(1, i1 - i0);
  const xAt = (i) => pad.l + ((i - i0) / span) * (w - pad.l - pad.r);
  const yAt = (s) => pad.t + (1 - (s - lo) / (hi - lo)) * (h - pad.t - pad.b);
  const name1 = [...vis].reverse().find((p) => p.metric)?.metric || "score";
  return { cv, w, h, n, pad, lo, hi, xAt, yAt, i0, i1, name1 };
}

// Score tab: every recorded pose, newest first; the best forward score is starred.
let trailListSig = "";
const fmtScoreRow = (v) => (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toPrecision(5));
function drawTrailList() {
  const ol = $("trail-list");
  if (!ol) return;
  const n = state.trail.length;
  const last = state.trail[n - 1];
  const sig = `${n}|${state.trailSel}|${last?.score}|${last?.label}`;
  if (sig === trailListSig) return;
  trailListSig = sig;
  if ($("trail-count")) $("trail-count").textContent = n ? `${n}` : "";
  if (!n) { ol.innerHTML = `<li class="empty">Match, Pose and Search record their poses here.</li>`; return; }
  let best = -1;
  state.trail.forEach((p, i) => { if (Number.isFinite(p.score) && (best < 0 || p.score > state.trail[best].score)) best = i; });
  const esc = (t) => String(t).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
  const rows = [];
  for (let i = n - 1; i >= 0; i--) {
    const p = state.trail[i];
    const v = [p.score, p.score2].filter(Number.isFinite).map(fmtScoreRow).join(" / ");
    const cls = [i === state.trailSel ? "sel" : "", i === best ? "best" : ""].filter(Boolean).join(" ");
    rows.push(`<li data-i="${i}" class="${cls}" title="${esc(trailCaption(p))}"><span class="k">${i + 1}</span><span class="lab">${esc(p.label || "")}</span><span class="v">${v}</span></li>`);
  }
  ol.innerHTML = rows.join("");
}

function drawTrail() {
  drawTrailList();
  const { cv, w, h, n, pad, lo, hi, xAt, yAt, i0, i1, name1 } = trailLayout();
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const dpr = devicePixelRatio || 1;
  cv.width = Math.max(1, w * dpr); cv.height = Math.max(1, h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0c0c0c";
  ctx.fillRect(0, 0, w, h);
  const ny = 4;
  ctx.strokeStyle = "#222";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= ny; i++) {
    const y = pad.t + (i / ny) * (h - pad.t - pad.b);
    ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y);
  }
  ctx.stroke();
  ctx.strokeStyle = "#3a3a3a";
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();
  ctx.font = "10px ui-monospace, Consolas, monospace";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#6a9ec8";
  ctx.textAlign = "right";
  ctx.fillText(fmtAxis(hi), pad.l - 5, pad.t);
  ctx.fillText(fmtAxis(lo), pad.l - 5, h - pad.b);
  ctx.save();
  ctx.translate(11, (pad.t + h - pad.b) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(name1, 0, 0);
  ctx.restore();
  ctx.fillStyle = "#8a8a8a";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("step", (pad.l + w - pad.r) / 2, h - 12);
  if (!n) {
    ctx.fillStyle = "#5c5c5c";
    ctx.font = "12px ui-sans-serif, system-ui";
    ctx.textBaseline = "middle";
    ctx.fillText("score vs step", (pad.l + w - pad.r) / 2, h * 0.48);
    return;
  }
  ctx.fillText(String(i0 + 1), xAt(i0), h - pad.b + 4);
  if (i1 !== i0) ctx.fillText(String(i1 + 1), xAt(i1), h - pad.b + 4);
  ctx.save();
  ctx.beginPath();
  ctx.rect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
  ctx.clip();
  const strokeSeries = (key, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.35;
    ctx.beginPath();
    let pen = false;
    state.trail.forEach((p, i) => {
      if (i < i0 - 1 || i > i1 + 1) return;
      const s = p[key];
      if (!Number.isFinite(s)) { pen = false; return; }
      const x = xAt(i), y = yAt(s);
      pen ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), pen = true);
    });
    ctx.stroke();
  };
  const dots = (key, color) => {
    state.trail.forEach((p, i) => {
      if (i < i0 || i > i1 || !Number.isFinite(p[key])) return;
      ctx.beginPath();
      ctx.arc(xAt(i), yAt(p[key]), i === state.trailSel ? 3.5 : 2.2, 0, 6.28);
      ctx.fillStyle = i === state.trailSel ? "#dcdcdc" : color;
      ctx.fill();
    });
  };
  strokeSeries("score", "#6a9ec8");
  strokeSeries("score2", "#c9864a");
  dots("score", "#6a9ec8");
  dots("score2", "#c9864a");
  ctx.restore();
}

function hitTrail(clientX) {
  const { cv, n, xAt, i0, i1 } = trailLayout();
  if (!n) return -1;
  const x = clientX - cv.getBoundingClientRect().left;
  let best = i0, bd = 1e9;
  for (let i = i0; i <= i1; i++) {
    const d = Math.abs(xAt(i) - x);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function recordTrail(score, label, force, extra) {
  if (state._restoring || !Number.isFinite(score)) return;
  const score2 = extra && Number.isFinite(extra.score2) ? extra.score2 : null;
  const score3 = extra && Number.isFinite(extra.score3) ? extra.score3 : null;
  const metric = extra?.metric || null;
  const metric2 = extra?.metric2 || null;
  const metric3 = extra?.metric3 || null;
  const last = state.trail[state.trail.length - 1];
  const close = (a, b) => (a == null && b == null) || (a != null && b != null && Math.abs(a - b) < 1e-8);
  if (last && last.label === label && close(last.score, score) && close(last.score2, score2) && close(last.score3, score3)) return;
  const rel = (a, b) => {
    if (a == null && b == null) return true;
    if (a == null || b == null) return false;
    return Math.abs(a - b) / Math.max(Math.abs(a), 1) < 0.005;
  };
  if (!force && last && rel(last.score, score) && rel(last.score2, score2) && rel(last.score3, score3)) return;
  state.trail.push({ score, score2, score3, metric, metric2, metric3, label, snap: snapshotPose() });
  if (state.trail.length > 240) state.trail.shift();
  state.trailSel = state.trail.length - 1;
  drawTrail();
  $("flex-meta").textContent = trailCaption(state.trail[state.trailSel]);
}

function setBusy(on, msg, kind) {
  if (on) job.stop = false;
  job.kind = on ? (kind || "job") : null;
  const el = $("busy");
  if (el) {
    el.hidden = !on;
    if (msg) el.textContent = msg;
  }
  // The controls stay locked for a whole stack run, between its jobs too.
  const ui = on || stackRun.active;
  const cancel = $("btn-cancel");
  if (cancel) cancel.hidden = !ui;
  const stop = $("btn-stop");
  if (stop) { stop.disabled = !ui; stop.hidden = !ui; }
  if ($("btn-auto")) $("btn-auto").hidden = ui;
  if ($("btn-stack-run")) $("btn-stack-run").hidden = ui;
  if ($("busy-msg")) $("busy-msg").textContent = ui ? (stackRun.active ? `${stackRun.msg || ""} ${on ? msg || "" : ""}` : msg || "") : "";
  for (const id of ["btn-auto", "btn-run", "btn-refit", "btn-grad", "btn-hho-align", "btn-open-left", "btn-open-right",
    "btn-stack-left", "btn-stack-right", "btn-slice-kp", "btn-slice-carry", "btn-stack-from", "btn-slice-prev", "btn-slice-next", "preset"]) {
    const b = $(id);
    if (b) b.disabled = ui;
  }
  const match = $("btn-rematch");
  if (match) {
    match.disabled = ui || !state.n1;
    match.classList.toggle("ready", !on && !!state.n1 && !state.matchJ);
  }
  markNext();
}

function markNext() {
  const step = !state.n1 ? "detect"
    : !state.matchJ ? "match"
    : !state.poseClimbed ? "climb"
    : "search";
  const done = { detect: !!state.n1, match: !!state.matchJ, climb: !!state.poseClimbed, search: false };
  document.querySelectorAll("[data-step]").forEach((el) => {
    el.classList.toggle("is-next", el.dataset.step === step && !!(state.left && state.right));
    el.classList.toggle("is-done", !!done[el.dataset.step] && el.dataset.step !== step);
  });
}

function markAuto(step) {
  document.querySelectorAll("[data-step]").forEach((el) => {
    el.classList.toggle("is-auto", !!step && el.dataset.step === step);
  });
}
markNext();

/** The pose in words, for the Pose card: scale, rotation, shift (fixed pixels). */
function poseBrief(H) {
  const s = Math.abs(H[8]) > 1e-12 ? H[8] : 1;
  const a = H[0] / s, b = H[1] / s, c = H[3] / s, d = H[4] / s;
  const ang = Math.atan2(c - b, a + d) * (180 / Math.PI);
  const persp = Math.abs(H[6] / s) + Math.abs(H[7] / s) > 1e-7 ? " · perspective" : "";
  return `scale ${Math.sqrt(Math.abs(a * d - b * c)).toFixed(3)} · rot ${ang >= 0 ? "+" : ""}${ang.toFixed(2)}° · shift ${(H[2] / s).toFixed(1)}, ${(H[5] / s).toFixed(1)}${persp}`;
}

async function fileToRgba(file, crop) {
  const bmp = await createImageBitmap(file);
  const c = document.createElement("canvas");
  c.width = bmp.width; c.height = bmp.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const im = ctx.getImageData(0, 0, bmp.width, bmp.height);
  const out = crop ? cropSpecimen(im.data, im.width, im.height) : { rgba: im.data, w: im.width, h: im.height };
  out.name = file.name;
  return out;
}

function withTimeout(p, ms, msg) {
  return Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error(msg)), ms)),
  ]);
}

function chip(s) { $("adapter").textContent = s; }

// Let the busy state paint before a long GPU job. requestAnimationFrame never
// fires in a background tab, so a job started there would wait forever.
function nextPaint() {
  return new Promise((r) => {
    const t = setTimeout(r, 50);
    requestAnimationFrame(() => { clearTimeout(t); r(); });
  });
}

async function main() {
  chip("booting…");
  log("booting…  origin " + location.origin + "  secure=" + window.isSecureContext);
  if (!window.isSecureContext) {
    throw new Error("WebGPU needs a secure origin. Open http://127.0.0.1:4174/ (npm run dev).");
  }
  if (!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
  chip("loading zcmir…");
  log("loading zcmir…");
  zc = await withTimeout(
    Zcmir.create({ wasm: new URL("../zcmir/zcmir.wasm", import.meta.url), onCompile: showCompiling }),
    30000,
    "Timed out creating the zcmir engine",
  );
  $("adapter").textContent = zc.state.info || "gpu";
  zc.device.addEventListener("uncapturederror", (ev) => log("GPU: " + ev.error.message));
  const f16Avail = zc.device.features.has("shader-f16");
  const halfEl = $("half");
  const halfLab = $("half-lab");
  if (f16Avail) {
    halfEl.disabled = false;
    halfEl.checked = true;
  } else {
    halfEl.disabled = true;
    halfEl.checked = false;
    if (halfLab) halfLab.title = "fp16 is not available on this GPU";
  }
  // Console / test handle.
  window.cmir = { zc, state };
  theme = +$("theme").value;
  // Offer a 2048 map when the FFT kernel fits (32 KB workgroup memory); canvases
  // over ~1024 px then keep one-pixel shift resolution.
  if ((zc.device.limits.maxComputeWorkgroupStorageSize || 0) >= 32768 && $("map-res") && !$("map-res").querySelector('option[value="2048"]')) {
    $("map-res").insertAdjacentHTML("afterbegin", '<option value="2048">2048</option>');
  }

  function num(id, d) {
    const el = $(id);
    const v = el && el.value !== "" ? +el.value : NaN;
    return Number.isFinite(v) ? v : d;
  }
  function int(id, d) { return Math.max(0, Math.round(num(id, d))); }

  /** Every engine setting from the page's controls (names as in src/settings.zig). */
  function appSettings() {
    const [rsLo, rsHi] = rsScaleBounds();
    const s = {
      metric: METRIC_KEY[$("metric").value] || "smi", exact: !!$("exact")?.checked, calibrated: !!$("map-cal")?.checked,
      symmetric: invOn(), group: state.group, map_res: int("map-res", 0), min_overlap: num("min-ov", 0.05),
      rs_smin: rsLo, rs_smax: rsHi,
      ffd: !!$("ffd")?.checked, ffd_grid: Math.max(2, Math.min(FFD_MAX, int("ffd-g", 4))), ffd_source: $("ffd-frame")?.value === "source",
      ffd_stiffness: Math.max(0, Math.min(1, num("ffd-stiff", 0.15))),
      hop: !!$("hop")?.checked, hop_fm: !!$("hop-fm")?.checked,
      g_a0: num("g-a0", 0.08), g_decay: num("g-decay", 0.5), g_steps: int("g-steps", 7), g_iters: Math.max(1, int("g-iters", 20)),
      clahe: !!$("clahe")?.checked, clahe_grid: 4, clahe_bins: 64,
      band: !!$("band")?.checked, bp_fine: num("bp-fine", 1) || 1, bp_coarse: num("bp-coarse", 6) || 6, invert: !!$("invert")?.checked,
      half: !!$("half")?.checked,
      detector: posGiftOn() ? "pos_gift" : "gls_mift",
      n_octaves: int("d-n_octaves", N_OCTAVES), max_points: int("d-max_points", MAX_POINTS), min_contrast: num("d-min_contrast", MIN_CONTRAST),
      tau: num("d-tau", TAU), radius: num("d-radius", RADIUS), nt: num("d-nt", 0),
      auto_nt: $("opt-nt") ? !!$("opt-nt").checked : true, second_ori: $("opt-second") ? !!$("opt-second").checked : true,
      inlier_px: num("d-inlier_px", INLIER_PX), scale_lo: num("d-scale_lo", 0.2), scale_hi: num("d-scale_hi", 5),
      n_trials: int("d-n_trials", N_TRIALS) || N_TRIALS, seed: int("d-seed", 12345),
      n_sigma: int("d-n_sigma", 4), n_angle: int("d-n_angle", 6), n_r: int("d-n_r", 3),
      match_method: $("match-method")?.value || "lofsc", per_octave: $("opt-octave") ? !!$("opt-octave").checked : true, mutual: !!$("mutual")?.checked,
      pg_search: $("pg-search") ? !!$("pg-search").checked : true, pg_corners: $("pg-corners")?.checked !== false,
      search_mode: searchMode(),
      sw_grid: int("sw-grid", 128), sw_nth: int("sw-nth", 90), sw_rot: num("sw-rot", 180), sw_ns: int("sw-ns", 20),
      sw_rot_range: $("sw-rot-mode")?.value === "range", sw_rot_lo: num("sw-rot-lo", -45), sw_rot_hi: num("sw-rot-hi", 45),
      sw_shear: !!$("sw-shear")?.checked, sw_shear_max: num("sw-shear-max", 0.1), sw_nshear: int("sw-nshear", 3),
      sw_aniso: !!$("sw-aniso")?.checked, sw_aniso_max: num("sw-aniso-max", 10) / 100, sw_naniso: int("sw-naniso", 3),
      sw_smin: num("sw-smin", 0.5), sw_smax: num("sw-smax", 2), sw_k: int("sw-k", 32), sw_fin: int("sw-fin", 4), sw_rel: !!$("sw-rel")?.checked,
      hho_auto: !!$("hho-auto")?.checked, hho_n: int("hho-n", 100), hho_t: int("hho-t", 20), hho_sg: num("hho-sg", 0.01), hho_sh: num("hho-sh", 0.01),
      hho_p: num("hho-p", 1e-4), hho_tx: num("peak-tx", 0.01), hho_th: num("peak-th", 0.01), hho_phi: num("hho-phi", 0),
      hho_seed: int("hho-seed", 1), hho_rel: !!$("hho-rel")?.checked,
    };
    const INT_KEYS = new Set(["n_octaves", "max_points", "n_orient", "n_rings", "n_scales", "pos_k"]);
    for (const [id, v, , , , , , key] of [...PG_DETECT, ...PG_ADV]) s[id] = INT_KEYS.has(key) ? int(`d-${id}`, v) : num(`d-${id}`, v);
    return s;
  }
  /** Run fn in order with the engine's other calls (operations and settings never interleave). */
  const zcDo = (fn) => zc._serial(fn);
  function setCpsNow() {
    const g = Math.max(2, Math.min(FFD_MAX, int("ffd-g", 4)));
    if (state.cps && state.cps.length === 2 * g * g) zc.setFfd(state.cps);
  }
  /** The page's settings and spline into the engine, queued behind any running operation. */
  function zcSync(extra = null) {
    return zcDo(() => {
      zc.configure(extra ? { ...appSettings(), ...extra } : appSettings());
      setCpsNow();
    });
  }
  /** The engine's keypoints (4 floats each) as the rows the views use: [x, y, score, scale]. */
  /** Keypoint rows x, y, score, scale, then the descriptor frame: direction (radians), radius (px). */
  function kpRows(a, fr) {
    const o = [];
    for (let i = 0, j = 0; i + 3 < a.length; i += 4, j += 2) o.push([a[i], a[i + 1], a[i + 2], a[i + 3], fr[j], fr[j + 1]]);
    return o;
  }
  /** An overlay from the engine as the pose record the drawing code takes. */
  function poseFromOverlay(ov) {
    return { ox: ov.ox, oy: ov.oy, ow: ov.ow, oh: ov.oh, Hc: ov.Hc, moving_np: ov.moving, fixed_np: ov.fixed, gen: ++poseGen };
  }
  /** Progress from engine operations: log lines, committed poses (the trail), search views. */
  function onZcEvent(ev) {
    if (ev.kind === "log") {
      log(ev.text);
      const m = /^kps \d+\/\d+\s+threshold ([\d.e+-]+) \/ ([\d.e+-]+)/.exec(ev.text);
      if (m && $("opt-nt")?.checked && $("d-nt")) $("d-nt").value = (+m[2]).toPrecision(3);
      return;
    }
    if (ev.kind === "trail" || ev.kind === "pose") {
      // The engine's pose; the overlay follows when the operation ends (calls are serialized).
      state.H0 = Float64Array.from(ev.H);
      zeroTan();
      if (ev.cps && $("ffd")?.checked && state.cps && ev.cps.length === state.cps.length) state.cps = Float32Array.from(ev.cps);
      writeH(state.H0);
      if (ev.kind === "trail") {
        recordTrail(ev.fwd, ev.label, true, invOn() && ev.inv != null ? { metric: "SMI", score2: ev.inv, metric2: "inverse" } : { metric: "SMI" });
      }
      return;
    }
    if (ev.kind === "sweep") {
      state.swarmHist = null;
      state.sweepView = {
        thetas: ev.thetas, sigmas: ev.sigmas, wrap: ev.wrap, best: Float32Array.from(ev.best),
        top: ev.top.map(([it, is]) => ({ it, is })), finalists: ev.finalists,
        result: ev.result ? { th: ev.result[0], s: ev.result[1] } : null,
        cands: (ev.cands || []).map((c) => ({ ...c, H: Float64Array.from(c.H) })), applied: ev.applied,
      };
      drawSwarm();
      return;
    }
    if (ev.kind === "swarm") {
      state.sweepView = null;
      if (!state.swarmHist) state.swarmHist = { rows: [], best: -Infinity };
      const row = ev.scores.map((v) => (v == null ? NaN : v));
      state.swarmHist.rows.push(row);
      for (const v of row) if (v > state.swarmHist.best) state.swarmHist.best = v;
      drawSwarm();
    }
  }

  const syncScore = () => {
    zcSync();
    const smiLike = /^SMI/.test($("metric").value);
    if ($("map-cal")) $("map-cal").disabled = !$("exact")?.checked || !smiLike;
    // Whitening applies to SMI only; E4 / λmax fix their own.
    if ($("exact-lab")) $("exact-lab").hidden = !smiLike;
    $("map-cal-lab")?.classList.toggle("field-dim", !!$("map-cal")?.disabled);
  };
  zc.onEvent(onZcEvent);
  log("WebGPU ready  " + $("adapter").textContent + "  " + (f16Avail ? "fp16 available" : "fp16 unavailable") + "  (zcmir engine)");
  $("cv-reg").addEventListener("contextmenu", (e) => e.preventDefault());

  onStageChange = () => {
    requestAnimationFrame(() => {
      showImages();
      drawTrail();
      drawCbar();
      if (state.pose) applyWarp(false);
      if (state.maps.shift) renderMaps();
      else if (state.left && state.right && mapDockOn()) refreshMap(false);
      if (popDockOn() && state.left && state.right) refreshTiles(currentH());
    });
  };

  let splitRaf = 0;
  function relayout() {
    if (splitRaf) return;
    splitRaf = requestAnimationFrame(() => {
      splitRaf = 0;
      showImages();
      drawTrail();
      drawCbar();
      if (state.pose) applyWarp(false);
      if (state.maps.shift) renderMaps();
      if (popDockOn() && state.left && state.right) refreshTiles(currentH());
      if (state.swarmHist) drawSwarm();
    });
  }
  bindSplitter($("split-pair"), { parent: $("pair"), prop: "--split-pair", axis: "y", min: 90, minOther: 90, ondrag: relayout });
  bindSplitter($("split-align"), {
    parent: $("view-main"), prop: "--split-align", min: 160, minOther: 206,
    ondrag: () => { state.userSplitAlign = true; relayout(); },
  });
  $("split-align")?.addEventListener("dblclick", () => {
    state.userSplitAlign = false;
    $("view-main").style.removeProperty("--split-align");
    state.mapViews.rs = null;
    relayout();
  });
  bindSplitter($("split-rail"), {
    parent: document.querySelector(".stagewrap"), prop: "--rail-w",
    axis: "x", fromEnd: true, min: 240, minOther: 280, unit: "px", ondrag: relayout,
  });
  $("btn-undo").onclick = () => {
    const s = state.undo.pop();
    if (s) { restorePose(s); applyFfd(); applyWarp(true); }
  };

  let warpTimer = 0, mapTimer = 0, flicker = true, flickRaf = 0, poseGen = 0;
  const scheduleWarp = () => { clearTimeout(warpTimer); warpTimer = setTimeout(() => applyWarp(false), 40); };
  const scheduleMap = () => { clearTimeout(mapTimer); mapTimer = setTimeout(() => refreshMap(true), 220); };
  let gpuBusy = false;
  let gpuPend = { warp: false, map: false, pin: false, trail: false };
  let gpuWait = [];
  function requestGpu(opts = {}) {
    if (opts.warp) gpuPend.warp = true;
    if (opts.map) gpuPend.map = true;
    if (opts.pin) gpuPend.pin = true;
    if (opts.trail) gpuPend.trail = true;
    const p = new Promise((resolve, reject) => gpuWait.push({ resolve, reject }));
    pumpGpu();
    return p;
  }
  async function pumpGpu() {
    if (gpuBusy) return;
    gpuBusy = true;
    try {
      while (gpuPend.warp || gpuPend.map) {
        const job = gpuPend;
        gpuPend = { warp: false, map: false, pin: false, trail: false };
        if (job.warp) await applyWarpUnlocked(job.trail);
        if (gpuPend.warp) continue;
        if (job.map || gpuPend.map) {
          const pin = job.pin || gpuPend.pin;
          gpuPend.map = false;
          gpuPend.pin = false;
          if (gpuPend.warp) continue;
          await refreshMapUnlocked(pin);
        }
      }
      const w = gpuWait.splice(0);
      for (const x of w) x.resolve();
    } catch (e) {
      const w = gpuWait.splice(0);
      for (const x of w) x.reject(e);
      log(e.stack || e);
    } finally {
      gpuBusy = false;
      if (gpuPend.warp || gpuPend.map) pumpGpu();
    }
  }
  const scheduleLiveMap = () => {
    if (!mapLiveOn() || !mapDockOn()) return;
    requestGpu({ map: true });
  };
  const flickPeriod = () => Math.max(40, Math.min(8000, +$("flick-ms").value || 2000));
  const flickerModes = new Set(["soft", "flicker"]);
  const checkerModes = new Set(["checker"]);
  const syncFlickerUi = () => {
    const mode = $("reg-mode").value;
    const flick = flickerModes.has(mode);
    const chk = checkerModes.has(mode);
    $("flick-ms").hidden = !flick;
    if ($("flick-unit")) $("flick-unit").hidden = !flick;
    if ($("chk-px")) $("chk-px").hidden = !chk;
    if ($("chk-unit")) $("chk-unit").hidden = !chk;
  };
  const armFlicker = () => {
    cancelAnimationFrame(flickRaf);
    syncFlickerUi();
    const mode0 = $("reg-mode").value;
    if (!flickerModes.has(mode0)) return;
    const tick = (now) => {
      flickRaf = requestAnimationFrame(tick);
      const mode = $("reg-mode").value;
      if (!state.pose || !flickerModes.has(mode)) return;
      if (mode === "flicker") flicker = ((now / flickPeriod()) | 0) % 2 === 0;
      paintReg(now);
    };
    flickRaf = requestAnimationFrame(tick);
  };
  armFlicker();
  $("flick-ms").onchange = armFlicker;
  if ($("chk-px")) {
    $("chk-px").oninput = () => {
      const t = +$("chk-px").value;
      if (!Number.isFinite(t) || t < 2) return;
      poseSpr.tile = -1;
      poseSprInv.tile = -1;
      if (state.pose) paintReg();
    };
    $("chk-px").onchange = () => {
      $("chk-px").value = String(checkerTile());
      poseSpr.tile = -1;
      poseSprInv.tile = -1;
      if (state.pose) paintReg();
    };
  }

  const detectBox = $("detect");
  for (const [k, v, lo, hi, st, tip, name] of DETECT) {
    detectBox.insertAdjacentHTML("beforeend", `<div><label title="${tip}">${name || k}</label><input type="number" id="d-${k}" value="${v}" min="${lo}" max="${hi}" step="${st}"></div>`);
  }
  const describeBox = $("describe");
  if (describeBox) for (const [k, v, lo, hi, st, tip, name] of DESCRIBE) {
    describeBox.insertAdjacentHTML("beforeend", `<div><label title="${tip}">${name || k}</label><input type="number" id="d-${k}" value="${v}" min="${lo}" max="${hi}" step="${st}"></div>`);
  }
  const matchBox = $("match");
  for (const [k, v, lo, hi, st, tip, name] of MATCH) {
    matchBox.insertAdjacentHTML("beforeend", `<div><label title="${tip}">${name || k}</label><input type="number" id="d-${k}" value="${v}" min="${lo}" max="${hi}" step="${st}"></div>`);
  }
  const fieldsInto = (box, list) => {
    if (box) for (const [k, v, lo, hi, st, tip, name] of list) {
      box.insertAdjacentHTML("beforeend", `<div><label title="${tip}">${name || k}</label><input type="number" id="d-${k}" value="${v}" min="${lo}" max="${hi}" step="${st}" title="${tip}"></div>`);
    }
  };
  fieldsInto($("pg-detect"), PG_DETECT);
  fieldsInto($("pg-adv"), PG_ADV);
  fieldsInto($("gls-adv"), GLS_ADV);
  function syncAutoNt() {
    const on = !!$("opt-nt")?.checked;
    const input = $("d-nt");
    if (!input) return;
    input.disabled = on;
    input.parentElement?.classList.toggle("field-dim", on);
  }
  $("opt-nt")?.addEventListener("change", syncAutoNt);
  syncAutoNt();
  function applyFfd() {
    const on = $("ffd").checked;
    const g = Math.max(2, Math.min(FFD_MAX, +$("ffd-g").value | 0));
    $("ffd-g").value = g;
    if (!state.cps || state.cpsGx !== g) {
      state.cps = new Float32Array(g * g * 2);
      state.cpsGx = g;
      state.cpsStart = Float32Array.from(state.cps);
      state.cpsGrad = null;
    } else if (on && (!state.cpsStart || state.cpsStart.length !== state.cps.length)) {
      state.cpsStart = Float32Array.from(state.cps);
    }
    if (zc) zcSync();
    // Remember the last change of the coefficients for the update arrows. A handle
    // drag counts once, from where the drag started.
    const applied = state._cpsApplied;
    if (!state._cpsDrag && applied && applied.length === state.cps.length && applied.some((v, i) => v !== state.cps[i])) {
      state.cpsStep = { from: applied, to: Float32Array.from(state.cps) };
    }
    if (!state._cpsDrag) state._cpsApplied = Float32Array.from(state.cps);
  }

  function ffdLattice(rw, rh, gx, gy) {
    const pts = [];
    for (let j = 0; j < gy; j++) for (let i = 0; i < gx; i++) {
      pts.push([(i * (rw - 1)) / Math.max(gx - 1, 1), (j * (rh - 1)) / Math.max(gy - 1, 1)]);
    }
    return pts;
  }

  async function showImages() {
    const vis = visMask();
    state._vis = vis;
    if (state.left) {
      state._viewL = fitDraw($("cv-left"), state.left.rgba, state.left.w, state.left.h);
      $("meta-left").textContent = `${state.left.w}×${state.left.h}`;
      paintKps($("cv-left"), state._viewL, state.kpsL, "L", vis.L);
    }
    if (state.right) {
      state._viewR = fitDraw($("cv-right"), state.right.rgba, state.right.w, state.right.h);
      $("meta-right").textContent = `${state.right.w}×${state.right.h}`;
      paintKps($("cv-right"), state._viewR, state.kpsR, "R", vis.R);
    }
    drawMatchLines(vis);
    if ($("kp-max-lbl")) $("kp-max-lbl").title = KP_MAX_TITLE + (vis.total ? ` Drawn: ${vis.shown} of ${vis.total}.` : "");
    if ($("kp-max-val") && $("kp-max")) $("kp-max-val").textContent = $("kp-max").value;
    drawKpPatch(state.kpSel);
  }

  /** A keypoint's descriptor patch: the disk it reads (kp[5] px), turned so the direction the
   *  descriptor starts from (kp[4]) points up, bilinear; the image outside the disk dimmed. */
  function blitPatch(cv, img, kp) {
    const rad = kp[5], th = kp[4];
    const c = Math.cos(th), s = Math.sin(th);
    const cx = kp[0] - 1, cy = kp[1] - 1;
    const ctx = cv.getContext("2d");
    const N = cv.width;
    const id = ctx.createImageData(N, N);
    const src = img.rgba, sw = img.w, sh = img.h;
    const at = (x, y, k) => src[(y * sw + x) * 4 + k];
    for (let y = 0; y < N; y++) {
      const py = ((y + 0.5) / N * 2 - 1) * rad;
      for (let x = 0; x < N; x++) {
        const px = ((x + 0.5) / N * 2 - 1) * rad;
        // patch up (0, −1) → (cos θ, sin θ); patch right → θ + 90°
        const ix = cx - px * s - py * c, iy = cy + px * c - py * s;
        const d = (y * N + x) * 4;
        const x0 = Math.floor(ix), y0 = Math.floor(iy);
        id.data[d + 3] = 255;
        if (x0 < 0 || y0 < 0 || x0 + 1 >= sw || y0 + 1 >= sh) continue;
        const fx = ix - x0, fy = iy - y0;
        const dim = px * px + py * py > rad * rad ? 0.35 : 1;
        for (let k = 0; k < 3; k++) {
          const top = at(x0, y0, k) * (1 - fx) + at(x0 + 1, y0, k) * fx;
          const bot = at(x0, y0 + 1, k) * (1 - fx) + at(x0 + 1, y0 + 1, k) * fx;
          id.data[d + k] = (top * (1 - fy) + bot * fy) * dim;
        }
      }
    }
    ctx.putImageData(id, 0, 0);
    ctx.strokeStyle = "rgba(230, 230, 232, .5)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(N / 2, N / 2, N / 2 - 0.5, 0, 2 * Math.PI); ctx.stroke();
    ctx.strokeStyle = "#c4a35a";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(N / 2, 1); ctx.lineTo(N / 2, 9); ctx.stroke();
  }

  /** The picked keypoint's patch and its match's, each in its descriptor's frame. */
  function drawKpPatch(sel) {
    const box = $("kp-patch");
    if (!box) return;
    if (!sel) { box.hidden = true; return; }
    const iL = sel.side === "L" ? sel.i : sel.pair;
    const iR = sel.side === "R" ? sel.i : sel.pair;
    const deg = (t) => `${Math.round((((t * 180 / Math.PI) % 360) + 540) % 360 - 180)}°`;
    const col = (key, img, kps, i, name) => {
      const kp = i >= 0 && img?.rgba ? kps?.[i] : null;
      $(`patch-col-${key}`).hidden = !kp;
      if (!kp) return false;
      blitPatch($(`cv-patch-${key}`), img, kp);
      $(`patch-sub-${key}`).textContent = `${name} #${i}
×${kpScale(kp)} · ${deg(kp[4])}`;
      return true;
    };
    const any = col("l", state.left, state.kpsL, iL, "moving") | col("r", state.right, state.kpsR, iR, "fixed");
    if (!any) { box.hidden = true; return; }
    $("patch-meta").textContent = iL < 0 || iR < 0 ? "unmatched" : state.inlier?.[iL] ? "inlier" : "outlier";
    box.hidden = false;
  }

  function invOn() { return !!$("sym")?.checked; }

  function showInv() { return !!$("inverse")?.checked; }

  /** Forward, inverse and mean pose score at H (the engine's pair score). */
  async function pairParts(H) {
    const s = await zc.score(H);
    return { fwd: s.fwd, inv: s.inv, mean: s.mean };
  }

  async function applyWarpUnlocked(trail) {
    if (!state.left || !state.right) return;
    // A pose change replaces any map preview (a redraw at the same pose keeps it).
    const base = state.preview?.base;
    if (base && currentH().some((v, i) => Math.abs(v - base[i]) > 1e-12)) {
      state.preview = null;
      state.mapMark = null;
      renderMaps();
    }
    try {
      const H = Float64Array.from(currentH());
      applyFfd();
      writeH(H);
      const ov = await zc.overlay(H);
      if (gpuPend.warp) return;
      const pose = poseFromOverlay(ov);
      state.pose = pose;
      let zInv = null;
      if (showInv()) {
        const inv = poseFromOverlay(await zc.overlay(H, { inverse: true }));
        if (gpuPend.warp) return;
        state.poseInv = inv;
      } else state.poseInv = null;
      paintReg();
      // The status line's nS is the forward score (NCC for NCC); the inverse and mean follow.
      const sc = await zc.score(H);
      if (gpuPend.warp) return;
      const z = sc.fwd;
      state._ns = z;
      if (invOn() && $("metric").value !== "NCC") zInv = { fwd: sc.fwd, inv: sc.inv, mean: sc.mean };
      const base = state._ns0;
      const d = base != null && base > 0 ? `  ${(z / base).toFixed(2)}×` : "";
      const invNote = zInv ? `  inverse ${fmtScore(zInv.inv)}  mean ${fmtScore(zInv.mean)}` : "";
      $("live").textContent = `${scoreName()} nS ${fmtScore(z)}${d}${invNote}`;
      const rp = $("res-pose");
      if (rp) {
        rp.textContent = base != null && base > 0 ? `${(z / base).toFixed(1)}×` : fmtScore(z);
        rp.title = `${scoreName()} ${fmtScore(z)}${base > 0 ? ` (${(z / base).toFixed(2)}× the identity's)` : ""}`;
      }
      $("sum-h").textContent = poseBrief(H);
      $("sum-h").title = summary(H);
      if (trail) {
        if (zInv) recordTrail(zInv.fwd, "SMI", false, { metric: "SMI", score2: zInv.inv, metric2: "inverse" });
        else recordTrail(z, $("metric").value, false, { metric: $("metric").value === "NCC" ? "NCC" : "dest nS" });
      }
    } catch (e) { log(e.stack || e); }
  }

  function applyWarp(doMap) {
    return requestGpu({ warp: true, map: !!doMap, pin: !!doMap, trail: !!doMap });
  }

  function paintReg(now = performance.now()) {
    const prev = state.preview;
    const poseFwd = prev ? prev.pose : state.pose;
    if (!poseFwd) return;
    const inv = !prev && !!(showInv() && state.poseInv);
    const pose = inv ? state.poseInv : poseFwd;
    const mode = $("reg-mode").value;
    const cv = $("cv-reg");
    const ctx = cv.getContext("2d");
    const dpr = devicePixelRatio || 1;
    const rw = cv.clientWidth, rh = cv.clientHeight;
    const H = prev ? prev.H : currentH();
    const view = inv
      ? overlayView(cv, pose, H, overlayBoxInv(pose, H))
      : overlayView(cv, pose, H);
    const tw = Math.max(1, (rw * dpr) | 0), th = Math.max(1, (rh * dpr) | 0);
    if (cv.width !== tw || cv.height !== th) { cv.width = tw; cv.height = th; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0c0c0c"; ctx.fillRect(0, 0, rw, rh);
    drawOverlay(ctx, pose, poseSprites(pose, prev ? previewSpr : inv ? poseSprInv : poseSpr), view, now);
    if (inv) {
      state._view = view;
      state._dst = [];
      state._cpsView = [];
      if (mapDockOn()) {
        const c = rsCenterDest(H);
        const at = mapPts(inv3(H), [c])[0];
        state._rsView = drawRsCenter(ctx, view, pose, H, at);
      } else state._rsView = null;
    } else drawHandles(view, pose, H);
    if (prev) drawPreviewNote(ctx, view, pose);
  }

  function drawOverlay(ctx, pose, spr, view, now) {
    const s = view.s, ox = view.ox, oy = view.oy;
    const dw = pose.ow * s, dh = pose.oh * s;
    const mode = $("reg-mode").value;
    ctx.imageSmoothingEnabled = false;
    if (mode === "soft") {
      const mix = softMix(now, flickPeriod());
      ctx.drawImage(spr.fix, ox, oy, dw, dh);
      ctx.globalAlpha = mix;
      ctx.drawImage(spr.mov, ox, oy, dw, dh);
      ctx.globalAlpha = 1;
    } else if (mode === "flicker") {
      ctx.drawImage(flicker ? spr.mov : spr.fix, ox, oy, dw, dh);
    } else if (mode === "blend") {
      ctx.drawImage(spr.fix, ox, oy, dw, dh);
      ctx.globalAlpha = 0.45;
      ctx.drawImage(spr.mov, ox, oy, dw, dh);
      ctx.globalAlpha = 1;
    } else if (mode === "checker") {
      ctx.drawImage(spr.chk, ox, oy, dw, dh);
    } else if (mode === "fixed") {
      ctx.drawImage(spr.fix, ox, oy, dw, dh);
    } else {
      ctx.drawImage(spr.mov, ox, oy, dw, dh);
    }
  }

  function drawHandles(view, pose, H) {
    const src = movingSrcCorners(state.left.w, state.left.h);
    const dst = mapPts(H, src).map(([x, y]) => [x - 1 - pose.ox, y - 1 - pose.oy]);
    const ctx = $("cv-reg").getContext("2d");
    const dpr = devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.beginPath();
    dst.forEach(([x, y], i) => {
      const px = view.ox + x * view.s, py = view.oy + y * view.s;
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.closePath();
    ctx.strokeStyle = "rgba(0,0,0,0.75)"; ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.strokeStyle = "#6a9ec8"; ctx.lineWidth = 1;
    ctx.stroke();
    dst.forEach(([x, y]) => {
      const px = view.ox + x * view.s, py = view.oy + y * view.s;
      ctx.beginPath(); ctx.arc(px, py, 3.5, 0, 6.28);
      ctx.fillStyle = "#dcdcdc"; ctx.fill();
      ctx.strokeStyle = "#0c0c0c"; ctx.lineWidth = 1; ctx.stroke();
    });
      const cps = [];
    if ($("ffd").checked && state.cps) {
      const gx = state.cpsGx, gy = gx;
      const srcF = ffdSource();
      const lw = srcF ? state.left.w : state.right.w;
      const lh = srcF ? state.left.h : state.right.h;
      const lat = ffdLattice(lw, lh, gx, gy);
      const [sx, sy] = destHalf(lw, lh);
      const mode = $("ffd-view")?.value || "grid";
      const arrows = $("ffd-arrows")?.value || "update";
      const toDest = (fx, fy) => {
        if (!srcF) return [fx - pose.ox, fy - pose.oy];
        const [X, Y] = mapPts(H, [[fx + 1, fy + 1]])[0];
        return [X - 1 - pose.ox, Y - 1 - pose.oy];
      };
      const toView = ([x, y]) => [view.ox + (x - pose.ox) * view.s, view.oy + (y - pose.oy) * view.s];
      if (mode !== "lattice") drawFfdField(ctx, view, pose, H, srcF, arrows === "update", mode === "jac", toView);
      const start = mode === "lattice" && state.cpsStart && state.cpsStart.length === state.cps.length ? state.cpsStart : null;
      lat.forEach(([fx, fy], i) => {
        // Control-point handles (coefficients, which a cubic B-spline does not interpolate).
        const [dx, dy] = toDest(fx + state.cps[i * 2] * sx, fy + state.cps[i * 2 + 1] * sy);
        cps.push([dx, dy]);
        const px = view.ox + dx * view.s, py = view.oy + dy * view.s;
        if (start) {
          const [rx, ry] = toDest(fx + start[i * 2] * sx, fy + start[i * 2 + 1] * sy);
          ctx.beginPath(); ctx.arc(view.ox + rx * view.s, view.oy + ry * view.s, 5, 0, 6.28);
          ctx.strokeStyle = "#e02020"; ctx.lineWidth = 1.6; ctx.stroke();
        }
        ctx.beginPath(); ctx.arc(px, py, mode === "lattice" ? 3 : 2.4, 0, 6.28);
        ctx.fillStyle = mode === "lattice" ? "#e02020" : "rgba(224,32,32,0.85)"; ctx.fill();
      });
      const gvec = state.cpsGrad;
      if (arrows === "grad" && gvec && gvec.length === state.cps.length) {
        const dir = cps.map((_, i) => {
          if (!srcF) return [gvec[i * 2] * sx, gvec[i * 2 + 1] * sy];
          const [fx, fy] = lat[i];
          const a = toDest(fx, fy);
          const b = toDest(fx + gvec[i * 2] * sx, fy + gvec[i * 2 + 1] * sy);
          return [b[0] - a[0], b[1] - a[1]];
        });
        drawLogVecs(ctx, view, cps, dir, "#00e040", 28);
      }
    }
    state._rsView = drawRsCenter(ctx, view, pose, H);
    ctx.restore();
    state._view = view; state._dst = dst; state._cpsView = cps;
  }

  /**
   * Deformed grid of the real B-spline displacement (exaggerated so the largest
   * displacement spans a quarter cell), optional det J fill, arrows of the last
   * coefficient update, and a legend with the true magnitudes.
   */
  function drawFfdField(ctx, view, pose, H, srcF, withUpdate, withJac, toView) {
    const step = withUpdate ? state.cpsStep : null;
    const prev = step && step.to.length === state.cps.length && step.to.every((v, i) => v === state.cps[i]) ? step.from : null;
    const hsh = (a) => { let h = 0; if (a) for (let i = 0; i < a.length; i++) h = (h * 31 + a[i] * 1e6) % 1e12; return h; };
    const key = [srcF, state.cpsGx, Array.from(H).join(","), hsh(state.cps), hsh(prev), state.left.w, state.left.h, state.right.w, state.right.h].join("|");
    if (!state._ffdVis || state._ffdVis.key !== key) {
      state._ffdVis = {
        key,
        geo: ffdOverlay({
          cps: state.cps, prev, gx: state.cpsGx, gy: state.cpsGx, frame: srcF ? "source" : "target", H,
          lw: state.left.w, lh: state.left.h, rw: state.right.w, rh: state.right.h, exag: "auto",
        }),
      };
    }
    const g = state._ffdVis.geo;
    const poly = (pts) => {
      ctx.beginPath();
      pts.forEach((p, i) => { const [x, y] = toView(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    };
    if (withJac) {
      for (const c of g.cells) {
        poly(c.poly);
        ctx.closePath();
        ctx.fillStyle = detColor(c.det, g.maxLog);
        ctx.fill();
      }
    }
    ctx.lineJoin = "round";
    for (const [w, col] of [[2.4, "rgba(0,0,0,0.55)"], [1, "rgba(255,206,84,0.9)"]]) {
      ctx.lineWidth = w;
      ctx.strokeStyle = col;
      for (const l of g.lines) { poly(l); ctx.stroke(); }
    }
    if (g.arrows.length && g.maxUpd > 1e-6) {
      ctx.strokeStyle = ctx.fillStyle = "#3ee07a";
      ctx.lineWidth = 1.5;
      for (const a of g.arrows) {
        if (Math.hypot(a.ux, a.uy) < 0.02 * g.maxUpd) continue;
        const [x0, y0] = toView(a.from), [x1, y1] = toView(a.to);
        drawArrow(ctx, x0, y0, x1, y1);
      }
    }
    const fmt = (v) => (v >= 10 ? v.toFixed(1) : v.toFixed(2));
    const parts = [`B-spline  max |d| ${fmt(g.maxD)} px  ·  drawn ×${fmt(g.k)}`,
      `det J ${g.minDet.toFixed(3)} … ${g.maxDet.toFixed(3)}${g.minDet <= 0 ? "  FOLDED" : ""}`];
    if (g.arrows.length && g.maxUpd > 1e-6) parts.push(`last update max ${fmt(g.maxUpd)} px ×${fmt(g.kUpd)}`);
    const cv = ctx.canvas, dpr = devicePixelRatio || 1;
    const yb = cv.height / dpr - 8;
    ctx.font = "11px ui-monospace, Consolas, monospace";
    ctx.textBaseline = "bottom";
    ctx.textAlign = "left";
    parts.forEach((t, i) => {
      const y = yb - (parts.length - 1 - i) * 14;
      ctx.fillStyle = "rgba(12,12,12,0.75)";
      ctx.fillRect(6, y - 13, ctx.measureText(t).width + 8, 14);
      ctx.fillStyle = i === 1 && g.minDet <= 0 ? "#ff5cff" : i === 2 ? "#3ee07a" : "#ffce54";
      ctx.fillText(t, 10, y);
    });
  }

  // The roto-scale map is made at the committed pose, so its centre is that pose's even while the
  // overlay shows a map preview; `pose` / `view` are the frame the overlay is drawn in (dragging uses them).
  function drawRsCenter(ctx, view, pose, H, at) {
    if (!mapDockOn()) return null;
    if (!state.left) return null;
    const c = at || rsCenterDest(currentH());
    const dx = c[0] - 1 - pose.ox, dy = c[1] - 1 - pose.oy;
    const px = view.ox + dx * view.s, py = view.oy + dy * view.s;
    ctx.save();
    ctx.fillStyle = "#3ec6c2";
    ctx.strokeStyle = "#3ec6c2";
    ctx.lineWidth = 2.8;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.arc(px, py, 4.2, 0, Math.PI * 2);
    ctx.fill();
    const r = 16, a0 = -0.35, a1 = Math.PI * 1.28, gap = 0.42;
    ctx.beginPath();
    ctx.arc(px, py, r, a0, a1 - gap);
    ctx.stroke();
    const tip = a1 - gap * 0.15;
    const hx = px + r * Math.cos(tip), hy = py + r * Math.sin(tip);
    const ux = -Math.sin(tip), uy = Math.cos(tip);
    const nx = -uy, ny = ux;
    ctx.beginPath();
    ctx.moveTo(hx + ux * 7, hy + uy * 7);
    ctx.lineTo(hx - ux * 3.2 - nx * 4.2, hy - uy * 3.2 - ny * 4.2);
    ctx.lineTo(hx - ux * 3.2 + nx * 4.2, hy - uy * 3.2 + ny * 4.2);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    return { px, py, dest: c, pose, view };
  }

  function rsCenterHit(x, y, which = state._rsView) {
    if (!which || !mapDockOn()) return false;
    return Math.hypot(which.px - x, which.py - y) < 22;
  }

  function showRsTip(on, x, y) {
    const tip = $("handle-tip");
    if (!tip) return;
    if (!on) { tip.hidden = true; return; }
    tip.hidden = false;
    tip.textContent = "roto-scale center · drag to move the log-polar origin";
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }

  $("cv-reg").addEventListener("pointermove", (ev) => {
    if (!state.pose || !state._view) return;
    const r = $("cv-reg").getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    const hit = rsCenterHit(x, y);
    $("cv-reg").classList.toggle("rs-over", hit);
    showRsTip(hit, x, y);
  });
  $("cv-reg").addEventListener("pointerleave", () => {
    $("cv-reg")?.classList.remove("rs-over");
    showRsTip(false);
  });

  $("cv-reg").addEventListener("pointerdown", (ev) => {
    // Editing corners starts from the committed pose; panning keeps the preview.
    if (state.preview && tweakOn()) { clearPreview(); return; }
    if (!state._dst || !state._view || !state.pose) return;
    $("cv-reg").focus();
    const r = $("cv-reg").getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    if (rsCenterHit(x, y)) {
      ev.preventDefault();
      // A roto-scale selection is a rotation about the old centre; a shift selection stays valid.
      if (state.preview?.kind === "rs") clearPreview();
      const move = (e) => {
        const xx = e.clientX - r.left, yy = e.clientY - r.top;
        // Pointer → canvas in the frame the handle was drawn in (the preview's while one is shown).
        const rv = state._rsView;
        const v = rv?.view || state._view, p = rv?.pose || state.pose;
        const ax = (xx - v.ox) / v.s + 1 + p.ox;
        const ay = (yy - v.oy) / v.s + 1 + p.oy;
        if (showInv() && state.poseInv && !state.preview) {
          const [fx, fy] = mapPts(currentH(), [[ax, ay]])[0];
          state.rsCenter = [fx, fy];
        } else {
          state.rsCenter = [ax, ay];
        }
        paintReg();
        showRsTip(true, xx, yy);
        scheduleLiveMap();
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        showRsTip(false);
        refreshMap(true);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
      return;
    }
    if (!tweakOn()) {
      const startX = x, startY = y;
      const ox0 = state.regView.x, oy0 = state.regView.y;
      const move = (e) => {
        const xx = e.clientX - r.left, yy = e.clientY - r.top;
        state.regView.x = ox0 + (xx - startX);
        state.regView.y = oy0 + (yy - startY);
        paintReg();
      };
      const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
      return;
    }
    const hitNear = (pts, rad) => {
      let hit = -1, best = rad;
      pts.forEach(([dx, dy], i) => {
        const px = state._view.ox + dx * state._view.s, py = state._view.oy + dy * state._view.s;
        const d = Math.hypot(px - x, py - y);
        if (d < best) { best = d; hit = i; }
      });
      return hit;
    };
    const cpHit = state._cpsView?.length ? hitNear(state._cpsView, 10) : -1;
    const hit = cpHit < 0 ? hitNear(state._dst, 12) : -1;
    pushUndo();
    if (cpHit >= 0) {
      const srcF = ffdSource();
      const lw = srcF ? state.left.w : state.right.w;
      const lh = srcF ? state.left.h : state.right.h;
      state._cpsDrag = Float32Array.from(state.cps);
      const move = (e) => {
        const xx = e.clientX - r.left, yy = e.clientY - r.top;
        let ax = (xx - state._view.ox) / state._view.s + state.pose.ox;
        let ay = (yy - state._view.oy) / state._view.s + state.pose.oy;
        if (srcF) {
          const [sx, sy] = mapPts(inv3(currentH()), [[ax + 1, ay + 1]])[0];
          ax = sx - 1; ay = sy - 1;
        }
        const gx = state.cpsGx;
        const lat = ffdLattice(lw, lh, gx, gx);
        const [sx, sy] = destHalf(lw, lh);
        state.cps[cpHit * 2] = (ax - lat[cpHit][0]) / sx;
        state.cps[cpHit * 2 + 1] = (ay - lat[cpHit][1]) / sy;
        applyFfd();
        paintReg();
        applyWarp(false);
        scheduleLiveMap();
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        const from = state._cpsDrag;
        state._cpsDrag = null;
        if (from && from.some((v, i) => v !== state.cps[i])) state.cpsStep = { from, to: Float32Array.from(state.cps) };
        state._cpsApplied = Float32Array.from(state.cps);
        applyWarp(true);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
      return;
    }
    if (hit >= 0) {
      const move = (e) => {
        const xx = e.clientX - r.left, yy = e.clientY - r.top;
        const ax = (xx - state._view.ox) / state._view.s + 1 + state.pose.ox;
        const ay = (yy - state._view.oy) / state._view.s + 1 + state.pose.oy;
        const src = movingSrcCorners(state.left.w, state.left.h);
        const cur = mapPts(currentH(), src);
        const nxt = cur.map((p) => p.slice());
        nxt[hit] = [ax, ay];
        const dest = state.group === "affine" ? parallelogramMove(cur, hit, [ax, ay]) : nxt;
        state.H0 = state.group === "affine" ? HAffineFromPts(src, dest) : HHomographyFromPts(src, dest);
        zeroTan();
        paintReg();
        applyWarp(false);
        scheduleLiveMap();
      };
      const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); applyWarp(true); };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
      return;
    }
    let lastX = x, lastY = y;
    const move = (e) => {
      const xx = e.clientX - r.left, yy = e.clientY - r.top;
      const dx = (xx - lastX) / state._view.s, dy = (yy - lastY) / state._view.s;
      lastX = xx; lastY = yy;
      applyNudge(state.tan, dragNudge(dx, dy, e, state.group));
      syncTan();
      paintReg();
      applyWarp(false);
      scheduleLiveMap();
    };
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); applyWarp(true); };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });

  $("cv-reg").addEventListener("wheel", (ev) => {
    if (!state.left || !state.right) return;
    ev.preventDefault();
    if (!tweakOn()) {
      const showing = showInv() && state.poseInv;
      const pose = showing ? state.poseInv : state.pose;
      if (!pose) return;
      const r = $("cv-reg").getBoundingClientRect();
      const mx = ev.clientX - r.left, my = ev.clientY - r.top;
      const rv = state.regView;
      const box = showing ? overlayBoxInv(pose, currentH()) : overlayBox(pose, currentH());
      const sFit = Math.min(r.width / Math.max(box.w, 1), r.height / Math.max(box.h, 1));
      const s0 = sFit * rv.z;
      const ox0 = (r.width - box.w * s0) / 2 + rv.x - box.x0 * s0;
      const oy0 = (r.height - box.h * s0) / 2 + rv.y - box.y0 * s0;
      const ix = (mx - ox0) / s0, iy = (my - oy0) / s0;
      rv.z = Math.min(16, Math.max(0.2, rv.z * (ev.deltaY < 0 ? 1.12 : 1 / 1.12)));
      const s1 = sFit * rv.z;
      rv.x = mx - (ix - box.x0) * s1 - (r.width - box.w * s1) / 2;
      rv.y = my - (iy - box.y0) * s1 - (r.height - box.h * s1) / 2;
      paintReg();
      return;
    }
    const d = wheelNudge(ev);
    if (!d) return;
    pushUndo();
    applyNudge(state.tan, d);
    syncTan();
    scheduleWarp();
    scheduleMap();
  }, { passive: false });

  $("cv-reg").addEventListener("dblclick", (ev) => {
    if (tweakOn()) return;
    ev.preventDefault();
    state.regView = { z: 1, x: 0, y: 0 };
    if (state.pose) paintReg();
  });

  window.addEventListener("keydown", (ev) => {
    if (typingTarget(ev.target)) return;
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "z") {
      ev.preventDefault();
      const s = state.undo.pop();
      if (s) { restorePose(s); applyFfd(); applyWarp(true); }
      return;
    }
    if (ev.key === "?" || (ev.shiftKey && ev.key === "/")) {
      toggleLog(true);
      log(HELP);
      return;
    }
    const d = tweakOn() ? keyNudge(ev, state.group) : null;
    if (!d) return;
    ev.preventDefault();
    pushUndo();
    applyNudge(state.tan, d);
    syncTan();
    scheduleWarp();
    scheduleMap();
  });

  function scoreName() {
    const m = $("metric").value;
    return m !== "NCC" && $("exact")?.checked ? `${m} exact` : m;
  }

  function fmtScore(v) {
    if (Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.001)) return v.toExponential(3);
    return v.toFixed(4);
  }

  function rsCaption(m) {
    return `${m.symmetric ? "sym " : ""}peak${m.calib ? " z" : ""} ${fmtScore(m.peak)}  Δθ ${((m.dth || 0) * 180 / Math.PI).toFixed(1)}°  ×${Math.exp(m.dsg || 0).toFixed(3)}`;
  }

  // Both maps about the current pose, the shift map first (it also sets the status line).
  async function refreshMapUnlocked(pin) {
    if (!state.left || !state.right) return;
    try {
      const t0 = performance.now();
      const name = $("metric").value;
      const minOv = +$("min-ov").value;
      const H = Float64Array.from(currentH());
      const lw = state.left.w, lh = state.left.h, rw = state.right.w, rh = state.right.h;
      await zcSync();
      const res = Math.max(1024, int("map-res", 1024));
      const m = await zc.shiftMap(H, { resolution: res });
      if (gpuPend.warp) return;
      const sh = {
        n: m.n, ny: m.ny || m.n, cw: m.cw, ch: m.ch, aw: m.canvasW, ah: m.canvasH, ox: m.ox, oy: m.oy, dx: m.dx, dy: m.dy,
        peak: m.peak, zero: m.zero, calib: m.calibrated, area: m.corrAreaPx, H,
      };
      if (name === "NCC") sh.ncc = m.map; else sh.score = m.map;
      state.maps.shift = sh;
      renderMap("shift");
      // Symmetric: the engine also computes the map in the moving frame (the fixed image turned
      // by H⁻¹ about H⁻¹c) and averages the two at the mirrored lag.
      const r = await zc.rsMap(H, { center: rsCenterDest(H), symmetric: invOn(), returnMaps: true, cap: res });
      if (gpuPend.warp) return;
      const lagOf = (i, n) => { const py = (i / n) | 0, px = i % n; return [px <= n / 2 ? px : px - n, py <= n / 2 ? py : py - n]; };
      const [ldx, ldy] = lagOf(r.peakIndex, r.n);
      const rs = {
        kind: "rs", score: r.map, n: r.n, nTh: r.nTh, nLam: r.nLam, dlam: r.dlam, r0: r.r0, r1: r.r1, dth: r.dth, dsg: r.dsg,
        dx: ldx, dy: ldy, peak: r.peak, zero: r.zero, calib: r.calibrated, symmetric: r.symmetric, cDest: [r.cx, r.cy], H,
      };
      state.maps.rs = rs;
      renderMap("rs");
      const zero = fmtScore(state._ns ?? sh.zero);
      $("live").textContent = `${scoreName()}  nS ${zero}   shift peak${sh.calib ? " z" : ""} ${fmtScore(sh.peak)} at (${sh.dx.toFixed(1)}, ${sh.dy.toFixed(1)})   roto-scale peak${rs.calib ? " z" : ""} ${fmtScore(rs.peak)}`;
      state._mapCap = `peak ${fmtScore(sh.peak)}  (${sh.dx.toFixed(1)}, ${sh.dy.toFixed(1)}) px`;
      state._rsCap = rsCaption(rs);
      $("map-meta").textContent = state._mapCap;
      $("rs-meta").textContent = state._rsCap;
      const cal = sh.calib ? `  calibrated z, correlation area ${sh.area.toFixed(1)} cells` : "";
      if (pin) log(`${name} maps  shift ${sh.n}² peak=${sh.peak.toFixed(4)} d=(${sh.dx.toFixed(1)},${sh.dy.toFixed(1)})${cal}  roto-scale ${rs.n}² peak=${rs.peak.toFixed(4)} ${rsCaption(rs)}  ${summary(H)}  ${((performance.now() - t0) / 1000).toFixed(2)}s`);
    } catch (e) { log(e.stack || e); }
  }

  function refreshMap(pin) {
    return requestGpu({ map: true, pin: !!pin });
  }

  // A map cell under the pointer → the lag it stands for and the pose it would give.
  function mapPick(kind, ev, snap = false) {
    const d = state.mapDraw[kind], out = state.maps[kind];
    if (!d || !out) return null;
    const r = $(MAP_PANES[kind].cv).getBoundingClientRect();
    const px = ev.clientX - r.left, py = ev.clientY - r.top;
    const f = d.frame;
    if (px < f.x || py < f.y || px > f.x + f.w || py > f.y + f.h) return null;
    let ix = Math.floor(mapIx(kind, d.crop, (px - d.v.ox) / d.v.sx));
    let iy = Math.floor((py - d.v.oy) / d.v.sy + d.crop.y0);
    if (ix < d.crop.x0 || iy < d.crop.y0 || ix >= d.crop.x0 + d.crop.w || iy >= d.crop.y0 + d.crop.h) return null;
    const n = out.n, ny = out.ny || n, h = n >> 1, hy = ny >> 1;
    const field = out.score || out.ncc;
    // Snap to the best cell within ~4 screen px (cells can be far smaller than a pixel);
    // Shift picks the exact cell.
    if (snap && field && !ev.shiftKey) {
      const rx = Math.min(24, Math.ceil(4 / d.v.sx)), ry = Math.min(24, Math.ceil(4 / d.v.sy));
      let best = -Infinity, bx = ix, by = iy;
      for (let y = Math.max(d.crop.y0, iy - ry); y <= Math.min(d.crop.y0 + d.crop.h - 1, iy + ry); y++) {
        for (let x = Math.max(d.crop.x0, ix - rx); x <= Math.min(d.crop.x0 + d.crop.w - 1, ix + rx); x++) {
          const v = field[((y + hy) % ny) * n + ((x + h) % n)];
          if (v > best) { best = v; bx = x; by = y; }
        }
      }
      ix = bx; iy = by;
    }
    const s = { kind, ix, iy, sx: ix - h, sy: iy - hy, score: field ? field[((iy + hy) % ny) * n + ((ix + h) % n)] : NaN };
    const H = out.H || currentH();
    if (kind === "rs") {
      const nTh = out.nTh || n;
      const dlam = out.dlam || (out.logSpan || 1) / Math.max(1, out.nLam || h);
      Object.assign(s, rsCorrToSim(s.sx, s.sy, nTh, dlam));
      s.H = composeSimC(H, out.cDest || rsCenterDest(H), s.dth, s.dsg, state.group);
      s.label = `Δθ ${(s.dth * 180 / Math.PI).toFixed(2)}°  ×${Math.exp(s.dsg).toFixed(4)}`;
    } else {
      // Lag in fixed-image pixels; the map peak is the moving image displaced by +d.
      s.dx = s.sx * ((out.aw || state.right.w) / out.cw);
      s.dy = s.sy * ((out.ah || state.right.h) / out.ch);
      s.H = mul3(new Float64Array([1, 0, -s.dx, 0, 1, -s.dy, 0, 0, 1]), H);
      s.label = `shift (${(-s.dx).toFixed(1)}, ${(-s.dy).toFixed(1)}) px`;
    }
    return s;
  }

  // Preview: the overlay shows the pose a map cell stands for; nothing is committed.
  let previewSeq = 0;
  async function showPreview(s) {
    if (!s || !state.left || !state.right) return;
    const seq = ++previewSeq;
    state.mapMark = { kind: s.kind, ix: s.ix, iy: s.iy };
    renderMaps();
    await requestGpu({});
    if (seq !== previewSeq) return;
    const pose = poseFromOverlay(await zc.overlay(s.H));
    const now = (await zc.score(s.H)).fwd;
    const then = (await zc.score(currentH())).fwd;
    if (seq !== previewSeq) return;
    state.preview = { H: Float64Array.from(s.H), base: Float64Array.from(currentH()), pose, label: s.label, score: now, current: then, kind: s.kind };
    paintReg();
  }

  // Cancel: drop the preview and put the committed pose back on the overlay.
  function clearPreview() {
    const had = !!(state.preview || state.mapMark);
    previewSeq++;
    state.preview = null;
    state.mapMark = null;
    if (!had) return;
    renderMaps();
    if (state.pose) applyWarp(false);
  }

  async function commitPose(H) {
    previewSeq++;
    state.preview = null;
    state.mapMark = null;
    pushUndo();
    state.H0 = Float64Array.from(H);
    zeroTan();
    await applyWarp(true);
  }

  // Pointer, wheel and keys on both map panes.
  for (const kind of ["shift", "rs"]) {
    const P = MAP_PANES[kind];
    const cv = $(P.cv);
    if (!cv) continue;
    let drag = null, dragged = false;
    const view = () => state.mapViews[kind] || state.mapDraw[kind]?.fit || null;
    cv.addEventListener("pointerdown", (ev) => {
      if (ev.button !== 0) return;
      const v = view();
      if (!v) return;
      drag = { x: ev.clientX, y: ev.clientY, v: { ...v }, id: ev.pointerId };
      dragged = false;
    });
    cv.addEventListener("pointermove", (ev) => {
      if (drag && ev.pointerId === drag.id && (ev.buttons & 1)) {
        const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
        if (!dragged && Math.hypot(dx, dy) > 3) { dragged = true; cv.setPointerCapture(ev.pointerId); cv.classList.add("panning"); }
        if (dragged) {
          const d = state.mapDraw[kind];
          if (state.mapViews[kind] && d) {
            state.mapViews[kind] = clampMapView({ ...drag.v, ox: drag.v.ox + dx, oy: drag.v.oy + dy }, d.frame, mapVis(state.maps[kind]));
            renderMap(kind);
          }
          return;
        }
      }
      const s = mapPick(kind, ev);
      if (!s) return;
      $(P.meta).textContent = `${s.label}  ·  ${fmtScore(s.score)}`;
    });
    const endDrag = (ev) => {
      if (drag && dragged) { try { cv.releasePointerCapture(ev.pointerId); } catch { /* not captured */ } }
      cv.classList.remove("panning");
      drag = null;
    };
    cv.addEventListener("pointerup", endDrag);
    cv.addEventListener("pointercancel", endDrag);
    cv.addEventListener("pointerleave", () => {
      $(P.meta).textContent = (kind === "rs" ? state._rsCap : state._mapCap) || "";
    });
    cv.addEventListener("click", (ev) => {
      if (dragged) { dragged = false; return; }
      showPreview(mapPick(kind, ev, true));
    });
    cv.addEventListener("dblclick", (ev) => {
      const s = mapPick(kind, ev, true);
      if (s) commitPose(s.H);
    });
    cv.addEventListener("contextmenu", (ev) => { ev.preventDefault(); clearPreview(); });
    cv.addEventListener("wheel", (ev) => {
      const d = state.mapDraw[kind];
      if (!d) return;
      ev.preventDefault();
      const r = cv.getBoundingClientRect();
      const mx = ev.clientX - r.left, my = ev.clientY - r.top;
      const v = view();
      const f = Math.exp(-ev.deltaY * (ev.deltaMode === 1 ? 0.05 : 0.0015));
      // zooming out stops at the fit view (the map filling its pane), never smaller
      const zoom = Math.min(64, Math.max(1, (v.sx * f) / d.fit.sx));
      if (zoom <= 1.0001) state.mapViews[kind] = null;
      else {
        const g = (zoom * d.fit.sx) / v.sx;
        state.mapViews[kind] = clampMapView({ sx: v.sx * g, sy: v.sy * g, ox: mx - (mx - v.ox) * g, oy: my - (my - v.oy) * g }, d.frame, mapVis(state.maps[kind]));
      }
      renderMap(kind);
    }, { passive: false });
    $(kind === "rs" ? "rs-fit" : "map-fit").onclick = () => { state.mapViews[kind] = null; renderMap(kind); };
  }
  window.addEventListener("keydown", (ev) => {
    if (typingTarget(ev.target) || !state.preview) return;
    if (ev.key === "Escape") clearPreview();
    else if (ev.key === "Enter") { ev.preventDefault(); commitPose(state.preview.H); }
  });

  // Preview note on the overlay: the committed footprint dashed, the scores, the keys.
  function drawPreviewNote(ctx, view, pose) {
    const p = state.preview;
    const src = movingSrcCorners(state.left.w, state.left.h);
    const dst = mapPts(currentH(), src).map(([x, y]) => [view.ox + (x - 1 - pose.ox) * view.s, view.oy + (y - 1 - pose.oy) * view.s]);
    const dpr = devicePixelRatio || 1;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = "rgba(230,230,230,0.8)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    dst.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);
    const better = p.score > p.current;
    const lines = [
      `preview  ${p.label}`,
      `nS ${fmtScore(p.score)}  (now ${fmtScore(p.current)}, ${better ? "+" : ""}${fmtScore(p.score - p.current)})`,
      "double-click or Enter: apply   right-click or Esc: cancel",
    ];
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 16;
    ctx.fillStyle = "rgba(12,12,12,0.82)";
    ctx.fillRect(8, 8, w, lines.length * 17 + 8);
    ctx.strokeStyle = "#e6c36a";
    ctx.strokeRect(8.5, 8.5, w - 1, lines.length * 17 + 7);
    ctx.textBaseline = "top";
    lines.forEach((l, i) => {
      ctx.fillStyle = i === 0 ? "#e6c36a" : i === 1 ? (better ? "#7dba6a" : "#d98a7a") : "#9a9a9a";
      ctx.fillText(l, 16, 13 + i * 17);
    });
    ctx.restore();
  }
  function grayToRgba(g) {
    const n = g.length;
    const rgba = new Uint8ClampedArray(n * 4);
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      const v = g[i] <= 0 ? 0 : g[i] >= 1 ? 255 : (g[i] * 255 + 0.5) | 0;
      rgba[j] = rgba[j + 1] = rgba[j + 2] = v;
      rgba[j + 3] = 255;
    }
    return rgba;
  }

  // The engine preprocesses each image per the settings (CLAHE, band pass, invert); the pane
  // shows the result.
  async function bakeSide(which) {
    const img = state[which];
    if (!img) return;
    await zcSync();
    img.rgba = grayToRgba(await zc.workImage(which === "left" ? 0 : 1, img.w, img.h));
  }

  async function rebakeAll() {
    if (state.left) await bakeSide("left");
    if (state.right) await bakeSide("right");
    await showImages();
    if (state.left && state.right) await applyWarp(true);
  }

  function syncBand() {
    const on = !!$("band")?.checked;
    if ($("bp-fine")) $("bp-fine").disabled = !on;
    if ($("bp-coarse")) $("bp-coarse").disabled = !on;
  }

  async function ingest(which, img) {
    const rgba = img.srcRgba || img.rgba;
    await zcSync();
    await zc.setImage(which === "left" ? 0 : 1, rgba, img.w, img.h);
    state[which] = { ...img, rgba, srcRgba: rgba };
    const fname = $(which === "left" ? "fname-left" : "fname-right");
    if (fname) {
      fname.textContent = `${img.name || "image"} · ${img.w}×${img.h}`;
      fname.title = fname.textContent;
      fname.classList.add("set");
    }
    $("app").dataset.images = String((state.left ? 1 : 0) + (state.right ? 1 : 0));
    $("app").dataset[which] = "1";
    await bakeSide(which);
    state._ns0 = null;
    state.rsCenter = null;
    state.poseClimbed = false;
    markNext();
    const note = [$("clahe").checked ? "CLAHE" : "", $("band")?.checked ? "band" : "", $("invert")?.checked ? "invert" : ""].filter(Boolean).join(" ");
    log(`loaded ${which} ${img.w}×${img.h}${note ? "  " + note : ""}`);
    await showImages();
    if (state.left && state.right) {
      state._ns0 = (await zc.score(identityH())).fwd;
      await applyWarp(true);
    }
  }


  function writeMatchStats(H) {
    const n1 = state.n1 || state.kpsL?.length || 0;
    const n2 = state.n2 || state.kpsR?.length || 0;
    $("sum-kps").textContent = n1 ? `${n1} / ${n2}` : "";
    if (!state.matchJ) {
      $("sum-match").textContent = "";
      $("sum-inl").textContent = "";
      if ($("h-summary")) $("h-summary").textContent = "";
      return;
    }
    let nMatch = 0, nInl = 0;
    for (let i = 0; i < state.matchJ.length; i++) {
      if (state.matchJ[i] !== 0xFFFFFFFF) nMatch++;
      if (state.inlier?.[i]) nInl++;
    }
    $("sum-match").textContent = String(nMatch);
    $("sum-inl").textContent = `${nInl} inliers`;
    $("sum-inl").parentElement.title = `${nInl} inliers of ${nMatch} matches`;
    if (H) {
      if ($("h-summary")) $("h-summary").textContent = poseLine(H);
    }
    if ($("status-extra")) $("status-extra").textContent = `inlier ${+$("d-inlier_px").value | 0} px`;
  }

  function applyMutual(mj, n1, n2) {
    const rev = new Int32Array(n2).fill(-1);
    for (let i = 0; i < n1; i++) {
      const j = mj[i];
      if (j !== 0xFFFFFFFF && j < n2 && rev[j] < 0) rev[j] = i;
    }
    if ($("mutual")?.checked) {
      for (let i = 0; i < n1; i++) {
        const j = mj[i];
        if (j !== 0xFFFFFFFF && j < n2 && rev[j] !== i) mj[i] = 0xFFFFFFFF;
      }
      rev.fill(-1);
      for (let i = 0; i < n1; i++) {
        const j = mj[i];
        if (j !== 0xFFFFFFFF && j < n2 && rev[j] < 0) rev[j] = i;
      }
    }
    return rev;
  }

  function refreshInliers() {
    if (!state.kpsL || !state.matchJ) return;
    state.inlier = kpInliers(state.kpsL, state.kpsR, state.matchJ, currentH(), +$("d-inlier_px").value || INLIER_PX);
    writeMatchStats(currentH());
    showImages();
  }

  async function ingestDetect(n1, n2) {
    state.n1 = n1; state.n2 = n2;
    state.poseClimbed = false;
    state.kpsL = kpRows(await zc.keypoints(0), await zc.keypointFrames(0));
    state.kpsR = kpRows(await zc.keypoints(1), await zc.keypointFrames(1));
    state.matchJ = state.matchJRaw = state.matchRev = state.inlier = null;
    state.matchScore = state._scoreRange = null;
    state.kpSel = null;
    writeMatchStats(currentH());
    await showImages();
  }

  /** The engine's matches (after a POS-GIFT rotation search, the fixed keypoints are the winning frame's, mapped back). */
  async function ingestMatches(n1, n2, H) {
    state.n1 = n1; state.n2 = n2;
    state.poseClimbed = false;
    state.H0 = H;
    zeroTan();
    // The fit is a new global model; a spline climbed on top of the old one no longer applies.
    if (state.cps) {
      state.cps.fill(0);
      state.cpsStart = Float32Array.from(state.cps);
      state.cpsGrad = null;
      applyFfd();
    }
    state.kpsL = kpRows(await zc.keypoints(0), await zc.keypointFrames(0));
    state.kpsR = kpRows(await zc.keypoints(1), await zc.keypointFrames(1));
    state.n1 = n1 = state.kpsL.length;
    state.n2 = n2 = state.kpsR.length;
    state.matchJRaw = await zc.matches();
    state.matchJ = Uint32Array.from(state.matchJRaw);
    state.matchRev = applyMutual(state.matchJ, n1, n2);
    state.inlier = kpInliers(state.kpsL, state.kpsR, state.matchJ, H, +$("d-inlier_px").value || INLIER_PX);
    state.matchScore = await zc.matchScores();
    state._scoreRange = matchScoreRange();
    state.kpSel = null;
    writeMatchStats(H);
    await showImages();
  }

  /** Forget everything tied to the previous image pair: pose, spline, keypoints, matches, trail. */
  function resetPairState() {
    state.H0 = identityH();
    zeroTan();
    state.cps = null;
    state.cpsStart = null;
    state.cpsGrad = null;
    applyFfd();  // + B-spline is the user's model choice and stays; the lattice restarts at rest
    state.trail = [];
    state.trailSel = -1;
    state.kpsL = state.kpsR = state.matchJ = state.matchJRaw = state.matchRev = state.inlier = state.kpSel = state.kpHover = null;
    state.matchScore = state._scoreRange = null;
    state.n1 = state.n2 = 0;
    state.poseClimbed = false;
    $("sum-kps").textContent = "";
    $("sum-match").textContent = "";
    $("sum-inl").textContent = "";
    if ($("res-pose")) $("res-pose").textContent = "";
    $("sum-h").textContent = "";
    if ($("h-summary")) $("h-summary").textContent = "";
    if ($("Htxt")) $("Htxt").textContent = "";
    if ($("status-extra")) $("status-extra").textContent = "";
    setDock("kp");
    state.regView = { z: 1, x: 0, y: 0 };
    applyFfd();
    drawTrail();
  }


  async function openFile(which, file) {
    if (!file || job.kind || stackRun.active) return;
    if (stackOn()) setMode("pair");  // a single image is a pair, not a slice
    const img = await fileToRgba(file, false);
    resetPairState();
    if (which === "left") state.stem = file.name.replace(/\.[^.]*$/, "");
    await ingest(which, img);
  }
  /** Two files dropped together: the first is moving, the second fixed. */
  async function openPair(files) {
    if (files.length < 2 || job.kind || stackRun.active) return false;
    if (stackOn()) setMode("pair");
    const [a, b] = await Promise.all([fileToRgba(files[0], false), fileToRgba(files[1], false)]);
    resetPairState();
    state.stem = files[0].name.replace(/\.[^.]*$/, "");
    await ingest("left", a);
    await ingest("right", b);
    log(`opened ${files[0].name} → moving, ${files[1].name} → fixed`);
    return true;
  }
  $("btn-open-left").onclick = () => $("file-left").click();
  $("btn-open-right").onclick = () => $("file-right").click();
  $("btn-empty-left").onclick = () => $("file-left").click();
  $("btn-empty-right").onclick = () => $("file-right").click();
  // The overlay takes drops too: two files are moving then fixed; one fills the missing side.
  {
    const el = $("pane-reg");
    el.addEventListener("dragover", (e) => { e.preventDefault(); el.classList.add("over"); });
    el.addEventListener("dragleave", () => el.classList.remove("over"));
    el.addEventListener("drop", async (e) => {
      e.preventDefault(); el.classList.remove("over");
      const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith("image/"));
      if (await openPair(files)) return;
      if (!files[0]) return;
      if (state.left && state.right) { log("drop on the moving or the fixed pane to replace one image"); return; }
      await openFile(state.left ? "right" : "left", files[0]);
    });
  }
  $("file-left").onchange = () => openFile("left", $("file-left").files[0]);
  $("file-right").onchange = () => openFile("right", $("file-right").files[0]);
  ["pane-left", "pane-right"].forEach((id) => {
    const el = $(id);
    el.addEventListener("dragover", (e) => { e.preventDefault(); el.classList.add("over"); });
    el.addEventListener("dragleave", () => el.classList.remove("over"));
    el.addEventListener("drop", async (e) => {
      e.preventDefault(); el.classList.remove("over");
      const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith("image/"));
      if (await openPair(files)) return;
      if (files[0]) await openFile(el.dataset.side === "left" ? "left" : "right", files[0]);
    });
  });
  const pickKp = (side, ev, persist) => {
    const cv = side === "L" ? $("cv-left") : $("cv-right");
    const view = side === "L" ? state._viewL : state._viewR;
    const kps = side === "L" ? state.kpsL : state.kpsR;
    const mask = side === "L" ? state._vis?.L : state._vis?.R;
    const r = cv.getBoundingClientRect();
    const i = hitKp(view, kps, ev.clientX - r.left, ev.clientY - r.top, mask);
    if (i < 0) {
      if (persist) {
        state.kpSel = null;
        drawKpPatch(null);
        showImages();
      } else if (state.kpHover) { state.kpHover = null; showImages(); }
      return;
    }
    let sel;
    if (side === "L") {
      const j = state.matchJ ? state.matchJ[i] : 0xFFFFFFFF;
      sel = { side: "L", i, pair: j === 0xFFFFFFFF ? -1 : j };
    } else {
      const li = state.matchRev ? state.matchRev[i] : -1;
      sel = { side: "R", i, pair: li };
    }
    if (persist) {
      state.kpSel = sel;
      const p = kps[i];
      log(`kp ${side} #${i}  (${p[0].toFixed(1)},${p[1].toFixed(1)})  scale ${kpScale(p).toFixed(2)}` +
        (sel.pair >= 0 ? `  ↔ ${sel.pair}` : "  unmatched") +
        (side === "L" && state.inlier?.[i] ? "  inlier" : ""));
      drawKpPatch(sel);
    } else {
      if (state.kpHover && state.kpHover.side === sel.side && state.kpHover.i === sel.i) return;
      state.kpHover = sel;
    }
    showImages();
  };
  $("cv-left").addEventListener("click", (e) => pickKp("L", e, true));
  $("cv-right").addEventListener("click", (e) => pickKp("R", e, true));
  $("cv-left").addEventListener("mousemove", (e) => pickKp("L", e, false));
  $("cv-right").addEventListener("mousemove", (e) => pickKp("R", e, false));
  $("cv-left").addEventListener("mouseleave", () => { if (state.kpHover) { state.kpHover = null; showImages(); } });
  $("cv-right").addEventListener("mouseleave", () => { if (state.kpHover) { state.kpHover = null; showImages(); } });
  state._viewL = state._viewR = null;
  window.addEventListener("resize", () => { showImages(); drawTrail(); drawCbar(); if (state.pose) applyWarp(false); });

  $("clahe").onchange = () => rebakeAll();
  if ($("band")) $("band").onchange = () => { syncBand(); rebakeAll(); };
  if ($("invert")) $("invert").onchange = () => rebakeAll();
  if ($("bp-fine")) $("bp-fine").onchange = () => { if ($("band").checked) rebakeAll(); };
  if ($("bp-coarse")) $("bp-coarse").onchange = () => { if ($("band").checked) rebakeAll(); };
  syncBand();
  document.querySelectorAll(".more-pop > button").forEach((btn) => {
    btn.addEventListener("click", () => { const menu = btn.closest("details"); if (menu) menu.open = false; });
  });
  // Menus are placed in the window (a pane would clip them), under their button and inside the
  // window's edges, and close on an outside click or a resize.
  document.querySelectorAll("details.more").forEach((menu) => {
    menu.addEventListener("toggle", () => {
      const pop = menu.querySelector(".more-pop");
      if (!menu.open || !pop) return;
      const r = menu.querySelector("summary").getBoundingClientRect();
      const w = pop.offsetWidth;
      pop.style.left = `${Math.round(Math.max(8, Math.min(innerWidth - w - 8, r.right - w)))}px`;
      pop.style.top = `${Math.round(r.bottom + 6)}px`;
    });
  });
  const closeMenus = (e) => {
    for (const menu of document.querySelectorAll("details.more[open]")) if (!e || !menu.contains(e.target)) menu.open = false;
  };
  document.addEventListener("pointerdown", closeMenus);
  window.addEventListener("resize", () => closeMenus());
  $("btn-swap").onclick = () => {
    if (!state.left || !state.right) return;
    [state.left, state.right] = [state.right, state.left];
    zcDo(() => zc.swap());
    [state.kpsL, state.kpsR] = [state.kpsR, state.kpsL];
    [state.levelsL, state.levelsR] = [state.levelsR, state.levelsL];
    [state.n1, state.n2] = [state.n2, state.n1];
    // Invert the whole pose (tangent step included) before the inlier recount uses it.
    state.H0 = inv3(currentH());
    zeroTan();
    if (state.matchJ) {
      const old = state.matchJ;
      const mj = new Uint32Array(state.n1).fill(0xffffffff);
      for (let j = 0; j < old.length; j++) {
        const i = old[j];
        if (i !== 0xffffffff && i < mj.length && mj[i] === 0xffffffff) mj[i] = j;
      }
      state.matchJ = mj;
      state.matchJRaw = Uint32Array.from(mj);
      state.matchRev = applyMutual(mj, state.n1, state.n2);
      refreshInliers();
    }
    if ($("ffd")?.checked && state.cps) {
      const frame = $("ffd-frame")?.value === "source" ? "target" : "source";
      if ($("ffd-frame")) $("ffd-frame").value = frame;
      for (let i = 0; i < state.cps.length; i++) state.cps[i] = -state.cps[i];
      if (state.cpsStart) for (let i = 0; i < state.cpsStart.length; i++) state.cpsStart[i] = -state.cpsStart[i];
      applyFfd();
    }
    state.rsCenter = null;
    showImages();
    writeH();
    applyWarp(true);
    log("swapped moving and fixed");
  };

  $("sym").onchange = () => {
    if (state.left && state.right) applyWarp(true);
    drawTrail();
  };
  $("inverse").onchange = () => {
    if (state.left && state.right) applyWarp(true);
    else paintReg();
  };

  $("half").onchange = async () => {
    if (!f16Avail) { $("half").checked = false; return; }
    log($("half").checked ? "half precision on (fp16)" : "half precision off (fp32)");
    if (state.left) await bakeSide("left");
    if (state.right) await bakeSide("right");
    await showImages();
    if (state.left && state.right) await applyWarp(true);
  };

  $("kp-max").addEventListener("input", () => { $("kp-max-val").textContent = $("kp-max").value; showImages(); });
  $("kp-show").onchange = () => showImages();
  $("kp-lines").onchange = () => showImages();
  if ($("kp-dots")) $("kp-dots").onchange = () => showImages();
  if ($("kp-line-color")) $("kp-line-color").onchange = () => showImages();
  $("d-inlier_px").addEventListener("input", refreshInliers);
  $("mutual").onchange = () => {
    if (!state.matchJRaw || !state.n1) return;
    state.matchJ = Uint32Array.from(state.matchJRaw);
    state.matchRev = applyMutual(state.matchJ, state.n1, state.n2);
    refreshInliers();
  };
  $("h-digits").onchange = () => { if (state.left && state.right) writeH(); };
  if ($("tweak")) $("tweak").onchange = () => { syncTweakUi(); };
  if ($("zoom-ov")) {
    $("zoom-ov").onchange = () => {
      state.regView = { z: 1, x: 0, y: 0 };
      if (state.pose) paintReg();
    };
  }
  if ($("map-log")) $("map-log").onchange = () => renderMaps();
  if ($("map-cbar")) $("map-cbar").onchange = () => { syncMapBar(); renderMap("shift"); };
  const onRsBound = () => {
    if (state.left && state.right) refreshMap(true);
    else renderMap("rs");
  };
  const onRsBoundDraw = () => renderMap("rs");
  if ($("rs-smin")) { $("rs-smin").oninput = onRsBoundDraw; $("rs-smin").onchange = onRsBound; }
  if ($("rs-smax")) { $("rs-smax").oninput = onRsBoundDraw; $("rs-smax").onchange = onRsBound; }
  $("btn-h-toggle").onclick = () => {
    const on = $("Htxt").hidden;
    $("Htxt").hidden = !on;
    $("btn-h-toggle").setAttribute("aria-pressed", String(on));
  };
  $("btn-copy-h").onclick = async () => {
    try {
      await navigator.clipboard.writeText($("Htxt").textContent || fmtH(currentH(), hDigits()));
      $("btn-copy-h").textContent = "copied";
      setTimeout(() => { $("btn-copy-h").textContent = "copy"; }, 900);
    } catch (e) { log(String(e)); }
  };
  function downloadBlob(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
  function exportWarp() {
    if (!state.left || !state.right) { log("need both images"); return; }
    applyFfd();
    const H = currentH();
    const on = !!$("ffd")?.checked && state.cps;
    const g = Math.max(2, state.cpsGx || +$("ffd-g")?.value || 4);
    const opts = {
      H,
      cps: on ? state.cps : null,
      gx: g, gy: g,
      frame: $("ffd-frame")?.value || "target",
      lw: state.left.w, lh: state.left.h,
      rw: state.right.w, rh: state.right.h,
      ffd: on,
    };
    const stem = (`${state.stem || "cmir"}`.replace(/[^\w.-]+/g, "_") || "cmir") + "_warp";
    // The engine writes the bundle (src/export.zig): NRRD field, and an ITK .tfm for a spline or an affine pose.
    const nfiles = on || isAffineHomography(H) ? 2 : 1;
    zcSync()
      .then(() => zcDo(() => { zc.pose = H; return zc.exportWarp(stem); }))
      .then((zip) => downloadBlob(new Blob([zip], { type: "application/zip" }), `${stem}.zip`))
      .catch((e) => log(String(e.stack || e)));
    const kind = on ? `B-spline ${g}×${g} ${opts.frame}` : (isAffineHomography(H) ? "affine" : "homography");
    log(`export  ${stem}.zip  NRRD displacement field${nfiles > 1 ? " + Insight .tfm" : ""}  ${kind}  ${state.right.w}×${state.right.h}`);
  }
  $("btn-export-warp").onclick = exportWarp;
  async function runMatch() {
    if (!state.left || !state.right || !state.n1) { log("detect first"); return; }
    setBusy(true, "matching…");
    try {
      const t0 = performance.now();
      await zcSync();
      const r = await zc.match();
      throwIfCanceled();
      await ingestMatches(state.n1, state.n2, Float64Array.from(r.H));
      log(`match  ${((performance.now() - t0) / 1000).toFixed(2)}s`);
      log(summary(r.H));
      await applyWarp(true);
      setDock("map");
    } catch (e) {
      if (e.canceled) log("match canceled");
      else log(String(e.stack || e));
    }
    setBusy(false);
  }
  $("btn-rematch").onclick = runMatch;
  $("btn-refit").onclick = () => {
    if (!state.kpsL || !state.matchJ || !state.inlier) return;
    const p = [], q = [];
    for (let i = 0; i < state.kpsL.length; i++) {
      if (!state.inlier[i]) continue;
      const j = state.matchJ[i];
      if (j === 0xFFFFFFFF) continue;
      p.push([state.kpsL[i][0], state.kpsL[i][1]]);
      q.push([state.kpsR[j][0], state.kpsR[j][1]]);
    }
    if (p.length < 3) { log("need ≥3 inliers to refit"); return; }
    pushUndo();
    state.H0 = (state.group === "affine" || p.length < 4) ? HAffineFromPts(p, q) : HHomographyFromPts(p, q);
    zeroTan();
    refreshInliers();
    applyWarp(true);
    state.poseClimbed = false;
    markNext();
    log(`refit from ${p.length} inliers  ${summary(state.H0)}`);
  };

  document.querySelectorAll("input[name=group]").forEach((el) => {
    el.onchange = () => { setGroup(el.value); applyWarp(true); };
  });
  // A different score makes the identity baseline (the "×" in the status line) stale.
  const onScoreChange = async () => {
    syncScore();
    if ($("exact-lab")) $("exact-lab").hidden = !/^SMI/.test($("metric").value);
    if (!state.left || !state.right) return;
    state._ns0 = (await zc.score(identityH())).fwd;
    await applyWarp(true);
  };
  $("metric").onchange = onScoreChange;
  if ($("exact")) $("exact").onchange = onScoreChange;
  if ($("map-res")) $("map-res").onchange = () => { syncScore(); if (state.left && state.right) refreshMap(true); };
  if ($("map-cal")) $("map-cal").onchange = () => { syncScore(); if (state.left && state.right) refreshMap(true); };
  $("reg-mode").onchange = () => { armFlicker(); applyWarp(false); };
  $("min-ov").onchange = () => refreshMap(true);
  $("theme").onchange = () => { theme = +$("theme").value; drawCbar(); renderMaps(); };
  function syncOptimizeUi() {
    $("line-params")?.classList.remove("lk-hide");
  }
  function syncMatchHint() {
    const m = $("match-method")?.value;
    const el = $("match-hint");
    if (!el) return;
    if (posGiftOn()) el.textContent = "POS-GIFT: nearest neighbours per level pair, affine FSC, then POS re-matching near the affine. The inlier radius recolors the overlay.";
    else if (m === "prosac") el.textContent = "PROSAC: top-ranked 4-point H, orientation + cheirality predicates, LO-DLT.";
    else if (m === "magsac") el.textContent = "MAGSAC++: σ-consensus weights instead of a hard inlier cut, then weighted DLT.";
    else el.textContent = "Lo-FSC: GPU 3-point affine + scale gate. inlier radius recolors live.";
  }
  if ($("hop")) $("hop").onchange = syncOptimizeUi;
  if ($("match-method")) $("match-method").onchange = syncMatchHint;
  syncOptimizeUi();
  syncMatchHint();
  syncClimbAuto = () => {
    const on = !!$("hho-auto")?.checked;
    const g = Math.max(2, +$("ffd-g")?.value | 0 || 4);
    const edge = 0.01;
    const spec = {
      "hho-n": 100,
      "hho-t": 20,
      "hho-sg": edge,
      "hho-sh": edge,
      "hho-p": edge * 0.01,
      "peak-tx": edge,
      "peak-th": edge,
      "hho-phi": +((0.05 * 2) / Math.max(g - 1, 1)).toFixed(4),
    };
    for (const [id, value] of Object.entries(spec)) {
      const el = $(id);
      if (!el) continue;
      el.disabled = on;
      if (on) el.value = value;
    }
  };
  if ($("hho-auto")) $("hho-auto").onchange = () => syncClimbAuto();
  syncClimbAuto();
  $("ffd").onchange = () => { syncModelUi(); applyFfd(); syncClimbAuto(); applyWarp(true); };
  syncModelUi();
  if ($("ffd-arrows")) $("ffd-arrows").onchange = () => { if (state.pose) paintReg(); };
  if ($("ffd-view")) $("ffd-view").onchange = () => { if (state.pose) paintReg(); };
  if ($("ffd-frame")) $("ffd-frame").onchange = () => { applyFfd(); applyWarp(true); };
  $("ffd-g").onchange = () => {
    state.cps = null;
    state.cpsStart = null;
    state.cpsGrad = null;
    applyFfd();
    syncClimbAuto();
    applyWarp(true);
  };
  $("btn-ffd-reset").onclick = () => {
    if (state.cps) state.cps.fill(0);
    state.cpsStart = state.cps ? Float32Array.from(state.cps) : null;
    state.cpsGrad = null;
    applyFfd();
    applyWarp(true);
  };
  $("btn-stop").onclick = () => requestCancel();
  $("btn-ident").onclick = () => { pushUndo(); state.H0 = identityH(); zeroTan(); applyWarp(true); };
  $("btn-reset-tan").onclick = () => { pushUndo(); zeroTan(); applyWarp(true); };
  $("btn-bake").onclick = () => { state.H0 = currentH(); zeroTan(); applyWarp(false); };
  // Snap to the shift-map peak (the translation it stands for, in fixed pixels).
  $("btn-snap").onclick = async () => {
    if (!state.maps.shift) await refreshMap(false);
    const m = state.maps.shift;
    if (!m) return;
    await commitPose(mul3(new Float64Array([1, 0, -m.dx, 0, 1, -m.dy, 0, 0, 1]), m.H || currentH()));
  };


  async function recordOptTrail(score, label, H, kind, force = true) {
    if (state.left && state.right && $("metric").value !== "NCC" && H) {
      try {
        const parts = await pairParts(H);
        recordTrail(parts.fwd, label, force, invOn()
          ? { metric: "SMI", score2: parts.inv, metric2: "inverse" }
          : { metric: "SMI" });
        return parts.fwd;
      } catch (e) { log(e.stack || e); }
    }
    recordTrail(score, label, force, { metric: "dest nS" });
    return score;
  }

  function drawTileHeat(cv, scores, G, label) {
    if (!cv || !scores) return;
    const dpr = devicePixelRatio || 1;
    const W = Math.max(1, cv.clientWidth), H = Math.max(1, cv.clientHeight);
    cv.width = W * dpr; cv.height = H * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0c0c0c";
    ctx.fillRect(0, 0, W, H);
    const n = G * G;
    let mx = 1e-8;
    for (let i = 0; i < n; i++) if (scores[i] > mx) mx = scores[i];
    const pad = 2, cell = Math.min((W - pad) / G, (H - pad) / G);
    const ox = (W - cell * G) * 0.5, oy = (H - cell * G) * 0.5;
    for (let ty = 0; ty < G; ty++) for (let tx = 0; tx < G; tx++) {
      const v = scores[ty * G + tx];
      const [r, g, b] = colormap(v / mx, 2);
      ctx.fillStyle = `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;
      ctx.fillRect(ox + tx * cell + 0.5, oy + ty * cell + 0.5, cell - 1, cell - 1);
    }
    if (label) $(label).textContent = `${G}×${G}  max ${mx.toFixed(1)}`;
  }

  // Search tab: the last sweep's climbed finalists, best first. A click tries that pose.
  function drawCandList() {
    const ol = $("cand-list");
    if (!ol) return;
    const v = state.sweepView, cands = v?.cands || [];
    if ($("cand-meta")) $("cand-meta").textContent = cands.length ? `${cands.length} climbed` : "";
    if (!cands.length) {
      ol.innerHTML = `<li class="empty">${state.swarmHist ? "The cloud search keeps one pose; its members are plotted above." : v ? "Searching…" : "Search (H) sweeps rotation × scale × shift, climbs the best peaks, and lists them here."}</li>`;
      return;
    }
    ol.innerHTML = cands.map((c, k) => {
      const lab = `${(c.th * 180 / Math.PI).toFixed(1)}°  ×${c.s.toFixed(3)}`;
      return `<li data-k="${k}" class="${k === v.applied ? "sel" : ""}${k === 0 ? " best" : ""}" title="Sweep peak #${c.peak}: rotation ${(c.th * 180 / Math.PI).toFixed(2)}°, scale ×${c.s.toFixed(4)}. Click to apply this pose (Ctrl+Z undoes)."><span class="k">${k + 1}</span><span class="lab">${lab}</span><span class="v">${fmtScoreRow(c.score)}</span></li>`;
    }).join("");
  }

  $("cand-list").onclick = async (ev) => {
    const li = ev.target.closest("li[data-k]");
    const v = state.sweepView;
    if (!li || !v?.cands || job.kind) return;
    const k = +li.dataset.k, c = v.cands[k];
    pushUndo();
    state.H0 = Float64Array.from(c.H);
    zeroTan();
    if (state.cps && $("ffd")?.checked) { state.cps.fill(0); applyFfd(); }
    v.applied = k;
    drawCandList();
    await recordOptTrail(c.score, `search #${k + 1}`, c.H, "dest");
    await applyWarp(true);
  };
  // The side-column plots redraw whenever their box changes (tab switch, splitter, window).
  if (typeof ResizeObserver === "function") {
    const ro = new ResizeObserver((ents) => {
      for (const e of ents) {
        if (e.target.id === "cv-swarm") drawSwarm();
        else if (e.target.id === "cv-flex") drawTrail();
      }
    });
    for (const id of ["cv-swarm", "cv-flex"]) if ($(id)) ro.observe($(id));
  }
  $("trail-list").onclick = (ev) => {
    const li = ev.target.closest("li[data-i]");
    if (li && !job.kind) restoreTrail(+li.dataset.i);
  };

  function drawSwarm() {
    drawCandList();
    const cv = $("cv-swarm");
    if (!cv) return;
    const hist = state.swarmHist;
    if (!hist && state.sweepView) { drawSweepView(cv, state.sweepView); return; }
    const dpr = devicePixelRatio || 1;
    const W = Math.max(1, cv.clientWidth), Hgt = Math.max(1, cv.clientHeight);
    cv.width = W * dpr; cv.height = Hgt * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0c0c0c";
    ctx.fillRect(0, 0, W, Hgt);
    const rows = hist?.rows || [];
    const nIt = rows.length;
    const pad = { l: 48, r: 12, t: 14, b: 22 };
    let lo = Infinity, hi = -Infinity;
    for (const row of rows) for (const s of row) if (Number.isFinite(s)) { if (s < lo) lo = s; if (s > hi) hi = s; }
    if (Number.isFinite(hist?.best)) { if (hist.best < lo) lo = hist.best; if (hist.best > hi) hi = hist.best; }
    if (!(hi > lo)) { hi = Number.isFinite(hi) ? hi + 1 : 1; lo = Number.isFinite(lo) ? lo - 1 : 0; }
    else { const g = 0.06 * (hi - lo); lo -= g; hi += g; }
    const xAt = (i) => pad.l + (nIt <= 1 ? 0.5 : i / (nIt - 1)) * (W - pad.l - pad.r);
    const yAt = (s) => pad.t + (1 - (s - lo) / (hi - lo)) * (Hgt - pad.t - pad.b);
    ctx.strokeStyle = "#2a2a2a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, Hgt - pad.b); ctx.lineTo(W - pad.r, Hgt - pad.b);
    ctx.stroke();
    ctx.font = "10px ui-monospace, Consolas, monospace";
    ctx.fillStyle = "#8a8a8a";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(fmtAxis(hi), pad.l - 4, pad.t);
    ctx.fillText(fmtAxis(lo), pad.l - 4, Hgt - pad.b);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("step", (pad.l + W - pad.r) / 2, Hgt - 12);
    if (!nIt) {
      ctx.fillStyle = "#5c5c5c";
      ctx.font = "12px ui-sans-serif, system-ui";
      ctx.textBaseline = "middle";
      ctx.fillText("score vs step", (pad.l + W - pad.r) / 2, Hgt * 0.48);
      if ($("swarm-meta")) $("swarm-meta").textContent = "score × step";
      return;
    }
    ctx.fillText("0", xAt(0), Hgt - pad.b + 3);
    if (nIt > 1) ctx.fillText(String(nIt - 1), xAt(nIt - 1), Hgt - pad.b + 3);
    const nMem = rows[0].length;
    ctx.fillStyle = "rgba(138, 164, 184, 0.85)";
    for (let m = 0; m < nMem; m++) {
      for (let i = 0; i < nIt; i++) {
        const s = rows[i][m];
        if (!Number.isFinite(s)) continue;
        ctx.beginPath();
        ctx.arc(xAt(i), yAt(s), 2.2, 0, 6.28);
        ctx.fill();
      }
    }
    if (Number.isFinite(hist.best)) {
      const y = yAt(hist.best);
      ctx.strokeStyle = "#e6c36a";
      ctx.lineWidth = 1.4;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(W - pad.r, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if ($("swarm-meta")) $("swarm-meta").textContent = `${nMem} seeds  best ${fmtAxis(hist.best)}`;
  }

  async function refreshTiles(H) {
    if (!state.left || !state.right || !popDockOn()) return;
    try {
      const heat = await zc.tileHeat(H || currentH(), { grid: TILE_GRID, minN: TILE_MIN_N });
      state.tiles = heat;
      drawTileHeat($("cv-tile-dest"), heat.dest, heat.G, "tile-dest-meta");
      drawTileHeat($("cv-tile-src"), heat.src, heat.G, "tile-src-meta");
    } catch (e) { log(e.stack || e); }
  }

  // Search (src/search.zig): "cloud" — a seed cloud around the pose (or the identity) climbing
  // together (with per-seed spline control points when the spline is on); "sweep" — a SIM(2)
  // sweep of the pose score, NMS, a batched Gauss–Newton screen and full climbs of the finalists
  // (spline off; a new pose resets it). The pose changes only when the search scores higher.
  async function runSearch(mode) {
    if (!state.left || !state.right) { log("need both images"); return; }
    if (mode === "sweep" && $("metric").value === "NCC") { log("search: the global sweep needs a moment score (SMI, E4 or λmax), not NCC."); return; }
    setBusy(true, mode === "sweep" ? "sweep…" : "climb…", mode === "sweep" ? "sweep" : "hho");
    pushUndo();
    setDock("pop");
    state.swarmHist = mode === "cloud" ? { rows: [], best: -Infinity } : null;
    state.sweepView = mode === "sweep" ? { thetas: [0], sigmas: [1], wrap: true, best: new Float32Array(1), top: [], finalists: [], result: null } : null;
    drawSwarm();
    try {
      await nextPaint();
      applyFfd();
      const H = Float64Array.from(currentH());
      await zcDo(() => { zc.configure({ ...appSettings(), search_mode: mode }); setCpsNow(); zc.pose = H; });
      const r = await zc.search();
      state.H0 = Float64Array.from(r.H);
      zeroTan();
      if ($("ffd")?.checked) {
        const cps = await zcDo(() => zc.ffd);
        if (state.cps && cps.length === state.cps.length) state.cps = Float32Array.from(cps);
      }
      applyFfd();
      drawSwarm();
      await applyWarp(true);
    } catch (e) {
      if (e.canceled) log("search stopped");
      else log(String(e.stack || e));
    } finally {
      setBusy(false);
    }
  }
  function runHho() { return runSearch("cloud"); }

  // Global Search: SIM(2) sweep → NMS → batched Gauss–Newton screen → full climbs.
  // Every score is the app's pose / shift-map score (whitening, edge weight and
  // symmetry as configured). The spline is off during the search; a new pose
  // resets it. The current pose is kept unless the search scores higher.
  function runSweep() { return runSearch("sweep"); }

  // θ × σ landscape of the last sweep: each cell is that candidate's best
  // translation score; rings mark the NMS peaks, orange the finalists, gold the result.
  function drawSweepView(cv, v) {
    const dpr = devicePixelRatio || 1;
    const W = Math.max(1, cv.clientWidth), Hgt = Math.max(1, cv.clientHeight);
    cv.width = W * dpr; cv.height = Hgt * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0c0c0c";
    ctx.fillRect(0, 0, W, Hgt);
    const nT = v.thetas.length, nS = v.sigmas.length;
    const pad = { l: 40, r: 12, t: 12, b: 24 };
    const pw = Math.max(1, W - pad.l - pad.r), ph = Math.max(1, Hgt - pad.t - pad.b);
    let hi = 0, lo = Infinity;
    for (const s of v.best) { if (s > hi) hi = s; if (s > 0 && s < lo) lo = s; }
    if (!(lo < hi)) lo = 0;
    const cw = pw / nT, chh = ph / nS;
    const xAt = (it) => pad.l + (it + 0.5) * cw;
    const yAt = (is) => pad.t + (nS - 0.5 - is) * chh;
    for (let i = 0; i < nT; i++) for (let j = 0; j < nS; j++) {
      const s = v.best[i * nS + j];
      if (!(s > 0)) continue;
      const [r, g, b] = colormap((s - lo) / (hi - lo || 1), theme);
      ctx.fillStyle = `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;
      ctx.fillRect(pad.l + i * cw, pad.t + (nS - 1 - j) * chh, Math.ceil(cw), Math.ceil(chh));
    }
    // Fractional grid position of a (θ, σ), for the refined result.
    const tStep = v.wrap ? 2 * Math.PI / nT : nT > 1 ? (v.thetas[nT - 1] - v.thetas[0]) / (nT - 1) : 1;
    const ls0 = Math.log(v.sigmas[0]), lsStep = nS > 1 ? (Math.log(v.sigmas[nS - 1]) - ls0) / (nS - 1) : 1;
    const ring = (x, y, r, color, w = 1.4) => {
      ctx.strokeStyle = color; ctx.lineWidth = w;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.stroke();
    };
    const rr = Math.max(3, Math.min(7, 0.45 * Math.min(cw * 2, chh)));
    v.top.forEach((p, k) => {
      const f = v.finalists.includes(k);
      ring(xAt(p.it), yAt(p.is), rr, f ? "#f0a050" : "rgba(255,255,255,0.75)", f ? 2 : 1);
    });
    if (v.result) {
      // the result's angle within the swept range (a range may run past ±180°)
      const th = v.thetas[0] + ((((v.result.th - v.thetas[0]) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI));
      let ft = (th - v.thetas[0]) / tStep;
      if (v.wrap) ft = ((ft % nT) + nT) % nT;
      const fs = nS > 1 ? (Math.log(v.result.s) - ls0) / lsStep : 0;
      ring(pad.l + (ft + 0.5) * cw, pad.t + (nS - 0.5 - fs) * chh, rr + 3, "#e6c36a", 2.2);
    }
    ctx.font = "10px ui-monospace, Consolas, monospace";
    ctx.fillStyle = "#8a8a8a";
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    const deg = (t) => `${Math.round(t * 180 / Math.PI)}°`;
    ctx.fillText(deg(v.thetas[0]), pad.l, Hgt - pad.b + 4);
    ctx.textAlign = "right";
    ctx.fillText(deg(v.wrap ? Math.PI : v.thetas[nT - 1]), W - pad.r, Hgt - pad.b + 4);
    ctx.textAlign = "center";
    ctx.fillText("rotation", pad.l + pw / 2, Hgt - pad.b + 4);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(`×${v.sigmas[nS - 1].toFixed(2)}`, pad.l - 4, yAt(nS - 1));
    if (nS > 1) ctx.fillText(`×${v.sigmas[0].toFixed(2)}`, pad.l - 4, yAt(0));
    if ($("swarm-meta")) $("swarm-meta").textContent = `sweep ${nT}θ × ${nS}σ  best coarse nS ${fmtAxis(hi)}  ${v.top.length} peaks`;
  }

  $("btn-hho-align").onclick = () => (searchMode() === "cloud" ? runHho() : runSweep());
  const syncSearchMode = () => {
    const cloud = searchMode() === "cloud";
    if ($("search-sweep")) $("search-sweep").hidden = cloud;
    if ($("search-cloud")) $("search-cloud").hidden = !cloud;
  };
  document.querySelectorAll('input[name="search-mode"]').forEach((r) => { r.onchange = syncSearchMode; });
  syncSearchMode();

  // ── POS-GIFT (js/posgift.js): the Detect tab's detector choice ──
  function posGiftOn() { return radioValue("detector") !== "gls"; }
  // The configuration that registered every EBSD ↔ BSE section (POS_GIFT_DEFAULTS); exposed: rotation
  // search (any angle) versus upright (faster, within ~7°), the detector counts, and the Match tab's
  // affine / homography choice for the final POS model.
  function posGiftSettings() {
    const st = {
      ...POS_GIFT_DEFAULTS,
      rotation: $("pg-search").checked ? "search" : 0,
      pos_model: state.group === "affine" ? "affine" : "homography", // the Match tab's model
      n_trials: +$("d-n_trials").value || N_TRIALS, seed: (+$("d-seed").value | 0) || 1,
    };
    for (const [id, v, lo, hi, step, , , key] of [...PG_DETECT, ...PG_ADV]) {
      const x = +$(`d-${id}`).value;
      st[key] = Number.isFinite(x) ? Math.min(hi, Math.max(lo, x)) : v;
    }
    // Orientations are even (sectors = 2 × orientations); structure counts are integers.
    st.n_orient = Math.max(4, Math.min(8, 2 * Math.round(st.n_orient / 2)));
    for (const k of ["n_rings", "n_scales", "pos_k", "n_octaves", "max_points"]) st[k] = Math.round(st[k]);
    st.det_map = $("pg-corners")?.checked === false ? "pcsum" : "min_moment";
    return st;
  }
  function syncDetector() {
    const pos = posGiftOn();
    for (const [id, show] of [["gls-opts", !pos], ["gls-match-opts", !pos], ["gls-match-opts2", !pos], ["pg-opts", pos]]) {
      if ($(id)) $(id).hidden = !show;
    }
    syncMatchHint();
  }
  {
    const onDetector = () => {
      syncDetector();
      // Keypoints and matches of the other detector no longer apply.
      if (state.n1) {
        state.n1 = state.n2 = 0;
        state.pg = null;
        state.kpsL = state.kpsR = state.matchJ = state.matchJRaw = state.matchRev = state.inlier = state.kpSel = null;
        state.matchScore = state._scoreRange = null;
        writeMatchStats(currentH());
        setBusy(false);
        showImages();
        log(`detector ${posGiftOn() ? "POS-GIFT" : "GLS-MIFT"}: run Detect`);
      }
    };
    document.querySelectorAll('input[name="detector"]').forEach((r) => { r.onchange = onDetector; });
  }
  syncDetector();

  async function runDetect() {
    if (!state.left || !state.right) { log("need both images"); return; }
    setBusy(true, "detect…", "match");
    setDock("kp");
    try {
      await nextPaint();
      const t0 = performance.now();
      await zcSync();
      const d = await zc.detect();
      throwIfCanceled();
      await ingestDetect(d.n1, d.n2);
      log(`detect ${posGiftOn() ? "POS-GIFT" : "GLS-MIFT"}  ${((performance.now() - t0) / 1000).toFixed(2)}s  · Match to correspond`);
    } catch (e) {
      if (e.canceled) log("detect canceled");
      else log(String(e.stack || e));
    }
    setBusy(false);
  }
  $("btn-run").onclick = runDetect;

  $("btn-stop").onclick = () => requestCancel();
  $("btn-cancel").onclick = () => requestCancel();

  // The pose climb (src/gclimb.zig, on the GPU): shift / roto-scale hops, Gauss–Newton (line
  // searches for E4), the spline. The steps arrive as trail events once per batch.
  // opts.global: climb the affine / homography alone even with + B-spline (Auto's first stage;
  // only while the spline is at rest, so the pose shown and the pose climbed agree).
  async function runGrad(opts = {}) {
    if (!state.left || !state.right) return;
    if (dockName() === "kp") setDock("map");
    const withSpline = $("ffd").checked && !opts.global;
    setBusy(true, withSpline ? `${state.group} + spline ascent…` : `${state.group} ascent…`, "grad");
    try {
      await nextPaint();
      applyFfd();
      if (withSpline && state.cps && (!state.cpsStart || state.cpsStart.length !== state.cps.length)) {
        state.cpsStart = Float32Array.from(state.cps);
      }
      const H = Float64Array.from(currentH());
      await zcDo(() => { zc.configure({ ...appSettings(), ffd: withSpline }); setCpsNow(); zc.pose = H; });
      const r = await zc.climb();
      state.H0 = Float64Array.from(r.H);
      zeroTan();
      if (withSpline) {
        const cps = await zcDo(() => zc.ffd);
        if (state.cps && cps.length === state.cps.length) state.cps = Float32Array.from(cps);
      }
      applyFfd();
      await applyWarp(true);
      state.poseClimbed = true;
    } catch (e) {
      if (e.canceled) log("ascent stopped");
      else log(String(e.stack || e));
    } finally {
      setBusy(false);
    }
  }
  $("btn-grad").onclick = () => runGrad();

  async function runAuto() {
    if (!state.left || !state.right) { log("need both images"); return; }
    if (job.kind) return;
    try {
      markAuto("detect");
      await runDetect();
      if (job.stop || !state.n1) return;
      markAuto("match");
      await runMatch();
      if (job.stop || !state.matchJ) return;
      markAuto("climb");
      // Coarse to fine: the global model first, then the spline together with it (Match left the
      // spline at rest, so the first stage climbs exactly the pose on screen).
      const spline = $("ffd").checked;
      await runGrad({ global: spline });
      if (!spline || job.stop) return;
      log(`${state.group} climbed · adding the spline`);
      await runGrad();
    } finally {
      markAuto("");
    }
  }
  $("btn-auto").onclick = runAuto;

  async function restoreTrail(i) {
    const p = state.trail[i];
    if (!p) return;
    pushUndo();
    state._restoring = true;
    state.trailSel = i;
    restorePose(p.snap);
    applyFfd();
    try { await applyWarp(true); }
    finally { state._restoring = false; }
    drawTrail();
    $("flex-meta").textContent = trailCaption(p, "  restored");
  }

  $("cv-flex").addEventListener("pointerdown", (ev) => {
    const { pad, w, i0, i1, n } = trailLayout();
    if (!n) return;
    const r = $("cv-flex").getBoundingClientRect();
    const x0 = ev.clientX;
    let dragged = false;
    const span0 = Math.max(1, i1 - i0);
    const i00 = i0, i10 = i1;
    const plotW = Math.max(1, w - pad.l - pad.r);
    const move = (e) => {
      const dx = e.clientX - x0;
      if (Math.abs(dx) < 4 && !dragged) return;
      dragged = true;
      const di = -dx / plotW * span0;
      let a = i00 + di, b = i10 + di;
      if (a < 0) { b -= a; a = 0; }
      if (b > n - 1) { a -= (b - (n - 1)); b = n - 1; }
      a = Math.max(0, a);
      state.trailView = { i0: Math.round(a), i1: Math.round(b) };
      drawTrail();
    };
    const up = (e) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      if (!dragged) {
        const i = hitTrail(e.clientX);
        if (i >= 0) restoreTrail(i);
      }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });
  $("cv-flex").addEventListener("wheel", (ev) => {
    const { pad, w, i0, i1, n } = trailLayout();
    if (!n) return;
    ev.preventDefault();
    const r = $("cv-flex").getBoundingClientRect();
    const x = ev.clientX - r.left;
    const plotW = Math.max(1, w - pad.l - pad.r);
    const frac = Math.min(1, Math.max(0, (x - pad.l) / plotW));
    const mid = i0 + frac * Math.max(1, i1 - i0);
    const span = Math.max(1, i1 - i0);
    const nspan = Math.max(2, Math.min(n - 1, span * (ev.deltaY > 0 ? 1.25 : 0.8)));
    let a = mid - frac * nspan, b = a + nspan;
    if (a < 0) { b -= a; a = 0; }
    if (b > n - 1) { a -= (b - (n - 1)); b = n - 1; }
    state.trailView = { i0: Math.round(Math.max(0, a)), i1: Math.round(Math.min(n - 1, b)) };
    drawTrail();
  }, { passive: false });
  $("cv-flex").addEventListener("dblclick", (ev) => { ev.preventDefault(); resetTrailView(); });
  $("cv-flex").addEventListener("mousemove", (ev) => {
    const i = hitTrail(ev.clientX);
    const p = state.trail[i];
    if (p) $("flex-meta").textContent = trailCaption(p, "  · click to restore · wheel zoom · drag pan");
  });
  if ($("btn-flex-reset")) $("btn-flex-reset").onclick = () => resetTrailView();
  syncTweakUi();
  syncMapBar();
  drawTrail();
  drawCbar();

  // Programmatic pipeline for the benchmark and the console. Images are
  // { rgba, w, h }; H maps moving (left) → fixed (right) pixels, 1-based.
  Object.assign(window.cmir, {
    async loadImages(moving, fixed) {
      resetPairState();
      await ingest("left", moving);
      await ingest("right", fixed);
    },
    runDetect, runMatch, runGrad, runAuto, runSweep, runHho,
    currentH: () => Float64Array.from(currentH()),
    state, zc,
    idle: () => requestGpu({}),
  });

  syncSummaries();
  // The Examples menu lists the current mode's examples only.
  let presetList = [];
  function fillPresets(mode) {
    const sel = $("preset");
    sel.replaceChildren(new Option("Examples…", ""));
    const groups = new Map();
    for (const p of presetList.filter((x) => (x.type === "stack") === (mode === "stack"))) {
      if (!groups.has(p.group)) groups.set(p.group, Object.assign(document.createElement("optgroup"), { label: p.group }));
      groups.get(p.group).append(new Option(p.name, p.id));
    }
    for (const g of groups.values()) sel.append(g);
    sel.hidden = !groups.size;  // no examples for this mode
  }

  // ── Stack mode: a moving stack registered to a fixed stack, slice pair by slice pair ──
  // Every view and card works on the current slice pair. Slices are decoded when visited (a stack
  // can be far larger than memory holds as RGBA). Per slice: { H, cps, score, method, redetected }
  // with method "keypoints" (detect → match → refine), "carry" (the previous slice's transform,
  // refined) or "edited" (changed by hand after registration).
  const stack = { mov: [], fix: [], res: [], i: -1, loaded: false, cache: new Map(), pending: null, nav: false };
  const stackN = () => Math.min(stack.mov.length, stack.fix.length);
  const stackOn = () => $("app").dataset.mode === "stack";
  const sliceScore = (sc) => (Number.isFinite(sc.mean) ? sc.mean : sc.fwd);
  const sameH = (A, B) => A && B && A.every((v, k) => Math.abs(v - B[k]) <= 1e-9 * (1 + Math.abs(v)));

  /** Decoded slices (and their preprocessed work images) are kept up to this many bytes, least
   *  recently used first out; the slice pair on screen always stays. */
  const CACHE_BYTES = 768 << 20;
  const sliceBytes = (img) => (img ? img.rgba.length + (img.baked ? img.baked.rgba.length : 0) : 0);
  function trimCache() {
    const keep = new Set(stack.i >= 0 ? [stack.mov[stack.i], stack.fix[stack.i]] : []);
    let total = 0;
    for (const e of stack.cache.values()) total += sliceBytes(e.img);
    for (const [item, e] of stack.cache) {
      if (total <= CACHE_BYTES) break;
      if (keep.has(item) || !e.img) continue;
      total -= sliceBytes(e.img);
      stack.cache.delete(item);
    }
  }
  function decodeSlice(item) {
    let e = stack.cache.get(item);
    if (e) {
      stack.cache.delete(item);  // most recently used last
      stack.cache.set(item, e);
      return e.p;
    }
    e = { img: null, p: null };
    e.p = (async () => {
      const blob = item.file || await (await fetch(item.url)).blob();
      e.img = Object.assign(await fileToRgba(blob, false), { name: item.name });
      trimCache();
      return e.img;
    })();
    e.p.catch(() => stack.cache.delete(item));
    stack.cache.set(item, e);
    return e.p;
  }
  /** Decode the slices around i ahead of time, so stepping and playing do not wait on decoding. */
  function prefetch(i) {
    for (const k of [i + 1, i - 1, i + 2, i + 3]) {
      if (k >= 0 && k < stackN()) { decodeSlice(stack.mov[k]); decodeSlice(stack.fix[k]); }
    }
  }
  /** The preprocessing that a cached work image was made with. */
  const prepKey = () => JSON.stringify(["clahe", "band", "invert", "half"].map((id) => !!$(id)?.checked).concat([$("bp-fine")?.value, $("bp-coarse")?.value]));
  /** One side of a slice into the engine and the views; the preprocessed work image is kept with the slice. */
  async function loadSide(which, img) {
    await zc.setImage(which === "left" ? 0 : 1, img.rgba, img.w, img.h);
    const key = prepKey();
    if (img.baked && img.bakedKey === key) { state[which] = img.baked; return; }
    state[which] = { ...img, srcRgba: img.rgba };
    await bakeSide(which);
    img.baked = state[which];
    img.bakedKey = key;
    trimCache();
  }

  function setMode(mode) {
    if ($("app").dataset.mode === mode) return;
    if (mode === "pair") { stack.loaded = false; stopPlay(); }
    fillPresets(mode);
    $("app").dataset.mode = mode;
    const r = document.querySelector(`input[name="mode"][value="${mode}"]`);
    if (r) r.checked = true;
    if (mode === "stack") {
      openCard("stack");
      if (stackN()) navTo(Math.max(0, stack.i));
    } else if (dockName() === "stack") setDock("kp");
    syncStackUi();
    requestAnimationFrame(() => onStageChange());
  }
  document.querySelectorAll('input[name="mode"]').forEach((r) => {
    r.onchange = () => {
      if (job.kind || stackRun.active) { document.querySelector(`input[name="mode"][value="${$("app").dataset.mode}"]`).checked = true; return; }
      setMode(r.value);
    };
  });

  function setStack(which, items) {
    stack[which] = items.slice().sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }));
    if (which === "mov") makeThumbs();
    stack.res = [];
    stack.i = -1;
    stack.loaded = false;
    stack.cache.clear();
    const el = $(which === "mov" ? "stack-name-left" : "stack-name-right");
    const list = stack[which];
    el.textContent = list.length ? `${list.length} images · ${list[0].name} …` : "no stack";
    el.title = list.map((x) => x.name).join("\n");
    el.classList.toggle("set", list.length > 0);
    if (stack.mov.length && stack.fix.length && stack.mov.length !== stack.fix.length) {
      log(`the stacks differ in length (${stack.mov.length} moving, ${stack.fix.length} fixed): using the first ${stackN()} of each`);
    }
    syncStackUi();
  }
  const openStack = async (which, input) => {
    setStack(which, [...input.files].filter((f) => f.type.startsWith("image/")).map((f) => ({ name: f.name, file: f })));
    input.value = "";
    if (stackN()) await navTo(0);
  };
  $("btn-stack-left").onclick = () => $("file-stack-left").click();
  $("btn-stack-right").onclick = () => $("file-stack-right").click();
  $("file-stack-left").onchange = () => openStack("mov", $("file-stack-left"));
  $("file-stack-right").onchange = () => openStack("fix", $("file-stack-right"));

  /** The current slice's pose into its record (a hand edit after registration becomes "edited"). */
  async function commitSlice() {
    const i = stack.i, r = stack.res[i];
    if (!stack.loaded || i < 0 || !r || !state.left || !state.right) return;
    const H = Float64Array.from(currentH());
    const cps = $("ffd").checked && state.cps ? Float32Array.from(state.cps) : null;
    const cpsSame = (!cps && !r.cps) || (cps && r.cps && cps.length === r.cps.length && cps.every((v, k) => v === r.cps[k]));
    if (sameH(r.H, H) && cpsSame) return;
    stack.res[i] = { ...r, H, cps, score: sliceScore(await zc.score(H)), method: "edited", mw: state.left.w, mh: state.left.h };
  }

  /** Show slice pair i at its registered pose (identity if not registered yet). Only what belongs to
   *  the slice changes: the side view, the zoom and the panel layout stay as they are, and the
   *  heavier refresh (maps, score history) waits until the slice stays put (settle). */
  async function goToSlice(i, settle = true) {
    const n = stackN();
    if (!n) return;
    i = Math.max(0, Math.min(n - 1, i));
    await commitSlice();
    const [a, b] = await Promise.all([decodeSlice(stack.mov[i]), decodeSlice(stack.fix[i])]);
    stack.i = i;
    // the previous slice's keypoints, matches, score history and map preview do not apply
    state.trail = [];
    state.trailSel = -1;
    state.kpsL = state.kpsR = state.matchJ = state.matchJRaw = state.matchRev = state.inlier = state.kpSel = state.kpHover = null;
    state.matchScore = state._scoreRange = null;
    state.n1 = state.n2 = 0;
    state.pg = null;
    state.preview = null;
    state.mapMark = null;
    state.rsCenter = null;
    await zcSync();
    await loadSide("left", a);
    await loadSide("right", b);
    stack.loaded = true;
    const app = $("app");
    app.dataset.images = "2";
    app.dataset.left = app.dataset.right = "1";
    state.stem = `slice_${String(i + 1).padStart(3, "0")}`;
    const r = stack.res[i];
    state.H0 = r ? Float64Array.from(r.H) : identityH();
    zeroTan();
    if (state.cps) {
      if (r?.cps && r.cps.length === state.cps.length) state.cps.set(r.cps); else state.cps.fill(0);
      state.cpsStart = Float32Array.from(state.cps);
      state.cpsGrad = null;
    }
    state.poseClimbed = !!r;
    state._ns0 = (await zc.score(identityH())).fwd;
    writeMatchStats(currentH());
    showImages();
    drawTrail();
    await applyWarp(false);
    markNext();
    syncStackUi();
    prefetch(i);
    if (settle) settleSoon();
  }
  let settleTimer = 0;
  /** Maps and the score history for the slice on screen, once the user stops moving through the stack. */
  function settleSoon() {
    clearTimeout(settleTimer);
    settleTimer = setTimeout(() => {
      if (job.kind || stackRun.active || stack.nav || playing) return settleSoon();
      applyWarp(true);
    }, 220);
  }
  /** Slider / keys / rows / playback: go to a slice, coalescing requests that arrive while one loads. */
  async function navTo(i, settle = true) {
    if (!stackN() || job.kind || stackRun.active) return;
    stack.pending = Math.max(0, Math.min(stackN() - 1, i));
    if (stack.nav) return;
    stack.nav = true;
    try {
      while (stack.pending != null) {
        const t = stack.pending;
        stack.pending = null;
        if (t !== stack.i || !stack.loaded) await goToSlice(t, settle);
      }
    } finally { stack.nav = false; }
  }
  const userNav = (i) => { stopPlay(); navTo(i); };

  // Playback: step through the stack at the chosen speed (loops); any other navigation stops it.
  let playing = false;
  function stopPlay() { playing = false; $("btn-slice-play").classList.remove("on"); }
  async function togglePlay() {
    if (playing) { stopPlay(); settleSoon(); return; }
    if (!stackN() || job.kind || stackRun.active) return;
    playing = true;
    $("btn-slice-play").classList.add("on");
    while (playing && stackN()) {
      const t0 = performance.now();
      await navTo((stack.i + 1) % stackN(), false);
      const wait = 1000 / (+$("slice-fps").value || 5) - (performance.now() - t0);
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    }
    settleSoon();
  }
  $("btn-slice-play").onclick = togglePlay;

  /** One slice pair from keypoints (Auto) or from the previous slice's transform (refine only). */
  async function registerSlice(i, how, start = null) {
    await goToSlice(i, false);
    const spline = $("ffd").checked;
    if (how === "carry") {
      state.H0 = Float64Array.from(start || stack.res[i - 1].H);
      zeroTan();
      if (state.cps) { state.cps.fill(0); applyFfd(); }
      await runGrad({ global: spline });
      if (spline && !stackRun.stop) await runGrad();
    } else {
      await runAuto();
    }
    const H = Float64Array.from(currentH());
    return { H, cps: spline && state.cps ? Float32Array.from(state.cps) : null, score: sliceScore(await zc.score(H)), method: how, mw: state.left.w, mh: state.left.h };
  }
  async function showResult(r) {
    state.H0 = Float64Array.from(r.H);
    zeroTan();
    if (state.cps) { if (r.cps && r.cps.length === state.cps.length) state.cps.set(r.cps); else state.cps.fill(0); applyFfd(); }
    await applyWarp(true);
  }

  // ── consistency: how far a slice's pose is from its neighbours' ──
  const mapPt = (H, x, y) => { const w = H[6] * x + H[7] * y + H[8]; return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w]; };
  const cornersOf = (r) => [[1, 1], [r.mw, 1], [r.mw, r.mh], [1, r.mh]];
  /** Largest corner distance (fixed px) between pose H and the reference poses' mean, on r's corners. */
  function poseGap(r, H, refs) {
    if (!refs.length) return null;
    let m = 0;
    for (const [x, y] of cornersOf(r)) {
      const [px, py] = mapPt(H, x, y);
      let qx = 0, qy = 0;
      for (const R of refs) { const [ax, ay] = mapPt(R, x, y); qx += ax / refs.length; qy += ay / refs.length; }
      m = Math.max(m, Math.hypot(px - qx, py - qy));
    }
    return m;
  }
  /** Off-trend error of slice k at pose H: the corners' distance from where the neighbours put
   *  them — their mean between two, a straight-line extrapolation at the ends (a stack that drifts
   *  steadily flags nothing). */
  const offTrend = (k, H) => {
    const r = stack.res[k];
    if (!r || !r.mw) return null;
    const a = stack.res[k - 1], b = stack.res[k + 1];
    if (a && b) return poseGap(r, H, [a.H, b.H]);
    for (const [near, far] of [[a, stack.res[k - 2]], [b, stack.res[k + 2]]]) {
      if (!near) continue;
      if (!far) return poseGap(r, H, [near.H]);
      // the extrapolated pose 2·near − far, on the corners
      let m = 0;
      for (const [x, y] of cornersOf(r)) {
        const [px, py] = mapPt(H, x, y), [nx, ny] = mapPt(near.H, x, y), [fx, fy] = mapPt(far.H, x, y);
        m = Math.max(m, Math.hypot(px - (2 * nx - fx), py - (2 * ny - fy)));
      }
      return m;
    }
    return null;
  };
  /** Per slice: off-trend error, the threshold (the user's, or 4× the median, at least 2 px), flags. */
  function trend() {
    const n = stackN();
    const e = Array.from({ length: n }, (_, k) => (stack.res[k] ? offTrend(k, stack.res[k].H) : null));
    const vals = e.filter((v) => v != null).sort((a, b) => a - b);
    const med = vals.length ? vals[vals.length >> 1] : 0;
    const raw = $("stack-jump").value.trim();
    const auto = raw === "" || !Number.isFinite(+raw);
    const thr = auto ? Math.max(2, 4 * med) : Math.max(0, +raw);
    return { e, thr, auto, flag: e.map((v) => v != null && vals.length >= 3 && v > thr) };
  }
  /** The jump threshold during a run: the user's, or 4× the median jump so far (at least 2 px; none before 3 jumps). */
  function jumpThreshold(jumps) {
    const raw = $("stack-jump").value.trim();
    if (raw !== "" && Number.isFinite(+raw)) return Math.max(0, +raw);
    if (jumps.length < 3) return Infinity;
    const v = jumps.slice().sort((a, b) => a - b);
    return Math.max(2, 4 * v[v.length >> 1]);
  }

  async function runStack(from = 0) {
    const n = stackN();
    if (!n || job.kind || stackRun.active) return;
    const every = radioValue("stack-init") === "keypoints";
    const drop = Math.max(0, num("stack-drop", 15)) / 100;
    Object.assign(stackRun, { active: true, stop: false });
    const t0 = performance.now();
    let done = 0, redetected = 0;
    // jumps between consecutive slices already registered (the automatic threshold's history)
    const jumps = [];
    for (let k = 1; k < from; k++) if (stack.res[k] && stack.res[k - 1] && stack.res[k].mw) jumps.push(poseGap(stack.res[k], stack.res[k].H, [stack.res[k - 1].H]));
    try {
      for (let i = from; i < n && !stackRun.stop; i++) {
        stackRun.msg = `slice ${i + 1}/${n}`;
        const prev = stack.res[i - 1];
        const how = every || !prev ? "keypoints" : "carry";
        let r = await registerSlice(i, how);
        if (stackRun.stop) break;
        const jump = prev ? poseGap(r, r.H, [prev.H]) : null;
        const dropped = how === "carry" && prev.score > 0 && r.score < prev.score * (1 - drop);
        const jumped = how === "carry" && jump != null && jump > jumpThreshold(jumps);
        if (dropped || jumped) {
          log(`slice ${i + 1}: ${dropped ? `${fmtScore(r.score)} is ${(100 * (1 - r.score / prev.score)).toFixed(0)}% below slice ${i}` : ""}` +
            `${dropped && jumped ? ", and " : ""}${jumped ? `the pose jumped ${jump.toFixed(1)} px` : ""}: trying keypoints`);
          const carried = r;
          const k = await registerSlice(i, "keypoints");
          if (stackRun.stop) break;
          r = { ...(k.score > carried.score ? k : carried), redetected: true };
          if (k.score <= carried.score) await showResult(carried);
          redetected++;
        }
        stack.res[i] = r;
        if (prev) jumps.push(poseGap(r, r.H, [prev.H]));
        done++;
        syncStackUi();
      }
      // backward pass: every slice of this run again from the next slice's pose; the better score stays
      if ($("stack-polish").checked && !stackRun.stop) {
        let better = 0;
        for (let i = Math.min(n - 2, stack.res.length - 2); i >= from && !stackRun.stop; i--) {
          const cur = stack.res[i], next = stack.res[i + 1];
          if (!cur || !next) continue;
          stackRun.msg = `polishing slice ${i + 1}`;
          const r = await registerSlice(i, "carry", next.H);
          if (stackRun.stop) break;
          if (r.score > cur.score * (1 + 1e-4)) {  // a real gain, not round-off
            stack.res[i] = { ...r, polished: true, redetected: cur.redetected };
            better++;
            log(`slice ${i + 1}: ${fmtScore(cur.score)} → ${fmtScore(r.score)} from slice ${i + 2}'s pose`);
          } else await showResult(cur);
          syncStackUi();
        }
        log(`polish: ${better} slice${better === 1 ? "" : "s"} improved`);
      }
    } finally {
      Object.assign(stackRun, { active: false, msg: "" });
      setBusy(false);
      syncStackUi();
      log(`stack: ${done} slice${done === 1 ? "" : "s"} in ${((performance.now() - t0) / 1000).toFixed(1)} s${redetected ? `, ${redetected} re-detected` : ""}${stackRun.stop ? " (stopped)" : ""}`);
    }
  }
  async function redoSlice(how) {
    const i = stack.i;
    if (i < 0 || job.kind || stackRun.active) return;
    if (how === "carry" && !stack.res[i - 1]) { log("the previous slice has no transform yet"); return; }
    Object.assign(stackRun, { active: true, stop: false, msg: `slice ${i + 1}` });
    try {
      const r = await registerSlice(i, how);
      if (!stackRun.stop) stack.res[i] = r;
    } finally {
      Object.assign(stackRun, { active: false, msg: "" });
      setBusy(false);
      syncStackUi();
    }
  }
  /** Flagged slices (off the trend of their neighbours) again from keypoints; the keypoint pose is
   *  kept when it fits its neighbours better. */
  async function redoFlagged() {
    if (job.kind || stackRun.active) return;
    await commitSlice();
    const ks = trend().flag.map((f, k) => (f ? k : -1)).filter((k) => k >= 0);
    if (!ks.length) { log("no flagged slices"); return; }
    Object.assign(stackRun, { active: true, stop: false });
    let fixed = 0;
    try {
      for (const k of ks) {
        if (stackRun.stop) break;
        stackRun.msg = `flagged slice ${k + 1}`;
        const old = stack.res[k], before = offTrend(k, old.H);
        const r = await registerSlice(k, "keypoints");
        if (stackRun.stop) break;
        const after = offTrend(k, r.H);
        const better = after != null && after < 0.9 * before;  // clearly closer to its neighbours
        if (better) { stack.res[k] = { ...r, redetected: true }; fixed++; }
        else await showResult(old);
        log(`slice ${k + 1}: off trend ${before.toFixed(1)} px → keypoints ${after == null ? "—" : after.toFixed(1)} px: ${better ? "kept the keypoints pose" : "kept the previous pose"}`);
        syncStackUi();
      }
    } finally {
      Object.assign(stackRun, { active: false, msg: "" });
      setBusy(false);
      syncStackUi();
      log(`redo flagged: ${fixed} of ${ks.length} improved`);
    }
  }
  $("btn-stack-flagged").onclick = redoFlagged;
  $("btn-stack-run").onclick = () => runStack(0);
  $("btn-stack-from").onclick = () => runStack(Math.max(0, stack.i));
  $("btn-slice-kp").onclick = () => redoSlice("keypoints");
  $("btn-slice-carry").onclick = () => redoSlice("carry");
  $("btn-slice-prev").onclick = () => userNav(stack.i - 1);
  $("btn-slice-next").onclick = () => userNav(stack.i + 1);
  $("slice-slider").addEventListener("input", () => userNav(+$("slice-slider").value - 1));
  // the wheel over the timeline steps through the slices (trackpads: one step per ~40 px)
  {
    let acc = 0;
    $("stack-bar").addEventListener("wheel", (ev) => {
      if (!stackN()) return;
      ev.preventDefault();
      acc += Math.abs(ev.deltaY) >= Math.abs(ev.deltaX) ? ev.deltaY : ev.deltaX;
      const steps = Math.trunc(acc / 40);
      if (!steps) return;
      acc = 0;
      userNav(Math.max(0, Math.min(stackN() - 1, (stack.pending ?? stack.i) + Math.sign(steps))));
    }, { passive: false });
  }
  // Overlay | Moving | Fixed: either stack alone (the moving one warped into the fixed frame), in one frame
  const VIEW_MODE = { moving: "warp", fixed: "fixed" };
  let viewMode = "soft";
  function setStackView(view) {
    const sel = $("reg-mode");
    const target = VIEW_MODE[view] || (["warp", "fixed"].includes(viewMode) ? "soft" : viewMode);
    if (!["warp", "fixed"].includes(sel.value)) viewMode = sel.value;
    if (sel.value === target) return;
    sel.value = target;
    sel.dispatchEvent(new Event("change"));
  }
  document.querySelectorAll('input[name="stack-view"]').forEach((r) => { r.onchange = () => setStackView(r.value); });
  $("reg-mode").addEventListener("change", () => {
    const v = $("reg-mode").value;
    const view = v === "warp" ? "moving" : v === "fixed" ? "fixed" : "overlay";
    const radio = document.querySelector(`input[name="stack-view"][value="${view}"]`);
    if (radio) radio.checked = true;
  });
  for (const id of ["stack-drop", "stack-jump"]) $(id).addEventListener("input", syncStackUi);
  $("stack-thumbs").onchange = () => {
    $("app").dataset.thumbs = $("stack-thumbs").checked ? "on" : "off";
    requestAnimationFrame(() => { drawStack(); onStageChange(); });
  };
  document.querySelectorAll('input[name="stack-init"]').forEach((r) => { r.onchange = syncStackUi; });
  window.addEventListener("keydown", (ev) => {
    if (!stackOn() || typingTarget(ev.target)) return;
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const free = !tweakOn();  // arrows edit the pose while edit pose is on
    if (ev.key === "PageDown" || (free && ev.key === "ArrowRight")) { ev.preventDefault(); userNav(stack.i + 1); }
    else if (ev.key === "PageUp" || (free && ev.key === "ArrowLeft")) { ev.preventDefault(); userNav(stack.i - 1); }
    else if (ev.key === "Home") { ev.preventDefault(); userNav(0); }
    else if (ev.key === "End") { ev.preventDefault(); userNav(stackN() - 1); }
    else if (ev.key === " " && free) { ev.preventDefault(); togglePlay(); }
    else if (free && "omf".includes(ev.key.toLowerCase()) && ev.key.length === 1) {
      ev.preventDefault();
      const view = { o: "overlay", m: "moving", f: "fixed" }[ev.key.toLowerCase()];
      document.querySelector(`input[name="stack-view"][value="${view}"]`).checked = true;
      setStackView(view);
    }
  });

  function transformsJson() {
    const t = trend();
    return JSON.stringify({
      convention: "H maps moving pixel coordinates to fixed pixel coordinates, row-major 3x3, 1-based pixel centres",
      model: state.group,
      off_trend_threshold_px: t.thr,
      slices: Array.from({ length: stackN() }, (_, i) => {
        const r = stack.res[i];
        return {
          index: i + 1, moving: stack.mov[i].name, fixed: stack.fix[i].name,
          H: r ? Array.from(r.H) : null, score: r ? r.score : null, method: r ? r.method : null, redetected: !!r?.redetected,
          off_trend_px: t.e[i], flagged: t.flag[i],
          bspline: r?.cps ? { lattice: Math.round(Math.sqrt(r.cps.length / 2)), frame: $("ffd-frame")?.value || "target", coefficients: Array.from(r.cps) } : null,
        };
      }),
    }, null, 1);
  }
  $("btn-stack-export").onclick = async () => {
    await commitSlice();
    downloadBlob(new Blob([transformsJson()], { type: "application/json" }), "stack_transforms.json");
  };

  /** RGBA of the moving image resampled on the fixed grid through field (x_moving − x_fixed, 0-based);
   *  bilinear, transparent where the moving image does not reach. */
  function warpRgba(src, sw, sh, field, fw, fh) {
    const out = new Uint8ClampedArray(fw * fh * 4);
    const xm = sw - 1, ym = sh - 1;
    for (let y = 0, k = 0; y < fh; y++) {
      for (let x = 0; x < fw; x++, k++) {
        const mx = x + field[2 * k], my = y + field[2 * k + 1];
        if (!(mx > -0.5 && my > -0.5 && mx < sw - 0.5 && my < sh - 0.5)) continue;
        const cx = Math.min(Math.max(mx, 0), xm), cy = Math.min(Math.max(my, 0), ym);
        const x0 = Math.min(Math.floor(cx), Math.max(xm - 1, 0)), y0 = Math.min(Math.floor(cy), Math.max(ym - 1, 0));
        const fx = cx - x0, fy = cy - y0;
        const x1 = Math.min(x0 + 1, xm), y1 = Math.min(y0 + 1, ym);
        const a = (y0 * sw + x0) * 4, b = (y0 * sw + x1) * 4, c = (y1 * sw + x0) * 4, d = (y1 * sw + x1) * 4;
        const o = 4 * k;
        for (let ch = 0; ch < 3; ch++) {
          out[o + ch] = (src[a + ch] * (1 - fx) + src[b + ch] * fx) * (1 - fy) + (src[c + ch] * (1 - fx) + src[d + ch] * fx) * fy;
        }
        out[o + 3] = 255;
      }
    }
    return out;
  }
  const stemOf = (name) => name.replace(/\.[^.]*$/, "").replace(/[^\w.-]+/g, "_");

  /** The registered moving stack: each registered slice's original moving image (its colours kept)
   *  resampled into its fixed frame by the engine's displacement field (the affine / homography and
   *  the B-spline), with the fixed stack if asked, and the transforms; one ZIP. Streamed: every page
   *  goes into a Blob as it is made (the browser keeps large Blobs on disk), so the stack is never
   *  in memory at once. */
  async function exportRegistered() {
    if (job.kind || stackRun.active) return;
    await commitSlice();
    const ks = Array.from({ length: stackN() }, (_, k) => k).filter((k) => stack.res[k]);
    if (!ks.length) { log("no registered slices to export"); return; }
    const tif = $("stack-format").value === "tif", withFixed = $("stack-with-fixed").checked;
    const back = stack.i;
    Object.assign(stackRun, { active: true, stop: false });
    const t0 = performance.now();
    const files = [];
    const tiffs = { mov: tif ? new TiffStream() : null, fix: tif && withFixed ? new TiffStream() : null };
    const chans = {};  // per stack: gray (1) or RGB (3), from its first page
    const addPng = async (name, rgba, w, h) => {
      const cv = new OffscreenCanvas(w, h);
      cv.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
      const blob = await cv.convertToBlob({ type: "image/png" });
      files.push({ name, parts: [blob], size: blob.size, crc: crc32(new Uint8Array(await blob.arrayBuffer())) });
    };
    const addPage = (which, rgba, w, h) => {
      chans[which] ??= channelsOf(rgba);
      tiffs[which].addPage(w, h, chans[which], pack(rgba, chans[which]));
    };
    try {
      for (const k of ks) {
        if (stackRun.stop) break;
        stackRun.msg = `exporting slice ${k + 1}`;
        setBusy(true, "", "export");
        await goToSlice(k, false);
        const r = stack.res[k];
        const [m, f] = await Promise.all([decodeSlice(stack.mov[k]), decodeSlice(stack.fix[k])]);
        const g = r.cps ? Math.round(Math.sqrt(r.cps.length / 2)) : 0;
        const field = await zcDo(() => {
          zc.configure({ ...appSettings(), ffd: !!r.cps, ...(g ? { ffd_grid: g } : {}) });
          if (r.cps) zc.setFfd(r.cps);
          zc.pose = Float64Array.from(r.H);
          return zc.displacement(f.w, f.h);
        });
        const warped = warpRgba(m.rgba, m.w, m.h, field, f.w, f.h);
        const num = String(k + 1).padStart(3, "0");
        if (tif) {
          addPage("mov", warped, f.w, f.h);
          if (withFixed) addPage("fix", new Uint8ClampedArray(f.rgba), f.w, f.h);
        } else {
          await addPng(`registered_moving/${num}_${stemOf(stack.mov[k].name)}.png`, warped, f.w, f.h);
          if (withFixed) await addPng(`fixed/${num}_${stemOf(stack.fix[k].name)}.png`, new Uint8ClampedArray(f.rgba), f.w, f.h);
        }
        setBusy(false);
      }
      if (stackRun.stop) { log("export stopped"); return; }
      if (tif) {
        files.push({ name: "registered_moving.tif", ...tiffs.mov.finish() });
        if (withFixed) files.push({ name: "fixed.tif", ...tiffs.fix.finish() });
      }
      files.push({ name: "stack_transforms.json", parts: [new TextEncoder().encode(transformsJson())] });
      downloadBlob(zipStore(files), "registered_stack.zip");
      log(`export: ${ks.length} registered slice${ks.length === 1 ? "" : "s"} as ${tif ? "TIFF stack" : "PNG files"}${withFixed ? " with the fixed stack" : ""} in ${((performance.now() - t0) / 1000).toFixed(1)} s${ks.length < stackN() ? ` (${stackN() - ks.length} not registered, left out)` : ""}`);
    } catch (e) { log(String(e.stack || e)); } finally {
      Object.assign(stackRun, { active: false, msg: "" });
      setBusy(false);
      await zcSync();
      if (back >= 0) await goToSlice(back);
    }
  }
  $("btn-stack-warped").onclick = exportRegistered;

  const METHOD = { keypoints: "keypoints", carry: "carried", edited: "edited" };
  /** Moving-slice thumbnails for the timeline: the browser decodes each straight to ~96 px high, one
   *  at a time in the background; a newer stack cancels the run. */
  stack.thumbs = [];
  stack.gen = 0;
  async function makeThumbs() {
    const gen = ++stack.gen;
    stack.thumbs = [];
    await new Promise((r) => setTimeout(r, 0));  // after setStack has filled the list
    for (let k = 0; k < stack.mov.length; k++) {
      if (gen !== stack.gen) return;
      const item = stack.mov[k];
      try {
        const blob = item.file || await (await fetch(item.url)).blob();
        stack.thumbs[k] = await createImageBitmap(blob, { resizeHeight: 96, resizeQuality: "medium" });
      } catch { /* no thumbnail for this slice */ }
      if (gen !== stack.gen) return;
      if (k % 4 === 3 || k === stack.mov.length - 1) drawStack();
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  const TRACK_PAD = 8;  // the slider thumb's half width: bar k sits above slider value k
  function syncStackUi() {
    const n = stackN(), i = stack.i;
    const done = stack.res.filter(Boolean).length;
    const every = radioValue("stack-init") === "keypoints";
    const drop = num("stack-drop", 15);
    $("slice-pos").textContent = n ? `slice ${Math.max(i, 0) + 1} / ${n}` : "no stack";
    $("slice-names").textContent = n && i >= 0 ? `${stack.mov[i].name} → ${stack.fix[i].name}` : "open a moving and a fixed stack";
    const t = trend();
    const flagged = t.flag.filter(Boolean).length;
    $("res-stack").textContent = n ? `${done} / ${n}${flagged ? ` · ${flagged} flagged` : ""}` : "";
    const sl = $("slice-slider");
    sl.max = String(Math.max(1, n));
    sl.value = String(Math.max(0, i) + 1);
    const jumpTxt = $("stack-jump").value.trim() === "" ? "auto" : `${$("stack-jump").value} px`;
    $("sum-stack").textContent = n ? `${n} slices · ${every ? "keypoints" : `carry · −${drop}% / jump ${jumpTxt}`}` : "open two stacks";
    $("stack-plan").textContent = every ? "keypoints on every slice" : `carry · re-detect at −${drop}% or ${jumpTxt === "auto" ? "a jump" : `${jumpTxt} jump`}`;
    if (!stackRun.active && !job.kind) $("btn-stack-run").disabled = !n;
    drawStack();
    // the slice table (built with textContent: file names are the user's)
    const ol = $("slice-list");
    ol.replaceChildren();
    if (n) {
      const head = document.createElement("li");
      head.className = "head";
      for (const [cls, text, tip] of [["k", "#", ""], ["lab", "moving → fixed", ""], ["v", "score", "the registration score of the slice pair"],
        ["d", "Δ", "change from the previous slice's score"], ["t", "off-trend", "how far the pose is from its neighbours' mean (moving-image corners, fixed px)"], ["m", "how", ""]]) {
        const sp = document.createElement("span");
        sp.className = cls;
        sp.textContent = text;
        if (tip) sp.title = tip;
        head.append(sp);
      }
      ol.append(head);
    }
    for (let k = 0; k < n; k++) {
      const r = stack.res[k], prev = stack.res[k - 1];
      const d = r && prev && prev.score > 0 ? (r.score / prev.score - 1) * 100 : null;
      const li = document.createElement("li");
      li.classList.toggle("sel", k === i);
      const cell = (cls, text) => { const s = document.createElement("span"); s.className = cls; s.textContent = text; li.append(s); return s; };
      cell("k", String(k + 1));
      cell("lab", `${stack.mov[k].name} → ${stack.fix[k].name}`).title = `${stack.mov[k].name} → ${stack.fix[k].name}`;
      cell("v", r ? fmtScore(r.score) : "—");
      const dc = cell("d", d == null ? "" : `${d >= 0 ? "+" : ""}${d.toFixed(1)}%`);
      if (d != null && d < -drop) dc.classList.add("bad");
      else if (d != null && d > 0) dc.classList.add("good");
      const tc = cell("t", t.e[k] == null ? "" : `${t.e[k].toFixed(1)} px`);
      if (t.flag[k]) { tc.classList.add("bad"); li.classList.add("flag"); li.title = `off the trend of its neighbours by more than ${t.thr.toFixed(1)} px`; }
      const how = cell("m", r ? `${r.polished ? "from next" : METHOD[r.method] || r.method}${r.redetected ? " ↻" : ""}` : "pending");
      if (r?.polished) how.title = "the backward pass improved it from the next slice's pose";
      li.onclick = () => userNav(k);
      ol.append(li);
    }
    $("slice-meta").textContent = n ? `${done} of ${n} registered${flagged ? ` · ${flagged} flagged` : ""} · off-trend limit ${Number.isFinite(t.thr) ? t.thr.toFixed(1) : "—"} px${t.auto ? " (auto)" : ""}` : "";
    $("btn-stack-flagged").disabled = stackRun.active || !!job.kind || !flagged;
  }

  // The timeline: one bar per slice (height: score, color: how it was registered), current slice outlined.
  function drawStack() {
    const cv = $("cv-stack");
    if (!cv || !stackOn()) return;
    const dpr = devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const n = stackN();
    const css = getComputedStyle(document.documentElement);
    g.font = "11px ui-sans-serif, system-ui";
    g.fillStyle = css.getPropertyValue("--faint");
    if (!n) { g.textBaseline = "middle"; g.fillText("open a moving and a fixed stack (several images each), or pick the stack example", 4, H / 2); return; }
    const color = { keypoints: css.getPropertyValue("--accent"), carry: css.getPropertyValue("--data"), edited: "#b48ead" };
    const bad = css.getPropertyValue("--bad");
    const max = Math.max(1e-12, ...stack.res.filter(Boolean).map((r) => r.score));
    const bw = (W - 2 * TRACK_PAD) / n, base = H - 11, gap = Math.min(2, bw * 0.15);
    // thumbnails on top (every m-th slice when the cells are narrow), score bars below
    const thumbs = $("app").dataset.thumbs === "on";
    const thumbH = thumbs ? Math.max(0, base - 24) : 0;
    const top = thumbs ? thumbH + 3 : 0;
    if (thumbs) {
      const m = Math.max(1, Math.ceil(16 / bw));
      for (let k = 0; k < n; k += m) {
        const t = stack.thumbs[k], cw = Math.min(m, n - k) * bw - 2;
        const x = TRACK_PAD + k * bw + 1;
        if (!t) { g.fillStyle = "#1d1d21"; g.fillRect(x, 0, cw, thumbH); continue; }
        const s = Math.min(cw / t.width, thumbH / t.height);
        const dw = t.width * s, dh = t.height * s;
        g.drawImage(t, x + (cw - dw) / 2, (thumbH - dh) / 2, dw, dh);
      }
    }
    const t = trend();
    for (let k = 0; k < n; k++) {
      const r = stack.res[k], x = TRACK_PAD + k * bw;
      const h = r ? Math.max(2, (base - top - 3) * Math.max(0, r.score) / max) : 3;
      g.fillStyle = r ? color[r.method] || "#888" : "#3a3a40";
      g.fillRect(x + gap, base - h, Math.max(1, bw - 2 * gap), h);
      if (r?.redetected) { g.fillStyle = bad; g.beginPath(); g.arc(x + bw / 2, Math.max(top + 2.5, base - h - 4), 2.2, 0, 2 * Math.PI); g.fill(); }
      if (t.flag[k]) { g.fillStyle = bad; g.fillRect(x + gap, base + 1, Math.max(1, bw - 2 * gap), 2); }
    }
    if (stack.i >= 0) {
      g.strokeStyle = "#e6e6e8";
      g.lineWidth = 1.5;
      g.strokeRect(TRACK_PAD + stack.i * bw + 0.75, 0.75, bw - 1.5, base + 2.5);
    }
    g.fillStyle = css.getPropertyValue("--faint");
    g.textBaseline = "alphabetic";
    const every = Math.max(1, Math.ceil(28 / bw));
    g.textAlign = "center";
    for (let k = 0; k < n; k += every) g.fillText(String(k + 1), TRACK_PAD + (k + 0.5) * bw, H - 1);
    g.textAlign = "start";
  }
  {
    const cv = $("cv-stack");
    const at = (ev) => Math.max(0, Math.min(stackN() - 1, Math.floor((ev.clientX - cv.getBoundingClientRect().left - TRACK_PAD) / ((cv.clientWidth - 2 * TRACK_PAD) / Math.max(1, stackN())))));
    cv.addEventListener("pointerdown", (ev) => { cv.setPointerCapture(ev.pointerId); userNav(at(ev)); });
    cv.addEventListener("pointermove", (ev) => {
      if (ev.buttons & 1) { userNav(at(ev)); return; }
      if (!stackN()) return;
      const k = at(ev), r = stack.res[k];
      cv.title = `slice ${k + 1}: ${stack.mov[k].name} → ${stack.fix[k].name}\n` +
        (r ? `${fmtScore(r.score)} · ${METHOD[r.method] || r.method}${r.redetected ? " (keypoints tried)" : ""}` : "not registered yet") +
        (trend().flag[k] ? "\nflagged: off the trend of its neighbours" : "") +
        "\nbars: gold keypoints · blue carried · purple edited · grey not yet · red underline: flagged";
    });
    window.addEventListener("resize", drawStack);
  }

  // Programmatic stacks (tests, the console): lists of URLs or Files / Blobs, slice order as given
  // (natural sort by name, as for opened files).
  Object.assign(window.cmir, {
    async openStack(moving, fixed) {
      setMode("stack");
      const items = (list) => list.map((x, k) => (typeof x === "string"
        ? { name: `${String(k + 1).padStart(4, "0")}_${x.split("/").pop()}`, url: x }
        : { name: x.name || `${String(k + 1).padStart(4, "0")}_slice`, file: x }));
      setStack("mov", items(moving));
      setStack("fix", items(fixed));
      await navTo(0);
    },
    registerStack: (from = 0) => runStack(from),
    stackResults: () => ({
      n: stackN(), i: stack.i, trend: trend(),
      slices: stack.res.map((r) => r && { H: Array.from(r.H), score: r.score, method: r.method, redetected: !!r.redetected }),
    }),
    exportStack: exportRegistered,
  });

  async function loadStackPreset(p) {
    setMode("stack");
    if (p.model) setGroup(p.model);
    const items = (list) => list.map((u) => ({ name: u.split("/").pop(), url: `./presets/${u}` }));
    setStack("mov", items(p.moving));
    setStack("fix", items(p.fixed));
    await navTo(0);
    log(`example stack: ${p.name} · ${stackN()} slice pairs · press Register stack`);
  }
  syncStackUi();

  // Example pairs (presets/presets.json; sources and licenses in presets/ATTRIBUTION.md).
  try {
    const list = (await (await fetch("./presets/presets.json")).json()).presets;
    presetList = list;
    fillPresets($("app").dataset.mode);
    const sel = $("preset");
    sel.onchange = async () => {
      const p = list.find((x) => x.id === sel.value);
      sel.value = "";
      if (!p || job.kind || stackRun.active) return;
      if (p.type === "stack") { await loadStackPreset(p); return; }
      setMode("pair");
      const load = async (url) => {
        const img = await fileToRgba(await (await fetch(`./presets/${url}`)).blob(), false);
        return Object.assign(img, { name: url.split("/")[0] + "/" + url.split("/")[1] });
      };
      setBusy(true, "loading example…");
      try {
        const [a, b] = await Promise.all([load(p.moving), load(p.fixed)]);
        setBusy(false);
        if (p.model) setGroup(p.model);
        resetPairState();
        state.stem = p.id;
        state.truth = p.truth || null;
        await ingest("left", a);
        await ingest("right", b);
        log(`example ${p.group} · ${p.name}${p.truth ? "  (ground truth known)" : ""} · press Auto`);
      } catch (e) { setBusy(false); log(String(e.stack || e)); }
    };
  } catch { $("preset").hidden = true; }
  log("open or drop a moving and a fixed image (two files dropped together: moving, then fixed), or pick an example");
}

main().catch((e) => {
  const msg = String(e.stack || e);
  log(msg);
  $("adapter").textContent = "failed";
  console.error(e);
});
