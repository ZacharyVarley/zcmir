/** Pose nudges from keyboard, wheel, and pointer — dest-pixel units. */

export function typingTarget(el) {
  if (!el || !el.tagName) return false;
  const t = el.tagName;
  if (t === "INPUT" || t === "TEXTAREA" || t === "SELECT") return true;
  return !!el.isContentEditable;
}

export function geoAmps(a0, ratio, n) {
  const o = [];
  let a = +a0 || 0.08;
  const r = Math.min(0.95, Math.max(0.2, +ratio || 0.5));
  const k = Math.max(2, Math.min(12, n | 0));
  for (let i = 0; i < k; i++) { o.push(a); a *= r; }
  return o;
}

const KEY = {
  ArrowLeft: ["tx", -1], ArrowRight: ["tx", 1],
  ArrowUp: ["ty", -1], ArrowDown: ["ty", 1],
  q: ["th", -0.015], e: ["th", 0.015], Q: ["th", -0.015], E: ["th", 0.015],
  w: ["sg", 0.02], s: ["sg", -0.02], W: ["sg", 0.02], S: ["sg", -0.02],
  a: ["al", -0.02], d: ["al", 0.02], A: ["al", -0.02], D: ["al", 0.02],
  z: ["ga", -0.02], x: ["ga", 0.02], Z: ["ga", -0.02], X: ["ga", 0.02],
  ",": ["px", -0.002], ".": ["px", 0.002],
  "[": ["py", -0.002], "]": ["py", 0.002],
};

export function keyNudge(ev, group) {
  const spec = KEY[ev.key];
  if (!spec) return null;
  if ((spec[0] === "px" || spec[0] === "py") && group !== "homography") return null;
  let k = 1;
  if (ev.shiftKey) k = 4;
  if (ev.altKey) k = 0.25;
  return { [spec[0]]: spec[1] * k };
}

export function wheelNudge(ev) {
  const d = Math.sign(ev.deltaY) || 0;
  if (!d) return null;
  if (ev.shiftKey) return { sg: -d * 0.025 };
  if (ev.altKey) return { al: d * 0.02 };
  if (ev.ctrlKey || ev.metaKey) return { tx: d * 4 };
  return { th: d * 0.02 };
}

export function dragNudge(dx, dy, ev, group) {
  if (ev.buttons === 2 || ev.ctrlKey || ev.metaKey) {
    if (group === "homography") return { px: dx * 0.00015, py: dy * 0.00015 };
    return { al: dx * 0.0025, ga: dy * 0.0025 };
  }
  if (ev.shiftKey) return { th: dx * 0.004 };
  if (ev.altKey) return { sg: -dy * 0.003 };
  return { tx: dx, ty: dy };
}

export function applyNudge(tan, delta) {
  for (const k of Object.keys(delta)) tan[k] = (tan[k] || 0) + delta[k];
}

export const HELP = [
  "Keys and mouse",
  "  steps        D detect · G refine · H brute force · Esc stop · Ctrl+Z undo",
  "  views        1–4 side tabs (keypoints, maps, score, brute force), 5 slices · L log",
  "  stack        ← → or PageUp / PageDown slices · Home / End · Space play · O / M / F overlay, moving, fixed",
  "  overlay      wheel zoom · drag pan · double-click fit",
  "  edit pose    (turn on 'edit pose' in the overlay bar)",
  "    keys       arrows translate · Q/E rotate · W/S scale · A/D, Z/X shear · , . [ ] perspective",
  "    mouse      drag translate · shift-drag rotate · alt-drag scale · ctrl-drag shear / perspective",
  "               wheel rotate · shift-wheel scale",
].join("\n");
