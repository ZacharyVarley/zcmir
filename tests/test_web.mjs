#!/usr/bin/env node
// The web app end to end in headless Chrome (WebGPU), driven over the DevTools protocol with no
// npm packages (Node 22+: the built-in WebSocket).
//
//   node tests/test_web.mjs                        the repo's app (zig build wasm first)
//   node tests/test_web.mjs --dir dist/web         a built site
//   CHROME=/path/to/chrome node tests/test_web.mjs
//   node tests/test_web.mjs --dir dist/web --save-pipelines dist/web/zcmir/pipelines.json
//                                                  … and record the pipelines it compiled: the
//                                                  site's first-visit warm-up (web/zcmir.js)
//
// A pair: the Landsat example (known ground truth) → Auto → the pose against the truth.
// A stack: two IN718 sections as a stack → Register stack → the export, whose ZIP entries'
// checksums and TIFF page counts are verified. Fails on any error the page logs.
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const dir = args.includes("--dir") ? args[args.indexOf("--dir") + 1] : null;
const savePipelines = args.includes("--save-pipelines") ? args[args.indexOf("--save-pipelines") + 1] : null;
const port = 4300 + Math.floor(Math.random() * 500);
const appUrl = `http://127.0.0.1:${port}/${dir ? "" : "web/app/"}`;

function chromePath() {
  if (process.env.CHROME) return process.env.CHROME;
  const c = {
    darwin: ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Chromium.app/Contents/MacOS/Chromium"],
    win32: ["C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"],
    linux: ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser"],
  }[process.platform] || [];
  const found = c.find((p) => existsSync(p));
  if (!found) throw new Error("no Chrome found; set CHROME");
  return found;
}

const failures = [];
const check = (name, ok, info = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${info ? "  " + info : ""}`);
  if (!ok) failures.push(name);
};

// ── the static server and the browser ──
const server = spawn(process.execPath, [join(ROOT, "scripts", "serve.mjs"), "--port", String(port), ...(dir ? ["--dir", dir] : [])], { stdio: "ignore" });
const profile = mkdtempSync(join(tmpdir(), "zcmir-chrome-"));
const flags = [
  "--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--no-first-run", "--no-default-browser-check",
  "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan", "--window-size=1400,900",
];
const chrome = spawn(chromePath(), [...flags, "about:blank"], { stdio: ["ignore", "ignore", "pipe"] });
const cleanup = () => {
  try { chrome.kill(); } catch { /* gone */ }
  try { server.kill(); } catch { /* gone */ }
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* locked */ }
};
process.on("exit", cleanup);

const wsUrl = await new Promise((res, rej) => {
  let err = "";
  const t = setTimeout(() => rej(new Error("Chrome did not start: " + err.slice(-500))), 30000);
  chrome.stderr.on("data", (d) => {
    err += d;
    const m = /DevTools listening on (ws:\/\/\S+)/.exec(err);
    if (m) { clearTimeout(t); res(m[1]); }
  });
});

// ── a minimal DevTools protocol client ──
const ws = new WebSocket(wsUrl);
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
let seq = 0;
const pending = new Map();
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    const { res, rej } = pending.get(m.id);
    pending.delete(m.id);
    if (m.error) rej(new Error(m.error.message)); else res(m.result);
  }
};
const send = (method, params = {}, sessionId) => new Promise((res, rej) => {
  const id = ++seq;
  pending.set(id, { res, rej });
  ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
});
const { targetId } = await send("Target.createTarget", { url: "about:blank" });
const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
const evaluate = async (expr, timeout = 300000) => {
  const r = await send("Runtime.evaluate", { expression: `(async () => { ${expr} })()`, awaitPromise: true, returnByValue: true, timeout }, sessionId);
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
};
await send("Runtime.enable", {}, sessionId);
await send("Page.navigate", { url: appUrl }, sessionId);

try {
  // the engine is up (or failed)
  const ready = await evaluate(`
    for (let i = 0; i < 600; i++) {
      const log = document.getElementById("log")?.textContent || "";
      if (/WebGPU ready/.test(log) || /error|no WebGPU|not available/i.test(document.getElementById("adapter")?.textContent || "")) break;
      await new Promise((r) => setTimeout(r, 100));
    }
    return { adapter: document.getElementById("adapter").textContent, log: document.getElementById("log").textContent.slice(-600) };`);
  check("the app starts on WebGPU", /WebGPU ready/.test(ready.log), `adapter: ${ready.adapter}`);
  if (!/WebGPU ready/.test(ready.log)) throw new Error("no WebGPU: " + ready.log);

  // a pair with ground truth: the Landsat example, Auto
  const pair = await evaluate(`
    const sel = document.getElementById("preset");
    sel.value = "landsat-b";
    sel.dispatchEvent(new Event("change"));
    for (let i = 0; i < 300 && !(window.cmir.state.left && window.cmir.state.right && window.cmir.state.truth); i++) await new Promise((r) => setTimeout(r, 50));
    const t0 = performance.now();
    await window.cmir.runAuto();
    const H = window.cmir.currentH(), T = window.cmir.state.truth.flat();
    const P = (M, x, y) => { const w = M[6] * x + M[7] * y + M[8]; return [(M[0] * x + M[1] * y + M[2]) / w, (M[3] * x + M[4] * y + M[5]) / w]; };
    const w = window.cmir.state.left.w, h = window.cmir.state.left.h;
    let e = 0;
    for (const [x, y] of [[1, 1], [w, 1], [w, h], [1, h]]) { const a = P(H, x, y), b = P(T, x, y); e = Math.max(e, Math.hypot(a[0] - b[0], a[1] - b[1])); }
    return { e, s: (performance.now() - t0) / 1000, inl: document.getElementById("sum-inl").textContent };`);
  check("Auto recovers the Landsat example's true pose", pair.e < 2, `${pair.e.toFixed(2)} px from the truth, ${pair.inl}, ${pair.s.toFixed(1)} s`);

  // a stack: two IN718 sections, Register stack, the export
  const st = await evaluate(`
    const { crc32 } = await import("./js/stackio.js");
    const got = [];
    const click = HTMLAnchorElement.prototype.click;
    HTMLAnchorElement.prototype.click = function () { if (this.download) { got.push(this.href); return; } return click.call(this); };
    await window.cmir.openStack(["./presets/in718-65/moving.png", "./presets/in718-85/moving.png"], ["./presets/in718-65/fixed.png", "./presets/in718-85/fixed.png"]);
    const t0 = performance.now();
    await window.cmir.registerStack();
    const res = window.cmir.stackResults();
    const secs = (performance.now() - t0) / 1000;
    await window.cmir.exportStack();
    const b = new Uint8Array(await (await fetch(got[0])).arrayBuffer());
    const v = new DataView(b.buffer);
    let e = b.length - 22;
    while (v.getUint32(e, true) !== 0x06054b50) e--;
    let p = v.getUint32(e + 16, true);
    const entries = [];
    for (let k = 0, n = v.getUint16(e + 10, true); k < n; k++) {
      const len = v.getUint16(p + 28, true), name = new TextDecoder().decode(b.subarray(p + 46, p + 46 + len));
      const crc = v.getUint32(p + 16, true), size = v.getUint32(p + 24, true), lo = v.getUint32(p + 42, true);
      const data = b.subarray(lo + 30 + v.getUint16(lo + 26, true), lo + 30 + v.getUint16(lo + 26, true) + size);
      let pages = 0;
      if (name.endsWith(".tif")) { const dv = new DataView(data.buffer, data.byteOffset); for (let ifd = dv.getUint32(4, true); ifd; ifd = dv.getUint32(ifd + 2 + 12 * dv.getUint16(ifd, true), true)) pages++; }
      entries.push({ name, ok: crc32(data) === crc, pages });
      p += 46 + len;
    }
    // scale: √(area of the moving image's outline under H / its own area)
    const img = window.cmir.state.left;
    const scale = (H) => {
      const P = (x, y) => { const w = H[6] * x + H[7] * y + H[8]; return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w]; };
      const q = [P(1, 1), P(img.w, 1), P(img.w, img.h), P(1, img.h)];
      let a = 0;
      for (let k = 0; k < 4; k++) { const [x0, y0] = q[k], [x1, y1] = q[(k + 1) % 4]; a += x0 * y1 - x1 * y0; }
      return Math.sqrt(Math.abs(a / 2) / ((img.w - 1) * (img.h - 1)));
    };
    return { n: res.n, methods: res.slices.map((s) => s && s.method), scales: res.slices.map((s) => s && scale(s.H)), secs, entries };`);
  check("Register stack: keypoints, then carried", st.n === 2 && st.methods.join() === "keypoints,carry", `${st.methods.join(", ")} in ${st.secs.toFixed(1)} s`);
  check("Register stack: the expected EBSD → BSE scale", st.scales.every((s) => Math.abs(s - 1.16) < 0.02), st.scales.map((s) => s.toFixed(4)).join(", "));
  check("the registered stack export", st.entries.length === 3 && st.entries.every((x) => x.ok) && st.entries.filter((x) => x.pages === 2).length === 2,
    st.entries.map((x) => `${x.name}${x.pages ? ` (${x.pages} pages)` : ""}${x.ok ? "" : " BAD CRC"}`).join(", "));

  const errs = await evaluate(`return document.getElementById("log").textContent.split("\\n").filter((l) => /error|failed|invalid/i.test(l)).slice(0, 5);`);
  check("no errors in the log", errs.length === 0, errs.join(" | "));

  if (savePipelines) {
    const rec = await evaluate(`return window.cmir.zc.warmList();`);
    check("the pipelines it compiled", rec?.list?.length > 0, `${rec?.list?.length} pipelines → ${savePipelines}`);
    if (rec?.list?.length) writeFileSync(savePipelines, JSON.stringify(rec));
  }
} catch (e) {
  check("the browser run", false, String(e.stack || e));
} finally {
  ws.close();
  cleanup();
}
console.log("web", failures.length ? `FAILED: ${failures.join(", ")}` : "OK");
process.exit(failures.length ? 1 : 0);
