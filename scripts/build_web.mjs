#!/usr/bin/env node
// Static web build: dist/web/ is the deployable site (any static host; WebGPU needs HTTPS or
// localhost). No bundler and no npm dependencies: the app is plain ES modules.
//   zig build wasm -Doptimize=ReleaseSmall && node scripts/build_web.mjs
//
// Layout (the same paths the app uses in the repo through scripts/serve.mjs aliases):
//   index.html css/ js/ presets/  web/app
//   zcmir/zcmir.wasm, zcmir.js    zig-out/web, web/zcmir.js (the engine; its WGSL is compiled in)
import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
const OUT = join(ROOT, "dist", "web");
const wasm = join(ROOT, "zig-out", "web", "zcmir.wasm");
if (!existsSync(wasm)) {
  console.error("zig-out/web/zcmir.wasm missing: run `zig build wasm -Doptimize=ReleaseSmall` first");
  process.exit(1);
}
rmSync(OUT, { recursive: true, force: true });
mkdirSync(join(OUT, "zcmir"), { recursive: true });
for (const d of ["index.html", "css", "js", "presets"]) cpSync(join(ROOT, "web", "app", d), join(OUT, d), { recursive: true });
cpSync(wasm, join(OUT, "zcmir", "zcmir.wasm"));
cpSync(join(ROOT, "web", "zcmir.js"), join(OUT, "zcmir", "zcmir.js"));
console.log(`built ${OUT}`);
