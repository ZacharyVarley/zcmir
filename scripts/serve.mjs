#!/usr/bin/env node
// Static dev server for the repo root (web/app, with the engine from web/zcmir.js and zig-out/web/).
// WebGPU needs a secure context; http://localhost and 127.0.0.1 count as one.
//   node scripts/serve.mjs [--port 4180]
//   node scripts/serve.mjs --dir dist/web --port 4190     (a built site as is, e.g. to check a deploy)
import { createServer } from "node:http";
import { createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = resolve(fileURLToPath(new URL("..", import.meta.url)));
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm",
  ".wgsl": "text/plain; charset=utf-8", ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
};
const args = process.argv.slice(2);
const port = +(args[args.indexOf("--port") + 1] || process.env.PORT || 4180);
const dir = args.includes("--dir") ? resolve(REPO, args[args.indexOf("--dir") + 1]) : null;
const ROOT = dir || REPO;

// Paths the pages fetch next to themselves, mapped onto the repo (scripts/build_web.mjs lays
// the deployed site out the same).
const ALIASES = dir ? [] : [
  // the app loads zcmir/zcmir.js and zcmir/zcmir.wasm next to its page (dist/web/zcmir/)
  ["/web/app/zcmir/zcmir.js", "/web/zcmir.js"], ["/web/app/zcmir/", "/zig-out/web/"],
];

createServer((req, res) => {
  let path = decodeURIComponent(req.url.split("?")[0]);
  if (path === "/" && !dir) { res.writeHead(302, { Location: "/web/app/" }).end(); return; }
  for (const [from, to] of ALIASES) if (path.startsWith(from)) { path = to + path.slice(from.length); break; }
  if (path.endsWith("/")) path += "index.html";
  const file = normalize(join(ROOT, path));
  if (!file.startsWith(ROOT + sep) || !existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404).end("not found");
    return;
  }
  res.writeHead(200, { "Content-Type": TYPES[extname(file)] || "application/octet-stream", "Cache-Control": "no-store" });
  createReadStream(file).pipe(res);
}).listen(port, "127.0.0.1", () => console.log(`zcmir dev server: http://127.0.0.1:${port}/`));
