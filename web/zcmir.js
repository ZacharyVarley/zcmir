/**
 * zcmir in the browser: loads zcmir.wasm (Zig) and gives it WebGPU.
 *
 * The module schedules every GPU operation itself; this adapter only carries out its requests
 * (buffers, pipelines, dispatches, copies, submits) through the browser's WebGPU API, and
 * resolves reads asynchronously. No registration logic lives here.
 *
 *   import { Zcmir } from "./zcmir.js";
 *   const zc = await Zcmir.create({ wasm: "zcmir.wasm" });
 *   zc.setImage(0, movingF32, w, h); zc.setImage(1, fixedF32, w2, h2);
 *   const m = await zc.shiftMap(H);            // { peak, dx, dy, n, map, … }
 */

const LIMIT_KEYS = ["maxStorageBufferBindingSize", "maxBufferSize", "maxComputeWorkgroupStorageSize", "maxStorageBuffersPerShaderStage"];
const utf8 = new TextDecoder();
const newStats = () => ({ dispatches: 0, submits: 0, reads: 0, waits: 0, waitMs: 0, compileWaitMs: 0, pipelines: 0, pipelineMs: 0 });
const utf8enc = new TextEncoder();
// match.zig Verdict, in order (null: it passes)
const FIT_VERDICTS = [null, "not a finite pose", "its horizon crosses the moving image (the image folds)", "it mirrors the image",
  "its scale is outside the scale limits", "it stretches the image beyond the stretch limit", "its perspective is beyond the perspective limit"];

// ── pipeline warm-up across page loads ──
// Browsers compile a WGSL pipeline the first time a page asks for it, which costs 20–1000 ms
// each. The adapter remembers (in localStorage) which pipelines this origin created, and at
// the next load starts compiling all of them in parallel while the page sets up, so the first
// detect / match / map finds them ready. The list belongs to one build of zcmir.wasm (its
// fingerprint), so a new build starts a new list. `warm: false` in Zcmir.create turns this off.
// A first visit starts from a list shipped with the site (`pipelines.json` beside the wasm,
// recorded by tests/test_web.mjs --save-pipelines) when it belongs to the same build.
const WARM_KEY = "zcmir.pipelines.v1";
const WARM_MAX = 400;

function hashCode(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) h = Math.imul(h ^ str.charCodeAt(i), 0x01000193);
  return (h >>> 0).toString(36) + "." + str.length.toString(36);
}

function hashBytes(bytes) {
  const u = new Uint8Array(bytes);
  let h = 0x811c9dc5;
  for (let i = 0; i < u.length; i++) h = Math.imul(h ^ u[i], 0x01000193);
  return (h >>> 0).toString(36) + "." + u.length.toString(36);
}

function loadWarm() {
  try { return JSON.parse(localStorage.getItem(WARM_KEY)); } catch { return null; }
}

async function fetchWarm(url) {
  try {
    const r = await fetch(url);
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

function warmRecord(w) {
  const list = w.list.slice(-WARM_MAX);
  const codes = {};
  for (const [h] of list) codes[h] = w.codes.get(h);
  return { fp: w.fp, codes, list };
}

function saveWarm(state) {
  const w = state.warm;
  if (!w || w.timer) return;
  w.timer = setTimeout(() => {
    w.timer = null;
    try { localStorage.setItem(WARM_KEY, JSON.stringify(warmRecord(w))); } catch { /* quota: skip */ }
  }, 1500);
}

function gpuImports(state) {
  const { device } = state;
  const bufs = [null];
  const freeIds = [];
  // pipelines: { pipeline, layout } once compiled (pipeline null while compiling or failed)
  const pipes = [null];
  const modules = new Map();
  // (code hash, entry) → compile promise, shared by the warm-up and createPipeline
  const compiled = new Map();
  let enc = null;
  let pass = null;
  let recorded = [];
  // Pipelines compile asynchronously and in parallel (createComputePipelineAsync). While a
  // pipeline that recorded work uses is still compiling, queue operations are held back in
  // order (`backlog`) and replayed once it is ready, so the module keeps recording ahead and
  // only a wait (a readback) blocks on the compiler.
  let backlog = null;
  let blocking = new Set();
  let drained = null;
  const mem = () => state.memory.buffer;
  const bytes = (ptr, len) => new Uint8Array(mem(), ptr, len);
  const endPass = () => { if (pass) { pass.end(); pass = null; } };
  const encoder = () => (enc ||= device.createCommandEncoder());
  // counters for zc.stats(): what an operation cost in GPU round trips
  const st = state.stats;
  device.addEventListener?.("uncapturederror", (e) => state.errors.push(String(e.error?.message || e.error)));

  const defer = (op) => {
    if (!backlog) {
      backlog = [];
      let resolve;
      drained = { promise: new Promise((r) => (resolve = r)), resolve };
    }
    backlog.push(op);
  };
  const replay = () => {
    const ops = backlog;
    backlog = null;
    for (const op of ops) op();
    drained.resolve();
    drained = null;
  };

  // ── operations, issued directly (or replayed from the backlog) ──
  // indirect: [buffer, byte offset] of the workgroup counts (3 × u32) instead of x, y, z
  const doDispatch = (p, x, y, z, entries, indirect) => {
    if (!p.pipeline) return; // failed to compile: reported through takeError
    if (!pass) pass = encoder().beginComputePass();
    pass.setPipeline(p.pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: p.layout, entries }));
    if (indirect) pass.dispatchWorkgroupsIndirect(indirect[0], indirect[1]);
    else pass.dispatchWorkgroups(Math.max(1, x), Math.max(1, y), Math.max(1, z));
  };
  const bindEntries = (bindsPtr, n) => {
    const v = new DataView(mem(), bindsPtr, n * 24);
    const entries = [];
    for (let i = 0; i < n; i++) {
      const o = i * 24;
      const size = Number(v.getBigUint64(o + 16, true));
      const resource = { buffer: bufs[v.getUint32(o + 4, true)], offset: Number(v.getBigUint64(o + 8, true)) };
      if (size) resource.size = size;
      entries.push({ binding: v.getUint32(o, true), resource });
    }
    return entries;
  };
  const doSubmit = () => {
    endPass();
    if (enc) { st.submits++; device.queue.submit([enc.finish()]); enc = null; }
    for (const b of recorded) b.destroy();
    recorded = [];
    for (const r of state.reads) {
      if (r.promise || !r.issued) continue;
      r.promise = r.staging.mapAsync(GPUMapMode.READ).then(() => {
        // wasm memory may have grown since the read was requested: take a fresh view.
        new Uint8Array(mem(), r.dst, r.size).set(new Uint8Array(r.staging.getMappedRange(), 0, r.size));
        r.staging.unmap();
        r.staging.destroy();
      });
    }
  };
  const doDestroy = (b) => { if (enc) recorded.push(b); else b.destroy(); }; // never under an open batch

  const moduleOf = (code) => {
    let module = modules.get(code);
    if (!module) { module = device.createShaderModule({ code }); modules.set(code, module); }
    return module;
  };
  // Compile progress for onCompile listeners: counts restart once everything started has finished.
  const cp = state.compiling;
  const told = () => { for (const fn of cp.listeners) try { fn({ done: cp.done, total: cp.total }); } catch (e) { console.error(e); } };
  const compile = (code, h, entryPoint) => {
    const key = h + "/" + entryPoint;
    let c = compiled.get(key);
    if (!c) {
      if (cp.done === cp.total) cp.done = cp.total = 0;
      cp.total++;
      told();
      c = device.createComputePipelineAsync({ label: entryPoint, layout: "auto", compute: { module: moduleOf(code), entryPoint } });
      c.finally(() => { cp.done++; told(); }).catch(() => {});
      compiled.set(key, c);
    }
    return c;
  };
  // Start compiling what this origin used before. Entries from an older zcmir may no longer
  // compile: their errors stay inside this error scope (not reported against a later call) and
  // they leave the list.
  const w = state.warm;
  if (w?.saved) {
    w.list = w.saved.list.slice();
    device.pushErrorScope("validation");
    for (const [h, entry] of w.saved.list) {
      const code = w.saved.codes[h];
      if (!code) continue;
      w.codes.set(h, code);
      compile(code, h, entry).catch(() => {
        const i = w.list.findIndex(([a, b]) => a === h && b === entry);
        if (i >= 0) w.list.splice(i, 1);
        saveWarm(state);
      });
    }
    device.popErrorScope().catch(() => {});
    w.saved = null;
  }

  return {
    limits(out) {
      const v = new DataView(mem(), out, 40);
      const L = device.limits;
      v.setBigUint64(0, BigInt(L.maxStorageBufferBindingSize), true);
      v.setBigUint64(8, BigInt(L.maxBufferSize), true);
      v.setUint32(16, L.maxComputeWorkgroupStorageSize, true);
      v.setUint32(20, L.maxStorageBuffersPerShaderStage, true);
      v.setUint32(24, L.minUniformBufferOffsetAlignment, true);
      v.setUint32(28, L.minStorageBufferOffsetAlignment, true);
      v.setUint32(32, device.features.has("shader-f16") ? 1 : 0, true);
      v.setUint32(36, navigator.gpu?.wgslLanguageFeatures?.has?.("packed_4x8_integer_dot_product") ? 1 : 0, true);
    },
    info(ptr, cap) {
      const b = utf8enc.encode(state.info).subarray(0, cap);
      bytes(ptr, b.length).set(b);
      return b.length;
    },
    createBuffer(size, usage) {
      const u = usage === 1
        ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT;
      const b = device.createBuffer({ size, usage: u });
      const id = freeIds.length ? freeIds.pop() : bufs.length;
      bufs[id] = b;
      return id;
    },
    destroyBuffer(id) {
      const b = bufs[id];
      if (!b) return;
      bufs[id] = null;
      freeIds.push(id);
      if (backlog) defer(() => doDestroy(b)); else doDestroy(b);
    },
    writeBuffer(id, offset, ptr, len) {
      const b = bufs[id];
      if (backlog) { const data = bytes(ptr, len).slice(); defer(() => device.queue.writeBuffer(b, offset, data)); }
      else device.queue.writeBuffer(b, offset, mem(), ptr, len);
    },
    // Starts compiling and returns at once; the first dispatch that needs it holds the queue
    // until it is ready.
    createPipeline(codePtr, codeLen, entryPtr, entryLen) {
      const t0 = performance.now();
      st.pipelines++;
      const code = utf8.decode(bytes(codePtr, codeLen));
      const entryPoint = utf8.decode(bytes(entryPtr, entryLen));
      const h = hashCode(code);
      if (w) {
        w.codes.set(h, code);
        const i = w.list.findIndex(([a, b]) => a === h && b === entryPoint);
        if (i >= 0) w.list.splice(i, 1);
        w.list.push([h, entryPoint]); // most recent last
        saveWarm(state);
      }
      const p = { pipeline: null, layout: null, ready: false };
      p.promise = compile(code, h, entryPoint)
        .then((pl) => { p.pipeline = pl; p.layout = pl.getBindGroupLayout(0); })
        .catch((e) => state.errors.push(`pipeline ${entryPoint}: ${e.message || e}`))
        .finally(() => {
          p.ready = true;
          if (blocking.delete(p) && backlog && blocking.size === 0) replay();
        });
      pipes.push(p);
      st.pipelineMs += performance.now() - t0;
      return pipes.length - 1;
    },
    dispatch(pipe, x, y, z, bindsPtr, n) {
      st.dispatches++;
      const p = pipes[pipe];
      const entries = bindEntries(bindsPtr, n);
      if (!p.ready) blocking.add(p);
      if (backlog || !p.ready) defer(() => doDispatch(p, x, y, z, entries));
      else doDispatch(p, x, y, z, entries);
    },
    dispatchIndirect(pipe, buf, offset, bindsPtr, n) {
      st.dispatches++;
      const p = pipes[pipe];
      const entries = bindEntries(bindsPtr, n);
      const ind = [bufs[buf], offset];
      if (!p.ready) blocking.add(p);
      if (backlog || !p.ready) defer(() => doDispatch(p, 0, 0, 0, entries, ind));
      else doDispatch(p, 0, 0, 0, entries, ind);
    },
    clearBuffer(id, offset, size) {
      const b = bufs[id];
      const op = () => { endPass(); encoder().clearBuffer(b, offset, size); };
      if (backlog) defer(op); else op();
    },
    copyBuffer(src, soff, dst, doff, size) {
      const a = bufs[src];
      const b = bufs[dst];
      const op = () => { endPass(); encoder().copyBufferToBuffer(a, soff, b, doff, size); };
      if (backlog) defer(op); else op();
    },
    readBuffer(src, offset, size, dst) {
      st.reads++;
      const a = bufs[src];
      const padded = Math.ceil(size / 4) * 4;
      const staging = device.createBuffer({ size: padded, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const r = { staging, size, dst, promise: null, issued: false };
      state.reads.push(r);
      const op = () => { endPass(); encoder().copyBufferToBuffer(a, offset, staging, 0, padded); r.issued = true; };
      if (backlog) defer(op); else op();
    },
    submit() {
      if (backlog) defer(doSubmit); else doSubmit();
    },
    takeError(ptr, cap) {
      if (!state.errors.length) return 0;
      const b = utf8enc.encode(state.errors.shift()).subarray(0, cap);
      bytes(ptr, b.length).set(b);
      return b.length;
    },
    // Suspending import (JS Promise Integration): the module's stack waits here while pipelines
    // still compiling and the submitted reads resolve, then resumes as if the call had blocked.
    wait: new WebAssembly.Suspending(async () => {
      const t0 = performance.now();
      st.waits++;
      try {
        if (drained) { await drained.promise; st.compileWaitMs += performance.now() - t0; }
        const reads = state.reads;
        state.reads = [];
        await Promise.all(reads.map((r) => r.promise));
        return 0;
      } catch (e) { state.errors.push(String(e)); return 1; } finally { st.waitMs += performance.now() - t0; }
    }),
  };
}

// Exports that never wait (plain calls); every other function export may suspend and is
// called through WebAssembly.promising.
const SYNC_EXPORTS = new Set(["zc_alloc", "zc_free", "zc_result_sizes", "zc_adapter", "zc_error", "zc_create", "zc_destroy", "zc_canvas",
  "zc_export_warp", "zc_displacement", "zc_image_size", "zc_cancel", "zc_swap",
  "zc_settings", "zc_configure", "zc_set_ffd", "zc_get_ffd", "zc_set_pose", "zc_get_pose", "zc_events"]);


// Result struct layouts (engine.zig), [name, type, byte offset].
const SHIFT = [["n", "u32", 0], ["cw", "u32", 4], ["ch", "u32", 8], ["canvasW", "u32", 12], ["canvasH", "u32", 16],
  ["ox", "i32", 20], ["oy", "i32", 24], ["peakIndex", "u32", 28], ["dx", "f64", 32], ["dy", "f64", 40], ["peak", "f64", 48],
  ["zero", "f64", 56], ["corrAreaPx", "f64", 64], ["exact", "bool", 72], ["calibrated", "bool", 76], ["ny", "u32", 80]];
const RS = [["n", "u32", 0], ["nTh", "u32", 4], ["nLam", "u32", 8], ["peakIndex", "u32", 12], ["dlam", "f64", 16],
  ["r0", "f64", 24], ["r1", "f64", 32], ["cx", "f64", 40], ["cy", "f64", 48], ["peak", "f64", 56], ["zero", "f64", 64],
  ["dth", "f64", 72], ["dsg", "f64", 80], ["symmetric", "bool", 88], ["calibrated", "bool", 92]];

export class Zcmir {
  /** Load the module and open a high-performance device with the limits the maps need. */
  static async create({ wasm = new URL("../zig-out/web/zcmir.wasm", import.meta.url), device = null, warm = true, onCompile = null } = {}) {
    let info = "";
    if (!device) {
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("WebGPU is not available");
      const requiredLimits = {};
      for (const k of LIMIT_KEYS) requiredLimits[k] = adapter.limits[k];
      const requiredFeatures = adapter.features.has("shader-f16") ? ["shader-f16"] : [];
      device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
      const ai = adapter.info || {};
      info = `${ai.vendor || "gpu"} ${ai.architecture || ai.device || ""}`.trim();
    }
    const compiling = { done: 0, total: 0, listeners: onCompile ? [onCompile] : [] };
    const state = { device, memory: null, reads: [], errors: [], info, listeners: [], stats: newStats(), compiling };
    const warmOn = warm && typeof localStorage !== "undefined";
    // This origin's own list when it belongs to this build; else the shipped one, downloaded
    // alongside the wasm on a first visit (or after the wasm changed).
    const own = warmOn ? loadWarm() : null;
    const shippedUrl = new URL("pipelines.json", new URL(wasm, location.href));
    const shipped = warmOn && !own ? fetchWarm(shippedUrl) : null;
    const bytes = await (await fetch(wasm)).arrayBuffer();
    if (warmOn) {
      const fp = hashBytes(bytes);
      let saved = own?.fp === fp ? own : null;
      if (!saved) {
        const ship = await (shipped || fetchWarm(shippedUrl));
        saved = ship?.fp === fp ? ship : null;
      }
      state.warm = { fp, saved, codes: new Map(), list: [], timer: null };
    }
    const env = {
      zc_log(ptr, len) { console.warn("zcmir:", utf8.decode(new Uint8Array(state.memory.buffer, ptr, len))); },
      // progress events from long operations (climb, search): one JSON object each
      zc_event(ptr, len) {
        const ev = JSON.parse(utf8.decode(new Uint8Array(state.memory.buffer, ptr, len)));
        for (const fn of state.listeners) try { fn(ev); } catch (e) { console.error(e); }
      },
    };
    if (typeof WebAssembly.Suspending !== "function") throw new Error("this browser lacks WebAssembly JS Promise Integration (Chrome / Edge 137+)");
    const { instance } = await WebAssembly.instantiate(bytes, { zcgpu: gpuImports(state), env });
    state.memory = instance.exports.memory;
    const x = {};
    for (const [k, v] of Object.entries(instance.exports)) {
      x[k] = typeof v === "function" && !SYNC_EXPORTS.has(k) ? WebAssembly.promising(v) : v;
    }
    return new Zcmir(x, state);
  }

  constructor(x, state) {
    this.x = x;
    this.state = state;
    this._queue = Promise.resolve();
    this.e = x.zc_create();
    if (!this.e) throw new Error("zcmir: engine creation failed");
    const sz = this._frame((f) => { const p = f.alloc(16); x.zc_result_sizes(p); return Array.from(new Uint32Array(this.mem, p, 4)); });
    this.sizes = { shift: sz[0], rs: sz[1], grad: sz[2] };
  }

  get device() { return this.state.device; }

  /** Call fn({ done, total }) as pipelines compile (total counts the current burst). */
  onCompile(fn) {
    this.state.compiling.listeners.push(fn);
    fn({ done: this.state.compiling.done, total: this.state.compiling.total });
  }

  /** The pipelines this page created, as the warm-up list (pipelines.json) records them. */
  warmList() {
    return this.state.warm ? warmRecord(this.state.warm) : null;
  }

  /** GPU traffic since the last call (dispatches, submits, reads, waits and the time suspended
   *  in them, pipelines created and the time that took), then reset. */
  stats() { const s = { ...this.state.stats }; Object.assign(this.state.stats, newStats()); return s; }
  get mem() { return this.state.memory.buffer; }

  /** Temporary wasm allocations for one call, freed afterwards. */
  _frame(fn) {
    const held = [];
    const f = {
      alloc: (n) => { const p = this.x.zc_alloc(Math.max(n, 8)); held.push([p, Math.max(n, 8)]); return p; },
      f64: (arr) => { const p = f.alloc(arr.length * 8); new Float64Array(this.mem, p, arr.length).set(Array.from(arr)); return p; },
      f32: (arr) => { const p = f.alloc(arr.length * 4); new Float32Array(this.mem, p, arr.length).set(arr); return p; },
      str: (s) => { const b = utf8enc.encode(s); const p = f.alloc(b.length); new Uint8Array(this.mem, p, b.length).set(b); return [p, b.length]; },
    };
    const done = () => { for (const [p, n] of held) this.x.zc_free(p, n); };
    let out;
    try { out = fn(f); } catch (e) { done(); throw e; }
    if (out instanceof Promise) return out.finally(done);
    done();
    return out;
  }

  _read(layout, p, size) {
    const v = new DataView(this.mem, p, size);
    const o = {};
    for (const [k, t, off] of layout) {
      o[k] = t === "u32" ? v.getUint32(off, true) : t === "i32" ? v.getInt32(off, true) : t === "bool" ? !!v.getUint32(off, true) : v.getFloat64(off, true);
    }
    return o;
  }

  _error() {
    return this._frame((f) => {
      const p = f.alloc(1024);
      const n = this.x.zc_error(this.e, p, 1024);
      return utf8.decode(new Uint8Array(this.mem, p, n));
    });
  }

  _check(rc, what) {
    const err = this._error();
    if (rc !== 0 || err) throw new Error(`${what}: ${err || rc}`);
  }

  _serial(fn) {
    const run = this._queue.then(fn, fn);
    this._queue = run.catch(() => {});
    return run;
  }

  get adapter() {
    return this._frame((f) => {
      const p = f.alloc(256);
      return utf8.decode(new Uint8Array(this.mem, p, this.x.zc_adapter(this.e, p, 256)));
    });
  }

  /** Apply settings overrides (see src/settings.zig for the keys). */
  configure(obj) {
    return this._frame((f) => {
      const [p, n] = f.str(JSON.stringify(obj));
      this._check(this.x.zc_configure(this.e, p, n), "configure");
    });
  }

  get settings() {
    return this._frame((f) => {
      let cap = 4096, p = f.alloc(cap), n = this.x.zc_settings(this.e, p, cap);
      if (n > cap) { cap = n; p = f.alloc(cap); n = this.x.zc_settings(this.e, p, cap); }
      return JSON.parse(utf8.decode(new Uint8Array(this.mem, p, n)));
    });
  }

  /** B-spline control points (2·g·g, g = settings.ffd_grid). */
  setFfd(cps) {
    this._frame((f) => this._check(this.x.zc_set_ffd(this.e, f.f32(Float32Array.from(cps)), cps.length), "set_ffd"));
  }

  /**
   * side 0 = moving, 1 = fixed. data: Float32Array(w·h) gray in [0, 1] (row-major) or
   * Uint8ClampedArray / Uint8Array(w·h·4) RGBA (ImageData.data). RGBA and { preprocess: true }
   * gray go through the settings' preprocessing (CLAHE, band pass, invert); plain gray is used
   * as the work image as is.
   */
  setImage(side, data, w, h, { preprocess = false } = {}) {
    return this._serial(() => this._frame(async (f) => {
      if (data instanceof Float32Array) {
        const p = f.f32(data.subarray(0, w * h));
        this._check(await (preprocess ? this.x.zc_set_image_gray(this.e, side, p, w, h, 1) : this.x.zc_set_image(this.e, side, p, w, h)), "set_image");
      } else {
        const n = w * h * 4, p = f.alloc(n);
        new Uint8Array(this.mem, p, n).set(data.subarray(0, n));
        this._check(await this.x.zc_set_image_rgba(this.e, side, p, w, h), "set_image");
      }
    }));
  }

  /** The preprocessed work image of a side as Float32Array(w·h). */
  workImage(side, w, h) {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(w * h * 4);
      this._check(await this.x.zc_work_image(this.e, side, p, w * h), "work_image");
      return new Float32Array(this.mem, p, w * h).slice();
    }));
  }

  /** Detect keypoints on both images: { n1, n2, levels1, levels2 }. */
  detect() {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(16);
      this._check(await this.x.zc_detect(this.e, p), "detect");
      const [n1, n2, levels1, levels2] = new Uint32Array(this.mem, p, 4);
      return { n1, n2, levels1, levels2 };
    }));
  }

  /**
   * Match and fit; the fit becomes the engine's pose. { H, inliers, correspondences } and for
   * POS-GIFT { affineInliers, posN, posInliers, rotation (degrees, rotation search), posKept
   * (H is POS's fit; false: the pooled affine fit scored higher and stays) }.
   */
  match() {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(112);
      this._check(await this.x.zc_match(this.e, p), "match");
      const v = new DataView(this.mem, p, 112);
      return {
        H: Array.from({ length: 9 }, (_, i) => v.getFloat64(i * 8, true)), inliers: v.getUint32(72, true), correspondences: v.getUint32(76, true),
        affineInliers: v.getUint32(80, true), posN: v.getUint32(84, true), posInliers: v.getUint32(88, true),
        rotation: v.getUint32(92, true) ? v.getFloat64(96, true) : null, posKept: !!v.getUint32(104, true),
      };
    }));
  }

  /**
   * Global search (settings.search_mode "sweep" or "cloud"): { H, score, prev, improved, peaks |
   * evals, finalists | seeds }. Progress, the sweep landscape ("sweep") and the cloud's scores
   * ("swarm") arrive as events.
   */
  search() {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(104);
      this._check(await this.x.zc_search(this.e, p), "search");
      const v = new DataView(this.mem, p, 104);
      return { H: Array.from({ length: 9 }, (_, i) => v.getFloat64(i * 8, true)), score: v.getFloat64(72, true), prev: v.getFloat64(80, true),
        improved: !!v.getUint32(88, true), countA: v.getUint32(92, true), countB: v.getUint32(96, true) };
    }));
  }

  /**
   * The overlay at H: { ox, oy, ow, oh, Hc, moving, fixed } — both images on the pose's union
   * canvas (Float32Array ow·oh in [0, 1]). inverse: the fixed image backprojected into the
   * moving frame.
   */
  overlay(H, { inverse = false } = {}) {
    return this._serial(() => this._frame(async (f) => {
      const hp = f.f64(H), cp = f.alloc(88);
      this._check(this.x.zc_canvas(this.e, hp, inverse ? 1 : 0, cp), "canvas");
      const n = new Uint32Array(this.mem, cp + 8, 2).reduce((a, b) => a * b, 1);
      const mp = f.alloc(n * 4), fp = f.alloc(n * 4);
      this._check(await this.x.zc_overlay(this.e, hp, inverse ? 1 : 0, cp, mp, fp, n), "overlay");
      const v = new DataView(this.mem, cp, 88);
      return {
        ox: v.getInt32(0, true), oy: v.getInt32(4, true), ow: v.getUint32(8, true), oh: v.getUint32(12, true),
        Hc: Array.from({ length: 9 }, (_, i) => v.getFloat64(16 + i * 8, true)),
        moving: new Float32Array(this.mem, mp, n).slice(), fixed: new Float32Array(this.mem, fp, n).slice(),
      };
    }));
  }

  /** Exchange-symmetric tile heat at H: { score, dest, src, G } (G×G each). */
  tileHeat(H, { grid = 8, minN = 12 } = {}) {
    return this._serial(() => this._frame(async (f) => {
      const G = Math.max(2, Math.min(16, grid | 0)), nT = G * G;
      const sp = f.alloc(8), dp = f.alloc(nT * 4), rp = f.alloc(nT * 4);
      this._check(await this.x.zc_tile_heat(this.e, f.f64(H), G, minN, sp, dp, rp, nT), "tile_heat");
      return { score: new Float64Array(this.mem, sp, 1)[0], dest: new Float32Array(this.mem, dp, nT).slice(), src: new Float32Array(this.mem, rp, nT).slice(), G };
    }));
  }

  /** The warp export at the engine's pose: a ZIP (Uint8Array) of {stem}_Warp.nrrd [+ {stem}.tfm]. */
  exportWarp(stem = "cmir_warp") {
    return this._frame((f) => {
      const [sp, sl] = f.str(stem);
      const n = Number(this.x.zc_export_warp(this.e, sp, sl, 0, 0));
      if (n < 0) this._check(-1, "export_warp");
      const p = f.alloc(n);
      this.x.zc_export_warp(this.e, sp, sl, p, n);
      return new Uint8Array(this.mem, p, n).slice();
    });
  }

  /** The displacement field over the fixed image (Float32Array w·h·2, x_moving − x_fixed). */
  displacement(w, h) {
    return this._frame((f) => {
      const p = f.alloc(w * h * 8);
      this._check(this.x.zc_displacement(this.e, p, w * h * 2), "displacement");
      return new Float32Array(this.mem, p, w * h * 2).slice();
    });
  }

  /** Detect → match → climb; resolves like climb(). */
  auto() {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(96);
      this._check(await this.x.zc_auto(this.e, p), "auto");
      const v = new DataView(this.mem, p, 96);
      return { H: Array.from({ length: 9 }, (_, i) => v.getFloat64(i * 8, true)), score: v.getFloat64(72, true), iterations: v.getUint32(80, true), moved: !!v.getUint32(84, true) };
    }));
  }

  /** Ask the running climb or search to stop early (it keeps its best). */
  cancel() { this.x.zc_cancel(this.e); }

  /** Exchange the moving and fixed images (features and GLS-MIFT keypoints follow). */
  swap() { this.x.zc_swap(this.e); }

  /** (w, h) of side 0 (moving) or 1 (fixed). */
  imageSize(side) {
    return this._frame((f) => { const p = f.alloc(8); this.x.zc_image_size(this.e, side, p); return Array.from(new Uint32Array(this.mem, p, 2)); });
  }

  /** Descriptor inner product of each moving keypoint with its match (−1e9: none), after match(). */
  matchScores(cap = 65536) {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(cap * 4);
      const n = await this.x.zc_match_scores(this.e, p, cap);
      if (n < 0) this._check(n, "match_scores");
      return new Float32Array(this.mem, p, n).slice();
    }));
  }

  /** Keypoints of a side as Float32Array(4·n): x, y (1-based pixels), score, 0. */
  keypoints(side, cap = 65536) {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(cap * 16);
      const n = await this.x.zc_keypoints(this.e, side, p, cap);
      if (n < 0) this._check(n, "keypoints");
      return new Float32Array(this.mem, p, 4 * Math.min(n, cap)).slice();
    }));
  }

  /** Descriptor frame of each keypoint of a side, in keypoints() order: Float32Array(2·n) of the
   *  direction it starts from (radians; x right, y down) and the radius it reads (pixels). */
  keypointFrames(side, cap = 65536) {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(cap * 8);
      const n = await this.x.zc_keypoint_frames(this.e, side, p, cap);
      if (n < 0) this._check(n, "keypoint_frames");
      return new Float32Array(this.mem, p, 2 * Math.min(n, cap)).slice();
    }));
  }

  /** The sanity check of a pose fitted to matches (the settings' pose limits on the moving image):
   *  null when it passes, else why it does not. */
  fitCheck(H) {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(72);
      new Float64Array(this.mem, p, 9).set(H);
      const v = await this.x.zc_fit_check(this.e, p);
      return v < FIT_VERDICTS.length ? FIT_VERDICTS[v] : "it fails the sanity check";
    }));
  }

  /** Matched fixed keypoint of each moving keypoint (0xffffffff: none) after match(). */
  matches(cap = 65536) {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(cap * 4);
      const n = await this.x.zc_matches(this.e, p, cap);
      if (n < 0) this._check(n, "matches");
      return new Uint32Array(this.mem, p, n).slice();
    }));
  }

  /** Listen to progress events ({kind: "log" | "trail" | "pose", …}); returns an unsubscribe. */
  onEvent(fn) {
    this.state.listeners.push(fn);
    return () => { this.state.listeners = this.state.listeners.filter((f) => f !== fn); };
  }

  /** The engine's pose (3×3 row-major, moving → fixed, 1-based). */
  get pose() {
    return this._frame((f) => { const p = f.alloc(72); this.x.zc_get_pose(this.e, p); return Array.from(new Float64Array(this.mem, p, 9)); });
  }
  set pose(H) {
    this._frame((f) => this.x.zc_set_pose(this.e, f.f64(H)));
  }

  get ffd() {
    return this._frame((f) => { const p = f.alloc(2048); const n = this.x.zc_get_ffd(this.e, p, 512); return Float32Array.from(new Float32Array(this.mem, p, n)); });
  }

  /** Climb from the engine's pose; resolves to { H, score, iterations, moved }. */
  climb() {
    return this._serial(() => this._frame(async (f) => {
      const p = f.alloc(96);
      this._check(await this.x.zc_climb(this.e, p), "climb");
      const v = new DataView(this.mem, p, 96);
      return { H: Array.from({ length: 9 }, (_, i) => v.getFloat64(i * 8, true)), score: v.getFloat64(72, true), iterations: v.getUint32(80, true), moved: !!v.getUint32(84, true) };
    }));
  }

  /** Pose score at H: { mean, fwd, inv }. */
  score(H) {
    return this._serial(() => this._frame(async (f) => {
      const out = f.alloc(24);
      this._check(await this.x.zc_score(this.e, f.f64(H), out), "score");
      const v = new Float64Array(this.mem, out, 3);
      return { mean: v[0], fwd: v[1], inv: Number.isNaN(v[2]) ? null : v[2] };
    }));
  }

  /** Score and tangent gradient at H: { score, n, grad[nk], hess[nk·nk] | null }. */
  gradient(H, { hess = false } = {}) {
    return this._serial(() => this._frame(async (f) => {
      const out = f.alloc(this.sizes.grad);
      this._check(await this.x.zc_gradient(this.e, f.f64(H), hess ? 1 : 0, out), "gradient");
      const v = new DataView(this.mem, out, this.sizes.grad);
      const nk = v.getUint32(612, true);
      const grad = Array.from({ length: nk }, (_, i) => v.getFloat64(32 + i * 8, true));
      const h = v.getUint32(608, true) ? Array.from({ length: nk * nk }, (_, i) => v.getFloat64(96 + i * 8, true)) : null;
      return { score: v.getFloat64(0, true), n: v.getFloat64(24, true), grad, hess: h };
    }));
  }

  /** Score and gradient over the spline control points (2·g·g) at H. */
  ffdGradient(H) {
    return this._serial(() => this._frame(async (f) => {
      const n = 2 * 16 * 16;
      const sp = f.alloc(8), gp = f.alloc(n * 8);
      this._check(await this.x.zc_ffd_gradient(this.e, f.f64(H), sp, gp, n), "ffd_gradient");
      const g = this.settings.ffd_grid;
      return { score: new Float64Array(this.mem, sp, 1)[0], grad: Array.from(new Float64Array(this.mem, gp, 2 * g * g)) };
    }));
  }

  /**
   * Dense shift map at pose H (3×3 row-major, moving → fixed, 1-based pixels). Without flags
   * the current settings decide (score, resolution, overlap floor); with { exact, calibrated }
   * those are explicit and there is no overlap floor.
   */
  shiftMap(H, { exact, calibrated, resolution = 1024, returnMap = true } = {}) {
    return this._serial(() => this._frame(async (f) => {
      const cap = resolution;
      const mapN = returnMap ? cap * cap : 0;
      const mp = mapN ? f.alloc(mapN * 4) : 0;
      const rp = f.alloc(this.sizes.shift);
      const hp = f.f64(H);
      const rc = exact === undefined && calibrated === undefined
        ? await this.x.zc_shift_map_set(this.e, hp, rp, mp, mapN)
        : await this.x.zc_shift_map(this.e, hp, (exact ?? true ? 1 : 0) | (calibrated ?? true ? 2 : 0), cap, rp, mp, mapN);
      this._check(rc, "shift_map");
      const out = this._read(SHIFT, rp, this.sizes.shift);
      if (mp) out.map = new Float32Array(this.mem, mp, out.n * out.ny).slice();
      return out;
    }));
  }

  /**
   * Roto-scale map at H about center (fixed pixels, 1-based; default the moving centroid).
   * { dth (rad), dsg (log scale), peak, n, nTh, nLam, dlam, r0, r1, map?, fwd?, inv? }.
   */
  rsMap(H, { center = null, symmetric = true, returnMaps = true, cap = 1024 } = {}) {
    return this._serial(() => this._frame(async (f) => {
      const len = returnMaps ? cap * cap : 0;
      const mp = len ? f.alloc(len * 4) : 0, fp = len ? f.alloc(len * 4) : 0, ip = len && symmetric ? f.alloc(len * 4) : 0;
      const rp = f.alloc(this.sizes.rs);
      const cp = center ? f.f64(center) : 0;
      this._check(await this.x.zc_rs_map(this.e, f.f64(H), cp, symmetric ? 1 : 0, rp, mp, fp, ip, len), "rs_map");
      const out = this._read(RS, rp, this.sizes.rs);
      const nn = out.n * out.n;
      if (mp) out.map = new Float32Array(this.mem, mp, nn).slice();
      if (fp) out.fwd = new Float32Array(this.mem, fp, nn).slice();
      if (ip) out.inv = new Float32Array(this.mem, ip, nn).slice();
      return out;
    }));
  }
}
