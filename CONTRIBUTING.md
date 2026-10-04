# Contributing to zcmir

Issues and pull requests are welcome. For anything beyond a small fix, open an issue first so we can agree on the approach.

## How it fits together

- **WGSL** (`shaders/`) does the registration math.
- **Zig** (`src/`) owns the schedule: which kernels run, in what order, on which buffers, and what is read back.
- **The GPU layer** (`src/gpu/`) connects that schedule to each environment:
  - natively through `webgpu.h`, using [wgpu-native](https://github.com/gfx-rs/wgpu-native) (Vulkan, Metal, Direct3D 12), loaded at run time;
  - in the browser through a JS adapter, `web/zcmir.js`, which forwards the module's requests to WebGPU.
- **The front ends** only pass images and settings in and take results out:
  - the web app (`web/app/`) runs `zcmir.wasm`;
  - the Python package (`python/zcmir/`) calls the native library through `ctypes`.

A fix to the registration therefore lands in both front ends at once.

```
shaders/            WGSL, the single copy (compiled into the library and the wasm)
src/gpu/            GPU layer: gpu.zig (policy), native.zig (webgpu.h), web.zig (browser imports)
src/fft.zig         mixed-radix Stockham FFTs (batched nx × ny, four-step); WGSL generated per length,
                    lengths from the 2^a·3^b ladder (setting fft_sizes; compact: any 13-smooth length)
src/smi.zig         SMI features, the dense shift map, the roto-scale (log-polar) map
src/pose.zig        pose moments, scores, gradients and Gauss–Newton matrices, B-spline gradients
src/pair.zig        the pair score: forward and inverse directions, symmetric mean
src/gclimb.zig      Refine on the GPU: batches of iterations (hops, Gauss–Newton, spline), gated
                    on the device's own done flag; shaders/climb.wgsl holds its decisions
src/climb.zig       what Refine and the searches share (the acceptance rule, the step ladder)
src/gls.zig         GLS-MIFT detection and matching; CLAHE, band pass, invert
src/posgift.zig     POS-GIFT (phase congruency, GIFT descriptors, rotation search, POS re-matching)
src/match.zig       PROSAC / MAGSAC++ robust fits
src/search.zig      brute force: the sweep (rotation × scale [× shear × stretch], each candidate's
                    FFTs sized to its footprint) and the seed cloud
src/overlay.zig     overlay images, NCC map, tile heat
src/export.zig      warp export: NRRD displacement field, ITK .tfm, stored ZIP
src/settings.zig    every option with the app's defaults (JSON in, JSON out)
src/engine.zig      the engine both front ends call; src/exports.zig is its C ABI
python/zcmir/       api.py (the functions), __init__.py (Registrar), tiff.py, app.py (the `zcmir` command),
                    examples.py (the app's example pairs as arrays)
web/app/            the app; js/stackio.js (TIFF / ZIP export), presets/ (examples)
tests/              test_api.py, test_app.py, test_web.mjs
examples/           the tutorials (jupytext percent scripts)
scripts/            fetch_deps.py, build_wheels.py, build_web.mjs, serve.mjs, make_presets.py,
                    run_tutorials.py
```

## Build

Everything is fetched into `.deps/` (git-ignored, checksummed). Nothing is installed system-wide. You need Python 3, and Node 22+ for the dev server, the site build and the browser test.

```bash
python scripts/fetch_deps.py                        # Zig 0.16.0 + wgpu-native for this machine
.deps/zig-0.16.0/zig build -Doptimize=ReleaseFast   # native library → zig-out/bin (Windows) or zig-out/lib
.deps/zig-0.16.0/zig build wasm                     # zig-out/web/zcmir.wasm
.deps/zig-0.16.0/zig build probe                    # open the GPU, run one kernel
```

`pip install .` builds and installs the wheel for this machine (`pyproject.toml` hands the build to `scripts/build_backend.py`, which runs `build_wheels.py` for the host platform). There is no editable install: reinstall after changing Zig or WGSL.

From a checkout, `node scripts/serve.mjs` serves `web/app` with the freshly built `zcmir.wasm` on <http://127.0.0.1:4180/web/app/>. `python -m zcmir`, run from `python/`, does the same.

**Other builds:**
- **The site.** `node scripts/build_web.mjs` writes `dist/web/`: the app, its examples, `zcmir.js` and `zcmir.wasm`. There is no bundler and there are no npm dependencies; any static server can host it, as long as it uses HTTPS or localhost (WebGPU requires it).
- **Wheels.** `python scripts/fetch_deps.py --all`, then `python scripts/build_wheels.py [platform …]`. Zig cross-compiles all six platforms from any host:
  - Windows x64 / ARM64;
  - manylinux 2.28 x64 / ARM64;
  - macOS 11 x64 / ARM64.

  Each wheel is `py3-none-<platform>` (one wheel serves every Python 3) and bundles the library, that platform's wgpu-native and the web app.
- **Examples.** `python scripts/make_presets.py SRC_DIR` rebuilds `web/app/presets/` from the downloaded sources. They are grayscale to keep the wheels small; [ATTRIBUTION.md](web/app/presets/ATTRIBUTION.md) lists the sources.

## Test

```bash
python tests/check_shaders.py                 # every shader kernel builds on this GPU (the compiler's message if not)
python tests/test_api.py                      # Python API on the GPU: synthetic pair and stack with known poses
python tests/test_api.py moving.png fixed.png # … on a pair of yours
python tests/test_app.py                      # the `zcmir` launcher serves every file of the app
node tests/test_web.mjs                       # the app in headless Chrome (WebGPU); --dir dist/web for a built site
python scripts/run_tutorials.py out/          # examples/*.py executed as notebooks → out/*.ipynb, out/*.html
```

- **`test_api`** registers a synthetic cross-modal pair with both detectors and with a spline. The pair is a texture and a copy warped by a known homography, with inverted, nonlinear contrast and noise. It checks the poses against the truth, then exercises the maps, brute force, warping, the warp export, and a stack with drifting known poses.
- **`test_web`** registers the Landsat example against its ground truth and a two-slice IN718 stack, then checks the export.
- **The tutorials** (`examples/`, jupytext percent scripts, outputs never committed) end with checks against the known pose and against the engine, so they fail like tests. They use the inspection calls `Registrar.features`, `moments`, `pos_gift_maps`, `pos_gift_descriptors` and `pos_gift_structure`, and `zcmir.examples.load`.

Changes to the registration should keep these passing. Report accuracy against ground truth, not only against a previous run.

**CI** (`.github/workflows/ci.yml`) runs on every push:
- on Linux: the library, the wasm, the site and wheels, and the launcher;
- on an Apple M1 runner (a real GPU through Metal): `check_shaders`, `test_api` and the tutorials on the installed wheel (the executed notebooks are the `tutorials` artifact), and `test_web` on the built site.

## Conventions

- **Pose.** `H` is a row-major 3×3 that maps *moving* pixel coordinates to *fixed* ones, 1-based: `(1, 1)` is the centre of the top-left pixel.
- **Shift-map lag.** `(dx, dy)` is in fixed-image pixels. The corrected pose is `translate(−dx, −dy) · H`.
- **Displacement field.** `x_moving = x_fixed + d(x_fixed)`, in 0-based pixels (ITK `DisplacementFieldTransform`).
- **Settings.** Every option lives in `src/settings.zig` with the app's default. A new option goes there first; both front ends pick it up by name.
- **GPU work:**
  - Buffers are allocated once and grown, and features stay on the GPU between calls.
  - Dispatches are batched into few submits.
  - Readbacks are few and batched: each costs about 3 ms in a browser.
  - Pipelines compile on first use, in parallel, and only a readback waits for them. The browser adapter remembers which ones a site used and starts compiling them as the page loads; a first visit starts from the shipped `pipelines.json` of the same build. The app shows compiles in the status bar.
  - Refine runs on the GPU (`gclimb.zig`, `climb.wgsl`): every step scores its candidates and the current pose, keeps the best that beats the current score (`climb_select`) and computes the gradient there, all on the device in f32. Eight iterations go into one submission as gated dispatches (`Gpu.beginGate`): a small kernel zeroes their workgroup counts once the climb's done flag is set, so a converged climb costs nothing more. The host reads the state once per batch for the trail, the pose and Stop.
  - Acceptance is relative: a score must beat the best by 1e-7 of it (`pair.beats`), about what the f32 moment sums resolve.
  - No dispatch exceeds 65535 workgroups per dimension.
- **Browser calls.** In the browser, calls are serialized through the adapter. Only `cancel()` may run while an operation waits on the GPU.
- **Profiling.**
  - Browser: `zc.stats()`.
  - Native: `zc_profile(e, 1)`, which times every dispatch with GPU timestamps, then `zc_profile_report`.

## Releasing

1. **Version.** Set the version in both `python/zcmir/__init__.py` (`__version__`) and `src/engine.zig` (`version`). The wheel build refuses a mismatch.
2. **Tag.** Push a tag `vX.Y.Z`. `.github/workflows/release.yml` then:
   - builds the six wheels and the site;
   - tests the wheels on Linux, Windows and macOS (on the M1, on its GPU, in headless Chrome, and by running the tutorials);
   - publishes to PyPI by trusted publishing;
   - attaches the wheels, the site and the executed tutorials (each `.ipynb`, plus a zip with their HTML) to a GitHub Release;
   - deploys the site to GitHub Pages: the app at <https://zacharyvarley.github.io/zcmir/>, the tutorials under `tutorials/`, and `zcmir/pipelines.json`, the pipelines the M1's browser test compiled (`test_web.mjs --save-pipelines`). A first visit starts compiling them as the page loads.

   Run by hand (Actions → release → Run workflow), it only builds and tests.
3. **One-time setup.**
   - On PyPI, add a trusted publisher for this repository: workflow `release.yml`, environment `pypi`.
   - In the repository settings, create the environment `pypi`, and under Pages set the source to **GitHub Actions**.
   - Under Environments → `github-pages` → Deployment branches and tags, add a tag rule `v*` (Pages allows only the default branch at first).
