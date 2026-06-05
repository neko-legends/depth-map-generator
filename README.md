# DepthMap AI Generator

DepthMap AI Generator is a local desktop batch tool for creating depth maps and
WebP files from images and folders. It is designed for AI image workflows that
need ControlNet-style depth maps without running ComfyUI.

The app uses a Tauri 2 desktop shell with a React UI. Rust handles the desktop
window, drag and drop, folder scanning, worker orchestration, and VRAM polling.
Python handles MDA and Depth Anything inference, first-run dependency setup,
model download, depth map export, and WebP conversion.

## Features

- Drag in one image, many images, or folders.
- Recursively scan image folders.
- Generate depth maps beside source images.
- Optional output folder override, with same-folder output as the default.
- Cancel active model downloads or batch jobs.
- Optional resize-before-depth mode with aspect ratio preserved.
- Depth outputs: 8-bit PNG, 16-bit PNG, or raw `.npy`.
- Optional source-image WebP conversion.
- PNG alpha is preserved with lossless WebP.
- JPG-style inputs use configurable WebP quality.
- Existing outputs get numbered filenames such as `_depth_2.png` when overwrite
  is off.
- The queue keeps output history per source image and shows recent output
  thumbnails first, so tuning changes can be compared without leaving the app.
- Model selector:
  - `sy000/MDA/mda_mog_sky_l2` by default.
  - `depth-anything/DA3-LARGE-1.1` as a smaller fallback.
  - `depth-anything/DA3-SMALL` for lower VRAM machines.
- First-launch model picker with Later option.
- Model download progress with percent and downloaded size.
- Optional Keep model loaded mode, enabled by default, to keep the selected model warm between
  jobs.
- Startup model preload when the selected model is already installed.
- Buttons to open the model install folder and package report.
- Device selector: Auto, CUDA, Apple MPS, CPU.
- NVIDIA VRAM usage display when `nvidia-smi` is available.
- Orange/black scalable 16:9 UI.

## Development

Prerequisites:

- Node.js 20.19+ or 22.12+
- Rust toolchain for Tauri builds
- Python 3.10, 3.11, or 3.12
- Git

## AI Agent Installation

Use these steps when installing or validating the app from an automated coding
agent.

1. Clone the repository.

```powershell
git clone https://github.com/flashosophy/DepthMapAIGenerator.git
cd DepthMapAIGenerator
```

2. Confirm required tools are available.

```powershell
node --version
npm --version
python --version
cargo --version
rustc --version
```

3. Install JavaScript dependencies from the committed lockfile.

```powershell
npm ci
```

4. Run source and package checks.

```powershell
npm audit
npm run build
python -m py_compile python\da3_worker.py
```

5. Start the desktop app in development mode.

```powershell
npm run tauri:dev
```

6. Build release artifacts.

```powershell
npm run tauri:build
```

Expected Windows outputs:

- `src-tauri\target\release\depth-map-ai-generator.exe`
- `src-tauri\target\release\bundle\nsis\DepthMap AI Generator_26.6.5_x64-setup.exe`

The release executable is the portable build. It embeds the app UI and Python
worker script; on first use it creates the app-managed Python environment and
downloads runtime packages and model weights into the app data folder.

Runtime notes for agents:

- Do not commit `node_modules`, `dist`, `dist-portable`, `src-tauri/target`, model
  folders, app-data Python environments, generated depth maps, or installer
  files.
- The app downloads model weights at runtime. Do not add Hugging Face model
  files to the repository.
- Direct Python runtime dependencies are pinned in `python/runtime-lock.txt`
  and mirrored by `python/da3_worker.py`.
- WebP-only runs install Pillow in the app-managed Python environment.
- Depth runs install the ML stack, Depth Anything 3, and the selected model.
  MDA also installs its runtime helpers, downloads the upstream MDA source, and
  downloads the `sy000/MDA` checkpoint.
- When `nvidia-smi` is available, the app-managed environment installs PyTorch
  from the CUDA 12.8 wheel index and pins NumPy below 2 for Depth Anything
  compatibility.
- Python 3.13+ is not selected for the app-managed environment because some ML
  dependencies may not publish compatible wheels yet.
- NVIDIA VRAM display depends on `nvidia-smi`; it is allowed to show unavailable
  on machines without NVIDIA drivers.
- Use `npm ci` for reproducible installs. Do not use `npm install` unless the
  task is specifically to update JavaScript packages and commit the resulting
  `package-lock.json` change.

Install JavaScript dependencies from the lockfile:

```powershell
npm ci
```

Run the frontend only:

```powershell
npm run dev
```

Run the desktop app:

```powershell
npm run tauri:dev
```

Build the NSIS setup installer and portable release executable:

```powershell
npm run tauri:build
```

## JavaScript Dependency Locking

The repository commits `package-lock.json` and pins direct JavaScript package
versions in `package.json`. This is intentional. Fresh installs should use
`npm ci`, which installs the exact package versions and integrity hashes from
the lockfile instead of resolving new package versions from npm.

This reduces supply-chain risk for distributed builds. npm packages can be
compromised after a project is working, including through maintainer account
compromise or malicious transitive releases. Locked installs do not make npm
perfectly safe, but they prevent accidental silent upgrades during normal
development and CI.

Direct runtime dependencies:

- `@tauri-apps/api` `2.11.0`: frontend API bridge into the Tauri shell.
- `@tauri-apps/plugin-dialog` `2.7.1`: native image/folder picker support.
- `@tauri-apps/plugin-opener` `2.5.4`: opens model/package folders and reports.
- `lucide-react` `0.468.0`: UI icons.
- `react` `18.3.1`: app UI framework.
- `react-dom` `18.3.1`: React DOM renderer.

Direct development dependencies:

- `@tauri-apps/cli` `2.11.2`: Tauri dev/build command.
- `@types/react` `18.3.28`: React TypeScript types.
- `@types/react-dom` `18.3.7`: React DOM TypeScript types.
- `@vitejs/plugin-react` `5.1.2`: React support for Vite.
- `typescript` `5.9.3`: TypeScript compiler.
- `vite` `7.3.3`: frontend dev server and production bundler.

When intentionally upgrading packages:

1. Update exact versions in `package.json`.
2. Run `npm install` to refresh `package-lock.json`.
3. Run `npm audit`.
4. Run `npm run build`.
5. Run `npm run tauri:build`.
6. Review and commit both `package.json` and `package-lock.json`.

The current locked JavaScript tree has been checked with `npm audit` and reports
zero known npm advisory vulnerabilities at the time of this update. A targeted
check for package names involved in recent npm supply-chain incidents found only
`debug@4.4.3` as a transitive development dependency of Babel through
`@vitejs/plugin-react`; the known compromised `debug@4.4.2` version is not in
the lockfile.

## Python Runtime Locking

The app creates its own Python environment at runtime instead of committing a
project `.venv`. Direct runtime packages are pinned in
`python/runtime-lock.txt` and mirrored in `python/da3_worker.py`. This is
intentional: unpinned Python installs can silently switch PyTorch wheel flavors,
pull NumPy versions incompatible with Depth Anything, or change MDA transitive
behavior.

Pinned runtime choices include:

- PyTorch `2.11.0` / torchvision `0.26.0`.
- CUDA builds from the PyTorch CUDA 12.8 wheel index when `nvidia-smi` is
  available.
- NumPy `1.26.4` for Depth Anything compatibility.
- Depth Anything 3 pinned to commit
  `41736238f5bced4debf3f2a12375d2466874866d`.
- MDA source pinned to commit
  `0a4ae22e6994a9794dfb24a1275c7fbf59112f2a`.

When intentionally upgrading Python runtime packages:

1. Update `python/runtime-lock.txt`.
2. Update matching constants in `python/da3_worker.py`.
3. Bump the relevant app-managed environment marker in `python/da3_worker.py`.
4. Run `python -m py_compile python\da3_worker.py`.
5. Run a CUDA availability check in the app-managed environment when NVIDIA
   support matters.
6. Run an MDA preload and one small MDA batch job.
7. Run `npm run tauri:build`.

## Runtime Behavior

On first launch, the app offers to download MDA-DA3-MoG-Sky, DA3-LARGE-1.1, or
DA3-SMALL. Choosing Later skips the download. If depth generation is later
started with no selected model installed, the worker downloads that model before
inference and shows download progress in the UI.

By default, outputs are written beside each source image. The Settings panel can
choose a single output folder instead. When overwrite is off and a target output
already exists, the worker writes the next available numbered filename instead
of replacing or skipping the file, for example `image_depth_2.png`.

Each queued source image keeps an in-session output history. New outputs appear
first as thumbnails in the Outputs column. Use the folder button on a row to open
the folder containing the newest output.

The Cancel button stops the active Python worker process tree. A canceled model
download may leave an incomplete temporary download file, which is replaced on
the next download attempt.

The first WebP or depth run creates an app-managed Python virtual environment in
the app data folder. It is separate from project `.venv` folders and avoids
mixing with custom Python environments used by other projects. The launcher
prefers Python 3.11, 3.12, or 3.10 and will not use Python 3.13+ for the managed
environment.

WebP-only runs install only Pillow. Depth-map runs install the ML runtime,
install Depth Anything 3 from GitHub, and download the selected model when
needed. MDA runs additionally download MDA's GitHub source and install the small
extra runtime packages its wrappers import. On NVIDIA machines, the app forces
the CUDA PyTorch wheel install so Auto can use the GPU instead of a CPU-only
PyTorch wheel.

Keep model loaded is enabled by default. When the selected model is already
installed, the app starts a persistent Python worker after launch, loads it, and
runs a tiny warm-up inference so the selected device is actually exercised. This
reduces repeat-run latency because later depth jobs can reuse the loaded model
instead of starting Python and loading it again. The model-loading indicator is
intentionally not a percentage bar because PyTorch model load does not expose
granular progress. Model downloads still show real percent progress.

The Runtime panel can open `package-report.txt` in the app data folder. That
report lists the direct install commands and, after setup has run, the exact
packages currently installed in the app-managed Python environment.

The app also installs `addict` because DA3 imports it in model/runtime modules
but it is not currently listed in DA3's project dependency metadata.

The repository does not include model weights, generated images, installers, or
local Python environments.

## Model And License Notes

The application code is MIT licensed. Downloaded MDA and Depth Anything models,
checkpoints, and source projects remain governed by their own licenses and
attribution requirements.

Model pages:

- https://huggingface.co/sy000/MDA
- https://huggingface.co/depth-anything/DA3-LARGE-1.1
- https://huggingface.co/depth-anything/DA3-SMALL

MDA:

- https://github.com/biansy000/MDA

Depth Anything 3:

- https://github.com/ByteDance-Seed/depth-anything-3
