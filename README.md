# DepthMap AI Generator

DepthMap AI Generator is a local desktop batch tool for creating depth maps and
WebP files from images and folders. It is designed for AI image workflows that
need ControlNet-style depth maps without running ComfyUI.

The app uses a Tauri 2 desktop shell with a React UI. Rust handles the desktop
window, drag and drop, folder scanning, worker orchestration, and VRAM polling.
Python handles Depth Anything 3 inference, first-run dependency setup, model
download, depth map export, and WebP conversion.

## Features

- Drag in one image, many images, or folders.
- Recursively scan image folders.
- Generate depth maps beside source images.
- Optional resize-before-depth mode with aspect ratio preserved.
- Depth outputs: 8-bit PNG, 16-bit PNG, or raw `.npy`.
- Optional source-image WebP conversion.
- PNG alpha is preserved with lossless WebP.
- JPG-style inputs use configurable WebP quality.
- Model selector:
  - `depth-anything/DA3-LARGE-1.1` by default.
  - `depth-anything/DA3-SMALL` for lower VRAM machines.
- First-launch model picker with Later option.
- Model download progress with percent and downloaded size.
- Buttons to open the model install folder and package report.
- Device selector: Auto, CUDA, Apple MPS, CPU.
- NVIDIA VRAM usage display when `nvidia-smi` is available.
- Orange/black scalable 16:9 UI.

## Development

Prerequisites:

- Node.js 20+
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

3. Install JavaScript dependencies.

```powershell
npm install
```

4. Run source checks.

```powershell
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
- `src-tauri\target\release\bundle\nsis\DepthMap AI Generator_0.1.0_x64-setup.exe`
- `src-tauri\target\release\bundle\msi\DepthMap AI Generator_0.1.0_x64_en-US.msi`

Runtime notes for agents:

- Do not commit `node_modules`, `dist`, `dist-portable`, `src-tauri/target`, model
  folders, app-data Python environments, generated depth maps, or installer
  files.
- The app downloads model weights at runtime. Do not add Hugging Face model
  files to the repository.
- WebP-only runs install Pillow in the app-managed Python environment.
- Depth runs install the ML stack, Depth Anything 3, and the selected Hugging
  Face model.
- Python 3.13+ is not selected for the app-managed environment because some ML
  dependencies may not publish compatible wheels yet.
- NVIDIA VRAM display depends on `nvidia-smi`; it is allowed to show unavailable
  on machines without NVIDIA drivers.

Install JavaScript dependencies:

```powershell
npm install
```

Run the frontend only:

```powershell
npm run dev
```

Run the desktop app:

```powershell
npm run tauri:dev
```

Build Windows installers and the release executable:

```powershell
npm run tauri:build
```

## Runtime Behavior

On first launch, the app offers to download either DA3-LARGE-1.1 or DA3-SMALL.
Choosing Later skips the download. If depth generation is later started with no
selected model installed, the worker downloads that model before inference and
shows download progress in the UI.

The first WebP or depth run creates an app-managed Python virtual environment in
the app data folder. It is separate from project `.venv` folders and avoids
mixing with custom Python environments used by other projects. The launcher
prefers Python 3.11, 3.12, or 3.10 and will not use Python 3.13+ for the managed
environment.

WebP-only runs install only Pillow. Depth-map runs install the ML runtime,
install Depth Anything 3 from GitHub, and download the selected Hugging Face
model when needed.

The Runtime panel can open `package-report.txt` in the app data folder. That
report lists the direct install commands and, after setup has run, the exact
packages currently installed in the app-managed Python environment.

The app also installs `addict` because DA3 imports it in model/runtime modules
but it is not currently listed in DA3's project dependency metadata.

The repository does not include model weights, generated images, installers, or
local Python environments.

## Model And License Notes

The application code is MIT licensed. The downloaded Depth Anything 3 models and
the Depth Anything 3 project remain governed by their own licenses and
attribution requirements.

Model pages:

- https://huggingface.co/depth-anything/DA3-LARGE-1.1
- https://huggingface.co/depth-anything/DA3-SMALL

Depth Anything 3:

- https://github.com/ByteDance-Seed/depth-anything-3
