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
- Device selector: Auto, CUDA, Apple MPS, CPU.
- NVIDIA VRAM usage display when `nvidia-smi` is available.
- Orange/black scalable 16:9 UI.

## Development

Prerequisites:

- Node.js 20+
- Rust toolchain for Tauri builds
- Python 3.10+
- Git

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

The first generation run creates a Python virtual environment in the app data
folder. WebP-only runs install only Pillow. Depth-map runs install the ML
runtime, install Depth Anything 3 from GitHub, and download the selected
Hugging Face model.

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
