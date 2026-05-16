from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any
from tempfile import TemporaryDirectory


DEFAULT_MODEL_REPO = "depth-anything/DA3-LARGE-1.1"
DA3_REPO = "git+https://github.com/ByteDance-Seed/depth-anything-3.git"


def emit(event_type: str, message: str | None = None, **payload: Any) -> None:
    event: dict[str, Any] = {"type": event_type}
    if message:
        event["message"] = message
    event.update(payload)
    print(json.dumps(event, ensure_ascii=False), flush=True)


def venv_python(env_dir: Path) -> Path:
    if platform.system() == "Windows":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def run_checked(command: list[str], message: str) -> None:
    emit("setup", message)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if line:
            emit("setup", line)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(command)}")


def ensure_env(env_dir: Path, need_depth: bool) -> Path:
    python_path = venv_python(env_dir)
    if not python_path.exists():
        emit("setup", f"Creating Python environment: {env_dir}")
        venv.EnvBuilder(with_pip=True).create(env_dir)

    webp_marker = env_dir / ".depthmapai-webp-ready"
    depth_marker = env_dir / ".depthmapai-depth-ready"
    old_marker = env_dir / ".depthmapai-ready"
    if old_marker.exists():
        webp_marker.write_text("ok\n", encoding="utf-8")
        depth_marker.write_text("ok\n", encoding="utf-8")
    if webp_marker.exists() and (not need_depth or depth_marker.exists()):
        return python_path

    pip = [str(python_path), "-m", "pip"]
    run_checked(pip + ["install", "--upgrade", "pip", "setuptools", "wheel"], "Updating pip")
    if not webp_marker.exists():
        run_checked(pip + ["install", "pillow"], "Installing WebP conversion package")
        webp_marker.write_text("ok\n", encoding="utf-8")
    if need_depth and not depth_marker.exists():
        run_checked(
            pip
            + [
                "install",
                "torch>=2",
                "torchvision",
                "huggingface_hub",
                "numpy<2",
                "pillow",
                "opencv-python",
                "safetensors",
            ],
            "Installing ML runtime packages",
        )
        run_checked(pip + ["install", DA3_REPO], "Installing Depth Anything 3")
        depth_marker.write_text("ok\n", encoding="utf-8")
    return python_path


def relaunch_in_env(job_path: Path, env_dir: Path, need_depth: bool) -> int:
    python_path = ensure_env(env_dir, need_depth)
    env = os.environ.copy()
    env["DEPTHMAPAI_VENV_ACTIVE"] = "1"
    process = subprocess.Popen(
        [str(python_path), str(Path(__file__).resolve()), "--job", str(job_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def choose_device(requested: str):
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS was requested, but it is not available.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def output_path_for(image_path: Path, suffix: str, output_format: str) -> Path:
    extension = ".npy" if output_format == "npy" else ".png"
    return image_path.with_name(f"{image_path.stem}{suffix}{extension}")


def webp_output_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".webp")


def has_alpha(image: Any) -> bool:
    if image.mode in ("RGBA", "LA"):
        return True
    if image.mode == "P" and "transparency" in image.info:
        return True
    return False


def convert_source_to_webp(image_path: Path, overwrite: bool, quality: int) -> tuple[Path, str]:
    from PIL import Image, ImageOps

    output_path = webp_output_path_for(image_path)
    if image_path.suffix.lower() == ".webp":
        return output_path, "Already WebP"
    if output_path.exists() and not overwrite:
        return output_path, "Skipped existing WebP"

    quality = max(1, min(100, int(quality)))
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        alpha = has_alpha(image)
        if alpha:
            image = image.convert("RGBA")
            image.save(output_path, "WEBP", lossless=True, method=6, exact=True)
            return output_path, "WebP saved with alpha"

        image = image.convert("RGB")
        if image_path.suffix.lower() == ".png":
            image.save(output_path, "WEBP", lossless=True, method=6)
            return output_path, "Lossless WebP saved"

        image.save(output_path, "WEBP", quality=quality, method=6)
        return output_path, f"WebP saved at quality {quality}"


def resized_input_for(image_path: Path, options: dict[str, Any], temp_dir: Path) -> Path:
    from PIL import Image

    mode = options.get("resizeMode", "none")
    if mode == "none":
        return image_path

    target = int(options.get("resizeValue") or 0)
    if target < 64:
        raise RuntimeError("Resize pixel value must be at least 64.")

    with Image.open(image_path) as image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise RuntimeError("Input image has invalid dimensions.")

        if mode == "width":
            scale = target / width
        elif mode == "height":
            scale = target / height
        elif mode == "long":
            scale = target / max(width, height)
        else:
            return image_path

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        if new_width == width and new_height == height:
            return image_path

        resized = image.convert("RGB").resize((new_width, new_height), Image.Resampling.LANCZOS)
        output_path = temp_dir / f"{image_path.stem}_{new_width}x{new_height}.png"
        resized.save(output_path)
        emit("setup", f"Resized {image_path.name}: {width}x{height} -> {new_width}x{new_height}")
        return output_path


def save_depth(depth: Any, output_path: Path, output_format: str, invert: bool) -> None:
    import numpy as np
    from PIL import Image

    depth_array = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth_array)
    if not finite.any():
        raise RuntimeError("Model returned a depth map with no finite values.")

    if output_format == "npy":
        np.save(output_path, depth_array)
        return

    valid = depth_array[finite]
    low, high = np.percentile(valid, [1.0, 99.0])
    if high <= low:
        low = float(valid.min())
        high = float(valid.max())
    if high <= low:
        normalized = np.zeros_like(depth_array, dtype=np.float32)
    else:
        normalized = np.clip((depth_array - low) / (high - low), 0.0, 1.0)

    if invert:
        normalized = 1.0 - normalized

    normalized[~finite] = 0.0

    if output_format == "png16":
        image = Image.fromarray((normalized * 65535.0).round().astype(np.uint16), mode="I;16")
    else:
        image = Image.fromarray((normalized * 255.0).round().astype(np.uint8), mode="L")
    image.save(output_path)


def ensure_model(model_repo: str, model_dir: Path) -> Path:
    model_file = model_dir / "model.safetensors"
    if model_file.exists():
        return model_dir

    from huggingface_hub import snapshot_download

    emit("model", f"Downloading {model_repo}")
    snapshot_download(
        repo_id=model_repo,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    return model_dir


def run_job(job: dict[str, Any]) -> None:
    options = job["options"]
    paths = [Path(path) for path in job["paths"]]
    generate_depth = bool(options.get("generateDepth", True))
    convert_webp = bool(options.get("convertWebp", False))
    model = None
    torch = None

    if not generate_depth and not convert_webp:
        raise RuntimeError("No outputs selected. Enable depth generation or WebP conversion.")

    if generate_depth:
        import torch as torch_module
        from depth_anything_3.api import DepthAnything3

        torch = torch_module
        model_repo = options.get("modelRepo", DEFAULT_MODEL_REPO)
        model_dir = ensure_model(model_repo, Path(job["model_dir"]))
        device = choose_device(options["device"])

        emit("model", f"Loading {model_repo.split('/')[-1]} on {device}")
        model = DepthAnything3.from_pretrained(str(model_dir))
        model = model.to(device=device)
        model.eval()

    for image_path in paths:
        output_path = output_path_for(image_path, options["suffix"], options["outputFormat"])
        outputs: list[str] = []

        if generate_depth and output_path.exists() and not options["overwrite"]:
            emit(
                "image_done",
                "Skipped existing output",
                path=str(image_path),
                output=str(output_path),
            )
            continue

        try:
            emit("image_start", f"Processing {image_path.name}", path=str(image_path))
            if convert_webp:
                webp_path, webp_note = convert_source_to_webp(
                    image_path,
                    bool(options["overwrite"]),
                    int(options.get("webpQuality", 92)),
                )
                outputs.append(str(webp_path))
                emit("setup", f"{image_path.name}: {webp_note}")

            if generate_depth:
                assert torch is not None
                assert model is not None
                with TemporaryDirectory(prefix="depthmapai-") as temp:
                    inference_path = resized_input_for(image_path, options, Path(temp))
                    with torch.inference_mode():
                        prediction = model.inference([str(inference_path)])
                save_depth(
                    prediction.depth[0],
                    output_path,
                    options["outputFormat"],
                    bool(options["invert"]),
                )
                outputs.append(str(output_path))

            emit(
                "image_done",
                f"Saved {len(outputs)} output{'' if len(outputs) == 1 else 's'}",
                path=str(image_path),
                output="; ".join(outputs),
            )
        except Exception as error:  # noqa: BLE001 - show per-file failures to the UI.
            emit("image_error", str(error), path=str(image_path))

    emit("done", "Batch finished.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job_path = Path(args.job)
    job = json.loads(job_path.read_text(encoding="utf-8"))

    if os.environ.get("DEPTHMAPAI_VENV_ACTIVE") != "1":
        need_depth = bool(job.get("options", {}).get("generateDepth", True))
        return relaunch_in_env(job_path, Path(job["env_dir"]), need_depth)

    try:
        run_job(job)
        return 0
    except Exception as error:  # noqa: BLE001 - top-level worker error.
        emit("error", str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
