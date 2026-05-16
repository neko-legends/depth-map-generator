from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any
from tempfile import TemporaryDirectory


DEFAULT_MODEL_REPO = "depth-anything/DA3-LARGE-1.1"
DA3_REPO = "git+https://github.com/ByteDance-Seed/depth-anything-3.git"
DEPTH_DEPS_MARKER = ".depthmapai-depth-deps-v2-ready"
MODEL_SIZES = {
    "depth-anything/DA3-LARGE-1.1": 1_643_843_860,
    "depth-anything/DA3-SMALL": 137_000_000,
}
MODEL_SIDE_FILES = (".gitattributes", "README.md", "config.json")
PYTHON_VERSION_RE = re.compile(r"Python\s+(\d+)\.(\d+)\.(\d+)")


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


def sanitize_installer_line(line: str) -> str | None:
    if line.startswith("Collecting "):
        package = line.removeprefix("Collecting ").split()[0]
        return f"Resolving dependency: {package}"
    if line.startswith("Downloading "):
        return "Downloading dependency packages"
    if line.startswith("Installing collected packages"):
        return "Installing dependency packages"
    if line.startswith("Successfully installed"):
        return "Dependency install step finished"
    if line.startswith("ERROR:") or " error:" in line.lower() or "failed" in line.lower():
        return line
    return None


def run_checked(command: list[str], message: str) -> None:
    emit("setup", f"{message}. This can take several minutes.")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    tail: list[str] = []
    last_status: str | None = None
    for line in process.stdout:
        line = line.strip()
        if line:
            tail.append(line)
            tail = tail[-18:]
            status = sanitize_installer_line(line)
            if status and status != last_status:
                emit("setup", status)
                last_status = status
    code = process.wait()
    if code != 0:
        details = "\n".join(tail) if tail else "No installer output was captured."
        raise RuntimeError(
            f"{message} failed. The app uses its own virtual environment, but it needs "
            f"a compatible system Python to create it. Install Python 3.10, 3.11, or 3.12 "
            f"and make it available on PATH, then try again.\n\nLast installer output:\n{details}"
        )


def parse_python_version(text: str) -> tuple[int, int, int] | None:
    match = PYTHON_VERSION_RE.search(text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def supported_python_version(version: tuple[int, int, int] | None) -> bool:
    return bool(version and version[0] == 3 and 10 <= version[1] <= 12)


def python_version(python_path: Path) -> tuple[int, int, int] | None:
    try:
        output = subprocess.check_output(
            [str(python_path), "--version"],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return None
    return parse_python_version(output)


def ensure_env(env_dir: Path, need_depth: bool) -> Path:
    python_path = venv_python(env_dir)
    if python_path.exists() and not supported_python_version(python_version(python_path)):
        emit("setup", "Recreating app Python environment with a supported Python version")
        shutil.rmtree(env_dir, ignore_errors=True)

    if not python_path.exists():
        emit("setup", f"Creating Python environment: {env_dir}")
        venv.EnvBuilder(with_pip=True).create(env_dir)

    if not supported_python_version(python_version(python_path)):
        raise RuntimeError(
            "DepthMap AI Generator requires Python 3.10, 3.11, or 3.12 for the app-managed "
            "virtual environment."
        )

    webp_marker = env_dir / ".depthmapai-webp-ready"
    depth_marker = env_dir / ".depthmapai-depth-ready"
    depth_deps_marker = env_dir / DEPTH_DEPS_MARKER
    old_marker = env_dir / ".depthmapai-ready"
    if old_marker.exists():
        webp_marker.write_text("ok\n", encoding="utf-8")
        depth_marker.write_text("ok\n", encoding="utf-8")
    if webp_marker.exists() and (
        not need_depth or (depth_marker.exists() and depth_deps_marker.exists())
    ):
        return python_path

    pip = [str(python_path), "-m", "pip"]
    run_checked(pip + ["install", "--upgrade", "pip", "setuptools", "wheel"], "Updating pip")
    if not webp_marker.exists():
        run_checked(pip + ["install", "pillow"], "Installing WebP conversion package")
        webp_marker.write_text("ok\n", encoding="utf-8")
    if need_depth and not depth_deps_marker.exists():
        run_checked(
            pip
            + [
                "install",
                "torch>=2",
                "torchvision",
                "huggingface_hub",
                "requests",
                "numpy<2",
                "pillow",
                "opencv-python",
                "safetensors",
                "addict",
            ],
            "Installing ML runtime packages",
        )
        run_checked(pip + ["install", DA3_REPO], "Installing Depth Anything 3")
        depth_marker.write_text("ok\n", encoding="utf-8")
        depth_deps_marker.write_text("ok\n", encoding="utf-8")
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


def save_depth(
    depth: Any,
    output_path: Path,
    output_format: str,
    invert: bool,
    normalize_low: float,
    normalize_high: float,
    gamma: float,
) -> None:
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
    normalize_low = max(0.0, min(99.0, float(normalize_low)))
    normalize_high = max(normalize_low + 0.1, min(100.0, float(normalize_high)))
    low, high = np.percentile(valid, [normalize_low, normalize_high])
    if high <= low:
        low = float(valid.min())
        high = float(valid.max())
    if high <= low:
        normalized = np.zeros_like(depth_array, dtype=np.float32)
    else:
        normalized = np.clip((depth_array - low) / (high - low), 0.0, 1.0)

    if invert:
        normalized = 1.0 - normalized

    gamma = max(0.2, min(4.0, float(gamma)))
    if abs(gamma - 1.0) > 0.001:
        normalized = np.power(normalized, gamma)

    normalized[~finite] = 0.0

    if output_format == "png16":
        image = Image.fromarray((normalized * 65535.0).round().astype(np.uint16), mode="I;16")
    else:
        image = Image.fromarray((normalized * 255.0).round().astype(np.uint8), mode="L")
    image.save(output_path)


def emit_download_progress(
    model_repo: str,
    downloaded: int,
    total: int,
    message: str | None = None,
) -> None:
    percent = min(100.0, round((downloaded / total) * 100, 1)) if total > 0 else None
    emit(
        "download_progress",
        message or f"Downloading {model_repo.split('/')[-1]}",
        modelRepo=model_repo,
        bytesDownloaded=downloaded,
        bytesTotal=total,
        percent=percent,
    )


def download_model_file(model_repo: str, model_dir: Path) -> None:
    import requests

    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "model.safetensors"
    temp_file = model_dir / "model.safetensors.incomplete"
    expected_size = MODEL_SIZES.get(model_repo, 0)
    url = f"https://huggingface.co/{model_repo}/resolve/main/model.safetensors?download=true"

    if temp_file.exists():
        temp_file.unlink()

    emit_download_progress(model_repo, 0, expected_size, f"Downloading {model_repo.split('/')[-1]}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or expected_size)
        downloaded = 0
        next_emit = 0
        with temp_file.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_emit or downloaded == total:
                    emit_download_progress(model_repo, downloaded, total)
                    next_emit = downloaded + max(total // 100, 8 * 1024 * 1024)

    actual_size = temp_file.stat().st_size
    if expected_size and actual_size < expected_size * 0.95:
        temp_file.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded model file looks incomplete: {actual_size} bytes of expected {expected_size}."
        )
    temp_file.replace(model_file)
    emit_download_progress(model_repo, actual_size, actual_size, f"Downloaded {model_repo.split('/')[-1]}")


def ensure_model(model_repo: str, model_dir: Path) -> Path:
    model_file = model_dir / "model.safetensors"
    if model_file.exists():
        emit_download_progress(model_repo, model_file.stat().st_size, model_file.stat().st_size, f"{model_repo.split('/')[-1]} already installed")
        return model_dir

    from huggingface_hub import hf_hub_download

    emit("model", f"Downloading {model_repo}")
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in MODEL_SIDE_FILES:
        try:
            hf_hub_download(
                repo_id=model_repo,
                filename=filename,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )
        except Exception as error:  # noqa: BLE001 - sidecar files vary by repo.
            emit("setup", f"Skipped {filename}: {error}")
    download_model_file(model_repo, model_dir)
    return model_dir


def run_job(job: dict[str, Any]) -> None:
    options = job["options"]
    paths = [Path(path) for path in job["paths"]]
    generate_depth = bool(options.get("generateDepth", True))
    convert_webp = bool(options.get("convertWebp", False))
    model = None
    torch = None
    download_only = bool(job.get("download_only", False))

    if download_only:
        model_repo = options.get("modelRepo", DEFAULT_MODEL_REPO)
        ensure_model(model_repo, Path(job["model_dir"]))
        emit("done", f"{model_repo.split('/')[-1]} is ready.")
        return

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
                        prediction = model.inference(
                            [str(inference_path)],
                            process_res=int(options.get("processRes", 504)),
                            process_res_method=options.get("processResMethod", "upper_bound_resize"),
                        )
                save_depth(
                    prediction.depth[0],
                    output_path,
                    options["outputFormat"],
                    bool(options["invert"]),
                    float(options.get("normalizeLow", 1.0)),
                    float(options.get("normalizeHigh", 99.0)),
                    float(options.get("depthGamma", 1.0)),
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
