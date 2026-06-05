from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import venv
import zipfile
from pathlib import Path
from typing import Any
from tempfile import TemporaryDirectory


MDA_MODEL_REPO = "sy000/MDA/mda_mog_sky_l2"
DEFAULT_MODEL_REPO = MDA_MODEL_REPO
DA3_COMMIT = "41736238f5bced4debf3f2a12375d2466874866d"
DA3_REPO = f"git+https://github.com/ByteDance-Seed/depth-anything-3.git@{DA3_COMMIT}"
MDA_HF_REPO = "sy000/MDA"
MDA_CHECKPOINT_FILENAME = "DA3_MOG_Sky_LogL2.ckpt"
MDA_MODEL_NAME = "mda_mog_sky_l2"
MDA_SOURCE_COMMIT = "0a4ae22e6994a9794dfb24a1275c7fbf59112f2a"
MDA_SOURCE_URL = f"https://github.com/biansy000/MDA/archive/{MDA_SOURCE_COMMIT}.zip"
DEPTH_DEPS_MARKER = ".depthmapai-depth-deps-v4-ready"
MDA_DEPS_MARKER = ".depthmapai-mda-deps-v2-ready"
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
BOOTSTRAP_PACKAGES = ["pip==26.1.1", "setuptools==70.2.0", "wheel==0.47.0"]
WEBP_PACKAGES = ["pillow==12.2.0"]
ML_PACKAGES = [
    "huggingface_hub==1.15.0",
    "requests==2.34.2",
    "numpy==1.26.4",
    "pillow==12.2.0",
    "opencv-python==4.11.0.86",
    "safetensors==0.7.0",
    "addict==2.4.0",
]
PYTORCH_CPU_PACKAGES = ["torch==2.11.0", "torchvision==0.26.0"]
PYTORCH_CUDA_PACKAGES = ["torch==2.11.0+cu128", "torchvision==0.26.0+cu128"]
MDA_RUNTIME_PACKAGES = [
    "PyYAML==6.0.3",
    "omegaconf==2.3.0",
    "einops==0.8.2",
    "scipy==1.17.1",
    "tqdm==4.67.3",
    "sympy==1.14.0",
    "lightning-bolts==0.7.0",
]
MODEL_SIZES = {
    MDA_MODEL_REPO: 5_423_360_059,
    "depth-anything/DA3-LARGE-1.1": 1_643_843_860,
    "depth-anything/DA3-SMALL": 137_000_000,
}
MODEL_SIDE_FILES = (".gitattributes", "README.md", "config.json")
PYTHON_VERSION_RE = re.compile(r"Python\s+(\d+)\.(\d+)\.(\d+)")
MDA_ACTIVE_PATHS: list[str] = []
os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
logging.getLogger("xformers").setLevel(logging.ERROR)


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


def nvidia_smi_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except Exception:
        return False
    return True


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
    run_checked(pip + ["install", "--upgrade", *BOOTSTRAP_PACKAGES], "Updating pip")
    if not webp_marker.exists():
        run_checked(pip + ["install", *WEBP_PACKAGES], "Installing WebP conversion package")
        webp_marker.write_text("ok\n", encoding="utf-8")
    if need_depth and not depth_deps_marker.exists():
        has_nvidia_gpu = nvidia_smi_available()
        torch_install = [
            "install",
            "--upgrade",
            *(PYTORCH_CUDA_PACKAGES if has_nvidia_gpu else PYTORCH_CPU_PACKAGES),
        ]
        if has_nvidia_gpu:
            torch_install += ["--force-reinstall", "--index-url", PYTORCH_CUDA_INDEX_URL]

        run_checked(
            pip + ["install", *ML_PACKAGES],
            "Installing ML runtime packages",
        )
        run_checked(
            pip + torch_install,
            "Installing CUDA PyTorch packages" if has_nvidia_gpu else "Installing PyTorch packages",
        )
        run_checked(pip + ["install", "--upgrade", "numpy==1.26.4"], "Pinning compatible NumPy")
        run_checked(pip + ["install", DA3_REPO], "Installing Depth Anything 3")
        depth_marker.write_text("ok\n", encoding="utf-8")
        depth_deps_marker.write_text("ok\n", encoding="utf-8")
    return python_path


def is_mda_model(model_repo: str) -> bool:
    return model_repo == MDA_MODEL_REPO


def model_display_name(model_repo: str) -> str:
    if is_mda_model(model_repo):
        return "MDA-DA3-MoG-Sky"
    return model_repo.split("/")[-1]


def ensure_mda_python_deps(env_dir: Path) -> None:
    marker = env_dir / MDA_DEPS_MARKER
    if marker.exists():
        return
    pip = [sys.executable, "-m", "pip"]
    run_checked(
        pip
        + ["install", *MDA_RUNTIME_PACKAGES],
        "Installing MDA runtime packages",
    )
    marker.write_text("ok\n", encoding="utf-8")


def ensure_mda_source(model_dir: Path) -> Path:
    import requests

    source_dir = model_dir / "source"
    if (source_dir / "demo.py").exists() and (source_dir / "src" / "testing" / "utils" / "model_choice.py").exists():
        return source_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    incomplete_dir = model_dir / "source.incomplete"
    shutil.rmtree(incomplete_dir, ignore_errors=True)

    emit("setup", "Downloading MDA source")
    with TemporaryDirectory(prefix="depthmapai-mda-source-") as temp:
        archive_path = Path(temp) / "mda.zip"
        with requests.get(MDA_SOURCE_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with archive_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(incomplete_dir)

    roots = [path for path in incomplete_dir.iterdir() if path.is_dir()]
    if len(roots) != 1:
        shutil.rmtree(incomplete_dir, ignore_errors=True)
        raise RuntimeError("Downloaded MDA source archive had an unexpected layout.")

    shutil.rmtree(source_dir, ignore_errors=True)
    roots[0].replace(source_dir)
    shutil.rmtree(incomplete_dir, ignore_errors=True)
    return source_dir


def ensure_mda_checkpoint(model_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    source_dir = ensure_mda_source(model_dir)
    checkpoint_dir = source_dir / "checkpoints" / "MDA"
    checkpoint_path = checkpoint_dir / MDA_CHECKPOINT_FILENAME
    if checkpoint_path.exists():
        emit_download_progress(
            MDA_MODEL_REPO,
            checkpoint_path.stat().st_size,
            checkpoint_path.stat().st_size,
            f"{model_display_name(MDA_MODEL_REPO)} already installed",
        )
        return checkpoint_path

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    emit_download_progress(
        MDA_MODEL_REPO,
        0,
        MODEL_SIZES[MDA_MODEL_REPO],
        f"Downloading {model_display_name(MDA_MODEL_REPO)}",
    )
    hf_hub_download(
        repo_id=MDA_HF_REPO,
        filename=MDA_CHECKPOINT_FILENAME,
        local_dir=str(checkpoint_dir),
        local_dir_use_symlinks=False,
    )
    if not checkpoint_path.exists():
        raise RuntimeError(f"MDA checkpoint download did not create {checkpoint_path}")
    emit_download_progress(
        MDA_MODEL_REPO,
        checkpoint_path.stat().st_size,
        checkpoint_path.stat().st_size,
        f"Downloaded {model_display_name(MDA_MODEL_REPO)}",
    )
    return checkpoint_path


def clear_depth_modules() -> None:
    prefixes = ("depth_anything_3", "src.depth_anything_3", "src.training", "src.testing", "src.stream3r")
    for name in list(sys.modules):
        if name in prefixes or any(name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(name, None)


def deactivate_mda_source() -> None:
    global MDA_ACTIVE_PATHS
    if not MDA_ACTIVE_PATHS:
        return
    active = {str(Path(path)) for path in MDA_ACTIVE_PATHS}
    sys.path[:] = [path for path in sys.path if str(Path(path)) not in active]
    MDA_ACTIVE_PATHS = []
    clear_depth_modules()


def activate_mda_source(source_dir: Path) -> None:
    global MDA_ACTIVE_PATHS
    deactivate_mda_source()
    paths = [str(source_dir), str(source_dir / "src")]
    for path in reversed(paths):
        sys.path.insert(0, path)
    MDA_ACTIVE_PATHS = paths
    clear_depth_modules()


def relaunch_in_env(job_path: Path, env_dir: Path, need_depth: bool, daemon: bool) -> int:
    python_path = ensure_env(env_dir, need_depth)
    env = os.environ.copy()
    env["DEPTHMAPAI_VENV_ACTIVE"] = "1"
    env.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    args = [str(python_path), str(Path(__file__).resolve())]
    if daemon:
        args.append("--daemon")
    args.extend(["--job", str(job_path)])
    process = subprocess.Popen(
        args,
        stdin=subprocess.PIPE if daemon else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert process.stdout is not None
    if daemon:
        assert process.stdin is not None

        def forward_stdin() -> None:
            try:
                for line in sys.stdin:
                    process.stdin.write(line)
                    process.stdin.flush()
            except Exception:
                pass
            finally:
                try:
                    process.stdin.close()
                except Exception:
                    pass

        threading.Thread(target=forward_stdin, daemon=True).start()

    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def choose_device(requested: str):
    import torch

    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU was requested, but torch.cuda is not available.")
        try:
            index = int(requested.split(":", 1)[1])
        except ValueError as error:
            raise RuntimeError(f"Invalid CUDA device: {requested}") from error
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {index} was requested, but {torch.cuda.device_count()} CUDA device(s) are available."
            )
        return torch.device(requested)
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


def output_dir_for(image_path: Path, options: dict[str, Any]) -> Path:
    output_dir = str(options.get("outputDir") or "").strip()
    directory = Path(output_dir).expanduser() if output_dir else image_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise RuntimeError(f"Output folder is not a directory: {directory}")
    return directory


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available output name for {path.name}.")


def final_output_path(path: Path, overwrite: bool) -> Path:
    return path if overwrite else unique_path(path)


def output_path_for(
    image_path: Path,
    suffix: str,
    output_format: str,
    options: dict[str, Any],
    overwrite: bool,
) -> Path:
    extension = ".npy" if output_format == "npy" else ".png"
    base_path = output_dir_for(image_path, options) / f"{image_path.stem}{suffix}{extension}"
    return final_output_path(base_path, overwrite)


def webp_output_path_for(image_path: Path, options: dict[str, Any], overwrite: bool) -> Path:
    base_path = output_dir_for(image_path, options) / f"{image_path.stem}.webp"
    return final_output_path(base_path, overwrite)


def has_alpha(image: Any) -> bool:
    if image.mode in ("RGBA", "LA"):
        return True
    if image.mode == "P" and "transparency" in image.info:
        return True
    return False


def convert_source_to_webp(
    image_path: Path,
    options: dict[str, Any],
    overwrite: bool,
    quality: int,
) -> tuple[Path, str]:
    from PIL import Image, ImageOps

    output_path = webp_output_path_for(image_path, options, overwrite)
    if image_path.suffix.lower() == ".webp" and output_path.resolve() == image_path.resolve():
        return output_path, "Already WebP"

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
    if is_mda_model(model_repo):
        ensure_mda_checkpoint(model_dir)
        return model_dir

    model_file = model_dir / "model.safetensors"
    if model_file.exists():
        emit_download_progress(model_repo, model_file.stat().st_size, model_file.stat().st_size, f"{model_display_name(model_repo)} already installed")
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


def prepare_mda_view(image_path: Path, size: int, patch_size: int, img_norm: Any) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    import numpy as np
    import torch
    from PIL import Image
    from PIL.ImageOps import exif_transpose

    with Image.open(image_path) as opened:
        image = exif_transpose(opened).convert("RGB")
        original_size = image.size
        width, height = image.size
        scale = max(1, int(size)) / max(width, height)
        resized_size = (
            max(patch_size, int(round(width * scale))),
            max(patch_size, int(round(height * scale))),
        )
        image = image.resize(resized_size, Image.Resampling.BICUBIC)

    target_width = max(patch_size, (image.size[0] // patch_size) * patch_size)
    target_height = max(patch_size, (image.size[1] // patch_size) * patch_size)
    if image.size != (target_width, target_height):
        image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)

    tensor = img_norm(image)[None]
    view = {
        "img": tensor,
        "ray_map": torch.full((tensor.shape[0], 6, tensor.shape[-2], tensor.shape[-1]), torch.nan),
        "true_shape": torch.from_numpy(np.int32([[tensor.shape[-2], tensor.shape[-1]]])),
        "idx": 0,
        "instance": "0",
        "camera_pose": torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(0),
        "img_mask": torch.tensor(True).unsqueeze(0),
        "ray_mask": torch.tensor(False).unsqueeze(0),
        "update": torch.tensor(True).unsqueeze(0),
        "reset": torch.tensor(False).unsqueeze(0),
    }
    return [view], original_size


def mda_depth_for_image(torch: Any, loaded: Any, image_path: Path, options: dict[str, Any], device: Any) -> Any:
    views, original_size = prepare_mda_view(
        image_path,
        int(options.get("processRes", 504)),
        int(getattr(loaded, "patch_size", 14)),
        loaded.img_norm,
    )
    with torch.enable_grad():
        prediction = loaded.model.inference(
            views,
            device,
            is_mog=True,
            use_sky_mask=True,
            crop_center_112=False,
            cam_inp=False,
            gt_cam_output=False,
            output_normalize=False,
        )
    try:
        loaded.model.zero_grad(set_to_none=True)
    except Exception:
        pass
    depth = prediction["depth"]
    if depth.ndim == 5:
        depth = depth.squeeze(0).squeeze(-1)
    else:
        depth = depth.squeeze(0)
    depth = depth[0].float().unsqueeze(0).unsqueeze(0)
    depth = torch.nn.functional.interpolate(
        depth,
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=False,
    )
    return depth[0, 0].detach().cpu().numpy()


class ModelCache:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.torch: Any | None = None
        self.model_repo: str | None = None
        self.device: str | None = None

    def clear(self) -> None:
        if self.model is not None:
            del self.model
        self.model = None
        if self.torch is not None and hasattr(self.torch, "cuda"):
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass
        self.model_repo = None
        self.device = None

    def load(self, options: dict[str, Any], model_dir: Path, env_dir: Path | None = None) -> tuple[Any, Any]:
        model_repo = options.get("modelRepo", DEFAULT_MODEL_REPO)
        emit("model", f"Preparing {model_display_name(model_repo)} runtime")
        import torch as torch_module

        device = choose_device(options["device"])
        device_key = str(device)

        if (
            self.model is not None
            and self.torch is not None
            and self.model_repo == model_repo
            and self.device == device_key
        ):
            emit(
                "model_ready",
                f"{model_display_name(model_repo)} is already loaded on {device_key}.",
                modelRepo=model_repo,
                device=device_key,
            )
            return self.torch, self.model

        self.clear()
        if is_mda_model(model_repo):
            if env_dir is not None:
                ensure_mda_python_deps(env_dir)
            ensure_mda_checkpoint(model_dir)
            source_dir = ensure_mda_source(model_dir)
            activate_mda_source(source_dir)
            import importlib

            model_choice = importlib.import_module("src.testing.utils.model_choice")
            model_choice.device = device
            emit("model", f"Loading {model_display_name(model_repo)} on {device_key}", modelRepo=model_repo, device=device_key)
            previous_cwd = Path.cwd()
            try:
                os.chdir(source_dir)
                model = model_choice.choose_model(MDA_MODEL_NAME)
            finally:
                os.chdir(previous_cwd)
            model = model._replace(model=model.model.to(device=device).eval())
        else:
            deactivate_mda_source()
            from depth_anything_3.api import DepthAnything3

            resolved_model_dir = ensure_model(model_repo, model_dir)
            emit("model", f"Loading {model_display_name(model_repo)} on {device_key}", modelRepo=model_repo, device=device_key)
            model = DepthAnything3.from_pretrained(str(resolved_model_dir))
            model = model.to(device=device)
            model.eval()

        self.torch = torch_module
        self.model = model
        self.model_repo = model_repo
        self.device = device_key
        emit(
            "model_ready",
            f"{model_display_name(model_repo)} loaded on {device_key}.",
            modelRepo=model_repo,
            device=device_key,
        )
        return torch_module, model


def warm_model(torch: Any, model: Any, options: dict[str, Any]) -> None:
    from PIL import Image

    model_repo = options.get("modelRepo", DEFAULT_MODEL_REPO)
    emit("model", "Warming model on a tiny test image")
    with TemporaryDirectory(prefix="depthmapai-warm-") as temp:
        warm_path = Path(temp) / "warmup.png"
        Image.new("RGB", (64, 64), (0, 0, 0)).save(warm_path)
        if is_mda_model(model_repo):
            mda_depth_for_image(torch, model, warm_path, {**options, "processRes": 112}, choose_device(options["device"]))
        else:
            with torch.inference_mode():
                model.inference(
                    [str(warm_path)],
                    process_res=336,
                    process_res_method=options.get("processResMethod", "upper_bound_resize"),
                )
    emit("model_ready", f"{model_display_name(model_repo)} warmed and ready.", modelRepo=model_repo)


def run_job(job: dict[str, Any], cache: ModelCache | None = None) -> None:
    options = job["options"]
    paths = [Path(path) for path in job["paths"]]
    generate_depth = bool(options.get("generateDepth", True))
    convert_webp = bool(options.get("convertWebp", False))
    model = None
    torch = None
    download_only = bool(job.get("download_only", False))
    preload_only = bool(job.get("preload_only", False))
    overwrite = bool(options.get("overwrite", False))

    if preload_only:
        if not generate_depth:
            emit("done", "Depth generation is disabled; model preload skipped.")
            return
        model_cache = cache or ModelCache()
        torch, model = model_cache.load(options, Path(job["model_dir"]), Path(job["env_dir"]))
        warm_model(torch, model, options)
        emit("done", f"{model_display_name(options.get('modelRepo', DEFAULT_MODEL_REPO))} is loaded and ready.")
        return

    if download_only:
        model_repo = options.get("modelRepo", DEFAULT_MODEL_REPO)
        ensure_model(model_repo, Path(job["model_dir"]))
        emit("done", f"{model_display_name(model_repo)} is ready.")
        return

    if not generate_depth and not convert_webp:
        raise RuntimeError("No outputs selected. Enable depth generation or WebP conversion.")

    if generate_depth:
        model_cache = cache or ModelCache()
        torch, model = model_cache.load(options, Path(job["model_dir"]), Path(job["env_dir"]))

    for image_path in paths:
        output_path = output_path_for(
            image_path,
            options["suffix"],
            options["outputFormat"],
            options,
            overwrite,
        )
        outputs: list[str] = []

        try:
            emit("image_start", f"Processing {image_path.name}", path=str(image_path))
            if convert_webp:
                webp_path, webp_note = convert_source_to_webp(
                    image_path,
                    options,
                    overwrite,
                    int(options.get("webpQuality", 92)),
                )
                outputs.append(str(webp_path))
                emit("setup", f"{image_path.name}: {webp_note}")

            if generate_depth:
                assert torch is not None
                assert model is not None
                with TemporaryDirectory(prefix="depthmapai-") as temp:
                    inference_path = resized_input_for(image_path, options, Path(temp))
                    if is_mda_model(options.get("modelRepo", DEFAULT_MODEL_REPO)):
                        depth = mda_depth_for_image(torch, model, inference_path, options, choose_device(options["device"]))
                    else:
                        with torch.inference_mode():
                            prediction = model.inference(
                                [str(inference_path)],
                                process_res=int(options.get("processRes", 504)),
                                process_res_method=options.get("processResMethod", "upper_bound_resize"),
                            )
                            depth = prediction.depth[0]
                save_depth(
                    depth,
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


def daemon_loop() -> int:
    cache = ModelCache()
    emit("model", "Persistent worker ready.")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception as error:
            emit("error", f"Invalid worker command: {error}")
            continue
        if job.get("shutdown"):
            cache.clear()
            return 0
        try:
            run_job(job, cache)
        except Exception as error:  # noqa: BLE001 - top-level daemon command error.
            emit("error", str(error))
    cache.clear()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job_path = Path(args.job)
    job = json.loads(job_path.read_text(encoding="utf-8"))

    if os.environ.get("DEPTHMAPAI_VENV_ACTIVE") != "1":
        need_depth = args.daemon or bool(job.get("options", {}).get("generateDepth", True))
        return relaunch_in_env(job_path, Path(job["env_dir"]), need_depth, args.daemon)

    try:
        if args.daemon:
            return daemon_loop()
        run_job(job)
        return 0
    except Exception as error:  # noqa: BLE001 - top-level worker error.
        emit("error", str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
