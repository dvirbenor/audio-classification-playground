"""Runtime presets for WavLM affect/disfluency inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import shutil
import sysconfig
from typing import Literal

import numpy as np

WAVLM_RUNTIME_PRESET_CHOICES = ("fast_exact", "compiled_static")
WAVLM_PREPROCESSING_POLICY = "wavlm_vectorized_znorm_v1"
WAVLM_STATIC_BATCH_PADDING_POLICY = "raw_zero_pad_before_znorm_v1"
WAVLM_COMPILED_STATIC_BATCH_SIZE = 256
WAVLM_COMPILE_MODE = "default"
WAVLM_LEGACY_COMPILE_MODE = "reduce-overhead"
WAVLM_WARMUP_WARNING_SEC = 60.0

WavLMRuntimePreset = Literal["fast_exact", "compiled_static", "custom"]


@dataclass(frozen=True)
class WavLMRuntimeSettings:
    """Resolved WavLM runtime knobs for worker-level presets."""

    requested_preset: str | None
    preset: WavLMRuntimePreset
    device: str
    task_batch_size: int | None
    autocast_dtype: str | None
    compile_model: bool
    compile_mode: str
    compile_dynamic: bool
    stream_layer_sum: bool
    allow_tf32: bool
    static_batch: bool
    warmup: bool


def resolve_wavlm_device(device: str | None) -> str:
    if device:
        return str(device)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def is_cuda_device(device: str | None) -> bool:
    return str(device or "").split(":", 1)[0].lower() == "cuda"


def has_custom_wavlm_runtime_knobs(
    *,
    autocast_dtype: str | None,
    compile_model: bool,
    compile_mode: str,
    compile_dynamic: bool,
    stream_layer_sum: bool,
    allow_tf32: bool,
) -> bool:
    """Return whether granular WavLM runtime knobs were requested."""
    return (
        autocast_dtype is not None
        or bool(compile_model)
        or str(compile_mode) != WAVLM_LEGACY_COMPILE_MODE
        or bool(compile_dynamic)
        or bool(stream_layer_sum)
        or bool(allow_tf32)
    )


def resolve_wavlm_runtime_settings(
    *,
    preset: str | None,
    device: str | None,
    autocast_dtype: str | None,
    compile_model: bool,
    compile_mode: str,
    compile_dynamic: bool,
    stream_layer_sum: bool,
    allow_tf32: bool,
    static_batch_size: int | None = None,
) -> WavLMRuntimeSettings:
    """Resolve orchestration WavLM presets to the existing low-level knobs.

    ``static_batch_size`` overrides the compiled-static batch dimension (default
    ``WAVLM_COMPILED_STATIC_BATCH_SIZE``). Larger fixed batches better fill big
    GPUs (e.g. Blackwell); the value is excluded from the semantic
    ``inference_config_hash`` so it never changes artifact identity.
    """
    resolved_device = resolve_wavlm_device(device)
    granular = has_custom_wavlm_runtime_knobs(
        autocast_dtype=autocast_dtype,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_dynamic=compile_dynamic,
        stream_layer_sum=stream_layer_sum,
        allow_tf32=allow_tf32,
    )
    if preset is not None and preset not in WAVLM_RUNTIME_PRESET_CHOICES:
        raise ValueError(
            "wavlm_runtime_preset must be one of: "
            + ", ".join(WAVLM_RUNTIME_PRESET_CHOICES)
        )
    if preset is not None and granular:
        raise ValueError(
            "WavLM runtime presets cannot be combined with granular WavLM knobs; "
            "omit the preset for custom experiments."
        )
    if static_batch_size is not None:
        if int(static_batch_size) <= 0:
            raise ValueError("wavlm_static_batch_size must be a positive integer")
        if granular:
            raise ValueError(
                "wavlm_static_batch_size only applies to the compiled_static "
                "preset; it cannot be combined with granular WavLM knobs."
            )
        if preset == "fast_exact":
            raise ValueError(
                "wavlm_static_batch_size requires the compiled_static preset "
                "(fast_exact does not use a static batch)."
            )

    if preset is None and granular:
        return WavLMRuntimeSettings(
            requested_preset=None,
            preset="custom",
            device=resolved_device,
            task_batch_size=None,
            autocast_dtype=autocast_dtype,
            compile_model=bool(compile_model),
            compile_mode=str(compile_mode),
            compile_dynamic=bool(compile_dynamic),
            stream_layer_sum=bool(stream_layer_sum),
            allow_tf32=bool(allow_tf32),
            static_batch=False,
            warmup=False,
        )

    requested = preset
    if preset is None:
        preset = (
            "compiled_static"
            if wavlm_compiled_static_is_eligible(resolved_device)
            else "fast_exact"
        )

    if preset == "compiled_static":
        if not wavlm_compiled_static_is_eligible(resolved_device):
            preset = "fast_exact"
        else:
            return WavLMRuntimeSettings(
                requested_preset=requested,
                preset="compiled_static",
                device=resolved_device,
                task_batch_size=(
                    int(static_batch_size)
                    if static_batch_size is not None
                    else WAVLM_COMPILED_STATIC_BATCH_SIZE
                ),
                # fp16 autocast is the validated default (event-level A/B passed on
                # affect/disfluency/emotion; in inference_config_hash so artifacts don't collide).
                autocast_dtype="fp16",
                compile_model=True,
                compile_mode=WAVLM_COMPILE_MODE,
                compile_dynamic=False,
                stream_layer_sum=False,
                allow_tf32=False,
                static_batch=True,
                warmup=True,
            )

    return WavLMRuntimeSettings(
        requested_preset=requested,
        preset="fast_exact",
        device=resolved_device,
        task_batch_size=None,
        autocast_dtype=None,
        compile_model=False,
        compile_mode=WAVLM_LEGACY_COMPILE_MODE,
        compile_dynamic=False,
        stream_layer_sum=False,
        allow_tf32=False,
        static_batch=False,
        warmup=False,
    )


def wavlm_compiled_static_is_eligible(device: str | None) -> bool:
    """Return whether this process can use the compiled-static WavLM preset."""
    if not is_cuda_device(device):
        return False
    try:
        import torch

        if not torch.cuda.is_available() or not hasattr(torch, "compile"):
            return False
    except Exception:
        return False
    if shutil.which("gcc") is None and shutil.which("cc") is None:
        return False
    include_dir = sysconfig.get_paths().get("include")
    if not include_dir or not (Path(include_dir) / "Python.h").is_file():
        return False
    return True


def inductor_cache_status() -> dict[str, object]:
    """Return lightweight diagnostics for the current Inductor cache env."""
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not cache_dir:
        return {"configured": False, "writable": False}
    path = Path(cache_dir)
    return {
        "configured": True,
        "path": str(path),
        "writable": path.exists() and os.access(path, os.W_OK),
    }


def configure_inductor_cache_namespace(*, preset: str) -> dict[str, object]:
    """Namespace ``TORCHINDUCTOR_CACHE_DIR`` for the current runtime stack."""
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not cache_dir:
        return {"configured": False, "writable": False}

    namespace = wavlm_inductor_cache_namespace(preset=preset)
    root = Path(os.environ.get("_ACP_WAVLM_INDUCTOR_CACHE_ROOT", cache_dir))
    os.environ["_ACP_WAVLM_INDUCTOR_CACHE_ROOT"] = str(root)
    path = root if root.name == namespace else root / namespace
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(path)
        return {
            "configured": True,
            "path": str(path),
            "namespace": namespace,
            "writable": False,
        }

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(path)
    return {
        "configured": True,
        "path": str(path),
        "namespace": namespace,
        "writable": os.access(path, os.W_OK),
    }


def wavlm_inductor_cache_namespace(*, preset: str) -> str:
    """Build the cache namespace from runtime versions that invalidate kernels."""
    torch_version = "unknown-torch"
    cuda_version = "unknown-cuda"
    gpu_arch = "unknown-arch"
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_version = str(getattr(torch.version, "cuda", None) or "cpu")
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            gpu_arch = f"sm{major}{minor}"
    except Exception:
        pass
    runtime_revision = (
        os.environ.get("IMAGE_REVISION")
        or os.environ.get("IMAGE_TAG")
        or os.environ.get("GIT_SHA")
        or os.environ.get("CODE_VERSION")
        or "unknown-runtime"
    )
    return "wavlm_inductor_" + "_".join(
        _safe_cache_token(part)
        for part in (torch_version, cuda_version, gpu_arch, runtime_revision, preset)
    )


def _safe_cache_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return token or "unknown"


def pad_windows_to_static_batch(
    windows: np.ndarray,
    *,
    batch_size: int,
    enabled: bool,
) -> tuple[np.ndarray, int]:
    """Zero-pad raw waveform windows to a static batch multiple."""
    n = int(len(windows))
    if not enabled or n == 0:
        return windows, n
    remainder = n % int(batch_size)
    if remainder == 0:
        return windows, n
    pad_count = int(batch_size) - remainder
    shape = (pad_count, *windows.shape[1:])
    padding = np.zeros(shape, dtype=np.float32)
    padded = np.concatenate(
        [np.asarray(windows, dtype=np.float32), padding],
        axis=0,
    )
    return padded, n
