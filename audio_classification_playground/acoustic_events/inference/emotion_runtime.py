"""Runtime presets for emotion2vec inference.

The production speedup comes from selecting an already-existing fast path:
resident direct scorer, fixed-size batches, PyTorch compile, and scoped TF32.
This module keeps those choices explicit so artifact manifests and actual
execution cannot drift apart.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Literal

EMOTION_RUNTIME_MODE_CHOICES = ("auto", "optimized", "fp32-eager", "custom")
DEFAULT_EMOTION_COMPILE_MODE = "default"
OPTIMIZED_EMOTION_BATCH_SIZE = 64
STANDALONE_EMOTION_BATCH_SIZE = 128

EmotionRuntimeMode = Literal["auto", "optimized", "fp32-eager", "custom"]


@dataclass(frozen=True)
class EmotionRuntimeSettings:
    """Resolved emotion2vec runtime knobs."""

    requested_mode: str
    mode: Literal["optimized", "fp32-eager", "custom"]
    device: str
    autocast_dtype: str | None
    compile_model: bool
    compile_mode: str
    allow_tf32: bool
    warmup: bool


def resolve_emotion_device(device: str | None) -> str:
    """Resolve the device string before loading FunASR."""
    if device:
        return str(device)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def is_cuda_device(device: str | None) -> bool:
    """Return whether a user/resolved device string targets CUDA."""
    return str(device or "").split(":", 1)[0].lower() == "cuda"


def _cuda_is_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def has_custom_emotion_runtime_knobs(
    *,
    autocast_dtype: str | None,
    compile_model: bool,
    compile_mode: str,
    allow_tf32: bool,
) -> bool:
    """Return whether granular emotion runtime knobs were requested."""
    return (
        autocast_dtype is not None
        or bool(compile_model)
        or str(compile_mode) != DEFAULT_EMOTION_COMPILE_MODE
        or bool(allow_tf32)
    )


def resolve_emotion_runtime_settings(
    *,
    mode: str | None,
    default_mode: str,
    device: str | None,
    autocast_dtype: str | None,
    compile_model: bool,
    compile_mode: str,
    allow_tf32: bool,
) -> EmotionRuntimeSettings:
    """Resolve emotion2vec runtime mode and low-level knobs.

    ``auto`` resolves to the optimized preset on CUDA and FP32 eager elsewhere.
    Explicit presets reject granular runtime knobs; use ``custom`` for
    experiments so the manifest exactly records the executed knobs.
    """
    resolved_device = resolve_emotion_device(device)
    granular = has_custom_emotion_runtime_knobs(
        autocast_dtype=autocast_dtype,
        compile_model=compile_model,
        compile_mode=compile_mode,
        allow_tf32=allow_tf32,
    )
    requested_mode = mode or ("custom" if granular else default_mode)
    if requested_mode not in EMOTION_RUNTIME_MODE_CHOICES:
        raise ValueError(
            "emotion_runtime_mode must be one of: "
            + ", ".join(EMOTION_RUNTIME_MODE_CHOICES)
        )
    if requested_mode != "custom" and granular:
        raise ValueError(
            "Emotion runtime presets cannot be combined with granular knobs; "
            "use emotion_runtime_mode='custom' for experiments."
        )

    if requested_mode == "auto":
        requested_mode = "optimized" if is_cuda_device(resolved_device) else "fp32-eager"

    if requested_mode == "optimized":
        if not is_cuda_device(resolved_device) or not _cuda_is_available():
            raise ValueError("emotion_runtime_mode='optimized' requires a CUDA device")
        return EmotionRuntimeSettings(
            requested_mode=mode or "auto",
            mode="optimized",
            device=resolved_device,
            # fp16 autocast is the validated default (emotion event-level A/B passed; in config hash).
            autocast_dtype="fp16",
            compile_model=True,
            compile_mode=DEFAULT_EMOTION_COMPILE_MODE,
            allow_tf32=True,
            warmup=True,
        )

    if requested_mode == "fp32-eager":
        return EmotionRuntimeSettings(
            requested_mode=mode or default_mode,
            mode="fp32-eager",
            device=resolved_device,
            autocast_dtype=None,
            compile_model=False,
            compile_mode=DEFAULT_EMOTION_COMPILE_MODE,
            allow_tf32=False,
            warmup=False,
        )

    return EmotionRuntimeSettings(
        requested_mode=mode or "custom",
        mode="custom",
        device=resolved_device,
        autocast_dtype=autocast_dtype,
        compile_model=bool(compile_model),
        compile_mode=str(compile_mode),
        allow_tf32=bool(allow_tf32),
        warmup=False,
    )


@contextmanager
def torch_matmul_precision(*, allow_tf32: bool) -> Iterator[None]:
    """Temporarily enable TF32 matmul/cudnn precision and restore state."""
    if not allow_tf32:
        with nullcontext():
            yield
        return
    import torch

    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_precision = (
        torch.get_float32_matmul_precision()
        if hasattr(torch, "get_float32_matmul_precision")
        else None
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
        if old_precision is not None:
            torch.set_float32_matmul_precision(old_precision)
