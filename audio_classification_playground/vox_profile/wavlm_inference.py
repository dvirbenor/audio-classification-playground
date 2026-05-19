"""Shared inference helpers for Vox-Profile WavLM wrappers."""
from __future__ import annotations

from contextlib import nullcontext

from speechbrain.integrations.huggingface import make_padding_masks

_AUTOCAST_DTYPES = {"fp16", "bf16"}


def prepare_wavlm_large_inputs(
    processor,
    x,
    length=None,
    *,
    sample_rate: int = 16_000,
    device=None,
):
    """Prepare a whole WavLM-large batch with legacy-equivalent normalization.

    The previous wrappers called ``processor`` once per window and stacked the
    returned tensors.  ``Wav2Vec2FeatureExtractor`` supports batched raw-speech
    input, so this keeps the same per-row normalization while avoiding a Python
    loop and many device round-trips.
    """
    inputs = processor(
        x.detach().cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    target_device = x.device if device is None else device
    signal = inputs["input_values"].to(target_device)

    attention_mask = None
    if length is not None:
        attention_mask = make_padding_masks(
            x,
            wav_len=length / length.max(),
        ).to(target_device)

    return signal, attention_mask


def validate_autocast_dtype(autocast_dtype: str | None) -> str | None:
    """Validate optional CUDA autocast dtype names used by WavLM predictors."""
    if autocast_dtype is None:
        return None
    if autocast_dtype not in _AUTOCAST_DTYPES:
        expected = ", ".join(sorted(_AUTOCAST_DTYPES))
        raise ValueError(f"autocast_dtype must be one of {expected}, got {autocast_dtype!r}")
    return autocast_dtype


def autocast_context(torch, device, autocast_dtype: str | None):
    """Return a CUDA autocast context for optional WavLM mixed precision."""
    dtype_name = validate_autocast_dtype(autocast_dtype)
    if dtype_name is None:
        return nullcontext()

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("WavLM autocast is only supported on CUDA devices")
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def compile_wavlm_backbone(wrapper, *, mode: str, dynamic: bool):
    """Compile only the torch WavLM backbone, leaving CPU preprocessing eager."""
    import torch

    wrapper.backbone_model = torch.compile(
        wrapper.backbone_model,
        mode=mode,
        dynamic=dynamic,
    )
    return wrapper
