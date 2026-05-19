"""Shared inference helpers for Vox-Profile WavLM wrappers."""
from __future__ import annotations

from speechbrain.integrations.huggingface import make_padding_masks


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
