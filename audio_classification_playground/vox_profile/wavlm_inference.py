"""Shared inference helpers for Vox-Profile WavLM wrappers."""
from __future__ import annotations

from contextlib import nullcontext

from speechbrain.integrations.huggingface import make_padding_masks
import torch
import transformers.models.wavlm.modeling_wavlm as wavlm

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

    For fixed 16 kHz waveform windows, ``Wav2Vec2FeatureExtractor`` only
    performs per-row zero-mean/unit-variance normalization before tensor
    conversion.  The vectorized Torch implementation below matches that
    fixed-window path, avoids the feature extractor's Python padding/container
    overhead, and stays on-device so it traces cleanly under
    ``torch.export`` / ``torch.onnx.export`` (a NumPy round-trip cannot).
    """
    target_device = x.device if device is None else device
    signal = _zero_mean_unit_var_norm(x).to(device=target_device, dtype=x.dtype)

    attention_mask = None
    if length is not None:
        attention_mask = make_padding_masks(
            x,
            wav_len=length / length.max(),
        ).to(target_device)

    return signal, attention_mask


def stream_wavlm_weighted_features(
    backbone_model,
    input_values,
    layer_weights,
    *,
    attention_mask=None,
    use_conv_output: bool,
):
    """Run WavLM while accumulating the learned layer mixture.

    The Vox-Profile heads only use a learned weighted sum over WavLM hidden
    states.  Asking Hugging Face for ``output_hidden_states=True`` materializes
    every layer, then stacks them into another large tensor.  This helper keeps
    the same layer order and model operations, but accumulates the weighted sum
    as the encoder advances so old layer outputs can be released earlier.
    """
    norm_weights = torch.nn.functional.softmax(layer_weights, dim=-1)
    expected_weights = (
        backbone_model.config.num_hidden_layers + 1
        if use_conv_output
        else backbone_model.config.num_hidden_layers
    )
    if norm_weights.numel() != expected_weights:
        raise ValueError(
            f"Expected {expected_weights} WavLM layer weights, got {norm_weights.numel()}"
        )

    extract_features = backbone_model.feature_extractor(input_values)
    extract_features = extract_features.transpose(1, 2)

    if attention_mask is not None:
        attention_mask = backbone_model._get_feature_vector_attention_mask(
            extract_features.shape[1],
            attention_mask,
            add_adapter=False,
        )

    hidden_states, _ = backbone_model.feature_projection(extract_features)
    hidden_states = backbone_model._mask_hidden_states(
        hidden_states,
        mask_time_indices=None,
        attention_mask=attention_mask,
    )
    return _stream_encoder_weighted_sum(
        backbone_model.encoder,
        hidden_states,
        attention_mask=attention_mask,
        norm_weights=norm_weights,
        use_conv_output=use_conv_output,
    )


def _stream_encoder_weighted_sum(
    encoder,
    hidden_states,
    *,
    attention_mask,
    norm_weights,
    use_conv_output: bool,
):
    is_stable = isinstance(encoder, wavlm.WavLMEncoderStableLayerNorm)
    weighted = None
    hidden_state_index = 0

    def add_hidden_state(value):
        nonlocal hidden_state_index, weighted
        if use_conv_output:
            weight_index = hidden_state_index
        elif hidden_state_index == 0:
            hidden_state_index += 1
            return
        else:
            weight_index = hidden_state_index - 1

        term = norm_weights[weight_index] * value
        weighted = term if weighted is None else weighted + term
        hidden_state_index += 1

    if attention_mask is not None:
        expand_attention_mask = attention_mask.unsqueeze(-1).repeat(
            1, 1, hidden_states.shape[2]
        )
        hidden_states[~expand_attention_mask] = 0

    position_embeddings = encoder.pos_conv_embed(hidden_states)
    hidden_states = hidden_states + position_embeddings
    if not is_stable:
        hidden_states = encoder.layer_norm(hidden_states)
    hidden_states = encoder.dropout(hidden_states)

    synced_gpus = wavlm.is_deepspeed_zero3_enabled() or wavlm.is_fsdp_managed_module(encoder)
    position_bias = None

    for i, layer in enumerate(encoder.layers):
        add_hidden_state(hidden_states)

        dropout_probability = torch.rand([])
        skip_the_layer = encoder.training and i > 0 and (
            dropout_probability < encoder.config.layerdrop
        )
        if not skip_the_layer or synced_gpus:
            layer_kwargs = {
                "attention_mask": attention_mask,
                "position_bias": position_bias,
                "output_attentions": False,
            }
            if not is_stable:
                layer_kwargs["index"] = i
            layer_outputs = layer(hidden_states, **layer_kwargs)
            hidden_states, position_bias = layer_outputs[:2]

    if is_stable:
        hidden_states = encoder.layer_norm(hidden_states)
    add_hidden_state(hidden_states)

    if weighted is None:
        raise RuntimeError("WavLM weighted feature accumulation produced no output")
    return weighted


def _zero_mean_unit_var_norm(values: torch.Tensor) -> torch.Tensor:
    """Match Wav2Vec2FeatureExtractor zero-mean/unit-var normalization.

    Pure Torch so the op traces under ``torch.export``: ``.numpy()`` is not
    supported on the fake/proxy tensor subclasses the exporter feeds through.
    Uses population variance (``unbiased=False``, i.e. NumPy's default ddof=0)
    to stay numerically equivalent to the feature extractor, up to float32
    reduction-order noise.
    """
    array = values.to(torch.float32)
    mean = array.mean(dim=1, keepdim=True)
    var = array.var(dim=1, keepdim=True, unbiased=False)
    return (array - mean) / torch.sqrt(var + 1e-7)


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

    if getattr(wrapper, "wavlm_stream_layer_sum", False):
        raise ValueError(
            "wavlm_stream_layer_sum and wavlm_compile are separate experiments; "
            "compile_wavlm_backbone cannot compile the streamed layer-sum path."
        )
    wrapper.backbone_model = torch.compile(
        wrapper.backbone_model,
        mode=mode,
        dynamic=dynamic,
    )
    return wrapper
