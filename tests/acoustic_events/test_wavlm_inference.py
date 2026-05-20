import numpy as np
import torch

from audio_classification_playground.vox_profile.wavlm_inference import (
    prepare_wavlm_large_inputs,
    stream_wavlm_weighted_features,
)
from audio_classification_playground.acoustic_events.inference.audio import (
    frame_audio,
    writable_contiguous_float32,
)


class _FakeFeatureExtractor:
    def __init__(self):
        self.call_shapes = []

    def __call__(self, raw_speech, *, sampling_rate, return_tensors, padding):
        assert sampling_rate == 16_000
        assert return_tensors == "pt"
        assert padding is True
        values = np.asarray(raw_speech, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        self.call_shapes.append(values.shape)
        normalized = np.stack([
            (row - row.mean()) / np.sqrt(row.var() + 1e-7)
            for row in values
        ]).astype(np.float32)
        return {"input_values": torch.from_numpy(normalized)}


def test_prepare_wavlm_large_inputs_matches_legacy_per_window_loop():
    x = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [2.0, 2.5, 3.5, 5.0],
        [-1.0, 0.0, 1.0, 2.0],
    ], dtype=torch.float32)

    legacy_processor = _FakeFeatureExtractor()
    legacy = torch.stack([
        legacy_processor(
            x[idx],
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )["input_values"][0]
        for idx in range(len(x))
    ])

    batch_processor = _FakeFeatureExtractor()
    signal, attention_mask = prepare_wavlm_large_inputs(batch_processor, x)

    assert attention_mask is None
    assert legacy_processor.call_shapes == [(1, 4), (1, 4), (1, 4)]
    assert batch_processor.call_shapes == []
    assert torch.equal(signal, legacy)


def test_prepare_wavlm_large_inputs_preserves_length_masks_when_provided():
    x = torch.ones((2, 5), dtype=torch.float32)
    length = torch.tensor([5, 3], dtype=torch.float32)

    _, attention_mask = prepare_wavlm_large_inputs(
        _FakeFeatureExtractor(),
        x,
        length=length,
    )

    assert attention_mask is not None
    assert attention_mask.tolist() == [
        [True, True, True, True, True],
        [True, True, True, False, False],
    ]


def test_prepare_wavlm_large_inputs_zero_windows_are_finite_zeros():
    x = torch.zeros((2, 16), dtype=torch.float32)

    signal, attention_mask = prepare_wavlm_large_inputs(_FakeFeatureExtractor(), x)

    assert attention_mask is None
    assert torch.isfinite(signal).all()
    assert torch.equal(signal, torch.zeros_like(signal))


def test_writable_contiguous_float32_copies_readonly_framed_views():
    samples = np.arange(16_000, dtype=np.float32)
    windows = frame_audio(
        samples,
        sample_rate=16_000,
        window_sec=0.25,
        hop_sec=0.125,
    )

    batch = writable_contiguous_float32(windows[:2])

    assert not windows.flags.writeable
    assert batch.flags.c_contiguous
    assert batch.flags.writeable
    assert batch.dtype == np.float32
    torch.from_numpy(batch)


def test_stream_wavlm_weighted_features_matches_hidden_state_stack():
    from transformers import WavLMConfig, WavLMModel

    for stable_layer_norm in (False, True):
        torch.manual_seed(0)
        config = WavLMConfig(
            conv_dim=(4,),
            conv_kernel=(3,),
            conv_stride=(2,),
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_hidden_layers=2,
            num_conv_pos_embeddings=4,
            num_conv_pos_embedding_groups=1,
            mask_time_prob=0.0,
            mask_feature_prob=0.0,
            hidden_dropout=0.0,
            feat_proj_dropout=0.0,
            attention_dropout=0.0,
            layerdrop=0.0,
            do_stable_layer_norm=stable_layer_norm,
        )
        model = WavLMModel(config).eval()
        input_values = torch.randn(2, 20)
        weights = torch.randn(config.num_hidden_layers + 1)

        with torch.inference_mode():
            hidden_states = model(input_values, output_hidden_states=True).hidden_states
            stacked = torch.stack(hidden_states, dim=0)
            expected = (
                torch.nn.functional.softmax(weights, dim=-1).view(-1, 1, 1, 1)
                * stacked
            ).sum(dim=0)
            actual = stream_wavlm_weighted_features(
                model,
                input_values,
                weights,
                use_conv_output=True,
            )

        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

        no_conv_weights = torch.randn(config.num_hidden_layers)
        with torch.inference_mode():
            expected_no_conv = (
                torch.nn.functional.softmax(no_conv_weights, dim=-1).view(-1, 1, 1, 1)
                * stacked[1:]
            ).sum(dim=0)
            actual_no_conv = stream_wavlm_weighted_features(
                model,
                input_values,
                no_conv_weights,
                use_conv_output=False,
            )

        assert torch.allclose(actual_no_conv, expected_no_conv, atol=1e-6, rtol=1e-6)
