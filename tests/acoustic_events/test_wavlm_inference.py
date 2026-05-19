import numpy as np
import torch

from audio_classification_playground.vox_profile.wavlm_inference import (
    prepare_wavlm_large_inputs,
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
    assert batch_processor.call_shapes == [(3, 4)]
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
