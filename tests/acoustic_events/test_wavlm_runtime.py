from unittest.mock import patch

import numpy as np
import pytest
import torch

from audio_classification_playground.acoustic_events.inference.models import (
    AffectPredictor,
    DisfluencyPredictor,
)
from audio_classification_playground.acoustic_events.inference.wavlm_runtime import (
    WAVLM_COMPILED_STATIC_BATCH_SIZE,
    WavLMRuntimeSettings,
    pad_windows_to_static_batch,
    resolve_wavlm_runtime_settings,
)


def test_resolve_wavlm_compiled_static_preset_defaults():
    with patch(
        "audio_classification_playground.acoustic_events.inference.wavlm_runtime."
        "wavlm_compiled_static_is_eligible",
        return_value=True,
    ):
        settings = resolve_wavlm_runtime_settings(
            preset=None,
            device="cuda",
            autocast_dtype=None,
            compile_model=False,
            compile_mode="reduce-overhead",
            compile_dynamic=False,
            stream_layer_sum=False,
            allow_tf32=False,
        )

    assert settings.preset == "compiled_static"
    assert settings.task_batch_size == WAVLM_COMPILED_STATIC_BATCH_SIZE
    assert settings.compile_model is True
    assert settings.compile_mode == "default"
    assert settings.compile_dynamic is False
    assert settings.static_batch is True
    assert settings.warmup is True


def test_wavlm_runtime_preset_rejects_granular_knobs():
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_wavlm_runtime_settings(
            preset="compiled_static",
            device="cuda",
            autocast_dtype=None,
            compile_model=True,
            compile_mode="reduce-overhead",
            compile_dynamic=False,
            stream_layer_sum=False,
            allow_tf32=False,
        )


def test_wavlm_static_batch_size_overrides_compiled_static_batch():
    with patch(
        "audio_classification_playground.acoustic_events.inference.wavlm_runtime."
        "wavlm_compiled_static_is_eligible",
        return_value=True,
    ):
        settings = resolve_wavlm_runtime_settings(
            preset="compiled_static",
            device="cuda",
            autocast_dtype=None,
            compile_model=False,
            compile_mode="reduce-overhead",
            compile_dynamic=False,
            stream_layer_sum=False,
            allow_tf32=False,
            static_batch_size=512,
        )

    assert settings.preset == "compiled_static"
    assert settings.task_batch_size == 512
    assert settings.static_batch is True


def test_wavlm_static_batch_size_rejects_fast_exact_preset():
    with pytest.raises(ValueError, match="requires the compiled_static preset"):
        resolve_wavlm_runtime_settings(
            preset="fast_exact",
            device="cuda",
            autocast_dtype=None,
            compile_model=False,
            compile_mode="reduce-overhead",
            compile_dynamic=False,
            stream_layer_sum=False,
            allow_tf32=False,
            static_batch_size=512,
        )


def test_wavlm_static_batch_size_rejects_granular_knobs():
    with pytest.raises(ValueError, match="cannot be combined with granular"):
        resolve_wavlm_runtime_settings(
            preset=None,
            device="cuda",
            autocast_dtype="fp16",
            compile_model=False,
            compile_mode="reduce-overhead",
            compile_dynamic=False,
            stream_layer_sum=False,
            allow_tf32=False,
            static_batch_size=512,
        )


def test_wavlm_static_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="positive integer"):
        resolve_wavlm_runtime_settings(
            preset="compiled_static",
            device="cuda",
            autocast_dtype=None,
            compile_model=False,
            compile_mode="reduce-overhead",
            compile_dynamic=False,
            stream_layer_sum=False,
            allow_tf32=False,
            static_batch_size=0,
        )


def test_pad_windows_to_static_batch_uses_raw_zero_rows():
    windows = np.ones((5, 3), dtype=np.float32)

    padded, valid_count = pad_windows_to_static_batch(
        windows,
        batch_size=4,
        enabled=True,
    )

    assert valid_count == 5
    assert padded.shape == (8, 3)
    assert np.array_equal(padded[:5], windows)
    assert np.array_equal(padded[5:], np.zeros((3, 3), dtype=np.float32))


def test_affect_predictor_static_padding_slices_outputs_and_skips_empty_forward():
    predictor = object.__new__(AffectPredictor)
    predictor.backbone = "wavlm"
    predictor.batch_size = 4
    predictor.wavlm_autocast_dtype = None
    predictor.wavlm_static_batch = True
    predictor._device = "cpu"
    predictor._model = _FakeAffectModel()

    empty = predictor(np.zeros((0, 3), dtype=np.float32))
    assert {name: values.shape for name, values in empty.items()} == {
        "arousal": (0,),
        "valence": (0,),
        "dominance": (0,),
    }
    assert predictor._model.seen_batches == []

    windows = np.arange(15, dtype=np.float32).reshape(5, 3) + 1.0
    out = predictor(windows)

    assert {name: values.shape for name, values in out.items()} == {
        "arousal": (5,),
        "valence": (5,),
        "dominance": (5,),
    }
    assert predictor._model.seen_batches == [4, 4]
    assert np.array_equal(predictor._model.seen_inputs[-1][1:], np.zeros((3, 3)))


def test_disfluency_predictor_static_padding_slices_outputs():
    predictor = object.__new__(DisfluencyPredictor)
    predictor.backbone = "wavlm"
    predictor.batch_size = 4
    predictor.wavlm_autocast_dtype = None
    predictor.wavlm_static_batch = True
    predictor._device = "cpu"
    predictor._model = _FakeDisfluencyModel()

    for n_windows in (1, 4, 5, 7, 8):
        windows = np.ones((n_windows, 3), dtype=np.float32)
        out = predictor(windows)
        assert out["fluency_logits"].shape == (n_windows, 2)
        assert out["disfluency_type_logits"].shape == (n_windows, 5)


class _FakeAffectModel:
    def __init__(self):
        self.seen_batches = []
        self.seen_inputs = []

    def __call__(self, batch):
        n = int(batch.shape[0])
        self.seen_batches.append(n)
        self.seen_inputs.append(batch.detach().cpu().numpy())
        base = torch.arange(n, dtype=torch.float32)
        return base, base + 10.0, base + 20.0


class _FakeDisfluencyModel:
    def __call__(self, batch, *, return_feature):
        assert return_feature is False
        n = int(batch.shape[0])
        return torch.zeros((n, 2)), torch.zeros((n, 5))


def _compiled_static_settings() -> WavLMRuntimeSettings:
    return WavLMRuntimeSettings(
        requested_preset=None,
        preset="compiled_static",
        device="cuda",
        task_batch_size=WAVLM_COMPILED_STATIC_BATCH_SIZE,
        autocast_dtype=None,
        compile_model=True,
        compile_mode="default",
        compile_dynamic=False,
        stream_layer_sum=False,
        allow_tf32=False,
        static_batch=True,
        warmup=True,
    )
