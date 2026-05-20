import unittest
from types import SimpleNamespace

import numpy as np
import torch

from audio_classification_playground.acoustic_events.inference.emotion2vec import (
    _pad_tensor_batch,
    _validate_compile_mode_for_device,
    predict_emotion2vec_scores,
    predict_emotion2vec_scores_from_audio,
)
from audio_classification_playground.acoustic_events.inference.audio import (
    frame_audio,
    frame_audio_geometry,
)
from audio_classification_playground.acoustic_events.inference.emotion_runtime import (
    torch_matmul_precision,
)


class FakeAutoModel:
    def __init__(self, model, labels):
        self.model = model
        self.kwargs = {
            "device": "cpu",
            "tokenizer": SimpleNamespace(token_list=labels),
        }


class FakeEmotion2vec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(normalize=False)
        self.proj = torch.nn.Linear(2, 3, bias=False)
        self.calls = []
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, 0.0],
                    ]
                )
            )

    def extract_features(self, source, padding_mask=None):
        self.calls.append(tuple(source.shape))
        return {"x": source[:, :2].unsqueeze(1)}


class GenerateOnlyAutoModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        batch = kwargs["input"]
        self.calls.append(
            {
                "len": len(batch),
                "batch_size": kwargs["batch_size"],
                "fs": kwargs["fs"],
                "disable_pbar": kwargs["disable_pbar"],
            }
        )
        return [
            {"labels": ["happy", "sad"], "scores": [0.25, 0.75]}
            for _ in batch
        ]


class Emotion2vecFastPathTest(unittest.TestCase):
    def test_direct_path_batches_transformer_work_and_filters_unuse_labels(self):
        model = FakeEmotion2vec()
        auto_model = FakeAutoModel(model, ["happy", "unuse_blank", "sad"])
        windows = np.asarray(
            [
                [1.0, 0.0, 9.0, 9.0],
                [0.0, 2.0, 9.0, 9.0],
                [3.0, 4.0, 9.0, 9.0],
                [5.0, 6.0, 9.0, 9.0],
                [7.0, 8.0, 9.0, 9.0],
            ],
            dtype=np.float32,
        )

        scores, labels = predict_emotion2vec_scores(
            auto_model,
            windows,
            sample_rate=16_000,
            batch_size=2,
        )

        self.assertEqual(model.calls, [(2, 4), (2, 4), (1, 4)])
        self.assertEqual(labels, ["happy", "sad"])
        self.assertEqual(scores.shape, (5, 2))
        np.testing.assert_allclose(scores.sum(axis=1), np.ones(5), rtol=1e-6)

    def test_audio_feed_path_matches_framed_window_path(self):
        sample_rate = 16_000
        samples = np.arange(8, dtype=np.float32)
        windows = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
            dtype=np.float32,
        )

        framed_model = FakeEmotion2vec()
        framed_scores, framed_labels = predict_emotion2vec_scores(
            FakeAutoModel(framed_model, ["happy", "unuse_blank", "sad"]),
            windows,
            sample_rate=sample_rate,
            batch_size=2,
        )
        audio_model = FakeEmotion2vec()
        audio_scores, audio_labels = predict_emotion2vec_scores_from_audio(
            FakeAutoModel(audio_model, ["happy", "unuse_blank", "sad"]),
            samples,
            sample_rate=sample_rate,
            window_sec=4 / sample_rate,
            hop_sec=1 / sample_rate,
            batch_size=2,
        )

        self.assertEqual(audio_model.calls, framed_model.calls)
        self.assertEqual(audio_labels, framed_labels)
        np.testing.assert_allclose(audio_scores, framed_scores, rtol=0.0, atol=0.0)

    def test_audio_feed_framing_is_bit_equal_to_numpy_framing(self):
        import torch.nn.functional as F

        sample_rate = 16_000
        samples = np.arange(9, dtype=np.float32)
        expected = np.asarray(
            frame_audio(
                samples,
                sample_rate=sample_rate,
                window_sec=4 / sample_rate,
                hop_sec=2 / sample_rate,
            )
        )
        n_frames, window_samples, hop_samples, pad_needed = frame_audio_geometry(
            len(samples),
            sample_rate=sample_rate,
            window_sec=4 / sample_rate,
            hop_sec=2 / sample_rate,
        )

        audio = torch.from_numpy(samples.copy())
        if pad_needed:
            audio = F.pad(audio, (0, pad_needed))
        actual = audio.as_strided(
            size=(n_frames, window_samples),
            stride=(hop_samples, 1),
        ).contiguous()

        np.testing.assert_array_equal(actual.numpy(), expected)

    def test_generate_fallback_sets_funasr_batch_size(self):
        auto_model = GenerateOnlyAutoModel()
        windows = np.ones((3, 4), dtype=np.float32)

        scores, labels = predict_emotion2vec_scores(
            auto_model,
            windows,
            sample_rate=8_000,
            batch_size=2,
        )

        self.assertEqual(labels, ["happy", "sad"])
        self.assertEqual(scores.shape, (3, 2))
        self.assertEqual(
            auto_model.calls,
            [
                {"len": 2, "batch_size": 2, "fs": 8_000, "disable_pbar": True},
                {"len": 1, "batch_size": 1, "fs": 8_000, "disable_pbar": True},
            ],
        )

    def test_reduce_overhead_compile_mode_is_rejected_for_cuda(self):
        with self.assertRaisesRegex(ValueError, "reduce-overhead"):
            _validate_compile_mode_for_device("reduce-overhead", torch.device("cuda"))

        _validate_compile_mode_for_device("default", torch.device("cuda"))
        _validate_compile_mode_for_device("reduce-overhead", torch.device("cpu"))

    def test_pad_tensor_batch_preserves_real_rows(self):
        batch = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        padded = _pad_tensor_batch(batch, 4)

        self.assertEqual(tuple(padded.shape), (4, 3))
        torch.testing.assert_close(padded[:2], batch)
        torch.testing.assert_close(padded[2:], torch.zeros((2, 3)))

    def test_torch_matmul_precision_context_restores_global_state(self):
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        old_precision = torch.get_float32_matmul_precision()

        with torch_matmul_precision(allow_tf32=True):
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
            with torch_matmul_precision(allow_tf32=True):
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
                self.assertEqual(torch.get_float32_matmul_precision(), "high")
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), "high")

        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, old_matmul)
        self.assertEqual(torch.backends.cudnn.allow_tf32, old_cudnn)
        self.assertEqual(torch.get_float32_matmul_precision(), old_precision)


if __name__ == "__main__":
    unittest.main()
