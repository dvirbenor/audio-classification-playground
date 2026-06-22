"""Triton Python backend — emotion2vec_plus_large.

Self-contained: only depends on funasr + modelscope (see requirements.txt).
No dependency on the audio_classification_playground package so the official
nvcr.io/nvidia/tritonserver image can be used without any custom build.

Receives batches of pre-segmented audio windows (float32, [batch, 48000])
assembled by Triton's dynamic batcher from concurrent worker requests.
Returns per-window probabilities ([batch, 9]) in CANONICAL_CHANNELS order:
  ["angry","disgusted","fearful","happy","neutral","other","sad","surprised","unknown"]

Run `scripts/export_models_onnx.py --task emotion` first — if that succeeds,
use the ONNX backend instead (faster, no Python GIL, no startup pip install).
This file is the fallback for when ONNX export is not possible.
"""
import numpy as np
import triton_python_backend_utils as pb_utils

SAMPLE_RATE = 16_000
EMOTION_WINDOW_SAMPLES = int(3.0 * SAMPLE_RATE)   # 48 000

# Canonical label order produced by emotion2vec_plus_large.
# Must match CANONICAL_CHANNELS in producers/emotion/config.py.
CANONICAL_CHANNELS = (
    "angry", "disgusted", "fearful", "happy", "neutral",
    "other", "sad", "surprised", "unknown",
)


class TritonPythonModel:
    def initialize(self, args):
        import torch
        from funasr import AutoModel

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._autocast_dtype = "fp16" if self._device == "cuda" else None

        self._auto_model = AutoModel(
            model="iic/emotion2vec_plus_large",
            batch_size=256,
            device=self._device,
            disable_update=True,
            disable_pbar=True,
        )

        # Warm up so the first real request is not slow.
        dummy = np.zeros((1, EMOTION_WINDOW_SAMPLES), dtype=np.float32)
        self._score_batch(dummy)

    def execute(self, requests):
        # Triton's dynamic batcher groups concurrent worker requests.
        # Concatenate inputs, run one forward pass, split outputs back.
        per_request_sizes = []
        all_windows = []
        for req in requests:
            windows = pb_utils.get_input_tensor_by_name(req, "input_values").as_numpy()
            per_request_sizes.append(len(windows))
            all_windows.append(windows)

        batch = np.concatenate(all_windows, axis=0)
        probs = self._score_batch(batch)   # [total, 9]

        responses = []
        idx = 0
        for n in per_request_sizes:
            chunk = probs[idx: idx + n]
            idx += n
            out = pb_utils.Tensor("probabilities", chunk.astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    def _score_batch(self, windows: np.ndarray) -> np.ndarray:
        """Run FunASR inference and return normalised probabilities in canonical order."""
        from funasr.utils.misc import deep_update

        # predict_emotion2vec_scores equivalent — call FunASR generate directly.
        result = self._auto_model.generate(
            input=windows,
            batch_size=len(windows),
            is_final=True,
            disable_pbar=True,
        )
        # result is a list of dicts with keys: "scores" (list), "labels" (list)
        raw_labels = result[0]["labels"]
        raw_scores = np.array([r["scores"] for r in result], dtype=np.float64)

        # Fold into canonical channel order and row-normalise.
        label_to_idx = {lbl: i for i, lbl in enumerate(CANONICAL_CHANNELS)}
        n_canonical = len(CANONICAL_CHANNELS)
        folded = np.zeros((len(raw_scores), n_canonical), dtype=np.float64)
        for src_i, lbl in enumerate(raw_labels):
            canonical = _normalise_label(lbl)
            dst_i = label_to_idx.get(canonical)
            if dst_i is None:
                raise RuntimeError(f"Unknown emotion2vec label {lbl!r} → {canonical!r}")
            folded[:, dst_i] += raw_scores[:, src_i]

        row_sums = folded.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)   # guard against all-zero rows
        return (folded / row_sums).astype(np.float32)

    def finalize(self):
        pass


def _normalise_label(label: str) -> str:
    """Mirror normalize_label() from producers/emotion/pipeline.py."""
    return label.strip().lower().replace(" ", "_").replace("-", "_")
