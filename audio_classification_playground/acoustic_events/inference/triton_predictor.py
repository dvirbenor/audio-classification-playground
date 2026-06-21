"""Triton Inference Server client predictors.

Drop-in replacements for the in-process model predictors. Each class satisfies
the ``predictor`` callable interface expected by the inference runners:

    affect / disfluency:  predictor(windows: np.ndarray) -> dict[str, np.ndarray]
    emotion:              predictor(windows: np.ndarray) -> tuple[np.ndarray, Sequence[str]]

Workers pass these as ``predictors={"affect": ..., "disfluency": ..., "emotion": ...}``
to ``run_all_inference()``, which injects them in place of direct model calls.
The GPU stays inside the Triton server; worker pods are CPU-only.

Multiple workers making concurrent requests to the same Triton server trigger
Triton's dynamic batcher, which groups simultaneous windows from different
archives into full GPU batches — the cross-archive batching benefit that MPS
cannot provide.

Usage (set via --triton-url flag on the orchestration worker):
    triton_url = "triton-inference.nlp-audio-understanding:8001"  # gRPC
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping

import numpy as np


_MAX_CHUNK = 256   # must not exceed max_batch_size in config.pbtxt


def _get_client(url: str):
    try:
        import tritonclient.grpc as grpcclient
    except ImportError as e:
        raise ImportError(
            "tritonclient[grpc] is required for Triton predictors. "
            "Install with: pip install tritonclient[grpc]"
        ) from e
    return grpcclient


class TritonAffectPredictor:
    """Calls the 'affect' ONNX model on the Triton server.

    Input:  windows [n, 56000] float32
    Output: {"arousal": [n], "valence": [n], "dominance": [n]}
    """

    def __init__(self, url: str) -> None:
        grpcclient = _get_client(url)
        self._client = grpcclient.InferenceServerClient(url=url)
        self._url = url

    def __call__(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        import tritonclient.grpc as grpcclient

        arousal, valence, dominance = [], [], []
        for chunk in _chunks(windows, _MAX_CHUNK):
            chunk = np.ascontiguousarray(chunk, dtype=np.float32)
            inp = grpcclient.InferInput("input_values", list(chunk.shape), "FP32")
            inp.set_data_from_numpy(chunk)
            result = self._client.infer(
                "affect",
                inputs=[inp],
                outputs=[
                    grpcclient.InferRequestedOutput("arousal"),
                    grpcclient.InferRequestedOutput("valence"),
                    grpcclient.InferRequestedOutput("dominance"),
                ],
            )
            arousal.append(result.as_numpy("arousal").reshape(-1))
            valence.append(result.as_numpy("valence").reshape(-1))
            dominance.append(result.as_numpy("dominance").reshape(-1))

        return {
            "arousal": np.concatenate(arousal),
            "valence": np.concatenate(valence),
            "dominance": np.concatenate(dominance),
        }


class TritonDisfluencyPredictor:
    """Calls the 'disfluency' ONNX model on the Triton server.

    Input:  windows [n, 48000] float32
    Output: {"fluency_logits": [n, 2], "disfluency_type_logits": [n, 5]}
    """

    def __init__(self, url: str) -> None:
        grpcclient = _get_client(url)
        self._client = grpcclient.InferenceServerClient(url=url)

    def __call__(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        import tritonclient.grpc as grpcclient

        fluency, disf_type = [], []
        for chunk in _chunks(windows, _MAX_CHUNK):
            chunk = np.ascontiguousarray(chunk, dtype=np.float32)
            inp = grpcclient.InferInput("input_values", list(chunk.shape), "FP32")
            inp.set_data_from_numpy(chunk)
            result = self._client.infer(
                "disfluency",
                inputs=[inp],
                outputs=[
                    grpcclient.InferRequestedOutput("fluency_logits"),
                    grpcclient.InferRequestedOutput("disfluency_type_logits"),
                ],
            )
            fluency.append(result.as_numpy("fluency_logits"))
            disf_type.append(result.as_numpy("disfluency_type_logits"))

        return {
            "fluency_logits": np.concatenate(fluency, axis=0),
            "disfluency_type_logits": np.concatenate(disf_type, axis=0),
        }


class TritonEmotionPredictor:
    """Calls the 'emotion' Python-backend model on the Triton server.

    The Python backend returns fully-processed probabilities in CANONICAL_CHANNELS
    order (9 classes). We return them as (probs, CANONICAL_CHANNELS) so the
    runner's emotion2vec_scores_to_probabilities() receives already-canonical,
    already-normalized data and its fold+normalise step is a no-op.

    Input:  windows [n, 48000] float32
    Output: (probabilities [n, 9], canonical_labels tuple[str, ...])
    """

    def __init__(self, url: str) -> None:
        grpcclient = _get_client(url)
        self._client = grpcclient.InferenceServerClient(url=url)

    def __call__(self, windows: np.ndarray) -> tuple[np.ndarray, Sequence[str]]:
        from ..producers.emotion.config import CANONICAL_CHANNELS
        import tritonclient.grpc as grpcclient

        probs_chunks = []
        for chunk in _chunks(windows, _MAX_CHUNK):
            chunk = np.ascontiguousarray(chunk, dtype=np.float32)
            inp = grpcclient.InferInput("input_values", list(chunk.shape), "FP32")
            inp.set_data_from_numpy(chunk)
            result = self._client.infer(
                "emotion",
                inputs=[inp],
                outputs=[grpcclient.InferRequestedOutput("probabilities")],
            )
            probs_chunks.append(result.as_numpy("probabilities"))

        return np.concatenate(probs_chunks, axis=0), CANONICAL_CHANNELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks(arr: np.ndarray, size: int):
    for start in range(0, max(len(arr), 1), size):
        yield arr[start: start + size]
