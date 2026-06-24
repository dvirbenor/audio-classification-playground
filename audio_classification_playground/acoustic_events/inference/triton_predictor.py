"""Triton Inference Server client predictors.

Drop-in replacements for the in-process model predictors. Each class satisfies
the ``predictor`` callable interface expected by the inference runners:

    affect / disfluency:  predictor(windows: np.ndarray) -> dict[str, np.ndarray]
    emotion:              predictor(windows: np.ndarray) -> tuple[np.ndarray, Sequence[str]]

Workers pass these as ``predictors={"affect": ..., "disfluency": ..., "emotion": ...}``
to ``run_all_inference()``, which injects them in place of direct model calls.
The GPU stays inside the Triton server; worker pods are CPU-only.

PIPELINING: a worker's windows are sent in chunks of ``_MAX_CHUNK``. A naive
loop sends one chunk and *blocks* on the round-trip before sending the next, so
the GPU sits idle (for this worker) during the network hop and the worker's CPU
idles waiting. We instead keep up to ``TRITON_PIPELINE_DEPTH`` chunk requests in
flight at once (a thread pool of synchronous ``infer`` calls — the gRPC client
is thread-safe), which hides the round-trip latency and lets a single worker
drive far more windows/s. Combined with the runner overlapping the three model
tasks, one CPU worker fills much more of the GPU than before.

Usage (set via --triton-url flag on the orchestration worker):
    triton_url = "triton-inference.nlp-audio-understanding:8001"  # gRPC
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np


_MAX_CHUNK = 128   # must not exceed max_batch_size in config.pbtxt (ONNX and TRT both 128)
# How many chunk requests to keep in flight per task, per worker. 4 hides the
# round-trip latency without flooding the server; tune via env if needed.
_PIPELINE_DEPTH = int(os.environ.get("TRITON_PIPELINE_DEPTH", "4"))


def _get_client(url: str):
    try:
        import tritonclient.grpc as grpcclient
    except ImportError as e:
        raise ImportError(
            "tritonclient[grpc] is required for Triton predictors. "
            "Install with: pip install tritonclient[grpc]"
        ) from e
    return grpcclient


def _run_pipeline(infer_chunk, chunks: list[np.ndarray]) -> list:
    """Run ``infer_chunk`` over chunks with up to _PIPELINE_DEPTH in flight,
    returning results in chunk order (ThreadPoolExecutor.map preserves order)."""
    if not chunks:
        return []
    depth = max(1, min(_PIPELINE_DEPTH, len(chunks)))
    if depth == 1:
        return [infer_chunk(c) for c in chunks]
    with ThreadPoolExecutor(max_workers=depth) as ex:
        return list(ex.map(infer_chunk, chunks))


def _prep_chunks(windows: np.ndarray) -> list[np.ndarray]:
    return [
        np.ascontiguousarray(c, dtype=np.float32)
        for c in _chunks(windows, _MAX_CHUNK)
        if len(c)
    ]


class TritonAffectPredictor:
    """Calls the 'affect' ONNX/TRT model on the Triton server.

    Input:  windows [n, 56000] float32
    Output: {"arousal": [n], "valence": [n], "dominance": [n]}
    """

    def __init__(self, url: str) -> None:
        grpcclient = _get_client(url)
        self._client = grpcclient.InferenceServerClient(url=url)

    def _infer_chunk(self, chunk: np.ndarray):
        import tritonclient.grpc as grpcclient

        inp = grpcclient.InferInput("input_values", list(chunk.shape), "FP32")
        inp.set_data_from_numpy(chunk)
        r = self._client.infer(
            "affect",
            inputs=[inp],
            outputs=[
                grpcclient.InferRequestedOutput("arousal"),
                grpcclient.InferRequestedOutput("valence"),
                grpcclient.InferRequestedOutput("dominance"),
            ],
        )
        return (r.as_numpy("arousal").reshape(-1),
                r.as_numpy("valence").reshape(-1),
                r.as_numpy("dominance").reshape(-1))

    def __call__(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        outs = _run_pipeline(self._infer_chunk, _prep_chunks(windows))
        if not outs:
            empty = np.empty((0,), dtype=np.float32)
            return {"arousal": empty, "valence": empty.copy(), "dominance": empty.copy()}
        return {
            "arousal": np.concatenate([o[0] for o in outs]),
            "valence": np.concatenate([o[1] for o in outs]),
            "dominance": np.concatenate([o[2] for o in outs]),
        }


class TritonDisfluencyPredictor:
    """Calls the 'disfluency' ONNX/TRT model on the Triton server.

    Input:  windows [n, 48000] float32
    Output: {"fluency_logits": [n, 2], "disfluency_type_logits": [n, 5]}
    """

    def __init__(self, url: str) -> None:
        grpcclient = _get_client(url)
        self._client = grpcclient.InferenceServerClient(url=url)

    def _infer_chunk(self, chunk: np.ndarray):
        import tritonclient.grpc as grpcclient

        inp = grpcclient.InferInput("input_values", list(chunk.shape), "FP32")
        inp.set_data_from_numpy(chunk)
        r = self._client.infer(
            "disfluency",
            inputs=[inp],
            outputs=[
                grpcclient.InferRequestedOutput("fluency_logits"),
                grpcclient.InferRequestedOutput("disfluency_type_logits"),
            ],
        )
        return (r.as_numpy("fluency_logits"), r.as_numpy("disfluency_type_logits"))

    def __call__(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        outs = _run_pipeline(self._infer_chunk, _prep_chunks(windows))
        if not outs:
            return {
                "fluency_logits": np.empty((0, 2), dtype=np.float32),
                "disfluency_type_logits": np.empty((0, 5), dtype=np.float32),
            }
        return {
            "fluency_logits": np.concatenate([o[0] for o in outs], axis=0),
            "disfluency_type_logits": np.concatenate([o[1] for o in outs], axis=0),
        }


class TritonEmotionPredictor:
    """Calls the 'emotion' ONNX/TRT model on the Triton server.

    The model emits raw per-class softmax scores over the 9 native emotion2vec
    labels (EMOTION2VEC_LABELS order). We return them as
    (scores, EMOTION2VEC_LABELS) so the runner's
    emotion2vec_scores_to_probabilities() folds them into CANONICAL_CHANNELS —
    exactly the same path as the direct (non-Triton) predictor.

    Input:  windows [n, 48000] float32
    Output: (scores [n, 9], EMOTION2VEC_LABELS tuple[str, ...])
    """

    def __init__(self, url: str) -> None:
        grpcclient = _get_client(url)
        self._client = grpcclient.InferenceServerClient(url=url)

    def _infer_chunk(self, chunk: np.ndarray):
        import tritonclient.grpc as grpcclient

        inp = grpcclient.InferInput("input_values", list(chunk.shape), "FP32")
        inp.set_data_from_numpy(chunk)
        r = self._client.infer(
            "emotion",
            inputs=[inp],
            outputs=[grpcclient.InferRequestedOutput("scores")],
        )
        return r.as_numpy("scores")

    def __call__(self, windows: np.ndarray) -> tuple[np.ndarray, Sequence[str]]:
        from .emotion2vec import EMOTION2VEC_LABELS

        outs = _run_pipeline(self._infer_chunk, _prep_chunks(windows))
        if not outs:
            return np.empty((0, len(EMOTION2VEC_LABELS)), dtype=np.float32), EMOTION2VEC_LABELS
        return np.concatenate(outs, axis=0), EMOTION2VEC_LABELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks(arr: np.ndarray, size: int):
    for start in range(0, max(len(arr), 1), size):
        yield arr[start: start + size]
