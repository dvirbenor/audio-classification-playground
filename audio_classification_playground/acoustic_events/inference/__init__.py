"""Reusable inference artifact layer for acoustic-event producers.

Typical CLI usage::

    uv run python -m audio_classification_playground.acoustic_events.inference run affect \
      --audio input.mp3 --backbone wavlm --out artifacts/

    uv run python -m audio_classification_playground.acoustic_events.inference run-all \
      --audio input.mp3 --affect-backbone wavlm --disfluency-backbone whisper \
      --out artifacts/ --reuse-cache

Typical Python usage::

    from audio_classification_playground.acoustic_events.inference import run_all_inference

    result = run_all_inference(
        "input.mp3",
        out_dir="artifacts",
        affect_backbone="wavlm",
        disfluency_backbone="whisper",
        reuse_cache=True,
    )

    affect_artifact = result.artifacts["affect"]

Artifacts are manifest-plus-NPZ directories that hold producer-ready model
evidence. This package does not create review sessions.
"""
from .adapters import (
    artifact_to_affect_signals,
    artifact_to_disfluency_logits,
    artifact_to_emotion_probabilities,
    artifact_to_vad,
)
from .artifacts import (
    InferenceRunResult,
    PredictionArtifact,
    decoded_audio_sha256,
    inference_config_hash,
    list_cached_artifacts,
    load_prediction_artifact,
)
from .runners import (
    run_affect_inference,
    run_all_inference,
    run_disfluency_inference,
    run_emotion_inference,
    run_vad,
)

__all__ = [
    "InferenceRunResult",
    "PredictionArtifact",
    "artifact_to_affect_signals",
    "artifact_to_disfluency_logits",
    "artifact_to_emotion_probabilities",
    "artifact_to_vad",
    "decoded_audio_sha256",
    "inference_config_hash",
    "list_cached_artifacts",
    "load_prediction_artifact",
    "run_affect_inference",
    "run_all_inference",
    "run_disfluency_inference",
    "run_emotion_inference",
    "run_vad",
]
