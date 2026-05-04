"""Reusable inference artifact layer for acoustic-event producers."""
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
