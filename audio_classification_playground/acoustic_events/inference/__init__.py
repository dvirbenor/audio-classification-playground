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
    inference_configs_match,
    list_cached_artifacts,
    load_prediction_artifact,
    semantic_inference_config,
)
from .audio import AudioData
from .emotion_runtime import (
    EMOTION_RUNTIME_MODE_CHOICES,
    OPTIMIZED_EMOTION_BATCH_SIZE,
    EmotionRuntimeSettings,
    resolve_emotion_runtime_settings,
)
from .models import (
    AffectPredictor,
    DisfluencyPredictor,
    EmotionPredictor,
    ModelSuite,
    VadDetector,
)
from .runners import (
    DEFAULT_VAD_FRAME_SPEECH_RATIO_THRESHOLD,
    DEFAULT_VAD_MIN_SILENCE_SEC,
    DEFAULT_VAD_MIN_SPEECH_SEC,
    DEFAULT_VAD_SPEECH_THRESHOLD,
    ShutdownRequested,
    compute_inference_config,
    resolve_task_batch_sizes,
    run_affect_inference,
    run_all_inference,
    run_disfluency_inference,
    run_emotion_inference,
    run_vad,
)

__all__ = [
    "AffectPredictor",
    "AudioData",
    "DisfluencyPredictor",
    "EmotionPredictor",
    "InferenceRunResult",
    "ModelSuite",
    "PredictionArtifact",
    "ShutdownRequested",
    "VadDetector",
    "DEFAULT_VAD_FRAME_SPEECH_RATIO_THRESHOLD",
    "DEFAULT_VAD_MIN_SILENCE_SEC",
    "DEFAULT_VAD_MIN_SPEECH_SEC",
    "DEFAULT_VAD_SPEECH_THRESHOLD",
    "EMOTION_RUNTIME_MODE_CHOICES",
    "EmotionRuntimeSettings",
    "OPTIMIZED_EMOTION_BATCH_SIZE",
    "artifact_to_affect_signals",
    "artifact_to_disfluency_logits",
    "artifact_to_emotion_probabilities",
    "artifact_to_vad",
    "compute_inference_config",
    "decoded_audio_sha256",
    "inference_config_hash",
    "inference_configs_match",
    "list_cached_artifacts",
    "load_prediction_artifact",
    "resolve_emotion_runtime_settings",
    "resolve_task_batch_sizes",
    "semantic_inference_config",
    "run_affect_inference",
    "run_all_inference",
    "run_disfluency_inference",
    "run_emotion_inference",
    "run_vad",
]
