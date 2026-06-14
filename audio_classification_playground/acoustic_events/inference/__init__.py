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
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

# Map each public symbol to the submodule that defines it. Imports are
# deferred to first access (PEP 562 ``__getattr__``) so that lightweight
# consumers -- e.g. the read-only orchestration CLI commands
# (progress/status/timings/errors), which only need ``artifacts`` -- don't
# drag in ``models``/``runners`` and the whole torch/transformers stack.
_SYMBOL_MODULES = {
    "artifact_to_affect_signals": ".adapters",
    "artifact_to_disfluency_logits": ".adapters",
    "artifact_to_emotion_probabilities": ".adapters",
    "artifact_to_vad": ".adapters",
    "InferenceRunResult": ".artifacts",
    "PredictionArtifact": ".artifacts",
    "decoded_audio_sha256": ".artifacts",
    "inference_config_hash": ".artifacts",
    "inference_configs_match": ".artifacts",
    "list_cached_artifacts": ".artifacts",
    "load_prediction_artifact": ".artifacts",
    "semantic_inference_config": ".artifacts",
    "AudioData": ".audio",
    "EMOTION_RUNTIME_MODE_CHOICES": ".emotion_runtime",
    "OPTIMIZED_EMOTION_BATCH_SIZE": ".emotion_runtime",
    "EmotionRuntimeSettings": ".emotion_runtime",
    "resolve_emotion_runtime_settings": ".emotion_runtime",
    "AffectPredictor": ".models",
    "DisfluencyPredictor": ".models",
    "EmotionPredictor": ".models",
    "ModelSuite": ".models",
    "VadDetector": ".models",
    "DEFAULT_VAD_FRAME_SPEECH_RATIO_THRESHOLD": ".runners",
    "DEFAULT_VAD_MIN_SILENCE_SEC": ".runners",
    "DEFAULT_VAD_MIN_SPEECH_SEC": ".runners",
    "DEFAULT_VAD_SPEECH_THRESHOLD": ".runners",
    "ShutdownRequested": ".runners",
    "compute_inference_config": ".runners",
    "resolve_task_batch_sizes": ".runners",
    "run_affect_inference": ".runners",
    "run_all_inference": ".runners",
    "run_disfluency_inference": ".runners",
    "run_emotion_inference": ".runners",
    "run_vad": ".runners",
}


def __getattr__(name: str):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)


def __dir__():
    return sorted(set(globals()) | set(_SYMBOL_MODULES))


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
