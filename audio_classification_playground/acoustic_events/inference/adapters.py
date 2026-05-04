"""Adapters from inference artifacts to existing producer inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..producers.affect.types import Signal, Vad
from .artifacts import PredictionArtifact, load_prediction_artifact


def artifact_to_affect_signals(
    artifact: PredictionArtifact | str | Path,
) -> list[Signal]:
    artifact = _coerce_artifact(artifact)
    _require_task(artifact, "affect")
    timing = artifact.manifest["timing"]
    return [
        Signal(name, artifact.arrays[name], timing["hop_sec"], timing["window_sec"])
        for name in ("arousal", "valence", "dominance")
    ]


def artifact_to_disfluency_logits(
    artifact: PredictionArtifact | str | Path,
) -> tuple[np.ndarray, np.ndarray, float, float, float | None]:
    artifact = _coerce_artifact(artifact)
    _require_task(artifact, "disfluency")
    timing = artifact.manifest["timing"]
    return (
        artifact.arrays["fluency_logits"],
        artifact.arrays["disfluency_type_logits"],
        float(timing["hop_sec"]),
        float(timing["window_sec"]),
        _duration_if_receptive_field_fits(artifact),
    )


def artifact_to_emotion_probabilities(
    artifact: PredictionArtifact | str | Path,
) -> tuple[np.ndarray, tuple[str, ...], float, float, float | None]:
    artifact = _coerce_artifact(artifact)
    _require_task(artifact, "emotion")
    timing = artifact.manifest["timing"]
    return (
        artifact.arrays["probabilities"],
        tuple(artifact.manifest["labels"]),
        float(timing["hop_sec"]),
        float(timing["window_sec"]),
        float(artifact.manifest["audio"]["duration_sec"]),
    )


def artifact_to_vad(artifact: PredictionArtifact | str | Path) -> Vad:
    artifact = _coerce_artifact(artifact)
    _require_task(artifact, "vad")
    intervals = artifact.arrays["intervals_sec"]
    return Vad(intervals=tuple((float(s), float(e)) for s, e in intervals))


def _coerce_artifact(artifact: PredictionArtifact | str | Path) -> PredictionArtifact:
    if isinstance(artifact, PredictionArtifact):
        return artifact
    return load_prediction_artifact(artifact)


def _require_task(artifact: PredictionArtifact, task: str) -> None:
    if artifact.task != task:
        raise ValueError(f"Expected {task!r} artifact, got {artifact.task!r}")


def _duration_if_receptive_field_fits(artifact: PredictionArtifact) -> float | None:
    duration = float(artifact.manifest["audio"]["duration_sec"])
    timing = artifact.manifest["timing"]
    receptive_end = timing.get("receptive_field_end_sec")
    if receptive_end is None:
        n_frames = int(timing["n_frames"])
        receptive_end = (n_frames - 1) * float(timing["hop_sec"]) + float(timing["window_sec"])
    if float(receptive_end) > duration + 1e-6:
        return None
    return duration
