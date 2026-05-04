"""Compose inference artifacts into review-package producer outputs."""
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass, replace
import json
from pathlib import Path
from typing import Mapping, Sequence

from ..inference import (
    artifact_to_affect_signals,
    artifact_to_disfluency_logits,
    artifact_to_emotion_probabilities,
    artifact_to_vad,
    load_prediction_artifact,
)
from ..inference.artifacts import PredictionArtifact, sanitize_for_filename
from ..inference.log import get_logger
from ..producers.affect import Config as AffectConfig
from ..producers.affect.pipeline import (
    DEFAULT_PRODUCER_ID as AFFECT_PRODUCER_ID,
    extract_events_with_tracks,
    producer_run as affect_producer_run,
)
from ..producers.affect.preprocessing import build_blocks
from ..producers.disfluency import DisfluencyConfig, produce_disfluency_events
from ..producers.emotion import Config as EmotionConfig, run_from_probabilities
from ..schema import Event, PredictionTrack, ProducerRun
from .jsonutil import jsonable
from .package import build_package_payload, tracks_meta_for_package, write_review_package


LOGGER = get_logger()
TASK_ORDER = ("affect", "disfluency", "emotion")


def compose_affect_from_artifacts(
    *,
    affect_artifact: PredictionArtifact | str | Path,
    vad_artifact: PredictionArtifact | str | Path,
    config: AffectConfig | Mapping | None = None,
) -> tuple[ProducerRun, list[PredictionTrack], list[Event]]:
    """Run the affect producer from inference artifacts."""
    affect = _coerce_artifact(affect_artifact)
    vad_source = _coerce_artifact(vad_artifact)
    _require_task(affect, "affect")
    _require_task(vad_source, "vad")
    _validate_same_audio([affect, vad_source])

    cfg = _resolve_config(AffectConfig.balanced(), config)
    signals = artifact_to_affect_signals(affect)
    vad = artifact_to_vad(vad_source)
    events, tracks = extract_events_with_tracks(
        signals,
        vad,
        cfg,
        producer_id=AFFECT_PRODUCER_ID,
    )
    blocks = build_blocks(vad, cfg)
    run = affect_producer_run(
        cfg,
        blocks=blocks,
        producer_id=AFFECT_PRODUCER_ID,
        source_model=_source_model(affect),
    )
    run = _with_outputs(
        run,
        inference_artifacts={
            "affect": _artifact_provenance(affect),
            "vad": _artifact_provenance(vad_source),
        },
        composition={"vad_applied": True},
    )
    return run, list(tracks), list(events)


def compose_disfluency_from_artifacts(
    *,
    disfluency_artifact: PredictionArtifact | str | Path,
    config: DisfluencyConfig | Mapping | None = None,
) -> tuple[ProducerRun, list[PredictionTrack], list[Event]]:
    """Run the disfluency producer from an inference artifact."""
    artifact = _coerce_artifact(disfluency_artifact)
    _require_task(artifact, "disfluency")
    cfg = _resolve_config(DisfluencyConfig.balanced(), config)
    fluency, types, hop_sec, window_sec, duration = artifact_to_disfluency_logits(artifact)
    run, tracks, events = produce_disfluency_events(
        fluency_logits=fluency,
        disfluency_type_logits=types,
        hop_sec=hop_sec,
        window_sec=window_sec,
        audio_duration_sec=duration,
        config=cfg,
        source_model=_source_model(artifact),
    )
    run = _with_outputs(
        run,
        inference_artifacts={"disfluency": _artifact_provenance(artifact)},
        composition={"vad_applied": False},
    )
    return run, list(tracks), list(events)


def compose_emotion_from_artifacts(
    *,
    emotion_artifact: PredictionArtifact | str | Path,
    config: EmotionConfig | Mapping | None = None,
) -> tuple[ProducerRun, list[PredictionTrack], list[Event]]:
    """Run the emotion producer from an inference artifact.

    VAD is intentionally not applied by default in this composition phase.
    """
    artifact = _coerce_artifact(emotion_artifact)
    _require_task(artifact, "emotion")
    cfg = _resolve_config(EmotionConfig.balanced(), config)
    probabilities, labels, hop_sec, window_sec, duration = artifact_to_emotion_probabilities(artifact)
    run, tracks, events = run_from_probabilities(
        probabilities,
        labels,
        hop_sec=hop_sec,
        window_sec=window_sec,
        audio_duration_sec=duration,
        vad_intervals=None,
        config=cfg,
        source_model=_source_model(artifact),
    )
    run = _with_outputs(
        run,
        inference_artifacts={"emotion": _artifact_provenance(artifact)},
        composition={"vad_applied": False},
    )
    return run, list(tracks), list(events)


def compose_review_package(
    *,
    affect_artifact: PredictionArtifact | str | Path,
    disfluency_artifact: PredictionArtifact | str | Path,
    emotion_artifact: PredictionArtifact | str | Path,
    vad_artifact: PredictionArtifact | str | Path,
    out_dir: str | Path,
    task_configs: Mapping[str, str | Path] | None = None,
) -> Path:
    """Run all producers from explicit artifacts and write a review package.

    Parameters
    ----------
    affect_artifact, disfluency_artifact, emotion_artifact, vad_artifact:
        Paths to artifact directories produced by
        ``audio_classification_playground.acoustic_events.inference`` or loaded
        ``PredictionArtifact`` objects.
    out_dir:
        Root directory for ``review_package.v1`` outputs. The concrete package
        path is ``<out_dir>/<recording_id>/<package_id>/``.
    task_configs:
        Optional mapping from task name to a JSON config file path, for example
        ``{"disfluency": "configs/disfluency-strict.json"}``. Supported task
        keys are ``affect``, ``disfluency``, and ``emotion``. The composer stores
        the resolved post-merge producer config in ``package.json``.
    """
    artifacts = {
        "affect": _coerce_artifact(affect_artifact),
        "disfluency": _coerce_artifact(disfluency_artifact),
        "emotion": _coerce_artifact(emotion_artifact),
        "vad": _coerce_artifact(vad_artifact),
    }
    _validate_artifact_set(artifacts)
    configs = _load_task_configs(task_configs or {})

    LOGGER.info("composing affect")
    affect_run, affect_tracks, affect_events = compose_affect_from_artifacts(
        affect_artifact=artifacts["affect"],
        vad_artifact=artifacts["vad"],
        config=configs.get("affect"),
    )
    LOGGER.info("affect: %d events, %d tracks", len(affect_events), len(affect_tracks))

    LOGGER.info("composing disfluency")
    disfluency_run, disfluency_tracks, disfluency_events = compose_disfluency_from_artifacts(
        disfluency_artifact=artifacts["disfluency"],
        config=configs.get("disfluency"),
    )
    LOGGER.info("disfluency: %d events, %d tracks", len(disfluency_events), len(disfluency_tracks))

    LOGGER.info("composing emotion")
    emotion_run, emotion_tracks, emotion_events = compose_emotion_from_artifacts(
        emotion_artifact=artifacts["emotion"],
        config=configs.get("emotion"),
    )
    LOGGER.info("emotion: %d events, %d tracks", len(emotion_events), len(emotion_tracks))

    runs = [affect_run, disfluency_run, emotion_run]
    tracks = [*affect_tracks, *disfluency_tracks, *emotion_tracks]
    events = _sort_events([*affect_events, *disfluency_events, *emotion_events])
    vad = artifact_to_vad(artifacts["vad"])
    package = build_package_payload(
        recording_id=_recording_id(artifacts.values()),
        audio=_audio_payload(artifacts["affect"]),
        vad_intervals=vad.intervals,
        inference_artifacts={
            task: _artifact_provenance(artifact)
            for task, artifact in sorted(artifacts.items())
        },
        producer_runs=[run.as_dict() for run in runs],
        events=[event.as_dict() for event in events],
        tracks_meta=tracks_meta_for_package(tracks),
    )
    path = write_review_package(out_dir=out_dir, package_payload=package, tracks=tracks)
    LOGGER.info("review package: %s (%s)", path, package["package_id"])
    return path


def _coerce_artifact(artifact: PredictionArtifact | str | Path) -> PredictionArtifact:
    if isinstance(artifact, PredictionArtifact):
        return artifact
    return load_prediction_artifact(artifact)


def _require_task(artifact: PredictionArtifact, task: str) -> None:
    if artifact.task != task:
        raise ValueError(f"Expected {task!r} artifact, got {artifact.task!r}")


def _validate_artifact_set(artifacts: Mapping[str, PredictionArtifact]) -> None:
    expected = {"affect", "disfluency", "emotion", "vad"}
    if set(artifacts) != expected:
        raise ValueError(f"Expected artifact keys {sorted(expected)}, got {sorted(artifacts)}")
    for task, artifact in artifacts.items():
        _require_task(artifact, task)
    _validate_same_audio(list(artifacts.values()))


def _validate_same_audio(artifacts: Sequence[PredictionArtifact]) -> None:
    hashes = {artifact.manifest["audio"]["sha256"] for artifact in artifacts}
    if len(hashes) != 1:
        raise ValueError(f"Artifacts have mixed audio_sha256 values: {sorted(hashes)}")


def _load_task_configs(paths: Mapping[str, str | Path]) -> dict[str, dict]:
    unknown = set(paths) - set(TASK_ORDER)
    if unknown:
        raise ValueError(f"Unknown producer config tasks: {sorted(unknown)}")
    out: dict[str, dict] = {}
    for task, path in paths.items():
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"{task} config must be a JSON object: {path}")
        out[task] = payload
    return out


def _resolve_config(default_config, override: Mapping | None):
    if override is None:
        return default_config
    if is_dataclass(override):
        return override
    allowed = {field.name for field in fields(default_config)}
    unknown = set(override) - allowed
    if unknown:
        raise ValueError(f"Unknown config fields for {type(default_config).__name__}: {sorted(unknown)}")
    return replace(default_config, **dict(override))


def _with_outputs(
    run: ProducerRun,
    *,
    inference_artifacts: Mapping,
    composition: Mapping,
) -> ProducerRun:
    outputs = dict(run.outputs)
    outputs["inference_artifacts"] = jsonable(inference_artifacts)
    outputs["composition"] = jsonable(composition)
    return ProducerRun(
        producer_id=run.producer_id,
        task=run.task,
        source_model=run.source_model,
        config=run.config,
        config_hash=run.config_hash,
        outputs=outputs,
    )


def _artifact_provenance(artifact: PredictionArtifact) -> dict:
    manifest = artifact.manifest
    return {
        "task": artifact.task,
        "manifest_path": str((artifact.path / "manifest.json").resolve()),
        "audio_sha256": manifest["audio"]["sha256"],
        "inference_config_hash": manifest["inference_config_hash"],
        "model": manifest.get("model", {}),
    }


def _source_model(artifact: PredictionArtifact) -> str:
    model = artifact.manifest.get("model", {})
    model_id = model.get("id", "")
    backbone = model.get("backbone")
    if backbone:
        return f"{model_id} ({backbone})"
    return str(model_id)


def _recording_id(artifacts) -> str:
    for artifact in artifacts:
        return sanitize_for_filename(str(artifact.manifest.get("recording_id") or artifact.path.parents[2].name))
    return "recording"


def _audio_payload(artifact: PredictionArtifact) -> dict:
    audio = artifact.manifest["audio"]
    return {
        "path": audio.get("path", ""),
        "sha256": audio["sha256"],
        "sample_rate": int(audio["sample_rate"]),
        "duration_sec": float(audio["duration_sec"]),
        "hash_semantics": audio.get("hash_semantics", "decoded_mono_16khz_float32"),
    }


def _sort_events(events: Sequence[Event]) -> list[Event]:
    task_rank = {task: i for i, task in enumerate(TASK_ORDER)}
    return sorted(
        events,
        key=lambda event: (
            float(event.start_sec),
            task_rank.get(event.task, 999),
            event.label,
            event.event_type,
            event.event_id,
        ),
    )
