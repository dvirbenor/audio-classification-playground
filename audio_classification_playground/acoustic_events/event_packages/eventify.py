"""Single-archive eventification for atomic event packages."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import json
from pathlib import Path
from typing import Literal, Mapping, Sequence

from ..composition.composer import (
    compose_affect_from_artifacts,
    compose_disfluency_from_artifacts,
    compose_emotion_from_artifacts,
)
from ..composition.jsonutil import jsonable
from ..inference.artifacts import PredictionArtifact, load_prediction_artifact
from ..orchestration.progress import TASKS, are_tasks_complete_by_artifact
from ..producers.affect import Config as AffectConfig
from ..producers.disfluency import DisfluencyConfig
from ..producers.disfluency.config import LABEL_TO_EVENT_LABEL
from ..producers.emotion import Config as EmotionConfig
from ..schema import Event, ProducerRun
from .package import (
    EventPackage,
    build_event_package_payload,
    event_config_fingerprint,
    input_fingerprint,
    load_event_package,
    write_event_package,
)


EVENT_POLICY = {
    "atomic_only": True,
    "affect_mode": "leaf_only",
    "affect_label_format": "axis_direction_compact",
    "vad_as_event": False,
}
NORMALIZER_VERSION = "event-package-normalizer.v1"
TASK_RANK = {"affect": 0, "disfluency": 1, "emotion": 2}
REQUIRED_TASKS = tuple(TASKS)

EventifyStatus = Literal["packaged", "skipped_complete", "not_ready"]


@dataclass(frozen=True)
class EventPackageConfigs:
    """Producer config overrides used by event packages."""

    affect: AffectConfig | Mapping | None = None
    disfluency: DisfluencyConfig | Mapping | None = None
    emotion: EmotionConfig | Mapping | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        affect: str | Path | None = None,
        disfluency: str | Path | None = None,
        emotion: str | Path | None = None,
    ) -> "EventPackageConfigs":
        return cls(
            affect=_read_json(affect) if affect else None,
            disfluency=_read_json(disfluency) if disfluency else None,
            emotion=_read_json(emotion) if emotion else None,
        )

    def resolved(self) -> dict[str, object]:
        return {
            "affect": _resolve_config(AffectConfig.balanced(), self.affect),
            "disfluency": _resolve_config(DisfluencyConfig.balanced(), self.disfluency),
            "emotion": _resolve_config(EmotionConfig.balanced(), self.emotion),
        }

    def resolved_dict(self) -> dict[str, dict]:
        return {
            task: _config_dict(config)
            for task, config in self.resolved().items()
        }

    def config_fingerprint(self) -> str:
        return event_config_fingerprint(
            producer_configs=self.resolved_dict(),
            event_policy=EVENT_POLICY,
            normalizer_version=NORMALIZER_VERSION,
        )


@dataclass(frozen=True)
class EventifyResult:
    """Outcome of eventifying one archive."""

    status: EventifyStatus
    package_path: Path | None = None
    package_payload: dict | None = None
    events: tuple[dict, ...] = ()
    reason: str = ""


def eventify_archive(
    *,
    inference_archive_dir: str | Path,
    output_archive_dir: str | Path,
    session_id: str,
    archive_id: str,
    date: str = "",
    configs: EventPackageConfigs | None = None,
    force: bool = False,
    validate_inputs: bool = False,
) -> EventifyResult:
    """Build one atomic event package if its source artifacts are ready.

    This is the reusable unit used both by the one-archive CLI and by the
    continuous CPU worker.  It never waits for missing artifacts.
    """
    configs = configs or EventPackageConfigs()
    inference_archive_dir = Path(inference_archive_dir)
    output_archive_dir = Path(output_archive_dir)
    event_cfg_fp = configs.config_fingerprint()

    if not force:
        existing = _load_complete_if_matching(
            output_archive_dir,
            event_config_fingerprint_value=event_cfg_fp,
            validate_inputs=validate_inputs,
            inference_archive_dir=inference_archive_dir,
        )
        if existing is not None:
            return EventifyResult(
                status="skipped_complete",
                package_path=existing.path,
                package_payload=existing.package,
                reason="complete",
            )

    if not required_artifacts_complete(inference_archive_dir):
        return EventifyResult(status="not_ready", reason="prediction_artifacts_incomplete")

    artifacts = load_archive_artifacts(inference_archive_dir)
    resolved = configs.resolved()

    affect_run, _, affect_events = compose_affect_from_artifacts(
        affect_artifact=artifacts["affect"],
        vad_artifact=artifacts["vad"],
        config=resolved["affect"],
    )
    disfluency_run, _, disfluency_events = compose_disfluency_from_artifacts(
        disfluency_artifact=artifacts["disfluency"],
        vad_artifact=artifacts["vad"],
        config=resolved["disfluency"],
    )
    emotion_run, _, emotion_events = compose_emotion_from_artifacts(
        emotion_artifact=artifacts["emotion"],
        vad_artifact=artifacts["vad"],
        config=resolved["emotion"],
    )
    events = normalize_events([*affect_events, *disfluency_events, *emotion_events])
    source_artifacts = {
        task: _artifact_provenance(artifact)
        for task, artifact in sorted(artifacts.items())
    }
    input_fp = input_fingerprint(
        source_artifacts=source_artifacts,
        event_config_fingerprint_value=event_cfg_fp,
    )
    runs = [affect_run, disfluency_run, emotion_run]
    payload = build_event_package_payload(
        session_id=session_id,
        archive_id=archive_id,
        date=date,
        audio=_audio_payload(artifacts["affect"]),
        source_artifacts=source_artifacts,
        producer_runs=[_producer_run_payload(run) for run in runs],
        event_policy=EVENT_POLICY,
        producer_configs=configs.resolved_dict(),
        event_config_fingerprint_value=event_cfg_fp,
        input_fingerprint_value=input_fp,
        events=events,
    )
    path = write_event_package(
        package_dir=output_archive_dir,
        package_payload=payload,
        events=events,
    )
    return EventifyResult(
        status="packaged",
        package_path=path,
        package_payload=payload,
        events=tuple(events),
    )


def required_artifacts_complete(inference_archive_dir: str | Path) -> bool:
    path = Path(inference_archive_dir)
    if len(path.parts) >= 2:
        # Use the existing orchestration helper against the parent layout:
        # <base>/<session_id>/<archive_id>/<task>
        output_base = path.parents[1]
        session_id = path.parent.name
        archive_id = path.name
        try:
            return are_tasks_complete_by_artifact(
                output_base,
                session_id,
                archive_id,
                REQUIRED_TASKS,
            )
        except ValueError:
            pass
    return all((path / task / "manifest.json").is_file() and (path / task / "predictions.npz").is_file()
               for task in REQUIRED_TASKS)


def load_archive_artifacts(inference_archive_dir: str | Path) -> dict[str, PredictionArtifact]:
    path = Path(inference_archive_dir)
    return {
        task: load_prediction_artifact(path / task)
        for task in REQUIRED_TASKS
    }


def normalize_events(events: Sequence[Event]) -> list[dict]:
    rows: list[dict] = []
    for event in events:
        if event.task == "affect":
            if event.event_type != "deviation":
                continue
            row = _normalize_affect_event(event)
        elif event.task == "disfluency":
            row = _normalize_generic_event(event, labels=_disfluency_labels(event))
        elif event.task == "emotion":
            row = _normalize_generic_event(event, labels=[event.label])
        else:
            row = _normalize_generic_event(event, labels=[event.label])
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            float(row["start_sec"]),
            TASK_RANK.get(str(row["task"]), 999),
            str(row["label"]),
            str(row["event_type"]),
            str(row["event_id"]),
        ),
    )


def _normalize_affect_event(event: Event) -> dict:
    axis = _affect_axis(event)
    direction = event.direction or ""
    label = f"{axis}{direction}" if axis and direction else event.label
    metadata = {"producer_label": event.label}
    row = _normalize_generic_event(event, labels=[label], label=label, metadata=metadata)
    row["axis"] = axis
    return row


def _normalize_generic_event(
    event: Event,
    *,
    labels: Sequence[str],
    label: str | None = None,
    metadata: Mapping | None = None,
) -> dict:
    compact_labels = _dedupe_labels([str(item) for item in labels if item])
    primary_label = label or event.label
    if primary_label not in compact_labels:
        compact_labels.insert(0, primary_label)
    row = {
        "event_id": event.event_id,
        "producer_id": event.producer_id,
        "task": event.task,
        "event_type": event.event_type,
        "label": primary_label,
        "labels": compact_labels,
        "start_sec": float(event.start_sec),
        "end_sec": float(event.end_sec),
        "duration_sec": float(event.duration_sec),
        "source_track_ids": list(event.source_track_ids),
        "score": float(event.score),
        "score_name": event.score_name,
        "direction": event.direction,
    }
    if metadata:
        row["metadata"] = dict(metadata)
    return jsonable(row)


def _disfluency_labels(event: Event) -> list[str]:
    labels = [event.label]
    for item in event.evidence.get("active_types", []) or []:
        if not isinstance(item, Mapping):
            continue
        raw = str(item.get("name") or "")
        normalized = LABEL_TO_EVENT_LABEL.get(raw)
        if normalized:
            labels.append(normalized)
    return _dedupe_labels(labels)


def _dedupe_labels(labels: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def _affect_axis(event: Event) -> str:
    if event.label.endswith("_deviation"):
        return event.label[: -len("_deviation")]
    for track_id in event.source_track_ids:
        if track_id.startswith("affect."):
            return track_id.split(".", 1)[1]
    return ""


def _load_complete_if_matching(
    package_dir: Path,
    *,
    event_config_fingerprint_value: str,
    validate_inputs: bool,
    inference_archive_dir: Path,
) -> EventPackage | None:
    try:
        package = load_event_package(package_dir)
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
        return None
    if package.event_config_fingerprint != event_config_fingerprint_value:
        return None
    if not validate_inputs:
        return package
    if not required_artifacts_complete(inference_archive_dir):
        return package
    artifacts = load_archive_artifacts(inference_archive_dir)
    source_artifacts = {
        task: _artifact_provenance(artifact)
        for task, artifact in sorted(artifacts.items())
    }
    observed = input_fingerprint(
        source_artifacts=source_artifacts,
        event_config_fingerprint_value=event_config_fingerprint_value,
    )
    return package if observed == package.input_fingerprint else None


def _resolve_config(default_config, override: Mapping | object | None):
    if override is None:
        return default_config
    if is_dataclass(override):
        return override
    allowed = {field.name for field in fields(default_config)}
    unknown = set(override) - allowed  # type: ignore[arg-type]
    if unknown:
        raise ValueError(
            f"Unknown config fields for {type(default_config).__name__}: {sorted(unknown)}"
        )
    return replace(default_config, **dict(override))  # type: ignore[arg-type]


def _config_dict(config) -> dict:
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def _producer_run_payload(run: ProducerRun) -> dict:
    return jsonable(run.as_dict())


def _artifact_provenance(artifact: PredictionArtifact) -> dict:
    manifest = artifact.manifest
    return {
        "task": artifact.task,
        "manifest_path": str((artifact.path / "manifest.json").resolve()),
        "audio_sha256": manifest["audio"]["sha256"],
        "inference_config_hash": manifest["inference_config_hash"],
        "model": manifest.get("model", {}),
    }


def _audio_payload(artifact: PredictionArtifact) -> dict:
    audio = artifact.manifest["audio"]
    return {
        "path": audio.get("path", ""),
        "sha256": audio["sha256"],
        "sample_rate": int(audio["sample_rate"]),
        "duration_sec": float(audio["duration_sec"]),
        "hash_semantics": audio.get("hash_semantics", "decoded_mono_16khz_float32"),
        **({"source_key": audio["source_key"]} if "source_key" in audio else {}),
    }


def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return payload
