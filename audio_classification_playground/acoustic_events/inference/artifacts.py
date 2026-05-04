"""Prediction artifact storage for acoustic-event inference runs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np


SAMPLE_RATE = 16_000
SCHEMA = "acoustic_predictions.v1"
PREDICTIONS_FILENAME = "predictions.npz"
MANIFEST_FILENAME = "manifest.json"
_COMPLETE_STATUS = "complete"


@dataclass(frozen=True)
class PredictionArtifact:
    """A loaded inference artifact: metadata plus numerical arrays."""

    task: str
    path: Path
    manifest: dict
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class InferenceRunResult:
    """Return value for a multi-task inference invocation."""

    artifacts: dict[str, PredictionArtifact]
    reused: dict[str, bool]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_for_filename(text: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_." else "_" for c in text)
    return out or "recording"


def decoded_audio_sha256(samples: np.ndarray) -> str:
    """Hash the exact mono 16 kHz float32 samples used as model input."""
    arr = np.ascontiguousarray(samples, dtype="<f4")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def inference_config_hash(config: Mapping) -> str:
    """Hash inference-only parameters, distinct from producer config hashes."""
    blob = json.dumps(dict(config), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def artifact_dir(
    out_dir: str | Path,
    *,
    recording_id: str,
    audio_sha256: str,
    task: str,
    inference_config_hash_value: str,
) -> Path:
    return (
        Path(out_dir)
        / sanitize_for_filename(recording_id)
        / audio_sha256
        / task
        / inference_config_hash_value
    )


def load_prediction_artifact(path: str | Path) -> PredictionArtifact:
    """Load a complete artifact from its directory or manifest path."""
    p = Path(path)
    artifact_path = p.parent if p.name == MANIFEST_FILENAME else p
    manifest_path = artifact_path / MANIFEST_FILENAME
    predictions_path = artifact_path / PREDICTIONS_FILENAME

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("status") != _COMPLETE_STATUS:
        raise ValueError(f"Artifact is not complete: {manifest_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Artifact predictions not found: {predictions_path}")

    with np.load(predictions_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    return PredictionArtifact(
        task=str(manifest["task"]),
        path=artifact_path,
        manifest=manifest,
        arrays=arrays,
    )


def write_prediction_artifact(
    artifact_path: str | Path,
    *,
    manifest: Mapping,
    arrays: Mapping[str, np.ndarray],
) -> PredictionArtifact:
    """Atomically write arrays first, then a complete manifest last."""
    path = Path(artifact_path)
    path.mkdir(parents=True, exist_ok=True)
    prepared_arrays = {
        str(name): np.asarray(values)
        for name, values in arrays.items()
    }
    final_npz = path / PREDICTIONS_FILENAME
    final_manifest = path / MANIFEST_FILENAME

    fd, tmp_npz_name = tempfile.mkstemp(
        prefix=f"{PREDICTIONS_FILENAME}.",
        suffix=".tmp.npz",
        dir=str(path),
    )
    os.close(fd)
    tmp_npz = Path(tmp_npz_name)
    try:
        np.savez_compressed(tmp_npz, **prepared_arrays)
        os.replace(tmp_npz, final_npz)
        _validate_written_npz(final_npz, prepared_arrays)

        payload = dict(manifest)
        payload["schema"] = SCHEMA
        payload["status"] = _COMPLETE_STATUS
        payload["predictions_path"] = PREDICTIONS_FILENAME
        payload["arrays"] = {
            name: {
                "shape": list(np.asarray(values).shape),
                "dtype": str(np.asarray(values).dtype),
            }
            for name, values in prepared_arrays.items()
        }
        _atomic_write_json(final_manifest, payload)
    finally:
        if tmp_npz.exists():
            tmp_npz.unlink()

    return load_prediction_artifact(path)


def list_cached_artifacts(
    out_dir: str | Path,
    *,
    recording_id: str | None = None,
    audio_sha256: str | None = None,
    task: str | None = None,
    inference_config_hash_value: str | None = None,
) -> list[PredictionArtifact]:
    """Return complete cached artifacts matching the provided filters."""
    root = Path(out_dir)
    if not root.exists():
        return []
    recording_id = sanitize_for_filename(recording_id) if recording_id is not None else None
    manifests = root.glob(f"**/{MANIFEST_FILENAME}")
    out: list[PredictionArtifact] = []
    for manifest_path in manifests:
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("status") != _COMPLETE_STATUS:
                continue
            if recording_id is not None and manifest.get("recording_id") != recording_id:
                continue
            if audio_sha256 is not None and manifest.get("audio", {}).get("sha256") != audio_sha256:
                continue
            if task is not None and manifest.get("task") != task:
                continue
            if (
                inference_config_hash_value is not None
                and manifest.get("inference_config_hash") != inference_config_hash_value
            ):
                continue
            if not (manifest_path.parent / PREDICTIONS_FILENAME).is_file():
                continue
            out.append(load_prediction_artifact(manifest_path.parent))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    out.sort(key=lambda artifact: str(artifact.path))
    return out


def find_cached_artifact(
    out_dir: str | Path,
    *,
    recording_id: str,
    audio_sha256: str,
    task: str,
    inference_config_hash_value: str,
) -> PredictionArtifact | None:
    path = artifact_dir(
        out_dir,
        recording_id=recording_id,
        audio_sha256=audio_sha256,
        task=task,
        inference_config_hash_value=inference_config_hash_value,
    )
    manifest_path = path / MANIFEST_FILENAME
    predictions_path = path / PREDICTIONS_FILENAME
    if not manifest_path.is_file() or not predictions_path.is_file():
        return None
    try:
        artifact = load_prediction_artifact(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    manifest = artifact.manifest
    if manifest.get("recording_id") != recording_id:
        return None
    if manifest.get("audio", {}).get("sha256") != audio_sha256:
        return None
    if manifest.get("task") != task:
        return None
    if manifest.get("inference_config_hash") != inference_config_hash_value:
        return None
    return artifact


def base_manifest(
    *,
    task: str,
    recording_id: str,
    audio_path: str | Path,
    audio_sha256: str,
    sample_rate: int,
    duration_sec: float,
    inference_config: Mapping,
    inference_config_hash_value: str,
    model: Mapping,
    timing: Mapping,
    runtime: Mapping,
    labels: Sequence[str] | None = None,
) -> dict:
    payload = {
        "schema": SCHEMA,
        "status": _COMPLETE_STATUS,
        "task": task,
        "recording_id": recording_id,
        "audio": {
            "path": str(Path(audio_path).resolve()),
            "sha256": audio_sha256,
            "sample_rate": int(sample_rate),
            "duration_sec": float(duration_sec),
            "hash_semantics": "decoded_mono_16khz_float32",
        },
        "inference_config": dict(inference_config),
        "inference_config_hash": inference_config_hash_value,
        "model": dict(model),
        "timing": dict(timing),
        "runtime": dict(runtime),
        "created_at": utc_now_iso(),
    }
    if labels is not None:
        payload["labels"] = [str(label) for label in labels]
    return payload


def _validate_written_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(arrays):
            raise ValueError(f"Unexpected arrays in {path}: {data.files}")
        for name, expected in arrays.items():
            observed = data[name]
            if observed.shape != np.asarray(expected).shape:
                raise ValueError(f"Array {name!r} changed shape during write")


def _atomic_write_json(path: Path, payload: Mapping) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dict(payload), f, indent=2, sort_keys=False, default=_json_default)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot JSON-serialize {type(obj).__name__}")
