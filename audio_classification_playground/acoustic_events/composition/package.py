"""Read and write deterministic review packages."""
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

from ..schema import MarkerTrack, PredictionTrack, RegularGridTrack, track_meta
from ..inference.artifacts import sanitize_for_filename
from .jsonutil import canonical_json_bytes, jsonable, pretty_json_text


PACKAGE_SCHEMA = "review_package.v1"
PACKAGE_JSON = "package.json"
LABELS_JSON = "labels.json"
TRACKS_DIR = "tracks"


@dataclass(frozen=True)
class ReviewPackage:
    """Typed view of a review package directory."""

    path: Path
    package: dict
    labels: dict

    @property
    def package_id(self) -> str:
        return str(self.package["package_id"])

    @property
    def package_fingerprint(self) -> str:
        return str(self.package["package_fingerprint"])

    @property
    def recording_id(self) -> str:
        return str(self.package["recording_id"])

    @property
    def events(self) -> list[dict]:
        return list(self.package.get("events", []))

    @property
    def tracks_meta(self) -> dict:
        return dict(self.package.get("tracks_meta", {}))


def build_package_payload(
    *,
    recording_id: str,
    audio: Mapping,
    vad_intervals: Sequence[Sequence[float]],
    inference_artifacts: Mapping,
    producer_runs: Sequence[Mapping],
    events: Sequence[Mapping],
    tracks_meta: Mapping,
) -> dict:
    """Build immutable package JSON and assign deterministic identity."""
    base = {
        "schema": PACKAGE_SCHEMA,
        "recording_id": sanitize_for_filename(recording_id),
        "audio": dict(audio),
        "vad_intervals": [[float(s), float(e)] for s, e in vad_intervals],
        "inference_artifacts": dict(inference_artifacts),
        "producer_runs": list(producer_runs),
        "events": list(events),
        "tracks_meta": dict(tracks_meta),
    }
    fingerprint = package_fingerprint(base)
    base["package_fingerprint"] = fingerprint
    base["package_id"] = fingerprint[:24]
    return jsonable(base)


def package_fingerprint(package_payload: Mapping) -> str:
    """Hash package content while excluding path-like fields and identity fields."""
    canonical = _fingerprint_payload(package_payload)
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def write_review_package(
    *,
    out_dir: str | Path,
    package_payload: Mapping,
    tracks: Sequence[PredictionTrack],
) -> Path:
    """Write a review package idempotently and return its directory."""
    payload = jsonable(package_payload)
    recording_id = sanitize_for_filename(str(payload["recording_id"]))
    package_id = str(payload["package_id"])
    package_dir = Path(out_dir) / recording_id / package_id
    package_json = package_dir / PACKAGE_JSON
    labels_json = package_dir / LABELS_JSON

    if package_json.is_file():
        existing = _read_json(package_json)
        if existing.get("package_fingerprint") == payload.get("package_fingerprint"):
            missing_tracks = _missing_track_files(package_dir, tracks)
            if missing_tracks:
                tracks_dir = package_dir / TRACKS_DIR
                tracks_dir.mkdir(parents=True, exist_ok=True)
                _write_track_arrays(tracks_dir, tracks)
            if not labels_json.is_file():
                _atomic_write_text(labels_json, pretty_json_text({}))
            return package_dir

    tracks_dir = package_dir / TRACKS_DIR
    tracks_dir.mkdir(parents=True, exist_ok=True)
    _write_track_arrays(tracks_dir, tracks)
    _atomic_write_text(package_json, pretty_json_text(payload))
    if not labels_json.exists():
        _atomic_write_text(labels_json, pretty_json_text({}))
    return package_dir


def load_review_package(path: str | Path) -> ReviewPackage:
    package_dir = Path(path).resolve()
    if not package_dir.is_dir():
        raise FileNotFoundError(f"review package directory not found: {package_dir}")
    package_json = package_dir / PACKAGE_JSON
    labels_json = package_dir / LABELS_JSON
    if not package_json.is_file():
        raise FileNotFoundError(f"package.json not found: {package_json}")
    package = _read_json(package_json)
    if package.get("schema") != PACKAGE_SCHEMA:
        raise ValueError(f"Unsupported review package schema: {package.get('schema')!r}")
    labels = _read_json(labels_json) if labels_json.is_file() else {}
    return ReviewPackage(path=package_dir, package=package, labels=labels)


def update_package_label(package_path: str | Path, event_id: str, label: Mapping) -> dict:
    pkg = load_review_package(package_path)
    event_ids = {event["event_id"] for event in pkg.events}
    if event_id not in event_ids:
        raise KeyError(f"event_id {event_id!r} not in package")
    payload = dict(label)
    if not payload.get("labeled_at"):
        payload["labeled_at"] = _utc_now_iso()
    labels = dict(pkg.labels)
    labels[event_id] = jsonable(payload)
    _atomic_write_text(pkg.path / LABELS_JSON, pretty_json_text(labels))
    return labels[event_id]


def clear_package_label(package_path: str | Path, event_id: str) -> None:
    pkg = load_review_package(package_path)
    labels = dict(pkg.labels)
    labels.pop(event_id, None)
    _atomic_write_text(pkg.path / LABELS_JSON, pretty_json_text(labels))


def tracks_meta_for_package(tracks: Sequence[PredictionTrack]) -> dict:
    meta: dict[str, dict] = {}
    for track in tracks:
        row = track_meta(track)
        if isinstance(track, RegularGridTrack):
            row["data_path"] = f"{TRACKS_DIR}/{_track_filename(track.producer_id)}"
        elif isinstance(track, MarkerTrack):
            row.pop("data_path", None)
        meta[track.track_id] = row
    return meta


def arrays_by_producer(tracks: Sequence[PredictionTrack]) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for track in tracks:
        if isinstance(track, RegularGridTrack):
            grouped.setdefault(track.producer_id, {})[track.track_id] = track.values.astype(
                np.float32,
                copy=False,
            )
    return grouped


def _missing_track_files(package_dir: Path, tracks: Sequence[PredictionTrack]) -> list[Path]:
    return [
        package_dir / TRACKS_DIR / _track_filename(producer_id)
        for producer_id in arrays_by_producer(tracks)
        if not (package_dir / TRACKS_DIR / _track_filename(producer_id)).is_file()
    ]


def _write_track_arrays(tracks_dir: Path, tracks: Sequence[PredictionTrack]) -> None:
    for producer_id, arrays in arrays_by_producer(tracks).items():
        final_path = tracks_dir / _track_filename(producer_id)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{final_path.name}.",
            suffix=".tmp.npz",
            dir=str(tracks_dir),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            np.savez_compressed(tmp_path, **arrays)
            os.replace(tmp_path, final_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


def _track_filename(producer_id: str) -> str:
    return f"{sanitize_for_filename(producer_id)}.npz"


def _fingerprint_payload(value):
    if isinstance(value, Mapping):
        return {
            str(k): _fingerprint_payload(v)
            for k, v in value.items()
            if k
            not in {
                "package_fingerprint",
                "package_id",
                "manifest_path",
                "path",
                "audio_path",
                "data_path",
            }
        }
    if isinstance(value, list):
        return [_fingerprint_payload(item) for item in value]
    return jsonable(value)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
