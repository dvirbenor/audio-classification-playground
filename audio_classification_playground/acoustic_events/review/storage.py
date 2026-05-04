"""Save / load / list labeling sessions.

Files on disk for a single session::

    <session_dir>/<recording_id>/<timestamp>__<short_uuid>.json
    <session_dir>/<recording_id>/<timestamp>__<short_uuid>.npz

The JSON is the canonical session record (human-readable, diffable, tiny).
The ``.npz`` holds regular-grid track arrays — too large for JSON, too small
to deserve their own subdirectory. Sparse marker tracks live in JSON metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from ..producers.affect.config import Config
from ..producers.affect.pipeline import DEFAULT_PRODUCER_ID, producer_run as affect_producer_run
from ..producers.affect.preprocessing import build_blocks
from ..producers.affect.types import Vad
from ..schema import Event, PredictionTrack, ProducerRun, RegularGridTrack, track_meta
from .models import Label, LabelingSession


_SESSION_SUFFIX = ".json"
_TRACKS_SUFFIX = ".npz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sanitize_for_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def config_hash(config: Config | dict) -> str:
    payload = asdict(config) if is_dataclass(config) and not isinstance(config, type) else dict(config)
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def session_fingerprint(
    producer_runs: Sequence[ProducerRun | dict],
    events: Sequence[Event | dict],
    tracks: Sequence[PredictionTrack],
) -> str:
    producers = [
        p.as_dict() if hasattr(p, "as_dict") else dict(p)
        for p in producer_runs
    ]
    payload = {
        "producers": [
            {
                "producer_id": p.get("producer_id"),
                "task": p.get("task"),
                "config_hash": p.get("config_hash"),
            }
            for p in sorted(producers, key=lambda x: x.get("producer_id", ""))
        ],
        "event_count": len(events),
        "track_count": len(tracks),
        "track_ids": sorted(t.track_id for t in tracks),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _audio_metadata(audio_path: Path) -> tuple[int, float]:
    """Return ``(sample_rate, duration_sec)`` without decoding the whole file."""
    import soundfile as sf

    try:
        info = sf.info(str(audio_path))
        return int(info.samplerate), float(info.frames) / float(info.samplerate)
    except Exception:
        # mp3 may need a librosa fallback on some systems
        import librosa

        sr = librosa.get_samplerate(str(audio_path))
        dur = librosa.get_duration(path=str(audio_path))
        return int(sr), float(dur)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False, default=_json_default)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Cannot JSON-serialize {type(obj).__name__}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_session(
    *,
    events: Sequence[Event],
    tracks: Sequence[PredictionTrack],
    vad: Vad,
    audio_path: str | Path,
    session_dir: str | Path,
    producer_runs: Sequence[ProducerRun | dict] | None = None,
    config: Config | dict | None = None,
    recording_id: str | None = None,
    inherit_from: str | Path | None = None,
    inherit_overlap_threshold: float = 0.5,
    notes: str = "",
) -> Path:
    """Persist a legacy session JSON (+ companion ``.npz``). Returns the JSON path.

    Deprecated: new workflows should compose ``review_package.v1`` directories
    via :mod:`audio_classification_playground.acoustic_events.composition`.

    If ``inherit_from`` points to a previous session, labels are pre-populated
    via semantic key + time-overlap matching.
    """
    audio_path = Path(audio_path).resolve()
    session_dir = Path(session_dir)
    if recording_id is None:
        recording_id = audio_path.stem
    recording_id = _sanitize_for_filename(recording_id)

    sr, duration = _audio_metadata(audio_path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_id = f"{ts}__{uuid.uuid4().hex[:8]}"

    out_dir = session_dir / recording_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- tracks → .npz + JSON metadata
    tracks_filename = f"{session_id}{_TRACKS_SUFFIX}"
    arrays = {
        t.track_id: t.values.astype(np.float32, copy=False)
        for t in tracks
        if isinstance(t, RegularGridTrack)
    }
    if arrays:
        np.savez_compressed(out_dir / tracks_filename, **arrays)
    else:
        np.savez_compressed(out_dir / tracks_filename)
    tracks_meta = {t.track_id: track_meta(t) for t in tracks}

    # --- events → list of dicts
    event_dicts = [e.as_dict() if hasattr(e, "as_dict") else asdict(e) for e in events]

    producer_run_dicts = _producer_run_dicts(
        producer_runs=producer_runs,
        tracks=tracks,
        vad=vad,
        config=config,
    )
    fingerprint = session_fingerprint(producer_run_dicts, event_dicts, tracks)

    # --- inheritance (if requested)
    labels: dict[str, dict] = {}
    if inherit_from is not None:
        from .inherit import inherit_labels

        prev = load_session_json(Path(inherit_from))
        labels = inherit_labels(
            prev_session=prev,
            new_events=event_dicts,
            overlap_threshold=inherit_overlap_threshold,
        )

    session = LabelingSession(
        session_id=session_id,
        recording_id=recording_id,
        audio_path=str(audio_path),
        audio_sr=sr,
        audio_duration_sec=duration,
        producer_runs=producer_run_dicts,
        session_fingerprint=fingerprint,
        tracks_meta=tracks_meta,
        tracks_data_path=tracks_filename,
        vad_intervals=[[float(s), float(e)] for s, e in vad.intervals],
        events=event_dicts,
        labels=labels,
        created_at=_utc_now_iso(),
        last_updated_at=_utc_now_iso(),
        event_schema="acoustic_events.v1",
        notes=notes,
    )

    json_path = out_dir / f"{session_id}{_SESSION_SUFFIX}"
    _atomic_write_json(json_path, session.to_dict())
    return json_path


def _producer_run_dicts(
    *,
    producer_runs: Sequence[ProducerRun | dict] | None,
    tracks: Sequence[PredictionTrack],
    vad: Vad,
    config: Config | dict | None,
) -> list[dict]:
    if producer_runs is not None:
        return [
            p.as_dict() if hasattr(p, "as_dict") else dict(p)
            for p in producer_runs
        ]

    producer_ids = sorted({t.producer_id for t in tracks})
    if not producer_ids and config is None:
        return []

    out: list[dict] = []
    for producer_id in producer_ids or [DEFAULT_PRODUCER_ID]:
        if producer_id == DEFAULT_PRODUCER_ID and config is not None:
            cfg = config if isinstance(config, Config) else Config(**dict(config))
            blocks = build_blocks(vad, cfg)
            out.append(affect_producer_run(cfg, blocks=blocks).as_dict())
        else:
            out.append(ProducerRun(
                producer_id=producer_id,
                task=_task_for_producer(producer_id, tracks),
                source_model="",
            ).as_dict())
    return out


def _task_for_producer(producer_id: str, tracks: Sequence[PredictionTrack]) -> str:
    for track in tracks:
        if track.producer_id == producer_id:
            return track.task
    return ""


# ---------------------------------------------------------------------------
# Load / list / update
# ---------------------------------------------------------------------------


def load_session_json(path: str | Path) -> dict:
    """Return the raw session dict (no schema validation beyond JSON parsing)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_session(path: str | Path) -> LabelingSession:
    return LabelingSession.from_dict(load_session_json(path))


def save_session_dict(path: str | Path, session: dict) -> None:
    session = dict(session)
    session["last_updated_at"] = _utc_now_iso()
    _atomic_write_json(Path(path), session)


def list_sessions(session_dir: str | Path) -> list[Path]:
    """Return all session JSON files under ``session_dir``, sorted newest-first."""
    root = Path(session_dir)
    if not root.exists():
        return []
    paths = list(root.glob(f"*/*{_SESSION_SUFFIX}"))
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


def update_label(session_path: str | Path, event_id: str, label: Label | dict) -> dict:
    """Atomically update a single event's label and persist."""
    p = Path(session_path)
    sess = load_session_json(p)
    if event_id not in {e["event_id"] for e in sess["events"]}:
        raise KeyError(f"event_id {event_id!r} not in session")
    payload = label.to_dict() if isinstance(label, Label) else dict(label)
    if not payload.get("labeled_at"):
        payload["labeled_at"] = _utc_now_iso()
    sess.setdefault("labels", {})[event_id] = payload
    save_session_dict(p, sess)
    return payload


def clear_label(session_path: str | Path, event_id: str) -> None:
    p = Path(session_path)
    sess = load_session_json(p)
    sess.get("labels", {}).pop(event_id, None)
    save_session_dict(p, sess)
