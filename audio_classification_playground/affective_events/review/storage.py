"""Save / load / list labeling sessions.

Files on disk for a single session::

    <session_dir>/<recording_id>/<config_hash>__<timestamp>.json
    <session_dir>/<recording_id>/<config_hash>__<timestamp>.npz

The JSON is the canonical session record (human-readable, diffable, tiny).
The ``.npz`` holds raw signal arrays — too large for JSON, too small to
deserve their own subdirectory.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..config import Config
from ..preprocessing import build_blocks
from ..types import Event, Signal, Vad
from .models import Label, LabelingSession


_SESSION_SUFFIX = ".json"
_SIGNALS_SUFFIX = ".npz"


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
    signals: Sequence[Signal],
    vad: Vad,
    config: Config,
    audio_path: str | Path,
    session_dir: str | Path,
    recording_id: str | None = None,
    inherit_from: str | Path | None = None,
    inherit_overlap_threshold: float = 0.5,
    notes: str = "",
) -> Path:
    """Persist a session JSON (+ companion ``.npz``). Returns the JSON path.

    If ``inherit_from`` points to a previous session, labels are pre-populated
    via time-overlap matching on the same signal name.
    """
    audio_path = Path(audio_path).resolve()
    session_dir = Path(session_dir)
    if recording_id is None:
        recording_id = audio_path.stem
    recording_id = _sanitize_for_filename(recording_id)

    sr, duration = _audio_metadata(audio_path)
    chash = config_hash(config)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_id = f"{chash}__{ts}"

    out_dir = session_dir / recording_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- signals → .npz
    signals_filename = f"{session_id}{_SIGNALS_SUFFIX}"
    np.savez_compressed(
        out_dir / signals_filename,
        **{s.name: s.values.astype(np.float32, copy=False) for s in signals},
    )
    signals_meta = {
        s.name: {
            "hop_sec": float(s.hop_sec),
            "window_sec": float(s.window_sec),
            "n_frames": int(s.n_frames),
        }
        for s in signals
    }

    # --- blocks (use the same merging the pipeline uses, so the UI matches)
    block_objs = build_blocks(vad, config)
    blocks = [
        {
            "block_id": b.block_id,
            "start_sec": b.start_sec,
            "end_sec": b.end_sec,
            "gap_before_sec": b.gap_before_sec if b.gap_before_sec != float("inf") else None,
            "gap_after_sec": b.gap_after_sec if b.gap_after_sec != float("inf") else None,
        }
        for b in block_objs
    ]

    # --- events → list of dicts
    event_dicts = [asdict(e) for e in events]

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
        config=asdict(config),
        config_hash=chash,
        signals_meta=signals_meta,
        signals_data_path=signals_filename,
        vad_intervals=[[float(s), float(e)] for s, e in vad.intervals],
        blocks=blocks,
        events=event_dicts,
        labels=labels,
        created_at=_utc_now_iso(),
        last_updated_at=_utc_now_iso(),
        notes=notes,
    )

    json_path = out_dir / f"{session_id}{_SESSION_SUFFIX}"
    _atomic_write_json(json_path, session.to_dict())
    return json_path


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
