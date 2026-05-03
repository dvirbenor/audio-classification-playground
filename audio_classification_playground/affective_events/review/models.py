"""Schema for labels and labeling sessions.

A *session* is the unit of label persistence: one (recording, config) pair.
Re-running detection with a different config produces a new session; labels
can be carried forward via :func:`inherit_labels` (see ``inherit.py``).

The on-disk format is JSON for the metadata + a sibling ``.npz`` for regular
track arrays. The split keeps the JSON small and human-readable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


VERDICTS: tuple[str, ...] = ("tp", "fp", "unclear", "partial")


@dataclass
class Label:
    """Per-event label."""

    verdict: str = ""        # one of VERDICTS, or "" for unset
    tags: list[str] = field(default_factory=list)
    comment: str = ""
    labeler: str = ""
    labeled_at: str = ""     # ISO-8601 UTC timestamp
    inherited_from: str | None = None
    inherited_match_score: float | None = None

    def is_set(self) -> bool:
        return bool(self.verdict)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "tags": list(self.tags),
            "comment": self.comment,
            "labeler": self.labeler,
            "labeled_at": self.labeled_at,
            "inherited_from": self.inherited_from,
            "inherited_match_score": self.inherited_match_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Label":
        return cls(
            verdict=d.get("verdict", ""),
            tags=list(d.get("tags") or []),
            comment=d.get("comment", "") or "",
            labeler=d.get("labeler", "") or "",
            labeled_at=d.get("labeled_at", "") or "",
            inherited_from=d.get("inherited_from"),
            inherited_match_score=d.get("inherited_match_score"),
        )


@dataclass
class LabelingSession:
    """Self-contained snapshot of one detection run + its accumulated labels.

    The ``tracks_data_path`` points to a sibling ``.npz`` file (relative to
    the session JSON) containing regular-grid track arrays. Everything else is
    inline so the JSON is portable and inspectable.
    """

    SCHEMA_VERSION: ClassVar[int] = 3

    session_id: str
    recording_id: str
    audio_path: str
    audio_sr: int
    audio_duration_sec: float

    producer_runs: list[dict]
    session_fingerprint: str

    tracks_meta: dict            # track_id -> serialized track metadata
    tracks_data_path: str        # filename, relative to session JSON's dir

    vad_intervals: list[list[float]]   # [[start, end], ...]

    events: list[dict]            # serialized acoustic Event dicts
    labels: dict[str, dict]       # event_id -> Label dict

    created_at: str
    last_updated_at: str
    schema_version: int = SCHEMA_VERSION
    event_schema: str = "acoustic_events.v1"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "recording_id": self.recording_id,
            "audio_path": self.audio_path,
            "audio_sr": self.audio_sr,
            "audio_duration_sec": self.audio_duration_sec,
            "producer_runs": self.producer_runs,
            "session_fingerprint": self.session_fingerprint,
            "tracks_meta": self.tracks_meta,
            "tracks_data_path": self.tracks_data_path,
            "vad_intervals": self.vad_intervals,
            "events": self.events,
            "labels": self.labels,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
            "event_schema": self.event_schema,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LabelingSession":
        return cls(
            schema_version=d.get("schema_version", cls.SCHEMA_VERSION),
            session_id=d["session_id"],
            recording_id=d["recording_id"],
            audio_path=d["audio_path"],
            audio_sr=int(d["audio_sr"]),
            audio_duration_sec=float(d["audio_duration_sec"]),
            producer_runs=list(d.get("producer_runs", [])),
            session_fingerprint=d.get("session_fingerprint", ""),
            tracks_meta=dict(d.get("tracks_meta", {})),
            tracks_data_path=d.get("tracks_data_path", ""),
            vad_intervals=[list(x) for x in d.get("vad_intervals", [])],
            events=list(d["events"]),
            labels=dict(d.get("labels", {})),
            created_at=d.get("created_at", ""),
            last_updated_at=d.get("last_updated_at", ""),
            event_schema=d.get("event_schema", "acoustic_events.v1"),
            notes=d.get("notes", "") or "",
        )
