"""Schema for labels and labeling sessions.

A *session* is the unit of label persistence: one (recording, config) pair.
Re-running detection with a different config produces a new session; labels
can be carried forward via :func:`inherit_labels` (see ``inherit.py``).

The on-disk format is JSON for the metadata + a sibling ``.npz`` for signal
arrays. The split keeps the JSON small and human-readable.
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

    The ``signals_data_path`` points to a sibling ``.npz`` file (relative to
    the session JSON) containing the raw signal arrays. Everything else is
    inline so the JSON is portable and inspectable.
    """

    SCHEMA_VERSION: ClassVar[int] = 2

    session_id: str
    recording_id: str
    audio_path: str
    audio_sr: int
    audio_duration_sec: float

    config: dict
    config_hash: str

    signals_meta: dict           # name -> {hop_sec, window_sec, n_frames}
    signals_data_path: str       # filename, relative to session JSON's dir

    vad_intervals: list[list[float]]   # [[start, end], ...]
    blocks: list[dict]                  # [{block_id, start_sec, end_sec}]

    events: list[dict]            # serialized v2 Event dicts
    labels: dict[str, dict]       # event_id -> Label dict

    created_at: str
    last_updated_at: str
    schema_version: int = SCHEMA_VERSION
    event_schema: str = "affective_events.v2"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "recording_id": self.recording_id,
            "audio_path": self.audio_path,
            "audio_sr": self.audio_sr,
            "audio_duration_sec": self.audio_duration_sec,
            "config": self.config,
            "config_hash": self.config_hash,
            "signals_meta": self.signals_meta,
            "signals_data_path": self.signals_data_path,
            "vad_intervals": self.vad_intervals,
            "blocks": self.blocks,
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
            config=dict(d["config"]),
            config_hash=d["config_hash"],
            signals_meta=dict(d["signals_meta"]),
            signals_data_path=d["signals_data_path"],
            vad_intervals=[list(x) for x in d.get("vad_intervals", [])],
            blocks=list(d.get("blocks", [])),
            events=list(d["events"]),
            labels=dict(d.get("labels", {})),
            created_at=d.get("created_at", ""),
            last_updated_at=d.get("last_updated_at", ""),
            event_schema=d.get("event_schema", "affective_events.v2"),
            notes=d.get("notes", "") or "",
        )
