"""Parquet schema, event thinning, and fingerprint utilities."""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

import pyarrow as pa

from .models import ArchiveEvents, SessionEventsRecord, ThinEvent

PARQUET_SCHEMA = pa.schema([
    ("session_id", pa.string()),
    ("date", pa.string()),
    ("archive_count", pa.int32()),
    ("event_count", pa.int32()),
    ("session_fingerprint", pa.string()),
    ("data", pa.string()),
])


def thin_event(raw: dict) -> ThinEvent:
    """Extract only semantically meaningful fields from a raw event dict."""
    fields: dict = {
        "start_sec": float(raw["start_sec"]),
        "end_sec": float(raw["end_sec"]),
        "duration_sec": float(raw["duration_sec"]),
        "task": raw["task"],
        "label": raw["label"],
        "score": float(raw["score"]),
        "score_name": raw["score_name"],
    }
    direction = raw.get("direction")
    if direction:
        fields["direction"] = direction
    axis = raw.get("axis")
    if axis:
        fields["axis"] = axis
    labels = raw.get("labels")
    if labels and len(labels) > 1:
        fields["labels"] = list(labels)
    return ThinEvent(**fields)


def build_session_record(
    *,
    session_id: str,
    date: str,
    archive_events: Sequence[tuple[str, list[dict]]],
) -> SessionEventsRecord:
    """Build a typed session record from loaded archive event lists.

    Parameters
    ----------
    session_id:
        The session identifier.
    date:
        The partition date for this record.
    archive_events:
        Sequence of ``(archive_id, raw_events)`` pairs where ``raw_events``
        is the list of dicts loaded from ``events.jsonl``.
    """
    items: list[ArchiveEvents] = []
    for archive_id, raw_events in sorted(archive_events, key=lambda x: x[0]):
        thinned = sorted(
            (thin_event(e) for e in raw_events),
            key=lambda ev: (ev.start_sec, ev.label),
        )
        items.append(ArchiveEvents(
            archive_id=archive_id,
            session_id=session_id,
            events_data=thinned,
        ))
    return SessionEventsRecord(session_id=session_id, date=date, event_items=items)


def session_fingerprint(archive_fingerprints: Sequence[tuple[str, str]]) -> str:
    """Deterministic hash of sorted (archive_id, event_package_fingerprint) pairs."""
    payload = sorted(archive_fingerprints)
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
