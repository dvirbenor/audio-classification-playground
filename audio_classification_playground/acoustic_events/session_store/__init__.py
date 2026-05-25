"""Session-level event store: date-partitioned parquet aggregation.

Aggregates per-archive event packages into session-level rows stored in
date-partitioned parquet files.  Each row contains thinned events for all
archives of a complete session.
"""
from __future__ import annotations

from .models import ArchiveEvents, SessionEventsRecord, ThinEvent
from .populate import populate_session_store
from .schema import PARQUET_SCHEMA, build_session_record, session_fingerprint, thin_event

__all__ = [
    "PARQUET_SCHEMA",
    "ArchiveEvents",
    "SessionEventsRecord",
    "ThinEvent",
    "build_session_record",
    "populate_session_store",
    "session_fingerprint",
    "thin_event",
]
