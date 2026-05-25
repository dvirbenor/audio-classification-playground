"""Pydantic models for the session event store.

These define the typed contract for the JSON stored in the parquet ``data``
column.  Consumers can validate with ``SessionEventsRecord.model_validate()``
or simply ``json.loads()`` for untyped dict access.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ThinEvent(BaseModel):
    """A single acoustic event with only semantically meaningful fields."""

    model_config = ConfigDict(extra="forbid")

    start_sec: float
    end_sec: float
    duration_sec: float
    task: str
    label: str
    score: float
    score_name: str
    direction: str | None = None
    axis: str | None = None
    labels: list[str] | None = None


class ArchiveEvents(BaseModel):
    """Events for one archive within a session."""

    model_config = ConfigDict(extra="forbid")

    archive_id: str
    session_id: str
    events_data: list[ThinEvent]


class SessionEventsRecord(BaseModel):
    """Complete session record stored as JSON in the parquet data column."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    date: str
    event_items: list[ArchiveEvents]
