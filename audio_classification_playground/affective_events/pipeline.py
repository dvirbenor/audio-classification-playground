"""Canonical affective-events extraction pipeline."""
from .v2.pipeline import (
    DEFAULT_PRODUCER_ID,
    extract_events,
    extract_events_with_tracks,
    producer_run,
    to_dataframe,
    tracks_from_signals,
)

__all__ = [
    "DEFAULT_PRODUCER_ID",
    "extract_events",
    "extract_events_with_tracks",
    "tracks_from_signals",
    "producer_run",
    "to_dataframe",
]
