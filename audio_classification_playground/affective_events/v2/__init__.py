"""Canonical affective-events detector internals.

Public callers should import from ``audio_classification_playground.affective_events``.
The ``v2`` package name is retained only as an implementation location during
the migration from the legacy detector.
"""
from .config import Config
from .pipeline import (
    extract_events,
    extract_events_with_tracks,
    producer_run,
    to_dataframe,
    tracks_from_signals,
)
from .types import Block, Event, ProducerRun, RegularGridTrack, Signal, Vad

__all__ = [
    "Block",
    "Config",
    "Event",
    "ProducerRun",
    "RegularGridTrack",
    "Signal",
    "Vad",
    "extract_events",
    "extract_events_with_tracks",
    "producer_run",
    "tracks_from_signals",
    "to_dataframe",
]
