"""Compatibility wrapper for canonical affect extraction pipeline."""
from audio_classification_playground.acoustic_events.producers.affect.pipeline import (
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
