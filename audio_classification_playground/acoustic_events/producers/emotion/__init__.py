"""Categorical emotion event producer."""
from .config import CANONICAL_CHANNELS, Config
from .pipeline import (
    DEFAULT_PRODUCER_ID,
    TRACK_ID,
    canonicalize_probabilities,
    extract_events,
    normalize_label,
    producer_run,
    run_from_probabilities,
    to_dataframe,
    tracks_from_probabilities,
)

__all__ = [
    "CANONICAL_CHANNELS",
    "Config",
    "DEFAULT_PRODUCER_ID",
    "TRACK_ID",
    "canonicalize_probabilities",
    "extract_events",
    "normalize_label",
    "producer_run",
    "run_from_probabilities",
    "to_dataframe",
    "tracks_from_probabilities",
]
