"""v2 affective-events detector.

This package implements the methodology in ``../METHODOLOGY.md`` alongside
the existing v1 package so callers can compare outputs before migration.
"""
from .config import Config
from .pipeline import extract_events, to_dataframe
from .types import Block, Event, Signal, Vad

__all__ = [
    "Block",
    "Config",
    "Event",
    "Signal",
    "Vad",
    "extract_events",
    "to_dataframe",
]
