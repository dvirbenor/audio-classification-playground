"""Emotional event extraction from continuous A/V/D speech predictions.

Quick start::

    from audio_classification_playground.affective_events import (
        extract_events, to_dataframe, Config, Signal, Vad,
    )

    events = extract_events(
        signals=[
            Signal("arousal",   arousal,   hop_sec=0.25, window_sec=3.5),
            Signal("valence",   valence,   hop_sec=0.25, window_sec=3.5),
            Signal("dominance", dominance, hop_sec=0.25, window_sec=3.5),
        ],
        vad=Vad.from_silero(silero_timestamps, sample_rate=16_000),
        config=Config.balanced(),
    )
    df = to_dataframe(events)
"""
from .config import Config
from .pipeline import extract_events, to_dataframe
from .types import Block, Event, Signal, Vad

__all__ = [
    "Config",
    "Signal",
    "Vad",
    "Block",
    "Event",
    "extract_events",
    "to_dataframe",
]
