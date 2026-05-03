"""Canonical affective event extraction from continuous A/V/D predictions.

Quick start::

    from audio_classification_playground.affective_events import (
        extract_events, tracks_from_signals, to_dataframe, Config, Signal, Vad,
    )

    signals = [
        Signal("arousal",   arousal,   hop_sec=0.25, window_sec=3.5),
        Signal("valence",   valence,   hop_sec=0.25, window_sec=3.5),
        Signal("dominance", dominance, hop_sec=0.25, window_sec=3.5),
    ]
    events = extract_events(
        signals=signals,
        vad=Vad.from_silero(silero_timestamps, sample_rate=16_000),
        config=Config.balanced(),
    )
    tracks = tracks_from_signals(signals)
    df = to_dataframe(events)
"""
from .schema import Event, MarkerItem, MarkerTrack, ProducerRun, RegularGridTrack
from .v2.config import Config
from .v2.pipeline import (
    extract_events,
    extract_events_with_tracks,
    producer_run,
    to_dataframe,
    tracks_from_signals,
)
from .v2.types import Block, Signal, Vad

__all__ = [
    "Config",
    "Signal",
    "Vad",
    "Block",
    "Event",
    "ProducerRun",
    "RegularGridTrack",
    "MarkerTrack",
    "MarkerItem",
    "extract_events",
    "extract_events_with_tracks",
    "tracks_from_signals",
    "producer_run",
    "to_dataframe",
]
