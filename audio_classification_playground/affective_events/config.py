"""Compatibility wrapper for canonical affect producer configuration."""
from audio_classification_playground.acoustic_events.producers.affect.config import (
    Config,
    ScalarOrPerSignal,
    value_for_signal,
)

__all__ = ["Config", "ScalarOrPerSignal", "value_for_signal"]
