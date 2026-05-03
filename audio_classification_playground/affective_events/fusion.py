"""Compatibility wrapper for canonical affect cross-signal fusion."""
from audio_classification_playground.acoustic_events.producers.affect.fusion import (
    attach_parent_ids,
    merge_cross_signal,
)

__all__ = ["attach_parent_ids", "merge_cross_signal"]
