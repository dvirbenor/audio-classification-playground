"""Compatibility wrapper for canonical affect pre-processing helpers."""
from audio_classification_playground.acoustic_events.producers.affect.preprocessing import (
    assign_frame_blocks,
    build_blocks,
    global_stats,
)

__all__ = ["assign_frame_blocks", "build_blocks", "global_stats"]
