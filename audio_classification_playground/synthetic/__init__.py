"""Synthetic audio generation module for multi-label audio classification testing."""

from .generator import generate_synthetic_audio, generate_test_audio, GenerationResult
from .audio_loader import load_audio_sources, AudioSources, AudioSample
from .timeline_generator import (
    Timeline, ActualTimeline,
    generate_timeline, generate_structured_timeline,
    validate_timeline
)
from .audio_mixer import mix_audio, render_info_to_actual_timeline, RenderInfo
from .label_generator import (
    generate_labels, generate_labels_from_actual,
    load_labels, save_labels
)
from .visualize_synthetic import (
    plot_labels,
    plot_labels_combined,
    plot_with_waveform,
    plot_comparison,
    visualize_from_files
)

__all__ = [
    # Main API
    "generate_synthetic_audio",
    "generate_test_audio",
    "GenerationResult",
    # Audio loading
    "load_audio_sources",
    "AudioSources",
    "AudioSample",
    # Timeline
    "Timeline",
    "ActualTimeline",
    "generate_timeline",
    "generate_structured_timeline",
    "validate_timeline",
    # Mixing
    "mix_audio",
    "render_info_to_actual_timeline",
    "RenderInfo",
    # Labels
    "generate_labels",
    "generate_labels_from_actual",
    "load_labels",
    "save_labels",
    # Visualization
    "plot_labels",
    "plot_labels_combined",
    "plot_with_waveform",
    "plot_comparison",
    "visualize_from_files",
]
