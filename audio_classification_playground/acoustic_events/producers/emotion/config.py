"""Configuration for categorical emotion event extraction.

This producer is calibrated for decisive emotion2vec-style probability outputs:
rows should already be normalized probabilities, often with a clear top class.
It is not intended for softer categorical logits without retuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


CANONICAL_CHANNELS = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "other",
    "sadness",
    "surprise",
)

DEFAULT_EVENT_LABELS = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
)

DEFAULT_BACKGROUND_LABELS = ("neutral", "other")


@dataclass(frozen=True)
class Config:
    """Extraction thresholds and structural support parameters."""

    absolute_min_probability: float = 0.60
    class_quantile: float = 0.90
    background_margin: float = 0.15
    min_duration_sec: float = 1.0
    support_close_gap_sec: float = 1.0
    probability_sum_tolerance: float = 1e-3
    event_labels: tuple[str, ...] = field(default_factory=lambda: DEFAULT_EVENT_LABELS)
    background_labels: tuple[str, ...] = field(default_factory=lambda: DEFAULT_BACKGROUND_LABELS)

    def __post_init__(self) -> None:
        event_labels = tuple(str(label).lower() for label in self.event_labels)
        background_labels = tuple(str(label).lower() for label in self.background_labels)
        object.__setattr__(self, "event_labels", event_labels)
        object.__setattr__(self, "background_labels", background_labels)

        unknown = sorted((set(event_labels) | set(background_labels)) - set(CANONICAL_CHANNELS))
        if unknown:
            raise ValueError(f"Unknown emotion labels in config: {unknown}")
        overlap = sorted(set(event_labels) & set(background_labels))
        if overlap:
            raise ValueError(f"Event labels cannot also be background labels: {overlap}")
        if not 0.0 <= self.class_quantile <= 1.0:
            raise ValueError("class_quantile must be between 0 and 1")
        if self.absolute_min_probability < 0.0 or self.absolute_min_probability > 1.0:
            raise ValueError("absolute_min_probability must be between 0 and 1")
        if self.background_margin < 0.0:
            raise ValueError("background_margin must be non-negative")
        if self.min_duration_sec <= 0.0:
            raise ValueError("min_duration_sec must be positive")
        if self.support_close_gap_sec < 0.0:
            raise ValueError("support_close_gap_sec must be non-negative")
        if self.probability_sum_tolerance <= 0.0:
            raise ValueError("probability_sum_tolerance must be positive")

    @classmethod
    def balanced(cls) -> "Config":
        return cls()

    @classmethod
    def exploratory(cls) -> "Config":
        return replace(
            cls(),
            absolute_min_probability=0.50,
            class_quantile=0.85,
            background_margin=0.10,
            min_duration_sec=0.75,
        )

    @classmethod
    def conservative(cls) -> "Config":
        return replace(
            cls(),
            absolute_min_probability=0.70,
            class_quantile=0.95,
            background_margin=0.25,
            min_duration_sec=1.5,
        )


__all__ = [
    "CANONICAL_CHANNELS",
    "DEFAULT_BACKGROUND_LABELS",
    "DEFAULT_EVENT_LABELS",
    "Config",
]
