"""Configuration for Vox-Profile disfluency event extraction.

The default suppression of pure ``Sound Repetition`` regions is based on a
listening audit of conversational/podcast-like audio, where those high-score
regions were mostly laughter, background, or otherwise non-target audio. For
clinical or stuttering-focused work where sound repetitions are themselves a
target event, set ``suppressed_types=()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


FLUENCY_LABELS = ("fluent", "disfluent")
DISFLUENCY_TYPE_LABELS = (
    "Block",
    "Prolongation",
    "Sound Repetition",
    "Word Repetition",
    "Interjection",
)

LABEL_TO_EVENT_LABEL = {
    "Block": "block",
    "Prolongation": "prolongation",
    "Word Repetition": "word_repetition",
    "Interjection": "interjection",
    "disfluent": "disfluent",
}


@dataclass(frozen=True)
class DisfluencyConfig:
    """Thresholds and support policy for disfluency event extraction.

    ``min_support_sec`` is converted to frames at runtime from the caller's
    hop size using ``max(1, ceil(min_support_sec / hop_sec))``. At coarse hops
    where one hop is already longer than ``min_support_sec``, this resolves to
    one frame and does not remove single-frame regions; the pooled model window
    itself provides the temporal context in that case.
    """

    seed_threshold: float = 0.70
    shoulder_threshold: float = 0.50
    min_support_sec: float = 0.50
    merge_gap_sec: float = 0.50
    type_threshold: float = 0.70
    suppressed_types: tuple[str, ...] = field(default_factory=lambda: ("Sound Repetition",))
    emit_unspecified: bool = False

    def __post_init__(self) -> None:
        suppressed_types = tuple(str(label) for label in self.suppressed_types)
        object.__setattr__(self, "suppressed_types", suppressed_types)

        for name in ("seed_threshold", "shoulder_threshold", "type_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.shoulder_threshold > self.seed_threshold:
            raise ValueError("shoulder_threshold must be <= seed_threshold")
        if self.min_support_sec < 0.0:
            raise ValueError("min_support_sec must be non-negative")
        if self.merge_gap_sec < 0.0:
            raise ValueError("merge_gap_sec must be non-negative")

        unknown = sorted(set(suppressed_types) - set(DISFLUENCY_TYPE_LABELS))
        if unknown:
            raise ValueError(f"Unknown suppressed disfluency types: {unknown}")

    @classmethod
    def balanced(cls) -> "DisfluencyConfig":
        return cls()

    @classmethod
    def exploratory(cls) -> "DisfluencyConfig":
        return replace(cls(), seed_threshold=0.50, shoulder_threshold=0.40)

    @classmethod
    def conservative(cls) -> "DisfluencyConfig":
        return replace(cls(), seed_threshold=0.85, shoulder_threshold=0.65)


__all__ = [
    "DISFLUENCY_TYPE_LABELS",
    "FLUENCY_LABELS",
    "LABEL_TO_EVENT_LABEL",
    "DisfluencyConfig",
]
