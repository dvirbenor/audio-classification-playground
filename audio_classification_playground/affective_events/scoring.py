"""Direction labels, strength helpers, and weighted-additive confidence."""
from __future__ import annotations

import math

from .config import Config


def direction_of(delta_z: float) -> str:
    return "+" if delta_z >= 0 else "-"


def strength_score(abs_z: float) -> float:
    """Map ``|z|`` to a smooth ``[0, 1]`` score (0.5 at z = 1.5)."""
    return 1.0 / (1.0 + math.exp(-1.5 * (abs_z - 1.5)))


def duration_score(duration_sec: float, model_window_sec: float) -> float:
    """Saturating score: low below the model's effective resolution, ~1 well above."""
    return max(0.0, min(1.0, duration_sec / (2.0 * model_window_sec)))


def coverage_score(mean_coverage: float) -> float:
    return max(0.0, min(1.0, (mean_coverage - 0.5) / 0.4))


def context_score(context_speech_sec: float, target_sec: float = 60.0) -> float:
    return max(0.0, min(1.0, context_speech_sec / target_sec))


def boundary_score(near_start: bool, near_end: bool) -> float:
    if near_start and near_end:
        return 0.5
    if near_start or near_end:
        return 0.75
    return 1.0


def combine_confidence(components: dict[str, float], config: Config) -> float:
    """Weighted average of named components in ``[0, 1]``.

    Components missing from ``confidence_weights`` are ignored. The result is
    clipped to ``[0, 1]`` so it can be displayed as-is.
    """
    weights = dict(config.confidence_weights)
    total_w = 0.0
    total = 0.0
    for name, w in weights.items():
        if name in components:
            total += w * components[name]
            total_w += w
    if total_w == 0.0:
        return 0.0
    return max(0.0, min(1.0, total / total_w))


def signature_for(deltas_z: dict[str, float], threshold: float = 0.5) -> str:
    """Compact label like ``A+ V- D0`` from a per-signal delta-z mapping.

    Signal names are reduced to their leading letter, capitalized; unknown
    names are kept verbatim. Only ``arousal``/``valence``/``dominance`` get
    their canonical short letters; this keeps signatures readable.
    """
    short = {"arousal": "A", "valence": "V", "dominance": "D"}
    parts: list[str] = []
    for name in ("arousal", "valence", "dominance"):
        if name in deltas_z:
            d = deltas_z[name]
            sym = "+" if d >= threshold else "-" if d <= -threshold else "0"
            parts.append(f"{short[name]}{sym}")
    for name, d in deltas_z.items():
        if name not in short:
            sym = "+" if d >= threshold else "-" if d <= -threshold else "0"
            parts.append(f"{name}{sym}")
    return " ".join(parts)
