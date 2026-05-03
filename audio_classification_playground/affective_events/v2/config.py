"""Configuration for the v2 affective-events detector."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping


ScalarOrPerSignal = float | Mapping[str, float]


@dataclass(frozen=True)
class Config:
    # --- Pre-processing ------------------------------------------------------
    vad_merge_gap_sec: float = 0.5
    min_speech_block_sec: float = 0.75

    # --- Block-aware baseline and scale -------------------------------------
    radius_sec: float = 120.0
    min_context_sec: float = 5.0
    scale_floor_frac: float = 0.5

    # --- Prominence detection ------------------------------------------------
    z_seed: ScalarOrPerSignal = 1.75
    seed_min_width_sec: ScalarOrPerSignal = 1.0
    z_return: ScalarOrPerSignal = 0.5
    min_duration_sec: ScalarOrPerSignal = 2.5
    merge_gap_sec: ScalarOrPerSignal = 0.5

    # --- Cross-signal joint merge -------------------------------------------
    cross_signal_min_overlap_sec: float = 1.0
    signature_z_threshold: float = 0.5

    @classmethod
    def balanced(cls) -> "Config":
        return cls()

    @classmethod
    def exploratory(cls) -> "Config":
        return replace(cls(), z_seed=1.5)

    @classmethod
    def conservative(cls) -> "Config":
        return replace(cls(), z_seed=2.25)


def value_for_signal(value: ScalarOrPerSignal, signal_name: str) -> float:
    if isinstance(value, Mapping):
        if signal_name in value:
            return float(value[signal_name])
        if "_default" in value:
            return float(value["_default"])
        raise KeyError(
            f"No value configured for signal {signal_name!r}; provide it or '_default'"
        )
    return float(value)
