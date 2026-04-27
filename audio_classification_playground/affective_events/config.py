"""Pipeline configuration.

The defaults reflect a 0.25 s hop / 3.5 s window emotion-dimension model
operating on speech recordings of ~hours in length. Three named presets
(``balanced``, ``exploratory``, ``conservative``) trade off recall and
precision; per-knob tuning is available but should rarely be needed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    # --- VAD merging into analysis blocks -----------------------------------
    vad_merge_gap_sec: float = 0.5
    min_speech_block_sec: float = 0.75

    # --- Per-frame quality (receptive-field speech coverage) ----------------
    usable_speech_coverage: float = 0.60
    core_speech_coverage: float = 0.80
    boundary_margin_sec: float = 1.5

    # --- Smoothing inside blocks --------------------------------------------
    smooth_median_sec: float = 1.0  # 0 disables

    # --- Robust normalization & local baseline ------------------------------
    scale_floor_frac: float = 0.5  # local scale floor as a fraction of global scale
    local_context_radius_sec: float = 90.0
    exclude_candidate_radius_sec: float = 8.0
    min_context_speech_sec: float = 20.0

    # --- Detector: block deviation ------------------------------------------
    min_block_for_deviation_sec: float = 3.5
    block_deviation_z_threshold: float = 1.75

    # --- Detector: within-block excursion -----------------------------------
    min_block_for_excursion_sec: float = 8.0
    excursion_enter_z: float = 2.0
    excursion_exit_z: float = 1.0
    excursion_min_duration_sec: float = 3.0
    excursion_merge_gap_sec: float = 1.5
    excursion_to_block_ratio: float = 0.7  # demote to block deviation above this

    # --- Detector: within-block regime shift --------------------------------
    min_block_for_regime_shift_sec: float = 20.0
    regime_shift_min_pre_sec: float = 6.0
    regime_shift_min_post_sec: float = 6.0
    regime_shift_min_effect_z: float = 1.25
    regime_shift_edge_margin_sec: float = 2.0

    # --- Detector: within-block ramp ----------------------------------------
    min_block_for_ramp_sec: float = 20.0
    ramp_min_duration_sec: float = 12.0
    ramp_min_total_change_z: float = 1.25
    ramp_min_monotonicity: float = 0.65

    # --- Detector: short-gap block transition -------------------------------
    short_gap_max_sec: float = 3.0
    short_gap_min_delta_z: float = 1.5

    # --- Per-signal episode aggregation -------------------------------------
    enable_episode_aggregation: bool = True
    episode_max_inter_block_gap_sec: float = 15.0
    episode_min_children: int = 2

    # --- Cross-signal joint merge -------------------------------------------
    enable_cross_signal_merge: bool = True
    cross_signal_min_overlap_sec: float = 1.0

    # --- Review window padding ----------------------------------------------
    review_pad_sec: float = 8.0

    # --- Confidence weighting (additive) ------------------------------------
    confidence_weights: tuple[tuple[str, float], ...] = (
        ("strength", 1.0),
        ("duration", 0.6),
        ("coverage", 0.5),
        ("context", 0.6),
        ("shape", 0.5),
        ("boundary", 0.4),
    )

    # ------------------------------------------------------------------ presets

    @classmethod
    def balanced(cls) -> "Config":
        return cls()

    @classmethod
    def exploratory(cls) -> "Config":
        return replace(
            cls(),
            block_deviation_z_threshold=1.4,
            excursion_enter_z=1.7,
            excursion_exit_z=0.8,
            regime_shift_min_effect_z=1.0,
            ramp_min_total_change_z=1.0,
            short_gap_min_delta_z=1.2,
        )

    @classmethod
    def conservative(cls) -> "Config":
        return replace(
            cls(),
            block_deviation_z_threshold=2.1,
            excursion_enter_z=2.4,
            excursion_exit_z=1.2,
            regime_shift_min_effect_z=1.6,
            ramp_min_total_change_z=1.6,
            short_gap_min_delta_z=1.9,
        )
