#!/usr/bin/env python3
"""Compare two emotion inference artifacts, optionally through event extraction."""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from audio_classification_playground.acoustic_events.composition.composer import (
    compose_emotion_from_artifacts,
)
from audio_classification_playground.acoustic_events.inference import (
    artifact_to_emotion_probabilities,
    load_prediction_artifact,
)


def top2_margins(scores: np.ndarray) -> np.ndarray:
    if scores.shape[1] < 2:
        return np.full(scores.shape[0], np.inf, dtype=np.float32)
    top2 = np.partition(scores, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def event_core(event) -> tuple:
    return (
        event.task,
        event.event_type,
        event.label,
        round(float(event.start_sec), 6),
        round(float(event.end_sec), 6),
    )


def compare_probabilities(reference_path: str, candidate_path: str, *, diagnostic_rows: int) -> None:
    reference = load_prediction_artifact(reference_path)
    candidate = load_prediction_artifact(candidate_path)
    ref_probs, ref_labels, ref_hop, ref_window, ref_duration = artifact_to_emotion_probabilities(reference)
    cand_probs, cand_labels, cand_hop, cand_window, cand_duration = artifact_to_emotion_probabilities(candidate)

    if ref_probs.shape != cand_probs.shape:
        raise ValueError(f"shape mismatch: reference={ref_probs.shape}, candidate={cand_probs.shape}")
    if ref_labels != cand_labels:
        raise ValueError(f"label mismatch: reference={ref_labels}, candidate={cand_labels}")
    if (ref_hop, ref_window) != (cand_hop, cand_window):
        raise ValueError("timing mismatch between emotion artifacts")
    if abs(float(ref_duration) - float(cand_duration)) > 1e-6:
        raise ValueError("audio duration mismatch between emotion artifacts")

    diff = np.abs(ref_probs - cand_probs)
    ref_top = np.argmax(ref_probs, axis=1)
    cand_top = np.argmax(cand_probs, axis=1)
    top1_match = ref_top == cand_top
    flip_rows = np.flatnonzero(~top1_match)

    print("=== probability comparison ===")
    print(f"reference_path: {reference.path}")
    print(f"candidate_path: {candidate.path}")
    print(f"reference_config_hash: {reference.manifest.get('inference_config_hash')}")
    print(f"candidate_config_hash: {candidate.manifest.get('inference_config_hash')}")
    print(f"reference_runtime_config: {_runtime_config_summary(reference.manifest)}")
    print(f"candidate_runtime_config: {_runtime_config_summary(candidate.manifest)}")
    print(f"labels_equal: {ref_labels == cand_labels}")
    print(f"shape: {ref_probs.shape}")
    print(f"max_abs_diff: {float(diff.max()):.10g}")
    print(f"mean_abs_diff: {float(diff.mean()):.10g}")
    print(f"p99_abs_diff: {float(np.quantile(diff, 0.99)):.10g}")
    print(f"top1_agreement: {float(np.mean(top1_match)):.6f}")
    print(f"top1_flip_count: {int(len(flip_rows))}")

    if len(flip_rows):
        ref_margins = top2_margins(ref_probs)
        cand_margins = top2_margins(cand_probs)
        print(f"top1_flip_reference_margin_min: {float(ref_margins[flip_rows].min()):.10g}")
        print(f"top1_flip_reference_margin_median: {float(np.median(ref_margins[flip_rows])):.10g}")
        print(f"top1_flip_candidate_margin_median: {float(np.median(cand_margins[flip_rows])):.10g}")
        for row in flip_rows[:max(0, diagnostic_rows)]:
            ref_i = int(ref_top[row])
            cand_i = int(cand_top[row])
            print(
                "top1_flip: "
                f"row={int(row)} "
                f"time_sec={row * ref_hop:.2f} "
                f"reference={ref_labels[ref_i]!r}:{ref_probs[row, ref_i]:.8f} "
                f"candidate={cand_labels[cand_i]!r}:{cand_probs[row, cand_i]:.8f} "
                f"reference_margin={ref_margins[row]:.8f} "
                f"row_max_abs_diff={diff[row].max():.8f}"
            )


def _runtime_config_summary(manifest: dict) -> dict:
    config = manifest.get("inference_config", {})
    return {
        key: config.get(key)
        for key in (
            "batch_size",
            "torch_allow_tf32",
            "torch_autocast_dtype",
            "torch_compile",
            "torch_compile_mode",
        )
        if key in config
    }


def compare_events(reference_path: str, candidate_path: str, vad_path: str) -> None:
    _, _, reference_events = compose_emotion_from_artifacts(
        emotion_artifact=reference_path,
        vad_artifact=vad_path,
    )
    _, _, candidate_events = compose_emotion_from_artifacts(
        emotion_artifact=candidate_path,
        vad_artifact=vad_path,
    )

    ref_counter = Counter(event_core(event) for event in reference_events)
    cand_counter = Counter(event_core(event) for event in candidate_events)
    missing = ref_counter - cand_counter
    added = cand_counter - ref_counter

    print("\n=== event comparison ===")
    print(f"reference_event_count: {len(reference_events)}")
    print(f"candidate_event_count: {len(candidate_events)}")
    print(f"event_core_equal: {not missing and not added}")
    print(f"missing_event_core_count: {sum(missing.values())}")
    print(f"added_event_core_count: {sum(added.values())}")

    common = set(ref_counter) & set(cand_counter)
    ref_by_core = {event_core(event): event for event in reference_events}
    cand_by_core = {event_core(event): event for event in candidate_events}
    if common:
        score_diffs = [
            abs(float(ref_by_core[core].score) - float(cand_by_core[core].score))
            for core in common
        ]
        print(f"matched_event_score_max_abs_diff: {max(score_diffs):.10g}")
        print(f"matched_event_score_mean_abs_diff: {float(np.mean(score_diffs)):.10g}")
    for label, counter in (("missing", missing), ("added", added)):
        for core, count in list(counter.items())[:10]:
            print(f"{label}_event: count={count} core={core}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-emotion", required=True)
    parser.add_argument("--candidate-emotion", required=True)
    parser.add_argument("--vad-artifact")
    parser.add_argument("--diagnostic-rows", type=int, default=10)
    args = parser.parse_args()

    compare_probabilities(
        args.reference_emotion,
        args.candidate_emotion,
        diagnostic_rows=args.diagnostic_rows,
    )
    if args.vad_artifact:
        compare_events(args.reference_emotion, args.candidate_emotion, args.vad_artifact)


if __name__ == "__main__":
    main()
