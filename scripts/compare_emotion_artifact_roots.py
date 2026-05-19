#!/usr/bin/env python3
"""Compare emotion artifacts for matching archives under two flat output roots."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from audio_classification_playground.acoustic_events.composition.composer import (
    compose_emotion_from_artifacts,
)
from audio_classification_playground.acoustic_events.inference import (
    artifact_to_emotion_probabilities,
    load_prediction_artifact,
)


def iter_emotion_artifacts(root: Path) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for path in root.glob("*/*/emotion"):
        if not (path / "manifest.json").is_file() or not (path / "predictions.npz").is_file():
            continue
        archive_id = path.parent.name
        session_id = path.parent.parent.name
        out[(session_id, archive_id)] = path
    return out


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


def compare_probability_artifacts(reference_path: Path, candidate_path: Path) -> dict[str, float | int]:
    reference = load_prediction_artifact(reference_path)
    candidate = load_prediction_artifact(candidate_path)
    ref_probs, ref_labels, ref_hop, ref_window, _ = artifact_to_emotion_probabilities(reference)
    cand_probs, cand_labels, cand_hop, cand_window, _ = artifact_to_emotion_probabilities(candidate)
    if ref_probs.shape != cand_probs.shape:
        raise ValueError(f"shape mismatch: {reference_path} vs {candidate_path}")
    if ref_labels != cand_labels or (ref_hop, ref_window) != (cand_hop, cand_window):
        raise ValueError(f"label/timing mismatch: {reference_path} vs {candidate_path}")

    diff = np.abs(ref_probs - cand_probs)
    ref_top = np.argmax(ref_probs, axis=1)
    cand_top = np.argmax(cand_probs, axis=1)
    flip_rows = np.flatnonzero(ref_top != cand_top)
    margins = top2_margins(ref_probs)
    return {
        "frames": int(ref_probs.shape[0]),
        "max_abs_diff": float(diff.max()),
        "p99_abs_diff": float(np.quantile(diff, 0.99)),
        "top1_flip_count": int(len(flip_rows)),
        "top1_flip_margin_min": float(margins[flip_rows].min()) if len(flip_rows) else 0.0,
    }


def compare_event_artifacts(reference_path: Path, candidate_path: Path, vad_path: Path) -> dict[str, int | float | bool]:
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
    common = set(ref_counter) & set(cand_counter)
    ref_by_core = {event_core(event): event for event in reference_events}
    cand_by_core = {event_core(event): event for event in candidate_events}
    score_diffs = [
        abs(float(ref_by_core[core].score) - float(cand_by_core[core].score))
        for core in common
    ]
    return {
        "reference_event_count": len(reference_events),
        "candidate_event_count": len(candidate_events),
        "event_core_equal": not missing and not added,
        "missing_event_core_count": int(sum(missing.values())),
        "added_event_core_count": int(sum(added.values())),
        "matched_event_score_max_abs_diff": max(score_diffs) if score_diffs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument(
        "--vad-root",
        type=Path,
        default=None,
        help="Root containing matching vad artifacts; defaults to --reference-root.",
    )
    parser.add_argument("--max-mismatches", type=int, default=10)
    args = parser.parse_args()

    reference = iter_emotion_artifacts(args.reference_root)
    candidate = iter_emotion_artifacts(args.candidate_root)
    common_keys = sorted(set(reference) & set(candidate))
    missing_candidate = sorted(set(reference) - set(candidate))
    added_candidate = sorted(set(candidate) - set(reference))
    vad_root = args.vad_root or args.reference_root

    totals = Counter()
    total_frames = 0
    total_flips = 0
    max_abs_diff = 0.0
    p99_values: list[float] = []
    event_mismatches: list[tuple[tuple[str, str], dict]] = []
    missing_vad: list[tuple[str, str]] = []

    for key in common_keys:
        prob = compare_probability_artifacts(reference[key], candidate[key])
        total_frames += int(prob["frames"])
        total_flips += int(prob["top1_flip_count"])
        max_abs_diff = max(max_abs_diff, float(prob["max_abs_diff"]))
        p99_values.append(float(prob["p99_abs_diff"]))

        vad_path = vad_root / key[0] / key[1] / "vad"
        if not (vad_path / "manifest.json").is_file():
            missing_vad.append(key)
            continue
        event = compare_event_artifacts(reference[key], candidate[key], vad_path)
        totals["reference_events"] += int(event["reference_event_count"])
        totals["candidate_events"] += int(event["candidate_event_count"])
        totals["missing_events"] += int(event["missing_event_core_count"])
        totals["added_events"] += int(event["added_event_core_count"])
        if not event["event_core_equal"]:
            event_mismatches.append((key, event))

    print("=== root comparison ===")
    print(f"reference_root: {args.reference_root}")
    print(f"candidate_root: {args.candidate_root}")
    print(f"reference_emotion_artifacts: {len(reference)}")
    print(f"candidate_emotion_artifacts: {len(candidate)}")
    print(f"common_archives: {len(common_keys)}")
    print(f"missing_candidate_archives: {len(missing_candidate)}")
    print(f"added_candidate_archives: {len(added_candidate)}")
    print(f"missing_vad_archives: {len(missing_vad)}")
    print(f"total_frames: {total_frames}")
    print(f"total_top1_flip_count: {total_flips}")
    print(f"max_abs_diff: {max_abs_diff:.10g}")
    print(f"max_archive_p99_abs_diff: {(max(p99_values) if p99_values else 0.0):.10g}")
    print(f"reference_event_count: {totals['reference_events']}")
    print(f"candidate_event_count: {totals['candidate_events']}")
    print(f"missing_event_core_count: {totals['missing_events']}")
    print(f"added_event_core_count: {totals['added_events']}")
    print(f"event_mismatch_archives: {len(event_mismatches)}")
    print(
        "PASS_EVENT_CORE_EQUAL_FOR_COMMON_WITH_VAD: "
        f"{len(event_mismatches) == 0 and not missing_vad}"
    )
    for key, event in event_mismatches[:max(0, args.max_mismatches)]:
        print(f"event_mismatch: archive={key} details={event}")
    for key in missing_vad[:max(0, args.max_mismatches)]:
        print(f"missing_vad: archive={key}")


if __name__ == "__main__":
    main()
