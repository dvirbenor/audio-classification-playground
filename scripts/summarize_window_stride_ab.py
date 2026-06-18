#!/usr/bin/env python3
"""Summarize results from the window/stride A/B experiment.

Reads results_run_*.json files written by ab_window_stride.py and prints
per-archive and aggregate tables.

Usage:
    uv run python scripts/summarize_window_stride_ab.py
    uv run python scripts/summarize_window_stride_ab.py --results-dir /efs/arno/experiments/ab-window-stride
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RESULTS_DIR = "/efs/arno/experiments/ab-window-stride"


def _load_results(results_dir: Path) -> list[dict]:
    files = sorted(results_dir.glob("results_run_*.json"))
    if not files:
        raise SystemExit(f"No results_run_*.json found in {results_dir}")
    results = []
    for f in files:
        data = json.loads(f.read_text())
        for entry in data["files"]:
            results.append(entry)
    print(f"Loaded {len(results)} archive(s) from {len(files)} file(s) in {results_dir}\n")
    return results


def _bar(value: float, lo: float = 0.0, hi: float = 1.0, width: int = 10) -> str:
    filled = int(round((value - lo) / (hi - lo) * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _print_variant_table(archives: list[dict]) -> None:
    variants = [v["name"] for v in archives[0]["variants"]]

    for task in ("affect", "disfluency"):
        print(f"  {'─' * 92}")
        print(f"  {task.upper()}")
        print(f"  {'─' * 92}")
        print(
            f"  {'variant':<12}  {'config':<18}  "
            f"{'recall':>6}  {'precis':>6}  {'label_ag':>8}  {'exact':>6}  "
            f"{'cnt_Δ%':>7}  {'bnd_p50':>7}  {'bnd_p99':>7}  {'bars (recall)':<12}"
        )
        print(f"  {'─' * 92}")

        for vname in variants:
            # Collect per-archive stats for this variant+task
            recalls, precisions, label_ags, exacts, deltas, p50s, p99s = [], [], [], [], [], [], []
            configs: list[str] = []
            for archive in archives:
                v = next(x for x in archive["variants"] if x["name"] == vname)
                d = v[task]
                recalls.append(d["recall"])
                precisions.append(d["precision"])
                label_ags.append(d["label_agreement"])
                exacts.append(d["exact_match_frac"])
                deltas.append(d["count_delta_pct"])
                p50s.append(d["boundary_drift_start_sec"]["p50"])
                p99s.append(d["boundary_drift_start_sec"]["p99"])
                w_key = "affect_window" if task == "affect" else "disf_window"
                h_key = "affect_hop" if task == "affect" else "disf_hop"
                configs.append(f"w={v[w_key]}s h={v[h_key]}s")

            config_str = configs[0]  # same across archives
            mean = lambda xs: sum(xs) / len(xs)

            r = mean(recalls)
            print(
                f"  {vname:<12}  {config_str:<18}  "
                f"{r:6.3f}  {mean(precisions):6.3f}  "
                f"{mean(label_ags):8.3f}  {mean(exacts):6.3f}  "
                f"{mean(deltas):+7.1f}  "
                f"{mean(p50s):7.3f}  {mean(p99s):7.3f}  "
                f"{_bar(r)}"
            )

        print()


def _print_per_archive(archives: list[dict]) -> None:
    for i, archive in enumerate(archives):
        label = Path(archive["audio"]).name
        print(f"Archive {i + 1}: {label}  ({archive['duration_sec']}s)")
        print(
            f"  baseline — affect: {archive['baseline_affect_events']} events  "
            f"disfluency: {archive['baseline_disf_events']} events"
        )

        for task in ("affect", "disfluency"):
            print(f"  {'variant':<12}  {'recall':>6}  {'precis':>6}  {'label_ag':>8}  {'exact':>6}  {'cnt_Δ%':>7}  {'bnd_p50':>7}  {'bnd_p99':>7}")
            print(f"  {'─' * 74}")
            for v in archive["variants"]:
                d = v[task]
                print(
                    f"  {v['name']:<12}  "
                    f"{d['recall']:6.3f}  {d['precision']:6.3f}  "
                    f"{d['label_agreement']:8.3f}  {d['exact_match_frac']:6.3f}  "
                    f"{d['count_delta_pct']:+7.1f}  "
                    f"{d['boundary_drift_start_sec']['p50']:7.3f}  "
                    f"{d['boundary_drift_start_sec']['p99']:7.3f}"
                )
            print()
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                    help="Directory containing results_run_*.json files.")
    ap.add_argument("--per-archive", action="store_true",
                    help="Also print per-archive breakdown.")
    args = ap.parse_args()

    archives = _load_results(Path(args.results_dir))

    print("=" * 94)
    print("AGGREGATE  (mean across archives)")
    print("=" * 94)
    _print_variant_table(archives)

    if args.per_archive:
        print("=" * 94)
        print("PER ARCHIVE")
        print("=" * 94)
        print()
        _print_per_archive(archives)


if __name__ == "__main__":
    main()
