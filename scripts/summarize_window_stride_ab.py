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


def _collect_variant(archives: list[dict], vname: str, task: str) -> tuple[dict, str]:
    """Aggregate per-archive dicts for a single variant+task. Returns (means, config_str)."""
    keys = ["recall", "precision", "label_agreement", "exact_match_frac", "count_delta_pct"]
    buckets: dict[str, list] = {k: [] for k in keys}
    bnd_p50, bnd_p99 = [], []
    tier_data: dict[str, dict[str, list]] = {"high": {}, "mid": {}, "low": {}}
    config_str = ""
    for archive in archives:
        v = next(x for x in archive["variants"] if x["name"] == vname)
        d = v[task]
        for k in keys:
            buckets[k].append(d[k])
        bnd_p50.append(d["boundary_drift_start_sec"]["p50"])
        bnd_p99.append(d["boundary_drift_start_sec"]["p99"])
        w_key = "affect_window" if task == "affect" else "disf_window"
        h_key = "affect_hop" if task == "affect" else "disf_hop"
        config_str = f"w={v[w_key]}s h={v[h_key]}s"
        for tier in ("high", "mid", "low"):
            if "score_tiers" not in d or tier not in d["score_tiers"]:
                continue
            td = d["score_tiers"][tier]
            tier_data[tier].setdefault("recall", []).append(td["recall"])
            tier_data[tier].setdefault("n_base", []).append(td["n_base"])
            tier_data[tier].setdefault("bnd_p50", []).append(td["boundary_drift_start_sec"]["p50"])
            tier_data[tier].setdefault("bnd_p99", []).append(td["boundary_drift_start_sec"]["p99"])
            if "cand_score_tiers" in d and tier in d["cand_score_tiers"]:
                ct = d["cand_score_tiers"][tier]
                tier_data[tier].setdefault("precision", []).append(ct["precision"])
                tier_data[tier].setdefault("n_cand", []).append(ct["n_cand"])

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    agg = {k: mean(v) for k, v in buckets.items()}
    agg["bnd_p50"] = mean(bnd_p50)
    agg["bnd_p99"] = mean(bnd_p99)
    agg["tiers"] = {
        tier: {
            "recall": mean(tier_data[tier].get("recall", [])),
            "n_base": mean(tier_data[tier].get("n_base", [])),
            "bnd_p50": mean(tier_data[tier].get("bnd_p50", [])),
            "bnd_p99": mean(tier_data[tier].get("bnd_p99", [])),
            "precision": mean(tier_data[tier].get("precision", [])),
            "n_cand": mean(tier_data[tier].get("n_cand", [])),
        }
        for tier in ("high", "mid", "low")
        if tier_data[tier]
    }
    return agg, config_str


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
            agg, config_str = _collect_variant(archives, vname, task)
            r = agg["recall"]
            print(
                f"  {vname:<12}  {config_str:<18}  "
                f"{r:6.3f}  {agg['precision']:6.3f}  "
                f"{agg['label_agreement']:8.3f}  {agg['exact_match_frac']:6.3f}  "
                f"{agg['count_delta_pct']:+7.1f}  "
                f"{agg['bnd_p50']:7.3f}  {agg['bnd_p99']:7.3f}  "
                f"{_bar(r)}"
            )

        print()

        # Score-tier breakdown: recall + boundary drift for high/mid/low confidence events
        has_tiers = any(
            "score_tiers" in v[task]
            for archive in archives
            for v in archive["variants"]
            if v["name"] == variants[0]
        )
        if not has_tiers:
            continue

        print(f"  {task.upper()} — by score tier  (high/mid/low = top 25% / middle 50% / bottom 25%)")
        print(f"  Recall = baseline tier found. Precision = candidate tier that's real.")
        print(f"  {'─' * 100}")
        print(
            f"  {'variant':<12}  {'tier':<5}  "
            f"{'n_base':>6}  {'recall':>6}  {'bnd_p50':>7}  {'bnd_p99':>7}  "
            f"{'n_cand':>6}  {'precis':>6}  {'bars (precis)':<12}"
        )
        print(f"  {'─' * 100}")
        for vname in variants:
            agg, _ = _collect_variant(archives, vname, task)
            for tier in ("high", "mid", "low"):
                if tier not in agg["tiers"]:
                    continue
                t = agg["tiers"][tier]
                p = t.get("precision", 0.0)
                print(
                    f"  {vname:<12}  {tier:<5}  "
                    f"{t['n_base']:6.1f}  {t['recall']:6.3f}  "
                    f"{t['bnd_p50']:7.3f}  {t['bnd_p99']:7.3f}  "
                    f"{t.get('n_cand', 0.0):6.1f}  {p:6.3f}  "
                    f"{_bar(p)}"
                )
            print()
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
