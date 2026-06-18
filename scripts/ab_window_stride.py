#!/usr/bin/env python3
"""A/B test varying window sizes and hop strides for affect and disfluency.

Baseline is the current production config:
  affect:     window=3.5s, hop=0.25s
  disfluency: window=3.0s, hop=0.25s

Five named variants are tested against that baseline per archive. The baseline
itself is computed once and reused for all comparisons. Results are expressed
as event-level accuracy metrics relative to baseline (no external ground truth).

Usage (local audio files, e.g. WAVs retrieved with retrieve_benchmark_audio.py):
  uv run python scripts/ab_window_stride.py \\
      --audio a.wav b.wav c.wav \\
      --device cuda \\
      --batch-size 128 \\
      --json-out results/ab_window_stride.json

Modeled after scripts/event_level_ab.py.
"""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from audio_classification_playground.acoustic_events.inference.artifacts import SAMPLE_RATE
from audio_classification_playground.acoustic_events.inference.audio import frame_audio, load_audio
from audio_classification_playground.acoustic_events.inference.models import (
    AffectPredictor,
    DisfluencyPredictor,
)
from audio_classification_playground.acoustic_events.producers.affect import (
    Config as AffectConfig,
    Signal,
    Vad,
    extract_events as affect_extract_events,
)
from audio_classification_playground.acoustic_events.producers.disfluency.pipeline import (
    produce_disfluency_events,
)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowConfig:
    name: str
    affect_window: float
    affect_hop: float
    disf_window: float
    disf_hop: float


BASELINE = WindowConfig(
    name="baseline",
    affect_window=3.5,
    affect_hop=0.25,
    disf_window=3.0,
    disf_hop=0.25,
)

VARIANTS: list[WindowConfig] = [
    WindowConfig("narrow",      affect_window=2.5, affect_hop=0.25, disf_window=2.0, disf_hop=0.25),
    WindowConfig("wide",        affect_window=5.0, affect_hop=0.25, disf_window=4.0, disf_hop=0.25),
    WindowConfig("coarse",      affect_window=3.5, affect_hop=0.50, disf_window=3.0, disf_hop=0.50),
    WindowConfig("hop_1s",      affect_window=3.5, affect_hop=1.00, disf_window=3.0, disf_hop=1.00),
    WindowConfig("narrow_fast", affect_window=2.5, affect_hop=0.50, disf_window=2.0, disf_hop=0.50),
]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _affect_arrays(windows: np.ndarray, device: str, bs: int) -> dict:
    p = AffectPredictor("wavlm", device=device, batch_size=bs)
    p(windows[:bs])  # warmup
    out = p(windows)
    del p
    _free()
    return out


def _disf_arrays(windows: np.ndarray, device: str, bs: int) -> dict:
    p = DisfluencyPredictor("wavlm", device=device, batch_size=bs)
    p(windows[:bs])  # warmup
    out = p(windows)
    del p
    _free()
    return out


def _affect_events(arrays: dict, duration_sec: float, window_sec: float, hop_sec: float) -> list:
    signals = [
        Signal(
            name=name,
            values=np.asarray(arrays[name], dtype=np.float32),
            hop_sec=hop_sec,
            window_sec=window_sec,
        )
        for name in ("arousal", "valence", "dominance")
    ]
    vad = Vad(intervals=((0.0, float(duration_sec)),))
    return affect_extract_events(signals, vad, AffectConfig.balanced())


def _disf_events(arrays: dict, duration_sec: float, window_sec: float, hop_sec: float) -> list:
    _, _, events = produce_disfluency_events(
        fluency_logits=np.asarray(arrays["fluency_logits"], dtype=np.float32),
        disfluency_type_logits=np.asarray(arrays["disfluency_type_logits"], dtype=np.float32),
        hop_sec=hop_sec,
        window_sec=window_sec,
        audio_duration_sec=float(duration_sec),
        vad_intervals=[(0.0, float(duration_sec))],
    )
    return events


# ---------------------------------------------------------------------------
# Diff / metrics
# ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    lo = max(a.start_sec, b.start_sec)
    hi = min(a.end_sec, b.end_sec)
    inter = max(0.0, hi - lo)
    union = (a.end_sec - a.start_sec) + (b.end_sec - b.start_sec) - inter
    return inter / union if union > 0 else 0.0


def diff_events(base: list, cand: list, *, boundary_tol: float) -> dict:
    """Greedy IoU match within same event_type; report accuracy metrics vs base."""
    base_s = sorted(base, key=lambda e: e.start_sec)
    cand_s = sorted(cand, key=lambda e: e.start_sec)
    used = [False] * len(cand_s)
    matches = []
    unmatched_base = []
    for b in base_s:
        best_j, best_iou = -1, 0.0
        for j, c in enumerate(cand_s):
            if used[j] or c.event_type != b.event_type:
                continue
            iou = _iou(b, c)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            used[best_j] = True
            matches.append((b, cand_s[best_j]))
        else:
            unmatched_base.append(b)

    n_base, n_cand, n_match = len(base_s), len(cand_s), len(matches)
    dstart = [abs(b.start_sec - c.start_sec) for b, c in matches]
    dend = [abs(b.end_sec - c.end_sec) for b, c in matches]
    label_agree = [b.label == c.label for b, c in matches]
    exact = [
        b.label == c.label and ds <= boundary_tol and de <= boundary_tol
        for (b, c), ds, de in zip(matches, dstart, dend)
    ]

    def _stat(xs):
        if not xs:
            return {"p50": 0.0, "p99": 0.0, "max": 0.0}
        return {
            "p50": float(np.quantile(xs, 0.50)),
            "p99": float(np.quantile(xs, 0.99)),
            "max": float(np.max(xs)),
        }

    # Score-stratified metrics.
    # Baseline tiers (for recall): percentiles of baseline scores.
    # Candidate tiers (for precision): percentiles of candidate scores.
    # Both use the same tier names but independent thresholds.
    matched_base_ids = {id(b) for b, _ in matches}
    matched_cand_ids = {id(c) for _, c in matches}
    matched_set = {id(b): (b, c, ds, de) for (b, c), ds, de in zip(matches, dstart, dend)}

    score_tiers: dict = {}
    if base_s:
        base_scores = np.array([e.score for e in base_s], dtype=np.float64)
        bp25, bp75 = float(np.quantile(base_scores, 0.25)), float(np.quantile(base_scores, 0.75))
        for tier_name, tier_base in [
            ("high", [e for e in base_s if e.score >= bp75]),
            ("mid",  [e for e in base_s if bp25 <= e.score < bp75]),
            ("low",  [e for e in base_s if e.score < bp25]),
        ]:
            tier_matched = [matched_set[id(b)] for b in tier_base if id(b) in matched_base_ids]
            tier_dstart = [ds for _, _, ds, _ in tier_matched]
            tier_dend = [de for _, _, _, de in tier_matched]
            score_tiers[tier_name] = {
                "n_base": len(tier_base),
                "n_matched": len(tier_matched),
                "recall": round(len(tier_matched) / max(len(tier_base), 1), 3),
                "boundary_drift_start_sec": _stat(tier_dstart),
                "boundary_drift_end_sec": _stat(tier_dend),
            }

    # Precision by candidate score tier: of the events the variant fires at
    # high/mid/low confidence, what fraction are real (match a baseline event)?
    cand_score_tiers: dict = {}
    if cand_s:
        cand_scores = np.array([e.score for e in cand_s], dtype=np.float64)
        cp25, cp75 = float(np.quantile(cand_scores, 0.25)), float(np.quantile(cand_scores, 0.75))
        for tier_name, tier_cand in [
            ("high", [e for e in cand_s if e.score >= cp75]),
            ("mid",  [e for e in cand_s if cp25 <= e.score < cp75]),
            ("low",  [e for e in cand_s if e.score < cp25]),
        ]:
            tier_matched = [c for c in tier_cand if id(c) in matched_cand_ids]
            cand_score_tiers[tier_name] = {
                "n_cand": len(tier_cand),
                "n_matched": len(tier_matched),
                "precision": round(len(tier_matched) / max(len(tier_cand), 1), 3),
            }

    by_type_base: dict[str, int] = {}
    by_type_cand: dict[str, int] = {}
    for e in base_s:
        by_type_base[e.event_type] = by_type_base.get(e.event_type, 0) + 1
    for e in cand_s:
        by_type_cand[e.event_type] = by_type_cand.get(e.event_type, 0) + 1

    return {
        "n_base": n_base,
        "n_cand": n_cand,
        "n_matched": n_match,
        "n_dropped": n_base - n_match,
        "n_added": n_cand - n_match,
        "count_delta_pct": round((n_cand - n_base) / max(n_base, 1) * 100, 1),
        "recall": round(n_match / max(n_base, 1), 3),
        "precision": round(n_match / max(n_cand, 1), 3),
        "label_agreement": round(float(np.mean(label_agree)) if label_agree else 1.0, 3),
        "exact_match_frac": round(float(np.mean(exact)) if exact else 1.0, 3),
        "boundary_drift_start_sec": _stat(dstart),
        "boundary_drift_end_sec": _stat(dend),
        "score_tiers": score_tiers,
        "cand_score_tiers": cand_score_tiers,
        "count_by_type_base": by_type_base,
        "count_by_type_cand": by_type_cand,
    }


# ---------------------------------------------------------------------------
# Per-archive runner
# ---------------------------------------------------------------------------

def run_archive(
    audio_path: str,
    *,
    device: str,
    bs: int,
    max_seconds: float | None,
    boundary_tol: float,
) -> dict:
    """Compute baseline + all variants for one archive."""
    audio = load_audio(audio_path, sample_rate=SAMPLE_RATE)
    samples = audio.samples
    if max_seconds:
        samples = samples[: int(max_seconds * SAMPLE_RATE)]
    duration = len(samples) / SAMPLE_RATE

    all_configs = [BASELINE] + VARIANTS

    # Pre-frame all unique (window, hop) combos to avoid redundant work
    affect_frames: dict[tuple[float, float], np.ndarray] = {}
    disf_frames: dict[tuple[float, float], np.ndarray] = {}
    for cfg in all_configs:
        ak = (cfg.affect_window, cfg.affect_hop)
        dk = (cfg.disf_window, cfg.disf_hop)
        if ak not in affect_frames:
            affect_frames[ak] = frame_audio(samples, sample_rate=SAMPLE_RATE,
                                            window_sec=cfg.affect_window, hop_sec=cfg.affect_hop)
        if dk not in disf_frames:
            disf_frames[dk] = frame_audio(samples, sample_rate=SAMPLE_RATE,
                                          window_sec=cfg.disf_window, hop_sec=cfg.disf_hop)

    # Run inference for each unique framing (models loaded/unloaded per call to free VRAM)
    affect_arrays_cache: dict[tuple[float, float], dict] = {}
    for key, windows in affect_frames.items():
        print(f"    affect inference  window={key[0]}s hop={key[1]}s  frames={len(windows)}", flush=True)
        affect_arrays_cache[key] = _affect_arrays(windows, device, bs)

    disf_arrays_cache: dict[tuple[float, float], dict] = {}
    for key, windows in disf_frames.items():
        print(f"    disf  inference  window={key[0]}s hop={key[1]}s  frames={len(windows)}", flush=True)
        disf_arrays_cache[key] = _disf_arrays(windows, device, bs)

    # Derive events for all configs
    events: dict[str, tuple[list, list]] = {}  # name -> (affect_events, disf_events)
    for cfg in all_configs:
        ae = _affect_events(affect_arrays_cache[(cfg.affect_window, cfg.affect_hop)],
                            duration, cfg.affect_window, cfg.affect_hop)
        de = _disf_events(disf_arrays_cache[(cfg.disf_window, cfg.disf_hop)],
                          duration, cfg.disf_window, cfg.disf_hop)
        events[cfg.name] = (ae, de)

    base_ae, base_de = events["baseline"]

    variants_out = []
    for cfg in VARIANTS:
        cand_ae, cand_de = events[cfg.name]
        variants_out.append({
            "name": cfg.name,
            "affect_window": cfg.affect_window,
            "affect_hop": cfg.affect_hop,
            "disf_window": cfg.disf_window,
            "disf_hop": cfg.disf_hop,
            "affect_frames": int(len(affect_frames[(cfg.affect_window, cfg.affect_hop)])),
            "disf_frames": int(len(disf_frames[(cfg.disf_window, cfg.disf_hop)])),
            "affect": diff_events(base_ae, cand_ae, boundary_tol=boundary_tol),
            "disfluency": diff_events(base_de, cand_de, boundary_tol=boundary_tol),
        })

    return {
        "audio": audio_path,
        "duration_sec": round(duration, 1),
        "baseline_affect_events": len(base_ae),
        "baseline_disf_events": len(base_de),
        "variants": variants_out,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(r: dict) -> None:
    print(f"\n  {'Variant':<12}  {'Task':<11}  {'recall':>6}  {'precis':>6}  {'label_ag':>8}  "
          f"{'exact':>6}  {'cnt_Δ%':>7}  {'bnd_p50':>7}  {'bnd_p99':>7}", flush=True)
    print("  " + "-" * 88, flush=True)
    for v in r["variants"]:
        for task, key in [("affect", "affect"), ("disfluency", "disfluency")]:
            d = v[task]
            print(
                f"  {v['name']:<12}  {task:<11}  "
                f"{d['recall']:6.3f}  {d['precision']:6.3f}  "
                f"{d['label_agreement']:8.3f}  {d['exact_match_frac']:6.3f}  "
                f"{d['count_delta_pct']:+7.1f}  "
                f"{d['boundary_drift_start_sec']['p50']:7.3f}  "
                f"{d['boundary_drift_start_sec']['p99']:7.3f}",
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", nargs="+", required=True, metavar="FILE",
                    help="Audio files to run (WAV/MP3). 3 archives recommended for speed.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-seconds", type=float, default=600.0,
                    help="Cap each file at N seconds (0 = full).")
    ap.add_argument("--boundary-tol", type=float, default=0.25,
                    help="Boundary tolerance for exact-match (default=1 hop = 0.25s).")
    ap.add_argument("--json-out", metavar="FILE",
                    help="Write full JSON report to this path.")
    args = ap.parse_args()
    max_seconds = args.max_seconds or None

    report = {
        "baseline": {"affect_window": BASELINE.affect_window, "affect_hop": BASELINE.affect_hop,
                     "disf_window": BASELINE.disf_window, "disf_hop": BASELINE.disf_hop},
        "boundary_tol_sec": args.boundary_tol,
        "files": [],
    }

    for path in args.audio:
        print(f"\n### {path}  (max {max_seconds or 'full'}s)", flush=True)
        r = run_archive(path, device=args.device, bs=args.batch_size,
                        max_seconds=max_seconds, boundary_tol=args.boundary_tol)
        report["files"].append(r)
        print(f"  baseline: affect={r['baseline_affect_events']} events  "
              f"disf={r['baseline_disf_events']} events  "
              f"duration={r['duration_sec']}s", flush=True)
        _print_result(r)

    print("\n" + "=" * 92, flush=True)
    print("AGGREGATE (mean across archives)", flush=True)
    print("=" * 92, flush=True)

    # Aggregate means across files
    agg: dict[str, dict[str, dict[str, list]]] = {}
    for r in report["files"]:
        for v in r["variants"]:
            agg.setdefault(v["name"], {"affect": {}, "disfluency": {}})
            for task in ("affect", "disfluency"):
                d = v[task]
                for metric in ("recall", "precision", "label_agreement", "exact_match_frac", "count_delta_pct"):
                    agg[v["name"]][task].setdefault(metric, []).append(d[metric])

    print(f"\n  {'Variant':<12}  {'Task':<11}  {'recall':>6}  {'precis':>6}  "
          f"{'label_ag':>8}  {'exact':>6}  {'cnt_Δ%':>7}", flush=True)
    print("  " + "-" * 70, flush=True)
    for vname, tasks in agg.items():
        for task in ("affect", "disfluency"):
            m = tasks[task]
            print(
                f"  {vname:<12}  {task:<11}  "
                f"{np.mean(m['recall']):6.3f}  {np.mean(m['precision']):6.3f}  "
                f"{np.mean(m['label_agreement']):8.3f}  {np.mean(m['exact_match_frac']):6.3f}  "
                f"{np.mean(m['count_delta_pct']):+7.1f}",
                flush=True,
            )

    print(flush=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
