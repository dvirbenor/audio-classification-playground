#!/usr/bin/env python3
"""Event-level A/B: does a precision/runtime change alter the events we emit?

Runs the real inference -> producer path at a baseline precision (fp32) and a
candidate autocast dtype (fp16/bf16) for the WavLM tasks (affect, disfluency),
then diffs the resulting *events* (count, type, start/end, score) -- the gate
the optimization plan calls "the one that actually matters". This measures
correctness, not speed; throughput is covered by the other scripts/ harnesses.

VAD is precision-independent, so the same coverage is fed to both sides; the
diff therefore isolates precision-induced event drift.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from audio_classification_playground.acoustic_events.inference.artifacts import SAMPLE_RATE
from audio_classification_playground.acoustic_events.inference.audio import frame_audio, load_audio
from audio_classification_playground.acoustic_events.inference.models import (
    AffectPredictor,
    DisfluencyPredictor,
)
from audio_classification_playground.acoustic_events.inference.runners import (
    AFFECT_WINDOW_SEC,
    DISFLUENCY_WINDOW_SEC,
    DEFAULT_HOP_SEC,
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


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def affect_arrays(windows: np.ndarray, dtype: str | None, device: str, bs: int) -> dict:
    p = AffectPredictor("wavlm", device=device, batch_size=bs, wavlm_autocast_dtype=dtype)
    p(windows[:bs])  # warmup
    out = p(windows)
    del p
    _free_cuda()
    return out


def disfluency_arrays(windows: np.ndarray, dtype: str | None, device: str, bs: int) -> dict:
    p = DisfluencyPredictor("wavlm", device=device, batch_size=bs, wavlm_autocast_dtype=dtype)
    p(windows[:bs])  # warmup
    out = p(windows)
    del p
    _free_cuda()
    return out


def affect_events(arrays: dict, duration_sec: float):
    signals = [
        Signal(name=name, values=np.asarray(arrays[name], dtype=np.float32),
               hop_sec=DEFAULT_HOP_SEC, window_sec=AFFECT_WINDOW_SEC)
        for name in ("arousal", "valence", "dominance")
    ]
    vad = Vad(intervals=((0.0, float(duration_sec)),))
    return affect_extract_events(signals, vad, AffectConfig.balanced())


def disfluency_events(arrays: dict, duration_sec: float):
    _, _, events = produce_disfluency_events(
        fluency_logits=np.asarray(arrays["fluency_logits"], dtype=np.float32),
        disfluency_type_logits=np.asarray(arrays["disfluency_type_logits"], dtype=np.float32),
        hop_sec=DEFAULT_HOP_SEC,
        window_sec=DISFLUENCY_WINDOW_SEC,
        audio_duration_sec=float(duration_sec),
        vad_intervals=[(0.0, float(duration_sec))],
    )
    return events


def _overlap(a, b) -> float:
    lo = max(a.start_sec, b.start_sec)
    hi = min(a.end_sec, b.end_sec)
    inter = max(0.0, hi - lo)
    union = (a.end_sec - a.start_sec) + (b.end_sec - b.start_sec) - inter
    return inter / union if union > 0 else 0.0


def diff_events(base: list, cand: list, *, boundary_tol: float) -> dict:
    """Greedy IoU match within the same event_type; report drift stats."""
    base = sorted(base, key=lambda e: e.start_sec)
    cand = sorted(cand, key=lambda e: e.start_sec)
    used = [False] * len(cand)
    matches = []
    for b in base:
        best_j, best_iou = -1, 0.0
        for j, c in enumerate(cand):
            if used[j] or c.event_type != b.event_type:
                continue
            iou = _overlap(b, c)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            used[best_j] = True
            matches.append((b, cand[best_j], best_iou))

    n_base, n_cand, n_match = len(base), len(cand), len(matches)
    dstart = [abs(b.start_sec - c.start_sec) for b, c, _ in matches]
    dend = [abs(b.end_sec - c.end_sec) for b, c, _ in matches]
    dscore = [abs(b.score - c.score) for b, c, _ in matches]
    label_agree = [b.label == c.label for b, c, _ in matches]
    exact = [
        b.label == c.label and ds <= boundary_tol and de <= boundary_tol
        for (b, c, _), ds, de in zip(matches, dstart, dend)
    ]

    def stat(xs):
        return {"max": float(np.max(xs)), "mean": float(np.mean(xs)), "p99": float(np.quantile(xs, 0.99))} if xs else {"max": 0.0, "mean": 0.0, "p99": 0.0}

    by_type_base: dict = {}
    by_type_cand: dict = {}
    for e in base:
        by_type_base[e.event_type] = by_type_base.get(e.event_type, 0) + 1
    for e in cand:
        by_type_cand[e.event_type] = by_type_cand.get(e.event_type, 0) + 1

    return {
        "n_base": n_base,
        "n_cand": n_cand,
        "n_matched": n_match,
        "n_dropped": n_base - n_match,   # in base, no candidate match
        "n_added": n_cand - n_match,     # in candidate, no base match
        "count_by_type_base": by_type_base,
        "count_by_type_cand": by_type_cand,
        "matched_label_agreement": float(np.mean(label_agree)) if label_agree else 1.0,
        "matched_exact_fraction": float(np.mean(exact)) if exact else 1.0,
        "delta_start_sec": stat(dstart),
        "delta_end_sec": stat(dend),
        "delta_score": stat(dscore),
    }


def run_one(audio_path: str, dtype: str, device: str, bs: int, max_seconds: float | None, boundary_tol: float) -> dict:
    audio = load_audio(audio_path, sample_rate=SAMPLE_RATE)
    samples = audio.samples
    if max_seconds:
        samples = samples[: int(max_seconds * SAMPLE_RATE)]
    duration = len(samples) / SAMPLE_RATE
    affect_w = frame_audio(samples, sample_rate=SAMPLE_RATE, window_sec=AFFECT_WINDOW_SEC, hop_sec=DEFAULT_HOP_SEC)
    disf_w = frame_audio(samples, sample_rate=SAMPLE_RATE, window_sec=DISFLUENCY_WINDOW_SEC, hop_sec=DEFAULT_HOP_SEC)

    # baseline fp32, then candidate dtype, one model resident at a time
    a_base = affect_arrays(affect_w, None, device, bs)
    a_cand = affect_arrays(affect_w, dtype, device, bs)
    d_base = disfluency_arrays(disf_w, None, device, bs)
    d_cand = disfluency_arrays(disf_w, dtype, device, bs)

    ev_a_base = affect_events(a_base, duration)
    ev_a_cand = affect_events(a_cand, duration)
    ev_d_base = disfluency_events(d_base, duration)
    ev_d_cand = disfluency_events(d_cand, duration)

    return {
        "audio": audio_path,
        "duration_sec": round(duration, 1),
        "affect_windows": int(len(affect_w)),
        "disfluency_windows": int(len(disf_w)),
        "affect": diff_events(ev_a_base, ev_a_cand, boundary_tol=boundary_tol),
        "disfluency": diff_events(ev_d_base, ev_d_cand, boundary_tol=boundary_tol),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--candidate-dtype", choices=("fp16", "bf16"), default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-seconds", type=float, default=600.0, help="Cap each archive (0 = full).")
    ap.add_argument("--boundary-tol", type=float, default=DEFAULT_HOP_SEC, help="Start/end tol for exact match (s).")
    ap.add_argument("--json-out")
    args = ap.parse_args()
    max_seconds = args.max_seconds or None

    report = {"candidate_dtype": args.candidate_dtype, "batch_size": args.batch_size,
              "boundary_tol_sec": args.boundary_tol, "files": []}
    for path in args.audio:
        print(f"\n### {path}", flush=True)
        r = run_one(path, args.candidate_dtype, args.device, args.batch_size, max_seconds, args.boundary_tol)
        report["files"].append(r)
        for task in ("affect", "disfluency"):
            d = r[task]
            print(f"  {task:11s} base={d['n_base']:4d} cand={d['n_cand']:4d} "
                  f"matched={d['n_matched']:4d} dropped={d['n_dropped']:3d} added={d['n_added']:3d} "
                  f"label_agree={d['matched_label_agreement']:.3f} exact={d['matched_exact_fraction']:.3f} "
                  f"d_start_max={d['delta_start_sec']['max']:.3f}s d_score_max={d['delta_score']['max']:.4f}", flush=True)

    print("\n" + json.dumps(report, indent=2), flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
