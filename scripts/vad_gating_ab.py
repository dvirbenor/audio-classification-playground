#!/usr/bin/env python3
"""VAD-gating A/B: event-identity + speed, full-timeline vs --vad-gating.

For each archive, runs the REAL ModelSuite through ``run_all_inference`` twice
(full timeline and VAD-gated, sharing one Silero VAD pass), then runs the
affect/disfluency/emotion producers on both artifact sets **using the real VAD
intervals** (the gate basis) and diffs the emitted events. Also records per-task
GPU inference time so we can compare overall archive processing speed.

The gate is event-safe iff it is a superset of every producer's consumed frame
set; this harness is the empirical confirmation (expect 0 dropped/added, 1.000
label agreement, 0 boundary/score drift).

Example:
    uv run python scripts/vad_gating_ab.py \
        --index benchmark_audio/index.json \
        --json-out optimization_research/baseline_results/vad_gating_ab.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from audio_classification_playground.acoustic_events.inference.artifacts import (
    SAMPLE_RATE,
    load_prediction_artifact,
)
from audio_classification_playground.acoustic_events.inference.audio import load_audio
from audio_classification_playground.acoustic_events.inference.models import ModelSuite
from audio_classification_playground.acoustic_events.inference.runners import (
    AFFECT_WINDOW_SEC,
    DISFLUENCY_WINDOW_SEC,
    EMOTION_WINDOW_SEC,
    DEFAULT_HOP_SEC,
    run_all_inference,
)
from audio_classification_playground.acoustic_events.inference.vad_gating import (
    VadGating,
    speech_window_mask,
)
from audio_classification_playground.acoustic_events.composition import (
    compose_affect_from_artifacts,
    compose_disfluency_from_artifacts,
    compose_emotion_from_artifacts,
)

GPU_TASKS = ("affect", "disfluency", "emotion")
WINDOW_SEC = {
    "affect": AFFECT_WINDOW_SEC,
    "disfluency": DISFLUENCY_WINDOW_SEC,
    "emotion": EMOTION_WINDOW_SEC,
}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --- event diff (same logic as scripts/event_level_ab.py) ------------------- #
def _overlap(a, b) -> float:
    lo = max(a.start_sec, b.start_sec)
    hi = min(a.end_sec, b.end_sec)
    inter = max(0.0, hi - lo)
    union = (a.end_sec - a.start_sec) + (b.end_sec - b.start_sec) - inter
    return inter / union if union > 0 else 0.0


def diff_events(base: list, cand: list, *, boundary_tol: float) -> dict:
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
            matches.append((b, cand[best_j]))
    dstart = [abs(b.start_sec - c.start_sec) for b, c in matches]
    dend = [abs(b.end_sec - c.end_sec) for b, c in matches]
    dscore = [abs(b.score - c.score) for b, c in matches]
    label_agree = [b.label == c.label for b, c in matches]
    return {
        "n_base": len(base),
        "n_cand": len(cand),
        "n_matched": len(matches),
        "n_dropped": len(base) - len(matches),
        "n_added": len(cand) - len(matches),
        "label_agreement": float(np.mean(label_agree)) if label_agree else 1.0,
        "d_start_max": float(np.max(dstart)) if dstart else 0.0,
        "d_end_max": float(np.max(dend)) if dend else 0.0,
        "d_score_max": float(np.max(dscore)) if dscore else 0.0,
    }


# --- producers (real composition stage; reads duration + intervals from the
#     artifacts, so full and gated are composed identically) ----------------- #
def _compose_events(root: Path, task: str) -> list:
    vad = root / "vad"
    if task == "affect":
        return compose_affect_from_artifacts(affect_artifact=root / "affect", vad_artifact=vad)[2]
    if task == "disfluency":
        return compose_disfluency_from_artifacts(disfluency_artifact=root / "disfluency", vad_artifact=vad)[2]
    return compose_emotion_from_artifacts(emotion_artifact=root / "emotion", vad_artifact=vad)[2]


def _run_inference(audio, out_root, suite, detector, gating, intervals):
    predictors = {"affect": suite.affect, "disfluency": suite.disfluency, "emotion": suite.emotion}
    _sync()
    t0 = time.perf_counter()
    res = run_all_inference(
        audio,
        out_dir=out_root,
        affect_backbone="wavlm",
        disfluency_backbone="wavlm",
        predictors=predictors,
        vad_detector=detector,
        artifact_path_fn=lambda task: out_root / task,
        reuse_cache=False,
        vad_gating=gating,
        vad_intervals=(intervals if gating is not None else None),
        progress=lambda m: None,
    )
    _sync()
    wall = time.perf_counter() - t0
    return res, wall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="benchmark_audio/index.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--bridge-sec", type=float, default=1.5)
    ap.add_argument("--boundary-tol", type=float, default=DEFAULT_HOP_SEC)
    ap.add_argument("--gate-tasks", nargs="+", default=list(GPU_TASKS),
                    choices=GPU_TASKS, help="Tasks to gate (default: all, to expose per-task drift).")
    ap.add_argument("--max-archives", type=int, default=0, help="0 = all")
    ap.add_argument("--out-root", default="/tmp/vad_gating_ab")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    archives = json.loads(Path(args.index).read_text())
    if args.max_archives:
        archives = archives[: args.max_archives]
    out_root = Path(args.out_root)

    print("loading ModelSuite (wavlm affect+disfluency, emotion2vec, silero vad) ...", flush=True)
    suite = ModelSuite(
        affect_backbone="wavlm", disfluency_backbone="wavlm",
        batch_size=args.batch_size, device=args.device, load_vad=True,
    )
    # warmup so neither timed run pays first-call autotune/alloc
    for task in GPU_TASKS:
        ws = int(round(WINDOW_SEC[task] * SAMPLE_RATE))
        dummy = np.zeros((min(args.batch_size, 64), ws), dtype=np.float32)
        getattr(suite, task)(dummy)
    _sync()
    print("warmup done\n", flush=True)

    results = []
    for i, a in enumerate(archives, 1):
        f = a.get("local_path")
        if not f or not Path(f).is_file():
            continue
        aid = a.get("archive_id", f"idx{i}")
        audio = load_audio(f, sample_rate=SAMPLE_RATE)
        dur = audio.duration_sec
        intervals = suite.vad(audio.samples, audio.sample_rate)
        speech = sum(e - s for s, e in intervals) / dur if dur else 0.0
        detector = lambda samples, sr, _iv=intervals: list(_iv)
        print(f"[{i}/{len(archives)}] {aid}  dur={dur/60:.1f}m  speech={speech*100:.1f}%  "
              f"intervals={len(intervals)}", flush=True)

        full_root = out_root / aid / "full"
        gated_root = out_root / aid / "gated"
        res_full, wall_full = _run_inference(audio, full_root, suite, detector, None, intervals)
        res_gated, wall_gated = _run_inference(
            audio, gated_root, suite, detector,
            VadGating(enabled=True, bridge_sec=args.bridge_sec, tasks=tuple(args.gate_tasks)),
            intervals,
        )

        rec = {"archive_id": aid, "duration_sec": round(dur, 1),
               "speech_fraction": round(speech, 4), "n_intervals": len(intervals),
               "tasks": {}, "events": {}}
        gpu_full = gpu_gated = 0.0
        for task in GPU_TASKS:
            n = load_prediction_artifact(full_root / task).arrays
            n_windows = (n["arousal"].shape[0] if task == "affect"
                         else n["fluency_logits"].shape[0] if task == "disfluency"
                         else n["probabilities"].shape[0])
            mask = speech_window_mask(n_windows, DEFAULT_HOP_SEC, WINDOW_SEC[task],
                                      intervals, bridge_sec=args.bridge_sec)
            tf = res_full.task_elapsed_sec.get(task, 0.0)
            tg = res_gated.task_elapsed_sec.get(task, 0.0)
            gpu_full += tf
            gpu_gated += tg
            rec["tasks"][task] = {
                "windows_full": int(n_windows),
                "windows_kept": int(mask.sum()),
                "kept_fraction": round(float(mask.mean()), 4),
                "sec_full": round(tf, 3),
                "sec_gated": round(tg, 3),
                "speedup": round(tf / tg, 2) if tg else None,
            }

        for task in GPU_TASKS:
            ev_full = _compose_events(full_root, task)
            ev_gated = _compose_events(gated_root, task)
            rec["events"][task] = diff_events(ev_full, ev_gated, boundary_tol=args.boundary_tol)

        rec["gpu_total_full_sec"] = round(gpu_full, 2)
        rec["gpu_total_gated_sec"] = round(gpu_gated, 2)
        rec["gpu_total_speedup"] = round(gpu_full / gpu_gated, 2) if gpu_gated else None
        rec["wall_full_sec"] = round(wall_full, 2)
        rec["wall_gated_sec"] = round(wall_gated, 2)
        results.append(rec)

        for task in GPU_TASKS:
            t = rec["tasks"][task]
            e = rec["events"][task]
            print(f"    {task:11s} keep={t['kept_fraction']*100:4.1f}%  "
                  f"{t['sec_full']:6.1f}s -> {t['sec_gated']:6.1f}s ({t['speedup']}x)  | "
                  f"events {e['n_base']}->{e['n_cand']} drop={e['n_dropped']} add={e['n_added']} "
                  f"label_agree={e['label_agreement']:.3f} dstart={e['d_start_max']:.3f} "
                  f"dscore={e['d_score_max']:.4f}", flush=True)
        print(f"    GPU total {rec['gpu_total_full_sec']}s -> {rec['gpu_total_gated_sec']}s "
              f"({rec['gpu_total_speedup']}x)\n", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # aggregate
    def agg(key_path):
        vals = []
        for r in results:
            v = r
            for k in key_path:
                v = v[k]
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    drops = sum(r["events"][t]["n_dropped"] for r in results for t in GPU_TASKS)
    adds = sum(r["events"][t]["n_added"] for r in results for t in GPU_TASKS)
    speedups = agg(["gpu_total_speedup"])
    summary = {
        "n_archives": len(results),
        "total_events_dropped": drops,
        "total_events_added": adds,
        "event_safe": drops == 0 and adds == 0,
        "mean_gpu_total_speedup": round(float(np.mean(speedups)), 2) if speedups else None,
        "mean_speech_fraction": round(float(np.mean(agg(["speech_fraction"]))), 4) if results else None,
    }
    print("SUMMARY:", json.dumps(summary, indent=2), flush=True)

    report = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "batch_size": args.batch_size, "bridge_sec": args.bridge_sec,
        "boundary_tol_sec": args.boundary_tol,
        "emotion_runtime": getattr(getattr(suite, "emotion", None), "compile_model", None),
        "archives": results, "summary": summary,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
