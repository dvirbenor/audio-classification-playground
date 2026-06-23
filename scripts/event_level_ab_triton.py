#!/usr/bin/env python3
"""Event-level A/B for the Triton TensorRT path.

Same gate as event_level_ab.py ("does the change alter the events we emit?") but
compares the two *served* backends instead of in-process fp32-vs-fp16:
  baseline  = ORT/ONNX server  (triton-inference)
  candidate = TensorRT server  (triton-trt)
Both are fed identical windows; the diff isolates ORT-vs-TRT event drift. Reuses
event_level_ab.py's event extraction + diff verbatim. The gate is dropped == 0,
added == 0, label_agree == 1.0 (events identical).

NOTE: covers affect + disfluency (the WavLM tasks the harness models). emotion is
a different producer and is not covered here — A/B it separately.

  uv run python scripts/event_level_ab_triton.py --audio benchmark_audio/*.wav --json-out trt_event_ab.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import event_level_ab as ab  # reuse affect_events / disfluency_events / diff_events

from audio_classification_playground.acoustic_events.inference.artifacts import SAMPLE_RATE
from audio_classification_playground.acoustic_events.inference.audio import frame_audio, load_audio
from audio_classification_playground.acoustic_events.inference.runners import (
    AFFECT_WINDOW_SEC, DISFLUENCY_WINDOW_SEC, DEFAULT_HOP_SEC,
)
from audio_classification_playground.acoustic_events.inference.triton_predictor import (
    TritonAffectPredictor, TritonDisfluencyPredictor,
)


def run_one(path, ort_url, trt_url, max_seconds, boundary_tol):
    audio = load_audio(path, sample_rate=SAMPLE_RATE)
    s = audio.samples
    if max_seconds:
        s = s[: int(max_seconds * SAMPLE_RATE)]
    dur = len(s) / SAMPLE_RATE
    aw = frame_audio(s, sample_rate=SAMPLE_RATE, window_sec=AFFECT_WINDOW_SEC, hop_sec=DEFAULT_HOP_SEC)
    dw = frame_audio(s, sample_rate=SAMPLE_RATE, window_sec=DISFLUENCY_WINDOW_SEC, hop_sec=DEFAULT_HOP_SEC)

    a_base = TritonAffectPredictor(ort_url)(aw)
    a_cand = TritonAffectPredictor(trt_url)(aw)
    d_base = TritonDisfluencyPredictor(ort_url)(dw)
    d_cand = TritonDisfluencyPredictor(trt_url)(dw)

    return {
        "audio": Path(path).name,
        "duration_sec": round(dur, 1),
        "affect": ab.diff_events(ab.affect_events(a_base, dur), ab.affect_events(a_cand, dur), boundary_tol=boundary_tol),
        "disfluency": ab.diff_events(ab.disfluency_events(d_base, dur), ab.disfluency_events(d_cand, dur), boundary_tol=boundary_tol),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--baseline-url", default="triton-inference.nlp-audio-understanding:8001")
    ap.add_argument("--candidate-url", default="triton-trt.nlp-audio-understanding:8001")
    ap.add_argument("--max-seconds", type=float, default=600.0, help="Cap each archive (0 = full).")
    ap.add_argument("--boundary-tol", type=float, default=DEFAULT_HOP_SEC)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    rep = {"baseline": args.baseline_url, "candidate": args.candidate_url, "files": []}
    tot_drop = tot_add = 0
    min_label_agree = 1.0
    for p in args.audio:
        print(f"### {p}", flush=True)
        r = run_one(p, args.baseline_url, args.candidate_url, args.max_seconds or None, args.boundary_tol)
        rep["files"].append(r)
        for t in ("affect", "disfluency"):
            d = r[t]
            tot_drop += d["n_dropped"]; tot_add += d["n_added"]
            min_label_agree = min(min_label_agree, d["matched_label_agreement"])
            print(f"  {t:11s} base={d['n_base']:4d} cand={d['n_cand']:4d} matched={d['n_matched']:4d} "
                  f"dropped={d['n_dropped']:3d} added={d['n_added']:3d} "
                  f"label_agree={d['matched_label_agreement']:.3f} exact={d['matched_exact_fraction']:.3f} "
                  f"d_start_max={d['delta_start_sec']['max']:.3f}s d_score_max={d['delta_score']['max']:.4f}", flush=True)

    verdict = "PASS (event-identical)" if (tot_drop == 0 and tot_add == 0 and min_label_agree == 1.0) \
        else "REVIEW: event drift detected"
    print(f"\n=== {verdict} | total dropped={tot_drop} added={tot_add} min_label_agree={min_label_agree:.3f} ===")
    rep["verdict"] = verdict
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
