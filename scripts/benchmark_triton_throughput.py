#!/usr/bin/env python3
"""Windows/second throughput benchmark for the *deployed* Triton server.

The Triton analog of ``benchmark_current_inference_real_audio.py``. That script
measures pure in-process GPU throughput (one big batch fed to a local model);
this one measures the **hosted** server as it is actually deployed: CPU clients
decode real archives, frame them into windows, and stream those windows to the
Triton Inference Server over gRPC. Triton's dynamic batcher merges concurrent
requests from many clients into full GPU batches — the cross-archive batching
that is the whole point of the Triton path (and that MPS cannot provide), so a
single sequential client *understates* it. We therefore sweep concurrency.

Two windows/second numbers are reported per task:
  * client win/s  — total windows / wall-clock across the C concurrent clients
                    (end-to-end, includes gRPC round-trip + serialization).
  * server win/s  — windows / GPU compute time, read from Triton's per-model
                    statistics (``compute_infer``). This is the clean GPU-only
                    number directly comparable to OPTIMIZATION_REPORT.md's
                    win/s table (e.g. affect GB202 = 1039 win/s).
Plus the realized average dynamic-batch size and mean queue delay per request.

Windows/sec is per-window, so it does NOT depend on hop; hop only sets how many
windows an archive yields. Windows are framed exactly like production (same
per-task window length + hop) so the request payloads match the live fleet.

Examples:
    # per-task concurrency sweep against the deployed ClusterIP service
    uv run python scripts/benchmark_triton_throughput.py \
        --url triton-inference.nlp-audio-understanding:8001 \
        --concurrency 1,4,8,16 --json-out triton_throughput.json

    # blended: all three tasks hammering the one GPU at once (deployed reality)
    uv run python scripts/benchmark_triton_throughput.py --blended --concurrency 8
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from audio_classification_playground.acoustic_events.inference.artifacts import SAMPLE_RATE
from audio_classification_playground.acoustic_events.inference.audio import frame_audio, load_audio
from audio_classification_playground.acoustic_events.inference.runners import (
    AFFECT_WINDOW_SEC,
    DISFLUENCY_WINDOW_SEC,
    EMOTION_WINDOW_SEC,
    DEFAULT_HOP_SEC,
)
from audio_classification_playground.acoustic_events.inference import triton_predictor
from audio_classification_playground.acoustic_events.inference.triton_predictor import (
    TritonAffectPredictor,
    TritonDisfluencyPredictor,
    TritonEmotionPredictor,
)

TASKS = {
    "affect": (TritonAffectPredictor, AFFECT_WINDOW_SEC),
    "disfluency": (TritonDisfluencyPredictor, DISFLUENCY_WINDOW_SEC),
    "emotion": (TritonEmotionPredictor, EMOTION_WINDOW_SEC),
}


# --------------------------------------------------------------------------- #
# Server-side statistics (the clean GPU-only number)
# --------------------------------------------------------------------------- #

def _stats_snapshot(url: str, model: str) -> dict:
    """Per-batch-size compute counters from Triton, keyed for before/after diff."""
    import tritonclient.grpc as grpcclient

    client = grpcclient.InferenceServerClient(url=url)
    raw = client.get_inference_statistics(model_name=model, as_json=True)
    stats = raw.get("model_stats", [])
    if not stats:
        return {"batches": {}, "queue_ns": 0, "queue_count": 0}
    s = stats[0]
    batches: dict[int, dict[str, int]] = {}
    for b in s.get("batch_stats", []):
        bs = int(b["batch_size"])
        ci = b.get("compute_infer", {})
        batches[bs] = {
            "count": int(ci.get("count", 0)),       # executions at this batch size
            "ns": int(ci.get("ns", 0)),             # compute_infer ns at this batch size
        }
    q = s.get("inference_stats", {}).get("queue", {})
    return {
        "batches": batches,
        "queue_ns": int(q.get("ns", 0)),
        "queue_count": int(q.get("count", 0)),
    }


def _stats_delta(before: dict, after: dict) -> dict:
    """Diff two snapshots → windows, executions, compute/queue time for the run."""
    total_windows = total_exec = total_compute_ns = 0
    for bs, a in after["batches"].items():
        b = before["batches"].get(bs, {"count": 0, "ns": 0})
        dcount = a["count"] - b["count"]
        if dcount <= 0:
            continue
        dns = a["ns"] - b["ns"]
        total_windows += bs * dcount
        total_exec += dcount
        total_compute_ns += dns
    dq_ns = after["queue_ns"] - before["queue_ns"]
    dq_count = after["queue_count"] - before["queue_count"]
    compute_sec = total_compute_ns / 1e9
    return {
        "server_windows": total_windows,
        "server_executions": total_exec,
        "server_compute_sec": round(compute_sec, 4),
        "server_windows_per_sec": round(total_windows / compute_sec, 1) if compute_sec else None,
        "avg_dynamic_batch": round(total_windows / total_exec, 1) if total_exec else None,
        "mean_queue_ms": round(dq_ns / 1e6 / dq_count, 2) if dq_count else None,
    }


# --------------------------------------------------------------------------- #
# Window pools
# --------------------------------------------------------------------------- #

def build_window_pool(files: list[dict], window_sec: float, hop_sec: float,
                      cap: int) -> np.ndarray:
    """Decode the selected archives and frame them into one [N, win] pool."""
    pools = []
    have = 0
    for f in files:
        audio = load_audio(f["local_path"], sample_rate=SAMPLE_RATE)
        w = frame_audio(audio.samples, sample_rate=audio.sample_rate,
                        window_sec=window_sec, hop_sec=hop_sec)
        pools.append(np.ascontiguousarray(w, dtype=np.float32))
        have += len(w)
        if cap and have >= cap:
            break
    pool = np.concatenate(pools, axis=0)
    if cap and len(pool) > cap:
        pool = pool[:cap]
    return pool


# --------------------------------------------------------------------------- #
# One timed run: C concurrent clients stream a window pool to one task
# --------------------------------------------------------------------------- #

def run_task(url: str, task: str, pool: np.ndarray, concurrency: int,
             warmup: int) -> dict:
    predictor_cls, _ = TASKS[task]
    # One predictor (== one gRPC client) per concurrent stream.
    predictors = [predictor_cls(url) for _ in range(concurrency)]
    shards = np.array_split(pool, concurrency)

    # Warm every client so connection setup / first-batch isn't timed.
    if warmup:
        for p, sh in zip(predictors, shards):
            if len(sh):
                p(sh[: min(warmup, len(sh))])

    before = _stats_snapshot(url, task)

    def _work(i: int):
        sh = shards[i]
        if not len(sh):
            return 0
        predictors[i](sh)
        return len(sh)

    start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            sent = sum(ex.map(_work, range(concurrency)))
    except Exception as e:  # server OOM / RPC failure: record, don't kill the sweep
        wall = time.perf_counter() - start
        return {
            "task": task, "concurrency": concurrency, "max_chunk": triton_predictor._MAX_CHUNK,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "client_windows": 0, "wall_sec": round(wall, 4),
            "client_windows_per_sec": None, "server_windows_per_sec": None,
            "avg_dynamic_batch": None, "mean_queue_ms": None,
        }
    wall = time.perf_counter() - start

    after = _stats_snapshot(url, task)
    server = _stats_delta(before, after)
    return {
        "task": task,
        "concurrency": concurrency,
        "max_chunk": triton_predictor._MAX_CHUNK,
        "client_windows": int(sent),
        "wall_sec": round(wall, 4),
        "client_windows_per_sec": round(sent / wall, 1) if wall else None,
        **server,
    }


def _fmt_row(r: dict) -> str:
    if r.get("error"):
        return (f"  {r['task']:11s} C={r['concurrency']:<3d} "
                f"chunk={r.get('max_chunk')}  FAILED: {r['error']}")
    return (f"  {r['task']:11s} C={r['concurrency']:<3d} "
            f"{r['client_windows']:7d} win  {r['wall_sec']:7.2f}s  "
            f"client {str(r['client_windows_per_sec']):>8s} win/s | "
            f"server {str(r['server_windows_per_sec']):>8s} win/s  "
            f"batch~{str(r['avg_dynamic_batch']):>5s}  queue {str(r['mean_queue_ms'])}ms")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="triton-inference.nlp-audio-understanding:8001",
                    help="Triton gRPC URL (host:8001).")
    ap.add_argument("--index", default="benchmark_audio/index.json",
                    help="index.json from retrieve_benchmark_audio.py.")
    ap.add_argument("--max-archives", type=int, default=0,
                    help="Use only the first N archives from the index. 0 = all.")
    ap.add_argument("--tasks", default="affect,disfluency,emotion")
    ap.add_argument("--concurrency", default="1,4,8,16",
                    help="Comma list of concurrent-client counts to sweep.")
    ap.add_argument("--windows-per-task", type=int, default=20000,
                    help="Cap windows per task (bounds runtime). 0 = all framed.")
    ap.add_argument("--hop-sec", type=float, default=DEFAULT_HOP_SEC)
    ap.add_argument("--max-chunk", type=int, default=triton_predictor._MAX_CHUNK,
                    help="Windows per gRPC request (== max batch the dynamic batcher "
                         "can form per client). Production default is 256; lower it if "
                         "the WavLM models OOM the shared GPU.")
    ap.add_argument("--warmup", type=int, default=64,
                    help="Windows per client to send (untimed) before measuring.")
    ap.add_argument("--blended", action="store_true",
                    help="Run all tasks concurrently at each concurrency level "
                         "(deployed reality: all three share one GPU) and report "
                         "the per-GPU blended win/s.")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    files = json.loads(Path(args.index).read_text())
    files = [f for f in files if f.get("local_path") and Path(f["local_path"]).exists()]
    if args.max_archives:
        files = files[: args.max_archives]
    if not files:
        print("no usable files in index", flush=True)
        return 1
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    concurrencies = [int(c) for c in args.concurrency.split(",") if c.strip()]
    # Per-request chunk == the largest batch a single client lets the dynamic
    # batcher form. Override the predictor's module-level default at runtime.
    triton_predictor._MAX_CHUNK = args.max_chunk
    total_audio_min = sum(f.get("duration_sec", 0) for f in files) / 60.0
    print(f"server: {args.url}   max_chunk (req batch): {args.max_chunk}", flush=True)
    print(f"archives: {len(files)}  (~{total_audio_min:.0f} min audio total)", flush=True)

    # Frame once per task; reuse the pool across the concurrency sweep.
    print("\nframing window pools ...", flush=True)
    pools = {}
    for t in tasks:
        _, window_sec = TASKS[t]
        pool = build_window_pool(files, window_sec, args.hop_sec, args.windows_per_task)
        pools[t] = pool
        print(f"  {t:11s} {len(pool):7d} windows  [{pool.shape[1]} samples/win]", flush=True)

    results = []

    if args.blended:
        # All tasks fire simultaneously at concurrency C → contend for the one
        # GPU exactly like the deployed fleet. Report each task + the per-GPU sum.
        for c in concurrencies:
            print(f"\n[blended] all tasks @ concurrency {c} each", flush=True)
            with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
                futs = {t: ex.submit(run_task, args.url, t, pools[t], c, args.warmup)
                        for t in tasks}
                rows = {t: f.result() for t, f in futs.items()}
            blended_client = 0.0
            for t in tasks:
                r = rows[t]
                r["mode"] = "blended"
                results.append(r)
                print(_fmt_row(r), flush=True)
                blended_client += r["client_windows_per_sec"] or 0
            print(f"  -> per-GPU blended client throughput: {blended_client:.0f} win/s", flush=True)
    else:
        for t in tasks:
            print(f"\n[{t}] concurrency sweep", flush=True)
            for c in concurrencies:
                r = run_task(args.url, t, pools[t], c, args.warmup)
                r["mode"] = "isolated"
                results.append(r)
                print(_fmt_row(r), flush=True)

    if args.json_out:
        report = {
            "url": args.url,
            "archives": [Path(f["local_path"]).name for f in files],
            "audio_minutes": round(total_audio_min, 1),
            "hop_sec": args.hop_sec,
            "max_chunk": args.max_chunk,
            "windows_per_task_cap": args.windows_per_task,
            "blended": args.blended,
            "results": results,
        }
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
