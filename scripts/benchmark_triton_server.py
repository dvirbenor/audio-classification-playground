#!/usr/bin/env python3
"""Latency + throughput benchmark against the deployed Triton inference server.

Sends synthetic float32 audio windows to all three models (affect, disfluency,
emotion) via gRPC, sweeping batch sizes and measuring p50/p90/p99 latency and
windows/s throughput.

Compare results against MPS_OPTIMIZATION.md §6 baseline:
  affect dedicated gated mean: 7.8 s/archive (~35 windows × 3.5 s/window solo)
  The Triton dynamic batcher collapses cross-archive windows into full GPU batches,
  so per-request latency should be much lower than per-archive MPS numbers.

Usage:
    # Port-forward first (runs in background; script will do it automatically):
    kubectl port-forward -n nlp-audio-understanding svc/triton-inference 8001:8001

    # Or let the script manage it:
    uv run python scripts/benchmark_triton_server.py
    uv run python scripts/benchmark_triton_server.py --url localhost:8001
    uv run python scripts/benchmark_triton_server.py --models affect --batches 1,8,32,128,256
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import signal
import atexit
from collections.abc import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Model specs — must match config.pbtxt / triton_predictor.py
# ---------------------------------------------------------------------------

MODEL_SPECS: dict[str, dict] = {
    "affect": {
        "input_name": "input_values",
        "window_samples": 56000,     # 3.5 s × 16 kHz
        "outputs": ["arousal", "valence", "dominance"],
        "description": "WavLM ONNX — affect (arousal/valence/dominance)",
    },
    "disfluency": {
        "input_name": "input_values",
        "window_samples": 48000,     # 3.0 s × 16 kHz
        "outputs": ["fluency_logits", "disfluency_type_logits"],
        "description": "WavLM ONNX — disfluency",
    },
    "emotion": {
        "input_name": "input_values",
        "window_samples": 48000,     # 3.0 s × 16 kHz
        "outputs": ["scores"],
        "description": "emotion2vec ONNX — 9-class softmax",
    },
}

DEFAULT_MODELS = ["affect", "disfluency", "emotion"]
DEFAULT_BATCHES = [1, 4, 16, 64, 128, 256]
DEFAULT_WARMUP = 3
DEFAULT_REPS = 20


# ---------------------------------------------------------------------------
# Port-forward helper
# ---------------------------------------------------------------------------

_pf_proc: subprocess.Popen | None = None


def start_port_forward(local_port: int = 8001, remote_port: int = 8001) -> None:
    global _pf_proc
    cmd = [
        "kubectl", "port-forward",
        "-n", "nlp-audio-understanding",
        "svc/triton-inference",
        f"{local_port}:{remote_port}",
    ]
    print(f"[port-forward] {' '.join(cmd)}")
    _pf_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    def _cleanup():
        if _pf_proc and _pf_proc.poll() is None:
            _pf_proc.terminate()

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda *_: (sys.exit(0)))

    # Give port-forward a moment to establish
    time.sleep(2.0)
    if _pf_proc.poll() is not None:
        raise RuntimeError("kubectl port-forward exited immediately — check your kubeconfig/namespace")
    print(f"[port-forward] ready on localhost:{local_port}")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _percentile(arr: list[float], p: int) -> float:
    return float(np.percentile(arr, p))


def benchmark_model(
    client,
    grpcclient,
    model_name: str,
    spec: dict,
    batch_sizes: list[int],
    warmup: int,
    reps: int,
) -> list[dict]:
    results = []
    for batch in batch_sizes:
        windows = np.random.randn(batch, spec["window_samples"]).astype(np.float32)
        inp = grpcclient.InferInput(spec["input_name"], list(windows.shape), "FP32")
        inp.set_data_from_numpy(np.ascontiguousarray(windows))
        outputs = [grpcclient.InferRequestedOutput(o) for o in spec["outputs"]]

        # Warmup — not timed
        for _ in range(warmup):
            client.infer(model_name, inputs=[inp], outputs=outputs)

        # Timed runs
        latencies_ms = []
        for _ in range(reps):
            t0 = time.perf_counter()
            client.infer(model_name, inputs=[inp], outputs=outputs)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        p50 = _percentile(latencies_ms, 50)
        p90 = _percentile(latencies_ms, 90)
        p99 = _percentile(latencies_ms, 99)
        mean = float(np.mean(latencies_ms))
        windows_per_s = batch / (mean / 1000)

        results.append({
            "model": model_name,
            "batch": batch,
            "mean_ms": mean,
            "p50_ms": p50,
            "p90_ms": p90,
            "p99_ms": p99,
            "windows_per_s": windows_per_s,
        })

    return results


def print_results(all_results: list[dict]) -> None:
    print()
    print(f"{'model':<12} {'batch':>5} {'mean ms':>9} {'p50 ms':>8} {'p90 ms':>8} {'p99 ms':>8} {'win/s':>8}")
    print("-" * 68)
    last_model = None
    for r in all_results:
        if r["model"] != last_model and last_model is not None:
            print()
        last_model = r["model"]
        print(
            f"{r['model']:<12} {r['batch']:>5} "
            f"{r['mean_ms']:>9.1f} {r['p50_ms']:>8.1f} "
            f"{r['p90_ms']:>8.1f} {r['p99_ms']:>8.1f} "
            f"{r['windows_per_s']:>8.1f}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--url", default=None,
        help="Triton gRPC URL (default: auto port-forward to localhost:8001)",
    )
    p.add_argument(
        "--models", default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated models to benchmark (default: {','.join(DEFAULT_MODELS)})",
    )
    p.add_argument(
        "--batches", default=",".join(str(b) for b in DEFAULT_BATCHES),
        help=f"Comma-separated batch sizes (default: {','.join(str(b) for b in DEFAULT_BATCHES)})",
    )
    p.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"Warmup requests per batch/model (default: {DEFAULT_WARMUP})",
    )
    p.add_argument(
        "--reps", type=int, default=DEFAULT_REPS,
        help=f"Timed repetitions per batch/model (default: {DEFAULT_REPS})",
    )
    p.add_argument(
        "--check-ready", action="store_true",
        help="Check server readiness and model metadata before benchmarking",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        print("ERROR: tritonclient[grpc] not installed. Run: pip install tritonclient[grpc]")
        sys.exit(1)

    url = args.url
    if url is None:
        start_port_forward(local_port=8001, remote_port=8001)
        url = "localhost:8001"

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    batch_sizes = [int(b) for b in args.batches.split(",") if b.strip()]

    print(f"\nTriton benchmark — url={url}")
    print(f"Models: {models}  Batches: {batch_sizes}  Warmup: {args.warmup}  Reps: {args.reps}\n")

    client = grpcclient.InferenceServerClient(url=url)

    if args.check_ready:
        print("Server ready:", client.is_server_ready())
        for m in models:
            print(f"  {m} ready:", client.is_model_ready(m))
            meta = client.get_model_metadata(m)
            print(f"  {m} inputs: {[(i.name, i.shape, i.datatype) for i in meta.inputs]}")
            print(f"  {m} outputs: {[(o.name, o.shape, o.datatype) for o in meta.outputs]}")
        print()

    all_results: list[dict] = []
    for model_name in models:
        if model_name not in MODEL_SPECS:
            print(f"WARNING: unknown model '{model_name}', skipping")
            continue
        spec = MODEL_SPECS[model_name]
        print(f"Benchmarking {model_name} — {spec['description']}")
        print(f"  input: [batch, {spec['window_samples']}] float32 — {spec['window_samples']/16000:.2f}s windows")
        results = benchmark_model(
            client, grpcclient, model_name, spec, batch_sizes,
            warmup=args.warmup, reps=args.reps,
        )
        all_results.extend(results)
        for r in results:
            print(
                f"  batch={r['batch']:>4}  mean={r['mean_ms']:.1f}ms  "
                f"p90={r['p90_ms']:.1f}ms  win/s={r['windows_per_s']:.1f}"
            )

    print_results(all_results)


if __name__ == "__main__":
    main()
