#!/usr/bin/env python3
"""Per-archive I/O (download) and processing (decode) benchmark.

Compares the *current* production path against the candidate optimizations,
one archive at a time, with per-trial granularity so the raw JSON can drive a
later report:

  download (I/O):   boto3 download_file (default)         [current]
                    boto3 download_file (tuned concurrency)
                    s5cmd cp                               [candidate]

  decode (CPU):     librosa.load(sr=16k, mono)            [current]
                    ffmpeg CLI -> f32le pipe (swresample) [candidate]

The decode comparison also records the numerical delta and the
``decoded_audio_sha256`` for each decoder, because the sha is the inference
cache key (CLAUDE.md): a decoder swap changes it and re-baselines the corpus,
so the drift must be gated by the event-level A/B before adoption.

Why ffmpeg via subprocess and not torchcodec: torchcodec 0.9.1 fails to load
against the pinned torch (``undefined symbol`` in libc10 -- a torch C++ ABI
mismatch, not an ffmpeg-version problem), so the ABI-independent ffmpeg CLI is
the working path here.

Example:
    uv run python scripts/benchmark_io_and_decode.py \
        --index benchmark_audio/index.json \
        --json-out optimization_research/baseline_results/io_decode_compare.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audio_classification_playground.acoustic_events.inference.artifacts import (
    SAMPLE_RATE,
    decoded_audio_sha256,
)

BUCKET = "riverside-pro-main"


def _median(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 4) if vals else None


# --------------------------------------------------------------------------- #
# Download paths
# --------------------------------------------------------------------------- #
def _boto3_client(max_pool: int = 64):
    import boto3
    from botocore.config import Config

    return boto3.client("s3", config=Config(max_pool_connections=max_pool))


def download_boto3(key: str, dst: Path, *, bucket: str, tuned: bool) -> float:
    """Download via boto3 download_file. Returns elapsed seconds."""
    from boto3.s3.transfer import TransferConfig

    client = _boto3_client()
    if tuned:
        cfg = TransferConfig(
            max_concurrency=64,
            multipart_threshold=8 * 1024 * 1024,
            multipart_chunksize=16 * 1024 * 1024,
        )
    else:
        cfg = TransferConfig()  # boto3 defaults: concurrency 10, 8 MiB chunks
    dst.unlink(missing_ok=True)
    start = time.perf_counter()
    client.download_file(bucket, key, str(dst), Config=cfg)
    return time.perf_counter() - start


def download_s5cmd(key: str, dst: Path, *, bucket: str, s5cmd: str) -> float:
    """Download via the s5cmd binary. Returns elapsed seconds."""
    dst.unlink(missing_ok=True)
    cmd = [s5cmd, "cp", f"s3://{bucket}/{key}", str(dst)]
    start = time.perf_counter()
    subprocess.run(cmd, check=True, capture_output=True)
    return time.perf_counter() - start


def _mbps(nbytes: int, sec: float) -> float | None:
    return round((nbytes / 1e6) / sec, 1) if sec and sec > 0 else None


def benchmark_download(
    key: str,
    *,
    bucket: str,
    s5cmd: str,
    tmp_dir: Path,
    trials: int,
    nbytes: int,
) -> dict:
    """Run interleaved download trials for all three methods."""
    methods = {
        "boto3_default": lambda dst: download_boto3(key, dst, bucket=bucket, tuned=False),
        "boto3_tuned": lambda dst: download_boto3(key, dst, bucket=bucket, tuned=True),
        "s5cmd": lambda dst: download_s5cmd(key, dst, bucket=bucket, s5cmd=s5cmd),
    }
    per_method: dict[str, list[dict]] = {name: [] for name in methods}
    dst = tmp_dir / "dl_probe.bin"
    # Interleave (trial outer, method inner) so network drift hits all methods evenly.
    for _ in range(trials):
        for name, fn in methods.items():
            try:
                sec = fn(dst)
                got = dst.stat().st_size if dst.exists() else 0
                per_method[name].append(
                    {"sec": round(sec, 4), "mbps": _mbps(got, sec), "bytes": got}
                )
            except Exception as exc:  # noqa: BLE001 - record and continue
                per_method[name].append({"error": f"{type(exc).__name__}: {exc}"[:200]})
            finally:
                dst.unlink(missing_ok=True)
    out: dict = {}
    for name, trials_list in per_method.items():
        secs = [t["sec"] for t in trials_list if "sec" in t]
        mbps = [t["mbps"] for t in trials_list if t.get("mbps") is not None]
        out[name] = {
            "trials": trials_list,
            "median_sec": _median(secs),
            "median_mbps": _median(mbps),
        }
    return out


# --------------------------------------------------------------------------- #
# Decode paths
# --------------------------------------------------------------------------- #
def decode_librosa(path: Path, sample_rate: int) -> np.ndarray:
    import librosa

    samples, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.ascontiguousarray(samples, dtype=np.float32)


def decode_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", str(path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype="<f4")


def _numeric_diff(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(a), len(b))
    if n == 0:
        return {"len_diff": len(a) - len(b), "overlap": 0}
    diff = np.abs(a[:n].astype(np.float64) - b[:n].astype(np.float64))
    return {
        "len_diff": int(len(a) - len(b)),
        "overlap": int(n),
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "rms_diff": float(np.sqrt(np.mean(diff ** 2))),
        "frac_gt_1e_3": float(np.mean(diff > 1e-3)),
        "frac_gt_1e_2": float(np.mean(diff > 1e-2)),
    }


def benchmark_decode(
    path: Path,
    *,
    sample_rate: int,
    duration_sec: float,
    trials: int,
) -> dict:
    """Time both decoders (warm) and capture the numerical delta + shas."""
    decoders = {"librosa": decode_librosa, "ffmpeg": decode_ffmpeg}
    arrays: dict[str, np.ndarray] = {}
    timings: dict[str, list[float]] = {name: [] for name in decoders}

    for name, fn in decoders.items():
        for trial in range(trials):
            start = time.perf_counter()
            arr = fn(path, sample_rate)
            elapsed = time.perf_counter() - start
            timings[name].append(round(elapsed, 4))
            if trial == 0:
                arrays[name] = arr  # keep one copy for the numeric diff
            else:
                del arr
                gc.collect()

    out: dict = {}
    for name in decoders:
        arr = arrays[name]
        med = _median(timings[name])
        out[name] = {
            "trials_sec": timings[name],
            "median_sec": med,
            "n_samples": int(len(arr)),
            "sha256": decoded_audio_sha256(np.ascontiguousarray(arr, dtype=np.float32)),
            # realtime factor: seconds of audio decoded per second of compute
            "rtf": round(duration_sec / med, 1) if med else None,
        }
    lib_med = out["librosa"]["median_sec"]
    ff_med = out["ffmpeg"]["median_sec"]
    out["speedup_ffmpeg_over_librosa"] = (
        round(lib_med / ff_med, 2) if lib_med and ff_med else None
    )
    numeric = _numeric_diff(arrays["librosa"], arrays["ffmpeg"])
    numeric["sha_match"] = out["librosa"]["sha256"] == out["ffmpeg"]["sha256"]
    out["numeric"] = numeric
    arrays.clear()
    gc.collect()
    return out


# --------------------------------------------------------------------------- #
def _versions(s5cmd: str) -> dict:
    info: dict = {"cpu_count": os.cpu_count()}
    try:
        import librosa

        info["librosa"] = librosa.__version__
    except Exception:
        pass
    try:
        import soxr

        info["soxr"] = soxr.__version__
    except Exception:
        pass
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        info["ffmpeg"] = out.stdout.splitlines()[0] if out.stdout else "?"
    except Exception:
        pass
    try:
        out = subprocess.run([s5cmd, "version"], capture_output=True, text=True)
        info["s5cmd"] = (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception:
        pass
    return info


def _ensure_local(archive: dict, *, bucket: str, s5cmd: str, tmp_dir: Path) -> Path | None:
    """Return a local decode source, downloading via s5cmd if absent."""
    local = archive.get("local_path")
    if local and Path(local).is_file():
        return Path(local)
    key = archive.get("s3_key")
    if not key:
        return None
    dst = tmp_dir / f"decode_src_{archive.get('archive_id', 'x')}.wav"
    download_s5cmd(key, dst, bucket=bucket, s5cmd=s5cmd)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="benchmark_audio/index.json")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--s5cmd", default=str(Path(".venv/bin/s5cmd").resolve()))
    ap.add_argument("--tmp-dir", default="/tmp/io_bench")
    ap.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    ap.add_argument("--download-trials", type=int, default=3)
    ap.add_argument("--decode-trials", type=int, default=3)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    archives = json.loads(Path(args.index).read_text())
    if not archives:
        print("no archives in index", flush=True)
        return 1

    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bucket": args.bucket,
        "sample_rate": args.sample_rate,
        "tmp_dir": str(tmp_dir),
        "download_trials": args.download_trials,
        "decode_trials": args.decode_trials,
        "s5cmd_path": args.s5cmd,
        **_versions(args.s5cmd),
    }
    print(f"env: {meta}\n", flush=True)

    results = []
    for i, archive in enumerate(archives, 1):
        aid = archive.get("archive_id", f"idx{i}")
        dur = float(archive.get("duration_sec") or 0.0)
        nbytes = int(archive.get("object_size_bytes") or 0)
        key = archive.get("s3_key")
        print(f"[{i}/{len(archives)}] {aid}  dur={dur/60:.1f}m  "
              f"size={nbytes/1e6:.0f}MB  sr={archive.get('sample_rate')}", flush=True)

        rec: dict = {
            "session_id": archive.get("session_id"),
            "archive_id": aid,
            "s3_key": key,
            "object_size_bytes": nbytes,
            "duration_sec": dur,
            "src_sample_rate": archive.get("sample_rate"),
            "src_channels": archive.get("channels"),
            "storage_class": archive.get("storage_class"),
        }

        if not args.skip_download and key:
            dl = benchmark_download(
                key, bucket=args.bucket, s5cmd=args.s5cmd,
                tmp_dir=tmp_dir, trials=args.download_trials, nbytes=nbytes,
            )
            rec["download"] = dl
            for name in ("boto3_default", "boto3_tuned", "s5cmd"):
                m = dl.get(name, {})
                print(f"    dl {name:14s} median {m.get('median_sec')}s  "
                      f"{m.get('median_mbps')} MB/s", flush=True)

        if not args.skip_decode:
            src = _ensure_local(archive, bucket=args.bucket, s5cmd=args.s5cmd, tmp_dir=tmp_dir)
            if src is None:
                print("    decode: no local source and no s3_key; skipping", flush=True)
            else:
                dec = benchmark_decode(
                    src, sample_rate=args.sample_rate,
                    duration_sec=dur, trials=args.decode_trials,
                )
                rec["decode"] = dec
                lib, ff = dec["librosa"], dec["ffmpeg"]
                nm = dec["numeric"]
                print(f"    decode librosa median {lib['median_sec']}s "
                      f"({lib['rtf']}x rt)  ffmpeg {ff['median_sec']}s "
                      f"({ff['rtf']}x rt)  speedup {dec['speedup_ffmpeg_over_librosa']}x",
                      flush=True)
                print(f"    numeric  max|Δ|={nm.get('max_abs_diff'):.3e}  "
                      f"rms={nm.get('rms_diff'):.3e}  "
                      f">1e-3:{nm.get('frac_gt_1e_3'):.3f}  sha_match={nm['sha_match']}",
                      flush=True)

        # end-to-end realistic combos from medians
        if "download" in rec and "decode" in rec:
            cur_dl = rec["download"]["boto3_default"]["median_sec"]
            opt_dl = rec["download"]["s5cmd"]["median_sec"]
            cur_dec = rec["decode"]["librosa"]["median_sec"]
            opt_dec = rec["decode"]["ffmpeg"]["median_sec"]
            if None not in (cur_dl, opt_dl, cur_dec, opt_dec):
                cur = cur_dl + cur_dec
                opt = opt_dl + opt_dec
                rec["end_to_end"] = {
                    "current_boto3_librosa_sec": round(cur, 3),
                    "optimized_s5cmd_ffmpeg_sec": round(opt, 3),
                    "speedup": round(cur / opt, 2) if opt else None,
                    "saved_sec": round(cur - opt, 3),
                }
                print(f"    end-to-end  current {cur:.1f}s -> optimized {opt:.1f}s  "
                      f"({rec['end_to_end']['speedup']}x, -{cur-opt:.1f}s)", flush=True)

        results.append(rec)
        gc.collect()
        print(flush=True)

    report = {"meta": meta, "archives": results, "summary": _summarize(results)}
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"wrote {out}", flush=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


def _summarize(results: list[dict]) -> dict:
    """Aggregate medians across archives for a quick headline."""
    def collect(path: list[str]) -> list[float]:
        out = []
        for r in results:
            node = r
            for k in path:
                node = node.get(k, {}) if isinstance(node, dict) else {}
            if isinstance(node, (int, float)):
                out.append(float(node))
        return out

    summary = {"n_archives": len(results)}
    dl_default = collect(["download", "boto3_default", "median_mbps"])
    dl_s5 = collect(["download", "s5cmd", "median_mbps"])
    if dl_default:
        summary["median_mbps_boto3_default"] = _median(dl_default)
    if dl_s5:
        summary["median_mbps_s5cmd"] = _median(dl_s5)
    speedups = collect(["decode", "speedup_ffmpeg_over_librosa"])
    if speedups:
        summary["median_decode_speedup_ffmpeg"] = _median(speedups)
    e2e = collect(["end_to_end", "speedup"])
    if e2e:
        summary["median_end_to_end_speedup"] = _median(e2e)
    maxdiffs = collect(["decode", "numeric", "max_abs_diff"])
    if maxdiffs:
        summary["max_decode_abs_diff"] = round(max(maxdiffs), 5)
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
