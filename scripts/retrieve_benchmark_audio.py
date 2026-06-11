#!/usr/bin/env python3
"""Retrieve a few real audio archives from S3 for inference benchmarking.

Samples non-deleted archives from the manifest parquet, resolves + downloads
the best audio per archive via the production resolver, and drops them in a
local directory with an ``index.json`` sidecar describing each file (S3 key,
size, sample rate, duration). Glacier / missing / failed archives are skipped
and the script keeps sampling until ``--n`` successes are collected.

Example:
    uv run python scripts/retrieve_benchmark_audio.py --n 5 --out-dir benchmark_audio
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from audio_classification_playground.acoustic_events.orchestration.audio_resolver import (
    AudioDownloadResult,
    resolve_and_download,
)

DEFAULT_PARQUET = "/efs/dvir/data/magic-clips-research/dataset-reference/all_archives.parquet"


def _probe(path: Path) -> dict:
    """Return sample_rate / duration_sec without a full decode when possible."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration_sec": round(float(info.frames) / float(info.samplerate), 3),
            "format": info.format,
        }
    except Exception as exc:  # mp3 / odd containers may not probe via soundfile
        try:
            import librosa

            return {"duration_sec": round(float(librosa.get_duration(path=str(path))), 3),
                    "probe_note": f"soundfile failed ({exc}); used librosa"}
        except Exception as exc2:
            return {"probe_error": f"{exc} / {exc2}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--n", type=int, default=5, help="Number of audios to retrieve.")
    ap.add_argument("--out-dir", default="benchmark_audio")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed for reproducibility.")
    ap.add_argument("--max-attempts", type=int, default=40,
                    help="Cap on archives tried before giving up.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["session_id", "archive_id", "file_parent_dir", "is_deleted", "date"]
    table = pq.read_table(args.parquet, columns=cols)
    rows = table.to_pylist()
    rows = [r for r in rows if not r.get("is_deleted")]
    print(f"manifest: {len(rows)} non-deleted archives", flush=True)

    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)

    collected: list[dict] = []
    attempts = 0
    for idx in order:
        if len(collected) >= args.n or attempts >= args.max_attempts:
            break
        attempts += 1
        r = rows[idx]
        sid, aid, fpd = r["session_id"], r["archive_id"], r["file_parent_dir"]
        res = resolve_and_download(sid, aid, fpd)
        if not isinstance(res, AudioDownloadResult):
            print(f"  skip {sid}/{aid}: {res.error_type} ({res.detail})", flush=True)
            continue
        ext = res.source_extension or Path(res.s3_key).suffix or ".wav"
        dest = out_dir / f"{sid}__{aid}{ext}"
        shutil.move(str(res.local_path), str(dest))
        meta = {
            "session_id": sid,
            "archive_id": aid,
            "file_parent_dir": fpd,
            "s3_key": res.s3_key,
            "local_path": str(dest),
            "object_size_bytes": res.object_size_bytes,
            "storage_class": res.storage_class,
            "download_sec": round(res.download_sec, 3),
            **_probe(dest),
        }
        collected.append(meta)
        size_mb = (res.object_size_bytes or 0) / 1e6
        print(f"  got  {dest.name}  {size_mb:.1f} MB  "
              f"{meta.get('duration_sec', '?')}s  sr={meta.get('sample_rate', '?')}", flush=True)

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(collected, indent=2))
    print(f"\nretrieved {len(collected)}/{args.n} in {attempts} attempts -> {index_path}", flush=True)
    return 0 if collected else 1


if __name__ == "__main__":
    raise SystemExit(main())
