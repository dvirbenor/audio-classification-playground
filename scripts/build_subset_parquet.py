#!/usr/bin/env python3
"""Build a small, fixed subset of all_archives.parquet for controlled benchmarks.

The manifest parquet has ~600k archives; a controlled A/B needs a bounded, FIXED set
so every arm does identical work. This deduplicates by (session_id, archive_id) and
takes the first N (or a seeded random sample).

Crucially for the MPS-vs-dedicated / VAD-gating benchmarks, ``--vad-present-under``
keeps only archives that ALREADY have a ``vad/`` artifact under a given output tree —
so you can build a subset on which gating is guaranteed available (no full-timeline
fallback), which is the whole point of measuring the gated regime.

Examples:
    # first 40 archives
    python scripts/build_subset_parquet.py --parquet all_archives.parquet \
        --out /scratch/subset.parquet --n 40

    # 40 archives that already have VAD computed in the prod tree (gating guaranteed)
    python scripts/build_subset_parquet.py --parquet all_archives.parquet \
        --out /scratch/subset_vad.parquet --n 40 \
        --vad-present-under /efs/.../models-inference
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pyarrow.parquet as pq

from audio_classification_playground.acoustic_events.orchestration.progress import (
    is_task_artifact_complete_for_archive,
)

COLUMNS = ["session_id", "archive_id", "file_parent_dir", "date"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True, help="Source all_archives.parquet")
    ap.add_argument("--out", required=True, help="Destination subset parquet")
    ap.add_argument("--n", type=int, default=40, help="Number of archives (default 40)")
    ap.add_argument("--seed", type=int, default=None,
                    help="If set, take a seeded random sample instead of the first N.")
    ap.add_argument("--vad-present-under", default=None,
                    help="Keep only archives that already have a vad/ artifact under "
                         "this output base (guarantees the gated regime).")
    args = ap.parse_args()

    table = pq.read_table(args.parquet, columns=COLUMNS)
    sid = table.column("session_id").to_pylist()
    aid = table.column("archive_id").to_pylist()

    # Dedup by (session_id, archive_id), preserving order.
    seen: set[tuple[str, str]] = set()
    candidates: list[int] = []
    for i, (s, a) in enumerate(zip(sid, aid)):
        key = (s, a)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(i)

    if args.vad_present_under:
        base = Path(args.vad_present_under)
        candidates = [
            i for i in candidates
            if is_task_artifact_complete_for_archive(base, sid[i], aid[i], "vad")
        ]
        print(f"{len(candidates)} archives have a vad/ artifact under {base}")
        if not candidates:
            raise SystemExit("No VAD-present archives found — run the vad task-group first.")

    if args.seed is not None:
        rng = random.Random(args.seed)
        rng.shuffle(candidates)

    keep = candidates[: args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(keep), out)
    print(f"wrote {len(keep)} archives -> {out}")


if __name__ == "__main__":
    main()
