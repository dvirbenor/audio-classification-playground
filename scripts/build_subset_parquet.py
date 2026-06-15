#!/usr/bin/env python3
"""Build a subset of all_archives.parquet, by count or by VAD presence/absence.

Two uses:

1. Controlled benchmarks — a bounded, FIXED set so every arm does identical work.
   Deduplicates by (session_id, archive_id) and takes the first N (or a seeded sample).
   ``--vad-present-under`` keeps only archives that ALREADY have a ``vad/`` artifact under
   a given output tree (so gating is guaranteed available — the point of the gated regime).

2. VAD backfill work-list — ``--vad-absent-under`` keeps only archives that are NOT yet
   VAD-done under an output tree, i.e. the UNDONE set. Feeding this to the backfill workers
   instead of all_archives.parquet means they scan only undone work (a ~100% claim hit rate
   at ANY completion level) instead of statting through the already-done majority every pass.
   Use with ``--all`` to keep every undone archive (no count cap). The presence/absence check
   is the SAME primitive the worker's ``--completion-policy exists`` uses, so the filter and
   the worker agree exactly — nothing is wrongly dropped.

Examples:
    # first 40 archives
    python scripts/build_subset_parquet.py --parquet all_archives.parquet \
        --out /scratch/subset.parquet --n 40

    # 40 archives that already have VAD computed in the prod tree (gating guaranteed)
    python scripts/build_subset_parquet.py --parquet all_archives.parquet \
        --out /scratch/subset_vad.parquet --n 40 \
        --vad-present-under /efs/.../models-inference

    # ALL archives still missing VAD (the backfill work-list)
    python scripts/build_subset_parquet.py --parquet all_archives.parquet \
        --out /workspace/undone.parquet --all \
        --vad-absent-under /efs/.../models-inference
"""
from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from audio_classification_playground.acoustic_events.orchestration.progress import (
    is_task_artifact_complete_for_archive,
)

COLUMNS = ["session_id", "archive_id", "file_parent_dir", "date"]
_SCAN_WORKERS = 64


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True, help="Source all_archives.parquet")
    ap.add_argument("--out", required=True, help="Destination subset parquet")
    ap.add_argument("--n", type=int, default=40, help="Number of archives (default 40)")
    ap.add_argument("--all", action="store_true",
                    help="Keep ALL matching archives, ignoring --n (for the backfill work-list).")
    ap.add_argument("--seed", type=int, default=None,
                    help="If set, take a seeded random sample instead of the first N.")
    ap.add_argument("--vad-present-under", default=None,
                    help="Keep only archives that already have a vad/ artifact under "
                         "this output base (guarantees the gated regime).")
    ap.add_argument("--vad-absent-under", default=None,
                    help="Keep only archives MISSING a vad/ artifact under this output "
                         "base (the backfill work-list / undone set).")
    ap.add_argument("--scan-workers", type=int, default=_SCAN_WORKERS,
                    help="Parallel EFS-stat workers for the presence/absence scan.")
    args = ap.parse_args()

    if args.vad_present_under and args.vad_absent_under:
        raise SystemExit("Pass at most one of --vad-present-under / --vad-absent-under.")

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

    base_path = args.vad_present_under or args.vad_absent_under
    if base_path:
        base = Path(base_path)
        want_present = args.vad_present_under is not None
        # Parallel EFS stat — same primitive the worker's completion-policy=exists uses, so
        # the filter and the worker agree. Single-threaded over ~600k archives is far too slow.
        def _has_vad(i: int) -> bool:
            return is_task_artifact_complete_for_archive(base, sid[i], aid[i], "vad")

        pool_size = min(args.scan_workers, len(candidates)) or 1
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            present_flags = list(pool.map(_has_vad, candidates))
        candidates = [
            i for i, present in zip(candidates, present_flags)
            if present == want_present
        ]
        kind = "present" if want_present else "absent (undone)"
        print(f"{len(candidates)} archives have vad {kind} under {base}")
        if not candidates and want_present:
            raise SystemExit("No VAD-present archives found — run the vad task-group first.")

    if args.seed is not None:
        rng = random.Random(args.seed)
        rng.shuffle(candidates)

    keep = candidates if args.all else candidates[: args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(keep), out)
    print(f"wrote {len(keep)} archives -> {out}")


if __name__ == "__main__":
    main()
