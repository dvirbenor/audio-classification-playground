#!/usr/bin/env python3
"""Verify the restore status of S3 Glacier / Deep Archive objects.

Reads a CSV with an ``s3_key`` column (the layout of ``wav_glacier_keys.csv`` /
``wav_full_keys.csv``) and heads each object to report whether its restore has
completed. The readiness signal is the ``Restore`` response header, the same
string ``audio_resolver`` gates on: ``ongoing-request="false"`` means a readable
copy exists and the inference fleet will pick the archive up on its next pass.

Per-key status:
  ready          restore complete, readable now (ongoing-request="false")
  in_progress    restore requested, copy not ready yet (ongoing-request="true")
  not_requested  still archived, no restore initiated
  not_archived   already in a live class (STANDARD/etc.) -- nothing to restore
  error:<code>   head_object failed (e.g. missing object, access denied)

Usage:
    uv run python scripts/check_glacier_restore.py --csv wav_glacier_keys.csv
    uv run python scripts/check_glacier_restore.py --csv wav_glacier_keys.csv --out-prefix restore_status
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Default source bucket, matching orchestration/audio_resolver.py:27.
BUCKET = "riverside-pro-main"

READY = "ready"
IN_PROGRESS = "in_progress"
NOT_REQUESTED = "not_requested"
NOT_ARCHIVED = "not_archived"


def _s3_client(max_pool: int):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        config=Config(
            max_pool_connections=max_pool,
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )


def _read_keys(csv_path: Path) -> list[str]:
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "s3_key" not in reader.fieldnames:
            raise SystemExit(f"{csv_path} must have an 's3_key' column; got {reader.fieldnames}")
        has_glacier = "is_glacier" in reader.fieldnames
        keys = []
        for row in reader:
            key = (row.get("s3_key") or "").strip()
            if not key:
                continue
            if has_glacier and (row.get("is_glacier") or "").strip().lower() not in ("true", "1", "yes"):
                continue
            keys.append(key)
    return list(dict.fromkeys(keys))


def check_one(s3, key: str, bucket: str) -> tuple[str, str]:
    from botocore.exceptions import ClientError

    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        return key, f"error:{exc.response.get('Error', {}).get('Code', exc)}"
    sc = head.get("StorageClass", "STANDARD")
    restore = head.get("Restore", "")
    if sc not in ("GLACIER", "DEEP_ARCHIVE"):
        return key, NOT_ARCHIVED
    if 'ongoing-request="false"' in restore:
        return key, READY
    if 'ongoing-request="true"' in restore:
        return key, IN_PROGRESS
    return key, NOT_REQUESTED


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("wav_glacier_keys.csv"),
                    help="CSV with an s3_key column (default: wav_glacier_keys.csv).")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--workers", type=int, default=32, help="Concurrent head_object requests.")
    ap.add_argument("--limit", type=int, default=None, help="Only check the first N keys.")
    ap.add_argument("--out-prefix", default=None,
                    help="If set, write <prefix>.<status>.txt files listing keys per status.")
    args = ap.parse_args(argv)

    keys = _read_keys(args.csv)
    if args.limit:
        keys = keys[: args.limit]
    print(f"Checking {len(keys)} key(s) against s3://{args.bucket}", file=sys.stderr)

    s3 = _s3_client(args.workers)
    by_status: dict[str, list[str]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(check_one, s3, k, args.bucket) for k in keys]
        for i, fut in enumerate(as_completed(futs), 1):
            key, status = fut.result()
            base = "error" if status.startswith("error") else status
            by_status[base].append(key)
            if i % 1000 == 0:
                print(f"  checked {i}/{len(keys)}", file=sys.stderr)

    counts = {k: len(v) for k, v in sorted(by_status.items())}
    total = sum(counts.values())
    ready = counts.get(READY, 0)
    print(json.dumps(counts, indent=2))
    print(f"\n{ready}/{total} ready ({100 * ready / total:.1f}%)" if total else "no keys", file=sys.stderr)

    if args.out_prefix:
        for status, ks in by_status.items():
            out = Path(f"{args.out_prefix}.{status}.txt")
            out.write_text("\n".join(ks) + "\n")
            print(f"  wrote {len(ks)} -> {out}", file=sys.stderr)

    # Exit 0 only when everything is ready (or nothing needed restoring).
    pending = total - ready - counts.get(NOT_ARCHIVED, 0)
    return 0 if pending == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
