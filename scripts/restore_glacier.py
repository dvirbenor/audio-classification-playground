#!/usr/bin/env python3
"""Initiate S3 Glacier / Deep Archive restores for the keys in a CSV.

Reads a CSV with an ``s3_key`` column (the layout of ``wav_full_keys.csv`` /
``wav_glacier_keys.csv``) and issues an asynchronous ``restore_object`` for each
object. Restore is per-object and async: the call returns immediately and the
temporary readable copy appears hours later (Deep Archive Bulk <= 48h, Standard
<= 12h). Once a restore completes, ``audio_resolver`` stops raising
``glacier_storage_class`` and the inference fleet picks the archive up on its
next pass automatically -- no code change needed.

This script is idempotent:
  * already restored / in-progress objects are treated as success (re-issuing
    just extends the readable window by ``--days``);
  * a ``--state`` JSON file records which keys have been requested so reruns
    skip them.

Usage:
    uv run python -m scripts.restore_glacier --csv wav_glacier_keys.csv
    uv run python -m scripts.restore_glacier --csv wav_glacier_keys.csv --status   # poll readiness
    uv run python -m scripts.restore_glacier --csv wav_glacier_keys.csv --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Default source bucket, matching orchestration/audio_resolver.py:27.
BUCKET = "riverside-pro-main"


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
            # If the file carries the is_glacier flag, only restore those.
            if has_glacier and (row.get("is_glacier") or "").strip().lower() not in ("true", "1", "yes"):
                continue
            keys.append(key)
    # De-dupe, preserve order.
    return list(dict.fromkeys(keys))


def _load_state(path: Path | None) -> set[str]:
    if path and path.exists():
        return set(json.loads(path.read_text()))
    return set()


def _save_state(path: Path | None, done: set[str]) -> None:
    if path:
        path.write_text(json.dumps(sorted(done)))


def _pick_tier(s3, key: str, bucket: str) -> tuple[str, str | None]:
    """For tier='fastest', head the object and pick the quickest tier it supports.

    Expedited (1-5 min) is only valid for GLACIER (Flexible Retrieval); Deep
    Archive does not support it, so Standard (<=12h) is the floor there.
    Returns (tier, skip_status) -- skip_status set if no restore is needed.
    """
    from botocore.exceptions import ClientError

    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        return "Standard", f"error:{exc.response.get('Error', {}).get('Code', exc)}"
    sc = head.get("StorageClass", "STANDARD")
    restore = head.get("Restore", "")
    if sc not in ("GLACIER", "DEEP_ARCHIVE"):
        return "Standard", "not_archived"
    if 'ongoing-request="false"' in restore:
        return "Standard", "already_restored"
    if 'ongoing-request="true"' in restore:
        return "Standard", "already_in_progress"
    return ("Expedited" if sc == "GLACIER" else "Standard"), None


def restore_one(s3, key: str, bucket: str, days: int, tier: str) -> tuple[str, str]:
    """Return (key, status). status in {requested, already_in_progress, already_restored, error:...}."""
    from botocore.exceptions import ClientError

    if tier == "fastest":
        tier, skip = _pick_tier(s3, key, bucket)
        if skip:
            return key, skip

    try:
        s3.restore_object(
            Bucket=bucket,
            Key=key,
            RestoreRequest={"Days": days, "GlacierJobParameters": {"Tier": tier}},
        )
        return key, f"requested:{tier}"
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "RestoreAlreadyInProgress":
            return key, "already_in_progress"
        # Object not in an archive class / already restored copy present.
        if code in ("InvalidObjectState", "RestoreAlreadyCompleted"):
            return key, "already_restored"
        return key, f"error:{code or exc}"


def check_one(s3, key: str, bucket: str) -> tuple[str, str]:
    """Poll readiness via head_object Restore header. Matches audio_resolver gate."""
    from botocore.exceptions import ClientError

    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        return key, f"error:{exc.response.get('Error', {}).get('Code', exc)}"
    sc = head.get("StorageClass", "STANDARD")
    restore = head.get("Restore", "")
    if sc not in ("GLACIER", "DEEP_ARCHIVE"):
        return key, "not_archived"
    if 'ongoing-request="false"' in restore:
        return key, "ready"
    if 'ongoing-request="true"' in restore:
        return key, "in_progress"
    return key, "not_requested"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("wav_glacier_keys.csv"),
                    help="CSV with an s3_key column (default: wav_glacier_keys.csv).")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--days", type=int, default=7, help="Days the restored copy stays readable.")
    ap.add_argument("--tier", default="fastest", choices=["fastest", "Expedited", "Standard", "Bulk"],
                    help="Retrieval tier. 'fastest' (default) heads each object and uses Expedited for "
                         "GLACIER (1-5 min) and Standard for DEEP_ARCHIVE (<=12h; Expedited unsupported).")
    ap.add_argument("--workers", type=int, default=32, help="Concurrent S3 requests.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N keys.")
    ap.add_argument("--sample", type=int, default=None,
                    help="Check a random sample of N keys instead of all (status estimate).")
    ap.add_argument("--state", type=Path, default=Path("restore_glacier_state.json"),
                    help="JSON file of already-requested keys (skipped on rerun). Pass '' to disable.")
    ap.add_argument("--status", action="store_true",
                    help="Poll readiness instead of issuing restores.")
    ap.add_argument("--dry-run", action="store_true", help="List what would be done and exit.")
    args = ap.parse_args(argv)

    state_path = args.state if str(args.state) else None
    keys = _read_keys(args.csv)
    if args.sample and args.sample < len(keys):
        import random

        keys = random.sample(keys, args.sample)
        print(f"random sample of {len(keys)} key(s) from {args.csv}", file=sys.stderr)
    if args.limit:
        keys = keys[: args.limit]
    if not args.sample:
        print(f"{len(keys)} key(s) from {args.csv}", file=sys.stderr)

    if args.status:
        s3 = _s3_client(args.workers)
        counts: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(check_one, s3, k, args.bucket) for k in keys]
            for i, fut in enumerate(as_completed(futs), 1):
                _, status = fut.result()
                base = status.split(":", 1)[0]
                counts[base] = counts.get(base, 0) + 1
                if i % 500 == 0:
                    print(f"  checked {i}/{len(keys)}", file=sys.stderr)
        total = sum(counts.values())
        print(json.dumps(counts, indent=2))
        if total:
            pct = {k: round(100 * v / total, 1) for k, v in sorted(counts.items())}
            print(f"pct: {json.dumps(pct)}", file=sys.stderr)
        return 0

    done = _load_state(state_path)
    pending = [k for k in keys if k not in done]
    print(f"{len(pending)} pending after state ({len(done)} already requested)", file=sys.stderr)

    if args.dry_run:
        for k in pending[:20]:
            print(f"  would restore tier={args.tier} days={args.days}: {k}")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        return 0

    s3 = _s3_client(args.workers)
    print(
        f"submitting {len(pending)} restore requests  "
        f"(workers={args.workers}, tier={args.tier}, days={args.days})",
        file=sys.stderr,
    )
    counts: dict[str, int] = {}
    lock = threading.Lock()
    processed = 0
    interval = max(100, min(500, len(pending) // 20))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(restore_one, s3, k, args.bucket, args.days, args.tier): k for k in pending}
        for fut in as_completed(futs):
            key, status = fut.result()
            base = status.split(":", 1)[0]
            with lock:
                counts[base] = counts.get(base, 0) + 1
                # Treat any non-hard-error outcome as "requested" for state purposes.
                if not status.startswith("error"):
                    done.add(key)
                processed += 1
                if processed % interval == 0:
                    summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                    print(f"  {processed}/{len(pending)}  {summary}", file=sys.stderr)
                    _save_state(state_path, done)
            if status.startswith("error"):
                print(f"  {status}  {key}", file=sys.stderr)

    _save_state(state_path, done)
    print(json.dumps(counts, indent=2))
    return 0 if not any(k.startswith("error") for k in counts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
