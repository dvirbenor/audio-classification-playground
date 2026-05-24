"""Command-line tools for atomic event packages."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import uuid

from ..orchestration.locking import reclaim_stale, release_claim, try_claim
from ..orchestration.manifest import ArchiveEntity, load_manifest
from .errors import (
    append_packaging_error,
    load_packaging_attempt_counts,
    packaging_error_archives,
)
from .eventify import (
    EventPackageConfigs,
    eventify_archive,
    required_artifacts_complete,
)
from .package import (
    append_completion_row,
    compact_completion_index,
    completion_row,
    completion_rows_from_index_and_shards,
    event_package_complete,
    load_event_package,
)


DEFAULT_POLL_INTERVAL_SEC = 300.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_LOG_EVERY = 1000
LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command.replace("-", "_")
    return globals()[f"_cmd_{command}"](args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acoustic-event-packages",
        description="Build compact atomic event packages from inference artifacts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eventify = sub.add_parser("eventify", help="Build one archive event package.")
    p_eventify.add_argument("--inference-archive", required=True, type=Path)
    p_eventify.add_argument("--events-archive", required=True, type=Path)
    p_eventify.add_argument("--session-id")
    p_eventify.add_argument("--archive-id")
    p_eventify.add_argument("--date", default="")
    _add_config_args(p_eventify)
    p_eventify.add_argument("--force", action="store_true")
    p_eventify.add_argument("--validate-inputs", action="store_true")

    p_run = sub.add_parser("run", help="Run a continuous or one-pass CPU event package worker.")
    p_run.add_argument("--parquet", required=True, type=Path)
    p_run.add_argument("--inference-output", required=True, type=Path)
    p_run.add_argument("--events-output", required=True, type=Path)
    p_run.add_argument("--watch", action="store_true")
    p_run.add_argument("--poll-interval-sec", type=float, default=DEFAULT_POLL_INTERVAL_SEC)
    p_run.add_argument("--num-shards", type=int, default=1)
    p_run.add_argument("--shard-index", type=int, default=0)
    p_run.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    p_run.add_argument("--verbose", action="store_true")
    p_run.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="With --verbose, log a pass-progress line every N in-shard archives. Use 0 to disable interval logs.",
    )
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--validate-inputs", action="store_true")
    p_run.add_argument("--retry-failed", action="store_true")
    p_run.add_argument("--reclaim-stale-minutes", type=float, default=60.0)
    _add_config_args(p_run)

    p_compact = sub.add_parser("compact-index", help="Compact completion shards to parquet.")
    p_compact.add_argument("--events-output", required=True, type=Path)

    p_progress = sub.add_parser("progress", help="Report event package coverage from indexes.")
    p_progress.add_argument("--parquet", required=True, type=Path)
    p_progress.add_argument("--events-output", required=True, type=Path)

    p_reconcile = sub.add_parser("reconcile-index", help="Compare or repair index rows from package truth.")
    p_reconcile.add_argument("--parquet", required=True, type=Path)
    p_reconcile.add_argument("--events-output", required=True, type=Path)
    p_reconcile.add_argument("--dry-run", action="store_true")

    p_reclaim = sub.add_parser("reclaim-stale-locks", help="Remove orphan event package locks.")
    p_reclaim.add_argument("--events-output", required=True, type=Path)
    p_reclaim.add_argument("--older-than", type=float, default=60.0)
    return parser


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--affect-config", type=Path)
    parser.add_argument("--disfluency-config", type=Path)
    parser.add_argument("--emotion-config", type=Path)


def _cmd_eventify(args: argparse.Namespace) -> int:
    session_id = args.session_id or args.inference_archive.parent.name
    archive_id = args.archive_id or args.inference_archive.name
    result = eventify_archive(
        inference_archive_dir=args.inference_archive,
        output_archive_dir=args.events_archive,
        session_id=session_id,
        archive_id=archive_id,
        date=args.date,
        configs=_configs(args),
        force=args.force,
        validate_inputs=args.validate_inputs,
    )
    print(json.dumps(_result_payload(result), indent=2, sort_keys=True, default=str))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    _configure_logging(verbose=args.verbose)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be >= 1")
    if args.log_every < 0:
        raise ValueError("--log-every must be >= 0")

    worker_id = _worker_id()
    configs = _configs(args)
    entities = load_manifest(args.parquet)
    LOGGER.info(
        "event-package worker start worker_id=%s manifest_entities=%d shard_index=%d num_shards=%d watch=%s",
        worker_id,
        len(entities),
        args.shard_index,
        args.num_shards,
        bool(args.watch),
    )
    pass_index = 0
    while True:
        pass_index += 1
        start = time.monotonic()
        summary = _run_one_pass(
            entities=entities,
            inference_output=args.inference_output,
            events_output=args.events_output,
            configs=configs,
            worker_id=worker_id,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            force=args.force,
            validate_inputs=args.validate_inputs,
            max_attempts=args.max_attempts,
            retry_failed=args.retry_failed,
            reclaim_stale_minutes=args.reclaim_stale_minutes,
            verbose=args.verbose,
            log_every=args.log_every,
        )
        elapsed = time.monotonic() - start
        LOGGER.info(
            "event-package pass complete pass=%d elapsed_sec=%.2f summary=%s",
            pass_index,
            elapsed,
            json.dumps(summary, sort_keys=True),
        )
        print(json.dumps(summary, sort_keys=True))
        if not args.watch:
            return 0
        time.sleep(max(0.0, float(args.poll_interval_sec)))


def _run_one_pass(
    *,
    entities: list[ArchiveEntity],
    inference_output: Path,
    events_output: Path,
    configs: EventPackageConfigs,
    worker_id: str,
    num_shards: int,
    shard_index: int,
    force: bool,
    validate_inputs: bool,
    max_attempts: int,
    retry_failed: bool,
    reclaim_stale_minutes: float,
    verbose: bool = False,
    log_every: int = DEFAULT_LOG_EVERY,
) -> dict:
    reclaim_stale(
        events_output,
        older_than_minutes=reclaim_stale_minutes,
        is_complete_fn=lambda sid, aid: event_package_complete(events_output / sid / aid),
    )
    attempt_counts = load_packaging_attempt_counts(events_output)
    event_cfg_fp = configs.config_fingerprint()
    indexed_complete = set()
    if not force and not validate_inputs:
        try:
            indexed_complete = {
                (str(row.get("session_id", "")), str(row.get("archive_id", "")))
                for row in completion_rows_from_index_and_shards(events_output)
                if row.get("event_config_fingerprint") == event_cfg_fp
            }
        except Exception:
            indexed_complete = set()
    if verbose:
        LOGGER.info(
            "event-package pass start worker_id=%s total_entities=%d shard_index=%d num_shards=%d indexed_complete=%d force=%s validate_inputs=%s",
            worker_id,
            len(entities),
            shard_index,
            num_shards,
            len(indexed_complete),
            bool(force),
            bool(validate_inputs),
        )
    summary = {
        "processed": 0,
        "packaged": 0,
        "skipped_complete": 0,
        "not_ready": 0,
        "locked": 0,
        "failed": 0,
        "failed_exhausted": 0,
        "out_of_shard": 0,
    }
    scanned = 0
    in_shard_seen = 0
    for entity in entities:
        scanned += 1
        if _shard_for(entity.session_id, entity.archive_id, num_shards) != shard_index:
            summary["out_of_shard"] += 1
            continue
        in_shard_seen += 1
        key = (entity.session_id, entity.archive_id)
        attempts = attempt_counts.get(key, 0)
        if attempts >= max_attempts and not retry_failed:
            summary["failed_exhausted"] += 1
            _log_verbose_progress(
                verbose=verbose,
                log_every=log_every,
                worker_id=worker_id,
                scanned=scanned,
                in_shard_seen=in_shard_seen,
                summary=summary,
            )
            continue
        if key in indexed_complete:
            summary["skipped_complete"] += 1
            _log_verbose_progress(
                verbose=verbose,
                log_every=log_every,
                worker_id=worker_id,
                scanned=scanned,
                in_shard_seen=in_shard_seen,
                summary=summary,
            )
            continue

        inference_archive = inference_output / entity.session_id / entity.archive_id
        events_archive = events_output / entity.session_id / entity.archive_id
        if not force and not validate_inputs and event_package_complete(events_archive):
            try:
                package = load_event_package(events_archive)
                if package.event_config_fingerprint == event_cfg_fp:
                    append_completion_row(
                        events_base=events_output,
                        worker_id=worker_id,
                        package_dir=events_archive,
                        package_payload=package.package,
                    )
                    summary["skipped_complete"] += 1
                    _log_verbose_progress(
                        verbose=verbose,
                        log_every=log_every,
                        worker_id=worker_id,
                        scanned=scanned,
                        in_shard_seen=in_shard_seen,
                        summary=summary,
                    )
                    continue
            except Exception:
                pass

        if not force and not required_artifacts_complete(inference_archive):
            summary["not_ready"] += 1
            _log_verbose_progress(
                verbose=verbose,
                log_every=log_every,
                worker_id=worker_id,
                scanned=scanned,
                in_shard_seen=in_shard_seen,
                summary=summary,
            )
            continue

        if not try_claim(events_output, entity, task_group="event-package"):
            summary["locked"] += 1
            _log_verbose_progress(
                verbose=verbose,
                log_every=log_every,
                worker_id=worker_id,
                scanned=scanned,
                in_shard_seen=in_shard_seen,
                summary=summary,
            )
            continue
        try:
            result = eventify_archive(
                inference_archive_dir=inference_archive,
                output_archive_dir=events_archive,
                session_id=entity.session_id,
                archive_id=entity.archive_id,
                date=entity.date,
                configs=configs,
                force=force,
                validate_inputs=validate_inputs,
            )
            summary["processed"] += 1
            summary[result.status] = summary.get(result.status, 0) + 1
            if result.package_payload is not None and result.package_path is not None:
                append_completion_row(
                    events_base=events_output,
                    worker_id=worker_id,
                    package_dir=result.package_path,
                    package_payload=result.package_payload,
                )
                if verbose and log_every == 1:
                    counts = result.package_payload.get("counts", {})
                    LOGGER.info(
                        "event-package wrote session_id=%s archive_id=%s events=%s path=%s",
                        entity.session_id,
                        entity.archive_id,
                        counts.get("events_total", len(result.events)),
                        result.package_path,
                    )
        except Exception as exc:
            summary["failed"] += 1
            append_packaging_error(
                events_output,
                session_id=entity.session_id,
                archive_id=entity.archive_id,
                error=exc,
            )
            attempt_counts[key] += 1
            LOGGER.exception(
                "event-package failed session_id=%s archive_id=%s",
                entity.session_id,
                entity.archive_id,
            )
        finally:
            release_claim(events_output, entity)
        _log_verbose_progress(
            verbose=verbose,
            log_every=log_every,
            worker_id=worker_id,
            scanned=scanned,
            in_shard_seen=in_shard_seen,
            summary=summary,
        )
    return summary


def _cmd_compact_index(args: argparse.Namespace) -> int:
    path = compact_completion_index(args.events_output)
    print(path)
    return 0


def _cmd_progress(args: argparse.Namespace) -> int:
    entities = load_manifest(args.parquet)
    rows = completion_rows_from_index_and_shards(args.events_output)
    completed = {(row["session_id"], row["archive_id"]) for row in rows}
    errors = packaging_error_archives(args.events_output)
    total_keys = {(e.session_id, e.archive_id) for e in entities}
    payload = {
        "total": len(total_keys),
        "completed": len(completed & total_keys),
        "packaging_errors": len(errors & total_keys),
        "remaining": len(total_keys - completed),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_reconcile_index(args: argparse.Namespace) -> int:
    entities = load_manifest(args.parquet)
    rows = completion_rows_from_index_and_shards(args.events_output)
    indexed = {
        (str(row["session_id"]), str(row["archive_id"])): row
        for row in rows
    }
    manifest_keys = {(entity.session_id, entity.archive_id) for entity in entities}
    missing: list[dict] = []
    indexed_missing_packages: list[dict] = []
    fingerprint_mismatches: list[dict] = []
    foreign_index_rows = [
        {"session_id": sid, "archive_id": aid}
        for sid, aid in sorted(set(indexed) - manifest_keys)
    ]
    worker_id = _worker_id()
    repaired = 0
    for entity in entities:
        key = (entity.session_id, entity.archive_id)
        package_dir = args.events_output / entity.session_id / entity.archive_id
        try:
            package = load_event_package(package_dir)
        except Exception:
            if key in indexed:
                indexed_missing_packages.append(
                    {"session_id": entity.session_id, "archive_id": entity.archive_id}
                )
            continue

        if key not in indexed:
            missing.append({"session_id": entity.session_id, "archive_id": entity.archive_id})
            if not args.dry_run:
                append_completion_row(
                    events_base=args.events_output,
                    worker_id=worker_id,
                    package_dir=package_dir,
                    package_payload=package.package,
                )
                repaired += 1
            continue

        expected = completion_row(package_dir, package.package)
        observed = indexed[key]
        fingerprint_fields = (
            "event_config_fingerprint",
            "input_fingerprint",
            "event_package_fingerprint",
        )
        if any(observed.get(field) != expected.get(field) for field in fingerprint_fields):
            fingerprint_mismatches.append(
                {
                    "session_id": entity.session_id,
                    "archive_id": entity.archive_id,
                    "indexed_event_package_fingerprint": observed.get("event_package_fingerprint", ""),
                    "package_event_package_fingerprint": expected.get("event_package_fingerprint", ""),
                }
            )
            if not args.dry_run:
                append_completion_row(
                    events_base=args.events_output,
                    worker_id=worker_id,
                    package_dir=package_dir,
                    package_payload=package.package,
                )
                repaired += 1

    payload = {
        "missing_index_rows": missing,
        "indexed_missing_packages": indexed_missing_packages,
        "fingerprint_mismatches": fingerprint_mismatches,
        "foreign_index_rows": foreign_index_rows,
        "counts": {
            "missing_index_rows": len(missing),
            "indexed_missing_packages": len(indexed_missing_packages),
            "fingerprint_mismatches": len(fingerprint_mismatches),
            "foreign_index_rows": len(foreign_index_rows),
            "repaired_rows_appended": repaired,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_reclaim_stale_locks(args: argparse.Namespace) -> int:
    reclaimed = reclaim_stale(
        args.events_output,
        older_than_minutes=args.older_than,
        is_complete_fn=lambda sid, aid: event_package_complete(args.events_output / sid / aid),
    )
    print(json.dumps({"reclaimed": reclaimed}, sort_keys=True))
    return 0


def _configs(args: argparse.Namespace) -> EventPackageConfigs:
    return EventPackageConfigs.from_paths(
        affect=args.affect_config,
        disfluency=args.disfluency_config,
        emotion=args.emotion_config,
    )


def _result_payload(result) -> dict:
    return {
        "status": result.status,
        "package_path": str(result.package_path) if result.package_path else None,
        "event_count": len(result.events),
        "reason": result.reason,
    }


def _worker_id() -> str:
    host = os.environ.get("HOSTNAME", "unknown")
    return f"{host}_{os.getpid()}_{uuid.uuid4().hex[:12]}"


def _shard_for(session_id: str, archive_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(f"{session_id}\0{archive_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(num_shards)


def _configure_logging(*, verbose: bool) -> None:
    if not verbose:
        logging.getLogger().setLevel(logging.WARNING)
        return
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(getattr(handler, "_event_packages_cli", False) for handler in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        handler._event_packages_cli = True  # type: ignore[attr-defined]
        root.addHandler(handler)


def _log_verbose_progress(
    *,
    verbose: bool,
    log_every: int,
    worker_id: str,
    scanned: int,
    in_shard_seen: int,
    summary: dict,
) -> None:
    if not verbose or log_every <= 0 or in_shard_seen % log_every != 0:
        return
    LOGGER.info(
        "event-package pass progress worker_id=%s scanned=%d in_shard_seen=%d packaged=%d skipped_complete=%d not_ready=%d locked=%d failed=%d failed_exhausted=%d",
        worker_id,
        scanned,
        in_shard_seen,
        summary.get("packaged", 0),
        summary.get("skipped_complete", 0),
        summary.get("not_ready", 0),
        summary.get("locked", 0),
        summary.get("failed", 0),
        summary.get("failed_exhausted", 0),
    )


if __name__ == "__main__":
    raise SystemExit(main())
