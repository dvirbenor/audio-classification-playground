"""Command-line interface for the batch orchestration pipeline.

Subcommands:

    run             Start a worker that processes archives.
    progress        Print a summary of the current pipeline state.
    errors          List audio or inference errors.
    reclaim-stale   Remove orphan lock files from crashed pods.

Example::

    python -m audio_classification_playground.acoustic_events.orchestration run \
        --parquet /efs/dvir/data/magic-clips-research/dataset-reference/all_archives.parquet \
        --output  /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
        --affect-backbone wavlm \
        --disfluency-backbone whisper \
        --batch-size 512
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )


def _cmd_run(args: argparse.Namespace) -> None:
    from .worker import run_worker

    run_worker(
        parquet_path=args.parquet,
        output_base=args.output,
        affect_backbone=args.affect_backbone,
        disfluency_backbone=args.disfluency_backbone,
        batch_size=args.batch_size,
        affect_batch_size=args.affect_batch_size,
        disfluency_batch_size=args.disfluency_batch_size,
        emotion_batch_size=args.emotion_batch_size,
        device=args.device,
        max_inference_attempts=args.max_retries,
        prefetch_workers=args.prefetch_workers,
        prefetch_lookahead=args.prefetch_lookahead,
        vad_prefetch_workers=args.vad_prefetch_workers,
        seed=args.seed,
    )


def _cmd_progress(args: argparse.Namespace) -> None:
    from .errors import load_inference_attempt_counts, load_permanent_error_set
    from .manifest import load_manifest
    from .progress import scan_progress

    output_base = Path(args.output)
    entities = load_manifest(args.parquet)
    perm_errors = load_permanent_error_set(output_base)
    inf_errors = load_inference_attempt_counts(output_base)

    summary = scan_progress(
        output_base, entities,
        permanent_audio_errors=perm_errors,
        inference_error_counts=dict(inf_errors),
    )
    lines = [
        f"Total entities:          {summary.total_entities}",
        f"Complete:                {summary.complete}",
        f"Partial:                 {summary.partial}",
        f"Permanent audio errors:  {summary.permanent_audio_errors}",
        f"Archives with inf. errs: {summary.inference_errors_by_archive}",
        f"Currently locked:        {summary.locked}",
        f"Remaining:               {summary.remaining}",
        "",
        "Per-task completion:",
    ]
    for task, count in sorted(summary.task_counts.items()):
        lines.append(f"  {task:12s} {count}")
    print("\n".join(lines))


def _cmd_errors(args: argparse.Namespace) -> None:
    output_base = Path(args.output)
    kind = args.kind

    if kind == "audio":
        from .errors import AUDIO_ERRORS_DIR

        errors_dir = output_base / AUDIO_ERRORS_DIR
    else:
        from .errors import INFERENCE_ERRORS_DIR

        errors_dir = output_base / INFERENCE_ERRORS_DIR

    if not errors_dir.is_dir():
        print(f"No {kind} errors directory found at {errors_dir}")
        return

    count = 0
    for f in sorted(errors_dir.iterdir()):
        if not f.name.endswith(".json"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sid = data.get("session_id", "?")
            aid = data.get("archive_id", "?")
            etype = data.get("error_type", "?")
            detail = data.get("detail", "")[:120]
            print(f"  {sid}/{aid}  [{etype}]  {detail}")
            count += 1
        except (OSError, json.JSONDecodeError):
            continue

    if args.summary:
        print(f"\nTotal {kind} error records: {count}")


def _cmd_reclaim_stale(args: argparse.Namespace) -> None:
    from .locking import reclaim_stale
    from .progress import is_archive_complete

    output_base = Path(args.output)
    reclaimed = reclaim_stale(
        output_base,
        older_than_minutes=args.older_than,
        is_complete_fn=lambda sid, aid: is_archive_complete(output_base, sid, aid),
    )
    print(f"Reclaimed {reclaimed} stale locks")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="acoustic-orchestration",
        description="Batch acoustic inference orchestration pipeline",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Start a worker pod")
    p_run.add_argument("--parquet", required=True, help="Path to all_archives.parquet")
    p_run.add_argument("--output", required=True, help="EFS output base directory")
    p_run.add_argument("--affect-backbone", required=True, choices=["wavlm", "whisper"])
    p_run.add_argument("--disfluency-backbone", required=True, choices=["wavlm", "whisper"])
    p_run.add_argument("--batch-size", type=int, default=512)
    p_run.add_argument("--affect-batch-size", type=int, default=None)
    p_run.add_argument("--disfluency-batch-size", type=int, default=None)
    p_run.add_argument("--emotion-batch-size", type=int, default=None)
    p_run.add_argument("--device", default=None)
    p_run.add_argument("--max-retries", type=int, default=3)
    p_run.add_argument("--prefetch-workers", type=int, default=4)
    p_run.add_argument("--prefetch-lookahead", type=int, default=4)
    p_run.add_argument("--vad-prefetch-workers", type=int, default=1)
    p_run.add_argument("--seed", type=int, default=None)

    # --- progress ---
    p_progress = sub.add_parser("progress", help="Show pipeline progress")
    p_progress.add_argument("--parquet", required=True)
    p_progress.add_argument("--output", required=True)

    # --- errors ---
    p_errors = sub.add_parser("errors", help="List error records")
    p_errors.add_argument("--output", required=True)
    p_errors.add_argument("--kind", choices=["audio", "inference"], default="audio")
    p_errors.add_argument("--summary", action="store_true")

    # --- reclaim-stale ---
    p_reclaim = sub.add_parser("reclaim-stale", help="Remove orphan lock files")
    p_reclaim.add_argument("--output", required=True)
    p_reclaim.add_argument("--older-than", type=float, default=60.0,
                           help="Minutes since lock creation (default: 60)")

    args = parser.parse_args(argv)
    _configure_logging(verbose=args.verbose)

    handlers = {
        "run": _cmd_run,
        "progress": _cmd_progress,
        "errors": _cmd_errors,
        "reclaim-stale": _cmd_reclaim_stale,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
