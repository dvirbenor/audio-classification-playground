"""Command-line interface for the batch orchestration pipeline.

Subcommands:

    run             Start a worker that processes archives.
    progress        Print a summary of the current pipeline state.
    errors          List audio or inference errors.
    timings         Summarise per-archive inference timing distributions.
    status          Fleet heartbeat — per-worker lock/pace dashboard.
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
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FMT_CONSOLE = "%(asctime)s %(name)s %(levelname)s %(message)s"
_LOG_FMT_FILE = "%(asctime)s [%(process)d] %(name)s %(levelname)s %(message)s"
_LOG_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
_LOG_BACKUP_COUNT = 3


def _configure_logging(
    verbose: bool = False,
    log_dir: Path | None = None,
    command: str | None = None,
) -> None:
    root = logging.getLogger()
    level = logging.DEBUG if verbose else logging.INFO
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FMT_CONSOLE))
    root.addHandler(console)

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            hostname = os.environ.get("HOSTNAME", "local")
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            prefix = command or "orchestration"
            filename = f"{prefix}_{hostname}_{os.getpid()}_{ts}.log"
            file_handler = RotatingFileHandler(
                log_dir / filename,
                maxBytes=_LOG_MAX_BYTES,
                backupCount=_LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(_LOG_FMT_FILE))
            root.addHandler(file_handler)
            root.info("Log file: %s", log_dir / filename)
        except OSError as exc:
            print(
                f"warning: could not create log file in {log_dir}: {exc}",
                file=sys.stderr,
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
        wavlm_autocast_dtype=args.wavlm_autocast_dtype,
        wavlm_compile=args.wavlm_compile,
        wavlm_compile_mode=args.wavlm_compile_mode,
        wavlm_compile_dynamic=args.wavlm_compile_dynamic,
        wavlm_stream_layer_sum=args.wavlm_stream_layer_sum,
        wavlm_runtime_preset=args.wavlm_runtime_preset,
        emotion_autocast_dtype=args.emotion_autocast_dtype,
        emotion_compile=args.emotion_compile,
        emotion_compile_mode=args.emotion_compile_mode,
        emotion_runtime_mode=args.emotion_runtime_mode,
        allow_tf32=args.allow_tf32,
        prefetch_workers=args.prefetch_workers,
        prefetch_lookahead=args.prefetch_lookahead,
        vad_prefetch_workers=args.vad_prefetch_workers,
        audio_cache_dir=args.audio_cache_dir,
        max_cache_bytes=args.max_cache_bytes,
        audio_cache_lock_stale_minutes=args.audio_cache_lock_stale_minutes,
        task_group=args.task_group,
        completion_policy=args.completion_policy,
        force_recompute=args.force_recompute,
        seed=args.seed,
    )


# ---------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------


def _print_full_progress(args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from .errors import load_inference_attempt_counts, load_permanent_error_set
    from .manifest import load_manifest
    from .progress import scan_progress

    output_base = Path(args.output)

    with ThreadPoolExecutor(max_workers=3) as pool:
        entities_future = pool.submit(load_manifest, args.parquet)
        perm_errors_future = pool.submit(load_permanent_error_set, output_base)
        inf_errors_future = pool.submit(load_inference_attempt_counts, output_base)

    entities = entities_future.result()
    perm_errors = perm_errors_future.result()
    inf_errors = inf_errors_future.result()

    summary = scan_progress(
        output_base, entities,
        permanent_audio_errors=perm_errors,
        inference_error_counts=dict(inf_errors),
        use_cache=not getattr(args, "no_cache", False),
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


def _print_fast_progress(args: argparse.Namespace) -> None:
    from .progress import quick_disk_summary

    output_base = Path(args.output)
    s = quick_disk_summary(output_base)

    lines = [
        f"Complete (all 4 tasks):  {s.complete}",
        f"Partially done:          {s.partial}",
        f"Lock files:              {s.lock_count}",
        "",
        "Per-task artifact count:",
    ]
    for task, count in sorted(s.task_counts.items()):
        lines.append(f"  {task:12s} {count}")

    lines.append("")
    lines.append(
        f"Audio errors:      {s.audio_error_records} records, "
        f"{s.permanent_audio_error_archives} permanent archives"
    )
    lines.append(
        f"Inference errors:  {s.inference_error_records} records "
        f"across {s.inference_error_archives} archives"
    )
    lines.append("")
    lines.append("(--fast mode: totals/remaining unavailable without --parquet)")
    print("\n".join(lines))


def _cmd_progress(args: argparse.Namespace) -> None:
    if args.fast and not args.parquet:
        _print_fast_progress(args)
    elif args.parquet:
        _print_full_progress(args)
    else:
        print("error: --parquet is required unless --fast is specified",
              file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


def _print_errors_flat(errors_dir: Path, kind: str, summary: bool) -> None:
    if not errors_dir.is_dir():
        print(f"No {kind} errors directory found at {errors_dir}")
        return

    count = 0
    for f in sorted(errors_dir.rglob("*.json")):
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

    if summary:
        print(f"\nTotal {kind} error records: {count}")


def _print_errors_grouped(errors_dir: Path, kind: str) -> None:
    from .errors import summarize_errors_grouped

    groups = summarize_errors_grouped(errors_dir)
    if not groups:
        print(f"No {kind} errors found.")
        return

    total_records = 0
    all_archives: set[tuple[str, str]] = set()

    print(f"{kind.capitalize()} errors by type:\n")
    for grp in groups:
        total_records += grp.record_count
        all_archives |= grp.unique_archives

        n_archives = len(grp.unique_archives)
        sid, aid = grp.example_archive
        suffix = ""
        if grp.is_permanent is True:
            suffix = " (permanent)"
        elif grp.is_permanent is False:
            suffix = " (transient)"

        print(
            f"  {grp.error_type:24s} "
            f"{grp.record_count:>4} records, {n_archives:>4} archives{suffix}"
        )
        print(f"    e.g. {sid}/{aid} \u2014 {grp.example_detail}")

    print(f"\nTotal: {total_records} records, {len(all_archives)} unique archives")


def _cmd_errors(args: argparse.Namespace) -> None:
    output_base = Path(args.output)
    kind = args.kind

    if kind == "audio":
        from .errors import AUDIO_ERRORS_DIR
        errors_dir = output_base / AUDIO_ERRORS_DIR
    else:
        from .errors import INFERENCE_ERRORS_DIR
        errors_dir = output_base / INFERENCE_ERRORS_DIR

    if args.group:
        _print_errors_grouped(errors_dir, kind)
    else:
        _print_errors_flat(errors_dir, kind, args.summary)


# ---------------------------------------------------------------------------
# timings
# ---------------------------------------------------------------------------


def _cmd_timings(args: argparse.Namespace) -> None:
    from .timings import (
        derive_vad_mode,
        format_timing_csv,
        format_timing_summary,
        load_timing_records,
        summarize_timings,
        summarize_timings_by_worker,
    )

    output_base = Path(args.output)
    records = load_timing_records(output_base)

    if not records:
        print("No timing records found.")
        return

    fields = None
    if args.fields:
        fields = tuple(f.strip() for f in args.fields.split(","))

    if args.min_audio_sec is not None:
        records = [r for r in records if r.get("audio_duration_sec", 0) >= args.min_audio_sec]
    if args.max_audio_sec is not None:
        records = [r for r in records if r.get("audio_duration_sec", 0) <= args.max_audio_sec]
    if args.worker:
        records = [r for r in records if args.worker in r.get("worker_id", "")]

    if not records:
        print("No timing records match the filters.")
        return

    if args.csv:
        print(format_timing_csv(records, fields), end="")
        return

    split_vad = args.split_by_vad_mode

    def _print_group(recs: list[dict], title: str | None = None) -> None:
        if split_vad:
            buckets: dict[str, list[dict]] = {}
            for r in recs:
                mode = derive_vad_mode(r)
                buckets.setdefault(mode, []).append(r)
            for mode in ("prefetched", "cached", "inline"):
                bucket = buckets.get(mode)
                if not bucket:
                    continue
                label = f"{title} (vad_mode={mode})" if title else f"vad_mode={mode}"
                print(format_timing_summary(summarize_timings(bucket, fields), title=label))
        else:
            print(format_timing_summary(summarize_timings(recs, fields), title=title))

    if args.by_worker:
        groups = summarize_timings_by_worker(records, fields)
        for wid in groups:
            worker_recs = [r for r in records if r.get("worker_id") == wid]
            _print_group(worker_recs, title=f"Worker: {wid}")
        print("---")
        _print_group(records, title="All workers (aggregate)")
    else:
        _print_group(records, title=f"Timing summary ({len(records)} records)")


def _cmd_status(args: argparse.Namespace) -> None:
    from .heartbeat import (
        build_fleet_heartbeat,
        count_error_files,
        format_heartbeat,
        load_recent_timings,
        parse_active_locks,
    )

    output_base = Path(args.output)
    locks = parse_active_locks(output_base)
    timings = load_recent_timings(output_base, tail=args.tail)
    heartbeat = build_fleet_heartbeat(locks, timings)

    if args.summary:
        from .progress import quick_disk_summary

        disk = quick_disk_summary(output_base)
        print(format_heartbeat(heartbeat, disk_summary=disk))
    else:
        errors = count_error_files(output_base)
        print(format_heartbeat(heartbeat, error_counts=errors))


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


def _cmd_warm_cache(args: argparse.Namespace) -> None:
    from .cache_warmer import warm_cache

    warm_cache(
        parquet_path=args.parquet,
        output_base=args.output,
        audio_cache_dir=args.audio_cache_dir,
        max_cache_bytes=args.max_cache_bytes,
        seed=args.seed,
        max_inference_attempts=args.max_retries,
        audio_cache_lock_stale_minutes=args.audio_cache_lock_stale_minutes,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="acoustic-orchestration",
        description="Batch acoustic inference orchestration pipeline",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--log-dir", type=Path, default=None,
        help="Directory for log files (default: {output}/_meta/logs/ for 'run')",
    )
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
    p_run.add_argument(
        "--wavlm-autocast-dtype",
        choices=("fp16", "bf16"),
        default=None,
        help="Opt-in autocast dtype for WavLM affect/disfluency; benchmark before production use.",
    )
    p_run.add_argument(
        "--wavlm-compile",
        action="store_true",
        help="Compile persistent WavLM backbones once; benchmark cold start and parity first.",
    )
    p_run.add_argument("--wavlm-compile-mode", default="reduce-overhead")
    p_run.add_argument(
        "--wavlm-compile-dynamic",
        action="store_true",
        help="Compile WavLM backbones with dynamic shapes to reduce last-batch recompiles.",
    )
    p_run.add_argument(
        "--wavlm-stream-layer-sum",
        action="store_true",
        help="Accumulate WavLM learned layer mixtures without materializing every hidden state.",
    )
    p_run.add_argument(
        "--wavlm-runtime-preset",
        choices=("fast_exact", "compiled_static"),
        default=None,
        help=(
            "Worker-level WavLM preset. If unset, workers choose "
            "compiled_static when CUDA/compiler prerequisites are available "
            "and fast_exact otherwise."
        ),
    )
    p_run.add_argument(
        "--emotion-runtime-mode",
        choices=["auto", "optimized", "fp32-eager", "custom"],
        default="auto",
        help=(
            "Emotion2vec runtime preset. Default 'auto' uses optimized "
            "compile+scoped-TF32 on CUDA and FP32 eager elsewhere. Use "
            "'fp32-eager' for old FP32 behavior, or 'custom' with granular "
            "experiment flags."
        ),
    )
    p_run.add_argument(
        "--emotion-autocast-dtype",
        choices=("fp16", "bf16"),
        default=None,
        help="Opt-in autocast dtype for emotion2vec; benchmark before production use.",
    )
    p_run.add_argument(
        "--emotion-compile",
        action="store_true",
        help="Compile the persistent emotion2vec inner torch model once.",
    )
    p_run.add_argument("--emotion-compile-mode", default="default")
    p_run.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Enable TF32 matmul precision for supported NVIDIA GPUs.",
    )
    p_run.add_argument("--max-retries", type=int, default=3)
    p_run.add_argument("--prefetch-workers", type=int, default=None)
    p_run.add_argument("--prefetch-lookahead", type=int, default=None)
    p_run.add_argument("--vad-prefetch-workers", type=int, default=None)
    p_run.add_argument(
        "--audio-cache-dir",
        type=Path,
        default=None,
        help="Enable shared decoded-audio cache at this directory.",
    )
    p_run.add_argument(
        "--max-cache-bytes",
        type=int,
        default=None,
        help="Required with --audio-cache-dir; shared decoded cache cap.",
    )
    p_run.add_argument(
        "--audio-cache-lock-stale-minutes",
        type=float,
        default=60.0,
        help="Minutes before abandoned cache locks may be reclaimed.",
    )
    from .task_groups import task_group_choices
    p_run.add_argument("--task-group", choices=task_group_choices(), default="all")
    p_run.add_argument(
        "--completion-policy",
        choices=("exists", "config"),
        default="exists",
        help="How workers decide whether a task artifact should be reused.",
    )
    p_run.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing task artifacts and recompute selected task-group work.",
    )
    p_run.add_argument("--seed", type=int, default=None)

    # --- progress ---
    p_progress = sub.add_parser("progress", help="Show pipeline progress")
    p_progress.add_argument("--parquet", default=None,
                            help="Path to all_archives.parquet (required unless --fast)")
    p_progress.add_argument("--output", required=True)
    p_progress.add_argument("--fast", action="store_true",
                            help="Quick disk-only summary without loading the manifest")
    p_progress.add_argument("--no-cache", action="store_true",
                            help="Bypass completion cache and re-verify all archives on disk")

    # --- errors ---
    p_errors = sub.add_parser("errors", help="List error records")
    p_errors.add_argument("--output", required=True)
    p_errors.add_argument("--kind", choices=["audio", "inference"], default="audio")
    p_errors.add_argument("--summary", action="store_true",
                          help="Append a total count (flat mode only)")
    p_errors.add_argument("--group", action="store_true",
                          help="Group errors by type with record/archive counts")

    # --- timings ---
    p_timings = sub.add_parser("timings", help="Summarise per-archive inference timing distributions")
    p_timings.add_argument("--output", required=True, help="EFS output base directory")
    p_timings.add_argument("--fields", default=None,
                           help="Comma-separated subset of timing fields to display")
    p_timings.add_argument("--csv", action="store_true",
                           help="Output as CSV instead of formatted table")
    p_timings.add_argument("--min-audio-sec", type=float, default=None,
                           help="Only include records with audio_duration_sec >= this value")
    p_timings.add_argument("--max-audio-sec", type=float, default=None,
                           help="Only include records with audio_duration_sec <= this value")
    p_timings.add_argument("--by-worker", action="store_true",
                           help="Print per-worker breakdown followed by aggregate")
    p_timings.add_argument("--worker", default=None,
                           help="Filter to a single worker_id (substring match)")
    p_timings.add_argument("--split-by-vad-mode", action="store_true", default=True,
                           dest="split_by_vad_mode",
                           help="Split stats by VAD mode: prefetched/cached/inline (default)")
    p_timings.add_argument("--no-split-by-vad-mode", action="store_false",
                           dest="split_by_vad_mode",
                           help="Disable VAD mode splitting")

    # --- status ---
    p_status = sub.add_parser(
        "status", help="Fleet heartbeat \u2014 per-worker lock/pace dashboard",
    )
    p_status.add_argument("--output", required=True, help="EFS output base directory")
    p_status.add_argument("--tail", type=int, default=20,
                          help="Recent timing records per worker for pace calculation")
    p_status.add_argument("--summary", action="store_true",
                          help="Include completed/partial counts (slower, walks output tree)")

    # --- reclaim-stale ---
    p_reclaim = sub.add_parser("reclaim-stale", help="Remove orphan lock files")
    p_reclaim.add_argument("--output", required=True)
    p_reclaim.add_argument("--older-than", type=float, default=60.0,
                           help="Minutes since lock creation (default: 60)")

    # --- warm-cache ---
    p_warm = sub.add_parser("warm-cache", help="Warm the shared decoded-audio cache")
    p_warm.add_argument("--parquet", required=True, help="Path to all_archives.parquet")
    p_warm.add_argument("--output", required=True, help="EFS output base directory")
    p_warm.add_argument("--audio-cache-dir", type=Path, required=True)
    p_warm.add_argument("--max-cache-bytes", type=int, required=True)
    p_warm.add_argument("--seed", type=int, default=None)
    p_warm.add_argument("--max-retries", type=int, default=3)
    p_warm.add_argument(
        "--audio-cache-lock-stale-minutes",
        type=float,
        default=60.0,
        help="Minutes before abandoned cache locks may be reclaimed.",
    )

    args = parser.parse_args(argv)
    _validate_emotion_runtime_args(parser, args)
    _validate_wavlm_runtime_args(parser, args)

    log_dir = args.log_dir
    if log_dir is None and args.command == "run":
        log_dir = Path(args.output) / "_meta" / "logs"

    _configure_logging(
        verbose=args.verbose,
        log_dir=log_dir,
        command=args.command,
    )

    handlers = {
        "run": _cmd_run,
        "progress": _cmd_progress,
        "errors": _cmd_errors,
        "timings": _cmd_timings,
        "status": _cmd_status,
        "reclaim-stale": _cmd_reclaim_stale,
        "warm-cache": _cmd_warm_cache,
    }
    handlers[args.command](args)


def _validate_emotion_runtime_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if getattr(args, "command", None) != "run":
        return
    if args.emotion_runtime_mode == "custom":
        return
    from ..inference.emotion_runtime import has_custom_emotion_runtime_knobs

    if has_custom_emotion_runtime_knobs(
        autocast_dtype=args.emotion_autocast_dtype,
        compile_model=args.emotion_compile,
        compile_mode=args.emotion_compile_mode,
        allow_tf32=args.allow_tf32,
    ):
        parser.error(
            "--emotion-runtime-mode presets cannot be mixed with granular "
            "emotion runtime flags; use --emotion-runtime-mode custom."
        )


def _validate_wavlm_runtime_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if getattr(args, "command", None) != "run":
        return
    if args.wavlm_runtime_preset is None:
        return
    from ..inference.wavlm_runtime import has_custom_wavlm_runtime_knobs

    if has_custom_wavlm_runtime_knobs(
        autocast_dtype=args.wavlm_autocast_dtype,
        compile_model=args.wavlm_compile,
        compile_mode=args.wavlm_compile_mode,
        compile_dynamic=args.wavlm_compile_dynamic,
        stream_layer_sum=args.wavlm_stream_layer_sum,
        allow_tf32=args.allow_tf32,
    ):
        parser.error(
            "--wavlm-runtime-preset cannot be mixed with granular WavLM "
            "runtime flags; omit the preset for custom experiments."
        )


if __name__ == "__main__":
    main()
