"""Command-line interface for the session event store."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .populate import populate_session_store

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command.replace("-", "_")
    return globals()[f"_cmd_{command}"](args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session-event-store",
        description="Populate date-partitioned session event parquet files from event packages.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_populate = sub.add_parser(
        "populate",
        help="Aggregate completed event packages into session-level parquet files.",
    )
    p_populate.add_argument(
        "--manifest", required=True, type=Path,
        help="Path to the source archive manifest parquet.",
    )
    p_populate.add_argument(
        "--events-output", required=True, type=Path,
        help="Root directory of event packages (contains <session_id>/<archive_id>/ dirs).",
    )
    p_populate.add_argument(
        "--store-output", required=True, type=Path,
        help="Output directory for date-partitioned parquet files.",
    )
    p_populate.add_argument(
        "--dates", nargs="*", default=None,
        help="Restrict processing to these date partitions only.",
    )
    p_populate.add_argument(
        "--force", action="store_true",
        help="Re-read and rewrite all sessions regardless of fingerprint.",
    )
    p_populate.add_argument(
        "--verbose", action="store_true",
        help="Enable per-session DEBUG logging.",
    )
    return parser


def _cmd_populate(args: argparse.Namespace) -> int:
    _configure_logging(verbose=args.verbose)
    summary = populate_session_store(
        manifest_path=args.manifest,
        events_output=args.events_output,
        store_output=args.store_output,
        dates=args.dates,
        force=args.force,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _configure_logging(*, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    if not any(getattr(h, "_session_store_cli", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        handler._session_store_cli = True  # type: ignore[attr-defined]
        root.addHandler(handler)
