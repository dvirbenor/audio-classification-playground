"""Command-line entry point for producer composition."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..inference.log import configure_stdout_logging
from .composer import compose_review_package


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_stdout_logging(verbose=args.verbose)

    task_configs = {
        task: path
        for task, path in {
            "affect": args.affect_config,
            "disfluency": args.disfluency_config,
            "emotion": args.emotion_config,
        }.items()
        if path is not None
    }
    compose_review_package(
        affect_artifact=args.affect_artifact,
        disfluency_artifact=args.disfluency_artifact,
        emotion_artifact=args.emotion_artifact,
        vad_artifact=args.vad_artifact,
        out_dir=args.out,
        task_configs=task_configs,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acoustic-events-compose",
        description="Compose inference artifacts into a review_package.v1 directory.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    compose = sub.add_parser("compose", help="Compose a review package.")
    compose.add_argument("--affect-artifact", required=True, type=Path)
    compose.add_argument("--disfluency-artifact", required=True, type=Path)
    compose.add_argument("--emotion-artifact", required=True, type=Path)
    compose.add_argument("--vad-artifact", required=True, type=Path)
    compose.add_argument("--out", required=True, type=Path)
    compose.add_argument("--affect-config", type=Path)
    compose.add_argument("--disfluency-config", type=Path)
    compose.add_argument("--emotion-config", type=Path)
    compose.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
