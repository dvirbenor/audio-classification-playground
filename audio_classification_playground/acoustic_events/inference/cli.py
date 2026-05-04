"""Command-line entry point for acoustic-event inference artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import list_cached_artifacts
from .audio import load_audio
from .runners import (
    run_affect_inference,
    run_all_inference,
    run_disfluency_inference,
    run_emotion_inference,
    run_vad,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    progress = print if getattr(args, "verbose", False) else None

    if args.command == "run":
        result = _run_single(args, progress)
        _print_artifact(args.task, result.artifact.path, result.reused)
        return 0
    if args.command == "run-all":
        result = run_all_inference(
            args.audio,
            out_dir=args.out,
            affect_backbone=args.affect_backbone,
            disfluency_backbone=args.disfluency_backbone,
            recording_id=args.recording_id,
            reuse_cache=args.reuse_cache,
            device=args.device,
            progress=progress,
        )
        for task, artifact in result.artifacts.items():
            _print_artifact(task, artifact.path, result.reused[task])
        return 0
    if args.command == "list-cached":
        audio_sha256 = None
        if args.audio is not None:
            audio_sha256 = load_audio(args.audio, recording_id=args.recording_id).audio_sha256
        artifacts = list_cached_artifacts(
            args.out,
            recording_id=args.recording_id,
            audio_sha256=audio_sha256,
            task=args.task,
            inference_config_hash_value=args.inference_config_hash,
        )
        for artifact in artifacts:
            print(f"{artifact.task}\t{artifact.path}")
        return 0
    parser.error("unknown command")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acoustic-events-inference",
        description="Run acoustic model inference into reusable prediction artifacts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one inference task.")
    run_sub = run.add_subparsers(dest="task", required=True)
    _add_common_run_args(run_sub.add_parser("affect", help="Run dimensional affect inference."))
    run_sub.choices["affect"].add_argument("--backbone", choices=("wavlm", "whisper"), required=True)
    _add_common_run_args(run_sub.add_parser("disfluency", help="Run disfluency inference."))
    run_sub.choices["disfluency"].add_argument("--backbone", choices=("wavlm", "whisper"), required=True)
    _add_common_run_args(run_sub.add_parser("emotion", help="Run emotion2vec inference."))
    _add_common_run_args(run_sub.add_parser("vad", help="Run shared VAD."))

    run_all = sub.add_parser("run-all", help="Run VAD and all model inference tasks.")
    _add_common_options(run_all)
    run_all.add_argument("--audio", required=True)
    run_all.add_argument("--affect-backbone", choices=("wavlm", "whisper"), required=True)
    run_all.add_argument("--disfluency-backbone", choices=("wavlm", "whisper"), required=True)

    cached = sub.add_parser("list-cached", help="List complete cached artifacts.")
    cached.add_argument("--out", required=True)
    cached.add_argument("--audio")
    cached.add_argument("--recording-id")
    cached.add_argument("--task", choices=("vad", "affect", "disfluency", "emotion"))
    cached.add_argument("--inference-config-hash")
    return parser


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    _add_common_options(parser)
    parser.add_argument("--audio", required=True)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="Artifact output root.")
    parser.add_argument("--recording-id")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--verbose", action="store_true")


def _run_single(args, progress):
    common = {
        "out_dir": args.out,
        "recording_id": args.recording_id,
        "reuse_cache": args.reuse_cache,
        "device": args.device,
        "progress": progress,
    }
    if args.task == "affect":
        return run_affect_inference(args.audio, backbone=args.backbone, **common)
    if args.task == "disfluency":
        return run_disfluency_inference(args.audio, backbone=args.backbone, **common)
    if args.task == "emotion":
        return run_emotion_inference(args.audio, **common)
    if args.task == "vad":
        return run_vad(args.audio, **common)
    raise ValueError(f"Unknown task {args.task!r}")


def _print_artifact(task: str, path: Path, reused: bool) -> None:
    status = "reused" if reused else "created"
    print(f"{task}: {status} {path}")


if __name__ == "__main__":
    raise SystemExit(main())
