"""Command-line entry point for acoustic-event inference artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import list_cached_artifacts
from .audio import load_audio
from .log import configure_stdout_logging
from .runners import (
    DEFAULT_VAD_MIN_SILENCE_SEC,
    DEFAULT_VAD_MIN_SPEECH_SEC,
    DEFAULT_VAD_SPEECH_THRESHOLD,
    run_affect_inference,
    run_all_inference,
    run_disfluency_inference,
    run_emotion_inference,
    run_vad,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_stdout_logging(verbose=getattr(args, "verbose", False))
    progress = logger.info

    if args.command == "run":
        result = _run_single(args, progress)
        _log_artifact(logger, args.task, result.artifact.path, result.reused)
        return 0
    if args.command == "run-all":
        result = run_all_inference(
            args.audio,
            out_dir=args.out,
            affect_backbone=args.affect_backbone,
            disfluency_backbone=args.disfluency_backbone,
            recording_id=args.recording_id,
            reuse_cache=args.reuse_cache,
            batch_size=128 if args.batch_size is None else args.batch_size,
            affect_batch_size=args.affect_batch_size,
            disfluency_batch_size=args.disfluency_batch_size,
            emotion_batch_size=args.emotion_batch_size,
            device=args.device,
            emotion_autocast_dtype=args.emotion_autocast_dtype,
            emotion_compile=args.emotion_compile,
            emotion_compile_mode=args.emotion_compile_mode,
            allow_tf32=args.allow_tf32,
            vad_threshold=args.vad_threshold,
            vad_min_speech_sec=args.vad_min_speech_sec,
            vad_min_silence_sec=args.vad_min_silence_sec,
            progress=progress,
        )
        for task, artifact in result.artifacts.items():
            _log_artifact(logger, task, artifact.path, result.reused[task])
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
            logger.info("%s\t%s", artifact.task, artifact.path)
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
    _add_emotion_runtime_options(run_sub.choices["emotion"])
    _add_common_run_args(run_sub.add_parser("vad", help="Run shared VAD."))
    _add_vad_options(run_sub.choices["vad"])

    run_all = sub.add_parser("run-all", help="Run VAD and all model inference tasks.")
    _add_common_options(run_all)
    run_all.add_argument("--audio", required=True)
    run_all.add_argument("--affect-backbone", choices=("wavlm", "whisper"), required=True)
    run_all.add_argument("--disfluency-backbone", choices=("wavlm", "whisper"), required=True)
    run_all.add_argument("--affect-batch-size", type=int)
    run_all.add_argument("--disfluency-batch-size", type=int)
    run_all.add_argument("--emotion-batch-size", type=int)
    _add_emotion_runtime_options(run_all)
    _add_vad_options(run_all)

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
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--verbose", action="store_true")


def _add_vad_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=DEFAULT_VAD_SPEECH_THRESHOLD,
        help="Silero speech probability threshold.",
    )
    parser.add_argument(
        "--vad-min-speech-sec",
        type=float,
        default=DEFAULT_VAD_MIN_SPEECH_SEC,
        help="Discard Silero speech regions shorter than this many seconds.",
    )
    parser.add_argument(
        "--vad-min-silence-sec",
        type=float,
        default=DEFAULT_VAD_MIN_SILENCE_SEC,
        help="Bridge Silero silence gaps shorter than this many seconds.",
    )


def _add_emotion_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--emotion-autocast-dtype",
        choices=("fp16", "bf16"),
        default=None,
        help="Opt-in autocast dtype for emotion2vec; benchmark before production use.",
    )
    parser.add_argument(
        "--emotion-compile",
        action="store_true",
        help="Compile the emotion2vec inner torch model before inference.",
    )
    parser.add_argument("--emotion-compile-mode", default="reduce-overhead")
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Enable TF32 matmul precision for supported NVIDIA GPUs.",
    )


def _run_single(args, progress):
    common = {
        "out_dir": args.out,
        "recording_id": args.recording_id,
        "reuse_cache": args.reuse_cache,
        "device": args.device,
        "progress": progress,
    }
    batch_kwargs = {}
    if args.batch_size is not None:
        batch_kwargs["batch_size"] = args.batch_size
    if args.task == "affect":
        return run_affect_inference(
            args.audio, backbone=args.backbone, **common, **batch_kwargs,
        )
    if args.task == "disfluency":
        return run_disfluency_inference(
            args.audio, backbone=args.backbone, **common, **batch_kwargs,
        )
    if args.task == "emotion":
        return run_emotion_inference(
            args.audio,
            autocast_dtype=args.emotion_autocast_dtype,
            compile_model=args.emotion_compile,
            compile_mode=args.emotion_compile_mode,
            allow_tf32=args.allow_tf32,
            **common,
            **batch_kwargs,
        )
    if args.task == "vad":
        return run_vad(
            args.audio,
            threshold=args.vad_threshold,
            min_speech_sec=args.vad_min_speech_sec,
            min_silence_sec=args.vad_min_silence_sec,
            **common,
        )
    raise ValueError(f"Unknown task {args.task!r}")


def _log_artifact(logger, task: str, path: Path, reused: bool) -> None:
    status = "reused" if reused else "created"
    logger.info("%s: %s %s", task, status, path)


if __name__ == "__main__":
    raise SystemExit(main())
