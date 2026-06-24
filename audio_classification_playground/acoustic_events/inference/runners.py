"""Inference runners that persist producer-ready prediction artifacts."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Mapping, Sequence

import numpy as np

from ...vox_profile.wavlm_inference import autocast_context, compile_wavlm_backbone
from ..producers.emotion.config import CANONICAL_CHANNELS, Config as EmotionConfig
from ..producers.emotion.pipeline import normalize_label
from .artifacts import (
    InferenceRunResult,
    MANIFEST_FILENAME,
    PREDICTIONS_FILENAME,
    PredictionArtifact,
    SAMPLE_RATE,
    artifact_dir,
    base_manifest,
    find_cached_artifact,
    inference_config_hash,
    inference_configs_match,
    load_prediction_artifact,
    write_prediction_artifact,
)
from .audio import (
    AudioData,
    count_audio_frames,
    frame_audio,
    load_audio,
    writable_contiguous_float32,
)
from .emotion2vec import predict_emotion2vec_scores, predict_emotion2vec_scores_from_audio
from .emotion_runtime import (
    DEFAULT_EMOTION_COMPILE_MODE,
    OPTIMIZED_EMOTION_BATCH_SIZE,
    STANDALONE_EMOTION_BATCH_SIZE,
    resolve_emotion_runtime_settings,
    torch_matmul_precision,
)
from .log import get_logger
from .vad_gating import (
    VadGating,
    gate_window_arrays,
    intervals_from_array,
    scatter_fill,
    speech_window_mask,
)
from .wavlm_runtime import (
    WAVLM_PREPROCESSING_POLICY,
    WAVLM_STATIC_BATCH_PADDING_POLICY,
    pad_windows_to_static_batch,
)


DEFAULT_AFFECT_MODELS = {
    "wavlm": "tiantiaf/wavlm-large-msp-podcast-emotion-dim",
    "whisper": "tiantiaf/whisper-large-v3-msp-podcast-emotion-dim",
}
DEFAULT_DISFLUENCY_MODELS = {
    "wavlm": "tiantiaf/wavlm-large-speech-flow",
    "whisper": "tiantiaf/whisper-large-v3-speech-flow",
}
DEFAULT_EMOTION_MODEL = "iic/emotion2vec_plus_large"
DEFAULT_VAD_MODEL = "silero-vad"

AFFECT_WINDOW_SEC = 3.5
DISFLUENCY_WINDOW_SEC = 3.0
EMOTION_WINDOW_SEC = 3.0
DEFAULT_HOP_SEC = 0.25
DEFAULT_VAD_SPEECH_THRESHOLD = 0.35
DEFAULT_VAD_MIN_SPEECH_SEC = 0.1
DEFAULT_VAD_MIN_SILENCE_SEC = 1.2
DEFAULT_VAD_FRAME_SPEECH_RATIO_THRESHOLD = 0.1

Backbone = str
ProgressFn = Callable[[str], None]
LOGGER = get_logger()


class ShutdownRequested(Exception):
    """Raised between tasks when the worker received a graceful shutdown signal."""


@dataclass(frozen=True)
class TaskRun:
    artifact: PredictionArtifact
    reused: bool


def _load_vad_intervals_from_artifact(
    vad_path: Path | None,
) -> list[tuple[float, float]] | None:
    """Load ``intervals_sec`` from an existing VAD artifact, or ``None``.

    Used by ``run_all_inference`` when gating is requested but the VAD task is
    not (re)run in this call (e.g. VAD already complete, only affect missing).
    """
    if vad_path is None:
        return None
    try:
        artifact = load_prediction_artifact(vad_path)
    except (OSError, ValueError, KeyError, FileNotFoundError, json.JSONDecodeError):
        return None
    if artifact.manifest.get("task") != "vad":
        return None
    return intervals_from_array(artifact.arrays.get("intervals_sec"))


def _task_gate(vad_gating: VadGating | None, task: str) -> VadGating | None:
    """Return the gating settings if *task* should be gated, else ``None``."""
    if vad_gating is not None and vad_gating.gates(task):
        return vad_gating
    return None


def _apply_gating(
    config: dict,
    vad_gating: VadGating | None,
    vad_intervals: Sequence[tuple[float, float]] | None = None,
) -> dict:
    """Merge the VAD-gating descriptor into an inference config (when it applies).

    Tagged only when gating is active *and* intervals are actually available —
    so an artifact is never labelled "gated" if it was in fact computed over the
    full timeline (intervals missing). Changes ``inference_config_hash`` so gated
    artifacts are a distinct lineage (immutability), mirroring ``autocast_dtype``.
    """
    if vad_gating is not None and vad_gating.active and vad_intervals is not None:
        return {**config, **vad_gating.config_extra()}
    return config


def run_affect_inference(
    audio_path: str | Path | AudioData,
    *,
    out_dir: str | Path,
    backbone: Backbone,
    recording_id: str | None = None,
    reuse_cache: bool = False,
    model_id: str | None = None,
    window_sec: float = AFFECT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    sample_rate: int = SAMPLE_RATE,
    batch_size: int = 128,
    device: str | None = None,
    wavlm_autocast_dtype: str | None = None,
    wavlm_compile: bool = False,
    wavlm_compile_mode: str = "reduce-overhead",
    wavlm_compile_dynamic: bool = False,
    wavlm_stream_layer_sum: bool = False,
    wavlm_static_batch: bool = False,
    wavlm_runtime_preset: str | None = None,
    allow_tf32: bool = False,
    predictor: Callable[[np.ndarray], Mapping[str, np.ndarray]] | None = None,
    progress: ProgressFn | None = None,
    cleanup_cuda: Callable[[], None] | None = None,
    artifact_path: Path | None = None,
    audio_path_override: str | None = None,
    audio_source_key: str | None = None,
    vad_intervals: Sequence[tuple[float, float]] | None = None,
    vad_gating: VadGating | None = None,
) -> TaskRun:
    """Run Vox-Profile dimensional affect inference and persist A/V/D arrays."""
    if backbone not in DEFAULT_AFFECT_MODELS:
        raise ValueError("backbone must be one of: wavlm, whisper")
    audio = _coerce_audio(audio_path, sample_rate=sample_rate, recording_id=recording_id)
    resolved_model_id = model_id or DEFAULT_AFFECT_MODELS[backbone]
    wavlm_effective = _wavlm_effective_runtime_config(
        backbone=backbone,
        predictor=predictor,
        autocast_dtype=wavlm_autocast_dtype,
        compile_model=wavlm_compile,
        compile_mode=wavlm_compile_mode,
        compile_dynamic=wavlm_compile_dynamic,
        stream_layer_sum=wavlm_stream_layer_sum,
        allow_tf32=allow_tf32,
    )
    config = _inference_config(
        task="affect",
        model_id=resolved_model_id,
        backbone=backbone,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
        batch_size=batch_size,
        transform_policy="vox_profile_affect_sigmoid_heads_v1",
        extra=_wavlm_runtime_config_extra(
            backbone=backbone,
            **wavlm_effective,
        ),
    )
    affect_gate = _task_gate(vad_gating, "affect")
    config = _apply_gating(config, affect_gate, vad_intervals)
    cached = _maybe_cached(out_dir, audio, "affect", config, reuse_cache, artifact_path)
    if cached is not None:
        _progress(progress, f"affect cache hit: {cached.path}")
        return TaskRun(cached, True)

    _progress(progress, "affect framing audio")
    windows = frame_audio(
        audio.samples,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )
    _progress(progress, f"affect windows: {len(windows)}")
    try:
        def _run_affect(batch_windows):
            if predictor is not None:
                return predictor(batch_windows)
            return _predict_affect(
                batch_windows,
                backbone=backbone,
                model_id=resolved_model_id,
                device=device,
                batch_size=batch_size,
                wavlm_autocast_dtype=wavlm_autocast_dtype,
                wavlm_compile=wavlm_compile,
                wavlm_compile_mode=wavlm_compile_mode,
                wavlm_compile_dynamic=wavlm_compile_dynamic,
                wavlm_stream_layer_sum=wavlm_stream_layer_sum,
                wavlm_static_batch=wavlm_static_batch,
                allow_tf32=allow_tf32,
                progress=progress,
            )

        arrays, _ = gate_window_arrays(
            _run_affect,
            windows,
            hop_sec=hop_sec,
            window_sec=window_sec,
            vad_intervals=vad_intervals,
            gating=affect_gate,
            fill=0.0,
        )
        arrays = _validate_affect_arrays(arrays)
        artifact = _write_task_artifact(
            out_dir,
            audio,
            task="affect",
            config=config,
            arrays=arrays,
            model={
                "family": "vox-profile-affect",
                "backbone": backbone,
                "id": resolved_model_id,
            },
            timing=_timing(sample_rate, window_sec, hop_sec, len(windows)),
            runtime=_runtime(
                device=device,
                batch_size=batch_size,
                extra=_wavlm_runtime_metadata(
                    backbone=backbone,
                    runtime_preset=wavlm_runtime_preset,
                    static_batch=wavlm_static_batch,
                    predictor=predictor,
                ),
            ),
            artifact_path=artifact_path,
            audio_path_override=audio_path_override,
            audio_source_key=audio_source_key,
        )
        _progress(progress, f"affect wrote: {artifact.path}")
        return TaskRun(artifact, False)
    finally:
        (cleanup_cuda or cleanup_torch_memory)()


def run_disfluency_inference(
    audio_path: str | Path | AudioData,
    *,
    out_dir: str | Path,
    backbone: Backbone,
    recording_id: str | None = None,
    reuse_cache: bool = False,
    model_id: str | None = None,
    window_sec: float = DISFLUENCY_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    sample_rate: int = SAMPLE_RATE,
    batch_size: int = 128,
    device: str | None = None,
    wavlm_autocast_dtype: str | None = None,
    wavlm_compile: bool = False,
    wavlm_compile_mode: str = "reduce-overhead",
    wavlm_compile_dynamic: bool = False,
    wavlm_stream_layer_sum: bool = False,
    wavlm_static_batch: bool = False,
    wavlm_runtime_preset: str | None = None,
    allow_tf32: bool = False,
    predictor: Callable[[np.ndarray], Mapping[str, np.ndarray]] | None = None,
    progress: ProgressFn | None = None,
    cleanup_cuda: Callable[[], None] | None = None,
    artifact_path: Path | None = None,
    audio_path_override: str | None = None,
    audio_source_key: str | None = None,
    vad_intervals: Sequence[tuple[float, float]] | None = None,
    vad_gating: VadGating | None = None,
) -> TaskRun:
    """Run Vox-Profile disfluency inference and persist raw logits."""
    if backbone not in DEFAULT_DISFLUENCY_MODELS:
        raise ValueError("backbone must be one of: wavlm, whisper")
    audio = _coerce_audio(audio_path, sample_rate=sample_rate, recording_id=recording_id)
    resolved_model_id = model_id or DEFAULT_DISFLUENCY_MODELS[backbone]
    wavlm_effective = _wavlm_effective_runtime_config(
        backbone=backbone,
        predictor=predictor,
        autocast_dtype=wavlm_autocast_dtype,
        compile_model=wavlm_compile,
        compile_mode=wavlm_compile_mode,
        compile_dynamic=wavlm_compile_dynamic,
        stream_layer_sum=wavlm_stream_layer_sum,
        allow_tf32=allow_tf32,
    )
    config = _inference_config(
        task="disfluency",
        model_id=resolved_model_id,
        backbone=backbone,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
        batch_size=batch_size,
        transform_policy="vox_profile_disfluency_raw_logits_v1",
        extra=_wavlm_runtime_config_extra(
            backbone=backbone,
            **wavlm_effective,
        ),
    )
    disfluency_gate = _task_gate(vad_gating, "disfluency")
    config = _apply_gating(config, disfluency_gate, vad_intervals)
    cached = _maybe_cached(out_dir, audio, "disfluency", config, reuse_cache, artifact_path)
    if cached is not None:
        _progress(progress, f"disfluency cache hit: {cached.path}")
        return TaskRun(cached, True)

    _progress(progress, "disfluency framing audio")
    windows = frame_audio(
        audio.samples,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )
    _progress(progress, f"disfluency windows: {len(windows)}")
    try:
        def _run_disfluency(batch_windows):
            if predictor is not None:
                return predictor(batch_windows)
            return _predict_disfluency(
                batch_windows,
                backbone=backbone,
                model_id=resolved_model_id,
                device=device,
                batch_size=batch_size,
                wavlm_autocast_dtype=wavlm_autocast_dtype,
                wavlm_compile=wavlm_compile,
                wavlm_compile_mode=wavlm_compile_mode,
                wavlm_compile_dynamic=wavlm_compile_dynamic,
                wavlm_stream_layer_sum=wavlm_stream_layer_sum,
                wavlm_static_batch=wavlm_static_batch,
                allow_tf32=allow_tf32,
                progress=progress,
            )

        arrays, _ = gate_window_arrays(
            _run_disfluency,
            windows,
            hop_sec=hop_sec,
            window_sec=window_sec,
            vad_intervals=vad_intervals,
            gating=disfluency_gate,
            fill=0.0,
        )
        arrays = _validate_disfluency_arrays(arrays)
        artifact = _write_task_artifact(
            out_dir,
            audio,
            task="disfluency",
            config=config,
            arrays=arrays,
            model={
                "family": "vox-profile-disfluency",
                "backbone": backbone,
                "id": resolved_model_id,
            },
            timing=_timing(sample_rate, window_sec, hop_sec, len(windows)),
            runtime=_runtime(
                device=device,
                batch_size=batch_size,
                extra=_wavlm_runtime_metadata(
                    backbone=backbone,
                    runtime_preset=wavlm_runtime_preset,
                    static_batch=wavlm_static_batch,
                    predictor=predictor,
                ),
            ),
            artifact_path=artifact_path,
            audio_path_override=audio_path_override,
            audio_source_key=audio_source_key,
        )
        _progress(progress, f"disfluency wrote: {artifact.path}")
        return TaskRun(artifact, False)
    finally:
        (cleanup_cuda or cleanup_torch_memory)()


def run_emotion_inference(
    audio_path: str | Path | AudioData,
    *,
    out_dir: str | Path,
    recording_id: str | None = None,
    reuse_cache: bool = False,
    model_id: str = DEFAULT_EMOTION_MODEL,
    window_sec: float = EMOTION_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    sample_rate: int = SAMPLE_RATE,
    batch_size: int | None = None,
    device: str | None = None,
    autocast_dtype: str | None = None,
    compile_model: bool = False,
    compile_mode: str = DEFAULT_EMOTION_COMPILE_MODE,
    emotion_runtime_mode: str | None = "auto",
    allow_tf32: bool = False,
    predictor: Callable[[np.ndarray], tuple[np.ndarray, Sequence[str]]] | None = None,
    progress: ProgressFn | None = None,
    cleanup_cuda: Callable[[], None] | None = None,
    artifact_path: Path | None = None,
    audio_path_override: str | None = None,
    audio_source_key: str | None = None,
    vad_intervals: Sequence[tuple[float, float]] | None = None,
    vad_gating: VadGating | None = None,
) -> TaskRun:
    """Run emotion2vec inference and persist canonical probabilities."""
    runtime_settings = resolve_emotion_runtime_settings(
        mode=emotion_runtime_mode,
        default_mode="auto",
        device=device,
        autocast_dtype=autocast_dtype,
        compile_model=compile_model,
        compile_mode=compile_mode,
        allow_tf32=allow_tf32,
    )
    resolved_batch_size = _resolve_emotion_batch_size(
        batch_size,
        settings=runtime_settings,
        fallback_batch_size=STANDALONE_EMOTION_BATCH_SIZE,
    )
    audio = _coerce_audio(audio_path, sample_rate=sample_rate, recording_id=recording_id)
    config = _inference_config(
        task="emotion",
        model_id=model_id,
        backbone=None,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
        batch_size=resolved_batch_size,
        transform_policy="emotion2vec_fold_row_normalize_v1",
        extra=_emotion_runtime_config_extra(
            autocast_dtype=runtime_settings.autocast_dtype,
            compile_model=runtime_settings.compile_model,
            compile_mode=runtime_settings.compile_mode,
            allow_tf32=runtime_settings.allow_tf32,
        ),
    )
    emotion_gate = _task_gate(vad_gating, "emotion")
    config = _apply_gating(config, emotion_gate, vad_intervals)
    cached = _maybe_cached(out_dir, audio, "emotion", config, reuse_cache, artifact_path)
    if cached is not None:
        _progress(progress, f"emotion cache hit: {cached.path}")
        return TaskRun(cached, True)

    n_windows = count_audio_frames(
        audio.samples,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )
    _progress(progress, f"emotion windows: {n_windows}")
    gate_on = emotion_gate is not None and vad_intervals is not None
    try:
        if gate_on:
            # Gate via the framed-windows path (numerically identical to the
            # predict_audio stride path on the same windows) so non-speech
            # windows are skipped and filled at the raw-score level.
            raw_scores, raw_labels = _emotion_scores_gated(
                audio.samples,
                predictor=predictor,
                model_id=model_id,
                sample_rate=sample_rate,
                window_sec=window_sec,
                hop_sec=hop_sec,
                batch_size=resolved_batch_size,
                runtime_settings=runtime_settings,
                vad_intervals=vad_intervals,
                vad_gating=vad_gating,
                progress=progress,
            )
        else:
            predict_audio = getattr(predictor, "predict_audio", None) if predictor is not None else None
            if callable(predict_audio):
                raw_scores, raw_labels = predict_audio(
                    audio.samples,
                    sample_rate=sample_rate,
                    window_sec=window_sec,
                    hop_sec=hop_sec,
                    progress=progress,
                )
            elif predictor is not None:
                _progress(progress, "emotion framing audio")
                windows = frame_audio(
                    audio.samples,
                    sample_rate=sample_rate,
                    window_sec=window_sec,
                    hop_sec=hop_sec,
                )
                raw_scores, raw_labels = predictor(windows)
            else:
                raw_scores, raw_labels = _predict_emotion2vec(
                    audio.samples,
                    model_id=model_id,
                    sample_rate=sample_rate,
                    window_sec=window_sec,
                    hop_sec=hop_sec,
                    batch_size=resolved_batch_size,
                    device=runtime_settings.device,
                    autocast_dtype=runtime_settings.autocast_dtype,
                    compile_model=runtime_settings.compile_model,
                    compile_mode=runtime_settings.compile_mode,
                    allow_tf32=runtime_settings.allow_tf32,
                    progress=progress,
                )
        probabilities, labels = emotion2vec_scores_to_probabilities(raw_scores, raw_labels)
        artifact = _write_task_artifact(
            out_dir,
            audio,
            task="emotion",
            config=config,
            arrays={"probabilities": probabilities},
            model={
                "family": "emotion2vec",
                "id": model_id,
            },
            timing=_timing(sample_rate, window_sec, hop_sec, n_windows),
            runtime=_runtime(device=runtime_settings.device, batch_size=resolved_batch_size),
            labels=labels,
            artifact_path=artifact_path,
            audio_path_override=audio_path_override,
            audio_source_key=audio_source_key,
        )
        _progress(progress, f"emotion wrote: {artifact.path}")
        return TaskRun(artifact, False)
    finally:
        (cleanup_cuda or cleanup_torch_memory)()


def run_vad(
    audio_path: str | Path | AudioData,
    *,
    out_dir: str | Path,
    recording_id: str | None = None,
    reuse_cache: bool = False,
    model_id: str = DEFAULT_VAD_MODEL,
    sample_rate: int = SAMPLE_RATE,
    threshold: float = DEFAULT_VAD_SPEECH_THRESHOLD,
    min_speech_sec: float = DEFAULT_VAD_MIN_SPEECH_SEC,
    min_silence_sec: float = DEFAULT_VAD_MIN_SILENCE_SEC,
    device: str | None = None,
    detector: Callable[[np.ndarray, int], Sequence[tuple[float, float]]] | None = None,
    progress: ProgressFn | None = None,
    cleanup_cuda: Callable[[], None] | None = None,
    artifact_path: Path | None = None,
    audio_path_override: str | None = None,
    audio_source_key: str | None = None,
) -> TaskRun:
    """Run shared Silero VAD and persist native-resolution intervals in seconds.

    The default threshold and duration settings mirror
    ``notebooks/vox-profile-emotion-dim.ipynb``. The notebook's
    frame-level speech-ratio threshold is recorded in the manifest because
    it is applied downstream when aligning these intervals to affect windows,
    not during Silero timestamp generation.
    """
    audio = _coerce_audio(audio_path, sample_rate=sample_rate, recording_id=recording_id)
    config = _inference_config(
        task="vad",
        model_id=model_id,
        backbone=None,
        sample_rate=sample_rate,
        window_sec=0.0,
        hop_sec=0.0,
        batch_size=0,
        transform_policy="silero_vad_intervals_sec_v1",
        extra={
            "threshold": float(threshold),
            "speech_threshold": float(threshold),
            "min_speech_sec": float(min_speech_sec),
            "min_silence_sec": float(min_silence_sec),
            "frame_speech_ratio_threshold": float(DEFAULT_VAD_FRAME_SPEECH_RATIO_THRESHOLD),
        },
    )
    cached = _maybe_cached(out_dir, audio, "vad", config, reuse_cache, artifact_path)
    if cached is not None:
        _progress(progress, f"vad cache hit: {cached.path}")
        return TaskRun(cached, True)

    try:
        intervals = (
            detector(audio.samples, sample_rate)
            if detector is not None
            else _detect_silero_vad(
                audio.samples,
                sample_rate=sample_rate,
                threshold=threshold,
                min_speech_sec=min_speech_sec,
                min_silence_sec=min_silence_sec,
                device=device,
            )
        )
        intervals_sec = _validate_intervals(intervals)
        artifact = _write_task_artifact(
            out_dir,
            audio,
            task="vad",
            config=config,
            arrays={"intervals_sec": intervals_sec},
            model={
                "family": "vad",
                "id": model_id,
            },
            timing={
                "sample_rate": int(sample_rate),
                "window_sec": 0.0,
                "hop_sec": 0.0,
                "n_frames": int(len(intervals_sec)),
                "window_semantics": "sparse_intervals_sec",
            },
            runtime=_runtime(device=device, batch_size=0),
            artifact_path=artifact_path,
            audio_path_override=audio_path_override,
            audio_source_key=audio_source_key,
        )
        _progress(progress, f"vad wrote: {artifact.path}")
        return TaskRun(artifact, False)
    finally:
        (cleanup_cuda or cleanup_torch_memory)()


def run_all_inference(
    audio_path: str | Path | AudioData,
    *,
    out_dir: str | Path,
    affect_backbone: Backbone,
    disfluency_backbone: Backbone,
    recording_id: str | None = None,
    reuse_cache: bool = False,
    sample_rate: int = SAMPLE_RATE,
    batch_size: int = 128,
    affect_batch_size: int | None = None,
    disfluency_batch_size: int | None = None,
    emotion_batch_size: int | None = None,
    device: str | None = None,
    vad_threshold: float = DEFAULT_VAD_SPEECH_THRESHOLD,
    vad_min_speech_sec: float = DEFAULT_VAD_MIN_SPEECH_SEC,
    vad_min_silence_sec: float = DEFAULT_VAD_MIN_SILENCE_SEC,
    wavlm_autocast_dtype: str | None = None,
    wavlm_compile: bool = False,
    wavlm_compile_mode: str = "reduce-overhead",
    wavlm_compile_dynamic: bool = False,
    wavlm_stream_layer_sum: bool = False,
    wavlm_static_batch: bool = False,
    wavlm_runtime_preset: str | None = None,
    emotion_autocast_dtype: str | None = None,
    emotion_compile: bool = False,
    emotion_compile_mode: str = DEFAULT_EMOTION_COMPILE_MODE,
    emotion_runtime_mode: str | None = "auto",
    allow_tf32: bool = False,
    progress: ProgressFn | None = None,
    predictors: Mapping[str, Callable] | None = None,
    vad_detector: Callable[[np.ndarray, int], Sequence[tuple[float, float]]] | None = None,
    cleanup_cuda: Callable[[], None] | None = None,
    artifact_path_fn: Callable[[str], Path] | None = None,
    audio_path_override: str | None = None,
    audio_source_key: str | None = None,
    shutdown_check: Callable[[], bool] | None = None,
    tasks_filter: Sequence[str] | None = None,
    vad_gating: VadGating | None = None,
    vad_intervals: Sequence[tuple[float, float]] | None = None,
) -> InferenceRunResult:
    """Run VAD, affect, disfluency, and emotion sequentially.

    Parameters
    ----------
    audio_path:
        A file path or a pre-loaded ``AudioData`` to avoid redundant decoding.
    artifact_path_fn:
        If provided, ``fn(task_name)`` returns the directory where that task's
        artifact should be stored, overriding the default deep layout.
    audio_path_override:
        Durable audio URI written into manifests (e.g. an ``s3://`` URI)
        instead of the ephemeral local temp path.
    audio_source_key:
        Bare S3 key recorded in the manifest's ``audio.source_key`` field.
    batch_size:
        Legacy default batch size for all sub-runners.
    affect_batch_size, disfluency_batch_size, emotion_batch_size:
        Optional per-task overrides. When provided, these win over
        ``batch_size`` for the corresponding task.
    shutdown_check:
        Called between tasks.  If it returns ``True``, a
        :class:`ShutdownRequested` exception is raised so the orchestrator
        can exit cleanly.

    Example::

        result = run_all_inference(
            "input.mp3",
            out_dir="artifacts",
            affect_backbone="wavlm",
            disfluency_backbone="whisper",
            reuse_cache=True,
        )

        affect_path = result.artifacts["affect"].path

    The return value is task-keyed by ``vad``, ``affect``, ``disfluency``,
    and ``emotion``. This function writes inference artifacts only; it does
    not run producers or save review sessions.
    """
    emotion_settings = resolve_emotion_runtime_settings(
        mode=emotion_runtime_mode,
        default_mode="auto",
        device=device,
        autocast_dtype=emotion_autocast_dtype,
        compile_model=emotion_compile,
        compile_mode=emotion_compile_mode,
        allow_tf32=allow_tf32,
    )
    resolved_batches = resolve_task_batch_sizes(
        batch_size=batch_size,
        affect_batch_size=affect_batch_size,
        disfluency_batch_size=disfluency_batch_size,
        emotion_batch_size=(
            OPTIMIZED_EMOTION_BATCH_SIZE
            if emotion_batch_size is None and emotion_settings.mode == "optimized"
            else emotion_batch_size
        ),
    )
    predictors = dict(predictors or {})
    if isinstance(audio_path, AudioData):
        audio = audio_path
    else:
        audio = load_audio(audio_path, sample_rate=sample_rate, recording_id=recording_id)
    _progress(progress, f"audio sha256: {audio.audio_sha256}")

    def _ap(task: str) -> Path | None:
        return artifact_path_fn(task) if artifact_path_fn is not None else None

    # VAD-gating: resolve the speech intervals once and feed them to the GPU
    # tasks.  Intervals come from (1) an explicit caller value, (2) the vad
    # step when it runs in this call, or (3) an existing vad artifact on disk.
    # If none are available, gating is a no-op for this run (full timeline).
    gating = vad_gating if (vad_gating is not None and vad_gating.active) else None
    gate_state = {
        "intervals": (list(vad_intervals) if vad_intervals is not None else None)
    }
    if gating is not None and gate_state["intervals"] is None:
        gate_state["intervals"] = _load_vad_intervals_from_artifact(_ap("vad"))

    artifacts: dict[str, PredictionArtifact] = {}
    reused: dict[str, bool] = {}
    steps = [
        (
            "vad",
            lambda: run_vad(
                audio,
                out_dir=out_dir,
                reuse_cache=reuse_cache,
                device=device,
                threshold=vad_threshold,
                min_speech_sec=vad_min_speech_sec,
                min_silence_sec=vad_min_silence_sec,
                detector=vad_detector,
                progress=progress,
                cleanup_cuda=cleanup_cuda,
                artifact_path=_ap("vad"),
                audio_path_override=audio_path_override,
                audio_source_key=audio_source_key,
            ),
        ),
        (
            "affect",
            lambda: run_affect_inference(
                audio,
                out_dir=out_dir,
                backbone=affect_backbone,
                reuse_cache=reuse_cache,
                batch_size=resolved_batches["affect"],
                device=device,
                wavlm_autocast_dtype=wavlm_autocast_dtype,
                wavlm_compile=wavlm_compile,
                wavlm_compile_mode=wavlm_compile_mode,
                wavlm_compile_dynamic=wavlm_compile_dynamic,
                wavlm_stream_layer_sum=wavlm_stream_layer_sum,
                wavlm_static_batch=wavlm_static_batch,
                wavlm_runtime_preset=wavlm_runtime_preset,
                allow_tf32=allow_tf32,
                predictor=predictors.get("affect"),
                progress=progress,
                cleanup_cuda=cleanup_cuda,
                artifact_path=_ap("affect"),
                audio_path_override=audio_path_override,
                audio_source_key=audio_source_key,
                vad_intervals=gate_state["intervals"],
                vad_gating=gating,
            ),
        ),
        (
            "disfluency",
            lambda: run_disfluency_inference(
                audio,
                out_dir=out_dir,
                backbone=disfluency_backbone,
                reuse_cache=reuse_cache,
                batch_size=resolved_batches["disfluency"],
                device=device,
                wavlm_autocast_dtype=wavlm_autocast_dtype,
                wavlm_compile=wavlm_compile,
                wavlm_compile_mode=wavlm_compile_mode,
                wavlm_compile_dynamic=wavlm_compile_dynamic,
                wavlm_stream_layer_sum=wavlm_stream_layer_sum,
                wavlm_static_batch=wavlm_static_batch,
                wavlm_runtime_preset=wavlm_runtime_preset,
                allow_tf32=allow_tf32,
                predictor=predictors.get("disfluency"),
                progress=progress,
                cleanup_cuda=cleanup_cuda,
                artifact_path=_ap("disfluency"),
                audio_path_override=audio_path_override,
                audio_source_key=audio_source_key,
                vad_intervals=gate_state["intervals"],
                vad_gating=gating,
            ),
        ),
        (
            "emotion",
            lambda: run_emotion_inference(
                audio,
                out_dir=out_dir,
                reuse_cache=reuse_cache,
                batch_size=resolved_batches["emotion"],
                device=emotion_settings.device,
                autocast_dtype=emotion_autocast_dtype,
                compile_model=emotion_compile,
                compile_mode=emotion_compile_mode,
                emotion_runtime_mode=emotion_runtime_mode,
                allow_tf32=allow_tf32,
                predictor=predictors.get("emotion"),
                progress=progress,
                cleanup_cuda=cleanup_cuda,
                artifact_path=_ap("emotion"),
                audio_path_override=audio_path_override,
                audio_source_key=audio_source_key,
                vad_intervals=gate_state["intervals"],
                vad_gating=gating,
            ),
        ),
    ]
    if tasks_filter is not None:
        wanted = set(tasks_filter)
        unknown = wanted - {task for task, _ in steps}
        if unknown:
            raise ValueError(f"Unknown tasks_filter entries: {sorted(unknown)}")
        steps = [(task, step) for task, step in steps if task in wanted]
    task_elapsed_sec: dict[str, float] = {}

    def _run_task(task: str, run_step):
        if shutdown_check is not None and shutdown_check():
            raise ShutdownRequested(f"shutdown requested before task {task!r}")
        _progress(progress, f"run-all task: {task}")
        task_started = time.perf_counter()
        batch_for_task = resolved_batches.get(task, 0)
        cuda_before = _cuda_memory_snapshot()
        _reset_cuda_peak_memory_stats()
        try:
            result = run_step()
        except Exception:
            LOGGER.info(
                "run-all task failed: task=%s batch_size=%d elapsed=%.3fs cuda=%s",
                task, batch_for_task, time.perf_counter() - task_started,
                _cuda_task_stats(cuda_before),
            )
            raise
        elapsed = time.perf_counter() - task_started
        LOGGER.info(
            "run-all task complete: task=%s reused=%s batch_size=%d elapsed=%.3fs cuda=%s",
            task, result.reused, batch_for_task, elapsed, _cuda_task_stats(cuda_before),
        )
        return result, elapsed

    def _record(task, result, elapsed):
        artifacts[task] = result.artifact
        reused[task] = result.reused
        task_elapsed_sec[task] = elapsed

    # VAD first (sequential): the gated model tasks read its speech intervals.
    for task, run_step in steps:
        if task != "vad":
            continue
        result, elapsed = _run_task(task, run_step)
        _record(task, result, elapsed)
        if gating is not None and gate_state["intervals"] is None:
            gate_state["intervals"] = intervals_from_array(
                result.artifact.arrays.get("intervals_sec")
            )

    # Model tasks (affect/disfluency/emotion). When every one runs through an
    # injected (Triton) predictor, overlap them concurrently. The win is NOT the
    # GPU running three models at once (at batch 128 each forward already
    # saturates the SMs, so co-resident executions just time-slice — cf. MPS
    # co-location nets only ~1.49x); it's that each thread mostly *waits* on its
    # own server round-trips + CPU framing, so overlapping hides those waits and
    # the per-archive wall drops from sum(tasks) toward max(task).
    # The in-process path (predictors=None: models run on the worker's own GPU)
    # keeps them sequential — naive threads would serialize on the default CUDA
    # stream and the GIL anyway, so it'd add contention with no real overlap
    # (true packing there needs MPS, a separate mechanism).
    model_steps = [(t, s) for t, s in steps if t != "vad"]
    triton_mode = bool(model_steps) and all(
        predictors.get(t) is not None for t, _ in model_steps
    )
    if triton_mode and len(model_steps) > 1:
        _progress(progress, f"run-all: {len(model_steps)} model tasks concurrent (triton)")
        with ThreadPoolExecutor(max_workers=len(model_steps)) as ex:
            futures = {ex.submit(_run_task, t, s): t for t, s in model_steps}
            collected = {futures[f]: f.result() for f in as_completed(futures)}
        for task, _ in model_steps:
            result, elapsed = collected[task]
            _record(task, result, elapsed)
    else:
        for task, run_step in model_steps:
            result, elapsed = _run_task(task, run_step)
            _record(task, result, elapsed)

    return InferenceRunResult(
        artifacts=artifacts, reused=reused, task_elapsed_sec=task_elapsed_sec,
    )


def emotion2vec_scores_to_probabilities(
    raw_scores,
    raw_labels: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Convert emotion2vec scores into producer-ready canonical probabilities."""
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"raw_scores must be 2-D [frames, classes], got {scores.shape}")
    if scores.shape[1] != len(raw_labels):
        raise ValueError(
            f"raw_scores has {scores.shape[1]} columns but {len(raw_labels)} labels"
        )
    if not np.isfinite(scores).all():
        raise ValueError("raw_scores must contain only finite values")
    if (scores < -1e-7).any():
        raise ValueError("raw_scores must be nonnegative for emotion2vec transform")

    folded = np.zeros((scores.shape[0], len(CANONICAL_CHANNELS)), dtype=np.float64)
    for src_i, raw_label in enumerate(raw_labels):
        label = normalize_label(raw_label)
        if label not in CANONICAL_CHANNELS:
            raise ValueError(f"Unsupported emotion2vec label {raw_label!r} -> {label!r}")
        folded[:, CANONICAL_CHANNELS.index(label)] += scores[:, src_i]

    row_sums = folded.sum(axis=1, keepdims=True)
    if (row_sums <= 0.0).any():
        raise ValueError("emotion2vec transform cannot normalize zero-score rows")
    probabilities = folded / row_sums
    tolerance = EmotionConfig.balanced().probability_sum_tolerance
    observed = probabilities.sum(axis=1)
    if not np.allclose(observed, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError(
            "emotion probabilities must sum to 1.0 after normalization; "
            f"observed min={observed.min():.6f}, max={observed.max():.6f}"
        )
    return probabilities.astype(np.float32), CANONICAL_CHANNELS


def resolve_task_batch_sizes(
    *,
    batch_size: int,
    affect_batch_size: int | None = None,
    disfluency_batch_size: int | None = None,
    emotion_batch_size: int | None = None,
) -> dict[str, int]:
    batches = {
        "affect": int(batch_size if affect_batch_size is None else affect_batch_size),
        "disfluency": int(batch_size if disfluency_batch_size is None else disfluency_batch_size),
        "emotion": int(batch_size if emotion_batch_size is None else emotion_batch_size),
    }
    for task, value in batches.items():
        if value <= 0:
            raise ValueError(f"{task} batch size must be positive")
    return batches


def _resolve_emotion_batch_size(
    batch_size: int | None,
    *,
    settings,
    fallback_batch_size: int,
) -> int:
    value = (
        OPTIMIZED_EMOTION_BATCH_SIZE
        if batch_size is None and settings.mode == "optimized"
        else fallback_batch_size if batch_size is None else int(batch_size)
    )
    if value <= 0:
        raise ValueError("emotion batch size must be positive")
    return value


def _emotion_runtime_config_extra(
    *,
    autocast_dtype: str | None,
    compile_model: bool,
    compile_mode: str,
    allow_tf32: bool,
) -> dict[str, object]:
    extra: dict[str, object] = {}
    if autocast_dtype is not None:
        extra["torch_autocast_dtype"] = autocast_dtype
    if compile_model:
        extra["torch_compile"] = True
        extra["torch_compile_mode"] = compile_mode
    if allow_tf32:
        extra["torch_allow_tf32"] = True
    return extra


def _wavlm_runtime_config_extra(
    *,
    backbone: str,
    autocast_dtype: str | None,
    compile_model: bool,
    compile_mode: str,
    compile_dynamic: bool,
    stream_layer_sum: bool,
    allow_tf32: bool,
) -> dict[str, object]:
    if backbone != "wavlm":
        return {}
    return {
        "torch_autocast_dtype": autocast_dtype,
        "torch_compile": bool(compile_model),
        "torch_compile_target": "wavlm_backbone",
        "torch_compile_mode": str(compile_mode),
        "torch_compile_dynamic": bool(compile_dynamic),
        "wavlm_stream_layer_sum": bool(stream_layer_sum),
        "torch_allow_tf32": bool(allow_tf32),
    }


def _wavlm_effective_runtime_config(
    *,
    backbone: str,
    predictor: object | None,
    autocast_dtype: str | None,
    compile_model: bool,
    compile_mode: str,
    compile_dynamic: bool,
    stream_layer_sum: bool,
    allow_tf32: bool,
) -> dict[str, object]:
    if backbone != "wavlm" or predictor is None:
        return {
            "autocast_dtype": autocast_dtype,
            "compile_model": compile_model,
            "compile_mode": compile_mode,
            "compile_dynamic": compile_dynamic,
            "stream_layer_sum": stream_layer_sum,
            "allow_tf32": allow_tf32,
        }
    return {
        "autocast_dtype": getattr(predictor, "wavlm_autocast_dtype", autocast_dtype),
        "compile_model": getattr(predictor, "wavlm_compile", compile_model),
        "compile_mode": getattr(predictor, "wavlm_compile_mode", compile_mode),
        "compile_dynamic": getattr(
            predictor,
            "wavlm_compile_dynamic",
            compile_dynamic,
        ),
        "stream_layer_sum": getattr(
            predictor,
            "wavlm_stream_layer_sum",
            stream_layer_sum,
        ),
        "allow_tf32": allow_tf32,
    }


def cleanup_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _coerce_audio(
    audio_path: str | Path | AudioData,
    *,
    sample_rate: int,
    recording_id: str | None,
) -> AudioData:
    if isinstance(audio_path, AudioData):
        return audio_path
    return load_audio(audio_path, sample_rate=sample_rate, recording_id=recording_id)


def _inference_config(
    *,
    task: str,
    model_id: str,
    backbone: str | None,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    batch_size: int,
    transform_policy: str,
    extra: Mapping | None = None,
) -> dict:
    out = {
        "task": task,
        "model_id": model_id,
        "backbone": backbone,
        "sample_rate": int(sample_rate),
        "window_sec": float(window_sec),
        "hop_sec": float(hop_sec),
        "batch_size": int(batch_size),
        "transform_policy": transform_policy,
    }
    if extra:
        out.update(dict(extra))
    return out


def _maybe_cached(
    out_dir: str | Path,
    audio: AudioData,
    task: str,
    config: Mapping,
    reuse_cache: bool,
    explicit_artifact_path: Path | None = None,
) -> PredictionArtifact | None:
    if not reuse_cache:
        return None
    if explicit_artifact_path is not None:
        return _validate_explicit_artifact(
            explicit_artifact_path, audio, task, config,
        )
    return find_cached_artifact(
        out_dir,
        recording_id=audio.recording_id,
        audio_sha256=audio.audio_sha256,
        task=task,
        inference_config_hash_value=inference_config_hash(config),
    )


def _validate_explicit_artifact(
    artifact_path: Path,
    audio: AudioData,
    task: str,
    config: Mapping,
) -> PredictionArtifact | None:
    """Validate an artifact at a caller-specified path against current config.

    With a flat layout the config hash is no longer embedded in the directory
    structure, so we must open the manifest and verify that the task, audio
    SHA256, and inference_config_hash all match the current run parameters.
    """
    manifest_path = artifact_path / MANIFEST_FILENAME
    predictions_path = artifact_path / PREDICTIONS_FILENAME
    if not manifest_path.is_file() or not predictions_path.is_file():
        return None
    try:
        artifact = load_prediction_artifact(artifact_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    m = artifact.manifest
    expected_hash = inference_config_hash(config)
    if m.get("task") != task:
        return None
    if m.get("audio", {}).get("sha256") != audio.audio_sha256:
        return None
    if m.get("inference_config_hash") == expected_hash:
        return artifact
    manifest_config = m.get("inference_config")
    if not isinstance(manifest_config, Mapping):
        return None
    if not inference_configs_match(manifest_config, config, ignore_batch_size=True):
        return None
    return artifact


def _write_task_artifact(
    out_dir: str | Path,
    audio: AudioData,
    *,
    task: str,
    config: Mapping,
    arrays: Mapping[str, np.ndarray],
    model: Mapping,
    timing: Mapping,
    runtime: Mapping,
    labels: Sequence[str] | None = None,
    artifact_path: Path | None = None,
    audio_path_override: str | None = None,
    audio_source_key: str | None = None,
) -> PredictionArtifact:
    config_hash = inference_config_hash(config)
    if artifact_path is not None:
        path = artifact_path
    else:
        path = artifact_dir(
            out_dir,
            recording_id=audio.recording_id,
            audio_sha256=audio.audio_sha256,
            task=task,
            inference_config_hash_value=config_hash,
        )
    manifest = base_manifest(
        task=task,
        recording_id=audio.recording_id,
        audio_path=audio.path,
        audio_sha256=audio.audio_sha256,
        sample_rate=audio.sample_rate,
        duration_sec=audio.duration_sec,
        inference_config=config,
        inference_config_hash_value=config_hash,
        model=model,
        timing=timing,
        runtime=runtime,
        labels=labels,
        audio_path_override=audio_path_override,
        audio_source_key=audio_source_key,
    )
    return write_prediction_artifact(path, manifest=manifest, arrays=arrays)


def _validate_affect_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = ("arousal", "valence", "dominance")
    out = {name: np.asarray(arrays[name], dtype=np.float32).reshape(-1) for name in required}
    n_frames = {values.shape[0] for values in out.values()}
    if len(n_frames) != 1:
        raise ValueError("affect arrays must have equal frame counts")
    for name, values in out.items():
        if not np.isfinite(values).all():
            raise ValueError(f"affect array {name!r} contains non-finite values")
    return out


def _validate_disfluency_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    fluency = np.asarray(arrays["fluency_logits"], dtype=np.float32)
    types = np.asarray(arrays["disfluency_type_logits"], dtype=np.float32)
    if fluency.ndim != 2 or fluency.shape[1] != 2:
        raise ValueError(f"fluency_logits must have shape [frames, 2], got {fluency.shape}")
    if types.ndim != 2 or types.shape[1] != 5:
        raise ValueError(f"disfluency_type_logits must have shape [frames, 5], got {types.shape}")
    if fluency.shape[0] != types.shape[0]:
        raise ValueError("disfluency logits must share frame count")
    if not np.isfinite(fluency).all() or not np.isfinite(types).all():
        raise ValueError("disfluency logits must be finite")
    return {"fluency_logits": fluency, "disfluency_type_logits": types}


def _validate_intervals(intervals: Sequence[tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(intervals, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"VAD intervals must have shape [n, 2], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("VAD intervals must be finite")
    if (arr[:, 1] <= arr[:, 0]).any():
        raise ValueError("VAD intervals must have positive duration")
    return arr


def _timing(sample_rate: int, window_sec: float, hop_sec: float, n_frames: int) -> dict:
    receptive_field_end_sec = (
        (int(n_frames) - 1) * float(hop_sec) + float(window_sec)
        if int(n_frames) > 0
        else 0.0
    )
    return {
        "sample_rate": int(sample_rate),
        "window_sec": float(window_sec),
        "hop_sec": float(hop_sec),
        "n_frames": int(n_frames),
        "receptive_field_end_sec": float(receptive_field_end_sec),
        "window_semantics": "frame summarizes [i*hop, i*hop + window]",
    }


def _runtime(
    *,
    device: str | None,
    batch_size: int,
    extra: Mapping[str, object] | None = None,
) -> dict:
    runtime = {
        "device": device or _default_device(),
        "batch_size": int(batch_size),
    }
    if extra:
        runtime.update(dict(extra))
    return runtime


def _wavlm_runtime_metadata(
    *,
    backbone: str,
    runtime_preset: str | None,
    static_batch: bool,
    predictor: object | None = None,
) -> dict[str, object]:
    if backbone != "wavlm":
        return {}
    actual_preset = getattr(predictor, "wavlm_runtime_preset", runtime_preset)
    actual_static_batch = bool(getattr(predictor, "wavlm_static_batch", static_batch))
    metadata: dict[str, object] = {
        "wavlm_preprocessing_policy": WAVLM_PREPROCESSING_POLICY,
    }
    if actual_preset is not None:
        metadata["wavlm_runtime_preset"] = actual_preset
    if actual_static_batch:
        metadata["wavlm_static_batch_padding_policy"] = (
            WAVLM_STATIC_BATCH_PADDING_POLICY
        )
    warmup_sec = getattr(predictor, "wavlm_warmup_sec", None)
    if warmup_sec is not None:
        metadata["wavlm_warmup_sec"] = float(warmup_sec)
    return metadata


def _cuda_memory_snapshot() -> dict[str, float] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated_gib": float(torch.cuda.memory_allocated()) / 2**30,
            "reserved_gib": float(torch.cuda.memory_reserved()) / 2**30,
            "free_gib": float(free) / 2**30,
            "total_gib": float(total) / 2**30,
        }
    except Exception:
        return None


def _reset_cuda_peak_memory_stats() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _cuda_task_stats(before: dict[str, float] | None) -> dict[str, object] | None:
    after = _cuda_memory_snapshot()
    if before is None and after is None:
        return None
    out: dict[str, object] = {
        "before": before,
        "after": after,
    }
    try:
        import torch

        if torch.cuda.is_available():
            out["peak_allocated_gib"] = float(torch.cuda.max_memory_allocated()) / 2**30
            out["peak_reserved_gib"] = float(torch.cuda.max_memory_reserved()) / 2**30
    except Exception:
        pass
    return out


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _predict_affect(
    windows: np.ndarray,
    *,
    backbone: str,
    model_id: str,
    device: str | None,
    batch_size: int,
    wavlm_autocast_dtype: str | None,
    wavlm_compile: bool,
    wavlm_compile_mode: str,
    wavlm_compile_dynamic: bool,
    wavlm_stream_layer_sum: bool,
    wavlm_static_batch: bool,
    allow_tf32: bool,
    progress: ProgressFn | None,
) -> dict[str, np.ndarray]:
    import torch
    from .models import configure_torch_matmul

    if len(windows) == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return {"arousal": empty.copy(), "valence": empty.copy(), "dominance": empty.copy()}

    configure_torch_matmul(allow_tf32=allow_tf32)
    wrapper = _load_affect_wrapper(backbone)
    run_device = device or _default_device()
    model = wrapper.from_pretrained(model_id).to(run_device).eval()
    if backbone == "wavlm" and wavlm_stream_layer_sum:
        model.wavlm_stream_layer_sum = True
    if backbone == "wavlm" and wavlm_compile:
        compile_wavlm_backbone(
            model,
            mode=wavlm_compile_mode,
            dynamic=wavlm_compile_dynamic,
        )
    windows, valid_count = pad_windows_to_static_batch(
        windows,
        batch_size=batch_size,
        enabled=backbone == "wavlm" and wavlm_static_batch,
    )
    arousal, valence, dominance = [], [], []
    with torch.inference_mode():
        for batch_np in _batches(windows, batch_size, progress, "affect"):
            batch = torch.from_numpy(writable_contiguous_float32(batch_np))
            if backbone != "wavlm":
                batch = batch.to(run_device)
            with autocast_context(
                torch,
                run_device,
                wavlm_autocast_dtype if backbone == "wavlm" else None,
            ):
                a, v, d = model(batch)
            arousal.append(a.detach().float().reshape(-1))
            valence.append(v.detach().float().reshape(-1))
            dominance.append(d.detach().float().reshape(-1))
    return {
        "arousal": torch.cat(arousal)[:valid_count].cpu().numpy(),
        "valence": torch.cat(valence)[:valid_count].cpu().numpy(),
        "dominance": torch.cat(dominance)[:valid_count].cpu().numpy(),
    }


def _predict_disfluency(
    windows: np.ndarray,
    *,
    backbone: str,
    model_id: str,
    device: str | None,
    batch_size: int,
    wavlm_autocast_dtype: str | None,
    wavlm_compile: bool,
    wavlm_compile_mode: str,
    wavlm_compile_dynamic: bool,
    wavlm_stream_layer_sum: bool,
    wavlm_static_batch: bool,
    allow_tf32: bool,
    progress: ProgressFn | None,
) -> dict[str, np.ndarray]:
    import torch
    from .models import configure_torch_matmul

    if len(windows) == 0:
        return {
            "fluency_logits": np.zeros((0, 2), dtype=np.float32),
            "disfluency_type_logits": np.zeros((0, 5), dtype=np.float32),
        }

    configure_torch_matmul(allow_tf32=allow_tf32)
    wrapper = _load_disfluency_wrapper(backbone)
    run_device = device or _default_device()
    model = wrapper.from_pretrained(model_id).to(run_device).eval()
    if backbone == "wavlm" and wavlm_stream_layer_sum:
        model.wavlm_stream_layer_sum = True
    if backbone == "wavlm" and wavlm_compile:
        compile_wavlm_backbone(
            model,
            mode=wavlm_compile_mode,
            dynamic=wavlm_compile_dynamic,
        )
    windows, valid_count = pad_windows_to_static_batch(
        windows,
        batch_size=batch_size,
        enabled=backbone == "wavlm" and wavlm_static_batch,
    )
    fluency, dysfluency = [], []
    with torch.inference_mode():
        for batch_np in _batches(windows, batch_size, progress, "disfluency"):
            batch = torch.from_numpy(writable_contiguous_float32(batch_np))
            if backbone != "wavlm":
                batch = batch.to(run_device)
            with autocast_context(
                torch,
                run_device,
                wavlm_autocast_dtype if backbone == "wavlm" else None,
            ):
                f, d = model(batch, return_feature=False)
            fluency.append(f.detach().float())
            dysfluency.append(d.detach().float())
    return {
        "fluency_logits": torch.cat(fluency, dim=0)[:valid_count].cpu().numpy(),
        "disfluency_type_logits": torch.cat(dysfluency, dim=0)[:valid_count].cpu().numpy(),
    }


def _predict_emotion2vec(
    samples: np.ndarray,
    *,
    model_id: str,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    batch_size: int,
    device: str | None,
    autocast_dtype: str | None,
    compile_model: bool,
    compile_mode: str,
    allow_tf32: bool,
    progress: ProgressFn | None,
) -> tuple[np.ndarray, Sequence[str]]:
    from funasr import AutoModel

    auto_kwargs = {
        "model": model_id,
        "batch_size": batch_size,
        "disable_update": True,
        "disable_pbar": True,
    }
    if device is not None:
        auto_kwargs["device"] = device
    with torch_matmul_precision(allow_tf32=allow_tf32):
        model = AutoModel(**auto_kwargs)
        return predict_emotion2vec_scores_from_audio(
            model,
            samples,
            sample_rate=sample_rate,
            window_sec=window_sec,
            hop_sec=hop_sec,
            batch_size=batch_size,
            autocast_dtype=autocast_dtype,
            compile_model=compile_model,
            compile_mode=compile_mode,
            progress=progress,
        )


def _predict_emotion2vec_windows(
    windows: np.ndarray,
    *,
    model_id: str,
    sample_rate: int,
    batch_size: int,
    device: str | None,
    autocast_dtype: str | None,
    compile_model: bool,
    compile_mode: str,
    allow_tf32: bool,
    progress: ProgressFn | None,
) -> tuple[np.ndarray, Sequence[str]]:
    """No-predictor emotion2vec scoring over an explicit windows array."""
    from funasr import AutoModel

    auto_kwargs = {
        "model": model_id,
        "batch_size": batch_size,
        "disable_update": True,
        "disable_pbar": True,
    }
    if device is not None:
        auto_kwargs["device"] = device
    with torch_matmul_precision(allow_tf32=allow_tf32):
        model = AutoModel(**auto_kwargs)
        return predict_emotion2vec_scores(
            model,
            windows,
            sample_rate=sample_rate,
            batch_size=batch_size,
            autocast_dtype=autocast_dtype,
            compile_model=compile_model,
            compile_mode=compile_mode,
            progress=progress,
        )


def _emotion_scores_gated(
    samples: np.ndarray,
    *,
    predictor,
    model_id: str,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    batch_size: int,
    runtime_settings,
    vad_intervals: Sequence[tuple[float, float]],
    vad_gating: VadGating,
    progress: ProgressFn | None,
) -> tuple[np.ndarray, Sequence[str]]:
    """Score only speech windows, scatter into a full-length raw-score array.

    Non-speech rows are filled with a uniform-positive sentinel (ones) so the
    existing fold+normalize+sum-to-1 conversion stays valid; the emotion
    producer never reads them (it gates on speech). Kept rows are identical to
    the ungated framed path.
    """
    windows = frame_audio(
        samples, sample_rate=sample_rate, window_sec=window_sec, hop_sec=hop_sec
    )
    n_windows = len(windows)
    mask = speech_window_mask(
        n_windows, hop_sec, window_sec, vad_intervals, bridge_sec=vad_gating.bridge_sec
    )
    kept = np.nonzero(mask)[0]
    _progress(progress, f"emotion gated windows: {kept.size}/{n_windows}")

    def _score(batch_windows):
        if predictor is not None:
            return predictor(batch_windows)
        return _predict_emotion2vec_windows(
            batch_windows,
            model_id=model_id,
            sample_rate=sample_rate,
            batch_size=batch_size,
            device=runtime_settings.device,
            autocast_dtype=runtime_settings.autocast_dtype,
            compile_model=runtime_settings.compile_model,
            compile_mode=runtime_settings.compile_mode,
            allow_tf32=runtime_settings.allow_tf32,
            progress=progress,
        )

    if kept.size:
        raw_sub, raw_labels = _score(windows[kept])
        raw_scores = scatter_fill(
            np.asarray(raw_sub, dtype=np.float32), kept, n_windows, fill=1.0
        )
    else:
        # All silence: score one probe window only to learn the label set,
        # then fill the whole timeline with the uniform sentinel.
        _, raw_labels = _score(windows[:1])
        raw_scores = np.full((n_windows, len(raw_labels)), 1.0, dtype=np.float32)
    return raw_scores, raw_labels


def _detect_silero_vad(
    samples: np.ndarray,
    *,
    sample_rate: int,
    threshold: float,
    min_speech_sec: float,
    min_silence_sec: float,
    device: str | None,
) -> list[tuple[float, float]]:
    import torch

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
        onnx=False,
    )
    get_speech_timestamps = utils[0]
    model = model.to("cpu")
    timestamps = get_speech_timestamps(
        torch.from_numpy(writable_contiguous_float32(samples)),
        model,
        sampling_rate=sample_rate,
        threshold=float(threshold),
        min_speech_duration_ms=int(float(min_speech_sec) * 1000),
        min_silence_duration_ms=int(float(min_silence_sec) * 1000),
        return_seconds=False,
    )
    return [
        (float(item["start"]) / float(sample_rate), float(item["end"]) / float(sample_rate))
        for item in timestamps
    ]


def _load_affect_wrapper(backbone: str):
    if backbone == "wavlm":
        from ...vox_profile.emotion.wavlm_emotion_dim import WavLMWrapper

        return WavLMWrapper
    if backbone == "whisper":
        from ...vox_profile.emotion.whisper_emotion_dim import WhisperWrapper

        return WhisperWrapper
    raise ValueError(f"Unknown affect backbone {backbone!r}")


def _load_disfluency_wrapper(backbone: str):
    if backbone == "wavlm":
        from ...vox_profile.fluency.wavlm_fluency import WavLMWrapper

        return WavLMWrapper
    if backbone == "whisper":
        from ...vox_profile.fluency.whisper_fluency import WhisperWrapper

        return WhisperWrapper
    raise ValueError(f"Unknown disfluency backbone {backbone!r}")


def _batches(
    windows: np.ndarray,
    batch_size: int,
    progress: ProgressFn | None,
    task: str,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    n = len(windows)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        _progress(progress, f"{task} batch {start}:{end} / {n}")
        yield windows[start:end]


def compute_inference_config(
    *,
    task: str,
    model_id: str,
    backbone: str | None,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    batch_size: int,
    transform_policy: str,
    extra: Mapping | None = None,
) -> dict:
    """Public wrapper around the internal config builder.

    The orchestrator uses this to compute expected inference_config_hash
    values for stale-artifact detection without duplicating the config
    construction logic.
    """
    return _inference_config(
        task=task,
        model_id=model_id,
        backbone=backbone,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
        batch_size=batch_size,
        transform_policy=transform_policy,
        extra=extra,
    )


def _progress(progress: ProgressFn | None, message: str) -> None:
    if progress is not None:
        progress(message)
    else:
        LOGGER.info(message)
