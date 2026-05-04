"""Inference runners that persist producer-ready prediction artifacts."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..producers.emotion.config import CANONICAL_CHANNELS, Config as EmotionConfig
from ..producers.emotion.pipeline import normalize_label
from .artifacts import (
    InferenceRunResult,
    PredictionArtifact,
    SAMPLE_RATE,
    artifact_dir,
    base_manifest,
    find_cached_artifact,
    inference_config_hash,
    write_prediction_artifact,
)
from .audio import AudioData, frame_audio, load_audio
from .log import get_logger


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
DEFAULT_VAD_SPEECH_THRESHOLD = 0.4
DEFAULT_VAD_MIN_SPEECH_SEC = 0.1
DEFAULT_VAD_MIN_SILENCE_SEC = 1.2
DEFAULT_VAD_FRAME_SPEECH_RATIO_THRESHOLD = 0.1

Backbone = str
ProgressFn = Callable[[str], None]
LOGGER = get_logger()


@dataclass(frozen=True)
class TaskRun:
    artifact: PredictionArtifact
    reused: bool


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
    predictor: Callable[[np.ndarray], Mapping[str, np.ndarray]] | None = None,
    progress: ProgressFn | None = None,
    cleanup_cuda: Callable[[], None] | None = None,
) -> TaskRun:
    """Run Vox-Profile dimensional affect inference and persist A/V/D arrays."""
    if backbone not in DEFAULT_AFFECT_MODELS:
        raise ValueError("backbone must be one of: wavlm, whisper")
    audio = _coerce_audio(audio_path, sample_rate=sample_rate, recording_id=recording_id)
    resolved_model_id = model_id or DEFAULT_AFFECT_MODELS[backbone]
    config = _inference_config(
        task="affect",
        model_id=resolved_model_id,
        backbone=backbone,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
        batch_size=batch_size,
        transform_policy="vox_profile_affect_sigmoid_heads_v1",
    )
    cached = _maybe_cached(out_dir, audio, "affect", config, reuse_cache)
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
        arrays = (
            predictor(windows)
            if predictor is not None
            else _predict_affect(
                windows,
                backbone=backbone,
                model_id=resolved_model_id,
                device=device,
                batch_size=batch_size,
                progress=progress,
            )
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
            runtime=_runtime(device=device, batch_size=batch_size),
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
    batch_size: int = 20,
    device: str | None = None,
    predictor: Callable[[np.ndarray], Mapping[str, np.ndarray]] | None = None,
    progress: ProgressFn | None = None,
    cleanup_cuda: Callable[[], None] | None = None,
) -> TaskRun:
    """Run Vox-Profile disfluency inference and persist raw logits."""
    if backbone not in DEFAULT_DISFLUENCY_MODELS:
        raise ValueError("backbone must be one of: wavlm, whisper")
    audio = _coerce_audio(audio_path, sample_rate=sample_rate, recording_id=recording_id)
    resolved_model_id = model_id or DEFAULT_DISFLUENCY_MODELS[backbone]
    config = _inference_config(
        task="disfluency",
        model_id=resolved_model_id,
        backbone=backbone,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
        batch_size=batch_size,
        transform_policy="vox_profile_disfluency_raw_logits_v1",
    )
    cached = _maybe_cached(out_dir, audio, "disfluency", config, reuse_cache)
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
        arrays = (
            predictor(windows)
            if predictor is not None
            else _predict_disfluency(
                windows,
                backbone=backbone,
                model_id=resolved_model_id,
                device=device,
                batch_size=batch_size,
                progress=progress,
            )
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
            runtime=_runtime(device=device, batch_size=batch_size),
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
    batch_size: int = 25,
    device: str | None = None,
    predictor: Callable[[np.ndarray], tuple[np.ndarray, Sequence[str]]] | None = None,
    progress: ProgressFn | None = None,
    cleanup_cuda: Callable[[], None] | None = None,
) -> TaskRun:
    """Run emotion2vec inference and persist canonical probabilities."""
    audio = _coerce_audio(audio_path, sample_rate=sample_rate, recording_id=recording_id)
    config = _inference_config(
        task="emotion",
        model_id=model_id,
        backbone=None,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
        batch_size=batch_size,
        transform_policy="emotion2vec_fold_row_normalize_v1",
    )
    cached = _maybe_cached(out_dir, audio, "emotion", config, reuse_cache)
    if cached is not None:
        _progress(progress, f"emotion cache hit: {cached.path}")
        return TaskRun(cached, True)

    _progress(progress, "emotion framing audio")
    windows = frame_audio(
        audio.samples,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )
    _progress(progress, f"emotion windows: {len(windows)}")
    try:
        raw_scores, raw_labels = (
            predictor(windows)
            if predictor is not None
            else _predict_emotion2vec(
                windows,
                model_id=model_id,
                sample_rate=sample_rate,
                batch_size=batch_size,
                progress=progress,
            )
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
            timing=_timing(sample_rate, window_sec, hop_sec, len(windows)),
            runtime=_runtime(device=device, batch_size=batch_size),
            labels=labels,
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
    cached = _maybe_cached(out_dir, audio, "vad", config, reuse_cache)
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
        )
        _progress(progress, f"vad wrote: {artifact.path}")
        return TaskRun(artifact, False)
    finally:
        (cleanup_cuda or cleanup_torch_memory)()


def run_all_inference(
    audio_path: str | Path,
    *,
    out_dir: str | Path,
    affect_backbone: Backbone,
    disfluency_backbone: Backbone,
    recording_id: str | None = None,
    reuse_cache: bool = False,
    sample_rate: int = SAMPLE_RATE,
    device: str | None = None,
    vad_threshold: float = DEFAULT_VAD_SPEECH_THRESHOLD,
    vad_min_speech_sec: float = DEFAULT_VAD_MIN_SPEECH_SEC,
    vad_min_silence_sec: float = DEFAULT_VAD_MIN_SILENCE_SEC,
    progress: ProgressFn | None = None,
    predictors: Mapping[str, Callable] | None = None,
    vad_detector: Callable[[np.ndarray, int], Sequence[tuple[float, float]]] | None = None,
    cleanup_cuda: Callable[[], None] | None = None,
) -> InferenceRunResult:
    """Run VAD, affect, disfluency, and emotion sequentially.

    Example:

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
    predictors = dict(predictors or {})
    audio = load_audio(audio_path, sample_rate=sample_rate, recording_id=recording_id)
    _progress(progress, f"audio sha256: {audio.audio_sha256}")

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
            ),
        ),
        (
            "affect",
            lambda: run_affect_inference(
                audio,
                out_dir=out_dir,
                backbone=affect_backbone,
                reuse_cache=reuse_cache,
                device=device,
                predictor=predictors.get("affect"),
                progress=progress,
                cleanup_cuda=cleanup_cuda,
            ),
        ),
        (
            "disfluency",
            lambda: run_disfluency_inference(
                audio,
                out_dir=out_dir,
                backbone=disfluency_backbone,
                reuse_cache=reuse_cache,
                device=device,
                predictor=predictors.get("disfluency"),
                progress=progress,
                cleanup_cuda=cleanup_cuda,
            ),
        ),
        (
            "emotion",
            lambda: run_emotion_inference(
                audio,
                out_dir=out_dir,
                reuse_cache=reuse_cache,
                device=device,
                predictor=predictors.get("emotion"),
                progress=progress,
                cleanup_cuda=cleanup_cuda,
            ),
        ),
    ]
    for task, run_step in steps:
        _progress(progress, f"run-all task: {task}")
        result = run_step()
        artifacts[task] = result.artifact
        reused[task] = result.reused
    return InferenceRunResult(artifacts=artifacts, reused=reused)


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
) -> PredictionArtifact | None:
    if not reuse_cache:
        return None
    return find_cached_artifact(
        out_dir,
        recording_id=audio.recording_id,
        audio_sha256=audio.audio_sha256,
        task=task,
        inference_config_hash_value=inference_config_hash(config),
    )


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
) -> PredictionArtifact:
    config_hash = inference_config_hash(config)
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


def _runtime(*, device: str | None, batch_size: int) -> dict:
    return {
        "device": device or _default_device(),
        "batch_size": int(batch_size),
    }


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
    progress: ProgressFn | None,
) -> dict[str, np.ndarray]:
    import torch

    wrapper = _load_affect_wrapper(backbone)
    run_device = device or _default_device()
    model = wrapper.from_pretrained(model_id).to(run_device).eval()
    arousal, valence, dominance = [], [], []
    with torch.inference_mode():
        for batch_np in _batches(windows, batch_size, progress, "affect"):
            batch = torch.from_numpy(np.ascontiguousarray(batch_np)).to(run_device)
            a, v, d = model(batch)
            arousal.append(a.detach().cpu().reshape(-1).numpy())
            valence.append(v.detach().cpu().reshape(-1).numpy())
            dominance.append(d.detach().cpu().reshape(-1).numpy())
    return {
        "arousal": np.concatenate(arousal),
        "valence": np.concatenate(valence),
        "dominance": np.concatenate(dominance),
    }


def _predict_disfluency(
    windows: np.ndarray,
    *,
    backbone: str,
    model_id: str,
    device: str | None,
    batch_size: int,
    progress: ProgressFn | None,
) -> dict[str, np.ndarray]:
    import torch

    wrapper = _load_disfluency_wrapper(backbone)
    run_device = device or _default_device()
    model = wrapper.from_pretrained(model_id).to(run_device).eval()
    fluency, dysfluency = [], []
    with torch.inference_mode():
        for batch_np in _batches(windows, batch_size, progress, "disfluency"):
            batch = torch.from_numpy(np.ascontiguousarray(batch_np)).to(run_device)
            f, d = model(batch, return_feature=False)
            fluency.append(f.detach().cpu().numpy())
            dysfluency.append(d.detach().cpu().numpy())
    return {
        "fluency_logits": np.concatenate(fluency, axis=0),
        "disfluency_type_logits": np.concatenate(dysfluency, axis=0),
    }


def _predict_emotion2vec(
    windows: np.ndarray,
    *,
    model_id: str,
    sample_rate: int,
    batch_size: int,
    progress: ProgressFn | None,
) -> tuple[np.ndarray, Sequence[str]]:
    from funasr import AutoModel

    model = AutoModel(model=model_id)
    all_scores = []
    labels = None
    for batch_np in _batches(windows, batch_size, progress, "emotion"):
        batch = [np.ascontiguousarray(batch_np[i]) for i in range(len(batch_np))]
        results = model.generate(
            input=batch,
            fs=sample_rate,
            granularity="utterance",
            extract_embedding=False,
        )
        if labels is None:
            labels = list(results[0]["labels"])
        all_scores.extend(result["scores"] for result in results)
    if labels is None:
        raise ValueError("emotion2vec produced no results")
    return np.asarray(all_scores, dtype=np.float32), labels


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
    run_device = device or _default_device()
    model = model.to(run_device)
    timestamps = get_speech_timestamps(
        torch.from_numpy(np.asarray(samples, dtype=np.float32)),
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


def _progress(progress: ProgressFn | None, message: str) -> None:
    if progress is not None:
        progress(message)
    else:
        LOGGER.info(message)
