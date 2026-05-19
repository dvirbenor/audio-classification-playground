"""Main worker loop for batch inference on a single Kubernetes pod.

The worker:

1. Installs a SIGTERM handler **immediately** on startup.
2. Loads the archive manifest and pre-existing error sets.
3. Loads persistent GPU models into memory.
4. Starts a background ``Prefetcher`` for claimed download, decode, and VAD.
5. Shuffles entities and claims a bounded lookahead queue via atomic locks.
6. Runs ``run_all_inference`` with injected persistent predictors and a
   custom ``artifact_path_fn`` for the flat output layout.
7. Handles ``ShutdownRequested``, ``torch.cuda.OutOfMemoryError``, and
   generic exceptions with appropriate error logging and lock release.
"""
from __future__ import annotations

import json
import logging
import os
import random
import signal
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from ..inference.artifacts import SAMPLE_RATE, inference_config_hash
from ..inference.runners import (
    DEFAULT_AFFECT_MODELS,
    DEFAULT_DISFLUENCY_MODELS,
    DEFAULT_EMOTION_MODEL,
    DEFAULT_VAD_MODEL,
    DEFAULT_VAD_SPEECH_THRESHOLD,
    DEFAULT_VAD_MIN_SPEECH_SEC,
    DEFAULT_VAD_MIN_SILENCE_SEC,
    DEFAULT_VAD_FRAME_SPEECH_RATIO_THRESHOLD,
    AFFECT_WINDOW_SEC,
    DISFLUENCY_WINDOW_SEC,
    EMOTION_WINDOW_SEC,
    DEFAULT_HOP_SEC,
    ShutdownRequested,
    cleanup_torch_memory,
    compute_inference_config,
    resolve_task_batch_sizes,
    run_all_inference,
)
from .audio_resolver import AudioResolutionError, BUCKET
from .errors import (
    append_audio_error,
    append_inference_error,
    count_inference_attempts_for,
    is_deterministic_error,
    load_inference_attempt_counts,
    load_permanent_error_set,
)
from .locking import release_claim, try_claim
from .manifest import ArchiveEntity, load_manifest
from .prefetch import PrefetchResult, Prefetcher
from .progress import is_archive_complete_for_config, is_task_complete_for_config

LOGGER = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 512
DEFAULT_EMOTION_BATCH_SIZE = 64
DEFAULT_MAX_INFERENCE_ATTEMPTS = 3
PREFETCH_LOOKAHEAD = 4
PREFETCH_WORKERS = 4
VAD_PREFETCH_WORKERS = 1
TIMINGS_DIR = "_meta/timings"


def _append_timing_record(jsonl_path: Path, record: dict) -> None:
    """Append a single JSON line to the worker's timing file.

    Catches ``OSError`` so a disk hiccup never kills the worker.
    """
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        LOGGER.warning("Failed to write timing record to %s", jsonl_path, exc_info=True)


def _emotion_runtime_config_extra(
    *,
    emotion_autocast_dtype: str | None,
    emotion_compile: bool,
    emotion_compile_mode: str,
    allow_tf32: bool,
) -> dict[str, object]:
    extra: dict[str, object] = {}
    if emotion_autocast_dtype is not None:
        extra["torch_autocast_dtype"] = emotion_autocast_dtype
    if emotion_compile:
        extra["torch_compile"] = True
        extra["torch_compile_mode"] = emotion_compile_mode
    if allow_tf32:
        extra["torch_allow_tf32"] = True
    return extra


def _wavlm_runtime_config_extra(
    *,
    backbone: str,
    wavlm_autocast_dtype: str | None,
    wavlm_compile: bool,
    wavlm_compile_mode: str,
    wavlm_compile_dynamic: bool,
    allow_tf32: bool,
) -> dict[str, object]:
    if backbone != "wavlm":
        return {}
    extra: dict[str, object] = {}
    if wavlm_autocast_dtype is not None:
        extra["torch_autocast_dtype"] = wavlm_autocast_dtype
    if wavlm_compile:
        extra["torch_compile"] = True
        extra["torch_compile_target"] = "wavlm_backbone"
        extra["torch_compile_mode"] = wavlm_compile_mode
        extra["torch_compile_dynamic"] = bool(wavlm_compile_dynamic)
    if allow_tf32:
        extra["torch_allow_tf32"] = True
    return extra


def build_expected_configs(
    *,
    affect_backbone: str,
    disfluency_backbone: str,
    batch_size: int,
    affect_batch_size: int | None = None,
    disfluency_batch_size: int | None = None,
    emotion_batch_size: int | None = None,
    sample_rate: int = SAMPLE_RATE,
    vad_threshold: float = DEFAULT_VAD_SPEECH_THRESHOLD,
    vad_min_speech_sec: float = DEFAULT_VAD_MIN_SPEECH_SEC,
    vad_min_silence_sec: float = DEFAULT_VAD_MIN_SILENCE_SEC,
    wavlm_autocast_dtype: str | None = None,
    wavlm_compile: bool = False,
    wavlm_compile_mode: str = "reduce-overhead",
    wavlm_compile_dynamic: bool = False,
    emotion_autocast_dtype: str | None = None,
    emotion_compile: bool = False,
    emotion_compile_mode: str = "reduce-overhead",
    allow_tf32: bool = False,
) -> dict[str, dict]:
    """Compute expected inference configs for each task.

    Uses the same ``compute_inference_config`` helper as the runners so
    worker stale-artifact detection cannot drift.
    """
    batches = resolve_task_batch_sizes(
        batch_size=batch_size,
        affect_batch_size=affect_batch_size,
        disfluency_batch_size=disfluency_batch_size,
        emotion_batch_size=(
            DEFAULT_EMOTION_BATCH_SIZE
            if emotion_batch_size is None
            else emotion_batch_size
        ),
    )
    configs: dict[str, dict] = {}
    configs["affect"] = compute_inference_config(
        task="affect",
        model_id=DEFAULT_AFFECT_MODELS[affect_backbone],
        backbone=affect_backbone,
        sample_rate=sample_rate,
        window_sec=AFFECT_WINDOW_SEC,
        hop_sec=DEFAULT_HOP_SEC,
        batch_size=batches["affect"],
        transform_policy="vox_profile_affect_sigmoid_heads_v1",
        extra=_wavlm_runtime_config_extra(
            backbone=affect_backbone,
            wavlm_autocast_dtype=wavlm_autocast_dtype,
            wavlm_compile=wavlm_compile,
            wavlm_compile_mode=wavlm_compile_mode,
            wavlm_compile_dynamic=wavlm_compile_dynamic,
            allow_tf32=allow_tf32,
        ),
    )
    configs["disfluency"] = compute_inference_config(
        task="disfluency",
        model_id=DEFAULT_DISFLUENCY_MODELS[disfluency_backbone],
        backbone=disfluency_backbone,
        sample_rate=sample_rate,
        window_sec=DISFLUENCY_WINDOW_SEC,
        hop_sec=DEFAULT_HOP_SEC,
        batch_size=batches["disfluency"],
        transform_policy="vox_profile_disfluency_raw_logits_v1",
        extra=_wavlm_runtime_config_extra(
            backbone=disfluency_backbone,
            wavlm_autocast_dtype=wavlm_autocast_dtype,
            wavlm_compile=wavlm_compile,
            wavlm_compile_mode=wavlm_compile_mode,
            wavlm_compile_dynamic=wavlm_compile_dynamic,
            allow_tf32=allow_tf32,
        ),
    )
    configs["emotion"] = compute_inference_config(
        task="emotion",
        model_id=DEFAULT_EMOTION_MODEL,
        backbone=None,
        sample_rate=sample_rate,
        window_sec=EMOTION_WINDOW_SEC,
        hop_sec=DEFAULT_HOP_SEC,
        batch_size=batches["emotion"],
        transform_policy="emotion2vec_fold_row_normalize_v1",
        extra=_emotion_runtime_config_extra(
            emotion_autocast_dtype=emotion_autocast_dtype,
            emotion_compile=emotion_compile,
            emotion_compile_mode=emotion_compile_mode,
            allow_tf32=allow_tf32,
        ),
    )
    configs["vad"] = compute_inference_config(
        task="vad",
        model_id=DEFAULT_VAD_MODEL,
        backbone=None,
        sample_rate=sample_rate,
        window_sec=0.0,
        hop_sec=0.0,
        batch_size=0,
        transform_policy="silero_vad_intervals_sec_v1",
        extra={
            "threshold": float(vad_threshold),
            "speech_threshold": float(vad_threshold),
            "min_speech_sec": float(vad_min_speech_sec),
            "min_silence_sec": float(vad_min_silence_sec),
            "frame_speech_ratio_threshold": float(DEFAULT_VAD_FRAME_SPEECH_RATIO_THRESHOLD),
        },
    )
    return configs


def build_expected_config_hashes(**kwargs) -> dict[str, str]:
    """Compute expected ``inference_config_hash`` for each task."""
    return {
        task: inference_config_hash(cfg)
        for task, cfg in build_expected_configs(**kwargs).items()
    }


def run_worker(
    *,
    parquet_path: str | Path,
    output_base: str | Path,
    affect_backbone: str,
    disfluency_backbone: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    affect_batch_size: int | None = None,
    disfluency_batch_size: int | None = None,
    emotion_batch_size: int | None = None,
    device: str | None = None,
    sample_rate: int = SAMPLE_RATE,
    max_inference_attempts: int = DEFAULT_MAX_INFERENCE_ATTEMPTS,
    vad_threshold: float = DEFAULT_VAD_SPEECH_THRESHOLD,
    vad_min_speech_sec: float = DEFAULT_VAD_MIN_SPEECH_SEC,
    vad_min_silence_sec: float = DEFAULT_VAD_MIN_SILENCE_SEC,
    wavlm_autocast_dtype: str | None = None,
    wavlm_compile: bool = False,
    wavlm_compile_mode: str = "reduce-overhead",
    wavlm_compile_dynamic: bool = False,
    emotion_autocast_dtype: str | None = None,
    emotion_compile: bool = False,
    emotion_compile_mode: str = "reduce-overhead",
    allow_tf32: bool = False,
    prefetch_workers: int = PREFETCH_WORKERS,
    prefetch_lookahead: int = PREFETCH_LOOKAHEAD,
    vad_prefetch_workers: int = VAD_PREFETCH_WORKERS,
    seed: int | None = None,
) -> None:
    """Entry point for a single worker pod.

    This function does not return until all entities are processed or a
    SIGTERM signal is received.
    """
    if prefetch_lookahead < 1:
        raise ValueError("prefetch_lookahead must be >= 1")
    if vad_prefetch_workers < 0:
        raise ValueError("vad_prefetch_workers must be >= 0")
    batches = resolve_task_batch_sizes(
        batch_size=batch_size,
        affect_batch_size=affect_batch_size,
        disfluency_batch_size=disfluency_batch_size,
        emotion_batch_size=(
            DEFAULT_EMOTION_BATCH_SIZE
            if emotion_batch_size is None
            else emotion_batch_size
        ),
    )

    output_base = Path(output_base)
    worker_id = f"{os.environ.get('HOSTNAME', 'unknown')}_{uuid.uuid4().hex[:8]}"
    timings_path = output_base / TIMINGS_DIR / f"{worker_id}.jsonl"
    shutdown_event = threading.Event()

    def _sigterm_handler(signum, frame):
        LOGGER.warning("SIGTERM received — finishing current archive then exiting")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    if shutdown_event.is_set():
        LOGGER.info("Shutdown requested before model loading; exiting early")
        return

    LOGGER.info("Loading manifest from %s", parquet_path)
    entities = load_manifest(parquet_path)
    permanent_errors = load_permanent_error_set(output_base)
    inference_attempts = load_inference_attempt_counts(output_base)

    expected_configs = build_expected_configs(
        affect_backbone=affect_backbone,
        disfluency_backbone=disfluency_backbone,
        batch_size=batch_size,
        affect_batch_size=batches["affect"],
        disfluency_batch_size=batches["disfluency"],
        emotion_batch_size=batches["emotion"],
        sample_rate=sample_rate,
        vad_threshold=vad_threshold,
        vad_min_speech_sec=vad_min_speech_sec,
        vad_min_silence_sec=vad_min_silence_sec,
        wavlm_autocast_dtype=wavlm_autocast_dtype,
        wavlm_compile=wavlm_compile,
        wavlm_compile_mode=wavlm_compile_mode,
        wavlm_compile_dynamic=wavlm_compile_dynamic,
        emotion_autocast_dtype=emotion_autocast_dtype,
        emotion_compile=emotion_compile,
        emotion_compile_mode=emotion_compile_mode,
        allow_tf32=allow_tf32,
    )
    expected_hashes = {
        task: inference_config_hash(cfg)
        for task, cfg in expected_configs.items()
    }

    if shutdown_event.is_set():
        LOGGER.info("Shutdown requested before model loading; exiting early")
        return

    LOGGER.info("Loading persistent inference models…")
    from ..inference.models import ModelSuite

    models = ModelSuite(
        affect_backbone=affect_backbone,
        disfluency_backbone=disfluency_backbone,
        batch_size=batch_size,
        affect_batch_size=batches["affect"],
        disfluency_batch_size=batches["disfluency"],
        emotion_batch_size=batches["emotion"],
        device=device,
        vad_threshold=vad_threshold,
        vad_min_speech_sec=vad_min_speech_sec,
        vad_min_silence_sec=vad_min_silence_sec,
        load_vad=vad_prefetch_workers == 0,
        wavlm_autocast_dtype=wavlm_autocast_dtype,
        wavlm_compile=wavlm_compile,
        wavlm_compile_mode=wavlm_compile_mode,
        wavlm_compile_dynamic=wavlm_compile_dynamic,
        emotion_autocast_dtype=emotion_autocast_dtype,
        emotion_compile=emotion_compile,
        emotion_compile_mode=emotion_compile_mode,
        allow_tf32=allow_tf32,
    )

    def _new_vad_detector():
        from ..inference.models import VadDetector

        return VadDetector(
            threshold=vad_threshold,
            min_speech_sec=vad_min_speech_sec,
            min_silence_sec=vad_min_silence_sec,
        )

    LOGGER.info("Persistent inference models loaded")

    prefetcher = Prefetcher(
        sample_rate=sample_rate,
        max_workers=prefetch_workers,
        vad_workers=vad_prefetch_workers,
        vad_detector_factory=_new_vad_detector,
        bucket=BUCKET,
    )

    rng = random.Random(seed)
    rng.shuffle(entities)

    def _shutdown_check() -> bool:
        return shutdown_event.is_set()

    def _artifact_path_fn(task: str, entity: ArchiveEntity) -> Path:
        return output_base / entity.session_id / entity.archive_id / task

    processed = 0
    skipped = 0
    failed = 0

    next_entity_idx = 0
    queued: deque[ArchiveEntity] = deque()

    def _release(entity: ArchiveEntity) -> None:
        release_claim(output_base, entity)
        prefetcher.discard(entity)

    def _fill_claimed_queue() -> None:
        nonlocal next_entity_idx, skipped
        while (
            not shutdown_event.is_set()
            and len(queued) < prefetch_lookahead
            and next_entity_idx < len(entities)
        ):
            entity = entities[next_entity_idx]
            next_entity_idx += 1
            sid, aid = entity.session_id, entity.archive_id
            entity_key = (sid, aid)

            if entity_key in permanent_errors:
                skipped += 1
                continue

            if inference_attempts.get(entity_key, 0) >= max_inference_attempts:
                skipped += 1
                continue

            if is_archive_complete_for_config(
                output_base,
                sid,
                aid,
                expected_hashes,
                expected_configs=expected_configs,
                ignore_batch_size=True,
            ):
                skipped += 1
                continue

            if not try_claim(output_base, entity):
                skipped += 1
                continue

            actual_attempts = count_inference_attempts_for(output_base, sid, aid)
            if actual_attempts >= max_inference_attempts:
                release_claim(output_base, entity)
                inference_attempts[entity_key] = actual_attempts
                skipped += 1
                continue

            if is_archive_complete_for_config(
                output_base,
                sid,
                aid,
                expected_hashes,
                expected_configs=expected_configs,
                ignore_batch_size=True,
            ):
                release_claim(output_base, entity)
                skipped += 1
                continue

            precompute_vad = (
                vad_prefetch_workers > 0
                and not is_task_complete_for_config(
                    output_base,
                    sid,
                    aid,
                    "vad",
                    expected_hashes["vad"],
                    expected_config=expected_configs["vad"],
                    ignore_batch_size=True,
                )
            )
            prefetcher.submit(entity, precompute_vad=precompute_vad)
            queued.append(entity)

    try:
        _fill_claimed_queue()

        while queued:
            if shutdown_event.is_set():
                LOGGER.info("Shutdown requested — releasing queued claims")
                break

            entity = queued.popleft()
            sid, aid = entity.session_id, entity.archive_id
            entity_key = (sid, aid)

            claim_released = False
            try:
                archive_started = time.perf_counter()
                wait_started = time.perf_counter()
                pf_result = prefetcher.get(entity)
                prefetch_wait_sec = time.perf_counter() - wait_started
                if isinstance(pf_result, AudioResolutionError):
                    append_audio_error(output_base, pf_result)
                    if pf_result.is_permanent:
                        permanent_errors.add(entity_key)
                    _release(entity)
                    claim_released = True
                    failed += 1
                    _fill_claimed_queue()
                    continue

                s3_uri = f"s3://{BUCKET}/{pf_result.s3_key}"
                vad_detector = _detector_for_result(
                    pf_result,
                    models,
                    allow_sync_vad=vad_prefetch_workers == 0,
                )

                inference_started = time.perf_counter()
                result = run_all_inference(
                    pf_result.audio,
                    out_dir=str(output_base),
                    affect_backbone=affect_backbone,
                    disfluency_backbone=disfluency_backbone,
                    reuse_cache=True,
                    batch_size=batch_size,
                    affect_batch_size=batches["affect"],
                    disfluency_batch_size=batches["disfluency"],
                    emotion_batch_size=batches["emotion"],
                    device=device,
                    sample_rate=sample_rate,
                    vad_threshold=vad_threshold,
                    vad_min_speech_sec=vad_min_speech_sec,
                    vad_min_silence_sec=vad_min_silence_sec,
                    wavlm_autocast_dtype=wavlm_autocast_dtype,
                    wavlm_compile=wavlm_compile,
                    wavlm_compile_mode=wavlm_compile_mode,
                    wavlm_compile_dynamic=wavlm_compile_dynamic,
                    emotion_autocast_dtype=emotion_autocast_dtype,
                    emotion_compile=emotion_compile,
                    emotion_compile_mode=emotion_compile_mode,
                    allow_tf32=allow_tf32,
                    predictors={
                        "affect": models.affect,
                        "disfluency": models.disfluency,
                        "emotion": models.emotion,
                    },
                    vad_detector=vad_detector,
                    cleanup_cuda=lambda: None,
                    artifact_path_fn=lambda task: _artifact_path_fn(task, entity),
                    audio_path_override=s3_uri,
                    audio_source_key=pf_result.s3_key,
                    shutdown_check=_shutdown_check,
                )
                inference_sec = time.perf_counter() - inference_started
                total_sec = time.perf_counter() - archive_started

                LOGGER.info(
                    "Archive timings %s/%s: prefetch_wait=%.3fs "
                    "download_decode=%.3fs vad=%.3fs inference=%.3fs total=%.3fs "
                    "precomputed_vad=%s",
                    sid,
                    aid,
                    prefetch_wait_sec,
                    pf_result.timings.download_decode_sec,
                    pf_result.timings.vad_sec,
                    inference_sec,
                    total_sec,
                    pf_result.vad_intervals is not None,
                )

                _append_timing_record(timings_path, {
                    "worker_id": worker_id,
                    "session_id": sid,
                    "archive_id": aid,
                    "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "audio_duration_sec": pf_result.audio.duration_sec,
                    "prefetch_wait_sec": prefetch_wait_sec,
                    "download_decode_sec": pf_result.timings.download_decode_sec,
                    "vad_precompute_sec": pf_result.timings.vad_sec,
                    "precomputed_vad": pf_result.vad_intervals is not None,
                    "vad_reused": result.reused.get("vad", False),
                    "affect_reused": result.reused.get("affect", False),
                    "disfluency_reused": result.reused.get("disfluency", False),
                    "emotion_reused": result.reused.get("emotion", False),
                    "vad_sec": result.task_elapsed_sec.get("vad", 0.0),
                    "affect_sec": result.task_elapsed_sec.get("affect", 0.0),
                    "disfluency_sec": result.task_elapsed_sec.get("disfluency", 0.0),
                    "emotion_sec": result.task_elapsed_sec.get("emotion", 0.0),
                    "inference_sec": inference_sec,
                    "total_sec": total_sec,
                })

                _release(entity)
                claim_released = True
                processed += 1
                if processed % 50 == 0:
                    LOGGER.info(
                        "Progress: processed=%d skipped=%d failed=%d",
                        processed, skipped, failed,
                    )

            except ShutdownRequested:
                LOGGER.info("ShutdownRequested during %s/%s — releasing lock", sid, aid)
                shutdown_event.set()
                if not claim_released:
                    _release(entity)
                    claim_released = True
                break

            except Exception as exc:
                _handle_inference_error(
                    exc, output_base, entity,
                    inference_attempts, max_inference_attempts,
                )
                failed += 1

            finally:
                if not claim_released:
                    _release(entity)
                _fill_claimed_queue()

    finally:
        while queued:
            _release(queued.popleft())
        prefetcher.shutdown()
        LOGGER.info(
            "Worker finished: processed=%d skipped=%d failed=%d",
            processed, skipped, failed,
        )


def _detector_for_result(result: PrefetchResult, models, *, allow_sync_vad: bool):
    if result.vad_intervals is None:
        return models.vad if allow_sync_vad else None

    intervals = tuple(result.vad_intervals)

    def _precomputed_vad(samples, sample_rate):
        return list(intervals)

    return _precomputed_vad


def _handle_inference_error(
    exc: Exception,
    output_base: Path,
    entity: ArchiveEntity,
    inference_attempts: dict,
    max_inference_attempts: int,
) -> None:
    import torch

    entity_key = (entity.session_id, entity.archive_id)

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        LOGGER.error("GPU OOM for %s/%s — cleaning GPU memory", entity.session_id, entity.archive_id)
        cleanup_torch_memory()
        append_inference_error(output_base, entity.session_id, entity.archive_id, exc)
        inference_attempts[entity_key] = inference_attempts.get(entity_key, 0) + 1
        return

    deterministic = is_deterministic_error(exc)
    append_inference_error(
        output_base, entity.session_id, entity.archive_id, exc,
        is_deterministic=deterministic,
    )
    if deterministic:
        inference_attempts[entity_key] = 9999
        LOGGER.error(
            "Deterministic error for %s/%s: %s — marked as permanently failed",
            entity.session_id, entity.archive_id, type(exc).__name__,
        )
    else:
        inference_attempts[entity_key] = inference_attempts.get(entity_key, 0) + 1
        LOGGER.error(
            "Inference error for %s/%s (attempt %d/%d): %s: %s",
            entity.session_id, entity.archive_id,
            inference_attempts[entity_key], max_inference_attempts,
            type(exc).__name__, str(exc)[:300],
        )
