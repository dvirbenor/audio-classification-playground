"""Main worker loop for batch inference on a single Kubernetes pod.

The worker:

1. Installs a SIGTERM handler **immediately** on startup.
2. Loads the archive manifest and pre-existing error sets.
3. Loads all four models into GPU memory (affect, disfluency, emotion, VAD).
4. Starts a background ``Prefetcher`` for S3 download + CPU decode.
5. Shuffles entities and iterates, claiming each via an atomic lock file.
6. Runs ``run_all_inference`` with injected persistent predictors and a
   custom ``artifact_path_fn`` for the flat output layout.
7. Handles ``ShutdownRequested``, ``torch.cuda.OutOfMemoryError``, and
   generic exceptions with appropriate error logging and lock release.
"""
from __future__ import annotations

import logging
import os
import random
import signal
import threading
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
from .prefetch import Prefetcher
from .progress import is_archive_complete_for_config

LOGGER = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 512
DEFAULT_MAX_INFERENCE_ATTEMPTS = 3
PREFETCH_LOOKAHEAD = 8
PREFETCH_WORKERS = 4


def build_expected_config_hashes(
    *,
    affect_backbone: str,
    disfluency_backbone: str,
    batch_size: int,
    sample_rate: int = SAMPLE_RATE,
    vad_threshold: float = DEFAULT_VAD_SPEECH_THRESHOLD,
    vad_min_speech_sec: float = DEFAULT_VAD_MIN_SPEECH_SEC,
    vad_min_silence_sec: float = DEFAULT_VAD_MIN_SILENCE_SEC,
) -> dict[str, str]:
    """Compute expected ``inference_config_hash`` for each task.

    Uses the same ``compute_inference_config`` helper as the runners so
    config hashes cannot drift.
    """
    configs: dict[str, dict] = {}
    configs["affect"] = compute_inference_config(
        task="affect",
        model_id=DEFAULT_AFFECT_MODELS[affect_backbone],
        backbone=affect_backbone,
        sample_rate=sample_rate,
        window_sec=AFFECT_WINDOW_SEC,
        hop_sec=DEFAULT_HOP_SEC,
        batch_size=batch_size,
        transform_policy="vox_profile_affect_sigmoid_heads_v1",
    )
    configs["disfluency"] = compute_inference_config(
        task="disfluency",
        model_id=DEFAULT_DISFLUENCY_MODELS[disfluency_backbone],
        backbone=disfluency_backbone,
        sample_rate=sample_rate,
        window_sec=DISFLUENCY_WINDOW_SEC,
        hop_sec=DEFAULT_HOP_SEC,
        batch_size=batch_size,
        transform_policy="vox_profile_disfluency_raw_logits_v1",
    )
    configs["emotion"] = compute_inference_config(
        task="emotion",
        model_id=DEFAULT_EMOTION_MODEL,
        backbone=None,
        sample_rate=sample_rate,
        window_sec=EMOTION_WINDOW_SEC,
        hop_sec=DEFAULT_HOP_SEC,
        batch_size=batch_size,
        transform_policy="emotion2vec_fold_row_normalize_v1",
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
    return {task: inference_config_hash(cfg) for task, cfg in configs.items()}


def run_worker(
    *,
    parquet_path: str | Path,
    output_base: str | Path,
    affect_backbone: str,
    disfluency_backbone: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    sample_rate: int = SAMPLE_RATE,
    max_inference_attempts: int = DEFAULT_MAX_INFERENCE_ATTEMPTS,
    vad_threshold: float = DEFAULT_VAD_SPEECH_THRESHOLD,
    vad_min_speech_sec: float = DEFAULT_VAD_MIN_SPEECH_SEC,
    vad_min_silence_sec: float = DEFAULT_VAD_MIN_SILENCE_SEC,
    prefetch_workers: int = PREFETCH_WORKERS,
    prefetch_lookahead: int = PREFETCH_LOOKAHEAD,
    seed: int | None = None,
) -> None:
    """Entry point for a single worker pod.

    This function does not return until all entities are processed or a
    SIGTERM signal is received.
    """
    output_base = Path(output_base)
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

    expected_hashes = build_expected_config_hashes(
        affect_backbone=affect_backbone,
        disfluency_backbone=disfluency_backbone,
        batch_size=batch_size,
        sample_rate=sample_rate,
        vad_threshold=vad_threshold,
        vad_min_speech_sec=vad_min_speech_sec,
        vad_min_silence_sec=vad_min_silence_sec,
    )

    if shutdown_event.is_set():
        LOGGER.info("Shutdown requested before model loading; exiting early")
        return

    LOGGER.info("Loading models into GPU memory…")
    from ..inference.models import ModelSuite

    models = ModelSuite(
        affect_backbone=affect_backbone,
        disfluency_backbone=disfluency_backbone,
        batch_size=batch_size,
        device=device,
        vad_threshold=vad_threshold,
        vad_min_speech_sec=vad_min_speech_sec,
        vad_min_silence_sec=vad_min_silence_sec,
    )
    LOGGER.info("All models loaded")

    prefetcher = Prefetcher(
        sample_rate=sample_rate,
        max_workers=prefetch_workers,
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

    lookahead_idx = 0

    def _submit_lookahead():
        nonlocal lookahead_idx
        while lookahead_idx < len(entities) and (
            lookahead_idx - processed - skipped - failed < prefetch_lookahead
        ):
            pf_entity = entities[lookahead_idx]
            pf_key = (pf_entity.session_id, pf_entity.archive_id)
            if pf_key not in permanent_errors and inference_attempts.get(pf_key, 0) < max_inference_attempts:
                prefetcher.submit(pf_entity)
            lookahead_idx += 1

    try:
        _submit_lookahead()

        for entity in entities:
            if shutdown_event.is_set():
                LOGGER.info("Shutdown requested — exiting main loop")
                break

            sid, aid = entity.session_id, entity.archive_id
            entity_key = (sid, aid)

            if entity_key in permanent_errors:
                skipped += 1
                _submit_lookahead()
                continue

            if inference_attempts.get(entity_key, 0) >= max_inference_attempts:
                skipped += 1
                _submit_lookahead()
                continue

            if is_archive_complete_for_config(output_base, sid, aid, expected_hashes):
                skipped += 1
                _submit_lookahead()
                continue

            if not try_claim(output_base, entity):
                skipped += 1
                _submit_lookahead()
                continue

            actual_attempts = count_inference_attempts_for(output_base, sid, aid)
            if actual_attempts >= max_inference_attempts:
                release_claim(output_base, entity)
                inference_attempts[entity_key] = actual_attempts
                skipped += 1
                _submit_lookahead()
                continue

            claim_released = False
            try:
                _submit_lookahead()

                pf_result = prefetcher.get(entity)
                if isinstance(pf_result, AudioResolutionError):
                    append_audio_error(output_base, pf_result)
                    if pf_result.is_permanent:
                        permanent_errors.add(entity_key)
                    release_claim(output_base, entity)
                    claim_released = True
                    failed += 1
                    _submit_lookahead()
                    continue

                audio_data, s3_key = pf_result
                s3_uri = f"s3://{BUCKET}/{s3_key}"

                run_all_inference(
                    audio_data,
                    out_dir=str(output_base),
                    affect_backbone=affect_backbone,
                    disfluency_backbone=disfluency_backbone,
                    reuse_cache=True,
                    batch_size=batch_size,
                    device=device,
                    sample_rate=sample_rate,
                    vad_threshold=vad_threshold,
                    vad_min_speech_sec=vad_min_speech_sec,
                    vad_min_silence_sec=vad_min_silence_sec,
                    predictors={
                        "affect": models.affect,
                        "disfluency": models.disfluency,
                        "emotion": models.emotion,
                    },
                    vad_detector=models.vad,
                    cleanup_cuda=lambda: None,
                    artifact_path_fn=lambda task: _artifact_path_fn(task, entity),
                    audio_path_override=s3_uri,
                    audio_source_key=s3_key,
                    shutdown_check=_shutdown_check,
                )

                release_claim(output_base, entity)
                claim_released = True
                processed += 1
                if processed % 50 == 0:
                    LOGGER.info(
                        "Progress: processed=%d skipped=%d failed=%d",
                        processed, skipped, failed,
                    )

            except ShutdownRequested:
                LOGGER.info("ShutdownRequested during %s/%s — releasing lock", sid, aid)
                if not claim_released:
                    release_claim(output_base, entity)
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
                    release_claim(output_base, entity)
                prefetcher.discard(entity)
                _submit_lookahead()

    finally:
        prefetcher.shutdown()
        LOGGER.info(
            "Worker finished: processed=%d skipped=%d failed=%d",
            processed, skipped, failed,
        )


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
