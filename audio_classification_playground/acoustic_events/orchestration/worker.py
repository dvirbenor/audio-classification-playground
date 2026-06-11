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
from ..inference.emotion_runtime import (
    DEFAULT_EMOTION_COMPILE_MODE,
    OPTIMIZED_EMOTION_BATCH_SIZE,
    resolve_emotion_runtime_settings,
)
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
from ..inference.vad_gating import DEFAULT_BRIDGE_SEC, DEFAULT_GATED_TASKS, VadGating
from ..inference.wavlm_runtime import (
    WAVLM_COMPILED_STATIC_BATCH_SIZE,
    configure_inductor_cache_namespace,
    inductor_cache_status,
    resolve_wavlm_runtime_settings,
)
from .audio_resolver import AudioResolutionError, BUCKET
from .audio_cache import SharedAudioCache
from .errors import (
    append_audio_error,
    append_inference_error,
    count_inference_attempts_for,
    is_deterministic_error,
    load_inference_attempt_counts,
    load_permanent_error_set,
)
from .locking import flat_lock_files, nested_lock_files, release_claim, try_claim
from .manifest import ArchiveEntity, load_manifest, sort_manifest_by_session
from .prefetch import PrefetchResult, Prefetcher
from .progress import (
    are_tasks_complete_by_artifact,
    incomplete_tasks_by_artifact,
    is_task_artifact_complete_for_archive,
    is_task_complete_for_config,
)
from .task_groups import (
    COMPLETION_POLICY_CONFIG,
    COMPLETION_POLICY_EXISTS,
    TASK_GROUP_ALL,
    TASK_GROUP_EMOTION,
    TASK_GROUP_EMOTION_VAD,
    TASK_GROUP_VAD,
    resolve_task_group,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 512
DEFAULT_EMOTION_BATCH_SIZE = OPTIMIZED_EMOTION_BATCH_SIZE
DEFAULT_MAX_INFERENCE_ATTEMPTS = 3
PREFETCH_LOOKAHEAD = 4
PREFETCH_WORKERS = 4
VAD_PREFETCH_WORKERS = 1
TIMINGS_DIR = "_meta/timings"
PREFETCH_SCHEDULER_ENV = "ACP_PREFETCH_SCHEDULER"
PREFETCH_SCHEDULER_READY_FIRST = "ready_first"
PREFETCH_SCHEDULER_FIFO = "fifo"
PREFETCH_WAIT_ANY_TIMEOUT_SEC = 0.5


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
    wavlm_stream_layer_sum: bool,
    allow_tf32: bool,
) -> dict[str, object]:
    if backbone != "wavlm":
        return {}
    return {
        "torch_autocast_dtype": wavlm_autocast_dtype,
        "torch_compile": bool(wavlm_compile),
        "torch_compile_target": "wavlm_backbone",
        "torch_compile_mode": str(wavlm_compile_mode),
        "torch_compile_dynamic": bool(wavlm_compile_dynamic),
        "wavlm_stream_layer_sum": bool(wavlm_stream_layer_sum),
        "torch_allow_tf32": bool(allow_tf32),
    }


def _resolve_worker_batch_sizes(
    *,
    batch_size: int,
    affect_batch_size: int | None,
    disfluency_batch_size: int | None,
    emotion_batch_size: int | None,
    affect_backbone: str,
    disfluency_backbone: str,
    wavlm_task_batch_size: int | None,
) -> dict[str, int]:
    """Resolve per-task batches, applying WavLM preset defaults last."""
    if wavlm_task_batch_size is not None:
        if (
            affect_backbone == "wavlm"
            and affect_batch_size is not None
            and int(affect_batch_size) != int(wavlm_task_batch_size)
        ):
            raise ValueError(
                "compiled_static WavLM preset requires affect_batch_size="
                f"{wavlm_task_batch_size}"
            )
        if (
            disfluency_backbone == "wavlm"
            and disfluency_batch_size is not None
            and int(disfluency_batch_size) != int(wavlm_task_batch_size)
        ):
            raise ValueError(
                "compiled_static WavLM preset requires disfluency_batch_size="
                f"{wavlm_task_batch_size}"
            )
    return resolve_task_batch_sizes(
        batch_size=batch_size,
        affect_batch_size=(
            wavlm_task_batch_size
            if affect_backbone == "wavlm" and wavlm_task_batch_size is not None
            else affect_batch_size
        ),
        disfluency_batch_size=(
            wavlm_task_batch_size
            if disfluency_backbone == "wavlm" and wavlm_task_batch_size is not None
            else disfluency_batch_size
        ),
        emotion_batch_size=(
            DEFAULT_EMOTION_BATCH_SIZE
            if emotion_batch_size is None
            else emotion_batch_size
        ),
    )


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
    wavlm_stream_layer_sum: bool = False,
    wavlm_runtime_preset: str | None = None,
    emotion_autocast_dtype: str | None = None,
    emotion_compile: bool = False,
    emotion_compile_mode: str = DEFAULT_EMOTION_COMPILE_MODE,
    emotion_runtime_mode: str | None = "auto",
    allow_tf32: bool = False,
    device: str | None = None,
    vad_gating: VadGating | None = None,
) -> dict[str, dict]:
    """Compute expected inference configs for each task.

    Uses the same ``compute_inference_config`` helper as the runners so
    worker stale-artifact detection cannot drift.
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
    wavlm_settings = resolve_wavlm_runtime_settings(
        preset=wavlm_runtime_preset,
        device=device,
        autocast_dtype=wavlm_autocast_dtype,
        compile_model=wavlm_compile,
        compile_mode=wavlm_compile_mode,
        compile_dynamic=wavlm_compile_dynamic,
        stream_layer_sum=wavlm_stream_layer_sum,
        allow_tf32=allow_tf32,
    )
    batches = _resolve_worker_batch_sizes(
        batch_size=batch_size,
        affect_batch_size=affect_batch_size,
        disfluency_batch_size=disfluency_batch_size,
        emotion_batch_size=emotion_batch_size,
        affect_backbone=affect_backbone,
        disfluency_backbone=disfluency_backbone,
        wavlm_task_batch_size=wavlm_settings.task_batch_size,
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
            wavlm_autocast_dtype=wavlm_settings.autocast_dtype,
            wavlm_compile=wavlm_settings.compile_model,
            wavlm_compile_mode=wavlm_settings.compile_mode,
            wavlm_compile_dynamic=wavlm_settings.compile_dynamic,
            wavlm_stream_layer_sum=wavlm_settings.stream_layer_sum,
            allow_tf32=wavlm_settings.allow_tf32,
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
            wavlm_autocast_dtype=wavlm_settings.autocast_dtype,
            wavlm_compile=wavlm_settings.compile_model,
            wavlm_compile_mode=wavlm_settings.compile_mode,
            wavlm_compile_dynamic=wavlm_settings.compile_dynamic,
            wavlm_stream_layer_sum=wavlm_settings.stream_layer_sum,
            allow_tf32=wavlm_settings.allow_tf32,
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
            emotion_autocast_dtype=emotion_settings.autocast_dtype,
            emotion_compile=emotion_settings.compile_model,
            emotion_compile_mode=emotion_settings.compile_mode,
            allow_tf32=emotion_settings.allow_tf32,
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
    # VAD gating applies to the GPU tasks only (not the VAD step that produces
    # the intervals); merge the descriptor so gated artifacts hash distinctly.
    if vad_gating is not None and vad_gating.active:
        for task in ("affect", "disfluency", "emotion"):
            if vad_gating.gates(task):
                configs[task] = {**configs[task], **vad_gating.config_extra()}
    return configs


def build_expected_config_hashes(**kwargs) -> dict[str, str]:
    """Compute expected ``inference_config_hash`` for each task."""
    return {
        task: inference_config_hash(cfg)
        for task, cfg in build_expected_configs(**kwargs).items()
    }


def _guard_against_mixed_lock_modes(output_base: Path, *, task_group: str) -> None:
    """Prevent all-in-one and task-fleet workers from sharing an output tree."""
    if task_group == TASK_GROUP_ALL:
        nested = nested_lock_files(output_base)
        if nested:
            raise RuntimeError(
                "Refusing to start all-in-one worker: task-scoped locks are active "
                f"under {output_base / '_meta' / 'locks'}"
            )
        return
    flat = flat_lock_files(output_base)
    if flat:
        raise RuntimeError(
            "Refusing to start task-fleet worker: archive-level all-mode locks are "
            f"active under {output_base / '_meta' / 'locks'}"
        )
    if task_group in (TASK_GROUP_VAD, TASK_GROUP_EMOTION, TASK_GROUP_EMOTION_VAD):
        legacy_emotion_vad_locks = (
            output_base / "_meta" / "locks" / TASK_GROUP_EMOTION_VAD
        )
        if legacy_emotion_vad_locks.is_dir() and any(
            legacy_emotion_vad_locks.glob("*.lock")
        ):
            raise RuntimeError(
                "Refusing to start split VAD/emotion worker while legacy "
                f"{TASK_GROUP_EMOTION_VAD!r} locks are active under "
                f"{legacy_emotion_vad_locks}"
            )


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
    wavlm_stream_layer_sum: bool = False,
    wavlm_runtime_preset: str | None = None,
    emotion_autocast_dtype: str | None = None,
    emotion_compile: bool = False,
    emotion_compile_mode: str = DEFAULT_EMOTION_COMPILE_MODE,
    emotion_runtime_mode: str | None = "auto",
    allow_tf32: bool = False,
    prefetch_workers: int | None = None,
    prefetch_lookahead: int | None = None,
    vad_prefetch_workers: int | None = None,
    audio_cache_dir: str | Path | None = None,
    max_cache_bytes: int | None = None,
    audio_cache_lock_stale_minutes: float = 60.0,
    task_group: str = TASK_GROUP_ALL,
    completion_policy: str = COMPLETION_POLICY_EXISTS,
    force_recompute: bool = False,
    vad_gating_enabled: bool = False,
    vad_gating_bridge_sec: float = DEFAULT_BRIDGE_SEC,
    vad_gating_tasks: tuple[str, ...] = DEFAULT_GATED_TASKS,
    seed: int | None = None,
) -> None:
    """Entry point for a single worker pod.

    This function does not return until all entities are processed or a
    SIGTERM signal is received.
    """
    group = resolve_task_group(task_group)
    if completion_policy not in (COMPLETION_POLICY_EXISTS, COMPLETION_POLICY_CONFIG):
        raise ValueError("completion_policy must be one of: exists, config")
    if prefetch_workers is None:
        prefetch_workers = group.prefetch_workers
    if prefetch_lookahead is None:
        prefetch_lookahead = group.prefetch_lookahead
    if vad_prefetch_workers is None:
        vad_prefetch_workers = group.vad_prefetch_workers
    if prefetch_lookahead < 1:
        raise ValueError("prefetch_lookahead must be >= 1")
    if vad_prefetch_workers < 0:
        raise ValueError("vad_prefetch_workers must be >= 0")
    if audio_cache_dir is not None and max_cache_bytes is None:
        raise ValueError("max_cache_bytes is required when audio_cache_dir is set")
    if max_cache_bytes is not None and max_cache_bytes <= 0:
        raise ValueError("max_cache_bytes must be > 0")
    prefetch_scheduler = os.environ.get(
        PREFETCH_SCHEDULER_ENV,
        PREFETCH_SCHEDULER_READY_FIRST,
    ).strip().lower()
    if prefetch_scheduler not in (
        PREFETCH_SCHEDULER_READY_FIRST,
        PREFETCH_SCHEDULER_FIFO,
    ):
        raise ValueError(
            f"{PREFETCH_SCHEDULER_ENV} must be one of: "
            f"{PREFETCH_SCHEDULER_READY_FIRST}, {PREFETCH_SCHEDULER_FIFO}"
        )
    emotion_settings = resolve_emotion_runtime_settings(
        mode=emotion_runtime_mode,
        default_mode="auto",
        device=device,
        autocast_dtype=emotion_autocast_dtype,
        compile_model=emotion_compile,
        compile_mode=emotion_compile_mode,
        allow_tf32=allow_tf32,
    )
    wavlm_settings = resolve_wavlm_runtime_settings(
        preset=wavlm_runtime_preset,
        device=device,
        autocast_dtype=wavlm_autocast_dtype,
        compile_model=wavlm_compile,
        compile_mode=wavlm_compile_mode,
        compile_dynamic=wavlm_compile_dynamic,
        stream_layer_sum=wavlm_stream_layer_sum,
        allow_tf32=allow_tf32,
    )
    if wavlm_runtime_preset is not None and wavlm_settings.preset != wavlm_runtime_preset:
        LOGGER.warning(
            "Requested WavLM runtime preset %s is unavailable on this worker; "
            "using %s",
            wavlm_runtime_preset,
            wavlm_settings.preset,
        )
    if wavlm_settings.compile_model:
        cache = configure_inductor_cache_namespace(preset=wavlm_settings.preset)
        if not cache.get("configured"):
            LOGGER.warning(
                "TORCHINDUCTOR_CACHE_DIR is not configured; WavLM compile warmup "
                "will be paid after each worker restart"
            )
        elif not cache.get("writable"):
            LOGGER.warning("TORCHINDUCTOR_CACHE_DIR is not writable: %s", cache)
        else:
            LOGGER.info("Using WavLM Inductor cache: %s", cache)
    else:
        cache = inductor_cache_status()
    batches = _resolve_worker_batch_sizes(
        batch_size=batch_size,
        affect_batch_size=affect_batch_size,
        disfluency_batch_size=disfluency_batch_size,
        emotion_batch_size=emotion_batch_size,
        affect_backbone=affect_backbone,
        disfluency_backbone=disfluency_backbone,
        wavlm_task_batch_size=wavlm_settings.task_batch_size,
    )
    LOGGER.info(
        "WavLM runtime: preset=%s requested=%s compile=%s mode=%s dynamic=%s "
        "static_batch=%s batches=%s cache=%s prefetch_scheduler=%s",
        wavlm_settings.preset,
        wavlm_settings.requested_preset,
        wavlm_settings.compile_model,
        wavlm_settings.compile_mode,
        wavlm_settings.compile_dynamic,
        wavlm_settings.static_batch,
        {
            "affect": batches["affect"],
            "disfluency": batches["disfluency"],
        },
        cache,
        prefetch_scheduler,
    )

    vad_gating = VadGating(
        enabled=bool(vad_gating_enabled),
        bridge_sec=float(vad_gating_bridge_sec),
        tasks=tuple(vad_gating_tasks),
    )
    if vad_gating.active:
        LOGGER.info(
            "VAD-gated inference enabled: bridge_sec=%.2f policy=%s tasks=%s",
            vad_gating.bridge_sec,
            vad_gating.policy,
            ",".join(vad_gating.tasks),
        )

    output_base = Path(output_base)
    _guard_against_mixed_lock_modes(output_base, task_group=group.name)
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
    inference_attempts = load_inference_attempt_counts(
        output_base,
        task_group=None if group.name == TASK_GROUP_ALL else group.name,
    )

    expected_wavlm_preset = (
        None if wavlm_settings.preset == "custom" else wavlm_settings.preset
    )
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
        wavlm_stream_layer_sum=wavlm_stream_layer_sum,
        wavlm_runtime_preset=expected_wavlm_preset,
        emotion_autocast_dtype=emotion_autocast_dtype,
        emotion_compile=emotion_compile,
        emotion_compile_mode=emotion_compile_mode,
        emotion_runtime_mode=emotion_runtime_mode,
        allow_tf32=allow_tf32,
        device=emotion_settings.device,
        vad_gating=vad_gating,
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
        load_vad=vad_prefetch_workers == 0 and "vad" in group.tasks,
        tasks_to_load=group.models,
        wavlm_autocast_dtype=wavlm_settings.autocast_dtype,
        wavlm_compile=wavlm_settings.compile_model,
        wavlm_compile_mode=wavlm_settings.compile_mode,
        wavlm_compile_dynamic=wavlm_settings.compile_dynamic,
        wavlm_stream_layer_sum=wavlm_settings.stream_layer_sum,
        wavlm_static_batch=wavlm_settings.static_batch,
        wavlm_warmup=wavlm_settings.warmup,
        wavlm_runtime_preset=wavlm_settings.preset,
        emotion_autocast_dtype=emotion_autocast_dtype,
        emotion_compile=emotion_compile,
        emotion_compile_mode=emotion_compile_mode,
        emotion_runtime_mode=emotion_runtime_mode,
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

    audio_cache = None
    if audio_cache_dir is not None:
        audio_cache = SharedAudioCache(
            audio_cache_dir,
            sample_rate=sample_rate,
            max_cache_bytes=int(max_cache_bytes),
            stale_lock_minutes=audio_cache_lock_stale_minutes,
            bucket=BUCKET,
        )

    prefetcher = Prefetcher(
        sample_rate=sample_rate,
        max_workers=prefetch_workers,
        vad_workers=vad_prefetch_workers,
        vad_detector_factory=_new_vad_detector,
        bucket=BUCKET,
        audio_cache=audio_cache,
    )

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(entities)
        LOGGER.info("Entity ordering: shuffled (seed=%d)", seed)
    else:
        entities = sort_manifest_by_session(entities)
        LOGGER.info("Entity ordering: session-grouped sort (date, session_id)")

    def _shutdown_check() -> bool:
        return shutdown_event.is_set()

    def _artifact_path_fn(task: str, entity: ArchiveEntity) -> Path:
        return output_base / entity.session_id / entity.archive_id / task

    processed = 0
    skipped = 0
    failed = 0

    next_entity_idx = 0
    queued: deque[ArchiveEntity] = deque()

    def _try_claim(
        entity: ArchiveEntity,
        *,
        task_names: tuple[str, ...],
    ) -> bool:
        if group.lock_namespaces is None:
            return try_claim(output_base, entity)
        acquired: list[str] = []
        namespaces_to_claim = tuple(
            namespace for namespace in group.lock_namespaces
            if namespace in task_names
        )
        if not namespaces_to_claim:
            return False
        for namespace in namespaces_to_claim:
            if try_claim(
                output_base,
                entity,
                namespace=namespace,
                task_group=group.name,
            ):
                acquired.append(namespace)
                continue
            for acquired_namespace in reversed(acquired):
                release_claim(output_base, entity, namespace=acquired_namespace)
            return False
        return True

    def _release(entity: ArchiveEntity) -> None:
        if group.lock_namespaces is None:
            release_claim(output_base, entity)
        else:
            for namespace in group.lock_namespaces:
                release_claim(output_base, entity, namespace=namespace)
        prefetcher.discard(entity)

    def _task_complete(sid: str, aid: str, task: str) -> bool:
        if force_recompute:
            return False
        if completion_policy == COMPLETION_POLICY_EXISTS:
            return is_task_artifact_complete_for_archive(output_base, sid, aid, task)
        return is_task_complete_for_config(
            output_base,
            sid,
            aid,
            task,
            expected_hashes[task],
            expected_config=expected_configs[task],
            ignore_batch_size=True,
        )

    def _group_complete(sid: str, aid: str) -> bool:
        if force_recompute:
            return False
        if completion_policy == COMPLETION_POLICY_EXISTS:
            return are_tasks_complete_by_artifact(output_base, sid, aid, group.tasks)
        return all(_task_complete(sid, aid, task) for task in group.tasks)

    def _missing_tasks(sid: str, aid: str) -> tuple[str, ...]:
        if force_recompute:
            return group.tasks
        if completion_policy == COMPLETION_POLICY_EXISTS:
            return incomplete_tasks_by_artifact(output_base, sid, aid, group.tasks)
        return tuple(task for task in group.tasks if not _task_complete(sid, aid, task))

    def _attempt_key(sid: str, aid: str):
        return (sid, aid) if group.name == TASK_GROUP_ALL else (sid, aid, group.name)

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

            attempt_key = _attempt_key(sid, aid)

            if inference_attempts.get(attempt_key, 0) >= max_inference_attempts:
                skipped += 1
                continue

            if _group_complete(sid, aid):
                skipped += 1
                continue

            tasks_to_claim = _missing_tasks(sid, aid)
            if not tasks_to_claim:
                skipped += 1
                continue

            if not _try_claim(entity, task_names=tasks_to_claim):
                skipped += 1
                continue

            actual_attempts = count_inference_attempts_for(
                output_base,
                sid,
                aid,
                task_group=None if group.name == TASK_GROUP_ALL else group.name,
            )
            if actual_attempts >= max_inference_attempts:
                _release(entity)
                inference_attempts[attempt_key] = actual_attempts
                skipped += 1
                continue

            if _group_complete(sid, aid):
                _release(entity)
                skipped += 1
                continue

            precompute_vad = (
                group.precompute_vad
                and
                vad_prefetch_workers > 0
                and "vad" in _missing_tasks(sid, aid)
            )
            prefetcher.submit(entity, precompute_vad=precompute_vad)
            queued.append(entity)

    def _pop_first_ready_entity() -> ArchiveEntity | None:
        for entity in tuple(queued):
            if prefetcher.is_ready(entity):
                queued.remove(entity)
                return entity
        return None

    def _select_next_entity() -> tuple[ArchiveEntity, float] | None:
        if not queued:
            return None
        if prefetch_scheduler == PREFETCH_SCHEDULER_FIFO:
            return queued.popleft(), 0.0

        ready_entity = _pop_first_ready_entity()
        if ready_entity is not None:
            return ready_entity, 0.0

        wait_started = time.perf_counter()
        while queued and not shutdown_event.is_set():
            prefetcher.wait_any(
                tuple(queued),
                timeout_sec=PREFETCH_WAIT_ANY_TIMEOUT_SEC,
            )
            ready_entity = _pop_first_ready_entity()
            if ready_entity is not None:
                return ready_entity, time.perf_counter() - wait_started
        return None

    try:
        _fill_claimed_queue()

        while queued:
            if shutdown_event.is_set():
                LOGGER.info("Shutdown requested — releasing queued claims")
                break

            selected = _select_next_entity()
            if selected is None:
                LOGGER.info("Shutdown requested while waiting for prefetch readiness")
                break
            entity, prefetch_scheduler_wait_sec = selected
            sid, aid = entity.session_id, entity.archive_id
            entity_key = (sid, aid)

            claim_released = False
            try:
                archive_started = time.perf_counter()
                wait_started = time.perf_counter()
                pf_result = prefetcher.get(entity)
                get_finished = time.perf_counter()
                prefetch_get_wait_sec = get_finished - wait_started
                prefetch_wait_sec = (
                    prefetch_scheduler_wait_sec + prefetch_get_wait_sec
                )
                ready_time = getattr(pf_result, "ready_time", 0.0)
                prefetch_ready_age_sec = (
                    max(0.0, get_finished - ready_time)
                    if ready_time
                    else 0.0
                )
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
                    allow_sync_vad=vad_prefetch_workers == 0 and "vad" in group.tasks,
                )
                tasks_to_run = _missing_tasks(sid, aid)
                if not tasks_to_run:
                    _release(entity)
                    claim_released = True
                    skipped += 1
                    _fill_claimed_queue()
                    continue

                inference_started = time.perf_counter()
                predictors = {}
                if getattr(models, "affect", None) is not None:
                    predictors["affect"] = models.affect
                if getattr(models, "disfluency", None) is not None:
                    predictors["disfluency"] = models.disfluency
                if getattr(models, "emotion", None) is not None:
                    predictors["emotion"] = models.emotion
                result = run_all_inference(
                    pf_result.audio,
                    out_dir=str(output_base),
                    affect_backbone=affect_backbone,
                    disfluency_backbone=disfluency_backbone,
                    reuse_cache=not force_recompute,
                    batch_size=batch_size,
                    affect_batch_size=batches["affect"],
                    disfluency_batch_size=batches["disfluency"],
                    emotion_batch_size=batches["emotion"],
                    device=device,
                    sample_rate=sample_rate,
                    vad_threshold=vad_threshold,
                    vad_min_speech_sec=vad_min_speech_sec,
                    vad_min_silence_sec=vad_min_silence_sec,
                    wavlm_autocast_dtype=wavlm_settings.autocast_dtype,
                    wavlm_compile=wavlm_settings.compile_model,
                    wavlm_compile_mode=wavlm_settings.compile_mode,
                    wavlm_compile_dynamic=wavlm_settings.compile_dynamic,
                    wavlm_stream_layer_sum=wavlm_settings.stream_layer_sum,
                    wavlm_static_batch=wavlm_settings.static_batch,
                    wavlm_runtime_preset=wavlm_settings.preset,
                    emotion_autocast_dtype=emotion_autocast_dtype,
                    emotion_compile=emotion_compile,
                    emotion_compile_mode=emotion_compile_mode,
                    emotion_runtime_mode=emotion_runtime_mode,
                    allow_tf32=allow_tf32,
                    predictors=predictors,
                    vad_detector=vad_detector,
                    cleanup_cuda=lambda: None,
                    artifact_path_fn=lambda task: _artifact_path_fn(task, entity),
                    audio_path_override=s3_uri,
                    audio_source_key=pf_result.s3_key,
                    shutdown_check=_shutdown_check,
                    tasks_filter=tasks_to_run,
                    vad_gating=vad_gating,
                    vad_intervals=pf_result.vad_intervals,
                )
                inference_sec = time.perf_counter() - inference_started
                total_sec = time.perf_counter() - archive_started

                LOGGER.info(
                    "Archive timings %s/%s: prefetch_wait=%.3fs "
                    "scheduler_wait=%.3fs get_wait=%.3fs download_decode=%.3fs "
                    "resolve=%.3fs head=%.3fs download=%.3fs decode=%.3fs "
                    "cache_wait=%.3fs vad_queue=%.3fs vad=%.3fs ready_age=%.3fs "
                    "inference=%.3fs total=%.3fs precomputed_vad=%s "
                    "cache_enabled=%s resolution_hit=%s object_hit=%s "
                    "cache_write=%s fallback=%s fallback_reason=%s",
                    sid,
                    aid,
                    prefetch_wait_sec,
                    prefetch_scheduler_wait_sec,
                    prefetch_get_wait_sec,
                    pf_result.timings.download_decode_sec,
                    pf_result.timings.resolve_sec,
                    pf_result.timings.head_sec,
                    pf_result.timings.download_sec,
                    pf_result.timings.decode_sec,
                    pf_result.timings.cache_wait_sec,
                    pf_result.timings.vad_queue_wait_sec,
                    pf_result.timings.vad_sec,
                    prefetch_ready_age_sec,
                    inference_sec,
                    total_sec,
                    pf_result.vad_intervals is not None,
                    pf_result.audio_cache_enabled,
                    pf_result.resolution_cache_hit,
                    pf_result.object_cache_hit,
                    pf_result.cache_write,
                    pf_result.cache_fallback,
                    pf_result.cache_fallback_reason or "-",
                )

                _append_timing_record(timings_path, {
                    "worker_id": worker_id,
                    "task_group": group.name,
                    "tasks_run": list(tasks_to_run),
                    "session_id": sid,
                    "archive_id": aid,
                    "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "s3_key": pf_result.s3_key,
                    "audio_source_extension": pf_result.audio_source_extension,
                    "audio_object_size_bytes": pf_result.audio_object_size_bytes,
                    "audio_storage_class": pf_result.audio_storage_class,
                    "audio_duration_sec": pf_result.audio.duration_sec,
                    "audio_cache_enabled": pf_result.audio_cache_enabled,
                    "audio_cache_payload_type": pf_result.audio_cache_payload_type,
                    "resolution_cache_hit": pf_result.resolution_cache_hit,
                    "object_cache_hit": pf_result.object_cache_hit,
                    "cache_write": pf_result.cache_write,
                    "cache_fallback": pf_result.cache_fallback,
                    "cache_fallback_reason": pf_result.cache_fallback_reason,
                    "decoded_bytes": pf_result.decoded_bytes,
                    "wavlm_runtime_preset": wavlm_settings.preset,
                    "wavlm_static_batch": wavlm_settings.static_batch,
                    "wavlm_batch_size": (
                        WAVLM_COMPILED_STATIC_BATCH_SIZE
                        if wavlm_settings.static_batch
                        else None
                    ),
                    "torchinductor_cache": cache,
                    "prefetch_scheduler": prefetch_scheduler,
                    "prefetch_scheduler_wait_sec": prefetch_scheduler_wait_sec,
                    "prefetch_get_wait_sec": prefetch_get_wait_sec,
                    "prefetch_wait_sec": prefetch_wait_sec,
                    "decode_queue_wait_sec": pf_result.timings.decode_queue_wait_sec,
                    "download_decode_sec": pf_result.timings.download_decode_sec,
                    "resolve_sec": pf_result.timings.resolve_sec,
                    "head_sec": pf_result.timings.head_sec,
                    "download_sec": pf_result.timings.download_sec,
                    "decode_sec": pf_result.timings.decode_sec,
                    "cache_wait_sec": pf_result.timings.cache_wait_sec,
                    "vad_queue_wait_sec": pf_result.timings.vad_queue_wait_sec,
                    "vad_precompute_sec": pf_result.timings.vad_sec,
                    "prefetch_submit_to_ready_sec": (
                        pf_result.timings.prefetch_submit_to_ready_sec
                    ),
                    "prefetch_ready_age_sec": prefetch_ready_age_sec,
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
                    task_group=group.name,
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
    *,
    task_group: str = TASK_GROUP_ALL,
) -> None:
    import torch

    entity_key = (
        (entity.session_id, entity.archive_id)
        if task_group == TASK_GROUP_ALL
        else (entity.session_id, entity.archive_id, task_group)
    )

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        LOGGER.error("GPU OOM for %s/%s — cleaning GPU memory", entity.session_id, entity.archive_id)
        cleanup_torch_memory()
        append_inference_error(
            output_base,
            entity.session_id,
            entity.archive_id,
            exc,
            task_group=None if task_group == TASK_GROUP_ALL else task_group,
        )
        inference_attempts[entity_key] = inference_attempts.get(entity_key, 0) + 1
        return

    deterministic = is_deterministic_error(exc)
    append_inference_error(
        output_base, entity.session_id, entity.archive_id, exc,
        is_deterministic=deterministic,
        task_group=None if task_group == TASK_GROUP_ALL else task_group,
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
