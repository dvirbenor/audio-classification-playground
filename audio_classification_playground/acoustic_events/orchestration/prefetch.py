"""Background audio download, decode, and optional VAD precomputation.

The ``Prefetcher`` downloads audio from S3, decodes it to a mono 16 kHz
``AudioData`` object, and can hand the decoded audio to a separate CPU VAD
executor.  VAD workers lazily create their own detector instances so a
stateful Silero model is never shared across concurrent calls.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from ..inference.audio import AudioData, load_audio
from .audio_resolver import AudioDownloadResult, AudioResolutionError, resolve_and_download
from .audio_cache import CachedAudio, SharedAudioCache
from .manifest import ArchiveEntity

VadDetectorFn = Callable[[np.ndarray, int], Sequence[tuple[float, float]]]


@dataclass(frozen=True)
class PrefetchTimings:
    """Wall-clock timings measured inside the prefetch pipeline."""

    decode_queue_wait_sec: float = 0.0
    download_decode_sec: float = 0.0
    resolve_sec: float = 0.0
    head_sec: float = 0.0
    download_sec: float = 0.0
    decode_sec: float = 0.0
    cache_wait_sec: float = 0.0
    vad_queue_wait_sec: float = 0.0
    vad_sec: float = 0.0
    prefetch_submit_to_ready_sec: float = 0.0
    total_sec: float = 0.0


@dataclass(frozen=True)
class PrefetchResult:
    """Ready-to-infer inputs for one claimed archive."""

    audio: AudioData
    s3_key: str
    audio_source_extension: str = ""
    audio_object_size_bytes: int | None = None
    audio_storage_class: str | None = None
    audio_cache_enabled: bool = False
    audio_cache_payload_type: str = ""
    resolution_cache_hit: bool = False
    object_cache_hit: bool = False
    cache_write: bool = False
    cache_fallback: bool = False
    cache_fallback_reason: str = ""
    decoded_bytes: int = 0
    vad_intervals: list[tuple[float, float]] | None = None
    timings: PrefetchTimings = PrefetchTimings()
    ready_time: float = 0.0
    _submitted_at: float = field(default=0.0, repr=False, compare=False)


@dataclass(frozen=True)
class _DecodedResult:
    audio: AudioData
    s3_key: str
    audio_source_extension: str
    audio_object_size_bytes: int | None
    audio_storage_class: str | None
    submitted_at: float
    decode_submitted_at: float
    decode_started_at: float
    download_decode_sec: float
    resolve_sec: float
    head_sec: float
    download_sec: float
    decode_sec: float
    cache_wait_sec: float = 0.0
    audio_cache_enabled: bool = False
    audio_cache_payload_type: str = ""
    resolution_cache_hit: bool = False
    object_cache_hit: bool = False
    cache_write: bool = False
    cache_fallback: bool = False
    cache_fallback_reason: str = ""
    decoded_bytes: int = 0


class Prefetcher:
    """Download + decode audio, optionally followed by CPU VAD.

    Usage::

        pf = Prefetcher(sample_rate=16000, max_workers=4, vad_workers=1)
        pf.submit(entity, precompute_vad=True)
        ...
        result = pf.get(entity)  # blocks until ready
        # result is PrefetchResult or AudioResolutionError
        pf.discard(entity)
        ...
        pf.shutdown()
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        max_workers: int = 4,
        s3_client=None,
        bucket: str = "riverside-pro-main",
        tmp_dir: str | None = None,
        vad_workers: int = 1,
        vad_detector_factory: Callable[[], VadDetectorFn] | None = None,
        prewarm_vad: bool = True,
        audio_cache: SharedAudioCache | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if vad_workers < 0:
            raise ValueError("vad_workers must be >= 0")
        self._sample_rate = sample_rate
        self._s3_client = s3_client
        self._bucket = bucket
        self._tmp_dir = tmp_dir
        self._audio_cache = audio_cache
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="prefetch")
        self._vad_pool = (
            ThreadPoolExecutor(max_workers=vad_workers, thread_name_prefix="vad-prefetch")
            if vad_workers > 0
            else None
        )
        self._vad_detector_factory = vad_detector_factory
        self._vad_local = threading.local()
        self._lock = threading.Lock()
        self._futures: dict[tuple[str, str], Future] = {}
        self._decode_futures: dict[tuple[str, str], Future] = {}
        self._vad_futures: dict[tuple[str, str], Future] = {}
        self._tmp_paths: dict[tuple[str, str], Path] = {}
        self._shared_paths: dict[tuple[str, str], Path] = {}
        self._discarded: set[tuple[str, str]] = set()

        if prewarm_vad and vad_workers > 0:
            self._make_vad_detector()

    def submit(self, entity: ArchiveEntity, *, precompute_vad: bool = False) -> None:
        """Queue *entity* for background download/decode and optional VAD."""
        key = (entity.session_id, entity.archive_id)
        submitted_at = time.perf_counter()
        with self._lock:
            if key in self._futures:
                return
            self._discarded.discard(key)
            final_future: Future = Future()
            self._futures[key] = final_future

        decode_submitted_at = time.perf_counter()
        decode_future = self._pool.submit(
            self._fetch_and_decode,
            entity,
            submitted_at,
            decode_submitted_at,
        )
        with self._lock:
            self._decode_futures[key] = decode_future

        def _after_decode(done: Future) -> None:
            try:
                decoded = done.result()
                if isinstance(decoded, AudioResolutionError):
                    self._set_final_result(final_future, decoded)
                    return
                if not precompute_vad:
                    self._set_final_result(
                        final_future,
                        self._prefetch_result(decoded, None, 0.0, 0.0),
                    )
                    return
                if self._vad_pool is None:
                    raise RuntimeError("precompute_vad=True requires vad_workers > 0")
                vad_submitted_at = time.perf_counter()
                vad_future = self._vad_pool.submit(
                    self._add_vad,
                    decoded,
                    vad_submitted_at,
                )
                with self._lock:
                    self._vad_futures[key] = vad_future

                def _after_vad(vad_done: Future) -> None:
                    try:
                        self._set_final_result(final_future, vad_done.result())
                    except BaseException as exc:
                        final_future.set_exception(exc)

                vad_future.add_done_callback(_after_vad)
            except BaseException as exc:
                final_future.set_exception(exc)

        decode_future.add_done_callback(_after_decode)

    def is_ready(self, entity: ArchiveEntity) -> bool:
        """Return whether a submitted prefetch has completed."""
        key = (entity.session_id, entity.archive_id)
        with self._lock:
            future = self._futures.get(key)
        return bool(future is not None and future.done())

    def wait_any(
        self,
        entities: Sequence[ArchiveEntity],
        timeout_sec: float | None = 0.5,
    ) -> bool:
        """Wait until any submitted entity is ready, returning ``False`` on timeout."""
        if not entities:
            return False
        keys = {(entity.session_id, entity.archive_id) for entity in entities}
        with self._lock:
            futures = [
                future
                for key, future in self._futures.items()
                if key in keys
            ]
        if not futures:
            return False
        done, _ = wait(futures, timeout=timeout_sec, return_when=FIRST_COMPLETED)
        return bool(done)

    def get(
        self, entity: ArchiveEntity
    ) -> PrefetchResult | AudioResolutionError:
        """Block until the result for *entity* is ready.

        Returns ``PrefetchResult`` on success, ``AudioResolutionError`` on
        audio resolution/decode failure, or raises for VAD failures.
        """
        key = (entity.session_id, entity.archive_id)
        with self._lock:
            future = self._futures.get(key)
        if future is None:
            submitted_at = time.perf_counter()
            decoded = self._fetch_and_decode(entity, submitted_at, submitted_at)
            if isinstance(decoded, AudioResolutionError):
                return self._mark_ready(decoded)
            return self._mark_ready(self._prefetch_result(decoded, None, 0.0, 0.0))
        return future.result()

    def discard(self, entity: ArchiveEntity) -> None:
        """Clean up temp files and memory for *entity*."""
        key = (entity.session_id, entity.archive_id)
        with self._lock:
            self._discarded.add(key)
            self._futures.pop(key, None)
            decode_future = self._decode_futures.pop(key, None)
            vad_future = self._vad_futures.pop(key, None)
            tmp = self._tmp_paths.pop(key, None)
            self._shared_paths.pop(key, None)
        if decode_future is not None:
            decode_future.cancel()
        if vad_future is not None:
            vad_future.cancel()
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def shutdown(self) -> None:
        """Shut down the thread pool and clean up all temp files."""
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self._vad_pool is not None:
            self._vad_pool.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            tmp_paths = list(self._tmp_paths.values())
            self._tmp_paths.clear()
            self._shared_paths.clear()
            self._futures.clear()
            self._decode_futures.clear()
            self._vad_futures.clear()
            self._discarded.clear()
        for tmp in tmp_paths:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _fetch_and_decode(
        self,
        entity: ArchiveEntity,
        submitted_at: float,
        decode_submitted_at: float,
    ) -> _DecodedResult | AudioResolutionError:
        decode_started_at = time.perf_counter()
        if self._audio_cache is not None:
            return self._fetch_from_cache(
                entity,
                submitted_at,
                decode_submitted_at,
                decode_started_at,
            )

        result = resolve_and_download(
            session_id=entity.session_id,
            archive_id=entity.archive_id,
            file_parent_dir=entity.file_parent_dir,
            s3_client=self._s3_client,
            bucket=self._bucket,
            tmp_dir=self._tmp_dir,
        )
        if isinstance(result, AudioResolutionError):
            return result

        if isinstance(result, AudioDownloadResult):
            download = result
        else:
            local_path, s3_key = result
            download = AudioDownloadResult(
                local_path=local_path,
                s3_key=s3_key,
                source_extension=Path(s3_key).suffix.lower(),
            )
        local_path = download.local_path
        s3_key = download.s3_key
        key = (entity.session_id, entity.archive_id)
        with self._lock:
            if key in self._discarded:
                local_path.unlink(missing_ok=True)
                return AudioResolutionError(
                    session_id=entity.session_id,
                    archive_id=entity.archive_id,
                    file_parent_dir=entity.file_parent_dir,
                    error_type="download_failed",
                    detail="Prefetch result was discarded before decode",
                    s3_key=s3_key,
                )
            self._tmp_paths[key] = local_path

        try:
            decode_only_started = time.perf_counter()
            audio = load_audio(
                local_path,
                sample_rate=self._sample_rate,
                recording_id=entity.archive_id,
            )
            decode_sec = time.perf_counter() - decode_only_started
        except Exception as exc:
            with self._lock:
                self._tmp_paths.pop(key, None)
            local_path.unlink(missing_ok=True)
            return AudioResolutionError(
                session_id=entity.session_id,
                archive_id=entity.archive_id,
                file_parent_dir=entity.file_parent_dir,
                error_type="download_failed",
                detail=f"Decode failed: {type(exc).__name__}: {exc}",
                s3_key=s3_key,
            )

        decode_finished_at = time.perf_counter()
        return _DecodedResult(
            audio=audio,
            s3_key=s3_key,
            audio_source_extension=download.source_extension,
            audio_object_size_bytes=download.object_size_bytes,
            audio_storage_class=download.storage_class,
            submitted_at=submitted_at,
            decode_submitted_at=decode_submitted_at,
            decode_started_at=decode_started_at,
            download_decode_sec=decode_finished_at - decode_started_at,
            resolve_sec=download.resolve_sec,
            head_sec=download.head_sec,
            download_sec=download.download_sec,
            decode_sec=decode_sec,
            decoded_bytes=int(audio.samples.nbytes),
        )

    def _fetch_from_cache(
        self,
        entity: ArchiveEntity,
        submitted_at: float,
        decode_submitted_at: float,
        decode_started_at: float,
    ) -> _DecodedResult | AudioResolutionError:
        assert self._audio_cache is not None
        result = self._audio_cache.get_decoded_audio(entity, tmp_dir=self._tmp_dir)
        if isinstance(result, AudioResolutionError):
            return result
        cached: CachedAudio = result
        key = (entity.session_id, entity.archive_id)
        if cached.cache_path is not None:
            with self._lock:
                self._shared_paths[key] = cached.cache_path
        with self._lock:
            if key in self._discarded:
                self._shared_paths.pop(key, None)
                return AudioResolutionError(
                    session_id=entity.session_id,
                    archive_id=entity.archive_id,
                    file_parent_dir=entity.file_parent_dir,
                    error_type="download_failed",
                    detail="Prefetch result was discarded before cache decode",
                    s3_key=cached.s3_key,
                    resolve_sec=cached.stats.resolve_sec,
                    head_sec=cached.stats.head_sec,
                    download_sec=cached.stats.download_sec,
                )

        total = time.perf_counter() - decode_started_at
        stats = cached.stats
        return _DecodedResult(
            audio=cached.audio,
            s3_key=cached.s3_key,
            audio_source_extension=cached.source_extension,
            audio_object_size_bytes=cached.object_size_bytes,
            audio_storage_class=cached.storage_class,
            submitted_at=submitted_at,
            decode_submitted_at=decode_submitted_at,
            decode_started_at=decode_started_at,
            download_decode_sec=total,
            resolve_sec=stats.resolve_sec,
            head_sec=stats.head_sec,
            download_sec=stats.download_sec,
            decode_sec=stats.decode_sec,
            cache_wait_sec=stats.cache_wait_sec,
            audio_cache_enabled=True,
            audio_cache_payload_type="decoded-f32-mono-v1",
            resolution_cache_hit=stats.resolution_cache_hit,
            object_cache_hit=stats.object_cache_hit,
            cache_write=stats.cache_write,
            cache_fallback=stats.cache_fallback,
            cache_fallback_reason=stats.cache_fallback_reason,
            decoded_bytes=stats.decoded_bytes,
        )

    def _add_vad(
        self,
        decoded: _DecodedResult,
        vad_submitted_at: float,
    ) -> PrefetchResult:
        started_at = time.perf_counter()
        detector = self._thread_vad_detector()
        intervals = detector(decoded.audio.samples, decoded.audio.sample_rate)
        vad_sec = time.perf_counter() - started_at
        return self._prefetch_result(
            decoded,
            intervals,
            vad_sec,
            started_at - vad_submitted_at,
        )

    def _prefetch_result(
        self,
        decoded: _DecodedResult,
        vad_intervals: Sequence[tuple[float, float]] | None,
        vad_sec: float,
        vad_queue_wait_sec: float,
    ) -> PrefetchResult:
        return PrefetchResult(
            audio=decoded.audio,
            s3_key=decoded.s3_key,
            audio_source_extension=decoded.audio_source_extension,
            audio_object_size_bytes=decoded.audio_object_size_bytes,
            audio_storage_class=decoded.audio_storage_class,
            audio_cache_enabled=decoded.audio_cache_enabled,
            audio_cache_payload_type=decoded.audio_cache_payload_type,
            resolution_cache_hit=decoded.resolution_cache_hit,
            object_cache_hit=decoded.object_cache_hit,
            cache_write=decoded.cache_write,
            cache_fallback=decoded.cache_fallback,
            cache_fallback_reason=decoded.cache_fallback_reason,
            decoded_bytes=decoded.decoded_bytes,
            vad_intervals=(
                None
                if vad_intervals is None
                else [(float(start), float(end)) for start, end in vad_intervals]
            ),
            timings=PrefetchTimings(
                decode_queue_wait_sec=(
                    decoded.decode_started_at - decoded.decode_submitted_at
                ),
                download_decode_sec=decoded.download_decode_sec,
                resolve_sec=decoded.resolve_sec,
                head_sec=decoded.head_sec,
                download_sec=decoded.download_sec,
                decode_sec=decoded.decode_sec,
                cache_wait_sec=decoded.cache_wait_sec,
                vad_queue_wait_sec=float(vad_queue_wait_sec),
                vad_sec=float(vad_sec),
                prefetch_submit_to_ready_sec=0.0,
                total_sec=0.0,
            ),
            _submitted_at=decoded.submitted_at,
        )

    def _set_final_result(
        self,
        future: Future,
        result: PrefetchResult | AudioResolutionError,
    ) -> None:
        future.set_result(self._mark_ready(result))

    def _mark_ready(
        self,
        result: PrefetchResult | AudioResolutionError,
    ) -> PrefetchResult | AudioResolutionError:
        ready_time = time.perf_counter()
        if isinstance(result, AudioResolutionError):
            return replace(result, ready_time=ready_time)
        submit_to_ready = ready_time - result._submitted_at
        timings = replace(
            result.timings,
            prefetch_submit_to_ready_sec=submit_to_ready,
            total_sec=submit_to_ready,
        )
        return replace(result, timings=timings, ready_time=ready_time)

    def _thread_vad_detector(self) -> VadDetectorFn:
        detector = getattr(self._vad_local, "detector", None)
        if detector is None:
            detector = self._make_vad_detector()
            self._vad_local.detector = detector
        return detector

    def _make_vad_detector(self) -> VadDetectorFn:
        if self._vad_detector_factory is not None:
            return self._vad_detector_factory()
        from ..inference.models import VadDetector

        return VadDetector()
