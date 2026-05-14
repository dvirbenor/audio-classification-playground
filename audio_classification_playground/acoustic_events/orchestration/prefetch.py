"""Background audio download, decode, and optional VAD precomputation.

The ``Prefetcher`` downloads audio from S3, decodes it to a mono 16 kHz
``AudioData`` object, and can hand the decoded audio to a separate CPU VAD
executor.  VAD workers lazily create their own detector instances so a
stateful Silero model is never shared across concurrent calls.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..inference.audio import AudioData, load_audio
from .audio_resolver import AudioResolutionError, resolve_and_download
from .manifest import ArchiveEntity

VadDetectorFn = Callable[[np.ndarray, int], Sequence[tuple[float, float]]]


@dataclass(frozen=True)
class PrefetchTimings:
    """Wall-clock timings measured inside the prefetch pipeline."""

    download_decode_sec: float = 0.0
    vad_sec: float = 0.0
    total_sec: float = 0.0


@dataclass(frozen=True)
class PrefetchResult:
    """Ready-to-infer inputs for one claimed archive."""

    audio: AudioData
    s3_key: str
    vad_intervals: list[tuple[float, float]] | None = None
    timings: PrefetchTimings = PrefetchTimings()


@dataclass(frozen=True)
class _DecodedResult:
    audio: AudioData
    s3_key: str
    download_decode_sec: float
    started_at: float


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
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if vad_workers < 0:
            raise ValueError("vad_workers must be >= 0")
        self._sample_rate = sample_rate
        self._s3_client = s3_client
        self._bucket = bucket
        self._tmp_dir = tmp_dir
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
        self._discarded: set[tuple[str, str]] = set()

        if prewarm_vad and vad_workers > 0:
            self._make_vad_detector()

    def submit(self, entity: ArchiveEntity, *, precompute_vad: bool = False) -> None:
        """Queue *entity* for background download/decode and optional VAD."""
        key = (entity.session_id, entity.archive_id)
        with self._lock:
            if key in self._futures:
                return
            self._discarded.discard(key)
            final_future: Future = Future()
            self._futures[key] = final_future

        decode_future = self._pool.submit(self._fetch_and_decode, entity)
        with self._lock:
            self._decode_futures[key] = decode_future

        def _after_decode(done: Future) -> None:
            try:
                decoded = done.result()
                if isinstance(decoded, AudioResolutionError):
                    final_future.set_result(decoded)
                    return
                if not precompute_vad:
                    final_future.set_result(self._prefetch_result(decoded, None, 0.0))
                    return
                if self._vad_pool is None:
                    raise RuntimeError("precompute_vad=True requires vad_workers > 0")
                vad_future = self._vad_pool.submit(self._add_vad, decoded)
                with self._lock:
                    self._vad_futures[key] = vad_future

                def _after_vad(vad_done: Future) -> None:
                    try:
                        final_future.set_result(vad_done.result())
                    except BaseException as exc:
                        final_future.set_exception(exc)

                vad_future.add_done_callback(_after_vad)
            except BaseException as exc:
                final_future.set_exception(exc)

        decode_future.add_done_callback(_after_decode)

    def get(
        self, entity: ArchiveEntity
    ) -> PrefetchResult | AudioResolutionError:
        """Block until the result for *entity* is ready.

        Returns ``PrefetchResult`` on success, ``AudioResolutionError`` on
        audio resolution/decode failure, or raises for VAD failures.
        """
        key = (entity.session_id, entity.archive_id)
        future = self._futures.get(key)
        if future is None:
            decoded = self._fetch_and_decode(entity)
            if isinstance(decoded, AudioResolutionError):
                return decoded
            return self._prefetch_result(decoded, None, 0.0)
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
        self, entity: ArchiveEntity
    ) -> _DecodedResult | AudioResolutionError:
        started_at = time.perf_counter()
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

        local_path, s3_key = result
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
                )
            self._tmp_paths[key] = local_path

        try:
            audio = load_audio(
                local_path,
                sample_rate=self._sample_rate,
                recording_id=entity.archive_id,
            )
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
            )

        return _DecodedResult(
            audio=audio,
            s3_key=s3_key,
            download_decode_sec=time.perf_counter() - started_at,
            started_at=started_at,
        )

    def _add_vad(self, decoded: _DecodedResult) -> PrefetchResult:
        started_at = time.perf_counter()
        detector = self._thread_vad_detector()
        intervals = detector(decoded.audio.samples, decoded.audio.sample_rate)
        vad_sec = time.perf_counter() - started_at
        return self._prefetch_result(decoded, intervals, vad_sec)

    def _prefetch_result(
        self,
        decoded: _DecodedResult,
        vad_intervals: Sequence[tuple[float, float]] | None,
        vad_sec: float,
    ) -> PrefetchResult:
        return PrefetchResult(
            audio=decoded.audio,
            s3_key=decoded.s3_key,
            vad_intervals=(
                None
                if vad_intervals is None
                else [(float(start), float(end)) for start, end in vad_intervals]
            ),
            timings=PrefetchTimings(
                download_decode_sec=decoded.download_decode_sec,
                vad_sec=float(vad_sec),
                total_sec=time.perf_counter() - decoded.started_at,
            ),
        )

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
