"""Background audio download and decode to hide S3 + CPU latency.

The ``Prefetcher`` downloads audio from S3 **and** decodes it to a mono
16 kHz float32 ``AudioData`` object in background threads.  The worker's
main loop submits entities ahead of time (lookahead window) and retrieves
ready results just before GPU inference.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from ..inference.audio import AudioData, load_audio
from .audio_resolver import AudioResolutionError, resolve_and_download
from .manifest import ArchiveEntity

LOGGER = logging.getLogger(__name__)


class Prefetcher:
    """Download + decode audio ahead of the main processing loop.

    Usage::

        pf = Prefetcher(output_base, sample_rate=16000, max_workers=4)
        pf.submit(entity)
        ...
        result = pf.get(entity)  # blocks until ready
        # result is (AudioData, s3_key) or AudioResolutionError
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
    ) -> None:
        self._sample_rate = sample_rate
        self._s3_client = s3_client
        self._bucket = bucket
        self._tmp_dir = tmp_dir
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="prefetch")
        self._futures: dict[tuple[str, str], Future] = {}
        self._tmp_paths: dict[tuple[str, str], Path] = {}

    def submit(self, entity: ArchiveEntity) -> None:
        """Queue *entity* for background download + decode."""
        key = (entity.session_id, entity.archive_id)
        if key in self._futures:
            return
        self._futures[key] = self._pool.submit(self._fetch_and_decode, entity)

    def get(
        self, entity: ArchiveEntity
    ) -> tuple[AudioData, str] | AudioResolutionError:
        """Block until the result for *entity* is ready.

        Returns ``(AudioData, s3_key)`` on success, or
        ``AudioResolutionError`` on failure.
        """
        key = (entity.session_id, entity.archive_id)
        future = self._futures.get(key)
        if future is None:
            return self._fetch_and_decode(entity)
        return future.result()

    def discard(self, entity: ArchiveEntity) -> None:
        """Clean up temp files and memory for *entity*."""
        key = (entity.session_id, entity.archive_id)
        self._futures.pop(key, None)
        tmp = self._tmp_paths.pop(key, None)
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def shutdown(self) -> None:
        """Shut down the thread pool and clean up all temp files."""
        self._pool.shutdown(wait=False, cancel_futures=True)
        for tmp in self._tmp_paths.values():
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        self._tmp_paths.clear()
        self._futures.clear()

    def _fetch_and_decode(
        self, entity: ArchiveEntity
    ) -> tuple[AudioData, str] | AudioResolutionError:
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
        self._tmp_paths[key] = local_path

        try:
            audio = load_audio(
                local_path,
                sample_rate=self._sample_rate,
                recording_id=entity.archive_id,
            )
        except Exception as exc:
            self._tmp_paths.pop(key, None)
            local_path.unlink(missing_ok=True)
            return AudioResolutionError(
                session_id=entity.session_id,
                archive_id=entity.archive_id,
                file_parent_dir=entity.file_parent_dir,
                error_type="download_failed",
                detail=f"Decode failed: {type(exc).__name__}: {exc}",
            )

        return audio, s3_key
