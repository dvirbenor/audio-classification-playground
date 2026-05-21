"""Shared decoded-audio cache for orchestration workers.

The cache is an optimization layer on shared storage.  Task artifacts and
task locks remain authoritative; cache misses and cache pressure fall back to
the normal worker-owned download/decode path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..inference.audio import AudioData, load_audio
from .audio_resolver import (
    BUCKET,
    AudioDownloadResult,
    AudioResolutionError,
    download_audio,
    resolve_audio_key,
)
from .locking import iter_lock_files
from .manifest import ArchiveEntity

LOGGER = logging.getLogger(__name__)

DECODED_CACHE_VERSION = "decoded-f32-mono-v1"
DEFAULT_CACHE_LOCK_STALE_MINUTES = 60.0
_POLL_SEC = 2.0


@dataclass(frozen=True)
class CacheAcquisitionStats:
    """Diagnostics for one shared-cache acquisition."""

    resolution_cache_hit: bool = False
    object_cache_hit: bool = False
    cache_write: bool = False
    cache_fallback: bool = False
    cache_fallback_reason: str = ""
    cache_wait_sec: float = 0.0
    resolve_sec: float = 0.0
    head_sec: float = 0.0
    download_sec: float = 0.0
    decode_sec: float = 0.0
    decoded_bytes: int = 0


@dataclass(frozen=True)
class CachedAudio:
    """Decoded audio plus source metadata returned by ``SharedAudioCache``."""

    audio: AudioData
    s3_key: str
    source_extension: str = ""
    object_size_bytes: int | None = None
    storage_class: str | None = None
    cache_path: Path | None = None
    stats: CacheAcquisitionStats = CacheAcquisitionStats()


@dataclass(frozen=True)
class _ResolvedAudio:
    s3_key: str
    source_extension: str
    object_size_bytes: int | None = None
    storage_class: str | None = None
    cache_hit: bool = False
    resolve_sec: float = 0.0


@dataclass(frozen=True)
class CleanupSummary:
    removed_objects: int = 0
    removed_bytes: int = 0
    removed_locks: int = 0
    removed_temps: int = 0


class SharedAudioCache:
    """Decoded 16 kHz mono float32 cache shared by orchestration pods."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        sample_rate: int,
        max_cache_bytes: int,
        stale_lock_minutes: float = DEFAULT_CACHE_LOCK_STALE_MINUTES,
        s3_client=None,
        bucket: str = BUCKET,
        poll_sec: float = _POLL_SEC,
    ) -> None:
        if max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be > 0")
        if stale_lock_minutes <= 0:
            raise ValueError("stale_lock_minutes must be > 0")
        self.cache_dir = Path(cache_dir)
        self.sample_rate = int(sample_rate)
        self.max_cache_bytes = int(max_cache_bytes)
        self.stale_lock_minutes = float(stale_lock_minutes)
        self._s3_client = s3_client
        self.bucket = bucket
        self.poll_sec = float(poll_sec)

    def get_decoded_audio(
        self,
        entity: ArchiveEntity,
        *,
        tmp_dir: str | None = None,
    ) -> CachedAudio | AudioResolutionError:
        """Return decoded audio from cache, materializing it if needed."""
        resolved = self.resolve(entity)
        if isinstance(resolved, AudioResolutionError):
            return resolved

        object_key = decoded_object_key(resolved.s3_key, self.sample_rate)
        object_path, metadata_path = self._object_paths(object_key)
        object_lock = self._object_lock_path(object_key)

        hit = self._load_ready_object(
            entity,
            resolved,
            object_path,
            metadata_path,
            resolution_cache_hit=resolved.cache_hit,
            waited_sec=0.0,
        )
        if hit is not None:
            return hit

        if self._current_cache_bytes() >= self.max_cache_bytes:
            return self._download_decode_without_cache(
                entity,
                resolved,
                tmp_dir=tmp_dir,
                resolution_cache_hit=resolved.cache_hit,
                fallback_reason="cache_full",
            )

        waited_sec = 0.0
        while True:
            if self._try_create_lock(object_lock, payload={
                "kind": "object",
                "object_key": object_key,
                "s3_key": resolved.s3_key,
                "created_at": time.time(),
            }):
                break

            wait_started = time.perf_counter()
            reclaimed = self._wait_for_ready_or_reclaim(
                lock_path=object_lock,
                ready_paths=(object_path, metadata_path),
            )
            waited_sec += time.perf_counter() - wait_started
            hit = self._load_ready_object(
                entity,
                resolved,
                object_path,
                metadata_path,
                resolution_cache_hit=resolved.cache_hit,
                waited_sec=waited_sec,
            )
            if hit is not None:
                return hit
            if reclaimed:
                continue

        try:
            hit = self._load_ready_object(
                entity,
                resolved,
                object_path,
                metadata_path,
                resolution_cache_hit=resolved.cache_hit,
                waited_sec=waited_sec,
            )
            if hit is not None:
                return hit

            return self._materialize_object(
                entity,
                resolved,
                object_key,
                object_path,
                metadata_path,
                tmp_dir=tmp_dir,
                resolution_cache_hit=resolved.cache_hit,
                waited_sec=waited_sec,
            )
        finally:
            self._release_lock(object_lock)

    def resolve(self, entity: ArchiveEntity) -> _ResolvedAudio | AudioResolutionError:
        """Resolve an entity to a source S3 key using the shared resolution cache."""
        resolution_key = resolution_cache_key(self.bucket, entity.file_parent_dir)
        record_path = self._resolution_path(resolution_key)
        lock_path = self._resolution_lock_path(resolution_key)

        record = self._read_json(record_path)
        if record:
            cached = self._resolved_from_record(entity, record, cache_hit=True)
            if cached is not None:
                return cached

        waited_sec = 0.0
        while True:
            if self._try_create_lock(lock_path, payload={
                "kind": "resolution",
                "resolution_key": resolution_key,
                "file_parent_dir": entity.file_parent_dir,
                "created_at": time.time(),
            }):
                break

            wait_started = time.perf_counter()
            reclaimed = self._wait_for_ready_or_reclaim(
                lock_path=lock_path,
                ready_paths=(record_path,),
            )
            waited_sec += time.perf_counter() - wait_started
            record = self._read_json(record_path)
            if record:
                cached = self._resolved_from_record(entity, record, cache_hit=True)
                if cached is not None:
                    return cached
            if reclaimed:
                continue

        try:
            record = self._read_json(record_path)
            if record:
                cached = self._resolved_from_record(entity, record, cache_hit=True)
                if cached is not None:
                    return cached

            if self._s3_client is None:
                self._s3_client = _get_s3_client()

            started = time.perf_counter()
            result = resolve_audio_key(
                self._s3_client,
                entity.file_parent_dir,
                bucket=self.bucket,
            )
            resolve_sec = time.perf_counter() - started + waited_sec
            if isinstance(result, AudioResolutionError):
                error = AudioResolutionError(
                    session_id=entity.session_id,
                    archive_id=entity.archive_id,
                    file_parent_dir=entity.file_parent_dir,
                    error_type=result.error_type,
                    detail=result.detail,
                    s3_key=result.s3_key,
                    resolve_sec=resolve_sec,
                )
                if error.is_permanent:
                    self._write_json_atomic(record_path, {
                        "status": "no_matching_file",
                        "bucket": self.bucket,
                        "file_parent_dir": entity.file_parent_dir,
                        "detail": error.detail,
                        "created_at": time.time(),
                    })
                return error

            resolved = _ResolvedAudio(
                s3_key=result,
                source_extension=Path(result).suffix.lower(),
                cache_hit=False,
                resolve_sec=resolve_sec,
            )
            self._write_json_atomic(record_path, {
                "status": "resolved",
                "bucket": self.bucket,
                "file_parent_dir": entity.file_parent_dir,
                "s3_key": resolved.s3_key,
                "source_extension": resolved.source_extension,
                "created_at": time.time(),
            })
            return resolved
        finally:
            self._release_lock(lock_path)

    def cache_bytes(self) -> int:
        """Return approximate ready-object cache bytes."""
        return self._ready_object_bytes()

    def cached_s3_key_for_entity(self, entity: ArchiveEntity) -> str | None:
        """Return a resolved S3 key only if it is already in the cache."""
        record = self._read_json(
            self._resolution_path(
                resolution_cache_key(self.bucket, entity.file_parent_dir),
            ),
        )
        if not record or record.get("status") != "resolved":
            return None
        s3_key = record.get("s3_key")
        return str(s3_key) if s3_key else None

    def cleanup(
        self,
        *,
        output_base: Path | None = None,
        terminal_entities: set[tuple[str, str]] | None = None,
        protected_s3_keys: set[str] | None = None,
        target_bytes: int | None = None,
    ) -> CleanupSummary:
        """Reclaim stale cache state and evict unused objects.

        Terminal objects are removed first.  If *target_bytes* is provided,
        least-recently-used non-terminal objects are removed until the ready
        cache is at or below that target.  Active task locks protect matching
        entities/S3 keys from eviction.
        """
        terminal_entities = terminal_entities or set()
        protected_s3_keys = set(protected_s3_keys or set())
        protected_entities: set[tuple[str, str]] = set()
        if output_base is not None:
            protected_entities = _locked_entities(output_base)

        removed_locks, removed_temps = self.reclaim_stale()
        removed_objects = 0
        removed_bytes = 0

        entries = self._object_entries()
        for entry in entries:
            if not self._can_evict(
                entry,
                terminal_entities=terminal_entities,
                protected_entities=protected_entities,
                protected_s3_keys=protected_s3_keys,
            ):
                continue
            if entry["entity_key"] not in terminal_entities:
                continue
            removed = self._remove_entry(entry)
            if removed:
                removed_objects += 1
                removed_bytes += removed

        if target_bytes is not None:
            for entry in self._object_entries():
                if self._ready_object_bytes() <= target_bytes:
                    break
                if not self._can_evict(
                    entry,
                    terminal_entities=terminal_entities,
                    protected_entities=protected_entities,
                    protected_s3_keys=protected_s3_keys,
                ):
                    continue
                removed = self._remove_entry(entry)
                if removed:
                    removed_objects += 1
                    removed_bytes += removed

        return CleanupSummary(
            removed_objects=removed_objects,
            removed_bytes=removed_bytes,
            removed_locks=removed_locks,
            removed_temps=removed_temps,
        )

    def reclaim_stale(self) -> tuple[int, int]:
        """Remove stale cache locks and temp files."""
        cutoff = time.time() - self.stale_lock_minutes * 60.0
        removed_locks = 0
        removed_temps = 0
        for lock_file in self._lock_root().rglob("*.lock"):
            if not lock_file.is_file():
                continue
            try:
                if lock_file.stat().st_mtime >= cutoff:
                    continue
            except OSError:
                pass
            shard_dir = self._associated_shard_dir(lock_file)
            try:
                lock_file.unlink(missing_ok=True)
                removed_locks += 1
            except OSError:
                LOGGER.warning("Could not remove stale cache lock %s", lock_file)
            if shard_dir is not None:
                removed_temps += self._remove_stale_temps(shard_dir, cutoff)
        removed_temps += self._remove_stale_temps(self._object_root(), cutoff)
        return removed_locks, removed_temps

    def _materialize_object(
        self,
        entity: ArchiveEntity,
        resolved: _ResolvedAudio,
        object_key: str,
        object_path: Path,
        metadata_path: Path,
        *,
        tmp_dir: str | None,
        resolution_cache_hit: bool,
        waited_sec: float,
    ) -> CachedAudio | AudioResolutionError:
        if self._s3_client is None:
            self._s3_client = _get_s3_client()

        dl_result = download_audio(
            self._s3_client,
            resolved.s3_key,
            bucket=self.bucket,
            tmp_dir=tmp_dir,
        )
        if isinstance(dl_result, AudioResolutionError):
            return AudioResolutionError(
                session_id=entity.session_id,
                archive_id=entity.archive_id,
                file_parent_dir=entity.file_parent_dir,
                error_type=dl_result.error_type,
                detail=dl_result.detail,
                s3_key=dl_result.s3_key,
                resolve_sec=resolved.resolve_sec,
                head_sec=dl_result.head_sec,
                download_sec=dl_result.download_sec,
            )
        self._update_resolution_record(entity, resolved, dl_result)

        decode_started = time.perf_counter()
        try:
            audio = load_audio(
                dl_result.local_path,
                sample_rate=self.sample_rate,
                recording_id=entity.archive_id,
            )
        except Exception as exc:
            return AudioResolutionError(
                session_id=entity.session_id,
                archive_id=entity.archive_id,
                file_parent_dir=entity.file_parent_dir,
                error_type="download_failed",
                detail=f"Decode failed: {type(exc).__name__}: {exc}",
                s3_key=resolved.s3_key,
                resolve_sec=resolved.resolve_sec,
                head_sec=dl_result.head_sec,
                download_sec=dl_result.download_sec,
            )
        finally:
            dl_result.local_path.unlink(missing_ok=True)
        decode_sec = time.perf_counter() - decode_started

        samples = np.ascontiguousarray(audio.samples, dtype="<f4")
        decoded_bytes = int(samples.nbytes)
        reserved = self._try_reserve(object_key, decoded_bytes)
        if not reserved:
            return CachedAudio(
                audio=AudioData(
                    path=audio.path,
                    recording_id=audio.recording_id,
                    samples=samples,
                    sample_rate=audio.sample_rate,
                    duration_sec=audio.duration_sec,
                    audio_sha256=audio.audio_sha256,
                ),
                s3_key=resolved.s3_key,
                source_extension=dl_result.source_extension,
                object_size_bytes=dl_result.object_size_bytes,
                storage_class=dl_result.storage_class,
                cache_path=None,
                stats=CacheAcquisitionStats(
                    resolution_cache_hit=resolution_cache_hit,
                    object_cache_hit=False,
                    cache_write=False,
                    cache_fallback=True,
                    cache_fallback_reason="capacity",
                    cache_wait_sec=waited_sec,
                    resolve_sec=resolved.resolve_sec,
                    head_sec=dl_result.head_sec,
                    download_sec=dl_result.download_sec,
                    decode_sec=decode_sec,
                    decoded_bytes=decoded_bytes,
                ),
            )

        try:
            try:
                self._write_decoded_object(
                    object_path,
                    metadata_path,
                    samples=samples,
                    metadata={
                        "version": DECODED_CACHE_VERSION,
                        "bucket": self.bucket,
                        "s3_key": resolved.s3_key,
                        "source_extension": dl_result.source_extension,
                        "source_object_size_bytes": dl_result.object_size_bytes,
                        "source_storage_class": dl_result.storage_class,
                        "audio_sha256": audio.audio_sha256,
                        "n_samples": int(len(samples)),
                        "decoded_bytes": decoded_bytes,
                        "sample_rate": int(audio.sample_rate),
                        "duration_sec": float(audio.duration_sec),
                        "first_entity": {
                            "session_id": entity.session_id,
                            "archive_id": entity.archive_id,
                            "file_parent_dir": entity.file_parent_dir,
                        },
                        "created_at": time.time(),
                    },
                )
            except Exception:
                LOGGER.warning("Decoded cache write failed for %s", resolved.s3_key, exc_info=True)
                return CachedAudio(
                    audio=AudioData(
                        path=audio.path,
                        recording_id=audio.recording_id,
                        samples=samples,
                        sample_rate=audio.sample_rate,
                        duration_sec=audio.duration_sec,
                        audio_sha256=audio.audio_sha256,
                    ),
                    s3_key=resolved.s3_key,
                    source_extension=dl_result.source_extension,
                    object_size_bytes=dl_result.object_size_bytes,
                    storage_class=dl_result.storage_class,
                    cache_path=None,
                    stats=CacheAcquisitionStats(
                        resolution_cache_hit=resolution_cache_hit,
                        object_cache_hit=False,
                        cache_write=False,
                        cache_fallback=True,
                        cache_fallback_reason="write_failed",
                        cache_wait_sec=waited_sec,
                        resolve_sec=resolved.resolve_sec,
                        head_sec=dl_result.head_sec,
                        download_sec=dl_result.download_sec,
                        decode_sec=decode_sec,
                        decoded_bytes=decoded_bytes,
                    ),
                )
        finally:
            self._release_reservation(object_key)

        return CachedAudio(
            audio=AudioData(
                path=object_path,
                recording_id=audio.recording_id,
                samples=samples,
                sample_rate=audio.sample_rate,
                duration_sec=audio.duration_sec,
                audio_sha256=audio.audio_sha256,
            ),
            s3_key=resolved.s3_key,
            source_extension=dl_result.source_extension,
            object_size_bytes=dl_result.object_size_bytes,
            storage_class=dl_result.storage_class,
            cache_path=object_path,
            stats=CacheAcquisitionStats(
                resolution_cache_hit=resolution_cache_hit,
                object_cache_hit=False,
                cache_write=True,
                cache_fallback=False,
                cache_wait_sec=waited_sec,
                resolve_sec=resolved.resolve_sec,
                head_sec=dl_result.head_sec,
                download_sec=dl_result.download_sec,
                decode_sec=decode_sec,
                decoded_bytes=decoded_bytes,
            ),
        )

    def _download_decode_without_cache(
        self,
        entity: ArchiveEntity,
        resolved: _ResolvedAudio,
        *,
        tmp_dir: str | None,
        resolution_cache_hit: bool,
        fallback_reason: str,
    ) -> CachedAudio | AudioResolutionError:
        if self._s3_client is None:
            self._s3_client = _get_s3_client()
        dl_result = download_audio(
            self._s3_client,
            resolved.s3_key,
            bucket=self.bucket,
            tmp_dir=tmp_dir,
        )
        if isinstance(dl_result, AudioResolutionError):
            return AudioResolutionError(
                session_id=entity.session_id,
                archive_id=entity.archive_id,
                file_parent_dir=entity.file_parent_dir,
                error_type=dl_result.error_type,
                detail=dl_result.detail,
                s3_key=dl_result.s3_key,
                resolve_sec=resolved.resolve_sec,
                head_sec=dl_result.head_sec,
                download_sec=dl_result.download_sec,
            )
        self._update_resolution_record(entity, resolved, dl_result)
        decode_started = time.perf_counter()
        try:
            audio = load_audio(
                dl_result.local_path,
                sample_rate=self.sample_rate,
                recording_id=entity.archive_id,
            )
        except Exception as exc:
            return AudioResolutionError(
                session_id=entity.session_id,
                archive_id=entity.archive_id,
                file_parent_dir=entity.file_parent_dir,
                error_type="download_failed",
                detail=f"Decode failed: {type(exc).__name__}: {exc}",
                s3_key=resolved.s3_key,
                resolve_sec=resolved.resolve_sec,
                head_sec=dl_result.head_sec,
                download_sec=dl_result.download_sec,
            )
        finally:
            dl_result.local_path.unlink(missing_ok=True)
        decode_sec = time.perf_counter() - decode_started
        decoded_bytes = int(audio.samples.nbytes)
        return CachedAudio(
            audio=audio,
            s3_key=resolved.s3_key,
            source_extension=dl_result.source_extension,
            object_size_bytes=dl_result.object_size_bytes,
            storage_class=dl_result.storage_class,
            cache_path=None,
            stats=CacheAcquisitionStats(
                resolution_cache_hit=resolution_cache_hit,
                object_cache_hit=False,
                cache_write=False,
                cache_fallback=True,
                cache_fallback_reason=fallback_reason,
                resolve_sec=resolved.resolve_sec,
                head_sec=dl_result.head_sec,
                download_sec=dl_result.download_sec,
                decode_sec=decode_sec,
                decoded_bytes=decoded_bytes,
            ),
        )

    def _load_ready_object(
        self,
        entity: ArchiveEntity,
        resolved: _ResolvedAudio,
        object_path: Path,
        metadata_path: Path,
        *,
        resolution_cache_hit: bool,
        waited_sec: float,
    ) -> CachedAudio | None:
        if not object_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = self._read_json(metadata_path)
            if not metadata or metadata.get("version") != DECODED_CACHE_VERSION:
                return None
            samples = np.load(object_path, allow_pickle=False)
            samples = np.ascontiguousarray(samples, dtype=np.float32)
            sample_rate = int(metadata["sample_rate"])
            if sample_rate != self.sample_rate:
                return None
            audio = AudioData(
                path=object_path,
                recording_id=entity.archive_id,
                samples=samples,
                sample_rate=sample_rate,
                duration_sec=float(metadata["duration_sec"]),
                audio_sha256=str(metadata["audio_sha256"]),
            )
            now = time.time()
            os.utime(object_path, (now, now))
            os.utime(metadata_path, (now, now))
            return CachedAudio(
                audio=audio,
                s3_key=str(metadata.get("s3_key") or resolved.s3_key),
                source_extension=str(metadata.get("source_extension") or resolved.source_extension),
                object_size_bytes=_int_or_none(metadata.get("source_object_size_bytes")),
                storage_class=(
                    None
                    if metadata.get("source_storage_class") is None
                    else str(metadata.get("source_storage_class"))
                ),
                cache_path=object_path,
                stats=CacheAcquisitionStats(
                    resolution_cache_hit=resolution_cache_hit,
                    object_cache_hit=True,
                    cache_wait_sec=waited_sec,
                    decoded_bytes=int(metadata.get("decoded_bytes") or samples.nbytes),
                ),
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def _resolved_from_record(
        self,
        entity: ArchiveEntity,
        record: dict,
        *,
        cache_hit: bool,
    ) -> _ResolvedAudio | AudioResolutionError | None:
        status = record.get("status")
        if status == "resolved":
            s3_key = str(record.get("s3_key") or "")
            if not s3_key:
                return None
            return _ResolvedAudio(
                s3_key=s3_key,
                source_extension=str(record.get("source_extension") or Path(s3_key).suffix.lower()),
                object_size_bytes=_int_or_none(record.get("object_size_bytes")),
                storage_class=(
                    None
                    if record.get("storage_class") is None
                    else str(record.get("storage_class"))
                ),
                cache_hit=cache_hit,
            )
        if status == "no_matching_file":
            return AudioResolutionError(
                session_id=entity.session_id,
                archive_id=entity.archive_id,
                file_parent_dir=entity.file_parent_dir,
                error_type="no_matching_file",
                detail=str(record.get("detail") or "No matching WAV/MP3"),
            )
        return None

    def _update_resolution_record(
        self,
        entity: ArchiveEntity,
        resolved: _ResolvedAudio,
        download: AudioDownloadResult,
    ) -> None:
        resolution_key = resolution_cache_key(self.bucket, entity.file_parent_dir)
        self._write_json_atomic(self._resolution_path(resolution_key), {
            "status": "resolved",
            "bucket": self.bucket,
            "file_parent_dir": entity.file_parent_dir,
            "s3_key": resolved.s3_key,
            "source_extension": download.source_extension or resolved.source_extension,
            "object_size_bytes": download.object_size_bytes,
            "storage_class": download.storage_class,
            "updated_at": time.time(),
        })

    def _write_decoded_object(
        self,
        object_path: Path,
        metadata_path: Path,
        *,
        samples: np.ndarray,
        metadata: dict,
    ) -> None:
        object_path.parent.mkdir(parents=True, exist_ok=True)
        token = f"{os.environ.get('HOSTNAME', 'unknown')}.{os.getpid()}.{uuid.uuid4().hex}"
        tmp_object = object_path.parent / f".tmp.{token}.npy"
        try:
            with open(tmp_object, "wb") as f:
                np.save(f, np.ascontiguousarray(samples, dtype="<f4"), allow_pickle=False)
            os.replace(str(tmp_object), str(object_path))
            self._write_json_atomic(metadata_path, metadata)
        finally:
            tmp_object.unlink(missing_ok=True)

    def _try_reserve(self, object_key: str, decoded_bytes: int) -> bool:
        reservation_path = self._reservation_path(object_key)
        capacity_lock = self._capacity_lock_path()
        self._wait_and_acquire_lock(capacity_lock, payload={
            "kind": "capacity",
            "created_at": time.time(),
        })
        try:
            current = self._ready_object_bytes() + self._reservation_bytes()
            if current + int(decoded_bytes) > self.max_cache_bytes:
                return False
            self._write_json_atomic(reservation_path, {
                "object_key": object_key,
                "bytes": int(decoded_bytes),
                "created_at": time.time(),
            })
            return True
        finally:
            self._release_lock(capacity_lock)

    def _release_reservation(self, object_key: str) -> None:
        self._reservation_path(object_key).unlink(missing_ok=True)

    def _current_cache_bytes(self) -> int:
        return self._ready_object_bytes() + self._reservation_bytes()

    def _ready_object_bytes(self) -> int:
        total = 0
        root = self._object_root()
        if not root.is_dir():
            return 0
        for path in root.rglob("*.npy"):
            if path.name.startswith(".tmp."):
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def _reservation_bytes(self) -> int:
        total = 0
        root = self._reservation_root()
        if not root.is_dir():
            return 0
        for path in root.rglob("*.json"):
            data = self._read_json(path)
            if data:
                total += int(data.get("bytes") or 0)
        return total

    def _wait_and_acquire_lock(self, lock_path: Path, *, payload: dict) -> None:
        while True:
            if self._try_create_lock(lock_path, payload=payload):
                return
            self._wait_for_ready_or_reclaim(lock_path=lock_path, ready_paths=())

    def _wait_for_ready_or_reclaim(
        self,
        *,
        lock_path: Path,
        ready_paths: Iterable[Path],
    ) -> bool:
        if any(path.is_file() for path in ready_paths):
            return False
        if self._is_stale(lock_path):
            shard_dir = self._associated_shard_dir(lock_path)
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                return False
            if shard_dir is not None:
                cutoff = time.time() - self.stale_lock_minutes * 60.0
                self._remove_stale_temps(shard_dir, cutoff)
            return True
        time.sleep(self.poll_sec)
        return False

    def _try_create_lock(self, lock_path: Path, *, payload: dict) -> bool:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            data = dict(payload)
            data.update({
                "worker": os.environ.get("HOSTNAME", "unknown"),
                "pid": os.getpid(),
                "time": time.time(),
            })
            json.dump(data, f, default=str)
        return True

    def _release_lock(self, lock_path: Path) -> None:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            LOGGER.warning("Failed to release cache lock %s", lock_path)

    def _is_stale(self, lock_path: Path) -> bool:
        try:
            return (
                time.time() - lock_path.stat().st_mtime
                > self.stale_lock_minutes * 60.0
            )
        except OSError:
            return True

    def _associated_shard_dir(self, lock_path: Path) -> Path | None:
        parent = lock_path.parent
        if not parent.name:
            return None
        if "object_locks" in lock_path.parts:
            return self._object_root() / parent.name
        if "resolution_locks" in lock_path.parts:
            return self._resolution_root() / parent.name
        return None

    def _remove_stale_temps(self, root: Path, cutoff: float) -> int:
        if not root.exists():
            return 0
        removed = 0
        for path in root.rglob(".tmp.*"):
            try:
                if path.stat().st_mtime > cutoff:
                    continue
                path.unlink(missing_ok=True)
                removed += 1
            except OSError:
                continue
        return removed

    def _object_entries(self) -> list[dict]:
        entries: list[dict] = []
        root = self._object_root()
        if not root.is_dir():
            return entries
        for object_path in root.rglob("*.npy"):
            if object_path.name.startswith(".tmp."):
                continue
            metadata_path = object_path.with_suffix(".json")
            metadata = self._read_json(metadata_path) or {}
            try:
                st = object_path.stat()
            except OSError:
                continue
            first = metadata.get("first_entity") or {}
            entity_key = (
                str(first.get("session_id") or ""),
                str(first.get("archive_id") or ""),
            )
            entries.append({
                "object_path": object_path,
                "metadata_path": metadata_path,
                "metadata": metadata,
                "bytes": int(st.st_size),
                "mtime": float(st.st_mtime),
                "s3_key": str(metadata.get("s3_key") or ""),
                "entity_key": entity_key,
            })
        entries.sort(key=lambda item: item["mtime"])
        return entries

    def _can_evict(
        self,
        entry: dict,
        *,
        terminal_entities: set[tuple[str, str]],
        protected_entities: set[tuple[str, str]],
        protected_s3_keys: set[str],
    ) -> bool:
        entity_key = entry.get("entity_key")
        s3_key = str(entry.get("s3_key") or "")
        if entity_key in protected_entities:
            return False
        if s3_key and s3_key in protected_s3_keys:
            return False
        return True

    def _remove_entry(self, entry: dict) -> int:
        object_path: Path = entry["object_path"]
        metadata_path: Path = entry["metadata_path"]
        try:
            size = object_path.stat().st_size
        except OSError:
            size = int(entry.get("bytes") or 0)
        try:
            object_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            return int(size)
        except OSError:
            return 0

    def _read_json(self, path: Path) -> dict | None:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _write_json_atomic(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".tmp.{path.name}.",
            suffix=".json",
            dir=str(path.parent),
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(str(tmp), str(path))
        finally:
            tmp.unlink(missing_ok=True)

    def _resolution_path(self, key: str) -> Path:
        return self._resolution_root() / key[:2] / f"{key}.json"

    def _resolution_lock_path(self, key: str) -> Path:
        return self._lock_root() / "resolution_locks" / key[:2] / f"{key}.lock"

    def _object_paths(self, key: str) -> tuple[Path, Path]:
        object_path = self._object_root() / key[:2] / f"{key}.npy"
        return object_path, object_path.with_suffix(".json")

    def _object_lock_path(self, key: str) -> Path:
        return self._lock_root() / "object_locks" / key[:2] / f"{key}.lock"

    def _reservation_path(self, key: str) -> Path:
        return self._reservation_root() / key[:2] / f"{key}.json"

    def _capacity_lock_path(self) -> Path:
        return self._lock_root() / "capacity.lock"

    def _resolution_root(self) -> Path:
        return self.cache_dir / "resolution"

    def _object_root(self) -> Path:
        return self.cache_dir / "objects"

    def _reservation_root(self) -> Path:
        return self.cache_dir / "reservations"

    def _lock_root(self) -> Path:
        return self.cache_dir / "locks"


def resolution_cache_key(bucket: str, file_parent_dir: str) -> str:
    return _hash_text(f"{bucket}\n{file_parent_dir}")


def decoded_object_key(s3_key: str, sample_rate: int) -> str:
    return _hash_text(f"{s3_key}\n{int(sample_rate)}\n{DECODED_CACHE_VERSION}")


def _hash_text(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).hexdigest()


def _get_s3_client():
    import boto3

    return boto3.client("s3")


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _locked_entities(output_base: Path) -> set[tuple[str, str]]:
    locked: set[tuple[str, str]] = set()
    for lock_file in iter_lock_files(output_base):
        stem = lock_file.stem
        if "__" not in stem:
            continue
        sid, _, aid = stem.partition("__")
        locked.add((sid, aid))
    return locked
