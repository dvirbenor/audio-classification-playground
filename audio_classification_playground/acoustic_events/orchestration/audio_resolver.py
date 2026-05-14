"""Resolve and download audio files from S3.

Resolution rules:
  1. List all objects under the archive's ``file_parent_dir`` prefix.
  2. Prefer ``*.wav`` files whose stem does not contain "enhanced", "cfr",
     or "CFR".
  3. Fall back to ``*.mp3`` files.
  4. If multiple candidates exist in a tier, pick the lexicographically
     first key for determinism.
  5. If no candidate is found, return ``no_matching_file``.
  6. If the chosen object is in Glacier / Deep Archive, return
     ``glacier_storage_class``.
  7. Transient download failures are retried with exponential backoff.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

BUCKET = "riverside-pro-main"
_EXCLUDED_WAV_PATTERN = re.compile(r"(?:enhanced|cfr|CFR)", re.IGNORECASE)
_MAX_DOWNLOAD_RETRIES = 3
_INITIAL_BACKOFF_SEC = 2.0


@dataclass(frozen=True)
class AudioResolutionError:
    """Describes why audio could not be fetched for an archive."""

    session_id: str
    archive_id: str
    file_parent_dir: str
    error_type: str  # "no_matching_file" | "glacier_storage_class" | "download_failed"
    detail: str = ""

    @property
    def is_permanent(self) -> bool:
        return self.error_type in ("no_matching_file", "glacier_storage_class")


def _get_s3_client():
    import boto3

    return boto3.client("s3")


def resolve_audio_key(
    s3_client,
    file_parent_dir: str,
    bucket: str = BUCKET,
) -> str | AudioResolutionError:
    """Find the best audio key under the given S3 prefix.

    Returns the chosen S3 key on success, or an ``AudioResolutionError``
    describing the failure.
    """
    prefix = file_parent_dir.rstrip("/") + "/"
    wavs: list[str] = []
    mp3s: list[str] = []

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            lower = key.lower()
            if lower.endswith(".wav"):
                stem = Path(key).stem
                if not _EXCLUDED_WAV_PATTERN.search(stem):
                    wavs.append(key)
            elif lower.endswith(".mp3"):
                mp3s.append(key)

    candidates = sorted(wavs) if wavs else sorted(mp3s)
    if not candidates:
        return AudioResolutionError(
            session_id="",
            archive_id="",
            file_parent_dir=file_parent_dir,
            error_type="no_matching_file",
            detail=f"No matching WAV/MP3 under s3://{bucket}/{prefix}",
        )
    return candidates[0]


def download_audio(
    s3_client,
    key: str,
    bucket: str = BUCKET,
    tmp_dir: str | None = None,
) -> Path | AudioResolutionError:
    """Download a single S3 object, retrying transient failures.

    Returns the local temp path on success, or an ``AudioResolutionError``
    on permanent failure.
    """
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
        storage_class = head.get("StorageClass", "STANDARD")
        restore_status = head.get("Restore", "")
        if storage_class in ("GLACIER", "DEEP_ARCHIVE") and 'ongoing-request="false"' not in restore_status:
            return AudioResolutionError(
                session_id="",
                archive_id="",
                file_parent_dir="",
                error_type="glacier_storage_class",
                detail=f"s3://{bucket}/{key} is in {storage_class}",
            )
    except Exception as exc:
        LOGGER.warning("head_object failed for %s: %s", key, exc)

    suffix = Path(key).suffix or ".wav"
    fd, tmp_path_str = tempfile.mkstemp(suffix=suffix, dir=tmp_dir)
    os.close(fd)
    tmp_path = Path(tmp_path_str)

    last_exc: Exception | None = None
    for attempt in range(_MAX_DOWNLOAD_RETRIES):
        try:
            s3_client.download_file(bucket, key, str(tmp_path))
            return tmp_path
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_DOWNLOAD_RETRIES - 1:
                wait = _INITIAL_BACKOFF_SEC * (2 ** attempt)
                LOGGER.warning(
                    "Download attempt %d/%d failed for %s: %s (retry in %.1fs)",
                    attempt + 1, _MAX_DOWNLOAD_RETRIES, key, exc, wait,
                )
                time.sleep(wait)

    tmp_path.unlink(missing_ok=True)
    return AudioResolutionError(
        session_id="",
        archive_id="",
        file_parent_dir="",
        error_type="download_failed",
        detail=f"s3://{bucket}/{key}: {last_exc}",
    )


def resolve_and_download(
    session_id: str,
    archive_id: str,
    file_parent_dir: str,
    s3_client=None,
    bucket: str = BUCKET,
    tmp_dir: str | None = None,
) -> tuple[Path, str] | AudioResolutionError:
    """Resolve the best audio key and download it.

    Returns ``(local_path, s3_key)`` on success, or an
    ``AudioResolutionError`` populated with entity identifiers.
    """
    if s3_client is None:
        s3_client = _get_s3_client()

    result = resolve_audio_key(s3_client, file_parent_dir, bucket=bucket)
    if isinstance(result, AudioResolutionError):
        return AudioResolutionError(
            session_id=session_id,
            archive_id=archive_id,
            file_parent_dir=file_parent_dir,
            error_type=result.error_type,
            detail=result.detail,
        )

    chosen_key = result
    dl_result = download_audio(s3_client, chosen_key, bucket=bucket, tmp_dir=tmp_dir)
    if isinstance(dl_result, AudioResolutionError):
        return AudioResolutionError(
            session_id=session_id,
            archive_id=archive_id,
            file_parent_dir=file_parent_dir,
            error_type=dl_result.error_type,
            detail=dl_result.detail,
        )

    return dl_result, chosen_key
