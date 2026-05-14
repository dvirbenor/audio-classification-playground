"""Load and preprocess the archive manifest from a parquet file.

This module reads the input parquet, deduplicates by ``(session_id,
archive_id)``, validates that IDs are safe for filesystem paths, and
returns a list of ``ArchiveEntity`` dataclass instances.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_UNSAFE_ID_PATTERN = re.compile(r"[/\\\x00]")


@dataclass(frozen=True)
class ArchiveEntity:
    """Atomic unit of work: one unique (session_id, archive_id) pair."""

    session_id: str
    archive_id: str
    file_parent_dir: str


def _validate_id(value: str, field: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field} must be a non-empty string, got {value!r}")
    if _UNSAFE_ID_PATTERN.search(value):
        raise ValueError(
            f"{field} contains unsafe path characters: {value!r}"
        )


def load_manifest(parquet_path: str | Path) -> list[ArchiveEntity]:
    """Read the parquet manifest and return a deduplicated entity list.

    Deduplication is by ``(session_id, archive_id)``.  When duplicates exist
    (e.g. across dates), the first ``file_parent_dir`` encountered is kept.
    ``is_deleted`` records are **not** filtered — they are expected to surface
    as inference failures if the underlying audio is inaccessible.
    """
    import pyarrow.parquet as pq

    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest parquet not found: {path}")

    table = pq.read_table(str(path), columns=["session_id", "archive_id", "file_parent_dir"])
    df = table.to_pydict()
    session_ids = df["session_id"]
    archive_ids = df["archive_id"]
    file_parent_dirs = df["file_parent_dir"]

    seen: dict[tuple[str, str], ArchiveEntity] = {}
    skipped = 0
    for sid, aid, fpd in zip(session_ids, archive_ids, file_parent_dirs):
        sid_str = str(sid).strip()
        aid_str = str(aid).strip()
        fpd_str = str(fpd).strip() if fpd else ""
        key = (sid_str, aid_str)
        if key in seen:
            skipped += 1
            continue
        try:
            _validate_id(sid_str, "session_id")
            _validate_id(aid_str, "archive_id")
        except ValueError:
            LOGGER.warning("Skipping invalid entity: session_id=%r archive_id=%r", sid_str, aid_str)
            skipped += 1
            continue
        seen[key] = ArchiveEntity(
            session_id=sid_str,
            archive_id=aid_str,
            file_parent_dir=fpd_str,
        )

    entities = list(seen.values())
    LOGGER.info(
        "Loaded %d unique entities from %d rows (%d duplicates/invalid)",
        len(entities),
        len(session_ids),
        skipped,
    )
    return entities
