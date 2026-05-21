"""Batch orchestration pipeline for large-scale acoustic inference.

See ``orchestration/README.md`` for architecture overview, usage, and
deployment instructions.
"""
from .manifest import ArchiveEntity, load_manifest
from .progress import (
    ProgressSummary,
    QuickSummary,
    completed_tasks_for_entity_keys,
    is_archive_complete,
    quick_disk_summary,
    scan_progress,
)
from .worker import run_worker

__all__ = [
    "ArchiveEntity",
    "ProgressSummary",
    "QuickSummary",
    "completed_tasks_for_entity_keys",
    "is_archive_complete",
    "load_manifest",
    "quick_disk_summary",
    "run_worker",
    "scan_progress",
]
