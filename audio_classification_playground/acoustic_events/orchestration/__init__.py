"""Batch orchestration pipeline for large-scale acoustic inference.

See ``orchestration/README.md`` for architecture overview, usage, and
deployment instructions.
"""
from typing import TYPE_CHECKING

from .manifest import ArchiveEntity, load_manifest
from .progress import (
    ProgressSummary,
    QuickSummary,
    completed_tasks_for_entity_keys,
    is_archive_complete,
    quick_disk_summary,
    scan_progress,
)

if TYPE_CHECKING:
    from .worker import run_worker


def __getattr__(name: str):
    # Lazily import ``run_worker`` so that read-only CLI commands
    # (progress/status/timings/errors) and other lightweight consumers don't
    # pull in ``worker`` and its heavy torch/inference dependency chain.
    if name == "run_worker":
        from .worker import run_worker

        return run_worker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
