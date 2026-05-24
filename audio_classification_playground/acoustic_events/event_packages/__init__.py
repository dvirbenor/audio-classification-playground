"""Atomic event packages for downstream transcript decoration.

Event packages are decisive, compact event collections.  They are distinct
from review packages: no labels file, no track arrays, no human review state.
"""
from __future__ import annotations

from .eventify import (
    EventPackageConfigs,
    EventifyResult,
    EventifyStatus,
    eventify_archive,
)
from .package import (
    EVENT_PACKAGE_SCHEMA,
    EVENTS_JSONL,
    PACKAGE_JSON,
    EventPackage,
    append_completion_row,
    build_event_package_payload,
    compact_completion_index,
    event_package_complete,
    load_event_package,
)

__all__ = [
    "EVENT_PACKAGE_SCHEMA",
    "EVENTS_JSONL",
    "PACKAGE_JSON",
    "EventPackage",
    "EventPackageConfigs",
    "EventifyResult",
    "EventifyStatus",
    "append_completion_row",
    "build_event_package_payload",
    "compact_completion_index",
    "event_package_complete",
    "eventify_archive",
    "load_event_package",
]
