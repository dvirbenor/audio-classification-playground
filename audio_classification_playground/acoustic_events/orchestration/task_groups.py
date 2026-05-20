"""Task-group definitions for orchestration workers."""
from __future__ import annotations

from dataclasses import dataclass


TASK_GROUP_ALL = "all"
TASK_GROUP_AFFECT = "affect"
TASK_GROUP_DISFLUENCY = "disfluency"
TASK_GROUP_EMOTION_VAD = "emotion-vad"

COMPLETION_POLICY_EXISTS = "exists"
COMPLETION_POLICY_CONFIG = "config"


@dataclass(frozen=True)
class TaskGroup:
    name: str
    tasks: tuple[str, ...]
    models: tuple[str, ...]
    lock_namespace: str | None
    precompute_vad: bool
    prefetch_lookahead: int
    prefetch_workers: int
    vad_prefetch_workers: int


TASK_GROUPS: dict[str, TaskGroup] = {
    TASK_GROUP_ALL: TaskGroup(
        name=TASK_GROUP_ALL,
        tasks=("vad", "affect", "disfluency", "emotion"),
        models=("affect", "disfluency", "emotion"),
        lock_namespace=None,
        precompute_vad=True,
        prefetch_lookahead=4,
        prefetch_workers=4,
        vad_prefetch_workers=1,
    ),
    TASK_GROUP_AFFECT: TaskGroup(
        name=TASK_GROUP_AFFECT,
        tasks=("affect",),
        models=("affect",),
        lock_namespace=TASK_GROUP_AFFECT,
        precompute_vad=False,
        prefetch_lookahead=8,
        prefetch_workers=8,
        vad_prefetch_workers=0,
    ),
    TASK_GROUP_DISFLUENCY: TaskGroup(
        name=TASK_GROUP_DISFLUENCY,
        tasks=("disfluency",),
        models=("disfluency",),
        lock_namespace=TASK_GROUP_DISFLUENCY,
        precompute_vad=False,
        prefetch_lookahead=8,
        prefetch_workers=8,
        vad_prefetch_workers=0,
    ),
    TASK_GROUP_EMOTION_VAD: TaskGroup(
        name=TASK_GROUP_EMOTION_VAD,
        tasks=("vad", "emotion"),
        models=("emotion",),
        lock_namespace=TASK_GROUP_EMOTION_VAD,
        precompute_vad=True,
        prefetch_lookahead=12,
        prefetch_workers=8,
        vad_prefetch_workers=1,
    ),
}


def resolve_task_group(name: str) -> TaskGroup:
    try:
        return TASK_GROUPS[name]
    except KeyError as exc:
        raise ValueError(
            f"task_group must be one of: {', '.join(sorted(TASK_GROUPS))}"
        ) from exc


def task_group_choices() -> tuple[str, ...]:
    return tuple(TASK_GROUPS)

