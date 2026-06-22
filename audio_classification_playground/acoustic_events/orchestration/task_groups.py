"""Task-group definitions for orchestration workers."""
from __future__ import annotations

from dataclasses import dataclass


TASK_GROUP_ALL = "all"
TASK_GROUP_AFFECT = "affect"
TASK_GROUP_DISFLUENCY = "disfluency"
TASK_GROUP_EMOTION = "emotion"
TASK_GROUP_VAD = "vad"
TASK_GROUP_EMOTION_VAD = "emotion-vad"
TASK_GROUP_MODELS = "models"

COMPLETION_POLICY_EXISTS = "exists"
COMPLETION_POLICY_CONFIG = "config"


@dataclass(frozen=True)
class TaskGroup:
    name: str
    tasks: tuple[str, ...]
    models: tuple[str, ...]
    lock_namespaces: tuple[str, ...] | None
    precompute_vad: bool
    prefetch_lookahead: int
    prefetch_workers: int
    vad_prefetch_workers: int


TASK_GROUPS: dict[str, TaskGroup] = {
    TASK_GROUP_ALL: TaskGroup(
        name=TASK_GROUP_ALL,
        tasks=("vad", "affect", "disfluency", "emotion"),
        models=("affect", "disfluency", "emotion"),
        lock_namespaces=None,
        precompute_vad=True,
        prefetch_lookahead=4,
        prefetch_workers=4,
        vad_prefetch_workers=1,
    ),
    TASK_GROUP_AFFECT: TaskGroup(
        name=TASK_GROUP_AFFECT,
        tasks=("affect",),
        models=("affect",),
        lock_namespaces=(TASK_GROUP_AFFECT,),
        precompute_vad=False,
        prefetch_lookahead=8,
        prefetch_workers=8,
        vad_prefetch_workers=0,
    ),
    TASK_GROUP_DISFLUENCY: TaskGroup(
        name=TASK_GROUP_DISFLUENCY,
        tasks=("disfluency",),
        models=("disfluency",),
        lock_namespaces=(TASK_GROUP_DISFLUENCY,),
        precompute_vad=False,
        prefetch_lookahead=8,
        prefetch_workers=8,
        vad_prefetch_workers=0,
    ),
    TASK_GROUP_EMOTION: TaskGroup(
        name=TASK_GROUP_EMOTION,
        tasks=("emotion",),
        models=("emotion",),
        lock_namespaces=(TASK_GROUP_EMOTION,),
        precompute_vad=False,
        prefetch_lookahead=28,
        prefetch_workers=14,
        vad_prefetch_workers=0,
    ),
    TASK_GROUP_VAD: TaskGroup(
        name=TASK_GROUP_VAD,
        tasks=("vad",),
        models=(),
        lock_namespaces=(TASK_GROUP_VAD,),
        precompute_vad=True,
        prefetch_lookahead=24,
        prefetch_workers=12,
        vad_prefetch_workers=1,
    ),
    TASK_GROUP_EMOTION_VAD: TaskGroup(
        name=TASK_GROUP_EMOTION_VAD,
        tasks=("vad", "emotion"),
        models=("emotion",),
        lock_namespaces=(TASK_GROUP_VAD, TASK_GROUP_EMOTION),
        precompute_vad=True,
        prefetch_lookahead=12,
        prefetch_workers=8,
        vad_prefetch_workers=1,
    ),
    # All three model tasks WITHOUT computing VAD (no vad task). For backfilling
    # the model artifacts when a separate VAD fleet has already populated vad/.
    # Pair with --vad-gating --require-precomputed-vad: inference is gated on the
    # EXISTING vad/ artifact (run_all_inference reads intervals_sec from disk),
    # and archives lacking vad/ are skipped. Note require_precomputed_vad is
    # honored here precisely because this group does not produce vad itself
    # (unlike "all"). Whole-archive lock like "all"; pair with
    # --completion-policy exists to skip done archives.
    TASK_GROUP_MODELS: TaskGroup(
        name=TASK_GROUP_MODELS,
        tasks=("affect", "disfluency", "emotion"),
        models=("affect", "disfluency", "emotion"),
        lock_namespaces=None,
        precompute_vad=False,
        prefetch_lookahead=8,
        prefetch_workers=8,
        vad_prefetch_workers=0,
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
