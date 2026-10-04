# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R4-R1 shared qualification helpers (provider validation parity)."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

from intergrax.runtime.long_running.scheduled_resume import (
    ScheduledResume,
    ScheduledResumePersistence,
    validate_scheduled_resume_for_persistence,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.state_x._r3_r4_support import MemoryScheduleStore, due_run_at

_REPO_ROOT = Path(__file__).resolve().parents[3]

FORBIDDEN_METADATA_PARITY_CASES: Tuple[Tuple[str, object], ...] = (
    ("human_approved", True),
    ("verdict", "approved"),
    ("run_id", "run-override"),
    ("tenant_id", "tenant-evil"),
    ("authorization_token", "secret"),
)


def scheduled_resume_persistence_implementations() -> List[Tuple[str, type]]:
    """Closed-world inventory of concrete ScheduledResumePersistence schedule() owners."""
    from tests.qualification.sched_01.test_sched_01_gates import _MemoryScheduleStore

    return [
        ("SQLiteTaskCheckpointStore", SQLiteTaskCheckpointStore),
        ("MemoryScheduleStore", MemoryScheduleStore),
        ("_MemoryScheduleStore", _MemoryScheduleStore),
    ]


def build_valid_scheduled_resume(**overrides: object) -> ScheduledResume:
    base = dict(
        schedule_id="sched_r1_valid",
        task_id="t1",
        tenant_id="tenant-a",
        resume_token="tok",
        run_at_utc=due_run_at(),
    )
    base.update(overrides)
    return ScheduledResume(**base)  # type: ignore[arg-type]


def post_construction_invalid_copy(valid: ScheduledResume, metadata: dict[str, object]) -> ScheduledResume:
    return valid.model_copy(update={"resume_metadata": metadata})


def parity_provider_factories(tmp_path: Path) -> Iterable[Tuple[str, Callable[[], ScheduledResumePersistence]]]:
    db = tmp_path / "r1parity.db"

    def sqlite_factory() -> ScheduledResumePersistence:
        return SQLiteTaskCheckpointStore(db_path=db)

    def memory_factory() -> ScheduledResumePersistence:
        return MemoryScheduleStore()

    return (
        ("sqlite", sqlite_factory),
        ("memory", memory_factory),
    )


def schedule_source_uses_canonical_validator(class_type: type) -> bool:
    source = inspect.getsource(class_type.schedule)
    return "validate_scheduled_resume_for_persistence" in source


def canonical_validator_definition_path() -> Path:
    return _REPO_ROOT / "intergrax/runtime/long_running/scheduled_resume.py"
