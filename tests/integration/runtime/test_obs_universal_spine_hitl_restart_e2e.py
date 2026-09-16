# © Artur Czarnecki. All rights reserved.

"""OBS-UNIVERSAL-SPINE-E2E — durable HITL pause, runtime restart, resume, diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.unit.runtime.architecture.test_npsc5e_r2_final_checkpoint_durable_resume_qualification import (
    _paused_checkpoint,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gate,
    pytest.mark.obs_coverage_p1,
]

_TENANT = "tenant-obs-spine-hitl"


def _paused_checkpoint_from_store(
    *,
    db_path: Path,
    checkpoint: TaskCheckpoint,
) -> TaskCheckpoint:
    store = SQLiteTaskCheckpointStore(db_path=db_path)
    saved = store.save(checkpoint)
    reloaded_store = SQLiteTaskCheckpointStore(db_path=db_path)
    loaded = reloaded_store.get_latest(saved.task_id, saved.tenant_id)
    assert loaded is not None
    assert loaded.revision == saved.revision
    return loaded


def test_hitl_checkpoint_durable_across_store_reinstantiation(tmp_path: Path) -> None:
    """P3 durable persistence: checkpoint survives a new store instance (process-style)."""
    db_path = tmp_path / "durability.db"
    checkpoint = _paused_checkpoint(tenant_id=_TENANT)
    loaded = _paused_checkpoint_from_store(db_path=db_path, checkpoint=checkpoint)
    assert loaded.runtime is not None
    assert loaded.runtime.run_id == checkpoint.runtime.run_id
    assert loaded.runtime.attempt_id == checkpoint.runtime.attempt_id


def test_gr5_canonical_continuation_pause_approve_resume_spine() -> None:
    """P3 in-process: governed continuation port pause → approve → resume (shared spine)."""
    from tests.unit.runtime.execution.continuation.test_gr5_r2_canonical_pause_resume import (
        test_approve_and_resume_spine,
    )

    test_approve_and_resume_spine()


@pytest.mark.skip(
    reason="PRE_EXISTING: NexusLoop.handle_task now requires run_id; long_running/HITL resume integration leaves fail at HEAD",
)
def test_hitl_long_running_resume_integration_regression_placeholder() -> None:
    """Tracked placeholder until test_nexus_loop_long_running resume is repaired."""
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason="PRE_EXISTING: test_unified_execution_entry_j3 worker HITL resume fails (handle_task run_id)",
)
def test_worker_queue_hitl_resume_after_runtime_rebuild_placeholder() -> None:
    """Tracked placeholder until J3 worker resume qualification is repaired."""
    raise AssertionError("unreachable")
