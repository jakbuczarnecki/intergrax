# © Artur Czarnecki. All rights reserved.

"""OBS-UNIVERSAL-SPINE-E2E — durable HITL pause, runtime restart, resume, diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.diagnostic_read_wiring import build_diagnostic_read_service
from intergrax.runtime.diagnostics.persistence_conformance import query_all_problems_for_tenant
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import TaskState
from testing_support.obs_universal_spine.hitl_restart_harness import (
    run_hitl_pause_resume_after_runtime_rebuild,
)
from tests.integration.applications.test_unified_execution_entry_j3 import (
    test_worker_checkpoint_resume_via_queue_payload,
)
from tests.integration.runtime.test_nexus_loop_long_running import (
    test_long_running_task_resumes_with_token,
)
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


@pytest.mark.asyncio
async def test_hitl_long_running_resume_integration_regression(tmp_path: Path) -> None:
    await test_long_running_task_resumes_with_token(tmp_path)


def test_worker_queue_hitl_resume_after_runtime_rebuild(tmp_path: Path) -> None:
    test_worker_checkpoint_resume_via_queue_payload(tmp_path)


@pytest.mark.asyncio
async def test_hitl_runtime_rebuild_resume_clean_no_false_problem(tmp_path: Path) -> None:
    checkpoint_db = tmp_path / "hitl_ckpt.db"
    events_db = tmp_path / "hitl_events.db"
    before, after, terminal_state, runtime_b = await run_hitl_pause_resume_after_runtime_rebuild(
        checkpoint_db=checkpoint_db,
        runtime_events_db=events_db,
        human_approved=True,
    )
    assert terminal_state is TaskState.COMPLETED
    assert before.tenant_id == after.tenant_id == _TENANT
    assert before.task_id == after.task_id
    assert before.run_id == after.run_id
    assert before.attempt_id == after.attempt_id
    events = runtime_b.runtime_event_store.list_for_run(before.run_id, tenant_id=_TENANT)
    assert any(event.event_type is RuntimeEventType.TASK_COMPLETED for event in events)
    assert query_all_problems_for_tenant(runtime_b.read_deps.problem_persistence, _TENANT) == ()
    read_service = build_diagnostic_read_service(runtime_b.read_deps)
    assert read_service.list_problems(tenant_id=_TENANT).total_count == 0


@pytest.mark.asyncio
async def test_hitl_runtime_rebuild_resume_failure_terminal_evidence(tmp_path: Path) -> None:
    checkpoint_db = tmp_path / "hitl_fail_ckpt.db"
    events_db = tmp_path / "hitl_fail_events.db"
    _, _, terminal_state, _runtime_b = await run_hitl_pause_resume_after_runtime_rebuild(
        checkpoint_db=checkpoint_db,
        runtime_events_db=events_db,
        human_approved=False,
        human_rejected=True,
    )
    assert terminal_state is TaskState.FAILED
