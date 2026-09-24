# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9-R1-R1-R1 — canonical runtime checkpoint before governed pause."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.runtime.long_running.checkpoint_builder import (
    mark_task_runtime_execution_tree_interrupted_for_pause,
    materialize_task_runtime_checkpoint_for_active_execution,
    resolve_task_runtime_checkpoint,
)
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    root_execution_id_from_tree,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointStatus,
)
from intergrax.runtime.long_running.resume_planner import (
    execution_identity_from_checkpoint,
)
from intergrax.runtime.human.agent_governance_pause_projection import (
    TaskAgentGovernancePauseProjectionAdapter,
)
from intergrax.contracts.agent_decision import HumanRequest
from tests.unit.runtime.architecture.test_uca_6c_r6_r5_foundation_hardening_gates import (
    _pending,
    _requirement,
)
from tests.unit.runtime.human.test_agent_governance_grant_lifecycle_h1 import (
    _MemoryTaskCheckpointStore,
    _task,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_materialize_runtime_checkpoint_under_active_identity() -> None:
    task = _task()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        task_id=str(task.task_id),
    )
    try:
        runtime = materialize_task_runtime_checkpoint_for_active_execution(task)
        assert runtime.run_id == run_id
        assert runtime.attempt_id == attempt_id
        assert resolve_task_runtime_checkpoint(task) is not None
        root_id = root_execution_id_from_tree(runtime.execution_tree)
        assert root_id == execution_id
    finally:
        reset_active_execution_identity(token)


def test_mark_interrupted_before_pause_updates_tree() -> None:
    task = _task()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        task_id=str(task.task_id),
    )
    try:
        materialize_task_runtime_checkpoint_for_active_execution(task)
        mark_task_runtime_execution_tree_interrupted_for_pause(task)
        runtime = resolve_task_runtime_checkpoint(task)
        assert runtime is not None
        root = runtime.execution_tree.entry_by_execution_id(execution_id)
        assert root is not None
        assert root.status is ExecutionCheckpointStatus.INTERRUPTED
    finally:
        reset_active_execution_identity(token)


def test_pause_projection_fail_closed_without_runtime() -> None:
    task = _task()
    with pytest.raises(ValueError, match="canonical task runtime checkpoint"):
        mark_task_runtime_execution_tree_interrupted_for_pause(task)


def test_persisted_checkpoint_includes_runtime_after_materialize() -> None:
    task = _task()
    store = _MemoryTaskCheckpointStore()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        task_id=str(task.task_id),
    )
    try:
        materialize_task_runtime_checkpoint_for_active_execution(task)
        mark_task_runtime_execution_tree_interrupted_for_pause(task)
        adapter = TaskAgentGovernancePauseProjectionAdapter(
            task=task,
            checkpoint_store=store,
        )
        pending = _pending(_requirement())
        human_request = HumanRequest(
            request_id=pending.human_request_id,
            prompt="approve?",
        )
        result = adapter.persist_pause_projection(
            pending=pending,
            human_request=human_request,
        )
        assert result.outcome.value == "applied"
        saved = store.get_latest(str(task.task_id), task.tenant_id)
        assert saved is not None
        assert saved.runtime is not None
        saved_run, saved_attempt = execution_identity_from_checkpoint(saved)
        assert saved_run == run_id
        assert saved_attempt == attempt_id
        assert (
            root_execution_id_from_tree(saved.runtime.execution_tree)
            == execution_id
        )
    finally:
        reset_active_execution_identity(token)
