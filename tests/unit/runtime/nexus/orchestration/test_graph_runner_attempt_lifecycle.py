# © Artur Czarnecki. All rights reserved.

"""GraphRunner durable attempt lifecycle integration (P0C-4)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.attempt_lifecycle import (
    AttemptLifecycleState,
    AttemptLifecycleStore,
    AttemptTransitionReason,
)
from intergrax.contracts.execution_identity import (
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.runtime.execution.attempt_lifecycle import AttemptLifecycleService, InMemoryAttemptLifecycleStore
from intergrax.runtime.nexus.orchestration.graph_runner import NexusGraphRunner
from intergrax.runtime.nexus.response.final_response_composer import FinalResponseComposer
from intergrax.runtime.task.task import Task


class _FailingAttemptLifecycleStore(AttemptLifecycleStore):
    @property
    def is_durable(self) -> bool:
        return False

    def load_raw(self, *, tenant_id: str, run_id: RunId) -> bytes | None:
        return None

    def compare_and_swap(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected: bytes | None,
        new_state: AttemptLifecycleState,
    ) -> bool:
        raise RuntimeError("store down")


def _build_runner(lifecycle_service: AttemptLifecycleService) -> NexusGraphRunner:
    return NexusGraphRunner(
        registry=MagicMock(),
        graph_executor=MagicMock(),
        validation_engine=MagicMock(),
        composer=FinalResponseComposer(),
        hitl=MagicMock(),
        events=MagicMock(),
        finish_task=AsyncMock(),
        finalize_trace=AsyncMock(),
        maybe_checkpoint=AsyncMock(),
        attempt_lifecycle=lifecycle_service,
    )


@pytest.mark.unit
def test_graph_runner_durable_transition_rebinds_context() -> None:
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_a1,
        execution_id=mint_execution_id(),
    )
    task = Task(task_id=mint_task_id(), tenant_id="tenant-a", user_id="user", message="hello")
    lifecycle_service = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    lifecycle_service.record_initial_attempt(
        tenant_id=task.tenant_id,
        run_id=run_id,
        attempt_id=attempt_a1,
    )
    runner = _build_runner(lifecycle_service)
    try:
        new_attempt_id = runner._transition_attempt_for_retry(
            task,
            run_id=run_id,
            expected_attempt_id=attempt_a1,
        )
    finally:
        reset_active_execution_identity(token)

    assert new_attempt_id is not None
    assert new_attempt_id != attempt_a1
    assert lifecycle_service.get_active_attempt_id(tenant_id=task.tenant_id, run_id=run_id) == new_attempt_id


@pytest.mark.unit
def test_graph_runner_transition_failure_leaves_context_unchanged() -> None:
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_a1,
        execution_id=mint_execution_id(),
    )
    task = Task(task_id=mint_task_id(), tenant_id="tenant-a", user_id="user", message="hello")
    runner = _build_runner(AttemptLifecycleService(_FailingAttemptLifecycleStore()))
    try:
        new_attempt_id = runner._transition_attempt_for_retry(
            task,
            run_id=run_id,
            expected_attempt_id=attempt_a1,
        )
        assert new_attempt_id is None
        assert peek_active_execution_identity() == (run_id, attempt_a1)
    finally:
        reset_active_execution_identity(token)
