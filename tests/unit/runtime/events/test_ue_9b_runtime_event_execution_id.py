# © Artur Czarnecki. All rights reserved.

"""UE-9B — RuntimeEvent binds to active Execution identity."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    require_active_execution_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from testing_support.runtime_events import runtime_event_test_identity
from intergrax.runtime.events.runtime_event_identity import require_bound_runtime_event_identity
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.nexus.retry.coordinator import RetryCoordinator
from intergrax.runtime.nexus.retry.retry_engine import RetryRecord
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUNTIME_EVENT_MODULE = _REPO_ROOT / "intergrax" / "runtime" / "events" / "runtime_event.py"


def _bind(
    *,
    run_id: str | None = None,
    attempt_id: str | None = None,
    execution_id: str | None = None,
    parent_execution_id: str | None = None,
):
    return bind_active_execution_identity(
        run_id=run_id or mint_run_id(),
        attempt_id=attempt_id or mint_attempt_id(),
        execution_id=execution_id or mint_execution_id(),
        parent_execution_id=parent_execution_id,
    )


def _event_for_active(task_id: str | None = None) -> RuntimeEvent:
    resolved_task_id, resolved_run_id, attempt_id, execution_id = require_bound_runtime_event_identity(
        task_id=task_id or mint_task_id(),
    )
    return RuntimeEvent(
        task_id=resolved_task_id,
        run_id=resolved_run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        event_type=RuntimeEventType.STEP_STARTED,
        phase=ExecutionPhase.STEP_EXECUTION,
    )


@pytest.mark.asyncio
async def test_root_execution_event_execution_id() -> None:
    root_id = mint_execution_id()
    token = _bind(execution_id=root_id)
    try:
        event = _event_for_active()
        assert event.execution_id == root_id
    finally:
        reset_active_execution_identity(token)


@pytest.mark.asyncio
async def test_child_execution_event_execution_id() -> None:
    root_id = mint_execution_id()
    child_id = mint_execution_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root_token = _bind(run_id=run_id, attempt_id=attempt_id, execution_id=root_id)
    child_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=child_id,
        parent_execution_id=root_id,
    )
    try:
        event = _event_for_active()
        assert event.execution_id == child_id
        assert event.execution_id != root_id
    finally:
        reset_active_execution_identity(child_token)
        reset_active_execution_identity(root_token)


@pytest.mark.asyncio
async def test_nested_child_execution_event_execution_id() -> None:
    root_id = mint_execution_id()
    child_id = mint_execution_id()
    nested_id = mint_execution_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root_token = _bind(run_id=run_id, attempt_id=attempt_id, execution_id=root_id)
    child_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=child_id,
        parent_execution_id=root_id,
    )
    nested_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=nested_id,
        parent_execution_id=child_id,
    )
    try:
        event = _event_for_active()
        assert event.execution_id == nested_id
    finally:
        reset_active_execution_identity(nested_token)
        reset_active_execution_identity(child_token)
        reset_active_execution_identity(root_token)


@pytest.mark.asyncio
async def test_parallel_child_executions_do_not_mix_execution_id() -> None:
    import asyncio

    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root_id = mint_execution_id()
    child_a = mint_execution_id()
    child_b = mint_execution_id()
    captured: dict[str, str] = {}

    async def _capture(label: str, execution_id: str) -> None:
        token = bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            parent_execution_id=root_id,
        )
        try:
            await asyncio.sleep(0)
            captured[label] = str(require_active_execution_id())
            event = _event_for_active()
            assert event.execution_id == execution_id
        finally:
            reset_active_execution_identity(token)

    root_token = _bind(run_id=run_id, attempt_id=attempt_id, execution_id=root_id)
    try:
        await asyncio.gather(_capture("a", child_a), _capture("b", child_b))
    finally:
        reset_active_execution_identity(root_token)
    assert captured["a"] == child_a
    assert captured["b"] == child_b
    assert captured["a"] != captured["b"]


def test_local_retry_preserves_execution_id() -> None:
    execution_id = mint_execution_id()
    token = _bind(execution_id=execution_id)
    try:
        first = _event_for_active()
        second = _event_for_active()
        assert first.execution_id == execution_id
        assert second.execution_id == execution_id
    finally:
        reset_active_execution_identity(token)


def test_redelivery_uses_new_execution_id() -> None:
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    execution_e1 = mint_execution_id()
    execution_e2 = mint_execution_id()
    task = Task(tenant_id="t", user_id="u", context=TaskContext())

    token_a1 = _bind(run_id=run_id, attempt_id=attempt_a1, execution_id=execution_e1)
    try:
        event_a1 = _event_for_active(task_id=task.task_id)
    finally:
        reset_active_execution_identity(token_a1)

    token_a2 = _bind(run_id=run_id, attempt_id=attempt_a2, execution_id=execution_e2)
    try:
        event_a2 = _event_for_active(task_id=task.task_id)
    finally:
        reset_active_execution_identity(token_a2)

    assert event_a1.attempt_id == attempt_a1
    assert event_a2.attempt_id == attempt_a2
    assert event_a1.execution_id == execution_e1
    assert event_a2.execution_id == execution_e2
    assert event_a1.execution_id != event_a2.execution_id

    coordinator = RetryCoordinator(max_run_retries=1, retry_run_on=frozenset())
    token_scheduled = _bind(run_id=run_id, attempt_id=attempt_a1, execution_id=execution_e1)
    try:
        scheduled = coordinator.scheduled_event_for_agent_retry(
            task,
            run_id=run_id,
            attempt_id=attempt_a1,
            record=RetryRecord(
                attempt=1,
                agent_id="a1",
                reason="validation_failed",
                alternate_agent_id="a2",
            ),
        )
        assert scheduled.execution_id == execution_e1
    finally:
        reset_active_execution_identity(token_scheduled)


def test_runtime_event_model_has_execution_id_field() -> None:
    assert "execution_id" in RuntimeEvent.model_fields


def test_runtime_event_module_does_not_mint_execution_id() -> None:
    tree = ast.parse(_RUNTIME_EVENT_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "mint_execution_id":
                pytest.fail("RuntimeEvent module must not mint execution_id")


def test_canonical_runtime_event_without_active_execution_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="active execution identity required"):
        require_bound_runtime_event_identity(task_id=mint_task_id())


def test_runtime_event_v2_requires_execution_id() -> None:
    with pytest.raises(ValueError, match="execution_id is required"):
        RuntimeEvent(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            event_type=RuntimeEventType.STEP_STARTED,
            phase=ExecutionPhase.STEP_EXECUTION,
            schema_version="runtime_event.v2",
        )


def test_runtime_event_serialization_roundtrip_preserves_execution_id() -> None:
    execution_id = mint_execution_id()
    token = _bind(execution_id=execution_id)
    try:
        event = _event_for_active()
        restored = RuntimeEvent.model_validate(event.model_dump(mode="json"))
        assert restored.execution_id == execution_id
        assert restored.schema_version == "runtime_event.v2"
    finally:
        reset_active_execution_identity(token)


def test_with_parent_preserves_child_execution_id() -> None:
    parent_execution = mint_execution_id()
    child_execution = mint_execution_id()
    parent_token = _bind(execution_id=parent_execution)
    try:
        parent = _event_for_active()
    finally:
        reset_active_execution_identity(parent_token)

    child_token = _bind(execution_id=child_execution)
    try:
        child = _event_for_active().with_parent(parent)
        assert child.parent_event_id == parent.event_id
        assert child.execution_id == child_execution
        assert child.execution_id != parent.execution_id
    finally:
        reset_active_execution_identity(child_token)


def test_runtime_event_v1_read_without_execution_id_still_loads() -> None:
    event = RuntimeEvent(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        event_type=RuntimeEventType.STEP_STARTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        schema_version="runtime_event.v1",
        execution_id=None,
    )
    assert event.execution_id is None
    assert event.schema_version == "runtime_event.v1"
