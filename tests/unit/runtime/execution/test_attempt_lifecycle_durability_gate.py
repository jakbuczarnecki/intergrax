# © Artur Czarnecki. All rights reserved.

"""P0C-4A production durability gate for attempt lifecycle retry transitions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.attempt_lifecycle import AttemptLifecycleError, AttemptTransitionReason
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
    wire_attempt_lifecycle_store,
)
from intergrax.runtime.execution.attempt_lifecycle.durability_policy import (
    DURABLE_ATTEMPT_LIFECYCLE_REQUIRED_MSG,
    validate_durable_attempt_lifecycle_for_composition,
)
from intergrax.runtime.execution.execution_terminal import (
    ExecutionTerminalService,
    InMemoryExecutionTerminalStore,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.orchestration.graph_runner import NexusGraphRunner
from intergrax.runtime.nexus.planning.task_planner import NexusPlan
from intergrax.runtime.nexus.response.final_response_composer import FinalResponseComposer
from intergrax.runtime.nexus.retry.retry_engine import RetryPolicy
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task


class _KvStore(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: bytes | None,
        new_value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        current = self.get(tenant_id, key)
        if expected is None and current is not None:
            return False
        if expected is not None and current != expected:
            return False
        self.set(tenant_id, key, new_value, ttl_seconds=ttl_seconds)
        return True


def _build_runner(
    lifecycle_service: AttemptLifecycleService,
    *,
    production_mode: bool = False,
) -> NexusGraphRunner:
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
        execution_terminal=ExecutionTerminalService(InMemoryExecutionTerminalStore()),
        production_mode=production_mode,
    )


@pytest.mark.unit
def test_composition_rejects_production_retry_with_in_memory_store() -> None:
    with pytest.raises(AttemptLifecycleError, match=DURABLE_ATTEMPT_LIFECYCLE_REQUIRED_MSG):
        NexusLoop(
            AgentRegistry(),
            production_mode=True,
            retry_policy=RetryPolicy(max_retries=1),
        )


@pytest.mark.unit
def test_composition_allows_production_without_retry_capability() -> None:
    loop = NexusLoop(
        AgentRegistry(),
        production_mode=True,
        retry_policy=RetryPolicy(max_retries=0),
        max_run_retries=0,
    )
    assert loop._attempt_lifecycle.store.is_durable is False  # noqa: SLF001


@pytest.mark.unit
def test_composition_allows_dev_retry_with_in_memory_store() -> None:
    loop = NexusLoop(
        AgentRegistry(),
        production_mode=False,
        retry_policy=RetryPolicy(max_retries=2),
        max_run_retries=1,
    )
    assert loop._attempt_lifecycle.store.is_durable is False  # noqa: SLF001


@pytest.mark.unit
def test_composition_allows_production_retry_with_kv_store() -> None:
    store = wire_attempt_lifecycle_store(kv_store=_KvStore())
    loop = NexusLoop(
        AgentRegistry(),
        production_mode=True,
        retry_policy=RetryPolicy(max_retries=1),
        attempt_lifecycle=AttemptLifecycleService(store),
    )
    assert loop._attempt_lifecycle.store.is_durable is True  # noqa: SLF001


@pytest.mark.unit
def test_composition_allows_production_retry_with_document_store() -> None:
    store = wire_attempt_lifecycle_store(document_store=InMemoryDocumentStore())
    loop = NexusLoop(
        AgentRegistry(),
        production_mode=True,
        max_run_retries=1,
        attempt_lifecycle=AttemptLifecycleService(store),
    )
    assert loop._attempt_lifecycle.store.is_durable is True  # noqa: SLF001


@pytest.mark.unit
def test_runtime_gate_rejects_production_transition_with_in_memory_store() -> None:
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
    runner = _build_runner(lifecycle_service, production_mode=True)
    try:
        with pytest.raises(AttemptLifecycleError, match=DURABLE_ATTEMPT_LIFECYCLE_REQUIRED_MSG):
            runner._transition_attempt_for_retry(
                task,
                run_id=run_id,
                expected_attempt_id=attempt_a1,
            )
        assert peek_active_execution_identity() == (run_id, attempt_a1)
        assert lifecycle_service.get_active_attempt_id(
            tenant_id=task.tenant_id,
            run_id=run_id,
        ) == attempt_a1
    finally:
        reset_active_execution_identity(token)


@pytest.mark.unit
def test_dev_runtime_transition_allows_in_memory_store() -> None:
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
    runner = _build_runner(lifecycle_service, production_mode=False)
    try:
        new_attempt_id = runner._transition_attempt_for_retry(
            task,
            run_id=run_id,
            expected_attempt_id=attempt_a1,
        )
        assert new_attempt_id is not None
        assert new_attempt_id != attempt_a1
    finally:
        reset_active_execution_identity(token)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_dynamic_graph_retry_denied_before_executor_rerun() -> None:
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

    graph_executor = MagicMock()
    execute_calls = {"count": 0}

    async def _execute(*_args, **_kwargs):
        execute_calls["count"] += 1
        if execute_calls["count"] == 1:
            on_retry = _kwargs.get("on_retry")
            assert on_retry is not None
            from intergrax.runtime.nexus.retry.retry_types import RetryRecord

            await on_retry(
                RetryRecord(
                    attempt=1,
                    reason="validation_failed",
                    agent_id="agent-a",
                ),
            )
        return ([], [], MagicMock(), False)

    graph_executor.execute = _execute
    graph_executor.execution_identity = MagicMock()
    graph_executor.execution_identity.require.return_value = (run_id, attempt_a1)
    graph_executor.set_retry_policy = MagicMock()

    runner = NexusGraphRunner(
        registry=MagicMock(),
        graph_executor=graph_executor,
        validation_engine=MagicMock(),
        composer=FinalResponseComposer(),
        hitl=MagicMock(),
        events=MagicMock(),
        finish_task=AsyncMock(),
        finalize_trace=AsyncMock(),
        maybe_checkpoint=AsyncMock(),
        attempt_lifecycle=lifecycle_service,
        execution_terminal=ExecutionTerminalService(InMemoryExecutionTerminalStore()),
        production_mode=True,
    )
    runner.events.publish = AsyncMock()
    plan = NexusPlan(task_id=task.task_id, classification="test", graph_retry_on_error=1)

    try:
        with pytest.raises(AttemptLifecycleError, match=DURABLE_ATTEMPT_LIFECYCLE_REQUIRED_MSG):
            await runner.run(
                task,
                plan=plan,
                graph=MagicMock(),
                lifecycle=MagicMock(),
                trace_emitter=MagicMock(),
            )
        assert execute_calls["count"] == 1
        assert peek_active_execution_identity() == (run_id, attempt_a1)
        assert lifecycle_service.get_active_attempt_id(
            tenant_id=task.tenant_id,
            run_id=run_id,
        ) == attempt_a1
    finally:
        reset_active_execution_identity(token)


@pytest.mark.unit
def test_validate_durable_attempt_lifecycle_uses_is_durable_capability() -> None:
    class _CustomDurableStore(InMemoryAttemptLifecycleStore):
        @property
        def is_durable(self) -> bool:
            return True

    validate_durable_attempt_lifecycle_for_composition(
        production_mode=True,
        store=_CustomDurableStore(),
        agent_retry_max=1,
        run_retry_max=0,
    )


@pytest.mark.unit
def test_attempt_lifecycle_service_require_durable_raises_for_non_durable_store() -> None:
    service = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    with pytest.raises(AttemptLifecycleError, match=DURABLE_ATTEMPT_LIFECYCLE_REQUIRED_MSG):
        service.require_durable()
