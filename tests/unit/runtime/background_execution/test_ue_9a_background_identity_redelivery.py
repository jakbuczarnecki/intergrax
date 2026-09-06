# © Artur Czarnecki. All rights reserved.

"""UE-9A — canonical identity across background execution and redelivery."""

from __future__ import annotations

from dataclasses import dataclass

from unittest.mock import patch

import pytest

from echo.echo_agent import EchoAgent
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    peek_active_execution_id,
    peek_active_execution_identity,
    require_active_execution_id,
    require_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.queueing.worker.execution import execute_logical_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import (
    BackgroundExecutionIdentity,
    bootstrap_background_execution,
)
from intergrax.runtime.background_execution.identity_admission import (
    BackgroundExecutionIdentityMismatchError,
)
from intergrax.runtime.background_execution.identity_persistence import (
    KvBackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.native_planner_action_context import NativePlannerRound
from intergrax.runtime.nexus.tools.tool_loop import (
    execute_planned_tool_calls,
    run_bounded_tool_loop,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.nexus_worker_execution import NexusWorkerRuntime
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import task_to_execution_payload
from intergrax.runtime.task.worker_payload import encode_execution_request
from intergrax.tools.core.tool_plan import PlannedToolCall
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.tools.core.tool_plan import ToolCallPlan
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision
from intergrax.tools.registry import ToolRegistry

from pydantic import BaseModel
from testing_support.builder import build_runtime_state_for_tests, tools_agent_make_contract

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _runtime_state(*, run_id: RunId) -> RuntimeState:
    return build_runtime_state_for_tests(run_id=str(run_id))


def _bind_canonical_context(
    *,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
) -> tuple[object, object]:
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    budget_token = bind_root_execution_budget(
        execution_id=execution_id,
        ledger=create_execution_budget_ledger(None),
    )
    return identity_token, budget_token


def _reset_canonical_context(identity_token: object, budget_token: object) -> None:
    reset_active_execution_budget(budget_token)
    reset_active_execution_identity(identity_token)


class _RecordingInvoker(RuntimeToolInvoker):
    def __init__(self) -> None:
        registry = ToolRegistry()
        registry.register(
            tools_agent_make_contract("probe.read", _In, _Out),
            _ReadHandler(),
        )
        super().__init__(
            registry=registry,
            executor=RegistryToolExecutor(registry=registry),
        )
        self.attempt_ids: list[AttemptId] = []

    def invoke(
        self,
        state: RuntimeState,
        request: ToolExecutionRequest[BaseModel],
        agent_id: str,
    ) -> ToolExecutionResult[BaseModel]:
        _ = state, request, agent_id
        _, active_attempt_id = require_active_execution_identity()
        self.attempt_ids.append(active_attempt_id)
        return ToolExecutionResult.ok(_Out(result=1))


class _TwoRoundPlanner:
    def __init__(self) -> None:
        self._round = 0

    def plan_tools(self, input_data, context=None, *, run_id, allowed_tool_ids=None):
        _ = input_data, context, run_id, allowed_tool_ids
        return ToolPlanDecision(final_answer=None, tool_plan=None, messages=[])

    def plan_native_round(
        self,
        messages: list[ChatMessage],
        *,
        allowed_tool_ids=None,
        run_id: str,
        tool_choice=None,
        protocol_config=None,
        **kwargs,
    ) -> NativePlannerRound:
        _ = messages, allowed_tool_ids, run_id, tool_choice, protocol_config, kwargs
        self._round += 1
        if self._round == 1:
            business_call = LLMToolCall.from_openai_shape(
                call_id="tc-1",
                name="probe.read",
                arguments={"value": 1},
            )
            return NativePlannerRound(
                response=LLMAdapterResponse(content="round-1", tool_calls=(business_call,)),
                business_tool_calls=(business_call,),
                tool_plan=ToolCallPlan(calls=[]),
                action_context=None,
            )
        return NativePlannerRound(
            response=LLMAdapterResponse(content="done", tool_calls=()),
            business_tool_calls=(),
            tool_plan=ToolCallPlan(calls=[]),
            action_context=None,
        )


class _KV(DistributedKVStore):
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


def _transport(
    *,
    tenant_id: str = "tenant-a",
    provider: str = "celery",
    transport_task_id: str = "transport-ue9a",
) -> BackgroundTransportExecutionRef:
    return BackgroundTransportExecutionRef(
        tenant_id=tenant_id,
        provider=provider,
        transport_task_id=transport_task_id,
    )


def _persistence(kv: _KV | None = None) -> KvBackgroundExecutionIdentityPersistence:
    return KvBackgroundExecutionIdentityPersistence(kv or _KV())


@dataclass(frozen=True, slots=True)
class _AttemptObservation:
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId


def _run_nexus_worker_attempt(
    *,
    persistence: KvBackgroundExecutionIdentityPersistence,
    transport: BackgroundTransportExecutionRef,
    capture: list[_AttemptObservation],
) -> BackgroundExecutionIdentity:
    identity = bootstrap_background_execution(
        transport_ref=transport,
        identity_persistence=persistence,
    )
    agent_registry = AgentRegistry()
    agent_registry.register(EchoAgent())
    runtime = NexusWorkerRuntime.from_registry(agent_registry)
    task = Task(
        task_id=str(identity.task_id),
        tenant_id=identity.tenant_id,
        user_id="user-1",
        message="ue-9a",
        context=TaskContext(capability="echo.basic"),
    )
    request = ExecutionRequest(
        run_id="queue-correlation-only",
        tenant_id=identity.tenant_id,
        user_id="user-1",
        input_payload=task_to_execution_payload(task),
    )
    payload = encode_execution_request(request)

    original_bind = bind_active_execution_identity

    def _capturing_bind(**kwargs: object) -> object:
        if kwargs.get("parent_execution_id") is not None:
            return original_bind(**kwargs)
        token = original_bind(**kwargs)
        capture.append(
            _AttemptObservation(
                task_id=identity.task_id,
                run_id=kwargs["run_id"],
                attempt_id=kwargs["attempt_id"],
                execution_id=kwargs["execution_id"],
            )
        )
        return token

    with patch(
        "intergrax.runtime.execution.boundary.bind_active_execution_identity",
        side_effect=_capturing_bind,
    ):
        runtime.execute_payload(
            payload,
            tenant_id=identity.tenant_id,
            run_id=str(identity.run_id),
            execution_identity=identity,
        )
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None
    return identity


def test_first_execution_mints_run_attempt_and_root_execution() -> None:
    persistence = _persistence()
    transport = _transport()
    observations: list[_AttemptObservation] = []

    identity = _run_nexus_worker_attempt(
        persistence=persistence,
        transport=transport,
        capture=observations,
    )

    assert len(observations) == 1
    obs = observations[0]
    assert obs.task_id == identity.task_id
    assert obs.run_id == identity.run_id
    assert obs.attempt_id == identity.attempt_id
    assert str(obs.execution_id).startswith("exec_")


def test_redelivery_preserves_run_task_and_attempt_but_mints_new_execution() -> None:
    persistence = _persistence()
    transport = _transport()
    observations: list[_AttemptObservation] = []

    first_identity = _run_nexus_worker_attempt(
        persistence=persistence,
        transport=transport,
        capture=observations,
    )
    second_identity = _run_nexus_worker_attempt(
        persistence=persistence,
        transport=transport,
        capture=observations,
    )

    assert len(observations) == 2
    first, second = observations
    assert second.task_id == first.task_id == first_identity.task_id == second_identity.task_id
    assert second.run_id == first.run_id == first_identity.run_id == second_identity.run_id
    assert second.attempt_id == first.attempt_id == first_identity.attempt_id == second_identity.attempt_id
    assert second.execution_id != first.execution_id


def test_task_id_unchanged_across_redelivery() -> None:
    persistence = _persistence()
    transport = _transport()
    first = bootstrap_background_execution(
        transport_ref=transport,
        identity_persistence=persistence,
    )
    second = bootstrap_background_execution(
        transport_ref=transport,
        identity_persistence=persistence,
    )
    third = bootstrap_background_execution(
        transport_ref=transport,
        identity_persistence=persistence,
    )

    assert second.task_id == first.task_id == third.task_id


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


class _ReadHandler:
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        return _Out(result=request.input.value)


def _bind_identity(
    *,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
) -> object:
    return bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )


def test_llm_retry_does_not_mint_new_attempt() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    identity_token, budget_token = _bind_canonical_context(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        state = _runtime_state(run_id=run_id)
        invoker = _RecordingInvoker()
        run_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=_TwoRoundPlanner(),
            planner_input=[ChatMessage(role="user", content="iterate")],
            allowed_tool_ids=("probe.read",),
            max_iterations=2,
        )
        active_run_id, active_attempt_id = require_active_execution_identity()
        assert active_run_id == run_id
        assert active_attempt_id == attempt_id
    finally:
        _reset_canonical_context(identity_token, budget_token)


def test_tool_retry_does_not_mint_new_attempt() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    identity_token, budget_token = _bind_canonical_context(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        state = _runtime_state(run_id=run_id)
        invoker = _RecordingInvoker()
        calls = [PlannedToolCall(step_id="s1", tool_id="probe.read", input=_In(value=1))]
        execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="retry-1",
        )
        execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="retry-2",
        )
        assert invoker.attempt_ids == [attempt_id, attempt_id]
        active_run_id, active_attempt_id = require_active_execution_identity()
        assert active_run_id == run_id
        assert active_attempt_id == attempt_id
    finally:
        _reset_canonical_context(identity_token, budget_token)


@dataclass(frozen=True)
class _ChildPing:
    value: str


@dataclass(frozen=True)
class _ChildPong:
    value: str


class _ChildDelegate:
    async def execute(self, request: _ChildPing) -> _ChildPong:
        return _ChildPong(value=request.value)


@pytest.mark.asyncio
async def test_child_execution_does_not_mint_new_attempt() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    ledger = create_execution_budget_ledger(RunBudget())
    child_runner = ChildExecutionRunner[_ChildPing, _ChildPong](ledger=ledger)

    class _RootDelegate:
        async def execute(self, request: _ChildPing) -> _ChildPong:
            return await child_runner.execute(request=request, delegate=_ChildDelegate())

    token = _bind_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        await ExecutionBoundary[_ChildPing, _ChildPong](
            _RootDelegate(),
            identity=ExecutionIdentityBinding(
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            ),
            authority=ParentExecutionAuthority.scoped(("read",)),
        ).execute(_ChildPing(value="child"))
        active_run_id, active_attempt_id = require_active_execution_identity()
        assert active_run_id == run_id
        assert active_attempt_id == attempt_id
    finally:
        reset_active_execution_identity(token)


def test_three_consecutive_redeliveries_keep_run_and_attempt() -> None:
    persistence = _persistence()
    transport = _transport(transport_task_id="transport-redelivery-chain")
    attempts: list[AttemptId] = []

    for _ in range(3):
        identity = bootstrap_background_execution(
            transport_ref=transport,
            identity_persistence=persistence,
        )
        attempts.append(identity.attempt_id)

    assert attempts[0] == attempts[1] == attempts[2]
    second = bootstrap_background_execution(
        transport_ref=transport,
        identity_persistence=persistence,
    )
    first_run = persistence.resolve_or_create(transport)
    assert second.run_id == first_run.run_id
    assert second.attempt_id == first_run.attempt_id


def test_parallel_executions_in_same_attempt_share_attempt_id() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    identity_token, budget_token = _bind_canonical_context(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        state = _runtime_state(run_id=run_id)
        invoker = _RecordingInvoker()
        calls = [
            PlannedToolCall(
                step_id=f"s{index}",
                tool_id="probe.read",
                input=_In(value=index),
            )
            for index in range(3)
        ]
        execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix="parallel",
            max_parallel_read_only=3,
        )
        assert len(invoker.attempt_ids) == 3
        assert set(invoker.attempt_ids) == {attempt_id}
    finally:
        _reset_canonical_context(identity_token, budget_token)


def test_worker_does_not_mint_new_run_id_on_redelivery() -> None:
    kv = _KV()
    persistence = KvBackgroundExecutionIdentityPersistence(kv)
    transport = _transport(transport_task_id="stable-run")

    first = bootstrap_background_execution(
        transport_ref=transport,
        identity_persistence=persistence,
    )
    second = bootstrap_background_execution(
        transport_ref=transport,
        identity_persistence=persistence,
    )

    assert second.run_id == first.run_id
    stored = kv.get("tenant-a", "bg_exec_identity:celery:stable-run")
    assert stored is not None
    assert str(first.run_id).encode("utf-8") in stored


def test_inconsistent_payload_run_id_fails_closed() -> None:
    persistence = _persistence()
    identity = bootstrap_background_execution(
        transport_ref=_transport(transport_task_id="inconsistent"),
        identity_persistence=persistence,
    )
    agent_registry = AgentRegistry()
    agent_registry.register(EchoAgent())
    runtime = NexusWorkerRuntime.from_registry(agent_registry)
    conflicting_run = mint_run_id()
    task = Task(
        task_id=str(identity.task_id),
        tenant_id=identity.tenant_id,
        user_id="user-1",
        message="conflict",
        context=TaskContext(capability="echo.basic"),
    )
    request = ExecutionRequest(
        run_id=str(conflicting_run),
        tenant_id=identity.tenant_id,
        user_id="user-1",
        input_payload=task_to_execution_payload(task),
    )
    payload = encode_execution_request(request)

    with pytest.raises(BackgroundExecutionIdentityMismatchError, match="payload run_id conflicts"):
        runtime.execute_payload(
            payload,
            tenant_id=identity.tenant_id,
            run_id=str(identity.run_id),
            execution_identity=identity,
        )


def test_inconsistent_handler_run_id_fails_closed() -> None:
    registry = TaskExecutionRegistry()
    persistence = _persistence()
    identity = bootstrap_background_execution(
        transport_ref=_transport(transport_task_id="handler-conflict"),
        identity_persistence=persistence,
    )

    class _Output(BaseModel):
        value: str = "ok"

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: str | None,
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[_Output]:
        _ = tenant_id, payload, idempotency_key, execution_identity
        return ToolExecutionResult.ok(_Output())

    registry.register("demo.task.v1", handler)

    with pytest.raises(BackgroundExecutionIdentityMismatchError, match="handler run_id"):
        execute_logical_task(
            registry=registry,
            logical_task_name="demo.task.v1",
            tenant_id=identity.tenant_id,
            run_id=str(mint_run_id()),
            payload=b"{}",
            idempotency_key=None,
            idempotency_store=None,
            execution_identity=identity,
        )


def test_contextvar_identity_does_not_leak_between_attempts() -> None:
    persistence = _persistence()
    transport = _transport(transport_task_id="ctx-leak")
    observations: list[_AttemptObservation] = []

    _run_nexus_worker_attempt(
        persistence=persistence,
        transport=transport,
        capture=observations,
    )
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None

    _run_nexus_worker_attempt(
        persistence=persistence,
        transport=transport,
        capture=observations,
    )
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None
    assert observations[1].attempt_id == observations[0].attempt_id


def test_queue_correlation_run_id_is_not_treated_as_canonical_conflict() -> None:
    persistence = _persistence()
    identity = bootstrap_background_execution(
        transport_ref=_transport(transport_task_id="queue-correlation"),
        identity_persistence=persistence,
    )
    agent_registry = AgentRegistry()
    agent_registry.register(EchoAgent())
    runtime = NexusWorkerRuntime.from_registry(agent_registry)
    task = Task(
        task_id=str(identity.task_id),
        tenant_id=identity.tenant_id,
        user_id="user-1",
        message="ok",
        context=TaskContext(capability="echo.basic"),
    )
    request = ExecutionRequest(
        run_id="queue-correlation-not-canonical",
        tenant_id=identity.tenant_id,
        user_id="user-1",
        input_payload=task_to_execution_payload(task),
    )
    payload = encode_execution_request(request)

    result_payload = runtime.execute_payload(
        payload,
        tenant_id=identity.tenant_id,
        run_id=str(identity.run_id),
        execution_identity=identity,
    )

    assert result_payload["run_id"] == str(identity.run_id)
