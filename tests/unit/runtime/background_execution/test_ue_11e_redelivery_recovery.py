# © Artur Czarnecki. All rights reserved.

"""UE-11E — background redelivery recovery qualification."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    peek_active_execution_id,
    peek_active_execution_identity,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.runtime.background_execution.bootstrap import (
    BackgroundExecutionIdentity,
    bootstrap_background_execution,
)
from intergrax.runtime.background_execution.identity_persistence import (
    KvBackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.runtime.execution.active_execution_budget import (
    peek_active_execution_budget,
    require_active_execution_budget,
)
from intergrax.runtime.execution.budget.consumption import consume_llm_call
from intergrax.runtime.execution.budget.persistence import (
    KvRunBudgetPersistence,
    create_durable_run_budget_ledger_factory,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.nexus_worker_execution import NexusWorkerRuntime
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import task_to_execution_payload
from intergrax.runtime.task.worker_payload import encode_execution_request
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-ue-11e"
_RUN_BUDGET = RunBudget(max_llm_calls=5)
_CONSUME_FIRST = 2
_CONSUME_SECOND = 1
_CONSUME_THIRD = 1
_AGENT_ID = "ue_11e_budget_workload"
_CAPABILITY = "ue.11e.budget_redelivery"


class _KV(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}
        self._lock = threading.Lock()

    def get(self, tenant_id: str, key: str) -> bytes | None:
        with self._lock:
            return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        del ttl_seconds
        with self._lock:
            self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        with self._lock:
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
        del ttl_seconds
        with self._lock:
            current = self._data.get((tenant_id, key))
            if expected is None and current is not None:
                return False
            if expected is not None and current != expected:
                return False
            self._data[(tenant_id, key)] = new_value
            return True


@dataclass(frozen=True, slots=True)
class _DeliveryObservation:
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    remaining_before: int
    remaining_after: int
    budget_execution_id: ExecutionId


def _transport(transport_task_id: str) -> BackgroundTransportExecutionRef:
    return BackgroundTransportExecutionRef(
        tenant_id=_TENANT,
        provider="celery",
        transport_task_id=transport_task_id,
    )


def _parse_consume_amount(message: str) -> int:
    prefix = "consume="
    if not message.startswith(prefix):
        raise ValueError(f"expected message prefix {prefix!r}")
    return int(message[len(prefix) :])


class _BudgetRedeliveryWorkloadAgent(Agent):
    """Deterministic workload that consumes governed budget inside worker execution."""

    __slots__ = ("_observations",)

    def __init__(self, *, observations: list[_DeliveryObservation]) -> None:
        self._observations = observations

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=_AGENT_ID,
            name=_AGENT_ID,
            description="UE-11E budget redelivery workload",
            capabilities=[_CAPABILITY],
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        if not isinstance(task_context, TaskContext):
            return CapabilityMatchResult(matched=False)
        if task_context.capability == _CAPABILITY:
            return CapabilityMatchResult(
                matched=True,
                agent_id=_AGENT_ID,
                matched_capabilities=[_CAPABILITY],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="ue-11e-budget-workload"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        del context
        return [
            AgentStep(
                step_id=f"{_AGENT_ID}_step",
                step_name=f"{_AGENT_ID}_step",
                step_index=0,
                trace_label=_CAPABILITY,
            )
        ]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        del step
        run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        budget_state = require_active_execution_budget()
        if budget_state.execution_id != execution_id:
            raise RuntimeError("active execution budget execution_id mismatch")

        message = (ctx.request.message or "") if ctx.request is not None else ""
        consume_amount = _parse_consume_amount(message)
        remaining_before = budget_state.ledger.snapshot_root_available().max_llm_calls
        for _ in range(consume_amount):
            consume_llm_call()
        remaining_after = budget_state.ledger.snapshot_root_available().max_llm_calls

        self._observations.append(
            _DeliveryObservation(
                task_id=ctx.task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                remaining_before=remaining_before,
                remaining_after=remaining_after,
                budget_execution_id=budget_state.execution_id,
            )
        )
        return StepOutput(
            step_id=f"{_AGENT_ID}_step",
            summary="ue-11e budget workload complete",
            data={"consume_amount": consume_amount},
        )

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        del step, output, ctx
        return AgentDecision(
            type=AgentDecisionType.COMPLETE,
            reason="ue-11e budget workload finished",
        )


def _run_worker_delivery(
    *,
    identity: BackgroundExecutionIdentity,
    kv: _KV,
    observations: list[_DeliveryObservation],
    consume_amount: int,
) -> None:
    agent_registry = AgentRegistry()
    agent_registry.register(_BudgetRedeliveryWorkloadAgent(observations=observations))
    runtime = NexusWorkerRuntime.from_registry(
        agent_registry,
        run_budget=_RUN_BUDGET,
        run_budget_persistence=KvRunBudgetPersistence(kv),
    )
    task = Task(
        task_id=str(identity.task_id),
        tenant_id=identity.tenant_id,
        user_id="user-1",
        message=f"consume={consume_amount}",
        context=TaskContext(capability=_CAPABILITY),
    )
    request = ExecutionRequest(
        run_id="queue-correlation-only",
        tenant_id=identity.tenant_id,
        user_id="user-1",
        input_payload=task_to_execution_payload(task),
    )
    payload = encode_execution_request(request)

    runtime.execute_payload(
        payload,
        tenant_id=identity.tenant_id,
        run_id=str(identity.run_id),
        execution_identity=identity,
    )
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None
    assert peek_active_execution_budget() is None


def test_ue_11e_redelivery_identity_and_budget_continuity() -> None:
    kv = _KV()
    identity_persistence = KvBackgroundExecutionIdentityPersistence(kv)
    transport = _transport("transport-ue-11e-redelivery")
    observations: list[_DeliveryObservation] = []

    identities: list[BackgroundExecutionIdentity] = []
    for _ in range(3):
        identities.append(
            bootstrap_background_execution(
                transport_ref=transport,
                identity_persistence=identity_persistence,
            )
        )

    consume_amounts = (_CONSUME_FIRST, _CONSUME_SECOND, _CONSUME_THIRD)
    for identity, consume_amount in zip(identities, consume_amounts, strict=True):
        _run_worker_delivery(
            identity=identity,
            kv=kv,
            observations=observations,
            consume_amount=consume_amount,
        )

    assert len(observations) == 3
    first, second, third = observations

    assert second.task_id == first.task_id == third.task_id
    assert second.run_id == first.run_id == third.run_id
    assert second.attempt_id == first.attempt_id == third.attempt_id
    assert len({first.execution_id, second.execution_id, third.execution_id}) == 3

    assert first.remaining_before == 5
    assert first.remaining_after == 3
    assert first.budget_execution_id == first.execution_id

    assert second.remaining_before == 3
    assert second.remaining_after == 2
    assert second.budget_execution_id == second.execution_id

    assert third.remaining_before == 2
    assert third.remaining_after == 1
    assert third.budget_execution_id == third.execution_id

    ledger_factory = create_durable_run_budget_ledger_factory(
        KvRunBudgetPersistence(kv),
        _RUN_BUDGET,
    )
    final_ledger = ledger_factory.create_ledger(
        _RUN_BUDGET,
        tenant_id=_TENANT,
        run_id=third.run_id,
        attempt_id=third.attempt_id,
    )
    assert final_ledger.snapshot_root_available().max_llm_calls == 1

    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None
    assert peek_active_execution_budget() is None
