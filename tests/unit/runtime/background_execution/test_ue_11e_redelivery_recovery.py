# © Artur Czarnecki. All rights reserved.

"""UE-11E — background redelivery recovery qualification."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    bind_active_execution_identity,
    mint_execution_id,
    peek_active_execution_id,
    peek_active_execution_identity,
)
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
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedgerFactory
from intergrax.runtime.execution.budget.models import (
    BudgetUsageTotals,
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
)
from intergrax.runtime.execution.budget.persistence import (
    KvRunBudgetPersistence,
    create_durable_run_budget_ledger_factory,
)
from intergrax.runtime.execution.active_execution_budget import peek_active_execution_budget
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.nexus_worker_execution import NexusWorkerRuntime
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import task_to_execution_payload
from intergrax.runtime.task.worker_payload import encode_execution_request
from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-ue-11e"
_RUN_BUDGET = RunBudget(max_llm_calls=5)
_CONSUME_FIRST = 2
_CONSUME_SECOND = 1
_CONSUME_THIRD = 1


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
    llm_calls_remaining_before_consume: int | None


def _transport(transport_task_id: str) -> BackgroundTransportExecutionRef:
    return BackgroundTransportExecutionRef(
        tenant_id=_TENANT,
        provider="celery",
        transport_task_id=transport_task_id,
    )


def _consume_llm_calls(
    ledger,
    *,
    root_execution_id: ExecutionId,
    amount: int,
) -> None:
    child_id = mint_execution_id()
    ledger.grant_child_budget(
        execution_id=child_id,
        parent_execution_id=root_execution_id,
        decision=ChildBudgetAllocationDecision(mode=ExecutionBudgetAllocationMode.SHARED),
    )
    ledger.consume_budget(child_id, BudgetUsageTotals(llm_calls=amount))
    ledger.release_child_budget(child_id)


def _run_worker_delivery(
    *,
    identity: BackgroundExecutionIdentity,
    capture: list[_DeliveryObservation],
    ledger_factory: ExecutionBudgetLedgerFactory,
    consume_amount: int,
) -> None:
    agent_registry = AgentRegistry()
    agent_registry.register(EchoAgent())
    runtime = NexusWorkerRuntime.from_registry(agent_registry)
    task = Task(
        task_id=str(identity.task_id),
        tenant_id=identity.tenant_id,
        user_id="user-1",
        message="ue-11e redelivery",
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
        execution_id = kwargs["execution_id"]
        ledger = ledger_factory.create_ledger(
            _RUN_BUDGET,
            tenant_id=identity.tenant_id,
            run_id=identity.run_id,
            attempt_id=identity.attempt_id,
        )
        remaining_before = ledger.snapshot_root_available().max_llm_calls
        _consume_llm_calls(ledger, root_execution_id=execution_id, amount=consume_amount)
        capture.append(
            _DeliveryObservation(
                task_id=identity.task_id,
                run_id=kwargs["run_id"],
                attempt_id=kwargs["attempt_id"],
                execution_id=execution_id,
                llm_calls_remaining_before_consume=remaining_before,
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


def test_ue_11e_redelivery_identity_and_budget_continuity() -> None:
    kv = _KV()
    identity_persistence = KvBackgroundExecutionIdentityPersistence(kv)
    ledger_factory = create_durable_run_budget_ledger_factory(
        KvRunBudgetPersistence(kv),
        _RUN_BUDGET,
    )
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
            capture=observations,
            ledger_factory=ledger_factory,
            consume_amount=consume_amount,
        )

    assert len(observations) == 3
    first, second, third = observations

    assert second.task_id == first.task_id == third.task_id
    assert second.run_id == first.run_id == third.run_id
    assert len({first.attempt_id, second.attempt_id, third.attempt_id}) == 3
    assert len({first.execution_id, second.execution_id, third.execution_id}) == 3

    assert first.llm_calls_remaining_before_consume == 5
    assert second.llm_calls_remaining_before_consume == 3
    assert third.llm_calls_remaining_before_consume == 2

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
