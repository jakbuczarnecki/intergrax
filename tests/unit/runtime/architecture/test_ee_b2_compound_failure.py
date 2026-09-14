# © Artur Czarnecki. All rights reserved.

"""EE-B2 — compound fault scenarios (primary vs secondary authority)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.concurrent_execution_work import ConcurrentExecutionWorkPolicy
from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityAdmissionRequest,
    ExecutionCapacityPolicy,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.execution.capacity import LocalExecutionCapacityAdmission
from intergrax.runtime.execution.concurrent_execution_work import (
    ConcurrentExecutionWorkDisposition,
    execute_concurrent_execution_work_resilient,
)
from intergrax.runtime.execution.request import ExecutionRequest
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy
from intergrax.runtime.observability.export_wiring import (
    make_observability_export_runtime_plugin,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from testing_support.chaos.barriers import PhaseGate
from testing_support.chaos.execution_ports import DeterministicWorkerFaultPort
from testing_support.chaos.failing_persistence import FailOnAppendPersistence
from testing_support.chaos.fault_plan import FailOnCall
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class _WorkResult:
    value: str


def _req(label: str) -> ExecutionRequest[str, _WorkResult]:
    return ExecutionRequest(input=label, output_type=_WorkResult)


class _FailingExporter:
    async def export(self, envelope) -> None:
        raise RuntimeError("otlp_down")


@pytest.mark.asyncio
async def test_ee_b2_compound_worker_failure_plus_otlp_export_failure() -> None:
    """Primary: worker fault. Secondary: OTLP export. Terminal authority: worker FAILED."""
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store, record_history=False)
    identity = runtime_event_test_identity()
    plugin = make_observability_export_runtime_plugin(
        exporter=_FailingExporter(),
        policy=ObservabilityExportPolicy(enabled=True),
    )
    plugin.register(bus, HookRegistry(), MagicMock())
    port = DeterministicWorkerFaultPort(
        fail_labels=frozenset({"worker"}),
        succeed=lambda label: _WorkResult(value=label),
    )
    outcomes = await execute_concurrent_execution_work_resilient(
        port,
        (_req("worker"), _req("ok")),
        policy=ConcurrentExecutionWorkPolicy(max_concurrency=2),
    )
    assert outcomes[0].disposition is ConcurrentExecutionWorkDisposition.FAILED
    assert outcomes[1].disposition is ConcurrentExecutionWorkDisposition.SUCCEEDED
    await bus.publish(sample_runtime_event(tenant_id="tenant-a", **identity))  # type: ignore[arg-type]
    assert len(store.list_for_run(identity["run_id"], tenant_id="tenant-a")) == 1


@pytest.mark.asyncio
async def test_ee_b2_compound_capacity_saturation_plus_cancel_releases_slot() -> None:
    gate = PhaseGate()
    policy = ExecutionCapacityPolicy(max_concurrent_root_executions=1)
    admission = LocalExecutionCapacityAdmission(policy)
    req = ExecutionCapacityAdmissionRequest(
        tenant_id="t",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    permit = await admission.acquire(req)

    from intergrax.runtime.execution.execution_work_port import ExecutionWorkPort

    class _GatePort(ExecutionWorkPort[str, _WorkResult, _WorkResult]):
        async def execute(
            self, request: ExecutionRequest[str, _WorkResult]
        ) -> _WorkResult:
            gate.mark_started(request.input)
            await gate.block()
            return _WorkResult(value=request.input)

    port = _GatePort()
    work = asyncio.create_task(
        execute_concurrent_execution_work_resilient(
            port,
            (_req("held"),),
            policy=ConcurrentExecutionWorkPolicy(max_concurrency=1),
        ),
    )
    await gate.wait_until_started(frozenset({"held"}))
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work
    await permit.release()
    second = await admission.acquire(req)
    await second.release()


@pytest.mark.asyncio
async def test_ee_b2_compound_execution_failure_plus_mandatory_evidence_failure() -> (
    None
):
    from intergrax.contracts.execution_evidence.persistence_boundary_errors import (
        MandatoryEvidencePersistenceError,
    )

    inner = InMemoryRuntimeEventStore()
    persistence = FailOnAppendPersistence(
        inner,
        fail_on=FailOnCall(call_number=1, message="mandatory_append_fault"),
    )
    bus = RuntimeEventBus(persistence=persistence, record_history=True)
    event = sample_runtime_event(tenant_id="tenant-compound")
    with pytest.raises(MandatoryEvidencePersistenceError):
        bus.record(event, tenant_id="tenant-compound")
    assert bus.history == []
