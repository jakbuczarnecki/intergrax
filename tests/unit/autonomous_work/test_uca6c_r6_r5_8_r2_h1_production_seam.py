# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.8-R2-H1 — governed worker execution production seam."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.autonomous_work.recovery_orchestration_ports import (
    CanonicalExecutionTerminalDisposition,
    PortAvailabilityDisposition,
    WorkerRecoveryCapabilityFulfillmentRequest,
    WorkerRecoveryCapabilityFulfillmentResult,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_episode_context import (
    WorkerRecoveryCapabilityFulfillmentEpisodeContext,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_request_builder import (
    WorkerRecoveryCapabilityFulfillmentRequestBuilder,
)
from intergrax.autonomous_work.worker_recovery_orchestration_service import (
    _episode_from_request,
)
from intergrax.contracts.autonomous_work.obstacle_recovery import RecoveryStrategy
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    WorkerRecoveryOrchestrationDisposition,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentDisposition,
    WorkerCapabilityFulfillmentResult,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryProvenance,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
    WorkerQualifiedCapabilityExecutionResult,
)
from intergrax.runtime.execution.governed_task_scoped_qualified_capability_execution_dispatch import (
    ActiveTaskRegistryGovernedExecutionTaskLookup,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    peek_governed_execution_task,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task
from tests.unit.autonomous_work.test_uca6b_worker_capability_recovery import (
    _recovery_decision,
    _worker_need,
)
from tests.unit.autonomous_work.test_uca6c_r6_r5_8_worker_consumer_e2e import (
    _PROFILE,
    _READ,
)
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.autonomous_work.test_worker_recovery_orchestration import (
    StubExecutionOutcomeReader,
    _decision,
    _harness,
    _orchestration_request,
)
from tests.unit.autonomous_work.test_uca6c_worker_qualified_capability_resume import (
    _EXEC_ID,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_active_task_registry() -> None:
    ActiveTaskRegistry.clear_for_tests()


_NOW = datetime(2026, 9, 23, 15, 0, tzinfo=UTC)
_REPO = Path(__file__).resolve().parents[3]
_BUILDER_PATH = (
    _REPO
    / "intergrax"
    / "autonomous_work"
    / "worker_recovery_capability_fulfillment_request_builder.py"
)


@dataclass
class _StaticEpisodeContext:
    context: WorkerRecoveryCapabilityFulfillmentEpisodeContext | None

    def resolve_episode_context(self, *, episode, request):
        if self.context is None:
            return None
        return replace(
            self.context,
            recovery_decision=request.decision,
        )


@dataclass
class _RecordingAsyncFulfillment:
    calls: int = 0
    legacy_acquisition_calls: int = 0

    async def fulfill_recovery_capability_async(
        self,
        handoff: WorkerRecoveryCapabilityFulfillmentRequest,
    ) -> WorkerRecoveryCapabilityFulfillmentResult:
        self.calls += 1
        provenance = WorkerCapabilityRecoveryProvenance(
            worker_need_id="need",
            canonical_need_id="canonical",
            discovery_correlation_id="corr",
            discovery_completion_outcome="missing",
            evidence_refs=(),
        )
        return WorkerRecoveryCapabilityFulfillmentResult(
            disposition=PortAvailabilityDisposition.AVAILABLE,
            fulfillment_result=WorkerCapabilityFulfillmentResult(
                disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED,
                provenance=provenance,
                execution_result=WorkerQualifiedCapabilityExecutionResult(
                    disposition=WorkerQualifiedCapabilityExecutionDisposition.DISPATCHED,
                    execution_request_id="worker-qualified-capability-execution:test",
                    execution_id=_EXEC_ID,
                ),
                decided_at=_NOW,
            ),
        )


def _episode_context() -> WorkerRecoveryCapabilityFulfillmentEpisodeContext:
    need = _worker_need()
    decision = _recovery_decision(need)
    return WorkerRecoveryCapabilityFulfillmentEpisodeContext(
        worker_need=need,
        recovery_decision=decision,
        capability_profile_ref=_PROFILE,
        tenant_id=_TENANT,
        task_id=_TASK_ID,
        requested_authority_scopes=(_READ,),
        allow_generic_acquisition=True,
    )


def _aligned_worker_need():
    worker_id = __import__(
        "tests.unit.autonomous_work.test_worker_recovery_orchestration",
        fromlist=["_WORKER_ID"],
    )._WORKER_ID
    return replace(
        _worker_need(),
        worker_instance_id=worker_id,
        obstacle_id=f"{worker_id}:obstacle:uca6b-1",
    )


def test_production_builder_projects_typed_request() -> None:
    need = _aligned_worker_need()
    decision = _recovery_decision(need)
    builder = WorkerRecoveryCapabilityFulfillmentRequestBuilder(
        episode_context=_StaticEpisodeContext(
            WorkerRecoveryCapabilityFulfillmentEpisodeContext(
                worker_need=need,
                recovery_decision=decision,
                capability_profile_ref=_PROFILE,
                tenant_id=_TENANT,
                task_id=_TASK_ID,
                requested_authority_scopes=(_READ,),
            ),
        ),
    )
    orch = _orchestration_request(decision=decision)
    episode = _episode_from_request(orch, started_at=_NOW)
    projected = builder.build_fulfillment_request(episode=episode, request=orch)
    assert projected is not None
    assert projected.tenant_id == _TENANT
    assert projected.task_id == _TASK_ID
    assert (
        projected.acquisition_request.need.recovery_episode_id
        == episode.recovery_episode_id
    )


def test_production_builder_fail_closed_on_missing_context() -> None:
    builder = WorkerRecoveryCapabilityFulfillmentRequestBuilder(
        episode_context=_StaticEpisodeContext(None),
    )
    orch = _orchestration_request(
        decision=_decision(strategy=RecoveryStrategy.ACQUIRE_CAPABILITY),
    )
    episode = _episode_from_request(orch, started_at=_NOW)
    assert builder.build_fulfillment_request(episode=episode, request=orch) is None


def test_builder_module_static_gate_no_tool_runtime_imports() -> None:
    tree = ast.parse(_BUILDER_PATH.read_text(encoding="utf-8"))
    modules = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert not any("nexus" in mod for mod in modules)


@pytest.mark.asyncio
async def test_orchestrate_uses_production_builder_and_async_fulfillment() -> None:
    need = _aligned_worker_need()
    decision = replace(
        _recovery_decision(need),
        strategy=RecoveryStrategy.ACQUIRE_CAPABILITY,
    )
    builder = WorkerRecoveryCapabilityFulfillmentRequestBuilder(
        episode_context=_StaticEpisodeContext(
            WorkerRecoveryCapabilityFulfillmentEpisodeContext(
                worker_need=need,
                recovery_decision=decision,
                capability_profile_ref=_PROFILE,
                tenant_id=_TENANT,
                task_id=_TASK_ID,
                requested_authority_scopes=(_READ,),
            ),
        ),
    )
    recording = _RecordingAsyncFulfillment()
    service, _ = _harness()
    service._recovery_capability_fulfillment_request_builder = builder
    service._recovery_capability_fulfillment_async = recording
    service._execution_outcome_reader = StubExecutionOutcomeReader(
        CanonicalExecutionTerminalDisposition.IN_PROGRESS,
    )
    orch = await service.orchestrate(_orchestration_request(decision=decision))
    assert recording.calls == 1
    assert orch.disposition is WorkerRecoveryOrchestrationDisposition.ATTEMPT_DISPATCHED


@pytest.mark.asyncio
async def test_active_task_registry_peek_binding_for_governed_lookup() -> None:
    task = Task(tenant_id=_TENANT, user_id="u1", message="h1", task_id=_TASK_ID)
    run_id = __import__(
        "intergrax.contracts.execution_identity",
        fromlist=["mint_run_id"],
    ).mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    binding = ActiveTaskRegistry.peek_binding(_TASK_ID)
    assert binding is not None
    assert binding.task.task_id == _TASK_ID
    lookup = ActiveTaskRegistryGovernedExecutionTaskLookup()
    assert lookup.resolve_task(_TASK_ID) is task
    assert peek_governed_execution_task() is None


def test_orchestration_source_has_no_nested_event_loop_workaround() -> None:
    path = (
        _REPO
        / "intergrax"
        / "autonomous_work"
        / "worker_recovery_orchestration_service.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "ThreadPoolExecutor" not in source
    assert "nest_asyncio" not in source
    assert "asyncio.run(" not in source
