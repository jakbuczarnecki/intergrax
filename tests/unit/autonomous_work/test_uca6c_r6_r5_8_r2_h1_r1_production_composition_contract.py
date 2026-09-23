# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.8-R2-H1-R1 — production composition contract hardening."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from intergrax.autonomous_work.capability_acquisition_ports import (
    StaticWorkerCapabilityProfileResolver,
    permissive_capability_policy,
)
from intergrax.autonomous_work.in_memory_repository import (
    InMemoryWorkerPrincipalBindingRepository,
)
from intergrax.autonomous_work.in_memory_worker_recovery_obstacle_capability_need_repository import (
    InMemoryWorkerRecoveryObstacleCapabilityNeedRepository,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_composition import (
    build_worker_qualified_capability_resume_coordinator,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_coordinator import (
    WorkerQualifiedCapabilityResumeCoordinator,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_ports import (
    WorkerQualifiedCapabilityAsyncExecutionPort,
    WorkerQualifiedCapabilityExecutionPort,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_episode_context_provider import (
    DurableWorkerRecoveryCapabilityFulfillmentEpisodeContextProvider,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_request_builder import (
    WorkerRecoveryCapabilityFulfillmentRequestBuilder,
)
from intergrax.autonomous_work.worker_recovery_governed_fulfillment_composition import (
    build_worker_recovery_governed_fulfillment_wiring,
)
from intergrax.autonomous_work.worker_recovery_orchestration_service import (
    _episode_from_request,
)
from intergrax.capability_qualification.qualified_capability_binding_service import (
    QualifiedCapabilityBindingService,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionResult,
    WorkerQualifiedCapabilityResumeRequest,
    WorkerQualifiedCapabilityResumeResult,
)
from intergrax.contracts.execution_identity import RunId, TaskId, mint_run_id
from intergrax.runtime.execution.qualified_capability_execution_composition import (
    build_qualified_capability_execution_dispatch_service,
)
from intergrax.runtime.execution.worker_qualified_capability_execution_async_adapter import (
    WorkerQualifiedCapabilityExecutionEngineAsyncAdapter,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.active_task_registry_fulfillment_task_context_reader import (
    ActiveTaskRegistryFulfillmentTaskContextReader,
)
from intergrax.runtime.task.task import Task
from tests.unit.autonomous_work import repository_contracts as contract_suite
from tests.unit.autonomous_work.test_uca6b_worker_capability_recovery import (
    _recovery_decision,
    _worker_need,
)
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.autonomous_work.test_uca6c_r6_r5_8_worker_consumer_e2e import (
    _PROFILE,
    _READ,
)
from tests.unit.autonomous_work.test_worker_recovery_orchestration import (
    _decision,
    _orchestration_request,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_active_task_registry() -> None:
    ActiveTaskRegistry.clear_for_tests()


_REPO = Path(__file__).resolve().parents[3]
_NOW = datetime(2026, 9, 23, 16, 0, tzinfo=UTC)
_COORDINATOR_PATH = (
    _REPO / "intergrax" / "autonomous_work" / "worker_qualified_capability_resume_coordinator.py"
)
_COMPOSITION_PATH = (
    _REPO
    / "intergrax"
    / "autonomous_work"
    / "worker_recovery_governed_fulfillment_composition.py"
)
_PROVIDER_PATH = (
    _REPO
    / "intergrax"
    / "autonomous_work"
    / "worker_recovery_capability_fulfillment_episode_context_provider.py"
)


@dataclass
class _FakeAsyncExecution:
    calls: int = 0

    async def execute_async(
        self,
        request,
    ) -> WorkerQualifiedCapabilityExecutionResult:
        self.calls += 1
        del request
        return WorkerQualifiedCapabilityExecutionResult(
            disposition=__import__(
                "intergrax.contracts.autonomous_work.worker_qualified_capability_resume",
                fromlist=["WorkerQualifiedCapabilityExecutionDisposition"],
            ).WorkerQualifiedCapabilityExecutionDisposition.FAILED,
            execution_request_id="fake",
            reason_detail="fake",
        )


@dataclass
class _FakeSyncExecution:
    def execute(self, request):
        del request
        return WorkerQualifiedCapabilityExecutionResult(
            disposition=__import__(
                "intergrax.contracts.autonomous_work.worker_qualified_capability_resume",
                fromlist=["WorkerQualifiedCapabilityExecutionDisposition"],
            ).WorkerQualifiedCapabilityExecutionDisposition.FAILED,
            execution_request_id="fake",
            reason_detail="fake",
        )


class _FakeBinding:
    def bind(self, request):
        del request
        return __import__(
            "intergrax.contracts.capability_qualification.qualified_capability_binding",
            fromlist=["QualifiedCapabilityBindingResult"],
        ).QualifiedCapabilityBindingResult(
            outcome=__import__(
                "intergrax.contracts.capability_qualification.qualified_capability_binding",
                fromlist=["QualifiedCapabilityBindingOutcome"],
            ).QualifiedCapabilityBindingOutcome.BLOCKED,
            binding_operation_id="bind:test",
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


def _production_provider_stack(
    *,
    tenant_id: str = _TENANT,
    task: Task | None = None,
    run_id: RunId | None = None,
):
    need = _aligned_worker_need()
    need_repo = InMemoryWorkerRecoveryObstacleCapabilityNeedRepository()
    need_repo.record_obstacle_capability_need(need)
    principal_repo = InMemoryWorkerPrincipalBindingRepository()
    worker_id = need.worker_instance_id
    principal_repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=worker_id,
            tenant_id=tenant_id,
        ),
    )
    task_reader = ActiveTaskRegistryFulfillmentTaskContextReader()
    profile_resolver = StaticWorkerCapabilityProfileResolver(
        permissive_capability_policy(_PROFILE),
    )
    provider = DurableWorkerRecoveryCapabilityFulfillmentEpisodeContextProvider(
        obstacle_capability_need_reader=need_repo,
        task_context_reader=task_reader,
        principal_binding_repository=principal_repo,
        capability_profile_resolver=profile_resolver,
    )
    return provider, need, task, run_id


@pytest.mark.asyncio
async def test_resume_coordinator_accepts_fake_async_port_without_runtime_import() -> None:
    coordinator = build_worker_qualified_capability_resume_coordinator(
        binding=_FakeBinding(),
        execution=_FakeSyncExecution(),
        async_execution=_FakeAsyncExecution(),
    )
    assert isinstance(coordinator, WorkerQualifiedCapabilityResumeCoordinator)
    tree = ast.parse(_COORDINATOR_PATH.read_text(encoding="utf-8"))
    modules = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert not any("worker_qualified_capability_execution_async_adapter" in mod for mod in modules)


def test_runtime_async_adapter_implements_async_execution_port() -> None:
    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=__import__(
            "intergrax.runtime.execution.qualified_capability_execution_handlers",
            fromlist=["QualifiedCapabilityExecutionBindingHandlerRegistry"],
        ).QualifiedCapabilityExecutionBindingHandlerRegistry(()),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    adapter = WorkerQualifiedCapabilityExecutionEngineAsyncAdapter(dispatch=dispatch)
    assert isinstance(adapter, WorkerQualifiedCapabilityAsyncExecutionPort)


def test_governed_composition_source_has_no_private_execution_mutation() -> None:
    source = _COMPOSITION_PATH.read_text(encoding="utf-8")
    assert "resume._execution" not in source
    assert "resume._async_execution" not in source
    assert "SLF001" not in source


def test_governed_composition_builds_resume_coordinator_via_public_constructor() -> None:
    from intergrax.autonomous_work.worker_capability_fulfillment_coordinator import (
        WorkerCapabilityFulfillmentCoordinator,
    )

    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=__import__(
            "intergrax.runtime.execution.qualified_capability_execution_handlers",
            fromlist=["QualifiedCapabilityExecutionBindingHandlerRegistry"],
        ).QualifiedCapabilityExecutionBindingHandlerRegistry(()),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    need_repo = InMemoryWorkerRecoveryObstacleCapabilityNeedRepository()
    principal_repo = InMemoryWorkerPrincipalBindingRepository()
    profile_resolver = StaticWorkerCapabilityProfileResolver(
        permissive_capability_policy(_PROFILE),
    )
    binding = QualifiedCapabilityBindingService(())
    fulfillment = Mock(spec=WorkerCapabilityFulfillmentCoordinator)
    (
        request_builder,
        _sync,
        _async,
        governed_dispatch,
        resume,
    ) = build_worker_recovery_governed_fulfillment_wiring(
        fulfillment_coordinator=fulfillment,
        inner_dispatch=dispatch,
        binding=binding,
        obstacle_capability_need_reader=need_repo,
        task_context_reader=ActiveTaskRegistryFulfillmentTaskContextReader(),
        principal_binding_repository=principal_repo,
        capability_profile_resolver=profile_resolver,
    )
    assert request_builder is not None
    assert governed_dispatch is not None
    assert isinstance(resume, WorkerQualifiedCapabilityResumeCoordinator)
    assert isinstance(resume._execution, WorkerQualifiedCapabilityExecutionPort)
    assert isinstance(resume._async_execution, WorkerQualifiedCapabilityAsyncExecutionPort)


@pytest.mark.asyncio
async def test_production_episode_context_provider_success() -> None:
    task = Task(tenant_id=_TENANT, user_id="u1", message="ctx", task_id=_TASK_ID)
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    provider, need, _, _ = _production_provider_stack(task=task, run_id=run_id)
    decision = _recovery_decision(need)
    orch = _orchestration_request(decision=decision)
    orch = replace(
        orch,
        resume_target=replace(orch.resume_target, run_id=run_id, requested_scopes=(_READ,)),
    )
    episode = _episode_from_request(orch, started_at=_NOW)
    context = provider.resolve_episode_context(episode=episode, request=orch)
    assert context is not None
    assert context.tenant_id == _TENANT
    assert context.task_id == _TASK_ID
    assert context.worker_need.worker_instance_id == need.worker_instance_id


@pytest.mark.asyncio
async def test_production_episode_context_missing_need_fail_closed() -> None:
    need_repo = InMemoryWorkerRecoveryObstacleCapabilityNeedRepository()
    principal_repo = InMemoryWorkerPrincipalBindingRepository()
    worker_id = __import__(
        "tests.unit.autonomous_work.test_worker_recovery_orchestration",
        fromlist=["_WORKER_ID"],
    )._WORKER_ID
    principal_repo.create(
        contract_suite.worker_principal_binding(worker_instance_id=worker_id),
    )
    provider = DurableWorkerRecoveryCapabilityFulfillmentEpisodeContextProvider(
        obstacle_capability_need_reader=need_repo,
        task_context_reader=ActiveTaskRegistryFulfillmentTaskContextReader(),
        principal_binding_repository=principal_repo,
        capability_profile_resolver=StaticWorkerCapabilityProfileResolver(
            permissive_capability_policy(_PROFILE),
        ),
    )
    from intergrax.contracts.autonomous_work.obstacle_recovery import RecoveryStrategy

    orch = _orchestration_request(decision=_decision(strategy=RecoveryStrategy.ACQUIRE_CAPABILITY))
    episode = _episode_from_request(orch, started_at=_NOW)
    assert provider.resolve_episode_context(episode=episode, request=orch) is None


@pytest.mark.asyncio
async def test_production_episode_context_tenant_mismatch_fail_closed() -> None:
    task = Task(tenant_id=_TENANT, user_id="u1", message="ctx", task_id=_TASK_ID)
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    provider, need, _, _ = _production_provider_stack(
        tenant_id="tenant-other",
        task=task,
        run_id=run_id,
    )
    decision = _recovery_decision(need)
    orch = replace(
        _orchestration_request(decision=decision),
        resume_target=replace(
            _orchestration_request(decision=decision).resume_target,
            run_id=run_id,
        ),
    )
    episode = _episode_from_request(orch, started_at=_NOW)
    assert provider.resolve_episode_context(episode=episode, request=orch) is None


@pytest.mark.asyncio
async def test_production_episode_context_task_missing_for_run_fail_closed() -> None:
    provider, need, _, _ = _production_provider_stack()
    decision = _recovery_decision(need)
    orch = replace(
        _orchestration_request(decision=decision),
        resume_target=replace(
            _orchestration_request(decision=decision).resume_target,
            run_id=mint_run_id(),
        ),
    )
    episode = _episode_from_request(orch, started_at=_NOW)
    assert provider.resolve_episode_context(episode=episode, request=orch) is None


def test_production_episode_context_decision_mismatch_fail_closed() -> None:
    provider, need, _, _ = _production_provider_stack()
    decision = _recovery_decision(need)
    wrong = replace(decision, decision_id="wrong-decision")
    orch = _orchestration_request(decision=wrong)
    episode = _episode_from_request(orch, started_at=_NOW)
    assert provider.resolve_episode_context(episode=episode, request=orch) is None


def test_production_episode_context_worker_need_mismatch_fail_closed() -> None:
    need_repo = InMemoryWorkerRecoveryObstacleCapabilityNeedRepository()
    need = _aligned_worker_need()
    mismatched = replace(need, recovery_decision_id="other-decision")
    need_repo.record_obstacle_capability_need(mismatched)
    principal_repo = InMemoryWorkerPrincipalBindingRepository()
    principal_repo.create(
        contract_suite.worker_principal_binding(worker_instance_id=need.worker_instance_id),
    )
    provider = DurableWorkerRecoveryCapabilityFulfillmentEpisodeContextProvider(
        obstacle_capability_need_reader=need_repo,
        task_context_reader=ActiveTaskRegistryFulfillmentTaskContextReader(),
        principal_binding_repository=principal_repo,
        capability_profile_resolver=StaticWorkerCapabilityProfileResolver(
            permissive_capability_policy(_PROFILE),
        ),
    )
    decision = _recovery_decision(need)
    orch = _orchestration_request(decision=decision)
    episode = _episode_from_request(orch, started_at=_NOW)
    assert provider.resolve_episode_context(episode=episode, request=orch) is None


@pytest.mark.asyncio
async def test_builder_with_production_provider_projects_request() -> None:
    task = Task(tenant_id=_TENANT, user_id="u1", message="ctx", task_id=_TASK_ID)
    run_id = mint_run_id()
    await ActiveTaskRegistry.register(task, run_id)
    provider, need, _, _ = _production_provider_stack(task=task, run_id=run_id)
    decision = _recovery_decision(need)
    orch = replace(
        _orchestration_request(decision=decision),
        resume_target=replace(
            _orchestration_request(decision=decision).resume_target,
            run_id=run_id,
            requested_scopes=(_READ,),
        ),
    )
    episode = _episode_from_request(orch, started_at=_NOW)
    builder = WorkerRecoveryCapabilityFulfillmentRequestBuilder(episode_context=provider)
    projected = builder.build_fulfillment_request(episode=episode, request=orch)
    assert projected is not None
    assert projected.task_id == _TASK_ID
    assert projected.tenant_id == _TENANT


def test_production_episode_context_provider_static_gate_no_discovery_acquisition() -> None:
    tree = ast.parse(_PROVIDER_PATH.read_text(encoding="utf-8"))
    modules = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    forbidden = ("discovery", "acquisition_service", "capability_acquisition_service")
    assert not any(
        mod and any(token in mod for token in forbidden) for mod in modules
    )
