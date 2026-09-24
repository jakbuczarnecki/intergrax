# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9-R1 — true process restart Host A/B/C worker governed execution."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.autonomous_work.capability_acquisition_ports import (
    StaticWorkerCapabilityProfileResolver,
    permissive_capability_policy,
)
from intergrax.autonomous_work.host_available_capability_binding_service import (
    HostAvailableCapabilityBindingService,
)
from intergrax.autonomous_work.in_memory_repository import (
    InMemoryWorkerPrincipalBindingRepository,
)
from intergrax.autonomous_work.repository import (
    WorkerInstanceRepository,
    WorkerPrincipalBindingRepository,
)
from intergrax.autonomous_work.persistence import AutonomousWorkRepositories
from intergrax.autonomous_work.worker_capability_direct_reuse_fulfillment_service import (
    WorkerCapabilityDirectReuseFulfillmentService,
)
from intergrax.autonomous_work.worker_capability_recovery_coordinator import (
    WorkerCapabilityRecoveryCoordinator,
)
from intergrax.autonomous_work.worker_recovery_governed_fulfillment_composition import (
    WorkerRecoveryGovernedFulfillmentWiring,
    build_worker_recovery_governed_fulfillment_wiring,
)
from intergrax.autonomous_work.worker_recovery_orchestration_service import (
    WorkerRecoveryOrchestrationService,
)
from intergrax.capability_acquisition.acquisition_service import (
    CapabilityAcquisitionService,
)
from intergrax.capability_acquisition.permit_acquisition_authorization import (
    PermitCapabilityAcquisitionAuthorizationPort,
)
from intergrax.capability_qualification.qualified_capability_binding_service import (
    QualifiedCapabilityBindingService,
)
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    WorkerRecoveryOrchestrationDisposition,
    derive_recovery_episode_id,
)
from intergrax.runtime.execution.continuation.durable_restart_identity_correlation import (
    correlate_durable_restart_execution_identity,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
)
from intergrax.contracts.execution_continuation import ExecutionContinuationLookup
from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdStore,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.runtime.codecraft.artifact_reference import artifact_reference_for_craft
from intergrax.runtime.codecraft.qualified_capability_binding_provider import (
    CodeCraftQualifiedCapabilityBindingProvider,
)
from intergrax.runtime.execution.qualified_capability_execution_composition import (
    build_qualified_capability_execution_dispatch_service,
)
from intergrax.runtime.execution.qualified_capability_execution_dispatch_service import (
    QualifiedCapabilityExecutionDispatchService,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandlerRegistry,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    peek_governed_execution_task,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
)
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    root_execution_id_from_tree,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.long_running.resume_planner import (
    build_checkpoint_resume_task,
    execution_identity_from_checkpoint,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.active_task_registry_fulfillment_task_context_reader import (
    ActiveTaskRegistryFulfillmentTaskContextReader,
)
from intergrax.runtime.task.task import Task, TaskState
from intergrax.autonomous_work.recovery_orchestration_ports import (
    CanonicalExecutionOutcomeReader,
    CanonicalExecutionTerminalDisposition,
)
from intergrax.runtime.execution.execution_terminal_outcome_by_execution_id import (
    build_canonical_execution_outcome_reader,
)
from intergrax.autonomous_work.lifecycle import WorkerLifecycleService
from intergrax.autonomous_work.worker_capability_need_projection import (
    project_worker_capability_need_to_capability_need,
)
from intergrax.autonomous_work.worker_recovery_capability_fulfillment_episode_context_ports import (
    WorkerRecoveryObstacleCapabilityNeedStorePort,
)
from intergrax.runtime.execution.continuation.composition import (
    ExecutionEngineContinuationDependencies,
)
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
)
from testing_support.uca6c_r6_r5_9_r1_true_restart_durable_backends import (
    Uca6cTrueRestartDurableBackends,
)
from tests.integration.autonomous_work.conftest import (
    drop_schema,
    open_bundle,
    resolve_postgresql_config,
)
from tests.unit.autonomous_work import repository_contracts as contract_suite
from tests.unit.autonomous_work.test_uca6b_worker_capability_recovery import (
    _PROFILE,
    _recovery_decision,
    _worker_need,
)
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.autonomous_work.test_uca6c_r6_r5_8_worker_consumer_e2e import (
    _StaticDiscovery,
)
from tests.unit.autonomous_work.test_uca6c_r_production_resume import (
    _PRINCIPAL,
    _WORKSPACE,
)
from tests.unit.autonomous_work.test_worker_recovery_orchestration import (
    RecordingHumanDecisionPort,
    RecordingRecoveryDispatchPort,
    _clock,
    _orchestration_request,
    _resume_target,
)
from tests.unit.autonomous_work.test_worker_recovery_orchestration import (
    _GOAL_ID,
    _RESP_ID,
    _WORKER_ID,
)
from tests.unit.autonomous_work.test_uca6c_worker_qualified_capability_resume import (
    _RecordingExecutionPort,
)
from tests.unit.autonomous_work.uca6c_worker_authority_fixtures import (
    _READ,
    build_worker_execution_admission_for_uca6c,
)
from tests.unit.autonomous_work.test_uca6c_r6_r5_8_r2_worker_governed_execution_e2e import (
    _AllowingMsePort,
    _ArtifactQualification,
    _CountingAsyncRecoveryFulfillment,
    _CountingCodeCraftBindingProvider,
    _CountingRootLauncher,
    _CraftAlignedAcquisitionStrategy,
    _continuation_for_worker_pause,
)
from tests.unit.autonomous_work.test_uca6c_r5_r2_strict_governance_composition import (
    _codecraft_context,
    _strict_tool_wiring,
)
from tests.unit.autonomous_work.uca6c_r5_r2_strict_fixtures import (
    uca6c_strict_worker_manifest,
    uca6c_strict_worker_registry,
)
from tests.unit.runtime.execution.test_uca6c_r6_r5_7_sequential_authority_generations import (
    _CountingToolExecutor,
    _approve_current_pause,
    _r57_env_profile,
)
from tests.unit.runtime.nexus.tools.test_gr10_r8_orchestration_inner_guard import (
    _RecordingGuard,
)
from intergrax.applications._shared.uca6c_codecraft_qualified_execution_composition import (
    bootstrap_uca6c_code_exec_catalog_tools,
)
from intergrax.runtime.codecraft.qualified_capability_execution_wiring import (
    build_codecraft_qualified_capability_execution_composition,
)
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications._shared.agent_runtime_governance_wiring import (
    capability_grants_from_application_manifest,
)
from intergrax.runtime.execution.execution_bound_catalog_tool_composition import (
    ExecutionBoundCatalogToolComposition,
    build_execution_bound_catalog_tool_composition,
)
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.sandbox.durable_sandbox_wiring_binding_resolver import (
    as_durable_wiring_binding_resolver,
)
from intergrax.runtime.sandbox.isolation_gate import sandbox_availability_provider
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.runtime.wiring.agent_runtime_governance_factory import (
    build_agent_runtime_governance_boundary,
)
from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID
from intergrax.contracts.idempotency_store import IdempotencyStore
from typing import cast

pytestmark = pytest.mark.unit

_SCHEMA_PREFIX = "autonomous_work_test_"
_NOW = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clear_active_task_registry() -> None:
    ActiveTaskRegistry.clear_for_tests()


@dataclass
class _CounterBundle:
    fulfillment: int = 0
    discovery: int = 0
    acquisition: int = 0
    qualification: int = 0
    binding: int = 0
    root_launch: int = 0
    backend: int = 0


@dataclass
class _GovernedStack:
    wiring: WorkerRecoveryGovernedFulfillmentWiring
    root_launcher: _CountingRootLauncher
    acquisition: _CraftAlignedAcquisitionStrategy
    qualification: _ArtifactQualification
    discovery: _StaticDiscovery
    binding: _CountingCodeCraftBindingProvider
    async_fulfillment: _CountingAsyncRecoveryFulfillment
    handler: object
    composition: ExecutionBoundCatalogToolComposition
    hitl: InternalOrchestrationContinuation
    checkpoint_store: TaskCheckpointPersistence
    backend: _CountingToolExecutor
    craft_id: str
    task: Task
    run_id: RunId
    attempt_id: AttemptId
    terminal_outcome_store: ExecutionTerminalOutcomeByExecutionIdStore
    need_repo: WorkerRecoveryObstacleCapabilityNeedStorePort


@dataclass
class _HostGraph:
    stack: _GovernedStack
    service: WorkerRecoveryOrchestrationService
    outcome_reader: CanonicalExecutionOutcomeReader
    orch_ctx: dict[str, WorkerInstanceRepository | object]
    continuation_deps: ExecutionEngineContinuationDependencies
    need_repo: WorkerRecoveryObstacleCapabilityNeedStorePort
    checkpoint_store: TaskCheckpointPersistence
    host_label: str


@dataclass(frozen=True)
class _AssertionScalars:
    task_id: str
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    continuation_id: str
    suspended_operation_id: str


def _aligned_worker_need():
    return replace(
        _worker_need(),
        worker_instance_id=_WORKER_ID,
        obstacle_id=f"{_WORKER_ID}:obstacle:uca6b-1",
    )


def _missing_capability_discovery() -> _StaticDiscovery:
    need = _aligned_worker_need()
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    from intergrax.contracts.capability_catalog.discovery_completion import (
        build_discovery_completion,
    )
    from intergrax.contracts.capability_catalog.federation import (
        CapabilityCatalogFederationCompleteness,
    )
    from intergrax.contracts.capability_catalog.discovery_completion import (
        DiscoveryCompletionOutcome,
    )

    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-r59-r1-restart",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.MISSING_CAPABILITY
    return _StaticDiscovery(completion)


def _seed_document_need(
    need_repo: WorkerRecoveryObstacleCapabilityNeedStorePort,
    principal_repo: InMemoryWorkerPrincipalBindingRepository,
) -> None:
    need = _aligned_worker_need()
    need_repo.record_obstacle_capability_need(need)
    principal_repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=_WORKER_ID,
            tenant_id=_TENANT,
        ),
    )


def _seed_aw_bundle(bundle: AutonomousWorkRepositories) -> None:
    definition = contract_suite.worker_definition()
    bundle.worker_definition.create(definition)
    worker = contract_suite.worker_instance(
        worker_instance_id=_WORKER_ID,
        worker_definition_id=definition.worker_definition_id,
        lifecycle_state=WorkerLifecycleState.WORKING,
    )
    bundle.worker_instance.create(worker)
    responsibility = contract_suite.responsibility(
        responsibility_id=_RESP_ID,
        worker_instance_id=_WORKER_ID,
    )
    bundle.responsibility.create(responsibility)
    goal = contract_suite.worker_goal(
        goal_id=_GOAL_ID,
        responsibility_id=_RESP_ID,
    )
    bundle.worker_goal.create(goal)
    bundle.work_continuity_state.create(
        contract_suite.continuity_state(worker_instance_ref=_WORKER_ID),
    )
    bundle.worker_principal_binding.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=_WORKER_ID,
            tenant_id=_TENANT,
        ),
    )


def _build_handler_for_restart(
    tmp_path: Path,
    backends: Uca6cTrueRestartDurableBackends,
    *,
    continuation_deps: ExecutionEngineContinuationDependencies,
    terminal_outcome_store: ExecutionTerminalOutcomeByExecutionIdStore,
    craft_id: str,
    sandbox_manager: SandboxSessionManager,
):
    ctx = _codecraft_context(
        tmp_path,
        craft_id,
        sandbox_manager=sandbox_manager,
    )
    manifest = uca6c_strict_worker_manifest()
    registry = uca6c_strict_worker_registry(manifest)
    tool_wiring = _strict_tool_wiring(ctx)
    bootstrap_uca6c_code_exec_catalog_tools(tool_wiring)
    env = _r57_env_profile()
    grants = capability_grants_from_application_manifest(
        manifest,
        tenant_id=_TENANT,
        agent_registry=registry,
    )
    guard = _RecordingGuard(allow=True)
    resolved_policy = wire_policy_bundle(env)
    checkpoint_store = backends.fresh_checkpoint_store()
    idempotency_store = cast(IdempotencyStore, backends.fresh_idempotency_store())
    base_executor = RegistryToolExecutor(registry=tool_wiring.registry)
    counting_executor = _CountingToolExecutor(base_executor)
    composition = build_execution_bound_catalog_tool_composition(
        registry=tool_wiring.registry,
        policy_bundle=resolved_policy,
        caller_agent_id="worker-uca6c-qualified",
        sandbox_availability=sandbox_availability_provider(tool_wiring.wiring_context),
        production_mode=True,
        scope_policy=StaticToolScopePolicy(allowed_tools={CODE_EXEC_TOOL_ID}),
        agent_runtime_governance=build_agent_runtime_governance_boundary(
            capability_grants=grants,
        ),
        canonical_inner_execution_guard=guard,
        meaningful_side_effect_authorization=_AllowingMsePort(),
        document_store=backends.document_store,
        continuation_dependencies=continuation_deps,
        reentry_claim_owner_id="uca6c:worker-uca6c-qualified",
        durable_wiring_binding_resolver=as_durable_wiring_binding_resolver(
            sandbox_manager,
        ),
        task_checkpoint_store=checkpoint_store,
        idempotency_store=idempotency_store,
        tool_executor=counting_executor,
        terminal_outcome_store=terminal_outcome_store,
    )
    codecraft_composition = build_codecraft_qualified_capability_execution_composition(
        tool_wiring.wiring_context,
        catalog_tool_invoker=composition.invoker,
        side_effect_recorder=[],
    )
    handler = codecraft_composition.handler
    hitl = InternalOrchestrationContinuation(
        port=continuation_deps.continuation,
        lifecycle_driver=continuation_deps.lifecycle_driver,
        suspended_work_reentry_coordinator=composition.suspended_work_reentry_coordinator,
    )
    return (
        handler,
        composition,
        craft_id,
        hitl,
        checkpoint_store,
        counting_executor,
        tool_wiring.wiring_context,
    )


def _build_governed_stack(
    tmp_path: Path,
    *,
    tool_wiring_context,
    craft_id: str,
    handler,
    terminal_outcome_store: ExecutionTerminalOutcomeByExecutionIdStore,
    need_repo: WorkerRecoveryObstacleCapabilityNeedStorePort,
    principal_repo: WorkerPrincipalBindingRepository,
    composition: ExecutionBoundCatalogToolComposition,
    hitl: InternalOrchestrationContinuation,
    checkpoint_store: TaskCheckpointPersistence,
    backend: _CountingToolExecutor,
) -> _GovernedStack:
    artifact = artifact_reference_for_craft(craft_id)
    dispatch, delegate, inner_launcher = (
        build_qualified_capability_execution_dispatch_service(
            handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry(
                (handler,)
            ),
            runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
            terminal_outcome_store=terminal_outcome_store,
        )
    )
    root_launcher = _CountingRootLauncher(inner_launcher)
    dispatch = QualifiedCapabilityExecutionDispatchService(
        root_execution_launcher=root_launcher,
        runtime_delegate=delegate,
    )
    binding_inner = CodeCraftQualifiedCapabilityBindingProvider(tool_wiring_context)
    binding_provider = _CountingCodeCraftBindingProvider(inner=binding_inner)
    binding = QualifiedCapabilityBindingService((binding_provider,))
    discovery = _missing_capability_discovery()
    strategy = _CraftAlignedAcquisitionStrategy(
        strategy_id="codecraft.synthesis.v1",
        artifact_reference=artifact,
    )
    acquisition = CapabilityAcquisitionService(
        strategies=(strategy,),
        authorization=PermitCapabilityAcquisitionAuthorizationPort(),
    )
    qualification = _ArtifactQualification(artifact=artifact)
    recovery = WorkerCapabilityRecoveryCoordinator(
        discovery=discovery,
        acquisition=acquisition,
        qualification=qualification,
    )
    authority_admission = build_worker_execution_admission_for_uca6c(
        worker_instance_id=_WORKER_ID,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    direct_reuse = WorkerCapabilityDirectReuseFulfillmentService(
        binding=HostAvailableCapabilityBindingService(()),
        execution=_RecordingExecutionPort(),
        authority_admission=authority_admission,
    )
    profile_resolver = StaticWorkerCapabilityProfileResolver(
        permissive_capability_policy(_PROFILE),
    )
    wiring = build_worker_recovery_governed_fulfillment_wiring(
        recovery=recovery,
        direct_reuse=direct_reuse,
        inner_dispatch=dispatch,
        binding=binding,
        obstacle_capability_need_reader=need_repo,
        task_context_reader=ActiveTaskRegistryFulfillmentTaskContextReader(),
        principal_binding_repository=principal_repo,
        capability_profile_resolver=profile_resolver,
        authority_admission=authority_admission,
    )
    async_fulfillment = _CountingAsyncRecoveryFulfillment(
        inner=wiring.fulfillment_async
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task = Task(tenant_id=_TENANT, user_id="u1", message="r59-r1", task_id=_TASK_ID)
    return _GovernedStack(
        wiring=wiring,
        root_launcher=root_launcher,
        acquisition=strategy,
        qualification=qualification,
        discovery=discovery,
        binding=binding_provider,
        async_fulfillment=async_fulfillment,
        handler=handler,
        composition=composition,
        hitl=hitl,
        checkpoint_store=checkpoint_store,
        backend=backend,
        craft_id=craft_id,
        task=task,
        run_id=run_id,
        attempt_id=attempt_id,
        terminal_outcome_store=terminal_outcome_store,
        need_repo=need_repo,
    )


def _build_orchestration_service(
    stack: _GovernedStack,
    bundle: AutonomousWorkRepositories,
    *,
    outcome_reader: CanonicalExecutionOutcomeReader,
) -> tuple[WorkerRecoveryOrchestrationService, dict[str, object]]:
    dispatch = RecordingRecoveryDispatchPort()
    human = RecordingHumanDecisionPort()
    service = WorkerRecoveryOrchestrationService(
        episode_repository=bundle.worker_recovery_episode,
        worker_instance_repository=bundle.worker_instance,
        worker_goal_repository=bundle.worker_goal,
        continuity_repository=bundle.work_continuity_state,
        lifecycle_service=WorkerLifecycleService(
            repository=bundle.worker_instance,
            clock=_clock,
        ),
        dispatch_port=dispatch,
        human_decision_port=human,
        recovery_capability_fulfillment_port=stack.wiring.fulfillment_sync,
        recovery_capability_fulfillment_async_port=stack.async_fulfillment,
        recovery_capability_fulfillment_request_builder=stack.wiring.request_builder,
        execution_outcome_reader=outcome_reader,
        clock=_clock,
    )
    return service, {
        "dispatch": dispatch,
        "episode_repo": bundle.worker_recovery_episode,
        "worker_repo": bundle.worker_instance,
    }


def _orch_request_for_stack(stack: _GovernedStack):
    need = _aligned_worker_need()
    decision = _recovery_decision(need)
    return _orchestration_request(
        decision=decision,
        resume_target=_resume_target(
            run_id=stack.run_id,
            requested_scopes=(_READ,),
        ),
    )


def _counters(stack: _GovernedStack) -> _CounterBundle:
    backend = stack.backend
    backend_calls = backend.calls if backend is not None else 0
    return _CounterBundle(
        fulfillment=stack.async_fulfillment.calls,
        discovery=stack.discovery.calls,
        acquisition=stack.acquisition.calls,
        qualification=stack.qualification.calls,
        binding=stack.binding.bind_calls,
        root_launch=stack.root_launcher.launch_count,
        backend=backend_calls,
    )


def _build_host_graph(
    tmp_path: Path,
    backends: Uca6cTrueRestartDurableBackends,
    bundle: AutonomousWorkRepositories,
    *,
    continuation_deps: ExecutionEngineContinuationDependencies,
    host_label: str,
    seed_document_need: bool,
) -> _HostGraph:
    terminal_outcome_store = backends.fresh_terminal_outcome_store()
    craft_id = f"craft-r59-r1-{host_label}"
    sandbox_manager = SandboxSessionManager(root=tmp_path)
    (
        handler,
        composition,
        craft_id,
        hitl,
        checkpoint_store,
        backend,
        tool_wiring_context,
    ) = _build_handler_for_restart(
        tmp_path,
        backends,
        continuation_deps=continuation_deps,
        terminal_outcome_store=terminal_outcome_store,
        craft_id=craft_id,
        sandbox_manager=sandbox_manager,
    )
    need_repo = backends.fresh_document_need_repository()
    if seed_document_need:
        principal_repo = InMemoryWorkerPrincipalBindingRepository()
        _seed_document_need(need_repo, principal_repo)
        principal_binding = principal_repo
    else:
        principal_binding = bundle.worker_principal_binding
    stack = _build_governed_stack(
        tmp_path,
        tool_wiring_context=tool_wiring_context,
        craft_id=craft_id,
        handler=handler,
        terminal_outcome_store=terminal_outcome_store,
        need_repo=need_repo,
        principal_repo=principal_binding,
        composition=composition,
        hitl=hitl,
        checkpoint_store=checkpoint_store,
        backend=backend,
    )
    outcome_reader = build_canonical_execution_outcome_reader(terminal_outcome_store)
    service, orch_ctx = _build_orchestration_service(
        stack,
        bundle,
        outcome_reader=outcome_reader,
    )
    return _HostGraph(
        stack=stack,
        service=service,
        outcome_reader=outcome_reader,
        orch_ctx=orch_ctx,
        continuation_deps=continuation_deps,
        need_repo=need_repo,
        checkpoint_store=checkpoint_store,
        host_label=host_label,
    )


async def _register_task(stack: _GovernedStack) -> None:
    await ActiveTaskRegistry.register(stack.task, stack.run_id)


def _destroy_host(host: _HostGraph | None) -> None:
    del host
    ActiveTaskRegistry.clear_for_tests()


def _clear_process_globals() -> None:
    ActiveTaskRegistry.clear_for_tests()
    assert peek_governed_execution_task() is None


def _load_task_from_durable_checkpoint(
    checkpoint_store: TaskCheckpointPersistence,
    *,
    expected_task_id: str,
) -> tuple[Task, TaskCheckpoint]:
    checkpoint = checkpoint_store.get_latest(expected_task_id, _TENANT)
    assert checkpoint is not None, "durable task checkpoint missing after Host A pause"
    return build_checkpoint_resume_task(checkpoint), checkpoint


def _orch_request_from_durable_worker_recovery_episode(
    bundle: AutonomousWorkRepositories,
) -> object:
    need = _aligned_worker_need()
    decision = _recovery_decision(need)
    episode_id = derive_recovery_episode_id(
        worker_instance_id=_WORKER_ID,
        obstacle_id=decision.obstacle_id,
        recovery_decision_id=decision.decision_id,
    )
    episode_repo = bundle.worker_recovery_episode
    episode = episode_repo.get(recovery_episode_id=episode_id)
    assert episode is not None, "durable WorkerRecoveryEpisode missing after Host A"
    assert episode.resume_target.run_id is not None, (
        "durable resume target missing canonical run correlation"
    )
    return replace(
        _orchestration_request(decision=decision),
        resume_target=episode.resume_target,
    )


def _open_aw_bundle():
    if resolve_postgresql_config() is None:
        pytest.skip(
            "PostgreSQL DSN required for durable WorkerRecoveryEpisode (Host C reconciliation)",
        )
    schema_name = f"{_SCHEMA_PREFIX}{uuid.uuid4().hex}"
    return schema_name, open_bundle(schema_name)


@pytest.mark.asyncio
async def test_true_restart_worker_governed_host_abc(tmp_path: Path) -> None:
    schema_name, bundle = _open_aw_bundle()
    backends = Uca6cTrueRestartDurableBackends.create(tmp_path)
    backends.aw_schema_name = schema_name
    _seed_aw_bundle(bundle)
    host_a: _HostGraph | None = None
    host_b: _HostGraph | None = None
    host_c: _HostGraph | None = None
    scalars: _AssertionScalars | None = None
    try:
        host_a = _build_host_graph(
            tmp_path,
            backends,
            bundle,
            continuation_deps=backends.host_a_continuation_dependencies(),
            host_label="a",
            seed_document_need=True,
        )
        await _register_task(host_a.stack)
        orch_request = _orch_request_for_stack(host_a.stack)
        gov_token = bind_active_execution_governance_identity(
            ActiveExecutionGovernanceIdentity(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                principal_id=_PRINCIPAL,
            ),
        )
        try:
            first = await host_a.service.orchestrate(orch_request)
            assert (
                first.disposition
                is WorkerRecoveryOrchestrationDisposition.ATTEMPT_DISPATCHED
            )
            execution_id = first.episode.last_execution_id
            assert execution_id is not None
            counters_a = _counters(host_a.stack)
            assert counters_a.fulfillment == 1
            assert counters_a.discovery == 1
            assert counters_a.acquisition == 1
            assert counters_a.qualification == 1
            assert counters_a.binding == 1
            assert counters_a.root_launch == 1
            assert counters_a.backend == 0
            worker_repo = host_a.orch_ctx["worker_repo"]
            assert isinstance(worker_repo, WorkerInstanceRepository)
            worker = worker_repo.get(worker_instance_id=_WORKER_ID)
            assert worker is not None
            assert worker.lifecycle_state is WorkerLifecycleState.WAITING_EXTERNAL
            continuation_id, descriptor = _continuation_for_worker_pause(
                host_a.stack.task,
                host_a.stack.composition,
            )
            checkpoint_a = host_a.checkpoint_store.get_latest(
                str(_TASK_ID),
                _TENANT,
            )
            assert checkpoint_a is not None
            assert checkpoint_a.runtime is not None
            cp_run_id, cp_attempt_id = execution_identity_from_checkpoint(
                checkpoint_a,
            )
            cp_execution_id = root_execution_id_from_tree(
                checkpoint_a.runtime.execution_tree,
            )
            assert cp_run_id == host_a.stack.run_id
            assert cp_execution_id == execution_id
            assert str(checkpoint_a.task_id) == str(_TASK_ID)
            scalars = _AssertionScalars(
                task_id=str(_TASK_ID),
                run_id=cp_run_id,
                attempt_id=cp_attempt_id,
                execution_id=execution_id,
                continuation_id=continuation_id,
                suspended_operation_id=str(descriptor.suspended_operation_id),
            )
            assert (
                host_a.need_repo.get_obstacle_capability_need(
                    worker_instance_id=_WORKER_ID,
                    obstacle_id=f"{_WORKER_ID}:obstacle:uca6b-1",
                )
                is not None
            )
            assert cp_execution_id == scalars.execution_id
            assert cp_attempt_id == scalars.attempt_id
            backends.seal_continuation_for_process_death()
        finally:
            reset_active_execution_governance_identity(gov_token)

        host_a_id = id(host_a.service)
        _destroy_host(host_a)
        host_a = None
        _clear_process_globals()
        bundle.close()

        bundle = open_bundle(schema_name)

        host_b = _build_host_graph(
            tmp_path,
            backends,
            bundle,
            continuation_deps=backends.host_b_continuation_dependencies(),
            host_label="b",
            seed_document_need=False,
        )
        assert id(host_b.service) != host_a_id
        task_b, checkpoint_b = _load_task_from_durable_checkpoint(
            host_b.checkpoint_store,
            expected_task_id=scalars.task_id,
        )
        assert str(task_b.task_id) == scalars.task_id
        pending = host_b.stack.hitl.port.get_pending(
            ExecutionContinuationLookup(continuation_id=scalars.continuation_id),
        )
        assert pending is not None
        reentry = host_b.stack.composition.suspended_work_reentry_coordinator
        assert reentry is not None
        descriptor_b = reentry.store.load_active_for_continuation(
            scalars.continuation_id
        )
        assert descriptor_b is not None
        assert (
            str(descriptor_b.suspended_operation_id) == scalars.suspended_operation_id
        )
        continuation_store = backends.continuation_state_persistence.load_state_store()
        identity_b = correlate_durable_restart_execution_identity(
            checkpoint=checkpoint_b,
            continuation_store=continuation_store,
            continuation_id=scalars.continuation_id,
            suspended_descriptor=descriptor_b,
        )
        restored_run_id = identity_b.run_id
        restored_attempt_id = identity_b.attempt_id
        restored_execution_id = identity_b.execution_id
        checkpoint_run_id, checkpoint_attempt_id = execution_identity_from_checkpoint(
            checkpoint_b,
        )
        assert restored_run_id == checkpoint_run_id
        assert restored_attempt_id == checkpoint_attempt_id
        assert restored_run_id == scalars.run_id
        assert restored_attempt_id == scalars.attempt_id
        assert restored_execution_id == scalars.execution_id
        await ActiveTaskRegistry.register(task_b, restored_run_id)
        host_b.stack = replace(host_b.stack, task=task_b)
        gov_token_b = bind_active_execution_governance_identity(
            ActiveExecutionGovernanceIdentity(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                principal_id=_PRINCIPAL,
            ),
        )
        try:
            for _ in range(6):
                if host_b.stack.backend.calls >= 1:
                    break
                if host_b.stack.task.state is not TaskState.WAITING_FOR_HUMAN:
                    break
                continuation_id, descriptor_b = _continuation_for_worker_pause(
                    host_b.stack.task,
                    host_b.stack.composition,
                )
                assert isinstance(descriptor_b, SuspendedExecutionOperationDescriptor)
                id_token = bind_active_execution_identity(
                    run_id=descriptor_b.identity.run_id,
                    attempt_id=descriptor_b.identity.attempt_id,
                    execution_id=restored_execution_id,
                )
                try:
                    _approve_current_pause(
                        host_b.stack.task,
                        hitl=host_b.stack.hitl,
                        continuation_id=continuation_id,
                        run_id=descriptor_b.identity.run_id,
                        attempt_id=descriptor_b.identity.attempt_id,
                        execution_id=restored_execution_id,
                        checkpoint_store=host_b.checkpoint_store,
                    )
                finally:
                    reset_active_execution_identity(id_token)
        finally:
            reset_active_execution_governance_identity(gov_token_b)
        counters_b = _counters(host_b.stack)
        assert counters_b.root_launch == 0
        assert counters_b.backend == 1
        assert counters_b.fulfillment == 0
        assert counters_b.discovery == 0
        terminal = host_b.outcome_reader.get_terminal_outcome(restored_execution_id)
        assert terminal.disposition is CanonicalExecutionTerminalDisposition.SUCCEEDED
        _destroy_host(host_b)
        host_b = None
        _clear_process_globals()
        bundle.close()

        bundle = open_bundle(schema_name)
        host_c = _build_host_graph(
            tmp_path,
            backends,
            bundle,
            continuation_deps=backends.host_b_continuation_dependencies(),
            host_label="c",
            seed_document_need=False,
        )
        orch_request_c = _orch_request_from_durable_worker_recovery_episode(bundle)
        host_c_run_id = orch_request_c.resume_target.run_id
        assert host_c_run_id is not None
        assert host_c_run_id == scalars.run_id
        gov_token_c = bind_active_execution_governance_identity(
            ActiveExecutionGovernanceIdentity(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                principal_id=_PRINCIPAL,
            ),
        )
        try:
            final = await host_c.service.orchestrate(orch_request_c)
            assert final.disposition is WorkerRecoveryOrchestrationDisposition.RESUMED
            assert final.episode.last_execution_id == scalars.execution_id
        finally:
            reset_active_execution_governance_identity(gov_token_c)
        counters_c = _counters(host_c.stack)
        assert counters_c.fulfillment == 0
        assert counters_c.root_launch == 0
        assert counters_c.backend == 0
        assert peek_governed_execution_task() is None
    finally:
        if bundle is not None:
            bundle.close()
        drop_schema(schema_name)
