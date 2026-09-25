# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9-R2-R1 — canonical reentry claim/fence authority vs terminal outcome."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator

import pytest

from intergrax.applications._shared.uca6c_codecraft_qualified_execution_composition import (
    bootstrap_uca6c_code_exec_catalog_tools,
)
from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdDisposition,
    ExecutionTerminalOutcomeByExecutionIdStore,
)
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationClaimOutcome,
    SuspendedOperationMutationOutcome,
)
from intergrax.contracts.execution.suspended_operation.claim_authority import (
    SuspendedOperationClaimAuthority,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.reentry import (
    ExecutionSuspendedWorkReentryDisposition,
    ExecutionSuspendedWorkReentryRequest,
)
from intergrax.contracts.execution_continuation import ExecutionContinuationLookup
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.runtime.codecraft.qualified_capability_execution_wiring import (
    build_codecraft_qualified_capability_execution_composition,
)
from intergrax.runtime.execution.document_store_execution_terminal_outcome_by_execution_id import (
    DocumentStoreExecutionTerminalOutcomeByExecutionIdStore,
)
from intergrax.runtime.execution.execution_bound_catalog_tool_composition import (
    ExecutionBoundCatalogToolComposition,
    build_execution_bound_catalog_tool_composition,
)
from intergrax.runtime.execution.suspended_operation.document_store_suspended_operation_store import (
    reconnect_document_store_suspended_operation_store,
)
from intergrax.runtime.execution.suspended_operation.reentry_coordinator import (
    ExecutionSuspendedWorkReentryCoordinator,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    bind_governed_execution_task,
    reset_governed_execution_task,
)
from intergrax.runtime.human.agent_governance_human_approval_grant import (
    AgentGovernanceHumanApprovalGrantCoordinator,
)
from intergrax.runtime.human.declarative_hitl_grant import (
    DeclarativeHitlGrantCoordinator,
)
from intergrax.runtime.human.governed_continuation_grant import (
    GovernedContinuationGrantCoordinator,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
    canonical_resume_after_authorization,
)
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.sandbox.isolation_gate import sandbox_availability_provider
from intergrax.runtime.task.task import Task
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.runtime.wiring.agent_runtime_governance_factory import (
    build_agent_runtime_governance_boundary,
)
from intergrax.applications._shared.agent_runtime_governance_wiring import (
    capability_grants_from_application_manifest,
)
from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.autonomous_work.test_uca6c_r5_r2_strict_governance_composition import (
    _codecraft_context,
    _strict_r6_kwargs,
    _strict_tool_wiring,
)
from tests.unit.autonomous_work.uca6c_r5_r2_strict_fixtures import (
    uca6c_strict_r6_durable_wiring,
    uca6c_strict_worker_manifest,
    uca6c_strict_worker_registry,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r2_multi_host_fencing import (
    OWNER_HOST_A,
    OWNER_HOST_B,
    advance_lease_clock,
    reclaim_as,
)
from tests.unit.runtime.execution.test_uca6c_r6_r5_7_sequential_authority_generations import (
    _CountingToolExecutor,
    _DeclarativeRequireHitlOnceHandler,
    _approve_current_pause,
    _declarative_policy_bundle,
    _MseRequireHumanOncePort,
    _start_gen1_pause,
)
from tests.unit.runtime.nexus.tools.test_gr10_r8_orchestration_inner_guard import (
    _RecordingGuard,
)

pytestmark = pytest.mark.unit


@dataclass
class ReentryFenceCounters:
    host_a_reentry_attempts: int = 0
    host_a_reentry_successes: int = 0
    host_b_reentry_attempts: int = 0
    host_b_reentry_successes: int = 0
    toolruntime_calls_a: int = 0
    toolruntime_calls_b: int = 0
    terminal_writes: int = 0


@dataclass
class DualHostReentryFixture:
    handler_a: object
    composition_a: ExecutionBoundCatalogToolComposition
    composition_b: ExecutionBoundCatalogToolComposition
    hitl_a: InternalOrchestrationContinuation
    hitl_b: InternalOrchestrationContinuation
    checkpoint_store: object
    document_store: object
    backend_a: _CountingToolExecutor
    backend_b: _CountingToolExecutor
    guard_a: _RecordingGuard
    guard_b: _RecordingGuard
    craft_id: str
    terminal_store: ExecutionTerminalOutcomeByExecutionIdStore
    counters: ReentryFenceCounters = field(default_factory=ReentryFenceCounters)
    document_store_a: object | None = None
    document_store_b: object | None = None


def _counting_terminal_store(
    backend: InMemoryDocumentStore,
    counters: ReentryFenceCounters,
) -> ExecutionTerminalOutcomeByExecutionIdStore:
    inner = DocumentStoreExecutionTerminalOutcomeByExecutionIdStore(backend)

    @dataclass(frozen=True, slots=True)
    class _CountingTerminalStore:
        _inner: DocumentStoreExecutionTerminalOutcomeByExecutionIdStore
        _counters: ReentryFenceCounters

        def record_terminal_disposition(
            self,
            execution_id: object,
            disposition: ExecutionTerminalOutcomeByExecutionIdDisposition,
        ) -> None:
            self._counters.terminal_writes += 1
            self._inner.record_terminal_disposition(execution_id, disposition)

        def get_recorded_disposition(
            self,
            execution_id: object,
        ) -> ExecutionTerminalOutcomeByExecutionIdDisposition | None:
            return self._inner.get_recorded_disposition(execution_id)

    return _CountingTerminalStore(inner, counters)


@dataclass
class _ReclaimBridge:
    descriptor: object
    store_a: object
    store_b: object
    counters: object = field(default_factory=lambda: type("C", (), {"reclaims": 0})())


def _build_host_composition(
    *,
    tmp_path: Path,
    r6_kwargs: dict[str, object],
    tool_wiring,
    mse_port: _MseRequireHumanOncePort,
    policy_bundle,
    claim_owner_id: str,
    terminal_store: DocumentStoreExecutionTerminalOutcomeByExecutionIdStore,
    guard: _RecordingGuard,
) -> tuple[
    ExecutionBoundCatalogToolComposition,
    _CountingToolExecutor,
    InternalOrchestrationContinuation,
]:
    base_executor = RegistryToolExecutor(registry=tool_wiring.registry)
    backend = _CountingToolExecutor(base_executor)
    composition = build_execution_bound_catalog_tool_composition(
        registry=tool_wiring.registry,
        policy_bundle=policy_bundle,
        caller_agent_id="worker-uca6c-qualified",
        sandbox_availability=sandbox_availability_provider(tool_wiring.wiring_context),
        production_mode=True,
        scope_policy=StaticToolScopePolicy(allowed_tools={CODE_EXEC_TOOL_ID}),
        agent_runtime_governance=build_agent_runtime_governance_boundary(
            capability_grants=capability_grants_from_application_manifest(
                uca6c_strict_worker_manifest(),
                tenant_id=_TENANT,
                agent_registry=uca6c_strict_worker_registry(
                    uca6c_strict_worker_manifest(),
                ),
            ),
        ),
        canonical_inner_execution_guard=guard,
        meaningful_side_effect_authorization=mse_port,
        document_store=r6_kwargs["document_store"],
        continuation_dependencies=r6_kwargs["continuation_dependencies"],
        reentry_claim_owner_id=claim_owner_id,
        durable_wiring_binding_resolver=r6_kwargs.get(
            "durable_wiring_binding_resolver",
        ),
        task_checkpoint_store=r6_kwargs["task_checkpoint_store"],
        idempotency_store=InMemoryIdempotencyStore(),
        tool_executor=backend,
        terminal_outcome_store=terminal_store,
    )
    deps = r6_kwargs["continuation_dependencies"]
    hitl = InternalOrchestrationContinuation(
        port=deps.continuation,
        lifecycle_driver=deps.lifecycle_driver,
        suspended_work_reentry_coordinator=composition.suspended_work_reentry_coordinator,
    )
    return composition, backend, hitl


def _build_dual_host_fixture(tmp_path: Path) -> DualHostReentryFixture:
    counters = ReentryFenceCounters()
    r6_kwargs = _strict_r6_kwargs(tmp_path)
    test_bundle = uca6c_strict_r6_durable_wiring(tmp_path)
    terminal_backend = InMemoryDocumentStore()
    terminal_store = _counting_terminal_store(terminal_backend, counters)
    craft_id = "craft-r59r2r1-reentry-fence"
    ctx = _codecraft_context(
        tmp_path,
        craft_id,
        sandbox_manager=test_bundle["sandbox_session_manager"],
    )
    tool_wiring = _strict_tool_wiring(ctx)
    bootstrap_uca6c_code_exec_catalog_tools(tool_wiring)
    counting = _DeclarativeRequireHitlOnceHandler()
    shared_mse = _MseRequireHumanOncePort()
    mse_a = shared_mse
    mse_b = shared_mse
    policy = _declarative_policy_bundle(
        always_require_hitl=False,
        counting_handler=counting,
    )
    guard_a = _RecordingGuard(allow=True)
    guard_b = _RecordingGuard(allow=True)
    composition_a, backend_a, hitl_a = _build_host_composition(
        tmp_path=tmp_path,
        r6_kwargs=r6_kwargs,
        tool_wiring=tool_wiring,
        mse_port=mse_a,
        policy_bundle=policy,
        claim_owner_id=OWNER_HOST_A,
        terminal_store=terminal_store,
        guard=guard_a,
    )
    composition_b, backend_b, hitl_b = _build_host_composition(
        tmp_path=tmp_path,
        r6_kwargs=r6_kwargs,
        tool_wiring=tool_wiring,
        mse_port=mse_b,
        policy_bundle=policy,
        claim_owner_id=OWNER_HOST_B,
        terminal_store=terminal_store,
        guard=guard_b,
    )
    codecraft_a = build_codecraft_qualified_capability_execution_composition(
        tool_wiring.wiring_context,
        catalog_tool_invoker=composition_a.invoker,
        side_effect_recorder=[],
    )
    assert composition_a.suspended_work_reentry_coordinator is not None
    assert composition_b.suspended_work_reentry_coordinator is not None
    assert id(composition_a.suspended_work_reentry_coordinator) != id(
        composition_b.suspended_work_reentry_coordinator
    )
    assert id(composition_a.suspended_work_reentry_coordinator.store) != id(
        composition_b.suspended_work_reentry_coordinator.store,
    )
    return DualHostReentryFixture(
        handler_a=codecraft_a.handler,
        composition_a=composition_a,
        composition_b=composition_b,
        hitl_a=hitl_a,
        hitl_b=hitl_b,
        checkpoint_store=r6_kwargs["task_checkpoint_store"],
        document_store=r6_kwargs["document_store"],
        backend_a=backend_a,
        backend_b=backend_b,
        guard_a=guard_a,
        guard_b=guard_b,
        craft_id=craft_id,
        terminal_store=terminal_store,
        counters=counters,
    )


def _sync_host_stores(fixture: DualHostReentryFixture) -> None:
    backing_a = fixture.document_store_a or fixture.document_store
    backing_b = fixture.document_store_b or fixture.document_store
    store_a = reconnect_document_store_suspended_operation_store(backing_a)
    store_b = reconnect_document_store_suspended_operation_store(backing_b)
    co_a = fixture.composition_a.suspended_work_reentry_coordinator
    co_b = fixture.composition_b.suspended_work_reentry_coordinator
    assert co_a is not None and co_b is not None
    new_co_a = replace(co_a, store=store_a)
    new_co_b = replace(co_b, store=store_b)
    fixture.composition_a = replace(
        fixture.composition_a,
        suspended_work_reentry_coordinator=new_co_a,
    )
    fixture.composition_b = replace(
        fixture.composition_b,
        suspended_work_reentry_coordinator=new_co_b,
    )
    fixture.hitl_a = replace(
        fixture.hitl_a,
        suspended_work_reentry_coordinator=new_co_a,
    )
    fixture.hitl_b = replace(
        fixture.hitl_b,
        suspended_work_reentry_coordinator=new_co_b,
    )


def _advance_to_gen3_blocked(
    fixture: DualHostReentryFixture,
    task: Task,
    *,
    run_id,
    attempt_id,
    execution_id,
) -> tuple[str, object]:
    store = fixture.composition_a.suspended_work_reentry_coordinator.store
    gen1 = _start_gen1_pause(fixture.handler_a, fixture.craft_id, execution_id)
    d1 = gen1.descriptor
    c1 = d1.continuation_id
    _approve_current_pause(
        task,
        hitl=fixture.hitl_a,
        continuation_id=c1,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        checkpoint_store=fixture.checkpoint_store,
    )
    d2 = store.load_active_for_logical_invocation(d1.logical_invocation_fingerprint)
    assert d2 is not None
    c2 = d2.continuation_id
    _approve_current_pause(
        task,
        hitl=fixture.hitl_a,
        continuation_id=c2,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        checkpoint_store=fixture.checkpoint_store,
    )
    d3 = store.load_active_for_logical_invocation(d1.logical_invocation_fingerprint)
    assert d3 is not None and d3.pause_generation == 3
    return d3.continuation_id, d3


def _resume_gen3_without_reentry(
    fixture: DualHostReentryFixture,
    task: Task,
    *,
    continuation_id: str,
    run_id,
    attempt_id,
    execution_id,
    hitl: InternalOrchestrationContinuation,
) -> None:
    pending = hitl.port.get_pending(
        ExecutionContinuationLookup(continuation_id=continuation_id),
    )
    pause_record = task.runtime.governance.pause_record
    human_request = task.runtime.governance.human_request
    assert pause_record is not None and human_request is not None
    if pending.governed_correlation is not None:
        task.runtime.governance.human_request = human_request.model_copy(
            update={"governed_continuation": pending.governed_correlation},
        )
    from intergrax.contracts.human_approver import local_development_approver_evidence

    approver = local_development_approver_evidence(tenant_id=task.tenant_id)
    authorized = HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
        task,
        HumanResponseVerdict.APPROVE,
        approver=approver,
        continuation=hitl.port,
        pause_id=pause_record.pause_id,
        human_request_id=human_request.request_id,
        run_id=str(run_id),
        attempt_id=str(attempt_id),
        execution_id=str(execution_id),
    )
    declarative_pending = task.runtime.governance.declarative_hitl_pending
    if (
        declarative_pending is not None
        and declarative_pending.pause_id == pause_record.pause_id
    ):
        DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
    agent_pending = task.runtime.governance.agent_governance_hitl_pending
    if (
        agent_pending is not None
        and fixture.checkpoint_store is not None
        and agent_pending.pause_id == pause_record.pause_id
    ):
        AgentGovernanceHumanApprovalGrantCoordinator.persist_available_grant_from_human_approve(
            task,
            checkpoint_store=fixture.checkpoint_store,
            approver=approver,
        )
    if task.runtime.governance.human_request is not None:
        GovernedContinuationGrantCoordinator.create_grant_from_approval(task)
    canonical_resume_after_authorization(
        task,
        authorized,
        capability=hitl,
    )


def _claim_authority_for_coordinator(
    coordinator: ExecutionSuspendedWorkReentryCoordinator,
    continuation_id: str,
) -> SuspendedOperationClaimAuthority:
    active = coordinator.store.load_active_for_continuation(continuation_id)
    if active is None:
        return SuspendedOperationClaimAuthority(
            owner_id=coordinator.claim_owner_id,
            fence=0,
            materialization_revision=0,
            pause_generation=1,
        )
    if (
        active.materialization_state is SuspendedOperationMaterializationState.CLAIMED
        and active.claim_ownership is not None
    ):
        return SuspendedOperationClaimAuthority.from_claimed_descriptor(active)
    return SuspendedOperationClaimAuthority.for_host_pending_claim(
        host_owner_id=coordinator.claim_owner_id,
        descriptor=active,
    )


def _reenter(
    coordinator: ExecutionSuspendedWorkReentryCoordinator,
    *,
    continuation_id: str,
    identity,
    task: Task | None,
    counters: ReentryFenceCounters,
    host: str,
    claim_authority: SuspendedOperationClaimAuthority | None = None,
) -> object:
    if host == "a":
        counters.host_a_reentry_attempts += 1
    else:
        counters.host_b_reentry_attempts += 1
    authority = claim_authority
    if authority is None:
        authority = _claim_authority_for_coordinator(coordinator, continuation_id)
    result = coordinator.reenter_after_resume(
        ExecutionSuspendedWorkReentryRequest(
            continuation_id=continuation_id,
            identity=identity,
            claim_authority=authority,
        ),
        task=task,
    )
    if result.disposition is ExecutionSuspendedWorkReentryDisposition.COMPLETED:
        if host == "a":
            counters.host_a_reentry_successes += 1
        else:
            counters.host_b_reentry_successes += 1
    return result


def _claim_host_a_short_lease(
    fixture: DualHostReentryFixture,
    d3,
) -> tuple[int, int, int]:
    store = fixture.composition_a.suspended_work_reentry_coordinator.store
    lease_at = datetime.now(UTC) + timedelta(minutes=5)
    claimed = store.claim(
        suspended_operation_id=d3.suspended_operation_id,
        expected_materialization_revision=d3.materialization_revision,
        owner_id=OWNER_HOST_A,
        lease_expires_at=lease_at,
    )
    assert claimed.outcome is SuspendedOperationClaimOutcome.CLAIMED
    assert claimed.descriptor is not None
    assert claimed.descriptor.claim_ownership is not None
    return (
        claimed.descriptor.claim_ownership.fence,
        claimed.descriptor.materialization_revision,
        claimed.descriptor.pause_generation,
    )


@contextmanager
def _multi_host_fixture(tmp_path: Path) -> Iterator[DualHostReentryFixture]:
    yield _build_dual_host_fixture(tmp_path)


def test_canonical_reentry_stale_host_blocked_current_host_completes(
    tmp_path: Path,
) -> None:
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT, user_id="u1", message="r59r2r1", task_id=_TASK_ID
        )
        id_token = bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        gov_token = bind_active_execution_governance_identity(
            ActiveExecutionGovernanceIdentity(
                tenant_id=_TENANT,
                workspace_id="workspace-uca6c",
                principal_id="principal-uca6c",
            ),
        )
        task_token = bind_governed_execution_task(task)
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            fence_a, revision_a, pause_gen = _claim_host_a_short_lease(fixture, d3)
            expired_now = datetime.now(UTC) + timedelta(hours=2)
            reclaim_bridge = _ReclaimBridge(
                descriptor=d3,
                store_a=fixture.composition_a.suspended_work_reentry_coordinator.store,
                store_b=fixture.composition_b.suspended_work_reentry_coordinator.store,
            )
            with advance_lease_clock(expired_now):
                reclaimed = reclaim_as(
                    reclaim_bridge,
                    "b",
                    expected_revision=revision_a,
                    owner_id=OWNER_HOST_B,
                    expected_fence=fence_a,
                    lease_at=expired_now + timedelta(minutes=5),
                )
            assert reclaimed.outcome is SuspendedOperationMutationOutcome.APPLIED
            assert reclaimed.descriptor is not None
            assert reclaimed.descriptor.claim_ownership is not None
            owner_b = reclaimed.descriptor.claim_ownership.owner_id
            fence_b = reclaimed.descriptor.claim_ownership.fence
            revision_b = reclaimed.descriptor.materialization_revision
            assert owner_b == OWNER_HOST_B
            assert fence_b > fence_a
            assert revision_b > revision_a
            assert reclaimed.descriptor.pause_generation == pause_gen
            assert reclaimed.descriptor.identity == d3.identity

            _sync_host_stores(fixture)

            _resume_gen3_without_reentry(
                fixture,
                task,
                continuation_id=c3,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                hitl=fixture.hitl_a,
            )
            root_guard_calls = fixture.guard_a.calls
            reentry_a = fixture.composition_a.suspended_work_reentry_coordinator
            assert reentry_a is not None
            stale_authority = SuspendedOperationClaimAuthority(
                owner_id=OWNER_HOST_A,
                fence=fence_a,
                materialization_revision=revision_b,
                pause_generation=pause_gen,
            )
            stale = _reenter(
                reentry_a,
                continuation_id=c3,
                identity=d3.identity,
                task=task,
                counters=fixture.counters,
                host="a",
                claim_authority=stale_authority,
            )
            assert stale.disposition is ExecutionSuspendedWorkReentryDisposition.FAILED
            assert stale.reason_detail == "stale_claim_owner"
            assert fixture.backend_a.calls == 0
            assert fixture.backend_b.calls == 0
            assert fixture.counters.terminal_writes == 0
            terminal_before = fixture.terminal_store.get_recorded_disposition(
                execution_id,
            )
            assert terminal_before is None

            reentry_b = fixture.composition_b.suspended_work_reentry_coordinator
            assert reentry_b is not None
            current = _reenter(
                reentry_b,
                continuation_id=c3,
                identity=d3.identity,
                task=task,
                counters=fixture.counters,
                host="b",
            )
            assert (
                current.disposition
                is ExecutionSuspendedWorkReentryDisposition.COMPLETED
            )
            assert fixture.backend_b.calls == 1
            assert fixture.backend_a.calls == 0
            assert fixture.counters.terminal_writes == 1
            terminal = fixture.terminal_store.get_recorded_disposition(execution_id)
            assert (
                terminal is ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED
            )
            consumed = reentry_b.store.load(d3.suspended_operation_id)
            assert consumed is not None
            assert (
                consumed.materialization_state
                is SuspendedOperationMaterializationState.CONSUMED
            )
            assert fixture.guard_a.calls == root_guard_calls
            assert fixture.guard_b.calls >= 1
            assert fixture.counters.host_a_reentry_successes == 0
            assert fixture.counters.host_b_reentry_successes == 1
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


def test_stale_host_after_terminal_cannot_reenter(tmp_path: Path) -> None:
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT, user_id="u1", message="r59r2r1-c", task_id=_TASK_ID
        )
        id_token = bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        gov_token = bind_active_execution_governance_identity(
            ActiveExecutionGovernanceIdentity(
                tenant_id=_TENANT,
                workspace_id="workspace-uca6c",
                principal_id="principal-uca6c",
            ),
        )
        task_token = bind_governed_execution_task(task)
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            fence_a, revision_a, _ = _claim_host_a_short_lease(fixture, d3)
            expired_now = datetime.now(UTC) + timedelta(hours=2)
            reclaim_bridge = _ReclaimBridge(
                descriptor=d3,
                store_a=fixture.composition_a.suspended_work_reentry_coordinator.store,
                store_b=fixture.composition_b.suspended_work_reentry_coordinator.store,
            )
            with advance_lease_clock(expired_now):
                reclaim_as(
                    reclaim_bridge,
                    "b",
                    expected_revision=revision_a,
                    owner_id=OWNER_HOST_B,
                    expected_fence=fence_a,
                    lease_at=expired_now + timedelta(minutes=5),
                )
            _sync_host_stores(fixture)
            _resume_gen3_without_reentry(
                fixture,
                task,
                continuation_id=c3,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                hitl=fixture.hitl_b,
            )
            reentry_b = fixture.composition_b.suspended_work_reentry_coordinator
            assert reentry_b is not None
            _reenter(
                reentry_b,
                continuation_id=c3,
                identity=d3.identity,
                task=task,
                counters=fixture.counters,
                host="b",
            )
            writes_after_b = fixture.counters.terminal_writes
            _sync_host_stores(fixture)
            reentry_a = fixture.composition_a.suspended_work_reentry_coordinator
            assert reentry_a is not None
            again = _reenter(
                reentry_a,
                continuation_id=c3,
                identity=d3.identity,
                task=None,
                counters=fixture.counters,
                host="a",
            )
            assert again.disposition in {
                ExecutionSuspendedWorkReentryDisposition.FAILED,
                ExecutionSuspendedWorkReentryDisposition.NOT_READY,
                ExecutionSuspendedWorkReentryDisposition.REJECTED,
            }
            assert fixture.counters.terminal_writes == writes_after_b
            assert fixture.backend_a.calls == 0
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


def test_stale_host_with_valid_grants_still_blocked(tmp_path: Path) -> None:
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT, user_id="u1", message="r59r2r1-d", task_id=_TASK_ID
        )
        id_token = bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        gov_token = bind_active_execution_governance_identity(
            ActiveExecutionGovernanceIdentity(
                tenant_id=_TENANT,
                workspace_id="workspace-uca6c",
                principal_id="principal-uca6c",
            ),
        )
        task_token = bind_governed_execution_task(task)
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            fence_a, revision_a, pause_gen = _claim_host_a_short_lease(fixture, d3)
            expired_now = datetime.now(UTC) + timedelta(hours=2)
            reclaim_bridge = _ReclaimBridge(
                descriptor=d3,
                store_a=fixture.composition_a.suspended_work_reentry_coordinator.store,
                store_b=fixture.composition_b.suspended_work_reentry_coordinator.store,
            )
            with advance_lease_clock(expired_now):
                reclaimed = reclaim_as(
                    reclaim_bridge,
                    "b",
                    expected_revision=revision_a,
                    owner_id=OWNER_HOST_B,
                    expected_fence=fence_a,
                    lease_at=expired_now + timedelta(minutes=5),
                )
            assert reclaimed.descriptor is not None
            revision_b = reclaimed.descriptor.materialization_revision
            _sync_host_stores(fixture)
            _resume_gen3_without_reentry(
                fixture,
                task,
                continuation_id=c3,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                hitl=fixture.hitl_a,
            )
            gov = task.runtime.governance
            assert (
                gov.agent_governance_human_approval_grant is not None
                or gov.governed_continuation_grant is not None
                or gov.declarative_hitl_grant is not None
            )
            reentry_a = fixture.composition_a.suspended_work_reentry_coordinator
            assert reentry_a is not None
            stale_authority = SuspendedOperationClaimAuthority(
                owner_id=OWNER_HOST_A,
                fence=fence_a,
                materialization_revision=revision_b,
                pause_generation=pause_gen,
            )
            stale = _reenter(
                reentry_a,
                continuation_id=c3,
                identity=d3.identity,
                task=task,
                counters=fixture.counters,
                host="a",
                claim_authority=stale_authority,
            )
            assert stale.disposition is ExecutionSuspendedWorkReentryDisposition.FAILED
            assert stale.reason_detail == "stale_claim_owner"
            assert fixture.counters.terminal_writes == 0
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


def test_reentry_rejects_identity_mismatch(tmp_path: Path) -> None:
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        wrong_execution = mint_execution_id()
        task = Task(
            tenant_id=_TENANT, user_id="u1", message="r59r2r1-h", task_id=_TASK_ID
        )
        id_token = bind_active_execution_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        gov_token = bind_active_execution_governance_identity(
            ActiveExecutionGovernanceIdentity(
                tenant_id=_TENANT,
                workspace_id="workspace-uca6c",
                principal_id="principal-uca6c",
            ),
        )
        task_token = bind_governed_execution_task(task)
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            fence_a, revision_a, _ = _claim_host_a_short_lease(fixture, d3)
            expired_now = datetime.now(UTC) + timedelta(hours=2)
            reclaim_bridge = _ReclaimBridge(
                descriptor=d3,
                store_a=fixture.composition_a.suspended_work_reentry_coordinator.store,
                store_b=fixture.composition_b.suspended_work_reentry_coordinator.store,
            )
            with advance_lease_clock(expired_now):
                reclaim_as(
                    reclaim_bridge,
                    "b",
                    expected_revision=revision_a,
                    owner_id=OWNER_HOST_B,
                    expected_fence=fence_a,
                    lease_at=expired_now + timedelta(minutes=5),
                )
            _sync_host_stores(fixture)
            _resume_gen3_without_reentry(
                fixture,
                task,
                continuation_id=c3,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                hitl=fixture.hitl_a,
            )
            reentry_a = fixture.composition_a.suspended_work_reentry_coordinator
            assert reentry_a is not None
            bad_identity = replace(d3.identity, execution_id=wrong_execution)
            bad_authority = _claim_authority_for_coordinator(reentry_a, c3)
            result = reentry_a.reenter_after_resume(
                ExecutionSuspendedWorkReentryRequest(
                    continuation_id=c3,
                    identity=bad_identity,
                    claim_authority=bad_authority,
                ),
                task=None,
            )
            assert (
                result.disposition is ExecutionSuspendedWorkReentryDisposition.REJECTED
            )
            assert result.reason_detail == "identity_mismatch"
            assert fixture.counters.terminal_writes == 0
            assert fixture.backend_a.calls == 0
            assert fixture.backend_b.calls == 0
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)
