# © Artur Czarnecki. All rights reserved.

"""Shared dual-host harness for UCA-6C-R6-R5.9-R3 crash window proofs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import BaseModel

from intergrax.applications._shared.uca6c_codecraft_qualified_execution_composition import (
    bootstrap_uca6c_code_exec_catalog_tools,
)
from intergrax.contracts.execution.crash_injection import (
    ExecutionSuspendedWorkReentryCrashInjectionPort,
    ToolRuntimeEffectCrashInjectionPort,
)
from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdDisposition,
    ExecutionTerminalOutcomeByExecutionIdStore,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.runtime.codecraft.qualified_capability_execution_wiring import (
    build_codecraft_qualified_capability_execution_composition,
)
from intergrax.runtime.execution.continuation.composition import (
    ExecutionEngineContinuationDependencies,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from testing_support.uca6c_r6_r5_9_r1_true_restart_durable_backends import (
    Uca6cTrueRestartDurableBackends,
)
from intergrax.runtime.execution.execution_bound_catalog_tool_composition import (
    ExecutionBoundCatalogToolComposition,
    build_execution_bound_catalog_tool_composition,
)
from intergrax.runtime.execution.suspended_operation.document_store_suspended_operation_store import (
    reconnect_document_store_suspended_operation_store,
)
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
)
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.sandbox.isolation_gate import sandbox_availability_provider
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.runtime.wiring.agent_runtime_governance_factory import (
    build_agent_runtime_governance_boundary,
)
from intergrax.applications._shared.agent_runtime_governance_wiring import (
    capability_grants_from_application_manifest,
)
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID
from intergrax.tools.tool_executor import ToolExecutor
from intergrax.contracts.idempotency_store import IdempotencyStore
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TENANT,
)
from tests.unit.autonomous_work.test_uca6c_r5_r2_strict_governance_composition import (
    _codecraft_context,
    _strict_r6_kwargs,
    _strict_tool_wiring,
)
from tests.unit.autonomous_work.uca6c_r5_r2_strict_fixtures import (
    uca6c_strict_worker_manifest,
    uca6c_strict_worker_registry,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r2_r1_canonical_reentry_fencing import (
    DualHostReentryFixture,
    ReentryFenceCounters,
    _counting_terminal_store,
    _declarative_policy_bundle,
    _DeclarativeRequireHitlOnceHandler,
    _MseRequireHumanOncePort,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r2_multi_host_fencing import (
    OWNER_HOST_A,
    OWNER_HOST_B,
)
from tests.unit.runtime.nexus.tools.test_gr10_r8_orchestration_inner_guard import (
    _RecordingGuard,
)

OWNER_HOST_A_CRASH = OWNER_HOST_A
OWNER_HOST_B_CRASH = OWNER_HOST_B


@dataclass
class R59DurableCrashHarnessKit:
    """Rebuild inputs for process-equivalent Host B after Host A process death."""

    tmp_path: Path
    tool_wiring: object
    policy: object
    shared_mse: _MseRequireHumanOncePort
    shared_backend: SharedPhysicalEffectExecutor
    terminal_store: ExecutionTerminalOutcomeByExecutionIdStore
    durable_backends: Uca6cTrueRestartDurableBackends
    durable_wiring_binding_resolver: object | None
    host_b_reentry_crash: ExecutionSuspendedWorkReentryCrashInjectionPort | None
    host_b_tool_crash: ToolRuntimeEffectCrashInjectionPort | None
    fresh_document_store: Callable[[], ConditionalDocumentStore] | None = None


@dataclass
class CrashWindowCounters:
    claim_attempts: int = 0
    claim_successes: int = 0
    reclaim_attempts: int = 0
    reclaim_successes: int = 0
    reentry_attempts: int = 0
    toolruntime_physical_attempts: int = 0
    backend_physical_attempts: int = 0
    backend_logical_effects: int = 0
    idempotency_replays: int = 0
    mark_consumed_attempts: int = 0
    mark_consumed_applied: int = 0
    terminal_writes: int = 0


class SharedPhysicalEffectExecutor(ToolExecutor):
    """Counts physical backend entry; optional shared logical effect counter."""

    def __init__(
        self,
        delegate: ToolExecutor,
        counters: CrashWindowCounters,
    ) -> None:
        self._delegate = delegate
        self._counters = counters

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        self._counters.backend_physical_attempts += 1
        self._counters.toolruntime_physical_attempts += 1
        result = self._delegate.execute(request)
        self._counters.backend_logical_effects += 1
        return result


def build_crash_dual_host_fixture(
    tmp_path: Path,
    *,
    host_a_reentry_crash: ExecutionSuspendedWorkReentryCrashInjectionPort | None = None,
    host_b_reentry_crash: ExecutionSuspendedWorkReentryCrashInjectionPort | None = None,
    host_a_tool_crash: ToolRuntimeEffectCrashInjectionPort | None = None,
    host_b_tool_crash: ToolRuntimeEffectCrashInjectionPort | None = None,
    shared_idempotency: IdempotencyStore | None = None,
) -> tuple[DualHostReentryFixture, CrashWindowCounters, InMemoryIdempotencyStore]:
    fence_counters = ReentryFenceCounters()
    crash_counters = CrashWindowCounters()
    r6_kwargs = _strict_r6_kwargs(tmp_path)
    terminal_backend = InMemoryDocumentStore()
    terminal_store = _counting_terminal_store(terminal_backend, fence_counters)
    idempotency = (
        shared_idempotency
        if isinstance(shared_idempotency, InMemoryIdempotencyStore)
        else InMemoryIdempotencyStore()
    )
    if shared_idempotency is None:
        shared_idempotency = idempotency
    craft_id = "craft-r59r3-crash"
    ctx = _codecraft_context(
        tmp_path,
        craft_id,
        sandbox_manager=r6_kwargs["sandbox_session_manager"],
    )
    tool_wiring = _strict_tool_wiring(ctx)
    bootstrap_uca6c_code_exec_catalog_tools(tool_wiring)
    counting = _DeclarativeRequireHitlOnceHandler()
    shared_mse = _MseRequireHumanOncePort()
    policy = _declarative_policy_bundle(
        always_require_hitl=False,
        counting_handler=counting,
    )
    base_executor = RegistryToolExecutor(registry=tool_wiring.registry)
    shared_backend = SharedPhysicalEffectExecutor(base_executor, crash_counters)

    def _host(
        claim_owner: str,
        reentry_crash: ExecutionSuspendedWorkReentryCrashInjectionPort | None,
        tool_crash: ToolRuntimeEffectCrashInjectionPort | None,
        guard: _RecordingGuard,
    ) -> tuple[
        ExecutionBoundCatalogToolComposition,
        InternalOrchestrationContinuation,
    ]:
        composition = build_execution_bound_catalog_tool_composition(
            registry=tool_wiring.registry,
            policy_bundle=policy,
            caller_agent_id="worker-uca6c-qualified",
            sandbox_availability=sandbox_availability_provider(
                tool_wiring.wiring_context
            ),
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
            meaningful_side_effect_authorization=shared_mse,
            document_store=r6_kwargs["document_store"],
            continuation_dependencies=r6_kwargs["continuation_dependencies"],
            reentry_claim_owner_id=claim_owner,
            durable_wiring_binding_resolver=r6_kwargs.get(
                "durable_wiring_binding_resolver",
            ),
            task_checkpoint_store=r6_kwargs["task_checkpoint_store"],
            idempotency_store=shared_idempotency,
            tool_executor=shared_backend,
            terminal_outcome_store=terminal_store,
            reentry_crash_injection=reentry_crash,
            tool_runtime_effect_crash_injection=tool_crash,
        )
        deps = r6_kwargs["continuation_dependencies"]
        hitl = InternalOrchestrationContinuation(
            port=deps.continuation,
            lifecycle_driver=deps.lifecycle_driver,
            suspended_work_reentry_coordinator=composition.suspended_work_reentry_coordinator,
        )
        return composition, hitl

    guard_a = _RecordingGuard(allow=True)
    guard_b = _RecordingGuard(allow=True)
    composition_a, hitl_a = _host(
        OWNER_HOST_A_CRASH,
        host_a_reentry_crash,
        host_a_tool_crash,
        guard_a,
    )
    composition_b, hitl_b = _host(
        OWNER_HOST_B_CRASH,
        host_b_reentry_crash,
        host_b_tool_crash,
        guard_b,
    )
    codecraft_a = build_codecraft_qualified_capability_execution_composition(
        tool_wiring.wiring_context,
        catalog_tool_invoker=composition_a.invoker,
        side_effect_recorder=[],
    )
    fixture = DualHostReentryFixture(
        handler_a=codecraft_a.handler,
        composition_a=composition_a,
        composition_b=composition_b,
        hitl_a=hitl_a,
        hitl_b=hitl_b,
        checkpoint_store=r6_kwargs["task_checkpoint_store"],
        document_store=r6_kwargs["document_store"],
        backend_a=shared_backend,
        backend_b=shared_backend,
        guard_a=guard_a,
        guard_b=guard_b,
        craft_id=craft_id,
        terminal_store=terminal_store,
        counters=fence_counters,
    )
    return fixture, crash_counters, idempotency


def build_r59_r4_durable_dual_host_fixture(
    tmp_path: Path,
    *,
    document_store: ConditionalDocumentStore | None = None,
    fresh_document_store: Callable[[], ConditionalDocumentStore] | None = None,
    host_a_reentry_crash: ExecutionSuspendedWorkReentryCrashInjectionPort | None = None,
    host_b_reentry_crash: ExecutionSuspendedWorkReentryCrashInjectionPort | None = None,
    host_a_tool_crash: ToolRuntimeEffectCrashInjectionPort | None = None,
    host_b_tool_crash: ToolRuntimeEffectCrashInjectionPort | None = None,
) -> tuple[
    DualHostReentryFixture,
    CrashWindowCounters,
    IdempotencyStore,
    R59DurableCrashHarnessKit,
]:
    """Dual-host fixture with SQLite idempotency, checkpoint, and continuation durability."""
    fence_counters = ReentryFenceCounters()
    crash_counters = CrashWindowCounters()
    if document_store is not None:
        durable_backends = Uca6cTrueRestartDurableBackends.create_with_document_store(
            tmp_path,
            document_store,
        )
    else:
        durable_backends = Uca6cTrueRestartDurableBackends.create(tmp_path)
    r6_kwargs = _strict_r6_kwargs(tmp_path)
    r6_kwargs["document_store"] = durable_backends.document_store
    checkpoint_store = durable_backends.fresh_checkpoint_store()
    r6_kwargs["task_checkpoint_store"] = checkpoint_store
    continuation_deps = durable_backends.host_a_continuation_dependencies()
    r6_kwargs["continuation_dependencies"] = continuation_deps
    shared_idempotency = durable_backends.fresh_idempotency_store()
    terminal_store = _counting_terminal_store(
        durable_backends.document_store,
        fence_counters,
    )
    craft_id = "craft-r59r4-distributed"
    ctx = _codecraft_context(
        tmp_path,
        craft_id,
        sandbox_manager=r6_kwargs["sandbox_session_manager"],
    )
    tool_wiring = _strict_tool_wiring(ctx)
    bootstrap_uca6c_code_exec_catalog_tools(tool_wiring)
    counting = _DeclarativeRequireHitlOnceHandler()
    shared_mse = _MseRequireHumanOncePort()
    policy = _declarative_policy_bundle(
        always_require_hitl=False,
        counting_handler=counting,
    )
    base_executor = RegistryToolExecutor(registry=tool_wiring.registry)
    shared_backend = SharedPhysicalEffectExecutor(base_executor, crash_counters)

    def _host(
        claim_owner: str,
        deps: ExecutionEngineContinuationDependencies,
        idempotency: IdempotencyStore,
        checkpoint: SQLiteTaskCheckpointStore,
        reentry_crash: ExecutionSuspendedWorkReentryCrashInjectionPort | None,
        tool_crash: ToolRuntimeEffectCrashInjectionPort | None,
        guard: _RecordingGuard,
    ) -> tuple[
        ExecutionBoundCatalogToolComposition,
        InternalOrchestrationContinuation,
    ]:
        composition = build_execution_bound_catalog_tool_composition(
            registry=tool_wiring.registry,
            policy_bundle=policy,
            caller_agent_id="worker-uca6c-qualified",
            sandbox_availability=sandbox_availability_provider(
                tool_wiring.wiring_context
            ),
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
            meaningful_side_effect_authorization=shared_mse,
            document_store=r6_kwargs["document_store"],
            continuation_dependencies=deps,
            reentry_claim_owner_id=claim_owner,
            durable_wiring_binding_resolver=r6_kwargs.get(
                "durable_wiring_binding_resolver",
            ),
            task_checkpoint_store=checkpoint,
            idempotency_store=idempotency,
            tool_executor=shared_backend,
            terminal_outcome_store=terminal_store,
            reentry_crash_injection=reentry_crash,
            tool_runtime_effect_crash_injection=tool_crash,
        )
        hitl = InternalOrchestrationContinuation(
            port=deps.continuation,
            lifecycle_driver=deps.lifecycle_driver,
            suspended_work_reentry_coordinator=composition.suspended_work_reentry_coordinator,
        )
        return composition, hitl

    guard_a = _RecordingGuard(allow=True)
    guard_b = _RecordingGuard(allow=True)
    composition_a, hitl_a = _host(
        OWNER_HOST_A_CRASH,
        continuation_deps,
        shared_idempotency,
        checkpoint_store,
        host_a_reentry_crash,
        host_a_tool_crash,
        guard_a,
    )
    composition_b, hitl_b = _host(
        OWNER_HOST_B_CRASH,
        continuation_deps,
        shared_idempotency,
        checkpoint_store,
        host_b_reentry_crash,
        host_b_tool_crash,
        guard_b,
    )
    codecraft_a = build_codecraft_qualified_capability_execution_composition(
        tool_wiring.wiring_context,
        catalog_tool_invoker=composition_a.invoker,
        side_effect_recorder=[],
    )
    fixture = DualHostReentryFixture(
        handler_a=codecraft_a.handler,
        composition_a=composition_a,
        composition_b=composition_b,
        hitl_a=hitl_a,
        hitl_b=hitl_b,
        checkpoint_store=checkpoint_store,
        document_store=r6_kwargs["document_store"],
        backend_a=shared_backend,
        backend_b=shared_backend,
        guard_a=guard_a,
        guard_b=guard_b,
        craft_id=craft_id,
        terminal_store=terminal_store,
        counters=fence_counters,
    )
    kit = R59DurableCrashHarnessKit(
        tmp_path=tmp_path,
        tool_wiring=tool_wiring,
        policy=policy,
        shared_mse=shared_mse,
        shared_backend=shared_backend,
        terminal_store=terminal_store,
        durable_backends=durable_backends,
        durable_wiring_binding_resolver=r6_kwargs.get(
            "durable_wiring_binding_resolver",
        ),
        host_b_reentry_crash=host_b_reentry_crash,
        host_b_tool_crash=host_b_tool_crash,
        fresh_document_store=fresh_document_store,
    )
    return fixture, crash_counters, shared_idempotency, kit


def seal_host_a_and_rebuild_host_b_process_equivalent(
    fixture: DualHostReentryFixture,
    kit: R59DurableCrashHarnessKit,
) -> None:
    """Simulate Host A process loss and wire Host B to durable continuation file."""
    kit.durable_backends.seal_continuation_for_process_death()
    continuation_deps = kit.durable_backends.host_b_continuation_dependencies()
    idempotency = kit.durable_backends.fresh_idempotency_store()
    checkpoint_store = kit.durable_backends.fresh_checkpoint_store()
    document_store_a = fixture.document_store
    document_store_b = fixture.document_store
    terminal_store_b = kit.terminal_store
    if kit.fresh_document_store is not None:
        host_a_store = fixture.document_store
        document_store_a = kit.fresh_document_store()
        document_store_b = kit.fresh_document_store()
        close_host_a = getattr(host_a_store, "close", None)
        if callable(close_host_a):
            close_host_a()
        terminal_store_b = _counting_terminal_store(
            document_store_b,
            fixture.counters,
        )
        kit.terminal_store = terminal_store_b
        fixture.terminal_store = terminal_store_b
        fixture.document_store = document_store_b
        fixture.document_store_a = document_store_a
        fixture.document_store_b = document_store_b
    guard_b = _RecordingGuard(allow=True)
    tool_wiring = kit.tool_wiring
    composition_b = build_execution_bound_catalog_tool_composition(
        registry=tool_wiring.registry,
        policy_bundle=kit.policy,
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
        canonical_inner_execution_guard=guard_b,
        meaningful_side_effect_authorization=kit.shared_mse,
        document_store=document_store_b,
        continuation_dependencies=continuation_deps,
        reentry_claim_owner_id=OWNER_HOST_B_CRASH,
        durable_wiring_binding_resolver=kit.durable_wiring_binding_resolver,
        task_checkpoint_store=checkpoint_store,
        idempotency_store=idempotency,
        tool_executor=kit.shared_backend,
        terminal_outcome_store=terminal_store_b,
        reentry_crash_injection=kit.host_b_reentry_crash,
        tool_runtime_effect_crash_injection=kit.host_b_tool_crash,
    )
    hitl_b = InternalOrchestrationContinuation(
        port=continuation_deps.continuation,
        lifecycle_driver=continuation_deps.lifecycle_driver,
        suspended_work_reentry_coordinator=composition_b.suspended_work_reentry_coordinator,
    )
    fixture.composition_b = composition_b
    fixture.hitl_b = hitl_b
    fixture.guard_b = guard_b
    fixture.checkpoint_store = checkpoint_store
    sync_hosts_after_restart(
        fixture,
        document_store_a=document_store_a,
        document_store_b=document_store_b,
    )


def sync_hosts_after_restart(
    fixture: DualHostReentryFixture,
    *,
    document_store_a: ConditionalDocumentStore | None = None,
    document_store_b: ConditionalDocumentStore | None = None,
) -> None:
    backing_a = document_store_a if document_store_a is not None else fixture.document_store
    backing_b = document_store_b if document_store_b is not None else fixture.document_store
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


def terminal_disposition(
    store: ExecutionTerminalOutcomeByExecutionIdStore,
    execution_id: object,
) -> ExecutionTerminalOutcomeByExecutionIdDisposition | None:
    return store.get_recorded_disposition(execution_id)
