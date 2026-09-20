# © Artur Czarnecki. All rights reserved.

"""GR-10-R12 — architecture gates for ORCHESTRATION Continuation authority."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationResolutionCommand,
    ExecutionContinuationResumeCommand,
    ExecutionContinuationTransition,
    ExecutionHumanVerdict,
    ExecutionPauseRequest,
    advance_continuation_lifecycle,
)
from intergrax.contracts.execution_continuation_state_store import (
    ExecutionContinuationStateStore,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
)
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.execution.continuation.composition import (
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.execution.continuation.durability_policy import (
    validate_execution_continuation_for_composition,
)
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    InMemoryExecutionContinuationStateStore,
    ReconstructedDurableExecutionContinuationStateStore,
    backing_execution_continuation_state_store,
    export_durable_continuation_state,
    execution_continuation_state_store_from_durable_export,
    restore_durable_continuation_backing,
)
from intergrax.runtime.human.governed_continuation_bridge import (
    apply_governed_continuation_pause,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalHitlContinuationCapabilityError,
    InternalOrchestrationContinuation,
    canonical_execution_is_resumed,
    establish_canonical_hitl_pause,
)
from intergrax.runtime.nexus.retry.retry_engine import RetryPolicy
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests
from testing_support.nexus_host_task_execution import (
    build_certified_internal_test_host_task_execution,
)
from intergrax.runtime.execution.attempt_lifecycle import (
    wire_attempt_lifecycle_store,
)
from tests.unit.runtime.background_execution.reentry_admission_doubles import InMemoryKVStore

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_SEMANTIC_CONSUMERS = (
    _REPO / "intergrax/runtime/policy/mse_hitl_effect_gate.py",
    _REPO / "intergrax/runtime/nexus/orchestration/graph_runner.py",
    _REPO / "intergrax/runtime/nexus/orchestration/intake_runner.py",
    _REPO / "intergrax/runtime/nexus/orchestration/governed_consequential_operation.py",
)
_SEED = "gr10-r12"
_TASK = canonical_task_id_for_tests(_SEED)
_RUN = canonical_run_id_for_tests(_SEED)


def _identity(
    *,
    task: TaskId | None = None,
    run: RunId | None = None,
    attempt: AttemptId | None = None,
    execution: ExecutionId | None = None,
) -> ExecutionContinuationIdentity:
    return ExecutionContinuationIdentity(
        task_id=task or _TASK,
        run_id=run or _RUN,
        attempt_id=attempt or mint_attempt_id(),
        execution_id=execution or mint_execution_id(),
    )


def _pause_to_waiting(port, driver, *, continuation_id: str, identity):
    port.request_pause(
        ExecutionPauseRequest(
            identity=identity,
            continuation_id=continuation_id,
            reason=ContinuationReason.COMPLIANCE,
            pause_id=f"pause_{continuation_id}",
            human_request_id=f"hr_{continuation_id}",
        ),
    )
    driver.record_execution_reached_safe_pause(
        continuation_id,
        execution_pause_established=True,
    )
    return driver.record_ready_for_human_resolution(continuation_id)


def _approve(port, pending):
    return port.apply_resolution(
        ExecutionContinuationResolutionCommand(
            continuation_id=pending.continuation_id,
            identity=pending.identity,
            expected_revision=pending.revision,
            verdict=ExecutionHumanVerdict.APPROVE,
            approver=local_development_approver_evidence(
                tenant_id="tenant-r12",
                actor_id="op-r12",
            ),
            human_request_id=pending.human_request_id or f"hr_{pending.continuation_id}",
            pause_id=pending.pause_id or f"pause_{pending.continuation_id}",
            resolved_at="2026-09-20T00:00:00Z",
        ),
    )


def test_gr10_r12_no_concrete_store_import_in_semantic_consumers() -> None:
    for path in _SEMANTIC_CONSUMERS:
        source = path.read_text(encoding="utf-8")
        assert "InMemoryExecutionContinuationStateStore" not in source
        assert "BackingExecutionContinuationStateStore" not in source
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "continuation.persistence" not in node.module


def test_gr10_r12_production_nexus_rejects_silent_lab_store() -> None:
    with pytest.raises(ExecutionContinuationError) as exc:
        NexusLoop(
            AgentRegistry(),
            production_mode=True,
            max_run_retries=0,
            retry_policy=RetryPolicy(max_retries=0),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE


def test_gr10_r12_production_nexus_rejects_explicit_non_durable_store() -> None:
    store = InMemoryExecutionContinuationStateStore()
    with pytest.raises(ExecutionContinuationError) as exc:
        NexusLoop(
            AgentRegistry(),
            production_mode=True,
            max_run_retries=0,
            retry_policy=RetryPolicy(max_retries=0),
            execution_continuation_state_store=store,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE
    assert "not durable" in str(exc.value).lower() or "is_durable" in str(exc.value)


def test_gr10_r12_production_nexus_accepts_durable_store() -> None:
    store = execution_continuation_state_store_from_durable_export(
        export_durable_continuation_state(ExecutionContinuationDurableBacking()),
    )
    assert store.is_durable is True
    loop = NexusLoop(
        AgentRegistry(),
        production_mode=True,
        max_run_retries=0,
        retry_policy=RetryPolicy(max_retries=0),
        execution_continuation_state_store=store,
    )
    assert loop.execution_continuation_state_store is store


def test_gr10_r12_lab_nexus_may_default_in_memory() -> None:
    loop = NexusLoop(AgentRegistry(), production_mode=False)
    assert loop.execution_continuation_state_store is not None
    assert loop.execution_continuation_state_store.is_durable is False


def test_gr10_r12_stale_c1_cannot_resume_after_c2() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    identity = _identity()
    c1 = _pause_to_waiting(
        deps.continuation,
        deps.lifecycle_driver,
        continuation_id="c1",
        identity=identity,
    )
    authorized = _approve(deps.continuation, c1)
    resumed_c1 = deps.continuation.resume(
        ExecutionContinuationResumeCommand(
            continuation_id=authorized.continuation_id,
            identity=identity,
            expected_revision=authorized.revision,
        ),
    )
    assert resumed_c1.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED
    c2 = _pause_to_waiting(
        deps.continuation,
        deps.lifecycle_driver,
        continuation_id="c2",
        identity=identity,
    )
    current = deps.continuation.get_pending(ExecutionContinuationLookup(identity=identity))
    assert current.continuation_id == "c2"
    with pytest.raises(ExecutionContinuationError) as exc:
        deps.continuation.resume(
            ExecutionContinuationResumeCommand(
                continuation_id="c1",
                identity=identity,
                expected_revision=resumed_c1.revision,
            ),
        )
    assert exc.value.code in {
        ExecutionContinuationErrorCode.INVALID_TRANSITION,
        ExecutionContinuationErrorCode.ALREADY_RESUMED,
        ExecutionContinuationErrorCode.STALE_REVISION,
    }
    cap = InternalOrchestrationContinuation(
        port=deps.continuation,
        lifecycle_driver=deps.lifecycle_driver,
    )
    assert canonical_execution_is_resumed(cap, identity=identity) is False
    assert c2.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN


def test_gr10_r12_wrong_attempt_blocked() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    attempt_a = mint_attempt_id()
    attempt_b = mint_attempt_id()
    identity = _identity(attempt=attempt_a)
    pending = _pause_to_waiting(
        deps.continuation,
        deps.lifecycle_driver,
        continuation_id="c-attempt",
        identity=identity,
    )
    wrong = _identity(attempt=attempt_b, execution=identity.execution_id)
    with pytest.raises(ExecutionContinuationError) as exc:
        deps.continuation.get_pending(
            ExecutionContinuationLookup(
                continuation_id=pending.continuation_id,
                identity=wrong,
            ),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.IDENTITY_MISMATCH


def test_gr10_r12_illegal_waiting_to_resumed_direct() -> None:
    with pytest.raises(ExecutionContinuationError) as exc:
        advance_continuation_lifecycle(
            ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
            ExecutionContinuationTransition.RESUME,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.INVALID_TRANSITION


def test_gr10_r12_bridge_without_store_fail_closed() -> None:
    task = Task(
        tenant_id="t",
        user_id="u",
        task_id=str(_TASK),
        message="m",
    )
    request = GovernedContinuationRequest(
        reason=ContinuationReason.COMPLIANCE,
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        source_agent_id="agent",
        prompt="approve?",
        operation_id="op-1",
    )
    with pytest.raises(InternalHitlContinuationCapabilityError):
        apply_governed_continuation_pause(task, request)


def test_gr10_r12_custom_store_replaceable() -> None:
    custom = InMemoryExecutionContinuationStateStore()
    deps = wire_execution_engine_continuation_dependencies(state_store=custom)
    assert deps.continuation_service.store is custom
    identity = _identity()
    pending = deps.continuation.request_pause(
        ExecutionPauseRequest(
            identity=identity,
            continuation_id="c-custom",
            reason=ContinuationReason.COMPLIANCE,
        ),
    )
    assert custom.load("c-custom") is pending


def test_gr10_r12_restart_from_durable_export() -> None:
    backing = ExecutionContinuationDurableBacking()
    live = backing_execution_continuation_state_store(backing)
    deps = wire_execution_engine_continuation_dependencies(state_store=live)
    identity = _identity()
    pending = _pause_to_waiting(
        deps.continuation,
        deps.lifecycle_driver,
        continuation_id="c-restart",
        identity=identity,
    )
    export = export_durable_continuation_state(backing)
    restored = execution_continuation_state_store_from_durable_export(export)
    assert restored.is_durable is True
    reloaded = restored.load("c-restart")
    assert reloaded is not None
    assert reloaded.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN
    assert reloaded.continuation_id == pending.continuation_id


def test_gr10_r12_validate_composition_helper() -> None:
    validate_execution_continuation_for_composition(
        production_mode=False,
        state_store=None,
        continuation_explicitly_wired=False,
        continuation_disabled=False,
    )
    with pytest.raises(ExecutionContinuationError) as missing:
        validate_execution_continuation_for_composition(
            production_mode=True,
            state_store=None,
            continuation_explicitly_wired=False,
            continuation_disabled=False,
        )
    assert missing.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE
    with pytest.raises(ExecutionContinuationError) as non_durable:
        validate_execution_continuation_for_composition(
            production_mode=True,
            state_store=InMemoryExecutionContinuationStateStore(),
            continuation_explicitly_wired=True,
            continuation_disabled=False,
        )
    assert non_durable.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE
    durable = execution_continuation_state_store_from_durable_export(
        export_durable_continuation_state(ExecutionContinuationDurableBacking()),
    )
    validate_execution_continuation_for_composition(
        production_mode=True,
        state_store=durable,
        continuation_explicitly_wired=True,
        continuation_disabled=False,
    )
    validate_execution_continuation_for_composition(
        production_mode=True,
        state_store=None,
        continuation_explicitly_wired=False,
        continuation_disabled=True,
    )


def test_gr10_r12_establish_pause_uses_port_not_task_authority() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    task = Task(tenant_id="t", user_id="u", task_id=str(_TASK), message="m")
    identity = _identity(task=_TASK)
    cap = InternalOrchestrationContinuation(
        port=deps.continuation,
        lifecycle_driver=deps.lifecycle_driver,
    )
    pending = establish_canonical_hitl_pause(
        task,
        identity=identity,
        continuation_id="c-pause",
        reason=ContinuationReason.COMPLIANCE,
        pause_id="pause_c",
        human_request_id="hr_c",
        capability=cap,
    )
    assert pending.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN
    assert task.runtime.governance.paused is True
    task.runtime.governance.paused = False
    assert canonical_execution_is_resumed(cap, identity=identity) is False

def _empty_durable_store() -> ReconstructedDurableExecutionContinuationStateStore:
    return execution_continuation_state_store_from_durable_export(
        export_durable_continuation_state(ExecutionContinuationDurableBacking()),
    )


def _strict_env() -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.product_defaults(profile_id="gr10.r12.r1.strict")


class _PluginDurableContinuationStore(ReconstructedDurableExecutionContinuationStateStore):
    """External-provider-shaped durable store (contract capability only)."""


class _PluginNonDurableContinuationStore(InMemoryExecutionContinuationStateStore):
    """Custom store rejected in production via is_durable=False."""


def test_gr10_r12_r1_production_rejects_backing_non_durable() -> None:
    store = backing_execution_continuation_state_store(ExecutionContinuationDurableBacking())
    assert store.is_durable is False
    with pytest.raises(ExecutionContinuationError) as exc:
        NexusLoop(
            AgentRegistry(),
            production_mode=True,
            max_run_retries=0,
            retry_policy=RetryPolicy(max_retries=0),
            execution_continuation_state_store=store,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE


def test_gr10_r12_r1_custom_durable_provider_accepted() -> None:
    store = _PluginDurableContinuationStore(
        restore_durable_continuation_backing(
            export_durable_continuation_state(ExecutionContinuationDurableBacking()),
        ),
    )
    assert store.is_durable is True
    loop = NexusLoop(
        AgentRegistry(),
        production_mode=True,
        max_run_retries=0,
        retry_policy=RetryPolicy(max_retries=0),
        execution_continuation_state_store=store,
    )
    assert loop.execution_continuation_state_store is store


def test_gr10_r12_r1_custom_non_durable_provider_rejected() -> None:
    store = _PluginNonDurableContinuationStore()
    with pytest.raises(ExecutionContinuationError) as exc:
        NexusLoop(
            AgentRegistry(),
            production_mode=True,
            max_run_retries=0,
            retry_policy=RetryPolicy(max_retries=0),
            execution_continuation_state_store=store,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE


def test_gr10_r12_r1_strict_factory_fails_without_durable_store() -> None:
    attempt_store = wire_attempt_lifecycle_store(kv_store=InMemoryKVStore())
    with pytest.raises(ExecutionContinuationError) as exc:
        build_nexus_loop_from_environment(
            AgentRegistry(),
            env=_strict_env(),
            attempt_lifecycle_store=attempt_store,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE


def test_gr10_r12_r1_strict_factory_accepts_durable_store() -> None:
    store = _empty_durable_store()
    attempt_store = wire_attempt_lifecycle_store(kv_store=InMemoryKVStore())
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=_strict_env(),
        execution_continuation_state_store=store,
        attempt_lifecycle_store=attempt_store,
    )
    assert loop.execution_continuation_state_store is store


def test_gr10_r12_r1_lab_factory_allows_implicit_in_memory() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="gr10.r12.r1.lab")
    assert env.execution_mode is not ExecutionMode.STRICT
    loop = build_nexus_loop_from_environment(AgentRegistry(), env=env)
    assert loop.execution_continuation_state_store is not None
    assert loop.execution_continuation_state_store.is_durable is False


def test_gr10_r12_r1_host_shares_nexus_store_identity() -> None:
    store = _empty_durable_store()
    loop = NexusLoop(
        AgentRegistry(),
        production_mode=True,
        max_run_retries=0,
        retry_policy=RetryPolicy(max_retries=0),
        execution_continuation_state_store=store,
    )
    host = build_certified_internal_test_host_task_execution(loop)
    assert loop.execution_continuation_state_store is store
    assert host._continuation_state_store is store  # noqa: SLF001


def test_gr10_r12_r1_explicit_port_uses_service_store_durability() -> None:
    store = _empty_durable_store()
    deps = wire_execution_engine_continuation_dependencies(state_store=store)
    loop = NexusLoop(
        AgentRegistry(),
        production_mode=True,
        max_run_retries=0,
        retry_policy=RetryPolicy(max_retries=0),
        execution_continuation=deps.continuation,
        continuation_lifecycle_driver=deps.lifecycle_driver,
    )
    assert loop.execution_continuation_state_store is store


def test_gr10_r12_r1_custom_port_without_store_fail_closed() -> None:
    class _OpaquePort:
        def request_pause(self, request):  # noqa: ANN001
            raise NotImplementedError

        def apply_resolution(self, command):  # noqa: ANN001
            raise NotImplementedError

        def resume(self, command):  # noqa: ANN001
            raise NotImplementedError

        def get_pending(self, lookup):  # noqa: ANN001
            raise NotImplementedError

        def transition(self, command):  # noqa: ANN001
            raise NotImplementedError

    deps = wire_execution_engine_continuation_dependencies(state_store=_empty_durable_store())
    with pytest.raises(ExecutionContinuationError) as exc:
        NexusLoop(
            AgentRegistry(),
            production_mode=True,
            max_run_retries=0,
            retry_policy=RetryPolicy(max_retries=0),
            execution_continuation=_OpaquePort(),  # type: ignore[arg-type]
            continuation_lifecycle_driver=deps.lifecycle_driver,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE


def test_gr10_r12_r1_restart_pause_resolve_resume() -> None:
    backing = ExecutionContinuationDurableBacking()
    live = backing_execution_continuation_state_store(backing)
    deps_a = wire_execution_engine_continuation_dependencies(state_store=live)
    identity = _identity()
    pending = _pause_to_waiting(
        deps_a.continuation,
        deps_a.lifecycle_driver,
        continuation_id="c-r1-restart",
        identity=identity,
    )
    assert pending.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN
    export = export_durable_continuation_state(backing)
    del deps_a, live, backing
    restored = execution_continuation_state_store_from_durable_export(export)
    assert restored.is_durable is True
    deps_b = wire_execution_engine_continuation_dependencies(state_store=restored)
    loaded = deps_b.continuation.get_pending(ExecutionContinuationLookup(identity=identity))
    authorized = _approve(deps_b.continuation, loaded)
    resumed = deps_b.continuation.resume(
        ExecutionContinuationResumeCommand(
            continuation_id=authorized.continuation_id,
            identity=identity,
            expected_revision=authorized.revision,
        ),
    )
    assert resumed.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED


def test_gr10_r12_r1_static_factory_no_implicit_wire() -> None:
    factory = (_REPO / "intergrax/applications/_shared/nexus_factory.py").read_text(
        encoding="utf-8",
    )
    assert "wire_execution_continuation_state_store" not in factory
    policy = (
        _REPO / "intergrax/runtime/execution/continuation/durability_policy.py"
    ).read_text(encoding="utf-8")
    assert "is_durable" in policy
    assert "isinstance" not in policy
    assert "InMemoryExecutionContinuationStateStore" not in policy


def test_gr10_r12_r1_plugin_durable_pause_resume() -> None:
    store: ExecutionContinuationStateStore = _PluginDurableContinuationStore(
        restore_durable_continuation_backing(
            export_durable_continuation_state(ExecutionContinuationDurableBacking()),
        ),
    )
    deps = wire_execution_engine_continuation_dependencies(state_store=store)
    identity = _identity()
    pending = _pause_to_waiting(
        deps.continuation,
        deps.lifecycle_driver,
        continuation_id="c-plugin",
        identity=identity,
    )
    authorized = _approve(deps.continuation, pending)
    resumed = deps.continuation.resume(
        ExecutionContinuationResumeCommand(
            continuation_id=authorized.continuation_id,
            identity=identity,
            expected_revision=authorized.revision,
        ),
    )
    assert resumed.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED

