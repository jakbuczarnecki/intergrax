# © Artur Czarnecki. All rights reserved.
"""TOOL-LIFE-X Batch C — stalled/abandoned/recovery qualification."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from pydantic import BaseModel
from testing_support.builder import (
    canonical_governed_execution_scope,
    canonical_run_id_for_tests,
)

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.dependency_concurrency_admission import DependencyConcurrencyKind
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationPhysicalState,
)
from intergrax.contracts.external_operation_identity import mint_stable_operation_id
from intergrax.contracts.idempotency_store import InvocationUncertaintyError
from intergrax.runtime.external_operations.external_operation_ownership import (
    ExternalOperationFencedCommitRejectedError,
    ProcessLocalExternalOperationOwner,
    mark_physical_running,
    mark_physical_terminal_if_owner_matches,
    reconcile_orphaned_running_operations,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    InMemoryExternalOperationStateStore,
)
from intergrax.runtime.external_operations.tool_external_operation_attempt import (
    ToolExternalOperationAttempt,
    tool_external_operation_identity,
)
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.tool_operation_recovery import (
    is_stalled_running_external_operation,
    reconcile_orphaned_tool_external_operations,
)
from intergrax.tools.core.contracts import (
    SideEffectRetrySafety,
    ToolContract,
    ToolRetryPolicy,
)
from intergrax.tools.execution_models import ToolExecutionRequest
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_BATCH_C_RUN_ID = canonical_run_id_for_tests("tool-life-x-batch-c")
_AGENT_ID = "batch-c-agent"


class ValueInput(BaseModel):
    value: int


class ValueOutput(BaseModel):
    result: int


class AllowAllScopePolicy:
    def is_allowed(self, *args: object, **kwargs: object) -> bool:
        return True


class DummyState:
    def __init__(self, *, tenant_id: str = "tenant_test") -> None:
        self.run_id = _BATCH_C_RUN_ID
        self.tenant_id = tenant_id
        self.declarative_hitl_grant = None
        self.request = type("Req", (), {"metadata": {}})()
        self._context = type(
            "Ctx",
            (),
            {
                "config": type(
                    "Cfg",
                    (),
                    {"policy_bundle": _enforce_allow_bundle(), "production_mode": False},
                )()
            },
        )()

    @property
    def context(self) -> object:
        return self._context

    @property
    def task_id(self) -> str | None:
        return None

    def trace_event(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


def _enforce_allow_bundle() -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="batch-c.allow")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    return wire_policy_bundle(env)


@contextmanager
def _governed_invoke_scope(*, governance_tenant_id: str = "tenant_test"):
    with canonical_governed_execution_scope(
        _BATCH_C_RUN_ID,
        governance_tenant_id=governance_tenant_id,
        governance_principal_id="principal-tool-life-x-batch-c",
    ):
        with patch(
            "intergrax.runtime.nexus.tools.invoker.require_meaningful_side_effect_authorization",
        ):
            yield


@pytest.fixture(autouse=True)
def _batch_c_active_execution_governance_identity() -> None:
    with _governed_invoke_scope():
        yield


def _request(
    *,
    tool_id: str,
    key: str | None = None,
) -> ToolExecutionRequest[ValueInput]:
    return ToolExecutionRequest(
        run_id=_BATCH_C_RUN_ID,
        step_id="step1",
        tool_id=tool_id,
        input=ValueInput(value=1),
        idempotency_key=key,
    )


def _tool_identity(tool_id: str = "demo.tool") -> object:
    execution_id = mint_execution_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        operation_id = mint_stable_operation_id(
            execution_id=execution_id,
            dependency_kind=DependencyConcurrencyKind.TOOL,
            dependency_identity=tool_id,
            logical_scope="step1",
        )
        return tool_external_operation_identity(
            tool_id=tool_id,
            step_id="step1",
            physical_attempt_sequence=1,
        )
    finally:
        reset_active_execution_identity(token)


def test_batch_c_l1_stale_worker_fenced_terminal_rejected() -> None:
    store = InMemoryExternalOperationStateStore()
    owner_a = ProcessLocalExternalOperationOwner.mint()
    owner_b = ProcessLocalExternalOperationOwner.mint()
    record = store.create_if_absent("op-1", owner_token=owner_a.token)
    running = mark_physical_running(
        store,
        operation_id="op-1",
        expected_revision=record.revision,
        owner=owner_a,
    )
    with pytest.raises(ExternalOperationFencedCommitRejectedError):
        mark_physical_terminal_if_owner_matches(
            store,
            operation_id="op-1",
            expected_revision=running.revision,
            physical_state=ExternalOperationPhysicalState.SUCCEEDED,
            owner=owner_b,
        )


def test_batch_c_l1_late_terminal_cannot_overwrite_prior_terminal() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    record = store.create_if_absent("op-2", owner_token=owner.token)
    running = mark_physical_running(
        store,
        operation_id="op-2",
        expected_revision=record.revision,
        owner=owner,
    )
    first = mark_physical_terminal_if_owner_matches(
        store,
        operation_id="op-2",
        expected_revision=running.revision,
        physical_state=ExternalOperationPhysicalState.UNKNOWN,
        owner=owner,
    )
    with pytest.raises(ExternalOperationFencedCommitRejectedError):
        mark_physical_terminal_if_owner_matches(
            store,
            operation_id="op-2",
            expected_revision=first.revision,
            physical_state=ExternalOperationPhysicalState.SUCCEEDED,
            owner=owner,
        )


def test_batch_c_l1_two_worker_reconcile_race_single_unknown() -> None:
    store = InMemoryExternalOperationStateStore()
    dead_owner = ProcessLocalExternalOperationOwner.mint()
    active_owner = ProcessLocalExternalOperationOwner.mint()
    record = store.create_if_absent("op-3", owner_token=dead_owner.token)
    mark_physical_running(
        store,
        operation_id="op-3",
        expected_revision=record.revision,
        owner=dead_owner,
    )
    barrier = threading.Barrier(2)
    results: list[tuple[ExternalOperationState, ...]] = []

    def worker() -> None:
        barrier.wait()
        results.append(
            reconcile_orphaned_tool_external_operations(
                store,
                active_owner=active_owner,
            ),
        )

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    loaded = store.load("op-3")
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.UNKNOWN
    assert sum(len(batch) for batch in results) >= 1


def test_batch_c_l1_stalled_classification_requires_running_and_deadline() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    record = store.create_if_absent("op-4", owner_token=owner.token)
    running = mark_physical_running(
        store,
        operation_id="op-4",
        expected_revision=record.revision,
        owner=owner,
    )
    assert is_stalled_running_external_operation(
        running,
        parent_execution_deadline_exceeded=False,
    ) is False
    assert is_stalled_running_external_operation(
        running,
        parent_execution_deadline_exceeded=True,
    ) is True


def test_batch_c_l1_uncertain_timeout_abandoned_not_retry_safe() -> None:
    class SlowExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            time.sleep(0.2)
            return ValueOutput(result=1)

    contract = ToolContract(
        tool_id="mutate.abandon",
        name="abandon",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        timeout_ms=50,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
        side_effect_retry_safety=SideEffectRetrySafety.NOT_RETRY_SAFE,
    )
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=SlowExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    result = invoker.invoke(
        state=DummyState(),
        agent_id=_AGENT_ID,
        request=_request(tool_id="mutate.abandon"),
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.ABANDONED
    assert result.error.error_message == "abandoned"


def test_batch_c_l1_unknown_outcome_blocks_blind_replay_after_abandon() -> None:
    class SlowExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            time.sleep(0.2)
            return ValueOutput(result=1)

    contract = ToolContract(
        tool_id="mutate.ledger",
        name="ledger",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        timeout_ms=50,
        retry_policy=ToolRetryPolicy(max_attempts=1, backoff_ms=0),
        side_effect_retry_safety=SideEffectRetrySafety.NOT_RETRY_SAFE,
    )
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=SlowExecutor(),
        pre_effect_coordinator=coordinator,
    )
    request = _request(tool_id="mutate.ledger", key="ledger-key")
    first = invoker.invoke(
        state=DummyState(),
        agent_id=_AGENT_ID,
        request=request,
    )
    assert first.error is not None
    assert first.error.error_code == RuntimeErrorCode.ABANDONED
    with pytest.raises(InvocationUncertaintyError):
        invoker.invoke(
            state=DummyState(),
            agent_id=_AGENT_ID,
            request=request,
        )


def test_batch_c_l1_invoker_reconciles_orphan_external_operation() -> None:
    store = InMemoryExternalOperationStateStore()
    active_owner = ProcessLocalExternalOperationOwner.mint()
    stale_owner = ProcessLocalExternalOperationOwner.mint()
    record = store.create_if_absent("orphan-op", owner_token=stale_owner.token)
    mark_physical_running(
        store,
        operation_id="orphan-op",
        expected_revision=record.revision,
        owner=stale_owner,
    )

    class OkExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            return ValueOutput(result=3)

    contract = ToolContract(
        tool_id="read.reconcile",
        name="read",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=False,
    )
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=OkExecutor(),
        scope_policy=AllowAllScopePolicy(),
        external_operation_store=store,
        external_operation_owner=active_owner,
    )
    result = invoker.invoke(
        state=DummyState(),
        agent_id=_AGENT_ID,
        request=_request(tool_id="read.reconcile"),
    )
    assert result.success is True
    loaded = store.load("orphan-op")
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.UNKNOWN


def test_batch_c_l1_tool_attempt_ignores_superseded_success_commit() -> None:
    store = InMemoryExternalOperationStateStore()
    owner_a = ProcessLocalExternalOperationOwner.mint()
    owner_b = ProcessLocalExternalOperationOwner.mint()
    identity = _tool_identity()
    assert identity.operation_id is not None
    record = store.create_if_absent(identity.operation_id, owner_token=owner_a.token)
    running = mark_physical_running(
        store,
        operation_id=identity.operation_id,
        expected_revision=record.revision,
        owner=owner_a,
    )
    reconcile_orphaned_running_operations(store, active_owner=owner_b)
    attempt = ToolExternalOperationAttempt(
        store=store,
        owner=owner_a,
        identity=identity,
    )
    attempt._revision = running.revision  # noqa: SLF001 — test stale worker
    attempt.mark_succeeded()
    loaded = store.load(identity.operation_id)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.UNKNOWN


def test_batch_c_l1_abandon_after_effect_started_retry_exhaustion() -> None:
    calls = {"n": 0}

    class FailAfterStart:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            raise RuntimeError("failed after backend")

    contract = ToolContract(
        tool_id="mutate.exhaust",
        name="exhaust",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
        side_effect_retry_safety=SideEffectRetrySafety.EXPLICITLY_RETRY_SAFE,
    )
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FailAfterStart(),
        pre_effect_coordinator=coordinator,
        scope_policy=AllowAllScopePolicy(),
    )
    result = invoker.invoke(
        state=DummyState(),
        agent_id=_AGENT_ID,
        request=_request(tool_id="mutate.exhaust", key="exhaust-key"),
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.ABANDONED
    assert calls["n"] == 3
