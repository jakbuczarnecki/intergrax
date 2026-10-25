# © Artur Czarnecki. All rights reserved.
"""TOOL-LIFE-X Batch B — retry, idempotency, unknown-outcome safety qualification."""

from __future__ import annotations

import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from testing_support.builder import (
    canonical_governed_execution_scope,
    canonical_run_id_for_tests,
)
from pydantic import BaseModel

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.idempotency_store import (
    IdempotencyOperationConflictError,
    InvocationUncertaintyError,
)
from datetime import datetime, timedelta, timezone

from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
from intergrax.runtime.execution.deadline_scope import (
    bind_active_execution_deadline_scope,
    reset_active_execution_deadline_scope,
)
from intergrax.runtime.execution.protected_work_admission import (
    CanonicalHardProtectedWorkAdmission,
    StaticCancellationView,
)
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.errors.tool_scope_violation_error import ToolScopeViolationError
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.operation_identity import compute_invocation_operation_identity
from intergrax.runtime.tools.tool_operation_retry_classification import (
    ToolOperationRetryClass,
    classify_tool_operation_retry_class,
    effective_max_retry_attempts,
)
from intergrax.tools.core.contracts import (
    SideEffectRetrySafety,
    ToolContract,
    ToolRetryPolicy,
)
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_BATCH_B_RUN_ID = canonical_run_id_for_tests("tool-life-x-batch-b")
_RUN_ID = _BATCH_B_RUN_ID
_AGENT_ID = "batch-b-agent"
_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"


class ValueInput(BaseModel):
    value: int


class ValueOutput(BaseModel):
    result: int


class AllowAllScopePolicy:
    def is_allowed(self, *args: object, **kwargs: object) -> bool:
        return True


@contextmanager
def _governed_invoke_scope(*, governance_tenant_id: str = "tenant_test"):
    with canonical_governed_execution_scope(
        _BATCH_B_RUN_ID,
        governance_tenant_id=governance_tenant_id,
        governance_principal_id="principal-tool-life-x-batch-b",
    ):
        with patch(
            "intergrax.runtime.nexus.tools.invoker.require_meaningful_side_effect_authorization",
        ):
            yield


@pytest.fixture(autouse=True)
def _batch_b_active_execution_governance_identity() -> None:
    with _governed_invoke_scope():
        yield


def _enforce_allow_bundle() -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="batch-b.allow")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    return wire_policy_bundle(env)


class DummyState:
    def __init__(
        self,
        *,
        tenant_id: str = "tenant_test",
        policy_bundle: object | None = None,
    ) -> None:
        self.run_id = _RUN_ID
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
                    {"policy_bundle": policy_bundle, "production_mode": False},
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


def _request(
    *,
    tool_id: str,
    key: str | None = None,
    value: int = 1,
) -> ToolExecutionRequest[ValueInput]:
    return ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=tool_id,
        input=ValueInput(value=value),
        idempotency_key=key,
    )


def test_batch_b_classify_read_only_and_non_idempotent_write() -> None:
    read_only = ToolContract(
        tool_id="read",
        name="read",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=False,
    )
    req = _request(tool_id="read")
    assert (
        classify_tool_operation_retry_class(read_only, req)
        is ToolOperationRetryClass.READ_ONLY
    )
    assert effective_max_retry_attempts(read_only, req) == 1

    mutate = ToolContract(
        tool_id="mutate",
        name="mutate",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=5),
    )
    assert (
        classify_tool_operation_retry_class(mutate, req)
        is ToolOperationRetryClass.NON_IDEMPOTENT_WRITE
    )
    assert effective_max_retry_attempts(mutate, req) == 1


def test_batch_b_idempotent_write_requires_key_for_multi_attempt() -> None:
    contract = ToolContract(
        tool_id="idem",
        name="idem",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=4, backoff_ms=0),
        side_effect_retry_safety=SideEffectRetrySafety.IDEMPOTENT,
    )
    without_key = _request(tool_id="idem")
    with_key = _request(tool_id="idem", key="k1")
    assert (
        classify_tool_operation_retry_class(contract, without_key)
        is ToolOperationRetryClass.IDEMPOTENT_WRITE
    )
    assert effective_max_retry_attempts(contract, without_key) == 1
    assert effective_max_retry_attempts(contract, with_key) == 4


def test_batch_b_classify_explicit_retry_safe_with_key_not_external_dedup() -> None:
    contract = ToolContract(
        tool_id="retry_safe",
        name="retry safe",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=5, backoff_ms=0),
        side_effect_retry_safety=SideEffectRetrySafety.EXPLICITLY_RETRY_SAFE,
    )
    with_key = _request(tool_id="retry_safe", key="idem-key")
    assert (
        classify_tool_operation_retry_class(contract, with_key)
        is ToolOperationRetryClass.EXPLICITLY_RETRY_SAFE_WRITE
    )
    assert effective_max_retry_attempts(contract, with_key) == 5


def test_batch_b_non_retry_safe_with_key_still_single_attempt() -> None:
    contract = ToolContract(
        tool_id="unsafe",
        name="unsafe",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=5, backoff_ms=0),
        side_effect_retry_safety=SideEffectRetrySafety.NOT_RETRY_SAFE,
    )
    with_key = _request(tool_id="unsafe", key="k-only")
    assert (
        classify_tool_operation_retry_class(contract, with_key)
        is ToolOperationRetryClass.NON_IDEMPOTENT_WRITE
    )
    assert effective_max_retry_attempts(contract, with_key) == 1


def test_batch_b_l1_non_idempotent_write_never_duplicates_on_transient() -> None:
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            raise RuntimeError("transient")

    contract = ToolContract(
        tool_id="mutate.default",
        name="mutate",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=5, backoff_ms=0),
    )
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    result = invoker.invoke(
        state=DummyState(policy_bundle=_enforce_allow_bundle()),
        agent_id=_AGENT_ID,
        request=_request(tool_id="mutate.default"),
    )
    assert result.success is False
    assert calls["n"] == 1


def test_batch_b_l1_unknown_outcome_timeout_blocks_blind_replay() -> None:
    calls = {"n": 0}

    class SlowExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            time.sleep(0.2)
            return ValueOutput(result=1)

    contract = ToolContract(
        tool_id="mutate.timeout",
        name="timeout",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        timeout_ms=50,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
        side_effect_retry_safety=SideEffectRetrySafety.EXPLICITLY_RETRY_SAFE,
    )
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=SlowExecutor(),
        pre_effect_coordinator=coordinator,
    )
    request = _request(tool_id="mutate.timeout", key="timeout-key")
    result = invoker.invoke(
        state=DummyState(policy_bundle=_enforce_allow_bundle()),
        agent_id=_AGENT_ID,
        request=request,
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.TIMEOUT
    assert calls["n"] == 1
    with pytest.raises(InvocationUncertaintyError):
        invoker.invoke(
            state=DummyState(policy_bundle=_enforce_allow_bundle()),
            agent_id=_AGENT_ID,
            request=request,
        )
    assert calls["n"] == 1


def test_batch_b_l1_retry_reauthorizes_each_attempt() -> None:
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return ValueOutput(result=7)

    contract = ToolContract(
        tool_id="mutate.retry_safe",
        name="retry safe",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
        side_effect_retry_safety=SideEffectRetrySafety.EXPLICITLY_RETRY_SAFE,
    )

    class AuthCountingInvoker(RuntimeToolInvoker):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            self.attempt_authorization_calls = 0

        def _require_current_attempt_authorization(self, **kwargs: object) -> None:
            self.attempt_authorization_calls += 1
            return super()._require_current_attempt_authorization(**kwargs)  # type: ignore[arg-type]

    invoker = AuthCountingInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    result = invoker.invoke(
        state=DummyState(policy_bundle=_enforce_allow_bundle()),
        agent_id=_AGENT_ID,
        request=_request(tool_id="mutate.retry_safe"),
    )
    assert result.success
    assert calls["n"] == 3
    assert invoker.attempt_authorization_calls == 3


def test_batch_b_l1_idempotency_key_scope_isolated_per_tenant() -> None:
    store = InMemoryIdempotencyStore()
    executor_calls: dict[str, int] = {"a": 0, "b": 0}

    class TenantExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            if request.input.value == 1:
                executor_calls["a"] += 1
            else:
                executor_calls["b"] += 1
            return ValueOutput(result=request.input.value)

    contract = ToolContract(
        tool_id="write",
        name="write",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
    )
    registry = ToolRegistry()
    registry.register(contract, TenantExecutor())
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=TenantExecutor(),
        pre_effect_coordinator=coordinator,
    )
    key = "shared-key"
    with _governed_invoke_scope(governance_tenant_id=_TENANT_A):
        invoker.invoke(
            state=DummyState(tenant_id=_TENANT_A, policy_bundle=_enforce_allow_bundle()),
            agent_id=_AGENT_ID,
            request=_request(tool_id="write", key=key, value=1),
        )
    with _governed_invoke_scope(governance_tenant_id=_TENANT_B):
        invoker.invoke(
            state=DummyState(tenant_id=_TENANT_B, policy_bundle=_enforce_allow_bundle()),
            agent_id=_AGENT_ID,
            request=_request(tool_id="write", key=key, value=2),
        )
    assert executor_calls["a"] == 1
    assert executor_calls["b"] == 1


def test_batch_b_same_key_different_payload_fails_closed() -> None:
    class WriteExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            return ValueOutput(result=request.input.value)

    store = InMemoryIdempotencyStore()
    contract = ToolContract(
        tool_id="write",
        name="write",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
    )
    registry = ToolRegistry()
    registry.register(contract, WriteExecutor())
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=WriteExecutor(),
        pre_effect_coordinator=coordinator,
    )
    state = DummyState(policy_bundle=_enforce_allow_bundle())
    first = _request(tool_id="write", key="payload-key", value=1)
    invoker.invoke(state=state, agent_id=_AGENT_ID, request=first)
    identity_b = compute_invocation_operation_identity("write", ValueInput(value=2))
    with pytest.raises(IdempotencyOperationConflictError):
        store.claim(
            state.tenant_id,
            "payload-key",
            "other-owner",
            lease_seconds=30,
            operation_identity=identity_b,
        )


def test_batch_b_retry_budget_exhausted_no_extra_executor_calls() -> None:
    calls = {"n": 0}

    class AlwaysFailExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            raise RuntimeError("fail")

    contract = ToolContract(
        tool_id="read.limit",
        name="read",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=False,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
    )
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=AlwaysFailExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    result = invoker.invoke(
        state=DummyState(),
        agent_id=_AGENT_ID,
        request=_request(tool_id="read.limit"),
    )
    assert result.success is False
    assert calls["n"] == 3


class _SystemMonotonicClock:
    def monotonic(self) -> float:
        return time.monotonic()


def test_batch_b_parent_deadline_blocks_inter_attempt_retry() -> None:
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            raise RuntimeError("transient")

    contract = ToolContract(
        tool_id="read.deadline",
        name="read",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=False,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=5_000),
    )
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    monotonic = _SystemMonotonicClock()
    remaining_s = 0.06
    bound = monotonic.monotonic() + remaining_s
    deadline_at = datetime.now(timezone.utc) + timedelta(seconds=remaining_s)
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=deadline_at,
        remaining_seconds=remaining_s,
        is_expired=False,
        global_deadline_monotonic=bound,
    )
    tokens = bind_active_execution_deadline_scope(
        projection=projection,
        admission=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=monotonic,
        ),
        monotonic_clock=monotonic,
    )
    try:
        result = invoker.invoke(
            state=DummyState(),
            agent_id=_AGENT_ID,
            request=_request(tool_id="read.deadline"),
        )
    finally:
        reset_active_execution_deadline_scope(*tokens)
    assert result.success is False
    assert calls["n"] == 1
    assert result.error is not None
    assert "deadline_exceeded" in result.error.error_message


def test_batch_b_cancellation_during_backoff_blocks_second_attempt() -> None:
    calls = {"n": 0}
    scope_revoked = {"v": False}

    class RevocableScope:
        def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
            del agent_id, tool_id
            return not scope_revoked["v"]

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            raise RuntimeError("transient")

    contract = ToolContract(
        tool_id="read.backoff",
        name="read",
        description="",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=False,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=50),
    )
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=RevocableScope(),
    )

    def _sleep_then_revoke(
        _seconds: float,
        *,
        should_abort: object = None,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        del should_abort, poll_interval_seconds
        scope_revoked["v"] = True

    with patch(
        "intergrax.runtime.nexus.tools.invoker.cooperative_delay_seconds",
        side_effect=_sleep_then_revoke,
    ):
        with pytest.raises(ToolScopeViolationError):
            invoker.invoke(
                state=DummyState(),
                agent_id=_AGENT_ID,
                request=_request(tool_id="read.backoff"),
            )
    assert calls["n"] == 1


def test_batch_b_idempotency_key_stable_across_retry_attempts() -> None:
    seen: list[str | None] = []

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            seen.append(request.idempotency_key)
            if len(seen) < 3:
                raise RuntimeError("transient")
            return ValueOutput(result=1)

    contract = ToolContract(
        tool_id="mutate.idempotent",
        name="idempotent",
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
        executor=FlakyExecutor(),
        pre_effect_coordinator=coordinator,
        scope_policy=AllowAllScopePolicy(),
    )
    key = "stable-key"
    result = invoker.invoke(
        state=DummyState(policy_bundle=_enforce_allow_bundle()),
        agent_id=_AGENT_ID,
        request=_request(tool_id="mutate.idempotent", key=key),
    )
    assert result.success
    assert seen == [key, key, key]
