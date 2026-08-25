# © Artur Czarnecki. All rights reserved.

"""TOOLS-SIDE-EFFECT-SAFETY — idempotency identity, retry safety, outcome states."""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from intergrax.contracts.idempotency_store import (
    ActiveInvocationClaimError,
    ClaimOutcome,
    IdempotencyOperationConflictError,
    InvocationClaim,
    InvocationStatus,
    InvocationUncertaintyError,
)
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.errors.tool_scope_violation_error import ToolScopeViolationError
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.tools.idempotent_invoker import IdempotentToolInvoker
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.operation_identity import compute_invocation_operation_identity
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.tools.core.contracts import (
    SideEffectRetrySafety,
    ToolContract,
    ToolRetryPolicy,
)
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import build_runtime_state_for_tests
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class ValueInput(BaseModel):
    value: int


class ValueOutput(BaseModel):
    result: int


class CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
        self.calls += 1
        return ValueOutput(result=request.input.value * 2)


class DummyHandler:
    def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
        return ValueOutput(result=request.input.value * 2)


class DummyState:
    def __init__(self, tenant_id: str = "tenant_test") -> None:
        self._tenant_id = tenant_id
        self.run_id = "run1"

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def context(self):
        return type("Ctx", (), {"config": type("Cfg", (), {"policy_bundle": None})()})()

    def trace_event(self, *args, **kwargs) -> None:
        del args, kwargs


class AllowAllScopePolicy:
    def is_allowed(self, *args, **kwargs) -> bool:
        return True


def _register_tool(
    registry: ToolRegistry,
    *,
    tool_id: str,
    side_effects: bool,
    retry_policy: ToolRetryPolicy = ToolRetryPolicy(),
    side_effect_retry_safety: SideEffectRetrySafety = SideEffectRetrySafety.NOT_RETRY_SAFE,
    timeout_ms: int = 30_000,
) -> None:
    registry.register(
        contract=ToolContract(
            tool_id=tool_id,
            name=tool_id,
            description=tool_id,
            input_schema=ValueInput,
            output_schema=ValueOutput,
            error_mapping={},
            side_effects=side_effects,
            retry_policy=retry_policy,
            side_effect_retry_safety=side_effect_retry_safety,
            timeout_ms=timeout_ms,
        ),
        handler=DummyHandler(),
    )


def _idempotent_invoker(
    executor: CountingExecutor,
    *,
    tools: tuple[str, ...] = ("tool_a",),
    side_effects: bool = True,
    store: InMemoryIdempotencyStore | None = None,
) -> tuple[IdempotentToolInvoker, InMemoryIdempotencyStore]:
    ledger = store or InMemoryIdempotencyStore()
    registry = ToolRegistry()
    for tool_id in tools:
        _register_tool(registry, tool_id=tool_id, side_effects=side_effects)
    base = RuntimeToolInvoker(registry=registry, executor=executor)
    invoker = IdempotentToolInvoker(
        base_invoker=base,
        idempotency_store=ledger,
    )
    return invoker, ledger


def _request(
    *,
    tool_id: str = "tool_a",
    value: int = 5,
    key: str = "key-x",
) -> ToolExecutionRequest[ValueInput]:
    return ToolExecutionRequest(
        run_id="run1",
        step_id="step1",
        tool_id=tool_id,
        input=ValueInput(value=value),
        idempotency_key=key,
    )


def test_cross_tool_same_key_fails_closed() -> None:
    executor = CountingExecutor()
    invoker, _ = _idempotent_invoker(executor, tools=("tool_a", "tool_b"))
    state = DummyState()

    invoker.invoke(state=state, agent_id="agent", request=_request(tool_id="tool_a"))
    with pytest.raises(IdempotencyOperationConflictError):
        invoker.invoke(state=state, agent_id="agent", request=_request(tool_id="tool_b"))
    assert executor.calls == 1


def test_cross_input_same_tool_key_fails_closed() -> None:
    executor = CountingExecutor()
    invoker, _ = _idempotent_invoker(executor)
    state = DummyState()

    invoker.invoke(state=state, agent_id="agent", request=_request(value=5))
    with pytest.raises(IdempotencyOperationConflictError):
        invoker.invoke(state=state, agent_id="agent", request=_request(value=9))
    assert executor.calls == 1


def test_same_canonical_input_replays() -> None:
    executor = CountingExecutor()
    invoker, _ = _idempotent_invoker(executor)
    state = DummyState()
    request = _request(value=5)

    r1 = invoker.invoke(state=state, agent_id="agent", request=request)
    r2 = invoker.invoke(state=state, agent_id="agent", request=request)
    assert r1.success and r2.success
    assert r1.output == r2.output
    assert executor.calls == 1


def test_read_only_tool_retries_per_policy() -> None:
    contract = ToolContract(
        tool_id="read_tool",
        name="read_tool",
        description="read",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=False,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
    )
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return ValueOutput(result=42)

    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    state = build_runtime_state_for_tests(run_id="run_retry")
    request = ToolExecutionRequest(
        run_id="run_retry",
        tool_id="read_tool",
        step_id="1",
        input=ValueInput(value=1),
    )
    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success
    assert calls["n"] == 3


def test_side_effect_tool_without_retry_proof_executes_once() -> None:
    contract = ToolContract(
        tool_id="mutate_tool",
        name="mutate_tool",
        description="mutate",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
    )
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            calls["n"] += 1
            raise RuntimeError("transient")

    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    state = build_runtime_state_for_tests(run_id="run_mutate")
    request = ToolExecutionRequest(
        run_id="run_mutate",
        tool_id="mutate_tool",
        step_id="1",
        input=ValueInput(value=1),
    )
    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success is False
    assert calls["n"] == 1


def test_side_effect_tool_explicit_retry_safe_retries() -> None:
    contract = ToolContract(
        tool_id="retry_safe_tool",
        name="retry_safe_tool",
        description="retry safe",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
        side_effect_retry_safety=SideEffectRetrySafety.EXPLICITLY_RETRY_SAFE,
    )
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return ValueOutput(result=7)

    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    state = build_runtime_state_for_tests(run_id="run_safe_retry")
    request = ToolExecutionRequest(
        run_id="run_safe_retry",
        tool_id="retry_safe_tool",
        step_id="1",
        input=ValueInput(value=1),
    )
    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success
    assert calls["n"] == 3


def test_side_effect_timeout_marks_uncertain_and_blocks_replay() -> None:
    contract = ToolContract(
        tool_id="slow_tool",
        name="slow_tool",
        description="slow",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        timeout_ms=50,
    )

    class SlowExecutor:
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            time.sleep(0.2)
            return ValueOutput(result=1)

    store = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    _register_tool(registry, tool_id="slow_tool", side_effects=True, timeout_ms=50)
    base = RuntimeToolInvoker(registry=registry, executor=SlowExecutor())
    invoker = IdempotentToolInvoker(base_invoker=base, idempotency_store=store)
    state = DummyState()
    request = _request(tool_id="slow_tool", key="timeout-key")

    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.TIMEOUT
    assert store.get_status("tenant_test", "timeout-key") == InvocationStatus.UNCERTAIN

    with pytest.raises(InvocationUncertaintyError):
        invoker.invoke(state=state, agent_id="agent", request=request)


def test_validation_failure_is_safe_terminal_replay() -> None:
    executor = CountingExecutor()
    store = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    _register_tool(registry, tool_id="tool_a", side_effects=True)
    base = RuntimeToolInvoker(registry=registry, executor=executor)
    invoker = IdempotentToolInvoker(base_invoker=base, idempotency_store=store)
    state = DummyState()
    bad_request = ToolExecutionRequest(
        run_id="run1",
        step_id="step1",
        tool_id="tool_a",
        input=ValueOutput(result=1),
        idempotency_key="validation-key",
    )
    r1 = invoker.invoke(state=state, agent_id="agent", request=bad_request)
    r2 = invoker.invoke(state=state, agent_id="agent", request=bad_request)
    assert r1.success is False
    assert r2.success is False
    assert executor.calls == 0
    assert store.get_status("tenant_test", "validation-key") == InvocationStatus.COMPLETED


def test_scope_denial_before_executor_is_safe_terminal() -> None:
    executor = CountingExecutor()
    store = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    _register_tool(registry, tool_id="tool_a", side_effects=True)
    base = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        scope_policy=StaticToolScopePolicy(allowed_tools=set()),
    )
    invoker = IdempotentToolInvoker(base_invoker=base, idempotency_store=store)
    state = DummyState()
    request = _request(key="scope-deny-key")

    with pytest.raises(ToolScopeViolationError):
        invoker.invoke(state=state, agent_id="agent", request=request)

    assert executor.calls == 0
    assert store.get_status("tenant_test", "scope-deny-key") == InvocationStatus.COMPLETED

    replay = invoker.invoke(state=state, agent_id="agent", request=request)
    assert replay.success is False
    assert executor.calls == 0


def test_output_validation_failure_after_executor_marks_uncertain() -> None:
    external_effects = {"n": 0}

    class BadOutputExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> BaseModel:
            external_effects["n"] += 1
            return ValueInput(value=999)

    store = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    _register_tool(registry, tool_id="tool_a", side_effects=True)
    base = RuntimeToolInvoker(registry=registry, executor=BadOutputExecutor())
    invoker = IdempotentToolInvoker(base_invoker=base, idempotency_store=store)
    state = DummyState()
    request = _request(key="bad-output-key")

    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success is False
    assert external_effects["n"] == 1
    assert store.get_status("tenant_test", "bad-output-key") == InvocationStatus.UNCERTAIN

    with pytest.raises(InvocationUncertaintyError):
        invoker.invoke(state=state, agent_id="agent", request=request)
    assert external_effects["n"] == 1


def test_mapped_validation_error_after_executor_marks_uncertain() -> None:
    external_effects = {"n": 0}

    class MutatingFailExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            external_effects["n"] += 1
            raise ValueError("mutation already applied")

    store = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    registry.register(
        contract=ToolContract(
            tool_id="tool_a",
            name="tool_a",
            description="tool_a",
            input_schema=ValueInput,
            output_schema=ValueOutput,
            error_mapping={ValueError: RuntimeErrorCode.VALIDATION_ERROR},
            side_effects=True,
        ),
        handler=DummyHandler(),
    )
    base = RuntimeToolInvoker(registry=registry, executor=MutatingFailExecutor())
    invoker = IdempotentToolInvoker(base_invoker=base, idempotency_store=store)
    state = DummyState()
    request = _request(key="mapped-validation-key")

    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.VALIDATION_ERROR
    assert external_effects["n"] == 1
    assert store.get_status("tenant_test", "mapped-validation-key") == InvocationStatus.UNCERTAIN

    with pytest.raises(InvocationUncertaintyError):
        invoker.invoke(state=state, agent_id="agent", request=request)
    assert external_effects["n"] == 1


def test_unknown_executor_failure_marks_uncertain() -> None:
    calls = {"n": 0}

    class FailExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            calls["n"] += 1
            raise RuntimeError("boom")

    store = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    _register_tool(registry, tool_id="tool_a", side_effects=True)
    base = RuntimeToolInvoker(registry=registry, executor=FailExecutor())
    invoker = IdempotentToolInvoker(base_invoker=base, idempotency_store=store)
    state = DummyState()
    request = _request(key="fail-key")

    result = invoker.invoke(state=state, agent_id="agent", request=request)
    assert result.success is False
    assert store.get_status("tenant_test", "fail-key") == InvocationStatus.UNCERTAIN

    with pytest.raises(InvocationUncertaintyError):
        invoker.invoke(state=state, agent_id="agent", request=request)
    assert calls["n"] == 1


def test_stale_owner_cannot_mark_uncertain() -> None:
    store = InMemoryIdempotencyStore()
    acquired = store.claim(
        "tenant-a",
        "uncertain-key",
        "owner-a",
        lease_seconds=30,
        operation_identity=None,
    )
    assert acquired.claim is not None
    stale_claim = InvocationClaim(
        tenant_id="tenant-a",
        key="uncertain-key",
        owner_id="owner-a",
        lease_expires_at=acquired.claim.lease_expires_at,
        fence=1,
    )
    entry = store._store[("tenant-a", "uncertain-key")]  # noqa: SLF001
    entry.claim = acquired.claim.model_copy(update={"fence": 2, "owner_id": "owner-b"})
    with pytest.raises(StaleClaimError):
        store.mark_uncertain_with_claim("tenant-a", "uncertain-key", stale_claim)


def test_active_claim_blocks_concurrent_owner() -> None:
    store = InMemoryIdempotencyStore()
    executor = CountingExecutor()
    invoker, _ = _idempotent_invoker(executor, store=store)
    state = DummyState()
    request = _request()

    store.claim("tenant_test", "key-x", "owner-a", lease_seconds=30)
    with pytest.raises(ActiveInvocationClaimError):
        invoker.invoke(state=state, agent_id="agent", request=request)
    assert executor.calls == 0


def test_completed_replays_with_matching_operation_identity() -> None:
    store = InMemoryIdempotencyStore()
    executor = CountingExecutor()
    invoker, ledger = _idempotent_invoker(executor, store=store)
    state = DummyState()
    request = _request()
    operation_identity = compute_invocation_operation_identity(
        request.tool_id,
        request.input,
    )

    invoker.invoke(state=state, agent_id="agent", request=request)
    outcome = ledger.claim(
        "tenant_test",
        "key-x",
        "owner-b",
        lease_seconds=30,
        operation_identity=operation_identity,
    )
    assert outcome.outcome == ClaimOutcome.REPLAY_COMPLETED
