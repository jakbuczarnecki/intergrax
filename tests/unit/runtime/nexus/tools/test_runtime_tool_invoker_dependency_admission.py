# © Artur Czarnecki. All rights reserved.

"""W2-B2 — RuntimeToolInvoker dependency admission integration."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import cast

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionRequest,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPermit,
    DependencyConcurrencyPolicy,
    DependencyConcurrencyPolicyMissingError,
)
from intergrax.contracts.idempotency_store import InvocationStatus
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.errors.tool_scope_violation_error import ToolScopeViolationError
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
)
from intergrax.runtime.resilience.local_dependency_concurrency_admission import (
    LocalDependencyConcurrencyAdmission,
)
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.tools.core.contracts import SideEffectRetrySafety, ToolContract, ToolRetryPolicy
from intergrax.tools.execution_models import (
    ToolEffectCertainty,
    ToolExecutionRequest,
)
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import (
    build_runtime_state_for_tests,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _enforce_allow_bundle() -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="w2b2.admission")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    return wire_policy_bundle(env)


def _state_for_side_effects(run_id: str):
    state = build_runtime_state_for_tests(run_id=run_id)
    state.context.config.policy_bundle = _enforce_allow_bundle()
    return state


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    result: int


class _WrongIn(BaseModel):
    other: int


class _Handler:
    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        return _Out(result=request.input.value)


class _CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0
        self.max_in_flight = 0
        self._in_flight = 0
        self._gate: threading.Event | None = None
        self._gate_tool_id: str | None = None
        self._lock = threading.Lock()

    def set_gate(self, gate: threading.Event, *, tool_id: str) -> None:
        self._gate = gate
        self._gate_tool_id = tool_id

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        with self._lock:
            self.calls += 1
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if self._gate is not None and request.tool_id == self._gate_tool_id:
                self._gate.wait(timeout=10)
            return _Out(result=request.input.value)
        finally:
            with self._lock:
                self._in_flight -= 1


class _CountingPermit:
    __slots__ = ("_counter", "_inner")

    def __init__(
        self,
        inner: DependencyConcurrencyPermit,
        counter: _CountingLocalAdmission,
    ) -> None:
        self._inner = inner
        self._counter = counter

    async def release(self) -> None:
        await self._inner.release()
        self._counter.release_count += 1
        self._counter.active_permits -= 1


class _CountingLocalAdmission:
    """Typed test double: local admission with acquire/release counters."""

    def __init__(
        self,
        policies: dict[DependencyConcurrencyIdentity, DependencyConcurrencyPolicy],
    ) -> None:
        self._inner = LocalDependencyConcurrencyAdmission(policies)
        self.acquire_count = 0
        self.release_count = 0
        self.active_permits = 0

    async def acquire(
        self,
        request: DependencyConcurrencyAdmissionRequest,
    ) -> DependencyConcurrencyPermit:
        inner = await self._inner.acquire(request)
        self.acquire_count += 1
        self.active_permits += 1
        return _CountingPermit(inner, self)


@dataclass
class _AdmissionHarness:
    invoker: RuntimeToolInvoker
    admission: _CountingLocalAdmission
    boundary: DependencyAttemptExecutionBoundary

    def close(self) -> None:
        self.invoker.close()


def _identity(tool_id: str) -> DependencyConcurrencyIdentity:
    return DependencyConcurrencyIdentity(
        kind=DependencyConcurrencyKind.TOOL,
        value=tool_id,
    )


def _reject(capacity: int) -> DependencyConcurrencyPolicy:
    return DependencyConcurrencyPolicy(
        max_concurrent_calls=capacity,
        overload_mode=DependencyConcurrencyOverloadMode.REJECT,
        wait_timeout_seconds=None,
    )


def _wait(capacity: int, timeout: float) -> DependencyConcurrencyPolicy:
    return DependencyConcurrencyPolicy(
        max_concurrent_calls=capacity,
        overload_mode=DependencyConcurrencyOverloadMode.WAIT_WITH_TIMEOUT,
        wait_timeout_seconds=timeout,
    )


def _harness(
    *,
    registry: ToolRegistry,
    executor: object,
    policies: dict[DependencyConcurrencyIdentity, DependencyConcurrencyPolicy],
    pre_effect: IdempotencyPreEffectCoordinator | None = None,
    scope_policy: StaticToolScopePolicy | None = None,
) -> _AdmissionHarness:
    admission = _CountingLocalAdmission(policies)
    boundary = DependencyAttemptExecutionBoundary(admission)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        pre_effect_coordinator=pre_effect,
        scope_policy=scope_policy,
        dependency_attempt_boundary=boundary,
    )
    return _AdmissionHarness(invoker=invoker, admission=admission, boundary=boundary)


def _register(registry: ToolRegistry, tool_id: str, **kwargs: object) -> ToolContract:
    contract = ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description=tool_id,
        input_schema=_In,
        output_schema=_Out,
        error_mapping=cast(
            dict[type[Exception], RuntimeErrorCode],
            kwargs.get("error_mapping", {}),
        ),
        side_effects=bool(kwargs.get("side_effects", False)),
        timeout_ms=int(kwargs.get("timeout_ms", 5000)),
        retry_policy=cast(
            ToolRetryPolicy,
            kwargs.get(
                "retry_policy",
                ToolRetryPolicy(max_attempts=1, backoff_ms=0),
            ),
        ),
        side_effect_retry_safety=cast(
            SideEffectRetrySafety,
            kwargs.get(
                "side_effect_retry_safety",
                SideEffectRetrySafety.NOT_RETRY_SAFE,
            ),
        ),
    )
    registry.register(contract=contract, handler=_Handler())
    return contract


def test_reject_before_submit_executor_not_called() -> None:
    registry = ToolRegistry()
    _register(registry, "tool-a")
    executor = _CountingExecutor()
    gate = threading.Event()
    executor.set_gate(gate, tool_id="tool-a")
    harness = _harness(
        registry=registry,
        executor=executor,
        policies={_identity("tool-a"): _reject(1)},
    )
    run_id = canonical_run_id_for_tests("reject-before-submit")
    state = build_runtime_state_for_tests(run_id=run_id)
    holder_state = build_runtime_state_for_tests(run_id=run_id)

    def _hold() -> None:
        with canonical_execution_identity_scope(run_id):
            harness.invoker.invoke(
                state=holder_state,
                agent_id="agent",
                request=ToolExecutionRequest(
                    run_id=run_id,
                    tool_id="tool-a",
                    step_id="1",
                    input=_In(value=1),
                ),
            )

    holder = threading.Thread(target=_hold)
    holder.start()
    time.sleep(0.15)
    with canonical_execution_identity_scope(run_id):
        result = harness.invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                tool_id="tool-a",
                step_id="2",
                input=_In(value=2),
            ),
        )
    gate.set()
    holder.join(timeout=10)
    harness.close()
    assert executor.calls == 1
    assert harness.admission.acquire_count >= 1
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.DEPENDENCY_ERROR
    assert result.effect_certainty is ToolEffectCertainty.NOT_STARTED


def test_tool_b_runs_when_tool_a_saturated() -> None:
    registry = ToolRegistry()
    _register(registry, "tool-a")
    _register(registry, "tool-b")
    gate = threading.Event()
    started = threading.Event()

    class _TrackingExecutor(_CountingExecutor):
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            if request.tool_id == "tool-a":
                started.set()
            return super().execute(request)

    tracking = _TrackingExecutor()
    tracking.set_gate(gate, tool_id="tool-a")
    harness = _harness(
        registry=registry,
        executor=tracking,
        policies={
            _identity("tool-a"): _reject(1),
            _identity("tool-b"): _reject(1),
        },
    )
    run_id = canonical_run_id_for_tests("noisy-neighbor")
    state = build_runtime_state_for_tests(run_id=run_id)
    holder_state = build_runtime_state_for_tests(run_id=run_id)

    def _hold_a() -> None:
        with canonical_execution_identity_scope(run_id):
            harness.invoker.invoke(
                state=holder_state,
                agent_id="agent",
                request=ToolExecutionRequest(
                    run_id=run_id,
                    tool_id="tool-a",
                    step_id="1",
                    input=_In(value=1),
                ),
            )

    holder = threading.Thread(target=_hold_a)
    holder.start()
    assert started.wait(timeout=2)
    with canonical_execution_identity_scope(run_id):
        result_b = harness.invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                tool_id="tool-b",
                step_id="2",
                input=_In(value=2),
            ),
        )
    gate.set()
    holder.join(timeout=10)
    harness.close()
    assert result_b.success is True
    assert tracking.calls == 2


def test_policy_missing_propagates() -> None:
    registry = ToolRegistry()
    _register(registry, "unknown-tool")
    executor = _CountingExecutor()
    harness = _harness(
        registry=registry,
        executor=executor,
        policies={_identity("other"): _reject(1)},
    )
    run_id = canonical_run_id_for_tests("policy-missing")
    state = build_runtime_state_for_tests(run_id=run_id)
    with canonical_execution_identity_scope(run_id):
        with pytest.raises(DependencyConcurrencyPolicyMissingError):
            harness.invoker.invoke(
                state=state,
                agent_id="agent",
                request=ToolExecutionRequest(
                    run_id=run_id,
                    tool_id="unknown-tool",
                    step_id="1",
                    input=_In(value=1),
                ),
            )
    harness.close()
    assert executor.calls == 0
    assert harness.admission.acquire_count == 0


def test_feature_off_legacy_unchanged() -> None:
    registry = ToolRegistry()
    _register(registry, "plain")
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(registry=registry, executor=executor)
    run_id = canonical_run_id_for_tests("feature-off")
    state = build_runtime_state_for_tests(run_id=run_id)
    with canonical_execution_identity_scope(run_id):
        result = invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                tool_id="plain",
                step_id="1",
                input=_In(value=3),
            ),
        )
    invoker.close()
    assert result.success is True
    assert executor.calls == 1


def test_exceeded_mapping_not_started_no_executor() -> None:
    registry = ToolRegistry()
    _register(registry, "tool-a")
    executor = _CountingExecutor()
    gate = threading.Event()
    executor.set_gate(gate, tool_id="tool-a")
    harness = _harness(
        registry=registry,
        executor=executor,
        policies={_identity("tool-a"): _reject(1)},
    )
    run_id = canonical_run_id_for_tests("exceeded-map")
    state = build_runtime_state_for_tests(run_id=run_id)
    holder_state = build_runtime_state_for_tests(run_id=run_id)

    def _hold() -> None:
        with canonical_execution_identity_scope(run_id):
            harness.invoker.invoke(
                state=holder_state,
                agent_id="agent",
                request=ToolExecutionRequest(
                    run_id=run_id,
                    tool_id="tool-a",
                    step_id="hold",
                    input=_In(value=1),
                ),
            )

    threading.Thread(target=_hold).start()
    time.sleep(0.15)
    with canonical_execution_identity_scope(run_id):
        result = harness.invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                tool_id="tool-a",
                step_id="reject",
                input=_In(value=2),
            ),
        )
    gate.set()
    harness.close()
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.DEPENDENCY_ERROR
    assert result.effect_certainty is ToolEffectCertainty.NOT_STARTED
    assert executor.calls == 1


def test_admission_timeout_mapping() -> None:
    registry = ToolRegistry()
    _register(registry, "tool-a")
    executor = _CountingExecutor()
    gate = threading.Event()
    executor.set_gate(gate, tool_id="tool-a")
    harness = _harness(
        registry=registry,
        executor=executor,
        policies={_identity("tool-a"): _wait(1, 0.05)},
    )
    run_id = canonical_run_id_for_tests("adm-timeout")
    state = build_runtime_state_for_tests(run_id=run_id)
    holder_state = build_runtime_state_for_tests(run_id=run_id)

    def _hold() -> None:
        with canonical_execution_identity_scope(run_id):
            harness.invoker.invoke(
                state=holder_state,
                agent_id="agent",
                request=ToolExecutionRequest(
                    run_id=run_id,
                    tool_id="tool-a",
                    step_id="1",
                    input=_In(value=1),
                ),
            )

    threading.Thread(target=_hold).start()
    time.sleep(0.1)
    with canonical_execution_identity_scope(run_id):
        result = harness.invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                tool_id="tool-a",
                step_id="2",
                input=_In(value=2),
            ),
        )
    gate.set()
    harness.close()
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.DEPENDENCY_ERROR
    assert result.effect_certainty is ToolEffectCertainty.NOT_STARTED
    assert executor.calls == 1


def test_capacity_two_limits_physical_concurrency() -> None:
    registry = ToolRegistry()
    _register(registry, "tool-a", timeout_ms=10_000)
    start_gate = threading.Event()
    proceed = threading.Event()

    class _SyncExecutor(_CountingExecutor):
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            start_gate.set()
            proceed.wait(timeout=5)
            return super().execute(request)

    sync_exec = _SyncExecutor()
    harness = _harness(
        registry=registry,
        executor=sync_exec,
        policies={_identity("tool-a"): _reject(2)},
    )
    run_id = canonical_run_id_for_tests("cap-two")
    errors: list[BaseException] = []

    def _call(step: str) -> None:
        try:
            st = build_runtime_state_for_tests(run_id=run_id)
            with canonical_execution_identity_scope(run_id):
                harness.invoker.invoke(
                    state=st,
                    agent_id="agent",
                    request=ToolExecutionRequest(
                        run_id=run_id,
                        tool_id="tool-a",
                        step_id=step,
                        input=_In(value=1),
                    ),
                )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_call, args=(str(i),)) for i in range(3)]
    for thread in threads:
        thread.start()
    assert start_gate.wait(timeout=3)
    time.sleep(0.3)
    proceed.set()
    for thread in threads:
        thread.join(timeout=15)
    harness.close()
    assert not errors
    assert sync_exec.max_in_flight <= 2


def test_wait_does_not_queue_second_physical_attempt() -> None:
    registry = ToolRegistry()
    _register(registry, "tool-a")
    executor = _CountingExecutor()
    gate = threading.Event()
    executor.set_gate(gate, tool_id="tool-a")
    harness = _harness(
        registry=registry,
        executor=executor,
        policies={_identity("tool-a"): _wait(1, 2.0)},
    )
    run_id = canonical_run_id_for_tests("wait-no-pool")
    holder_state = build_runtime_state_for_tests(run_id=run_id)

    def _hold() -> None:
        with canonical_execution_identity_scope(run_id):
            harness.invoker.invoke(
                state=holder_state,
                agent_id="agent",
                request=ToolExecutionRequest(
                    run_id=run_id,
                    tool_id="tool-a",
                    step_id="1",
                    input=_In(value=1),
                ),
            )

    threading.Thread(target=_hold).start()
    time.sleep(0.2)
    assert executor.calls == 1
    gate.set()
    harness.close()


def test_retry_reacquire_after_release() -> None:
    registry = ToolRegistry()
    _register(
        registry,
        "flaky",
        side_effects=False,
        retry_policy=ToolRetryPolicy(max_attempts=2, backoff_ms=10),
        error_mapping={ConnectionError: RuntimeErrorCode.TOOL_ERROR},
    )

    class _FlakyExecutor:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transient")
            return _Out(result=7)

    flaky = _FlakyExecutor()
    harness = _harness(
        registry=registry,
        executor=flaky,
        policies={_identity("flaky"): _reject(1)},
    )
    run_id = canonical_run_id_for_tests("retry-reacquire")
    state = build_runtime_state_for_tests(run_id=run_id)
    with canonical_execution_identity_scope(run_id):
        result = harness.invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                tool_id="flaky",
                step_id="1",
                input=_In(value=1),
            ),
        )
    harness.close()
    assert result.success is True
    assert flaky.calls == 2
    assert harness.admission.acquire_count == 2
    assert harness.admission.release_count == 2


def test_no_permit_held_during_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ToolRegistry()
    _register(
        registry,
        "flaky",
        side_effects=False,
        retry_policy=ToolRetryPolicy(max_attempts=2, backoff_ms=50),
        error_mapping={ConnectionError: RuntimeErrorCode.TOOL_ERROR},
    )

    class _FlakyExecutor:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transient")
            return _Out(result=1)

    harness = _harness(
        registry=registry,
        executor=_FlakyExecutor(),
        policies={_identity("flaky"): _reject(1)},
    )
    observed_during_backoff: list[int] = []

    original_sleep = time.sleep

    def _sleep(seconds: float) -> None:
        if seconds >= 0.05:
            observed_during_backoff.append(harness.admission.active_permits)
        original_sleep(seconds)

    monkeypatch.setattr(time, "sleep", _sleep)
    run_id = canonical_run_id_for_tests("backoff-no-permit")
    state = build_runtime_state_for_tests(run_id=run_id)
    with canonical_execution_identity_scope(run_id):
        harness.invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                tool_id="flaky",
                step_id="1",
                input=_In(value=1),
            ),
        )
    harness.close()
    assert observed_during_backoff
    assert all(value == 0 for value in observed_during_backoff)


def test_timeout_holds_capacity_until_worker_finishes() -> None:
    registry = ToolRegistry()
    _register(registry, "slow-a", timeout_ms=100)
    executor = _CountingExecutor()
    gate = threading.Event()
    executor.set_gate(gate, tool_id="slow-a")
    harness = _harness(
        registry=registry,
        executor=executor,
        policies={_identity("slow-a"): _reject(1)},
    )
    run_id = canonical_run_id_for_tests("timeout-holds")
    state = build_runtime_state_for_tests(run_id=run_id)
    with canonical_execution_identity_scope(run_id):
        result = harness.invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                tool_id="slow-a",
                step_id="1",
                input=_In(value=1),
            ),
        )
        assert result.error is not None
        assert result.error.error_code == RuntimeErrorCode.TIMEOUT
        rejected = harness.invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                tool_id="slow-a",
                step_id="2",
                input=_In(value=2),
            ),
        )
    assert rejected.error is not None
    assert rejected.error.error_code == RuntimeErrorCode.DEPENDENCY_ERROR
    gate.set()
    harness.close()
    assert executor.calls == 1


def test_timeout_on_a_does_not_block_tool_b() -> None:
    registry = ToolRegistry()
    _register(registry, "slow-a", timeout_ms=100)
    _register(registry, "tool-b")
    gate = threading.Event()
    executor = _CountingExecutor()
    executor.set_gate(gate, tool_id="slow-a")
    harness = _harness(
        registry=registry,
        executor=executor,
        policies={
            _identity("slow-a"): _reject(1),
            _identity("tool-b"): _reject(1),
        },
    )
    run_id = canonical_run_id_for_tests("timeout-b-free")
    state = build_runtime_state_for_tests(run_id=run_id)
    with canonical_execution_identity_scope(run_id):
        harness.invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                tool_id="slow-a",
                step_id="1",
                input=_In(value=1),
            ),
        )
        result_b = harness.invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                tool_id="tool-b",
                step_id="2",
                input=_In(value=2),
            ),
        )
    gate.set()
    harness.close()
    assert result_b.success is True
    assert executor.calls == 2


def test_idempotency_replay_skips_admission() -> None:
    registry = ToolRegistry()
    _register(registry, "idem-tool", side_effects=True)
    executor = _CountingExecutor()
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    harness = _harness(
        registry=registry,
        executor=executor,
        policies={_identity("idem-tool"): _reject(1)},
        pre_effect=coordinator,
    )
    run_id = canonical_run_id_for_tests("idem-replay")
    state = _state_for_side_effects(run_id)
    request = ToolExecutionRequest(
        run_id=run_id,
        tool_id="idem-tool",
        step_id="1",
        input=_In(value=3),
        idempotency_key="key-1",
    )
    with canonical_execution_identity_scope(run_id):
        first = harness.invoker.invoke(state=state, agent_id="agent", request=request)
        second = harness.invoker.invoke(state=state, agent_id="agent", request=request)
    harness.close()
    assert first.success is True
    assert second.success is True
    assert executor.calls == 1
    assert harness.admission.acquire_count == 1


def test_claimed_admission_reject_finalizes_not_started() -> None:
    registry = ToolRegistry()
    _register(registry, "tool-a", side_effects=True)
    executor = _CountingExecutor()
    gate = threading.Event()
    executor.set_gate(gate, tool_id="tool-a")
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    harness = _harness(
        registry=registry,
        executor=executor,
        policies={_identity("tool-a"): _reject(1)},
        pre_effect=coordinator,
    )
    run_id = canonical_run_id_for_tests("claim-reject")
    holder_state = _state_for_side_effects(run_id)
    state = _state_for_side_effects(run_id)

    def _hold() -> None:
        with canonical_execution_identity_scope(run_id):
            harness.invoker.invoke(
                state=holder_state,
                agent_id="agent",
                request=ToolExecutionRequest(
                    run_id=run_id,
                    tool_id="tool-a",
                    step_id="hold",
                    input=_In(value=1),
                    idempotency_key="hold-key",
                ),
            )

    threading.Thread(target=_hold).start()
    time.sleep(0.15)
    with canonical_execution_identity_scope(run_id):
        result = harness.invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                tool_id="tool-a",
                step_id="reject",
                input=_In(value=2),
                idempotency_key="reject-key",
            ),
        )
    gate.set()
    harness.close()
    assert result.success is False
    assert result.effect_certainty is ToolEffectCertainty.NOT_STARTED
    assert store.get_status(state.tenant_id, "reject-key") == InvocationStatus.COMPLETED


def test_policy_missing_post_claim_not_started() -> None:
    registry = ToolRegistry()
    _register(registry, "tool-a", side_effects=True)
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    harness = _harness(
        registry=registry,
        executor=_CountingExecutor(),
        policies={_identity("other"): _reject(1)},
        pre_effect=coordinator,
    )
    run_id = canonical_run_id_for_tests("policy-missing-claim")
    state = _state_for_side_effects(run_id)
    with canonical_execution_identity_scope(run_id):
        with pytest.raises(DependencyConcurrencyPolicyMissingError):
            harness.invoker.invoke(
                state=state,
                agent_id="agent",
                request=ToolExecutionRequest(
                    run_id=run_id,
                    tool_id="tool-a",
                    step_id="1",
                    input=_In(value=1),
                    idempotency_key="pm-key",
                ),
            )
    harness.close()
    assert store.get_status(state.tenant_id, "pm-key") == InvocationStatus.COMPLETED


def test_governance_denial_skips_admission() -> None:
    registry = ToolRegistry()
    _register(registry, "denied-tool")
    harness = _harness(
        registry=registry,
        executor=_CountingExecutor(),
        policies={_identity("denied-tool"): _reject(1)},
        scope_policy=StaticToolScopePolicy(allowed_tools={"other-tool"}),
    )
    run_id = canonical_run_id_for_tests("gov-deny")
    state = build_runtime_state_for_tests(run_id=run_id)
    with canonical_execution_identity_scope(run_id):
        with pytest.raises(ToolScopeViolationError):
            harness.invoker.invoke(
                state=state,
                agent_id="agent",
                request=ToolExecutionRequest(
                    run_id=run_id,
                    tool_id="denied-tool",
                    step_id="1",
                    input=_In(value=1),
                ),
            )
    harness.close()
    assert harness.admission.acquire_count == 0


def test_invalid_input_skips_admission() -> None:
    registry = ToolRegistry()
    _register(registry, "tool-a")
    harness = _harness(
        registry=registry,
        executor=_CountingExecutor(),
        policies={_identity("tool-a"): _reject(1)},
    )
    run_id = canonical_run_id_for_tests("invalid-input")
    state = build_runtime_state_for_tests(run_id=run_id)
    bad_request = ToolExecutionRequest(
        run_id=run_id,
        tool_id="tool-a",
        step_id="1",
        input=cast(_In, _WrongIn(other=1)),
    )
    with canonical_execution_identity_scope(run_id):
        result = harness.invoker.invoke(
            state=state,
            agent_id="agent",
            request=bad_request,
        )
    harness.close()
    assert result.success is False
    assert harness.admission.acquire_count == 0


def test_submit_failure_releases_permit(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ToolRegistry()
    _register(registry, "tool-a")
    harness = _harness(
        registry=registry,
        executor=_CountingExecutor(),
        policies={_identity("tool-a"): _reject(1)},
    )
    run_id = canonical_run_id_for_tests("submit-fail")
    state = build_runtime_state_for_tests(run_id=run_id)

    def _broken_submit(*args: object, **kwargs: object) -> object:
        raise RuntimeError("submit failed")

    monkeypatch.setattr(harness.invoker._execution_pool, "submit", _broken_submit)
    with canonical_execution_identity_scope(run_id):
        result = harness.invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                tool_id="tool-a",
                step_id="1",
                input=_In(value=1),
            ),
        )
    assert result.success is False
    assert harness.admission.acquire_count == 1
    assert harness.admission.release_count == 1
    assert harness.admission.active_permits == 0
    harness.close()


def test_close_after_detached_timeout_drains() -> None:
    registry = ToolRegistry()
    _register(registry, "slow-a", timeout_ms=80)
    gate = threading.Event()
    executor = _CountingExecutor()
    executor.set_gate(gate, tool_id="slow-a")
    harness = _harness(
        registry=registry,
        executor=executor,
        policies={_identity("slow-a"): _reject(1)},
    )
    run_id = canonical_run_id_for_tests("close-detached")
    state = build_runtime_state_for_tests(run_id=run_id)
    with canonical_execution_identity_scope(run_id):
        harness.invoker.invoke(
            state=state,
            agent_id="agent",
            request=ToolExecutionRequest(
                run_id=run_id,
                tool_id="slow-a",
                step_id="1",
                input=_In(value=1),
            ),
        )
    gate.set()
    harness.close()
    assert harness.admission.active_permits == 0
