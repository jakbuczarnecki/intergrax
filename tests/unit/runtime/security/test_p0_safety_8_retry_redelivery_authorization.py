# © Artur Czarnecki. All rights reserved.

"""P0-SAFETY-8 — retry / redelivery authorization conformance."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.idempotency_store import InvocationUncertaintyError
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.errors.tool_scope_violation_error import ToolScopeViolationError
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.sandbox.isolation_errors import SandboxIsolationRequiredError
from intergrax.runtime.sandbox.isolation_gate import SandboxIsolationAvailability
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.core.contracts import (
    SideEffectRetrySafety,
    ToolContract,
    ToolIsolationRequirement,
    ToolRetryPolicy,
)
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_RUN_ID = "p0-safety-8-run"
_AGENT_ID = "p0-safety-8-agent"


class ValueInput(BaseModel):
    value: int


class ValueOutput(BaseModel):
    result: int


class AllowAllScopePolicy:
    def is_allowed(self, *args: object, **kwargs: object) -> bool:
        return True


class CountingScopePolicy:
    def __init__(self, *, allow_calls: int) -> None:
        self._allow_calls = allow_calls
        self.calls = 0

    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        del agent_id, tool_id
        self.calls += 1
        return self.calls <= self._allow_calls


class RevocableScopePolicy:
    def __init__(self) -> None:
        self._revoked = False

    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        del agent_id, tool_id
        return not self._revoked

    def revoke(self) -> None:
        self._revoked = True


class HealthyOnceSandboxAvailability:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> SandboxIsolationAvailability:
        self.calls += 1
        return SandboxIsolationAvailability(
            session_configured=True,
            host_configured=False,
            healthy=self.calls == 1,
        )

class AuthCountingInvoker(RuntimeToolInvoker):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.attempt_authorization_calls = 0

    def _require_current_attempt_authorization(self, **kwargs: object) -> None:
        self.attempt_authorization_calls += 1
        return super()._require_current_attempt_authorization(**kwargs)  # type: ignore[arg-type]


class EventTrackingInvoker(AuthCountingInvoker):
    def __init__(self, events: list[str], *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._events = events

    def _require_current_attempt_authorization(self, **kwargs: object) -> None:
        self._events.append("authorization")
        return super()._require_current_attempt_authorization(**kwargs)  # type: ignore[arg-type]


def _enforce_allow_bundle() -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="p0s8.allow")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    return wire_policy_bundle(env)


class DummyState:
    def __init__(self, *, policy_bundle: object | None = None) -> None:
        self.run_id = _RUN_ID
        self.tenant_id = "tenant_test"
        self.declarative_hitl_grant = None
        self._context = type(
            "Ctx",
            (),
            {"config": type("Cfg", (), {"policy_bundle": policy_bundle})()},
        )()

    @property
    def context(self) -> object:
        return self._context

    @property
    def task_id(self) -> str | None:
        return None

    def trace_event(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


def _state_with_allow() -> DummyState:
    return DummyState(policy_bundle=_enforce_allow_bundle())


def _request(
    *,
    tool_id: str,
    key: str | None = None,
) -> ToolExecutionRequest[ValueInput]:
    return ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=tool_id,
        input=ValueInput(value=1),
        idempotency_key=key,
    )


def test_default_side_effect_tool_executes_once_despite_retry_policy() -> None:
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            raise RuntimeError("transient")

    contract = ToolContract(
        tool_id="mutate.default",
        name="mutate.default",
        description="mutate",
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
        state=_state_with_allow(),
        agent_id=_AGENT_ID,
        request=_request(tool_id="mutate.default"),
    )
    assert result.success is False
    assert calls["n"] == 1


def test_read_only_tool_retries_per_policy() -> None:
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return ValueOutput(result=42)

    contract = ToolContract(
        tool_id="read.retry",
        name="read.retry",
        description="read",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=False,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
    )
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    result = invoker.invoke(
        state=DummyState(),
        agent_id=_AGENT_ID,
        request=_request(tool_id="read.retry"),
    )
    assert result.success
    assert calls["n"] == 3


def test_explicit_retry_safe_side_effect_retries_when_authorized() -> None:
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
        name="mutate.retry_safe",
        description="retry safe",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
        side_effect_retry_safety=SideEffectRetrySafety.EXPLICITLY_RETRY_SAFE,
    )
    invoker = AuthCountingInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    result = invoker.invoke(
        state=_state_with_allow(),
        agent_id=_AGENT_ID,
        request=_request(tool_id="mutate.retry_safe"),
    )
    assert result.success
    assert calls["n"] == 3
    assert invoker.attempt_authorization_calls == 3


def test_authority_revoked_during_backoff_blocks_retry_with_ordering_proof() -> None:
    events: list[str] = []
    calls = {"n": 0}
    scope = RevocableScopePolicy()

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            events.append("executor")
            raise RuntimeError("transient")

    contract = ToolContract(
        tool_id="read.backoff_revoke",
        name="read.backoff_revoke",
        description="read",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=False,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=50),
    )
    invoker = EventTrackingInvoker(
        events,
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=scope,
    )

    def _sleep_then_revoke(_seconds: float) -> None:
        events.append("sleep")
        scope.revoke()

    with patch("intergrax.runtime.nexus.tools.invoker.time.sleep", side_effect=_sleep_then_revoke):
        with pytest.raises(ToolScopeViolationError):
            invoker.invoke(
                state=DummyState(),
                agent_id=_AGENT_ID,
                request=_request(tool_id="read.backoff_revoke"),
            )

    assert calls["n"] == 1
    assert events == ["authorization", "executor", "sleep", "authorization"]


def test_fresh_scope_authorization_blocks_retry_when_revoked() -> None:
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            raise RuntimeError("transient")

    contract = ToolContract(
        tool_id="read.scope_retry",
        name="read.scope_retry",
        description="read",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=False,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
    )
    scope = CountingScopePolicy(allow_calls=1)
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=scope,
    )
    with pytest.raises(ToolScopeViolationError):
        invoker.invoke(
            state=DummyState(),
            agent_id=_AGENT_ID,
            request=_request(tool_id="read.scope_retry"),
        )
    assert calls["n"] == 1
    assert scope.calls == 2


def test_fresh_sandbox_authorization_blocks_retry_when_unavailable() -> None:
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            raise RuntimeError("transient")

    contract = ToolContract(
        tool_id="plugin.isolated",
        name="plugin.isolated",
        description="isolated",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
        side_effect_retry_safety=SideEffectRetrySafety.EXPLICITLY_RETRY_SAFE,
        isolation_requirement=ToolIsolationRequirement.SANDBOX,
    )
    availability = HealthyOnceSandboxAvailability()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
        sandbox_availability=availability,
    )

    with patch(
        "intergrax.runtime.nexus.tools.invoker.require_meaningful_side_effect_authorization",
    ):
        with pytest.raises(SandboxIsolationRequiredError):
            invoker.invoke(
                state=_state_with_allow(),
                agent_id=_AGENT_ID,
                request=_request(tool_id="plugin.isolated"),
            )
    assert calls["n"] == 1
    assert availability.calls == 2


def test_side_effect_timeout_does_not_blind_retry() -> None:
    calls = {"n": 0}

    class SlowExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            time.sleep(0.2)
            return ValueOutput(result=1)

    contract = ToolContract(
        tool_id="mutate.timeout",
        name="mutate.timeout",
        description="timeout",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        timeout_ms=50,
        retry_policy=ToolRetryPolicy(max_attempts=3, backoff_ms=0),
        side_effect_retry_safety=SideEffectRetrySafety.EXPLICITLY_RETRY_SAFE,
    )
    store = InMemoryIdempotencyStore()
    registry = ToolRegistry()
    registry.register(contract, SlowExecutor())
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=SlowExecutor(),
        pre_effect_coordinator=coordinator,
    )
    request = _request(tool_id="mutate.timeout", key="timeout-key")
    result = invoker.invoke(
        state=_state_with_allow(),
        agent_id=_AGENT_ID,
        request=request,
    )
    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == RuntimeErrorCode.TIMEOUT
    assert calls["n"] == 1

    with pytest.raises(InvocationUncertaintyError):
        invoker.invoke(
            state=_state_with_allow(),
            agent_id=_AGENT_ID,
            request=request,
        )
    assert calls["n"] == 1


def test_plugin_default_side_effect_retry_safety_is_single_attempt() -> None:
    calls = {"n": 0}

    class FlakyExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            raise RuntimeError("transient")

    contract = ToolContract(
        tool_id="plugin.custom",
        name="plugin.custom",
        description="plugin",
        input_schema=ValueInput,
        output_schema=ValueOutput,
        error_mapping={},
        side_effects=True,
        retry_policy=ToolRetryPolicy(max_attempts=4, backoff_ms=0),
    )
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=FlakyExecutor(),
        scope_policy=AllowAllScopePolicy(),
    )
    result = invoker.invoke(
        state=_state_with_allow(),
        agent_id=_AGENT_ID,
        request=_request(tool_id="plugin.custom"),
    )
    assert result.success is False
    assert calls["n"] == 1


def test_retry_limit_is_centrally_enforced() -> None:
    calls = {"n": 0}

    class AlwaysFailExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            del request
            calls["n"] += 1
            raise RuntimeError("always fails")

    contract = ToolContract(
        tool_id="read.limit",
        name="read.limit",
        description="read",
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


def test_idempotency_key_preserved_across_retry_attempts() -> None:
    seen_keys: list[str | None] = []

    class KeyCapturingExecutor:
        def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
            seen_keys.append(request.idempotency_key)
            if len(seen_keys) < 3:
                raise RuntimeError("transient")
            return ValueOutput(result=1)

    contract = ToolContract(
        tool_id="mutate.idempotent",
        name="mutate.idempotent",
        description="idempotent",
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
        executor=KeyCapturingExecutor(),
        scope_policy=AllowAllScopePolicy(),
        pre_effect_coordinator=coordinator,
    )
    request = _request(tool_id="mutate.idempotent", key="stable-key")
    result = invoker.invoke(
        state=_state_with_allow(),
        agent_id=_AGENT_ID,
        request=request,
    )
    assert result.success
    assert seen_keys == ["stable-key", "stable-key", "stable-key"]
    assert request.idempotency_key == "stable-key"
