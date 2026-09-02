# © Artur Czarnecki. All rights reserved.

"""P0-SAFETY-4 — fresh side-effect authorization conformance at RuntimeToolInvoker."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyHitlRequiredError,
    DeclarativePolicyViolationError,
)
from intergrax.runtime.nexus.errors.tool_scope_violation_error import ToolScopeViolationError
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativeHitlScopeAssignmentState,
    UniqueDeclarativeHitlCandidate,
    maybe_assign_declarative_hitl_scope,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
    PreEffectClaimContext,
    PreEffectCoordinationResult,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.registry import ToolRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TASK_ID = mint_task_id()
_RUN_ID = mint_run_id()
_TOOL_ID = "p0.safety.side_effect"
_RULE_ID = "p0.safety.rule"
_PLUGIN_TOOL_ID = "plugin.custom.side_effect"


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


class PluginHandler:
    def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
        return ValueOutput(result=request.input.value + 1)


class DummyHandler:
    def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
        return ValueOutput(result=request.input.value)


class MutableScopePolicy:
    def __init__(self, *, allowed: bool = True) -> None:
        self._allowed = allowed
        self.calls = 0

    def deny(self) -> None:
        self._allowed = False

    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        del agent_id, tool_id
        self.calls += 1
        return self._allowed


class GateOrderLog:
    def __init__(self) -> None:
        self.events: list[str] = []

    def record(self, event: str) -> None:
        self.events.append(event)


class RecordingScopePolicy:
    def __init__(self, *, order: GateOrderLog, inner: MutableScopePolicy) -> None:
        self._order = order
        self._inner = inner

    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        self._order.record("scope_authorization")
        return self._inner.is_allowed(agent_id=agent_id, tool_id=tool_id)


class RecordingPreEffectCoordinator:
    def __init__(
        self,
        *,
        order: GateOrderLog,
        inner: IdempotencyPreEffectCoordinator,
    ) -> None:
        self._order = order
        self._inner = inner

    def before_external_effect(self, **kwargs: object) -> PreEffectCoordinationResult:
        self._order.record("idempotency_before_external_effect")
        return self._inner.before_external_effect(**kwargs)  # type: ignore[arg-type]

    def after_external_effect(
        self,
        *,
        claim_context: PreEffectClaimContext,
        contract: ToolContract,
        result: ToolExecutionResult[BaseModel],
    ) -> None:
        self._inner.after_external_effect(
            claim_context=claim_context,
            contract=contract,
            result=result,
        )

    def on_post_claim_exception(
        self,
        *,
        claim_context: PreEffectClaimContext,
        contract: ToolContract,
        effect_may_have_started: bool,
    ) -> None:
        self._inner.on_post_claim_exception(
            claim_context=claim_context,
            contract=contract,
            effect_may_have_started=effect_may_have_started,
        )


class GovernanceDummyState:
    def __init__(
        self,
        *,
        tenant_id: str = "tenant_test",
        run_id: str = _RUN_ID,
        task_id: str = _TASK_ID,
        order: GateOrderLog | None = None,
    ) -> None:
        self._tenant_id = tenant_id
        self.run_id = run_id
        self.request = type("Req", (), {"task_id": task_id})()
        self.declarative_hitl_grant: DeclarativeHitlApprovalGrant | None = None
        self._order = order
        self._governance_context = type(
            "Ctx",
            (),
            {"config": type("Cfg", (), {"policy_bundle": None})()},
        )()

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def context(self):
        return self._governance_context

    @property
    def task_id(self) -> str:
        return self.request.task_id

    def trace_event(self, *args: object, **kwargs: object) -> None:
        if self._order is not None and kwargs.get("step") == "declarative_policy_evaluation":
            self._order.record("declarative_policy_authorization")


def _register_side_effect_tool(
    registry: ToolRegistry,
    *,
    tool_id: str = _TOOL_ID,
) -> None:
    registry.register(
        contract=ToolContract(
            tool_id=tool_id,
            name=tool_id,
            description=tool_id,
            input_schema=ValueInput,
            output_schema=ValueOutput,
            error_mapping={},
            side_effects=True,
        ),
        handler=DummyHandler(),
    )


def _policy_bundle(*, action: str, tool_id: str = _TOOL_ID, rule_id: str = _RULE_ID) -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=f"p0.{action}")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": rule_id,
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": tool_id,
                "action": action,
            }
        ],
        policy_enforcement_mode="enforce",
    )
    return wire_policy_bundle(env)


def _hitl_grant(*, bundle: object, key: str, tool_id: str = _TOOL_ID) -> DeclarativeHitlApprovalGrant:
    provenance = bundle.declarative_policy_runtime.provenance.rules_digest_sha256
    return DeclarativeHitlApprovalGrant(
        grant_id="grant-p0-safety",
        invocation_scope_id="dhr_scope",
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=tool_id,
        idempotency_key=key,
        matched_rule_ids=(_RULE_ID,),
        human_request_id="hr-p0-safety",
        policy_provenance_digest=provenance,
        pause_id="pause-p0-safety",
        approved_at="2026-08-30T00:00:00+00:00",
    )


def _invoker(
    executor: CountingExecutor,
    *,
    scope_policy: MutableScopePolicy | RecordingScopePolicy | None = None,
    coordinator: IdempotencyPreEffectCoordinator | RecordingPreEffectCoordinator | None = None,
    tool_id: str = _TOOL_ID,
) -> RuntimeToolInvoker:
    registry = ToolRegistry()
    _register_side_effect_tool(registry, tool_id=tool_id)
    return RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        scope_policy=scope_policy,
        pre_effect_coordinator=coordinator,
    )


def _request(
    *,
    tool_id: str = _TOOL_ID,
    value: int = 5,
    key: str = "p0-key",
    scope_id: str | None = None,
) -> ToolExecutionRequest[ValueInput]:
    return ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=tool_id,
        input=ValueInput(value=value),
        idempotency_key=key,
        declarative_hitl_invocation_scope_id=scope_id,
    )


def test_scope_denied_before_replay_blocks_not_replays() -> None:
    executor = CountingExecutor()
    scope = MutableScopePolicy(allowed=True)
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    invoker = _invoker(executor, scope_policy=scope, coordinator=coordinator)
    state = GovernanceDummyState()
    request = _request(key="scope-replay-key")

    first = invoker.invoke(state=state, agent_id="agent-a", request=request)
    assert first.success
    assert executor.calls == 1

    scope.deny()
    with pytest.raises(ToolScopeViolationError):
        invoker.invoke(state=state, agent_id="agent-a", request=request)
    assert executor.calls == 1


def test_policy_deny_overrides_prior_approval_grant() -> None:
    executor = CountingExecutor()
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    invoker = _invoker(executor, coordinator=coordinator)
    state = GovernanceDummyState()
    hitl_bundle = _policy_bundle(action="require_hitl")
    state.context.config.policy_bundle = hitl_bundle
    base_request = _request(key="deny-over-approval-key")

    with pytest.raises(DeclarativePolicyHitlRequiredError):
        invoker.invoke(state=state, agent_id="agent-a", request=base_request)
    assert executor.calls == 0

    state.declarative_hitl_grant = _hitl_grant(bundle=hitl_bundle, key="deny-over-approval-key")
    scoped_request = maybe_assign_declarative_hitl_scope(
        base_request,
        state=state,
        assignment_state=DeclarativeHitlScopeAssignmentState(),
        unique_candidate=UniqueDeclarativeHitlCandidate(candidate_index=0),
        request_index=0,
    )
    state.context.config.policy_bundle = _policy_bundle(action="deny")

    with pytest.raises(DeclarativePolicyViolationError):
        invoker.invoke(state=state, agent_id="agent-a", request=scoped_request)
    assert executor.calls == 0


def test_hitl_approval_requires_matching_invocation_scope() -> None:
    executor = CountingExecutor()
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    invoker = _invoker(executor, coordinator=coordinator)
    state = GovernanceDummyState()
    hitl_bundle = _policy_bundle(action="require_hitl")
    state.context.config.policy_bundle = hitl_bundle
    state.declarative_hitl_grant = _hitl_grant(bundle=hitl_bundle, key="scope-bind-key")
    wrong_scope_request = _request(
        key="scope-bind-key",
        scope_id="dhr_other_scope",
    )

    with pytest.raises(DeclarativePolicyHitlRequiredError):
        invoker.invoke(state=state, agent_id="agent-a", request=wrong_scope_request)
    assert executor.calls == 0


def test_unauthorized_scope_denial_never_invokes_handler() -> None:
    executor = CountingExecutor()
    scope = MutableScopePolicy(allowed=False)
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    invoker = _invoker(executor, scope_policy=scope, coordinator=coordinator)
    state = GovernanceDummyState()

    with pytest.raises(ToolScopeViolationError):
        invoker.invoke(state=state, agent_id="agent-a", request=_request(key="unauth-key"))
    assert executor.calls == 0
    assert store.get_status("tenant_test", "unauth-key") is None


def test_plugin_side_effect_tool_uses_same_invocation_gate() -> None:
    executor = CountingExecutor()
    scope = MutableScopePolicy(allowed=False)
    registry = ToolRegistry()
    registry.register(
        contract=ToolContract(
            tool_id=_PLUGIN_TOOL_ID,
            name=_PLUGIN_TOOL_ID,
            description="plugin tool",
            input_schema=ValueInput,
            output_schema=ValueOutput,
            error_mapping={},
            side_effects=True,
        ),
        handler=PluginHandler(),
    )
    coordinator = IdempotencyPreEffectCoordinator(
        idempotency_store=InMemoryIdempotencyStore(),
    )
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        scope_policy=scope,
        pre_effect_coordinator=coordinator,
    )
    state = GovernanceDummyState()

    with pytest.raises(ToolScopeViolationError):
        invoker.invoke(
            state=state,
            agent_id="agent-a",
            request=_request(tool_id=_PLUGIN_TOOL_ID, key="plugin-key"),
        )
    assert executor.calls == 0


class RecordingExecutor(CountingExecutor):
    def __init__(self, order: GateOrderLog) -> None:
        super().__init__()
        self._order = order

    def execute(self, request: ToolExecutionRequest[ValueInput]) -> ValueOutput:
        self._order.record("handler_execution")
        return super().execute(request)


def test_pre_effect_gate_ordering_is_authorization_before_idempotency_before_handler() -> None:
    order = GateOrderLog()
    scope = MutableScopePolicy(allowed=True)
    store = InMemoryIdempotencyStore()
    inner_coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    coordinator = RecordingPreEffectCoordinator(order=order, inner=inner_coordinator)
    state = GovernanceDummyState(order=order)
    allow_env = ApplicationEnvironmentProfile.lab_defaults(profile_id="p0.order")
    allow_env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    state.context.config.policy_bundle = wire_policy_bundle(allow_env)
    invoker = _invoker(
        RecordingExecutor(order),
        scope_policy=RecordingScopePolicy(order=order, inner=scope),
        coordinator=coordinator,
    )

    result = invoker.invoke(
        state=state,
        agent_id="agent-a",
        request=_request(key="ordering-key"),
    )
    assert result.success
    assert order.events == [
        "scope_authorization",
        "declarative_policy_authorization",
        "idempotency_before_external_effect",
        "handler_execution",
    ]
