# © Artur Czarnecki. All rights reserved.

"""BLOCK B E2E: declarative policy through standard host wiring and tool invoker."""

from __future__ import annotations

import importlib.metadata

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import reset_entry_point_spec_cache_for_tests
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyViolationError,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from testing_support.builder import build_runtime_state_for_tests
from intergrax.runtime.policy.rules.schema import PolicyRuleAction
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_GROUP = "intergrax.policy_rules"
_TOOL_ID = "e2e.blocked.tool"


class _Input(BaseModel):
    value: int


class _Output(BaseModel):
    value: int


class _CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        self.calls += 1
        return _Output(value=request.input.value)


class _EntryPoint:
    def __init__(self, name: str, value: str, group: str) -> None:
        self.name = name
        self.value = value
        self.group = group


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


class _AlphaHandler:
    rule_id = "alpha-rule"

    def evaluate(self, rule: object, *, context: object) -> object:
        from intergrax.runtime.policy.rules.schema import PolicyRuleAction

        return PolicyRuleAction.ALLOW


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


def _install_eps(monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _ep(name: str, attr: str) -> _EntryPoint:
    return _EntryPoint(name, f"{__name__}:{attr}", _GROUP)


def _deny_profile(*, mode: str = "enforce") -> PolicyRulesProfile:
    return PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": _TOOL_ID,
                "action": "deny",
            }
        ],
        policy_enforcement_mode=mode,
    )


def _build_state(bundle: object) -> tuple[object, RuntimeToolInvoker, _CountingExecutor]:
    contract = ToolContract(
        tool_id=_TOOL_ID,
        name=_TOOL_ID,
        description="e2e",
        input_schema=_Input,
        output_schema=_Output,
        side_effects=True,
        error_mapping={},
        risk_level=ToolRiskLevel.LOW,
    )
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=executor,
        scope_policy=None,
    )
    state = build_runtime_state_for_tests(run_id="run-e2e")
    state.context.config.policy_bundle = bundle
    return state, invoker, executor


def test_enforce_deny_blocks_tool_invocation() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="policy.e2e.enforce")
    env.policy_rules = _deny_profile(mode="enforce")
    bundle = wire_policy_bundle(env)

    state, invoker, executor = _build_state(bundle)
    request = ToolExecutionRequest(
        run_id="run-e2e",
        tool_id=_TOOL_ID,
        step_id="1",
        input=_Input(value=1),
    )

    with pytest.raises(DeclarativePolicyViolationError) as exc:
        invoker.invoke(state=state, agent_id="agent-e2e", request=request)

    assert executor.calls == 0
    assert "deny_tool" in exc.value.matched_rule_ids
    trace = next(e for e in state.trace_events if e.step == "declarative_policy_evaluation")
    payload = trace.payload
    assert payload.action == "deny"
    assert payload.enforced is True


def test_audit_only_permits_but_records_deny() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="policy.e2e.audit")
    env.policy_rules = _deny_profile(mode="audit_only")
    bundle = wire_policy_bundle(env)

    state, invoker, executor = _build_state(bundle)
    request = ToolExecutionRequest(
        run_id="run-e2e",
        tool_id=_TOOL_ID,
        step_id="1",
        input=_Input(value=2),
    )

    result = invoker.invoke(state=state, agent_id="agent-e2e", request=request)

    assert result.success is True
    assert executor.calls == 1
    trace = next(e for e in state.trace_events if e.step == "declarative_policy_evaluation")
    payload = trace.payload
    assert payload.would_deny is True
    assert payload.enforced is False
    assert "audit_only_bypass" in payload.reasons


def test_unknown_handler_enforce_denies_without_side_effect() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="policy.e2e.unknown")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": "nonexistent_handler",
                "resource_kind": "tool",
                "resource_id": _TOOL_ID,
                "action": "allow",
            }
        ],
        policy_enforcement_mode="enforce",
    )
    bundle = wire_policy_bundle(env)
    state, invoker, executor = _build_state(bundle)
    request = ToolExecutionRequest(
        run_id="run-e2e",
        tool_id=_TOOL_ID,
        step_id="1",
        input=_Input(value=3),
    )

    with pytest.raises(DeclarativePolicyViolationError):
        invoker.invoke(state=state, agent_id="agent-e2e", request=request)

    assert executor.calls == 0
    trace = next(e for e in state.trace_events if e.step == "declarative_policy_evaluation")
    assert trace.payload.unknown_handler_ids == ("nonexistent_handler",)


def test_require_hitl_blocks_tool_before_orchestration_bridge() -> None:
    """REQUIRE_HITL blocks handler execution at invoker boundary."""
    from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
        DeclarativePolicyHitlRequiredError,
    )

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="policy.e2e.hitl")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": _TOOL_ID,
                "action": "require_hitl",
            }
        ],
        policy_enforcement_mode="enforce",
    )
    bundle = wire_policy_bundle(env)
    state, invoker, executor = _build_state(bundle)
    request = ToolExecutionRequest(
        run_id="run-e2e",
        tool_id=_TOOL_ID,
        step_id="1",
        input=_Input(value=4),
    )

    with pytest.raises(DeclarativePolicyHitlRequiredError) as exc:
        invoker.invoke(state=state, agent_id="agent-e2e", request=request)

    assert executor.calls == 0
    assert "deny_tool" in exc.value.matched_rule_ids


def test_allowlist_rejects_unlisted_external_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_DISCOVER_PLUGINS", "true")
    _install_eps(monkeypatch, [_ep("alpha", "_AlphaHandler")])
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="policy.e2e.allowlist")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        allowed_handler_ids=["other-rule"],
    )
    bundle = wire_policy_bundle(env)
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert "alpha-rule" not in runtime.registry.handler_ids()
    assert runtime.load_report.rejected[0].reason_code is PluginAdmissionReasonCode.NOT_IN_ALLOWLIST
    assert "alpha-rule" in runtime.provenance.rejected_handler_ids


def test_scope_deny_still_wins_over_declarative_allow() -> None:
    from intergrax.runtime.nexus.errors.tool_scope_violation_error import ToolScopeViolationError
    from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="policy.e2e.scope")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": "allowed.tool",
                "action": "allow",
            }
        ],
        policy_enforcement_mode="enforce",
    )
    bundle = wire_policy_bundle(env)

    contract = ToolContract(
        tool_id="blocked.tool",
        name="blocked.tool",
        description="scope",
        input_schema=_Input,
        output_schema=_Output,
        side_effects=False,
        error_mapping={},
        risk_level=ToolRiskLevel.LOW,
    )
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(
        registry=FakeRegistry(contract),
        executor=executor,
        scope_policy=StaticToolScopePolicy(allowed_tools={"allowed.tool"}),
    )
    state = build_runtime_state_for_tests(run_id="run-scope")
    state.context.config.policy_bundle = bundle
    request = ToolExecutionRequest(
        run_id="run-scope",
        tool_id="blocked.tool",
        step_id="1",
        input=_Input(value=1),
    )

    with pytest.raises(ToolScopeViolationError):
        invoker.invoke(state=state, agent_id="agent-e2e", request=request)
    assert executor.calls == 0


def test_require_hitl_satisfied_by_matching_grant() -> None:
    from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
    from intergrax.runtime.policy.declarative_enforcer import DeclarativePolicyEnforcer
    from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="policy.e2e.grant")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": _TOOL_ID,
                "action": "require_hitl",
            }
        ],
        policy_enforcement_mode="enforce",
    )
    bundle = wire_policy_bundle(env)
    assert bundle.declarative_policy_runtime is not None
    enforcer = DeclarativePolicyEnforcer(runtime=bundle.declarative_policy_runtime)
    grant = DeclarativeHitlApprovalGrant(
        grant_id="grant-1",
        invocation_scope_id="dhr_scope",
        task_id="task-1",
        run_id="run-e2e",
        step_id="1",
        tool_id=_TOOL_ID,
        idempotency_key=None,
        matched_rule_ids=("deny_tool",),
        human_request_id="hr-1",
        policy_provenance_digest=bundle.declarative_policy_runtime.provenance.rules_digest_sha256,
        pause_id="pause-1",
        approved_at="2026-08-14T00:00:00+00:00",
    )
    decision = enforcer.evaluate_tool_invocation(
        context=PolicyEvaluationContext(
            tool_id=_TOOL_ID,
            task_id="task-1",
            run_id="run-e2e",
            step_id="1",
            invocation_scope_id="dhr_scope",
            approval_grant=grant,
        )
    )
    assert decision.action.value == "allow"
    assert decision.should_block_execution is False


def test_deny_overrides_matching_grant() -> None:
    from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
    from intergrax.runtime.policy.declarative_enforcer import DeclarativePolicyEnforcer
    from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="policy.e2e.deny_grant")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": _TOOL_ID,
                "action": "deny",
            }
        ],
        policy_enforcement_mode="enforce",
    )
    bundle = wire_policy_bundle(env)
    assert bundle.declarative_policy_runtime is not None
    enforcer = DeclarativePolicyEnforcer(runtime=bundle.declarative_policy_runtime)
    grant = DeclarativeHitlApprovalGrant(
        grant_id="grant-1",
        invocation_scope_id="dhr_scope",
        task_id="task-1",
        run_id="run-e2e",
        step_id="1",
        tool_id=_TOOL_ID,
        idempotency_key=None,
        matched_rule_ids=("deny_tool",),
        human_request_id="hr-1",
        policy_provenance_digest=None,
        pause_id="pause-1",
        approved_at="2026-08-14T00:00:00+00:00",
    )
    decision = enforcer.evaluate_tool_invocation(
        context=PolicyEvaluationContext(
            tool_id=_TOOL_ID,
            invocation_scope_id="dhr_scope",
            approval_grant=grant,
        )
    )
    assert decision.should_block_execution is True
    assert decision.action.value == "deny"


def test_grant_satisfaction_requires_identity_dimensions() -> None:
    from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
    from intergrax.runtime.policy.declarative_enforcer import DeclarativePolicyEnforcer
    from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="policy.e2e.grant_strict")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": _TOOL_ID,
                "action": "require_hitl",
            }
        ],
        policy_enforcement_mode="enforce",
    )
    bundle = wire_policy_bundle(env)
    enforcer = DeclarativePolicyEnforcer(runtime=bundle.declarative_policy_runtime)
    grant = DeclarativeHitlApprovalGrant(
        grant_id="grant-1",
        invocation_scope_id="dhr_scope",
        task_id="task-1",
        run_id="run-e2e",
        step_id="1",
        tool_id=_TOOL_ID,
        idempotency_key=None,
        matched_rule_ids=("deny_tool",),
        human_request_id="hr-1",
        policy_provenance_digest=bundle.declarative_policy_runtime.provenance.rules_digest_sha256,
        pause_id="pause-1",
        approved_at="2026-08-14T00:00:00+00:00",
    )
    decision = enforcer.evaluate_tool_invocation(
        context=PolicyEvaluationContext(
            tool_id=_TOOL_ID,
            invocation_scope_id="dhr_scope",
            approval_grant=grant,
        )
    )
    assert decision.should_block_execution is True
