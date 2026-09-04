# © Artur Czarnecki. All rights reserved.

"""LKW declarative policy wiring for meaningful side-effect authorization (UE-11G-C1-R4-F-D1)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyViolationError,
)
from intergrax.runtime.policy.declarative_enforcer import (
    DeclarativePolicyEnforcer,
    resolve_declarative_policy_enforcer,
)
from intergrax.runtime.policy.rules.evaluation import (
    PolicyEnforcementMode,
    PolicyEvaluationContext,
)
from intergrax.runtime.policy.rules.schema import PolicyRuleAction
from intergrax.runtime.policy.side_effect_authorization_errors import (
    MeaningfulSideEffectAuthorizationRequiredError,
    SideEffectAuthorizationFailureReason,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.rag.ingest_service import RAG_INGEST_TOOL_ID
from intergrax.tools.providers.rag.service import RAG_TOOL_ID
from intergrax.tools.registry import ToolRegistry
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.policy_profile import (
    LKW_RAG_INGEST_ALLOW_RULE_ID,
    build_local_workspace_policy_rules_profile,
    local_workspace_meaningful_side_effect_tool_ids,
)
from testing_support.builder import (
    build_runtime_state_for_tests,
    canonical_execution_identity_scope,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TASK_ID = mint_task_id()
_RUN_ID = mint_run_id()
_AGENT_ID = "local_indexer"
_UNKNOWN_SIDE_EFFECT_TOOL_ID = "lkw.probe.unknown_side_effect"


class _ValueInput(BaseModel):
    value: int


class _ValueOutput(BaseModel):
    result: int


class _AllowAllScopePolicy:
    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        return True


class _OkHandler:
    def execute(self, request: ToolExecutionRequest[_ValueInput]) -> _ValueOutput:
        return _ValueOutput(result=request.input.value)


def _register_tool(
    registry: ToolRegistry,
    *,
    tool_id: str,
    side_effects: bool,
) -> None:
    registry.register(
        contract=ToolContract(
            tool_id=tool_id,
            name=tool_id,
            description=tool_id,
            input_schema=_ValueInput,
            output_schema=_ValueOutput,
            error_mapping={},
            side_effects=side_effects,
        ),
        handler=_OkHandler(),
    )


def _lkw_policy_bundle() -> object:
    env = build_local_workspace_environment_profile()
    return wire_policy_bundle(env)


def _state_with_lkw_policy():
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    state.context.config.policy_bundle = _lkw_policy_bundle()
    return state


def test_lkw_environment_profile_declares_enforce_policy_rules() -> None:
    profile = build_local_workspace_policy_rules_profile()
    assert profile.policy_enforcement_mode is PolicyEnforcementMode.ENFORCE
    assert profile.rules_path is not None
    assert profile.rules_path.name == "product.yaml"


def test_lkw_environment_wiring_builds_declarative_policy_runtime() -> None:
    env = build_local_workspace_environment_profile()
    bundle = wire_policy_bundle(env)
    runtime = bundle.declarative_policy_runtime
    assert runtime is not None
    assert runtime.enforcement_mode is PolicyEnforcementMode.ENFORCE
    rule_ids = {rule.rule_id for rule in runtime.rules}
    assert LKW_RAG_INGEST_ALLOW_RULE_ID in rule_ids


def test_lkw_host_composition_resolves_declarative_policy_enforcer() -> None:
    env = build_local_workspace_environment_profile()
    bundle = wire_policy_bundle(env)
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    state.context.config.policy_bundle = bundle
    enforcer = resolve_declarative_policy_enforcer(state)
    assert enforcer is not None
    assert enforcer.runtime.enforcement_mode is PolicyEnforcementMode.ENFORCE


def test_lkw_policy_allows_rag_ingest_document_for_local_indexer() -> None:
    bundle = _lkw_policy_bundle()
    enforcer = DeclarativePolicyEnforcer(runtime=bundle.declarative_policy_runtime)
    decision = enforcer.evaluate_tool_invocation(
        context=PolicyEvaluationContext(
            tool_id=RAG_INGEST_TOOL_ID,
            agent_id=_AGENT_ID,
            task_id=_TASK_ID,
            run_id=_RUN_ID,
            step_id="step1",
        ),
    )
    assert decision.action is PolicyRuleAction.ALLOW
    assert LKW_RAG_INGEST_ALLOW_RULE_ID in decision.matched_rule_ids or not decision.matched_rule_ids


def test_lkw_runtime_invoker_allows_rag_ingest_document() -> None:
    registry = ToolRegistry()
    _register_tool(registry, tool_id=RAG_INGEST_TOOL_ID, side_effects=True)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=_OkHandler(),
        scope_policy=_AllowAllScopePolicy(),
        pre_effect_coordinator=IdempotencyPreEffectCoordinator(
            idempotency_store=InMemoryIdempotencyStore(),
        ),
    )
    state = _state_with_lkw_policy()
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=RAG_INGEST_TOOL_ID,
        input=_ValueInput(value=1),
        idempotency_key="lkw-ingest-key",
    )
    with canonical_execution_identity_scope(_RUN_ID):
        result = invoker.invoke(state=state, agent_id=_AGENT_ID, request=request)
    assert result.success is True


def test_lkw_runtime_invoker_denies_explicitly_denied_side_effect() -> None:
    env = build_local_workspace_environment_profile()
    deny_profile = PolicyRulesProfile(
        rules_path=env.policy_rules.rules_path if env.policy_rules else None,
        inline_rules=[
            {
                "rule_id": "lkw.test.deny.ingest",
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": RAG_INGEST_TOOL_ID,
                "action": "deny",
            }
        ],
        policy_enforcement_mode=PolicyEnforcementMode.ENFORCE,
    )
    env = env.model_copy(update={"policy_rules": deny_profile})
    bundle = wire_policy_bundle(env)
    registry = ToolRegistry()
    _register_tool(registry, tool_id=RAG_INGEST_TOOL_ID, side_effects=True)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=_OkHandler(),
        scope_policy=_AllowAllScopePolicy(),
    )
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    state.context.config.policy_bundle = bundle
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=RAG_INGEST_TOOL_ID,
        input=_ValueInput(value=1),
        idempotency_key="lkw-deny-key",
    )
    with canonical_execution_identity_scope(_RUN_ID):
        with pytest.raises(DeclarativePolicyViolationError):
            invoker.invoke(state=state, agent_id=_AGENT_ID, request=request)


def test_lkw_runtime_invoker_denies_unknown_side_effect_when_explicitly_denied() -> None:
    env = build_local_workspace_environment_profile()
    deny_profile = PolicyRulesProfile(
        inline_rules=[
            {
                "rule_id": "lkw.test.deny.unknown",
                "handler_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": _UNKNOWN_SIDE_EFFECT_TOOL_ID,
                "action": "deny",
            }
        ],
        policy_enforcement_mode=PolicyEnforcementMode.ENFORCE,
    )
    env = env.model_copy(update={"policy_rules": deny_profile})
    bundle = wire_policy_bundle(env)
    registry = ToolRegistry()
    _register_tool(registry, tool_id=_UNKNOWN_SIDE_EFFECT_TOOL_ID, side_effects=True)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=_OkHandler(),
        scope_policy=_AllowAllScopePolicy(),
    )
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    state.context.config.policy_bundle = bundle
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=_UNKNOWN_SIDE_EFFECT_TOOL_ID,
        input=_ValueInput(value=1),
        idempotency_key="lkw-unknown-key",
    )
    with canonical_execution_identity_scope(_RUN_ID):
        with pytest.raises(DeclarativePolicyViolationError):
            invoker.invoke(state=state, agent_id=_AGENT_ID, request=request)


def test_unconfigured_application_fails_closed_for_side_effect_tools() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="unconfigured.product")
    assert env.policy_rules is None
    bundle = wire_policy_bundle(env)
    assert bundle.declarative_policy_runtime is None
    registry = ToolRegistry()
    _register_tool(registry, tool_id=RAG_INGEST_TOOL_ID, side_effects=True)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=_OkHandler(),
        scope_policy=_AllowAllScopePolicy(),
    )
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    state.context.config.policy_bundle = bundle
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=RAG_INGEST_TOOL_ID,
        input=_ValueInput(value=1),
        idempotency_key="unconfigured-key",
    )
    with canonical_execution_identity_scope(_RUN_ID):
        with pytest.raises(MeaningfulSideEffectAuthorizationRequiredError) as exc_info:
            invoker.invoke(state=state, agent_id=_AGENT_ID, request=request)
    assert exc_info.value.reason is SideEffectAuthorizationFailureReason.NOT_CONFIGURED


def test_read_only_rag_retrieve_does_not_require_side_effect_authorization() -> None:
    registry = ToolRegistry()
    _register_tool(registry, tool_id=RAG_TOOL_ID, side_effects=False)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=_OkHandler(),
        scope_policy=_AllowAllScopePolicy(),
    )
    state = build_runtime_state_for_tests(run_id=_RUN_ID)
    state.context.config.policy_bundle = None
    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id=RAG_TOOL_ID,
        input=_ValueInput(value=1),
    )
    with canonical_execution_identity_scope(_RUN_ID):
        result = invoker.invoke(state=state, agent_id=_AGENT_ID, request=request)
    assert result.success is True


def test_lab_environment_does_not_inherit_lkw_policy_rules() -> None:
    from intergrax.applications._shared.lab_environment_profile import (
        _LAB_POLICY_RULES,
        build_lab_environment_profile,
    )
    from lab_application.host.settings import LabApplicationSettings

    lab_env = build_lab_environment_profile(LabApplicationSettings.from_env())
    lkw_env = build_local_workspace_environment_profile()
    assert lkw_env.policy_rules is not None
    assert lkw_env.policy_rules.rules_path is not None
    assert "local_workspace_application" in str(lkw_env.policy_rules.rules_path)
    if lab_env.policy_rules is not None and lab_env.policy_rules.rules_path is not None:
        assert lab_env.policy_rules.rules_path == _LAB_POLICY_RULES
        assert lab_env.policy_rules.rules_path != lkw_env.policy_rules.rules_path


def test_lkw_meaningful_side_effect_inventory_is_least_privilege_for_c1() -> None:
    authorized = local_workspace_meaningful_side_effect_tool_ids()
    assert authorized == (RAG_INGEST_TOOL_ID,)
