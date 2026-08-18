# © Artur Czarnecki. All rights reserved.

"""Built-in Tool Invocation Control policy catalog and composition (G2C-2B)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.policy_catalog import PolicyDefinition, PolicyDefinitionSource
from intergrax.contracts.tool_invocation_control_policy import (
    TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
    ToolInvocationControlConfig,
)
from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.core.plugins.discovery import EP_POLICY_RULES
from intergrax.runtime.policy.builtin_catalog import (
    TOOL_INVOCATION_CONTROL_HANDLER_ID,
    TOOL_INVOCATION_CONTROL_POLICY_ID,
    TOOL_INVOCATION_CONTROL_VERSION,
    ToolInvocationControlCompositionError,
    build_builtin_policy_catalog,
    built_in_policy_definitions,
    compose_tool_invocation_control_rule,
)
from intergrax.runtime.policy.catalog import (
    UnknownPolicyDefinitionError,
    UnsupportedPolicyDefinitionVersionError,
)
from intergrax.runtime.policy.declarative_enforcer import DeclarativePolicyEnforcer
from intergrax.runtime.policy.policy_bundle import DeclarativePolicyRuntime
from intergrax.runtime.policy.rules.evaluation import (
    PolicyEnforcementMode,
    PolicyEvaluationContext,
)
from intergrax.runtime.policy.rules.provenance import PolicyBundleProvenance
from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry
from intergrax.runtime.policy.rules.schema import DeclarativePolicyRule, PolicyRuleAction

pytestmark = pytest.mark.unit

_CONFIGURED_RULE_ID = "finance.block_admin_delete"
_EMPTY_PROVENANCE = PolicyBundleProvenance(
    source_kind="inline",
    rules_path=None,
    rules_digest_sha256="g2c-2b-test",
    handler_provenance=(),
)


def _resolved_definition() -> PolicyDefinition:
    catalog = build_builtin_policy_catalog()
    return catalog.resolve(
        policy_id=TOOL_INVOCATION_CONTROL_POLICY_ID,
        version=TOOL_INVOCATION_CONTROL_VERSION,
    )


def _enforcer_for_rule(rule: DeclarativePolicyRule) -> DeclarativePolicyEnforcer:
    runtime = DeclarativePolicyRuntime(
        registry=PolicyRuleRegistry(),
        rules=(rule,),
        load_report=DomainPluginLoadReport.empty(EP_POLICY_RULES),
        enforcement_mode=PolicyEnforcementMode.ENFORCE,
        provenance=_EMPTY_PROVENANCE,
    )
    return DeclarativePolicyEnforcer(runtime=runtime)


def test_builtin_catalog_contains_exactly_one_definition() -> None:
    definitions = built_in_policy_definitions()
    assert len(definitions) == 1
    definition = definitions[0]
    assert definition.policy_id == TOOL_INVOCATION_CONTROL_POLICY_ID
    assert definition.version == TOOL_INVOCATION_CONTROL_VERSION
    assert definition.handler_id == TOOL_INVOCATION_CONTROL_HANDLER_ID
    assert (
        definition.configuration_contract_id
        == TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID
    )
    assert definition.source is PolicyDefinitionSource.BUILT_IN


def test_build_builtin_policy_catalog_resolves_exact_version() -> None:
    catalog = build_builtin_policy_catalog()
    resolved = catalog.resolve(
        policy_id=TOOL_INVOCATION_CONTROL_POLICY_ID,
        version=TOOL_INVOCATION_CONTROL_VERSION,
    )
    assert resolved is catalog.definitions()[0]


def test_catalog_unknown_policy_id_fails() -> None:
    catalog = build_builtin_policy_catalog()
    with pytest.raises(UnknownPolicyDefinitionError):
        catalog.resolve(policy_id="missing", version="1")


def test_catalog_unsupported_version_fails() -> None:
    catalog = build_builtin_policy_catalog()
    with pytest.raises(UnsupportedPolicyDefinitionVersionError):
        catalog.resolve(
            policy_id=TOOL_INVOCATION_CONTROL_POLICY_ID,
            version="2",
        )


def test_tool_invocation_control_config_is_immutable_and_extra_forbid() -> None:
    config = ToolInvocationControlConfig(
        tool_id="admin.delete_user",
        action=PolicyRuleAction.DENY,
    )
    with pytest.raises(ValidationError):
        ToolInvocationControlConfig(
            tool_id="admin.delete_user",
            action=PolicyRuleAction.DENY,
            extra_field=True,
        )
    with pytest.raises(ValidationError):
        config.tool_id = "other"


def test_tool_invocation_control_config_rejects_blank_tool_id() -> None:
    with pytest.raises(ValidationError):
        ToolInvocationControlConfig(tool_id="   ", action=PolicyRuleAction.DENY)


def test_tool_invocation_control_config_accepts_wildcard_tool_id() -> None:
    config = ToolInvocationControlConfig(tool_id="*", action=PolicyRuleAction.ALLOW)
    assert config.tool_id == "*"


def test_compose_rule_maps_config_to_declarative_rule() -> None:
    definition = _resolved_definition()
    config = ToolInvocationControlConfig(
        tool_id="admin.delete_user",
        action=PolicyRuleAction.DENY,
    )
    rule = compose_tool_invocation_control_rule(
        rule_id=_CONFIGURED_RULE_ID,
        definition=definition,
        config=config,
    )
    assert rule == DeclarativePolicyRule(
        rule_id=_CONFIGURED_RULE_ID,
        handler_id=TOOL_INVOCATION_CONTROL_HANDLER_ID,
        resource_kind="tool",
        resource_id="admin.delete_user",
        action=PolicyRuleAction.DENY,
        conditions={},
    )


def test_identity_layers_remain_distinct() -> None:
    definition = _resolved_definition()
    rule = compose_tool_invocation_control_rule(
        rule_id=_CONFIGURED_RULE_ID,
        definition=definition,
        config=ToolInvocationControlConfig(
            tool_id="admin.delete_user",
            action=PolicyRuleAction.DENY,
        ),
    )
    assert definition.policy_id == TOOL_INVOCATION_CONTROL_POLICY_ID
    assert rule.rule_id == _CONFIGURED_RULE_ID
    assert rule.handler_id == TOOL_INVOCATION_CONTROL_HANDLER_ID
    assert (
        definition.policy_id
        != rule.rule_id
        != rule.handler_id
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("policy_id", "other_policy"),
        ("version", "2"),
        ("handler_id", "other_handler"),
        ("configuration_contract_id", "other.contract.v1"),
        ("source", PolicyDefinitionSource.PLUGIN),
    ],
)
def test_compose_rejects_mismatched_definition(
    field_name: str,
    value: object,
) -> None:
    definition = _resolved_definition().model_copy(update={field_name: value})
    config = ToolInvocationControlConfig(
        tool_id="blocked.tool",
        action=PolicyRuleAction.DENY,
    )
    with pytest.raises(ToolInvocationControlCompositionError):
        compose_tool_invocation_control_rule(
            rule_id=_CONFIGURED_RULE_ID,
            definition=definition,
            config=config,
        )


def test_compose_rejects_blank_rule_id() -> None:
    definition = _resolved_definition()
    config = ToolInvocationControlConfig(
        tool_id="blocked.tool",
        action=PolicyRuleAction.DENY,
    )
    with pytest.raises(ToolInvocationControlCompositionError):
        compose_tool_invocation_control_rule(
            rule_id="   ",
            definition=definition,
            config=config,
        )


def test_enforcement_deny_via_catalog_composition_path() -> None:
    definition = _resolved_definition()
    rule = compose_tool_invocation_control_rule(
        rule_id=_CONFIGURED_RULE_ID,
        definition=definition,
        config=ToolInvocationControlConfig(
            tool_id="blocked.tool",
            action=PolicyRuleAction.DENY,
        ),
    )
    decision = _enforcer_for_rule(rule).evaluate_tool_invocation(
        context=PolicyEvaluationContext(tool_id="blocked.tool"),
    )
    assert decision.action is PolicyRuleAction.DENY
    assert decision.matched_rule_ids == (_CONFIGURED_RULE_ID,)
    assert decision.should_block_execution is True
    assert TOOL_INVOCATION_CONTROL_POLICY_ID not in decision.matched_rule_ids
    assert TOOL_INVOCATION_CONTROL_HANDLER_ID not in decision.matched_rule_ids


def test_enforcement_require_hitl_via_catalog_composition_path() -> None:
    definition = _resolved_definition()
    rule = compose_tool_invocation_control_rule(
        rule_id=_CONFIGURED_RULE_ID,
        definition=definition,
        config=ToolInvocationControlConfig(
            tool_id="governed.tool",
            action=PolicyRuleAction.REQUIRE_HITL,
        ),
    )
    decision = _enforcer_for_rule(rule).evaluate_tool_invocation(
        context=PolicyEvaluationContext(tool_id="governed.tool"),
    )
    assert decision.action is PolicyRuleAction.REQUIRE_HITL
    assert decision.requires_hitl is True
    assert decision.matched_rule_ids == (_CONFIGURED_RULE_ID,)
    assert decision.should_block_execution is True


def test_enforcement_allow_via_catalog_composition_path() -> None:
    definition = _resolved_definition()
    rule = compose_tool_invocation_control_rule(
        rule_id=_CONFIGURED_RULE_ID,
        definition=definition,
        config=ToolInvocationControlConfig(
            tool_id="allowed.tool",
            action=PolicyRuleAction.ALLOW,
        ),
    )
    decision = _enforcer_for_rule(rule).evaluate_tool_invocation(
        context=PolicyEvaluationContext(tool_id="allowed.tool"),
    )
    assert decision.action is PolicyRuleAction.ALLOW
    assert decision.matched_rule_ids == ()
