# © Artur Czarnecki. All rights reserved.

"""Unit tests for DeclarativePolicyEnforcer (BLOCK B / CAND-007)."""

from __future__ import annotations

import pytest

from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.core.plugins.discovery import EP_POLICY_RULES
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

_EMPTY_PROVENANCE = PolicyBundleProvenance(
    source_kind="inline",
    rules_path=None,
    rules_digest_sha256="abc",
    handler_provenance=(),
)


def _runtime(
    rules: tuple[DeclarativePolicyRule, ...],
    *,
    mode: PolicyEnforcementMode = PolicyEnforcementMode.ENFORCE,
) -> DeclarativePolicyRuntime:
    return DeclarativePolicyRuntime(
        registry=PolicyRuleRegistry(),
        rules=rules,
        load_report=DomainPluginLoadReport.empty(EP_POLICY_RULES),
        enforcement_mode=mode,
        provenance=_EMPTY_PROVENANCE,
    )


def test_no_matching_rules_allow() -> None:
    runtime = _runtime(())
    enforcer = DeclarativePolicyEnforcer(runtime=runtime)
    decision = enforcer.evaluate_tool_invocation(
        context=PolicyEvaluationContext(tool_id="any"),
    )
    assert decision.action is PolicyRuleAction.ALLOW
    assert decision.matched_rule_ids == ()


def test_exact_tool_match_deny() -> None:
    rule = DeclarativePolicyRule(
        rule_id="deny_tool",
        resource_kind="tool",
        resource_id="blocked.tool",
        action=PolicyRuleAction.DENY,
    )
    enforcer = DeclarativePolicyEnforcer(runtime=_runtime((rule,)))
    decision = enforcer.evaluate_tool_invocation(
        context=PolicyEvaluationContext(tool_id="blocked.tool"),
    )
    assert decision.action is PolicyRuleAction.DENY
    assert decision.matched_rule_ids == ("deny_tool",)


def test_wildcard_match() -> None:
    rule = DeclarativePolicyRule(
        rule_id="deny_tool",
        resource_kind="tool",
        resource_id="*",
        action=PolicyRuleAction.REQUIRE_HITL,
    )
    enforcer = DeclarativePolicyEnforcer(runtime=_runtime((rule,)))
    decision = enforcer.evaluate_tool_invocation(
        context=PolicyEvaluationContext(tool_id="any.tool"),
    )
    assert decision.action is PolicyRuleAction.REQUIRE_HITL
    assert decision.requires_hitl is True


def test_precedence_deny_over_hitl_over_allow() -> None:
    rules = (
        DeclarativePolicyRule(
            rule_id="deny_tool",
            resource_kind="tool",
            resource_id="x",
            action=PolicyRuleAction.ALLOW,
        ),
        DeclarativePolicyRule(
            rule_id="deny_tool",
            resource_kind="tool",
            resource_id="x",
            action=PolicyRuleAction.REQUIRE_HITL,
        ),
        DeclarativePolicyRule(
            rule_id="deny_tool",
            resource_kind="tool",
            resource_id="x",
            action=PolicyRuleAction.DENY,
        ),
    )
    enforcer = DeclarativePolicyEnforcer(runtime=_runtime(rules))
    decision = enforcer.evaluate_tool_invocation(
        context=PolicyEvaluationContext(tool_id="x"),
    )
    assert decision.action is PolicyRuleAction.DENY


def test_unknown_handler_deny() -> None:
    rule = DeclarativePolicyRule(
        rule_id="nonexistent_handler",
        resource_kind="tool",
        resource_id="x",
        action=PolicyRuleAction.ALLOW,
    )
    enforcer = DeclarativePolicyEnforcer(runtime=_runtime((rule,)))
    decision = enforcer.evaluate_tool_invocation(
        context=PolicyEvaluationContext(tool_id="x"),
    )
    assert decision.action is PolicyRuleAction.DENY
    assert decision.unknown_handler_ids == ("nonexistent_handler",)


def test_audit_only_records_but_does_not_enforce() -> None:
    rule = DeclarativePolicyRule(
        rule_id="deny_tool",
        resource_kind="tool",
        resource_id="blocked",
        action=PolicyRuleAction.DENY,
    )
    runtime = _runtime((rule,), mode=PolicyEnforcementMode.AUDIT_ONLY)
    decision = DeclarativePolicyEnforcer(runtime=runtime).evaluate_tool_invocation(
        context=PolicyEvaluationContext(tool_id="blocked"),
    )
    assert decision.would_deny is True
    assert decision.enforced is False
    assert decision.should_block_execution is False
    assert "audit_only_bypass" in decision.reasons


def test_enforce_blocks() -> None:
    rule = DeclarativePolicyRule(
        rule_id="deny_tool",
        resource_kind="tool",
        resource_id="blocked",
        action=PolicyRuleAction.DENY,
    )
    runtime = _runtime((rule,), mode=PolicyEnforcementMode.ENFORCE)
    decision = DeclarativePolicyEnforcer(runtime=runtime).evaluate_tool_invocation(
        context=PolicyEvaluationContext(tool_id="blocked"),
    )
    assert decision.enforced is True
    assert decision.should_block_execution is True
