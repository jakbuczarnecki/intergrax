# © Artur Czarnecki. All rights reserved.

"""Unit tests for PolicyRuleRegistry (BLOCK B / CAND-007/008)."""

from __future__ import annotations

import pytest

from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext
from intergrax.runtime.policy.rules.registry import (
    DenyToolRuleHandler,
    PolicyRuleRegistry,
)
from intergrax.runtime.policy.rules.schema import DeclarativePolicyRule, PolicyRuleAction

pytestmark = pytest.mark.unit


class _ExplodingHandler:
    rule_id = "explode"

    def evaluate(self, rule: DeclarativePolicyRule, *, context: PolicyEvaluationContext) -> PolicyRuleAction:
        raise RuntimeError("boom")


class _DuplicateHandler:
    rule_id = "dup-rule"

    def evaluate(self, rule: DeclarativePolicyRule, *, context: PolicyEvaluationContext) -> PolicyRuleAction:
        return PolicyRuleAction.ALLOW


def test_shipped_deny_tool_handler_exists() -> None:
    registry = PolicyRuleRegistry()
    assert registry.resolve("deny_tool") is not None


def test_unknown_handler_fail_closed() -> None:
    registry = PolicyRuleRegistry()
    rule = DeclarativePolicyRule(
        rule_id="missing",
        resource_kind="tool",
        resource_id="x",
        action=PolicyRuleAction.ALLOW,
    )
    outcome = registry.evaluate_rule(
        rule,
        context=PolicyEvaluationContext(tool_id="x"),
    )
    assert outcome.action is PolicyRuleAction.DENY
    assert outcome.unknown_handler is True
    assert "unknown_handler:missing" in outcome.reasons


def test_duplicate_handler_id_rejected() -> None:
    registry = PolicyRuleRegistry()
    first = registry.register(_DuplicateHandler())
    second = registry.register(_DuplicateHandler())
    assert first.accepted is True
    assert second.accepted is False
    assert second.reason_code is PluginAdmissionReasonCode.PLUGIN_ID_COLLISION


def test_shipped_handler_cannot_be_overridden() -> None:
    registry = PolicyRuleRegistry()

    class _FakeDeny:
        rule_id = "deny_tool"

        def evaluate(
            self,
            rule: DeclarativePolicyRule,
            *,
            context: PolicyEvaluationContext,
        ) -> PolicyRuleAction:
            return PolicyRuleAction.ALLOW

    result = registry.register(_FakeDeny())
    assert result.accepted is False
    assert result.reason_code is PluginAdmissionReasonCode.SHIPPED_ID_COLLISION


def test_handler_exception_fail_closed() -> None:
    registry = PolicyRuleRegistry()
    registry.register(_ExplodingHandler())
    rule = DeclarativePolicyRule(
        rule_id="explode",
        resource_kind="tool",
        resource_id="*",
        action=PolicyRuleAction.ALLOW,
    )
    outcome = registry.evaluate_rule(
        rule,
        context=PolicyEvaluationContext(tool_id="any"),
    )
    assert outcome.action is PolicyRuleAction.DENY
    assert outcome.handler_exception is True


def test_unsupported_conditions_fail_closed() -> None:
    registry = PolicyRuleRegistry()
    rule = DeclarativePolicyRule(
        rule_id="deny_tool",
        resource_kind="tool",
        resource_id="blocked",
        action=PolicyRuleAction.DENY,
        conditions={"tenant": "x"},
    )
    outcome = registry.evaluate_rule(
        rule,
        context=PolicyEvaluationContext(tool_id="blocked"),
    )
    assert outcome.action is PolicyRuleAction.DENY
    assert outcome.unsupported_conditions is True
