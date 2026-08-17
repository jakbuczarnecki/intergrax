# © Artur Czarnecki. All rights reserved.

"""Unit tests for PolicyRuleRegistry (BLOCK B / CAND-007/008)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext
from intergrax.runtime.policy.rules.loader import load_policy_rules_from_path
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
        rule_id="finance.block_upload",
        handler_id="missing_handler",
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
    assert outcome.rule_id == "finance.block_upload"
    assert "unknown_handler:missing_handler" in outcome.reasons


def test_rule_id_not_used_for_handler_lookup() -> None:
    registry = PolicyRuleRegistry()
    rule = DeclarativePolicyRule(
        rule_id="deny_tool",
        handler_id="missing_handler",
        resource_kind="tool",
        resource_id="blocked",
        action=PolicyRuleAction.DENY,
    )
    outcome = registry.evaluate_rule(
        rule,
        context=PolicyEvaluationContext(tool_id="blocked"),
    )
    assert outcome.action is PolicyRuleAction.DENY
    assert outcome.unknown_handler is True
    assert outcome.rule_id == "deny_tool"
    assert "unknown_handler:missing_handler" in outcome.reasons


def test_resolve_handler_by_handler_id_with_distinct_rule_id() -> None:
    registry = PolicyRuleRegistry()
    rule = DeclarativePolicyRule(
        rule_id="finance.block_external_uploads",
        handler_id="deny_tool",
        resource_kind="tool",
        resource_id="blocked.tool",
        action=PolicyRuleAction.DENY,
    )
    outcome = registry.evaluate_rule(
        rule,
        context=PolicyEvaluationContext(tool_id="blocked.tool"),
    )
    assert outcome.action is PolicyRuleAction.DENY
    assert outcome.rule_id == "finance.block_external_uploads"


def test_two_rules_same_handler_distinct_evidence() -> None:
    registry = PolicyRuleRegistry()
    rule_a = DeclarativePolicyRule(
        rule_id="finance.block_upload",
        handler_id="deny_tool",
        resource_kind="tool",
        resource_id="tool.a",
        action=PolicyRuleAction.DENY,
    )
    rule_b = DeclarativePolicyRule(
        rule_id="legal.block_export",
        handler_id="deny_tool",
        resource_kind="tool",
        resource_id="tool.b",
        action=PolicyRuleAction.DENY,
    )
    outcome_a = registry.evaluate_rule(
        rule_a,
        context=PolicyEvaluationContext(tool_id="tool.a"),
    )
    outcome_b = registry.evaluate_rule(
        rule_b,
        context=PolicyEvaluationContext(tool_id="tool.b"),
    )
    assert outcome_a.rule_id == "finance.block_upload"
    assert outcome_b.rule_id == "legal.block_export"
    assert outcome_a.action is PolicyRuleAction.DENY
    assert outcome_b.action is PolicyRuleAction.DENY


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
        rule_id="explode_rule",
        handler_id="explode",
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
    assert outcome.rule_id == "explode_rule"


def test_unsupported_conditions_fail_closed() -> None:
    registry = PolicyRuleRegistry()
    rule = DeclarativePolicyRule(
        rule_id="finance.block_upload",
        handler_id="deny_tool",
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
    assert outcome.rule_id == "finance.block_upload"


def test_old_payload_without_handler_id_rejected() -> None:
    with pytest.raises(ValidationError):
        DeclarativePolicyRule.model_validate(
            {
                "rule_id": "deny_tool",
                "resource_kind": "tool",
                "resource_id": "blocked",
                "action": "deny",
            }
        )


def test_loader_accepts_explicit_handler_id(tmp_path: Path) -> None:
    payload = [
        {
            "rule_id": "finance.block_external_uploads",
            "handler_id": "deny_tool",
            "resource_kind": "tool",
            "resource_id": "some_tool",
            "action": "deny",
        }
    ]
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rules = load_policy_rules_from_path(path)
    assert len(rules) == 1
    assert rules[0].rule_id == "finance.block_external_uploads"
    assert rules[0].handler_id == "deny_tool"


def test_loader_rejects_old_coupled_payload(tmp_path: Path) -> None:
    payload = [
        {
            "rule_id": "deny_tool",
            "resource_kind": "tool",
            "resource_id": "blocked",
            "action": "deny",
        }
    ]
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_policy_rules_from_path(path)
