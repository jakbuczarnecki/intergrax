# © Artur Czarnecki. All rights reserved.

"""Typed policy rule handler registry (Phase H-APP.2.4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.runtime.policy.rules.schema import DeclarativePolicyRule, PolicyRuleAction


@runtime_checkable
class PolicyRuleHandler(Protocol):
    rule_id: str

    def evaluate(self, rule: DeclarativePolicyRule, *, context: dict[str, str]) -> PolicyRuleAction: ...


class DenyToolRuleHandler:
    """Deny matching tool_id rules."""

    rule_id = "deny_tool"

    def evaluate(self, rule: DeclarativePolicyRule, *, context: dict[str, str]) -> PolicyRuleAction:
        if rule.resource_kind != "tool":
            return PolicyRuleAction.ALLOW
        tool_id = context.get("tool_id", "")
        if rule.resource_id == "*" or rule.resource_id == tool_id:
            return rule.action
        return PolicyRuleAction.ALLOW


class PolicyRuleRegistry:
    """Registry of typed rule handlers — no eval/getattr."""

    def __init__(self) -> None:
        self._handlers: dict[str, PolicyRuleHandler] = {}
        self.register(DenyToolRuleHandler())

    def register(self, handler: PolicyRuleHandler) -> None:
        self._handlers[handler.rule_id] = handler

    def evaluate_rule(
        self,
        rule: DeclarativePolicyRule,
        *,
        context: dict[str, str],
    ) -> PolicyRuleAction:
        handler = self._handlers.get(rule.rule_id)
        if handler is None:
            return PolicyRuleAction.ALLOW
        return handler.evaluate(rule, context=context)
