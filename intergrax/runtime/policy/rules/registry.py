# © Artur Czarnecki. All rights reserved.

"""Typed policy rule handler registry (Phase H-APP.2.4, BLOCK B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.core.plugins.admission import PluginAdmissionReasonCode
from intergrax.core.plugins.discovery import ConflictPolicy
from intergrax.runtime.policy.rules.evaluation import PolicyEvaluationContext, PolicyRuleEvaluationOutcome
from intergrax.runtime.policy.rules.schema import DeclarativePolicyRule, PolicyRuleAction

_SHIPPED_HANDLER_IDS = frozenset({"deny_tool"})


@runtime_checkable
class PolicyRuleHandler(Protocol):
    rule_id: str

    def evaluate(
        self,
        rule: DeclarativePolicyRule,
        *,
        context: PolicyEvaluationContext,
    ) -> PolicyRuleAction: ...


class DenyToolRuleHandler:
    """Deny matching tool_id rules."""

    rule_id = "deny_tool"

    def evaluate(
        self,
        rule: DeclarativePolicyRule,
        *,
        context: PolicyEvaluationContext,
    ) -> PolicyRuleAction:
        if rule.resource_kind != "tool":
            return PolicyRuleAction.ALLOW
        if rule.resource_id == "*" or rule.resource_id == context.tool_id:
            return rule.action
        return PolicyRuleAction.ALLOW


@dataclass(frozen=True, slots=True)
class PolicyRuleHandlerRegistrationResult:
    accepted: bool
    handler_id: str
    reason: str | None = None
    reason_code: PluginAdmissionReasonCode | None = None


class PolicyRuleRegistry:
    """Registry of typed rule handlers — no eval/getattr."""

    def __init__(self) -> None:
        self._handlers: dict[str, PolicyRuleHandler] = {}
        self.register(DenyToolRuleHandler())

    def handler_ids(self) -> frozenset[str]:
        return frozenset(self._handlers)

    def resolve(self, handler_id: str) -> PolicyRuleHandler | None:
        return self._handlers.get(handler_id)

    def register(
        self,
        handler: PolicyRuleHandler,
        *,
        conflict_policy: ConflictPolicy = "error",
    ) -> PolicyRuleHandlerRegistrationResult:
        handler_id = handler.rule_id
        existing = self._handlers.get(handler_id)
        if existing is not None:
            if handler_id in _SHIPPED_HANDLER_IDS:
                return PolicyRuleHandlerRegistrationResult(
                    accepted=False,
                    handler_id=handler_id,
                    reason=f"Shipped policy handler {handler_id!r} cannot be overridden.",
                    reason_code=PluginAdmissionReasonCode.SHIPPED_ID_COLLISION,
                )
            if conflict_policy == "skip":
                return PolicyRuleHandlerRegistrationResult(
                    accepted=False,
                    handler_id=handler_id,
                    reason=f"Policy handler {handler_id!r} already registered.",
                    reason_code=PluginAdmissionReasonCode.ALREADY_REGISTERED,
                )
            if conflict_policy == "override":
                self._handlers[handler_id] = handler
                return PolicyRuleHandlerRegistrationResult(
                    accepted=True,
                    handler_id=handler_id,
                )
            return PolicyRuleHandlerRegistrationResult(
                accepted=False,
                handler_id=handler_id,
                reason=f"Policy handler {handler_id!r} already registered.",
                reason_code=PluginAdmissionReasonCode.PLUGIN_ID_COLLISION,
            )
        self._handlers[handler_id] = handler
        return PolicyRuleHandlerRegistrationResult(
            accepted=True,
            handler_id=handler_id,
        )

    def evaluate_rule(
        self,
        rule: DeclarativePolicyRule,
        *,
        context: PolicyEvaluationContext,
    ) -> PolicyRuleEvaluationOutcome:
        if rule.conditions:
            return PolicyRuleEvaluationOutcome(
                rule_id=rule.rule_id,
                action=PolicyRuleAction.DENY,
                reasons=(f"unsupported_conditions:rule={rule.rule_id}",),
                unsupported_conditions=True,
            )

        handler = self._handlers.get(rule.handler_id)
        if handler is None:
            return PolicyRuleEvaluationOutcome(
                rule_id=rule.rule_id,
                action=PolicyRuleAction.DENY,
                reasons=(f"unknown_handler:{rule.handler_id}",),
                unknown_handler=True,
            )

        try:
            action = handler.evaluate(rule, context=context)
        except Exception as exc:
            return PolicyRuleEvaluationOutcome(
                rule_id=rule.rule_id,
                action=PolicyRuleAction.DENY,
                reasons=(f"handler_exception:{type(exc).__name__}",),
                handler_exception=True,
            )

        if not isinstance(action, PolicyRuleAction):
            return PolicyRuleEvaluationOutcome(
                rule_id=rule.rule_id,
                action=PolicyRuleAction.DENY,
                reasons=(f"invalid_handler_outcome:{rule.rule_id}",),
                handler_exception=True,
            )

        return PolicyRuleEvaluationOutcome(
            rule_id=rule.rule_id,
            action=action,
        )
