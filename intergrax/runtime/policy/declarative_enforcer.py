# © Artur Czarnecki. All rights reserved.

"""Declarative policy enforcement at the tool invocation boundary (BLOCK B / CAND-007)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.runtime.policy.policy_bundle import DeclarativePolicyRuntime
from intergrax.runtime.policy.rules.evaluation import (
    PolicyEnforcementDecision,
    PolicyEnforcementMode,
    PolicyEvaluationContext,
    PolicyRuleEvaluationOutcome,
)
from intergrax.runtime.policy.rules.schema import DeclarativePolicyRule, PolicyRuleAction

if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

_ACTION_PRECEDENCE = {
    PolicyRuleAction.DENY: 3,
    PolicyRuleAction.REQUIRE_HITL: 2,
    PolicyRuleAction.ALLOW: 1,
}


def _rule_applies_to_tool(rule: DeclarativePolicyRule, tool_id: str) -> bool:
    if rule.resource_kind != "tool":
        return False
    return rule.resource_id == "*" or rule.resource_id == tool_id


def _aggregate_action(outcomes: tuple[PolicyRuleEvaluationOutcome, ...]) -> PolicyRuleAction:
    best = PolicyRuleAction.ALLOW
    best_rank = _ACTION_PRECEDENCE[best]
    for outcome in outcomes:
        rank = _ACTION_PRECEDENCE[outcome.action]
        if rank > best_rank:
            best = outcome.action
            best_rank = rank
    return best


def _grant_satisfies_hitl(
    grant: DeclarativeHitlApprovalGrant,
    *,
    context: PolicyEvaluationContext,
    decision: PolicyEnforcementDecision,
) -> bool:
    if context.invocation_scope_id is None:
        return False
    if context.invocation_scope_id != grant.invocation_scope_id:
        return False
    if context.task_id is None:
        return False
    if context.task_id != grant.task_id:
        return False
    if context.run_id is None:
        return False
    if context.run_id != grant.run_id:
        return False
    if context.step_id is None:
        return False
    if context.step_id != grant.step_id:
        return False
    if context.tool_id != grant.tool_id:
        return False
    if grant.idempotency_key is not None:
        if context.idempotency_key != grant.idempotency_key:
            return False
    if set(decision.matched_rule_ids) != set(grant.matched_rule_ids):
        return False
    if (
        decision.provenance_digest is not None
        and decision.provenance_digest != grant.policy_provenance_digest
    ):
        return False
    return True


@dataclass(frozen=True, slots=True)
class DeclarativePolicyEnforcer:
    """Evaluate declarative rules for tool invocations without mutating runtime state."""

    runtime: DeclarativePolicyRuntime

    def evaluate_tool_invocation(
        self,
        *,
        context: PolicyEvaluationContext,
    ) -> PolicyEnforcementDecision:
        applicable_rules = tuple(
            rule
            for rule in self.runtime.rules
            if _rule_applies_to_tool(rule, context.tool_id)
        )
        if not applicable_rules:
            return PolicyEnforcementDecision(
                action=PolicyRuleAction.ALLOW,
                matched_rule_ids=(),
                reasons=(),
                enforcement_mode=self.runtime.enforcement_mode,
                enforced=False,
                would_deny=False,
                requires_hitl=False,
                provenance_digest=self.runtime.provenance.rules_digest_sha256,
            )

        outcomes: list[PolicyRuleEvaluationOutcome] = []
        matched_rule_ids: list[str] = []
        reasons: list[str] = []
        unknown_handler_ids: list[str] = []

        for rule in applicable_rules:
            outcome = self.runtime.registry.evaluate_rule(rule, context=context)
            outcomes.append(outcome)
            if outcome.action is not PolicyRuleAction.ALLOW:
                matched_rule_ids.append(rule.rule_id)
            reasons.extend(outcome.reasons)
            if outcome.unknown_handler:
                unknown_handler_ids.append(rule.handler_id)

        final_action = _aggregate_action(tuple(outcomes))
        would_deny = final_action in (PolicyRuleAction.DENY, PolicyRuleAction.REQUIRE_HITL)
        enforced = (
            self.runtime.enforcement_mode is PolicyEnforcementMode.ENFORCE and would_deny
        )

        audit_reasons = list(reasons)
        if would_deny and not enforced:
            audit_reasons.append("audit_only_bypass")

        decision = PolicyEnforcementDecision(
            action=final_action,
            matched_rule_ids=tuple(matched_rule_ids),
            reasons=tuple(audit_reasons),
            enforcement_mode=self.runtime.enforcement_mode,
            enforced=enforced,
            would_deny=would_deny,
            requires_hitl=final_action is PolicyRuleAction.REQUIRE_HITL,
            unknown_handler_ids=tuple(unknown_handler_ids),
            provenance_digest=self.runtime.provenance.rules_digest_sha256,
        )

        if (
            decision.action is PolicyRuleAction.REQUIRE_HITL
            and decision.enforced
            and isinstance(context.approval_grant, DeclarativeHitlApprovalGrant)
            and _grant_satisfies_hitl(context.approval_grant, context=context, decision=decision)
        ):
            return PolicyEnforcementDecision(
                action=PolicyRuleAction.ALLOW,
                matched_rule_ids=decision.matched_rule_ids,
                reasons=decision.reasons,
                enforcement_mode=decision.enforcement_mode,
                enforced=False,
                would_deny=decision.would_deny,
                requires_hitl=True,
                unknown_handler_ids=decision.unknown_handler_ids,
                provenance_digest=decision.provenance_digest,
            )

        return decision


def resolve_declarative_policy_enforcer(
    state: RuntimeState,
) -> DeclarativePolicyEnforcer | None:
    bundle = state.context.config.policy_bundle
    if bundle is None or bundle.declarative_policy_runtime is None:
        return None
    return DeclarativePolicyEnforcer(runtime=bundle.declarative_policy_runtime)
