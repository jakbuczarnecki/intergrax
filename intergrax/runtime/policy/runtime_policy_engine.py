# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Runtime governance policy engine (architecture §42.11).

Evaluates ``AgentDecision`` and ``ExecutionInterrupt`` against Tier-3 rules.
Use ``PolicyEngine`` facade for unified runtime + replay policy entry (§42.11).
"""

from __future__ import annotations

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.execution_interrupt import ExecutionInterrupt
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.meaningful_side_effect_policy import MeaningfulSideEffectPolicyRule
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
from intergrax.contracts.runtime_execution_policy_admission import (
    RootExecutionAdmissionPolicyRule,
)
from intergrax.contracts.runtime_policy_context import (
    AgentDecisionPolicyContext,
    PreModelPhase,
    PreModelPolicyContext,
)


def _meaningful_side_effect_action_rank(action: PolicyAction) -> int:
    """Explicit strictness ordering — aligned with collaborative policy composition."""
    if action is PolicyAction.DENY:
        return 0
    if action is PolicyAction.REQUIRE_HUMAN:
        return 1
    if action is PolicyAction.ESCALATE:
        return 2
    if action is PolicyAction.ALLOW:
        return 3
    return 0


def _meaningful_side_effect_rule_specificity(rule: MeaningfulSideEffectPolicyRule) -> int:
    """Lower is more specific (action match vs wildcard)."""
    return 0 if rule.action is not None else 1


def _normalize_meaningful_side_effect_rule_action(
    rule: MeaningfulSideEffectPolicyRule,
) -> PolicyAction:
    if rule.decision is PolicyAction.MODIFY:
        return PolicyAction.DENY
    return rule.decision


def _root_execution_admission_rule_specificity(rule: RootExecutionAdmissionPolicyRule) -> int:
    return 0 if rule.execution_operation is not None else 1


def _normalize_root_execution_admission_rule_action(
    rule: RootExecutionAdmissionPolicyRule,
) -> PolicyAction:
    if rule.decision is PolicyAction.MODIFY:
        return PolicyAction.DENY
    return rule.decision


def _root_execution_admission_controlling_rule(
    candidates: list[tuple[RootExecutionAdmissionPolicyRule, PolicyAction]],
) -> tuple[RootExecutionAdmissionPolicyRule, PolicyAction]:
    best_rank = min(_meaningful_side_effect_action_rank(action) for _, action in candidates)
    rank_candidates = [
        pair for pair in candidates if _meaningful_side_effect_action_rank(pair[1]) == best_rank
    ]
    best_specificity = min(
        _root_execution_admission_rule_specificity(rule) for rule, _ in rank_candidates
    )
    specific_candidates = [
        pair
        for pair in rank_candidates
        if _root_execution_admission_rule_specificity(pair[0]) == best_specificity
    ]
    return min(
        specific_candidates,
        key=lambda pair: (
            1 if pair[0].decision is PolicyAction.MODIFY else 0,
            pair[0].rule_id,
        ),
    )


def _meaningful_side_effect_controlling_rule(
    candidates: list[tuple[MeaningfulSideEffectPolicyRule, PolicyAction]],
) -> tuple[MeaningfulSideEffectPolicyRule, PolicyAction]:
    best_rank = min(_meaningful_side_effect_action_rank(action) for _, action in candidates)
    rank_candidates = [
        pair for pair in candidates if _meaningful_side_effect_action_rank(pair[1]) == best_rank
    ]
    best_specificity = min(
        _meaningful_side_effect_rule_specificity(rule) for rule, _ in rank_candidates
    )
    specific_candidates = [
        pair
        for pair in rank_candidates
        if _meaningful_side_effect_rule_specificity(pair[0]) == best_specificity
    ]
    return min(
        specific_candidates,
        key=lambda pair: (
            1 if pair[0].decision is PolicyAction.MODIFY else 0,
            pair[0].rule_id,
        ),
    )


class RuntimePolicyEngine:
    """Rule-based policy evaluation for live task execution."""

    def __init__(
        self,
        *,
        meaningful_side_effect_rules: tuple[MeaningfulSideEffectPolicyRule, ...] = (),
        root_execution_admission_rules: tuple[RootExecutionAdmissionPolicyRule, ...] = (),
    ) -> None:
        self._meaningful_side_effect_rules = meaningful_side_effect_rules
        self._root_execution_admission_rules = root_execution_admission_rules

    def has_root_execution_admission_rules(self) -> bool:
        return bool(self._root_execution_admission_rules)

    def evaluate_root_execution_admission(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_id: str,
        collaborative_authority_scopes: tuple[str, ...],
        execution_operation: str,
        resource_scope: str | None = None,
    ) -> tuple[PolicyDecision, tuple[str, ...] | None]:
        """Authorize worker root execution admission — fail closed when unconfigured."""
        if not self._root_execution_admission_rules:
            return (
                PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="root_execution_admission_unconfigured",
                    enforcement_level=EnforcementLevel.MANDATORY,
                    policy_rule_id="default.root_execution_admission.unconfigured",
                ),
                None,
            )
        if not principal_id.strip():
            return (
                PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="root_execution_admission_principal_missing",
                    enforcement_level=EnforcementLevel.MANDATORY,
                    policy_rule_id="default.root_execution_admission.principal",
                ),
                None,
            )

        applicable: list[tuple[RootExecutionAdmissionPolicyRule, PolicyAction]] = []
        for rule in self._root_execution_admission_rules:
            if (
                rule.execution_operation is not None
                and rule.execution_operation != execution_operation
            ):
                continue
            applicable.append(
                (rule, _normalize_root_execution_admission_rule_action(rule)),
            )

        if not applicable:
            return (
                PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="root_execution_admission_indeterminate",
                    enforcement_level=EnforcementLevel.MANDATORY,
                    policy_rule_id="default.root_execution_admission.indeterminate",
                    audit_payload={"execution_operation": execution_operation},
                ),
                None,
            )

        controlling_rule, controlling_action = _root_execution_admission_controlling_rule(
            applicable,
        )
        if (
            controlling_action is PolicyAction.DENY
            and controlling_rule.decision is PolicyAction.MODIFY
        ):
            reason = "root_execution_admission_unsupported_decision"
        else:
            reason = (
                controlling_rule.reason
                or f"root_execution_admission:{controlling_action.value}"
            )

        approved_scopes: tuple[str, ...] | None = None
        if controlling_action is PolicyAction.ALLOW:
            approved_scopes = controlling_rule.approved_scopes

        return (
            PolicyDecision(
                action=controlling_action,
                reason=reason,
                enforcement_level=EnforcementLevel.MANDATORY,
                policy_rule_id=controlling_rule.rule_id,
                audit_payload={
                    "tenant_id": tenant_id,
                    "workspace_id": workspace_id,
                    "principal_id": principal_id,
                    "execution_operation": execution_operation,
                    "resource_scope": resource_scope,
                    "collaborative_authority_scopes": list(collaborative_authority_scopes),
                },
            ),
            approved_scopes,
        )

    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        """Authorize a proposed external side effect — fail closed by default.

        Unlike ``evaluate_decision`` (default allow), missing identity or no
        matching rule yields DENY. Does not execute the side effect.
        """
        if not (request.task_id or "").strip() or not (request.run_id or "").strip():
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="meaningful_side_effect_identity_missing",
                enforcement_level=EnforcementLevel.MANDATORY,
                policy_rule_id="default.meaningful_side_effect.identity",
                audit_payload={
                    "action": request.action,
                    "task_id": request.task_id,
                    "run_id": request.run_id,
                },
            )
        if not (request.principal_id or "").strip():
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="meaningful_side_effect_principal_missing",
                enforcement_level=EnforcementLevel.MANDATORY,
                policy_rule_id="default.meaningful_side_effect.principal",
                audit_payload={"action": request.action},
            )

        applicable: list[tuple[MeaningfulSideEffectPolicyRule, PolicyAction]] = []
        for rule in self._meaningful_side_effect_rules:
            if rule.action is not None and rule.action != request.action:
                continue
            applicable.append(
                (rule, _normalize_meaningful_side_effect_rule_action(rule)),
            )

        if not applicable:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="meaningful_side_effect_indeterminate",
                enforcement_level=EnforcementLevel.MANDATORY,
                policy_rule_id="default.meaningful_side_effect.indeterminate",
                audit_payload={
                    "action": request.action,
                    "kinds": [k.value for k in request.kinds],
                },
            )

        controlling_rule, controlling_action = _meaningful_side_effect_controlling_rule(
            applicable,
        )
        matched_rule_ids = [rule.rule_id for rule, _ in applicable]
        if (
            controlling_action is PolicyAction.DENY
            and controlling_rule.decision is PolicyAction.MODIFY
        ):
            reason = "meaningful_side_effect_unsupported_decision"
        else:
            reason = controlling_rule.reason or f"meaningful_side_effect:{controlling_action.value}"

        return PolicyDecision(
            action=controlling_action,
            reason=reason,
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id=controlling_rule.rule_id,
            audit_payload={
                "action": request.action,
                "kinds": [k.value for k in request.kinds],
                "external_target": request.external_target,
                "matched_rule_ids": matched_rule_ids,
                "resolution_reason": "conservative_precedence",
            },
        )

    def evaluate_decision(
        self,
        decision: AgentDecision,
        *,
        context: AgentDecisionPolicyContext | None = None,
    ) -> PolicyDecision:
        ctx = context or AgentDecisionPolicyContext()
        if decision.type == AgentDecisionType.INTERRUPT and decision.severity.value == "critical":
            if ctx.require_human_on_critical:
                return PolicyDecision(
                    action=PolicyAction.REQUIRE_HUMAN,
                    reason="critical_interrupt_requires_human",
                    enforcement_level=EnforcementLevel.MANDATORY,
                    policy_rule_id="default.critical_interrupt",
                )
        if decision.type == AgentDecisionType.COMPLETE and ctx.has_unresolved_critical_interrupt:
            return PolicyDecision(
                action=PolicyAction.REQUIRE_HUMAN,
                reason="unresolved_critical_interrupt",
                enforcement_level=EnforcementLevel.MANDATORY,
                policy_rule_id="default.block_complete_on_critical",
            )
        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="default_allow",
            policy_rule_id="default.allow",
        )

    def evaluate_pre_llm(
        self,
        *,
        tenant_id: str,
        agent_id: str,
        message_count: int,
        context: PreModelPolicyContext | None = None,
    ) -> PolicyDecision:
        ctx = context or PreModelPolicyContext()
        if message_count < 1:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="pre_llm_empty_context",
                policy_rule_id="default.pre_llm_context",
            )
        if ctx.phase is PreModelPhase.NEXUS_PLANNING:
            planner_model_id = ctx.planner_model_id.strip()
            denied = {item.strip() for item in ctx.denied_planner_model_ids if item.strip()}
            if planner_model_id and planner_model_id in denied:
                return PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="planner_model_denied",
                    policy_rule_id="reasoning.planner_model_denied",
                    audit_payload={
                        "tenant_id": tenant_id,
                        "planner_model_id": planner_model_id,
                    },
                )
        if ctx.phase is PreModelPhase.AGENT_STEP:
            model_id = ctx.model_id.strip()
            denied = {item.strip() for item in ctx.denied_model_ids if item.strip()}
            if model_id and model_id in denied:
                return PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="agent_model_denied",
                    policy_rule_id="runtime.agent_model_denied",
                    audit_payload={
                        "tenant_id": tenant_id,
                        "agent_id": agent_id,
                        "model_id": model_id,
                    },
                )
        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="pre_llm_default_allow",
            policy_rule_id="default.pre_llm_allow",
            audit_payload={"tenant_id": tenant_id, "agent_id": agent_id},
        )

    def evaluate_pre_output(
        self,
        *,
        tenant_id: str,
        agent_id: str,
        output_chars: int,
    ) -> PolicyDecision:
        if output_chars <= 0:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="pre_output_empty",
                policy_rule_id="default.pre_output_empty",
            )
        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="pre_output_default_allow",
            policy_rule_id="default.pre_output_allow",
            audit_payload={"tenant_id": tenant_id, "agent_id": agent_id},
        )

    def evaluate_interrupt(self, interrupt: ExecutionInterrupt) -> PolicyDecision:
        if interrupt.blocking:
            return PolicyDecision(
                action=PolicyAction.REQUIRE_HUMAN,
                reason=f"blocking_interrupt:{interrupt.interrupt_type.value}",
                enforcement_level=EnforcementLevel.MANDATORY,
                policy_rule_id="default.blocking_interrupt",
                audit_payload={"interrupt_id": interrupt.interrupt_id},
            )
        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="non_blocking_interrupt",
            policy_rule_id="default.non_blocking_interrupt",
        )
