# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Runtime governance policy engine (architecture §42.11).

Evaluates ``AgentDecision`` and ``ExecutionInterrupt`` against Tier-3 rules.
Use ``PolicyEngine`` facade for unified runtime + replay policy entry (§42.11).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.execution_interrupt import ExecutionInterrupt
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision


class RuntimePolicyEngine:
    """Rule-based policy evaluation for live task execution."""

    def __init__(self, rules: Optional[List[Dict[str, Any]]] = None) -> None:
        self._rules = rules or []

    def evaluate_decision(
        self,
        decision: AgentDecision,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        ctx = context or {}
        if decision.type == AgentDecisionType.INTERRUPT and decision.severity.value == "critical":
            if ctx.get("require_human_on_critical", True):
                return PolicyDecision(
                    action=PolicyAction.REQUIRE_HUMAN,
                    reason="critical_interrupt_requires_human",
                    enforcement_level=EnforcementLevel.MANDATORY,
                    policy_rule_id="default.critical_interrupt",
                )
        if decision.type == AgentDecisionType.COMPLETE and ctx.get("has_unresolved_critical_interrupt"):
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
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        ctx = context or {}
        if message_count < 1:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="pre_llm_empty_context",
                policy_rule_id="default.pre_llm_context",
            )
        if ctx.get("phase") == "nexus_planning":
            planner_model_id = str(ctx.get("planner_model_id", "")).strip()
            denied = {
                str(item).strip()
                for item in ctx.get("denied_planner_model_ids", ())
                if str(item).strip()
            }
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
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        _ = context
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

    def evaluate_critic_verdict(
        self,
        *,
        passed: bool,
        recommended_action: str,
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        ctx = context or {}
        governance = ctx.get("critic_governance") or {}
        if recommended_action == "escalate_hitl":
            return PolicyDecision(
                action=PolicyAction.REQUIRE_HUMAN,
                reason="critic_escalate_hitl",
                enforcement_level=EnforcementLevel.MANDATORY,
                policy_rule_id="critic.l2_escalation",
            )
        if governance.get("require_critic_on_completion") and not passed:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="critic_completion_required",
                enforcement_level=EnforcementLevel.MANDATORY,
                policy_rule_id="critic.require_on_completion",
            )
        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="critic_default_allow",
            policy_rule_id="critic.allow",
        )
