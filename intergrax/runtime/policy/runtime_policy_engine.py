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
        _ = context
        if message_count < 1:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="pre_llm_empty_context",
                policy_rule_id="default.pre_llm_context",
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
