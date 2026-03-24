# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.agents_packages.legal_agent.steps.base.legal_base_step import LegalBaseStep
from intergrax.agents_packages.legal_agent.legal_agent_state import (
    DecisionStatus,
    LegalAgentState,
)
from intergrax.agents_packages.legal_agent.tracing.legal_decision_enforcement_step_diag_v1 import (
    LegalDecisionEnforcementStepDiagV1,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.policies.runtime_policies import ExecutionKind
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class LegalDecisionEnforcementStep(LegalBaseStep):
    """
    Deterministic guardrails after :class:`LegalDecisionStep`.

    Overrides LLM decision when objective signals require it (policy breaches,
    failed legal checks). Mutates ``agent_state.decision`` in place.
    """

    def execution_kind(self) -> ExecutionKind | None:
        return None

    async def run_step(
        self,
        state: RuntimeState,
        agent_state: LegalAgentState,
    ) -> None:

        decision = agent_state.decision
        if decision is None:
            state.trace_event(
                component=TraceComponent.STEP,
                step="LegalDecisionEnforcementStep",
                message="Skipped: no decision on agent state (run LegalDecisionStep first).",
                level=TraceLevel.INFO,
            )
            return

        violations = agent_state.policy_violations or []
        checks = agent_state.legal_checks

        has_high_risk = any(not c.valid for c in checks)
        has_policy_violation = len(violations) > 0

        original_status: DecisionStatus = decision.status

        # Policy breach → at least CONDITIONAL (cannot stay APPROVE as-is).
        if has_policy_violation and decision.status not in ("REJECT", "CONDITIONAL"):
            decision.status = "CONDITIONAL"
            decision.blocking_issues = list(decision.blocking_issues) + [
                "Enforcement: organization policy violation(s) detected.",
            ]

        # Failed legal checks → REJECT (strongest signal).
        if has_high_risk:
            decision.status = "REJECT"
            decision.blocking_issues = list(decision.blocking_issues) + [
                "Enforcement: one or more legal checks marked invalid.",
            ]

        agent_state.decision_pre_enforcement_status = original_status
        agent_state.decision_enforcement_modified = decision.status != original_status

        status_after = decision.status
        diag = LegalDecisionEnforcementStepDiagV1(
            step_name="LegalDecisionEnforcementStep",
            decision_status_before=original_status,
            decision_status_after=status_after,
            enforcement_modified=agent_state.decision_enforcement_modified,
            has_high_risk_checks=has_high_risk,
            has_policy_violations=has_policy_violation,
            legal_checks_count=len(checks),
            policy_violations_count=len(violations),
            decision_confidence=decision.confidence,
        )

        if decision.status != original_status:
            state.trace_event(
                component=TraceComponent.STEP,
                step="LegalDecisionEnforcementStep",
                message=(
                    f"Decision overridden: {original_status} → {decision.status}"
                ),
                level=TraceLevel.WARNING,
                payload=diag,
            )
        else:
            state.trace_event(
                component=TraceComponent.STEP,
                step="LegalDecisionEnforcementStep",
                message="Decision unchanged after enforcement rules.",
                level=TraceLevel.INFO,
                payload=diag,
            )
