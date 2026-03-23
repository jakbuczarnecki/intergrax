# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from intergrax.agents_packages.legal_agent.steps.base.legal_base_step import LegalBaseStep
from intergrax.agents_packages.legal_agent.legal_agent_state import (
    Clause,
    LegalAgentState,
    PolicyViolation,
)
from intergrax.agents_packages.legal_agent.tracing.legal_policy_compliance_step_diag_v1 import (
    LegalPolicyComplianceStepDiagV1,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LegalPolicyComplianceResult(BaseModel):
    violations: List[PolicyViolation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class LegalPolicyComplianceStep(LegalBaseStep):

    async def run_step(
        self,
        state: RuntimeState,
        agent_state: LegalAgentState,
    ) -> None:

        if not agent_state.clauses:
            state.trace_event(
                component=TraceComponent.STEP,
                step="LegalPolicyComplianceStep",
                message="Skipped: no clauses on agent state.",
                level=TraceLevel.INFO,
            )
            return

        llm = state.context.config.llm_adapter

        clauses_block = self._format_clauses(agent_state.clauses)

        policy_text = agent_state.config.organization_compliance_policy.strip()
        if not policy_text:
            state.trace_event(
                component=TraceComponent.STEP,
                step="LegalPolicyComplianceStep",
                message="Skipped: organization_compliance_policy is empty.",
                level=TraceLevel.INFO,
            )
            return

        system = (
            "You are a legal compliance system.\n"
            "Check whether the contract clauses violate the organization policy.\n\n"
            "Return structured JSON only.\n\n"
            "For each violation:\n"
            "- clause_id must match input\n"
            "- policy_rule: short rule name\n"
            "- violation: what is wrong\n"
            "- suggested_fix: how to correct it\n"
            "- severity: LOW / MEDIUM / HIGH\n"
        )

        user = f"{policy_text}\n\nClauses:\n{clauses_block}\n"

        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]

        result = llm.generate_structured(
            messages,
            LegalPolicyComplianceResult,
            run_id=state.run_id,
        )

        if not isinstance(result, LegalPolicyComplianceResult):
            raise TypeError("Invalid LLM response type in LegalPolicyComplianceStep.")

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        agent_state.policy_violations = list(result.violations)

        high_sev = sum(1 for v in result.violations if v.severity == "HIGH")
        state.trace_event(
            component=TraceComponent.STEP,
            step="LegalPolicyComplianceStep",
            message=f"Detected {len(result.violations)} policy violations.",
            level=TraceLevel.INFO,
            payload=LegalPolicyComplianceStepDiagV1(
                step_name="LegalPolicyComplianceStep",
                outcome="completed",
                clauses_count=len(agent_state.clauses),
                violations_count=len(result.violations),
                high_severity_violations=high_sev,
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_clauses(self, clauses: List[Clause]) -> str:
        lines: List[str] = []
        for c in clauses:
            cat = c.category if c.category else "n/a"
            lines.append(
                f"id={c.id}\ncategory={cat}\ntext={c.text}\n---"
            )
        return "\n".join(lines)