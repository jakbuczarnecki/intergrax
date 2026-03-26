# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from pydantic import BaseModel

from intergrax.agents_packages.legal_agent.prompts.legal_agent_llm_prompts import (
    DECISION_SYSTEM,
    decision_user,
)
from intergrax.agents_packages.legal_agent.steps.base.legal_base_step import LegalBaseStep
from intergrax.agents_packages.legal_agent.domain.legal_agent_state import (
    LegalAgentState,
    LegalCheck,
    LegalDecision,
    SensitiveFlag,
)
from intergrax.agents_packages.legal_agent.tracing.legal_decision_step_diag_v1 import LegalDecisionStepDiagV1
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LegalDecisionResult(BaseModel):
    decision: LegalDecision


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class LegalDecisionStep(LegalBaseStep):

    async def run_step(
        self,
        state: RuntimeState,
        agent_state: LegalAgentState,
    ) -> None:

        if not agent_state.legal_checks:
            state.trace_event(
                component=TraceComponent.STEP,
                step="LegalDecisionStep",
                message="Skipped: no legal_checks available.",
                level=TraceLevel.INFO,
            )
            return

        decision_before = (
            agent_state.decision.status if agent_state.decision is not None else None
        )

        llm = state.context.config.llm_adapter

        checks_block = self._format_checks(agent_state.legal_checks)
        flags_block = self._format_flags(agent_state.sensitive_flags)

        messages = [
            ChatMessage(role="system", content=DECISION_SYSTEM),
            ChatMessage(
                role="user",
                content=decision_user(
                    checks_block=checks_block,
                    flags_block=flags_block,
                ),
            ),
        ]

        result = llm.generate_structured(
            messages,
            LegalDecisionResult,
            run_id=state.run_id,
        )

        if not isinstance(result, LegalDecisionResult):
            raise TypeError("Invalid LLM response type in LegalDecisionStep.")

        agent_state.decision = result.decision
        agent_state.decision_pre_enforcement_status = None
        agent_state.decision_enforcement_modified = False

        violations = agent_state.policy_violations or []
        high_risk_count = sum(1 for c in agent_state.legal_checks if not c.valid)
        high_severity_violations = sum(1 for v in violations if v.severity == "HIGH")

        state.trace_event(
            component=TraceComponent.STEP,
            step="LegalDecisionStep",
            message=(
                f"Decision computed: {result.decision.status} "
                f"(confidence={result.decision.confidence})"
            ),
            level=TraceLevel.INFO,
            payload=LegalDecisionStepDiagV1(
                step_name="LegalDecisionStep",
                decision_status=result.decision.status,
                decision_confidence=result.decision.confidence,
                legal_checks_count=len(agent_state.legal_checks),
                high_risk_count=high_risk_count,
                policy_violations_count=len(violations),
                high_severity_violations=high_severity_violations,
                recommendations_count=len(agent_state.recommendations),
                decision_before=decision_before,
                decision_after=result.decision.status,
                enforcement_triggered=False,
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_checks(self, checks: List[LegalCheck]) -> str:
        if not checks:
            return "(none)"
        lines: List[str] = []
        for c in checks:
            src = c.source if c.source is not None else "n/a"
            det = c.details if c.details is not None else "n/a"
            lines.append(
                f"id={c.clause_id}\nvalid={c.valid}\nsource={src}\n"
                f"details={det}\n---"
            )
        return "\n".join(lines)

    def _format_flags(self, flags: List[SensitiveFlag]) -> str:
        if not flags:
            return "(none)"
        lines: List[str] = []
        for f in flags:
            lines.append(f"id={f.clause_id}\nreason={f.reason}\n---")
        return "\n".join(lines)