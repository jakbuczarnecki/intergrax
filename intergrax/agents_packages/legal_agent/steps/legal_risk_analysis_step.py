# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from intergrax.agents_packages.legal_agent.prompts.legal_agent_llm_prompts import (
    RISK_ANALYSIS_SYSTEM,
    risk_analysis_user,
)
from intergrax.agents_packages.legal_agent.steps.base.legal_base_step import LegalBaseStep
from intergrax.agents_packages.legal_agent.domain.legal_agent_state import (
    Clause,
    LegalAgentState,
    LegalCheck,
    SensitiveFlag,
)
from intergrax.agents_packages.legal_agent.tracing.legal_risk_analysis_step_diag_v1 import (
    LegalRiskAnalysisStepDiagV1,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


# ---------------------------------------------------------------------------
# Models (structured LLM output → maps directly into LegalAgentState)
# ---------------------------------------------------------------------------


class LegalRiskAnalysisResult(BaseModel):
    """
    One-shot risk review over clauses already stored on ``LegalAgentState``.

    Field semantics align with :class:`LegalCheck` and :class:`SensitiveFlag`.
    """

    legal_checks: List[LegalCheck] = Field(default_factory=list)
    sensitive_flags: List[SensitiveFlag] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class LegalRiskAnalysisStep(LegalBaseStep):

    async def run_step(
        self,
        state: RuntimeState,
        agent_state: LegalAgentState,
    ) -> None:

        if not agent_state.clauses:
            state.trace_event(
                component=TraceComponent.STEP,
                step="LegalRiskAnalysisStep",
                message="Skipped: no clauses on agent state.",
                level=TraceLevel.INFO,
            )
            return

        llm = state.context.config.llm_adapter

        clauses_block = self._format_clauses_for_prompt(agent_state.clauses)

        messages = [
            ChatMessage(role="system", content=RISK_ANALYSIS_SYSTEM),
            ChatMessage(
                role="user",
                content=risk_analysis_user(clauses_block=clauses_block),
            ),
        ]

        result = llm.generate_structured(
            messages,
            LegalRiskAnalysisResult,
            run_id=state.run_id,
        )

        if not isinstance(result, LegalRiskAnalysisResult):
            raise TypeError("Invalid LLM response type in LegalRiskAnalysisStep.")

        agent_state.legal_checks.extend(result.legal_checks)
        agent_state.sensitive_flags.extend(result.sensitive_flags)

        high_risk_added = sum(1 for c in result.legal_checks if not c.valid)
        state.trace_event(
            component=TraceComponent.STEP,
            step="LegalRiskAnalysisStep",
            message=(
                f"Risk analysis done: {len(result.legal_checks)} legal_checks, "
                f"{len(result.sensitive_flags)} sensitive_flags."
            ),
            level=TraceLevel.INFO,
            payload=LegalRiskAnalysisStepDiagV1(
                step_name="LegalRiskAnalysisStep",
                clauses_input_count=len(agent_state.clauses),
                legal_checks_added_count=len(result.legal_checks),
                sensitive_flags_added_count=len(result.sensitive_flags),
                high_risk_checks_added_count=high_risk_added,
            ),
        )

    def _format_clauses_for_prompt(self, clauses: List[Clause]) -> str:
        lines: List[str] = []
        for c in clauses:
            cat = c.category if c.category is not None else "n/a"
            lines.append(
                f"id={c.id}\ncategory={cat}\ntext={c.text}\n"
                f"pre_flagged_sensitive={c.is_sensitive}\n---"
            )
        return "\n".join(lines)
