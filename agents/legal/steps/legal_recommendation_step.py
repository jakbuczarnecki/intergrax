# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, model_validator

from legal.prompts.legal_agent_llm_prompts import (
    RECOMMENDATION_SYSTEM,
    recommendation_user,
)
from legal.steps.base.legal_base_step import LegalBaseStep
from legal.domain.legal_agent_state import (
    Clause,
    LegalAgentState,
    LegalCheck,
    LegalRecommendation,
    PolicyViolation,
    SensitiveFlag,
)
from legal.tracing.legal_recommendation_step_diag_v1 import (
    LegalRecommendationStepDiagV1,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


# ---------------------------------------------------------------------------
# Models (LLM structured output)
# ---------------------------------------------------------------------------


class LegalRecommendationResult(BaseModel):
    recommendations: List[LegalRecommendation] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_root_shape(cls, data: object) -> object:
        """
        Ollama / small models sometimes return one recommendation object or a bare
        array instead of ``{"recommendations": [...]}``.
        """
        if isinstance(data, list):
            return {"recommendations": data}
        if isinstance(data, dict) and "recommendations" not in data:
            if "clause_id" in data:
                return {"recommendations": [data]}
        return data


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class LegalRecommendationStep(LegalBaseStep):

    async def run_step(
        self,
        state: RuntimeState,
        agent_state: LegalAgentState,
    ) -> None:

        violations = agent_state.policy_violations or []
        if (
            not agent_state.legal_checks
            and not agent_state.sensitive_flags
            and not violations
        ):
            state.trace_event(
                component=TraceComponent.STEP,
                step="LegalRecommendationStep",
                message=(
                    "Skipped: no legal_checks, sensitive_flags, or policy_violations."
                ),
                level=TraceLevel.INFO,
            )
            return

        llm = state.context.config.llm_adapter

        checks_block = self._format_checks(agent_state.legal_checks)
        flags_block = self._format_flags(agent_state.sensitive_flags)
        violations_block = self._format_violations(violations)
        clauses_block = self._format_clauses(agent_state.clauses)

        messages = [
            ChatMessage(role="system", content=RECOMMENDATION_SYSTEM),
            ChatMessage(
                role="user",
                content=recommendation_user(
                    clauses_block=clauses_block,
                    checks_block=checks_block,
                    flags_block=flags_block,
                    violations_block=violations_block,
                ),
            ),
        ]

        result = llm.generate_structured(
            messages,
            LegalRecommendationResult,
            run_id=state.run_id,
        )

        if not isinstance(result, LegalRecommendationResult):
            raise TypeError("Invalid LLM response type in LegalRecommendationStep.")

        agent_state.recommendations.extend(result.recommendations)

        high_pri = sum(1 for r in result.recommendations if r.priority == "HIGH")
        state.trace_event(
            component=TraceComponent.STEP,
            step="LegalRecommendationStep",
            message=f"Generated {len(result.recommendations)} recommendations.",
            level=TraceLevel.INFO,
            payload=LegalRecommendationStepDiagV1(
                step_name="LegalRecommendationStep",
                clauses_context_count=len(agent_state.clauses),
                legal_checks_count=len(agent_state.legal_checks),
                sensitive_flags_count=len(agent_state.sensitive_flags),
                policy_violations_count=len(violations),
                recommendations_added_count=len(result.recommendations),
                high_priority_recommendations_count=high_pri,
            ),
        )

    def _format_clauses(self, clauses: List[Clause]) -> str:
        if not clauses:
            return "(none)"
        lines: List[str] = []
        for c in clauses:
            cat = c.category if c.category is not None else "n/a"
            lines.append(f"id={c.id}\ncategory={cat}\ntext={c.text}\n---")
        return "\n".join(lines)

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

    def _format_violations(self, violations: List[PolicyViolation]) -> str:
        if not violations:
            return "(none)"
        lines: List[str] = []
        for v in violations:
            lines.append(
                f"id={v.clause_id}\nrule={v.policy_rule}\n"
                f"violation={v.violation}\nseverity={v.severity}\n"
                f"suggested_fix={v.suggested_fix}\n---"
            )
        return "\n".join(lines)
