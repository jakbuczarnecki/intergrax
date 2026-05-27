# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from legal.prompts.legal_agent_llm_prompts import (
    NORMALIZE_CLAUSES_SYSTEM,
    normalize_clauses_user,
)
from legal.steps.base.legal_base_step import LegalBaseStep
from legal.domain.legal_agent_state import (
    Clause,
    LegalAgentState,
)
from legal.tracing.legal_normalize_clauses_step_diag_v1 import (
    LegalNormalizeClausesStepDiagV1,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class NormalizedClause(BaseModel):
    original_ids: List[str] = Field(default_factory=list)
    normalized_text: str
    category: Optional[str] = None


class LegalNormalizeClausesResult(BaseModel):
    normalized_clauses: List[NormalizedClause] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class LegalNormalizeClausesStep(LegalBaseStep):

    async def run_step(
        self,
        state: RuntimeState,
        agent_state: LegalAgentState,
    ) -> None:

        clauses = agent_state.clauses

        if not clauses:
            state.trace_event(
                component=TraceComponent.STEP,
                step="LegalNormalizeClausesStep",
                message="Skipped: no clauses on agent state.",
                level=TraceLevel.INFO,
                payload=LegalNormalizeClausesStepDiagV1(
                    step_name="LegalNormalizeClausesStep",
                    outcome="skipped_no_clauses",
                    input_clauses_count=0,
                    output_clauses_count=0,
                    duplicates_removed_count=0,
                ),
            )
            return

        if len(clauses) == 1:
            state.trace_event(
                component=TraceComponent.STEP,
                step="LegalNormalizeClausesStep",
                message="Skipped: single clause, no normalization needed.",
                level=TraceLevel.INFO,
                payload=LegalNormalizeClausesStepDiagV1(
                    step_name="LegalNormalizeClausesStep",
                    outcome="skipped_single_clause",
                    input_clauses_count=1,
                    output_clauses_count=1,
                    duplicates_removed_count=0,
                ),
            )
            return

        llm = state.context.config.llm_adapter
        clauses_block = self._format_clauses(clauses)
        input_count = len(clauses)

        messages = [
            ChatMessage(role="system", content=NORMALIZE_CLAUSES_SYSTEM),
            ChatMessage(
                role="user",
                content=normalize_clauses_user(clauses_block=clauses_block),
            ),
        ]

        result = llm.generate_structured(
            messages,
            LegalNormalizeClausesResult,
            run_id=state.run_id,
        )

        if not isinstance(result, LegalNormalizeClausesResult):
            raise TypeError("Invalid LLM response type in LegalNormalizeClausesStep.")

        if not result.normalized_clauses:
            state.trace_event(
                component=TraceComponent.STEP,
                step="LegalNormalizeClausesStep",
                message="Fallback: normalization returned empty result.",
                level=TraceLevel.WARNING,
                payload=LegalNormalizeClausesStepDiagV1(
                    step_name="LegalNormalizeClausesStep",
                    outcome="fallback_empty_llm",
                    input_clauses_count=input_count,
                    output_clauses_count=input_count,
                    duplicates_removed_count=0,
                ),
            )
            return

        by_id: Dict[str, Clause] = {c.id: c for c in clauses}
        normalized: List[Clause] = []
        for nc in result.normalized_clauses:
            sensitive_any = any(
                by_id[oid].is_sensitive
                for oid in nc.original_ids
                if oid in by_id
            )
            normalized.append(
                Clause(
                    id=uuid.uuid4().hex,
                    text=nc.normalized_text,
                    category=nc.category,
                    is_sensitive=sensitive_any,
                )
            )

        new_count = len(normalized)
        removed = max(0, input_count - new_count)

        agent_state.clauses = normalized

        state.trace_event(
            component=TraceComponent.STEP,
            step="LegalNormalizeClausesStep",
            message=(
                f"Normalized clauses: {input_count} → {new_count} "
                f"(removed ~{removed} duplicates)."
            ),
            level=TraceLevel.INFO,
            payload=LegalNormalizeClausesStepDiagV1(
                step_name="LegalNormalizeClausesStep",
                outcome="normalized",
                input_clauses_count=input_count,
                output_clauses_count=new_count,
                duplicates_removed_count=removed,
            ),
        )

    def _format_clauses(self, clauses: List[Clause]) -> str:
        lines: List[str] = []
        for c in clauses:
            cat = c.category if c.category else "n/a"
            lines.append(
                f"id={c.id}\ncategory={cat}\ntext={c.text}\n---"
            )
        return "\n".join(lines)
