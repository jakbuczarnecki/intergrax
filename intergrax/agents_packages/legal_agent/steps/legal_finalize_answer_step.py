# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from typing import List

from pydantic import BaseModel

from intergrax.agents_packages.legal_agent.steps.legal_base_step import LegalBaseStep
from intergrax.agents_packages.legal_agent.legal_agent_state import Clause, LegalAgentState
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RouteInfo, RuntimeAnswer, RuntimeStats
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class FinalAnswerModel(BaseModel):
    answer: str


class LegalFinalizeAnswerStep(LegalBaseStep):

    async def run_step(
        self,
        state: RuntimeState,
        agent_state: LegalAgentState,
    ) -> None:

        clauses = agent_state.clauses

        llm = state.context.config.llm_adapter

        if not clauses:
            clauses_text = "No clauses detected."
        else:
            lines: List[str] = []
            for idx, clause in enumerate(clauses, start=1):
                lines.append(f"{idx}. {self._clause_repr(clause)}")
            clauses_text = "\n".join(lines)

        user_msg = (state.request.message or "").strip()
        human = (
            f"User request:\n{user_msg or '[none]'}\n\n"
            f"Extracted clauses (workspace):\n{clauses_text}\n"
        )

        system = (
            "You are a legal analysis assistant. Based on the extracted clauses below, "
            "produce a final user-facing summary. Focus on clarity, key risks, and important "
            "clauses. Return structured JSON only matching the provided schema."
        )

        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=human),
        ]

        result = llm.generate_structured(
                messages,
                FinalAnswerModel,
                run_id=state.run_id,
            )

        if not isinstance(result, FinalAnswerModel):
            raise TypeError("Invalid LLM response type in LegalFinalizeAnswerStep.")

        answer_text = result.answer.strip() or "[ERROR] Empty legal finalize answer."
        state.raw_answer = answer_text

        used_attachments = bool(state.request.attachments)
        if used_attachments:
            state.used_attachments_context = True

        route = RouteInfo(
            used_rag=state.used_rag,
            used_websearch=False,
            used_tools=False,
            used_user_profile=state.used_user_profile,
            used_user_longterm_memory=state.used_user_longterm_memory,
            strategy="legal_agent",
            extra={"clauses_count": len(clauses)},
        )

        state.runtime_answer = RuntimeAnswer(
            answer=answer_text,
            citations=[],
            route=route,
            tool_calls=[],
            stats=RuntimeStats(),
            raw_model_output=None,
        )

        state.trace_event(
            component=TraceComponent.STEP,
            step="LegalFinalizeAnswerStep",
            message="Final answer generated from clauses.",
            level=TraceLevel.INFO,
        )

    def _clause_repr(self, clause: Clause) -> str:
        return json.dumps(clause.model_dump(mode="json"), ensure_ascii=False)
