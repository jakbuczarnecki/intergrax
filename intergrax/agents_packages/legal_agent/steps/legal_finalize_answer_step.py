# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from typing import List, Sequence

from pydantic import BaseModel

from intergrax.agents_packages.legal_agent.steps.base.legal_base_step import LegalBaseStep
from intergrax.agents_packages.legal_agent.legal_agent_state import (
    Clause,
    LegalAgentState,
    LegalDecision,
    LegalOpinion,
)
from intergrax.agents_packages.legal_agent.tracing.legal_finalize_answer_step_diag_v1 import (
    LegalFinalizeAnswerStepDiagV1,
)
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

        llm = state.context.config.llm_adapter

        workspace = self._build_workspace_text(agent_state)

        user_msg = (state.request.message or "").strip()
        human = (
            f"User request:\n{user_msg or '[none]'}\n\n"
            f"Legal agent workspace (all prior steps; JSON per section):\n{workspace}\n"
        )

        system = (
            "You are a legal analysis assistant. Synthesize a single clear, accurate "
            "user-facing answer using the full workspace: clauses, risk/legal checks, "
            "sensitive flags, compliance results, uncertainties, policy violations, "
            "structured recommendations, the formal decision (if present), and the "
            "Decision enforcement section (LegalDecisionEnforcementStep: whether the "
            "LLM decision was tightened using policy / legal-check rules). "
            "Treat the post-enforcement decision.status and blocking_issues as authoritative "
            "for approval posture; call out enforcement-driven changes explicitly in prose "
            "when decision_enforcement_modified is true. "
            "Do not invent facts; if a section is empty, say what was not analyzed. "
            "Return structured JSON only matching the provided schema."
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

        stripped = result.answer.strip()
        empty_fallback = not stripped
        answer_text = stripped or "[ERROR] Empty legal finalize answer."
        state.raw_answer = answer_text

        used_attachments = bool(state.request.attachments)
        if used_attachments:
            state.used_attachments_context = True

        violations = agent_state.policy_violations or []
        route = RouteInfo(
            used_rag=state.used_rag,
            used_websearch=False,
            used_tools=False,
            used_user_profile=state.used_user_profile,
            used_user_longterm_memory=state.used_user_longterm_memory,
            strategy="legal_agent",
            extra={
                "clauses_count": len(agent_state.clauses),
                "legal_checks_count": len(agent_state.legal_checks),
                "sensitive_flags_count": len(agent_state.sensitive_flags),
                "compliance_results_count": len(agent_state.compliance_results),
                "uncertainties_count": len(agent_state.uncertainties),
                "policy_violations_count": len(violations),
                "recommendations_count": len(agent_state.recommendations),
                "decision_status": (
                    agent_state.decision.status if agent_state.decision else None
                ),
            },
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
            message="Final answer generated from full legal workspace.",
            level=TraceLevel.INFO,
            payload=LegalFinalizeAnswerStepDiagV1(
                step_name="LegalFinalizeAnswerStep",
                outcome="empty_fallback" if empty_fallback else "ok",
                answer_length_chars=len(answer_text),
                used_rag=state.used_rag,
                used_attachments_context=used_attachments,
                clauses_count=len(agent_state.clauses),
                legal_checks_count=len(agent_state.legal_checks),
                sensitive_flags_count=len(agent_state.sensitive_flags),
                compliance_results_count=len(agent_state.compliance_results),
                uncertainties_count=len(agent_state.uncertainties),
                policy_violations_count=len(violations),
                recommendations_count=len(agent_state.recommendations),
                decision_status=(
                    agent_state.decision.status if agent_state.decision else None
                ),
                decision_enforcement_modified=agent_state.decision_enforcement_modified,
            ),
        )

    def _build_workspace_text(self, agent_state: LegalAgentState) -> str:
        sections: List[str] = [
            self._section_clauses(agent_state.clauses),
            self._section_models("Legal checks", agent_state.legal_checks),
            self._section_models("Sensitive flags", agent_state.sensitive_flags),
            self._section_models("Compliance results", agent_state.compliance_results),
            self._section_models("Uncertainties", agent_state.uncertainties),
            self._section_models(
                "Policy violations",
                agent_state.policy_violations or [],
            ),
            self._section_models(
                "Structured recommendations",
                agent_state.recommendations,
            ),
            self._section_legal_decision(agent_state.decision),
            self._section_decision_enforcement(agent_state),
            self._section_legal_opinion(agent_state.final_opinion),
        ]
        return "\n\n".join(sections)

    def _section_clauses(self, clauses: List[Clause]) -> str:
        header = "## Clauses"
        if not clauses:
            return f"{header}\n(none)\n"
        lines: List[str] = [header]
        for idx, clause in enumerate(clauses, start=1):
            lines.append(f"{idx}. {self._clause_repr(clause)}")
        return "\n".join(lines)

    def _section_models(self, title: str, items: Sequence[BaseModel]) -> str:
        header = f"## {title}"
        if not items:
            return f"{header}\n(none)\n"
        payload = [m.model_dump(mode="json") for m in items]
        return f"{header}\n{json.dumps(payload, ensure_ascii=False, indent=2)}"

    def _section_legal_decision(self, decision: LegalDecision | None) -> str:
        header = "## Decision (final, after LegalDecisionEnforcementStep)"
        if decision is None:
            return f"{header}\n(none)\n"
        return f"{header}\n{json.dumps(decision.model_dump(mode='json'), ensure_ascii=False, indent=2)}"

    def _section_decision_enforcement(self, agent_state: LegalAgentState) -> str:
        header = "## Decision enforcement (LegalDecisionEnforcementStep)"
        if agent_state.decision is None:
            return f"{header}\n(none) — no decision to enforce\n"

        lines: List[str] = [
            header,
            f"status_before_enforcement: {agent_state.decision_pre_enforcement_status!s}",
            f"status_after_enforcement: {agent_state.decision.status}",
            f"enforcement_modified_status: {agent_state.decision_enforcement_modified}",
        ]
        enf = [
            b
            for b in agent_state.decision.blocking_issues
            if str(b).startswith("Enforcement:")
        ]
        if enf:
            lines.append("enforcement_blocking_issues:")
            lines.extend(f"  - {x}" for x in enf)
        else:
            lines.append("enforcement_blocking_issues: (none)")

        lines.append(
            "Note: The JSON in the Decision section above is the authoritative "
            "post-enforcement outcome."
        )
        return "\n".join(lines) + "\n"

    def _section_legal_opinion(self, opinion: LegalOpinion | None) -> str:
        header = "## Legal opinion (prior step, if any)"
        if opinion is None:
            return f"{header}\n(none)\n"
        return f"{header}\n{json.dumps(opinion.model_dump(mode='json'), ensure_ascii=False, indent=2)}"

    def _clause_repr(self, clause: Clause) -> str:
        return json.dumps(clause.model_dump(mode="json"), ensure_ascii=False)
