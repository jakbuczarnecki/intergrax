# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from enum import Enum
from typing import List, Literal

from pydantic import BaseModel

from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer
from intergrax.llm.messages import ChatMessage


# =========================
# Structured Output Models
# =========================

class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class ClauseItem(BaseModel):
    name: str
    importance: str
    description: str


class RiskItem(BaseModel):
    title: str
    severity: str
    explanation: str
    recommendation: str


class MissingClause(BaseModel):
    name: str
    reason: str


class Decision(BaseModel):
    sign_recommendation: Literal["approve", "review", "reject"]
    overall_risk: RiskLevel


class LegalAnalysisOutput(BaseModel):
    summary: str
    contract_type: str
    key_clauses: List[ClauseItem]
    risks: List[RiskItem]
    missing_clauses: List[MissingClause]
    decision: Decision


class LegalAnalysisPipeline(RuntimePipeline):
    """
    Legal analysis pipeline (v2 - structured output).

    - single LLM call
    - uses LLMAdapter.generate_structured(...)
    - returns validated Pydantic model
    """

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        cfg = state.context.config
        llm = cfg.llm_adapter

        if llm is None:
            raise RuntimeError("LegalAnalysisPipeline: llm_adapter is not configured.")

        contract_text = state.request.message or ""

        system_prompt = (
            "You are a senior legal expert specializing in business contracts. "
            "Analyze contracts for companies and provide structured legal insights."
        )

        user_prompt = (
            "Analyze the contract below.\n\n"
            "Focus on:\n"
            "- key clauses\n"
            "- legal risks\n"
            "- missing protections\n"
            "- recommendation whether to sign\n\n"
            "Contract:\n"
            f"{contract_text}"
        )

        messages: List[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        output = llm.generate_structured(
            messages=messages,
            output_model=LegalAnalysisOutput,
            run_id=state.run_id,
        )

        json = output.model_dump_json()

        # Store structured + raw representation
        state.raw_answer = json

        answer = RuntimeAnswer(
            run_id=state.run_id,
            answer=json
        )

        state.runtime_answer = answer

        return state.runtime_answer