# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Legal-tier tool / retrieval intent plan.

Tier-1 :class:`~intergrax.tools.core.tool_plan.ToolCallPlan` is produced inside
:class:`~intergrax.runtime.nexus.runtime_steps.tools_step.ToolsStep` when tools run.
This Pydantic model drives *whether* to run RAG / websearch / tools and carries
intent + confidence for routing feedback.

Canonical (Phase O.5): ``tool_ids`` e.g. ``["rag.retrieve", "websearch.query"]``.
Legacy booleans remain for LLM structured output compatibility.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID

LegalToolIntent = Literal[
    "llm_only",
    "rag",
    "tools",
    "websearch",
    "combination",
]


def compute_legal_tool_intent_from_layers(
    *,
    use_rag: bool,
    use_tools: bool,
    use_websearch: bool,
) -> LegalToolIntent:
    """
    Derive :class:`LegalToolIntent` from enabled Nexus layer flags (single source of truth).
    Used by organization governance, dynamic policy caps, and similar clamps.
    """
    n = int(use_rag) + int(use_tools) + int(use_websearch)
    if n == 0:
        return "llm_only"
    if n == 1:
        if use_rag:
            return "rag"
        if use_tools:
            return "tools"
        return "websearch"
    return "combination"


class LegalToolPlan(BaseModel):
    """
    Structured output from :mod:`tool_decision_component`.

    Tier-1 :class:`~intergrax.tools.core.tool_plan.ToolCallPlan` is still produced inside
    :class:`~intergrax.runtime.nexus.runtime_steps.tools_step.ToolsStep` when
    ``use_tools`` is true (no duplicate executor here).
    """

    intent: LegalToolIntent = Field(
        description="Primary strategy for this turn before legal stage routing.",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Planner confidence.")
    tool_ids: list[str] = Field(
        default_factory=list,
        description="Canonical catalog tool ids (rag.retrieve, websearch.query, …).",
    )
    use_rag: bool = Field(description="Run Nexus RagStep when infrastructure allows.")
    use_tools: bool = Field(description="Run Nexus ToolsStep when tools are configured.")
    use_websearch: bool = Field(
        description="Run Nexus WebsearchStep when websearch is configured.",
    )
    reasoning_summary: str = Field(
        default="",
        description="Short rationale for logs and routing context.",
    )

    @model_validator(mode="after")
    def _sync_layers_and_tool_ids(self) -> Self:
        ids = list(self.tool_ids)
        use_rag = self.use_rag
        use_websearch = self.use_websearch

        if use_rag and RAG_RETRIEVE_TOOL_ID not in ids:
            ids.append(RAG_RETRIEVE_TOOL_ID)
        if use_websearch and WEBSEARCH_QUERY_TOOL_ID not in ids:
            ids.append(WEBSEARCH_QUERY_TOOL_ID)

        if RAG_RETRIEVE_TOOL_ID in ids:
            use_rag = True
        if WEBSEARCH_QUERY_TOOL_ID in ids:
            use_websearch = True

        self.tool_ids = list(dict.fromkeys(ids))
        self.use_rag = use_rag
        self.use_websearch = use_websearch
        self.intent = compute_legal_tool_intent_from_layers(
            use_rag=use_rag,
            use_tools=self.use_tools,
            use_websearch=use_websearch,
        )
        return self

    def resolved_tool_ids(self) -> list[str]:
        return list(self.tool_ids)

    @classmethod
    def default_llm_only(cls) -> LegalToolPlan:
        return cls(
            intent="llm_only",
            confidence=1.0,
            use_rag=False,
            use_tools=False,
            use_websearch=False,
            reasoning_summary="tool decision disabled or skipped",
        )

    @classmethod
    def from_tool_ids(cls, tool_ids: list[str], *, use_tools: bool = False) -> LegalToolPlan:
        return cls(
            intent="llm_only",
            confidence=1.0,
            tool_ids=tool_ids,
            use_rag=False,
            use_tools=use_tools,
            use_websearch=False,
        )
