# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Legal-tier tool / retrieval intent plan.

Tier-1 :class:`~intergrax.tools.core.tool_plan.ToolCallPlan` is produced inside
:class:`~intergrax.runtime.nexus.runtime_steps.tools_step.ToolsStep` when tools run.
This Pydantic model only drives *whether* to run RAG / websearch / tools and carries
intent + confidence for routing feedback.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LegalToolIntent = Literal[
    "llm_only",
    "rag",
    "tools",
    "websearch",
    "combination",
]


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
    use_rag: bool = Field(description="Run Nexus RagStep when infrastructure allows.")
    use_tools: bool = Field(description="Run Nexus ToolsStep when tools are configured.")
    use_websearch: bool = Field(
        description="Run Nexus WebsearchStep when websearch is configured.",
    )
    reasoning_summary: str = Field(
        default="",
        description="Short rationale for logs and routing context.",
    )

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
