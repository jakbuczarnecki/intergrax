# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Planner output model (Tier-1 tools step; distinct from §42.7 ``AgentDecision``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from intergrax.llm.messages import ChatMessage
from intergrax.tools.core.tool_plan import ToolCallPlan


@dataclass(slots=True)
class ToolPlanDecision:
    """Tools-agent planner output (not ``contracts.agent_decision.AgentDecision``)."""

    final_answer: Optional[str]
    tool_plan: Optional[ToolCallPlan]
    messages: List[ChatMessage]
