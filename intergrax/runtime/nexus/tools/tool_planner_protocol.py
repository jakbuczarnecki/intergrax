# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Tool planning contract for ToolsStep (Phase Q+-L.2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Protocol, Union, runtime_checkable

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.nexus.tools.tool_planning_policy import NativeToolChoice
from intergrax.tools.core.tool_plan import ToolCallPlan
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision


@runtime_checkable
class ToolPlannerProtocol(Protocol):
    """Plans catalog tool calls; execution stays in :class:`RuntimeToolInvoker`."""

    def plan_tools(
        self,
        input_data: Union[str, list[ChatMessage]],
        context: Optional[Any] = None,
        *,
        run_id: str,
        allowed_tool_ids: Sequence[str] | None = None,
    ) -> ToolPlanDecision:
        ...


@runtime_checkable
class IterativeToolPlannerProtocol(ToolPlannerProtocol, Protocol):
    """Planner capable of native iterative tool rounds (bounded ReAct)."""

    def plan_native_round(
        self,
        messages: list[ChatMessage],
        *,
        allowed_tool_ids: Sequence[str] | None = None,
        run_id: str,
        tool_choice: NativeToolChoice | None = None,
    ) -> tuple[LLMAdapterResponse, ToolCallPlan]:
        ...
