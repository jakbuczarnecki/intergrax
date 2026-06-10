# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Tool planning contract for ToolsStep (Phase Q+-L.2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Protocol, Union, runtime_checkable

from intergrax.llm.messages import ChatMessage

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
