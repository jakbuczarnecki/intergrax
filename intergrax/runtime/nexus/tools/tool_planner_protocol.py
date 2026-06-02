# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Tool planning contract for ToolsStep (Phase Q+-L.2)."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from intergrax.tools.core.tool_plan_decision import ToolPlanDecision


@runtime_checkable
class ToolPlannerProtocol(Protocol):
    """Plans catalog tool calls; execution stays in :class:`RuntimeToolInvoker`."""

    def plan_tools(
        self,
        input_data: str,
        context: Optional[Any] = None,
        *,
        run_id: str,
    ) -> ToolPlanDecision:
        ...
