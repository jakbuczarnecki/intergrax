# © Artur Czarnecki. All rights reserved.

"""Tool invocation orchestration plugin contract (TOOL-ENG-16 · ADR-TOOL-003)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState, ToolCallTrace
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.tools.core.tool_plan import ToolCallPlan

ToolInvocationStopReason = Literal[
    "empty_tool_calls",
    "max_iterations",
    "budget_exceeded",
    "planner_final_answer",
    "legacy_single_pass",
]


@dataclass(slots=True)
class ToolInvocationResult:
    """Canonical batch orchestration result (Plane 3 — orchestration 2a)."""

    tool_traces: list[ToolCallTrace] = field(default_factory=list)
    loop_iterations: int = 0
    stop_reason: ToolInvocationStopReason = "legacy_single_pass"
    appended_messages: list[ChatMessage] = field(default_factory=list)
    used_native_tool_messages: bool = False


@runtime_checkable
class ToolInvocationPattern(Protocol):
    """Orchestrates multi-call tool plans before atomic invoke (2b unchanged)."""

    @property
    def pattern_id(self) -> str:
        """Stable identifier for trace and config (e.g. ``single_pass``)."""
        ...

    def execute(
        self,
        *,
        state: RuntimeState,
        invoker: RuntimeToolInvoker,
        planner: ToolPlannerProtocol,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[ChatMessage],
    ) -> ToolInvocationResult:
        ...


def pattern_for_mode(mode: ToolInvocationMode) -> ToolInvocationPattern:
    """Resolve shipped invocation pattern from runtime config."""
    from intergrax.runtime.nexus.tools.patterns.bounded_react import BoundedReactPattern
    from intergrax.runtime.nexus.tools.patterns.single_pass import SinglePassPattern

    if mode == ToolInvocationMode.BOUNDED_REACT:
        return BoundedReactPattern()
    if mode in (
        ToolInvocationMode.PARALLEL_BATCH,
        ToolInvocationMode.DETERMINISTIC_CHAIN,
        ToolInvocationMode.PARALLEL_SEMANTIC_BATCH,
    ):
        raise NotImplementedError(
            f"ToolInvocationMode.{mode.value} is registered but not yet shipped; "
            "see TOOL-ENG-9/20/25."
        )
    return SinglePassPattern()
