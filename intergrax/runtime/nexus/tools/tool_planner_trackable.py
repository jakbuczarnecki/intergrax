# © Artur Czarnecki. All rights reserved.

"""Protocol for planners that expose an LLM adapter for usage tracking (Phase U-Typ.3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


@runtime_checkable
class ToolPlannerTrackable(Protocol):
    """Tool planner that registers a secondary LLM adapter on the usage tracker."""

    @property
    def llm(self) -> LLMAdapter:
        ...
