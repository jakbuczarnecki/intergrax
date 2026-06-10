# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Planner input assembly for ToolsStep (TOOL-ENG-11)."""

from __future__ import annotations

from typing import List, Union

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config_types import ToolsContextScope
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


def resolve_tool_planner_input(
    state: RuntimeState,
) -> Union[str, List[ChatMessage]]:
    """Build planner input from ``tools_context_scope`` on runtime config."""
    scope = state.context.config.tools_context_scope
    message = (state.request.message or "").strip()

    if scope == ToolsContextScope.CONVERSATION:
        history = list(state.base_history or [])
        if message:
            history.append(ChatMessage(role="user", content=message))
        if history:
            return history
        return message

    if scope == ToolsContextScope.FULL:
        if state.messages_for_llm:
            return list(state.messages_for_llm)
        return message

    return message
