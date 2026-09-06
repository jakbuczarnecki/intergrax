# © Artur Czarnecki. All rights reserved.

"""Native tool-choice projection at provider boundary."""

from __future__ import annotations

from intergrax.llm_adapters.contracts.native_tool_choice import (
    NativeForcedFunctionChoice,
    project_native_tool_choice_for_provider,
)
from intergrax.runtime.nexus.tools.atomic_planner_round import PLANNER_ROUND_TOOL_ID


def test_forced_function_projects_to_ollama_required() -> None:
    projected = project_native_tool_choice_for_provider(
        NativeForcedFunctionChoice(function_name=PLANNER_ROUND_TOOL_ID),
        provider="ollama",
    )
    assert projected == "required"


def test_forced_function_projects_to_openai_function_choice() -> None:
    projected = project_native_tool_choice_for_provider(
        NativeForcedFunctionChoice(function_name=PLANNER_ROUND_TOOL_ID),
        provider="openai",
    )
    assert projected == {"type": "function", "name": PLANNER_ROUND_TOOL_ID}


def test_literal_auto_passthrough() -> None:
    assert project_native_tool_choice_for_provider("auto", provider="openai") == "auto"
