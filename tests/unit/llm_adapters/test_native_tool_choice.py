# © Artur Czarnecki. All rights reserved.

"""Native tool-choice projection at provider boundary."""

from __future__ import annotations

from intergrax.llm_adapters._shared.openai_tool_choice_projection import (
    project_openai_compatible_tool_choice,
)
from intergrax.llm_adapters.contracts.native_tool_choice import NativeForcedFunctionChoice
from intergrax.llm_adapters.providers._ollama_tool_choice_projection import (
    project_ollama_native_tool_choice,
)
from intergrax.runtime.nexus.tools.atomic_planner_round import PLANNER_ROUND_TOOL_ID


def test_forced_function_projects_to_ollama_required() -> None:
    projected = project_ollama_native_tool_choice(
        NativeForcedFunctionChoice(function_name=PLANNER_ROUND_TOOL_ID),
    )
    assert projected == "required"


def test_forced_function_projects_to_openai_function_choice() -> None:
    projected = project_openai_compatible_tool_choice(
        NativeForcedFunctionChoice(function_name=PLANNER_ROUND_TOOL_ID),
    )
    assert projected == {"type": "function", "name": PLANNER_ROUND_TOOL_ID}


def test_literal_auto_passthrough() -> None:
    assert project_openai_compatible_tool_choice("auto") == "auto"
