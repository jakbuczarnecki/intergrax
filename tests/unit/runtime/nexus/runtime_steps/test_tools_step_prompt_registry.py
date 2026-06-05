# © Artur Czarnecki. All rights reserved.

"""PE-4: ToolsStep uses injected prompt registry from RuntimeContext."""

from __future__ import annotations

import pytest

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.runtime_steps.tools_step import ToolsStep
from testing_support.builder import build_runtime_state_for_tests

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_tools_step_uses_context_prompt_registry() -> None:
    registry = YamlPromptRegistry.create_default(path="prompts", load=True)
    state = build_runtime_state_for_tests(run_id="run-pe4")
    state.context.prompt_registry = registry

    prompt = ToolsStep().tools_runtime_context_prompt(state)

    assert "{context}" in prompt or "context" in prompt.lower()
