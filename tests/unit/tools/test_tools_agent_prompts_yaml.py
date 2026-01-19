# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.tools.tools_agent import (
    SYSTEM_PROMPT,
    PLANNER_PROMPT,
    SYSTEM_CONTEXT_TEMPLATE,
)
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _assert_non_empty_str(value: str) -> None:
    assert isinstance(value, str)
    assert value.strip()


# ----------------------------------------------------------------------
# Registry level contracts
# ----------------------------------------------------------------------

def test_tools_agent_yaml_registry_contains_all_prompts() -> None:
    registry = YamlPromptRegistry.create_default(load=True)

    assert registry.resolve_localized("tools_agent_system")
    assert registry.resolve_localized("tools_agent_planner")
    assert registry.resolve_localized("tools_agent_context")


# ----------------------------------------------------------------------
# SYSTEM_PROMPT
# ----------------------------------------------------------------------

def test_tools_agent_system_prompt_exact_contract() -> None:
    text = SYSTEM_PROMPT()

    # Exact legacy contract
    assert text.rstrip() == (
        "You are a capable assistant. Use tools when helpful. "
        "If you call a tool, do not fabricate results—wait for tool outputs."
    ).rstrip()


# ----------------------------------------------------------------------
# PLANNER_PROMPT
# ----------------------------------------------------------------------

def test_tools_agent_planner_prompt_exact_contract() -> None:
    text = PLANNER_PROMPT()
    _assert_non_empty_str(text)

    # Core structural requirements from legacy prompt
    assert "You do not have native tool-calling." in text
    assert '{"call_tool":' in text
    assert '{"final_answer":' in text
    assert "Never include commentary outside JSON." in text


def test_tools_agent_planner_prompt_json_shape_stability() -> None:
    """
    Ensure the prompt still contains the exact JSON shape examples
    required by the legacy parser.
    """
    text = PLANNER_PROMPT()

    assert '"name": "<tool_name>"' in text
    assert '"arguments": {...}' in text
    assert '"<text>"' in text


# ----------------------------------------------------------------------
# SYSTEM_CONTEXT_TEMPLATE
# ----------------------------------------------------------------------

def test_tools_agent_context_template_exact_contract() -> None:
    template = SYSTEM_CONTEXT_TEMPLATE()
    assert template.rstrip() == "Session context:\n{context}".rstrip()

    rendered = template.format(context="ABC")
    assert rendered.rstrip() == "Session context:\nABC".rstrip()
