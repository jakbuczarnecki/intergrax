# © Artur Czarnecki. All rights reserved.

"""YAML contracts for catalog tool planning prompts (Tier-1 path)."""

from __future__ import annotations

import pytest

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.tools.tool_planning_prompts import (
    investigation_policy_prompt,
    planner_prompt,
    system_context_template,
    system_prompt,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _assert_non_empty_str(value: str) -> None:
    assert isinstance(value, str)
    assert value.strip()


def test_tools_agent_yaml_registry_contains_all_prompts() -> None:
    registry = YamlPromptRegistry.create_default(load=True)

    assert registry.resolve_localized("tools_agent_system")
    assert registry.resolve_localized("tools_agent_planner")
    assert registry.resolve_localized("tools_agent_context")
    assert registry.resolve_localized("tools_investigation_policy")


def test_tool_planning_system_prompt_exact_contract() -> None:
    text = system_prompt()
    assert text.rstrip() == (
        "You are a capable assistant. Use tools when helpful. "
        "If you call a tool, do not fabricate results—wait for tool outputs."
    ).rstrip()


def test_tool_planning_planner_prompt_exact_contract() -> None:
    text = planner_prompt()
    _assert_non_empty_str(text)
    assert "You do not have native tool-calling." in text
    assert '{"call_tool":' in text
    assert '{"final_answer":' in text
    assert "Never include commentary outside JSON." in text


def test_tool_planning_planner_prompt_json_shape_stability() -> None:
    text = planner_prompt()
    assert '"name": "<tool_name>"' in text
    assert '"arguments": {...}' in text
    assert '"<text>"' in text


def test_tool_planning_context_template_exact_contract() -> None:
    template = system_context_template()
    assert template.rstrip() == "Session context:\n{context}".rstrip()

    rendered = template.format(context="ABC")
    assert rendered.rstrip() == "Session context:\nABC".rstrip()


def test_tools_investigation_policy_yaml_contract() -> None:
    text = investigation_policy_prompt()
    lowered = text.lower()
    _assert_non_empty_str(text)
    assert "investigation and evidence policy" in lowered
    assert "observation" in lowered
    assert "observed facts" in lowered
    assert "inferred" in lowered
    assert "evidence gap" in lowered or "material evidence" in lowered
    assert "contradict" in lowered
    assert "correlation" in lowered
    assert "causation" in lowered
    assert "uncertainty" in lowered or "limitation" in lowered
    assert "stop when" in lowered or "unlikely to materially change" in lowered
    assert "chain-of-thought" not in lowered
    assert "private reasoning" not in lowered
