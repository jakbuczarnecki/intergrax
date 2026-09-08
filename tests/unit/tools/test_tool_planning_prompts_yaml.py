# © Artur Czarnecki. All rights reserved.

"""YAML contracts for catalog tool planning prompts (Tier-1 path)."""

from __future__ import annotations

import pytest

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.tools.atomic_planner_round import (
    ATOMIC_PLANNER_ROUND_SCHEMA_FIELD_NAMES,
    PLANNER_ROUND_TOOL_ID,
)
from intergrax.runtime.nexus.tools.investigation_proof import (
    format_investigation_follow_up_context,
)
from intergrax.runtime.nexus.tools.tool_planning_prompts import (
    GENERIC_INVESTIGATION_POLICY_PROMPT_ID,
    composed_investigation_policy_prompt,
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


_LEGACY_TRANSPORT_MARKERS = (
    "EVIDENCE_BASIS:",
    "PURPOSE:",
    "public decision note in assistant content",
)


def test_tools_investigation_policy_atomic_transport_contract() -> None:
    text = investigation_policy_prompt()
    for field_name in ATOMIC_PLANNER_ROUND_SCHEMA_FIELD_NAMES:
        assert field_name in text
    assert PLANNER_ROUND_TOOL_ID in text
    for marker in _LEGACY_TRANSPORT_MARKERS:
        assert marker not in text
    assert "`purpose`" in text or "field name `purpose`" in text
    assert "public_purpose" in text
    assert "Do not use `public_purpose`" in text
    assert "Do not call business tools directly" in text
    assert "planning protocol metadata" in text


def test_atomic_planner_prompt_schema_field_parity() -> None:
    text = investigation_policy_prompt()
    for field_name in sorted(ATOMIC_PLANNER_ROUND_SCHEMA_FIELD_NAMES):
        assert field_name in text


def test_certified_planner_context_uses_purpose_not_public_purpose() -> None:
    generic = investigation_policy_prompt()
    follow_up = format_investigation_follow_up_context(
        round_index=2,
        available_evidence_references=("evidence.a",),
    )
    assert "Use the exact field name `purpose`" in generic
    assert "Do not use `public_purpose`" in generic
    assert "and purpose" in follow_up
    assert "do not use public_purpose" in follow_up.lower()


def test_incident_composed_planner_prompt_single_atomic_transport_owner() -> None:
    composed = composed_investigation_policy_prompt(
        overlay_prompt_id="incident_investigation_policy",
    )
    generic = investigation_policy_prompt()
    overlay = investigation_policy_prompt(prompt_id="incident_investigation_policy")
    assert composed.startswith(generic.strip())
    assert overlay.strip() in composed
    assert composed.count(PLANNER_ROUND_TOOL_ID) == generic.count(PLANNER_ROUND_TOOL_ID)
    for marker in _LEGACY_TRANSPORT_MARKERS:
        assert marker not in composed
    assert "production.comparison.read" in composed
    assert "incident_window" in composed
