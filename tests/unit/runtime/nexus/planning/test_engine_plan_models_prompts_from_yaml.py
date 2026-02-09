# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.planning.engine_plan_models import (
    BASE_PLANNER_SYSTEM_PROMPT,
    DEFAULT_PLANNER_SYSTEM_PROMPT,
    DEFAULT_PLANNER_REPLAN_SYSTEM_PROMPT,
    DEFAULT_PLANNER_NEXT_STEP_RULES_PROMPT,
    DEFAULT_PLANNER_FALLBACK_CLARIFY_QUESTION,
)

pytestmark = pytest.mark.unit


def _assert_non_empty_str(value: str) -> None:
    assert isinstance(value, str)
    assert value.strip()


def test_base_planner_system_prompt_is_loaded_from_yaml() -> None:
    """
    Contract: base planner system prompt is resolved from YAML and is non-empty.
    """
    text = BASE_PLANNER_SYSTEM_PROMPT()
    _assert_non_empty_str(text)

    # Minimal semantic anchors (stable, not too brittle)
    low = text.lower()
    assert "json" in low
    assert "schema" in low or "must match" in low


def test_default_planner_system_prompt_is_loaded_from_yaml() -> None:
    """
    Contract: default planner system prompt is resolved from YAML and is non-empty.
    """
    text = DEFAULT_PLANNER_SYSTEM_PROMPT()
    _assert_non_empty_str(text)

    low = text.lower()
    assert "json" in low or "schema" in low


def test_replan_system_prompt_template_contains_placeholder() -> None:
    """
    Contract:
    - replan prompt is resolved from YAML,
    - returned string is a template compatible with legacy `.format(replan_json=...)`,
      i.e. it contains `{replan_json}`.
    """
    template = DEFAULT_PLANNER_REPLAN_SYSTEM_PROMPT()
    _assert_non_empty_str(template)

    assert "{replan_json}" in template

    # Ensure formatting actually works (and doesn't throw)
    rendered = template.format(replan_json='{"reason":"unit-test"}')
    _assert_non_empty_str(rendered)
    assert '{"reason":"unit-test"}' in rendered


def test_next_step_rules_prompt_is_loaded_from_yaml() -> None:
    """
    Contract: next-step rules prompt is resolved from YAML and is non-empty.
    """
    text = DEFAULT_PLANNER_NEXT_STEP_RULES_PROMPT()
    _assert_non_empty_str(text)

    low = text.lower()
    # Minimal semantic anchors
    assert "next" in low or "step" in low


def test_fallback_clarify_question_prompt_is_loaded_from_yaml() -> None:
    """
    Contract: fallback clarify question prompt is resolved from YAML and is non-empty.
    """
    text = DEFAULT_PLANNER_FALLBACK_CLARIFY_QUESTION()
    _assert_non_empty_str(text)

    # Must look like a question (heuristic, but stable)
    assert "?" in text
