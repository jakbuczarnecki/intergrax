# © Artur Czarnecki. All rights reserved.

"""Tests for skill-driven Tier-1 tool profile extension."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.skill_tool_profile import (
    extend_tool_profile_for_skills,
    tool_ids_referenced_by_skill_profile,
)
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry.profile import ToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_tool_ids_referenced_by_legal_skill_bundle() -> None:
    tool_ids = tool_ids_referenced_by_skill_profile(SkillProfile(enabled_bundles=["legal"]))
    assert "rag.retrieve" in tool_ids
    assert "websearch.query" in tool_ids


def test_extend_tool_profile_adds_skill_tools_without_duplicates() -> None:
    profile = extend_tool_profile_for_skills(
        ToolProfile(enabled=["rag.retrieve"]),
        SkillProfile(enabled_bundles=["legal"]),
    )
    assert profile.enabled.count("rag.retrieve") == 1
    assert "websearch.query" in profile.enabled


def test_empty_skill_profile_is_noop() -> None:
    base = ToolProfile(enabled=["database.query"])
    assert extend_tool_profile_for_skills(base, SkillProfile()) is base
