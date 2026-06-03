# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import SkillResolutionError, SkillResolver

pytestmark = pytest.mark.unit


def test_requires_skills_expands_dependencies_before_root() -> None:
    base = SkillManifest(
        skill_id="base.pack",
        description="base",
        tool_ids=("tool.a",),
    )
    derived = SkillManifest(
        skill_id="derived.pack",
        description="derived",
        tool_ids=("tool.b",),
        requires_skills=("base.pack",),
    )
    registry = SkillRegistry()
    registry.register(base)
    registry.register(derived)
    pack = SkillResolver(registry).resolve(["derived.pack"])
    assert pack.skill_ids == ("base.pack", "derived.pack")
    assert pack.tool_ids == frozenset({"tool.a", "tool.b"})


def test_requires_skills_unknown_dependency_raises() -> None:
    registry = SkillRegistry()
    registry.register(
        SkillManifest(
            skill_id="orphan",
            description="x",
            requires_skills=("missing.pack",),
        )
    )
    with pytest.raises(SkillResolutionError, match="Unknown skill_id"):
        SkillResolver(registry).resolve(["orphan"])
