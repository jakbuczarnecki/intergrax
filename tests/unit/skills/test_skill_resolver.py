# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import SkillResolutionError, SkillResolver
@pytest.mark.unit
def test_skill_resolver_merges_two_skills() -> None:
    skills = SkillRegistry()
    skills.register(
        SkillManifest(skill_id="a.pack", description="a", tool_ids=("t1", "t2"))
    )
    skills.register(
        SkillManifest(skill_id="b.pack", description="b", tool_ids=("t2", "t3"))
    )

    pack = SkillResolver(skills).resolve(["a.pack", "b.pack"])
    assert pack.tool_ids == frozenset({"t1", "t2", "t3"})
    assert pack.merged_allowed_tools() == ("t1", "t2", "t3")


@pytest.mark.unit
def test_skill_resolver_validates_tools_against_registry() -> None:
    skills = SkillRegistry()
    skills.register(
        SkillManifest(skill_id="a.pack", description="a", tool_ids=("known.tool",))
    )

    class _StubToolRegistry:
        def has(self, tool_id: str) -> bool:
            return tool_id == "known.tool"

    pack = SkillResolver(skills, _StubToolRegistry()).resolve(["a.pack"])
    assert "known.tool" in pack.tool_ids

    skills.register(
        SkillManifest(skill_id="b.pack", description="b", tool_ids=("missing.tool",))
    )
    with pytest.raises(SkillResolutionError):
        SkillResolver(skills, _StubToolRegistry()).resolve(["b.pack"])


@pytest.mark.unit
def test_skill_resolver_unknown_skill_raises() -> None:
    resolver = SkillResolver(SkillRegistry())
    with pytest.raises(SkillResolutionError):
        resolver.validate_skill_ids(["missing.skill"])
