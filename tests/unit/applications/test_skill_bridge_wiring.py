# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.applications._shared.skill_bridge_wiring import (
    merge_skill_policy_fragments,
    skill_prompt_metadata,
)
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.skills.core.contracts import SkillRiskTier
from intergrax.skills.resolver import ResolvedSkillPack

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_skill_prompt_metadata() -> None:
    pack = ResolvedSkillPack(
        skill_ids=("s1",),
        prompt_instruction_ids=frozenset({"rag.hybrid_qa.system"}),
        policy_fragment_ids=frozenset(),
        tool_ids=frozenset(),
        risk_tier=SkillRiskTier.LOW,
    )
    meta = skill_prompt_metadata(pack)
    assert meta["skill_prompt_instruction_ids"] == ["rag.hybrid_qa.system"]


def test_merge_skill_policy_fragments() -> None:
    pack = ResolvedSkillPack(
        skill_ids=("s1",),
        prompt_instruction_ids=frozenset(),
        policy_fragment_ids=frozenset({"policy.frag.a"}),
        tool_ids=frozenset(),
        risk_tier=SkillRiskTier.LOW,
    )
    bundle = merge_skill_policy_fragments(RuntimePolicyBundle(), pack)
    assert "policy.frag.a" in bundle.domain_fragments
