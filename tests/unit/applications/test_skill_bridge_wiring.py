# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.applications._shared.skill_bridge_wiring import (
    merge_skill_policy_fragments,
    skill_prompt_metadata,
)
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier
from intergrax.skills.core.version_binding import (
    ResolvedSkillRef,
    ResolvedSkillRole,
    SkillVersionResolutionMode,
)
from intergrax.skills.resolver import ResolvedSkillPack

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _pack_for_bridge_test(
    *,
    prompt_instruction_ids: frozenset[str] = frozenset(),
    policy_fragment_ids: frozenset[str] = frozenset(),
) -> ResolvedSkillPack:
    ref = ResolvedSkillRef.from_manifest(
        SkillManifest(skill_id="s1", description="s1"),
        resolution_mode=SkillVersionResolutionMode.MATERIALIZED,
        role=ResolvedSkillRole.ROOT,
    )
    return ResolvedSkillPack(
        resolved_skills=(ref,),
        prompt_instruction_ids=prompt_instruction_ids,
        policy_fragment_ids=policy_fragment_ids,
        tool_ids=frozenset(),
        risk_tier=SkillRiskTier.LOW,
        snapshot_digest="sha256:test",
    )


def test_skill_prompt_metadata() -> None:
    pack = _pack_for_bridge_test(prompt_instruction_ids=frozenset({"rag.hybrid_qa.system"}))
    meta = skill_prompt_metadata(pack)
    assert meta["skill_prompt_instruction_ids"] == ["rag.hybrid_qa.system"]


def test_merge_skill_policy_fragments() -> None:
    pack = _pack_for_bridge_test(policy_fragment_ids=frozenset({"policy.frag.a"}))
    bundle = merge_skill_policy_fragments(RuntimePolicyBundle(), pack)
    assert "policy.frag.a" in bundle.domain_fragments
