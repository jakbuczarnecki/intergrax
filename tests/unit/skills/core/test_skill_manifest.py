# © Artur Czarnecki. All rights reserved.

import pytest
from pydantic import ValidationError

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier


@pytest.mark.unit
def test_skill_manifest_round_trip_json() -> None:
    manifest = SkillManifest(
        skill_id="legal.contract_review",
        version="1.0.0",
        description="Review contracts",
        tool_ids=("rag.retrieve", "websearch.query"),
        prompt_instruction_ids=("legal.contract_review.system",),
        policy_fragment_id="legal.policy",
        risk_tier=SkillRiskTier.HIGH,
        tags=("legal",),
    )
    payload = manifest.model_dump(mode="json")
    restored = SkillManifest.model_validate(payload)
    assert restored == manifest
    assert restored.qualified_id == "legal.contract_review@1.0.0"


@pytest.mark.unit
def test_skill_manifest_rejects_duplicate_tool_ids() -> None:
    with pytest.raises(ValidationError):
        SkillManifest(
            skill_id="x.y",
            description="d",
            tool_ids=("a", "a"),
        )
