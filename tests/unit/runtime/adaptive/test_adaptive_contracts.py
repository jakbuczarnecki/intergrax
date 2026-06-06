# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-1.1–1.2: Adaptive contract validation tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.adaptive.contracts import (
    ProcessPatternAction,
    ProcessPatternProposal,
    ProfileArtifactType,
    ProfileVersionDraft,
    ProfileVersionRecord,
    ProfileVersionStatus,
    UtilityWeights,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_profile_version_draft_requires_version_id() -> None:
    with pytest.raises(ValueError):
        ProfileVersionDraft(
            version_id="  ",
            artifact_type=ProfileArtifactType.RAG,
        )


def test_profile_version_record_roundtrip() -> None:
    record = ProfileVersionRecord(
        version_id="v1",
        artifact_type=ProfileArtifactType.ORCHESTRATION,
        artifact_payload={"long_running_enabled": True},
        created_by="proposal_1",
        status=ProfileVersionStatus.SHADOW,
    )
    restored = ProfileVersionRecord.model_validate(record.model_dump())
    assert restored.status == ProfileVersionStatus.SHADOW


def test_process_pattern_proposal_requires_description() -> None:
    with pytest.raises(ValueError):
        ProcessPatternProposal(
            description="",
            suggested_action=ProcessPatternAction.TUNE_ROUTING,
        )


def test_utility_weights_defaults_match_ahia() -> None:
    weights = UtilityWeights()
    assert weights.w_quality == pytest.approx(0.50)
    assert weights.w_business == pytest.approx(0.0)
