# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 4 ranking contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.capability_catalog import (
    CapabilityRankingContext,
    CapabilityRankingEvidence,
    CapabilityRankingSignal,
)

pytestmark = pytest.mark.unit


def test_ranking_evidence_requires_contiguous_position() -> None:
    CapabilityRankingEvidence(
        ranker_id="stable.identity",
        rank_position=1,
        signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
    )
    with pytest.raises(ValidationError):
        CapabilityRankingEvidence(
            ranker_id="stable.identity",
            rank_position=0,
            signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
        )


def test_ranking_context_is_optional_semantic_need() -> None:
    context = CapabilityRankingContext()
    assert context.semantic_need is None
    assert CapabilityRankingContext(semantic_need="echo tool").semantic_need == "echo tool"
