# © Artur Czarnecki. All rights reserved.

"""UCA-6A — CodeCraft gap synthesis public contract invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.need import CapabilityNeed
from intergrax.contracts.codecraft.gap_synthesis import (
    CodeCraftGapSynthesisOutcome,
    CodeCraftGapSynthesisRequest,
    CodeCraftGapSynthesisResult,
)

pytestmark = pytest.mark.unit


def test_succeeded_without_subject_raises() -> None:
    with pytest.raises(ValidationError):
        CodeCraftGapSynthesisResult(
            operation_id="op-1",
            gap_id="gap-1",
            outcome=CodeCraftGapSynthesisOutcome.SUCCEEDED,
        )


def test_succeeded_with_artifact_reference_is_valid() -> None:
    result = CodeCraftGapSynthesisResult(
        operation_id="op-1",
        gap_id="gap-1",
        outcome=CodeCraftGapSynthesisOutcome.SUCCEEDED,
        artifact_reference="codecraft:artifact:craft-abc",
    )
    assert result.artifact_reference == "codecraft:artifact:craft-abc"


def test_gap_synthesis_request_requires_goal() -> None:
    need = CapabilityNeed(
        need_id="need-1",
        kinds=(CapabilityKind.TOOL,),
        intent_summary="build csv parser tool",
    )
    req = CodeCraftGapSynthesisRequest(
        operation_id="op-1",
        gap_id="gap-1",
        canonical_discovery_correlation_id="disc-corr",
        capability_need=need,
        synthesis_goal="build csv parser tool",
    )
    assert req.target_kind is CapabilityKind.TOOL
