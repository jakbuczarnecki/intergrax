# © Artur Czarnecki. All rights reserved.

"""ME-5 search contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.capability_catalog import (
    CapabilitySearchEvidence,
    CapabilitySearchQuery,
    CapabilitySearchSignal,
)

pytestmark = pytest.mark.unit


def test_search_evidence_requires_strategy_id() -> None:
    CapabilitySearchEvidence(
        search_strategy_id="catalog.entry_text",
        signal=CapabilitySearchSignal.PASS_THROUGH,
    )
    with pytest.raises(ValidationError):
        CapabilitySearchEvidence(
            search_strategy_id="",
            signal=CapabilitySearchSignal.PASS_THROUGH,
        )


def test_search_query_allows_optional_text() -> None:
    assert CapabilitySearchQuery().text is None
    assert CapabilitySearchQuery(text="echo").text == "echo"
