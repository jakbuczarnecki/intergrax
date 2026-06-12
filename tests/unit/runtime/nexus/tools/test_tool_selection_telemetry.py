# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-32 — tool selection trace telemetry."""

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.tracing.tools.tool_selection import (
    ToolSelectionCandidateDiagV1,
    ToolSelectionDiagV1,
)

pytestmark = pytest.mark.unit


def test_tool_selection_diag_schema() -> None:
    payload = ToolSelectionDiagV1(
        strategy_id="semantic",
        selection_mode="semantic",
        candidate_tool_ids=["rag.retrieve"],
        candidates=[ToolSelectionCandidateDiagV1(tool_id="rag.retrieve", score=0.91)],
    )
    assert payload.schema_id() == "intergrax.diag.tools.selection"
    data = payload.to_dict()
    assert data["ops"] == "tool_selection"
    assert data["candidates"][0]["score"] == 0.91
