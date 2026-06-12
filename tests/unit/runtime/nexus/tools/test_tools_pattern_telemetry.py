# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-27 — tool invocation pattern trace payload."""

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.tracing.tools.tools_summary import ToolsSummaryDiagV1

pytestmark = pytest.mark.unit


def test_tools_summary_includes_pattern_fields() -> None:
    payload = ToolsSummaryDiagV1(
        tools_mode="auto",
        used_tools=True,
        tool_calls_count=1,
        tool_names=["rag.retrieve"],
        warning=None,
        error_type=None,
        error_message=None,
        pattern_id="parallel_batch",
        stop_reason="legacy_single_pass",
    )
    data = payload.to_dict()
    assert data["ops"] == "tool_invocation_pattern"
    assert data["pattern_id"] == "parallel_batch"
    assert data["stop_reason"] == "legacy_single_pass"
