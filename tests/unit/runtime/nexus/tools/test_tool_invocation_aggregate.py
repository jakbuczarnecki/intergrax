# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-29 — ToolInvocationAggregate acceptance tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace
from intergrax.runtime.nexus.tools.tool_invocation_aggregate import ToolInvocationAggregate

pytestmark = pytest.mark.unit


def test_aggregate_merges_traces_in_order() -> None:
    traces = (
        ToolCallTrace(
            tool_name="alpha.tool",
            arguments={"value": 1},
            output_preview='{"result": 1}',
            success=True,
            error_message=None,
            raw_trace={},
        ),
        ToolCallTrace(
            tool_name="beta.tool",
            arguments={"value": 2},
            output_preview=None,
            success=False,
            error_message="boom",
            raw_trace={},
        ),
    )
    aggregate = ToolInvocationAggregate.from_traces(traces)
    assert aggregate.success_count == 1
    assert aggregate.failure_count == 1
    assert "alpha.tool" in aggregate.combined_context
    assert "beta.tool" in aggregate.combined_context
    assert "boom" in aggregate.combined_context
    assert aggregate.traces == traces
