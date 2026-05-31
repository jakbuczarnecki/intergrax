# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.integrations.contracts.observability_backend import TraceQueryResult, TraceRecord
from intergrax.tools.providers.observability.contracts import TracesQueryInput
from intergrax.tools.providers.observability.service import traces_query
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_observability_query_traces_delegates_to_backend() -> None:
    backend = MagicMock()
    backend.query_traces.return_value = TraceQueryResult(
        traces=[TraceRecord(trace_id="t1", name="document_parser_trace")]
    )
    out = traces_query(ToolWiringContext(observability_backend=backend), TracesQueryInput(limit=5))
    assert out.total == 1
    assert out.traces[0].trace_id == "t1"
