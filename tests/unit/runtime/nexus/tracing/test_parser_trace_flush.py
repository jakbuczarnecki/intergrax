# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from intergrax.runtime.nexus.tracing.parser_trace_flush import export_parser_traces_from_events

pytestmark = pytest.mark.gate


@dataclass
class _TaggedEvent:
    tags: dict[str, Any] = field(default_factory=dict)


def test_export_parser_traces_reads_tags_from_dataclass_event() -> None:
    trace_payload = {"spans": [{"name": "parse"}]}
    events = [
        _TaggedEvent(
            tags={
                "integration_parser_trace": trace_payload,
                "source": "unit_test",
            }
        )
    ]
    with patch(
        "intergrax.runtime.nexus.tracing.parser_trace_flush.export_parser_trace"
    ) as export_mock:
        export_parser_traces_from_events(events)
        export_mock.assert_called_once_with(source="unit_test", trace=trace_payload)


def test_export_parser_traces_ignores_events_without_trace_tag() -> None:
    with patch(
        "intergrax.runtime.nexus.tracing.parser_trace_flush.export_parser_trace"
    ) as export_mock:
        export_parser_traces_from_events([{"tags": {"other": True}}])
        export_mock.assert_not_called()
