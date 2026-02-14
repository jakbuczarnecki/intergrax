# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Unit tests for TraceEvent serialization.

These tests define the behavioral contract for TraceEvent.to_dict():
- enum fields are serialized using their `.value`,
- DiagnosticPayload is exported as:
    - payload_schema_id (from payload.__class__.schema_id())
    - payload_schema_version (from payload.__class__.schema_version())
    - payload (from payload.to_dict())
- when payload is None, payload-related fields must be None (not missing, not {}).
- tags are preserved as-is.

Why this matters:
TraceEvent JSON is a production boundary for logs, exports, persistence and diagnostics.
Breaking this contract silently corrupts observability and stored traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from intergrax.runtime.nexus.tracing.trace_models import (
    DiagnosticPayload,
    TraceComponent,
    TraceEvent,
    TraceLevel,
)


pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _DummyPayload(DiagnosticPayload):
    value: int

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.test.dummy_payload"

    @classmethod
    def schema_version(cls) -> int:
        return 7

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value}


def test_trace_event_to_dict_without_payload_has_null_payload_fields() -> None:
    """
    When payload is None, payload_schema_id/version and payload must be None.
    """
    ev = TraceEvent(
        event_id="e1",
        run_id="r1",
        seq=1,
        ts_utc="2026-01-01T00:00:00+00:00",
        level=TraceLevel.INFO,
        component=TraceComponent.RUNTIME,
        step="S",
        message="M",
        payload=None,
        tags={"x": 1},
    )

    d = ev.to_dict()

    assert d["event_id"] == "e1"
    assert d["run_id"] == "r1"
    assert d["seq"] == 1
    assert d["ts_utc"] == "2026-01-01T00:00:00+00:00"

    # Enums must serialize to `.value`
    assert d["level"] == "info"
    assert d["component"] == "runtime"

    assert d["step"] == "S"
    assert d["message"] == "M"

    # Payload export contract
    assert d["payload_schema_id"] is None
    assert d["payload_schema_version"] is None
    assert d["payload"] is None

    # tags preserved
    assert d["tags"] == {"x": 1}


def test_trace_event_to_dict_with_payload_exports_schema_and_payload_dict() -> None:
    """
    When payload is present, schema id/version and payload dict must be exported.
    """
    payload = _DummyPayload(value=123)

    ev = TraceEvent(
        event_id="e2",
        run_id="r2",
        seq=2,
        ts_utc="2026-01-01T00:00:01+00:00",
        level=TraceLevel.DEBUG,
        component=TraceComponent.PLANNER,
        step="plan",
        message="built",
        payload=payload,
        tags={"phase": "unit"},
    )

    d = ev.to_dict()

    assert d["level"] == "debug"
    assert d["component"] == "planner"

    assert d["payload_schema_id"] == "intergrax.test.dummy_payload"
    assert d["payload_schema_version"] == 7
    assert d["payload"] == {"value": 123}

    assert d["tags"] == {"phase": "unit"}
