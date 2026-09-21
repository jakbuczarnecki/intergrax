# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.persisted_run_trace import (
    PersistedRunErrorCode,
    PersistedTraceEvent,
    decode_persisted_run_error,
    decode_persisted_run_stats,
    decode_persisted_trace_event,
    parse_persisted_run_error_code,
    persisted_trace_event_to_wire,
)

pytestmark = pytest.mark.unit


def test_decode_persisted_run_stats_normalizes_llm_usage() -> None:
    stats = decode_persisted_run_stats(
        {"duration_ms": 10, "llm_usage": {"total_tokens": 4, "cost": 0.01}}
    )
    assert stats.duration_ms == 10
    assert stats.llm_usage["total_tokens"] == 4


def test_unknown_error_code_maps_to_unknown_enum() -> None:
    err = decode_persisted_run_error({"error_type": "not_a_real_code", "message": "m"})
    assert err.error_type is PersistedRunErrorCode.UNKNOWN
    assert parse_persisted_run_error_code("") is PersistedRunErrorCode.UNKNOWN


def test_trace_event_wire_roundtrip() -> None:
    raw = {
        "event_id": "e",
        "run_id": "r",
        "seq": 2,
        "ts_utc": "t",
        "level": "INFO",
        "component": "tools",
        "step": "tool.run",
        "message": "done",
        "payload": None,
        "tags": {},
        "artifact_refs": [],
    }
    event = decode_persisted_trace_event(raw)
    assert persisted_trace_event_to_wire(event)["step"] == "tool.run"
