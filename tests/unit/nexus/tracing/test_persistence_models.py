# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunMetadata
from intergrax.runtime.nexus.tracing.trace_models import (
    DiagnosticPayload,
    TraceComponent,
    TraceEvent,
    TraceLevel,
)
from tests._support.builder import build_engine_harness, build_runtime_config_deterministic

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _DummyDiag(DiagnosticPayload):
    value: int

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.tests.dummy"

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value}


def test_trace_event_to_dict_serializes_payload_schema_and_dict() -> None:
    evt = TraceEvent(
        event_id="evt-1",
        run_id="run-1",
        seq=1,
        ts_utc="2026-01-01T00:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.ENGINE,
        step="test",
        message="hello",
        payload=_DummyDiag(value=42),
        tags={"k": "v"},
    )

    data = evt.to_dict()

    assert data["event_id"] == "evt-1"
    assert data["run_id"] == "run-1"
    assert data["seq"] == 1
    assert data["ts_utc"] == "2026-01-01T00:00:00Z"
    assert data["level"] == "info"
    assert data["component"] == "engine"
    assert data["step"] == "test"
    assert data["message"] == "hello"

    assert data["payload_schema_id"] == "intergrax.tests.dummy"
    assert data["payload_schema_version"] == 1
    assert data["payload"] == {"value": 42}

    assert data["tags"] == {"k": "v"}


def test_persisted_run_holds_metadata_and_serialized_events() -> None:
    meta = RunMetadata(
        run_id="run-1",
        session_id="sess-1",
        user_id="user-1",
        tenant_id="tenant-1",
        started_at_utc="2026-01-01T00:00:00Z",
    )

    run = PersistedRun(metadata=meta, events=[])

    assert run.metadata.run_id == "run-1"
    assert run.events == []


@pytest.mark.asyncio
async def test_runtime_finalizes_run_metadata() -> None:
    cfg = build_runtime_config_deterministic()

    harness = build_engine_harness(cfg=cfg)
    store = InMemoryRunTraceStore()

    # Inject trace store into already-built context
    harness.engine.context.trace_writer = store

    request = RuntimeRequest(
        user_id="user-1",
        session_id="sess-1",
        message="hello",
    )

    answer = await harness.engine.run(request)

    run = store.read_run(answer.run_id)

    assert run.metadata.run_id == answer.run_id
    assert run.metadata.session_id == "sess-1"
    assert run.metadata.user_id == "user-1"    
    assert run.metadata.tenant_id in ("test-tenant", "", "tenant-1")
    assert run.metadata.started_at_utc != ""