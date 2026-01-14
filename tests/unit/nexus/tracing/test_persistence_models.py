# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import asyncio

import pytest
from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.errors.classifier import ErrorClassifier
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.pipelines.pipeline_factory import PipelineFactory
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunMetadata, RunStats
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


class _FailingPipeline(RuntimePipeline):
    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        raise RuntimeError("boom")


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
        stats=RunStats(
            duration_ms=0,
            llm_usage={},
        )
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

    # ensure run_abort is not emitted on success
    steps = [e.step for e in run.events]
    assert "run_abort" not in steps



@pytest.mark.asyncio
async def test_runtime_persists_error_metadata_on_exception(monkeypatch) -> None:
    cfg = build_runtime_config_deterministic()
    harness = build_engine_harness(cfg=cfg)
    store = InMemoryRunTraceStore()
    harness.engine.context.trace_writer = store

    def _fake_build_pipeline(*, state):
        return _FailingPipeline()

    monkeypatch.setattr(PipelineFactory, "build_pipeline", _fake_build_pipeline)

    request = RuntimeRequest(
        user_id="user-1",
        session_id="sess-1",
        message="hello",
    )

    with pytest.raises(RuntimeError):
        await harness.engine.run(request)

    # We don't have RuntimeAnswer, so we must locate run_id by reading store content.
    # InMemoryRunTraceStore already keys by run_id; use its internal API only via reader methods:
    # Here we assert that exactly one run was finalized.
    assert len(store._metadata_by_run) == 1  # if you prefer, expose a public method later (separate change)

    run_id = next(iter(store._metadata_by_run.keys()))
    run = store.read_run(run_id)

    assert run.metadata.error is not None
    assert run.metadata.error.error_type == RuntimeErrorCode.INTERNAL_ERROR
    assert "boom" in run.metadata.error.message


def test_error_classifier_validation() -> None:
    assert ErrorClassifier.classify(ValueError("x")) == RuntimeErrorCode.VALIDATION_ERROR


class _SleepingPipeline(RuntimePipeline):
    async def _inner_run(self, state):
        await asyncio.sleep(0.05)
        raise AssertionError("Pipeline should have timed out before producing an answer.")


@pytest.mark.asyncio
async def test_runtime_persists_timeout_error(monkeypatch) -> None:
    cfg = build_runtime_config_deterministic()
    cfg.runtime_timeout_ms = 1

    harness = build_engine_harness(cfg=cfg)
    store = InMemoryRunTraceStore()
    harness.engine.context.trace_writer = store

    def _fake_build_pipeline(*, state):
        return _SleepingPipeline()

    monkeypatch.setattr(PipelineFactory, "build_pipeline", _fake_build_pipeline)

    request = RuntimeRequest(
        user_id="user-1",
        session_id="sess-1",
        message="hello",
    )

    with pytest.raises(asyncio.TimeoutError):
        await harness.engine.run(request)

    assert len(store._metadata_by_run) == 1
    run_id = next(iter(store._metadata_by_run.keys()))
    run = store.read_run(run_id)

    assert run.metadata.error is not None
    assert run.metadata.error.error_type == RuntimeErrorCode.TIMEOUT