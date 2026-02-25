# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

import pytest
from typing import Iterable, List

from intergrax.runtime.replay.contracts.artifact_dto import ArtifactDTO
from intergrax.runtime.replay.contracts.artifact_store import ReplayArtifactStore
from intergrax.runtime.replay.contracts.run_record_dto import RunRecordDTO
from intergrax.runtime.replay.contracts.run_record_store import RunRecordStore
from intergrax.runtime.replay.contracts.trace_event_dto import TraceEventDTO
from intergrax.runtime.replay.contracts.trace_event_store import TraceEventStore
from intergrax.runtime.replay.replay_engine import ReplayEngine
from intergrax.runtime.replay.models import ReconstructedRun

pytestmark = pytest.mark.unit


# ---------------- FAKE STORES ----------------

TENANT = "tenant_test"

class FakeRunStore(RunRecordStore):
    def get(self, tenant_id: str, run_id: str) -> RunRecordDTO:
        return RunRecordDTO(
            run_id=run_id,
            started_at=0.0,
            finished_at=10.0,
            status="finished",
            final_answer="Hello World",
        )


class FakeTraceStore(TraceEventStore):
    def get_events(self, tenant_id: str, run_id: str) -> Iterable[TraceEventDTO]:
        return [
            TraceEventDTO(
                run_id=run_id,
                step_id="step-1",
                event_type="STEP_STARTED",
                timestamp=1.0,
                payload={"step_type": "planner"},
            ),
            TraceEventDTO(
                run_id=run_id,
                step_id="step-1",
                event_type="LLM_CALL",
                timestamp=2.0,
                payload={
                    "model": "gpt-4",
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "finish_reason": "stop",
                },
            ),
            TraceEventDTO(
                run_id=run_id,
                step_id="step-1",
                event_type="TOOL_EXECUTED",
                timestamp=3.0,
                payload={
                    "tool_id": "search_tool",
                    "input": {"query": "test"},
                    "output": {"result": "ok"},
                    "success": True,
                },
            ),
            TraceEventDTO(
                run_id=run_id,
                step_id="step-1",
                event_type="STEP_FINISHED",
                timestamp=4.0,
                payload={"status": "success"},
            ),
        ]


class FakeArtifactStore(ReplayArtifactStore):
    def list_for_run(self, tenant_id: str, run_id: str) -> Iterable[ArtifactDTO]:
        return [
            ArtifactDTO(
                artifact_id="art-1",
                name="doc",
                type="text",
                produced_by_step="step-1",
                metadata={"size": 123},
            )
        ]


# ---------------- TEST ----------------


def test_replay_engine_reconstructs_run():
    engine = ReplayEngine(
        run_store=FakeRunStore(),
        trace_store=FakeTraceStore(),
        artifact_store=FakeArtifactStore(),
    )

    result: ReconstructedRun = engine.reconstruct(TENANT, "run-1")

    assert result.run_id == "run-1"
    assert result.final_answer == "Hello World"

    assert len(result.steps) == 1
    step = result.steps[0]

    assert step.step_id == "step-1"
    assert step.status == "success"
    assert step.started_at == 1.0
    assert step.finished_at == 4.0

    assert len(step.llm_calls) == 1
    assert step.llm_calls[0].model == "gpt-4"

    assert len(step.tool_calls) == 1
    assert step.tool_calls[0].tool_id == "search_tool"

    assert len(step.artifacts) == 1
    assert step.artifacts[0].artifact_id == "art-1"


def test_replay_engine_run_without_tool_calls():
    class NoToolTraceStore(TraceEventStore):
        def get_events(self, tenant_id: str, run_id: str) -> Iterable[TraceEventDTO]:
            return [
                TraceEventDTO(
                    run_id=run_id,
                    step_id="step-1",
                    event_type="STEP_STARTED",
                    timestamp=1.0,
                    payload={"step_type": "core"},
                ),
                TraceEventDTO(
                    run_id=run_id,
                    step_id="step-1",
                    event_type="LLM_CALL",
                    timestamp=2.0,
                    payload={
                        "model": "gpt-4",
                        "prompt_tokens": 5,
                        "completion_tokens": 5,
                        "total_tokens": 10,
                        "finish_reason": "stop",
                    },
                ),
                TraceEventDTO(
                    run_id=run_id,
                    step_id="step-1",
                    event_type="STEP_FINISHED",
                    timestamp=3.0,
                    payload={"status": "finished"},
                ),
            ]

    class EmptyArtifactStore(ReplayArtifactStore):
        def list_for_run(self, tenant_id: str, run_id: str) -> Iterable[ArtifactDTO]:
            return []

    engine = ReplayEngine(
        run_store=FakeRunStore(),
        trace_store=NoToolTraceStore(),
        artifact_store=EmptyArtifactStore(),
    )

    result = engine.reconstruct(TENANT, "run-2")

    assert len(result.steps) == 1
    step = result.steps[0]

    assert len(step.tool_calls) == 0
    assert len(step.llm_calls) == 1
    assert result.tool_calls == []


def test_replay_engine_step_without_finish_event():
    class NoFinishTraceStore(TraceEventStore):
        def get_events(self, tenant_id: str, run_id: str) -> Iterable[TraceEventDTO]:
            return [
                TraceEventDTO(
                    run_id=run_id,
                    step_id="step-1",
                    event_type="STEP_STARTED",
                    timestamp=1.0,
                    payload={"step_type": "core"},
                ),
                TraceEventDTO(
                    run_id=run_id,
                    step_id="step-1",
                    event_type="LLM_CALL",
                    timestamp=2.0,
                    payload={
                        "model": "gpt-4",
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                        "total_tokens": 7,
                        "finish_reason": "stop",
                    },
                ),
            ]

    class EmptyArtifactStore(ReplayArtifactStore):
        def list_for_run(self, tenant_id: str, run_id: str) -> Iterable[ArtifactDTO]:
            return []

    engine = ReplayEngine(
        run_store=FakeRunStore(),
        trace_store=NoFinishTraceStore(),
        artifact_store=EmptyArtifactStore(),
    )

    result = engine.reconstruct(TENANT, "run-3")

    assert len(result.steps) == 1
    step = result.steps[0]

    assert step.status == "running"
    assert step.finished_at is None
    assert len(step.llm_calls) == 1
    assert len(step.tool_calls) == 0


def test_replay_engine_multiple_steps_ordering():
    class MultiStepTraceStore(TraceEventStore):
        def get_events(self, tenant_id: str, run_id: str) -> Iterable[TraceEventDTO]:
            return [
                # Step B starts later
                TraceEventDTO(
                    run_id=run_id,
                    step_id="step-B",
                    event_type="STEP_STARTED",
                    timestamp=5.0,
                    payload={"step_type": "secondary"},
                ),
                # Step A starts first
                TraceEventDTO(
                    run_id=run_id,
                    step_id="step-A",
                    event_type="STEP_STARTED",
                    timestamp=1.0,
                    payload={"step_type": "primary"},
                ),
                TraceEventDTO(
                    run_id=run_id,
                    step_id="step-A",
                    event_type="STEP_FINISHED",
                    timestamp=2.0,
                    payload={"status": "finished"},
                ),
                TraceEventDTO(
                    run_id=run_id,
                    step_id="step-B",
                    event_type="STEP_FINISHED",
                    timestamp=6.0,
                    payload={"status": "finished"},
                ),
            ]

    class EmptyArtifactStore(ReplayArtifactStore):
        def list_for_run(self, tenant_id: str, run_id: str) -> Iterable[ArtifactDTO]:
            return []

    engine = ReplayEngine(
        run_store=FakeRunStore(),
        trace_store=MultiStepTraceStore(),
        artifact_store=EmptyArtifactStore(),
    )

    result = engine.reconstruct(TENANT, "run-4")

    assert len(result.steps) == 2

    # Must be ordered by started_at
    assert result.steps[0].step_id == "step-A"
    assert result.steps[1].step_id == "step-B"

    assert result.steps[0].started_at == 1.0
    assert result.steps[1].started_at == 5.0


def test_replay_engine_ignores_events_without_step_started():
    class NoStartTraceStore(TraceEventStore):
        def get_events(self, tenant_id: str, run_id: str) -> Iterable[TraceEventDTO]:
            return [
                TraceEventDTO(
                    run_id=run_id,
                    step_id="step-1",
                    event_type="LLM_CALL",
                    timestamp=1.0,
                    payload={
                        "model": "gpt-4",
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                        "finish_reason": "stop",
                    },
                ),
                TraceEventDTO(
                    run_id=run_id,
                    step_id="step-1",
                    event_type="STEP_FINISHED",
                    timestamp=2.0,
                    payload={"status": "finished"},
                ),
            ]

    class EmptyArtifactStore(ReplayArtifactStore):
        def list_for_run(self, tenant_id: str, run_id: str) -> Iterable[ArtifactDTO]:
            return []

    engine = ReplayEngine(
        run_store=FakeRunStore(),
        trace_store=NoStartTraceStore(),
        artifact_store=EmptyArtifactStore(),
    )

    result = engine.reconstruct(TENANT, "run-5")

    # No STEP_STARTED → no reconstructed steps
    assert result.steps == []
    assert result.llm_calls == []
    assert result.tool_calls == []


