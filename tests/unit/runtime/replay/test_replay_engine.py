# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from typing import List

from intergrax.runtime.replay.contracts.artifact_dto import ArtifactDTO
from intergrax.runtime.replay.contracts.run_record_dto import RunRecordDTO
from intergrax.runtime.replay.contracts.trace_event_dto import TraceEventDTO
from intergrax.runtime.replay.replay_engine import ReplayEngine
from intergrax.runtime.replay.models import ReconstructedRun


# ---------------- FAKE STORES ----------------


class FakeRunStore:
    def get(self, run_id: str) -> RunRecordDTO:
        return RunRecordDTO(
            run_id=run_id,
            started_at=0.0,
            finished_at=10.0,
            status="finished",
            final_answer="Hello World",
        )


class FakeTraceStore:
    def get_events(self, run_id: str) -> List[TraceEventDTO]:
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


class FakeArtifactStore:
    def list_for_run(self, run_id: str) -> List[ArtifactDTO]:
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

    result: ReconstructedRun = engine.reconstruct("run-1")

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
