# © Artur Czarnecki. All rights reserved.

"""CRIT-V-2 eval.judge and eval.trajectory tool tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.architecture.online_evaluation_registry import InMemoryOnlineEvaluationRegistry
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunMetadata, RunStats
from intergrax.tools.providers.eval.contracts import (
    EvalJudgeInput,
    EvalListObservationsInput,
    EvalTrajectoryInput,
)
from intergrax.tools.providers.eval.judge import _JudgeLLMResult, eval_judge
from intergrax.tools.providers.eval.service import eval_list_observations
from intergrax.tools.providers.eval.trajectory import eval_trajectory
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FakeTraceReader:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self._events = events

    def read_run(self, run_id: str, tenant_id: str) -> PersistedRun:
        metadata = RunMetadata(
            run_id=run_id,
            session_id="s1",
            user_id="u1",
            tenant_id=tenant_id,
            started_at_utc="2026-06-07T00:00:00Z",
            stats=RunStats(duration_ms=100, llm_usage={}),
        )
        return PersistedRun(metadata=metadata, events=self._events)


def test_eval_judge_scores_with_llm_adapter() -> None:
    llm = FakeLLMAdapter(
        fake_structured_data=_JudgeLLMResult(score=0.92, passed=True, reasons=["complete answer"]),
    )
    ctx = ToolWiringContext(extras={"llm_adapter": llm})
    out = eval_judge(
        ctx,
        EvalJudgeInput(
            output_text="The contract is valid.",
            rubric_id="legal.summary",
            criteria=["mentions validity"],
            min_score=0.75,
        ),
    )
    assert out.passed is True
    assert out.score == pytest.approx(0.92)
    assert out.reasons == ["complete answer"]


def test_eval_judge_records_observation_when_requested() -> None:
    llm = FakeLLMAdapter(
        fake_structured_data=_JudgeLLMResult(score=0.5, passed=False, reasons=["incomplete"]),
    )
    registry = InMemoryOnlineEvaluationRegistry()
    ctx = ToolWiringContext(
        evaluation_registry=registry,
        extras={"llm_adapter": llm},
    )
    out = eval_judge(
        ctx,
        EvalJudgeInput(
            output_text="partial",
            rubric_id="case.a",
            min_score=0.75,
            run_id="run-judge-1",
            agent_id="agent-a",
            record_observation=True,
            observation_id="obs-judge-1",
        ),
    )
    assert out.passed is False
    assert out.observation_recorded is True
    listed = eval_list_observations(ctx, EvalListObservationsInput(limit=10))
    assert listed.total == 1
    assert listed.average_score == pytest.approx(0.5)


def test_eval_judge_requires_llm_adapter() -> None:
    with pytest.raises(RuntimeError, match="llm_adapter_not_configured"):
        eval_judge(
            ToolWiringContext(),
            EvalJudgeInput(output_text="x", rubric_id="r1"),
        )


def test_eval_trajectory_scores_trace_with_duplicates() -> None:
    events = [
        {"step": "tool_invocation_start", "message": "rag.retrieve", "payload": {"tool_name": "rag.retrieve"}},
        {"step": "tool_invocation_start", "message": "rag.retrieve", "payload": {"tool_name": "rag.retrieve"}},
        {"step": "tool_invocation_error", "message": "failed", "payload": {}},
    ]
    ctx = ToolWiringContext(trace_reader=_FakeTraceReader(events))
    out = eval_trajectory(
        ctx,
        EvalTrajectoryInput(run_id="run-traj-1", tenant_id="t1", min_score=0.5),
    )
    assert out.tool_call_count == 2
    assert out.duplicate_tool_calls == 1
    assert out.error_count == 1
    assert out.score < 1.0


def test_eval_trajectory_records_observation() -> None:
    events = [
        {"step": "tool_invocation_start", "message": "ok", "payload": {"tool_name": "harness.echo"}},
    ]
    registry = InMemoryOnlineEvaluationRegistry()
    ctx = ToolWiringContext(
        evaluation_registry=registry,
        trace_reader=_FakeTraceReader(events),
    )
    out = eval_trajectory(
        ctx,
        EvalTrajectoryInput(
            run_id="run-traj-2",
            tenant_id="t1",
            agent_id="agent-b",
            record_observation=True,
            observation_id="obs-traj-1",
        ),
    )
    assert out.observation_recorded is True
    listed = eval_list_observations(ctx, EvalListObservationsInput(limit=10))
    assert listed.total == 1


def test_eval_trajectory_requires_trace_reader() -> None:
    with pytest.raises(RuntimeError, match="trace_reader_not_configured"):
        eval_trajectory(
            ToolWiringContext(),
            EvalTrajectoryInput(run_id="run-x", tenant_id="t1"),
        )
