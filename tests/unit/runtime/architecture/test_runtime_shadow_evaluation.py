# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.runtime.architecture.online_evaluation_registry import InMemoryOnlineEvaluationRegistry
from intergrax.runtime.architecture.runtime_governance_bridge import RuntimeArchitectureGovernanceBridge
from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.responses.response_schema import RouteInfo, RuntimeAnswer, RuntimeRequest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_maybe_record_harness_shadow_evaluation_emits_trace() -> None:
    engine = RuntimeEngine(MagicMock())
    request = RuntimeRequest(
        agent_id="echo",
        user_id="u1",
        session_id="s1",
        message="hi",
        tenant_id="t1",
        metadata={
            "harness_shadow_eval": {
                "scenario_id": "harness.echo",
                "passed": True,
                "score": 0.95,
            }
        },
    )
    state = MagicMock()
    state.run_id = "run-1"
    traces: list[str] = []

    def _trace_event(*, step: str, **kwargs) -> None:
        traces.append(step)

    state.trace_event = _trace_event
    answer = RuntimeAnswer(answer="ok", route=RouteInfo())

    engine._maybe_record_harness_shadow_evaluation(
        request=request,
        state=state,
        runtime_answer=answer,
    )
    assert "harness_shadow_eval_recorded" in traces


def test_governance_bridge_records_shadow_observation() -> None:
    registry = InMemoryOnlineEvaluationRegistry()
    bridge = RuntimeArchitectureGovernanceBridge(evaluation_registry=registry)
    obs = bridge.record_shadow_run_evaluation(
        run_id="run-2",
        agent_id="echo",
        scenario_id="harness.echo",
        passed=True,
        score=1.0,
    )
    assert obs.run_id == "run-2"
    assert obs.passed is True
    assert len(registry.list_observations()) == 1
