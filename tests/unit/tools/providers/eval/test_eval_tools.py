# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.architecture.online_evaluation_models import OnlineEvaluationMode
from intergrax.runtime.architecture.online_evaluation_registry import InMemoryOnlineEvaluationRegistry
from intergrax.tools.providers.eval.contracts import (
    EvalListObservationsInput,
    EvalRecordObservationInput,
    EvalSummarizeReleaseInput,
)
from intergrax.tools.providers.eval.service import (
    eval_list_observations,
    eval_record_observation,
    eval_summarize_release,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


def test_eval_record_and_list_observations() -> None:
    registry = InMemoryOnlineEvaluationRegistry()
    ctx = ToolWiringContext(evaluation_registry=registry)
    recorded = eval_record_observation(
        ctx,
        EvalRecordObservationInput(
            observation_id="obs-1",
            run_id="run-rel-1",
            agent_id="agent-a",
            mode=OnlineEvaluationMode.SHADOW.value,
            scenario_id="rel-2026.06.07-smoke",
            passed=True,
            score=0.9,
            candidate_profile_version_id="v1",
        ),
    )
    assert recorded.recorded is True
    listed = eval_list_observations(ctx, EvalListObservationsInput(limit=10))
    assert listed.total == 1
    assert listed.pass_rate == 1.0
    assert listed.average_score == pytest.approx(0.9)


def test_eval_summarize_release_matches_scenario_prefix() -> None:
    registry = InMemoryOnlineEvaluationRegistry()
    ctx = ToolWiringContext(evaluation_registry=registry)
    eval_record_observation(
        ctx,
        EvalRecordObservationInput(
            observation_id="obs-1",
            run_id="run-1",
            agent_id="agent-a",
            mode=OnlineEvaluationMode.ONLINE.value,
            scenario_id="rel-2026.06.07-smoke",
            passed=True,
            score=1.0,
            candidate_profile_version_id="v1",
        ),
    )
    eval_record_observation(
        ctx,
        EvalRecordObservationInput(
            observation_id="obs-2",
            run_id="run-2",
            agent_id="agent-a",
            mode=OnlineEvaluationMode.ONLINE.value,
            scenario_id="other",
            passed=False,
            score=0.2,
            candidate_profile_version_id="v1",
        ),
    )
    summary = eval_summarize_release(ctx, EvalSummarizeReleaseInput(release_id="rel-2026.06.07"))
    assert summary.observation_count == 1
    assert summary.passed_count == 1
    assert summary.failed_count == 0


def test_eval_registry_not_configured() -> None:
    with pytest.raises(RuntimeError, match="evaluation_registry_not_configured"):
        eval_list_observations(ToolWiringContext(), EvalListObservationsInput())
