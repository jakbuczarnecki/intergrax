# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.architecture.online_evaluation import (
    OnlineEvaluationBatch,
    OnlineEvaluationMode,
    append_online_evaluation_to_trend,
    record_shadow_observation,
)
from intergrax.runtime.architecture.online_evaluation_registry import InMemoryOnlineEvaluationRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_shadow_observation_appends_to_trend() -> None:
    registry = InMemoryOnlineEvaluationRegistry()
    obs = record_shadow_observation(
        run_id="run-1",
        agent_id="echo",
        scenario_id="harness.smoke",
        passed=True,
        score=0.9,
        registry=registry,
    )
    batch = OnlineEvaluationBatch(release_id="2026.06.02-rc1", observations=[obs])
    snapshot, comparisons = append_online_evaluation_to_trend(
        existing_snapshots=[],
        batch=batch,
    )
    assert snapshot.release_id == "2026.06.02-rc1"
    assert len(snapshot.automated_report.records) == 1
    assert snapshot.automated_report.records[0].final_passed is True
    assert comparisons == []
    assert obs.mode == OnlineEvaluationMode.SHADOW
