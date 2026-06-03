# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.architecture.online_evaluation import record_shadow_observation
from intergrax.runtime.architecture.online_evaluation_registry import InMemoryOnlineEvaluationRegistry
from intergrax.runtime.architecture.online_evaluation_trend import (
    export_shadow_evaluation_trend,
    load_evaluation_release_snapshots,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_export_shadow_evaluation_trend_builds_comparisons(tmp_path: Path) -> None:
    registry = InMemoryOnlineEvaluationRegistry()
    snapshots_path = tmp_path / "snapshots.json"

    record_shadow_observation(
        run_id="run-1",
        agent_id="echo",
        scenario_id="harness.a",
        passed=True,
        score=0.9,
        registry=registry,
    )
    first = export_shadow_evaluation_trend(
        "2026.06.01-rc1",
        registry=registry,
        snapshots_path=snapshots_path,
        clear_registry_after_export=True,
    )
    assert len(first.snapshots) == 1
    assert first.comparisons == []

    record_shadow_observation(
        run_id="run-2",
        agent_id="echo",
        scenario_id="harness.b",
        passed=False,
        score=0.3,
        registry=registry,
    )
    second = export_shadow_evaluation_trend(
        "2026.06.02-rc1",
        registry=registry,
        snapshots_path=snapshots_path,
        clear_registry_after_export=True,
    )
    assert len(second.snapshots) == 2
    assert len(second.comparisons) == 1
    assert registry.list_observations() == []
    assert len(load_evaluation_release_snapshots(snapshots_path)) == 2
