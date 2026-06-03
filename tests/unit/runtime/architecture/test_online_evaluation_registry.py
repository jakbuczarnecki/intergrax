# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.architecture.online_evaluation import record_shadow_observation
from intergrax.runtime.architecture.online_evaluation_registry import (
    FileOnlineEvaluationRegistry,
    InMemoryOnlineEvaluationRegistry,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_file_registry_persists_shadow_observation(tmp_path: Path) -> None:
    registry = FileOnlineEvaluationRegistry(tmp_path / "observations.json")
    record_shadow_observation(
        run_id="run-a",
        agent_id="echo",
        scenario_id="harness.echo",
        passed=True,
        score=1.0,
        registry=registry,
    )
    stored = registry.list_observations()
    assert len(stored) == 1
    assert stored[0].run_id == "run-a"
    assert stored[0].scenario_id == "harness.echo"


def test_in_memory_registry_append() -> None:
    registry = InMemoryOnlineEvaluationRegistry()
    obs = record_shadow_observation(
        run_id="run-b",
        agent_id="signoff_probe",
        scenario_id="harness.smoke",
        passed=False,
        score=0.2,
        registry=registry,
    )
    assert registry.list_observations() == [obs]
