# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-E (F-D5-01, F-D5-12): the legacy registry surface sees the untenanted namespace only."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.runtime.architecture.online_evaluation_models import (
    OnlineEvaluationMode,
    OnlineEvaluationObservation,
)
from intergrax.runtime.architecture.online_evaluation_registry import (
    FileOnlineEvaluationRegistry,
    OnlineEvaluationTenantBoundaryError,
)

pytestmark = pytest.mark.unit


def _obs(observation_id: str, run_id: str = "run-1") -> OnlineEvaluationObservation:
    return OnlineEvaluationObservation(
        observation_id=observation_id,
        run_id=run_id,
        agent_id="agent:research",
        mode=OnlineEvaluationMode.SHADOW,
        scenario_id="s-math",
        passed=True,
        score=1.0,
    )


def _registry(tmp_path: Path) -> FileOnlineEvaluationRegistry:
    return FileOnlineEvaluationRegistry(tmp_path / "observations.json")


def test_legacy_list_never_returns_tenanted_rows(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.append(_obs("legacy"))
    registry.append_for_tenant(_obs("a-1"), "tenant-a")
    registry.append_for_tenant(_obs("b-1"), "tenant-b")
    assert [o.observation_id for o in registry.list_observations()] == ["legacy"]
    assert [o.observation_id for o in registry.list_observations_for_tenant("tenant-a")] == ["a-1"]
    assert [o.observation_id for o in registry.list_observations_for_tenant("tenant-b")] == ["b-1"]


def test_legacy_clear_leaves_tenanted_rows(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.append(_obs("legacy"))
    registry.append_for_tenant(_obs("a-1"), "tenant-a")
    registry.clear()
    assert registry.list_observations() == []
    assert [o.observation_id for o in registry.list_observations_for_tenant("tenant-a")] == ["a-1"]


def test_untenanted_legacy_behaviour_is_unchanged(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.append(_obs("o1"))
    registry.append(_obs("o2"))
    assert [o.observation_id for o in registry.list_observations()] == ["o1", "o2"]
    registry.clear()
    assert registry.list_observations() == []


def test_same_observation_id_in_two_tenants_resolves_per_tenant(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.append_for_tenant(_obs("shadow:run-1:s", run_id="run-a"), "tenant-a")
    registry.append_for_tenant(_obs("shadow:run-1:s", run_id="run-b"), "tenant-b")
    assert registry.get_observation("shadow:run-1:s", tenant_id="tenant-a").run_id == "run-a"
    assert registry.get_observation("shadow:run-1:s", tenant_id="tenant-b").run_id == "run-b"


def test_id_held_only_by_another_tenant_is_still_denied(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.append_for_tenant(_obs("only-a"), "tenant-a")
    with pytest.raises(OnlineEvaluationTenantBoundaryError):
        registry.get_observation("only-a", tenant_id="tenant-b")
    with pytest.raises(KeyError):
        registry.get_observation("nope", tenant_id="tenant-b")
    with pytest.raises(ValueError, match="tenant_id"):
        registry.get_observation("only-a", tenant_id=" ")
