# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-E (F-D5-01): writers stamp the tenant they hold; nothing drops it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from intergrax.runtime.architecture.online_evaluation import record_shadow_observation
from intergrax.runtime.architecture.online_evaluation_registry import (
    FileOnlineEvaluationRegistry,
    OnlineEvaluationTenantBoundaryError,
    TenantScopedOnlineEvaluationRegistry,
)
from intergrax.runtime.architecture.user_satisfaction import (
    SatisfactionSignal,
    UserSatisfactionEvent,
    record_user_satisfaction,
)

pytestmark = pytest.mark.unit


def _event(tenant: str) -> UserSatisfactionEvent:
    return UserSatisfactionEvent(
        tenant_id=tenant,
        task_id="task-1",
        run_id="run-1",
        signal=SatisfactionSignal.THUMBS_UP,
        score=9.0,
    )


def _registry(tmp_path: Path) -> FileOnlineEvaluationRegistry:
    return FileOnlineEvaluationRegistry(tmp_path / "observations.json")


def test_user_satisfaction_keeps_the_event_tenant(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    record_user_satisfaction(_event("tenant-a"), registry=registry)
    assert len(registry.list_observations_for_tenant("tenant-a")) == 1
    assert registry.list_observations_for_tenant("tenant-b") == []
    assert registry.list_observations() == []
    assert TenantScopedOnlineEvaluationRegistry(registry, "tenant-b").list_observations() == []


@pytest.mark.parametrize("blank", ["", "   "])
def test_user_satisfaction_with_blank_tenant_fails_closed(tmp_path: Path, blank: str) -> None:
    registry = _registry(tmp_path)
    with pytest.raises(ValueError, match="tenant_id"):
        record_user_satisfaction(_event(blank), registry=registry)
    assert registry.list_observations() == []
    assert not (tmp_path / "observations.json").exists()


def test_user_satisfaction_into_a_view_of_another_tenant_is_denied(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    view: Any = TenantScopedOnlineEvaluationRegistry(registry, "tenant-b")
    with pytest.raises(OnlineEvaluationTenantBoundaryError):
        record_user_satisfaction(_event("tenant-a"), registry=view)
    assert registry.list_observations_for_tenant("tenant-a") == []
    assert registry.list_observations_for_tenant("tenant-b") == []


def test_user_satisfaction_into_a_view_of_the_same_tenant_is_stored(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    view: Any = TenantScopedOnlineEvaluationRegistry(registry, "tenant-a")
    record_user_satisfaction(_event("tenant-a"), registry=view)
    assert len(registry.list_observations_for_tenant("tenant-a")) == 1


def _shadow(registry: Any, **kwargs: Any) -> None:
    record_shadow_observation(
        run_id="run-1",
        agent_id="agent:research",
        scenario_id="s-math",
        passed=True,
        score=1.0,
        registry=registry,
        **kwargs,
    )


def test_shadow_observation_with_tenant_is_stamped(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _shadow(registry, tenant_id="tenant-a")
    assert len(registry.list_observations_for_tenant("tenant-a")) == 1
    assert registry.list_observations() == []


def test_shadow_observation_through_a_view_of_the_same_tenant(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    view = TenantScopedOnlineEvaluationRegistry(registry, "tenant-a")
    _shadow(view, tenant_id="tenant-a")
    _shadow(view)
    assert len(registry.list_observations_for_tenant("tenant-a")) == 2


def test_shadow_observation_with_conflicting_tenant_is_denied(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    view = TenantScopedOnlineEvaluationRegistry(registry, "tenant-a")
    with pytest.raises(OnlineEvaluationTenantBoundaryError):
        _shadow(view, tenant_id="tenant-b")
    assert registry.list_observations_for_tenant("tenant-a") == []
    assert registry.list_observations_for_tenant("tenant-b") == []


def test_shadow_observation_with_blank_tenant_fails_closed(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    with pytest.raises(ValueError, match="tenant_id"):
        _shadow(registry, tenant_id="  ")
    assert registry.list_observations() == []


def test_shadow_observation_without_tenant_stays_untenanted(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _shadow(registry)
    assert len(registry.list_observations()) == 1
