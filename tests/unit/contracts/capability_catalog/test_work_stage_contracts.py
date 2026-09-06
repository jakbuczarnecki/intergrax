# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 8 work-stage contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    WorkStageCapabilityNeed,
)

pytestmark = pytest.mark.unit


def _scope() -> CapabilityDiscoveryScope:
    return CapabilityDiscoveryScope(
        organization_id="org-acme",
        tenant_id="tenant-a",
        application_id="app-research",
        mode=CapabilityDiscoveryScopeMode.ENTERPRISE,
    )


def test_work_stage_capability_need_requires_non_empty_references() -> None:
    with pytest.raises(ValidationError, match="work_reference"):
        WorkStageCapabilityNeed(
            work_reference="",
            stage_reference="stage.collect",
            goal_objective="resolve customer incident",
            stage_objective="collect evidence",
            discovery_query=CapabilityDiscoveryQuery(scope=_scope()),
        )


def test_work_stage_capability_need_distinguishes_goal_and_stage_objectives() -> None:
    need = WorkStageCapabilityNeed(
        work_reference="work.incident-42",
        stage_reference="stage.collect",
        goal_objective="resolve customer incident",
        stage_objective="collect evidence",
        discovery_query=CapabilityDiscoveryQuery(scope=_scope()),
    )
    assert need.goal_objective != need.stage_objective
    assert need.requests_capabilities is True


def test_work_stage_capability_need_empty_query_means_no_capability_request() -> None:
    need = WorkStageCapabilityNeed(
        work_reference="work.incident-42",
        stage_reference="stage.wait",
        goal_objective="resolve customer incident",
        stage_objective="await human approval",
    )
    assert need.discovery_query is None
    assert need.requests_capabilities is False


def test_work_stage_capability_need_is_immutable() -> None:
    need = WorkStageCapabilityNeed(
        work_reference="work.incident-42",
        stage_reference="stage.collect",
        goal_objective="resolve customer incident",
        stage_objective="collect evidence",
    )
    with pytest.raises(ValidationError):
        need.stage_reference = "stage.other"  # type: ignore[misc]
