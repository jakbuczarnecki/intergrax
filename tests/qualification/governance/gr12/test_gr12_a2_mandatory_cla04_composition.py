# © Artur Czarnecki. All rights reserved.

"""GR-12-A2 — mandatory CLA-04 composition and live wiring gates."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi import FastAPI

from intergrax.applications._shared.control_plane_composition import (
    ControlPlaneCompositionError,
)
from intergrax.applications._shared.harness_control_plane_governance_wiring import (
    build_harness_control_plane_governance,
    resolve_harness_task_control_mutation_boundary,
)
from intergrax.applications._shared.production_capacity_governance_wiring import (
    ProductionCapacityGovernance,
    build_production_capacity_governance,
)
from intergrax.applications._shared.production_capacity_wiring import (
    resolve_production_capacity_wiring,
)
from intergrax.applications._shared.task_control_wiring import wire_harness_task_control
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.governance.control_plane_mutation_approval import (
    ApprovalConsumingControlPlaneMutationEvaluator,
)
from tests.qualification.governance.gr12.catalog import (
    GR12_CONTROL_PLANE_SURFACES,
    Gr12CoverageStatus,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass
class _RecordingEvaluator:
    decision: PolicyDecision = field(
        default_factory=lambda: PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="external_allow",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="external.test_allow",
            decision_id="dec-external",
        )
    )
    calls: list[ControlPlaneMutationRequest] = field(default_factory=list)

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        return self.decision


def test_gr12_a2_host_boundary_catalog_wired_not_qualified() -> None:
    row = next(r for r in GR12_CONTROL_PLANE_SURFACES if r.path_id == "CP-HOST-BOUNDARY-OPTIONAL")
    assert row.coverage is Gr12CoverageStatus.WIRED_NOT_QUALIFIED


def test_gr12_a2_ecp_boundary_catalog_wired_not_qualified() -> None:
    row = next(r for r in GR12_CONTROL_PLANE_SURFACES if r.path_id == "CP-ECP-BOUNDARY-OPTIONAL")
    assert row.coverage is Gr12CoverageStatus.WIRED_NOT_QUALIFIED


def test_gr12_a2_product_task_control_wiring_requires_canonical_boundary() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="tenant-a2")
    app = FastAPI()
    wire_harness_task_control(
        app,
        enabled=True,
        host_execution=object(),
        env=env,
    )
    boundary = resolve_harness_task_control_mutation_boundary(
        build_harness_control_plane_governance(env),
    )
    assert boundary is not None


def test_gr12_a2_external_evaluator_injected_without_domain_changes() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="tenant-a2")
    external = ControlPlaneMutationAuthorizationBoundary(evaluator=_RecordingEvaluator())
    governance = build_harness_control_plane_governance(
        env,
        mutation_authorization_boundary=external,
    )
    boundary = resolve_harness_task_control_mutation_boundary(governance)
    assert boundary is not None
    evaluator = boundary.evaluator
    assert isinstance(evaluator, ApprovalConsumingControlPlaneMutationEvaluator)
    assert evaluator.inner is external.evaluator


def test_gr12_a2_ecp_product_without_authority_fails_at_wiring() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    governance = build_production_capacity_governance(env)
    assert governance.mutation_authorization_boundary is None
    with pytest.raises(ControlPlaneCompositionError) as exc_info:
        resolve_production_capacity_wiring(env, governance=governance)
    assert exc_info.value.blocker_code == "ECP_BLOCKED_MISSING_BOUNDARY"


def test_gr12_a2_ecp_enabled_without_boundary_fails_at_wiring() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    base = build_production_capacity_governance(env)
    stripped = ProductionCapacityGovernance(
        principal=base.principal,
        mutation_authorization_boundary=None,
        tenant_resolver=base.tenant_resolver,
        tenant_id=base.tenant_id,
    )
    with pytest.raises(ControlPlaneCompositionError) as exc_info:
        resolve_production_capacity_wiring(env, governance=stripped)
    assert exc_info.value.blocker_code == "ECP_BLOCKED_MISSING_BOUNDARY"


def test_gr12_a2_non_product_task_control_may_wire_without_composition_boundary() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="dev-a2")
    app = FastAPI()
    wire_harness_task_control(
        app,
        enabled=True,
        host_execution=object(),
        env=env,
    )
