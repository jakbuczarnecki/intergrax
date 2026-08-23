# © Artur Czarnecki. All rights reserved.

"""CP13 — typed core fields on control-plane mutation contracts."""

from __future__ import annotations

import inspect

import pytest

from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationResult,
    ControlPlaneMutationAuthorizationScope,
    ControlPlaneMutationRequest,
    ControlPlaneMutationRisk,
    GovernanceEvaluationPoint,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType

pytestmark = pytest.mark.unit


def test_cp13_authority_critical_fields_are_typed_not_dict_bags() -> None:
    for model in (
        ControlPlaneMutationRequest,
        ControlPlaneMutationAuthorizationEvidence,
        ControlPlaneMutationAuthorizationScope,
        ControlPlaneMutationAuthorizationResult,
    ):
        hints = model.model_fields
        for field in hints.values():
            annotation = field.annotation
            assert annotation is not None
            assert "dict[str, Any]" not in str(annotation)
            assert "Mapping[str, Any]" not in str(annotation)

    request_fields = set(ControlPlaneMutationRequest.model_fields)
    assert "principal" in request_fields
    assert "mutation_id" in request_fields
    assert "resource_type" in request_fields
    assert "resource_id" in request_fields
    assert "current_revision" in request_fields
    assert "target_revision" in request_fields
    assert "risk_classification" in request_fields
    assert "context" not in request_fields
    assert "metadata" not in request_fields


def test_cp13_governance_evaluation_point_is_typed_enum() -> None:
    assert GovernanceEvaluationPoint.CONTROL_PLANE_MUTATION.value == "control_plane_mutation"
    assert inspect.isclass(GovernanceEvaluationPoint)


def test_cp13_risk_is_explicit_enum_not_inferred() -> None:
    principal = RequestIdentity(
        tenant_id="tenant-a",
        user_id="user-1",
        principal_type=PrincipalType.USER,
        auth_subject="user-1",
    )
    request = ControlPlaneMutationRequest(
        mutation_id="mut-1",
        mutation_type="activate_profile",
        principal=principal,
        resource_scope="workspace-a",
        resource_type="agent_distribution_activation",
        resource_id="profile-123",
        current_revision="4",
        target_revision="5",
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )
    assert request.risk_classification is ControlPlaneMutationRisk.HIGH
