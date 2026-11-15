# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-A / F-D1-03 (authorization part): the emergency revocation entry point is governed."""

from __future__ import annotations

import inspect

import pytest
from intergrax.agent_distribution.admin_models import AgentPlatformAdminGovernanceBlockedError
from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.emergency_revocation_response import (
    EmergencyTrustResponseAction,
)
from tests.qualification.pdc_x.d1 import _d1_harness as h
from tests.qualification.pdc_x.d1_repair import _support as s

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _revoked_serving_stack() -> h.D1Stack:
    stack = s.stack_with_validated_revision()
    s.serving_two_revisions(stack)
    s.revoke_digest(stack, s.V11)  # the serving release is revoked; v1.0.0 is a trusted way back
    return stack


def test_entry_point_requires_a_principal_and_takes_no_caller_snapshot() -> None:
    parameters = inspect.signature(AgentPlatformAdminService.respond_to_emergency_revocation).parameters
    assert "principal" in parameters
    assert parameters["principal"].default is inspect.Parameter.empty
    for forbidden in ("request", "revocation_state", "trust_policy", "evaluated_at"):
        assert forbidden not in parameters


def test_foreign_tenant_principal_cannot_force_a_rollback() -> None:
    stack = _revoked_serving_stack()

    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as denied:
        stack.service.respond_to_emergency_revocation(  # type: ignore[call-arg]
            application_id=h.APP,
            application_environment_id=h.ENV,
            principal=h.OTHER_TENANT_PRINCIPAL,
        )

    assert denied.value.blocker_code == "AP-11_BLOCKED_BY_TENANT_AUTHORITY"
    assert s.traffic_revision(stack) == "rev-2"


def test_policy_deny_blocks_the_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _revoked_serving_stack()
    monkeypatch.setattr(stack.service, "_mutation_authorization_boundary", s.deny_mutation_boundary())

    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as denied:
        stack.service.respond_to_emergency_revocation(  # type: ignore[call-arg]
            application_id=h.APP,
            application_environment_id=h.ENV,
            principal=h.principal(),
        )

    assert denied.value.blocker_code == "AP-11_BLOCKED_BY_GOVERNANCE_DENY"
    assert s.traffic_revision(stack) == "rev-2"


def test_authorized_principal_rolls_back_using_only_the_injected_revocation_source() -> None:
    stack = _revoked_serving_stack()

    response = stack.service.respond_to_emergency_revocation(  # type: ignore[call-arg]
        application_id=h.APP,
        application_environment_id=h.ENV,
        principal=h.principal(),
    )

    assert response.action is EmergencyTrustResponseAction.ROLLBACK
    assert s.traffic_revision(stack) == "rev-1"


def test_nothing_revoked_means_no_action_even_for_an_authorized_principal() -> None:
    stack = s.stack_with_validated_revision()
    s.serving_two_revisions(stack)

    response = stack.service.respond_to_emergency_revocation(  # type: ignore[call-arg]
        application_id=h.APP,
        application_environment_id=h.ENV,
        principal=h.principal(),
    )

    assert response.action is EmergencyTrustResponseAction.NO_ACTION
    assert s.traffic_revision(stack) == "rev-2"
