# © Artur Czarnecki. All rights reserved.

"""Runtime/Governance root execution authority admission tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.runtime_execution_admission import (
    RootExecutionAuthorityAdmissionDisposition,
    RootExecutionAuthorityAdmissionRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.root_execution_authority_admission import (
    DenyingRootExecutionAuthorityAdmission,
    RootExecutionAuthorityAdmissionService,
    UnavailableRootExecutionAuthorityAdmission,
)

pytestmark = pytest.mark.unit


def _request(*, action: PolicyAction) -> RootExecutionAuthorityAdmissionRequest:
    return RootExecutionAuthorityAdmissionRequest(
        tenant_id="tenant-a",
        workspace_id="workspace-x",
        principal_id="principal-1",
        collaborative_authority_scopes=("workspace.read",),
            effective_authority_decision=EffectiveAuthorityDecision(
                decision=PolicyDecision(action=action, reason="test"),
            ),
    )


def test_allow_mints_scoped_trusted_authority() -> None:
    service = RootExecutionAuthorityAdmissionService()
    result = service.authorize(_request(action=PolicyAction.ALLOW))
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.ALLOWED
    assert result.trusted_parent_execution_authority is not None
    assert result.trusted_parent_execution_authority.permission_scopes == (
        "workspace.read",
    )


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (PolicyAction.DENY, RootExecutionAuthorityAdmissionDisposition.DENIED),
        (PolicyAction.REQUIRE_HUMAN, RootExecutionAuthorityAdmissionDisposition.REQUIRE_HUMAN),
        (PolicyAction.ESCALATE, RootExecutionAuthorityAdmissionDisposition.ESCALATE),
        (PolicyAction.MODIFY, RootExecutionAuthorityAdmissionDisposition.DENIED),
    ],
)
def test_non_allow_fail_closed(
    action: PolicyAction,
    expected: RootExecutionAuthorityAdmissionDisposition,
) -> None:
    service = RootExecutionAuthorityAdmissionService()
    result = service.authorize(_request(action=action))
    assert result.disposition is expected
    assert result.trusted_parent_execution_authority is None


def test_denying_adapter() -> None:
    service = DenyingRootExecutionAuthorityAdmission()
    result = service.authorize(_request(action=PolicyAction.ALLOW))
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.DENIED


def test_unavailable_adapter() -> None:
    service = UnavailableRootExecutionAuthorityAdmission()
    result = service.authorize(_request(action=PolicyAction.ALLOW))
    assert result.disposition is RootExecutionAuthorityAdmissionDisposition.UNAVAILABLE
