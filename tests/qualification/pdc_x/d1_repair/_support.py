# © Artur Czarnecki. All rights reserved.

"""Shared scenario helpers for the T-IGX-041-A (PDC-X B-P7-D1) repair tests.

Builds on the B-D1 harness (``tests/qualification/pdc_x/d1/_d1_harness.py``): real
``AgentPlatformAdminService`` over in-memory stores, no mocks of the unit under test.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.admin_models import AgentUpgradeOutcome
from intergrax.agent_distribution.installation import AgentInstallationRecord
from intergrax.agent_distribution.trust import AgentPackageTrustRevocationState
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from tests.qualification.pdc_x.d1 import _d1_harness as h

V1, V11 = "1.0.0", "1.1.0"


@dataclass
class _DenyEvaluator:
    def evaluate(self, request: object) -> PolicyDecision:
        del request
        return PolicyDecision(action=PolicyAction.DENY, reason="test_deny")


def deny_mutation_boundary() -> ControlPlaneMutationAuthorizationBoundary:
    return ControlPlaneMutationAuthorizationBoundary(evaluator=_DenyEvaluator())


def stack_with_validated_revision() -> h.D1Stack:
    """v1.0.0 installed, bound, enabled; rev-1 built (not activated)."""
    stack = h.build_stack()
    h.install_bind_enable(stack, h.release(V1))
    h.build_revision(stack, "rev-1")
    return stack


def two_revisions(stack: h.D1Stack) -> None:
    """rev-1 (v1.0.0) active; slot upgraded to v1.1.0 with a built rev-2."""
    h.activate_revision(stack, "rev-1")
    candidate = h.release(V11, contract_version="1.1.0")
    stack.register(candidate)
    upgraded = stack.service.upgrade_agent(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=candidate.upgrade_request(expected_active=h.installation_id(V1)),
        principal=h.principal(),
    )
    assert upgraded.outcome is AgentUpgradeOutcome.UPGRADED
    h.build_revision(stack, "rev-2")


def serving_two_revisions(stack: h.D1Stack) -> None:
    """rev-2 (v1.1.0) serving with rev-1 (v1.0.0) as the prior traffic revision."""
    two_revisions(stack)
    h.activate_revision(stack, "rev-2", expected_prior="rev-1", expected_pointer=1)


def revoke_digest(stack: h.D1Stack, version: str) -> None:
    stack.revocation.state = AgentPackageTrustRevocationState(revoked_package_digests=frozenset({h.DIGESTS[version]}))


def traffic_revision(stack: h.D1Stack) -> str | None:
    return stack.service.inspect_serving(
        application_id=h.APP, application_environment_id=h.ENV
    ).traffic_serving_revision_id


def drop_trust_record(stack: h.D1Stack, version: str) -> None:
    """Persist the installation of ``version`` without its trust record (fail-closed probe)."""
    record: AgentInstallationRecord | None = stack.installation_store.get_installation(h.installation_id(version))
    assert record is not None
    stack.installation_store.persist_installation(record.model_copy(update={"trust_record": None}))
