# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-A / F-D1-02: activation and rollback re-check revocation of the target revision."""

from __future__ import annotations

import pytest
from intergrax.agent_distribution.admin_models import (
    AgentPlatformAdminGovernanceBlockedError,
    HandlePostCutoverFailureRequest,
    RollbackRuntimeRevisionRequest,
)
from intergrax.agent_distribution.errors import AgentPackageTrustError
from intergrax.agent_distribution.trust import AgentPackageTrustRevocationState
from tests.qualification.pdc_x.d1 import _d1_harness as h
from tests.qualification.pdc_x.d1_repair import _support as s

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REVOCATIONS = {
    "digest": AgentPackageTrustRevocationState(revoked_package_digests=frozenset({h.DIGESTS[s.V1]})),
    "publisher": AgentPackageTrustRevocationState(revoked_publisher_ids=frozenset({"publisher:acme"})),
    "evidence": AgentPackageTrustRevocationState(revoked_evidence_ids=frozenset({f"evidence:{s.V1}"})),
}


def _rollback(stack: h.D1Stack, *, principal=None):
    return stack.service.rollback_revision(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=RollbackRuntimeRevisionRequest(
            mutation_id="mut-rollback-repair",
            expected_current_traffic_revision_id="rev-2",
            expected_serving_pointer_revision=2,
        ),
        principal=principal or h.principal(),
    )


@pytest.mark.parametrize("kind", sorted(_REVOCATIONS))
def test_rollback_to_a_revoked_prior_revision_is_refused_and_serving_is_unchanged(kind: str) -> None:
    stack = s.stack_with_validated_revision()
    s.serving_two_revisions(stack)
    stack.revocation.state = _REVOCATIONS[kind]

    with pytest.raises(AgentPackageTrustError):
        _rollback(stack)

    serving = stack.service.inspect_serving(application_id=h.APP, application_environment_id=h.ENV)
    assert serving.traffic_serving_revision_id == "rev-2"
    assert serving.serving_pointer_revision == 2


def test_rollback_to_a_trusted_prior_revision_still_works() -> None:
    stack = s.stack_with_validated_revision()
    s.serving_two_revisions(stack)
    s.revoke_digest(stack, s.V11)  # revoking the serving release must not block the way back

    result = _rollback(stack)

    assert result.restored_revision_id == "rev-1"
    assert s.traffic_revision(stack) == "rev-1"


def test_rollback_target_without_a_trust_record_is_refused_fail_closed() -> None:
    stack = s.stack_with_validated_revision()
    s.serving_two_revisions(stack)
    s.drop_trust_record(stack, s.V1)

    with pytest.raises(AgentPackageTrustError):
        _rollback(stack)

    assert s.traffic_revision(stack) == "rev-2"


def test_foreign_tenant_gets_the_tenant_error_not_revocation_state() -> None:
    stack = s.stack_with_validated_revision()
    s.serving_two_revisions(stack)
    s.revoke_digest(stack, s.V1)

    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        _rollback(stack, principal=h.OTHER_TENANT_PRINCIPAL)

    assert s.traffic_revision(stack) == "rev-2"


def test_post_cutover_failure_rollback_to_a_revoked_prior_revision_is_refused() -> None:
    stack = s.stack_with_validated_revision()
    s.serving_two_revisions(stack)
    s.revoke_digest(stack, s.V1)

    with pytest.raises(AgentPackageTrustError):
        stack.service.handle_post_cutover_failure(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=HandlePostCutoverFailureRequest(
                mutation_id="mut-post-cutover-repair",
                recovery_mutation_id="mut-recovery-repair",
                runtime_revision_id="rev-2",
                failure_evidence_ref="evidence://post-cutover",
                attempt_rollback=True,
            ),
            principal=h.principal(),
        )

    assert s.traffic_revision(stack) == "rev-2"


def test_activation_of_a_revision_revoked_after_build_is_refused_before_any_deployment_work() -> None:
    stack = s.stack_with_validated_revision()
    s.revoke_digest(stack, s.V1)  # revoked between build and activation

    with pytest.raises(AgentPackageTrustError):
        h.activate_revision(stack, "rev-1")

    assert stack.adapter.prepare_count == {}
    assert stack.deployment_store.get_instance(h.APP, h.ENV, "rev-1") is None
    assert s.traffic_revision(stack) is None


def test_activation_of_an_unrevoked_revision_still_works() -> None:
    stack = s.stack_with_validated_revision()

    h.activate_revision(stack, "rev-1")

    assert s.traffic_revision(stack) == "rev-1"


def test_foreign_tenant_activation_is_denied_before_the_trust_check() -> None:
    stack = s.stack_with_validated_revision()
    s.revoke_digest(stack, s.V1)

    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        h.activate_revision(stack, "rev-1", principal_value=h.OTHER_TENANT_PRINCIPAL)


def test_public_revision_trust_check_denies_revoked_and_accepts_trusted_revisions() -> None:
    stack = s.stack_with_validated_revision()
    revision = stack.revision_store.get_revision("rev-1")
    assert revision is not None

    stack.service.assert_revision_trust_admissible(revision)  # trusted: no exception

    s.revoke_digest(stack, s.V1)
    with pytest.raises(AgentPackageTrustError):
        stack.service.assert_revision_trust_admissible(revision)
