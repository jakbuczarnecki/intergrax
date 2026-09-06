# © Artur Czarnecki. All rights reserved.

"""AC-6 Phase 5 — production-path trust + lifecycle end-to-end closure."""

from __future__ import annotations

from datetime import timedelta

import pytest

from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    BindAgentRequest,
    BuildApplicationRevisionRequest,
    InstallAgentRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.ed25519_package_attestation_verifier import (
    Ed25519PackageAttestationVerifier,
)
from intergrax.agent_distribution.emergency_revocation_response import (
    AgentEmergencyRevocationRequest,
    EmergencyTrustResponseAction,
    EmergencyTrustResponseReasonCode,
)
from intergrax.agent_distribution.errors import AgentPackageTrustError
from intergrax.agent_distribution.installation import InstallationState
from intergrax.agent_distribution.package_attestation import (
    AgentPackageAttestationAlgorithm,
    AgentPackageAttestationVerificationOutcome,
    AgentPackageAttestationVerificationRequest,
    StaticPublisherVerificationKeyProvider,
    is_verified_signature_qualification_evidence,
)
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.agent_distribution.trust import (
    AgentPackageTrustOutcome,
    AgentPackageTrustPolicy,
    AgentPackageTrustReasonCode,
    AgentPackageTrustRevocationState,
)
from testing_support.ac6_trust_lifecycle_composition import (
    AC6_APP,
    AC6_ARTIFACT,
    AC6_BINDING,
    AC6_DIGEST_D1,
    AC6_DIGEST_D2,
    AC6_ENV,
    AC6_EVAL_FRESH,
    AC6_EVAL_STALE,
    AC6_FIXED_AT,
    AC6_META_REF,
    AC6_PUBLISHER_ID,
    AC6_QUALIFIED_AT,
    AC6_SLOT,
    Ac6AdminStack,
    ac6_admin_principal,
    ac6_evaluate_trust,
    ac6_package_identity,
    ac6_qualification,
    ac6_require_trust_record,
    build_ac6_admin_stack,
)
from testing_support.agent_package_attestation import build_test_attestation_keypair

pytestmark = [pytest.mark.integration, pytest.mark.gate]


def _install_request(
    *,
    installation_id: str,
    digest: str,
    trust_record: object,
) -> InstallAgentRequest:
    return InstallAgentRequest(
        mutation_id=f"mut-install-{installation_id}",
        installation_id=installation_id,
        installation_slot_id=AC6_SLOT,
        package_identity=ac6_package_identity(digest),
        artifact_store_ref=f"store://artifacts/{installation_id}",
        trust_record=trust_record,
        agent_project_metadata_ref=AC6_META_REF,
    )


def _bind_request() -> BindAgentRequest:
    return BindAgentRequest(
        mutation_id="mut-bind-ac6",
        application_binding_id=AC6_BINDING,
        logical_agent_id="researcher",
        installation_slot_id=AC6_SLOT,
    )


def _build_request(revision_id: str) -> BuildApplicationRevisionRequest:
    from intergrax.agent_distribution.dependency import RepositoryDependencyDeclaration
    from intergrax.agent_distribution.runtime_revision import MaterializationTopology

    return BuildApplicationRevisionRequest(
        mutation_id=f"mut-build-{revision_id}",
        runtime_revision_id=revision_id,
        application_release_id="rel-1",
        platform_version="0.1.0",
        python_version="3.12",
        source_context_root="/tmp/src",
        output_root="/tmp/out",
        application_source_root=f"applications/{AC6_APP}",
        materialization_topology=MaterializationTopology.OCI_IMAGE,
        repository_declaration=RepositoryDependencyDeclaration(
            application_release_id="rel-1",
            direct_dependencies=(),
        ),
        resolver_algorithm_id="intergrax.test-resolver",
        resolver_algorithm_version="1.0.0",
    )


def _activate_request(
    revision_id: str,
    *,
    expected_serving_pointer_revision: int = 0,
) -> ActivateRuntimeRevisionRequest:
    return ActivateRuntimeRevisionRequest(
        mutation_id=f"mut-activate-{revision_id}",
        runtime_revision_id=revision_id,
        artifact_locator="test://artifact",
        expected_artifact_digest=AC6_ARTIFACT,
        expected_serving_pointer_revision=expected_serving_pointer_revision,
    )


def _install_bind_enable_build_activate(
    stack: Ac6AdminStack,
    *,
    installation_id: str,
    digest: str,
    trust_record: object,
    revision_id: str,
    expected_serving_pointer_revision: int = 0,
) -> None:
    principal = ac6_admin_principal()
    stack.service.install_agent(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=_install_request(
            installation_id=installation_id,
            digest=digest,
            trust_record=trust_record,
        ),
        principal=principal,
    )
    stack.service.bind_agent(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=_bind_request(),
        principal=principal,
    )
    stack.service.enable_binding(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        application_binding_id=AC6_BINDING,
        request=SetAgentEnablementRequest(
            mutation_id="mut-enable-ac6",
            expected_revision=0,
        ),
        principal=principal,
    )
    stack.service.build_application_revision(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=_build_request(revision_id),
        principal=principal,
    )
    stack.service.activate_revision(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=_activate_request(
            revision_id,
            expected_serving_pointer_revision=expected_serving_pointer_revision,
        ),
        principal=principal,
    )


def test_ac6_signed_package_reaches_active_revision_through_canonical_trust_path() -> (
    None
):
    package = ac6_package_identity(AC6_DIGEST_D2)
    qualification = ac6_qualification(package)
    decision = ac6_evaluate_trust(
        build_ac6_admin_stack().coordinator,
        package,
        qualification=qualification,
    )
    assert decision.outcome is AgentPackageTrustOutcome.ALLOW
    assert decision.reason_code is AgentPackageTrustReasonCode.QUALIFIED
    trust_record = ac6_require_trust_record(decision)
    assert trust_record.package_digest == AC6_DIGEST_D2
    assert is_verified_signature_qualification_evidence(
        qualification.evidence[0],
        expected_package_digest=AC6_DIGEST_D2,
    )

    stack = build_ac6_admin_stack()
    _install_bind_enable_build_activate(
        stack,
        installation_id="inst-ac6-d2",
        digest=AC6_DIGEST_D2,
        trust_record=trust_record,
        revision_id="rev-ac6-n",
    )

    serving = stack.service.inspect_serving(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
    )
    revision = stack.service.inspect_revision(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        runtime_revision_id="rev-ac6-n",
    )
    installation = stack.service.inspect_installation(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        installation_id="inst-ac6-d2",
    )
    assert serving.traffic_serving_revision_id == "rev-ac6-n"
    assert revision.revision_state is RuntimeRevisionState.ACTIVE
    assert installation.installation_state is InstallationState.INSTALLED_ACTIVE


def test_ac6_invalid_signature_cannot_reach_installation() -> None:
    signed_package = ac6_package_identity(AC6_DIGEST_D1)
    target_package = ac6_package_identity(AC6_DIGEST_D2)
    qualification_for_d1 = ac6_qualification(signed_package)
    forged_signature_b64 = qualification_for_d1.evidence[0].signature_b64
    _, public_key_bytes = build_test_attestation_keypair()
    verifier = Ed25519PackageAttestationVerifier(
        key_provider=StaticPublisherVerificationKeyProvider(
            {("publisher:acme", "test-publisher-key-1"): public_key_bytes}
        )
    )
    verification = verifier.verify(
        AgentPackageAttestationVerificationRequest(
            package_identity=target_package,
            publisher_id=AC6_PUBLISHER_ID,
            attestation_id="attest-bad",
            key_id="test-publisher-key-1",
            algorithm=AgentPackageAttestationAlgorithm.ED25519,
            signature_b64=forged_signature_b64,
        )
    )
    assert verification.outcome is AgentPackageAttestationVerificationOutcome.INVALID
    assert not is_verified_signature_qualification_evidence(
        qualification_for_d1.evidence[0],
        expected_package_digest=AC6_DIGEST_D2,
    )

    stack = build_ac6_admin_stack()
    decision = ac6_evaluate_trust(
        stack.coordinator,
        target_package,
        qualification=qualification_for_d1,
    )
    assert decision.outcome is AgentPackageTrustOutcome.DENY
    assert decision.trust_record is None

    assert "inst-bad-sig" not in stack.state.installations
    assert AC6_BINDING not in stack.state.bindings
    assert not stack.state.revisions
    assert (
        stack.service.inspect_serving(
            application_id=AC6_APP,
            application_environment_id=AC6_ENV,
        ).traffic_serving_revision_id
        is None
    )


def test_ac6_revoked_digest_blocked_before_install() -> None:
    package = ac6_package_identity(AC6_DIGEST_D2)
    qualification = ac6_qualification(package)
    stack = build_ac6_admin_stack(
        revocation_state=AgentPackageTrustRevocationState(
            revoked_package_digests=frozenset({AC6_DIGEST_D2}),
        ),
    )
    decision = ac6_evaluate_trust(
        stack.coordinator,
        package,
        qualification=qualification,
        revocation_state=stack.revocation_state["state"],
    )
    assert decision.outcome is AgentPackageTrustOutcome.DENY
    assert decision.reason_code is AgentPackageTrustReasonCode.PACKAGE_DIGEST_REVOKED

    with pytest.raises(AgentPackageTrustError) as exc_info:
        stack.service.install_agent(
            application_id=AC6_APP,
            application_environment_id=AC6_ENV,
            request=_install_request(
                installation_id="inst-revoked",
                digest=AC6_DIGEST_D2,
                trust_record=ac6_require_trust_record(
                    ac6_evaluate_trust(
                        build_ac6_admin_stack().coordinator,
                        package,
                        qualification=qualification,
                    )
                ),
            ),
            principal=ac6_admin_principal(),
        )
    assert exc_info.value.reason_code == "package_digest_revoked"
    assert "inst-revoked" not in stack.state.installations
    assert not stack.state.revisions


def test_ac6_stale_qualification_blocks_new_runtime_revision_but_preserves_active_n() -> (
    None
):
    package = ac6_package_identity(AC6_DIGEST_D2)
    qualification = ac6_qualification(package)
    stack = build_ac6_admin_stack(
        evaluation_at=AC6_EVAL_FRESH,
        policy=AgentPackageTrustPolicy(max_qualification_age=timedelta(days=7)),
    )
    trust_record = ac6_require_trust_record(
        ac6_evaluate_trust(stack.coordinator, package, qualification=qualification)
    )
    _install_bind_enable_build_activate(
        stack,
        installation_id="inst-ac6-stale",
        digest=AC6_DIGEST_D2,
        trust_record=trust_record,
        revision_id="rev-ac6-active",
    )
    serving_before = stack.service.inspect_serving(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
    )
    assert serving_before.traffic_serving_revision_id == "rev-ac6-active"
    assert "inst-ac6-stale" in stack.state.installations

    stack.evaluation_times["at"] = AC6_EVAL_STALE
    with pytest.raises(AgentPackageTrustError) as exc_info:
        stack.service.build_application_revision(
            application_id=AC6_APP,
            application_environment_id=AC6_ENV,
            request=_build_request("rev-ac6-blocked"),
            principal=ac6_admin_principal(),
        )
    assert exc_info.value.reason_code == "qualification_expired"
    assert "rev-ac6-blocked" not in stack.state.revisions
    serving_after = stack.service.inspect_serving(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
    )
    assert serving_after.traffic_serving_revision_id == "rev-ac6-active"
    assert (
        serving_after.serving_pointer_revision
        == serving_before.serving_pointer_revision
    )


def _bootstrap_two_revision_stack() -> Ac6AdminStack:
    stack = build_ac6_admin_stack(evaluation_at=AC6_EVAL_FRESH)
    package_d1 = ac6_package_identity(AC6_DIGEST_D1)
    package_d2 = ac6_package_identity(AC6_DIGEST_D2, version="2.0.0")
    trust_d1 = ac6_require_trust_record(
        ac6_evaluate_trust(
            stack.coordinator,
            package_d1,
            qualification=ac6_qualification(package_d1),
        )
    )
    principal = ac6_admin_principal()
    stack.service.install_agent(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=_install_request(
            installation_id="inst-ac6-d1",
            digest=AC6_DIGEST_D1,
            trust_record=trust_d1,
        ),
        principal=principal,
    )
    stack.service.bind_agent(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=_bind_request(),
        principal=principal,
    )
    stack.service.enable_binding(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        application_binding_id=AC6_BINDING,
        request=SetAgentEnablementRequest(
            mutation_id="mut-enable-d1",
            expected_revision=0,
        ),
        principal=principal,
    )
    stack.service.build_application_revision(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=_build_request("rev-ac6-n1"),
        principal=principal,
    )
    stack.service.activate_revision(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=_activate_request("rev-ac6-n1", expected_serving_pointer_revision=0),
        principal=principal,
    )
    trust_d2 = ac6_require_trust_record(
        ac6_evaluate_trust(
            stack.coordinator,
            package_d2,
            qualification=ac6_qualification(package_d2),
        )
    )
    stack.service.install_agent(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=_install_request(
            installation_id="inst-ac6-d2",
            digest=AC6_DIGEST_D2,
            trust_record=trust_d2,
        ),
        principal=principal,
    )
    stack.service.build_application_revision(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=_build_request("rev-ac6-n2"),
        principal=principal,
    )
    stack.service.activate_revision(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=ActivateRuntimeRevisionRequest(
            mutation_id="mut-activate-rev-ac6-n2",
            runtime_revision_id="rev-ac6-n2",
            artifact_locator="test://artifact",
            expected_artifact_digest=AC6_ARTIFACT,
            expected_serving_pointer_revision=1,
            expected_prior_traffic_revision_id="rev-ac6-n1",
        ),
        principal=principal,
    )
    return stack


def test_ac6_active_digest_revocation_rolls_back_to_currently_trusted_prior_revision() -> (
    None
):
    stack = _bootstrap_two_revision_stack()
    serving_before = stack.service.inspect_serving(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
    )
    assert serving_before.traffic_serving_revision_id == "rev-ac6-n2"

    response = stack.service.respond_to_emergency_revocation(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=AgentEmergencyRevocationRequest(
            application_id=AC6_APP,
            application_environment_id=AC6_ENV,
            expected_current_traffic_revision_id="rev-ac6-n2",
            expected_serving_pointer_revision=serving_before.serving_pointer_revision,
            evaluated_at=AC6_FIXED_AT,
            revocation_state=AgentPackageTrustRevocationState(
                revoked_package_digests=frozenset({AC6_DIGEST_D2}),
            ),
        ),
    )
    serving_after = stack.service.inspect_serving(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
    )
    assert response.action is EmergencyTrustResponseAction.ROLLBACK
    assert response.response_reason_code is (
        EmergencyTrustResponseReasonCode.SAFE_ROLLBACK_COMPLETED
    )
    assert serving_after.traffic_serving_revision_id == "rev-ac6-n1"
    restored = stack.service.inspect_revision(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        runtime_revision_id="rev-ac6-n1",
    )
    superseded = stack.service.inspect_revision(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        runtime_revision_id="rev-ac6-n2",
    )
    assert restored.revision_state is RuntimeRevisionState.ACTIVE
    assert superseded.revision_state is RuntimeRevisionState.SUPERSEDED


def test_ac6_revoked_prior_revision_is_not_emergency_rollback_target() -> None:
    stack = _bootstrap_two_revision_stack()
    response = stack.service.respond_to_emergency_revocation(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=AgentEmergencyRevocationRequest(
            application_id=AC6_APP,
            application_environment_id=AC6_ENV,
            evaluated_at=AC6_FIXED_AT,
            revocation_state=AgentPackageTrustRevocationState(
                revoked_package_digests=frozenset({AC6_DIGEST_D1, AC6_DIGEST_D2}),
            ),
        ),
    )
    serving = stack.service.inspect_serving(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
    )
    assert response.action is EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET
    assert response.response_reason_code is (
        EmergencyTrustResponseReasonCode.ROLLBACK_TARGET_UNTRUSTED
    )
    assert serving.traffic_serving_revision_id == "rev-ac6-n2"


def test_ac6_qualification_expiry_alone_does_not_trigger_emergency_rollback() -> None:
    stack = build_ac6_admin_stack(
        evaluation_at=AC6_EVAL_FRESH,
        policy=AgentPackageTrustPolicy(max_qualification_age=timedelta(days=7)),
    )
    package = ac6_package_identity(AC6_DIGEST_D2)
    trust_record = ac6_require_trust_record(
        ac6_evaluate_trust(
            stack.coordinator,
            package,
            qualification=ac6_qualification(
                package,
                qualified_at=AC6_QUALIFIED_AT,
            ),
        )
    )
    _install_bind_enable_build_activate(
        stack,
        installation_id="inst-ac6-expiry",
        digest=AC6_DIGEST_D2,
        trust_record=trust_record,
        revision_id="rev-ac6-expiry",
    )
    stack.evaluation_times["at"] = AC6_EVAL_STALE
    response = stack.service.respond_to_emergency_revocation(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=AgentEmergencyRevocationRequest(
            application_id=AC6_APP,
            application_environment_id=AC6_ENV,
            evaluated_at=AC6_EVAL_STALE,
            revocation_state=AgentPackageTrustRevocationState(),
            trust_policy=AgentPackageTrustPolicy(
                max_qualification_age=timedelta(days=7)
            ),
        ),
    )
    serving = stack.service.inspect_serving(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
    )
    assert response.action is EmergencyTrustResponseAction.NO_ACTION
    assert response.response_reason_code is (
        EmergencyTrustResponseReasonCode.NO_ACTIVE_REVOCATION
    )
    assert serving.traffic_serving_revision_id == "rev-ac6-expiry"


def test_ac6_stale_emergency_request_cannot_override_newer_activation() -> None:
    stack = _bootstrap_two_revision_stack()
    stale_request = AgentEmergencyRevocationRequest(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        expected_current_traffic_revision_id="rev-ac6-n1",
        expected_serving_pointer_revision=0,
        evaluated_at=AC6_FIXED_AT,
        revocation_state=AgentPackageTrustRevocationState(
            revoked_package_digests=frozenset({AC6_DIGEST_D2}),
        ),
    )
    response = stack.service.respond_to_emergency_revocation(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
        request=stale_request,
    )
    serving = stack.service.inspect_serving(
        application_id=AC6_APP,
        application_environment_id=AC6_ENV,
    )
    assert response.action is EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET
    assert response.response_reason_code is (
        EmergencyTrustResponseReasonCode.SERVING_POINTER_MISMATCH
    )
    assert serving.traffic_serving_revision_id == "rev-ac6-n2"
