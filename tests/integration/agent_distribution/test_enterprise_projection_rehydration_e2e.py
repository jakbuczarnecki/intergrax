# © Artur Czarnecki. All rights reserved.

"""Enterprise durable projection rehydration E2E proofs (F3/F4/F5 + corruption)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    InstallAgentRequest,
)
from intergrax.agent_distribution.emergency_revocation_response import (
    AgentEmergencyRevocationRequest,
    EmergencyTrustResponseAction,
    EmergencyTrustResponseReasonCode,
)
from intergrax.agent_distribution.errors import RuntimeActivationConflict
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.package_trust import AgentPackageTrustError
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentPackageTrustRevocationState,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    build_production_registry_projection_input_bundle_for_revision,
)
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.core.qualification import QualificationStatus
from testing_support.agent_platform_admin_harness import admin_test_principal
from testing_support.canonical_agent_lifecycle_composition import (
    default_stage15_proof_config,
)
from testing_support.enterprise_agent_lifecycle_composition import (
    EnterpriseAgentLifecycleProofStack,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_REVOKED_DIGEST = "sha256:" + ("9" * 64)
_DIGEST_N1 = "sha256:" + ("b" * 64)


@pytest.fixture(autouse=True)
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def test_enterprise_durable_f4_failed_activation_preserves_serving_after_reopen(
    tmp_path: Path,
) -> None:
    config = default_stage15_proof_config()
    db_path = tmp_path / "f4.db"
    shared_root = tmp_path / "shared-artifacts"
    stack = EnterpriseAgentLifecycleProofStack.build(shared_root, db_path=db_path)
    result = stack.run_happy_path()
    serving_revision = result.runtime_revision_id
    expected_answer = result.execution_answer
    built_n1 = stack.build_revision(revision_id="rev-enterprise-f4-n1")
    serving_before = stack.admin.inspect_serving(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    input_bundle = build_production_registry_projection_input_bundle_for_revision(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
        runtime_revision_id=built_n1.runtime_revision_id,
        manifest=stack.canonical.manifest,
        build_context=ApplicationBuildContext.for_manifest(stack.canonical.manifest),
        authority=stack.durable_runtime.registry_projection_authority,
    )
    activation_request = ActivateRuntimeRevisionRequest(
        mutation_id="mut-enterprise-f4-fail",
        runtime_revision_id=built_n1.runtime_revision_id,
        artifact_locator=built_n1.artifact_locator,
        expected_artifact_digest=built_n1.materialization_artifact_digest,
        expected_serving_pointer_revision=999,
        expected_prior_traffic_revision_id=serving_before.traffic_serving_revision_id,
    )
    with pytest.raises(RuntimeActivationConflict):
        stack.launcher.deploy_and_activate(
            projection_input=input_bundle,
            activation_request=activation_request,
            principal=stack.canonical.governance.principal,
            admission_mutation_id=reference_admission_mutation_id(
                built_n1.runtime_revision_id,
            ),
        )
    serving_after = stack.admin.inspect_serving(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    assert serving_after.traffic_serving_revision_id == serving_revision
    del stack
    reopened = EnterpriseAgentLifecycleProofStack.reopen(shared_root, db_path, config)
    _, answer = asyncio.run(reopened.execute_canonical())
    assert answer == expected_answer


def test_enterprise_durable_f3_revoked_install_rejected_serving_unchanged_after_reopen(
    tmp_path: Path,
) -> None:
    config = default_stage15_proof_config()
    db_path = tmp_path / "f3.db"
    shared_root = tmp_path / "shared-artifacts"
    stack = EnterpriseAgentLifecycleProofStack.build(shared_root, db_path=db_path)
    result = stack.run_happy_path()
    serving_before = stack.admin.inspect_serving(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    trust = getattr(stack.admin, "_package_trust_coordinator")
    original = trust.assert_install_admission

    def _deny_revoked(**kwargs):
        package = kwargs.get("package_identity")
        if package is not None and package.package_digest == _REVOKED_DIGEST:
            raise AgentPackageTrustError(
                "package digest revoked",
                reason_code="package_digest_revoked",
            )
        return original(**kwargs)

    trust.assert_install_admission = _deny_revoked  # type: ignore[method-assign]
    with pytest.raises(AgentPackageTrustError) as exc_info:
        stack.admin.install_agent(
            application_id=config.application_id,
            application_environment_id=config.environment_id,
            request=InstallAgentRequest(
                mutation_id="mut-enterprise-f3-revoked",
                installation_id="inst-revoked-f3",
                installation_slot_id="slot-revoked-f3",
                package_identity=AgentPackageIdentity(
                    distribution_package_id=config.distribution_package_id,
                    package_version="9.9.9",
                    package_digest=_REVOKED_DIGEST,
                ),
                artifact_store_ref="store://artifacts/revoked-f3",
                trust_record=AgentInstallationTrustRecord(
                    qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
                    package_digest=_REVOKED_DIGEST,
                    publisher_identity_ref="publisher:stage15",
                    source_provider_id=config.catalog_source_id,
                    trust_evidence_refs=(
                        AgentTrustEvidenceRef(
                            evidence_id="evidence:revoked-f3",
                            kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                        ),
                    ),
                ),
                agent_project_metadata_ref=config.metadata_ref,
            ),
            principal=admin_test_principal(),
        )
    assert exc_info.value.reason_code == "package_digest_revoked"
    serving_after = stack.admin.inspect_serving(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    assert (
        serving_after.traffic_serving_revision_id
        == serving_before.traffic_serving_revision_id
    )
    del stack
    reopened = EnterpriseAgentLifecycleProofStack.reopen(shared_root, db_path, config)
    _, answer = asyncio.run(reopened.execute_canonical())
    assert answer == result.execution_answer


def test_enterprise_durable_f5_emergency_rollback_rehydrates_prior_revision(
    tmp_path: Path,
) -> None:
    config = default_stage15_proof_config()
    db_path = tmp_path / "f5.db"
    shared_root = tmp_path / "shared-artifacts"
    stack = EnterpriseAgentLifecycleProofStack.build(shared_root, db_path=db_path)
    result_n = stack.run_happy_path()
    principal = admin_test_principal()
    stack.admin.install_agent(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
        request=InstallAgentRequest(
            mutation_id="mut-enterprise-f5-install-n1",
            installation_id="inst-enterprise-f5-n1",
            installation_slot_id=config.installation_slot_id,
            package_identity=AgentPackageIdentity(
                distribution_package_id=config.distribution_package_id,
                package_version="2.0.0",
                package_digest=_DIGEST_N1,
            ),
            artifact_store_ref="store://artifacts/inst-enterprise-f5-n1",
            trust_record=AgentInstallationTrustRecord(
                qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
                package_digest=_DIGEST_N1,
                publisher_identity_ref="publisher:stage15",
                source_provider_id=config.catalog_source_id,
                trust_evidence_refs=(
                    AgentTrustEvidenceRef(
                        evidence_id="evidence:f5-n1",
                        kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                    ),
                ),
            ),
            agent_project_metadata_ref=config.metadata_ref,
        ),
        principal=principal,
    )
    built_n1 = stack.build_revision(revision_id="rev-enterprise-f5-n1")
    stack.register_projection_and_activate(built_n1)
    serving_n1 = stack.admin.inspect_serving(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    assert serving_n1.traffic_serving_revision_id == built_n1.runtime_revision_id
    response = stack.admin.respond_to_emergency_revocation(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
        request=AgentEmergencyRevocationRequest(
            application_id=config.application_id,
            application_environment_id=config.environment_id,
            expected_current_traffic_revision_id=built_n1.runtime_revision_id,
            expected_serving_pointer_revision=serving_n1.serving_pointer_revision,
            evaluated_at=datetime.now(UTC),
            revocation_state=AgentPackageTrustRevocationState(
                revoked_package_digests=frozenset({_DIGEST_N1}),
            ),
        ),
    )
    assert response.action is EmergencyTrustResponseAction.ROLLBACK
    assert response.response_reason_code is (
        EmergencyTrustResponseReasonCode.SAFE_ROLLBACK_COMPLETED
    )
    serving_rolled = stack.admin.inspect_serving(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    assert serving_rolled.traffic_serving_revision_id == result_n.runtime_revision_id
    del stack
    reopened = EnterpriseAgentLifecycleProofStack.reopen(shared_root, db_path, config)
    serving = reopened.admin.inspect_serving(
        application_id=config.application_id,
        application_environment_id=config.environment_id,
    )
    assert serving.traffic_serving_revision_id == result_n.runtime_revision_id
    _, answer = asyncio.run(reopened.execute_canonical())
    assert answer == result_n.execution_answer
