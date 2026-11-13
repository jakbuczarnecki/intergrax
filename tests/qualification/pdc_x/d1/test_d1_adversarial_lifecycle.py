# © Artur Czarnecki. All rights reserved.

"""PDC-D1 adversarial lifecycle coverage: R003, R004, R007, R012, R013.

Each test drives real distribution-plane code (trust coordinator, attestation verifier,
admin service, activation, emergency revocation) over in-memory stores.
"""

from __future__ import annotations

import ast
import inspect
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from intergrax.agent_distribution.activation_domain_coordinator import (
    ServingExecutionAdmissionRequest,
)
from intergrax.agent_distribution.admin_models import (
    AgentPlatformAdminGovernanceBlockedError,
    AgentUpgradeOutcome,
    HandlePostCutoverFailureRequest,
    InstallAgentRequest,
    RetireAgentRequest,
    RollbackRuntimeRevisionRequest,
)
from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.catalog import CatalogProviderKind, CatalogSourceIdentity
from intergrax.agent_distribution.deployment import DeploymentInstanceState
from intergrax.agent_distribution.ed25519_package_attestation_verifier import (
    Ed25519PackageAttestationVerifier,
)
from intergrax.agent_distribution.emergency_revocation_response import (
    EmergencyTrustResponseAction,
    EmergencyTrustResponseReasonCode,
)
from intergrax.agent_distribution.errors import (
    AgentPackageAttestationError,
    AgentPackageTrustError,
    InstallationSlotConflict,
    RuntimeReadinessError,
    ServingExecutionAdmissionError,
)
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentInstallationStore,
)
from intergrax.agent_distribution.installation import (
    AgentInstallationRecord,
    InstallationState,
)
from intergrax.agent_distribution.package_attestation import (
    AgentPackageAttestationAlgorithm,
    AgentPackageAttestationVerificationOutcome,
    AgentPackageAttestationVerificationReasonCode,
    AgentPackageAttestationVerificationRequest,
)
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.agent_distribution.serving_execution_admission import (
    verify_current_serving_execution_admission_eligibility,
)
from intergrax.agent_distribution.trust import (
    AgentDeliverySource,
    AgentPackageQualificationResult,
    AgentPackageTrustOutcome,
    AgentPackageTrustPolicy,
    AgentPackageTrustPosture,
    AgentPackageTrustReasonCode,
    AgentPackageTrustRevocationState,
    AgentPublisherIdentity,
    AgentQualificationEvidenceKind,
)
from intergrax.core.qualification import QualificationEvidence, QualificationStatus
from testing_support.agent_package_attestation import (
    build_test_attestation_keypair,
    build_test_attestation_trust_coordinator,
    sign_package_attestation_statement,
    verified_signature_qualification_evidence,
)
from tests.qualification.pdc_x.d1 import _d1_harness as h

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_V1, _V11 = "1.0.0", "1.1.0"
_PUBLISHER = AgentPublisherIdentity(publisher_id="publisher:acme", display_name="ACME")
_SOURCE = CatalogSourceIdentity(catalog_source_id="builtin", provider_kind=CatalogProviderKind.BUILTIN)
_KEY_ID = "publisher-key-1"
_QUALIFIED_AT = datetime(2026, 8, 6, 12, 0, 0, tzinfo=UTC)
_SIGNATURE_POLICY = AgentPackageTrustPolicy(
    posture=AgentPackageTrustPosture.PRODUCTION,
    required_evidence_kinds=frozenset(
        {
            AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            AgentQualificationEvidenceKind.REVOCATION_CHECK,
        }
    ),
)


def _upgrade_request(candidate: h.Release, expected: str):
    return candidate.upgrade_request(expected_active=expected)


# --------------------------------------------------------------------------- R003 provenance


@pytest.mark.parametrize("tamper", ["package_version", "package_digest", "publisher_id"])
def test_r003_tampered_package_fails_attestation_verification(tamper: str) -> None:
    identity = h.package(_V1)
    private_key, public_key = build_test_attestation_keypair()
    signature = sign_package_attestation_statement(
        package_identity=identity,
        publisher_id="publisher:acme",
        key_id=_KEY_ID,
        private_key=private_key,
    )
    publisher = "publisher:acme"
    presented = identity
    if tamper == "publisher_id":
        publisher = "publisher:evil"
    elif tamper == "package_version":
        presented = identity.model_copy(update={"package_version": "9.9.9"})
    else:
        presented = identity.model_copy(update={"package_digest": h.DIGESTS[_V11]})
    verifier = Ed25519PackageAttestationVerifier()
    request = AgentPackageAttestationVerificationRequest(
        package_identity=presented,
        publisher_id=publisher,
        attestation_id="attest-1",
        key_id=_KEY_ID,
        algorithm=AgentPackageAttestationAlgorithm.ED25519,
        signature_b64=signature,
        public_key_bytes=public_key,
    )
    result = verifier.verify(request)
    assert result.outcome is AgentPackageAttestationVerificationOutcome.INVALID
    assert result.reason_code is AgentPackageAttestationVerificationReasonCode.INVALID_SIGNATURE
    with pytest.raises(AgentPackageAttestationError):
        verifier.verify_qualification_evidence(request)  # no evidence for a tampered package


def test_r003_unsigned_package_is_denied_by_the_trust_coordinator() -> None:
    coordinator = build_test_attestation_trust_coordinator()
    decision = coordinator.evaluate(
        package_identity=h.package(_V1),
        catalog_source=_SOURCE,
        delivery_source=AgentDeliverySource.BUILTIN,
        publisher=_PUBLISHER,
        policy=_SIGNATURE_POLICY,
        qualification=None,
    )
    assert decision.outcome is AgentPackageTrustOutcome.DENY
    assert decision.reason_code is AgentPackageTrustReasonCode.MISSING_REQUIRED_EVIDENCE
    assert not decision.installable


def test_r003_evidence_signed_for_another_digest_does_not_transfer_to_a_tampered_package() -> None:
    coordinator = build_test_attestation_trust_coordinator()
    genuine = verified_signature_qualification_evidence(package_identity=h.package(_V1), publisher_id="publisher:acme")
    revocation_ok = QualificationEvidence(
        kind=AgentQualificationEvidenceKind.REVOCATION_CHECK, code="revocation_ok", ref="rev"
    )
    qualification = AgentPackageQualificationResult(
        publisher=_PUBLISHER,
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(genuine, revocation_ok),
        reason="verified",
        delivery_source=AgentDeliverySource.BUILTIN,
        qualified_at=_QUALIFIED_AT,
    )
    tampered = h.package(_V1).model_copy(update={"package_digest": h.DIGESTS[_V11]})
    decision = coordinator.evaluate(
        package_identity=tampered,
        catalog_source=_SOURCE,
        delivery_source=AgentDeliverySource.BUILTIN,
        publisher=_PUBLISHER,
        policy=_SIGNATURE_POLICY,
        qualification=qualification,
        evidence_package_digest=h.DIGESTS[_V1],
    )
    assert decision.outcome is AgentPackageTrustOutcome.DENY
    assert decision.reason_code is AgentPackageTrustReasonCode.EVIDENCE_DIGEST_MISMATCH


def test_r003_install_rejects_digest_mismatch_unqualified_and_evidence_free_trust() -> None:
    stack = h.build_stack()
    first = h.release(_V1)
    stack.register(first)
    request = first.install_request()

    with pytest.raises(AgentPackageTrustError):  # trust record bound to another digest
        InstallAgentRequest(
            **{**_install_kwargs(request), "trust_record": h.trust(_V11)},
        )

    unqualified = request.model_copy(
        update={
            "trust_record": request.trust_record.model_copy(
                update={"qualification_status": QualificationStatus.NOT_QUALIFIED}
            )
        }
    )
    evidence_free = request.model_copy(
        update={"trust_record": request.trust_record.model_copy(update={"trust_evidence_refs": ()})}
    )
    for bad in (unqualified, evidence_free):
        with pytest.raises(AgentPackageTrustError):
            stack.service.install_agent(
                application_id=h.APP,
                application_environment_id=h.ENV,
                request=bad,
                principal=h.principal(),
            )
    assert stack.state.installations == {}  # nothing persisted for rejected provenance


def _install_kwargs(request: InstallAgentRequest) -> dict[str, object]:
    return {
        "mutation_id": request.mutation_id,
        "installation_id": request.installation_id,
        "installation_slot_id": request.installation_slot_id,
        "package_identity": request.package_identity,
        "artifact_store_ref": request.artifact_store_ref,
        "trust_record": request.trust_record,
        "agent_project_metadata_ref": request.agent_project_metadata_ref,
        "package_contract_authority": request.package_contract_authority,
    }


def test_r003_upgrade_candidate_with_revoked_digest_is_refused_before_any_store_write() -> None:
    stack = h.build_stack()
    h.install_bind_enable(stack, h.release(_V1))
    stack.revocation.state = AgentPackageTrustRevocationState(revoked_package_digests=frozenset({h.DIGESTS[_V11]}))
    candidate = h.release(_V11, contract_version="1.1.0")
    stack.register(candidate)
    with pytest.raises(AgentPackageTrustError) as exc_info:
        stack.service.upgrade_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=_upgrade_request(candidate, h.installation_id(_V1)),
            principal=h.principal(),
        )
    assert exc_info.value.reason_code == "package_digest_revoked"
    assert h.installation_id(_V11) not in stack.state.installations
    active = stack.installation_service.resolve_active_for_slot(h.ENV, h.SLOT)
    assert active is not None and active.installation_id == h.installation_id(_V1)


# --------------------------------------------------------------------------- R004 no shadow registry

_PRODUCT_ROOTS = ("intergrax", "agents", "applications")
_AGENT_DISTRIBUTION = _REPO_ROOT / "intergrax" / "agent_distribution"


def _python_files(root: str) -> list[Path]:
    return sorted(
        path
        for path in (_REPO_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts and ".venv" not in path.parts
    )


def test_r004_agent_registry_is_defined_exactly_once() -> None:
    definitions: list[str] = []
    for root in _PRODUCT_ROOTS:
        for path in _python_files(root):
            text = path.read_text(encoding="utf-8")
            if "AgentRegistry" not in text:
                continue
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "AgentRegistry":
                    definitions.append(path.relative_to(_REPO_ROOT).as_posix())
    assert definitions == ["intergrax/runtime/registry/agent_registry.py"]


def test_r004_distribution_plane_defines_no_registry_and_never_imports_agent_registry() -> None:
    offenders: list[str] = []
    for path in sorted(_AGENT_DISTRIBUTION.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and re.fullmatch(r"\w*Agent\w*Registry", node.name):
                offenders.append(f"{path.name}: class {node.name}")
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("intergrax.runtime.registry.agent_registry"):
                    offenders.append(f"{path.name}: imports {node.module}")
                if any(alias.name == "AgentRegistry" for alias in node.names):
                    offenders.append(f"{path.name}: imports AgentRegistry")
    assert not offenders, offenders


def test_r004_runtime_registry_is_a_projection_without_install_api() -> None:
    from intergrax.runtime.registry.agent_registry import AgentRegistry

    public = {name for name, _ in inspect.getmembers(AgentRegistry, inspect.isfunction) if not name.startswith("_")}
    lifecycle_verbs = ("install", "upgrade", "promote", "activate", "rollback", "revoke", "bind")
    assert not {name for name in public if any(verb in name for verb in lifecycle_verbs)}
    for path in sorted((_REPO_ROOT / "intergrax" / "runtime" / "registry").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "intergrax.agent_distribution" not in text, path.name  # projection only, no authority


def test_r004_lifecycle_operations_live_only_on_the_admin_service_and_its_facade() -> None:
    owners: dict[str, set[str]] = {"upgrade_agent": set(), "retire_agent": set(), "install_agent": set()}
    for path in sorted(_AGENT_DISTRIBUTION.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if (
                        isinstance(item, ast.FunctionDef)
                        and item.name in owners
                        and not node.name.endswith("Port")  # Protocol ports declare, never implement
                    ):
                        owners[item.name].add(f"{path.name}:{node.name}")
    for name, where in owners.items():
        assert where == {
            "admin_service.py:AgentPlatformAdminService",
            "agent_manager_command_facade.py:AgentManagerCommandFacade",
        }, (name, where)
    assert AgentPlatformAdminService.upgrade_agent and AgentPlatformAdminService.retire_agent


# --------------------------------------------------------------------------- R007 tenant isolation


def _stack_with_validated_revision() -> h.D1Stack:
    stack = h.build_stack()
    h.install_bind_enable(stack, h.release(_V1))
    h.build_revision(stack, "rev-1")
    return stack


def test_r007_cross_tenant_install_bind_upgrade_retire_are_denied_without_mutation() -> None:
    stack = h.build_stack()
    first = h.release(_V1)
    stack.register(first)
    foreign = h.OTHER_TENANT_PRINCIPAL

    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as install_denied:
        stack.service.install_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=first.install_request(),
            principal=foreign,
        )
    assert install_denied.value.blocker_code == "AP-11_BLOCKED_BY_TENANT_AUTHORITY"
    assert stack.state.installations == {}

    h.install_bind_enable(stack, first)
    candidate = h.release(_V11, contract_version="1.1.0")
    stack.register(candidate)
    before_bindings = stack.service.list_bindings(application_id=h.APP, application_environment_id=h.ENV)

    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as bind_denied:
        stack.service.bind_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=h.bind_request(binding_id="bind-foreign", mutation_id="mut-bind-foreign"),
            principal=foreign,
        )
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as upgrade_denied:
        stack.service.upgrade_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=_upgrade_request(candidate, h.installation_id(_V1)),
            principal=foreign,
        )
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError) as retire_denied:
        stack.service.retire_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=RetireAgentRequest(
                mutation_id="mut-retire-foreign",
                installation_slot_id=h.SLOT,
                expected_active_installation_id=h.installation_id(_V1),
            ),
            principal=foreign,
        )
    for denied in (bind_denied, upgrade_denied, retire_denied):
        assert denied.value.blocker_code == "AP-11_BLOCKED_BY_TENANT_AUTHORITY"
        assert denied.value.tenant_scope_denial is not None
        assert denied.value.tenant_scope_denial.tenant_id == h.OTHER_TENANT
    assert stack.service.list_bindings(application_id=h.APP, application_environment_id=h.ENV) == before_bindings
    active = stack.installation_service.resolve_active_for_slot(h.ENV, h.SLOT)
    assert active is not None and active.installation_id == h.installation_id(_V1)
    assert h.installation_id(_V11) not in stack.state.installations


def test_r007_cross_tenant_activate_and_rollback_are_denied_before_any_deployment_work() -> None:
    stack = _stack_with_validated_revision()
    foreign = h.OTHER_TENANT_PRINCIPAL

    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        h.activate_revision(stack, "rev-1", principal_value=foreign)

    # the foreign principal must not even trigger candidate preparation
    assert stack.adapter.prepare_count == {}
    assert stack.deployment_store.get_instance(h.APP, h.ENV, "rev-1") is None
    serving = stack.service.inspect_serving(application_id=h.APP, application_environment_id=h.ENV)
    assert serving.traffic_serving_revision_id is None

    _two_revisions(stack)
    h.activate_revision(stack, "rev-2", expected_prior="rev-1", expected_pointer=1)
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.rollback_revision(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=RollbackRuntimeRevisionRequest(
                mutation_id="mut-rollback-foreign",
                expected_current_traffic_revision_id="rev-2",
                expected_serving_pointer_revision=2,
            ),
            principal=foreign,
        )
    assert (
        stack.service.inspect_serving(
            application_id=h.APP, application_environment_id=h.ENV
        ).traffic_serving_revision_id
        == "rev-2"
    )


def test_r007_foreign_principal_learns_nothing_about_rollback_state() -> None:
    stack = _stack_with_validated_revision()
    h.activate_revision(stack, "rev-1")  # single revision: no rollback target exists
    with pytest.raises(AgentPlatformAdminGovernanceBlockedError):
        stack.service.rollback_revision(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=RollbackRuntimeRevisionRequest(
                mutation_id="mut-rollback-probe",
                expected_current_traffic_revision_id="rev-1",
                expected_serving_pointer_revision=1,
            ),
            principal=h.OTHER_TENANT_PRINCIPAL,
        )


def test_r007_other_application_cannot_see_or_retire_a_slot_it_does_not_own() -> None:
    stack = h.build_stack()
    h.install_bind_enable(stack, h.release(_V1))
    with pytest.raises(Exception) as excinfo:
        stack.service.retire_agent(
            application_id=h.APP_B,
            application_environment_id=h.ENV,
            request=RetireAgentRequest(
                mutation_id="mut-retire-app-b",
                installation_slot_id=h.SLOT,
                expected_active_installation_id=h.installation_id(_V1),
            ),
            principal=h.principal(),
        )
    assert type(excinfo.value).__name__ == "AgentDistributionNotFoundError"
    active = stack.installation_service.resolve_active_for_slot(h.ENV, h.SLOT)
    assert active is not None and active.installation_state is InstallationState.INSTALLED_ACTIVE


# --------------------------------------------------------------------------- R012 rollback


def _two_revisions(stack: h.D1Stack) -> None:
    """v1.0.0 serving as rev-1; upgraded to v1.1.0 with a validated rev-2."""
    h.activate_revision(stack, "rev-1")
    upgrade = stack.service.upgrade_agent(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=_prepare_upgrade(stack),
        principal=h.principal(),
    )
    assert upgrade.outcome is AgentUpgradeOutcome.UPGRADED
    h.build_revision(stack, "rev-2")


def _prepare_upgrade(stack: h.D1Stack):
    candidate = h.release(_V11, contract_version="1.1.0")
    stack.register(candidate)
    return _upgrade_request(candidate, h.installation_id(_V1))


def test_r012_failed_activation_leaves_traffic_on_the_previous_revision_and_slot_rolls_back() -> None:
    stack = _stack_with_validated_revision()
    _two_revisions(stack)
    stack.adapter.fail_readiness("rev-2")

    with pytest.raises(RuntimeReadinessError):
        h.activate_revision(stack, "rev-2", expected_prior="rev-1", expected_pointer=1)

    serving = stack.service.inspect_serving(application_id=h.APP, application_environment_id=h.ENV)
    assert serving.traffic_serving_revision_id == "rev-1"
    assert serving.serving_pointer_revision == 1
    rev1 = stack.service.inspect_revision(
        application_id=h.APP, application_environment_id=h.ENV, runtime_revision_id="rev-1"
    )
    rev2 = stack.service.inspect_revision(
        application_id=h.APP, application_environment_id=h.ENV, runtime_revision_id="rev-2"
    )
    assert rev1.revision_state is RuntimeRevisionState.ACTIVE
    assert rev2.revision_state is RuntimeRevisionState.VALIDATED
    failed_instance = stack.deployment_store.get_instance(h.APP, h.ENV, "rev-2")
    assert failed_instance is not None
    assert failed_instance.instance_state is DeploymentInstanceState.FAILED

    # operator rolls the slot back to the previous release; the next build is v1.0.0 again
    restored = stack.installation_service.rollback_slot_to_previous(
        h.ENV, h.SLOT, expected_active_installation_id=h.installation_id(_V11)
    )
    assert restored.value.installation_id == h.installation_id(_V1)
    h.build_revision(stack, "rev-3")
    rev3 = stack.service.inspect_revision(
        application_id=h.APP, application_environment_id=h.ENV, runtime_revision_id="rev-3"
    )
    assert rev3.installed_agent_package_digests == (h.DIGESTS[_V1],)


def test_r012_post_cutover_failure_rolls_back_to_the_prior_revision() -> None:
    stack = _stack_with_validated_revision()
    _two_revisions(stack)
    h.activate_revision(stack, "rev-2", expected_prior="rev-1", expected_pointer=1)

    result = stack.service.handle_post_cutover_failure(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=HandlePostCutoverFailureRequest(
            mutation_id="mut-post-cutover",
            recovery_mutation_id="mut-recovery",
            runtime_revision_id="rev-2",
            failure_evidence_ref="evidence://post-cutover",
            attempt_rollback=True,
        ),
        principal=h.principal(),
    )

    assert result.rollback_result is not None
    assert result.rollback_result.restored_revision_id == "rev-1"
    serving = stack.service.inspect_serving(application_id=h.APP, application_environment_id=h.ENV)
    assert serving.traffic_serving_revision_id == "rev-1"


class _PromotionFailingInstallationStore(InMemoryAgentInstallationStore):
    def __init__(self, state: AgentDistributionStoreState) -> None:
        super().__init__(state)
        self.fail_next_promotion = False

    def atomic_promote_active_installation(self, **kwargs):  # type: ignore[no-untyped-def]
        if self.fail_next_promotion:
            raise RuntimeError("simulated promotion failure")
        return super().atomic_promote_active_installation(**kwargs)


def test_r012_upgrade_whose_promotion_fails_leaves_the_previous_revision_active() -> None:
    stack = h.build_stack(installation_store_factory=_PromotionFailingInstallationStore)
    h.install_bind_enable(stack, h.release(_V1))
    store = stack.installation_store
    assert isinstance(store, _PromotionFailingInstallationStore)
    store.fail_next_promotion = True
    candidate = h.release(_V11, contract_version="1.1.0")
    stack.register(candidate)

    with pytest.raises(RuntimeError, match="simulated promotion failure"):
        stack.service.upgrade_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=_upgrade_request(candidate, h.installation_id(_V1)),
            principal=h.principal(),
        )

    active = stack.installation_service.resolve_active_for_slot(h.ENV, h.SLOT)
    assert active is not None
    assert active.installation_id == h.installation_id(_V1)
    assert active.installation_state is InstallationState.INSTALLED_ACTIVE
    failed: AgentInstallationRecord | None = store.get_installation(h.installation_id(_V11))
    assert failed is not None and failed.installation_state is InstallationState.FAILED_CANDIDATE
    binding = stack.service.list_bindings(application_id=h.APP, application_environment_id=h.ENV).bindings[0]
    assert binding.active_installation_id == h.installation_id(_V1)


def test_r012_concurrent_slot_change_is_detected_and_does_not_clobber_the_active_release() -> None:
    stack = h.build_stack()
    h.install_bind_enable(stack, h.release(_V1))
    candidate = h.release(_V11, contract_version="1.1.0")
    stack.register(candidate)
    with pytest.raises(InstallationSlotConflict):
        stack.service.upgrade_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=_upgrade_request(candidate, "inst-someone-else"),
            principal=h.principal(),
        )
    active = stack.installation_service.resolve_active_for_slot(h.ENV, h.SLOT)
    assert active is not None and active.installation_id == h.installation_id(_V1)


# --------------------------------------------------------------------------- R013 revocation


def _serving_two_revisions(stack: h.D1Stack) -> None:
    _two_revisions(stack)
    h.activate_revision(stack, "rev-2", expected_prior="rev-1", expected_pointer=1)


def _admission_request(stack: h.D1Stack, revision_id: str) -> ServingExecutionAdmissionRequest:
    instance = stack.deployment_store.get_instance(h.APP, h.ENV, revision_id)
    assert instance is not None and instance.serving_unit_ref is not None
    return ServingExecutionAdmissionRequest(
        application_id=h.APP,
        application_environment_id=h.ENV,
        runtime_revision_id=revision_id,
        serving_unit_ref=instance.serving_unit_ref,
        materialization_artifact_digest=h.ARTIFACT_DIGEST,
    )


def _admit(stack: h.D1Stack, revision_id: str):
    request = _admission_request(stack, revision_id)
    return verify_current_serving_execution_admission_eligibility(
        serving_store=stack.serving_store,
        deployment_instance_store=stack.deployment_store,
        revision_store=stack.revision_store,
        application_id=request.application_id,
        application_environment_id=request.application_environment_id,
        runtime_revision_id=request.runtime_revision_id,
        serving_unit_ref=request.serving_unit_ref,
        materialization_artifact_digest=request.materialization_artifact_digest,
    )


def test_r013_revocation_while_serving_rolls_back_and_closes_admission_on_the_revoked_revision() -> None:
    stack = _stack_with_validated_revision()
    _serving_two_revisions(stack)
    assert _admit(stack, "rev-2") is not None  # executing under the soon-revoked release

    stack.revocation.state = AgentPackageTrustRevocationState(revoked_package_digests=frozenset({h.DIGESTS[_V11]}))
    response = stack.service.respond_to_emergency_revocation(application_id=h.APP, application_environment_id=h.ENV)

    assert response.action is EmergencyTrustResponseAction.ROLLBACK
    assert response.response_reason_code is EmergencyTrustResponseReasonCode.SAFE_ROLLBACK_COMPLETED
    serving = stack.service.inspect_serving(application_id=h.APP, application_environment_id=h.ENV)
    assert serving.traffic_serving_revision_id == "rev-1"
    with pytest.raises(ServingExecutionAdmissionError):  # new admissions on the revoked revision fail closed
        _admit(stack, "rev-2")
    drained = stack.deployment_store.get_instance(h.APP, h.ENV, "rev-2")
    assert drained is not None and drained.instance_state is DeploymentInstanceState.DRAINING
    assert _admit(stack, "rev-1") is not None


def test_r013_revocation_without_a_trusted_rollback_target_is_blocked_explicitly_and_blocks_new_builds() -> None:
    stack = _stack_with_validated_revision()
    _serving_two_revisions(stack)
    stack.revocation.state = AgentPackageTrustRevocationState(
        revoked_package_digests=frozenset({h.DIGESTS[_V1], h.DIGESTS[_V11]})
    )

    response = stack.service.respond_to_emergency_revocation(application_id=h.APP, application_environment_id=h.ENV)

    assert response.action is EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET
    assert response.response_reason_code is EmergencyTrustResponseReasonCode.ROLLBACK_TARGET_UNTRUSTED
    # the revocation reaches the build path too: no new revision may include the revoked package
    with pytest.raises(AgentPackageTrustError):
        h.build_revision(stack, "rev-after-revocation")
    assert "rev-after-revocation" not in stack.state.revisions
    # and no trust-evidence-free install of the revoked digest is possible
    with pytest.raises(AgentPackageTrustError):
        stack.service.install_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=h.release(_V1)
            .install_request()
            .model_copy(update={"installation_id": "inst-reinstall", "installation_slot_id": "slot-other"}),
            principal=h.principal(),
        )


def test_r013_revoking_a_non_serving_installation_takes_no_action() -> None:
    stack = _stack_with_validated_revision()
    _serving_two_revisions(stack)
    stack.revocation.state = AgentPackageTrustRevocationState(
        revoked_package_digests=frozenset({"sha256:" + ("9" * 64)})
    )
    response = stack.service.respond_to_emergency_revocation(application_id=h.APP, application_environment_id=h.ENV)
    assert response.action is EmergencyTrustResponseAction.NO_ACTION
    serving = stack.service.inspect_serving(application_id=h.APP, application_environment_id=h.ENV)
    assert serving.traffic_serving_revision_id == "rev-2"
