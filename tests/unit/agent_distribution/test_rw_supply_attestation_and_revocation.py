# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-N (T-AF2-RW-SUPPLY-AB): attestation on install and upgrade, mandatory trust composition,
revocation source, lifecycle evidence consumer and the no-target serving stop (NT-RW-SUP-01..04).

Everything runs in process: the Ed25519 key is a throwaway key generated at run time (never printed),
the revocation states are in-process stubs that stand in for a revocation feed, and the evidence
plane is an in-memory runtime event store. No network, no real publisher key, no live backend.

Red-first: names introduced by this ticket are imported inside the helpers that need them, so the
nodes that use only the existing service API (03a, 03c, 04c) run at the pre-change commit and fail
there with an assertion. Every refusal is captured with ``_capture`` and checked with ``assert``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    AgentUpgradeOutcome,
    InstallationMutationResult,
    UpgradeAgentRequest,
    UpgradeAgentResult,
)
from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.control_plane_governance import StaticApplicationEnvironmentTenantResolver
from intergrax.agent_distribution.deployment import DeploymentInstanceState
from intergrax.agent_distribution.ed25519_package_attestation_verifier import Ed25519PackageAttestationVerifier
from intergrax.agent_distribution.emergency_revocation_response import (
    AgentEmergencyRevocationRequest,
    AgentEmergencyRevocationResponse,
    AgentEmergencyRevocationService,
    EmergencyTrustResponseAction,
    EmergencyTrustResponseReasonCode,
)
from intergrax.agent_distribution.errors import (
    AgentPackageTrustError,
    RuntimeActivationConflict,
    RuntimeRollbackError,
    ServingExecutionAdmissionError,
)
from intergrax.agent_distribution.events import AgentDistributionEvent
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.installation import InstallationState
from intergrax.agent_distribution.materialization_service import RuntimeMaterializationService
from intergrax.agent_distribution.package_attestation import StaticPublisherVerificationKeyProvider
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.serving_execution_admission import (
    verify_current_serving_execution_admission_eligibility,
)
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentPackageTrustPolicy,
    AgentPackageTrustReasonCode,
    AgentPackageTrustRevocationState,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.applications._shared.production_agent_platform_runtime import ProductionAgentPlatformRuntime
from intergrax.applications._shared.production_lifecycle import ProductionLifecycleLauncher
from intergrax.applications._shared.production_process_composition import (
    create_reference_production_process_composition,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    build_production_registry_projection_input_bundle_for_revision,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleLauncher,
    wire_reference_production_lifecycle_services,
)
from intergrax.applications._shared.registry_projection_input_bundle import reference_admission_mutation_id
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.security.security_audit_trail import (
    EvidencePlaneSecurityAuditTrail,
    SecurityAuditSink,
    SecurityAuditTrail,
)
from testing_support import canonical_agent_lifecycle_composition as canonical_module
from testing_support.agent_package_attestation import sign_package_attestation_statement
from testing_support.agent_platform_admin_harness import (
    AgentProjectMetadataTestProvider,
    DeterministicAgentDistributionAdapter,
    allow_mutation_boundary,
)
from testing_support.agent_platform_dependency_resolver import make_identity_dependency_resolver
from testing_support.canonical_agent_lifecycle_composition import default_stage15_proof_config
from testing_support.enterprise_agent_lifecycle_composition import EnterpriseAgentLifecycleProofStack
from tests.qualification.pdc_x.d1 import _d1_harness as h
from tests.qualification.pdc_x.d1_repair import _support as s

from governed_contractor_application.host import single_node_profile_v1_strict_startup as strict_startup

if TYPE_CHECKING:
    from intergrax.applications._shared.production_agent_capability_runtime import (
        ProductionPackageTrustComposition,
    )

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PUBLISHER = "publisher:acme"
_OTHER_PUBLISHER = "publisher:other"
_KEY_ID = "rw-sup-throwaway-key"
_STOP_EVENT = "agent_trust.emergency_serving_stopped"
_SUPERSEDED_EVENT = "agent_trust.emergency_serving_stop_superseded"


def _capture(action: Callable[[], object]) -> Exception | None:
    """Run ``action``; return the exception it raised (or ``None``) so refusals are asserted."""
    try:
        action()
    except Exception as exc:  # the caller asserts the exact type
        return exc
    return None


# --------------------------------------------------------------------------- production factory


@dataclass
class _Signer:
    private_key: Ed25519PrivateKey
    public_key_bytes: bytes


def _throwaway_signer() -> _Signer:
    private_key = Ed25519PrivateKey.generate()
    return _Signer(private_key=private_key, public_key_bytes=private_key.public_key().public_bytes_raw())


@dataclass
class _FactoryStack:
    admin: AgentPlatformAdminService
    runtime: ProductionAgentPlatformRuntime
    metadata: AgentProjectMetadataTestProvider

    def register(self, *releases: h.Release) -> None:
        for item in releases:
            self.metadata._records[h.meta_ref(item.version)] = item.metadata()

    def installation_states(self) -> dict[str, InstallationState]:
        return {
            record.installation_id: record.installation_state
            for record in self.runtime.distribution_state.installations.values()
        }


def _non_lab_trust(
    signer: _Signer,
    *,
    trail: SecurityAuditSink | None = None,
    revocation: Callable[[], AgentPackageTrustRevocationState] | None = None,
    extra_keys: dict[tuple[str, str], bytes] | None = None,
) -> ProductionPackageTrustComposition:
    from intergrax.applications._shared.production_agent_capability_runtime import (
        ProductionPackageTrustComposition,
    )

    keys = {(_PUBLISHER, _KEY_ID): signer.public_key_bytes, **(extra_keys or {})}
    return ProductionPackageTrustComposition(
        attestation_verifier=Ed25519PackageAttestationVerifier(
            key_provider=StaticPublisherVerificationKeyProvider(keys)
        ),
        revocation_state_source=revocation or AgentPackageTrustRevocationState,
        policy_source=AgentPackageTrustPolicy,
        evaluation_time_source=lambda: h.EVALUATED_AT,
        lifecycle_evidence_trail=trail if trail is not None else SecurityAuditTrail(),
        is_lab=False,
    )


def _factory_stack(trust: ProductionPackageTrustComposition) -> _FactoryStack:
    from intergrax.applications._shared.production_agent_capability_runtime import (
        ProductionAgentPlatformAdminConfig,
        build_production_agent_platform_admin_service,
    )

    base = create_reference_production_process_composition()
    metadata = AgentProjectMetadataTestProvider({})
    admin = build_production_agent_platform_admin_service(
        agent_platform_runtime=base.agent_platform_runtime,
        lifecycle_services=wire_reference_production_lifecycle_services(base),
        admin_config=ProductionAgentPlatformAdminConfig(
            metadata_provider=metadata,
            materialization_service=RuntimeMaterializationService(
                {MaterializationTopology.OCI_IMAGE: DeterministicAgentDistributionAdapter()}
            ),
            dependency_resolver=make_identity_dependency_resolver(),
            mutation_authorization_boundary=allow_mutation_boundary(),
            environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(h.TENANT),
            package_trust=trust,
        ),
    )
    return _FactoryStack(admin=admin, runtime=base.agent_platform_runtime, metadata=metadata)


def _signed_record(
    release: h.Release,
    signer: _Signer,
    *,
    attestation_id: str = "attest-rw-sup-1",
    signed_identity: AgentPackageIdentity | None = None,
    material_publisher: str = _PUBLISHER,
    extra_refs: tuple[AgentTrustEvidenceRef, ...] = (),
) -> AgentInstallationTrustRecord:
    from intergrax.agent_distribution.trust import AgentTrustAttestationMaterial

    signature_b64 = sign_package_attestation_statement(
        package_identity=signed_identity or release.identity,
        publisher_id=material_publisher,
        key_id=_KEY_ID,
        private_key=signer.private_key,
    )
    base = h.trust(release.version)
    return AgentInstallationTrustRecord(
        qualification_status=base.qualification_status,
        package_digest=base.package_digest,
        publisher_identity_ref=base.publisher_identity_ref,
        source_provider_id=base.source_provider_id,
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id=attestation_id,
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
            *extra_refs,
        ),
        attestation_material=(
            AgentTrustAttestationMaterial(
                attestation_id=attestation_id,
                key_id=_KEY_ID,
                algorithm="ED25519",
                publisher_id=material_publisher,
                signature_b64=signature_b64,
            ),
        ),
    )


def _install(
    stack: _FactoryStack,
    release: h.Release,
    record: AgentInstallationTrustRecord,
    *,
    slot: str = h.SLOT,
) -> InstallationMutationResult:
    stack.register(release)
    request = release.install_request(slot=slot).model_copy(update={"trust_record": record})
    return stack.admin.install_agent(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=request,
        principal=h.principal(),
    )


def _bind(stack: _FactoryStack) -> None:
    """Bind the slot to the application: an upgrade is scoped to the application that owns the slot."""
    stack.admin.bind_agent(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=h.bind_request(),
        principal=h.principal(),
    )


def _upgrade(stack: _FactoryStack, release: h.Release, record: AgentInstallationTrustRecord) -> UpgradeAgentResult:
    stack.register(release)
    request: UpgradeAgentRequest = release.upgrade_request(expected_active=h.installation_id(s.V1)).model_copy(
        update={"trust_record": record}
    )
    return stack.admin.upgrade_agent(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=request,
        principal=h.principal(),
    )


def _untyped_replace(value: object, **changes: object) -> object:
    """``dataclasses.replace`` without static checking, to pass the ``None`` a caller could pass."""
    replace = cast("Callable[..., object]", dataclasses.replace)
    return replace(value, **changes)


def _trust_reason(caught: Exception | None) -> str | None:
    assert isinstance(caught, AgentPackageTrustError), f"expected a trust refusal, got {caught!r}"
    return caught.reason_code


# --------------------------------------------------------------------------- NT-RW-SUP-01 (EX-D1-01)


def test_sup_01a_unsigned_install_refused() -> None:
    stack = _factory_stack(_non_lab_trust(_throwaway_signer()))
    release = h.release(s.V1)

    caught = _capture(lambda: _install(stack, release, h.trust(s.V1)))

    assert _trust_reason(caught) == AgentPackageTrustReasonCode.UNSIGNED_FORBIDDEN.value
    assert stack.installation_states() == {}
    assert stack.runtime.distribution_state.artifact_metadata == {}


def test_sup_01b_tampered_upgrade_refused() -> None:
    signer = _throwaway_signer()
    stack = _factory_stack(_non_lab_trust(signer))
    first = h.release(s.V1)
    _install(stack, first, _signed_record(first, signer))
    _bind(stack)
    candidate = h.release(s.V11, contract_version="1.1.0")
    # signed over another digest: the signature does not match the digest-pinned candidate
    tampered = _signed_record(candidate, signer, signed_identity=first.identity)

    caught = _capture(lambda: _upgrade(stack, candidate, tampered))

    assert _trust_reason(caught) == AgentPackageTrustReasonCode.MALFORMED_EVIDENCE.value
    assert stack.installation_states() == {h.installation_id(s.V1): InstallationState.INSTALLED_ACTIVE}


def test_sup_01c_forged_evidence_ref_refused() -> None:
    signer = _throwaway_signer()
    other_signer = _throwaway_signer()
    stack = _factory_stack(
        _non_lab_trust(signer, extra_keys={(_OTHER_PUBLISHER, _KEY_ID): other_signer.public_key_bytes})
    )
    release = h.release(s.V1)

    # (1) a signature-verification evidence ref with no attestation material behind it
    ref_only = _capture(lambda: _install(stack, release, h.trust(s.V1)))
    # (2) valid material whose publisher differs from the record's publisher
    foreign = _capture(
        lambda: _install(stack, release, _signed_record(release, other_signer, material_publisher=_OTHER_PUBLISHER))
    )
    # (3) valid material plus a second signature ref that no material backs
    dangling = _capture(
        lambda: _install(
            stack,
            release,
            _signed_record(
                release,
                signer,
                extra_refs=(
                    AgentTrustEvidenceRef(
                        evidence_id="attest-forged",
                        kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                    ),
                ),
            ),
        )
    )

    assert _trust_reason(ref_only) == AgentPackageTrustReasonCode.UNSIGNED_FORBIDDEN.value
    assert _trust_reason(foreign) == AgentPackageTrustReasonCode.MALFORMED_EVIDENCE.value
    assert _trust_reason(dangling) == AgentPackageTrustReasonCode.MALFORMED_EVIDENCE.value
    assert stack.installation_states() == {}


def test_sup_01d_valid_attestation_admitted() -> None:
    signer = _throwaway_signer()
    stack = _factory_stack(_non_lab_trust(signer))
    first = h.release(s.V1)
    _install(stack, first, _signed_record(first, signer))
    _bind(stack)
    candidate = h.release(s.V11, contract_version="1.1.0")

    upgraded = _upgrade(stack, candidate, _signed_record(candidate, signer, attestation_id="attest-rw-sup-2"))

    assert upgraded.outcome is AgentUpgradeOutcome.UPGRADED
    assert stack.installation_states()[h.installation_id(s.V11)] is InstallationState.INSTALLED_ACTIVE


# --------------------------------------------------------------------------- NT-RW-SUP-02 (EX-D1-02)


def test_sup_02a_factory_requires_trust_composition() -> None:
    from intergrax.applications._shared.production_agent_capability_runtime import (
        ProductionAgentPlatformAdminConfig,
        ProductionPackageTrustComposition,
    )

    complete = _non_lab_trust(_throwaway_signer())
    # called through untyped handles on purpose: these calls omit a required input or pass None
    build_config = cast("Callable[..., object]", ProductionAgentPlatformAdminConfig)
    build_trust = cast("Callable[..., object]", ProductionPackageTrustComposition)
    no_trust = _capture(
        lambda: build_config(
            metadata_provider=AgentProjectMetadataTestProvider({}),
            materialization_service=RuntimeMaterializationService({}),
            dependency_resolver=make_identity_dependency_resolver(),
            mutation_authorization_boundary=allow_mutation_boundary(),
            environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(h.TENANT),
        )
    )
    none_member = _capture(lambda: _untyped_replace(complete, revocation_state_source=None))
    no_is_lab = _capture(
        lambda: build_trust(
            attestation_verifier=complete.attestation_verifier,
            revocation_state_source=complete.revocation_state_source,
            policy_source=complete.policy_source,
            evaluation_time_source=complete.evaluation_time_source,
            lifecycle_evidence_trail=complete.lifecycle_evidence_trail,
        )
    )
    permissive = _capture(
        lambda: dataclasses.replace(
            complete, policy_source=lambda: AgentPackageTrustPolicy(forbid_unsigned_or_unqualified=False)
        )
    )

    assert isinstance(no_trust, TypeError)
    assert isinstance(none_member, ValueError) and "revocation_state_source" in str(none_member)
    assert isinstance(no_is_lab, TypeError)
    assert isinstance(permissive, ValueError) and "forbid_unsigned_or_unqualified" in str(permissive)


@pytest.fixture
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr("intergrax.applications._shared.llm_resolver.resolve_llm_adapter", _resolve)


class _LauncherScenario:
    """Enterprise proof stack whose factory receives a package-trust composition with a revocation source.

    The composition is the lab one with an in-process revocation stub (labelled: it stands in for a
    revocation feed); the canonical install carries no attestation material, which the lab policy permits.
    """

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.revocation = h.MutableRevocation()
        self._compose_revocation_source(monkeypatch)
        self.config = default_stage15_proof_config()
        self.stack = EnterpriseAgentLifecycleProofStack.build(tmp_path, db_path=tmp_path / "rw-supply-ab.db")
        self.stack.install_from_catalog()
        self.stack.bind_enabled_agent()
        built = self.stack.build_revision()
        before = self.stack.admin.inspect_serving(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
        )
        assert built.artifact_locator is not None and built.materialization_artifact_digest is not None
        self.revision_id = built.runtime_revision_id
        self.input_bundle = build_production_registry_projection_input_bundle_for_revision(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
            runtime_revision_id=built.runtime_revision_id,
            manifest=self.stack.canonical.manifest,
            build_context=ApplicationBuildContext.for_manifest(self.stack.canonical.manifest),
            authority=self.stack.durable_runtime.registry_projection_authority,
        )
        self.activation_request = ActivateRuntimeRevisionRequest(
            mutation_id="mut-rw-supply-ab",
            runtime_revision_id=built.runtime_revision_id,
            artifact_locator=built.artifact_locator,
            expected_artifact_digest=built.materialization_artifact_digest,
            expected_serving_pointer_revision=before.serving_pointer_revision,
            expected_prior_traffic_revision_id=before.traffic_serving_revision_id,
        )
        self.writes: list[str] = []
        services = self.stack.launcher.services
        for owner, name in (
            (services.revision_service, "persist_candidate_revision"),
            (services.revision_service, "mark_validated"),
            (services.projection_input_store, "register"),
            (services.activation_service, "prepare_candidate"),
        ):
            monkeypatch.setattr(owner, name, self._recording(name, getattr(owner, name)))

    def _compose_revocation_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from intergrax.applications._shared.production_agent_capability_runtime import (
            lab_package_trust_composition,
        )

        composed = dataclasses.replace(lab_package_trust_composition(), revocation_state_source=self.revocation)
        monkeypatch.setattr(canonical_module, "lab_package_trust_composition", lambda: composed)

    def _recording(self, name: str, original: Callable[..., object]) -> Callable[..., object]:
        def _call(*args: object, **kwargs: object) -> object:
            self.writes.append(name)
            return original(*args, **kwargs)

        return _call

    def revoke_serving_package(self) -> None:
        self.revocation.state = AgentPackageTrustRevocationState(
            revoked_package_digests=frozenset({self.config.package_digest})
        )

    def deploy(self, launcher: ProductionLifecycleLauncher | ReferenceProductionLifecycleLauncher) -> object:
        return launcher.deploy_and_activate(
            projection_input=self.input_bundle,
            activation_request=self.activation_request,
            principal=self.stack.canonical.governance.principal,
            admission_mutation_id=reference_admission_mutation_id(self.revision_id),
        )

    def production_launcher(self) -> ProductionLifecycleLauncher:
        reference = self.stack.launcher
        return ProductionLifecycleLauncher(
            reference.process_composition,
            services=reference.services,
            mutation_authorization_boundary=allow_mutation_boundary(),
            environment_tenant_resolver=self.stack.canonical.governance.environment_tenant_resolver,
        )

    def traffic_revision(self) -> str | None:
        return self.stack.admin.inspect_serving(
            application_id=self.config.application_id,
            application_environment_id=self.config.environment_id,
        ).traffic_serving_revision_id


@pytest.mark.usefixtures("_stub_host_llm")
def test_sup_02b_revoked_digest_refused_production_launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _LauncherScenario(tmp_path, monkeypatch)
    scenario.revoke_serving_package()

    caught = _capture(lambda: scenario.deploy(scenario.production_launcher()))

    assert _trust_reason(caught) == AgentPackageTrustReasonCode.PACKAGE_DIGEST_REVOKED.value
    assert scenario.writes == []
    assert scenario.traffic_revision() is None


@pytest.mark.usefixtures("_stub_host_llm")
def test_sup_02c_revoked_digest_refused_reference_launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _LauncherScenario(tmp_path, monkeypatch)
    scenario.revoke_serving_package()

    caught = _capture(lambda: scenario.deploy(scenario.stack.launcher))

    assert _trust_reason(caught) == AgentPackageTrustReasonCode.PACKAGE_DIGEST_REVOKED.value
    assert scenario.writes == []
    assert scenario.traffic_revision() is None


class _LoaderReached(RuntimeError):
    pass


def test_sup_02d_strict_startup_refuses_lab_trust_composition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.applications._shared.production_agent_capability_runtime import (
        AgentCapabilityApplicationComposition,
    )

    reached: list[str] = []

    def _sentinel_loader(**_: object) -> object:
        reached.append("profile_loader")
        raise _LoaderReached("profile loader reached")

    monkeypatch.setattr(
        strict_startup,
        "load_validate_and_admit_single_node_production_profile_v1_from_env",
        _sentinel_loader,
    )
    lab_composition: AgentCapabilityApplicationComposition = canonical_module._build_application_composition(
        default_stage15_proof_config(),
        tmp_path,
        canonical_module._StaticCatalogProvider((), resolutions_by_version={}),
        canonical_module._MetadataProvider({}),
    )

    caught = _capture(
        lambda: strict_startup.open_gca_strict_production_serving_session_from_single_node_production_profile_v1(
            environment=cast("strict_startup.StrictProductionServingEnvironment", object()),
            governance=cast("strict_startup.ProductionControlPlaneGovernance", object()),
            deployment_adapter=cast("strict_startup.RuntimeDeploymentAdapter", object()),
            application_composition=lab_composition,
        )
    )

    assert reached == []
    assert isinstance(caught, ValueError) and "package_trust" in str(caught)


# --------------------------------------------------------------------------- NT-RW-SUP-03 (EX-D1-04)


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentDistributionEvent] = []

    def __call__(self, events: tuple[AgentDistributionEvent, ...]) -> None:
        self.events.extend(events)

    def of_type(self, event_type: str) -> list[AgentDistributionEvent]:
        return [event for event in self.events if event.event_type == event_type]


def _with_sink(stack: h.D1Stack, monkeypatch: pytest.MonkeyPatch, sink: Callable[..., None]) -> None:
    monkeypatch.setattr(stack.service, "_lifecycle_event_sink", sink)


def _respond(
    stack: h.D1Stack, *, expected_current_traffic_revision_id: str | None = None
) -> AgentEmergencyRevocationResponse:
    return stack.service.respond_to_emergency_revocation(
        application_id=h.APP,
        application_environment_id=h.ENV,
        principal=h.principal(),
        expected_current_traffic_revision_id=expected_current_traffic_revision_id,
    )


def _instance_state(stack: h.D1Stack, revision_id: str) -> DeploymentInstanceState | None:
    instance = stack.deployment_store.get_instance(h.APP, h.ENV, revision_id)
    return instance.instance_state if instance is not None else None


def _failure_ref(stack: h.D1Stack, revision_id: str) -> str | None:
    instance = stack.deployment_store.get_instance(h.APP, h.ENV, revision_id)
    return instance.failure_evidence_ref if instance is not None else None


def _admit(stack: h.D1Stack, revision_id: str) -> object:
    instance = stack.deployment_store.get_instance(h.APP, h.ENV, revision_id)
    assert instance is not None and instance.serving_unit_ref is not None
    return verify_current_serving_execution_admission_eligibility(
        serving_store=stack.serving_store,
        deployment_instance_store=stack.deployment_store,
        revision_store=stack.revision_store,
        application_id=h.APP,
        application_environment_id=h.ENV,
        runtime_revision_id=revision_id,
        serving_unit_ref=instance.serving_unit_ref,
        materialization_artifact_digest=h.ARTIFACT_DIGEST,
    )


def _single_revision_serving() -> h.D1Stack:
    """rev-1 (v1.0.0) serving, no prior traffic revision."""
    stack = s.stack_with_validated_revision()
    h.activate_revision(stack, "rev-1")
    return stack


def _two_revisions_serving() -> h.D1Stack:
    stack = s.stack_with_validated_revision()
    s.serving_two_revisions(stack)
    return stack


def _revoke(stack: h.D1Stack, *versions: str) -> None:
    stack.revocation.state = AgentPackageTrustRevocationState(
        revoked_package_digests=frozenset(h.DIGESTS[version] for version in versions)
    )


def _assert_stopped(
    stack: h.D1Stack,
    response: AgentEmergencyRevocationResponse,
    sink: _RecordingSink,
    *,
    revision_id: str,
    reason: EmergencyTrustResponseReasonCode,
) -> None:
    assert response.action is EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET
    assert response.response_reason_code is reason
    assert _instance_state(stack, revision_id) is DeploymentInstanceState.FAILED
    assert _failure_ref(stack, revision_id) == f"emergency-revocation:{reason.value}"
    assert isinstance(_capture(lambda: _admit(stack, revision_id)), ServingExecutionAdmissionError)
    assert _STOP_EVENT in response.audit_event_types
    assert s.traffic_revision(stack) == revision_id  # the serving pointer is unchanged
    stops = sink.of_type(_STOP_EVENT)
    assert len(stops) == 1
    stop = stops[0]
    assert stop.aggregate_id == revision_id
    assert stop.attributes == {
        "runtime_revision_id": revision_id,
        "reason_code": reason.value,
        "failure_evidence_ref": f"emergency-revocation:{reason.value}",
        "instance_state": DeploymentInstanceState.FAILED.value,
        "application_id": h.APP,
        "application_environment_id": h.ENV,
    }


def test_sup_03a_revoked_without_target_stops_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    lone = _single_revision_serving()
    lone_sink = _RecordingSink()
    _with_sink(lone, monkeypatch, lone_sink)
    _revoke(lone, s.V1)
    no_prior = _respond(lone)

    pair = _two_revisions_serving()
    pair_sink = _RecordingSink()
    _with_sink(pair, monkeypatch, pair_sink)
    _revoke(pair, s.V1, s.V11)
    untrusted = _respond(pair)

    _assert_stopped(
        lone, no_prior, lone_sink, revision_id="rev-1", reason=EmergencyTrustResponseReasonCode.NO_PRIOR_REVISION
    )
    _assert_stopped(
        pair,
        untrusted,
        pair_sink,
        revision_id="rev-2",
        reason=EmergencyTrustResponseReasonCode.ROLLBACK_TARGET_UNTRUSTED,
    )


def test_sup_03b_stale_request_does_not_stop_serving() -> None:
    stack = _two_revisions_serving()
    _revoke(stack, s.V11)

    response = _respond(stack, expected_current_traffic_revision_id="rev-1")

    assert response.response_reason_code is EmergencyTrustResponseReasonCode.SERVING_POINTER_MISMATCH
    assert _instance_state(stack, "rev-2") is DeploymentInstanceState.SERVING
    assert _STOP_EVENT not in response.audit_event_types
    assert _admit(stack, "rev-2") is not None


def test_sup_03c_evidence_unavailable_and_rollback_failed_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    unavailable = _two_revisions_serving()
    unavailable_sink = _RecordingSink()
    _with_sink(unavailable, monkeypatch, unavailable_sink)
    s.drop_trust_record(unavailable, s.V11)  # an enabled entry of the serving revision without a trust record
    no_evidence = _respond(unavailable)

    failing = _two_revisions_serving()
    failing_sink = _RecordingSink()
    _with_sink(failing, monkeypatch, failing_sink)
    _revoke(failing, s.V11)  # v1.0.0 stays trusted, so a rollback is attempted

    def _rollback_fails(**_: object) -> object:
        raise RuntimeRollbackError("stub: rollback failed")

    monkeypatch.setattr(failing.activation, "rollback", _rollback_fails)
    rollback_failed = _respond(failing)

    _assert_stopped(
        unavailable,
        no_evidence,
        unavailable_sink,
        revision_id="rev-2",
        reason=EmergencyTrustResponseReasonCode.ACTIVE_TRUST_EVIDENCE_UNAVAILABLE,
    )
    _assert_stopped(
        failing,
        rollback_failed,
        failing_sink,
        revision_id="rev-2",
        reason=EmergencyTrustResponseReasonCode.ROLLBACK_FAILED,
    )

    # missing deployment instance: nothing to mark, the stop event records instance_state=absent
    absent = _two_revisions_serving()
    absent_sink = _RecordingSink()
    _with_sink(absent, monkeypatch, absent_sink)
    _revoke(absent, s.V1, s.V11)
    for key in [key for key in absent.state.deployment_instances if key[1] == "rev-2"]:
        del absent.state.deployment_instances[key]
    missing = _respond(absent)

    assert missing.response_reason_code is EmergencyTrustResponseReasonCode.ROLLBACK_TARGET_UNTRUSTED
    assert _STOP_EVENT in missing.audit_event_types
    stops = absent_sink.of_type(_STOP_EVENT)
    assert len(stops) == 1
    stop = stops[0]
    assert stop.attributes["instance_state"] == "absent"
    assert stop.attributes["failure_evidence_ref"] == "emergency-revocation:rollback_target_untrusted"


class _ServingChangesAfterResponse:
    """Emergency service double: the real response, then the serving revision moves on (a newer state)."""

    def __init__(self, inner: AgentEmergencyRevocationService, stack: h.D1Stack) -> None:
        self._inner = inner
        self._stack = stack

    def respond_to_current_revocation(
        self, request: AgentEmergencyRevocationRequest
    ) -> AgentEmergencyRevocationResponse:
        response = self._inner.respond_to_current_revocation(request)
        serving = self._stack.serving_store.get_serving_record(h.APP, h.ENV)
        assert serving is not None and serving.traffic_serving_revision_id is not None
        self._stack.activation.rollback(
            application_id=h.APP,
            application_environment_id=h.ENV,
            expected_current_traffic_revision_id=serving.traffic_serving_revision_id,
            expected_serving_pointer_revision=serving.serving_pointer_revision,
        )
        return response


def test_sup_03d_conflict_and_concurrent_change_do_not_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    conflicted = _two_revisions_serving()
    _revoke(conflicted, s.V11)

    def _rollback_conflicts(**_: object) -> object:
        raise RuntimeActivationConflict("stub: serving pointer moved")

    monkeypatch.setattr(conflicted.activation, "rollback", _rollback_conflicts)
    conflict = _respond(conflicted)

    moved = _two_revisions_serving()
    moved_sink = _RecordingSink()
    _with_sink(moved, monkeypatch, moved_sink)
    _revoke(moved, s.V1, s.V11)
    monkeypatch.setattr(
        moved.service,
        "_emergency_revocation_service",
        _ServingChangesAfterResponse(moved.service._emergency_revocation_service, moved),
    )
    moved_responses: list[AgentEmergencyRevocationResponse] = []
    superseded = _capture(lambda: moved_responses.append(_respond(moved)))

    assert conflict.response_reason_code is EmergencyTrustResponseReasonCode.ROLLBACK_CONFLICT
    assert _instance_state(conflicted, "rev-2") is DeploymentInstanceState.SERVING
    assert _STOP_EVENT not in conflict.audit_event_types
    assert superseded is None
    assert _instance_state(moved, "rev-2") is not DeploymentInstanceState.FAILED
    assert s.traffic_revision(moved) == "rev-1"
    assert moved_sink.of_type(_STOP_EVENT) == []
    superseded_events = moved_sink.of_type(_SUPERSEDED_EVENT)
    assert len(superseded_events) == 1
    assert _SUPERSEDED_EVENT in moved_responses[0].audit_event_types
    event = superseded_events[0]
    assert event.attributes == {
        "runtime_revision_id": "rev-2",
        "reason_code": EmergencyTrustResponseReasonCode.ROLLBACK_TARGET_UNTRUSTED.value,
        "instance_state": DeploymentInstanceState.DRAINING.value,  # the newer state drained rev-2
        "application_id": h.APP,
        "application_environment_id": h.ENV,
    }


def test_sup_03e_trusted_activation_after_stop_recovers() -> None:
    stack = _single_revision_serving()
    _revoke(stack, s.V1)
    _respond(stack)
    candidate = h.release(s.V11, contract_version="1.1.0")
    stack.register(candidate)
    stack.service.upgrade_agent(
        application_id=h.APP,
        application_environment_id=h.ENV,
        request=candidate.upgrade_request(expected_active=h.installation_id(s.V1)),
        principal=h.principal(),
    )
    h.build_revision(stack, "rev-2")

    h.activate_revision(stack, "rev-2", expected_prior="rev-1", expected_pointer=1)

    assert s.traffic_revision(stack) == "rev-2"
    assert _admit(stack, "rev-2") is not None


# --------------------------------------------------------------------------- NT-RW-SUP-04 (PDC-D1-R017)


def test_sup_04a_install_event_reaches_evidence_plane() -> None:
    trail = EvidencePlaneSecurityAuditTrail(InMemoryRuntimeEventStore())
    signer = _throwaway_signer()
    factory = _factory_stack(_non_lab_trust(signer, trail=trail))
    release = h.release(s.V1)
    _install(factory, release, _signed_record(release, signer))

    # the stop event of a revoked serving revision without a target reaches the same production sink
    stopped = _single_revision_serving()
    stopped.service._lifecycle_event_sink = factory.admin._lifecycle_event_sink  # the production-composed sink
    _revoke(stopped, s.V1)
    _respond(stopped)

    entries = trail.list_entries(tenant_id=h.TENANT)
    actions = [entry.action for entry in entries]
    assert actions[:3] == ["installation.created", "installation.verified", "installation.activated"]
    assert _STOP_EVENT in actions
    for entry in entries[:3]:
        assert entry.resource == f"installation:{h.installation_id(s.V1)}"
        assert entry.actor_id == "agent_distribution.lifecycle"
        assert entry.metadata["application_id"] == h.APP
        assert entry.metadata["application_environment_id"] == h.ENV
    report = trail.verify_chain(h.TENANT)
    assert report.intact and report.fully_chained


def test_sup_04b_missing_trail_refused_at_composition() -> None:
    complete = _non_lab_trust(_throwaway_signer())

    caught = _capture(lambda: _untyped_replace(complete, lifecycle_evidence_trail=None))

    assert isinstance(caught, ValueError) and "lifecycle_evidence_trail" in str(caught)


class _SinkBoom(RuntimeError):
    pass


def _raising_sink(events: tuple[AgentDistributionEvent, ...]) -> None:
    del events
    raise _SinkBoom("stub: evidence sink unavailable")


def test_sup_04c_sink_failure_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    # failure path: the upgrade fails after verification; the original error stays, the sink error is its cause
    upgrading = s.stack_with_validated_revision()
    _with_sink(upgrading, monkeypatch, _raising_sink)

    def _boom(installation: object) -> object:
        raise RuntimeError("resolve-failed")

    monkeypatch.setattr(upgrading.service, "_resolve_package_contracts", _boom)
    candidate = h.release(s.V11, contract_version="1.1.0")
    upgrading.register(candidate)
    original = _capture(
        lambda: upgrading.service.upgrade_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=candidate.upgrade_request(expected_active=h.installation_id(s.V1)),
            principal=h.principal(),
        )
    )

    assert isinstance(original, RuntimeError) and str(original) == "resolve-failed"
    assert isinstance(original.__cause__, _SinkBoom)

    # success path: the install is committed, the caller sees a typed error with the sink error chained
    from intergrax.agent_distribution.admin_service import LifecycleEvidenceSinkError

    installing = h.build_stack()
    _with_sink(installing, monkeypatch, _raising_sink)
    first = h.release(s.V1)
    installing.register(first)
    success_path = _capture(
        lambda: installing.service.install_agent(
            application_id=h.APP,
            application_environment_id=h.ENV,
            request=first.install_request(),
            principal=h.principal(),
        )
    )

    assert isinstance(success_path, LifecycleEvidenceSinkError)
    assert isinstance(success_path.__cause__, _SinkBoom)
    persisted = installing.installation_store.get_installation(h.installation_id(s.V1))
    assert persisted is not None and persisted.installation_state is InstallationState.INSTALLED_ACTIVE
