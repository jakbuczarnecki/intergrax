# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.core.qualification import QualificationStatus

from intergrax.agent_distribution.activation import (
    FakeRuntimeServingProjectionCoordinator,
)
from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    BindAgentRequest,
    BuildApplicationRevisionRequest,
)
from intergrax.agent_distribution.agent_discovery import (
    AgentDiscoveryCandidate,
    AgentDiscoveryRequest,
    AgentDiscoveryStrategyId,
    StaticAgentDiscoveryStrategy,
    project_package_contract_capabilities,
    project_to_capability_candidate,
)
from intergrax.agent_distribution.agent_project_metadata import (
    AgentPackageContractDeclaration,
)
from intergrax.agent_distribution.agent_selection import (
    DeterministicIdentitySelectionStrategy,
    SelectionOutcome,
    build_agent_selection_request,
    require_selected_identity,
)
from intergrax.agent_distribution.capability_matching import CapabilityMatcher
from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    AgentDiscoveryCandidateIdentity,
    CatalogPackageResolution,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.dependency import RepositoryDependencyDeclaration
from intergrax.agent_distribution.dynamic_acquisition import (
    CatalogSourceProviderRegistry,
    DynamicAgentAcquisitionActivationError,
    DynamicAgentAcquisitionInstallIntent,
    DynamicAgentAcquisitionOutcome,
    DynamicAgentAcquisitionRequest,
    DynamicAgentAcquisitionResolutionError,
    DynamicAgentAcquisitionService,
    assert_exact_discovery_candidate_match,
    resolve_discovery_candidate_exact,
)
from intergrax.agent_distribution.errors import AgentPackageTrustError
from intergrax.agent_distribution.federated_discovery import (
    FederatedAgentDiscoveryStrategy,
)
from intergrax.agent_distribution.identity import AgentPackageCandidate
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.task_capability_resolution import (
    build_deterministic_task_capability_resolver,
    build_task_capability_resolution_request,
    build_task_capability_rule,
)
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    AdminStack,
    _APP,
    _ARTIFACT,
    _DIGEST,
    _ENV,
    _META_REF,
    _PACKAGE,
    _PACKAGE_ID,
    admin_test_principal,
    build_admin_stack,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SOURCE_ID = "builtin-1"
_CATALOG_ENTRY_ID = "cat-researcher"
_SLOT = "slot-search"
_INSTALL_ID = "inst-1"
_BINDING_ID = "bind-search"
_LOGICAL_AGENT = "researcher"


def _source(
    kind: CatalogProviderKind = CatalogProviderKind.BUILTIN,
) -> CatalogSourceIdentity:
    return CatalogSourceIdentity(catalog_source_id=_SOURCE_ID, provider_kind=kind)


def _candidate(
    *,
    version: str = "1.0.0",
    digest: str | None = _DIGEST,
) -> AgentPackageCandidate:
    return AgentPackageCandidate(
        distribution_package_id=_PACKAGE_ID,
        package_version=version,
        package_digest=digest,
    )


def _identity(
    *,
    version: str = "1.0.0",
    digest: str | None = _DIGEST,
) -> AgentDiscoveryCandidateIdentity:
    return AgentDiscoveryCandidateIdentity(
        source=_source(),
        package=_candidate(version=version, digest=digest),
    )


def _catalog_entry() -> AgentCatalogEntry:
    return AgentCatalogEntry(
        catalog_entry_id=_CATALOG_ENTRY_ID,
        catalog_source=_source(),
        display_name="Researcher",
        package_id_line=_PACKAGE_ID,
    )


def _resolution(
    *,
    version: str = "1.0.0",
    digest: str | None = _DIGEST,
) -> CatalogPackageResolution:
    return CatalogPackageResolution(
        entry=_catalog_entry(),
        package_candidate=_candidate(version=version, digest=digest),
        artifact_locator="catalog://artifact/researcher",
    )


class _ExactCatalog:
    def __init__(
        self,
        *,
        entry: AgentCatalogEntry,
        resolution_by_selector: dict[str, CatalogPackageResolution],
    ) -> None:
        self._entry = entry
        self._resolution_by_selector = resolution_by_selector

    @property
    def catalog_source_id(self) -> str:
        return self._entry.catalog_source.catalog_source_id

    def list_entries(self, filters: object | None = None) -> list[AgentCatalogEntry]:
        del filters
        return [self._entry]

    def resolve_package(
        self,
        entry: AgentCatalogEntry,
        *,
        version_selector: str,
    ) -> CatalogPackageResolution:
        if entry.catalog_entry_id != self._entry.catalog_entry_id:
            raise ValueError("unexpected catalog entry")
        try:
            return self._resolution_by_selector[version_selector]
        except KeyError as exc:
            raise ValueError(f"no resolution for selector {version_selector}") from exc

    def health(self) -> None:
        return None


def _trust() -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=_DIGEST,
        publisher_identity_ref="publisher:acme",
        source_provider_id="builtin",
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id="evidence:service:0",
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
        ),
    )


def _build_request(
    revision_id: str, *, mutation_id: str = "mut-build"
) -> BuildApplicationRevisionRequest:
    return BuildApplicationRevisionRequest(
        mutation_id=mutation_id,
        runtime_revision_id=revision_id,
        application_release_id="rel-1",
        platform_version="0.1.0",
        python_version="3.12",
        source_context_root="/tmp/src",
        output_root="/tmp/out",
        application_source_root="applications/app-a",
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
    pointer_revision: int = 0,
    prior_revision_id: str | None = None,
    mutation_id: str = "mut-activate",
) -> ActivateRuntimeRevisionRequest:
    return ActivateRuntimeRevisionRequest(
        mutation_id=mutation_id,
        runtime_revision_id=revision_id,
        artifact_locator="test://artifact",
        expected_artifact_digest=_ARTIFACT,
        expected_serving_pointer_revision=pointer_revision,
        expected_prior_traffic_revision_id=prior_revision_id,
    )


def _acquisition_request(
    revision_id: str,
    *,
    identity: AgentDiscoveryCandidateIdentity | None = None,
    install_mutation_id: str = "mut-install",
    bind_mutation_id: str = "mut-bind",
    pointer_revision: int = 0,
    prior_revision_id: str | None = None,
) -> DynamicAgentAcquisitionRequest:
    return DynamicAgentAcquisitionRequest(
        selected_identity=identity or _identity(),
        application_id=_APP,
        application_environment_id=_ENV,
        catalog_entry_id=_CATALOG_ENTRY_ID,
        install=DynamicAgentAcquisitionInstallIntent(
            mutation_id=install_mutation_id,
            installation_id=_INSTALL_ID,
            installation_slot_id=_SLOT,
            artifact_store_ref="store://artifacts/inst-1",
            trust_record=_trust(),
            agent_project_metadata_ref=_META_REF,
        ),
        bind=BindAgentRequest(
            mutation_id=bind_mutation_id,
            application_binding_id=_BINDING_ID,
            logical_agent_id=_LOGICAL_AGENT,
            installation_slot_id=_SLOT,
            enablement=True,
        ),
        build=_build_request(revision_id),
        activate=_activate_request(
            revision_id,
            pointer_revision=pointer_revision,
            prior_revision_id=prior_revision_id,
        ),
    )


@dataclass
class AcquisitionHarness:
    stack: AdminStack
    service: DynamicAgentAcquisitionService
    catalog: _ExactCatalog


def build_acquisition_harness(
    *,
    resolution_by_selector: dict[str, CatalogPackageResolution] | None = None,
) -> AcquisitionHarness:
    stack = build_admin_stack(with_catalog=False)
    catalog = _ExactCatalog(
        entry=_catalog_entry(),
        resolution_by_selector=resolution_by_selector or {"1.0.0": _resolution()},
    )
    registry = CatalogSourceProviderRegistry({_SOURCE_ID: catalog})
    service = DynamicAgentAcquisitionService(
        catalog_registry=registry,
        lifecycle=stack.service,
    )
    return AcquisitionHarness(stack=stack, service=service, catalog=catalog)


def test_exact_resolution_proceeds() -> None:
    harness = build_acquisition_harness()
    resolution = resolve_discovery_candidate_exact(
        identity=_identity(),
        catalog_entry_id=_CATALOG_ENTRY_ID,
        registry=harness.service._catalog_registry,
    )
    assert resolution.package_candidate.distribution_package_id == _PACKAGE_ID
    assert resolution.package_candidate.package_version == "1.0.0"


def test_resolution_version_mismatch_fails_closed() -> None:
    harness = build_acquisition_harness(
        resolution_by_selector={"1.0.0": _resolution(version="1.0.1")},
    )
    with pytest.raises(DynamicAgentAcquisitionResolutionError, match="version"):
        resolve_discovery_candidate_exact(
            identity=_identity(version="1.0.0"),
            catalog_entry_id=_CATALOG_ENTRY_ID,
            registry=harness.service._catalog_registry,
        )


def test_resolution_digest_mismatch_fails_closed() -> None:
    other_digest = "sha256:" + ("b" * 64)
    harness = build_acquisition_harness(
        resolution_by_selector={"1.0.0": _resolution(digest=other_digest)},
    )
    with pytest.raises(DynamicAgentAcquisitionResolutionError, match="digest"):
        resolve_discovery_candidate_exact(
            identity=_identity(digest=_DIGEST),
            catalog_entry_id=_CATALOG_ENTRY_ID,
            registry=harness.service._catalog_registry,
        )


def test_source_mismatch_fails_closed() -> None:
    resolution = _resolution()
    mismatched_entry = resolution.entry.model_copy(
        update={
            "catalog_source": CatalogSourceIdentity(
                catalog_source_id="other-source",
                provider_kind=CatalogProviderKind.OFFICIAL_CATALOG,
            ),
        },
    )
    mismatched = resolution.model_copy(update={"entry": mismatched_entry})
    with pytest.raises(DynamicAgentAcquisitionResolutionError, match="source"):
        assert_exact_discovery_candidate_match(
            identity=_identity(),
            resolution=mismatched,
        )


def test_install_bind_and_activate_through_canonical_services() -> None:
    harness = build_acquisition_harness()
    result = harness.service.acquire(
        _acquisition_request("rev-acquire-1"),
        principal=admin_test_principal(),
    )
    assert result.outcome is DynamicAgentAcquisitionOutcome.ACQUIRED_ACTIVE
    assert result.resolved_package_identity == _PACKAGE
    assert result.installation_reused is False
    assert result.binding_reused is False
    installation = harness.stack.service._installation_store.get_installation(
        _INSTALL_ID
    )
    assert installation is not None
    binding = harness.stack.service._binding_store.get_binding(_BINDING_ID)
    assert binding is not None
    assert binding.enablement is True
    revision = harness.stack.service._revision_store.get_revision("rev-acquire-1")
    assert revision is not None
    assert revision.revision_state is RuntimeRevisionState.ACTIVE
    serving = harness.stack.service.inspect_serving(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert serving.traffic_serving_revision_id == "rev-acquire-1"


def test_idempotent_reacquisition_reuses_install_and_binding() -> None:
    harness = build_acquisition_harness()
    first = harness.service.acquire(
        _acquisition_request("rev-acquire-1"),
        principal=admin_test_principal(),
    )
    assert first.installation_reused is False
    assert first.binding_reused is False
    second = harness.service.acquire(
        _acquisition_request(
            "rev-acquire-2",
            install_mutation_id="mut-install-2",
            bind_mutation_id="mut-bind-2",
            pointer_revision=1,
            prior_revision_id="rev-acquire-1",
        ),
        principal=admin_test_principal(),
    )
    assert second.installation_reused is True
    assert second.binding_reused is True
    assert second.outcome is DynamicAgentAcquisitionOutcome.ACQUIRED_ACTIVE
    assert (
        len(
            harness.stack.service.list_installed(
                application_id=_APP, application_environment_id=_ENV
            ).installations
        )
        == 1
    )
    assert (
        len(
            harness.stack.service.list_bindings(
                application_id=_APP, application_environment_id=_ENV
            ).bindings
        )
        == 1
    )

    third = harness.service.acquire(
        _acquisition_request(
            "rev-acquire-2",
            install_mutation_id="mut-install-3",
            bind_mutation_id="mut-bind-3",
            pointer_revision=2,
            prior_revision_id="rev-acquire-2",
        ),
        principal=admin_test_principal(),
    )
    assert third.installation_reused is True
    assert third.binding_reused is True
    assert third.outcome is DynamicAgentAcquisitionOutcome.ALREADY_ACTIVE


def test_failed_prepare_preserves_active_revision() -> None:
    harness = build_acquisition_harness()
    harness.service.acquire(
        _acquisition_request("rev-active"),
        principal=admin_test_principal(),
    )
    coordinator = FakeRuntimeServingProjectionCoordinator()
    coordinator.fail_prepare("rev-candidate-fail")
    harness.stack.service._activation_service._projection_coordinator = coordinator
    with pytest.raises(DynamicAgentAcquisitionActivationError):
        harness.service.acquire(
            _acquisition_request(
                "rev-candidate-fail",
                install_mutation_id="mut-install-2",
                bind_mutation_id="mut-bind-2",
                pointer_revision=1,
                prior_revision_id="rev-active",
            ),
            principal=admin_test_principal(),
        )
    serving = harness.stack.service.inspect_serving(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert serving.traffic_serving_revision_id == "rev-active"


def test_trust_rejection_propagates_through_install_path() -> None:
    harness = build_acquisition_harness()
    request = _acquisition_request("rev-trust")
    request = request.model_copy(
        update={
            "install": request.install.model_copy(
                update={
                    "trust_record": _trust().model_copy(
                        update={"package_digest": "sha256:" + ("c" * 64)},
                    ),
                },
            ),
        },
    )
    with pytest.raises(AgentPackageTrustError):
        harness.service.acquire(request, principal=admin_test_principal())


def _discovery_candidate(capability_ids: tuple[str, ...]) -> AgentDiscoveryCandidate:
    return AgentDiscoveryCandidate(
        identity=_identity(),
        capabilities=project_package_contract_capabilities(
            AgentPackageContractDeclaration(
                contract_id="contract.v1",
                contract_version="1",
                capabilities=capability_ids,
            ),
        ),
        catalog_entry_id=_CATALOG_ENTRY_ID,
    )


def test_task_pipeline_through_acquisition_to_active() -> None:
    harness = build_acquisition_harness()
    resolver = build_deterministic_task_capability_resolver(
        rules=(
            build_task_capability_rule(
                rule_id="rule.document.ocr.v1",
                task_kind="document.ocr",
                required=("document.ocr",),
                rule_version="1",
            ),
        ),
    )
    resolution = resolver.resolve(
        build_task_capability_resolution_request(task_kind="document.ocr"),
    )
    discovery = FederatedAgentDiscoveryStrategy(
        strategies=(
            StaticAgentDiscoveryStrategy(
                strategy_id=AgentDiscoveryStrategyId(value="static.test"),
                candidates=(_discovery_candidate(("document.ocr",)),),
            ),
        ),
    )
    discovered = discovery.discover(
        AgentDiscoveryRequest(requirement=resolution.capability_requirement),
    )
    matcher = CapabilityMatcher()
    match = matcher.match(
        requirement=resolution.capability_requirement,
        candidate=project_to_capability_candidate(discovered.candidates[0]),
    )
    decision = DeterministicIdentitySelectionStrategy().select(
        build_agent_selection_request(
            requirement=resolution.capability_requirement,
            eligible_matches=(match,) if match.eligible else (),
        ),
    )
    assert decision.outcome is SelectionOutcome.SELECTED
    selected = require_selected_identity(decision)
    result = harness.service.acquire(
        _acquisition_request("rev-pipeline-1", identity=selected),
        principal=admin_test_principal(),
    )
    assert result.outcome is DynamicAgentAcquisitionOutcome.ACQUIRED_ACTIVE
    assert result.selected_identity == selected
    assert result.resolved_package_identity.package_digest == _DIGEST
    serving = harness.stack.service.inspect_serving(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert serving.traffic_serving_revision_id == "rev-pipeline-1"


def test_dynamic_acquisition_module_has_no_registry_or_nexus_coupling() -> None:
    import intergrax.agent_distribution.dynamic_acquisition as module

    source_path = Path(module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    forbidden_prefixes = (
        "intergrax.harness",
        "intergrax.nexus",
        "agents",
        "applications",
    )
    violations = sorted(
        imported
        for imported in imported_modules
        if any(
            imported == prefix or imported.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        )
    )
    assert not violations, f"forbidden imports: {violations}"
