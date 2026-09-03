# © Artur Czarnecki. All rights reserved.

"""ADR-AGENT-007 Phase 4E — historical projection isolation across revisions."""

from __future__ import annotations

import inspect
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.agent_distribution.admin_models import (
    BindAgentRequest,
    BuildApplicationRevisionRequest,
    InstallAgentRequest,
    SetAgentEnablementRequest,
    UpdateAgentBindingRequest,
)
from intergrax.agent_distribution.agent_project_metadata import AgentProjectMetadata
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.dependency import RepositoryDependencyDeclaration
from intergrax.agent_distribution.effective_roster_authority import (
    EffectiveRosterAuthorityService,
)
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.materialization import (
    MaterializationInput,
    MaterializationOutput,
)
from intergrax.agent_distribution.materialization_service import (
    RuntimeMaterializationService,
)
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_context_staging import (
    RUNTIME_LOCK_MANIFEST_FILENAME,
    directory_content_digest,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    ProductionRegistryProjectionInputError,
    build_production_registry_projection_for_revision,
    build_production_registry_projection_input_bundle_for_revision,
    production_test_artifact_locator,
)
from intergrax.applications._shared.registry_projection import (
    RegistryProjectionInputBundle,
)
from intergrax.applications._shared.registry_projection_authority_resolver import (
    RegistryProjectionAuthorityResolver,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.core.qualification import QualificationStatus
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    AdminStack,
    _ENV,
    _META_REF,
    admin_test_principal,
    build_admin_stack,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APP = "app_a"

_RELEASE = "rel-1"
_LOGICAL_AGENT_ID = "researcher"
_SLOT_ID = "slot-search"
_BINDING_ID = "bind-search"

_DIGEST_A = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)
_PACKAGE_ID_A = "package-a"
_PACKAGE_ID_B = "package-b"
_INSTALL_A = "inst-a"
_INSTALL_B = "inst-b"

_CONFIG_A = {"model": {"temperature": 0.1}, "limits": {"rpm": 10}}
_CONFIG_B = {"model": {"temperature": 0.8}, "limits": {"rpm": 100}}

_FACTORY_REF_A = AgentBindingFactoryReference(
    factory_path="example_agent.factory_a.build_agent",
)
_FACTORY_REF_B = AgentBindingFactoryReference(
    factory_path="example_agent.factory_b.build_agent",
)

_REVISION_N = "rev-phase4e-n"
_REVISION_N_PLUS_1 = "rev-phase4e-n-plus-1"


@dataclass(frozen=True, slots=True)
class _HistoricalScenario:
    installation_id: str
    package_digest: str
    distribution_package_id: str
    factory_reference: AgentBindingFactoryReference
    merged_config: dict[str, object]
    effective_enablement: bool = True


_SCENARIO_A = _HistoricalScenario(
    installation_id=_INSTALL_A,
    package_digest=_DIGEST_A,
    distribution_package_id=_PACKAGE_ID_A,
    factory_reference=_FACTORY_REF_A,
    merged_config=_CONFIG_A,
)
_SCENARIO_B = _HistoricalScenario(
    installation_id=_INSTALL_B,
    package_digest=_DIGEST_B,
    distribution_package_id=_PACKAGE_ID_B,
    factory_reference=_FACTORY_REF_B,
    merged_config=_CONFIG_B,
)


class _Phase4eVenvBundleMaterializer:
    topology = MaterializationTopology.VENV_BUNDLE
    materializer_id = "intergrax.phase4e-historical-proof"
    materializer_version = "1.0.0"

    def __init__(self, artifact_root_base: Path) -> None:
        self._artifact_root_base = artifact_root_base

    def materialize(
        self, materialization_input: MaterializationInput
    ) -> MaterializationOutput:
        revision_id = materialization_input.runtime_revision.runtime_revision_id
        artifact_root = self._artifact_root_base / revision_id
        site_packages = artifact_root / "site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        package_dir = site_packages / "example_agent"
        package_dir.mkdir(exist_ok=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")

        for entry in materialization_input.effective_roster.entries:
            if not entry.effective_enablement:
                continue
            factory_reference = entry.factory_reference
            if factory_reference is None or factory_reference.factory_path is None:
                continue
            module_name, function_name = factory_reference.factory_path.rsplit(".", 1)
            relative_module = module_name.removeprefix("example_agent.")
            marker = entry.package_digest[-8:]
            (package_dir / f"{relative_module}.py").write_text(
                textwrap.dedent(
                    f"""
                    from echo.echo_agent import EchoAgent

                    MARKER = {marker!r}

                    def {function_name}(ctx, binding):
                        return EchoAgent()
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

        lock = materialization_input.materialized_runtime_lock
        (artifact_root / RUNTIME_LOCK_MANIFEST_FILENAME).write_text(
            json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (artifact_root / ".intergrax-runtime-graph.json").write_text(
            "{}\n", encoding="utf-8"
        )
        digest = directory_content_digest(artifact_root)
        return MaterializationOutput(
            materialization_artifact_digest=digest,
            artifact_locator=production_test_artifact_locator(artifact_root),
            health_check_evidence_ref=production_test_artifact_locator(artifact_root),
            runtime_graph_manifest_path=".intergrax-runtime-graph.json",
            topology=self.topology,
        )


def _trust_record(package_digest: str) -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=package_digest,
        publisher_identity_ref="publisher:acme",
        source_provider_id="builtin",
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id=f"evidence:{package_digest[-8:]}",
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
        ),
    )


def _package_identity(scenario: _HistoricalScenario) -> AgentPackageIdentity:
    return AgentPackageIdentity(
        distribution_package_id=scenario.distribution_package_id,
        package_version="1.0.0",
        package_digest=scenario.package_digest,
    )


def _manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id=_APP,
        name="Phase 4E Historical Proof",
        agents=[
            AgentBinding(
                import_path="echo.echo_agent.EchoAgent",
                contract_id=_LOGICAL_AGENT_ID,
                builder_key=_LOGICAL_AGENT_ID,
            )
        ],
    )


def _build_request(
    revision_id: str, *, mutation_id: str
) -> BuildApplicationRevisionRequest:
    return BuildApplicationRevisionRequest(
        mutation_id=mutation_id,
        runtime_revision_id=revision_id,
        application_release_id=_RELEASE,
        platform_version="0.1.0",
        python_version="3.12",
        source_context_root="/tmp/src",
        output_root="/tmp/out",
        application_source_root="applications/app_a",
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        repository_declaration=RepositoryDependencyDeclaration(
            application_release_id=_RELEASE,
            direct_dependencies=(),
        ),
        resolver_algorithm_id="intergrax.phase4e-resolver",
        resolver_algorithm_version="1.0.0",
    )


def _build_phase4e_stack(tmp_path: Path) -> AdminStack:
    stack = build_admin_stack()
    stack.service._metadata_provider._records[_META_REF] = AgentProjectMetadata(
        distribution_package_id=_PACKAGE_ID_A,
        dependencies=(),
    )
    stack.service._metadata_provider._records["meta://package-b"] = (
        AgentProjectMetadata(
            distribution_package_id=_PACKAGE_ID_B,
            dependencies=(),
        )
    )
    stack.service._materialization_service = RuntimeMaterializationService(
        {MaterializationTopology.VENV_BUNDLE: _Phase4eVenvBundleMaterializer(tmp_path)}
    )
    return stack


def _install_agent(
    stack: AdminStack,
    scenario: _HistoricalScenario,
    *,
    mutation_id: str,
    metadata_ref: str = _META_REF,
) -> None:
    stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=InstallAgentRequest(
            mutation_id=mutation_id,
            installation_id=scenario.installation_id,
            installation_slot_id=_SLOT_ID,
            package_identity=_package_identity(scenario),
            artifact_store_ref=f"store://artifacts/{scenario.installation_id}",
            trust_record=_trust_record(scenario.package_digest),
            agent_project_metadata_ref=metadata_ref,
        ),
        principal=admin_test_principal(),
    )


def _bind_agent(
    stack: AdminStack, scenario: _HistoricalScenario, *, mutation_id: str
) -> None:
    stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=BindAgentRequest(
            mutation_id=mutation_id,
            application_binding_id=_BINDING_ID,
            logical_agent_id=_LOGICAL_AGENT_ID,
            installation_slot_id=_SLOT_ID,
            config=scenario.merged_config,
            factory_reference=scenario.factory_reference,
            enablement=True,
        ),
        principal=admin_test_principal(),
    )


def _build_revision(stack: AdminStack, revision_id: str, *, mutation_id: str):
    return stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request(revision_id, mutation_id=mutation_id),
        principal=admin_test_principal(),
    )


def _setup_state_a(stack: AdminStack) -> None:
    _install_agent(stack, _SCENARIO_A, mutation_id="mut-phase4e-install-a")
    _bind_agent(stack, _SCENARIO_A, mutation_id="mut-phase4e-bind-a")


def _mutate_to_state_b(stack: AdminStack) -> None:
    _install_agent(
        stack,
        _SCENARIO_B,
        mutation_id="mut-phase4e-install-b",
        metadata_ref="meta://package-b",
    )
    binding = stack.service._binding_store.get_binding(_BINDING_ID)
    assert binding is not None
    updated_binding = binding.model_copy(
        update={
            "factory_reference": _SCENARIO_B.factory_reference,
            "binding_revision": binding.binding_revision + 1,
        }
    )
    stack.service._binding_store.persist_binding(
        updated_binding,
        expected_revision=binding.binding_revision,
    )
    stack.service.update_binding_config(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id=_BINDING_ID,
        request=UpdateAgentBindingRequest(
            mutation_id="mut-phase4e-config-b",
            expected_revision=updated_binding.binding_revision,
            config=_SCENARIO_B.merged_config,
        ),
        principal=admin_test_principal(),
    )
    stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id=_BINDING_ID,
        request=SetAgentEnablementRequest(
            mutation_id="mut-phase4e-enable-b",
            expected_revision=updated_binding.binding_revision + 1,
        ),
        principal=admin_test_principal(),
    )


def _authority_resolver(stack: AdminStack) -> RegistryProjectionAuthorityResolver:
    return RegistryProjectionAuthorityResolver(
        revision_store=stack.service._revision_store,
        effective_roster_authority=EffectiveRosterAuthorityService(
            snapshot_store=stack.effective_roster_snapshot_store,
        ),
        lock_store=stack.service._lock_store,
        materialization_store=stack.materialization_store,
    )


def _project_bundle(
    stack: AdminStack,
    *,
    runtime_revision_id: str,
) -> RegistryProjectionInputBundle:
    manifest = _manifest()
    return build_production_registry_projection_input_bundle_for_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=runtime_revision_id,
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        authority=_authority_resolver(stack),
    )


def _enabled_entry(bundle: RegistryProjectionInputBundle) -> EffectiveRosterEntry:
    enabled = [
        entry for entry in bundle.effective_roster.entries if entry.effective_enablement
    ]
    assert len(enabled) == 1
    return enabled[0]


def _assert_projection_matches_scenario(
    bundle: RegistryProjectionInputBundle,
    *,
    scenario: _HistoricalScenario,
    expected_roster_revision_id: str,
    expected_revision_id: str,
    other: _HistoricalScenario,
) -> None:
    revision = bundle.runtime_revision
    entry = _enabled_entry(bundle)

    assert revision.runtime_revision_id == expected_revision_id
    assert revision.application_id == _APP
    assert revision.application_environment_id == _ENV
    assert revision.application_release_id == _RELEASE
    assert revision.effective_roster_revision_id == expected_roster_revision_id
    assert entry.logical_agent_id == _LOGICAL_AGENT_ID
    assert entry.installation_slot_id == _SLOT_ID
    assert entry.active_installation_id == scenario.installation_id
    assert entry.package_digest == scenario.package_digest
    assert entry.distribution_package_id == scenario.distribution_package_id
    assert entry.merged_config == scenario.merged_config
    assert entry.factory_reference == scenario.factory_reference
    assert entry.effective_enablement == scenario.effective_enablement

    assert entry.active_installation_id != other.installation_id
    assert entry.package_digest != other.package_digest
    assert entry.distribution_package_id != other.distribution_package_id
    assert entry.merged_config != other.merged_config
    assert entry.factory_reference != other.factory_reference


def _assert_lock_and_materialization(
    stack: AdminStack,
    bundle: RegistryProjectionInputBundle,
    *,
    expected_revision_id: str,
    other_revision_id: str,
) -> None:
    revision = bundle.runtime_revision
    authority = _authority_resolver(stack)
    resolved = authority.require_for_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=expected_revision_id,
    )
    other_revision = stack.service._revision_store.get_revision(other_revision_id)
    assert other_revision is not None

    assert (
        resolved.materialized_runtime_lock.lock_id
        == revision.materialized_runtime_lock_id
    )
    assert (
        resolved.materialized_runtime_lock.lock_digest
        == revision.materialized_runtime_lock_digest
    )
    assert resolved.runtime_materialization.runtime_revision_id == expected_revision_id
    assert (
        resolved.runtime_materialization.materialization_artifact_digest
        == revision.materialization_artifact_digest
    )
    assert (
        bundle.materialization_artifact_digest
        == revision.materialization_artifact_digest
    )
    assert (
        resolved.materialized_runtime_lock.lock_id
        != other_revision.materialized_runtime_lock_id
    )
    assert (
        resolved.runtime_materialization.runtime_revision_id
        != other_revision.runtime_revision_id
    )


@pytest.fixture
def phase4e_stack(tmp_path: Path) -> AdminStack:
    return _build_phase4e_stack(tmp_path)


def test_public_projection_api_has_no_effective_roster_param() -> None:
    signature = inspect.signature(
        build_production_registry_projection_input_bundle_for_revision
    )
    assert "effective_roster" not in signature.parameters
    projection_signature = inspect.signature(
        build_production_registry_projection_for_revision
    )
    assert "effective_roster" not in projection_signature.parameters


def test_historical_projection_isolation_across_n_and_n_plus_one(
    phase4e_stack: AdminStack,
) -> None:
    stack = phase4e_stack
    _setup_state_a(stack)
    built_n = _build_revision(stack, _REVISION_N, mutation_id="mut-phase4e-build-n")
    snapshot_a = stack.effective_roster_snapshot_store.get_by_revision(
        built_n.effective_roster_revision_id
    )
    assert snapshot_a is not None
    snapshot_a_copy = snapshot_a.model_copy(deep=True)

    _mutate_to_state_b(stack)
    built_n_plus_1 = _build_revision(
        stack,
        _REVISION_N_PLUS_1,
        mutation_id="mut-phase4e-build-n-plus-1",
    )
    snapshot_b = stack.effective_roster_snapshot_store.get_by_revision(
        built_n_plus_1.effective_roster_revision_id
    )
    assert snapshot_b is not None

    assert (
        built_n.effective_roster_revision_id
        != built_n_plus_1.effective_roster_revision_id
    )
    assert (
        stack.effective_roster_snapshot_store.get_by_revision(
            built_n.effective_roster_revision_id
        )
        is not None
    )
    assert (
        stack.effective_roster_snapshot_store.get_by_revision(
            built_n_plus_1.effective_roster_revision_id
        )
        is not None
    )

    reloaded_a = stack.effective_roster_snapshot_store.get_by_revision(
        built_n.effective_roster_revision_id
    )
    assert reloaded_a == snapshot_a_copy

    bundle_n_plus_1 = _project_bundle(stack, runtime_revision_id=_REVISION_N_PLUS_1)
    bundle_n = _project_bundle(stack, runtime_revision_id=_REVISION_N)

    _assert_projection_matches_scenario(
        bundle_n,
        scenario=_SCENARIO_A,
        expected_roster_revision_id=built_n.effective_roster_revision_id,
        expected_revision_id=_REVISION_N,
        other=_SCENARIO_B,
    )
    _assert_projection_matches_scenario(
        bundle_n_plus_1,
        scenario=_SCENARIO_B,
        expected_roster_revision_id=built_n_plus_1.effective_roster_revision_id,
        expected_revision_id=_REVISION_N_PLUS_1,
        other=_SCENARIO_A,
    )
    _assert_lock_and_materialization(
        stack,
        bundle_n,
        expected_revision_id=_REVISION_N,
        other_revision_id=_REVISION_N_PLUS_1,
    )
    _assert_lock_and_materialization(
        stack,
        bundle_n_plus_1,
        expected_revision_id=_REVISION_N_PLUS_1,
        other_revision_id=_REVISION_N,
    )

    def _failing_build_roster(**kwargs: object) -> EffectiveRoster:
        del kwargs
        raise AssertionError("current desired-state roster lookup must not be used")

    stack.service._build_roster = _failing_build_roster  # type: ignore[method-assign]
    bundle_n_after_guard = _project_bundle(stack, runtime_revision_id=_REVISION_N)
    _assert_projection_matches_scenario(
        bundle_n_after_guard,
        scenario=_SCENARIO_A,
        expected_roster_revision_id=built_n.effective_roster_revision_id,
        expected_revision_id=_REVISION_N,
        other=_SCENARIO_B,
    )

    projection_n = build_production_registry_projection_for_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=_REVISION_N,
        manifest=_manifest(),
        build_context=ApplicationBuildContext.for_manifest(_manifest()),
        authority=_authority_resolver(stack),
    )
    assert projection_n.agent_registry.list_agent_ids() == [_LOGICAL_AGENT_ID]


def test_projection_order_independence_n_plus_one_before_n(
    phase4e_stack: AdminStack,
) -> None:
    stack = phase4e_stack
    _setup_state_a(stack)
    built_n = _build_revision(stack, _REVISION_N, mutation_id="mut-phase4e-order-n")
    _mutate_to_state_b(stack)
    built_n_plus_1 = _build_revision(
        stack,
        _REVISION_N_PLUS_1,
        mutation_id="mut-phase4e-order-n-plus-1",
    )

    bundle_n_plus_1 = _project_bundle(stack, runtime_revision_id=_REVISION_N_PLUS_1)
    bundle_n = _project_bundle(stack, runtime_revision_id=_REVISION_N)

    _assert_projection_matches_scenario(
        bundle_n,
        scenario=_SCENARIO_A,
        expected_roster_revision_id=built_n.effective_roster_revision_id,
        expected_revision_id=_REVISION_N,
        other=_SCENARIO_B,
    )
    _assert_projection_matches_scenario(
        bundle_n_plus_1,
        scenario=_SCENARIO_B,
        expected_roster_revision_id=built_n_plus_1.effective_roster_revision_id,
        expected_revision_id=_REVISION_N_PLUS_1,
        other=_SCENARIO_A,
    )


def test_missing_historical_snapshot_fails_closed_after_n_plus_one(
    phase4e_stack: AdminStack,
) -> None:
    stack = phase4e_stack
    _setup_state_a(stack)
    built_n = _build_revision(stack, _REVISION_N, mutation_id="mut-phase4e-negative-n")
    _mutate_to_state_b(stack)
    _build_revision(
        stack, _REVISION_N_PLUS_1, mutation_id="mut-phase4e-negative-n-plus-1"
    )

    roster_revision_id = built_n.effective_roster_revision_id
    del stack.state.effective_roster_snapshots[roster_revision_id]

    with pytest.raises(
        ProductionRegistryProjectionInputError,
        match="canonical effective roster snapshot authority",
    ):
        _project_bundle(stack, runtime_revision_id=_REVISION_N)


def test_missing_canonical_lock_n_fails_closed_while_n_plus_one_exists(
    phase4e_stack: AdminStack,
) -> None:
    stack = phase4e_stack
    _setup_state_a(stack)
    built_n = _build_revision(stack, _REVISION_N, mutation_id="mut-phase4e-lock-n")
    _mutate_to_state_b(stack)
    _build_revision(stack, _REVISION_N_PLUS_1, mutation_id="mut-phase4e-lock-n-plus-1")

    lock_id = built_n.materialized_runtime_lock_id
    assert lock_id is not None
    del stack.state.locks[lock_id]

    with pytest.raises(
        ProductionRegistryProjectionInputError,
        match="canonical materialized runtime lock not found",
    ):
        _project_bundle(stack, runtime_revision_id=_REVISION_N)


def test_missing_materialization_n_fails_closed_while_n_plus_one_exists(
    phase4e_stack: AdminStack,
) -> None:
    stack = phase4e_stack
    _setup_state_a(stack)
    _build_revision(stack, _REVISION_N, mutation_id="mut-phase4e-mat-n")
    _mutate_to_state_b(stack)
    _build_revision(stack, _REVISION_N_PLUS_1, mutation_id="mut-phase4e-mat-n-plus-1")

    del stack.state.materializations[_REVISION_N]

    with pytest.raises(
        ProductionRegistryProjectionInputError,
        match="missing canonical materialization record",
    ):
        _project_bundle(stack, runtime_revision_id=_REVISION_N)
