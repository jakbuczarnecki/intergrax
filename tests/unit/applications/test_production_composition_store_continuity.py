# © Artur Czarnecki. All rights reserved.

"""AC-3-FIX-2 / AGENT-CONSOLIDATION-3-ARCH production AP store composition continuity proofs."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.agent_distribution.activation import ActivationService
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter
from intergrax.agent_distribution.in_memory_stores import (
    InMemoryApplicationEnvironmentActivationStore,
    InMemoryDeploymentInstanceStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_registry_authority import HarnessHostRegistryAuthorityError
from intergrax.applications._shared.production_agent_platform_runtime import (
    build_production_agent_platform_runtime,
)
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
from intergrax.applications._shared.registry_projection import (
    ApplicationRegistryProjectionCoordinator,
    InMemoryRegistryProjectionInputStore,
    MaterializedRegistryProjection,
)
from research_application.host.agent_builders import RESEARCH_AGENT_BUILDERS
from research_application.host.main import create_research_process_app
from research_application.host.settings import ResearchBackendSettings
from research_application.host.wiring import build_research_environment_profile
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST
from research_application.tests.research_ac3_projection import build_research_test_registry_projection
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection_bundle

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO = Path(__file__).resolve().parents[3]

_PRODUCTION_BOOTSTRAP_SOURCES = (
    REPO / "applications" / "legal_application" / "host" / "main.py",
    REPO / "applications" / "research_application" / "host" / "main.py",
    REPO / "applications" / "local_workspace_application" / "host" / "main.py",
    REPO / "applications" / "governed_contractor_application" / "host" / "main.py",
    REPO / "applications" / "dispute_sim_application" / "host" / "main.py",
    REPO / "applications" / "local_workspace_application" / "hosting" / "runtime.py",
)

_PRODUCTION_FACTORY_SOURCES = (
    REPO / "applications" / "legal_application" / "host" / "factory.py",
    REPO / "applications" / "research_application" / "host" / "factory.py",
    REPO / "applications" / "local_workspace_application" / "host" / "factory.py",
    REPO / "applications" / "governed_contractor_application" / "host" / "factory.py",
    REPO / "applications" / "dispute_sim_application" / "host" / "factory.py",
)

_CANONICAL_RUNTIME_CONSTRUCTOR_SOURCES = (
    REPO / "intergrax" / "applications" / "_shared" / "production_process_composition.py",
)

_RUNTIME_CONSTRUCTOR_SYMBOLS = frozenset(
    {
        "build_production_agent_platform_runtime",
        "create_process_local_agent_platform_runtime",
    }
)


def _seed_active_projection(
    *,
    serving_store,
    projection_store,
    application_id: str,
    application_environment_id: str,
    projection: MaterializedRegistryProjection,
) -> str:
    revision_id = projection.evidence.runtime_revision_id
    projection_store.put(projection)
    serving_store.atomic_swap_serving_revision(
        application_id=application_id,
        application_environment_id=application_environment_id,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=revision_id,
        prior_revision_id=None,
        committed_at=datetime.now(UTC),
    )
    return revision_id


def _activate_projection_via_ap_lifecycle(
    *,
    composition: ProductionProcessComposition,
    bundle,
) -> str:
    activation, _coordinator, revision_service = _wire_activation_on_composition(composition)
    revision_service.persist_candidate_revision(bundle.runtime_revision)
    revision_service.mark_validated(
        bundle.runtime_revision.runtime_revision_id,
        validated_revision=bundle.runtime_revision.model_copy(
            update={"revision_state": RuntimeRevisionState.VALIDATED}
        ),
    )
    input_store = activation._projection_coordinator._input_store  # noqa: SLF001
    input_store.register(bundle)
    revision = bundle.runtime_revision
    activation.prepare_candidate(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        artifact_locator="test://artifact",
    )
    activation.commit_activation(
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        runtime_revision_id=revision.runtime_revision_id,
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
        expected_artifact_digest=bundle.materialization_artifact_digest or "sha256:" + ("d" * 64),
    )
    return revision.runtime_revision_id


def _research_activation_bundle(
    settings: ResearchBackendSettings,
    *,
    revision_id: str,
    enabled_contract_ids: tuple[str, ...] | None = None,
    application_id: str | None = None,
    application_environment_id: str | None = None,
):
    manifest = RESEARCH_APPLICATION_MANIFEST
    if application_id is not None:
        manifest = manifest.model_copy(update={"app_id": application_id})
    env = manifest.environment or build_research_environment_profile(settings)
    if application_environment_id is not None:
        env = env.model_copy(update={"profile_id": application_environment_id})
    return build_test_registry_projection_bundle(
        manifest,
        env,
        builders=RESEARCH_AGENT_BUILDERS,
        revision_id=revision_id,
        enabled_contract_stems=frozenset(enabled_contract_ids) if enabled_contract_ids else None,
        settings=settings,
    )


def _wire_activation_on_composition(
    composition: ProductionProcessComposition,
) -> tuple[ActivationService, ApplicationRegistryProjectionCoordinator, RuntimeRevisionService]:
    state = composition.agent_platform_runtime.distribution_state
    stores = composition.agent_platform_runtime.stores
    revision_store = InMemoryRuntimeRevisionStore(state)
    input_store = InMemoryRegistryProjectionInputStore()
    coordinator = ApplicationRegistryProjectionCoordinator(
        revision_store=revision_store,
        input_store=input_store,
        projection_store=stores.registry_projection_store,
    )
    activation = ActivationService(
        revision_store=revision_store,
        deployment_instance_store=InMemoryDeploymentInstanceStore(state),
        serving_store=stores.serving_store,
        activation_store=InMemoryApplicationEnvironmentActivationStore(state),
        deployment_adapter=FakeInMemoryRuntimeDeploymentAdapter(),
        projection_coordinator=coordinator,
    )
    return activation, coordinator, RuntimeRevisionService(revision_store)


def test_production_process_composition_owns_single_runtime_bundle() -> None:
    composition = create_reference_production_process_composition()
    assert isinstance(composition, ProductionProcessComposition)
    assert composition.agent_platform_runtime.stores.serving_store is not None
    assert composition.agent_platform_runtime.stores.registry_projection_store is not None


def test_production_store_continuity_resolves_active_projection_and_nexus_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "test-store-continuity-key")
    settings = ResearchBackendSettings(use_nexus_loop=True)
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    composition = create_reference_production_process_composition()
    bundle = _research_activation_bundle(
        settings,
        revision_id="rev-store-continuity",
        enabled_contract_ids=("research",),
    )
    revision_id = _activate_projection_via_ap_lifecycle(
        composition=composition,
        bundle=bundle,
    )

    resolved = bootstrap_production_registry_projection(
        application_id=manifest.app_id,
        application_environment_id=env.profile_id,
        stores=composition.agent_platform_runtime.stores,
    )
    assert resolved.evidence.runtime_revision_id == revision_id
    assert resolved.agent_registry.list_agent_ids() == ["research"]

    app = create_research_process_app(process_composition=composition)
    runtime = build_harness_host_runtime(
        manifest.model_copy(update={"environment": env}),
        env,
        settings=settings,
        registry_projection=resolved,
        use_in_memory_trace=True,
    )
    assert runtime.registry_projection_evidence.runtime_revision_id == revision_id
    assert runtime.nexus_loop.registry.list_agent_ids() == ["research"]
    assert app is not None


def test_multi_application_environments_share_process_stores_without_collision() -> None:
    settings = ResearchBackendSettings(use_nexus_loop=True)
    composition = create_reference_production_process_composition()
    stores = composition.agent_platform_runtime.stores

    research_manifest = RESEARCH_APPLICATION_MANIFEST
    research_env = research_manifest.environment or build_research_environment_profile(settings)
    research_bundle = _research_activation_bundle(
        settings,
        revision_id="rev-app-a",
        enabled_contract_ids=("research",),
    )
    legal_bundle = _research_activation_bundle(
        settings,
        revision_id="rev-app-b",
        enabled_contract_ids=("research",),
        application_id="legal",
        application_environment_id="prod",
    )

    _activate_projection_via_ap_lifecycle(composition=composition, bundle=research_bundle)
    _activate_projection_via_ap_lifecycle(composition=composition, bundle=legal_bundle)

    resolved_a = bootstrap_production_registry_projection(
        application_id=research_manifest.app_id,
        application_environment_id=research_env.profile_id,
        stores=stores,
    )
    resolved_b = bootstrap_production_registry_projection(
        application_id="legal",
        application_environment_id="prod",
        stores=stores,
    )
    assert resolved_a.evidence.runtime_revision_id == "rev-app-a"
    assert resolved_b.evidence.runtime_revision_id == "rev-app-b"


def test_fresh_runtime_store_cannot_resolve_seeded_projection() -> None:
    settings = ResearchBackendSettings(use_nexus_loop=True)
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    seeded_runtime = build_production_agent_platform_runtime()
    fresh_runtime = build_production_agent_platform_runtime()
    projection = build_research_test_registry_projection(
        settings,
        revision_id="rev-fresh-negative",
        enabled_contract_ids=("research",),
    )
    _seed_active_projection(
        serving_store=seeded_runtime.stores.serving_store,
        projection_store=seeded_runtime.stores.registry_projection_store,
        application_id=manifest.app_id,
        application_environment_id=env.profile_id,
        projection=projection,
    )

    with pytest.raises(HarnessHostRegistryAuthorityError, match="no active traffic-serving"):
        bootstrap_production_registry_projection(
            application_id=manifest.app_id,
            application_environment_id=env.profile_id,
            stores=fresh_runtime.stores,
        )


def test_fresh_process_composition_fails_closed_for_strict_process_app() -> None:
    composition = create_reference_production_process_composition()
    with pytest.raises(HarnessHostRegistryAuthorityError, match="no active traffic-serving"):
        create_research_process_app(process_composition=composition)


def test_multi_app_process_wiring_respects_shared_composition_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "test-process-wiring-key")
    settings = ResearchBackendSettings(use_nexus_loop=True)
    composition = create_reference_production_process_composition()
    research_bundle = _research_activation_bundle(
        settings,
        revision_id="rev-process-a",
        enabled_contract_ids=("research",),
    )
    legal_bundle = _research_activation_bundle(
        settings,
        revision_id="rev-process-b",
        enabled_contract_ids=("research",),
        application_id="legal",
        application_environment_id="prod",
    )
    _activate_projection_via_ap_lifecycle(composition=composition, bundle=research_bundle)
    _activate_projection_via_ap_lifecycle(composition=composition, bundle=legal_bundle)

    research_app = create_research_process_app(process_composition=composition)
    legal_resolved = bootstrap_production_registry_projection(
        application_id="legal",
        application_environment_id="prod",
        stores=composition.agent_platform_runtime.stores,
    )
    assert legal_resolved.evidence.runtime_revision_id == "rev-process-b"
    assert research_app is not None


def _bootstrap_calls_missing_stores(tree: ast.AST) -> list[int]:
    missing: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "bootstrap_production_registry_projection":
            if "stores" not in {keyword.arg for keyword in node.keywords}:
                missing.append(node.lineno)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "bootstrap_production_registry_projection"
        ):
            if "stores" not in {keyword.arg for keyword in node.keywords}:
                missing.append(node.lineno)
    return missing


def _runtime_owner_violations(source_path: Path) -> list[int]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    violations: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in _RUNTIME_CONSTRUCTOR_SYMBOLS:
            violations.append(node.lineno)
        if isinstance(func, ast.Attribute) and func.attr in _RUNTIME_CONSTRUCTOR_SYMBOLS:
            violations.append(node.lineno)
    return violations


@pytest.mark.parametrize("source_path", _PRODUCTION_BOOTSTRAP_SOURCES, ids=lambda p: p.name)
def test_production_bootstrap_sources_wire_agent_platform_stores(source_path: Path) -> None:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    missing = _bootstrap_calls_missing_stores(tree)
    assert not missing, (
        f"{source_path} calls bootstrap_production_registry_projection without stores= "
        f"at lines {missing}"
    )


@pytest.mark.parametrize("source_path", _PRODUCTION_FACTORY_SOURCES, ids=lambda p: p.name)
def test_production_factories_do_not_construct_agent_platform_runtime(source_path: Path) -> None:
    violations = _runtime_owner_violations(source_path)
    assert not violations, (
        f"{source_path} must not construct agent platform runtime at lines {violations}"
    )


@pytest.mark.parametrize("source_path", _PRODUCTION_BOOTSTRAP_SOURCES, ids=lambda p: p.name)
def test_production_bootstrap_sources_do_not_construct_agent_platform_runtime(
    source_path: Path,
) -> None:
    violations = _runtime_owner_violations(source_path)
    assert not violations, (
        f"{source_path} must not construct agent platform runtime at lines {violations}"
    )


def test_canonical_runtime_constructor_sites_are_process_composition_only() -> None:
    allowed_violations = {
        str(path): _runtime_owner_violations(path) for path in _CANONICAL_RUNTIME_CONSTRUCTOR_SOURCES
    }
    assert all(allowed_violations.values()), "canonical process composition must construct runtime"
    for source_path in _PRODUCTION_BOOTSTRAP_SOURCES:
        assert not _runtime_owner_violations(source_path), source_path
    for source_path in _PRODUCTION_FACTORY_SOURCES:
        assert not _runtime_owner_violations(source_path), source_path
