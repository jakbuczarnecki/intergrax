# © Artur Czarnecki. All rights reserved.

"""AC-3-FIX-2 / AGENT-CONSOLIDATION-3-ARCH production AP store composition continuity proofs."""

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

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
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from research_application.host.factory import create_research_backend_app
from research_application.host.settings import ResearchBackendSettings
from research_application.host.wiring import build_research_environment_profile
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST
from research_application.tests.research_ac3_projection import build_research_test_registry_projection

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

# AC-3-FIX-3 will relocate runtime construction into ProductionProcessComposition.
_KNOWN_RUNTIME_OWNER_VIOLATIONS_UNTIL_FIX3 = _PRODUCTION_BOOTSTRAP_SOURCES


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
    platform_runtime = composition.agent_platform_runtime
    projection = build_research_test_registry_projection(
        settings,
        revision_id="rev-store-continuity",
        enabled_contract_ids=("research",),
    )
    revision_id = _seed_active_projection(
        serving_store=platform_runtime.stores.serving_store,
        projection_store=platform_runtime.stores.registry_projection_store,
        application_id=manifest.app_id,
        application_environment_id=env.profile_id,
        projection=projection,
    )

    resolved = bootstrap_production_registry_projection(
        application_id=manifest.app_id,
        application_environment_id=env.profile_id,
        stores=composition.agent_platform_runtime.stores,
    )
    assert resolved.evidence.runtime_revision_id == revision_id
    assert resolved.agent_registry.list_agent_ids() == ["research"]

    app = create_research_backend_app(registry_projection=resolved, settings=settings)
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
    research_projection = build_research_test_registry_projection(
        settings,
        revision_id="rev-app-a",
        enabled_contract_ids=("research",),
    )
    legal_manifest_id = "legal"
    legal_env_id = "prod"
    legal_projection = build_research_test_registry_projection(
        settings,
        revision_id="rev-app-b",
        enabled_contract_ids=("research",),
    )
    legal_projection = replace(
        legal_projection,
        evidence=legal_projection.evidence.model_copy(
            update={
                "application_id": legal_manifest_id,
                "application_environment_id": legal_env_id,
            }
        ),
    )

    _seed_active_projection(
        serving_store=stores.serving_store,
        projection_store=stores.registry_projection_store,
        application_id=research_manifest.app_id,
        application_environment_id=research_env.profile_id,
        projection=research_projection,
    )
    _seed_active_projection(
        serving_store=stores.serving_store,
        projection_store=stores.registry_projection_store,
        application_id=legal_manifest_id,
        application_environment_id=legal_env_id,
        projection=legal_projection,
    )

    resolved_a = bootstrap_production_registry_projection(
        application_id=research_manifest.app_id,
        application_environment_id=research_env.profile_id,
        stores=stores,
    )
    resolved_b = bootstrap_production_registry_projection(
        application_id=legal_manifest_id,
        application_environment_id=legal_env_id,
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
        if isinstance(func, ast.Name) and func.id == "build_production_agent_platform_runtime":
            violations.append(node.lineno)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "build_production_agent_platform_runtime"
        ):
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


def test_known_runtime_owner_violations_documented_until_ac3_fix3() -> None:
    remaining = [
        path
        for path in _KNOWN_RUNTIME_OWNER_VIOLATIONS_UNTIL_FIX3
        if _runtime_owner_violations(path)
    ]
    assert remaining == list(_KNOWN_RUNTIME_OWNER_VIOLATIONS_UNTIL_FIX3)

