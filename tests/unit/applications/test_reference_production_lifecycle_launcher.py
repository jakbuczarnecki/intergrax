# © Artur Czarnecki. All rights reserved.

"""Reference production lifecycle launcher proofs (AGENT-CONSOLIDATION-3-FIX-4)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_registry_authority import HarnessHostRegistryAuthorityError
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_process_composition import (
    create_reference_production_process_composition,
)
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications._shared.registry_projection_input_bundle import (
    build_reference_activation_request,
    build_reference_registry_projection_input_bundle,
    reference_admission_mutation_id,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    wire_governed_reference_production_launcher,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleLauncher,
)
from intergrax.applications._shared.harness_host_runtime_compat import (
    resolve_harness_host_nexus_loop_legacy,
)
from research_application.host.agent_builders import RESEARCH_AGENT_BUILDERS
from research_application.host.main import create_research_process_app
from research_application.host.reference_lifecycle_input import build_research_reference_lifecycle_input
from research_application.host.settings import ResearchBackendSettings
from research_application.host.wiring import build_research_environment_profile
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _governed_launcher(
    composition,
    *,
    settings: ResearchBackendSettings | None = None,
    application_id: str | None = None,
    application_environment_id: str | None = None,
) -> tuple[ReferenceProductionLifecycleLauncher, object]:
    resolved_settings = settings or ResearchBackendSettings(use_nexus_loop=True)
    manifest = RESEARCH_APPLICATION_MANIFEST
    if application_id is not None:
        manifest = manifest.model_copy(update={"app_id": application_id})
    env = manifest.environment or build_research_environment_profile(resolved_settings)
    if application_environment_id is not None:
        env = env.model_copy(update={"profile_id": application_environment_id})
    return wire_governed_reference_production_launcher(composition, env)


def _deploy_launcher(
    launcher: ReferenceProductionLifecycleLauncher,
    projection_input,
    activation_request,
    *,
    principal,
):
    return launcher.deploy_and_activate(
        projection_input,
        activation_request,
        principal=principal,
        admission_mutation_id=reference_admission_mutation_id(
            projection_input.runtime_revision.runtime_revision_id
        ),
    )


def _deploy_and_activate(
    composition,
    projection_input,
    activation_request,
    *,
    settings: ResearchBackendSettings | None = None,
    application_id: str | None = None,
    application_environment_id: str | None = None,
):
    launcher, governance = _governed_launcher(
        composition,
        settings=settings,
        application_id=application_id,
        application_environment_id=application_environment_id,
    )
    return launcher.deploy_and_activate(
        projection_input,
        activation_request,
        principal=governance.principal,
        admission_mutation_id=reference_admission_mutation_id(
            projection_input.runtime_revision.runtime_revision_id
        ),
    )


def _research_bundle(
    settings: ResearchBackendSettings,
    *,
    revision_id: str,
    application_id: str | None = None,
    application_environment_id: str | None = None,
    enabled_contract_ids: tuple[str, ...] | None = ("research",),
):
    manifest = RESEARCH_APPLICATION_MANIFEST
    if application_id is not None:
        manifest = manifest.model_copy(update={"app_id": application_id})
    env = manifest.environment or build_research_environment_profile(settings)
    if application_environment_id is not None:
        env = env.model_copy(update={"profile_id": application_environment_id})
    return build_reference_registry_projection_input_bundle(
        manifest,
        env,
        builders=RESEARCH_AGENT_BUILDERS,
        runtime_revision_id=revision_id,
        enabled_contract_stems=frozenset(enabled_contract_ids) if enabled_contract_ids else None,
        settings=settings,
    )


def _seed_active_projection_without_activation(
    *,
    serving_store,
    projection_store,
    application_id: str,
    application_environment_id: str,
    projection: MaterializedRegistryProjection,
) -> None:
    projection_store.put(projection)
    serving_store.atomic_swap_serving_revision(
        application_id=application_id,
        application_environment_id=application_environment_id,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=projection.evidence.runtime_revision_id,
        prior_revision_id=None,
        committed_at=datetime.now(UTC),
    )


def test_reference_lifecycle_research_e2e_without_seed_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "test-lifecycle-launcher-key")
    settings = ResearchBackendSettings(use_nexus_loop=True)
    composition = create_reference_production_process_composition()
    projection_input, activation_request = build_research_reference_lifecycle_input(
        settings,
        runtime_revision_id="rev-lifecycle-e2e",
    )
    launcher, governance = _governed_launcher(composition, settings=settings)
    result = _deploy_launcher(
        launcher,
        projection_input,
        activation_request,
        principal=governance.principal,
    )

    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    resolved = bootstrap_production_registry_projection(
        application_id=manifest.app_id,
        application_environment_id=env.profile_id,
        stores=composition.agent_platform_runtime.stores,
    )
    assert result.runtime_revision_id == "rev-lifecycle-e2e"
    assert resolved.evidence.runtime_revision_id == "rev-lifecycle-e2e"
    assert resolved.agent_registry.list_agent_ids() == ["research"]

    app = create_research_process_app(process_composition=composition)
    runtime = build_harness_host_runtime(
        manifest.model_copy(update={"environment": env}),
        env,
        settings=settings,
        registry_projection=resolved,
        use_in_memory_trace=True,
    )
    assert runtime.registry_projection_evidence.runtime_revision_id == "rev-lifecycle-e2e"
    assert resolve_harness_host_nexus_loop_legacy(runtime).registry.list_agent_ids() == ["research"]
    assert app is not None
    assert launcher.process_composition is composition


def test_fresh_composition_serve_without_activation_fails() -> None:
    composition = create_reference_production_process_composition()
    with pytest.raises(HarnessHostRegistryAuthorityError, match="no active traffic-serving"):
        create_research_process_app(process_composition=composition)


def test_projection_without_activation_pointer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ResearchBackendSettings(use_nexus_loop=True)
    composition = create_reference_production_process_composition()
    projection_input, _ = build_research_reference_lifecycle_input(
        settings,
        runtime_revision_id="rev-projection-only",
    )
    from intergrax.applications._shared.registry_projection import build_registry_projection

    projection = build_registry_projection(projection_input)
    stores = composition.agent_platform_runtime.stores
    stores.registry_projection_store.put(projection)
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    with pytest.raises(HarnessHostRegistryAuthorityError, match="no active traffic-serving"):
        bootstrap_production_registry_projection(
            application_id=manifest.app_id,
            application_environment_id=env.profile_id,
            stores=stores,
        )


def test_serving_pointer_without_projection_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = ResearchBackendSettings(use_nexus_loop=True)
    composition = create_reference_production_process_composition()
    projection_input, activation_request = build_research_reference_lifecycle_input(
        settings,
        runtime_revision_id="rev-missing-projection",
    )
    launcher, governance = _governed_launcher(composition, settings=settings)
    _deploy_launcher(
        launcher,
        projection_input,
        activation_request,
        principal=governance.principal,
    )
    stores = composition.agent_platform_runtime.stores
    stores.registry_projection_store.put(
        launcher.services.projection_coordinator.get_projection("rev-missing-projection")
    )
    # Remove projection while serving pointer still references revision.
    stores.registry_projection_store._projections.clear()  # noqa: SLF001
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile(settings)
    with pytest.raises(HarnessHostRegistryAuthorityError, match="registry projection missing"):
        bootstrap_production_registry_projection(
            application_id=manifest.app_id,
            application_environment_id=env.profile_id,
            stores=stores,
        )


def test_app_env_mismatch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = ResearchBackendSettings(use_nexus_loop=True)
    composition = create_reference_production_process_composition()
    projection_input, activation_request = build_research_reference_lifecycle_input(
        settings,
        runtime_revision_id="rev-env-mismatch",
    )
    result = _deploy_and_activate(composition, projection_input, activation_request, settings=settings)
    stores = composition.agent_platform_runtime.stores
    stores.serving_store.atomic_swap_serving_revision(
        application_id="legal",
        application_environment_id="prod",
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=result.runtime_revision_id,
        prior_revision_id=None,
        committed_at=datetime.now(UTC),
    )
    with pytest.raises(HarnessHostRegistryAuthorityError, match="does not match requested"):
        bootstrap_production_registry_projection(
            application_id="legal",
            application_environment_id="prod",
            stores=stores,
        )


def test_cross_composition_serve_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = ResearchBackendSettings(use_nexus_loop=True)
    composition_a = create_reference_production_process_composition()
    composition_b = create_reference_production_process_composition()
    projection_input, activation_request = build_research_reference_lifecycle_input(
        settings,
        runtime_revision_id="rev-cross-composition",
    )
    _deploy_and_activate(composition_a, projection_input, activation_request, settings=settings)
    with pytest.raises(HarnessHostRegistryAuthorityError, match="no active traffic-serving"):
        create_research_process_app(process_composition=composition_b)


def test_multi_app_same_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = ResearchBackendSettings(use_nexus_loop=True)
    composition = create_reference_production_process_composition()
    launcher, governance = _governed_launcher(composition, settings=settings)
    research_bundle = _research_bundle(settings, revision_id="rev-multi-a")
    legal_bundle = _research_bundle(
        settings,
        revision_id="rev-multi-b",
        application_id="legal",
        application_environment_id="prod",
    )
    _deploy_launcher(
        launcher,
        research_bundle,
        build_reference_activation_request(research_bundle),
        principal=governance.principal,
    )
    legal_launcher, legal_governance = _governed_launcher(
        composition,
        settings=settings,
        application_id="legal",
        application_environment_id="prod",
    )
    _deploy_launcher(
        legal_launcher,
        legal_bundle,
        build_reference_activation_request(legal_bundle),
        principal=legal_governance.principal,
    )
    stores = composition.agent_platform_runtime.stores
    research_manifest = RESEARCH_APPLICATION_MANIFEST
    research_env = research_manifest.environment or build_research_environment_profile(settings)
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
    assert resolved_a.evidence.runtime_revision_id == "rev-multi-a"
    assert resolved_b.evidence.runtime_revision_id == "rev-multi-b"


def test_lifecycle_and_serving_share_store_instances() -> None:
    composition = create_reference_production_process_composition()
    services = ReferenceProductionLifecycleLauncher(composition).services
    stores = composition.agent_platform_runtime.stores
    assert services.activation_service._serving_store is stores.serving_store  # noqa: SLF001
    assert services.projection_coordinator._projection_store is stores.registry_projection_store  # noqa: SLF001
