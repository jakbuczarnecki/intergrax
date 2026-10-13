# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R4 — sanctioned GCA production startup path (create_governed_contractor_process_app)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from governed_contractor_application.host.main import create_governed_contractor_process_app
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)
from intergrax.applications._shared.harness_registry_authority import HarnessHostRegistryAuthorityError
from intergrax.applications._shared.harness_host_orchestration_topology_wiring import (
    HarnessHostOrchestrationTopologyReliabilityCompositionError,
)
from intergrax.applications._shared.production_process_composition import (
    create_reference_production_process_composition,
)
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from testing_support.host_fixture_wiring import install_diagnostic_cursor_secret
from testing_support.orchestration.orchestration_consequential_effect_reliability_doubles import (
    DurableTestProviderInvocationStore,
)

pytestmark = [pytest.mark.unit]


def _seed_active_registry_projection(
    composition,
    *,
    application_id: str,
    application_environment_id: str,
    projection,
) -> None:
    stores = composition.agent_platform_runtime.stores
    stores.registry_projection_store.put(projection)
    stores.serving_store.atomic_swap_serving_revision(
        application_id=application_id,
        application_environment_id=application_environment_id,
        expected_current_revision_id=None,
        expected_pointer_revision=0,
        new_revision_id=projection.evidence.runtime_revision_id,
        prior_revision_id=None,
        committed_at=datetime.now(UTC),
    )


def test_sanctioned_process_app_fails_closed_without_activated_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical entry requires AP-9/AP-10 traffic-serving projection before factory wiring."""
    install_diagnostic_cursor_secret(monkeypatch)
    composition = create_reference_production_process_composition(
        provider_invocation_store=DurableTestProviderInvocationStore(),
    )
    with pytest.raises(HarnessHostRegistryAuthorityError, match="no active traffic-serving"):
        create_governed_contractor_process_app(process_composition=composition)


def test_sanctioned_process_app_fails_closed_without_durable_continuation_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """After lifecycle stores are seeded, strict factory still requires explicit durable continuation."""
    from governed_contractor_application.host.environment_profile import (
        build_governed_contractor_environment_profile,
    )
    from governed_contractor_application.manifest import build_governed_contractor_manifest

    install_diagnostic_cursor_secret(monkeypatch)
    monkeypatch.setenv("INTERGRAX_RELATIONAL_DB", str(tmp_path / "isolated-relational.db"))
    composition = create_reference_production_process_composition(
        provider_invocation_store=DurableTestProviderInvocationStore(),
    )
    from governed_contractor_application.host.settings import GovernedContractorBackendSettings

    manifest = build_governed_contractor_manifest()
    settings = GovernedContractorBackendSettings.from_env()
    env = manifest.environment or build_governed_contractor_environment_profile(settings)
    projection = build_governed_contractor_test_registry_projection(
        revision_id="prod-q-r1-r4-process-app",
    )
    _seed_active_registry_projection(
        composition,
        application_id=manifest.app_id,
        application_environment_id=env.profile_id,
        projection=projection,
    )
    with pytest.raises(ExecutionContinuationError) as exc:
        create_governed_contractor_process_app(
            process_composition=composition,
            settings=settings,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE


def test_strict_process_composition_requires_provider_invocation_store_at_factory() -> None:
    """Negative matrix slice — reuses R1-R1 admission evidence boundary; provider store is separate."""
    composition = create_reference_production_process_composition()
    with pytest.raises(HarnessHostOrchestrationTopologyReliabilityCompositionError):
        create_governed_contractor_backend_app(
            registry_projection=build_governed_contractor_test_registry_projection(),
            process_composition=composition,
            document_store=InMemoryDocumentStore(),
        )
