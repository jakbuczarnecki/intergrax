# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-DUR-02 — GCA production composition continuation binding."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from governed_contractor_application.host.main import create_governed_contractor_process_app
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)
from intergrax.applications._shared.harness_host_orchestration_topology_wiring import (
    HarnessHostOrchestrationTopologyReliabilityCompositionError,
)
from intergrax.applications._shared.production_execution_continuation_store import (
    open_production_execution_continuation_state_store,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    execution_continuation_state_store_from_durable_export,
    export_durable_continuation_state,
)
from testing_support.host_fixture_wiring import install_diagnostic_cursor_secret
from testing_support.orchestration.orchestration_consequential_effect_reliability_doubles import (
    DurableTestProviderInvocationStore,
)

pytestmark = [pytest.mark.unit]


class _HarnessStartupCaptured(RuntimeError):
    pass


def _durable_reconstructed_store():
    return execution_continuation_state_store_from_durable_export(
        export_durable_continuation_state(ExecutionContinuationDurableBacking()),
    )


def _seed_active_registry_projection(composition, *, application_id, application_environment_id, projection):
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


def test_composition_serialized_seal_store_reaches_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_diagnostic_cursor_secret(monkeypatch)
    state_file = tmp_path / "continuation.json"
    durable_store = open_production_execution_continuation_state_store(
        state_file,
        missing_file_policy="empty",
    )
    composition = create_reference_production_process_composition(
        provider_invocation_store=DurableTestProviderInvocationStore(),
        execution_continuation_state_store=durable_store,
    )
    captured: dict[str, object] = {}

    def _capture(*_args, **kwargs):
        captured.update(kwargs)
        raise _HarnessStartupCaptured()

    with patch(
        "governed_contractor_application.host.factory.build_harness_host_runtime",
        side_effect=_capture,
    ):
        with pytest.raises(_HarnessStartupCaptured):
            create_governed_contractor_backend_app(
                registry_projection=build_governed_contractor_test_registry_projection(),
                process_composition=composition,
                document_store=InMemoryDocumentStore(),
            )
    assert captured.get("execution_continuation_state_store") is durable_store
    durable_store.close()


def test_conflicting_kwarg_and_composition_store_rejected() -> None:
    composition_store = _durable_reconstructed_store()
    other_store = _durable_reconstructed_store()
    composition = ProductionProcessComposition(
        agent_platform_runtime=create_reference_production_process_composition().agent_platform_runtime,
        provider_invocation_store=DurableTestProviderInvocationStore(),
        execution_continuation_state_store=composition_store,
    )
    with pytest.raises(HarnessHostOrchestrationTopologyReliabilityCompositionError):
        create_governed_contractor_backend_app(
            registry_projection=build_governed_contractor_test_registry_projection(),
            process_composition=composition,
            document_store=InMemoryDocumentStore(),
            execution_continuation_state_store=other_store,
        )


def test_sanctioned_process_app_forwards_composition_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from governed_contractor_application.host.environment_profile import (
        build_governed_contractor_environment_profile,
    )
    from governed_contractor_application.host.settings import GovernedContractorBackendSettings
    from governed_contractor_application.manifest import build_governed_contractor_manifest

    install_diagnostic_cursor_secret(monkeypatch)
    monkeypatch.setenv("INTERGRAX_RELATIONAL_DB", str(tmp_path / "relational.db"))
    state_file = tmp_path / "continuation.json"
    durable_store = open_production_execution_continuation_state_store(
        state_file,
        missing_file_policy="empty",
    )
    composition = create_reference_production_process_composition(
        provider_invocation_store=DurableTestProviderInvocationStore(),
        execution_continuation_state_store=durable_store,
    )
    manifest = build_governed_contractor_manifest()
    settings = GovernedContractorBackendSettings.from_env()
    env = manifest.environment or build_governed_contractor_environment_profile(settings)
    projection = build_governed_contractor_test_registry_projection(
        revision_id="prod-q-r1-dur-02-process-app",
    )
    _seed_active_registry_projection(
        composition,
        application_id=manifest.app_id,
        application_environment_id=env.profile_id,
        projection=projection,
    )
    captured: dict[str, object] = {}

    def _capture(*_args, **kwargs):
        captured.update(kwargs)
        raise _HarnessStartupCaptured()

    with patch(
        "governed_contractor_application.host.factory.build_harness_host_runtime",
        side_effect=_capture,
    ):
        with pytest.raises(_HarnessStartupCaptured):
            create_governed_contractor_process_app(
                process_composition=composition,
                settings=settings,
            )
    assert captured.get("execution_continuation_state_store") is durable_store
    durable_store.close()


def test_startup_failure_after_open_does_not_pin_writer_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_diagnostic_cursor_secret(monkeypatch)
    state_file = tmp_path / "continuation.json"
    durable_store = open_production_execution_continuation_state_store(
        state_file,
        missing_file_policy="empty",
    )
    composition = create_reference_production_process_composition(
        provider_invocation_store=DurableTestProviderInvocationStore(),
        execution_continuation_state_store=durable_store,
    )
    with patch(
        "governed_contractor_application.host.factory.build_harness_host_runtime",
        side_effect=RuntimeError("injected startup failure"),
    ):
        with pytest.raises(RuntimeError, match="injected startup failure"):
            create_governed_contractor_backend_app(
                registry_projection=build_governed_contractor_test_registry_projection(),
                process_composition=composition,
                document_store=InMemoryDocumentStore(),
            )
    durable_store.close()
    reopened = open_production_execution_continuation_state_store(
        state_file,
        missing_file_policy="empty",
    )
    reopened.close()
