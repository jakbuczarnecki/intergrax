# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.applications._shared.application_composition_context import (
    ApplicationCompositionContext,
    composition_for_factory_context,
)
from intergrax.applications._shared.diagnostic_read_wiring import (
    build_diagnostic_read_service,
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ReliabilityProfile,
)
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.contracts.execution_lineage import (
    ExecutionLineageConfigurationError,
    ExecutionLineagePersistenceProvider,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.execution.lineage.document_store_persistence import (
    DocumentStoreExecutionLineagePersistence,
)
from intergrax.runtime.execution.lineage.wiring import resolve_execution_lineage_persistence
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@dataclass
class _EnvWiring:
    composition: ApplicationCompositionContext


@dataclass
class _Observability:
    runtime_event_store: InMemoryRuntimeEventStore


@dataclass
class _RuntimeStub:
    environment: ApplicationEnvironmentProfile
    env_wiring: _EnvWiring
    observability: _Observability


def _runtime(*, provider: ExecutionLineagePersistenceProvider | None) -> _RuntimeStub:
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="diag.lineage.read")
    environment.reliability_profile = ReliabilityProfile(
        execution_lineage_persistence_provider=provider,
    )
    document_store = InMemoryDocumentStore()
    return _RuntimeStub(
        environment=environment,
        env_wiring=_EnvWiring(
            composition_for_factory_context(
                ApplicationBuildContext.for_manifest(
                    ApplicationManifest.lab(
                        app_id="diag_lineage_read",
                        name="Diag Lineage Read",
                        agents=[],
                    ),
                ),
                tool_wiring_context=ToolWiringContext(document_store=document_store),
            )
        ),
        observability=_Observability(InMemoryRuntimeEventStore()),
    )


def test_lineage_disabled_keeps_reader_none() -> None:
    runtime = _runtime(provider=None)
    dependencies = resolve_host_diagnostic_read_dependencies(runtime)
    assert dependencies.execution_lineage_reader is None
    assert build_diagnostic_read_service(dependencies) is not None


def test_lineage_enabled_uses_document_store_reader() -> None:
    runtime = _runtime(provider=ExecutionLineagePersistenceProvider.DOCUMENT_STORE)
    dependencies = resolve_host_diagnostic_read_dependencies(runtime)
    reader = dependencies.execution_lineage_reader
    assert isinstance(reader, DocumentStoreExecutionLineagePersistence)


def test_lineage_provider_fail_closed_without_document_store() -> None:
    runtime = _runtime(provider=ExecutionLineagePersistenceProvider.DOCUMENT_STORE)
    runtime.env_wiring.composition.tool_wiring_context.document_store = None
    with pytest.raises(ValueError):
        resolve_host_diagnostic_read_dependencies(runtime)


def test_resolve_execution_lineage_persistence_fail_closed_incompatible_store() -> None:
    with pytest.raises(ExecutionLineageConfigurationError):
        resolve_execution_lineage_persistence(
            document_store=object(),
            provider=ExecutionLineagePersistenceProvider.DOCUMENT_STORE,
        )
