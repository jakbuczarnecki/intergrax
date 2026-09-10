# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.applications._shared.diagnostic_read_wiring import (
    build_diagnostic_read_service,
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ReliabilityProfile,
)
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
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    wire_causal_evidence_persistence,
)

pytestmark = pytest.mark.unit


@dataclass
class _ToolWiringContext:
    document_store: object | None


@dataclass
class _BuildContext:
    tool_wiring_context: _ToolWiringContext


@dataclass
class _EnvWiring:
    build_context: _BuildContext


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
        env_wiring=_EnvWiring(_BuildContext(_ToolWiringContext(document_store))),
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
    runtime.env_wiring.build_context.tool_wiring_context.document_store = None
    with pytest.raises(ValueError):
        resolve_host_diagnostic_read_dependencies(runtime)


def test_resolve_execution_lineage_persistence_fail_closed_incompatible_store() -> None:
    with pytest.raises(ExecutionLineageConfigurationError):
        resolve_execution_lineage_persistence(
            document_store=object(),
            provider=ExecutionLineagePersistenceProvider.DOCUMENT_STORE,
        )
