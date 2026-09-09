# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.nexus_factory import (
    build_nexus_loop_from_environment,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ReliabilityProfile,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageConfigurationError,
    ExecutionLineagePersistenceProvider,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.runtime.execution.lineage.document_store_persistence import (
    DocumentStoreExecutionLineagePersistence,
)
from intergrax.runtime.execution.lineage.wiring import (
    resolve_execution_lineage_persistence,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry


def test_lineage_disabled_when_provider_unset() -> None:
    store = InMemoryDocumentStore()
    resolved = resolve_execution_lineage_persistence(
        document_store=store,
        provider=None,
    )
    assert resolved is None


def test_lineage_enabled_with_document_store_provider() -> None:
    store = InMemoryDocumentStore()
    resolved = resolve_execution_lineage_persistence(
        document_store=store,
        provider=ExecutionLineagePersistenceProvider.DOCUMENT_STORE,
    )
    assert isinstance(resolved, DocumentStoreExecutionLineagePersistence)


def test_incompatible_document_store_raises_when_explicitly_enabled() -> None:
    class _PlainStore:
        pass

    with pytest.raises(ExecutionLineageConfigurationError):
        resolve_execution_lineage_persistence(
            document_store=_PlainStore(),  # type: ignore[arg-type]
            provider=ExecutionLineagePersistenceProvider.DOCUMENT_STORE,
        )


def test_nexus_factory_wires_lineage_when_provider_configured() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(
        profile_id="dg001.lineage.provider",
    ).model_copy(
        update={
            "reliability_profile": ReliabilityProfile(
                execution_lineage_persistence_provider=(
                    ExecutionLineagePersistenceProvider.DOCUMENT_STORE
                ),
            ),
        },
    )
    store = InMemoryDocumentStore()
    loop = build_nexus_loop_from_environment(
        registry=AgentRegistry(),
        env=env,
        document_store=store,
    )
    assert isinstance(
        loop.execution_lineage_persistence, DocumentStoreExecutionLineagePersistence
    )


def test_nexus_factory_leaves_lineage_none_without_provider() -> None:
    store = InMemoryDocumentStore()
    loop = build_nexus_loop_from_environment(
        registry=AgentRegistry(),
        env=ApplicationEnvironmentProfile.lab_defaults(
            profile_id="dg001.lineage.disabled"
        ),
        document_store=store,
    )
    assert loop.execution_lineage_persistence is None
