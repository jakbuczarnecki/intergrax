# © Artur Czarnecki. All rights reserved.

"""Synthetic scale backend probe proving generic runner pluginability (S1-N)."""

from __future__ import annotations

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from tests.system.functional_diagnostics_scale.backend import (
    BackendResourceObservation,
    ScaleBackendIdentity,
)


class SyntheticFunctionalDiagnosticsScaleProbe:
    """Minimal second probe — not used for canonical S1 backend measurement."""

    def __init__(self) -> None:
        self._store: ConditionalDocumentStore | None = None

    @property
    def provider_id(self) -> str:
        return "synthetic-in-memory"

    def prepare(self) -> None:
        return None

    def build_document_store(self) -> ConditionalDocumentStore:
        self._store = assert_conditional_document_store(InMemoryDocumentStore())
        return self._store

    def backend_identity(self) -> ScaleBackendIdentity:
        return ScaleBackendIdentity(
            provider_id=self.provider_id,
            document_store_type="InMemoryDocumentStore",
            database_name="synthetic",
            collection_name="synthetic",
        )

    def collect_backend_metrics(self) -> BackendResourceObservation:
        return BackendResourceObservation(
            document_count=0,
            storage_size_bytes=0,
            indexes=(),
        )

    def observe_execution_query_efficiency(
        self,
        *,
        tenant_id: str,
        task_id: str,
        run_id: str,
    ) -> None:
        return None

    def cleanup(self) -> None:
        self._store = None

    def close_document_store(self, store: ConditionalDocumentStore) -> None:
        store.close()
        self._store = None


__all__ = ["SyntheticFunctionalDiagnosticsScaleProbe"]
