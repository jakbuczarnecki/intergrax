# © Artur Czarnecki. All rights reserved.

"""Integration composition for attestation_demo (minimal lab profile)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.integration_wiring import bootstrap_application_integration_catalog
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter


@dataclass(frozen=True)
class AttestationDemoIntegrationWiring:
    profile: IntegrationProfile
    document_store: DocumentStore
    trace_store: RunTraceWriter
    trace_db_path: Path | None


def wire_attestation_demo_integrations(
    *,
    db_path: Path | None = None,
    document_store: DocumentStore | None = None,
) -> AttestationDemoIntegrationWiring:
    """Partner PoC — lab integration profile, in-memory trace, shared document store."""
    bootstrap_application_integration_catalog()
    profile = IntegrationProfile.lab()
    resolved_store = document_store or InMemoryDocumentStore()
    if db_path is None:
        trace_store: RunTraceWriter = InMemoryRunTraceStore()
        trace_db_path = None
    else:
        from intergrax.integrations.providers.relational_store.sqlite.bundle import (
            create_sqlite_integration,
        )

        sqlite_bundle = create_sqlite_integration(trace_db=db_path)
        trace_store = sqlite_bundle.trace_store  # type: ignore[assignment]
        trace_db_path = db_path
    return AttestationDemoIntegrationWiring(
        profile=profile,
        document_store=resolved_store,
        trace_store=trace_store,
        trace_db_path=trace_db_path,
    )
