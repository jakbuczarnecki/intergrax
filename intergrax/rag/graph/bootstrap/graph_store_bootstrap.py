# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Adapt Integration GraphStore to RAG GraphStore (GraphRAG semantics)."""

from __future__ import annotations

from intergrax.integrations.contracts.graph_store import GraphStore as IntegrationGraphStore
from intergrax.rag.graph.contracts.graph_store import GraphStore as RagGraphStore
from intergrax.rag.graph.providers.cypher_rag_graph_store import CypherRagGraphStore
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore


def create_rag_graph_store(
    *,
    integration_graph_store: IntegrationGraphStore | None = None,
    tenant_id: str | None = None,
) -> RagGraphStore:
    """
    Build a RAG GraphStore from an Integration graph capability or local in-memory harness.

    Provider selection and materialization belong to Integrations; this module only adapts
    ``IntegrationGraphStore`` → ``RagGraphStore``.
    """
    if integration_graph_store is None:
        return InMemoryGraphStore(tenant_id=tenant_id)
    return CypherRagGraphStore(integration_graph_store, tenant_id=tenant_id)
