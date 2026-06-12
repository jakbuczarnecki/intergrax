# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sync vector index lifecycle operations to document knowledge graph (M-RAG.40)."""

from __future__ import annotations

from typing import Optional, Sequence

from intergrax.rag.graph.contracts.graph_store import GraphStore


def sync_graph_delete_documents(
    graph_store: Optional[GraphStore],
    document_ids: Sequence[str],
) -> int:
    """Unlink chunk ids from graph and prune orphan entities. Returns affected count."""
    if graph_store is None:
        return 0
    ids = [item.strip() for item in document_ids if item.strip()]
    if not ids:
        return 0
    return graph_store.unlink_chunks(ids)


def sync_graph_purge_collection(
    graph_store: Optional[GraphStore],
    *,
    tenant_id: str | None = None,
) -> int:
    """Purge graph artifacts for a tenant or entire graph when tenant is unset."""
    if graph_store is None:
        return 0
    return graph_store.purge_graph(tenant_id=tenant_id)
