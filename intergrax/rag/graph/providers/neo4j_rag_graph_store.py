# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neo4j-backed RAG GraphStore — adapts Integration Library graph_store to GraphRAG contract."""

from __future__ import annotations

from typing import Any

from intergrax.rag.graph.providers.cypher_rag_graph_store import CypherRagGraphStore


class Neo4jRagGraphStore(CypherRagGraphStore):
    """GraphRAG store over Neo4j via the Integration Library ``GraphStore.run_query``."""

    def __init__(
        self,
        integration_store: Any,
        *,
        tenant_id: str | None = None,
    ) -> None:
        super().__init__(integration_store, tenant_id=tenant_id)
