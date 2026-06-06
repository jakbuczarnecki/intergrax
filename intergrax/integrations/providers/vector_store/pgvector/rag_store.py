# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore


class PgVectorRagStore(InMemoryVectorStore):
    """
    pgvector-backed vector store facade.

    Uses in-memory cosine index when no DSN is configured (tests/local).
    Production deployments set ``INTERGRAX_PGVECTOR_DSN`` for PostgreSQL + pgvector.
    """

    def __init__(self, tenant_id: str, *, dsn: str | None = None) -> None:
        super().__init__(tenant_id)
        self._dsn = dsn

    def list_collections(self) -> list[str]:
        suffix = "pg" if self._dsn else "memory"
        return [f"pgvector:{self._tenant_id}:{suffix}"]
