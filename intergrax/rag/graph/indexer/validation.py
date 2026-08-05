# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Native Graph RAG batch validation shared by graph indexers."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.graph.contracts.graph_store import GraphStore


def validate_graph_index_batch(
    store: GraphStore,
    documents: Sequence[KnowledgeDocument],
    chunk_ids: Sequence[str] | None,
) -> tuple[tuple[KnowledgeDocument, ...], tuple[str, ...]]:
    """Validate the complete native batch before any GraphStore write."""
    materialized = tuple(documents)
    if not materialized:
        return (), ()

    validated: list[KnowledgeDocument] = []
    for document in materialized:
        if not isinstance(document, KnowledgeDocument):
            raise TypeError("documents must contain only KnowledgeDocument values")
        try:
            validated.append(
                KnowledgeDocument.model_validate(document.model_dump(mode="python"))
            )
        except Exception as exc:
            raise ValueError("document failed full revalidation") from exc

    first_scope = validated[0].scope
    if any(
        document.scope.tenant_id != first_scope.tenant_id
        or document.scope.namespace != first_scope.namespace
        for document in validated[1:]
    ):
        raise ValueError("documents must share the same tenant and namespace")

    store_tenant = store.tenant_id
    if store_tenant is not None and first_scope.tenant_id != store_tenant:
        raise ValueError("document tenant_id differs from bound graph store")

    if chunk_ids is None:
        resolved_ids = tuple(document.identity.document_id for document in validated)
    else:
        if len(chunk_ids) != len(validated):
            raise ValueError("chunk_ids length must match documents length")
        resolved_ids = tuple(chunk_ids)

    if any(not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in resolved_ids):
        raise ValueError("chunk_ids must contain only non-empty strings")
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError("chunk_ids must be unique")

    return tuple(validated), resolved_ids
