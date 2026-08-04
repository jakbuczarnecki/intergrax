# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cross-tenant isolation contract for vector-store backends (M-RAG.35)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.contracts.vector_store import VectorStore
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

TENANT_ISOLATION_CONTRACT_BACKENDS: tuple[str, ...] = (
    "inmemory",
    "pgvector",
    "weaviate",
    "qdrant",
    "chroma",
    "lancedb",
    "typesense",
)

StoreFactory = Callable[[str, str], VectorStore]


@dataclass(frozen=True)
class TenantIsolationContractResult:
    slug: str
    cross_query_isolated: bool
    ingest_mismatch_rejected: bool
    reason: str = ""


def _embedding(dim: int = 4) -> list[float]:
    return [0.1, 0.2, 0.3, 0.4][:dim] + [0.0] * max(0, dim - 4)


def _document(document_id: str, tenant_id: str, content: str) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {"tenant_id": tenant_id},
            "content": content,
            "metadata": {},
            "provenance": {
                "source_kind": "tenant-isolation-contract",
                "source_id": document_id,
            },
        }
    )


def run_tenant_isolation_contract(
    factory: StoreFactory,
    *,
    slug: str,
    collection_name: str = "tenant_iso_contract",
    tenant_a: str = "tenant_A",
    tenant_b: str = "tenant_B",
) -> TenantIsolationContractResult:
    """
    Verify tenant A data is invisible to tenant B and ingest metadata mismatch is rejected.
    """
    scope_a = VectorStoreScope(tenant_id=tenant_a)
    scope_b = VectorStoreScope(tenant_id=tenant_b)
    manager_a = VectorstoreManager(
        store=factory(tenant_a, collection_name),
        scope=scope_a,
    )
    manager_b = VectorstoreManager(
        store=factory(tenant_b, collection_name),
        scope=scope_b,
    )
    vector = _embedding()
    secret = _document("chunk-secret-1", tenant_a, "tenant-secret-chunk")

    try:
        manager_a.add_records(
            [VectorStoreRecord(document=secret, embedding=vector, vector_id=secret.identity.document_id)],
            scope=scope_a,
        )
    except Exception as exc:
        return TenantIsolationContractResult(
            slug=slug,
            cross_query_isolated=False,
            ingest_mismatch_rejected=False,
            reason=f"tenant_a_ingest_failed:{exc}",
        )

    try:
        hits = manager_b.query(query_embedding=vector, top_k=5, scope=scope_b)
    except Exception as exc:
        return TenantIsolationContractResult(
            slug=slug,
            cross_query_isolated=False,
            ingest_mismatch_rejected=False,
            reason=f"tenant_b_query_failed:{exc}",
        )

    if hits:
        return TenantIsolationContractResult(
            slug=slug,
            cross_query_isolated=False,
            ingest_mismatch_rejected=False,
            reason="tenant_b_leaked_tenant_a_chunks",
        )

    mismatch_doc = _document("chunk-mismatch", tenant_b, "mismatch")
    ingest_rejected = False
    try:
        manager_a.add_records(
            [
                VectorStoreRecord(
                    document=mismatch_doc,
                    embedding=vector,
                    vector_id=mismatch_doc.identity.document_id,
                )
            ],
            scope=scope_a,
        )
    except ValueError:
        ingest_rejected = True
    except Exception as exc:
        return TenantIsolationContractResult(
            slug=slug,
            cross_query_isolated=True,
            ingest_mismatch_rejected=False,
            reason=f"ingest_mismatch_unexpected_error:{exc}",
        )

    if not ingest_rejected:
        return TenantIsolationContractResult(
            slug=slug,
            cross_query_isolated=True,
            ingest_mismatch_rejected=False,
            reason="ingest_metadata_mismatch_not_rejected",
        )

    query_mismatch_rejected = False
    try:
        manager_a.query(
            query_embedding=vector,
            top_k=1,
            metadata_filter=MetadataFilter(conditions={"tenant_id": tenant_b}),
            scope=scope_a,
        )
    except ValueError:
        query_mismatch_rejected = True

    # Some backends coerce tenant filter instead of raising — isolation already proven.
    if not query_mismatch_rejected:
        try:
            retry_hits = manager_a.query(
                query_embedding=vector,
                top_k=5,
                metadata_filter=MetadataFilter(conditions={"tenant_id": tenant_b}),
                scope=scope_a,
            )
            if retry_hits:
                return TenantIsolationContractResult(
                    slug=slug,
                    cross_query_isolated=True,
                    ingest_mismatch_rejected=True,
                    reason="query_filter_leaked_with_foreign_tenant",
                )
        except ValueError:
            query_mismatch_rejected = True

    return TenantIsolationContractResult(
        slug=slug,
        cross_query_isolated=True,
        ingest_mismatch_rejected=True,
        reason="ok",
    )
