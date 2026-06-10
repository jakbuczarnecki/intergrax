# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cross-tenant isolation contract for vector-store backends (M-RAG.35)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from langchain_core.documents import Document

from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStore

TENANT_ISOLATION_CONTRACT_BACKENDS: tuple[str, ...] = (
    "inmemory",
    "pgvector",
    "weaviate",
    "qdrant",
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
    store_a = factory(tenant_a, collection_name)
    store_b = factory(tenant_b, collection_name)
    vector = _embedding()
    secret = Document(page_content="tenant-secret-chunk", metadata={"tenant_id": tenant_a})

    try:
        store_a.add_documents([secret], [vector], ids=["chunk-secret-1"])
    except Exception as exc:
        return TenantIsolationContractResult(
            slug=slug,
            cross_query_isolated=False,
            ingest_mismatch_rejected=False,
            reason=f"tenant_a_ingest_failed:{exc}",
        )

    try:
        hits = store_b.query(query_embedding=vector, top_k=5)
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

    mismatch_doc = Document(
        page_content="mismatch",
        metadata={"tenant_id": tenant_b},
    )
    ingest_rejected = False
    try:
        store_a.add_documents([mismatch_doc], [vector], ids=["chunk-mismatch"])
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
        store_a.query(
            query_embedding=vector,
            top_k=1,
            metadata_filter=MetadataFilter(conditions={"tenant_id": tenant_b}),
        )
    except ValueError:
        query_mismatch_rejected = True

    # Some backends coerce tenant filter instead of raising — isolation already proven.
    if not query_mismatch_rejected:
        try:
            retry_hits = store_a.query(
                query_embedding=vector,
                top_k=5,
                metadata_filter=MetadataFilter(conditions={"tenant_id": tenant_b}),
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
