# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vector-store production SLO soak contract (M-RAG.30)."""

from __future__ import annotations

import importlib
import time
import uuid
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document

from intergrax.integrations.contracts.base import IntegrationStatus
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStore

STABLE_PROD_SLO_SLUGS: tuple[str, ...] = ("qdrant", "pgvector", "chroma", "weaviate")

BETA_PROMOTION_CANDIDATE_SLUGS: tuple[str, ...] = ("pinecone", "milvus", "vespa")

_SLUG_MANIFEST_MODULE = "intergrax.integrations.providers.vector_store.{slug}.manifest"


@dataclass(frozen=True)
class SoakConfig:
    """Harness soak scenario for vector-store lifecycle + query SLO."""

    document_count: int = 50
    query_rounds: int = 5
    top_k: int = 5
    embedding_dim: int = 8
    max_p95_query_ms: float = 2_000.0


@dataclass(frozen=True)
class SoakResult:
    passed: bool
    slug: str = ""
    documents_indexed: int = 0
    queries_executed: int = 0
    p95_query_ms: float = 0.0
    reason: str = ""


def manifest_status_for_slug(slug: str) -> IntegrationStatus:
    module = importlib.import_module(_SLUG_MANIFEST_MODULE.format(slug=slug))
    manifest = module.MANIFEST
    return manifest.status


def _docs(count: int) -> list[Document]:
    return [
        Document(page_content=f"soak_doc_{i}", metadata={"group": i % 3, "batch": "soak"})
        for i in range(count)
    ]


def _embeddings(count: int, dim: int) -> list[list[float]]:
    return [[float((i + j) % 7) for j in range(dim)] for i in range(count)]


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    index = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[index]


def run_vectorstore_soak(
    store: VectorStore,
    *,
    config: SoakConfig | None = None,
    slug: str = "",
) -> SoakResult:
    """
    Execute ingest → query → metadata filter → delete soak on any ``VectorStore``.

    Used by gate unit tests (in-memory harness) and integration probes for stable backends.
    """
    cfg = config or SoakConfig()
    docs = _docs(cfg.document_count)
    embs = _embeddings(cfg.document_count, cfg.embedding_dim)

    try:
        store.add_documents(docs, embs)
    except Exception as exc:
        return SoakResult(passed=False, slug=slug, reason=f"ingest_failed:{exc}")

    try:
        indexed = store.count()
    except Exception as exc:
        return SoakResult(passed=False, slug=slug, reason=f"count_failed:{exc}")

    if indexed != cfg.document_count:
        return SoakResult(
            passed=False,
            slug=slug,
            documents_indexed=indexed,
            reason=f"count_mismatch:{indexed}!={cfg.document_count}",
        )

    latencies_ms: list[float] = []
    queries_executed = 0

    try:
        for round_idx in range(cfg.query_rounds):
            query_vec = embs[round_idx % len(embs)]
            started = time.perf_counter()
            hits = store.query(
                query_embedding=query_vec,
                top_k=cfg.top_k,
                include_embeddings=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1_000.0
            latencies_ms.append(elapsed_ms)
            queries_executed += 1

            if len(hits) == 0:
                return SoakResult(
                    passed=False,
                    slug=slug,
                    documents_indexed=indexed,
                    queries_executed=queries_executed,
                    reason="empty_query_hits",
                )

            for rank, hit in enumerate(hits):
                if hit.rank != rank:
                    return SoakResult(
                        passed=False,
                        slug=slug,
                        documents_indexed=indexed,
                        queries_executed=queries_executed,
                        reason=f"rank_mismatch:{hit.rank}!={rank}",
                    )
                if hit.similarity_score < -1e-6 or hit.similarity_score > 1.0 + 1e-6:
                    return SoakResult(
                        passed=False,
                        slug=slug,
                        documents_indexed=indexed,
                        queries_executed=queries_executed,
                        reason=f"score_out_of_range:{hit.similarity_score}",
                    )

        filtered = store.query(
            query_embedding=embs[0],
            top_k=cfg.document_count,
            metadata_filter=MetadataFilter(conditions={"group": 1}),
        )
        if not filtered or any(hit.metadata.get("group") != 1 for hit in filtered):
            return SoakResult(
                passed=False,
                slug=slug,
                documents_indexed=indexed,
                queries_executed=queries_executed,
                reason="metadata_filter_failed",
            )

        probe = store.query(query_embedding=embs[0], top_k=1)
        store.delete([probe[0].id])
        if store.count() != indexed - 1:
            return SoakResult(
                passed=False,
                slug=slug,
                documents_indexed=indexed,
                queries_executed=queries_executed,
                reason="delete_count_mismatch",
            )
    except Exception as exc:
        return SoakResult(
            passed=False,
            slug=slug,
            documents_indexed=indexed,
            queries_executed=queries_executed,
            reason=f"query_failed:{exc}",
        )

    p95_ms = _p95(latencies_ms)
    if p95_ms > cfg.max_p95_query_ms:
        return SoakResult(
            passed=False,
            slug=slug,
            documents_indexed=indexed,
            queries_executed=queries_executed,
            p95_query_ms=p95_ms,
            reason=f"slo_latency_exceeded:p95={p95_ms:.2f}>{cfg.max_p95_query_ms}",
        )

    return SoakResult(
        passed=True,
        slug=slug,
        documents_indexed=indexed,
        queries_executed=queries_executed,
        p95_query_ms=p95_ms,
        reason="ok",
    )


def unique_soak_collection(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"
