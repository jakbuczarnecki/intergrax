# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""
Configurable RAG profile (Tier-0) — retriever, reranker, routing, ingest options.

No vendor or parser is hardcoded; slugs and strategy ids come from env or explicit
``RagProfile`` instances passed through ``ToolWiringContext`` / ``RuntimeConfig``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

RouteMode = Literal["off", "auto"]
QueryExpansionMode = Literal["off", "deterministic", "llm"]
ContextualEnrichMode = Literal["off", "on"]
GraphIndexerMode = Literal["heuristic", "llm", "heuristic_then_llm"]
AgenticQueryMode = Literal["deterministic", "llm"]

HARNESS_GRAPH_STORE_BACKEND = "inmemory"
PRODUCTION_GRAPH_STORE_BACKEND = "neo4j"
APPROVED_PRODUCTION_GRAPH_STORE_SLUGS: tuple[str, ...] = ("neo4j", "memgraph")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_csv_tuple(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _env_optional_float(name: str) -> Optional[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class RagProfile:
    """Platform defaults for retrieval, rerank, ingest, and routing."""

    # Retrieval
    retriever_id: str = "hybrid"
    fast_retriever_id: str = "vector_similarity"
    deep_retriever_id: str = "fusion"
    reranker_id: str = "embedding_cosine"
    enable_rerank: bool = True
    prefetch_top_k: int = 20
    final_top_k: int = 8
    score_threshold: Optional[float] = None

    # Adaptive routing (fast / standard / deep)
    route_mode: RouteMode = "auto"
    deep_query_min_words: int = 12
    llm_route_enabled: bool = False

    # Ingest (no fixed parser — optional integration slug)
    chunking_strategy_id: str = "langchain_recursive"
    hierarchical_index_enabled: bool = False
    sync_ingest_max_bytes: int = 50_000_000
    semantic_chunking_max_chars: int = 100_000
    async_ingest_workflow_id: str = "rag-ingest"
    document_parser_slug: Optional[str] = None
    contextual_enrich: ContextualEnrichMode = "off"
    query_expansion: QueryExpansionMode = "deterministic"

    # Agentic deep retrieval (M-RAG.13)
    agentic_enabled: bool = False
    agentic_max_iterations: int = 3
    agentic_min_chunks: int = 2
    agentic_min_score: float = 0.35
    agentic_query_mode: AgenticQueryMode = "deterministic"
    agentic_iteration_retriever_ids: tuple[str, ...] = ()
    agentic_max_total_latency_ms: Optional[float] = None

    # Native hybrid (BM25 + dense via store)
    native_hybrid_enabled: bool = True
    qdrant_sparse_enabled: bool = False

    # GraphRAG
    graph_rag_enabled: bool = False
    graph_rag_hops: int = 1
    graph_rag_seed_top_k: int = 5
    graph_indexer_mode: GraphIndexerMode = "heuristic"
    graph_store_backend: str = "inmemory"

    # Sparse encoding for native hybrid (Qdrant sparse vectors)
    sparse_encoder: str = "bm25_hash"

    # Weaviate native hybrid (when client configured)
    weaviate_native_hybrid: bool = True

    # Governance / ops
    embedding_model_version: Optional[str] = None
    embedding_version_warn_on_ingest: bool = True
    embedding_version_filter_on_retrieve: bool = False
    max_context_chars: int = 4000

    extras: dict[str, str] = field(default_factory=dict)

    def uses_hierarchical_index(self) -> bool:
        if self.hierarchical_index_enabled:
            return True
        retriever_ids = {
            self.retriever_id,
            self.fast_retriever_id,
            self.deep_retriever_id,
        }
        return "hierarchical" in retriever_ids

    def effective_retriever(self, *, route_tier: str) -> str:
        if route_tier == "fast":
            return self.fast_retriever_id
        if route_tier == "deep":
            if self.graph_rag_enabled:
                return "graph_rag"
            if self.query_expansion != "off":
                return "multiquery"
            return self.deep_retriever_id
        return self.retriever_id


def production_rag_profile() -> RagProfile:
    """
    Harness / lab GraphRAG preset (AUDIT-IDEAL-14.1).

    Uses **in-memory** graph store — suitable for gate tests and lab hosts only.
    Tier-3 product hosts MUST use ``production_graph_rag_profile()`` with
    ``IntegrationProfile.graph_store=neo4j`` (M-RAG.33).
    """
    return RagProfile(
        retriever_id="hybrid",
        deep_retriever_id="fusion",
        graph_rag_enabled=True,
        graph_rag_hops=1,
        graph_indexer_mode="heuristic",
        graph_store_backend=HARNESS_GRAPH_STORE_BACKEND,
        enable_rerank=True,
        route_mode="auto",
    )


def production_graph_rag_profile() -> RagProfile:
    """
    Tier-3 GraphRAG production preset — durable ``neo4j`` graph backend required.

    Pair with ``IntegrationProfile.graph_store`` slug ``neo4j`` and
    ``create_rag_graph_store(profile=..., integration_graph_store=...)``.
    """
    return RagProfile(
        retriever_id="hybrid",
        deep_retriever_id="fusion",
        graph_rag_enabled=True,
        graph_rag_hops=1,
        graph_indexer_mode="heuristic",
        graph_store_backend=PRODUCTION_GRAPH_STORE_BACKEND,
        enable_rerank=True,
        route_mode="auto",
    )


def is_harness_graph_rag_profile(profile: RagProfile) -> bool:
    return (
        profile.graph_rag_enabled
        and profile.graph_store_backend == HARNESS_GRAPH_STORE_BACKEND
    )


def validate_graph_rag_production_wiring(
    profile: RagProfile,
    *,
    graph_store_slug: str | None,
) -> str | None:
    """
    Return an error reason when GraphRAG is enabled on a product host without neo4j.

    ``None`` means wiring is valid or GraphRAG is disabled.
    """
    if not profile.graph_rag_enabled:
        return None
    if profile.graph_store_backend not in APPROVED_PRODUCTION_GRAPH_STORE_SLUGS:
        return "graph_store_backend_not_approved_for_production"
    if graph_store_slug is not None and graph_store_slug not in APPROVED_PRODUCTION_GRAPH_STORE_SLUGS:
        return f"integration_graph_store_not_approved:{graph_store_slug}"
    return None


def rag_profile_from_env() -> RagProfile:
    """Build profile from ``INTERGRAX_RAG_*`` environment variables."""
    threshold_raw = os.getenv("INTERGRAX_RAG_SCORE_THRESHOLD", "").strip()
    score_threshold: Optional[float] = None
    if threshold_raw:
        try:
            score_threshold = float(threshold_raw)
        except ValueError:
            score_threshold = None

    route_raw = os.getenv("INTERGRAX_RAG_ROUTE_MODE", "auto").strip().lower()
    route_mode: RouteMode = "auto" if route_raw != "off" else "off"

    contextual_raw = os.getenv("INTERGRAX_RAG_CONTEXTUAL_ENRICH", "off").strip().lower()
    contextual: ContextualEnrichMode = "on" if contextual_raw in ("1", "true", "on", "yes") else "off"

    expansion_raw = os.getenv("INTERGRAX_RAG_QUERY_EXPANSION", "deterministic").strip().lower()
    if expansion_raw not in ("off", "deterministic", "llm"):
        expansion_raw = "deterministic"
    query_expansion: QueryExpansionMode = expansion_raw  # type: ignore[assignment]

    graph_mode_raw = os.getenv("INTERGRAX_RAG_GRAPH_INDEXER_MODE", "heuristic").strip().lower()
    if graph_mode_raw not in ("heuristic", "llm", "heuristic_then_llm"):
        graph_mode_raw = "heuristic"
    graph_indexer_mode: GraphIndexerMode = graph_mode_raw  # type: ignore[assignment]

    agentic_q_raw = os.getenv("INTERGRAX_RAG_AGENTIC_QUERY_MODE", "deterministic").strip().lower()
    if agentic_q_raw not in ("deterministic", "llm"):
        agentic_q_raw = "deterministic"
    agentic_query_mode: AgenticQueryMode = agentic_q_raw  # type: ignore[assignment]

    parser_slug = os.getenv("INTERGRAX_RAG_DOCUMENT_PARSER_SLUG", "").strip() or None

    return RagProfile(
        retriever_id=os.getenv("INTERGRAX_RAG_RETRIEVER_ID", "hybrid").strip() or "hybrid",
        fast_retriever_id=os.getenv("INTERGRAX_RAG_FAST_RETRIEVER_ID", "vector_similarity").strip()
        or "vector_similarity",
        deep_retriever_id=os.getenv("INTERGRAX_RAG_DEEP_RETRIEVER_ID", "fusion").strip() or "fusion",
        reranker_id=os.getenv("INTERGRAX_RAG_RERANKER_ID", "embedding_cosine").strip()
        or "embedding_cosine",
        enable_rerank=_env_bool("INTERGRAX_RAG_ENABLE_RERANK", True),
        prefetch_top_k=_env_int("INTERGRAX_RAG_PREFETCH_TOP_K", 20),
        final_top_k=_env_int("INTERGRAX_RAG_FINAL_TOP_K", 8),
        score_threshold=score_threshold,
        route_mode=route_mode,
        deep_query_min_words=_env_int("INTERGRAX_RAG_DEEP_QUERY_MIN_WORDS", 12),
        llm_route_enabled=_env_bool("INTERGRAX_RAG_LLM_ROUTE_ENABLED", False),
        chunking_strategy_id=os.getenv("INTERGRAX_RAG_CHUNKING_STRATEGY", "langchain_recursive").strip()
        or "langchain_recursive",
        hierarchical_index_enabled=_env_bool("INTERGRAX_RAG_HIERARCHICAL_INDEX", False),
        sync_ingest_max_bytes=_env_int("INTERGRAX_RAG_SYNC_INGEST_MAX_BYTES", 50_000_000),
        semantic_chunking_max_chars=_env_int(
            "INTERGRAX_RAG_SEMANTIC_CHUNKING_MAX_CHARS", 100_000
        ),
        async_ingest_workflow_id=os.getenv("INTERGRAX_RAG_ASYNC_INGEST_WORKFLOW_ID", "rag-ingest").strip()
        or "rag-ingest",
        document_parser_slug=parser_slug,
        contextual_enrich=contextual,
        query_expansion=query_expansion,
        embedding_model_version=os.getenv("INTERGRAX_RAG_EMBEDDING_MODEL_VERSION", "").strip() or None,
        embedding_version_warn_on_ingest=_env_bool("INTERGRAX_RAG_EMBEDDING_VERSION_WARN_INGEST", True),
        embedding_version_filter_on_retrieve=_env_bool(
            "INTERGRAX_RAG_EMBEDDING_VERSION_FILTER_RETRIEVE",
            False,
        ),
        max_context_chars=_env_int("INTERGRAX_RAG_MAX_CONTEXT_CHARS", 4000),
        agentic_enabled=_env_bool("INTERGRAX_RAG_AGENTIC_ENABLED", False),
        agentic_max_iterations=_env_int("INTERGRAX_RAG_AGENTIC_MAX_ITERATIONS", 3),
        agentic_min_chunks=_env_int("INTERGRAX_RAG_AGENTIC_MIN_CHUNKS", 2),
        agentic_min_score=float(os.getenv("INTERGRAX_RAG_AGENTIC_MIN_SCORE", "0.35") or "0.35"),
        native_hybrid_enabled=_env_bool("INTERGRAX_RAG_NATIVE_HYBRID", True),
        qdrant_sparse_enabled=_env_bool("INTERGRAX_RAG_QDRANT_SPARSE", False),
        graph_rag_enabled=_env_bool("INTERGRAX_RAG_GRAPH_ENABLED", False),
        graph_rag_hops=_env_int("INTERGRAX_RAG_GRAPH_HOPS", 1),
        graph_indexer_mode=graph_indexer_mode,
        graph_store_backend=os.getenv("INTERGRAX_RAG_GRAPH_STORE", "inmemory").strip().lower()
        or "inmemory",
        sparse_encoder=os.getenv("INTERGRAX_RAG_SPARSE_ENCODER", "bm25_hash").strip().lower()
        or "bm25_hash",
        agentic_query_mode=agentic_query_mode,
        agentic_iteration_retriever_ids=_env_csv_tuple("INTERGRAX_RAG_AGENTIC_ITERATION_RETRIEVERS"),
        agentic_max_total_latency_ms=_env_optional_float(
            "INTERGRAX_RAG_AGENTIC_MAX_TOTAL_LATENCY_MS"
        ),
        weaviate_native_hybrid=_env_bool("INTERGRAX_RAG_WEAVIATE_NATIVE_HYBRID", True),
        extras={
            "metrics_enabled": "true"
            if _env_bool("INTERGRAX_RAG_METRICS_ENABLED", False)
            else "false",
        },
    )
