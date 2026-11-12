# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from intergrax.rag.governance.source_freshness import RetrievalFreshnessAssessment
    from intergrax.rag.retrieval.citation import Citation


@dataclass(frozen=True)
class RetrievalChunk:
    id: str
    text: str
    score: float
    rank: int = 0
    channel: str = "unknown"
    vector_id: str | None = None
    scope: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    user_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_ref(self) -> str:
        """Stable source reference for retrieval poisoning defense."""
        for bucket in (self.metadata, self.user_metadata, self.provenance):
            for key in ("url", "source", "source_id"):
                value = bucket.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return self.id


@dataclass
class RetrievalTrace:
    route_tier: str = "standard"
    route_classifier: str | None = None
    retriever_id: str = ""
    reranker_id: str | None = None
    rerank_enabled: bool = False
    candidates_before_rerank: int = 0
    candidates_after_rerank: int = 0
    retrieval_latency_ms: float | None = None
    rerank_latency_ms: float | None = None
    agentic_iteration: int | None = None
    agentic_stopped: str | None = None
    agentic_total_latency_ms: float | None = None
    agentic_per_iteration_retriever_ids: list[str] = field(default_factory=list)
    agentic_per_iteration_latency_ms: list[float] = field(default_factory=list)
    agentic_refine_calls: int = 0
    agentic_latency_budget_ms: float | None = None
    hybrid_used: bool = False
    recall_at_k: float | None = None
    attempted_retriever_ids: list[str] = field(default_factory=list)
    fallback_applied: bool = False
    retrieval_error_kind: str | None = None
    embedding_version_filtered_count: int = 0
    embedding_version_warnings: list[str] = field(default_factory=list)
    channel_contributions: dict[str, list[str]] = field(default_factory=dict)
    graph_expanded_node_ids: list[str] = field(default_factory=list)
    graph_provenance_summary: str = ""
    graph_provenance_records: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RetrievalResult:
    chunks: list[RetrievalChunk]
    used: bool
    reason: str
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)
    citations: list[Citation] = field(default_factory=list)
    freshness: RetrievalFreshnessAssessment | None = None
    """Typed source-staleness verdicts; None unless the service has a source observer (PDC-D2-R005)."""
