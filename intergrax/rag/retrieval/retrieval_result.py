# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from intergrax.rag.retrieval.citation import Citation


@dataclass(frozen=True)
class RetrievalChunk:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    route_tier: str = "standard"
    route_classifier: Optional[str] = None
    retriever_id: str = ""
    reranker_id: Optional[str] = None
    rerank_enabled: bool = False
    candidates_before_rerank: int = 0
    candidates_after_rerank: int = 0
    retrieval_latency_ms: Optional[float] = None
    rerank_latency_ms: Optional[float] = None
    agentic_iteration: Optional[int] = None
    agentic_stopped: Optional[str] = None
    agentic_total_latency_ms: Optional[float] = None
    agentic_per_iteration_retriever_ids: List[str] = field(default_factory=list)
    agentic_per_iteration_latency_ms: List[float] = field(default_factory=list)
    agentic_refine_calls: int = 0
    agentic_latency_budget_ms: Optional[float] = None
    hybrid_used: bool = False
    recall_at_k: Optional[float] = None
    attempted_retriever_ids: List[str] = field(default_factory=list)
    fallback_applied: bool = False
    retrieval_error_kind: Optional[str] = None
    embedding_version_filtered_count: int = 0
    embedding_version_warnings: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    chunks: List[RetrievalChunk]
    used: bool
    reason: str
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)
    citations: List["Citation"] = field(default_factory=list)
