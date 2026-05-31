# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RetrievalChunk:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    route_tier: str = "standard"
    retriever_id: str = ""
    reranker_id: Optional[str] = None
    rerank_enabled: bool = False
    candidates_before_rerank: int = 0
    candidates_after_rerank: int = 0
    retrieval_latency_ms: Optional[float] = None
    rerank_latency_ms: Optional[float] = None
    agentic_iteration: Optional[int] = None
    agentic_stopped: Optional[str] = None


@dataclass
class RetrievalResult:
    chunks: List[RetrievalChunk]
    used: bool
    reason: str
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)
