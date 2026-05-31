# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""In-process RAG retrieval metrics (Prometheus text + OTLP-style JSON snapshot)."""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Optional, Tuple

_rag_metrics_enabled_override: Optional[bool] = None


def _rag_metrics_enabled() -> bool:
    return os.getenv("INTERGRAX_RAG_METRICS_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def set_rag_metrics_enabled(enabled: bool) -> None:
    global _rag_metrics_enabled_override
    _rag_metrics_enabled_override = enabled


def is_rag_metrics_enabled() -> bool:
    if _rag_metrics_enabled_override is not None:
        return _rag_metrics_enabled_override
    return _rag_metrics_enabled()


@dataclass
class _RagCounter:
    calls: int = 0
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    agentic_iterations: int = 0
    hybrid_calls: int = 0
    hits_total: int = 0
    recall_at_k_sum: float = 0.0
    recall_at_k_samples: int = 0


class RagMetricsCollector:
    """Thread-safe aggregator keyed by (tenant_id, retriever_id, route_tier)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_key: DefaultDict[Tuple[str, str, str], _RagCounter] = defaultdict(_RagCounter)

    def record(
        self,
        *,
        tenant_id: str = "_platform",
        retriever_id: str = "unknown",
        route_tier: str = "standard",
        retrieval_latency_ms: float = 0.0,
        rerank_latency_ms: float = 0.0,
        agentic_iterations: int = 0,
        hybrid_used: bool = False,
        hits: int = 0,
        recall_at_k: Optional[float] = None,
    ) -> None:
        tenant = (tenant_id or "_platform").strip() or "_platform"
        key = (tenant, retriever_id or "unknown", route_tier or "standard")
        with self._lock:
            c = self._by_key[key]
            c.calls += 1
            c.retrieval_latency_ms += float(retrieval_latency_ms or 0.0)
            c.rerank_latency_ms += float(rerank_latency_ms or 0.0)
            c.agentic_iterations += int(agentic_iterations or 0)
            if hybrid_used:
                c.hybrid_calls += 1
            c.hits_total += int(hits or 0)
            if recall_at_k is not None:
                c.recall_at_k_sum += float(recall_at_k)
                c.recall_at_k_samples += 1

    def snapshot_for_tenant(self, tenant_id: str) -> Dict[str, Dict[str, Any]]:
        tenant = (tenant_id or "_platform").strip() or "_platform"
        with self._lock:
            out: Dict[str, Dict[str, Any]] = {}
            for (t, retriever, tier), c in self._by_key.items():
                if t != tenant:
                    continue
                avg_recall = (
                    c.recall_at_k_sum / c.recall_at_k_samples if c.recall_at_k_samples else None
                )
                out[f"{retriever}:{tier}"] = {
                    "calls": c.calls,
                    "retrieval_latency_ms": round(c.retrieval_latency_ms, 2),
                    "rerank_latency_ms": round(c.rerank_latency_ms, 2),
                    "agentic_iterations": c.agentic_iterations,
                    "hybrid_calls": c.hybrid_calls,
                    "hits_total": c.hits_total,
                    "recall_at_k_avg": avg_recall,
                }
            return out

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            out: Dict[str, Dict[str, Any]] = {}
            for (tenant, retriever, tier), c in self._by_key.items():
                avg_recall = (
                    c.recall_at_k_sum / c.recall_at_k_samples if c.recall_at_k_samples else None
                )
                out[f"{tenant}:{retriever}:{tier}"] = {
                    "calls": c.calls,
                    "retrieval_latency_ms": round(c.retrieval_latency_ms, 2),
                    "rerank_latency_ms": round(c.rerank_latency_ms, 2),
                    "agentic_iterations": c.agentic_iterations,
                    "hybrid_calls": c.hybrid_calls,
                    "hits_total": c.hits_total,
                    "recall_at_k_avg": avg_recall,
                }
            return out


_collector = RagMetricsCollector()


def get_rag_metrics_collector() -> RagMetricsCollector:
    return _collector


def record_retrieval(
    *,
    tenant_id: str = "_platform",
    retriever_id: str = "unknown",
    route_tier: str = "standard",
    retrieval_latency_ms: float = 0.0,
    rerank_latency_ms: float = 0.0,
    agentic_iterations: int = 0,
    hybrid_used: bool = False,
    hits: int = 0,
    recall_at_k: Optional[float] = None,
) -> None:
    if not is_rag_metrics_enabled():
        return
    get_rag_metrics_collector().record(
        tenant_id=tenant_id,
        retriever_id=retriever_id,
        route_tier=route_tier,
        retrieval_latency_ms=retrieval_latency_ms,
        rerank_latency_ms=rerank_latency_ms,
        agentic_iterations=agentic_iterations,
        hybrid_used=hybrid_used,
        hits=hits,
        recall_at_k=recall_at_k,
    )
