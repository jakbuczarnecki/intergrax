# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.rag.retrieval.retrieval_result import RetrievalTrace
from intergrax.rag.tracking.metrics import (
    get_rag_metrics_collector,
    record_retrieval,
    set_rag_metrics_enabled,
)

pytestmark = pytest.mark.unit


def test_rag_metrics_collector_records_hybrid_and_recall() -> None:
    set_rag_metrics_enabled(True)
    collector = get_rag_metrics_collector()
    record_retrieval(
        tenant_id="t1",
        retriever_id="hybrid",
        route_tier="standard",
        retrieval_latency_ms=12.5,
        rerank_latency_ms=3.0,
        hybrid_used=True,
        hits=4,
        recall_at_k=0.75,
    )
    snap = collector.snapshot_for_tenant("t1")
    assert "hybrid:standard" in snap
    row = snap["hybrid:standard"]
    assert row["calls"] >= 1
    assert row["hybrid_calls"] >= 1
    assert row["recall_at_k_avg"] == 0.75


def test_retrieval_trace_extended_fields() -> None:
    trace = RetrievalTrace(
        hybrid_used=True,
        agentic_total_latency_ms=42.0,
        recall_at_k=1.0,
    )
    assert trace.hybrid_used is True
    assert trace.agentic_total_latency_ms == 42.0
