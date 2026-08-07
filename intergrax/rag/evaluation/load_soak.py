# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Concurrent retrieval load/soak SLO contract for CI and nightly probes (M-RAG.36)."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.evaluation.metrics import recall_at_k
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager


@dataclass(frozen=True)
class SoakQuery:
    query: str
    relevant_ids: frozenset[str]
    k: int
    min_recall: float


@dataclass(frozen=True)
class LoadSoakConfig:
    """Harness concurrent retrieve scenario for latency + recall regression."""

    concurrent_workers: int = 4
    queries_per_worker: int = 3
    max_p95_latency_ms: float = 2_000.0
    min_recall_floor: float = 0.0


@dataclass(frozen=True)
class LoadSoakResult:
    passed: bool
    concurrent_workers: int = 0
    queries_executed: int = 0
    p95_latency_ms: float = 0.0
    min_observed_recall: float = 1.0
    reason: str = ""


def soak_queries_from_golden_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    scenarios: frozenset[str] = frozenset({"retrieval"}),
) -> List[SoakQuery]:
    queries: List[SoakQuery] = []
    for case in cases:
        scenario = str(case.get("scenario", "retrieval")).lower()
        if scenario not in scenarios:
            continue
        queries.append(
            SoakQuery(
                query=str(case["query"]),
                relevant_ids=frozenset(case.get("relevant_ids", [])),
                k=int(case.get("k", 2)),
                min_recall=float(case.get("min_recall", 1.0)),
            )
        )
    return queries


def build_soak_retrieval_service(
    cases: Sequence[Mapping[str, Any]],
    *,
    profile: RagProfile | None = None,
    scenarios: frozenset[str] = frozenset({"retrieval"}),
) -> RetrievalService:
    """Merge golden retrieval documents into one in-memory service for load probes."""
    from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
        InMemoryVectorStore,
    )

    resolved = profile or RagProfile(enable_rerank=False, route_mode="off", retriever_id="hybrid")
    docs_by_id: Dict[str, KnowledgeDocument] = {}
    for case in cases:
        scenario = str(case.get("scenario", "retrieval")).lower()
        if scenario not in scenarios:
            continue
        for item in case.get("documents", []):
            doc_id = str(item["id"])
            docs_by_id[doc_id] = KnowledgeDocument.model_validate(
                {
                    "schema_version": 1,
                    "identity": {
                        "document_id": doc_id,
                        "root_document_id": doc_id,
                    },
                    "scope": {"tenant_id": "soak"},
                    "content": str(item["text"]),
                    "metadata": {"doc_id": doc_id},
                    "provenance": {
                        "source_kind": "load_soak_fixture",
                        "source_id": doc_id,
                    },
                }
            )

    store = InMemoryVectorStore(tenant_id="soak")
    scope = VectorStoreScope(tenant_id="soak")
    manager = VectorstoreManager(store=store, scope=scope)
    docs = list(docs_by_id.values())
    ids = list(docs_by_id.keys())
    embedder = _FakeEmbedder()
    embeddings = [embedder.embed_one(document.content) for document in docs]
    manager.add_records(
        [
            VectorStoreRecord(
                document=document,
                embedding=embedding,
                vector_id=doc_id,
            )
            for document, embedding, doc_id in zip(docs, embeddings, ids)
        ],
        scope=scope,
    )

    from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import (
        create_default_retriever_manager,
    )

    retriever_manager = create_default_retriever_manager(
        vector_store=manager,
        embedding_manager=embedder,
        graph_store=None,
        profile=resolved,
    )
    return RetrievalService(
        retriever_manager=retriever_manager,
        reranker_manager=None,
        profile=resolved,
    )


def run_retrieval_load_soak(
    service: RetrievalService,
    queries: Sequence[SoakQuery],
    *,
    config: LoadSoakConfig | None = None,
) -> LoadSoakResult:
    """
    Execute concurrent ``RetrievalService.retrieve`` calls and enforce SLO budgets.

    Fails when p95 latency exceeds ``max_p95_latency_ms`` or any query recall drops
    below its per-query ``min_recall`` / global ``min_recall_floor``.
    """
    cfg = config or LoadSoakConfig()
    if not queries:
        return LoadSoakResult(passed=False, reason="no_soak_queries")

    workers = max(1, int(cfg.concurrent_workers))
    per_worker = max(1, int(cfg.queries_per_worker))
    task_count = workers * per_worker

    latencies_ms: list[float] = []
    recalls: list[float] = []
    executed = 0

    def _run_one(query: SoakQuery) -> tuple[SoakQuery, float, float]:
        started = time.perf_counter()
        response = service.retrieve(
            RetrievalRequest(
                query=query.query,
                top_k=query.k,
                scope=VectorStoreScope(tenant_id="soak"),
            )
        )
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        retrieved_ids = [chunk.id for chunk in response.chunks]
        rec = recall_at_k(retrieved_ids, set(query.relevant_ids), query.k)
        return query, elapsed_ms, rec

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_run_one, queries[i % len(queries)])
                for i in range(task_count)
            ]
            for future in as_completed(futures):
                query, elapsed_ms, rec = future.result()
                latencies_ms.append(elapsed_ms)
                recalls.append(rec)
                executed += 1
                if rec < query.min_recall:
                    min_observed = min(recalls) if recalls else rec
                    return LoadSoakResult(
                        passed=False,
                        concurrent_workers=workers,
                        queries_executed=executed,
                        p95_latency_ms=_p95(latencies_ms),
                        min_observed_recall=min_observed,
                        reason=f"recall_regression:{rec:.3f}<{query.min_recall:.3f}",
                    )
    except Exception as exc:
        return LoadSoakResult(
            passed=False,
            concurrent_workers=workers,
            queries_executed=executed,
            reason=f"retrieve_failed:{exc}",
        )

    min_observed = min(recalls) if recalls else 0.0

    if min_observed < float(cfg.min_recall_floor):
        return LoadSoakResult(
            passed=False,
            concurrent_workers=workers,
            queries_executed=executed,
            p95_latency_ms=_p95(latencies_ms),
            min_observed_recall=min_observed,
            reason=f"recall_floor:{min_observed:.3f}<{cfg.min_recall_floor:.3f}",
        )

    p95_ms = _p95(latencies_ms)
    if p95_ms > cfg.max_p95_latency_ms:
        return LoadSoakResult(
            passed=False,
            concurrent_workers=workers,
            queries_executed=executed,
            p95_latency_ms=p95_ms,
            min_observed_recall=min_observed,
            reason=f"slo_latency_exceeded:p95={p95_ms:.2f}>{cfg.max_p95_latency_ms}",
        )

    return LoadSoakResult(
        passed=True,
        concurrent_workers=workers,
        queries_executed=executed,
        p95_latency_ms=p95_ms,
        min_observed_recall=min_observed,
        reason="ok",
    )


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    index = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[index]


def export_load_soak_report(
    result: LoadSoakResult,
    path: Path,
    *,
    generated_at: str,
) -> None:
    """Write nightly load/soak evidence artifact (RAG-MAINT-02)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": generated_at,
        "passed": result.passed,
        "concurrent_workers": result.concurrent_workers,
        "queries_executed": result.queries_executed,
        "p95_latency_ms": result.p95_latency_ms,
        "min_observed_recall": result.min_observed_recall,
        "reason": result.reason,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class _FakeEmbedder:
    def embed_one(self, text: str) -> List[float]:
        vector = [0.0] * 64
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:2], "big") % 64
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_one(text) for text in texts]
