# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Golden retrieval regression harness (M-RAG.11)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from langchain_core.documents import Document

from intergrax.rag.evaluation.metrics import recall_at_k
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import create_default_retriever_manager
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager


@dataclass(frozen=True)
class GoldenCaseResult:
    name: str
    recall: float
    min_recall: float
    passed: bool
    retrieved_ids: List[str]


@dataclass(frozen=True)
class GoldenRunReport:
    passed: bool
    results: List[GoldenCaseResult]


def load_golden_cases(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("cases", []))


def run_golden_retrieval(
    cases: Sequence[Dict[str, Any]],
    *,
    profile: RagProfile | None = None,
) -> GoldenRunReport:
    profile = profile or RagProfile(enable_rerank=False, route_mode="off", retriever_id="hybrid")
    results: List[GoldenCaseResult] = []

    for case in cases:
        store = InMemoryVectorStore(tenant_id="golden")
        manager = VectorstoreManager(store=store)
        docs = [
            Document(page_content=item["text"], metadata={"doc_id": item["id"]})
            for item in case["documents"]
        ]
        texts = [d.page_content for d in docs]
        embeddings = [[0.1, 0.2, 0.3] for _ in texts]
        for i, d in enumerate(docs):
            d.metadata["tenant_id"] = "golden"
        ids = [item["id"] for item in case["documents"]]
        manager.add_documents(docs, embeddings, ids=ids)

        retriever_manager = create_default_retriever_manager(
            vector_store=manager,
            embedding_manager=_FakeEmbedder(),
        )
        service = RetrievalService(
            retriever_manager=retriever_manager,
            reranker_manager=None,
            profile=profile,
        )
        response = service.retrieve(
            RetrievalRequest(query=case["query"], top_k=int(case.get("k", 2)))
        )
        retrieved_ids = [c.id for c in response.chunks]
        relevant = set(case.get("relevant_ids", []))
        k = int(case.get("k", 2))
        min_recall = float(case.get("min_recall", 1.0))
        rec = recall_at_k(retrieved_ids, relevant, k)
        results.append(
            GoldenCaseResult(
                name=str(case.get("name", "case")),
                recall=rec,
                min_recall=min_recall,
                passed=rec >= min_recall,
                retrieved_ids=retrieved_ids,
            )
        )

    return GoldenRunReport(passed=all(r.passed for r in results), results=results)


class _FakeEmbedder:
    def embed_one(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]
