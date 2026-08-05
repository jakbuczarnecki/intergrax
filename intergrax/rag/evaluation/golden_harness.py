# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Golden retrieval regression harness (M-RAG.11 + extended scenarios)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.evaluation.metrics import recall_at_k
from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager


@dataclass(frozen=True)
class GoldenCaseResult:
    name: str
    scenario: str
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
    results: List[GoldenCaseResult] = []

    for case in cases:
        scenario = str(case.get("scenario", "retrieval")).lower()
        case_profile = _profile_for_case(case, profile)
        if scenario == "graph_rag":
            results.append(_run_graph_case(case, profile=case_profile))
        elif scenario == "agentic":
            results.append(_run_agentic_case(case, profile=case_profile))
        elif scenario == "multi_hop":
            results.append(_run_multi_hop_case(case, profile=case_profile))
        else:
            results.append(_run_retrieval_case(case, profile=case_profile))

    return GoldenRunReport(passed=all(r.passed for r in results), results=results)


def _profile_for_case(case: Dict[str, Any], profile: RagProfile | None) -> RagProfile:
    base = profile or RagProfile(enable_rerank=False, route_mode="off", retriever_id="hybrid")
    overrides = case.get("profile") or {}
    if not overrides:
        return base
    fields = {f.name for f in RagProfile.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    merged = {k: v for k, v in base.__dict__.items() if k in fields}
    merged.update({k: v for k, v in overrides.items() if k in fields})
    return RagProfile(**merged)


def _run_retrieval_case(case: Dict[str, Any], *, profile: RagProfile) -> GoldenCaseResult:
    service, _graph, _manager = _build_service(case, profile=profile, graph_store=None)
    return _evaluate_case(case, service, scenario="retrieval")


def _run_graph_case(case: Dict[str, Any], *, profile: RagProfile) -> GoldenCaseResult:
    tenant_id = str(case.get("tenant_id", "golden"))
    graph = InMemoryGraphStore(tenant_id=tenant_id)
    service, built_graph, manager = _build_service(
        case, profile=profile, graph_store=graph, tenant_id=tenant_id
    )
    if case.get("graph_index", True):
        docs = _documents_from_case(case, tenant_id=tenant_id)
        ids = [item["id"] for item in case["documents"]]
        HeuristicGraphIndexer(built_graph).index_documents(docs, chunk_ids=ids)
    pre_delete = [str(chunk_id) for chunk_id in case.get("pre_delete_chunk_ids", [])]
    if pre_delete:
        built_graph.unlink_chunks(pre_delete)
        manager.delete(pre_delete)
    return _evaluate_case(case, service, scenario="graph_rag")


def _run_multi_hop_case(case: Dict[str, Any], *, profile: RagProfile) -> GoldenCaseResult:
    tenant_id = str(case.get("tenant_id", "golden"))
    graph = InMemoryGraphStore(tenant_id=tenant_id)
    service, _, _manager = _build_service(
        case,
        profile=profile,
        graph_store=graph,
        tenant_id=tenant_id,
    )
    docs = _documents_from_case(case, tenant_id=tenant_id)
    ids = [item["id"] for item in case["documents"]]
    HeuristicGraphIndexer(graph).index_documents(docs, chunk_ids=ids)
    for edge in case.get("graph_edges", []):
        from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode

        src = str(edge["source"])
        tgt = str(edge["target"])
        graph.upsert_node(GraphNode(id=src, label=src.replace("ent:", ""), node_type="entity"))
        graph.upsert_node(GraphNode(id=tgt, label=tgt.replace("ent:", ""), node_type="entity"))
        graph.upsert_edge(GraphEdge(source_id=src, target_id=tgt, relation=str(edge.get("relation", "related_to"))))
        for doc_id in edge.get("chunk_ids", []):
            graph.link_chunk(src, doc_id)
            graph.link_chunk(tgt, doc_id)
    return _evaluate_case(case, service, scenario="multi_hop")


def _run_agentic_case(case: Dict[str, Any], *, profile: RagProfile) -> GoldenCaseResult:
    agentic_profile = RagProfile(
        **{
            **profile.__dict__,
            "agentic_enabled": True,
            "route_mode": "auto",
            "deep_query_min_words": int(case.get("deep_query_min_words", 3)),
        }
    )
    service, _, _manager = _build_service(case, profile=agentic_profile, graph_store=None)
    return _evaluate_case(case, service, scenario="agentic")


def _build_service(
    case: Dict[str, Any],
    *,
    profile: RagProfile,
    graph_store: InMemoryGraphStore | None,
    tenant_id: str = "golden",
):
    from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
        InMemoryVectorStore,
    )

    store = InMemoryVectorStore(tenant_id=tenant_id)
    scope = VectorStoreScope(
        tenant_id=tenant_id,
        namespace=case.get("namespace"),
        workspace_id=case.get("workspace_id"),
    )
    manager = VectorstoreManager(store=store, scope=scope)
    docs = _documents_from_case(case, tenant_id=tenant_id)
    texts = [d.content for d in docs]
    embeddings = [[0.1, 0.2, 0.3] for _ in texts]
    ids = [item["id"] for item in case["documents"]]
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
        embedding_manager=_FakeEmbedder(),
        graph_store=graph_store,
        profile=profile,
    )
    service = RetrievalService(
        retriever_manager=retriever_manager,
        reranker_manager=None,
        profile=profile,
    )
    return service, graph_store, manager


def _documents_from_case(
    case: Dict[str, Any],
    *,
    tenant_id: str,
) -> list[KnowledgeDocument]:
    return [
        KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": {
                    "document_id": str(item["id"]),
                    "root_document_id": str(item["id"]),
                },
                "scope": {
                    "tenant_id": tenant_id,
                    "namespace": case.get("namespace"),
                    "workspace_id": case.get("workspace_id"),
                },
                "content": str(item["text"]),
                "metadata": {"doc_id": str(item["id"])},
                "provenance": {
                    "source_kind": "golden_fixture",
                    "source_id": str(item["id"]),
                },
            }
        )
        for item in case["documents"]
    ]


def _evaluate_case(
    case: Dict[str, Any],
    service: RetrievalService,
    *,
    scenario: str,
) -> GoldenCaseResult:
    response = service.retrieve(
        RetrievalRequest(query=case["query"], top_k=int(case.get("k", 2)))
    )
    retrieved_ids = [c.id for c in response.chunks]
    relevant = set(case.get("relevant_ids", []))
    k = int(case.get("k", 2))
    min_recall = float(case.get("min_recall", 1.0))
    rec = recall_at_k(retrieved_ids, relevant, k)
    response.trace.recall_at_k = rec
    if case.get("expect_empty"):
        passed = len(retrieved_ids) == 0
    else:
        passed = rec >= min_recall
    return GoldenCaseResult(
        name=str(case.get("name", "case")),
        scenario=scenario,
        recall=rec,
        min_recall=min_recall,
        passed=passed,
        retrieved_ids=retrieved_ids,
    )


class _FakeEmbedder:
    def embed_one(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]
