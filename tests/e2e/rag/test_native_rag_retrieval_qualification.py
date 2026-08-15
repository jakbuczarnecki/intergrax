from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)
from intergrax.rag.document_loaders.bootstrap.default_loader import (
    create_default_documents_loader,
)
from intergrax.rag.document_splitters.bootstrap.default_chunking_engine import (
    create_default_document_splitter,
)
from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.evaluation.metrics import mean_reciprocal_rank, recall_at_k
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import (
    create_default_retriever_manager,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager


pytestmark = [pytest.mark.e2e, pytest.mark.gate]

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "rag_qual3"
CASES_PATH = FIXTURE_DIR / "retrieval_cases.json"
TENANT_ID = "rag-qual-3-tenant"
NAMESPACE = "rag-qual-3"
WORKSPACE_ID = "rag-qual-3-workspace"
RETRIEVAL_K = 3
MIN_CORPUS_RECALL_AT_3 = 0.95
MIN_CORPUS_MRR = 0.90
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class _DeterministicTokenEmbeddingManager(BaseEmbeddingManager):
    """Small offline embedding used only by this qualification gate."""

    dimension = 2048

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN_PATTERN.findall(text.lower()):
                digest = hashlib.blake2b(
                    token.encode("utf-8"),
                    digest_size=8,
                ).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                vectors[row, index] += 1.0 if digest[4] & 1 else -1.0

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        np.divide(vectors, norms, out=vectors, where=norms != 0)
        return vectors

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_documents(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> EmbeddingResult:
        documents_tuple = tuple(documents)
        return EmbeddingResult(
            documents=documents_tuple,
            embeddings=self.embed_texts([doc.content for doc in documents_tuple]),
        )


class _RecordingInMemoryVectorStore(InMemoryVectorStore):
    """Observation-only wrapper; query and ranking remain provider-native."""

    def __init__(self, tenant_id: str) -> None:
        super().__init__(tenant_id=tenant_id)
        self.records: list[VectorStoreRecord] = []

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        self.records.extend(records)
        return super().add_records(records, scope=scope)


def _load_cases() -> list[dict[str, object]]:
    return list(json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"])


def _build_qualification_pipeline() -> tuple[
    IngestPipeline,
    _DeterministicTokenEmbeddingManager,
    _RecordingInMemoryVectorStore,
    VectorstoreManager,
    VectorStoreScope,
    RagProfile,
]:
    profile = RagProfile(
        retriever_id="vector_similarity",
        fast_retriever_id="vector_similarity",
        deep_retriever_id="vector_similarity",
        enable_rerank=False,
        route_mode="off",
        query_expansion="off",
        native_hybrid_enabled=False,
    )
    scope = VectorStoreScope(
        tenant_id=TENANT_ID,
        namespace=NAMESPACE,
        workspace_id=WORKSPACE_ID,
    )
    embedding_manager = _DeterministicTokenEmbeddingManager()
    store = _RecordingInMemoryVectorStore(tenant_id=TENANT_ID)
    vectorstore = VectorstoreManager(store=store, scope=scope)
    pipeline = IngestPipeline(
        loader=create_default_documents_loader(),
        splitter=create_default_document_splitter(),
        embedding_manager=embedding_manager,
        vectorstore=vectorstore,
        profile=profile,
    )
    return pipeline, embedding_manager, store, vectorstore, scope, profile


def _source_id(document: KnowledgeDocument) -> str:
    return str(document.provenance.source_id)


def _matches_source(document: KnowledgeDocument, source_file: str) -> bool:
    return Path(_source_id(document)).name == source_file


def test_native_rag_retrieval_qualification_from_real_fixture_files() -> None:
    cases = _load_cases()
    (
        ingest_pipeline,
        embedding_manager,
        store,
        vectorstore,
        scope,
        profile,
    ) = _build_qualification_pipeline()

    fixture_files = sorted(FIXTURE_DIR.glob("*.txt"))
    assert len(fixture_files) == 3

    ingest_results = [
        ingest_pipeline.run(
            IngestRequest(
                source_path=str(path),
                base_metadata={
                    "tenant_id": TENANT_ID,
                    "namespace": NAMESPACE,
                    "qualification": "RAG-AUD-QUAL-3",
                },
                workspace_id=WORKSPACE_ID,
            )
        )
        for path in fixture_files
    ]

    assert all(result.used for result in ingest_results)
    assert all(result.reason == "ok" for result in ingest_results)
    assert all(result.parser_id for result in ingest_results)
    assert all(
        isinstance(record.document, KnowledgeDocument) for record in store.records
    )
    assert all(
        record.document.scope.tenant_id == TENANT_ID
        and record.document.scope.namespace == NAMESPACE
        and record.document.scope.workspace_id == WORKSPACE_ID
        for record in store.records
    )
    assert vectorstore.count() == len(store.records) > 3

    retriever_manager = create_default_retriever_manager(
        vector_store=vectorstore,
        embedding_manager=embedding_manager,
        profile=profile,
    )
    retrieval_service = RetrievalService(
        retriever_manager=retriever_manager,
        profile=profile,
    )

    recalls_at_3: list[float] = []
    reciprocal_ranks: list[float] = []
    for case in cases:
        source_file = str(case["source_file"])
        fact_marker = str(case["fact_marker"])
        relevant_ids = {
            record.vector_id
            for record in store.records
            if fact_marker in record.document.content
            and _matches_source(record.document, source_file)
        }
        assert relevant_ids, f"fixture fact was not indexed: {case['name']}"
        if case.get("chunk_boundary"):
            assert len(relevant_ids) > 1, case["name"]

        response = retrieval_service.retrieve(
            RetrievalRequest(
                query=str(case["query"]),
                final_top_k=RETRIEVAL_K,
                prefetch_k=RETRIEVAL_K,
                scope=scope,
                route_tier_override="standard",
            )
        )
        assert response.used is True, case["name"]
        retrieved_ids = [chunk.id for chunk in response.chunks]
        recall_at_1 = recall_at_k(retrieved_ids, relevant_ids, 1)
        recall_at_3 = recall_at_k(retrieved_ids, relevant_ids, RETRIEVAL_K)
        mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)
        recalls_at_3.append(recall_at_3)
        reciprocal_ranks.append(mrr)

        diagnostic = (
            f"{case['name']}: retrieved="
            f"{[(chunk.id, chunk.provenance.get('source_id'), chunk.text) for chunk in response.chunks]} "
            f"relevant={sorted(relevant_ids)}"
        )
        assert recall_at_1 >= float(case["min_recall_at_1"]), diagnostic
        assert recall_at_3 >= float(case["min_recall_at_3"]), diagnostic
        assert mrr >= float(case["min_mrr"]), diagnostic
        assert any(
            fact_marker in chunk.text
            and chunk.provenance.get("source_id", "").endswith(source_file)
            for chunk in response.chunks
        ), case["name"]
        assert all(
            chunk.scope.get("tenant_id") == TENANT_ID
            and chunk.scope.get("namespace") == NAMESPACE
            and chunk.scope.get("workspace_id") == WORKSPACE_ID
            for chunk in response.chunks
        ), case["name"]

    assert sum(recalls_at_3) / len(recalls_at_3) >= MIN_CORPUS_RECALL_AT_3
    assert sum(reciprocal_ranks) / len(reciprocal_ranks) >= MIN_CORPUS_MRR
