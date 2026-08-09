from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import (
    create_default_retriever_manager,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = [pytest.mark.integration, pytest.mark.gate]

TENANT_ID = "reingest-7b-tenant"
NAMESPACE = "reingest-7b"
WORKSPACE_ID = "reingest-7b-workspace"
SCOPE = VectorStoreScope(
    tenant_id=TENANT_ID,
    namespace=NAMESPACE,
    workspace_id=WORKSPACE_ID,
)


class _FileLoader:
    def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
        del kwargs
        source_path = Path(source).resolve()
        source_id = str(source_path)
        root_id = "root-" + hashlib.sha256(source_id.encode()).hexdigest()[:16]
        return [
            KnowledgeDocument.model_validate(
                {
                    "schema_version": 1,
                    "identity": {
                        "document_id": root_id,
                        "root_document_id": root_id,
                    },
                    "scope": {
                        "tenant_id": TENANT_ID,
                        "namespace": NAMESPACE,
                        "workspace_id": WORKSPACE_ID,
                    },
                    "content": source_path.read_text(encoding="utf-8"),
                    "metadata": {},
                    "provenance": {
                        "source_kind": "file",
                        "source_id": source_id,
                    },
                }
            )
        ]


class _FileSplitter:
    def split_documents(
        self,
        documents: Sequence[KnowledgeDocument],
        strategy_id: str | None = None,
    ) -> list[KnowledgeDocument]:
        del strategy_id
        chunks: list[KnowledgeDocument] = []
        for document in documents:
            chunks.extend(
                build_derived_chunk(
                    document,
                    content=content.strip(),
                    strategy_id="reingest-7b",
                    chunk_index=index,
                )
                for index, content in enumerate(document.content.split("|"))
                if content.strip()
            )
        return chunks


class _MarkerEmbeddingManager(BaseEmbeddingManager):
    dimension = 3

    def __init__(self) -> None:
        self.fail_on: str | None = None

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if self.fail_on and any(self.fail_on in text for text in texts):
            raise RuntimeError("controlled embedding failure")
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for index, text in enumerate(texts):
            if "A-old" in text:
                vectors[index, 0] = 1.0
            elif "A-new" in text:
                vectors[index, 1] = 1.0
            elif "B-stable" in text:
                vectors[index, 2] = 1.0
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
            embeddings=self.embed_texts([document.content for document in documents_tuple]),
        )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build() -> tuple[
    IngestPipeline,
    _MarkerEmbeddingManager,
    VectorstoreManager,
    RetrievalService,
]:
    profile = RagProfile(
        retriever_id="vector_similarity",
        fast_retriever_id="vector_similarity",
        deep_retriever_id="vector_similarity",
        enable_rerank=False,
        route_mode="off",
        native_hybrid_enabled=False,
    )
    embedding = _MarkerEmbeddingManager()
    vectorstore = VectorstoreManager(
        InMemoryVectorStore(tenant_id=TENANT_ID),
        scope=SCOPE,
    )
    pipeline = IngestPipeline(
        loader=_FileLoader(),
        splitter=_FileSplitter(),
        embedding_manager=embedding,
        vectorstore=vectorstore,
        profile=profile,
    )
    retrieval = RetrievalService(
        retriever_manager=create_default_retriever_manager(
            vector_store=vectorstore,
            embedding_manager=embedding,
            profile=profile,
        ),
        profile=profile,
    )
    return pipeline, embedding, vectorstore, retrieval


def _ingest(pipeline: IngestPipeline, path: Path) -> list[str]:
    result = pipeline.run(
        IngestRequest(
            source_path=str(path),
            base_metadata={"tenant_id": TENANT_ID, "namespace": NAMESPACE},
            workspace_id=WORKSPACE_ID,
        )
    )
    assert result.used is True
    assert result.reason == "ok"
    return result.vector_ids


def _source_ids(vectorstore: VectorstoreManager, path: Path) -> set[str]:
    return set(
        vectorstore.list_source_record_ids(
            source_id=str(path.resolve()),
            scope=SCOPE,
        )
    )


def test_source_reingest_replaces_only_current_source_safely(tmp_path: Path) -> None:
    pipeline, embedding, vectorstore, retrieval = _build()
    source_a = tmp_path / "a" / "same.txt"
    source_b = tmp_path / "b" / "same.txt"

    _write(source_a, "A-old alpha|A-old tail")
    _write(source_b, "B-stable beta")
    a_v1 = set(_ingest(pipeline, source_a))
    b_v1 = set(_ingest(pipeline, source_b))
    assert len(a_v1) == 2
    assert len(b_v1) == 1
    assert _source_ids(vectorstore, source_a) == a_v1

    _write(source_a, "A-new alpha|A-new tail")
    a_v2 = set(_ingest(pipeline, source_a))
    assert len(a_v2) == 2
    assert a_v1.isdisjoint(a_v2)
    assert _source_ids(vectorstore, source_a) == a_v2
    assert _source_ids(vectorstore, source_b) == b_v1

    _write(source_a, "A-new alpha")
    a_v3 = set(_ingest(pipeline, source_a))
    assert len(a_v3) == 1
    assert _source_ids(vectorstore, source_a) == a_v3
    assert vectorstore.count(scope=SCOPE) == len(a_v3) + len(b_v1)

    repeated = set(_ingest(pipeline, source_a))
    assert repeated == a_v3
    assert _source_ids(vectorstore, source_a) == a_v3
    assert vectorstore.count(scope=SCOPE) == len(a_v3) + len(b_v1)

    current_a = retrieval.retrieve(
        RetrievalRequest(
            query="A-new",
            scope=SCOPE,
            final_top_k=5,
            prefetch_k=5,
            route_tier_override="standard",
        )
    )
    assert current_a.used is True
    assert any("A-new" in chunk.text for chunk in current_a.chunks)
    assert all("A-old" not in chunk.text for chunk in current_a.chunks)
    assert all(
        chunk.scope.get("tenant_id") == TENANT_ID
        and chunk.scope.get("namespace") == NAMESPACE
        and chunk.scope.get("workspace_id") == WORKSPACE_ID
        for chunk in current_a.chunks
    )

    old_a = retrieval.retrieve(
        RetrievalRequest(
            query="A-old",
            scope=SCOPE,
            final_top_k=5,
            prefetch_k=5,
            route_tier_override="standard",
        )
    )
    assert all("A-old" not in chunk.text for chunk in old_a.chunks)

    current_b = retrieval.retrieve(
        RetrievalRequest(
            query="B-stable",
            scope=SCOPE,
            final_top_k=5,
            prefetch_k=5,
            route_tier_override="standard",
        )
    )
    assert current_b.used is True
    assert any("B-stable" in chunk.text for chunk in current_b.chunks)
    assert all("A-old" not in chunk.text for chunk in current_b.chunks)

    ids_before_failure = _source_ids(vectorstore, source_a)
    _write(source_a, "A-failing alpha")
    embedding.fail_on = "A-failing"
    with pytest.raises(RuntimeError, match="controlled embedding failure"):
        pipeline.run(
            IngestRequest(
                source_path=str(source_a),
                base_metadata={"tenant_id": TENANT_ID, "namespace": NAMESPACE},
                workspace_id=WORKSPACE_ID,
            )
        )
    embedding.fail_on = None
    assert _source_ids(vectorstore, source_a) == ids_before_failure
    preserved = retrieval.retrieve(
        RetrievalRequest(
            query="A-new",
            scope=SCOPE,
            final_top_k=5,
            prefetch_k=5,
            route_tier_override="standard",
        )
    )
    assert any("A-new" in chunk.text for chunk in preserved.chunks)
