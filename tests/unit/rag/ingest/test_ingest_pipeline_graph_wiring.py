# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.ingest import ingest_pipeline as ingest_pipeline_module
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile

pytestmark = pytest.mark.unit


class _NativeLoader:
    def __init__(self, document: KnowledgeDocument) -> None:
        self.document = document

    def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
        del source, kwargs
        return [self.document]


class _NativeSplitter:
    def __init__(self) -> None:
        self.received: list[KnowledgeDocument] | None = None

    def split_documents(
        self,
        documents: Sequence[KnowledgeDocument],
        strategy_id: str | None = None,
    ) -> list[KnowledgeDocument]:
        del strategy_id
        self.received = list(documents)
        return list(documents)


class _EmbeddingManager:
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)

    def embed_documents(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> EmbeddingResult:
        native_documents = tuple(documents)
        return EmbeddingResult(
            documents=native_documents,
            embeddings=np.ones((len(native_documents), 2), dtype=np.float32),
        )


class _VectorstoreManager:
    def __init__(self) -> None:
        self.records: list[object] = []
        self.scope: object = None

    def add_records(self, records: Sequence[object], *, scope: object = None) -> list[str]:
        self.records.extend(records)
        self.scope = scope
        return [record.vector_id for record in records]  # type: ignore[attr-defined]

    def list_source_record_ids(
        self,
        *,
        source_id: str,
        scope: object,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                record.vector_id
                for record in self.records
                if record.document.provenance.source_id == source_id  # type: ignore[attr-defined]
                and scope.matches_document(record.document)  # type: ignore[attr-defined]
            )
        )

    def count(self, *, scope: object) -> int:
        return sum(
            scope.matches_document(record.document)  # type: ignore[attr-defined]
            for record in self.records
        )

    def delete(self, ids: Sequence[str], *, scope: object) -> None:
        stale_ids = set(ids)
        self.records = [
            record
            for record in self.records
            if record.vector_id not in stale_ids  # type: ignore[attr-defined]
            or not scope.matches_document(record.document)  # type: ignore[attr-defined]
        ]


class _RecordingGraphIndexer:
    def __init__(self) -> None:
        self.documents: list[KnowledgeDocument] | None = None
        self.chunk_ids: list[str] | None = None

    def index_documents(
        self,
        documents: Sequence[KnowledgeDocument],
        *,
        chunk_ids: Sequence[str] | None = None,
    ) -> None:
        self.documents = list(documents)
        self.chunk_ids = list(chunk_ids or [])


def _native_document(source: Path, document_id: str = "document-1") -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {"tenant_id": "tenant-a", "namespace": "docs"},
            "content": "native document content",
            "metadata": {"loader_marker": "native"},
            "provenance": {
                "source_kind": "file",
                "source_id": str(source),
                "provider_id": "test-loader",
            },
        }
    )


def _build_pipeline(
    *,
    source: Path,
    dual_index: bool,
    graph_store: object | None,
    graph_rag_enabled: bool,
) -> tuple[IngestPipeline, KnowledgeDocument, _NativeSplitter, _RecordingGraphIndexer]:
    native_document = _native_document(source)
    splitter = _NativeSplitter()
    graph_indexer = _RecordingGraphIndexer()
    vectorstore = _VectorstoreManager()
    toc_vectorstore = _VectorstoreManager() if dual_index else None

    pipeline = IngestPipeline(
        loader=_NativeLoader(native_document),
        splitter=splitter,
        embedding_manager=_EmbeddingManager(),
        vectorstore=vectorstore,
        toc_vectorstore=toc_vectorstore,
        profile=RagProfile(
            graph_rag_enabled=graph_rag_enabled,
            hierarchical_index_enabled=dual_index,
            chunking_strategy_id="recursive",
        ),
        graph_store=graph_store,
    )
    return pipeline, native_document, splitter, graph_indexer


@pytest.mark.parametrize("dual_index", [False, True], ids=["single-index", "dual-index"])
def test_ingest_passes_native_chunks_to_graph_in_all_index_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dual_index: bool,
) -> None:
    source = tmp_path / "native.txt"
    source.write_text("native document content", encoding="utf-8")
    graph_store = object()
    pipeline, original_document, splitter, graph_indexer = _build_pipeline(
        source=source,
        dual_index=dual_index,
        graph_store=graph_store,
        graph_rag_enabled=True,
    )
    factory_calls: list[tuple[object, RagProfile]] = []

    def _recording_factory(
        store: object,
        profile: RagProfile,
        *,
        llm: object = None,
    ) -> _RecordingGraphIndexer:
        del llm
        factory_calls.append((store, profile))
        return graph_indexer

    monkeypatch.setattr(ingest_pipeline_module, "resolve_graph_indexer", _recording_factory)
    original_payload = deepcopy(original_document.model_dump(mode="python"))

    result = pipeline.run(
        IngestRequest(
            source_path=str(source),
            base_metadata={
                "tenant_id": "tenant-a",
                "namespace": "docs",
                "workspace_id": "spoofed-metadata-workspace",
                "user_label": "user metadata",
            },
            workspace_id="workspace-a",
            chunking_strategy_id="recursive",
        )
    )

    assert result.used is True
    persisted_records = pipeline._vectorstore.records  # type: ignore[attr-defined]
    persisted_ids = [record.vector_id for record in persisted_records]
    assert result.vector_ids == persisted_ids
    assert all(
        record.vector_id == record.document.identity.document_id
        for record in persisted_records
    )
    assert graph_indexer.documents is not None
    assert graph_indexer.chunk_ids == result.vector_ids
    assert all(isinstance(document, KnowledgeDocument) for document in graph_indexer.documents)
    assert all(type(document).__name__ != "Document" for document in graph_indexer.documents)
    assert splitter.received is not None
    assert graph_indexer.documents[0].scope.tenant_id == "tenant-a"
    assert graph_indexer.documents[0].scope.namespace == "docs"
    assert graph_indexer.documents[0].scope.workspace_id == "workspace-a"
    assert graph_indexer.documents[0].identity.document_id == "document-1"
    assert graph_indexer.documents[0].content == "native document content"
    assert graph_indexer.documents[0].provenance.source_id == str(source)
    assert graph_indexer.documents[0].metadata["user_label"] == "user metadata"
    assert "workspace_id" not in graph_indexer.documents[0].metadata
    assert original_document.model_dump(mode="python") == original_payload
    assert factory_calls == [(graph_store, pipeline._profile)]  # type: ignore[attr-defined]


def test_ingest_ids_are_not_based_on_same_stem_for_distinct_documents(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first" / "same.txt"
    second_source = tmp_path / "second" / "same.txt"
    first_source.parent.mkdir()
    second_source.parent.mkdir()
    first_source.write_text("first", encoding="utf-8")
    second_source.write_text("second", encoding="utf-8")

    pipeline, _original_document, _splitter, _graph_indexer = _build_pipeline(
        source=first_source,
        dual_index=False,
        graph_store=None,
        graph_rag_enabled=False,
    )
    loader = pipeline._loader  # type: ignore[attr-defined]

    first_result = pipeline.run(
        IngestRequest(
            source_path=str(first_source),
            base_metadata={"tenant_id": "tenant-a"},
            chunking_strategy_id="recursive",
        )
    )

    loader.document = _native_document(second_source, document_id="document-2")
    second_result = pipeline.run(
        IngestRequest(
            source_path=str(second_source),
            base_metadata={"tenant_id": "tenant-a"},
            chunking_strategy_id="recursive",
        )
    )

    assert first_result.vector_ids != second_result.vector_ids
    assert all(
        not vector_id.startswith("ingest-same-")
        for vector_id in first_result.vector_ids + second_result.vector_ids
    )


@pytest.mark.parametrize(
    ("graph_store", "graph_rag_enabled"),
    [(None, True), (object(), False)],
    ids=["no-graph-store", "graph-disabled"],
)
def test_ingest_does_not_call_graph_indexer_when_graph_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    graph_store: object | None,
    graph_rag_enabled: bool,
) -> None:
    source = tmp_path / "native.txt"
    source.write_text("native document content", encoding="utf-8")
    pipeline, _original_document, _splitter, graph_indexer = _build_pipeline(
        source=source,
        dual_index=False,
        graph_store=graph_store,
        graph_rag_enabled=graph_rag_enabled,
    )
    factory_calls = 0

    def _unexpected_factory(*args: object, **kwargs: object) -> _RecordingGraphIndexer:
        del args, kwargs
        nonlocal factory_calls
        factory_calls += 1
        return graph_indexer

    monkeypatch.setattr(ingest_pipeline_module, "resolve_graph_indexer", _unexpected_factory)

    result = pipeline.run(
        IngestRequest(
            source_path=str(source),
            base_metadata={"tenant_id": "tenant-a", "namespace": "docs"},
            workspace_id="workspace-a",
            chunking_strategy_id="recursive",
        )
    )

    assert result.used is True
    assert factory_calls == 0
    assert graph_indexer.documents is None
