from __future__ import annotations

import importlib.metadata
from collections.abc import Sequence

import numpy as np
import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)
from intergrax.rag.document_splitters.bootstrap.default_chunking_engine import (
    create_default_document_splitter,
)
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import (
    BaseChunkingStrategy,
)
from intergrax.rag.document_splitters.registry.strategy_registry import (
    ChunkingStrategyRegistry,
)
from intergrax.rag.document_splitters.strategies.recursive_chunking_strategy import (
    RecursiveChunkingStrategy,
)
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
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    BaseRetrieverPlugin,
    RetrievalHit,
    RetrieverQuery,
)
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry
from intergrax.rag.rerankers.bootstrap.reranker_bootstrap import (
    create_default_reranker_engine,
)
from intergrax.rag.rerankers.contracts.base_reranker import (
    BaseReranker,
    BaseRerankerPlugin,
)
from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
    RerankerResult,
)
from intergrax.rag.rerankers.re_ranker_manager import ReRankerManager
from intergrax.rag.rerankers.registry.reranker_registry import RerankerRegistry
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = pytest.mark.unit


class _EntryPoint:
    def __init__(self, name: str, value: str, group: str) -> None:
        self.name = name
        self.value = value
        self.group = group


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


def _patch_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    plugins: dict[str, Sequence[type]],
) -> None:
    entries = [
        _EntryPoint(
            f"{group.rsplit('.', maxsplit=1)[-1]}-{index}",
            f"{plugin.__module__}:{plugin.__name__}",
            group,
        )
        for group, plugin_types in plugins.items()
        for index, plugin in enumerate(plugin_types)
    ]
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints(entries),
    )


def _source(
    document_id: str = "doc-plugin",
    *,
    namespace: str | None = None,
) -> KnowledgeDocument:
    scope = {"tenant_id": "tenant.plugin"}
    if namespace is not None:
        scope["namespace"] = namespace
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": scope,
            "content": "external plugin content",
            "metadata": {"source": document_id},
            "provenance": {
                "source_kind": "test",
                "source_id": document_id,
            },
        }
    )


class _ExternalChunker(BaseChunkingStrategy):
    @classmethod
    def strategy_id(cls) -> str:
        return "external_chunker"

    def chunk(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> Sequence[KnowledgeDocument]:
        return [
            build_derived_chunk(
                document,
                content=f"external:{document.content}",
                strategy_id=self.strategy_id(),
                chunk_index=0,
            )
            for document in documents
        ]


class _ExternalLoader:
    def __init__(self, document: KnowledgeDocument | None = None) -> None:
        self.document = document or _source("doc-plugin-ingest")

    def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
        del source, kwargs
        return [self.document]


class _OfflineEmbeddingManager(BaseEmbeddingManager):
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [
                [float("external" in text), float("plugin" in text)]
                for text in texts
            ],
            dtype=np.float32,
        )

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


class _ConflictingChunker(_ExternalChunker):
    @classmethod
    def strategy_id(cls) -> str:
        return "recursive"


class _DuplicateChunker(_ExternalChunker):
    @classmethod
    def strategy_id(cls) -> str:
        return "duplicate_chunker"


class _ExternalRetriever(BaseRetriever):
    requires_query_embedding = False

    @classmethod
    def name(cls) -> str:
        return "external_retriever"

    def retrieve(self, query: RetrieverQuery) -> Sequence[RetrievalHit]:
        return [
            RetrievalHit(
                document=_source("retrieved-document"),
                score=0.91,
                rank=0,
                channel="external",
                retriever_name=self.name(),
                query_text=query.query_text,
            )
        ]


class _DependencyAwareRetriever(BaseRetriever):
    requires_query_embedding = False

    def __init__(self, vector_store: object) -> None:
        self.vector_store = vector_store

    @classmethod
    def name(cls) -> str:
        return "external_retriever"

    def retrieve(self, query: RetrieverQuery) -> Sequence[RetrievalHit]:
        return [
            RetrievalHit(
                document=_source("retrieved-document"),
                score=0.91,
                rank=0,
                channel="external",
                retriever_name=self.name(),
                query_text=query.query_text,
            )
        ]


class _ExternalRetrieverPlugin(BaseRetrieverPlugin):
    received_vector_store: object | None = None

    @classmethod
    def create(
        cls,
        *,
        vector_store: object,
        embedding_manager: object,
        graph_store: object | None = None,
        profile: RagProfile | None = None,
        llm_for_query_expansion: object | None = None,
        toc_vector_store: object | None = None,
    ) -> BaseRetriever:
        del embedding_manager, graph_store, profile, llm_for_query_expansion, toc_vector_store
        cls.received_vector_store = vector_store
        return _DependencyAwareRetriever(vector_store)


class _ExternalReranker(BaseReranker):
    @classmethod
    def name(cls) -> str:
        return "external_reranker"

    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RerankerCandidate],
        limit: int | None = None,
    ) -> Sequence[RerankerResult]:
        del query
        results = [
            RerankerResult(
                candidate=candidate,
                rerank_score=0.99,
                fusion_score=None,
                rank=index,
            )
            for index, candidate in enumerate(candidates)
        ]
        return results if limit is None else results[:limit]


class _DependencyAwareReranker(BaseReranker):
    def __init__(self, embedding_manager: object) -> None:
        self.embedding_manager = embedding_manager

    @classmethod
    def name(cls) -> str:
        return "external_reranker"

    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RerankerCandidate],
        limit: int | None = None,
    ) -> Sequence[RerankerResult]:
        del query
        results = [
            RerankerResult(
                candidate=candidate,
                rerank_score=0.99,
                fusion_score=None,
                rank=index,
            )
            for index, candidate in enumerate(candidates)
        ]
        return results if limit is None else results[:limit]


class _ExternalRerankerPlugin(BaseRerankerPlugin):
    received_embedding_manager: object | None = None

    @classmethod
    def create(cls, *, embedding_manager: object) -> BaseReranker:
        cls.received_embedding_manager = embedding_manager
        return _DependencyAwareReranker(embedding_manager)


def test_external_chunker_entry_point_uses_normal_splitter_and_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        {"intergrax.rag.chunkers": (_ExternalChunker,)},
    )
    splitter = create_default_document_splitter(
        registry=ChunkingStrategyRegistry(),
        discover_entry_points=True,
    )
    profile = RagProfile(chunking_strategy_id="external_chunker")

    chunks = splitter.split_documents(
        [_source()],
        strategy_id=profile.chunking_strategy_id,
    )

    assert len(chunks) == 1
    assert isinstance(chunks[0], KnowledgeDocument)
    assert chunks[0].content == "external:external plugin content"


def test_external_chunker_entry_point_flows_through_ingest_and_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _patch_entry_points(
        monkeypatch,
        {"intergrax.rag.chunkers": (_ExternalChunker,)},
    )
    profile = RagProfile(
        chunking_strategy_id="external_chunker",
        retriever_id="vector_similarity",
        fast_retriever_id="vector_similarity",
        deep_retriever_id="vector_similarity",
        enable_rerank=False,
        route_mode="off",
        query_expansion="off",
        native_hybrid_enabled=False,
    )
    scope = VectorStoreScope(
        tenant_id="tenant.plugin",
        namespace="plugin",
        workspace_id="workspace.plugin",
    )
    embedding_manager = _OfflineEmbeddingManager()
    vectorstore = VectorstoreManager(
        store=InMemoryVectorStore(tenant_id=scope.tenant_id),
        scope=scope,
    )
    splitter = create_default_document_splitter(discover_entry_points=True)
    source = tmp_path / "plugin.txt"
    source.write_text("plugin fixture", encoding="utf-8")
    pipeline = IngestPipeline(
        loader=_ExternalLoader(
            _source("doc-plugin-ingest", namespace=scope.namespace)
        ),
        splitter=splitter,
        embedding_manager=embedding_manager,
        vectorstore=vectorstore,
        profile=profile,
    )

    ingest = pipeline.run(
        IngestRequest(
            source_path=str(source),
            base_metadata={
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
            },
            workspace_id=scope.workspace_id,
        )
    )

    assert ingest.used is True
    assert ingest.vector_ids
    assert ingest.num_chunks == 1

    retriever_manager = create_default_retriever_manager(
        vector_store=vectorstore,
        embedding_manager=embedding_manager,
        profile=profile,
        discover_entry_points=False,
    )
    service = RetrievalService(
        retriever_manager=retriever_manager,
        profile=profile,
    )
    result = service.retrieve(
        RetrievalRequest(
            query="external plugin content",
            final_top_k=1,
            prefetch_k=1,
            scope=scope,
            route_tier_override="standard",
        )
    )

    assert result.used is True
    assert result.chunks[0].text == "external:external plugin content"
    assert result.chunks[0].metadata["chunk_strategy"] == "external_chunker"
    assert result.chunks[0].scope == {
        "tenant_id": scope.tenant_id,
        "namespace": scope.namespace,
        "workspace_id": scope.workspace_id,
    }


def test_external_retriever_entry_point_uses_retrieval_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        {"intergrax.rag.retrievers": (_ExternalRetrieverPlugin,)},
    )
    profile = RagProfile(
        retriever_id="external_retriever",
        fast_retriever_id="external_retriever",
        deep_retriever_id="external_retriever",
        enable_rerank=False,
        route_mode="off",
        query_expansion="off",
    )
    vector_store = object()
    registry = RetrieverRegistry()
    manager = create_default_retriever_manager(
        vector_store=vector_store,
        embedding_manager=object(),
        registry=registry,
        profile=profile,
        discover_entry_points=True,
    )
    service = RetrievalService(retriever_manager=manager, profile=profile)

    result = service.retrieve(RetrievalRequest(query="plugin query"))

    assert result.used is True
    assert result.chunks[0].text == "external plugin content"
    assert result.trace.retriever_id == "external_retriever"
    assert _ExternalRetrieverPlugin.received_vector_store is vector_store
    assert registry.get("external_retriever").vector_store is vector_store


def test_external_reranker_entry_point_uses_retrieval_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        {"intergrax.rag.rerankers": (_ExternalRerankerPlugin,)},
    )
    profile = RagProfile(
        retriever_id="external_retriever",
        fast_retriever_id="external_retriever",
        deep_retriever_id="external_retriever",
        reranker_id="external_reranker",
        enable_rerank=True,
        route_mode="off",
        query_expansion="off",
        prefetch_top_k=1,
        final_top_k=1,
    )
    retriever_manager = create_default_retriever_manager(
        vector_store=object(),
        embedding_manager=object(),
        registry=RetrieverRegistry([_ExternalRetriever()]),
        profile=profile,
        discover_entry_points=False,
    )
    reranker_registry = RerankerRegistry()
    embedding_manager = object()
    reranker_engine = create_default_reranker_engine(
        embedding_manager=embedding_manager,
        registry=reranker_registry,
        discover_entry_points=True,
    )
    reranker_manager = ReRankerManager(engine=reranker_engine)
    service = RetrievalService(
        retriever_manager=retriever_manager,
        reranker_manager=reranker_manager,
        profile=profile,
    )

    result = service.retrieve(RetrievalRequest(query="plugin query"))

    assert result.used is True
    assert result.trace.reranker_id == "external_reranker"
    assert result.trace.rerank_enabled is True
    assert result.chunks[0].score == pytest.approx(0.99)
    assert _ExternalRerankerPlugin.received_embedding_manager is embedding_manager
    assert (
        reranker_registry.get("external_reranker").embedding_manager
        is embedding_manager
    )


def test_rag_entry_point_discovery_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        {"intergrax.rag.chunkers": (_ExternalChunker,)},
    )
    splitter = create_default_document_splitter(discover_entry_points=False)
    source = _source()

    chunks = splitter.split_documents([source], strategy_id="recursive")

    assert chunks
    assert all(chunk.metadata["chunk_strategy"] == "recursive" for chunk in chunks)
    with pytest.raises(RuntimeError, match="Chunking strategy not registered"):
        splitter.split_documents([source], strategy_id="external_chunker")


def test_rag_entry_point_discovery_uses_canonical_environment_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_DISCOVER_PLUGINS", "true")
    _patch_entry_points(
        monkeypatch,
        {"intergrax.rag.chunkers": (_ExternalChunker,)},
    )
    splitter = create_default_document_splitter(
        registry=ChunkingStrategyRegistry(),
    )

    chunks = splitter.split_documents(
        [_source()],
        strategy_id="external_chunker",
    )

    assert chunks[0].content == "external:external plugin content"


def test_rag_entry_point_id_collisions_do_not_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        {"intergrax.rag.chunkers": (_ConflictingChunker,)},
    )
    registry = ChunkingStrategyRegistry([RecursiveChunkingStrategy()])

    with pytest.raises(ValueError, match="Chunking strategy already registered: recursive"):
        create_default_document_splitter(
            registry=registry,
            discover_entry_points=True,
        )

    _patch_entry_points(
        monkeypatch,
        {"intergrax.rag.chunkers": (_DuplicateChunker, _DuplicateChunker)},
    )
    with pytest.raises(ValueError, match="Chunking strategy already registered: duplicate_chunker"):
        create_default_document_splitter(
            registry=ChunkingStrategyRegistry(),
            discover_entry_points=True,
        )
