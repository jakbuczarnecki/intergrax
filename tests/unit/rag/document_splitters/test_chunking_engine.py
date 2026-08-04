# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Sequence

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy
from intergrax.rag.document_splitters.engine.chunking_engine import ChunkingEngine
from intergrax.rag.document_splitters.registry.strategy_registry import ChunkingStrategyRegistry


pytestmark = pytest.mark.unit


def _source(document_id: str, content: str) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {"tenant_id": "tenant.test"},
            "content": content,
            "metadata": {"source": document_id},
            "provenance": {
                "source_kind": "file",
                "source_id": document_id,
            },
        }
    )


class _PassthroughStrategy(BaseChunkingStrategy):
    @classmethod
    def strategy_id(cls) -> str:
        return "passthrough"

    def chunk(self, documents: Sequence[KnowledgeDocument]) -> Sequence[KnowledgeDocument]:
        return [
            build_derived_chunk(doc, content=doc.content, strategy_id=self.strategy_id(), chunk_index=0)
            for doc in documents
        ]


class _FirstOnlyStrategy(BaseChunkingStrategy):
    @classmethod
    def strategy_id(cls) -> str:
        return "first_only"

    def chunk(self, documents: Sequence[KnowledgeDocument]) -> Sequence[KnowledgeDocument]:
        if not documents:
            return []
        doc = documents[0]
        return [
            build_derived_chunk(doc, content=doc.content, strategy_id=self.strategy_id(), chunk_index=0)
        ]


class _BadTypeStrategy(BaseChunkingStrategy):
    @classmethod
    def strategy_id(cls) -> str:
        return "bad_type"

    def chunk(self, documents: Sequence[KnowledgeDocument]) -> Sequence[object]:
        return ["not-a-document"]


class _BadParentStrategy(BaseChunkingStrategy):
    @classmethod
    def strategy_id(cls) -> str:
        return "bad_parent"

    def chunk(self, documents: Sequence[KnowledgeDocument]) -> Sequence[KnowledgeDocument]:
        doc = documents[0]
        chunk = build_derived_chunk(doc, content=doc.content, strategy_id=self.strategy_id(), chunk_index=0)
        return [
            KnowledgeDocument.model_validate(
                {
                    **chunk.model_dump(mode="python"),
                    "identity": {
                        "document_id": chunk.identity.document_id,
                        "root_document_id": chunk.identity.root_document_id,
                        "parent_document_id": "missing-parent",
                    },
                }
            )
        ]


class _ExplodingStrategy(BaseChunkingStrategy):
    @classmethod
    def strategy_id(cls) -> str:
        return "explode"

    def chunk(self, documents: Sequence[KnowledgeDocument]) -> Sequence[KnowledgeDocument]:
        raise RuntimeError("strategy failed")


def _engine(strategy: BaseChunkingStrategy) -> ChunkingEngine:
    registry = ChunkingStrategyRegistry([strategy])
    return ChunkingEngine(registry=registry)


def test_chunking_engine_preserves_source_order_and_unique_ids() -> None:
    docs = [_source("doc-a", "alpha"), _source("doc-b", "beta")]
    chunks = _engine(_PassthroughStrategy()).chunk(docs, strategy_id="passthrough")

    assert len(chunks) == 2
    assert chunks[0].identity.parent_document_id == "doc-a"
    assert chunks[1].identity.parent_document_id == "doc-b"
    assert len({chunk.identity.document_id for chunk in chunks}) == 2


def test_chunking_engine_rejects_non_knowledge_document() -> None:
    docs = [_source("doc-a", "alpha")]
    with pytest.raises(TypeError, match="non-KnowledgeDocument"):
        _engine(_BadTypeStrategy()).chunk(docs, strategy_id="bad_type")


def test_chunking_engine_rejects_unknown_parent_id() -> None:
    docs = [_source("doc-a", "alpha")]
    with pytest.raises(ValueError, match="parent_document_id"):
        _engine(_BadParentStrategy()).chunk(docs, strategy_id="bad_parent")


def test_chunking_engine_per_source_fallback_is_derivative() -> None:
    docs = [_source("doc-a", "alpha"), _source("doc-b", "beta")]
    chunks = _engine(_FirstOnlyStrategy()).chunk(docs, strategy_id="first_only")

    assert len(chunks) == 2
    assert chunks[0].identity.parent_document_id == "doc-a"
    fallback = chunks[1]
    assert fallback.identity.parent_document_id == "doc-b"
    assert fallback.identity.document_id != "doc-b"
    assert fallback.metadata.get("chunk_fallback") is True


def test_chunking_engine_does_not_mask_strategy_exceptions() -> None:
    docs = [_source("doc-a", "alpha")]
    with pytest.raises(RuntimeError, match="strategy failed"):
        _engine(_ExplodingStrategy()).chunk(docs, strategy_id="explode")
