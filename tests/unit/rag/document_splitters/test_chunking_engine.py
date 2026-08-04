# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Sequence

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import (
    build_derived_chunk,
    validate_derived_chunk,
)
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
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


def _bad_chunk(doc: KnowledgeDocument, strategy_id: str, **updates: object) -> KnowledgeDocument:
    chunk = build_derived_chunk(
        doc,
        content=doc.content,
        strategy_id=strategy_id,
        chunk_index=0,
    )
    payload = chunk.model_dump(mode="python")
    for key, value in updates.items():
        if key in {"metadata", "identity", "provenance"}:
            payload[key] = {**payload[key], **value} if isinstance(value, dict) else value
        else:
            payload[key] = value
    return KnowledgeDocument.model_validate(payload)


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


class _WrongChunkIdStrategy(BaseChunkingStrategy):
    @classmethod
    def strategy_id(cls) -> str:
        return "wrong_chunk_id"

    def chunk(self, documents: Sequence[KnowledgeDocument]) -> Sequence[KnowledgeDocument]:
        doc = documents[0]
        return [
            _bad_chunk(
                doc,
                self.strategy_id(),
                metadata={ChunkMetadataKey.CHUNK_ID.value: "not-the-document-id"},
            )
        ]


class _WrongChunkStrategyStrategy(BaseChunkingStrategy):
    @classmethod
    def strategy_id(cls) -> str:
        return "wrong_chunk_strategy"

    def chunk(self, documents: Sequence[KnowledgeDocument]) -> Sequence[KnowledgeDocument]:
        doc = documents[0]
        return [
            _bad_chunk(
                doc,
                self.strategy_id(),
                metadata={ChunkMetadataKey.CHUNK_STRATEGY.value: "other-strategy"},
            )
        ]


class _WrongChunkSizeStrategy(BaseChunkingStrategy):
    @classmethod
    def strategy_id(cls) -> str:
        return "wrong_chunk_size"

    def chunk(self, documents: Sequence[KnowledgeDocument]) -> Sequence[KnowledgeDocument]:
        doc = documents[0]
        return [
            _bad_chunk(
                doc,
                self.strategy_id(),
                metadata={ChunkMetadataKey.CHUNK_SIZE.value: 0},
            )
        ]


class _WrongContentHashStrategy(BaseChunkingStrategy):
    @classmethod
    def strategy_id(cls) -> str:
        return "wrong_content_hash"

    def chunk(self, documents: Sequence[KnowledgeDocument]) -> Sequence[KnowledgeDocument]:
        doc = documents[0]
        return [
            _bad_chunk(
                doc,
                self.strategy_id(),
                provenance={"content_hash": "0" * 64},
            )
        ]


class _NondeterministicIdStrategy(BaseChunkingStrategy):
    @classmethod
    def strategy_id(cls) -> str:
        return "nondeterministic_id"

    def chunk(self, documents: Sequence[KnowledgeDocument]) -> Sequence[KnowledgeDocument]:
        doc = documents[0]
        forged_id = "forged-chunk-id-000000000001"
        return [
            _bad_chunk(
                doc,
                self.strategy_id(),
                identity={"document_id": forged_id},
                metadata={ChunkMetadataKey.CHUNK_ID.value: forged_id},
            )
        ]


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


def test_chunking_engine_rejects_duplicate_source_document_ids() -> None:
    docs = [_source("doc-a", "alpha"), _source("doc-a", "beta")]
    with pytest.raises(ValueError, match="Duplicate source document_id: doc-a"):
        _engine(_PassthroughStrategy()).chunk(docs, strategy_id="passthrough")


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


def test_chunking_engine_rejects_fallback_document_id_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = [_source("doc-a", "alpha"), _source("doc-b", "beta")]
    raw_chunk = build_derived_chunk(
        docs[0],
        content="alpha",
        strategy_id="first_only",
        chunk_index=0,
    )
    duplicate_id = raw_chunk.identity.document_id
    original_build = build_derived_chunk

    def _colliding_build(source: KnowledgeDocument, **kwargs: object) -> KnowledgeDocument:
        if source.identity.document_id == "doc-b":
            chunk = original_build(source, **kwargs)
            payload = chunk.model_dump(mode="python")
            payload["identity"]["document_id"] = duplicate_id
            payload["metadata"][ChunkMetadataKey.CHUNK_ID.value] = duplicate_id
            return KnowledgeDocument.model_validate(payload)
        return original_build(source, **kwargs)

    def _passthrough_validate(
        source: KnowledgeDocument,
        chunk: KnowledgeDocument,
        *,
        strategy_id: str,
    ) -> KnowledgeDocument:
        if source.identity.document_id == "doc-b":
            return KnowledgeDocument.model_validate(chunk.model_dump(mode="python"))
        return validate_derived_chunk(source, chunk, strategy_id=strategy_id)

    monkeypatch.setattr(
        "intergrax.rag.document_splitters.engine.chunking_engine.build_derived_chunk",
        _colliding_build,
    )
    monkeypatch.setattr(
        "intergrax.rag.document_splitters.engine.chunking_engine.validate_derived_chunk",
        _passthrough_validate,
    )

    with pytest.raises(ValueError, match="Duplicate chunk document_id"):
        _engine(_FirstOnlyStrategy()).chunk(docs, strategy_id="first_only")


def test_chunking_engine_rejects_wrong_metadata_chunk_id() -> None:
    docs = [_source("doc-a", "alpha")]
    with pytest.raises(ValueError, match="chunk_id must match identity.document_id"):
        _engine(_WrongChunkIdStrategy()).chunk(docs, strategy_id="wrong_chunk_id")


def test_chunking_engine_rejects_wrong_metadata_chunk_strategy() -> None:
    docs = [_source("doc-a", "alpha")]
    with pytest.raises(ValueError, match="chunk_strategy must match the requested strategy_id"):
        _engine(_WrongChunkStrategyStrategy()).chunk(docs, strategy_id="wrong_chunk_strategy")


def test_chunking_engine_rejects_wrong_metadata_chunk_size() -> None:
    docs = [_source("doc-a", "alpha")]
    with pytest.raises(ValueError, match="chunk_size must match len\\(content\\)"):
        _engine(_WrongChunkSizeStrategy()).chunk(docs, strategy_id="wrong_chunk_size")


def test_chunking_engine_rejects_wrong_content_hash() -> None:
    docs = [_source("doc-a", "alpha")]
    with pytest.raises(ValueError, match="content_hash must match SHA-256 of content"):
        _engine(_WrongContentHashStrategy()).chunk(docs, strategy_id="wrong_content_hash")


def test_chunking_engine_rejects_nondeterministic_document_id() -> None:
    docs = [_source("doc-a", "alpha")]
    with pytest.raises(ValueError, match="deterministic derived id"):
        _engine(_NondeterministicIdStrategy()).chunk(docs, strategy_id="nondeterministic_id")


def test_chunking_engine_does_not_mask_strategy_exceptions() -> None:
    docs = [_source("doc-a", "alpha")]
    with pytest.raises(RuntimeError, match="strategy failed"):
        _engine(_ExplodingStrategy()).chunk(docs, strategy_id="explode")
