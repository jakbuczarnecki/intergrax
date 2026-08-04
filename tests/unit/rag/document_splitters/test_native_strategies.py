# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    attach_parser_native_handle,
)
from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.document_splitters.strategies.docling_chunking_strategy import (
    DoclingChunkingStrategy,
)
from intergrax.rag.document_splitters.strategies.parent_child_chunking_strategy import (
    ParentChildChunkingStrategy,
)
from intergrax.rag.document_splitters.strategies.recursive_chunking_strategy import (
    RecursiveChunkingStrategy,
)
from intergrax.rag.document_splitters.strategies.semantic_chunking_strategy import (
    SemanticChunkingStrategy,
)


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


def _assert_common_chunk_properties(chunk: KnowledgeDocument, source: KnowledgeDocument) -> None:
    assert isinstance(chunk, KnowledgeDocument)
    assert chunk.identity.root_document_id == source.identity.root_document_id
    assert chunk.identity.parent_document_id == source.identity.document_id
    assert chunk.scope == source.scope
    assert chunk.provenance.source_kind == source.provenance.source_kind
    assert chunk.provenance.source_id == source.provenance.source_id
    assert DocumentMetadataKey.DOCLING_DOCUMENT_META.value not in chunk.metadata


def test_recursive_strategy_native_contract() -> None:
    source = _source("doc-rec", "abcdefghij" * 30)
    strategy = RecursiveChunkingStrategy(chunk_size=50, chunk_overlap=10)
    chunks = strategy.chunk([source])

    assert chunks
    indices = [chunk.metadata[ChunkMetadataKey.CHUNK_INDEX.value] for chunk in chunks]
    assert indices == list(range(len(chunks)))
    first_run = [chunk.identity.document_id for chunk in chunks]
    second_run = [chunk.identity.document_id for chunk in strategy.chunk([source])]
    assert first_run == second_run
    for chunk in chunks:
        _assert_common_chunk_properties(chunk, source)


def test_semantic_strategy_native_contract() -> None:
    source = _source("doc-sem", "One. Two. Three. Four.")
    embedding_manager = MagicMock()
    embedding_manager.embed_texts.return_value = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.0, 1.0],
        ]
    )
    strategy = SemanticChunkingStrategy(embedding_manager=embedding_manager, similarity_threshold=0.95)
    chunks = strategy.chunk([source])

    assert chunks
    for chunk in chunks:
        _assert_common_chunk_properties(chunk, source)


def test_parent_child_strategy_metadata_and_indexes() -> None:
    source_a = _source("doc-a", "word " * 200)
    source_b = _source("doc-b", "term " * 200)
    strategy = ParentChildChunkingStrategy(parent_size=120, child_size=40, child_overlap=5)
    chunks = strategy.chunk([source_a, source_b])

    parents_a = {
        chunk.metadata[ChunkMetadataKey.PARENT_CHUNK_ID.value]
        for chunk in chunks
        if chunk.identity.parent_document_id == "doc-a"
    }
    parents_b = {
        chunk.metadata[ChunkMetadataKey.PARENT_CHUNK_ID.value]
        for chunk in chunks
        if chunk.identity.parent_document_id == "doc-b"
    }
    assert parents_a.isdisjoint(parents_b)

    a_indices = [
        chunk.metadata[ChunkMetadataKey.CHUNK_INDEX.value]
        for chunk in chunks
        if chunk.identity.parent_document_id == "doc-a"
    ]
    assert a_indices == list(range(len(a_indices)))

    sources = {"doc-a": source_a, "doc-b": source_b}
    for chunk in chunks:
        parent_id = chunk.identity.parent_document_id
        assert parent_id is not None
        _assert_common_chunk_properties(chunk, sources[parent_id])
        assert chunk.metadata[ChunkMetadataKey.CHUNK_SIZE.value] == len(chunk.content)


def test_langchain_recursive_strategy_native_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    class _FakeSplitter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def split_text(self, text: str) -> list[str]:
            return [text[i : i + 10] for i in range(0, len(text), 10)]

    fake_module = types.ModuleType("langchain_text_splitters")
    fake_module.RecursiveCharacterTextSplitter = _FakeSplitter
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", fake_module)

    from intergrax.rag.document_splitters.strategies.langchain_recursive_chunking_strategy import (
        LangChainRecursiveChunkingStrategy,
    )

    source = _source("doc-lc", "abcdefghij" * 20)
    strategy = LangChainRecursiveChunkingStrategy(chunk_size=40, chunk_overlap=5)
    chunks = strategy.chunk([source])

    assert chunks
    for chunk in chunks:
        _assert_common_chunk_properties(chunk, source)


def test_docling_strategy_uses_private_handle_and_skips_empty_items(monkeypatch: pytest.MonkeyPatch) -> None:
    import docling_core.types.doc as doc_mod

    class FakeSectionHeader:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeTextItem:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakePictureItem:
        label = ""

    class _FakeDoc:
        def iterate_items(self):
            yield FakeSectionHeader("Intro"), 1
            yield FakeTextItem("Body text"), 1
            yield FakeSectionHeader("Next"), 1
            yield FakePictureItem(), 1

    monkeypatch.setattr(doc_mod, "SectionHeaderItem", FakeSectionHeader, raising=False)
    monkeypatch.setattr(doc_mod, "TextItem", FakeTextItem, raising=False)
    monkeypatch.setattr(doc_mod, "ListItem", FakeTextItem, raising=False)
    monkeypatch.setattr(doc_mod, "TableItem", FakeTextItem, raising=False)
    monkeypatch.setattr(doc_mod, "CodeItem", FakeTextItem, raising=False)
    monkeypatch.setattr(doc_mod, "FormulaItem", FakeTextItem, raising=False)
    monkeypatch.setattr(doc_mod, "PictureItem", FakePictureItem, raising=False)

    source = attach_parser_native_handle(_source("doc-dl", "placeholder"), _FakeDoc())
    strategy = DoclingChunkingStrategy()
    chunks = strategy.chunk([source])

    assert chunks
    assert all(DocumentMetadataKey.DOCLING_DOCUMENT_META.value not in chunk.metadata for chunk in chunks)
    assert chunks[0].metadata.get(ChunkMetadataKey.SECTION.value) == "Intro"
    assert "Body text" in chunks[0].content
    assert all(chunk.identity.parent_document_id == "doc-dl" for chunk in chunks)
