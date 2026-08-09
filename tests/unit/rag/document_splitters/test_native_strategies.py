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
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
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


def _reconstruct_chunks(
    chunks: list[KnowledgeDocument],
    chunk_overlap: int,
) -> str:
    if not chunks:
        return ""

    reconstructed = chunks[0].content
    for chunk in chunks[1:]:
        maximum_overlap = min(chunk_overlap, len(reconstructed), len(chunk.content))
        overlap = next(
            (
                length
                for length in range(maximum_overlap, 0, -1)
                if reconstructed[-length:] == chunk.content[:length]
            ),
            0,
        )
        reconstructed += chunk.content[overlap:]

    return reconstructed


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


def test_recursive_strategy_keeps_short_document_as_one_chunk() -> None:
    content = "A short document stays intact. " * 20
    source = _source("doc-short", content)

    chunks = RecursiveChunkingStrategy(chunk_size=700, chunk_overlap=200).chunk([source])

    assert [chunk.content for chunk in chunks] == [content]


def test_recursive_strategy_prefers_paragraph_boundaries() -> None:
    first = "First paragraph has useful context."
    second = "Second paragraph has more useful context."
    source = _source("doc-paragraphs", f"{first}\n\n{second}")

    chunks = RecursiveChunkingStrategy(chunk_size=45, chunk_overlap=0).chunk([source])

    assert [chunk.content for chunk in chunks] == [f"{first}\n\n", second]


def test_recursive_strategy_prefers_line_boundaries_after_paragraphs() -> None:
    source = _source(
        "doc-lines",
        "First line has words.\nSecond line has words.\nThird line has words.",
    )

    chunks = RecursiveChunkingStrategy(chunk_size=25, chunk_overlap=0).chunk([source])

    assert len(chunks) == 3
    assert all(chunk.content.endswith("\n") for chunk in chunks[:2])
    assert "".join(chunk.content for chunk in chunks) == source.content


def test_recursive_strategy_preserves_word_boundaries() -> None:
    source = _source("doc-words", "alpha beta gamma delta")

    chunks = RecursiveChunkingStrategy(chunk_size=12, chunk_overlap=0).chunk([source])

    assert [chunk.content for chunk in chunks] == ["alpha beta ", "gamma delta"]


def test_recursive_strategy_hard_fallback_bounds_unbroken_text() -> None:
    source = _source("doc-token", "x" * 25)

    chunks = RecursiveChunkingStrategy(chunk_size=7, chunk_overlap=0).chunk([source])

    assert "".join(chunk.content for chunk in chunks) == source.content
    assert all(0 < len(chunk.content) <= 7 for chunk in chunks)


def test_recursive_strategy_preserves_meaningful_content_around_whitespace_gap() -> None:
    prefix = "meaningful-prefix"
    suffix = "meaningful-suffix"
    source = _source("doc-whitespace-gap", prefix + (" " * 32) + suffix)
    strategy = RecursiveChunkingStrategy(chunk_size=16, chunk_overlap=0)

    chunks = list(strategy.chunk([source]))
    emitted_content = "".join(chunk.content for chunk in chunks)

    assert chunks
    assert prefix in emitted_content
    assert suffix in emitted_content
    assert emitted_content.index(prefix) < emitted_content.index(suffix)
    assert all(0 < len(chunk.content) <= 16 for chunk in chunks)
    assert all(chunk.content.strip() for chunk in chunks)
    with pytest.raises(ValueError, match="chunk content must be a non-empty string"):
        build_derived_chunk(
            source,
            content=" " * 3,
            strategy_id=RecursiveChunkingStrategy.strategy_id(),
            chunk_index=0,
        )


def test_recursive_strategy_overlap_preserves_bounded_context_and_progress() -> None:
    source = _source(
        "doc-overlap",
        "alpha beta gamma delta epsilon zeta eta theta",
    )
    strategy = RecursiveChunkingStrategy(chunk_size=20, chunk_overlap=5)

    chunks = list(strategy.chunk([source]))

    assert len(chunks) > 1
    assert all(0 < len(chunk.content) <= 20 for chunk in chunks)
    assert all(
        previous.content[-5:] == current.content[:5]
        for previous, current in zip(chunks, chunks[1:])
    )
    assert all(previous.content != current.content for previous, current in zip(chunks, chunks[1:]))


def test_recursive_strategy_zero_overlap_reconstructs_exact_source() -> None:
    source = _source("doc-zero-overlap", "one two three four five six")

    chunks = RecursiveChunkingStrategy(chunk_size=10, chunk_overlap=0).chunk([source])

    assert _reconstruct_chunks(list(chunks), 0) == source.content


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "chunk_size must be a positive integer"),
        (-1, 0, "chunk_size must be a positive integer"),
        (10, -1, "chunk_overlap must be a non-negative integer"),
        (10, 10, "chunk_overlap must be smaller than chunk_size"),
        (10, 11, "chunk_overlap must be smaller than chunk_size"),
    ],
)
def test_recursive_strategy_rejects_invalid_configuration(
    chunk_size: int,
    chunk_overlap: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RecursiveChunkingStrategy(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


@pytest.mark.parametrize("content", ["", "   \r\n\t  "])
def test_recursive_strategy_ignores_empty_and_whitespace_documents(content: str) -> None:
    source = MagicMock(spec=KnowledgeDocument)
    source.content = content

    chunks = RecursiveChunkingStrategy(chunk_size=20, chunk_overlap=5).chunk([source])

    assert chunks == []


def test_recursive_strategy_supports_unicode_and_crlf_losslessly() -> None:
    content = "Nagłówek\r\n\r\nTreść — café.\r\nDruga linia."
    source = _source("doc-unicode", content)

    chunks = RecursiveChunkingStrategy(chunk_size=20, chunk_overlap=0).chunk([source])

    assert "".join(chunk.content for chunk in chunks) == content
    assert all(len(chunk.content) <= 20 for chunk in chunks)
    assert chunks[0].content.endswith("\r\n\r\n")


def test_recursive_strategy_is_deterministic_in_content_and_identity() -> None:
    source = _source("doc-deterministic", "alpha beta gamma delta " * 5)

    first_run = RecursiveChunkingStrategy(chunk_size=24, chunk_overlap=6).chunk([source])
    second_run = RecursiveChunkingStrategy(chunk_size=24, chunk_overlap=6).chunk([source])

    assert [(chunk.content, chunk.identity.document_id) for chunk in first_run] == [
        (chunk.content, chunk.identity.document_id) for chunk in second_run
    ]


def test_recursive_strategy_preserves_metadata_and_provenance() -> None:
    source = _source("doc-provenance", "metadata and provenance remain attached")

    chunks = RecursiveChunkingStrategy(chunk_size=20, chunk_overlap=3).chunk([source])

    assert chunks
    for chunk in chunks:
        _assert_common_chunk_properties(chunk, source)
        assert chunk.metadata["source"] == "doc-provenance"
        assert chunk.metadata[ChunkMetadataKey.CHUNK_STRATEGY.value] == "recursive"
        assert chunk.metadata[ChunkMetadataKey.CHUNK_SIZE.value] == len(chunk.content)
        assert chunk.provenance.content_hash


def test_recursive_strategy_content_integrity_with_overlap() -> None:
    content = (
        "section-00 alpha\n\n"
        "section-01 beta\n\n"
        "section-02 gamma\n\n"
        "section-03 delta"
    )
    source = _source("doc-integrity", content)

    chunks = RecursiveChunkingStrategy(chunk_size=28, chunk_overlap=7).chunk([source])

    assert _reconstruct_chunks(list(chunks), 7) == content


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
