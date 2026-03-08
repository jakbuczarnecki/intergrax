# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from intergrax.rag.document_splitters.strategies.recursive_chunking_strategy import (
    RecursiveChunkingStrategy,
)
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import (
    ChunkMetadataKey,
)

pytestmark = pytest.mark.integration


FIXTURES_DIR = Path("tests/fixtures/documents")


def _load_text_documents() -> list[Document]:

    docs: list[Document] = []

    for file in FIXTURES_DIR.rglob("*.txt"):

        text = file.read_text(encoding="utf-8")

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file),
                },
            )
        )

    return docs


def test_recursive_chunking_produces_chunks():

    documents = _load_text_documents()

    strategy = RecursiveChunkingStrategy(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = strategy.chunk(documents)

    assert len(chunks) > 0


def test_chunk_metadata_contains_strategy():

    documents = _load_text_documents()

    strategy = RecursiveChunkingStrategy()

    chunks = strategy.chunk(documents)

    for chunk in chunks:

        assert ChunkMetadataKey.CHUNK_STRATEGY in chunk.metadata


def test_chunk_metadata_contains_index():

    documents = _load_text_documents()

    strategy = RecursiveChunkingStrategy()

    chunks = strategy.chunk(documents)

    for chunk in chunks:

        assert ChunkMetadataKey.CHUNK_INDEX in chunk.metadata


def test_chunk_order_is_preserved():

    documents = _load_text_documents()

    strategy = RecursiveChunkingStrategy()

    chunks = strategy.chunk(documents)

    chunks_by_source = {}

    for chunk in chunks:

        source = chunk.metadata["source"]

        chunks_by_source.setdefault(source, []).append(chunk)

    for source_chunks in chunks_by_source.values():

        indices = [
            chunk.metadata[ChunkMetadataKey.CHUNK_INDEX]
            for chunk in source_chunks
        ]

        assert indices == sorted(indices)