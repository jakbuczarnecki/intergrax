# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.document_splitters.strategies.recursive_chunking_strategy import (
    RecursiveChunkingStrategy,
)

pytestmark = pytest.mark.integration


FIXTURES_DIR = Path("tests/fixtures/documents")


def _load_knowledge_documents() -> list[KnowledgeDocument]:
    docs: list[KnowledgeDocument] = []

    for index, file in enumerate(sorted(FIXTURES_DIR.rglob("*.txt"))):
        text = file.read_text(encoding="utf-8")
        document_id = f"fixture-{index:04d}-{file.stem}"

        docs.append(
            KnowledgeDocument.model_validate(
                {
                    "schema_version": 1,
                    "identity": {
                        "document_id": document_id,
                        "root_document_id": document_id,
                    },
                    "scope": {"tenant_id": "fixture-tenant"},
                    "content": text,
                    "metadata": {"source": str(file)},
                    "provenance": {
                        "source_kind": "file",
                        "source_id": str(file),
                    },
                }
            )
        )

    return docs


def test_recursive_chunking_produces_chunks() -> None:
    documents = _load_knowledge_documents()

    strategy = RecursiveChunkingStrategy(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = strategy.chunk(documents)

    assert len(chunks) > 0


def test_chunk_metadata_contains_strategy() -> None:
    documents = _load_knowledge_documents()

    strategy = RecursiveChunkingStrategy()

    chunks = strategy.chunk(documents)

    for chunk in chunks:
        assert ChunkMetadataKey.CHUNK_STRATEGY in chunk.metadata


def test_chunk_metadata_contains_index() -> None:
    documents = _load_knowledge_documents()

    strategy = RecursiveChunkingStrategy()

    chunks = strategy.chunk(documents)

    for chunk in chunks:
        assert ChunkMetadataKey.CHUNK_INDEX in chunk.metadata


def test_chunk_order_is_preserved() -> None:
    documents = _load_knowledge_documents()

    strategy = RecursiveChunkingStrategy()

    chunks = strategy.chunk(documents)

    chunks_by_source: dict[str, list[KnowledgeDocument]] = {}

    for chunk in chunks:
        source = str(chunk.metadata["source"])

        chunks_by_source.setdefault(source, []).append(chunk)

    for source_chunks in chunks_by_source.values():
        indices = [
            chunk.metadata[ChunkMetadataKey.CHUNK_INDEX]
            for chunk in source_chunks
        ]

        assert indices == sorted(indices)
