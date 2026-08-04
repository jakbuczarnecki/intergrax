# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    attach_parser_native_handle,
)
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey


pytestmark = pytest.mark.unit


def _source_document(**overrides: object) -> KnowledgeDocument:
    payload: dict[str, object] = {
        "schema_version": 1,
        "identity": {
            "document_id": "source-doc-1234567890ab",
            "root_document_id": "source-doc-1234567890ab",
        },
        "scope": {"tenant_id": "tenant.test", "namespace": "workspace-1"},
        "content": "Source document content for chunking.",
        "metadata": {"source": "/tmp/example.txt", "parser": "tests.dummy"},
        "provenance": {
            "source_kind": "file",
            "source_id": "/tmp/example.txt",
            "provider_id": "tests.dummy",
            "content_hash": "abc123sourcehash0000000000000000000000000000000000000000",
        },
    }
    payload.update(overrides)
    return KnowledgeDocument.model_validate(payload)


def test_build_derived_chunk_identity_and_metadata() -> None:
    source = _source_document()
    chunk = build_derived_chunk(
        source,
        content="chunk one",
        strategy_id="recursive",
        chunk_index=0,
    )

    assert chunk.identity.root_document_id == "source-doc-1234567890ab"
    assert chunk.identity.parent_document_id == "source-doc-1234567890ab"
    assert chunk.identity.document_id != source.identity.document_id
    assert chunk.metadata[ChunkMetadataKey.CHUNK_ID.value] == chunk.identity.document_id
    assert chunk.metadata[ChunkMetadataKey.CHUNK_INDEX.value] == 0
    assert chunk.metadata[ChunkMetadataKey.CHUNK_STRATEGY.value] == "recursive"
    assert chunk.metadata[ChunkMetadataKey.CHUNK_SIZE.value] == len("chunk one")
    assert chunk.scope == source.scope
    assert chunk.provenance.source_kind == source.provenance.source_kind
    assert chunk.provenance.source_id == source.provenance.source_id
    assert chunk.provenance.provider_id == source.provenance.provider_id
    assert chunk.provenance.content_hash != source.provenance.content_hash


def test_build_derived_chunk_deterministic_id() -> None:
    source = _source_document()
    first = build_derived_chunk(
        source,
        content="same",
        strategy_id="recursive",
        chunk_index=0,
    )
    second = build_derived_chunk(
        source,
        content="same",
        strategy_id="recursive",
        chunk_index=0,
    )
    assert first.identity.document_id == second.identity.document_id


def test_build_derived_chunk_id_varies_by_strategy_index_and_content() -> None:
    source = _source_document()
    base = build_derived_chunk(source, content="alpha", strategy_id="recursive", chunk_index=0)
    other_strategy = build_derived_chunk(source, content="alpha", strategy_id="semantic", chunk_index=0)
    other_index = build_derived_chunk(source, content="alpha", strategy_id="recursive", chunk_index=1)
    other_content = build_derived_chunk(source, content="beta", strategy_id="recursive", chunk_index=0)

    ids = {
        base.identity.document_id,
        other_strategy.identity.document_id,
        other_index.identity.document_id,
        other_content.identity.document_id,
    }
    assert len(ids) == 4


def test_build_derived_chunk_does_not_mutate_source_or_copy_runtime_handle() -> None:
    handle = object()
    source = attach_parser_native_handle(_source_document(), handle)
    before = source.model_dump(mode="python")

    build_derived_chunk(source, content="chunk", strategy_id="recursive", chunk_index=0)

    assert source.model_dump(mode="python") == before


def test_build_derived_chunk_rejects_empty_content() -> None:
    source = _source_document()
    with pytest.raises(ValueError, match="non-empty"):
        build_derived_chunk(source, content="   ", strategy_id="recursive", chunk_index=0)
