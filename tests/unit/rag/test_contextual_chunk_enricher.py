from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.contextual.chunk_enricher import ContextualChunkEnricher


pytestmark = pytest.mark.unit


def _document(*, content: str = "chunk body") -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": "chunk-1",
                "root_document_id": "root-1",
                "parent_document_id": "parent-1",
            },
            "scope": {"tenant_id": "tenant.test", "namespace": "workspace.test"},
            "content": content,
            "metadata": {"source": "fixture", "position": 1},
            "provenance": {
                "source_kind": "file",
                "source_id": "source-1",
                "provider_id": "fixture",
                "content_hash": "sha256:canonical",
            },
        }
    )


def test_enriches_native_chunk_without_mutating_lineage() -> None:
    llm = MagicMock()
    llm.generate_messages.return_value = SimpleNamespace(content="Situating context")
    document = _document(content="source body")
    chunk = _document()

    result = ContextualChunkEnricher(llm).enrich([document], [chunk])

    enriched = result[0]
    assert isinstance(enriched, KnowledgeDocument)
    assert enriched.content == "Situating context\n\nchunk body"
    assert enriched.metadata["contextual_enrich"] is True
    assert enriched.schema_version == chunk.schema_version
    assert enriched.identity == chunk.identity
    assert enriched.scope == chunk.scope
    assert enriched.provenance == chunk.provenance
    assert enriched.provenance.content_hash == "sha256:canonical"
    assert chunk.content == "chunk body"
    assert "contextual_enrich" not in chunk.metadata


def test_missing_llm_returns_original_chunks() -> None:
    chunk = _document()

    result = ContextualChunkEnricher().enrich([chunk], [chunk])

    assert result == [chunk]
    assert result[0] is chunk


def test_empty_llm_response_returns_original_chunk() -> None:
    llm = MagicMock()
    llm.generate_messages.return_value = SimpleNamespace(content="")
    chunk = _document()

    result = ContextualChunkEnricher(llm).enrich([chunk], [chunk])

    assert result == [chunk]
    assert result[0] is chunk


def test_llm_failure_is_fail_soft() -> None:
    llm = MagicMock()
    llm.generate_messages.side_effect = RuntimeError("offline")
    chunk = _document()

    result = ContextualChunkEnricher(llm).enrich([chunk], [chunk])

    assert result == [chunk]
    assert result[0] is chunk


def test_empty_chunk_does_not_call_llm() -> None:
    llm = MagicMock()
    empty_chunk = MagicMock(spec=KnowledgeDocument)
    empty_chunk.content = " "

    result = ContextualChunkEnricher(llm).enrich(
        [_document(content="source body")],
        [empty_chunk],
    )

    assert result == [empty_chunk]
    llm.generate_messages.assert_not_called()
