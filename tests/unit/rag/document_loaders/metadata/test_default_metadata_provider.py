# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey
from intergrax.rag.document_loaders.metadata.default_metadata_provider import (
    DefaultMetadataProvider,
)


pytestmark = pytest.mark.unit

_TENANT = "tenant.test"


def _sample_doc(**metadata) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": "docid1234567890ab",
                "root_document_id": "docid1234567890ab",
            },
            "scope": {"tenant_id": _TENANT},
            "content": "content",
            "metadata": {
                "source": "file.pdf",
                "parser": "tests.dummy",
                "position": 0,
                **metadata,
            },
            "provenance": {
                "source_kind": "file",
                "source_id": "file.pdf",
                "provider_id": "tests.dummy",
            },
        }
    )


def test_default_metadata_provider_returns_knowledge_document(tmp_path: Path):

    source = tmp_path / "sample.pdf"
    source.write_text("pdf", encoding="utf-8")

    result = DefaultMetadataProvider().enrich([_sample_doc()], source)

    assert len(result) == 1
    assert isinstance(result[0], KnowledgeDocument)


def test_default_metadata_provider_does_not_mutate_input(tmp_path: Path):

    source = tmp_path / "sample.pdf"
    source.write_text("pdf", encoding="utf-8")

    source_doc = _sample_doc()
    original_metadata = dict(source_doc.metadata)

    DefaultMetadataProvider().enrich([source_doc], source)

    assert dict(source_doc.metadata) == original_metadata


def test_default_metadata_provider_preserves_content_identity_scope_provenance(
    tmp_path: Path,
):

    source = tmp_path / "sample.pdf"
    source.write_text("pdf", encoding="utf-8")

    source_doc = _sample_doc()

    result = DefaultMetadataProvider().enrich([source_doc], source)[0]

    assert result.content == source_doc.content
    assert result.identity == source_doc.identity
    assert result.scope == source_doc.scope
    assert result.provenance == source_doc.provenance


def test_default_metadata_provider_adds_source_fields(tmp_path: Path):

    source = tmp_path / "sample.pdf"
    source.write_text("pdf", encoding="utf-8")

    result = DefaultMetadataProvider().enrich([_sample_doc()], source)[0]

    assert result.metadata[DocumentMetadataKey.SOURCE_PATH] == str(source.resolve())
    assert result.metadata[DocumentMetadataKey.SOURCE_NAME] == "sample.pdf"
    assert result.metadata[DocumentMetadataKey.EXTENSION] == ".pdf"


def test_default_metadata_provider_preserves_existing_values_via_setdefault(
    tmp_path: Path,
):

    source = tmp_path / "sample.pdf"
    source.write_text("pdf", encoding="utf-8")

    source_doc = _sample_doc(
        **{
            DocumentMetadataKey.SOURCE_PATH: "existing/path.pdf",
            DocumentMetadataKey.SOURCE_NAME: "existing.pdf",
            DocumentMetadataKey.EXTENSION: ".txt",
            DocumentMetadataKey.PARENT_ID: "existing-parent",
        }
    )

    result = DefaultMetadataProvider().enrich([source_doc], source)[0]

    assert result.metadata[DocumentMetadataKey.SOURCE_PATH] == "existing/path.pdf"
    assert result.metadata[DocumentMetadataKey.SOURCE_NAME] == "existing.pdf"
    assert result.metadata[DocumentMetadataKey.EXTENSION] == ".txt"
    assert result.metadata[DocumentMetadataKey.PARENT_ID] == "existing-parent"


def test_default_metadata_provider_moves_page_to_page_index(tmp_path: Path):

    source = tmp_path / "sample.pdf"
    source.write_text("pdf", encoding="utf-8")

    source_doc = _sample_doc(page=3)

    result = DefaultMetadataProvider().enrich([source_doc], source)[0]

    assert "page" not in result.metadata
    assert result.metadata[DocumentMetadataKey.PAGE_INDEX] == 3


def test_default_metadata_provider_preserves_existing_page_index(tmp_path: Path):

    source = tmp_path / "sample.pdf"
    source.write_text("pdf", encoding="utf-8")

    source_doc = _sample_doc(page=3, page_index=7)

    result = DefaultMetadataProvider().enrich([source_doc], source)[0]

    assert result.metadata[DocumentMetadataKey.PAGE_INDEX] == 7
    assert result.metadata["page"] == 3


def test_default_metadata_provider_generates_stable_parent_id(tmp_path: Path):

    source = tmp_path / "sample.pdf"
    source.write_text("pdf", encoding="utf-8")

    provider = DefaultMetadataProvider()
    first = provider.enrich([_sample_doc()], source)[0]
    second = provider.enrich([_sample_doc()], source)[0]

    expected = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:16]

    assert first.metadata[DocumentMetadataKey.PARENT_ID] == expected
    assert second.metadata[DocumentMetadataKey.PARENT_ID] == expected


def test_default_metadata_provider_handles_missing_path_and_uri():

    provider = DefaultMetadataProvider()

    missing_path = provider.enrich([_sample_doc()], "/no/such/file.pdf")[0]
    uri_result = provider.enrich(
        [_sample_doc()],
        "https://example.com/docs/report.pdf",
    )[0]

    assert missing_path.metadata[DocumentMetadataKey.SOURCE_PATH] == "/no/such/file.pdf"
    assert missing_path.metadata[DocumentMetadataKey.SOURCE_NAME] == "file.pdf"
    assert missing_path.metadata[DocumentMetadataKey.EXTENSION] == ""

    assert uri_result.metadata[DocumentMetadataKey.SOURCE_PATH] == (
        "https://example.com/docs/report.pdf"
    )
    assert uri_result.metadata[DocumentMetadataKey.SOURCE_NAME] == "report.pdf"
