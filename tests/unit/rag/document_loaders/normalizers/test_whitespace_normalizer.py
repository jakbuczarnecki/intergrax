# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
from pydantic import ValidationError

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_loaders.normalizers.whitespace_normalizer import (
    WhitespaceNormalizer,
)


pytestmark = pytest.mark.unit

_TENANT = "tenant.test"


def _sample_doc(content: str, **metadata) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": "docid1234567890ab",
                "root_document_id": "docid1234567890ab",
            },
            "scope": {"tenant_id": _TENANT},
            "content": content,
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


def test_whitespace_normalizer_returns_knowledge_document():

    docs = [_sample_doc("This   is     a   test")]

    result = WhitespaceNormalizer().normalize(docs, source="file.txt")

    assert len(result) == 1
    assert isinstance(result[0], KnowledgeDocument)


def test_whitespace_normalizer_collapses_spaces_tabs_and_newlines():

    docs = [_sample_doc("This   is\t\ta   test\n\n\n\nLine2")]

    result = WhitespaceNormalizer().normalize(docs, source="file.txt")

    assert result[0].content == "This is a test\n\nLine2"


def test_whitespace_normalizer_does_not_mutate_input():

    source_doc = _sample_doc("A   B")
    original_content = source_doc.content
    original_metadata = dict(source_doc.metadata)

    WhitespaceNormalizer().normalize([source_doc], source="file.txt")

    assert source_doc.content == original_content
    assert dict(source_doc.metadata) == original_metadata


def test_whitespace_normalizer_preserves_identity():

    docs = [_sample_doc("A   B")]

    result = WhitespaceNormalizer().normalize(docs, source="file.txt")

    assert result[0].identity == docs[0].identity


def test_whitespace_normalizer_preserves_scope():

    docs = [_sample_doc("A   B")]

    result = WhitespaceNormalizer().normalize(docs, source="file.txt")

    assert result[0].scope == docs[0].scope


def test_whitespace_normalizer_preserves_metadata():

    docs = [_sample_doc("A   B", custom="unit-test")]

    result = WhitespaceNormalizer().normalize(docs, source="file.txt")

    assert result[0].metadata["custom"] == "unit-test"
    assert result[0].metadata["source"] == "file.pdf"


def test_whitespace_normalizer_preserves_provenance():

    docs = [_sample_doc("A   B")]

    result = WhitespaceNormalizer().normalize(docs, source="file.txt")

    assert result[0].provenance == docs[0].provenance


def test_whitespace_normalizer_rejects_empty_output():

    base = _sample_doc("placeholder")
    doc = KnowledgeDocument.model_construct(
        schema_version=base.schema_version,
        identity=base.identity,
        scope=base.scope,
        content="   \t\n  ",
        metadata=dict(base.metadata),
        provenance=base.provenance,
    )

    with pytest.raises(ValidationError):
        WhitespaceNormalizer().normalize([doc], source="file.txt")
