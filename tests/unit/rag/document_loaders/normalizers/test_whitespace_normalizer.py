# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
from langchain_core.documents import Document

from intergrax.rag.document_loaders.normalizers.whitespace_normalizer import (
    WhitespaceNormalizer,
)


pytestmark = pytest.mark.unit


def test_whitespace_normalizer_collapses_multiple_spaces():

    docs = [
        Document(
            page_content="This   is     a   test",
            metadata={"id": 1},
        )
    ]

    normalizer = WhitespaceNormalizer()

    result = normalizer.normalize(docs, source="file.txt")

    assert len(result) == 1
    assert "   " not in result[0].page_content


def test_whitespace_normalizer_collapses_multiple_newlines():

    docs = [
        Document(
            page_content="Line1\n\n\n\nLine2",
            metadata={"id": 1},
        )
    ]

    normalizer = WhitespaceNormalizer()

    result = normalizer.normalize(docs, source="file.txt")

    assert len(result) == 1
    assert "\n\n\n" not in result[0].page_content


def test_whitespace_normalizer_preserves_metadata():

    docs = [
        Document(
            page_content="A   B",
            metadata={"source": "unit-test"},
        )
    ]

    normalizer = WhitespaceNormalizer()

    result = normalizer.normalize(docs, source="file.txt")

    assert result[0].metadata["source"] == "unit-test"


def test_whitespace_normalizer_preserves_document_count():

    docs = [
        Document(page_content="A   B", metadata={}),
        Document(page_content="C   D", metadata={}),
    ]

    normalizer = WhitespaceNormalizer()

    result = normalizer.normalize(docs, source="file.txt")

    assert len(result) == 2