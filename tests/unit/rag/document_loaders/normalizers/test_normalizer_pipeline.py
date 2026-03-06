# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from pathlib import Path
from typing import Sequence

import pytest
from langchain_core.documents import Document

from intergrax.rag.document_loaders.normalizer_pipeline import NormalizerPipeline
from intergrax.rag.document_loaders.contracts.base_document_normalizer import BaseDocumentNormalizer


pytestmark = pytest.mark.unit


class DummyNormalizer(BaseDocumentNormalizer):

    def __init__(self, tag: str):
        self.tag = tag

    def normalize(
        self,
        documents: Sequence[Document],
        source: Path | str,
    ) -> Sequence[Document]:
        result = []
        for d in documents:
            md = dict(d.metadata or {})
            md[self.tag] = True

            result.append(
                Document(
                    page_content=d.page_content,
                    metadata=md,
                )
            )
        return result


def test_pipeline_executes_normalizers_in_order():

    docs = [
        Document(page_content="text", metadata={"a": 1})
    ]

    pipeline = NormalizerPipeline(
        normalizers=[
            DummyNormalizer("n1"),
            DummyNormalizer("n2"),
        ]
    )

    result = pipeline.normalize(docs, source="file.txt")

    assert len(result) == 1
    assert result[0].metadata["n1"] is True
    assert result[0].metadata["n2"] is True


def test_pipeline_preserves_document_count():

    docs = [
        Document(page_content="a", metadata={}),
        Document(page_content="b", metadata={}),
        Document(page_content="c", metadata={}),
    ]

    pipeline = NormalizerPipeline(
        normalizers=[DummyNormalizer("tag")]
    )

    result = pipeline.normalize(docs, source="file.txt")

    assert len(result) == 3


def test_pipeline_allows_empty_normalizer_list():

    docs = [
        Document(page_content="text", metadata={"x": 1})
    ]

    pipeline = NormalizerPipeline(normalizers=[])

    result = pipeline.normalize(docs, source="file.txt")

    assert result[0].page_content == "text"
    assert result[0].metadata["x"] == 1


def test_pipeline_preserves_metadata():

    docs = [
        Document(page_content="text", metadata={"source": "unit-test"})
    ]

    pipeline = NormalizerPipeline(
        normalizers=[DummyNormalizer("normalized")]
    )

    result = pipeline.normalize(docs, source="file.txt")

    assert result[0].metadata["source"] == "unit-test"
    assert result[0].metadata["normalized"] is True