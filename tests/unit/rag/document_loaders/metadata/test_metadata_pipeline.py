# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from intergrax.rag.document_loaders.metadata_pipeline import MetadataPipeline
from intergrax.rag.document_loaders.contracts.metadata_provider import BaseMetadataProvider


pytestmark = pytest.mark.unit


class _DummyProvider(BaseMetadataProvider):

    def __init__(self, tag: str):
        self.tag = tag

    def enrich(
        self,
        documents: Sequence[Document],
        source: Path | str,
    ) -> Sequence[Document]:

        out = []
        for d in documents:
            meta = dict(d.metadata)
            meta[self.tag] = True
            out.append(
                Document(
                    page_content=d.page_content,
                    metadata=meta,
                )
            )
        return out


class _FailingProvider(BaseMetadataProvider):

    def enrich(
        self,
        documents: Sequence[Document],
        source: Path | str,
    ) -> Sequence[Document]:
        raise RuntimeError("provider failure")


def test_pipeline_executes_providers_in_sequence():

    docs = [Document(page_content="a", metadata={})]

    p1 = _DummyProvider("p1")
    p2 = _DummyProvider("p2")

    pipeline = MetadataPipeline([p1, p2])

    result = pipeline.enrich(docs, "file.txt")

    assert result[0].metadata["p1"] is True
    assert result[0].metadata["p2"] is True


def test_pipeline_propagates_documents():

    docs = [Document(page_content="a", metadata={})]

    pipeline = MetadataPipeline([_DummyProvider("x")])

    result = pipeline.enrich(docs, "file.txt")

    assert len(result) == 1
    assert result[0].metadata["x"] is True


def test_pipeline_returns_input_when_no_providers():

    docs = [Document(page_content="a", metadata={})]

    pipeline = MetadataPipeline([])

    result = pipeline.enrich(docs, "file.txt")

    assert result == docs


def test_pipeline_passes_source_to_providers():

    captured = {}

    class _SourceCaptureProvider(BaseMetadataProvider):

        def enrich(self, documents, source):
            captured["source"] = source
            return documents

    pipeline = MetadataPipeline([_SourceCaptureProvider()])

    pipeline.enrich([Document(page_content="a")], "abc.txt")

    assert captured["source"] == "abc.txt"


def test_pipeline_propagates_provider_exception():

    pipeline = MetadataPipeline([_FailingProvider()])

    with pytest.raises(RuntimeError):
        pipeline.enrich([Document(page_content="a")], "file.txt")