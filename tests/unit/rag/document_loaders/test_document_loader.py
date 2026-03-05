# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

import pytest

from langchain_core.documents import Document

from intergrax.rag.document_loaders.documents_loader import DocumentsLoader
from intergrax.rag.document_loaders.contracts.base_document_handler import (
    BaseDocumentHandler,
)
from intergrax.rag.document_loaders.registry.document_handler_registry import (
    DocumentHandlerRegistry,
)


pytestmark = pytest.mark.unit


class _DummyHandler(BaseDocumentHandler):

    def __init__(self, docs: Sequence[Document]):
        self.docs = list(docs)
        self.load_called = False

    def supports(self, source: str) -> bool:
        return True

    def confidence(self, source: str) -> float:
        return 1.0

    def build_parsers(self):
        return []

    def load(self, source: str):
        self.load_called = True
        return list(self.docs)


class _DummyMetadataPipeline:

    def __init__(self):
        self.called = False

    def enrich(self, docs, source):
        self.called = True
        return docs


def test_loader_calls_handler_load():

    docs = [Document(page_content="A")]

    handler = _DummyHandler(docs)

    registry = DocumentHandlerRegistry()
    registry.register(handler)

    pipeline = _DummyMetadataPipeline()

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=pipeline,
    )

    result = loader.load_document("file.pdf")

    assert handler.load_called
    assert result == docs


def test_loader_runs_metadata_pipeline():

    docs = [Document(page_content="A")]

    handler = _DummyHandler(docs)

    registry = DocumentHandlerRegistry()
    registry.register(handler)

    pipeline = _DummyMetadataPipeline()

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=pipeline,
    )

    loader.load_document("file.pdf")

    assert pipeline.called


def test_loader_returns_empty_when_handler_returns_none():

    handler = _DummyHandler([])

    registry = DocumentHandlerRegistry()
    registry.register(handler)

    pipeline = _DummyMetadataPipeline()

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=pipeline,
    )

    result = loader.load_document("file.pdf")

    assert result == []


def test_loader_returns_empty_on_exception():

    class _FailingHandler(_DummyHandler):

        def load(self, source: str):
            raise RuntimeError("boom")

    handler = _FailingHandler([])

    registry = DocumentHandlerRegistry()
    registry.register(handler)

    pipeline = _DummyMetadataPipeline()

    loader = DocumentsLoader(
        registry=registry,
        metadata_pipeline=pipeline,
    )

    result = loader.load_document("file.pdf")

    assert result == []