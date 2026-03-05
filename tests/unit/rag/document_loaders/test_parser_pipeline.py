# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from typing import Sequence

from langchain_core.documents import Document

from intergrax.rag.document_loaders.parsers.parser_pipeline import ParserPipeline
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser


pytestmark = pytest.mark.unit


class _DummyParser(BaseDocumentParser):

    _ID = "tests.dummy"

    def __init__(
        self,
        *,
        available: bool = True,
        docs: Sequence[Document] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._available = available
        self._docs = docs
        self._error = error

    @classmethod
    def parser_id(cls) -> str:
        return cls._ID

    def is_available(self) -> bool:
        return self._available

    def load(self, source: str) -> Sequence[Document]:
        if self._error is not None:
            raise self._error
        return self._docs or []


def test_pipeline_requires_at_least_one_parser():

    with pytest.raises(ValueError):
        ParserPipeline([])


def test_pipeline_skips_unavailable_parsers():

    parser1 = _DummyParser(available=False)
    parser2 = _DummyParser(docs=[Document(page_content="ok")])

    pipeline = ParserPipeline([parser1, parser2])

    docs = pipeline.parse("file")

    assert len(docs) == 1
    assert docs[0].page_content == "ok"


def test_pipeline_returns_first_successful_parser():

    parser1 = _DummyParser(docs=[Document(page_content="first")])
    parser2 = _DummyParser(docs=[Document(page_content="second")])

    pipeline = ParserPipeline([parser1, parser2])

    docs = pipeline.parse("file")

    assert docs[0].page_content == "first"


def test_pipeline_tries_next_parser_on_exception():

    parser1 = _DummyParser(error=RuntimeError("fail"))
    parser2 = _DummyParser(docs=[Document(page_content="ok")])

    pipeline = ParserPipeline([parser1, parser2])

    docs = pipeline.parse("file")

    assert docs[0].page_content == "ok"


def test_pipeline_raises_last_error_if_all_parsers_fail():

    parser1 = _DummyParser(error=ValueError("a"))
    parser2 = _DummyParser(error=RuntimeError("b"))

    pipeline = ParserPipeline([parser1, parser2])

    with pytest.raises(RuntimeError):
        pipeline.parse("file")


def test_pipeline_raises_runtime_error_when_no_parser_produces_docs():

    parser1 = _DummyParser(docs=None)
    parser2 = _DummyParser(docs=None)

    pipeline = ParserPipeline([parser1, parser2])

    with pytest.raises(RuntimeError):
        pipeline.parse("file")