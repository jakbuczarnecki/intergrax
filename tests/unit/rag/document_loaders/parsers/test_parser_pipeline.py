# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from typing import Sequence

from langchain_core.documents import Document

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey
from intergrax.rag.document_loaders.pipeline.parser_pipeline import (
    TRACE_METADATA_KEY,
    ParserPipeline,
)


pytestmark = pytest.mark.unit


class _DummyParser(BaseDocumentParser):

    _ID = "tests.dummy"

    def __init__(
        self,
        *,
        available: bool = True,
        fragments: Sequence[ParsedDocumentFragment] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._available = available
        self._fragments = fragments
        self._error = error

    @classmethod
    def parser_id(cls) -> str:
        return cls._ID

    def is_available(self) -> bool:
        return self._available

    def load(self, source: str) -> Sequence[ParsedDocumentFragment]:
        if self._error is not None:
            raise self._error
        return self._fragments or []


def _fragment(text: str = "ok", **metadata: object) -> ParsedDocumentFragment:
    return ParsedDocumentFragment(text=text, metadata=dict(metadata))


def test_pipeline_requires_at_least_one_parser():

    with pytest.raises(ValueError):
        ParserPipeline([])


def test_pipeline_skips_unavailable_parsers():

    parser1 = _DummyParser(available=False)
    parser2 = _DummyParser(fragments=[_fragment("ok")])

    pipeline = ParserPipeline([parser1, parser2])

    fragments = pipeline.parse("file")

    assert len(fragments) == 1
    assert fragments[0].text == "ok"


def test_pipeline_returns_first_successful_parser():

    parser1 = _DummyParser(fragments=[_fragment("first")])
    parser2 = _DummyParser(fragments=[_fragment("second")])

    pipeline = ParserPipeline([parser1, parser2])

    fragments = pipeline.parse("file")

    assert fragments[0].text == "first"


def test_pipeline_tries_next_parser_on_exception():

    parser1 = _DummyParser(error=RuntimeError("fail"))
    parser2 = _DummyParser(fragments=[_fragment("ok")])

    pipeline = ParserPipeline([parser1, parser2])

    fragments = pipeline.parse("file")

    assert fragments[0].text == "ok"


def test_pipeline_raises_last_error_if_all_parsers_fail():

    parser1 = _DummyParser(error=ValueError("a"))
    parser2 = _DummyParser(error=RuntimeError("b"))

    pipeline = ParserPipeline([parser1, parser2])

    with pytest.raises(RuntimeError):
        pipeline.parse("file")


def test_pipeline_raises_runtime_error_when_no_parser_produces_docs():

    parser1 = _DummyParser(fragments=None)
    parser2 = _DummyParser(fragments=None)

    pipeline = ParserPipeline([parser1, parser2])

    with pytest.raises(RuntimeError):
        pipeline.parse("file")


def test_pipeline_attaches_trace_without_mutating_input_fragment():

    input_fragment = _fragment("body", source="file")
    original_metadata = dict(input_fragment.metadata)
    parser = _DummyParser(fragments=[input_fragment])

    fragments = ParserPipeline([parser]).parse("file")

    assert TRACE_METADATA_KEY in fragments[0].metadata
    assert fragments[0].metadata["integration_parser_id"] == _DummyParser.parser_id()
    assert input_fragment.metadata == original_metadata
    assert fragments[0].text == "body"


def test_pipeline_preserves_native_handle_in_trace_enrichment():

    handle = object()
    input_fragment = ParsedDocumentFragment(text="x", metadata={}, native_handle=handle)
    parser = _DummyParser(fragments=[input_fragment])

    fragments = ParserPipeline([parser]).parse("file")

    assert fragments[0].native_handle is handle


class _BridgeHandler(BaseDocumentHandler):
    def __init__(self, parsers: list[BaseDocumentParser]) -> None:
        self._parsers = parsers

    def supports(self, source: str) -> bool:
        return True

    def confidence(self, source: str) -> float:
        return 1.0

    def build_parsers(self) -> list[BaseDocumentParser]:
        return self._parsers


def test_handler_legacy_bridge_returns_langchain_document():

    handle = {"doc": 1}
    fragment = ParsedDocumentFragment(
        text="hello",
        metadata={"source": "file", "parser": "tests.dummy"},
        native_handle=handle,
    )
    handler = _BridgeHandler([_DummyParser(fragments=[fragment])])

    docs = handler.load("file")

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "hello"
    assert docs[0].metadata["source"] == "file"
    assert docs[0].metadata[DocumentMetadataKey.DOCLING_DOCUMENT_META] is handle
    assert fragment.metadata == {"source": "file", "parser": "tests.dummy"}


def test_handler_legacy_bridge_preserves_trace_metadata():

    traced = ParsedDocumentFragment(
        text="ok",
        metadata={
            TRACE_METADATA_KEY: {"parser_id": "tests.dummy"},
            "integration_parser_id": "tests.dummy",
        },
    )
    handler = _BridgeHandler([_DummyParser(fragments=[traced])])

    docs = handler.load("file")

    assert docs[0].metadata[TRACE_METADATA_KEY]["parser_id"] == "tests.dummy"
