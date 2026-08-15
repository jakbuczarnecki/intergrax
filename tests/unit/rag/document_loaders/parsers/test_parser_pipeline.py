# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest
from typing import Sequence

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.knowledge.contracts import (
    KnowledgeDocument,
    KnowledgeDocumentScope,
)
from intergrax.knowledge.contracts.document import dump_knowledge_document
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    get_parser_native_handle,
)
from intergrax.rag.document_loaders.contracts.base_document_handler import (
    BaseDocumentHandler,
    _fragment_to_knowledge_document,
    _resolve_source_kind,
)
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey
from intergrax.rag.document_loaders.contracts.metadata_contract import build_loader_metadata
from intergrax.rag.document_loaders.handlers.text_smart_document_handler import (
    TextSmartDocumentHandler,
)
from intergrax.rag.document_loaders.pipeline.parser_pipeline import (
    TRACE_METADATA_KEY,
    ParserPipeline,
)


pytestmark = pytest.mark.unit

_SCOPE = KnowledgeDocumentScope(
    tenant_id="tenant.test",
    namespace="ns.test",
    workspace_id="workspace.test",
)


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


def _fragment(
    text: str = "ok",
    *,
    source: str = "file",
    position: int = 0,
    **metadata: object,
) -> ParsedDocumentFragment:
    base = build_loader_metadata(source=source, parser="tests.dummy", position=position)
    base.update(metadata)
    return ParsedDocumentFragment(text=text, metadata=base)


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


def test_handler_returns_knowledge_document():

    fragment = _fragment("hello", source="C:\\docs\\a.pdf", position=0)
    handler = _BridgeHandler([_DummyParser(fragments=[fragment])])

    docs = handler.load("C:\\docs\\a.pdf", scope=_SCOPE)

    assert len(docs) == 1
    assert isinstance(docs[0], KnowledgeDocument)
    assert docs[0].content == "hello"
    assert docs[0].scope.tenant_id == "tenant.test"
    assert docs[0].scope.namespace == "ns.test"
    assert docs[0].scope.workspace_id == "workspace.test"
    assert docs[0].provenance.source_kind == "file"
    assert docs[0].provenance.source_id == "C:\\docs\\a.pdf"
    assert docs[0].provenance.provider_id == "tests.dummy"
    assert DocumentMetadataKey.DOCUMENT_ID.value not in docs[0].metadata
    assert docs[0].metadata["parser"] == "tests.dummy"
    assert docs[0].identity.root_document_id == docs[0].identity.document_id
    assert docs[0].identity.parent_document_id is None
    expected_id = build_loader_metadata(
        source="C:\\docs\\a.pdf", parser="tests.dummy", position=0
    )["document_id"]
    assert docs[0].identity.document_id == expected_id


@pytest.mark.parametrize("workspace_id", ["workspace-a", None])
def test_text_handler_preserves_full_scope(
    tmp_path: Path,
    workspace_id: str | None,
) -> None:
    source = tmp_path / "scope.txt"
    source.write_text("scoped text", encoding="utf-8")

    documents = TextSmartDocumentHandler().load(
        str(source),
        scope=KnowledgeDocumentScope(
            tenant_id="tenant-a",
            namespace="namespace-a",
            workspace_id=workspace_id,
        ),
    )

    assert documents
    assert all(document.scope.tenant_id == "tenant-a" for document in documents)
    assert all(document.scope.namespace == "namespace-a" for document in documents)
    assert all(document.scope.workspace_id == workspace_id for document in documents)


def test_handler_multi_fragment_unique_ids():

    source = "/home/a.pdf"
    fragments = [
        _fragment("one", source=source, position=0),
        _fragment("two", source=source, position=1),
    ]
    handler = _BridgeHandler([_DummyParser(fragments=fragments)])

    docs = handler.load(source, scope=_SCOPE)

    assert len(docs) == 2
    assert docs[0].identity.document_id != docs[1].identity.document_id


def test_handler_reload_preserves_document_ids():

    source = "s3://bucket/a"
    fragment = _fragment("body", source=source, position=0)
    handler = _BridgeHandler([_DummyParser(fragments=[fragment])])

    first = handler.load(source, scope=_SCOPE)
    second = handler.load(source, scope=_SCOPE)

    assert first[0].identity.document_id == second[0].identity.document_id


def test_handler_uri_source_kind_mapping():

    assert _resolve_source_kind("https://example.com/a") == "https"
    assert _resolve_source_kind("s3://bucket/a") == "s3"
    assert _resolve_source_kind("/home/a.pdf") == "file"


def test_handler_rejects_whitespace_only_content():

    fragment = _fragment("   ", source="file", position=0)

    with pytest.raises(ValueError, match="content must be a non-empty string"):
        _fragment_to_knowledge_document(fragment, source="file", scope=_SCOPE)


def test_handler_rejects_reserved_parser_metadata():

    fragment = _fragment(
        "ok",
        source="file",
        position=0,
        **{"workspace_id": "leak"},
    )

    with pytest.raises(ValueError, match="reserved KnowledgeDocument key"):
        _fragment_to_knowledge_document(fragment, source="file", scope=_SCOPE)


@pytest.mark.parametrize("reserved_key", ["tenant_id", "namespace", "workspace_id"])
def test_handler_rejects_all_reserved_scope_metadata(reserved_key: str):

    fragment = _fragment(
        "ok",
        source="file",
        position=0,
        **{reserved_key: "leak"},
    )

    with pytest.raises(ValueError, match="reserved KnowledgeDocument key"):
        _fragment_to_knowledge_document(fragment, source="file", scope=_SCOPE)


def test_handler_preserves_none_workspace():

    fragment = _fragment("ok", source="file", position=0)
    scope = KnowledgeDocumentScope(tenant_id="tenant.test", namespace="ns.test")

    document = _fragment_to_knowledge_document(fragment, source="file", scope=scope)

    assert document.scope.tenant_id == "tenant.test"
    assert document.scope.namespace == "ns.test"
    assert document.scope.workspace_id is None


def test_handler_attaches_native_handle_as_private_runtime_state():

    handle = {"doc": 1}
    fragment = ParsedDocumentFragment(
        text="hello",
        metadata=build_loader_metadata(source="file", parser="tests.dummy", position=0),
        native_handle=handle,
    )
    handler = _BridgeHandler([_DummyParser(fragments=[fragment])])

    docs = handler.load("file", scope=_SCOPE)

    assert isinstance(docs[0], KnowledgeDocument)
    assert DocumentMetadataKey.DOCLING_DOCUMENT_META not in docs[0].metadata
    assert DocumentMetadataKey.DOCLING_DOCUMENT_META.value not in docs[0].model_dump()
    dumped = dump_knowledge_document(docs[0]).decode("utf-8")
    assert '"_docling_document"' not in dumped

    assert get_parser_native_handle(docs[0]) is handle
    assert DocumentMetadataKey.DOCLING_DOCUMENT_META.value not in docs[0].metadata


def test_handler_preserves_trace_metadata():

    traced = ParsedDocumentFragment(
        text="ok",
        metadata={
            **build_loader_metadata(source="file", parser="tests.dummy", position=0),
            TRACE_METADATA_KEY: {"parser_id": "tests.dummy"},
            "integration_parser_id": "tests.dummy",
        },
    )
    handler = _BridgeHandler([_DummyParser(fragments=[traced])])

    docs = handler.load("file", scope=_SCOPE)

    assert docs[0].metadata[TRACE_METADATA_KEY]["parser_id"] == "tests.dummy"
