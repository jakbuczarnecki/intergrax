# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite import google_workspace
from intergrax.integrations.providers.collaboration_suite.google_workspace.config import (
    GoogleWorkspaceCollaborationSuiteIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceSourceKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read import (
    GOOGLE_DOCS_NATIVE_MIME_TYPE,
    GOOGLE_DOCS_SOURCE_KIND,
    GoogleDocsBlockKind,
    GoogleDocsInlineKind,
    GoogleDocsKnowledgeReader,
    GoogleDocsNamedStyleType,
    GoogleDocsSegmentKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.docs import (
    GoogleDocsBullet,
    GoogleDocsDocument,
    GoogleDocsParagraph,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspaceErrorKind,
)

_UNEXPECTED_MESSAGE = "unexpected Google Docs provider response"
_INVALID_ID_MESSAGE = "invalid Google Docs document identifier"
_REQUEST_FAILED_MESSAGE = "Google Docs provider request failed"

_DOCUMENT_ID = "doc-main-1"
_DOCUMENT_TITLE = "Structured Doc"


@dataclass
class _RecordingTransport:
    responses: list[dict[str, object]] = field(default_factory=list)
    calls: list[dict[str, object]] = field(default_factory=list)
    exception: Exception | None = None

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "source_kind": source_kind,
                "relative_path": relative_path,
                "params": dict(params or {}),
                "headers": dict(headers or {}),
            }
        )
        if self.exception is not None:
            raise self.exception
        if not self.responses:
            return {}
        return self.responses.pop(0)


def _text_run(content: str, start: int, end: int) -> dict[str, object]:
    return {
        "startIndex": start,
        "endIndex": end,
        "textRun": {"content": content},
    }


def _paragraph_block(
    start: int,
    end: int,
    elements: list[dict[str, object]],
    **paragraph_fields: object,
) -> dict[str, object]:
    paragraph: dict[str, object] = {"elements": elements}
    paragraph.update(paragraph_fields)
    return {"startIndex": start, "endIndex": end, "paragraph": paragraph}


def _document_tab(
    body_content: list[dict[str, object]],
    **extra: object,
) -> dict[str, object]:
    tab: dict[str, object] = {"body": {"content": body_content}}
    tab.update(extra)
    return tab


def _tab(
    tab_id: str,
    title: str,
    index: int,
    nesting_level: int,
    document_tab: dict[str, object],
    *,
    parent_tab_id: str | None = None,
    child_tabs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    props: dict[str, object] = {
        "tabId": tab_id,
        "title": title,
        "index": index,
        "nestingLevel": nesting_level,
    }
    if parent_tab_id is not None:
        props["parentTabId"] = parent_tab_id
    payload: dict[str, object] = {
        "tabProperties": props,
        "documentTab": document_tab,
    }
    if child_tabs is not None:
        payload["childTabs"] = child_tabs
    return payload


def _document_payload(
    *,
    document_id: str = _DOCUMENT_ID,
    title: str = _DOCUMENT_TITLE,
    tabs: list[dict[str, object]],
    revision_id: str | None = "rev-42",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "documentId": document_id,
        "title": title,
        "suggestionsViewMode": "PREVIEW_WITHOUT_SUGGESTIONS",
        "tabs": tabs,
    }
    if revision_id is not None:
        payload["revisionId"] = revision_id
    return payload


def _assert_safe_dependency_error(exc_info: pytest.ExceptionInfo[IntegrationDependencyError]) -> None:
    assert exc_info.value.__cause__ is None
    message = str(exc_info.value)
    assert message in {_UNEXPECTED_MESSAGE, _REQUEST_FAILED_MESSAGE}


def _reader_with_payload(payload: dict[str, object]) -> tuple[GoogleDocsKnowledgeReader, _RecordingTransport]:
    transport = _RecordingTransport(responses=[payload])
    return GoogleDocsKnowledgeReader(transport=transport), transport


def test_constants() -> None:
    assert GOOGLE_DOCS_SOURCE_KIND == "docs"
    assert GOOGLE_DOCS_NATIVE_MIME_TYPE == "application/vnd.google-apps.document"


def test_single_tab_document_success() -> None:
    body = [
        _paragraph_block(
            1,
            12,
            [_text_run("Hello", 1, 6)],
            paragraphStyle={"namedStyleType": "NORMAL_TEXT"},
        ),
        _paragraph_block(
            12,
            24,
            [_text_run("Heading", 12, 19)],
            paragraphStyle={"namedStyleType": "HEADING_1", "headingId": "heading-1"},
        ),
        _paragraph_block(
            24,
            36,
            [_text_run("Item", 24, 28)],
            bullet={"listId": "list-1", "nestingLevel": 0},
        ),
    ]
    document_tab = _document_tab(
        body,
        lists={"list-1": {"listProperties": {"nestingLevels": []}}},
    )
    tab = _tab("tab-1", "Main", 0, 0, document_tab)
    payload = _document_payload(tabs=[tab])
    reader, _ = _reader_with_payload(payload)
    document = reader.read_document(document_id=_DOCUMENT_ID)

    assert document.document_id == _DOCUMENT_ID
    assert document.title == _DOCUMENT_TITLE
    assert document.revision_id == "rev-42"
    assert document.suggestions_view_mode == "PREVIEW_WITHOUT_SUGGESTIONS"
    assert len(document.tabs) == 1
    tab_model = document.tabs[0]
    body_segment = tab_model.segments[0]
    assert body_segment.kind is GoogleDocsSegmentKind.BODY
    assert body_segment.segment_id is None
    assert body_segment.blocks[0].paragraph.elements[0].text == "Hello"
    assert body_segment.blocks[1].paragraph.named_style_type is GoogleDocsNamedStyleType.HEADING_1
    assert body_segment.blocks[1].paragraph.heading_id == "heading-1"
    assert body_segment.blocks[2].paragraph.bullet == GoogleDocsBullet(list_id="list-1", nesting_level=0)


def test_nested_tabs_preorder() -> None:
    child_tab = _tab(
        "tab-child",
        "Child",
        0,
        1,
        _document_tab([_paragraph_block(1, 5, [_text_run("C", 1, 2)])]),
        parent_tab_id="tab-root",
    )
    grandchild_tab = _tab(
        "tab-grandchild",
        "Grandchild",
        0,
        2,
        _document_tab([_paragraph_block(1, 5, [_text_run("G", 1, 2)])]),
        parent_tab_id="tab-child",
    )
    child_tab["childTabs"] = [grandchild_tab]
    root_tab = _tab(
        "tab-root",
        "Root",
        0,
        0,
        _document_tab([_paragraph_block(1, 5, [_text_run("R", 1, 2)])]),
    )
    root_tab["childTabs"] = [child_tab]
    second_root = _tab(
        "tab-root-2",
        "Root Two",
        1,
        0,
        _document_tab([_paragraph_block(1, 5, [_text_run("S", 1, 2)])]),
    )
    payload = _document_payload(tabs=[root_tab, second_root], revision_id=None)
    reader, _ = _reader_with_payload(payload)
    document = reader.read_document(document_id=_DOCUMENT_ID)

    tab_ids = [tab.tab_id for tab in document.tabs]
    assert tab_ids == ["tab-root", "tab-child", "tab-grandchild", "tab-root-2"]
    assert document.tabs[0].nesting_level == 0
    assert document.tabs[1].parent_tab_id == "tab-root"
    assert document.tabs[1].nesting_level == 1
    assert document.tabs[2].parent_tab_id == "tab-child"
    assert document.tabs[2].nesting_level == 2
    assert document.tabs[3].index == 1
    assert document.revision_id is None


def test_segments_sorted_by_map_key() -> None:
    body = [_paragraph_block(1, 5, [_text_run("B", 1, 2)])]
    headers = {
        "header-b": {"headerId": "header-b", "content": [_paragraph_block(1, 5, [_text_run("H2", 1, 3)])]},
        "header-a": {"headerId": "header-a", "content": [_paragraph_block(1, 5, [_text_run("H1", 1, 3)])]},
    }
    footers = {
        "footer-b": {"footerId": "footer-b", "content": [_paragraph_block(1, 5, [_text_run("F2", 1, 3)])]},
        "footer-a": {"footerId": "footer-a", "content": [_paragraph_block(1, 5, [_text_run("F1", 1, 3)])]},
    }
    footnotes = {
        "fn-b": {"footnoteId": "fn-b", "content": [_paragraph_block(1, 5, [_text_run("N2", 1, 3)])]},
        "fn-a": {"footnoteId": "fn-a", "content": [_paragraph_block(1, 5, [_text_run("N1", 1, 3)])]},
    }
    document_tab = _document_tab(body, headers=headers, footers=footers, footnotes=footnotes)
    payload = _document_payload(tabs=[_tab("tab-1", "Main", 0, 0, document_tab)])
    reader, _ = _reader_with_payload(payload)
    document = reader.read_document(document_id=_DOCUMENT_ID)
    segments = document.tabs[0].segments
    kinds = [segment.kind for segment in segments]
    assert kinds[:1] == [GoogleDocsSegmentKind.BODY]
    assert kinds[1:3] == [GoogleDocsSegmentKind.HEADER, GoogleDocsSegmentKind.HEADER]
    assert [segment.segment_id for segment in segments[1:3]] == ["header-a", "header-b"]
    assert kinds[3:5] == [GoogleDocsSegmentKind.FOOTER, GoogleDocsSegmentKind.FOOTER]
    assert [segment.segment_id for segment in segments[3:5]] == ["footer-a", "footer-b"]
    assert kinds[5:7] == [GoogleDocsSegmentKind.FOOTNOTE, GoogleDocsSegmentKind.FOOTNOTE]
    assert [segment.segment_id for segment in segments[5:7]] == ["fn-a", "fn-b"]


def test_structural_blocks_and_tables() -> None:
    nested_table_cell_content = [
        _paragraph_block(31, 34, [_text_run("N", 31, 32)]),
    ]
    nested_table = {
        "startIndex": 30,
        "endIndex": 35,
        "table": {
            "rows": 1,
            "columns": 1,
            "tableRows": [
                {
                    "startIndex": 30,
                    "endIndex": 35,
                    "tableCells": [
                        {
                            "startIndex": 30,
                            "endIndex": 35,
                            "content": nested_table_cell_content,
                        },
                    ],
                },
            ],
        },
    }
    outer_table = {
        "startIndex": 30,
        "endIndex": 50,
        "table": {
            "rows": 2,
            "columns": 3,
            "tableRows": [
                {
                    "startIndex": 30,
                    "endIndex": 40,
                    "tableCells": [
                        {"startIndex": 30, "endIndex": 35, "columnSpan": 2, "content": [nested_table]},
                        {"startIndex": 35, "endIndex": 40, "content": [_paragraph_block(35, 38, [_text_run("X", 35, 36)])]},
                    ],
                },
                {
                    "startIndex": 40,
                    "endIndex": 50,
                    "tableCells": [
                        {"startIndex": 40, "endIndex": 45, "content": [_paragraph_block(40, 43, [_text_run("Y", 40, 41)])]},
                        {"startIndex": 45, "endIndex": 48, "columnSpan": 2, "content": [_paragraph_block(45, 47, [_text_run("Z", 45, 46)])]},
                    ],
                },
            ],
        },
    }
    toc_child = _paragraph_block(22, 26, [_text_run("T", 22, 23)])
    body = [
        _paragraph_block(1, 10, [_text_run("P", 1, 2)]),
        {"startIndex": 10, "endIndex": 11, "sectionBreak": {"sectionStyle": {"sectionType": "CONTINUOUS"}}},
        {
            "startIndex": 11,
            "endIndex": 28,
            "tableOfContents": {"content": [toc_child]},
        },
        outer_table,
    ]
    payload = _document_payload(tabs=[_tab("tab-1", "Main", 0, 0, _document_tab(body))])
    reader, _ = _reader_with_payload(payload)
    document = reader.read_document(document_id=_DOCUMENT_ID)
    blocks = document.tabs[0].segments[0].blocks
    assert blocks[0].kind is GoogleDocsBlockKind.PARAGRAPH
    assert blocks[1].kind is GoogleDocsBlockKind.SECTION_BREAK
    assert blocks[2].kind is GoogleDocsBlockKind.TABLE_OF_CONTENTS
    assert blocks[2].children[0].paragraph.elements[0].text == "T"
    table_block = blocks[3]
    assert table_block.kind is GoogleDocsBlockKind.TABLE
    assert table_block.table.rows == 2
    assert table_block.table.columns == 3
    assert table_block.table.table_rows[0].cells[0].column_span == 2


def test_all_inline_kinds() -> None:
    elements: list[dict[str, object]] = [
        _text_run("  spaced\ttext\n", 1, 20),
        {"startIndex": 20, "endIndex": 21, "autoText": {"type": "PAGE_NUMBER"}},
        {"startIndex": 21, "endIndex": 22, "pageBreak": {}},
        {"startIndex": 22, "endIndex": 23, "columnBreak": {}},
        {
            "startIndex": 23,
            "endIndex": 24,
            "footnoteReference": {"footnoteId": "fn-1", "footnoteNumber": "1"},
        },
        {"startIndex": 24, "endIndex": 25, "horizontalRule": {}},
        {"startIndex": 25, "endIndex": 26, "equation": {}},
        {"startIndex": 26, "endIndex": 27, "inlineObjectElement": {"inlineObjectId": "inline-1"}},
        {
            "startIndex": 27,
            "endIndex": 28,
            "person": {"personId": "person-1", "email": "user@example.com", "name": "User Name"},
        },
        {
            "startIndex": 28,
            "endIndex": 29,
            "richLink": {
                "richLinkId": "rich-1",
                "richLinkProperties": {
                    "title": "Example",
                    "uri": "https://example.com/secret",
                    "mimeType": "text/html",
                },
            },
        },
        {
            "startIndex": 29,
            "endIndex": 30,
            "date": {
                "dateId": "date-1",
                "dateElementProperties": {
                    "displayText": "Jan 1",
                    "timestamp": "2024-01-01T00:00:00Z",
                },
            },
        },
    ]
    body = [_paragraph_block(1, 31, elements, positionedObjectIds=["pos-1"])]
    document_tab = _document_tab(
        body,
        lists={"list-1": {"listProperties": {}}},
        inlineObjects={"inline-1": {"objectId": "inline-1", "inlineObjectProperties": {}}},
        positionedObjects={"pos-1": {"positionedObjectProperties": {}}},
        footnotes={"fn-1": {"footnoteId": "fn-1", "content": [_paragraph_block(1, 5, [_text_run("F", 1, 2)])]}},
    )
    payload = _document_payload(tabs=[_tab("tab-1", "Main", 0, 0, document_tab)])
    reader, _ = _reader_with_payload(payload)
    document = reader.read_document(document_id=_DOCUMENT_ID)
    inline = document.tabs[0].segments[0].blocks[0].paragraph.elements
    kinds = [element.kind for element in inline]
    assert kinds == [
        GoogleDocsInlineKind.TEXT_RUN,
        GoogleDocsInlineKind.AUTO_TEXT,
        GoogleDocsInlineKind.PAGE_BREAK,
        GoogleDocsInlineKind.COLUMN_BREAK,
        GoogleDocsInlineKind.FOOTNOTE_REFERENCE,
        GoogleDocsInlineKind.HORIZONTAL_RULE,
        GoogleDocsInlineKind.EQUATION,
        GoogleDocsInlineKind.INLINE_OBJECT,
        GoogleDocsInlineKind.PERSON,
        GoogleDocsInlineKind.RICH_LINK,
        GoogleDocsInlineKind.DATE,
    ]
    assert inline[0].text == "  spaced\ttext\n"
    assert inline[1].reference_id == "PAGE_NUMBER"
    assert inline[4].reference_id == "fn-1"
    assert inline[7].reference_id == "inline-1"
    assert inline[8].text == "User Name"
    assert inline[9].text == "Example"
    assert inline[9].mime_type == "text/html"
    assert inline[10].auxiliary_text == "2024-01-01T00:00:00Z"
    assert document.tabs[0].segments[0].blocks[0].paragraph.positioned_object_ids == ("pos-1",)


def test_sensitive_fields_hidden_from_repr_and_frozen() -> None:
    payload = _document_payload(
        tabs=[
            _tab(
                "tab-1",
                "Main",
                0,
                0,
                    _document_tab([_paragraph_block(1, 8, [_text_run("Secret", 1, 7)])]),
            ),
        ],
    )
    reader, _ = _reader_with_payload(payload)
    document = reader.read_document(document_id=_DOCUMENT_ID)
    rendered = repr(document)
    assert _DOCUMENT_ID not in rendered
    assert _DOCUMENT_TITLE not in rendered
    assert "Secret" not in rendered
    assert "rev-42" not in rendered
    with pytest.raises(ValidationError):
        document.title = "changed"  # type: ignore[misc]


def test_exact_transport_request() -> None:
    document_id = "doc+special=id"
    payload = _document_payload(
        document_id=document_id,
        tabs=[_tab("tab-1", "Main", 0, 0, _document_tab([_paragraph_block(1, 5, [_text_run("A", 1, 2)])]))],
    )
    transport = _RecordingTransport(responses=[payload])
    reader = GoogleDocsKnowledgeReader(transport=transport)
    reader.read_document(document_id=document_id)
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["source_kind"] is GoogleWorkspaceSourceKind.DOCS
    assert call["relative_path"] == "/documents/doc%2Bspecial%3Did"
    assert call["params"] == {
        "includeTabsContent": True,
        "suggestionsViewMode": "PREVIEW_WITHOUT_SUGGESTIONS",
    }
    assert call["headers"] == {}


def test_reader_construction_and_invalid_transport() -> None:
    transport = _RecordingTransport()
    GoogleDocsKnowledgeReader(transport=transport)
    assert transport.calls == []
    with pytest.raises(IntegrationConfigurationError):
        GoogleDocsKnowledgeReader(transport=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "document_id",
    ["", "   ", 123, True, "a\x00b", "a\x7fb", "x" * 1025, "path/segment"],
)
def test_invalid_document_id_rejected(document_id: object) -> None:
    transport = _RecordingTransport()
    reader = GoogleDocsKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationConfigurationError, match=_INVALID_ID_MESSAGE):
        reader.read_document(document_id=document_id)  # type: ignore[arg-type]
    assert transport.calls == []


def test_transport_api_error_propagates() -> None:
    api_error = GoogleWorkspaceApiError(
        kind=GoogleWorkspaceErrorKind.NOT_FOUND,
        status_code=404,
        retry_after_seconds=None,
        safe_reason="not_found",
        attempts=1,
    )
    transport = _RecordingTransport(exception=api_error)
    reader = GoogleDocsKnowledgeReader(transport=transport)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        reader.read_document(document_id=_DOCUMENT_ID)
    assert exc_info.value is api_error


def test_transport_runtime_error_normalized() -> None:
    transport = _RecordingTransport(exception=RuntimeError("network blew up"))
    reader = GoogleDocsKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError, match=_REQUEST_FAILED_MESSAGE) as exc_info:
        reader.read_document(document_id=_DOCUMENT_ID)
    assert "network" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_malformed_top_level_response_type() -> None:
    transport = _RecordingTransport()
    transport.get_json = lambda **kwargs: "not-a-dict"  # type: ignore[method-assign, assignment]
    reader = GoogleDocsKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.read_document(document_id=_DOCUMENT_ID)
    _assert_safe_dependency_error(exc_info)


@pytest.mark.parametrize(
    "mutation",
    [
        {"documentId": None},
        {"title": ""},
        {"suggestionsViewMode": "DEFAULT"},
        {"tabs": []},
        {"comments": [{"id": "c1"}]},
        {"suggestions": [{"id": "s1"}]},
        {"body": {"content": []}},
    ],
)
def test_malformed_top_level_fields(mutation: dict[str, object]) -> None:
    base = _document_payload(
        tabs=[_tab("tab-1", "Main", 0, 0, _document_tab([_paragraph_block(1, 5, [_text_run("A", 1, 2)])]))],
    )
    base.update(mutation)
    reader, _ = _reader_with_payload(base)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE) as exc_info:
        reader.read_document(document_id=_DOCUMENT_ID)
    _assert_safe_dependency_error(exc_info)


def test_document_id_mismatch_rejected() -> None:
    payload = _document_payload(
        document_id="other-id",
        tabs=[_tab("tab-1", "Main", 0, 0, _document_tab([_paragraph_block(1, 5, [_text_run("A", 1, 2)])]))],
    )
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_document(document_id=_DOCUMENT_ID)


def test_duplicate_tab_id_rejected() -> None:
    tab = _tab("tab-1", "Main", 0, 0, _document_tab([_paragraph_block(1, 5, [_text_run("A", 1, 2)])]))
    payload = _document_payload(tabs=[tab, tab])
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_document(document_id=_DOCUMENT_ID)


def test_dangling_references_rejected() -> None:
    body = [
        _paragraph_block(
            1,
            10,
            [
                {
                    "startIndex": 1,
                    "endIndex": 2,
                    "inlineObjectElement": {"inlineObjectId": "missing-inline"},
                },
            ],
        ),
    ]
    payload = _document_payload(tabs=[_tab("tab-1", "Main", 0, 0, _document_tab(body))])
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_document(document_id=_DOCUMENT_ID)


def test_unknown_structural_union_rejected() -> None:
    body = [{"startIndex": 1, "endIndex": 2, "unknownFuture": {}}]
    payload = _document_payload(tabs=[_tab("tab-1", "Main", 0, 0, _document_tab(body))])
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_document(document_id=_DOCUMENT_ID)


def test_unsafe_text_control_rejected() -> None:
    body = [_paragraph_block(1, 5, [_text_run("bad\x00text", 1, 9)])]
    payload = _document_payload(tabs=[_tab("tab-1", "Main", 0, 0, _document_tab(body))])
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_document(document_id=_DOCUMENT_ID)


def test_text_budget_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read import docs as docs_module

    monkeypatch.setattr(docs_module, "_MAX_TEXT_CHARS", 3)
    body = [_paragraph_block(1, 10, [_text_run("four", 1, 5)])]
    payload = _document_payload(tabs=[_tab("tab-1", "Main", 0, 0, _document_tab(body))])
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_document(document_id=_DOCUMENT_ID)


def test_block_budget_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read import docs as docs_module

    monkeypatch.setattr(docs_module, "_MAX_BLOCKS", 1)
    body = [
        _paragraph_block(1, 5, [_text_run("A", 1, 2)]),
        _paragraph_block(5, 9, [_text_run("B", 5, 6)]),
    ]
    payload = _document_payload(tabs=[_tab("tab-1", "Main", 0, 0, _document_tab(body))])
    reader, _ = _reader_with_payload(payload)
    with pytest.raises(IntegrationDependencyError, match=_UNEXPECTED_MESSAGE):
        reader.read_document(document_id=_DOCUMENT_ID)


def test_model_construct_rejected() -> None:
    with pytest.raises(ValidationError):
        GoogleDocsDocument(
            document_id=_DOCUMENT_ID,
            title=_DOCUMENT_TITLE,
            suggestions_view_mode="PREVIEW_WITHOUT_SUGGESTIONS",
            tabs=(),
        )


def test_direct_model_mutation_rejected() -> None:
    paragraph = GoogleDocsParagraph(elements=())
    with pytest.raises(ValidationError):
        paragraph.elements = ()  # type: ignore[misc]


class _HostileString(str):
    def strip(self) -> str:
        raise RuntimeError("hostile strip")


def test_hostile_document_id_strip_rejected() -> None:
    transport = _RecordingTransport()
    reader = GoogleDocsKnowledgeReader(transport=transport)
    with pytest.raises(IntegrationConfigurationError, match=_INVALID_ID_MESSAGE):
        reader.read_document(document_id=_HostileString(" doc-1 "))  # type: ignore[arg-type]
    assert transport.calls == []


# --- Integration delegation ---


@dataclass(frozen=True, slots=True)
class _FakeTransport:
    responses: list[dict[str, object]] = field(default_factory=list)
    calls: list[dict[str, object]] = field(default_factory=list)

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "source_kind": source_kind,
                "relative_path": relative_path,
                "params": dict(params or {}),
            }
        )
        if not self.responses:
            return {}
        return self.responses.pop(0)


@dataclass(frozen=True, slots=True)
class _FakeClientFamily:
    _transport: _FakeTransport

    @property
    def transport(self) -> _FakeTransport:
        return self._transport


def test_integration_read_docs_document_delegates_transport() -> None:
    payload = _document_payload(
        tabs=[_tab("tab-1", "Main", 0, 0, _document_tab([_paragraph_block(1, 5, [_text_run("A", 1, 2)])]))],
    )
    transport = _FakeTransport(responses=[payload])
    family = _FakeClientFamily(_transport=transport)
    integration = GoogleWorkspaceCollaborationSuiteIntegration.from_client(
        family,  # type: ignore[arg-type]
        enabled=True,
    )
    document = integration.read_docs_document(document_id=_DOCUMENT_ID)
    assert document.document_id == _DOCUMENT_ID
    assert len(transport.calls) == 1
    assert transport.calls[0]["source_kind"] is GoogleWorkspaceSourceKind.DOCS


def test_disabled_integration_read_docs_document_fails() -> None:
    integration = GoogleWorkspaceCollaborationSuiteIntegration.for_provider(
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        display_name="Google Workspace",
        config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(enabled=False),
    )
    with pytest.raises(IntegrationConfigurationError, match="disabled"):
        integration.read_docs_document(document_id=_DOCUMENT_ID)


def test_knowledge_read_package_exports_docs_symbols() -> None:
    from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read import (
        GoogleDocsKnowledgeReader as PackageReader,
    )

    assert PackageReader is GoogleDocsKnowledgeReader
    assert GOOGLE_DOCS_SOURCE_KIND == "docs"


def test_lazy_docs_exports_resolve() -> None:
    assert google_workspace.GOOGLE_DOCS_SOURCE_KIND == "docs"
    assert google_workspace.GoogleDocsKnowledgeReader is GoogleDocsKnowledgeReader


def test_existing_drive_exports_remain() -> None:
    public_names = set(google_workspace.__all__)
    assert "GoogleDriveKnowledgeReader" in public_names
    assert "GoogleDocsKnowledgeReader" in public_names
