# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for GoogleWorkspaceDocsKnowledgeAdapter."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceBinaryPayload,
    GoogleWorkspaceBinaryTransport,
    GoogleWorkspaceSourceKind,
    GoogleWorkspaceTransport,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.docs import (
    GOOGLE_DOCS_NATIVE_MIME_TYPE,
    GOOGLE_DOCS_SOURCE_KIND,
    GoogleDocsDocument,
    GoogleDocsKnowledgeReader,
    GoogleDocsTab,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspaceErrorKind,
)
from intergrax.runtime.vendor_knowledge.adapters import (
    GOOGLE_DOCS_CURSOR_VERSION,
    GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE,
    GOOGLE_DOCS_ITEM_METADATA_VERSION,
    GOOGLE_DOCS_STRUCTURED_RECORD_MIME_TYPE,
    GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA,
    GOOGLE_DRIVE_USER_SCOPE_TYPE,
    GoogleWorkspaceDocsKnowledgeAdapter,
    GoogleWorkspaceDriveKnowledgeAdapter,
    register_google_workspace_docs_knowledge_adapter,
    register_google_workspace_drive_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_docs import (
    _build_structured_record,
    _compute_content_hash,
)
from intergrax.runtime.vendor_knowledge.contracts import VendorKnowledgeAdapter
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChangeKind,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_DOCUMENT_ID = "doc-main-1"
_DOCUMENT_TITLE = "Structured Doc"
_SECRET_REVISION = "secret-revision-42"
_SECRET_TITLE = "Secret Document Title"
_SECRET_URI = "https://example.com/secret-uri"

_INVALID_SCOPE_MESSAGE = "Google Workspace Docs knowledge source scope is invalid"
_INVALID_CURSOR_MESSAGE = "Google Workspace Docs knowledge cursor is invalid"
_COMPLETE_CURSOR_MESSAGE = (
    "Google Workspace Docs reconciliation cursor is complete; restart reconciliation"
)
_INVALID_PROVIDER_RESPONSE_MESSAGE = (
    "Google Workspace Docs knowledge provider response is invalid"
)
_INVALID_DESCRIPTOR_MESSAGE = "Google Workspace Docs document descriptor is invalid"
_CONFIGURATION_ERROR_MESSAGE = "Google Workspace Docs knowledge page limit is invalid"
_DEPENDENCY_UNAVAILABLE_MESSAGE = (
    "Google Workspace Docs knowledge dependency is unavailable"
)
_CONTENT_HASH_MISMATCH_MESSAGE = (
    "Google Workspace Docs document content changed since descriptor creation"
)
_UNSUPPORTED_PERMISSIONS_MESSAGE = (
    "Authoritative Google Docs permissions projection is not implemented"
)

_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "structured_record_schema",
        "native_mime_type",
        "tab_count",
    }
)


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


def _table_cell(
    start: int,
    end: int,
    content: list[dict[str, object]],
    *,
    row_span: int = 1,
    column_span: int = 1,
) -> dict[str, object]:
    cell: dict[str, object] = {
        "startIndex": start,
        "endIndex": end,
        "content": content,
    }
    if row_span != 1 or column_span != 1:
        cell["tableCellStyle"] = {"rowSpan": row_span, "columnSpan": column_span}
    return cell


def _document_payload(
    *,
    document_id: str = _DOCUMENT_ID,
    title: str = _DOCUMENT_TITLE,
    tabs: list[dict[str, object]],
    revision_id: str | None = _SECRET_REVISION,
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


def _representative_payload() -> dict[str, object]:
    inline_elements: list[dict[str, object]] = [
        _text_run("Body text", 1, 10),
        {
            "startIndex": 10,
            "endIndex": 11,
            "footnoteReference": {"footnoteId": "fn-1", "footnoteNumber": "1"},
        },
        {
            "startIndex": 11,
            "endIndex": 12,
            "person": {
                "personId": "person-1",
                "personProperties": {
                    "name": "User Name",
                    "email": "user@example.com",
                },
            },
        },
        {
            "startIndex": 12,
            "endIndex": 13,
            "richLink": {
                "richLinkId": "rich-1",
                "richLinkProperties": {
                    "title": "Example",
                    "uri": _SECRET_URI,
                    "mimeType": "text/html",
                },
            },
        },
        {
            "startIndex": 13,
            "endIndex": 14,
            "dateElement": {
                "dateId": "date-1",
                "dateElementProperties": {
                    "displayText": "Jan 1",
                    "timestamp": "2024-01-01T00:00:00Z",
                },
            },
        },
    ]
    body = [
        _paragraph_block(1, 14, inline_elements),
        _paragraph_block(
            14,
            24,
            [_text_run("Item", 14, 18)],
            bullet={"listId": "list-1", "nestingLevel": 0},
        ),
        {
            "startIndex": 24,
            "endIndex": 44,
            "table": {
                "rows": 1,
                "columns": 2,
                "tableRows": [
                    {
                        "startIndex": 24,
                        "endIndex": 44,
                        "tableCells": [
                            _table_cell(
                                24,
                                34,
                                [_paragraph_block(24, 29, [_text_run("A", 24, 25)])],
                                column_span=2,
                            ),
                        ],
                    },
                ],
            },
        },
    ]
    headers = {
        "header-a": {
            "headerId": "header-a",
            "content": [_paragraph_block(1, 7, [_text_run("Header", 1, 7)])],
        },
    }
    main_tab = _document_tab(
        body,
        headers=headers,
        lists={"list-1": {"listProperties": {}}},
        footnotes={
            "fn-1": {
                "footnoteId": "fn-1",
                "content": [_paragraph_block(1, 5, [_text_run("F", 1, 2)])],
            },
        },
    )
    second_tab = _document_tab([_paragraph_block(1, 5, [_text_run("Tab2", 1, 5)])])
    child_tab = _tab(
        "tab-child",
        "Child",
        0,
        1,
        _document_tab([_paragraph_block(1, 5, [_text_run("C", 1, 2)])]),
        parent_tab_id="tab-root",
    )
    root_tab = _tab("tab-root", "Root", 0, 0, main_tab)
    root_tab["childTabs"] = [child_tab]
    second_root = _tab("tab-root-2", "Second", 1, 0, second_tab)
    return _document_payload(tabs=[root_tab, second_root])


def _document_from_payload(
    payload: dict[str, object],
    *,
    document_id: str = _DOCUMENT_ID,
) -> GoogleDocsDocument:
    transport = _RecordingTransport(responses=[payload])
    reader = GoogleDocsKnowledgeReader(transport=transport)
    return reader.read_document(document_id=document_id)


def _representative_document() -> GoogleDocsDocument:
    return _document_from_payload(_representative_payload())


def _source(
    *,
    remote_scope_id: str = _DOCUMENT_ID,
    remote_scope_type: str = GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE,
    parameters: dict[str, Any] | None = None,
    provider_id: str = GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    integration_kind: IntegrationCategory = IntegrationCategory.COLLABORATION_SUITE,
    source_kind: str = GOOGLE_DOCS_SOURCE_KIND,
    safe_display_name: str = "Quarterly Plan",
) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        scope=KnowledgeSourceScope(
            remote_scope_id=remote_scope_id,
            remote_scope_type=remote_scope_type,
            safe_display_name=safe_display_name,
            parameters=parameters or {},
        ),
    )


class _FakeGoogleWorkspaceIntegration(CollaborationSuite):
    def __init__(
        self,
        *,
        documents_by_id: dict[str, GoogleDocsDocument] | None = None,
        read_sequence: list[GoogleDocsDocument] | None = None,
    ) -> None:
        self._documents_by_id = dict(documents_by_id or {})
        self._read_sequence = list(read_sequence or [])
        self.docs_calls: list[dict[str, Any]] = []

    def read_docs_document(
        self,
        *,
        document_id: str,
    ) -> GoogleDocsDocument:
        self.docs_calls.append({"document_id": document_id})
        if self._read_sequence:
            return self._read_sequence.pop(0)
        return self._documents_by_id[document_id]

    def get_message(self, user_id: str, message_id: str):
        raise NotImplementedError

    def list_messages(self, user_id: str, *, folder: str = "inbox", limit: int = 25):
        raise NotImplementedError

    def send_mail(self, user_id: str, *, subject: str, body: str, to):
        raise NotImplementedError

    def list_calendar_events(self, user_id: str, *, start: str, end: str, limit: int = 50):
        raise NotImplementedError

    def get_user(self, user_id: str):
        raise NotImplementedError

    def reply_message(self, user_id: str, message_id: str, *, body: str) -> None:
        raise NotImplementedError

    def create_event(
        self,
        user_id: str,
        *,
        subject: str,
        start: str,
        end: str,
        location: str = "",
        attendees=(),
    ):
        raise NotImplementedError


class _StubBinaryTransport(GoogleWorkspaceBinaryTransport):
    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        raise NotImplementedError

    def get_binary(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        max_bytes: int,
        range_limited: bool,
    ) -> GoogleWorkspaceBinaryPayload:
        raise NotImplementedError


class _StubClientFamily:
    def __init__(self) -> None:
        self.transport: GoogleWorkspaceTransport = _StubBinaryTransport()


class _BoundGoogleWorkspaceIntegration(GoogleWorkspaceCollaborationSuiteIntegration):
    _bound_fake: _FakeGoogleWorkspaceIntegration = PrivateAttr()

    @classmethod
    def from_fake(cls, fake: _FakeGoogleWorkspaceIntegration) -> _BoundGoogleWorkspaceIntegration:
        bound = cls.from_client(_StubClientFamily(), enabled=True)
        bound._bound_fake = fake
        return bound

    def read_docs_document(
        self,
        *,
        document_id: str,
    ) -> GoogleDocsDocument:
        return self._bound_fake.read_docs_document(document_id=document_id)


def _integration(fake: _FakeGoogleWorkspaceIntegration) -> GoogleWorkspaceCollaborationSuiteIntegration:
    return _BoundGoogleWorkspaceIntegration.from_fake(fake)


def _scope_fingerprint(document_id: str) -> str:
    payload = f"google_workspace\x00docs\x00{document_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _encode_cursor_payload(payload: dict[str, object]) -> KnowledgeCursor:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return KnowledgeCursor(value=encoded, version=GOOGLE_DOCS_CURSOR_VERSION)


def _complete_cursor(document_id: str = _DOCUMENT_ID) -> KnowledgeCursor:
    return _encode_cursor_payload(
        {
            "schema_version": GOOGLE_DOCS_CURSOR_VERSION,
            "scope_fingerprint": _scope_fingerprint(document_id),
            "complete": True,
        }
    )


def _descriptor_for_document(document: GoogleDocsDocument) -> KnowledgeItemDescriptor:
    record = _build_structured_record(document)
    content_hash = _compute_content_hash(record)
    return KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(
            remote_id=document.document_id,
            parent_remote_id=None,
            logical_key=None,
        ),
        revision=KnowledgeItemRevision(
            version=None,
            etag=None,
            content_hash=content_hash,
            acl_hash=None,
            updated_at=None,
        ),
        title=document.title,
        item_type="google_workspace_docs_document",
        content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=GOOGLE_DOCS_SOURCE_KIND,
            remote_id=document.document_id,
            web_url=None,
            safe_locator=None,
        ),
        metadata={
            "schema_version": GOOGLE_DOCS_ITEM_METADATA_VERSION,
            "structured_record_schema": GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA,
            "native_mime_type": GOOGLE_DOCS_NATIVE_MIME_TYPE,
            "tab_count": len(document.tabs),
        },
    )


def _assert_canonical_error_identity(err: VendorKnowledgeError) -> None:
    assert err.provider_id == GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID
    assert err.source_kind == GOOGLE_DOCS_SOURCE_KIND


def _assert_no_document_id_leak(rendered: str) -> None:
    assert _DOCUMENT_ID not in rendered


def _assert_invalid_scope_boundary(
    exc_info: pytest.ExceptionInfo[VendorKnowledgeError],
    *,
    fake: _FakeGoogleWorkspaceIntegration,
) -> None:
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert err.retryable is False
    assert err.safe_message == _INVALID_SCOPE_MESSAGE
    assert err.__cause__ is None
    _assert_canonical_error_identity(err)
    assert fake.docs_calls == []
    _assert_no_document_id_leak(f"{err!r} {err.safe_message}")


def _assert_invalid_cursor_boundary(
    exc_info: pytest.ExceptionInfo[VendorKnowledgeError],
    *,
    fake: _FakeGoogleWorkspaceIntegration,
) -> None:
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert err.retryable is False
    assert err.__cause__ is None
    _assert_canonical_error_identity(err)
    assert fake.docs_calls == []
    _assert_no_document_id_leak(f"{err!r} {err.safe_message}")


def _assert_invalid_provider_response_boundary(
    exc_info: pytest.ExceptionInfo[VendorKnowledgeError],
    *,
    fake: _FakeGoogleWorkspaceIntegration,
) -> None:
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert err.retryable is False
    assert err.safe_message == _INVALID_PROVIDER_RESPONSE_MESSAGE
    assert err.__cause__ is None
    _assert_canonical_error_identity(err)
    _assert_no_document_id_leak(f"{err!r} {err.safe_message}")


async def test_adapter_identity() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    assert adapter.provider_id == GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID
    assert adapter.integration_kind is IntegrationCategory.COLLABORATION_SUITE
    assert adapter.source_kind == GOOGLE_DOCS_SOURCE_KIND
    assert isinstance(adapter, VendorKnowledgeAdapter)


async def test_capabilities_exact_set() -> None:
    caps = GoogleWorkspaceDocsKnowledgeAdapter().capabilities
    assert caps.full_inventory is True
    assert caps.incremental_changes is False
    assert caps.content_fetch is True
    assert caps.binary_content is False
    assert caps.rich_text_content is False
    assert caps.structured_content is True
    assert caps.permissions is False
    assert caps.tombstones is False
    assert caps.remote_versions is False
    assert caps.reconciliation is True


async def test_valid_scope_inspect() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    source = _source()
    info = await adapter.inspect_scope(integration=_integration(fake), source=source)
    assert info.source.scope.remote_scope_id == _DOCUMENT_ID
    assert info.safe_display_name == "Quarterly Plan"
    assert info.capabilities == adapter.capabilities
    assert fake.docs_calls == []


async def test_read_page_single_upsert() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    document = _representative_document()
    fake = _FakeGoogleWorkspaceIntegration(documents_by_id={_DOCUMENT_ID: document})
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    assert len(fake.docs_calls) == 1
    assert fake.docs_calls[0]["document_id"] == _DOCUMENT_ID
    assert len(page.changes) == 1
    change = page.changes[0]
    assert change.kind is KnowledgeChangeKind.UPSERT
    assert change.remote_id == _DOCUMENT_ID
    assert change.descriptor is not None
    assert page.has_more is False
    assert page.next_cursor is None
    assert page.proposed_checkpoint is not None
    assert page.proposed_checkpoint.version == GOOGLE_DOCS_CURSOR_VERSION


async def test_descriptor_mapping() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    document = _representative_document()
    fake = _FakeGoogleWorkspaceIntegration(documents_by_id={_DOCUMENT_ID: document})
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=1,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.identity.remote_id == _DOCUMENT_ID
    assert descriptor.identity.parent_remote_id is None
    assert descriptor.identity.logical_key is None
    assert descriptor.title == _DOCUMENT_TITLE
    assert descriptor.item_type == "google_workspace_docs_document"
    assert descriptor.content_mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert descriptor.content_available is True
    assert descriptor.provenance.remote_id == _DOCUMENT_ID
    assert descriptor.provenance.provider_id == GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID
    assert descriptor.provenance.source_kind == GOOGLE_DOCS_SOURCE_KIND
    assert descriptor.provenance.web_url is None
    assert descriptor.revision.version is None
    assert descriptor.revision.etag is None
    assert descriptor.revision.acl_hash is None
    assert descriptor.revision.updated_at is None
    assert descriptor.revision.content_hash is not None
    assert len(descriptor.revision.content_hash) == 64
    assert set(descriptor.metadata.keys()) == _METADATA_KEYS
    assert descriptor.metadata["schema_version"] == GOOGLE_DOCS_ITEM_METADATA_VERSION
    assert descriptor.metadata["structured_record_schema"] == GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA
    assert descriptor.metadata["native_mime_type"] == GOOGLE_DOCS_NATIVE_MIME_TYPE
    assert descriptor.metadata["tab_count"] == len(document.tabs)
    dumped = json.dumps(descriptor.model_dump(mode="json"))
    assert "revision_id" not in dumped
    assert "revisionId" not in dumped


async def test_fetch_content_structured_record() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    document = _representative_document()
    fake = _FakeGoogleWorkspaceIntegration(
        read_sequence=[document, document],
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=descriptor,
    )
    assert len(fake.docs_calls) == 2
    assert fake.docs_calls[1]["document_id"] == _DOCUMENT_ID
    assert content.mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert content.mime_type == GOOGLE_DOCS_STRUCTURED_RECORD_MIME_TYPE
    record = content.structured_record
    assert record["schema_version"] == GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA
    assert record["document_id"] == _DOCUMENT_ID
    assert record["title"] == _DOCUMENT_TITLE
    assert len(record["tabs"]) == len(document.tabs)
    assert content.content_hash == descriptor.revision.content_hash
    assert "revision_id" not in record
    assert "revisionId" not in json.dumps(record)
    assert _SECRET_URI not in json.dumps(record)


async def test_revision_independence() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    base = _representative_document()
    revised_payload = _representative_payload()
    revised_payload["revisionId"] = "different-revision"
    revised = _document_from_payload(revised_payload)
    fake_a = _FakeGoogleWorkspaceIntegration(documents_by_id={_DOCUMENT_ID: base})
    fake_b = _FakeGoogleWorkspaceIntegration(documents_by_id={_DOCUMENT_ID: revised})
    page_a = await adapter.read_page(
        integration=_integration(fake_a),
        source=_source(),
        cursor=None,
        limit=50,
    )
    page_b = await adapter.read_page(
        integration=_integration(fake_b),
        source=_source(),
        cursor=None,
        limit=50,
    )
    desc_a = page_a.changes[0].descriptor
    desc_b = page_b.changes[0].descriptor
    assert desc_a is not None and desc_b is not None
    assert desc_a.revision.content_hash == desc_b.revision.content_hash
    content_a = await adapter.fetch_content(
        integration=_integration(fake_a),
        source=_source(),
        item=desc_a,
    )
    content_b = await adapter.fetch_content(
        integration=_integration(fake_b),
        source=_source(),
        item=desc_b,
    )
    assert content_a.structured_record == content_b.structured_record
    assert content_a.content_hash == content_b.content_hash
    for artifact in (page_a, page_b, desc_a, desc_b, content_a, content_b):
        dumped = json.dumps(artifact.model_dump(mode="json"))
        assert "revision_id" not in dumped
        assert "revisionId" not in dumped
        assert _SECRET_REVISION not in dumped


async def test_materialization_race() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    original = _representative_document()
    changed_payload = _representative_payload()
    changed_payload["tabs"][0]["documentTab"]["body"]["content"][0]["paragraph"]["elements"][0][
        "textRun"
    ]["content"] = "Changed body text"
    changed = _document_from_payload(changed_payload)
    fake = _FakeGoogleWorkspaceIntegration(read_sequence=[original, changed])
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=descriptor,
        )
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert err.retryable is True
    assert err.safe_message == _CONTENT_HASH_MISMATCH_MESSAGE


async def test_complete_cursor_rejected_as_continuation() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=_complete_cursor(),
            limit=50,
        )
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert err.safe_message == _COMPLETE_CURSOR_MESSAGE
    assert fake.docs_calls == []


@pytest.mark.parametrize(
    ("cursor", "message"),
    [
        (
            KnowledgeCursor(value="!!!", version=GOOGLE_DOCS_CURSOR_VERSION),
            _INVALID_CURSOR_MESSAGE,
        ),
        (
            KnowledgeCursor(
                value="not-valid-base64$$",
                version=GOOGLE_DOCS_CURSOR_VERSION,
            ),
            _INVALID_CURSOR_MESSAGE,
        ),
        (
            _encode_cursor_payload({"schema_version": "wrong.version", "complete": True}),
            _INVALID_CURSOR_MESSAGE,
        ),
        (
            _encode_cursor_payload(
                {
                    "schema_version": GOOGLE_DOCS_CURSOR_VERSION,
                    "scope_fingerprint": "deadbeef",
                    "complete": True,
                }
            ),
            _INVALID_CURSOR_MESSAGE,
        ),
    ],
)
async def test_invalid_cursors_rejected(
    cursor: KnowledgeCursor,
    message: str,
) -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert err.safe_message == message
    assert err.retryable is False
    assert fake.docs_calls == []
    _assert_no_document_id_leak(f"{err!r} {err.safe_message}")


async def test_cursor_does_not_store_raw_document_id() -> None:
    cursor = _complete_cursor()
    assert _DOCUMENT_ID not in cursor.value


async def test_wrong_integration_object_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=object(),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


@pytest.mark.parametrize(
    "source",
    [
        _source(provider_id="other"),
        _source(integration_kind=IntegrationCategory.ISSUE_TRACKER),
        _source(source_kind="drive"),
        _source(remote_scope_type=GOOGLE_DRIVE_USER_SCOPE_TYPE),
        _source(remote_scope_id="doc\x01"),
        _source(remote_scope_id="doc\x7f"),
        _source(remote_scope_id="doc/id"),
        _source(remote_scope_id="doc\\id"),
        _source(remote_scope_id="x" * 1025),
        _source(parameters={"x": "y"}),
    ],
)
async def test_invalid_scope_rejected(source: KnowledgeSourceRef) -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=source,
            cursor=None,
            limit=50,
        )
    _assert_invalid_scope_boundary(exc_info, fake=fake)


async def test_blank_document_id_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    source = KnowledgeSourceRef.model_construct(
        tenant_id="tenant-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_DOCS_SOURCE_KIND,
        scope=KnowledgeSourceScope.model_construct(
            remote_scope_id="",
            remote_scope_type=GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE,
            safe_display_name="Doc",
            parameters={},
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=source,
            cursor=None,
            limit=50,
        )
    _assert_invalid_scope_boundary(exc_info, fake=fake)


class _KnowledgeSourceRefSubclass(KnowledgeSourceRef):
    pass


async def test_foreign_source_subclass_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    source = _KnowledgeSourceRefSubclass(
        tenant_id="tenant-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_DOCS_SOURCE_KIND,
        scope=KnowledgeSourceScope(
            remote_scope_id=_DOCUMENT_ID,
            remote_scope_type=GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE,
            safe_display_name="Doc",
            parameters={},
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=source,
            cursor=None,
            limit=50,
        )
    _assert_invalid_scope_boundary(exc_info, fake=fake)


async def test_malformed_model_construct_source_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    source = KnowledgeSourceRef.model_construct(
        tenant_id="tenant-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_DOCS_SOURCE_KIND,
        scope=KnowledgeSourceScope.model_construct(
            remote_scope_id=_DOCUMENT_ID,
            remote_scope_type=GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE,
            safe_display_name=123,
            parameters={},
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=source,
            cursor=None,
            limit=50,
        )
    _assert_invalid_scope_boundary(exc_info, fake=fake)


@pytest.mark.parametrize("limit", [True, 0, -1, 1001, 1.0, "100"])
async def test_invalid_limit_rejected(limit: object) -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=limit,  # type: ignore[arg-type]
        )
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert err.safe_message == _CONFIGURATION_ERROR_MESSAGE
    assert fake.docs_calls == []


@pytest.mark.parametrize("limit", [1, 1000])
async def test_valid_limits_accepted(limit: int) -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    document = _representative_document()
    fake = _FakeGoogleWorkspaceIntegration(documents_by_id={_DOCUMENT_ID: document})
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=limit,
    )
    assert len(page.changes) == 1


async def test_fetch_permissions_unsupported() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    document = _representative_document()
    descriptor = _descriptor_for_document(document)
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_permissions(
            integration=_integration(fake),
            source=_source(),
            item=descriptor,
        )
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert err.safe_message == _UNSUPPORTED_PERMISSIONS_MESSAGE
    assert fake.docs_calls == []


async def test_registry_registration() -> None:
    registry = KnowledgeAdapterRegistry()
    adapter = register_google_workspace_docs_knowledge_adapter(registry)
    assert isinstance(adapter, GoogleWorkspaceDocsKnowledgeAdapter)
    resolved = registry.resolve(source=_source())
    assert resolved is adapter
    drive_registry = KnowledgeAdapterRegistry()
    register_google_workspace_drive_knowledge_adapter(drive_registry)
    assert isinstance(
        drive_registry.resolve(
            source=KnowledgeSourceRef(
                tenant_id="tenant-1",
                provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                source_kind="drive",
                scope=KnowledgeSourceScope(
                    remote_scope_id="user",
                    remote_scope_type="google_workspace_drive_user",
                    safe_display_name="My Drive",
                    parameters={},
                ),
            )
        ),
        GoogleWorkspaceDriveKnowledgeAdapter,
    )


async def test_package_exports() -> None:
    from intergrax.runtime.vendor_knowledge import adapters as package

    assert package.GOOGLE_DOCS_CURSOR_VERSION == GOOGLE_DOCS_CURSOR_VERSION
    assert package.GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE == GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE
    assert package.GOOGLE_DOCS_ITEM_METADATA_VERSION == GOOGLE_DOCS_ITEM_METADATA_VERSION
    assert package.GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA == GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA
    assert package.GOOGLE_DOCS_STRUCTURED_RECORD_MIME_TYPE == GOOGLE_DOCS_STRUCTURED_RECORD_MIME_TYPE
    assert package.GoogleWorkspaceDocsKnowledgeAdapter is GoogleWorkspaceDocsKnowledgeAdapter
    assert (
        package.register_google_workspace_docs_knowledge_adapter
        is register_google_workspace_docs_knowledge_adapter
    )


@pytest.mark.parametrize(
    ("kind", "expected_code", "retryable"),
    [
        (GoogleWorkspaceErrorKind.AUTHENTICATION, VendorKnowledgeErrorCode.AUTHENTICATION_FAILED, False),
        (GoogleWorkspaceErrorKind.AUTHORIZATION, VendorKnowledgeErrorCode.AUTHORIZATION_DENIED, False),
        (GoogleWorkspaceErrorKind.NOT_FOUND, VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND, False),
        (GoogleWorkspaceErrorKind.RATE_LIMITED, VendorKnowledgeErrorCode.RATE_LIMITED, True),
        (GoogleWorkspaceErrorKind.TEMPORARY, VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE, True),
        (GoogleWorkspaceErrorKind.MALFORMED_RESPONSE, VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE, False),
        (GoogleWorkspaceErrorKind.UNEXPECTED_REDIRECT, VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE, False),
        (GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE, VendorKnowledgeErrorCode.CONFIGURATION_ERROR, False),
        (GoogleWorkspaceErrorKind.INVALID_REQUEST, VendorKnowledgeErrorCode.CONFIGURATION_ERROR, False),
    ],
)
async def test_google_api_error_mapping(
    kind: GoogleWorkspaceErrorKind,
    expected_code: VendorKnowledgeErrorCode,
    retryable: bool,
) -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_docs_document(self, *, document_id: str) -> GoogleDocsDocument:
            raise GoogleWorkspaceApiError(
                kind=kind,
                status_code=500,
                retry_after_seconds=None,
                safe_reason="secret-provider-detail",
                attempts=1,
            )

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    err = exc_info.value
    assert err.code is expected_code
    assert err.retryable is retryable
    assert "secret-provider-detail" not in str(err)
    assert _DOCUMENT_ID not in str(err)
    assert _SECRET_TITLE not in str(err)


async def test_integration_configuration_error_translated() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_docs_document(self, *, document_id: str) -> GoogleDocsDocument:
            raise IntegrationConfigurationError("misconfigured")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR


async def test_integration_dependency_error_translated() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_docs_document(self, *, document_id: str) -> GoogleDocsDocument:
            raise IntegrationDependencyError("temporary outage")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert "temporary outage" not in str(exc_info.value)


async def test_generic_exception_translated() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_docs_document(self, *, document_id: str) -> GoogleDocsDocument:
            raise RuntimeError("unexpected runtime failure")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert "unexpected runtime failure" not in str(exc_info.value)


async def test_public_objects_do_not_leak_secrets() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    document = _representative_document()
    fake = _FakeGoogleWorkspaceIntegration(read_sequence=[document, document])
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=descriptor,
    )
    checkpoint = page.proposed_checkpoint
    assert checkpoint is not None
    for artifact in (page, descriptor, content, checkpoint):
        blob = json.dumps(artifact.model_dump(mode="json"))
        for secret in (
            "revision_id",
            "revisionId",
            _SECRET_REVISION,
            _SECRET_URI,
            "Authorization",
            "Bearer",
            "access_token",
            "refresh_token",
            "client_secret",
            "x-goog-api-key",
            "connection_ref",
        ):
            assert secret not in blob
    assert _DOCUMENT_ID not in checkpoint.value
    assert _SECRET_TITLE not in repr(descriptor)
    assert _DOCUMENT_TITLE not in repr(content)
    assert "Body text" not in repr(content)
    assert checkpoint.value not in repr(checkpoint)


async def test_provider_wrong_type_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()

    class _WrongType:
        pass

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_docs_document(self, *, document_id: str):
            return _WrongType()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)


async def test_provider_subclass_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()

    class _DocsSubclass(GoogleDocsDocument):
        pass

    document = _representative_document()
    subclass = _DocsSubclass(**document.model_dump())

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_docs_document(self, *, document_id: str) -> GoogleDocsDocument:
            return subclass

    fake = _FailingSuite()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)


async def test_provider_document_id_mismatch_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    document = _representative_document()
    payload = document.model_dump()
    payload["document_id"] = "other-doc-id"
    mismatched = GoogleDocsDocument(**payload)

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_docs_document(self, *, document_id: str) -> GoogleDocsDocument:
            return mismatched

    fake = _FailingSuite()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)


async def test_descriptor_wrong_remote_id_rejected_before_provider() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    document = _representative_document()
    descriptor = _descriptor_for_document(document)
    bad = descriptor.model_copy(
        update={
            "identity": descriptor.identity.model_copy(update={"remote_id": "other-doc"}),
        }
    )
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=bad,
        )
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert err.safe_message == _INVALID_DESCRIPTOR_MESSAGE
    assert fake.docs_calls == []


async def test_descriptor_extra_metadata_key_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    document = _representative_document()
    descriptor = _descriptor_for_document(document)
    metadata = dict(descriptor.metadata)
    metadata["extra"] = "x"
    bad = descriptor.model_copy(update={"metadata": metadata})
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=bad,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert fake.docs_calls == []


class _KnowledgeItemDescriptorSubclass(KnowledgeItemDescriptor):
    pass


async def test_descriptor_subclass_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    document = _representative_document()
    base = _descriptor_for_document(document)
    bad = _KnowledgeItemDescriptorSubclass(**base.model_dump())
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=bad,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert fake.docs_calls == []


async def test_provider_model_construct_malformed_tab_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    malformed_tab = GoogleDocsTab.model_construct(
        tab_id="tab-1",
        title="T",
        parent_tab_id=None,
        index=0,
        nesting_level=0,
        segments=(),
    )
    malformed = GoogleDocsDocument.model_construct(
        document_id=_DOCUMENT_ID,
        title=_DOCUMENT_TITLE,
        revision_id=_SECRET_REVISION,
        suggestions_view_mode="PREVIEW_WITHOUT_SUGGESTIONS",
        tabs=(malformed_tab,),
    )

    class _FailingSuite(_FakeGoogleWorkspaceIntegration):
        def read_docs_document(self, *, document_id: str) -> GoogleDocsDocument:
            return malformed

    fake = _FailingSuite()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    _assert_invalid_provider_response_boundary(exc_info, fake=fake)


async def test_cursor_foreign_version_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    cursor = KnowledgeCursor(value="abc", version="foreign.version")
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    _assert_invalid_cursor_boundary(exc_info, fake=fake)


async def test_cursor_extra_field_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    cursor = _encode_cursor_payload(
        {
            "schema_version": GOOGLE_DOCS_CURSOR_VERSION,
            "scope_fingerprint": _scope_fingerprint(_DOCUMENT_ID),
            "complete": True,
            "extra": "field",
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    _assert_invalid_cursor_boundary(exc_info, fake=fake)


async def test_cursor_missing_field_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    cursor = _encode_cursor_payload(
        {
            "schema_version": GOOGLE_DOCS_CURSOR_VERSION,
            "scope_fingerprint": _scope_fingerprint(_DOCUMENT_ID),
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    _assert_invalid_cursor_boundary(exc_info, fake=fake)


class _KnowledgeCursorSubclass(KnowledgeCursor):
    pass


async def test_cursor_subclass_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    base = _complete_cursor()
    cursor = _KnowledgeCursorSubclass(**base.model_dump())
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    _assert_invalid_cursor_boundary(exc_info, fake=fake)


async def test_cursor_model_construct_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    cursor = KnowledgeCursor.model_construct(
        value=_complete_cursor().value,
        version=GOOGLE_DOCS_CURSOR_VERSION,
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    _assert_invalid_cursor_boundary(exc_info, fake=fake)


async def test_cursor_non_object_json_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    encoded = base64.urlsafe_b64encode(b"[1,2,3]").decode("ascii").rstrip("=")
    cursor = KnowledgeCursor(value=encoded, version=GOOGLE_DOCS_CURSOR_VERSION)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    _assert_invalid_cursor_boundary(exc_info, fake=fake)


async def test_cursor_over_limit_value_rejected() -> None:
    adapter = GoogleWorkspaceDocsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    encoded = "A" * 25_000
    cursor = KnowledgeCursor(value=encoded, version=GOOGLE_DOCS_CURSOR_VERSION)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    _assert_invalid_cursor_boundary(exc_info, fake=fake)
