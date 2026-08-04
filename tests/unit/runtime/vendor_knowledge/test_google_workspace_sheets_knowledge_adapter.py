# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for GoogleWorkspaceSheetsKnowledgeAdapter."""

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
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.sheets import (
    GOOGLE_SHEETS_NATIVE_MIME_TYPE,
    GOOGLE_SHEETS_SOURCE_KIND,
    GoogleSheetsKnowledgeReader,
    GoogleSheetsSpreadsheet,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspaceErrorKind,
)
from intergrax.runtime.vendor_knowledge.adapters import (
    GOOGLE_SHEETS_CURSOR_VERSION,
    GOOGLE_SHEETS_ITEM_METADATA_VERSION,
    GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE,
    GOOGLE_SHEETS_STRUCTURED_RECORD_MIME_TYPE,
    GOOGLE_SHEETS_STRUCTURED_RECORD_SCHEMA,
    GoogleWorkspaceSheetsKnowledgeAdapter,
    register_google_workspace_sheets_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_sheets import (
    _build_structured_record,
    _compute_content_hash,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
    to_source_ref,
)
from intergrax.runtime.vendor_knowledge.contracts import VendorKnowledgeAdapter
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
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

_SPREADSHEET_ID = "sheet-main-1"
_SPREADSHEET_TITLE = "Structured Spreadsheet"
_SHEET_TITLE = "GridSheet"
_FORMULA = "=SUM(1,2)"
_CELL_TEXT = "hello"
_NOTE = "line one\nline two"
_ERROR_MESSAGE = "division by zero"
_CONNECTION_REF = "conn-google-1"
_INVALID_SCOPE_MESSAGE = "Google Workspace Sheets knowledge source scope is invalid"
_INVALID_CURSOR_MESSAGE = "Google Workspace Sheets knowledge cursor is invalid"
_COMPLETE_CURSOR_MESSAGE = (
    "Google Workspace Sheets reconciliation cursor is complete; restart reconciliation"
)
_INVALID_PROVIDER_RESPONSE_MESSAGE = (
    "Google Workspace Sheets knowledge provider response is invalid"
)
_INVALID_DESCRIPTOR_MESSAGE = "Google Workspace Sheets spreadsheet descriptor is invalid"
_CONFIGURATION_ERROR_MESSAGE = "Google Workspace Sheets knowledge page limit is invalid"
_CONTENT_HASH_MISMATCH_MESSAGE = (
    "Google Workspace Sheets spreadsheet content changed since descriptor creation"
)
_UNSUPPORTED_PERMISSIONS_MESSAGE = (
    "Authoritative Google Sheets permissions projection is not implemented"
)


@dataclass
class _RecordingTransport:
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
                "headers": dict(headers or {}),
            }
        )
        return self.responses.pop(0)


def _representative_payload(
    *,
    spreadsheet_id: str = _SPREADSHEET_ID,
    title: str = _SPREADSHEET_TITLE,
    formula: str = _FORMULA,
    note: str = _NOTE,
    merge_end_row: int = 2,
    named_range_name: str = "BoundedRange",
) -> dict[str, object]:
    values = [
        {
            "userEnteredValue": {"stringValue": _CELL_TEXT},
            "effectiveValue": {"stringValue": _CELL_TEXT},
        },
        {
            "userEnteredValue": {"numberValue": 42},
            "effectiveValue": {"numberValue": 42},
        },
        {
            "userEnteredValue": {"boolValue": True},
            "effectiveValue": {"boolValue": True},
        },
        {
            "userEnteredValue": {"formulaValue": formula},
            "effectiveValue": {"numberValue": 3},
            "formattedValue": "3",
        },
        {
            "userEnteredValue": {"numberValue": 99.5},
            "effectiveValue": {"numberValue": 99.5},
            "formattedValue": "$99.50",
            "effectiveFormat": {
                "numberFormat": {"type": "CURRENCY", "pattern": "$#,##0.00"},
            },
        },
        {
            "userEnteredValue": {"numberValue": 45000},
            "effectiveValue": {"numberValue": 45000},
            "formattedValue": "1/15/2024",
            "effectiveFormat": {
                "numberFormat": {"type": "DATE", "pattern": "M/d/yyyy"},
            },
            "note": note,
        },
        {
            "userEnteredValue": {"formulaValue": "=1/0"},
            "effectiveValue": {
                "errorValue": {
                    "type": "DIVIDE_BY_ZERO",
                    "message": _ERROR_MESSAGE,
                },
            },
        },
    ]
    return {
        "spreadsheetId": spreadsheet_id,
        "properties": {
            "title": title,
            "locale": "en_US",
            "timeZone": "America/New_York",
            "autoRecalc": "ON_CHANGE",
        },
        "sheets": [
            {
                "properties": {
                    "sheetId": 100,
                    "title": _SHEET_TITLE,
                    "index": 0,
                    "sheetType": "GRID",
                    "gridProperties": {
                        "rowCount": 10,
                        "columnCount": 10,
                        "frozenRowCount": 1,
                        "frozenColumnCount": 1,
                    },
                    "hidden": True,
                    "rightToLeft": True,
                },
                "data": [
                    {
                        "startRow": 2,
                        "startColumn": 3,
                        "rowData": [{"values": values}],
                    },
                ],
                "merges": [
                    {
                        "sheetId": 100,
                        "startRowIndex": 0,
                        "endRowIndex": merge_end_row,
                        "startColumnIndex": 0,
                        "endColumnIndex": 2,
                    },
                ],
            },
            {
                "properties": {
                    "sheetId": 101,
                    "title": "ObjectSheet",
                    "index": 1,
                    "sheetType": "OBJECT",
                },
            },
        ],
        "namedRanges": [
            {
                "namedRangeId": "nr-bounded",
                "name": named_range_name,
                "range": {
                    "sheetId": 100,
                    "startRowIndex": 0,
                    "endRowIndex": 3,
                    "startColumnIndex": 0,
                    "endColumnIndex": 2,
                },
            },
        ],
    }


def _spreadsheet_from_payload(payload: dict[str, object]) -> GoogleSheetsSpreadsheet:
    reader = GoogleSheetsKnowledgeReader(
        transport=_RecordingTransport(responses=[payload]),
    )
    return reader.read_spreadsheet(spreadsheet_id=payload["spreadsheetId"])


def _representative_spreadsheet(**kwargs: object) -> GoogleSheetsSpreadsheet:
    return _spreadsheet_from_payload(_representative_payload(**kwargs))


def _source(
    *,
    remote_scope_id: str = _SPREADSHEET_ID,
    remote_scope_type: str = GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE,
    parameters: dict[str, Any] | None = None,
    provider_id: str = GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    integration_kind: IntegrationCategory = IntegrationCategory.COLLABORATION_SUITE,
    source_kind: str = GOOGLE_SHEETS_SOURCE_KIND,
    connection_ref: str | None = _CONNECTION_REF,
    safe_display_name: str = "Quarterly Plan",
) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        connection_ref=connection_ref,
        scope=KnowledgeSourceScope(
            remote_scope_id=remote_scope_id,
            remote_scope_type=remote_scope_type,
            safe_display_name=safe_display_name,
            parameters=parameters or {},
        ),
    )


class _FakeGoogleWorkspaceIntegration:
    def __init__(
        self,
        *,
        spreadsheets: list[GoogleSheetsSpreadsheet] | None = None,
    ) -> None:
        self._spreadsheets = list(spreadsheets or [])
        self.sheets_calls: list[dict[str, Any]] = []

    def read_sheets_spreadsheet(
        self,
        *,
        spreadsheet_id: str,
    ) -> GoogleSheetsSpreadsheet:
        self.sheets_calls.append({"spreadsheet_id": spreadsheet_id})
        return self._spreadsheets.pop(0)


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
    def from_fake(
        cls,
        fake: _FakeGoogleWorkspaceIntegration,
    ) -> _BoundGoogleWorkspaceIntegration:
        bound = cls.from_client(_StubClientFamily(), enabled=True)
        bound._bound_fake = fake
        return bound

    def read_sheets_spreadsheet(
        self,
        *,
        spreadsheet_id: str,
    ) -> GoogleSheetsSpreadsheet:
        return self._bound_fake.read_sheets_spreadsheet(
            spreadsheet_id=spreadsheet_id,
        )


def _integration(
    fake: _FakeGoogleWorkspaceIntegration,
) -> GoogleWorkspaceCollaborationSuiteIntegration:
    return _BoundGoogleWorkspaceIntegration.from_fake(fake)


def _binding() -> KnowledgeSourceBinding:
    return KnowledgeSourceBinding(
        binding_id="binding-sheets-1",
        tenant_id="tenant-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_SHEETS_SOURCE_KIND,
        connection_ref=_CONNECTION_REF,
        safe_display_name="Quarterly Plan",
        scope=KnowledgeSourceScope(
            remote_scope_id=_SPREADSHEET_ID,
            remote_scope_type=GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE,
            safe_display_name="Quarterly Plan",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
    )


class _RecordingResolver:
    def __init__(self, integration: GoogleWorkspaceCollaborationSuiteIntegration) -> None:
        self.integration = integration
        self.received_sources: list[KnowledgeSourceRef] = []

    def resolve(self, *, source: KnowledgeSourceRef) -> object:
        self.received_sources.append(source)
        return self.integration


def _scope_fingerprint(spreadsheet_id: str = _SPREADSHEET_ID) -> str:
    return hashlib.sha256(
        f"google_workspace\x00sheets\x00{spreadsheet_id}".encode("utf-8")
    ).hexdigest()


def _cursor_payload(
    payload: dict[str, object],
    *,
    version: str = GOOGLE_SHEETS_CURSOR_VERSION,
) -> KnowledgeCursor:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    value = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return KnowledgeCursor(value=value, version=version)


def _complete_cursor() -> KnowledgeCursor:
    return _cursor_payload(
        {
            "schema_version": GOOGLE_SHEETS_CURSOR_VERSION,
            "scope_fingerprint": _scope_fingerprint(),
            "complete": True,
        }
    )


def _descriptor_for(
    spreadsheet: GoogleSheetsSpreadsheet | None = None,
) -> KnowledgeItemDescriptor:
    spreadsheet = spreadsheet or _representative_spreadsheet()
    record = _build_structured_record(spreadsheet)
    return KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(
            remote_id=spreadsheet.spreadsheet_id,
            parent_remote_id=None,
            logical_key=None,
        ),
        revision=KnowledgeItemRevision(
            version=None,
            etag=None,
            content_hash=_compute_content_hash(record),
            acl_hash=None,
            updated_at=None,
        ),
        title=spreadsheet.title,
        item_type="google_workspace_sheets_spreadsheet",
        content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=GOOGLE_SHEETS_SOURCE_KIND,
            remote_id=spreadsheet.spreadsheet_id,
            web_url=None,
            safe_locator=None,
        ),
        metadata={
            "schema_version": GOOGLE_SHEETS_ITEM_METADATA_VERSION,
            "structured_record_schema": GOOGLE_SHEETS_STRUCTURED_RECORD_SCHEMA,
            "native_mime_type": GOOGLE_SHEETS_NATIVE_MIME_TYPE,
            "sheet_count": len(spreadsheet.sheets),
            "named_range_count": len(spreadsheet.named_ranges),
        },
    )


async def _read_page(
    adapter: GoogleWorkspaceSheetsKnowledgeAdapter,
    fake: _FakeGoogleWorkspaceIntegration,
    *,
    source: KnowledgeSourceRef | None = None,
    cursor: KnowledgeCursor | None = None,
    limit: int = 50,
):
    return await adapter.read_page(
        integration=_integration(fake),
        source=source or _source(),
        cursor=cursor,
        limit=limit,
    )


async def test_identity_and_capabilities() -> None:
    adapter = GoogleWorkspaceSheetsKnowledgeAdapter()
    assert isinstance(adapter, VendorKnowledgeAdapter)
    assert adapter.provider_id == GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID
    assert adapter.integration_kind is IntegrationCategory.COLLABORATION_SUITE
    assert adapter.source_kind == GOOGLE_SHEETS_SOURCE_KIND
    assert adapter.capabilities.model_dump() == {
        "full_inventory": True,
        "incremental_changes": False,
        "content_fetch": True,
        "binary_content": False,
        "rich_text_content": False,
        "structured_content": True,
        "permissions": False,
        "tombstones": False,
        "remote_versions": False,
        "reconciliation": True,
    }


async def test_inspection_preserves_exact_source_without_provider_call() -> None:
    adapter = GoogleWorkspaceSheetsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    source = _source()
    info = await adapter.inspect_scope(
        integration=_integration(fake),
        source=source,
    )
    assert info.source == source
    assert info.source.connection_ref == _CONNECTION_REF
    assert info.safe_display_name == source.scope.safe_display_name
    assert info.capabilities == adapter.capabilities
    assert fake.sheets_calls == []


async def test_binding_source_and_facade_preserve_equality_boundary() -> None:
    adapter = GoogleWorkspaceSheetsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    binding_source = to_source_ref(_binding())
    assert (await adapter.inspect_scope(
        integration=_integration(fake),
        source=binding_source,
    )).source == binding_source

    integration = _integration(fake)
    resolver = _RecordingResolver(integration)
    registry = KnowledgeAdapterRegistry()
    register_google_workspace_sheets_knowledge_adapter(registry)
    facade = VendorKnowledgeFacadeService(
        tenant_id="tenant-1",
        resolver=resolver,
        adapter_registry=registry,
    )
    result = await facade.inspect_source(source=binding_source)
    assert result.source == binding_source
    assert result.source.connection_ref == _CONNECTION_REF
    assert resolver.received_sources == [binding_source]
    assert fake.sheets_calls == []


@pytest.mark.parametrize(
    "source",
    [
        _source(provider_id="other"),
        _source(integration_kind=IntegrationCategory.ISSUE_TRACKER),
        _source(source_kind="drive"),
        _source(remote_scope_type="other"),
        _source(parameters={"unexpected": "value"}),
        _source(remote_scope_id="sheet\x00main-1"),
        _source(remote_scope_id="sheet\x7fmain-1"),
        _source(remote_scope_id="sheet/main-1"),
        _source(remote_scope_id="sheet\\main-1"),
        _source(remote_scope_id="x" * 1025),
    ],
)
async def test_invalid_scopes_are_safe_and_call_no_provider(
    source: KnowledgeSourceRef,
) -> None:
    adapter = GoogleWorkspaceSheetsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await _read_page(adapter, fake, source=source)
    error = exc_info.value
    assert error.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert error.retryable is False
    assert error.safe_message == _INVALID_SCOPE_MESSAGE
    assert error.__cause__ is None
    assert fake.sheets_calls == []
    assert _SPREADSHEET_ID not in repr(error)
    assert _CONNECTION_REF not in repr(error)


async def test_malformed_source_models_and_subclasses_are_rejected() -> None:
    adapter = GoogleWorkspaceSheetsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()

    trimmed = KnowledgeSourceRef.model_construct(
        tenant_id="tenant-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_SHEETS_SOURCE_KIND,
        connection_ref=_CONNECTION_REF,
        scope=KnowledgeSourceScope.model_construct(
            remote_scope_id=" sheet-main-1",
            remote_scope_type=GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE,
            safe_display_name="Sheet",
            parameters={},
        ),
    )
    blank = KnowledgeSourceRef.model_construct(
        tenant_id="tenant-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_SHEETS_SOURCE_KIND,
        connection_ref=_CONNECTION_REF,
        scope=KnowledgeSourceScope.model_construct(
            remote_scope_id="",
            remote_scope_type=GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE,
            safe_display_name="Sheet",
            parameters={},
        ),
    )
    malformed = KnowledgeSourceRef.model_construct(
        tenant_id="tenant-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_SHEETS_SOURCE_KIND,
        connection_ref=_CONNECTION_REF,
        scope=KnowledgeSourceScope.model_construct(
            remote_scope_id=_SPREADSHEET_ID,
            remote_scope_type=GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE,
            safe_display_name=123,
            parameters={},
        ),
    )

    class _SourceSubclass(KnowledgeSourceRef):
        pass

    subclass = _SourceSubclass(
        tenant_id="tenant-1",
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_SHEETS_SOURCE_KIND,
        connection_ref=_CONNECTION_REF,
        scope=KnowledgeSourceScope(
            remote_scope_id=_SPREADSHEET_ID,
            remote_scope_type=GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE,
            safe_display_name="Sheet",
            parameters={},
        ),
    )
    for source in (trimmed, blank, malformed, subclass):
        with pytest.raises(VendorKnowledgeError) as exc_info:
            await _read_page(adapter, fake, source=source)
        assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert fake.sheets_calls == []


async def test_read_page_returns_one_structured_upsert_and_complete_checkpoint() -> None:
    spreadsheet = _representative_spreadsheet()
    fake = _FakeGoogleWorkspaceIntegration(spreadsheets=[spreadsheet])
    page = await _read_page(GoogleWorkspaceSheetsKnowledgeAdapter(), fake)
    assert fake.sheets_calls == [{"spreadsheet_id": _SPREADSHEET_ID}]
    assert len(page.changes) == 1
    change = page.changes[0]
    assert change.kind is KnowledgeChangeKind.UPSERT
    assert change.remote_id == _SPREADSHEET_ID
    assert change.descriptor is not None
    assert page.has_more is False
    assert page.next_cursor is None
    assert page.proposed_checkpoint is not None
    assert page.proposed_checkpoint.version == GOOGLE_SHEETS_CURSOR_VERSION

    descriptor = change.descriptor
    assert descriptor.identity.remote_id == _SPREADSHEET_ID
    assert descriptor.identity.parent_remote_id is None
    assert descriptor.identity.logical_key is None
    assert descriptor.title == _SPREADSHEET_TITLE
    assert descriptor.item_type == "google_workspace_sheets_spreadsheet"
    assert descriptor.content_mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert descriptor.content_available is True
    assert descriptor.revision.version is None
    assert descriptor.revision.etag is None
    assert descriptor.revision.acl_hash is None
    assert descriptor.revision.updated_at is None
    assert len(descriptor.revision.content_hash or "") == 64
    assert descriptor.provenance.provider_id == (
        GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID
    )
    assert descriptor.provenance.source_kind == GOOGLE_SHEETS_SOURCE_KIND
    assert descriptor.provenance.remote_id == _SPREADSHEET_ID
    assert descriptor.provenance.web_url is None
    assert descriptor.provenance.safe_locator is None
    assert descriptor.metadata == {
        "schema_version": GOOGLE_SHEETS_ITEM_METADATA_VERSION,
        "structured_record_schema": GOOGLE_SHEETS_STRUCTURED_RECORD_SCHEMA,
        "native_mime_type": GOOGLE_SHEETS_NATIVE_MIME_TYPE,
        "sheet_count": 2,
        "named_range_count": 1,
    }


async def test_structured_content_preserves_nested_sheet_data() -> None:
    spreadsheet = _representative_spreadsheet()
    fake = _FakeGoogleWorkspaceIntegration(spreadsheets=[spreadsheet, spreadsheet])
    adapter = GoogleWorkspaceSheetsKnowledgeAdapter()
    page = await _read_page(adapter, fake)
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=descriptor,
    )
    assert content.mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert content.mime_type == GOOGLE_SHEETS_STRUCTURED_RECORD_MIME_TYPE
    assert content.content_hash == descriptor.revision.content_hash
    record = content.structured_record
    assert record["schema_version"] == GOOGLE_SHEETS_STRUCTURED_RECORD_SCHEMA
    assert record["spreadsheet_id"] == _SPREADSHEET_ID
    assert record["title"] == _SPREADSHEET_TITLE
    assert record["locale"] == "en_US"
    assert record["time_zone"] == "America/New_York"
    assert record["recalculation_interval"] == "ON_CHANGE"
    assert len(record["sheets"]) == 2
    sheet = record["sheets"][0]
    assert sheet["title"] == _SHEET_TITLE
    cell = sheet["grid_data"][0]["rows"][0]["cells"][0]
    assert cell["row_index"] == 2
    assert cell["column_index"] == 3
    assert cell["user_entered_value"]["text"] == _CELL_TEXT
    assert cell["user_entered_value"]["kind"] == "STRING"
    formula_cell = sheet["grid_data"][0]["rows"][0]["cells"][3]
    assert formula_cell["user_entered_value"]["text"] == _FORMULA
    assert formula_cell["effective_value"]["number"] == 3.0
    assert formula_cell["formatted_value"] == "3"
    formatted_cell = sheet["grid_data"][0]["rows"][0]["cells"][4]
    assert formatted_cell["effective_number_format"]["format_type"] == "CURRENCY"
    note_cell = sheet["grid_data"][0]["rows"][0]["cells"][5]
    assert note_cell["note"] == _NOTE
    error_cell = sheet["grid_data"][0]["rows"][0]["cells"][6]
    assert error_cell["effective_value"]["error"]["error_type"] == "DIVIDE_BY_ZERO"
    assert error_cell["effective_value"]["error"]["message"] == _ERROR_MESSAGE
    assert sheet["merged_ranges"][0]["end_row_index"] == 2
    assert record["named_ranges"][0]["name"] == "BoundedRange"


async def test_content_hash_is_canonical_and_ignores_source_display_fields() -> None:
    spreadsheet = _representative_spreadsheet()
    record = _build_structured_record(spreadsheet)
    reordered = {key: record[key] for key in reversed(tuple(record))}
    assert _compute_content_hash(record) == _compute_content_hash(reordered)

    adapter = GoogleWorkspaceSheetsKnowledgeAdapter()
    fake_a = _FakeGoogleWorkspaceIntegration(spreadsheets=[spreadsheet])
    fake_b = _FakeGoogleWorkspaceIntegration(
        spreadsheets=[_representative_spreadsheet()],
    )
    page_a = await _read_page(
        adapter,
        fake_a,
        source=_source(connection_ref="connection-other", safe_display_name="Other"),
    )
    page_b = await _read_page(
        adapter,
        fake_b,
        source=_source(connection_ref="connection-third", safe_display_name="Third"),
    )
    assert page_a.changes[0].descriptor.revision.content_hash == (
        page_b.changes[0].descriptor.revision.content_hash
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"formula": "=SUM(9,9)"},
        {"merge_end_row": 3},
        {"named_range_name": "ChangedRange"},
        {"title": "Changed Spreadsheet"},
    ],
)
async def test_content_bearing_changes_change_hash(kwargs: dict[str, object]) -> None:
    base = _representative_spreadsheet()
    changed = _representative_spreadsheet(**kwargs)
    assert _compute_content_hash(_build_structured_record(base)) != (
        _compute_content_hash(_build_structured_record(changed))
    )


async def test_fetch_content_consistency_fence_fails_closed() -> None:
    original = _representative_spreadsheet()
    changed = _representative_spreadsheet(formula="=SUM(9,9)")
    fake = _FakeGoogleWorkspaceIntegration(spreadsheets=[original, changed])
    adapter = GoogleWorkspaceSheetsKnowledgeAdapter()
    page = await _read_page(adapter, fake)
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=descriptor,
        )
    error = exc_info.value
    assert error.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert error.retryable is True
    assert error.safe_message == _CONTENT_HASH_MISMATCH_MESSAGE
    assert error.__cause__ is None
    assert fake.sheets_calls == [
        {"spreadsheet_id": _SPREADSHEET_ID},
        {"spreadsheet_id": _SPREADSHEET_ID},
    ]
    rendered = repr(error)
    assert _SPREADSHEET_ID not in rendered
    assert _SPREADSHEET_TITLE not in rendered
    assert _FORMULA not in rendered


async def test_complete_cursor_is_rejected_without_provider_call() -> None:
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await _read_page(
            GoogleWorkspaceSheetsKnowledgeAdapter(),
            fake,
            cursor=_complete_cursor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert exc_info.value.safe_message == _COMPLETE_CURSOR_MESSAGE
    assert fake.sheets_calls == []


@pytest.mark.parametrize(
    "cursor",
    [
        KnowledgeCursor(value="!!!", version=GOOGLE_SHEETS_CURSOR_VERSION),
        KnowledgeCursor(value="abc=", version=GOOGLE_SHEETS_CURSOR_VERSION),
        KnowledgeCursor(value="abc$", version=GOOGLE_SHEETS_CURSOR_VERSION),
        _cursor_payload(
            {
                "schema_version": "wrong.version",
                "scope_fingerprint": _scope_fingerprint(),
                "complete": True,
            }
        ),
        _cursor_payload(
            {
                "schema_version": GOOGLE_SHEETS_CURSOR_VERSION,
                "scope_fingerprint": "deadbeef",
                "complete": True,
            }
        ),
        KnowledgeCursor(
            value=_complete_cursor().value + "A",
            version=GOOGLE_SHEETS_CURSOR_VERSION,
        ),
    ],
)
async def test_invalid_cursors_are_rejected_without_provider_call(
    cursor: KnowledgeCursor,
) -> None:
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await _read_page(
            GoogleWorkspaceSheetsKnowledgeAdapter(),
            fake,
            cursor=cursor,
        )
    error = exc_info.value
    assert error.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert error.safe_message == _INVALID_CURSOR_MESSAGE
    assert error.retryable is False
    assert error.__cause__ is None
    assert fake.sheets_calls == []
    assert _SPREADSHEET_ID not in repr(error)


async def test_cursor_is_canonical_bounded_and_contains_no_raw_scope() -> None:
    cursor = _complete_cursor()
    assert cursor.version == GOOGLE_SHEETS_CURSOR_VERSION
    assert "=" not in cursor.value
    assert len(cursor.value) <= 24_576
    assert _SPREADSHEET_ID not in cursor.value

    raw = base64.urlsafe_b64decode(cursor.value + "===")
    decoded = json.loads(raw)
    assert decoded == {
        "complete": True,
        "schema_version": GOOGLE_SHEETS_CURSOR_VERSION,
        "scope_fingerprint": _scope_fingerprint(),
    }

    class _CursorSubclass(KnowledgeCursor):
        pass

    fake = _FakeGoogleWorkspaceIntegration()
    subclass = _CursorSubclass(value=cursor.value, version=cursor.version)
    constructed = KnowledgeCursor.model_construct(
        value=cursor.value,
        version=cursor.version,
    )
    for invalid in (subclass, constructed):
        with pytest.raises(VendorKnowledgeError) as exc_info:
            await _read_page(
                GoogleWorkspaceSheetsKnowledgeAdapter(),
                fake,
                cursor=invalid,
            )
        assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert fake.sheets_calls == []


@pytest.mark.parametrize("limit", [True, 0, -1, 1001, 1.0, "50"])
async def test_invalid_limits_call_no_provider(limit: object) -> None:
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await _read_page(
            GoogleWorkspaceSheetsKnowledgeAdapter(),
            fake,
            limit=limit,  # type: ignore[arg-type]
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert exc_info.value.safe_message == _CONFIGURATION_ERROR_MESSAGE
    assert fake.sheets_calls == []


@pytest.mark.parametrize("field", ["identity", "provenance", "revision"])
async def test_malformed_descriptor_nested_models_call_no_provider(field: str) -> None:
    adapter = GoogleWorkspaceSheetsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    descriptor = _descriptor_for()
    malformed = descriptor.model_copy(
        update={
            field: {"unexpected": "malformed"},
        },
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=malformed,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert exc_info.value.safe_message == _INVALID_DESCRIPTOR_MESSAGE
    assert fake.sheets_calls == []


def _descriptor_mutations(
    descriptor: KnowledgeItemDescriptor,
) -> list[KnowledgeItemDescriptor]:
    return [
        descriptor.model_copy(
            update={
                "identity": descriptor.identity.model_copy(
                    update={"remote_id": "other"},
                ),
            }
        ),
        descriptor.model_copy(
            update={
                "identity": descriptor.identity.model_copy(
                    update={"parent_remote_id": "parent"},
                ),
            }
        ),
        descriptor.model_copy(
            update={
                "identity": descriptor.identity.model_copy(
                    update={"logical_key": "logical"},
                ),
            }
        ),
        descriptor.model_copy(
            update={
                "provenance": descriptor.provenance.model_copy(
                    update={"provider_id": "other"},
                ),
            }
        ),
        descriptor.model_copy(
            update={
                "provenance": descriptor.provenance.model_copy(
                    update={"source_kind": "drive"},
                ),
            }
        ),
        descriptor.model_copy(update={"item_type": "other"}),
        descriptor.model_copy(update={"content_mode": KnowledgeContentMode.BINARY}),
        descriptor.model_copy(update={"content_available": False}),
        descriptor.model_copy(
            update={
                "revision": descriptor.revision.model_copy(
                    update={"version": "v1"},
                ),
            }
        ),
        descriptor.model_copy(
            update={
                "revision": descriptor.revision.model_copy(
                    update={"etag": "etag"},
                ),
            }
        ),
        descriptor.model_copy(
            update={
                "revision": descriptor.revision.model_copy(
                    update={"content_hash": "bad"},
                ),
            }
        ),
        descriptor.model_copy(
            update={
                "revision": descriptor.revision.model_copy(
                    update={"acl_hash": "acl"},
                ),
            }
        ),
        descriptor.model_copy(
            update={
                "revision": descriptor.revision.model_copy(
                    update={"updated_at": "2024-01-01T00:00:00Z"},
                ),
            }
        ),
        descriptor.model_copy(
            update={
                "provenance": descriptor.provenance.model_copy(
                    update={"web_url": "https://example.com"},
                ),
            }
        ),
        descriptor.model_copy(
            update={
                "provenance": descriptor.provenance.model_copy(
                    update={"safe_locator": "locator"},
                ),
            }
        ),
        descriptor.model_copy(
            update={"metadata": {**descriptor.metadata, "unexpected": True}},
        ),
        descriptor.model_copy(
            update={
                "metadata": {
                    **descriptor.metadata,
                    "schema_version": "wrong",
                },
            }
        ),
        descriptor.model_copy(
            update={
                "metadata": {
                    **descriptor.metadata,
                    "structured_record_schema": "wrong",
                },
            }
        ),
        descriptor.model_copy(
            update={
                "metadata": {
                    **descriptor.metadata,
                    "native_mime_type": "text/plain",
                },
            }
        ),
        descriptor.model_copy(
            update={
                "metadata": {
                    **descriptor.metadata,
                    "sheet_count": "2",
                },
            }
        ),
        descriptor.model_copy(
            update={
                "metadata": {
                    **descriptor.metadata,
                    "named_range_count": True,
                },
            }
        ),
    ]


async def test_descriptor_mutations_are_rejected_before_provider_call() -> None:
    adapter = GoogleWorkspaceSheetsKnowledgeAdapter()
    fake = _FakeGoogleWorkspaceIntegration()
    descriptor = _descriptor_for()
    for mutated in _descriptor_mutations(descriptor):
        with pytest.raises(VendorKnowledgeError) as exc_info:
            await adapter.fetch_content(
                integration=_integration(fake),
                source=_source(),
                item=mutated,
            )
        assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
        assert exc_info.value.safe_message == _INVALID_DESCRIPTOR_MESSAGE
    assert fake.sheets_calls == []


async def test_fetch_permissions_is_explicitly_unsupported() -> None:
    fake = _FakeGoogleWorkspaceIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await GoogleWorkspaceSheetsKnowledgeAdapter().fetch_permissions(
            integration=_integration(fake),
            source=_source(),
            item=_descriptor_for(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert exc_info.value.retryable is False
    assert exc_info.value.safe_message == _UNSUPPORTED_PERMISSIONS_MESSAGE
    assert fake.sheets_calls == []


async def test_wrong_integration_type_is_rejected() -> None:
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await GoogleWorkspaceSheetsKnowledgeAdapter().read_page(
            integration=object(),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert exc_info.value.retryable is False
    assert exc_info.value.safe_message != ""


async def test_malformed_spreadsheet_and_id_mismatch_are_invalid_provider_response() -> None:
    adapter = GoogleWorkspaceSheetsKnowledgeAdapter()
    malformed = GoogleSheetsSpreadsheet.model_construct(
        spreadsheet_id=_SPREADSHEET_ID,
        title=_SPREADSHEET_TITLE,
    )
    wrong_id = _representative_spreadsheet(spreadsheet_id="other-sheet")
    for spreadsheet in (malformed, wrong_id):
        fake = _FakeGoogleWorkspaceIntegration(spreadsheets=[spreadsheet])
        with pytest.raises(VendorKnowledgeError) as exc_info:
            await _read_page(adapter, fake)
        assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
        assert exc_info.value.safe_message == _INVALID_PROVIDER_RESPONSE_MESSAGE


@pytest.mark.parametrize(
    ("kind", "code", "retryable"),
    [
        (
            GoogleWorkspaceErrorKind.AUTHENTICATION,
            VendorKnowledgeErrorCode.AUTHENTICATION_FAILED,
            False,
        ),
        (
            GoogleWorkspaceErrorKind.AUTHORIZATION,
            VendorKnowledgeErrorCode.AUTHORIZATION_DENIED,
            False,
        ),
        (
            GoogleWorkspaceErrorKind.NOT_FOUND,
            VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND,
            False,
        ),
        (
            GoogleWorkspaceErrorKind.RATE_LIMITED,
            VendorKnowledgeErrorCode.RATE_LIMITED,
            True,
        ),
        (
            GoogleWorkspaceErrorKind.TEMPORARY,
            VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
            True,
        ),
        (
            GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            False,
        ),
        (
            GoogleWorkspaceErrorKind.UNEXPECTED_REDIRECT,
            VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            False,
        ),
        (
            GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE,
            VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
            False,
        ),
        (
            GoogleWorkspaceErrorKind.INVALID_REQUEST,
            VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
            False,
        ),
    ],
)
async def test_google_error_mapping(
    kind: GoogleWorkspaceErrorKind,
    code: VendorKnowledgeErrorCode,
    retryable: bool,
) -> None:
    class _FailingIntegration(_FakeGoogleWorkspaceIntegration):
        def read_sheets_spreadsheet(
            self,
            *,
            spreadsheet_id: str,
        ) -> GoogleSheetsSpreadsheet:
            raise GoogleWorkspaceApiError(
                kind=kind,
                status_code=500,
                retry_after_seconds=None,
                safe_reason="private provider reason",
                attempts=1,
            )

    fake = _FailingIntegration()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await _read_page(GoogleWorkspaceSheetsKnowledgeAdapter(), fake)
    error = exc_info.value
    assert error.code is code
    assert error.retryable is retryable
    assert error.__cause__ is None
    rendered = repr(error)
    assert "private provider reason" not in rendered
    assert _SPREADSHEET_ID not in rendered
    assert _CONNECTION_REF not in rendered


@pytest.mark.parametrize(
    ("exception", "code", "retryable"),
    [
        (
            IntegrationConfigurationError("private configuration"),
            VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
            False,
        ),
        (
            IntegrationDependencyError("private dependency"),
            VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
            True,
        ),
        (
            RuntimeError("private runtime"),
            VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
            True,
        ),
    ],
)
async def test_integration_exceptions_are_safe(
    exception: Exception,
    code: VendorKnowledgeErrorCode,
    retryable: bool,
) -> None:
    class _FailingIntegration(_FakeGoogleWorkspaceIntegration):
        def read_sheets_spreadsheet(
            self,
            *,
            spreadsheet_id: str,
        ) -> GoogleSheetsSpreadsheet:
            raise exception

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await _read_page(
            GoogleWorkspaceSheetsKnowledgeAdapter(),
            _FailingIntegration(),
        )
    error = exc_info.value
    assert error.code is code
    assert error.retryable is retryable
    assert error.__cause__ is None
    assert "private" not in repr(error)
    assert _SPREADSHEET_ID not in repr(error)


async def test_registry_registration_and_package_exports() -> None:
    registry = KnowledgeAdapterRegistry()
    adapter = register_google_workspace_sheets_knowledge_adapter(registry)
    assert isinstance(adapter, GoogleWorkspaceSheetsKnowledgeAdapter)
    assert registry.resolve(source=_source()) is adapter

    import intergrax.runtime.vendor_knowledge.adapters as package

    assert package.GOOGLE_SHEETS_CURSOR_VERSION == GOOGLE_SHEETS_CURSOR_VERSION
    assert (
        package.GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE
        == GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE
    )
    assert (
        package.GOOGLE_SHEETS_ITEM_METADATA_VERSION
        == GOOGLE_SHEETS_ITEM_METADATA_VERSION
    )
    assert (
        package.GOOGLE_SHEETS_STRUCTURED_RECORD_MIME_TYPE
        == GOOGLE_SHEETS_STRUCTURED_RECORD_MIME_TYPE
    )
    assert (
        package.GOOGLE_SHEETS_STRUCTURED_RECORD_SCHEMA
        == GOOGLE_SHEETS_STRUCTURED_RECORD_SCHEMA
    )
    assert (
        package.GoogleWorkspaceSheetsKnowledgeAdapter
        is GoogleWorkspaceSheetsKnowledgeAdapter
    )
    assert (
        package.register_google_workspace_sheets_knowledge_adapter
        is register_google_workspace_sheets_knowledge_adapter
    )


async def test_public_objects_do_not_leak_private_values() -> None:
    spreadsheet = _representative_spreadsheet()
    fake = _FakeGoogleWorkspaceIntegration(spreadsheets=[spreadsheet, spreadsheet])
    adapter = GoogleWorkspaceSheetsKnowledgeAdapter()
    page = await _read_page(adapter, fake)
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
        assert _CONNECTION_REF not in blob
        assert "Authorization" not in blob
        assert "Bearer" not in blob
        assert "access_token" not in blob
        assert "refresh_token" not in blob
        assert "client_secret" not in blob
    assert _SPREADSHEET_ID not in checkpoint.value
    assert checkpoint.value not in repr(checkpoint)
    assert _SPREADSHEET_TITLE not in repr(descriptor)
    assert _SHEET_TITLE not in repr(descriptor)
    assert _FORMULA not in repr(descriptor)
    assert _CELL_TEXT not in repr(descriptor)
    assert _NOTE not in repr(descriptor)
    assert _ERROR_MESSAGE not in repr(descriptor)
    assert _SPREADSHEET_TITLE not in repr(content)
    assert _CELL_TEXT not in repr(content)
