# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""End-to-end Google Workspace Sheets durable reconciliation proof."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from dataclasses import dataclass

import pytest

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceSourceKind,
    GoogleWorkspaceTransport,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
    GoogleWorkspaceCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.sheets import (
    GOOGLE_SHEETS_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.adapters import (
    GOOGLE_SHEETS_CURSOR_VERSION,
    GOOGLE_SHEETS_ITEM_METADATA_VERSION,
    GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE,
    GOOGLE_SHEETS_STRUCTURED_RECORD_MIME_TYPE,
    GOOGLE_SHEETS_STRUCTURED_RECORD_SCHEMA,
    register_google_workspace_sheets_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
    to_source_ref,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChangeKind,
    KnowledgeContentMode,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from intergrax.runtime.vendor_knowledge.sync_coordinator import (
    VendorKnowledgeSyncCoordinator,
)
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeRemoteItemStateRepository,
    DocumentStoreKnowledgeSourceLeaseRepository,
    DocumentStoreKnowledgeSyncCheckpointRepository,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeRemoteItemStatus,
    KnowledgeSyncMode,
    KnowledgeSyncRunStatus,
)
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    IdempotentRecordingSink,
    RecordingBindingService,
    durable_reconciliation_coordinator_kwargs,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_TENANT_ID = "tenant-1"
_BINDING_ID = "google-sheets-binding"
_SPREADSHEET_ID = "sheet-proof-1"
_SPREADSHEET_TITLE = "Sheets Proof Spreadsheet"
_SHEET_TITLE = "GridSheet"
_CONNECTION_REF = "conn-google-sheets-1"

_FORMULA_V1 = "=SUM(1,2)"
_FORMULA_V2 = "=SUM(9,9)"
_CELL_TEXT = "hello"
_NOTE = "line one\nline two"
_ERROR_MESSAGE = "division by zero"

_OPERATION_INITIAL = "sheets-op-initial"
_OPERATION_UPDATE = "sheets-op-update"
_CONTENT_HASH_MISMATCH_MESSAGE = (
    "Google Workspace Sheets spreadsheet content changed since descriptor creation"
)
_SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/sheet-proof-1/edit"
_PROVIDER_REVISION = "provider-revision-sheets-1"

_SHEETS_FIELDS = (
    "spreadsheetId,"
    "properties(title,locale,timeZone,autoRecalc),"
    "sheets("
    "properties("
    "sheetId,title,index,sheetType,"
    "gridProperties("
    "rowCount,columnCount,frozenRowCount,frozenColumnCount"
    "),"
    "hidden,rightToLeft"
    "),"
    "data("
    "startRow,startColumn,"
    "rowData(values("
    "userEnteredValue,"
    "effectiveValue,"
    "formattedValue,"
    "note,"
    "effectiveFormat(numberFormat(type,pattern))"
    "))"
    "),"
    "merges"
    "),"
    "namedRanges(namedRangeId,name,range)"
)

_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")


def _representative_payload(
    *,
    formula: str = _FORMULA_V1,
    formula_result: int = 3,
    formula_formatted: str = "3",
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
            "effectiveValue": {"numberValue": formula_result},
            "formattedValue": formula_formatted,
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
            "note": _NOTE,
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
        "spreadsheetId": _SPREADSHEET_ID,
        "properties": {
            "title": _SPREADSHEET_TITLE,
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
                        "endRowIndex": 2,
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
                "name": "BoundedRange",
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


def _binding() -> KnowledgeSourceBinding:
    return KnowledgeSourceBinding(
        binding_id=_BINDING_ID,
        tenant_id=_TENANT_ID,
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_SHEETS_SOURCE_KIND,
        connection_ref=_CONNECTION_REF,
        safe_display_name="Google Sheets Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id=_SPREADSHEET_ID,
            remote_scope_type=GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE,
            safe_display_name="Sheets Proof",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
        broad_scope=False,
    )


class _DeterministicSheetsTransport:
    """Strict ordered queue transport for the real Sheets knowledge reader."""

    def __init__(self) -> None:
        self._queue: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []

    def queue_response(self, payload: dict[str, object]) -> None:
        self._queue.append(copy.deepcopy(payload))

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        call = {
            "source_kind": source_kind,
            "relative_path": relative_path,
            "params": dict(params or {}),
            "headers": dict(headers or {}),
        }
        self.calls.append(call)
        assert call["source_kind"] is GoogleWorkspaceSourceKind.SHEETS
        assert call["relative_path"] == f"/spreadsheets/{_SPREADSHEET_ID}"
        assert call["params"] == {"fields": _SHEETS_FIELDS}
        assert call["headers"] == {}
        if not self._queue:
            raise AssertionError("no queued Sheets response remains")
        return copy.deepcopy(self._queue.pop(0))


class _StubClientFamily:
    def __init__(self, transport: GoogleWorkspaceTransport) -> None:
        self.transport = transport


class _SheetsProviderScenario:
    def __init__(self) -> None:
        self.transport = _DeterministicSheetsTransport()

    @property
    def client_family(self) -> _StubClientFamily:
        return _StubClientFamily(self.transport)

    @property
    def sheets_calls(self) -> list[dict[str, object]]:
        return self.transport.calls

    def queue_response(self, payload: dict[str, object]) -> None:
        self.transport.queue_response(payload)

    def queue_descriptor_content_pair(
        self,
        payload: dict[str, object],
    ) -> None:
        self.queue_response(payload)
        self.queue_response(payload)


class _RecordingResolver:
    def __init__(
        self,
        integration: GoogleWorkspaceCollaborationSuiteIntegration,
    ) -> None:
        self._integration = integration
        self.received_sources: list[KnowledgeSourceRef] = []

    def resolve(
        self,
        *,
        source: KnowledgeSourceRef,
    ) -> GoogleWorkspaceCollaborationSuiteIntegration:
        self.received_sources.append(source)
        return self._integration


@dataclass
class _RuntimeBundle:
    coordinator: VendorKnowledgeSyncCoordinator
    resolver: _RecordingResolver
    integration: GoogleWorkspaceCollaborationSuiteIntegration


def _build_runtime(
    *,
    scenario: _SheetsProviderScenario,
    document_store: InMemoryDocumentStore,
    sink: IdempotentRecordingSink,
    binding: KnowledgeSourceBinding,
    owner_id: str,
) -> _RuntimeBundle:
    integration = GoogleWorkspaceCollaborationSuiteIntegration.from_client(
        scenario.client_family,
        enabled=True,
    )
    resolver = _RecordingResolver(integration=integration)
    registry = KnowledgeAdapterRegistry()
    register_google_workspace_sheets_knowledge_adapter(registry)
    facade = VendorKnowledgeFacadeService(
        tenant_id=_TENANT_ID,
        resolver=resolver,
        adapter_registry=registry,
    )
    lease_repo = DocumentStoreKnowledgeSourceLeaseRepository(document_store)
    checkpoint_repo = DocumentStoreKnowledgeSyncCheckpointRepository(document_store)
    state_repo = DocumentStoreKnowledgeRemoteItemStateRepository(document_store)
    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id=_TENANT_ID,
        owner_id=owner_id,
        binding_service=RecordingBindingService(binding=binding),  # type: ignore[arg-type]
        facade=facade,
        lease_repository=lease_repo,
        checkpoint_repository=checkpoint_repo,
        item_state_repository=state_repo,
        sink=sink,
        lease_ttl_seconds=30,
        **durable_reconciliation_coordinator_kwargs(
            state_repository=state_repo,
            document_store=document_store,
        ),
    )
    return _RuntimeBundle(
        coordinator=coordinator,
        resolver=resolver,
        integration=integration,
    )


def _fresh_checkpoint(
    document_store: InMemoryDocumentStore,
) -> object:
    return DocumentStoreKnowledgeSyncCheckpointRepository(document_store).get(
        tenant_id=_TENANT_ID,
        binding_id=_BINDING_ID,
    )


def _fresh_state(
    document_store: InMemoryDocumentStore,
) -> object:
    return DocumentStoreKnowledgeRemoteItemStateRepository(document_store).get(
        tenant_id=_TENANT_ID,
        binding_id=_BINDING_ID,
        remote_id=_SPREADSHEET_ID,
    )


def _envelope_for(batch: object) -> object:
    return next(
        envelope
        for envelope in batch.envelopes  # type: ignore[attr-defined]
        if envelope.remote_id == _SPREADSHEET_ID
    )


def _decode_cursor(value: str) -> dict[str, object]:
    padding = "=" * (-len(value) % 4)
    raw = base64.urlsafe_b64decode(value + padding)
    return json.loads(raw.decode("utf-8"))


def _scope_fingerprint() -> str:
    return hashlib.sha256(
        f"google_workspace\x00sheets\x00{_SPREADSHEET_ID}".encode("utf-8")
    ).hexdigest()


def _public_blob(value: object) -> str:
    return json.dumps(value, default=str)


def _assert_descriptor_contract(descriptor: object, *, content_hash: str) -> None:
    assert descriptor.identity.remote_id == _SPREADSHEET_ID  # type: ignore[attr-defined]
    assert descriptor.provenance.remote_id == _SPREADSHEET_ID  # type: ignore[attr-defined]
    assert descriptor.item_type == "google_workspace_sheets_spreadsheet"  # type: ignore[attr-defined]
    assert descriptor.content_mode is KnowledgeContentMode.STRUCTURED_RECORD  # type: ignore[attr-defined]
    assert descriptor.content_available is True  # type: ignore[attr-defined]
    assert descriptor.revision.content_hash == content_hash  # type: ignore[attr-defined]
    assert _SHA256_HEX_RE.fullmatch(content_hash)
    assert descriptor.revision.version is None  # type: ignore[attr-defined]
    assert descriptor.revision.etag is None  # type: ignore[attr-defined]
    assert descriptor.revision.acl_hash is None  # type: ignore[attr-defined]
    assert descriptor.revision.updated_at is None  # type: ignore[attr-defined]
    assert descriptor.provenance.web_url is None  # type: ignore[attr-defined]
    assert descriptor.provenance.safe_locator is None  # type: ignore[attr-defined]
    assert descriptor.metadata == {
        "schema_version": GOOGLE_SHEETS_ITEM_METADATA_VERSION,
        "structured_record_schema": GOOGLE_SHEETS_STRUCTURED_RECORD_SCHEMA,
        "native_mime_type": "application/vnd.google-apps.spreadsheet",
        "sheet_count": 2,
        "named_range_count": 1,
    }


def _assert_structured_record(
    record: dict[str, object],
    *,
    formula: str,
    formula_result: float,
    formula_formatted: str,
) -> None:
    assert record["schema_version"] == GOOGLE_SHEETS_STRUCTURED_RECORD_SCHEMA
    assert record["spreadsheet_id"] == _SPREADSHEET_ID
    assert record["title"] == _SPREADSHEET_TITLE
    assert record["locale"] == "en_US"
    assert record["time_zone"] == "America/New_York"
    assert record["recalculation_interval"] == "ON_CHANGE"

    sheets = record["sheets"]
    assert isinstance(sheets, list)
    assert len(sheets) == 2
    grid_sheet = sheets[0]
    assert grid_sheet["sheet_id"] == 100
    assert grid_sheet["title"] == _SHEET_TITLE
    assert grid_sheet["index"] == 0
    assert grid_sheet["sheet_type"] == "GRID"
    assert grid_sheet["row_count"] == 10
    assert grid_sheet["column_count"] == 10
    assert grid_sheet["frozen_row_count"] == 1
    assert grid_sheet["frozen_column_count"] == 1

    grid_data = grid_sheet["grid_data"]
    assert grid_data[0]["start_row_index"] == 2
    assert grid_data[0]["start_column_index"] == 3
    cells = grid_data[0]["rows"][0]["cells"]
    assert cells[0]["row_index"] == 2
    assert cells[0]["column_index"] == 3
    assert cells[0]["user_entered_value"]["text"] == _CELL_TEXT
    assert cells[0]["effective_value"]["text"] == _CELL_TEXT
    assert cells[1]["effective_value"]["number"] == 42.0
    assert cells[2]["effective_value"]["boolean"] is True

    formula_cell = cells[3]
    assert formula_cell["user_entered_value"]["text"] == formula
    assert formula_cell["effective_value"]["number"] == formula_result
    assert formula_cell["formatted_value"] == formula_formatted

    formatted_cell = cells[4]
    assert formatted_cell["effective_number_format"]["format_type"] == "CURRENCY"
    assert formatted_cell["effective_number_format"]["pattern"] == "$#,##0.00"
    note_cell = cells[5]
    assert note_cell["note"] == _NOTE
    error_cell = cells[6]
    assert error_cell["effective_value"]["error"]["error_type"] == "DIVIDE_BY_ZERO"
    assert error_cell["effective_value"]["error"]["message"] == _ERROR_MESSAGE

    assert grid_sheet["merged_ranges"][0] == {
        "sheet_id": 100,
        "start_row_index": 0,
        "end_row_index": 2,
        "start_column_index": 0,
        "end_column_index": 2,
    }
    assert sheets[1]["sheet_type"] == "OBJECT"
    assert record["named_ranges"][0]["name"] == "BoundedRange"
    assert record["named_ranges"][0]["grid_range"]["sheet_id"] == 100
    assert record["named_ranges"][0]["grid_range"]["end_row_index"] == 3


def _assert_no_private_data(blob: str, raw_payload: dict[str, object]) -> None:
    forbidden = (
        _CONNECTION_REF,
        "Authorization",
        "Bearer",
        "access_token",
        "refresh_token",
        "client_secret",
        "credential_ref",
        _SPREADSHEET_URL,
        _PROVIDER_REVISION,
        "revisionId",
    )
    for value in forbidden:
        assert value not in blob
    assert "bearer" not in blob.lower()
    assert _SPREADSHEET_ID not in _decode_cursor(
        json.loads(blob)["checkpoint"]["cursor"]["value"]
    ).get("scope_fingerprint", "")
    assert "spreadsheetId" not in blob
    assert json.dumps(raw_payload, sort_keys=True) not in blob


def _assert_no_fence_leak(rendered: str) -> None:
    for value in (
        _SPREADSHEET_ID,
        _SPREADSHEET_TITLE,
        _SHEET_TITLE,
        _FORMULA_V1,
        _FORMULA_V2,
        _CELL_TEXT,
        _NOTE,
        _ERROR_MESSAGE,
        _CONNECTION_REF,
    ):
        assert value not in rendered


@pytest.mark.asyncio
async def test_google_sheets_facade_coordinator_durable_restart_update_and_fence() -> None:
    scenario = _SheetsProviderScenario()
    document_store = InMemoryDocumentStore()
    sink = IdempotentRecordingSink()
    binding = _binding()
    expected_source = to_source_ref(binding)
    payload_v1 = _representative_payload()
    payload_v2 = _representative_payload(
        formula=_FORMULA_V2,
        formula_result=18,
        formula_formatted="18",
    )
    scenario.queue_descriptor_content_pair(payload_v1)

    runtime_a = _build_runtime(
        scenario=scenario,
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-runtime-a",
    )
    assert isinstance(runtime_a.integration, GoogleWorkspaceCollaborationSuiteIntegration)
    phase_a = await runtime_a.coordinator.reconcile_once(
        binding_id=_BINDING_ID,
        restart=True,
        operation_id=_OPERATION_INITIAL,
    )
    assert all(source == expected_source for source in runtime_a.resolver.received_sources)
    assert all(
        source.connection_ref == _CONNECTION_REF
        for source in runtime_a.resolver.received_sources
    )
    assert runtime_a.resolver.received_sources

    assert phase_a.status is KnowledgeSyncRunStatus.COMPLETED
    assert phase_a.mode is KnowledgeSyncMode.RECONCILIATION
    assert phase_a.has_more is False
    assert phase_a.checkpoint_advanced is True
    assert phase_a.tombstone_count == 0
    assert phase_a.delivery_id is not None
    assert len(scenario.sheets_calls) == 2
    assert len(sink.calls) == 1
    assert len(sink.durable_delivery_ids) == 1

    batch_1 = sink.calls[0]
    assert len(batch_1.envelopes) == 1
    assert batch_1.envelopes[0].change_kind is KnowledgeChangeKind.UPSERT
    envelope_1 = _envelope_for(batch_1)
    assert envelope_1.descriptor is not None
    assert envelope_1.content is not None
    assert batch_1.source == expected_source
    assert sink.durable_delivery_ids == [phase_a.delivery_id]

    hash_v1 = envelope_1.descriptor.revision.content_hash
    _assert_descriptor_contract(envelope_1.descriptor, content_hash=hash_v1)
    assert envelope_1.content.mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert envelope_1.content.mime_type == GOOGLE_SHEETS_STRUCTURED_RECORD_MIME_TYPE
    assert envelope_1.content.content_hash == hash_v1
    _assert_structured_record(
        envelope_1.content.structured_record,
        formula=_FORMULA_V1,
        formula_result=3.0,
        formula_formatted="3",
    )

    checkpoint_a = _fresh_checkpoint(document_store)
    assert checkpoint_a is not None
    assert checkpoint_a.binding_configuration_version == 1
    assert checkpoint_a.cursor.version == GOOGLE_SHEETS_CURSOR_VERSION
    assert _SPREADSHEET_ID not in checkpoint_a.cursor.value
    assert _decode_cursor(checkpoint_a.cursor.value) == {
        "schema_version": GOOGLE_SHEETS_CURSOR_VERSION,
        "scope_fingerprint": _scope_fingerprint(),
        "complete": True,
    }
    state_a = _fresh_state(document_store)
    assert state_a is not None
    assert state_a.status is KnowledgeRemoteItemStatus.ACTIVE
    assert state_a.remote_id == _SPREADSHEET_ID
    assert state_a.revision.content_hash == hash_v1
    assert state_a.last_delivery_id == phase_a.delivery_id

    # Phase B — fresh-runtime terminal replay.
    calls_after_a = len(scenario.sheets_calls)
    sink_calls_after_a = len(sink.calls)
    deliveries_after_a = list(sink.durable_delivery_ids)
    runtime_b = _build_runtime(
        scenario=scenario,
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-runtime-b",
    )
    phase_b = await runtime_b.coordinator.reconcile_once(
        binding_id=_BINDING_ID,
        restart=True,
        operation_id=_OPERATION_INITIAL,
    )
    assert runtime_b.resolver.received_sources == []
    assert phase_b == phase_a
    assert len(scenario.sheets_calls) == calls_after_a
    assert len(sink.calls) == sink_calls_after_a
    assert sink.durable_delivery_ids == deliveries_after_a
    assert _fresh_checkpoint(document_store) == checkpoint_a
    assert _fresh_state(document_store) == state_a

    # Phase C — fresh-runtime reconciliation after a content-bearing change.
    scenario.queue_descriptor_content_pair(payload_v2)
    runtime_c = _build_runtime(
        scenario=scenario,
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-runtime-c",
    )
    phase_c = await runtime_c.coordinator.reconcile_once(
        binding_id=_BINDING_ID,
        restart=True,
        operation_id=_OPERATION_UPDATE,
    )
    assert phase_c.status is KnowledgeSyncRunStatus.COMPLETED
    assert phase_c.tombstone_count == 0
    assert phase_c.delivery_id is not None
    assert phase_c.delivery_id != phase_a.delivery_id
    assert len(scenario.sheets_calls) == 4
    assert len(sink.calls) == 2
    assert sink.durable_delivery_ids == [phase_a.delivery_id, phase_c.delivery_id]

    envelope_2 = _envelope_for(sink.calls[1])
    assert envelope_2.change_kind is KnowledgeChangeKind.UPSERT
    hash_v2 = envelope_2.descriptor.revision.content_hash
    assert hash_v2 != hash_v1
    assert envelope_2.content.content_hash == hash_v2
    _assert_structured_record(
        envelope_2.content.structured_record,
        formula=_FORMULA_V2,
        formula_result=18.0,
        formula_formatted="18",
    )
    assert envelope_2.descriptor.identity.remote_id == _SPREADSHEET_ID
    assert envelope_2.descriptor.provenance.remote_id == _SPREADSHEET_ID
    assert envelope_2.descriptor.item_type == envelope_1.descriptor.item_type

    state_c = _fresh_state(document_store)
    assert state_c is not None
    assert state_c.status is KnowledgeRemoteItemStatus.ACTIVE
    assert state_c.remote_id == _SPREADSHEET_ID
    assert state_c.revision.content_hash == hash_v2
    assert state_c.last_delivery_id == phase_c.delivery_id
    checkpoint_c = _fresh_checkpoint(document_store)
    assert checkpoint_c == checkpoint_a

    # Phase D — Sheets does not reinterpret a complete reconciliation cursor as
    # an incremental cursor and does not mutate durable state.
    calls_before_d = len(scenario.sheets_calls)
    sink_calls_before_d = len(sink.calls)
    deliveries_before_d = list(sink.durable_delivery_ids)
    runtime_d = _build_runtime(
        scenario=scenario,
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-runtime-d",
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await runtime_d.coordinator.sync_once(binding_id=_BINDING_ID)
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert exc_info.value.retryable is False
    assert len(scenario.sheets_calls) == calls_before_d
    assert len(sink.calls) == sink_calls_before_d
    assert sink.durable_delivery_ids == deliveries_before_d
    assert _fresh_checkpoint(document_store) == checkpoint_c
    assert _fresh_state(document_store) == state_c

    durable_blob = _public_blob(
        {
            "results": [
                phase_a.model_dump(mode="json"),
                phase_b.model_dump(mode="json"),
                phase_c.model_dump(mode="json"),
            ],
            "descriptor": envelope_2.descriptor.model_dump(mode="json"),
            "content": envelope_2.content.model_dump(mode="json"),
            "checkpoint": checkpoint_c.model_dump(mode="json"),
            "state": state_c.model_dump(mode="json"),
            "delivery_id": phase_c.delivery_id,
        }
    )
    _assert_no_private_data(durable_blob, payload_v2)
    assert _CONNECTION_REF not in durable_blob
    assert _CONNECTION_REF not in json.dumps(
        envelope_2.descriptor.model_dump(mode="json")
    )
    assert _CONNECTION_REF not in json.dumps(envelope_2.content.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_google_sheets_descriptor_content_race_fails_closed_before_durable_write() -> None:
    scenario = _SheetsProviderScenario()
    document_store = InMemoryDocumentStore()
    sink = IdempotentRecordingSink()
    binding = _binding()
    scenario.queue_response(_representative_payload())
    scenario.queue_response(
        _representative_payload(
            formula=_FORMULA_V2,
            formula_result=18,
            formula_formatted="18",
        )
    )

    runtime = _build_runtime(
        scenario=scenario,
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-fence",
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await runtime.coordinator.reconcile_once(
            binding_id=_BINDING_ID,
            restart=True,
            operation_id="sheets-content-fence",
        )

    error = exc_info.value
    assert error.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert error.retryable is True
    assert error.safe_message == _CONTENT_HASH_MISMATCH_MESSAGE
    assert error.__cause__ is None
    _assert_no_fence_leak(f"{error!r} {error.safe_message}")

    assert len(scenario.sheets_calls) == 2
    assert len(sink.calls) == 0
    assert sink.durable_delivery_ids == []
    assert _fresh_checkpoint(document_store) is None
    assert _fresh_state(document_store) is None
