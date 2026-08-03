# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""End-to-end Google Workspace Docs knowledge adapter proof through facade and coordinator."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

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
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.docs import (
    GOOGLE_DOCS_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.adapters import (
    GOOGLE_DOCS_CURSOR_VERSION,
    GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE,
    GOOGLE_DOCS_ITEM_METADATA_VERSION,
    GOOGLE_DOCS_STRUCTURED_RECORD_MIME_TYPE,
    GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA,
    register_google_workspace_docs_knowledge_adapter,
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
_BINDING_ID = "google-docs-binding"
_DOCUMENT_ID = "doc-proof-1"
_CONNECTION_REF = "conn-google-1"
_DOCUMENT_TITLE = "Docs Proof Document"
_BODY_TEXT_V1 = "Body text"
_BODY_TEXT_V2 = "Body text v2"

_OPERATION_V1 = "google-docs-proof-v1"
_OPERATION_V2 = "google-docs-proof-v2"

_DESCRIPTOR_REVISION_V1 = "descriptor-revision-v1"
_CONTENT_REVISION_V1 = "content-revision-v1"
_DESCRIPTOR_REVISION_V2 = "descriptor-revision-v2"
_CONTENT_REVISION_V2 = "content-revision-v2"

_RICH_LINK_URI = "https://example.com/secret-docs-uri"

_CONTENT_HASH_MISMATCH_MESSAGE = (
    "Google Workspace Docs document content changed since descriptor creation"
)

_FORBIDDEN_RECORD_KEYS = frozenset(
    {
        "plain_text",
        "html",
        "markdown",
        "raw_payload",
        "revision_id",
        "revisionId",
        "uri",
    }
)


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
    revision_id: str | None = None,
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


def _representative_payload(*, body_text: str = _BODY_TEXT_V1) -> dict[str, object]:
    inline_elements: list[dict[str, object]] = [
        _text_run(body_text, 1, 10),
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
                    "uri": _RICH_LINK_URI,
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


def _scope_fingerprint(document_id: str) -> str:
    payload = f"google_workspace\x00docs\x00{document_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decode_docs_cursor_payload(cursor_value: str) -> dict[str, Any]:
    padding = "=" * (-len(cursor_value) % 4)
    raw = base64.urlsafe_b64decode(cursor_value + padding)
    return json.loads(raw.decode("utf-8"))


def _binding() -> KnowledgeSourceBinding:
    return KnowledgeSourceBinding(
        binding_id=_BINDING_ID,
        tenant_id=_TENANT_ID,
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=GOOGLE_DOCS_SOURCE_KIND,
        connection_ref=_CONNECTION_REF,
        safe_display_name="Google Docs Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id=_DOCUMENT_ID,
            remote_scope_type=GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE,
            safe_display_name="Docs Proof",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
        broad_scope=False,
    )


class _DeterministicDocsTransport:
    """Strict ordered queue transport for real GoogleDocsKnowledgeReader."""

    def __init__(self, scenario: _GoogleDocsProviderScenario) -> None:
        self._scenario = scenario
        self.calls: list[dict[str, object]] = []

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "source_kind": source_kind,
                "relative_path": relative_path,
                "params": dict(params or {}),
                "headers": dict(headers or {}),
            }
        )
        call = self.calls[-1]
        assert call["source_kind"] is GoogleWorkspaceSourceKind.DOCS
        assert call["relative_path"] == f"/documents/{_DOCUMENT_ID}"
        assert call["params"] == {
            "includeTabsContent": True,
            "suggestionsViewMode": "PREVIEW_WITHOUT_SUGGESTIONS",
        }
        header_map = call["headers"]
        assert "Authorization" not in header_map
        assert "authorization" not in header_map
        if not self._scenario._queue:
            raise AssertionError("no queued Docs response remains")
        return self._scenario._queue.pop(0)


class _StubClientFamily:
    def __init__(self, transport: GoogleWorkspaceTransport) -> None:
        self.transport = transport


class _GoogleDocsProviderScenario:
    """Deterministic provider scenario with strict call-order enforcement."""

    def __init__(self) -> None:
        self._queue: list[dict[str, object]] = []
        self.transport = _DeterministicDocsTransport(self)

    @property
    def client_family(self) -> _StubClientFamily:
        return _StubClientFamily(transport=self.transport)

    @property
    def docs_calls(self) -> list[dict[str, object]]:
        return self.transport.calls

    def queue_response(self, payload: dict[str, object]) -> None:
        self._queue.append(copy.deepcopy(payload))

    def queue_descriptor_content_pair(
        self,
        base_payload: dict[str, object],
        *,
        descriptor_revision: str,
        content_revision: str,
    ) -> None:
        descriptor_payload = copy.deepcopy(base_payload)
        descriptor_payload["revisionId"] = descriptor_revision
        content_payload = copy.deepcopy(base_payload)
        content_payload["revisionId"] = content_revision
        self.queue_response(descriptor_payload)
        self.queue_response(content_payload)


class _RecordingResolver:
    def __init__(
        self,
        integration: GoogleWorkspaceCollaborationSuiteIntegration,
    ) -> None:
        self._integration = integration
        self.received_sources: list[KnowledgeSourceRef] = []

    def resolve(self, *, source: KnowledgeSourceRef) -> GoogleWorkspaceCollaborationSuiteIntegration:
        self.received_sources.append(source)
        return self._integration


@dataclass
class _RuntimeBundle:
    coordinator: VendorKnowledgeSyncCoordinator
    resolver: _RecordingResolver
    integration: GoogleWorkspaceCollaborationSuiteIntegration


def _build_runtime(
    *,
    scenario: _GoogleDocsProviderScenario,
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
    register_google_workspace_docs_knowledge_adapter(registry)
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


def _fresh_checkpoint_repo(
    document_store: InMemoryDocumentStore,
) -> DocumentStoreKnowledgeSyncCheckpointRepository:
    return DocumentStoreKnowledgeSyncCheckpointRepository(document_store)


def _fresh_state_repo(
    document_store: InMemoryDocumentStore,
) -> DocumentStoreKnowledgeRemoteItemStateRepository:
    return DocumentStoreKnowledgeRemoteItemStateRepository(document_store)


def _envelope_for(batch: object, remote_id: str) -> object:
    return next(
        envelope for envelope in batch.envelopes if envelope.remote_id == remote_id  # type: ignore[attr-defined]
    )


def _public_blob(value: object) -> str:
    return json.dumps(value, default=str)


def _assert_no_public_secrets(blob: str) -> None:
    forbidden = (
        _DESCRIPTOR_REVISION_V1,
        _CONTENT_REVISION_V1,
        _DESCRIPTOR_REVISION_V2,
        _CONTENT_REVISION_V2,
        _RICH_LINK_URI,
        "Authorization",
        "Bearer",
        "access_token",
        "refresh_token",
        "client_secret",
        "x-goog-api-key",
        "credential_ref",
    )
    for item in forbidden:
        assert item not in blob
    assert "bearer" not in blob.lower()


def _assert_connection_ref_scope(blob: str, *, descriptor: object, content: object, cursor: object) -> None:
    descriptor_blob = json.dumps(descriptor.model_dump(mode="json"))  # type: ignore[attr-defined]
    content_blob = json.dumps(content.model_dump(mode="json"))  # type: ignore[attr-defined]
    cursor_blob = json.dumps(cursor.model_dump(mode="json"))  # type: ignore[attr-defined]
    assert _CONNECTION_REF not in descriptor_blob
    assert _CONNECTION_REF not in content_blob
    assert _CONNECTION_REF not in cursor_blob
    provenance = descriptor.provenance  # type: ignore[attr-defined]
    assert _CONNECTION_REF not in json.dumps(provenance.model_dump(mode="json"))


def _assert_sensitive_repr(
    descriptor: object,
    content: object,
    cursor: object,
) -> None:
    assert _DOCUMENT_TITLE not in repr(descriptor)
    record = content.structured_record  # type: ignore[attr-defined]
    if record is not None:
        assert json.dumps(record) not in repr(content)
    assert cursor.value not in repr(cursor)  # type: ignore[attr-defined]


def _collect_forbidden_keys(obj: object, found: set[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _FORBIDDEN_RECORD_KEYS:
                found.add(key)
            _collect_forbidden_keys(value, found)
    elif isinstance(obj, list):
        for item in obj:
            _collect_forbidden_keys(item, found)


def _assert_structured_record_shape(
    record: dict[str, object],
    *,
    body_text: str,
) -> None:
    forbidden: set[str] = set()
    _collect_forbidden_keys(record, forbidden)
    assert not forbidden

    assert record["schema_version"] == GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA
    assert record["document_id"] == _DOCUMENT_ID
    assert record["title"] == _DOCUMENT_TITLE
    assert record["suggestions_view_mode"] == "PREVIEW_WITHOUT_SUGGESTIONS"

    tabs = record["tabs"]
    assert isinstance(tabs, list)
    assert [tab["tab_id"] for tab in tabs] == ["tab-root", "tab-child", "tab-root-2"]
    assert tabs[1]["parent_tab_id"] == "tab-root"

    root_tab = tabs[0]
    segment_kinds = [segment["kind"] for segment in root_tab["segments"]]
    assert segment_kinds[0] == "BODY"
    assert "HEADER" in segment_kinds
    assert "FOOTNOTE" in segment_kinds
    assert segment_kinds.index("BODY") < segment_kinds.index("HEADER")
    assert segment_kinds.index("HEADER") < segment_kinds.index("FOOTNOTE")

    body_segment = root_tab["segments"][0]
    paragraph_block = body_segment["blocks"][0]
    assert paragraph_block["kind"] == "PARAGRAPH"
    elements = paragraph_block["paragraph"]["elements"]
    text_run = next(element for element in elements if element["kind"] == "TEXT_RUN")
    assert text_run["text"] == body_text

    footnote_ref = next(
        element for element in elements if element["kind"] == "FOOTNOTE_REFERENCE"
    )
    assert footnote_ref["reference_id"] == "fn-1"
    assert footnote_ref["text"] == "1"

    person = next(element for element in elements if element["kind"] == "PERSON")
    assert person["reference_id"] == "person-1"
    assert person["text"] == "User Name"
    assert person["auxiliary_text"] == "user@example.com"

    rich_link = next(element for element in elements if element["kind"] == "RICH_LINK")
    assert rich_link["reference_id"] == "rich-1"
    assert rich_link["text"] == "Example"
    assert rich_link["mime_type"] == "text/html"

    date_elem = next(element for element in elements if element["kind"] == "DATE")
    assert date_elem["reference_id"] == "date-1"
    assert date_elem["text"] == "Jan 1"
    assert date_elem["auxiliary_text"] == "2024-01-01T00:00:00Z"

    bullet_block = root_tab["segments"][0]["blocks"][1]
    assert bullet_block["kind"] == "PARAGRAPH"
    bullet = bullet_block["paragraph"]["bullet"]
    assert bullet["list_id"] == "list-1"
    assert bullet["nesting_level"] == 0

    table_block = root_tab["segments"][0]["blocks"][2]
    assert table_block["kind"] == "TABLE"
    cell = table_block["table"]["table_rows"][0]["cells"][0]
    assert cell["column_span"] == 2
    assert cell["row_span"] == 1


def _assert_descriptor_contract(descriptor: object, *, content_hash: str) -> None:
    assert descriptor.identity.remote_id == _DOCUMENT_ID  # type: ignore[attr-defined]
    assert descriptor.item_type == "google_workspace_docs_document"  # type: ignore[attr-defined]
    assert descriptor.content_mode is KnowledgeContentMode.STRUCTURED_RECORD  # type: ignore[attr-defined]
    assert descriptor.content_available is True  # type: ignore[attr-defined]
    assert descriptor.revision.version is None  # type: ignore[attr-defined]
    assert descriptor.revision.etag is None  # type: ignore[attr-defined]
    assert descriptor.revision.acl_hash is None  # type: ignore[attr-defined]
    assert descriptor.revision.updated_at is None  # type: ignore[attr-defined]
    assert descriptor.revision.content_hash == content_hash  # type: ignore[attr-defined]
    assert _SHA256_HEX_RE.match(content_hash)
    metadata = descriptor.metadata  # type: ignore[attr-defined]
    assert metadata == {
        "schema_version": GOOGLE_DOCS_ITEM_METADATA_VERSION,
        "structured_record_schema": GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA,
        "native_mime_type": "application/vnd.google-apps.document",
        "tab_count": 3,
    }
    dumped = json.dumps(descriptor.model_dump(mode="json"))  # type: ignore[attr-defined]
    assert "revisionId" not in dumped
    assert "revision_id" not in dumped


_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")


def _assert_no_fence_leak(rendered: str) -> None:
    assert _DOCUMENT_ID not in rendered
    assert _DOCUMENT_TITLE not in rendered
    assert _BODY_TEXT_V1 not in rendered
    assert _BODY_TEXT_V2 not in rendered
    assert _DESCRIPTOR_REVISION_V1 not in rendered
    assert _CONTENT_REVISION_V1 not in rendered
    assert _RICH_LINK_URI not in rendered
    assert _CONNECTION_REF not in rendered


@pytest.mark.asyncio
async def test_google_docs_facade_coordinator_restart_update_and_structured_content() -> None:
    scenario = _GoogleDocsProviderScenario()
    document_store = InMemoryDocumentStore()
    sink = IdempotentRecordingSink()
    binding = _binding()
    expected_source = to_source_ref(binding)

    payload_v1 = _representative_payload(body_text=_BODY_TEXT_V1)
    scenario.queue_descriptor_content_pair(
        payload_v1,
        descriptor_revision=_DESCRIPTOR_REVISION_V1,
        content_revision=_CONTENT_REVISION_V1,
    )

    # Phase A — initial durable reconciliation
    runtime_a = _build_runtime(
        scenario=scenario,
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-runtime-a",
    )
    phase_a = await runtime_a.coordinator.reconcile_once(
        binding_id=_BINDING_ID,
        restart=True,
        operation_id=_OPERATION_V1,
    )
    assert all(source == expected_source for source in runtime_a.resolver.received_sources)
    assert all(
        source.connection_ref == _CONNECTION_REF
        for source in runtime_a.resolver.received_sources
    )
    assert all(
        source.scope.remote_scope_id == _DOCUMENT_ID
        for source in runtime_a.resolver.received_sources
    )
    sources_snapshot_a = list(runtime_a.resolver.received_sources)
    assert sources_snapshot_a
    assert all(
        runtime_a.resolver.resolve(source=source) is runtime_a.integration
        for source in sources_snapshot_a
    )
    del runtime_a

    assert phase_a.status is KnowledgeSyncRunStatus.COMPLETED
    assert phase_a.mode is KnowledgeSyncMode.RECONCILIATION
    assert phase_a.has_more is False
    assert phase_a.checkpoint_advanced is True
    assert phase_a.tombstone_count == 0
    assert phase_a.delivery_id is not None

    assert len(scenario.docs_calls) == 2
    assert all(call["relative_path"] == f"/documents/{_DOCUMENT_ID}" for call in scenario.docs_calls)

    batch_1 = sink.calls[0]
    assert len(batch_1.envelopes) == 1
    assert batch_1.tenant_id == _TENANT_ID
    assert batch_1.binding_id == _BINDING_ID
    assert batch_1.source.connection_ref == _CONNECTION_REF
    envelope_1 = _envelope_for(batch_1, _DOCUMENT_ID)
    assert envelope_1.change_kind is KnowledgeChangeKind.UPSERT
    assert envelope_1.permissions is None
    assert envelope_1.descriptor is not None
    assert envelope_1.content is not None

    hash_v1 = envelope_1.descriptor.revision.content_hash
    _assert_descriptor_contract(envelope_1.descriptor, content_hash=hash_v1)
    assert envelope_1.content.mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert envelope_1.content.mime_type == GOOGLE_DOCS_STRUCTURED_RECORD_MIME_TYPE
    assert envelope_1.content.content_hash == hash_v1
    _assert_structured_record_shape(
        envelope_1.content.structured_record,
        body_text=_BODY_TEXT_V1,
    )

    checkpoint_repo_a = _fresh_checkpoint_repo(document_store)
    checkpoint_a = checkpoint_repo_a.get(tenant_id=_TENANT_ID, binding_id=_BINDING_ID)
    assert checkpoint_a is not None
    assert checkpoint_a.tenant_id == _TENANT_ID
    assert checkpoint_a.binding_id == _BINDING_ID
    assert checkpoint_a.binding_configuration_version == 1
    assert checkpoint_a.cursor.version == GOOGLE_DOCS_CURSOR_VERSION
    assert _DOCUMENT_ID not in checkpoint_a.cursor.value
    decoded_cursor_a = _decode_docs_cursor_payload(checkpoint_a.cursor.value)
    assert decoded_cursor_a == {
        "schema_version": GOOGLE_DOCS_CURSOR_VERSION,
        "scope_fingerprint": _scope_fingerprint(_DOCUMENT_ID),
        "complete": True,
    }

    state_repo_a = _fresh_state_repo(document_store)
    state_a = state_repo_a.get(
        tenant_id=_TENANT_ID,
        binding_id=_BINDING_ID,
        remote_id=_DOCUMENT_ID,
    )
    assert state_a is not None
    assert state_a.status is KnowledgeRemoteItemStatus.ACTIVE
    assert state_a.revision is not None
    assert state_a.revision.version is None
    assert state_a.revision.content_hash == hash_v1
    assert state_a.last_delivery_id == phase_a.delivery_id

    delivery_id_v1 = phase_a.delivery_id
    assert len(sink.durable_delivery_ids) == 1

    # Phase B — completed-run runtime replay
    docs_calls_after_a = len(scenario.docs_calls)
    sink_calls_after_a = len(sink.calls)
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
        operation_id=_OPERATION_V1,
    )
    assert runtime_b.resolver.received_sources == []
    del runtime_b

    assert phase_b.status is KnowledgeSyncRunStatus.COMPLETED
    assert phase_b.mode is KnowledgeSyncMode.RECONCILIATION
    assert phase_b.has_more is False
    assert phase_b.checkpoint_advanced is True
    assert phase_b.delivery_id == delivery_id_v1
    assert len(scenario.docs_calls) == docs_calls_after_a
    assert len(sink.calls) == sink_calls_after_a
    assert len(sink.durable_delivery_ids) == 1

    checkpoint_b = _fresh_checkpoint_repo(document_store).get(
        tenant_id=_TENANT_ID,
        binding_id=_BINDING_ID,
    )
    assert checkpoint_b is not None
    assert checkpoint_b.cursor.value == checkpoint_a.cursor.value

    state_b = _fresh_state_repo(document_store).get(
        tenant_id=_TENANT_ID,
        binding_id=_BINDING_ID,
        remote_id=_DOCUMENT_ID,
    )
    assert state_b == state_a

    # Phase C — new reconciliation after content change
    payload_v2 = _representative_payload(body_text=_BODY_TEXT_V2)
    scenario.queue_descriptor_content_pair(
        payload_v2,
        descriptor_revision=_DESCRIPTOR_REVISION_V2,
        content_revision=_CONTENT_REVISION_V2,
    )

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
        operation_id=_OPERATION_V2,
    )
    del runtime_c

    assert phase_c.status is KnowledgeSyncRunStatus.COMPLETED
    assert phase_c.mode is KnowledgeSyncMode.RECONCILIATION
    assert phase_c.has_more is False
    assert phase_c.checkpoint_advanced is True
    assert phase_c.tombstone_count == 0
    assert len(scenario.docs_calls) == 4
    assert len(sink.calls) == 2
    assert len(sink.durable_delivery_ids) == 2
    assert phase_c.delivery_id is not None
    assert phase_c.delivery_id != delivery_id_v1

    batch_2 = sink.calls[1]
    assert len(batch_2.envelopes) == 1
    envelope_2 = _envelope_for(batch_2, _DOCUMENT_ID)
    assert envelope_2.change_kind is KnowledgeChangeKind.UPSERT
    hash_v2 = envelope_2.descriptor.revision.content_hash  # type: ignore[union-attr]
    assert hash_v2 != hash_v1
    assert envelope_2.content.content_hash == hash_v2  # type: ignore[union-attr]
    _assert_structured_record_shape(
        envelope_2.content.structured_record,  # type: ignore[union-attr]
        body_text=_BODY_TEXT_V2,
    )

    state_c = _fresh_state_repo(document_store).get(
        tenant_id=_TENANT_ID,
        binding_id=_BINDING_ID,
        remote_id=_DOCUMENT_ID,
    )
    assert state_c is not None
    assert state_c.status is KnowledgeRemoteItemStatus.ACTIVE
    assert state_c.revision is not None
    assert state_c.revision.content_hash == hash_v2
    assert state_c.last_delivery_id == phase_c.delivery_id
    assert state_c.remote_id == _DOCUMENT_ID

    checkpoint_c = _fresh_checkpoint_repo(document_store).get(
        tenant_id=_TENANT_ID,
        binding_id=_BINDING_ID,
    )
    assert checkpoint_c is not None
    assert checkpoint_c.cursor.value == checkpoint_a.cursor.value
    assert _decode_docs_cursor_payload(checkpoint_c.cursor.value)["complete"] is True

    # Identity preservation across reconciliations
    assert envelope_1.descriptor.identity.remote_id == envelope_2.descriptor.identity.remote_id  # type: ignore[union-attr]
    assert envelope_1.descriptor.provenance.remote_id == envelope_2.descriptor.provenance.remote_id  # type: ignore[union-attr]
    assert envelope_1.descriptor.item_type == envelope_2.descriptor.item_type  # type: ignore[union-attr]

    # Phase D — incremental capability boundary
    docs_calls_before_d = len(scenario.docs_calls)
    sink_calls_before_d = len(sink.calls)
    checkpoint_before_d = checkpoint_c
    state_before_d = state_c

    runtime_d = _build_runtime(
        scenario=scenario,
        document_store=document_store,
        sink=sink,
        binding=binding,
        owner_id="owner-runtime-d",
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await runtime_d.coordinator.sync_once(binding_id=_BINDING_ID)
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert err.retryable is False
    assert len(scenario.docs_calls) == docs_calls_before_d
    assert len(sink.calls) == sink_calls_before_d

    checkpoint_after_d = _fresh_checkpoint_repo(document_store).get(
        tenant_id=_TENANT_ID,
        binding_id=_BINDING_ID,
    )
    state_after_d = _fresh_state_repo(document_store).get(
        tenant_id=_TENANT_ID,
        binding_id=_BINDING_ID,
        remote_id=_DOCUMENT_ID,
    )
    assert checkpoint_after_d == checkpoint_before_d
    assert state_after_d == state_before_d
    del runtime_d

    # Public and durable safety
    public_proof = _public_blob(
        {
            "runs": [phase_a.model_dump(mode="json"), phase_c.model_dump(mode="json")],
            "sink": [batch.model_dump(mode="json") for batch in sink.calls],
            "checkpoint": checkpoint_c.model_dump(mode="json"),
            "state": state_c.model_dump(mode="json"),
        }
    )
    _assert_no_public_secrets(public_proof)
    _assert_connection_ref_scope(
        public_proof,
        descriptor=envelope_2.descriptor,
        content=envelope_2.content,
        cursor=checkpoint_c.cursor,
    )
    _assert_sensitive_repr(
        envelope_2.descriptor,
        envelope_2.content,
        checkpoint_c.cursor,
    )


@pytest.mark.asyncio
async def test_google_docs_coordinator_rejects_descriptor_content_race_before_durable_write() -> None:
    scenario = _GoogleDocsProviderScenario()
    document_store = InMemoryDocumentStore()
    sink = IdempotentRecordingSink()
    binding = _binding()

    payload_v1 = _representative_payload(body_text=_BODY_TEXT_V1)
    payload_changed = _representative_payload(body_text=_BODY_TEXT_V2)
    scenario.queue_response(copy.deepcopy(payload_v1))
    scenario.queue_response(copy.deepcopy(payload_changed))

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
            operation_id="google-docs-fence",
        )

    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert err.retryable is True
    assert err.safe_message == _CONTENT_HASH_MISMATCH_MESSAGE
    assert err.__cause__ is None
    _assert_no_fence_leak(f"{err!r} {err.safe_message}")

    assert len(scenario.docs_calls) == 2
    assert len(sink.calls) == 0
    assert sink.durable_delivery_ids == []

    checkpoint = _fresh_checkpoint_repo(document_store).get(
        tenant_id=_TENANT_ID,
        binding_id=_BINDING_ID,
    )
    assert checkpoint is None

    state = _fresh_state_repo(document_store).get(
        tenant_id=_TENANT_ID,
        binding_id=_BINDING_ID,
        remote_id=_DOCUMENT_ID,
    )
    assert state is None
