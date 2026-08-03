# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""End-to-end Microsoft Graph Drive knowledge adapter proof through facade and coordinator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pytest

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MSGRAPH_DRIVE_SOURCE_KIND,
    MsGraphDriveDeltaPage,
    MsGraphDriveFileContent,
    MsGraphDriveItem,
    MsGraphDriveItemKind,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_drive import (
    MSGRAPH_DRIVE_CURSOR_VERSION,
    MSGRAPH_DRIVE_SCOPE_TYPE,
    register_msgraph_drive_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
)
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeContentMode,
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
    KnowledgeSyncRunStatus,
)
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    IdempotentRecordingSink,
    RecordingBindingService,
    durable_reconciliation_coordinator_kwargs,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_DRIVE_ID = "b!drive-reconcile"
_DELTA_URL = (
    "https://graph.microsoft.com/v1.0/drives/b%21drive-reconcile/root/delta?"
    "$deltatoken=checkpoint-token-1"
)
_NEXT_URL = (
    "https://graph.microsoft.com/v1.0/drives/b%21drive-reconcile/root/delta?"
    "$skiptoken=page-token-1"
)
_INCREMENTAL_DELTA_URL = (
    "https://graph.microsoft.com/v1.0/drives/b%21drive-reconcile/root/delta?"
    "$deltatoken=checkpoint-token-2"
)
_TS = "2026-05-29T10:15:30+00:00"
_SECRET_TOKEN = "checkpoint-token-1"
_SKIP_TOKEN = "page-token-1"


def _item(
    *,
    remote_id: str,
    kind: MsGraphDriveItemKind,
    name: str,
    c_tag: str | None = '"ctag-1"',
) -> MsGraphDriveItem:
    if kind == MsGraphDriveItemKind.DELETED:
        return MsGraphDriveItem(remote_id=remote_id, drive_id=_DRIVE_ID, kind=kind)
    return MsGraphDriveItem(
        remote_id=remote_id,
        drive_id=_DRIVE_ID,
        parent_remote_id="parent-1",
        kind=kind,
        name=name,
        e_tag='"etag-1"',
        c_tag=c_tag,
        size_bytes=5,
        mime_type="application/pdf" if kind == MsGraphDriveItemKind.FILE else None,
        created_at=None,
        last_modified_at=__import__("datetime").datetime.fromisoformat(_TS),
        web_url="https://contoso.sharepoint.com/file",
        is_root=False,
        deleted_state=None,
    )


def _page(
    *,
    items: tuple[MsGraphDriveItem, ...],
    continuation_kind: MsGraphKnowledgeContinuationKind,
    url: str,
) -> MsGraphDriveDeltaPage:
    return MsGraphDriveDeltaPage(
        items=items,
        continuation=MsGraphKnowledgeContinuation(kind=continuation_kind, url=url),
    )


def _content(remote_id: str, data: bytes) -> MsGraphDriveFileContent:
    return MsGraphDriveFileContent(
        drive_id=_DRIVE_ID,
        remote_id=remote_id,
        content_revision='"ctag-1"',
        data=data,
        size_bytes=len(data),
        mime_type="application/pdf",
        content_hash=hashlib.sha256(data).hexdigest(),
    )


class _DriveFakeCollaborationSuite(CollaborationSuite):
    def __init__(self) -> None:
        self.delta_calls: list[dict[str, Any]] = []
        self.content_calls: list[dict[str, Any]] = []
        self._reconcile_pages = [
            _page(
                items=(
                    _item(
                        remote_id="file-1", kind=MsGraphDriveItemKind.FILE, name="A.pdf"
                    ),
                    _item(
                        remote_id="folder-1",
                        kind=MsGraphDriveItemKind.FOLDER,
                        name="Docs",
                        c_tag=None,
                    ),
                ),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            ),
            _page(
                items=(
                    _item(
                        remote_id="file-2", kind=MsGraphDriveItemKind.FILE, name="B.pdf"
                    ),
                ),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            ),
        ]
        self._incremental_page = _page(
            items=(
                _item(
                    remote_id="file-2",
                    kind=MsGraphDriveItemKind.FILE,
                    name="B-updated.pdf",
                ),
                _item(
                    remote_id="file-3",
                    kind=MsGraphDriveItemKind.DELETED,
                    name="ignored",
                ),
            ),
            continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=_INCREMENTAL_DELTA_URL,
        )
        self._content = {
            "file-1": _content("file-1", b"file-one"),
            "file-2": _content("file-2", b"file-two"),
        }

    def read_drive_delta_page(
        self,
        *,
        drive_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int,
    ) -> MsGraphDriveDeltaPage:
        self.delta_calls.append(
            {"drive_id": drive_id, "continuation": continuation, "limit": limit}
        )
        if (
            continuation is not None
            and continuation.kind == MsGraphKnowledgeContinuationKind.DELTA
        ):
            if continuation.url == _DELTA_URL:
                return self._incremental_page
        if not self._reconcile_pages:
            raise AssertionError("unexpected reconcile page request")
        return self._reconcile_pages.pop(0)

    def read_drive_file_content(self, *, item: MsGraphDriveItem, max_bytes: int):
        self.content_calls.append({"item": item, "max_bytes": max_bytes})
        return self._content[item.remote_id]

    def get_message(self, user_id: str, message_id: str):
        raise NotImplementedError

    def list_messages(self, user_id: str, *, folder: str = "inbox", limit: int = 25):
        raise NotImplementedError

    def send_mail(self, user_id: str, *, subject: str, body: str, to):
        raise NotImplementedError

    def list_calendar_events(
        self, user_id: str, *, start: str, end: str, limit: int = 50
    ):
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


@dataclass
class _GraphResolver:
    integration: Ms365GraphCollaborationSuiteIntegration

    def resolve(self, *, source) -> Ms365GraphCollaborationSuiteIntegration:
        return self.integration


def _public_blob(value: object) -> str:
    return json.dumps(value, default=str)


def _assert_no_secrets(blob: str) -> None:
    forbidden = (
        _SECRET_TOKEN,
        _SKIP_TOKEN,
        _NEXT_URL,
        _DELTA_URL,
        "Authorization",
        "skiptoken",
        "deltatoken",
    )
    for item in forbidden:
        assert item not in blob
    assert "credential_ref" not in blob


def _build_coordinator(fake: _DriveFakeCollaborationSuite):
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        fake, enabled=True
    )
    registry = KnowledgeAdapterRegistry()
    register_msgraph_drive_knowledge_adapter(registry)
    facade = VendorKnowledgeFacadeService(
        tenant_id="tenant-1",
        resolver=_GraphResolver(integration=integration),
        adapter_registry=registry,
    )
    document_store = InMemoryDocumentStore()
    lease_repo = DocumentStoreKnowledgeSourceLeaseRepository(document_store)
    checkpoint_repo = DocumentStoreKnowledgeSyncCheckpointRepository(document_store)
    state_repo = DocumentStoreKnowledgeRemoteItemStateRepository(document_store)
    sink = IdempotentRecordingSink()
    binding = KnowledgeSourceBinding(
        binding_id="msgraph-drive-binding",
        tenant_id="tenant-1",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_DRIVE_SOURCE_KIND,
        connection_ref="conn-1",
        safe_display_name="Drive Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id=_DRIVE_ID,
            remote_scope_type=MSGRAPH_DRIVE_SCOPE_TYPE,
            safe_display_name="Finance Drive",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
    )
    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id="tenant-1",
        owner_id="owner-1",
        binding_service=RecordingBindingService(binding=binding),  # type: ignore[arg-type]
        facade=facade,
        lease_repository=lease_repo,
        checkpoint_repository=checkpoint_repo,
        item_state_repository=state_repo,
        sink=sink,
        lease_ttl_seconds=30,
        **durable_reconciliation_coordinator_kwargs(
            state_repository=state_repo, document_store=document_store
        ),
    )
    return coordinator, sink, checkpoint_repo, state_repo, fake


@pytest.mark.asyncio
async def test_msgraph_drive_facade_coordinator_reconciliation_and_incremental() -> (
    None
):
    coordinator, sink, checkpoint_repo, state_repo, fake = _build_coordinator(
        _DriveFakeCollaborationSuite()
    )

    first = await coordinator.reconcile_once(
        binding_id="msgraph-drive-binding",
        restart=True,
        operation_id="msgraph-drive-recon",
    )
    second = await coordinator.reconcile_once(
        binding_id="msgraph-drive-binding",
        restart=False,
        operation_id="msgraph-drive-recon",
        trigger_delivery_id=first.delivery_id,
    )

    assert first.status is KnowledgeSyncRunStatus.COMPLETED
    assert second.status is KnowledgeSyncRunStatus.COMPLETED
    assert first.has_more is True
    assert second.has_more is False
    assert len(sink.calls) == 2
    assert sink.calls[0].delivery_id != sink.calls[1].delivery_id

    file_envelopes = []
    folder_envelopes = []
    for batch in sink.calls:
        assert batch.mode.value == "reconciliation"
        for envelope in batch.envelopes:
            assert envelope.permissions is None
            descriptor = envelope.descriptor
            assert descriptor is not None
            if descriptor.item_type == "msgraph_drive_file":
                file_envelopes.append(envelope)
                assert envelope.content is not None
                assert envelope.content.mode is KnowledgeContentMode.BINARY
                assert envelope.content.binary is not None
            elif descriptor.item_type == "msgraph_drive_folder":
                folder_envelopes.append(envelope)
                assert envelope.content is None
            _assert_no_secrets(_public_blob(envelope.model_dump(mode="json")))

    assert len(file_envelopes) == 2
    assert len(folder_envelopes) == 1

    checkpoint = checkpoint_repo.get(
        tenant_id="tenant-1", binding_id="msgraph-drive-binding"
    )
    assert checkpoint is not None
    assert checkpoint.cursor is not None
    assert checkpoint.cursor.version == MSGRAPH_DRIVE_CURSOR_VERSION
    assert _SECRET_TOKEN not in _public_blob(checkpoint.model_dump(mode="json"))

    assert fake.delta_calls[0]["continuation"] is None
    assert fake.delta_calls[1]["continuation"] is not None
    assert (
        fake.delta_calls[1]["continuation"].kind
        == MsGraphKnowledgeContinuationKind.NEXT_PAGE
    )

    for remote_id in ("file-1", "file-2", "folder-1"):
        state = state_repo.get(
            tenant_id="tenant-1",
            binding_id="msgraph-drive-binding",
            remote_id=remote_id,
        )
        assert state is not None

    incremental = await coordinator.sync_once(binding_id="msgraph-drive-binding")
    assert incremental.status is KnowledgeSyncRunStatus.COMPLETED
    assert fake.delta_calls[-1]["continuation"] is not None
    assert (
        fake.delta_calls[-1]["continuation"].kind
        == MsGraphKnowledgeContinuationKind.DELTA
    )
    assert fake.delta_calls[-1]["continuation"].url == _DELTA_URL

    incremental_batch = sink.calls[-1]
    kinds = {envelope.change_kind.value for envelope in incremental_batch.envelopes}
    assert "upsert" in kinds
    assert "deleted" in kinds
    deleted_envelope = next(
        envelope
        for envelope in incremental_batch.envelopes
        if envelope.change_kind.value == "deleted"
    )
    assert deleted_envelope.content is None
    assert deleted_envelope.descriptor is None

    deleted_state = state_repo.get(
        tenant_id="tenant-1",
        binding_id="msgraph-drive-binding",
        remote_id="file-3",
    )
    assert deleted_state is not None
    assert deleted_state.status is KnowledgeRemoteItemStatus.DELETED

    updated_checkpoint = checkpoint_repo.get(
        tenant_id="tenant-1", binding_id="msgraph-drive-binding"
    )
    assert updated_checkpoint is not None
    assert updated_checkpoint.cursor is not None
    assert updated_checkpoint.cursor.version == MSGRAPH_DRIVE_CURSOR_VERSION

    public_proof = _public_blob(
        {
            "first": first.model_dump(mode="json"),
            "second": second.model_dump(mode="json"),
            "incremental": incremental.model_dump(mode="json"),
            "sink": [batch.model_dump(mode="json") for batch in sink.calls],
        }
    )
    _assert_no_secrets(public_proof)
