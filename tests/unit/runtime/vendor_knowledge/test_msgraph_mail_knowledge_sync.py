# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""End-to-end Microsoft Graph Mail knowledge adapter proof through facade and coordinator."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.client import (
    GraphRestClient,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MSGRAPH_MAIL_SOURCE_KIND,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphMailImportance,
    MsGraphMailMessageChange,
    MsGraphMailMessageChangeKind,
    MsGraphMailMessageContent,
    MsGraphMailMessageDeltaPage,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_mail import (
    MSGRAPH_MAIL_CURSOR_VERSION,
    MSGRAPH_MAIL_SCOPE_TYPE,
    encode_msgraph_mail_folder_scope_id,
    register_msgraph_mail_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
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

_MAILBOX_USER_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_FOLDER_ID = "AQMkAGI2TG93AAA="
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_FOLDER = quote(_FOLDER_ID, safe="")
_DELTA_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
    f"{_QUOTED_FOLDER}/messages/delta?$deltatoken=checkpoint-token-1"
)
_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
    f"{_QUOTED_FOLDER}/messages/delta?$skiptoken=page-token-1"
)
_INCREMENTAL_DELTA_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
    f"{_QUOTED_FOLDER}/messages/delta?$deltatoken=checkpoint-token-2"
)
_TS = "2026-05-29T10:15:30+00:00"
_SECRET_TOKEN = "checkpoint-token-1"
_SKIP_TOKEN = "page-token-1"
_CHANGE_KEY_A = "ck-msg-a"
_CHANGE_KEY_B = "ck-msg-b"
_CHANGE_KEY_B_UPDATED = "ck-msg-b-v2"
_CHANGE_KEY_C = "ck-msg-c"


def _graph_config() -> Ms365GraphIntegrationConfig:
    return Ms365GraphIntegrationConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret",
        graph_base_url=DEFAULT_GRAPH_BASE_URL,
    )


def _encode_message_remote_id(*, message_id: str) -> str:
    payload = {
        "schema_version": "msgraph.mail.message-id.v1",
        "mailbox_user_id": _MAILBOX_USER_ID,
        "folder_id": _FOLDER_ID,
        "message_id": message_id,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _active_message(
    *,
    remote_id: str,
    change_key: str,
    subject: str,
) -> MsGraphMailMessageChange:
    return MsGraphMailMessageChange(
        mailbox_user_id=_MAILBOX_USER_ID,
        scope_folder_id=_FOLDER_ID,
        remote_id=remote_id,
        kind=MsGraphMailMessageChangeKind.ACTIVE,
        parent_folder_id=_FOLDER_ID,
        change_key=change_key,
        subject=subject,
        last_modified_at=datetime.fromisoformat(_TS),
        is_read=False,
        is_draft=False,
        has_attachments=False,
        importance=MsGraphMailImportance.NORMAL,
    )


def _removed_message(*, remote_id: str) -> MsGraphMailMessageChange:
    return MsGraphMailMessageChange(
        mailbox_user_id=_MAILBOX_USER_ID,
        scope_folder_id=_FOLDER_ID,
        remote_id=remote_id,
        kind=MsGraphMailMessageChangeKind.REMOVED,
        removed_reason="deleted",
    )


def _page(
    *,
    items: tuple[MsGraphMailMessageChange, ...],
    continuation_kind: MsGraphKnowledgeContinuationKind,
    url: str,
) -> MsGraphMailMessageDeltaPage:
    return MsGraphMailMessageDeltaPage(
        items=items,
        continuation=MsGraphKnowledgeContinuation(kind=continuation_kind, url=url),
    )


def _content(
    *,
    remote_id: str,
    change_key: str,
    subject: str,
    body_text: str,
) -> MsGraphMailMessageContent:
    return MsGraphMailMessageContent(
        mailbox_user_id=_MAILBOX_USER_ID,
        remote_id=remote_id,
        parent_folder_id=_FOLDER_ID,
        content_revision=change_key,
        subject=subject,
        body_text=body_text,
    )


class _MailFakeCollaborationSuite(GraphRestClient):
    def __init__(self) -> None:
        super().__init__(_graph_config(), http_client=MagicMock())
        self.delta_calls: list[dict[str, Any]] = []
        self.content_calls: list[dict[str, Any]] = []
        self.attachment_calls: list[dict[str, Any]] = []
        self._reconcile_pages = [
            _page(
                items=(
                    _active_message(
                        remote_id="msg-a",
                        change_key=_CHANGE_KEY_A,
                        subject="Message A",
                    ),
                    _active_message(
                        remote_id="msg-b",
                        change_key=_CHANGE_KEY_B,
                        subject="Message B",
                    ),
                ),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            ),
            _page(
                items=(
                    _active_message(
                        remote_id="msg-c",
                        change_key=_CHANGE_KEY_C,
                        subject="Message C",
                    ),
                ),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            ),
        ]
        self._incremental_page = _page(
            items=(
                _active_message(
                    remote_id="msg-b",
                    change_key=_CHANGE_KEY_B_UPDATED,
                    subject="Message B updated",
                ),
                _removed_message(remote_id="msg-c"),
            ),
            continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=_INCREMENTAL_DELTA_URL,
        )
        self._content = {
            ("msg-a", _CHANGE_KEY_A): _content(
                remote_id="msg-a",
                change_key=_CHANGE_KEY_A,
                subject="Message A",
                body_text="body-a",
            ),
            ("msg-b", _CHANGE_KEY_B): _content(
                remote_id="msg-b",
                change_key=_CHANGE_KEY_B,
                subject="Message B",
                body_text="body-b",
            ),
            ("msg-c", _CHANGE_KEY_C): _content(
                remote_id="msg-c",
                change_key=_CHANGE_KEY_C,
                subject="Message C",
                body_text="body-c",
            ),
            ("msg-b", _CHANGE_KEY_B_UPDATED): _content(
                remote_id="msg-b",
                change_key=_CHANGE_KEY_B_UPDATED,
                subject="Message B updated",
                body_text="body-b-updated",
            ),
        }

    def read_mail_messages_delta_page(
        self,
        *,
        mailbox_user_id: str,
        folder_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int,
    ) -> MsGraphMailMessageDeltaPage:
        self.delta_calls.append(
            {
                "mailbox_user_id": mailbox_user_id,
                "folder_id": folder_id,
                "continuation": continuation,
                "limit": limit,
            }
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

    def read_mail_message_content(
        self,
        *,
        message: MsGraphMailMessageChange,
        max_chars: int,
    ) -> MsGraphMailMessageContent:
        self.content_calls.append({"message": message, "max_chars": max_chars})
        return self._content[(message.remote_id, message.change_key)]

    def read_mail_attachments_page(
        self,
        *,
        message: MsGraphMailMessageChange,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ):
        self.attachment_calls.append(
            {
                "message": message,
                "continuation": continuation,
                "limit": limit,
            }
        )
        raise AssertionError("attachment methods must not be called")

    def read_mail_file_attachment_content(
        self,
        *,
        message: MsGraphMailMessageChange,
        attachment: object,
        max_bytes: int = 100,
    ):
        self.attachment_calls.append(
            {
                "message": message,
                "attachment": attachment,
                "max_bytes": max_bytes,
            }
        )
        raise AssertionError("attachment methods must not be called")

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
        _CHANGE_KEY_A,
        _CHANGE_KEY_B,
        _CHANGE_KEY_B_UPDATED,
        _CHANGE_KEY_C,
        "Authorization",
        "skiptoken",
        "deltatoken",
    )
    for item in forbidden:
        assert item not in blob
    assert "credential_ref" not in blob


def _build_coordinator(fake: _MailFakeCollaborationSuite):
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        fake, enabled=True
    )
    registry = KnowledgeAdapterRegistry()
    register_msgraph_mail_knowledge_adapter(registry)
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
        binding_id="msgraph-mail-binding",
        tenant_id="tenant-1",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_MAIL_SOURCE_KIND,
        connection_ref="conn-1",
        safe_display_name="Mail Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id=encode_msgraph_mail_folder_scope_id(
                mailbox_user_id=_MAILBOX_USER_ID,
                folder_id=_FOLDER_ID,
            ),
            remote_scope_type=MSGRAPH_MAIL_SCOPE_TYPE,
            safe_display_name="Inbox Folder",
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
async def test_msgraph_mail_facade_coordinator_reconciliation_and_incremental() -> None:
    coordinator, sink, checkpoint_repo, state_repo, fake = _build_coordinator(
        _MailFakeCollaborationSuite()
    )

    first = await coordinator.reconcile_once(
        binding_id="msgraph-mail-binding",
        restart=True,
        operation_id="msgraph-mail-recon",
    )
    second = await coordinator.reconcile_once(
        binding_id="msgraph-mail-binding",
        restart=False,
        operation_id="msgraph-mail-recon",
        trigger_delivery_id=first.delivery_id,
    )

    assert first.status is KnowledgeSyncRunStatus.COMPLETED
    assert second.status is KnowledgeSyncRunStatus.COMPLETED
    assert first.has_more is True
    assert second.has_more is False
    assert len(sink.calls) == 2
    assert sink.calls[0].delivery_id != sink.calls[1].delivery_id

    message_envelopes = []
    for batch in sink.calls:
        assert batch.mode.value == "reconciliation"
        for envelope in batch.envelopes:
            assert envelope.permissions is None
            descriptor = envelope.descriptor
            assert descriptor is not None
            assert descriptor.item_type == "msgraph_mail_message"
            message_envelopes.append(envelope)
            assert envelope.content is not None
            assert envelope.content.mode is KnowledgeContentMode.STRUCTURED_RECORD
            assert envelope.content.structured_record is not None
            _assert_no_secrets(_public_blob(envelope.model_dump(mode="json")))

    assert len(message_envelopes) == 3

    checkpoint = checkpoint_repo.get(
        tenant_id="tenant-1", binding_id="msgraph-mail-binding"
    )
    assert checkpoint is not None
    assert checkpoint.cursor is not None
    assert checkpoint.cursor.version == MSGRAPH_MAIL_CURSOR_VERSION
    assert _SECRET_TOKEN not in _public_blob(checkpoint.model_dump(mode="json"))

    assert fake.delta_calls[0]["continuation"] is None
    assert fake.delta_calls[1]["continuation"] is not None
    assert (
        fake.delta_calls[1]["continuation"].kind
        == MsGraphKnowledgeContinuationKind.NEXT_PAGE
    )
    assert fake.attachment_calls == []

    for message_id in ("msg-a", "msg-b", "msg-c"):
        state = state_repo.get(
            tenant_id="tenant-1",
            binding_id="msgraph-mail-binding",
            remote_id=_encode_message_remote_id(message_id=message_id),
        )
        assert state is not None

    incremental = await coordinator.sync_once(binding_id="msgraph-mail-binding")
    assert incremental.status is KnowledgeSyncRunStatus.COMPLETED
    assert fake.delta_calls[-1]["continuation"] is not None
    assert (
        fake.delta_calls[-1]["continuation"].kind
        == MsGraphKnowledgeContinuationKind.DELTA
    )
    assert fake.delta_calls[-1]["continuation"].url == _DELTA_URL
    assert fake.attachment_calls == []

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
        binding_id="msgraph-mail-binding",
        remote_id=_encode_message_remote_id(message_id="msg-c"),
    )
    assert deleted_state is not None
    assert deleted_state.status is KnowledgeRemoteItemStatus.DELETED

    updated_checkpoint = checkpoint_repo.get(
        tenant_id="tenant-1", binding_id="msgraph-mail-binding"
    )
    assert updated_checkpoint is not None
    assert updated_checkpoint.cursor is not None
    assert updated_checkpoint.cursor.version == MSGRAPH_MAIL_CURSOR_VERSION

    public_proof = _public_blob(
        {
            "first": first.model_dump(mode="json"),
            "second": second.model_dump(mode="json"),
            "incremental": incremental.model_dump(mode="json"),
            "sink": [batch.model_dump(mode="json") for batch in sink.calls],
        }
    )
    _assert_no_secrets(public_proof)


@pytest.mark.asyncio
async def test_msgraph_mail_sink_failure_does_not_commit_checkpoint_or_state() -> None:
    coordinator, sink, checkpoint_repo, state_repo, fake = _build_coordinator(
        _MailFakeCollaborationSuite()
    )
    sink.fail_times = 1
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="msgraph-mail-binding",
            restart=True,
            operation_id="msgraph-mail-recon-restart",
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert len(sink.calls) == 1
    assert sink.durable_delivery_ids == []
    assert (
        checkpoint_repo.get(tenant_id="tenant-1", binding_id="msgraph-mail-binding")
        is None
    )
    for message_id in ("msg-a", "msg-b", "msg-c"):
        assert (
            state_repo.get(
                tenant_id="tenant-1",
                binding_id="msgraph-mail-binding",
                remote_id=_encode_message_remote_id(message_id=message_id),
            )
            is None
        )
    assert fake.delta_calls
