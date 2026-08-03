# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""End-to-end Microsoft Graph Teams Chat knowledge adapter proof through facade and coordinator."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import pytest

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphTeamsChatBodyKind,
    MsGraphTeamsChatImportance,
    MsGraphTeamsChatMessage,
    MsGraphTeamsChatMessageSnapshotPage,
    MsGraphTeamsChatMessageState,
    MsGraphTeamsChatMessageType,
    MsGraphTeamsChatMessageWindow,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_chat import (
    MSGRAPH_TEAMS_CHAT_CURSOR_VERSION,
    MSGRAPH_TEAMS_CHAT_SCOPE_TYPE,
    encode_msgraph_teams_chat_scope_id,
    register_msgraph_teams_chat_knowledge_adapter,
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
)
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    IdempotentRecordingSink,
    RecordingBindingService,
    durable_reconcile_until_complete,
    durable_reconciliation_coordinator_kwargs,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_MAILBOX_USER_ID = "user-abc-123"
_CHAT_ID = "chat-abc-123"
_MSG_1 = "msg-001"
_MSG_2 = "msg-002"
_MSG_3 = "msg-003"
_ETAG_1 = "etag-msg-1"
_ETAG_2 = "etag-msg-2"
_ETAG_3 = "etag-msg-3"
_SECRET_SKIP = "super-secret-skiptoken"
_WINDOW_START = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
_WINDOW_END = datetime(2024, 2, 1, 0, 0, tzinfo=timezone.utc)
_TS = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)
_CREATED_TS = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_CHAT = quote(_CHAT_ID, safe="")
_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/"
    f"{_QUOTED_CHAT}/messages?$skiptoken={_SECRET_SKIP}"
)


def _window() -> MsGraphTeamsChatMessageWindow:
    return MsGraphTeamsChatMessageWindow(start_at=_WINDOW_START, end_at=_WINDOW_END)


def _encode_message_remote_id(*, message_remote_id: str) -> str:
    payload = {
        "schema_version": "msgraph.teams-chat.message-id.v1",
        "mailbox_user_id": _MAILBOX_USER_ID,
        "chat_remote_id": _CHAT_ID,
        "message_remote_id": message_remote_id,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _active_message(
    *,
    remote_id: str,
    revision: str,
    subject: str,
    body_content: str,
) -> MsGraphTeamsChatMessage:
    return MsGraphTeamsChatMessage(
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_remote_id=_CHAT_ID,
        remote_id=remote_id,
        revision=revision,
        state=MsGraphTeamsChatMessageState.ACTIVE,
        message_type=MsGraphTeamsChatMessageType.MESSAGE,
        importance=MsGraphTeamsChatImportance.NORMAL,
        created_at=_CREATED_TS,
        last_modified_at=_TS,
        subject=subject,
        body_kind=MsGraphTeamsChatBodyKind.TEXT,
        body_content=body_content,
    )


def _deleted_message(*, remote_id: str, revision: str) -> MsGraphTeamsChatMessage:
    return MsGraphTeamsChatMessage(
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_remote_id=_CHAT_ID,
        remote_id=remote_id,
        revision=revision,
        state=MsGraphTeamsChatMessageState.DELETED,
        message_type=MsGraphTeamsChatMessageType.MESSAGE,
        importance=MsGraphTeamsChatImportance.NORMAL,
        created_at=_CREATED_TS,
        last_modified_at=_TS,
        deleted_at=_TS,
    )


def _snapshot_page(
    *,
    items: tuple[MsGraphTeamsChatMessage, ...],
    continuation_url: str | None = None,
) -> MsGraphTeamsChatMessageSnapshotPage:
    continuation = None
    if continuation_url is not None:
        continuation = MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=continuation_url,
        )
    return MsGraphTeamsChatMessageSnapshotPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_remote_id=_CHAT_ID,
        window=_window(),
        items=items,
        continuation=continuation,
    )


class _TeamsChatFakeCollaborationSuite(CollaborationSuite):
    def __init__(self) -> None:
        self.snapshot_calls: list[dict[str, Any]] = []
        self.content_calls: list[dict[str, Any]] = []
        self.forbidden_calls: list[str] = []
        self._snapshot_pages = [
            _snapshot_page(
                items=(
                    _active_message(
                        remote_id=_MSG_1,
                        revision=_ETAG_1,
                        subject="Message one",
                        body_content="body-one",
                    ),
                    _active_message(
                        remote_id=_MSG_2,
                        revision=_ETAG_2,
                        subject="Message two",
                        body_content="body-two",
                    ),
                ),
                continuation_url=_NEXT_URL,
            ),
            _snapshot_page(
                items=(_deleted_message(remote_id=_MSG_3, revision=_ETAG_3),),
            ),
        ]
        self._snapshot_pages_backup = list(self._snapshot_pages)
        self._content = {
            (_MSG_1, _ETAG_1): _active_message(
                remote_id=_MSG_1,
                revision=_ETAG_1,
                subject="Message one",
                body_content="body-one",
            ),
            (_MSG_2, _ETAG_2): _active_message(
                remote_id=_MSG_2,
                revision=_ETAG_2,
                subject="Message two",
                body_content="body-two",
            ),
        }

    def _reset_if_needed(self) -> None:
        if not self._snapshot_pages and self._snapshot_pages_backup:
            self._snapshot_pages = list(self._snapshot_pages_backup)

    def read_teams_chat_messages_snapshot_page_by_reference(
        self,
        *,
        chat,
        window: MsGraphTeamsChatMessageWindow,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int,
        max_chars_per_message: int,
    ) -> MsGraphTeamsChatMessageSnapshotPage:
        if continuation is None:
            self._reset_if_needed()
        self.snapshot_calls.append(
            {
                "chat": chat,
                "window": window,
                "continuation": continuation,
                "limit": limit,
                "max_chars_per_message": max_chars_per_message,
            }
        )
        if not self._snapshot_pages:
            raise AssertionError("unexpected snapshot page request")
        return self._snapshot_pages.pop(0)

    def read_teams_chat_message_content(
        self, *, message, max_chars: int
    ) -> MsGraphTeamsChatMessage:
        self.content_calls.append({"message": message, "max_chars": max_chars})
        return self._content[(message.remote_id, message.revision)]

    def read_teams_chats_page(self, **kwargs: Any):
        self.forbidden_calls.append("chat_inventory")
        raise AssertionError("chat inventory must not be called")

    def read_teams_chat_members_page(self, **kwargs: Any):
        self.forbidden_calls.append("chat_members")
        raise AssertionError("chat members must not be called")

    def read_teams_chat_hosted_contents_page(self, **kwargs: Any):
        self.forbidden_calls.append("hosted_content")
        raise AssertionError("hosted content must not be called")

    def read_teams_channel_root_messages_page_by_reference(self, **kwargs: Any):
        self.forbidden_calls.append("teams_channel")
        raise AssertionError("teams channel must not be called")

    def read_mail_messages_delta_page(self, **kwargs: Any):
        self.forbidden_calls.append("mail")
        raise AssertionError("mail must not be called")

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


class _TeamsChatTestIntegration(Ms365GraphCollaborationSuiteIntegration):
    def _graph_base_url_for_teams_chat_validation(self) -> str:
        return DEFAULT_GRAPH_BASE_URL


@dataclass
class _GraphResolver:
    integration: Ms365GraphCollaborationSuiteIntegration

    def resolve(self, *, source) -> Ms365GraphCollaborationSuiteIntegration:
        return self.integration


def _public_blob(value: object) -> str:
    return json.dumps(value, default=str)


def _assert_no_secrets(blob: str) -> None:
    forbidden = (
        _SECRET_SKIP,
        _NEXT_URL,
        _ETAG_1,
        _ETAG_2,
        _ETAG_3,
        "Authorization",
        "skiptoken",
    )
    for item in forbidden:
        assert item not in blob


def _build_coordinator(fake: _TeamsChatFakeCollaborationSuite):
    integration = _TeamsChatTestIntegration.from_client(fake, enabled=True)
    registry = KnowledgeAdapterRegistry()
    register_msgraph_teams_chat_knowledge_adapter(registry)
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
        binding_id="msgraph-teams-chat-binding",
        tenant_id="tenant-1",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
        connection_ref="conn-1",
        safe_display_name="Teams Chat Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id=encode_msgraph_teams_chat_scope_id(
                mailbox_user_id=_MAILBOX_USER_ID,
                chat_remote_id=_CHAT_ID,
                window=_window(),
            ),
            remote_scope_type=MSGRAPH_TEAMS_CHAT_SCOPE_TYPE,
            safe_display_name="Project Chat",
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
    return coordinator, sink, checkpoint_repo, state_repo, fake, integration


async def _reconcile_until_complete(
    coordinator: VendorKnowledgeSyncCoordinator,
) -> list:
    return await durable_reconcile_until_complete(
        coordinator,
        binding_id="msgraph-teams-chat-binding",
        operation_id="msgraph-teams-chat-recon",
    )


@pytest.mark.asyncio
async def test_msgraph_teams_chat_facade_coordinator_reconciliation() -> None:
    coordinator, sink, checkpoint_repo, state_repo, fake, integration = (
        _build_coordinator(_TeamsChatFakeCollaborationSuite())
    )

    results = await _reconcile_until_complete(coordinator)
    assert len(results) == 2
    assert all(result.has_more for result in results[:-1])
    assert results[-1].has_more is False
    assert len(sink.calls) == 2
    assert len({batch.delivery_id for batch in sink.calls}) == 2

    traversal: list[tuple[str, str | None]] = []
    for batch in sink.calls:
        assert batch.mode.value == "reconciliation"
        for envelope in batch.envelopes:
            _assert_no_secrets(_public_blob(envelope.model_dump(mode="json")))
            assert envelope.permissions is None
            if envelope.change_kind.value == "deleted":
                assert envelope.descriptor is None
                assert envelope.content is None
                traversal.append(("deleted", envelope.remote_id))
                continue
            descriptor = envelope.descriptor
            assert descriptor is not None
            assert descriptor.item_type == "msgraph_teams_chat_message"
            assert descriptor.identity.parent_remote_id is None
            assert envelope.content is not None
            assert envelope.content.mode is KnowledgeContentMode.STRUCTURED_RECORD
            assert envelope.content.structured_record is not None
            assert envelope.content.structured_record["schema"] == (
                "msgraph.teams-chat.message.knowledge.v1"
            )
            traversal.append(("active", descriptor.title))

    assert traversal == [
        ("active", "Message one"),
        ("active", "Message two"),
        ("deleted", _encode_message_remote_id(message_remote_id=_MSG_3)),
    ]

    checkpoint = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id="msgraph-teams-chat-binding",
    )
    assert checkpoint is not None
    assert checkpoint.cursor is not None
    assert checkpoint.cursor.version == MSGRAPH_TEAMS_CHAT_CURSOR_VERSION
    assert _SECRET_SKIP not in _public_blob(checkpoint.model_dump(mode="json"))

    assert fake.snapshot_calls[0]["continuation"] is None
    assert fake.snapshot_calls[1]["continuation"] is not None
    assert fake.snapshot_calls[1]["continuation"].url == _NEXT_URL
    assert fake.forbidden_calls == []

    for message_id in (_MSG_1, _MSG_2):
        state = state_repo.get(
            tenant_id="tenant-1",
            binding_id="msgraph-teams-chat-binding",
            remote_id=_encode_message_remote_id(message_remote_id=message_id),
        )
        assert state is not None
    deleted_state = state_repo.get(
        tenant_id="tenant-1",
        binding_id="msgraph-teams-chat-binding",
        remote_id=_encode_message_remote_id(message_remote_id=_MSG_3),
    )
    assert deleted_state is not None
    assert deleted_state.status is KnowledgeRemoteItemStatus.DELETED

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="msgraph-teams-chat-binding")
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY

    restart_result = await coordinator.reconcile_once(
        binding_id="msgraph-teams-chat-binding",
        restart=True,
        operation_id="msgraph-teams-chat-incremental",
    )
    assert restart_result.has_more is True
    assert fake.snapshot_calls[-1]["continuation"] is None

    public_proof = _public_blob(
        {
            "results": [result.model_dump(mode="json") for result in results],
            "sink": [batch.model_dump(mode="json") for batch in sink.calls],
            "integration_id": id(integration),
        }
    )
    _assert_no_secrets(public_proof)


@pytest.mark.asyncio
async def test_msgraph_teams_chat_sink_failure_does_not_commit_checkpoint_or_state() -> (
    None
):
    coordinator, sink, checkpoint_repo, state_repo, fake, integration = (
        _build_coordinator(_TeamsChatFakeCollaborationSuite())
    )
    integration_id = id(integration)
    sink.fail_times = 1
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="msgraph-teams-chat-binding",
            restart=True,
            operation_id="msgraph-teams-chat-sink-fail",
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert len(sink.calls) == 1
    assert sink.durable_delivery_ids == []
    assert (
        checkpoint_repo.get(
            tenant_id="tenant-1",
            binding_id="msgraph-teams-chat-binding",
        )
        is None
    )
    for message_id in (_MSG_1, _MSG_2):
        assert (
            state_repo.get(
                tenant_id="tenant-1",
                binding_id="msgraph-teams-chat-binding",
                remote_id=_encode_message_remote_id(message_remote_id=message_id),
            )
            is None
        )
    assert len(fake.snapshot_calls) == 1
    assert fake.snapshot_calls[0]["continuation"] is None
    assert id(integration) == integration_id


@pytest.mark.asyncio
async def test_msgraph_teams_chat_retry_same_page_after_sink_failure() -> None:
    fake = _TeamsChatFakeCollaborationSuite()
    single_page = _snapshot_page(
        items=(
            _active_message(
                remote_id=_MSG_1,
                revision=_ETAG_1,
                subject="Message one",
                body_content="body-one",
            ),
        ),
    )
    fake._snapshot_pages = [single_page]
    fake._snapshot_pages_backup = [single_page]
    coordinator, sink, checkpoint_repo, state_repo, fake, integration = (
        _build_coordinator(fake)
    )
    integration_id = id(integration)
    sink.fail_times = 1
    with pytest.raises(VendorKnowledgeError):
        await coordinator.reconcile_once(
            binding_id="msgraph-teams-chat-binding",
            restart=True,
            operation_id="msgraph-teams-chat-sink-fail",
        )
    first_delivery = sink.calls[0].delivery_id
    first_snapshot_calls = len(fake.snapshot_calls)
    result = await coordinator.reconcile_once(
        binding_id="msgraph-teams-chat-binding",
        restart=False,
        operation_id="msgraph-teams-chat-sink-fail",
    )
    assert result.delivery_id == first_delivery
    assert sink.calls[1].delivery_id == first_delivery
    assert len(sink.durable_delivery_ids) == 1
    assert sink.durable_delivery_ids[0] == first_delivery
    assert len(fake.snapshot_calls) == first_snapshot_calls + 1
    assert fake.snapshot_calls[0] == fake.snapshot_calls[first_snapshot_calls]
    checkpoint = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id="msgraph-teams-chat-binding",
    )
    assert checkpoint is not None
    assert checkpoint.cursor is not None
    assert (
        state_repo.get(
            tenant_id="tenant-1",
            binding_id="msgraph-teams-chat-binding",
            remote_id=_encode_message_remote_id(message_remote_id=_MSG_1),
        )
        is not None
    )
    assert id(integration) == integration_id


@pytest.mark.asyncio
async def test_msgraph_teams_chat_absence_is_not_deletion() -> None:
    fake = _TeamsChatFakeCollaborationSuite()
    coordinator, sink, checkpoint_repo, state_repo, fake, _integration = (
        _build_coordinator(fake)
    )
    await _reconcile_until_complete(coordinator)
    active_remote_id = _encode_message_remote_id(message_remote_id=_MSG_1)
    active_state = state_repo.get(
        tenant_id="tenant-1",
        binding_id="msgraph-teams-chat-binding",
        remote_id=active_remote_id,
    )
    assert active_state is not None
    assert active_state.status is KnowledgeRemoteItemStatus.ACTIVE
    prior_sink_calls = len(sink.calls)
    fake._snapshot_pages = [
        _snapshot_page(
            items=(
                _active_message(
                    remote_id=_MSG_2,
                    revision=_ETAG_2,
                    subject="Message two",
                    body_content="body-two",
                ),
            ),
            continuation_url=_NEXT_URL,
        ),
    ]
    partial = await coordinator.reconcile_once(
        binding_id="msgraph-teams-chat-binding",
        restart=True,
        operation_id="msgraph-teams-chat-absence-partial",
    )
    assert partial.has_more is True
    assert len(sink.calls) == prior_sink_calls + 1
    latest_batch = sink.calls[-1]
    assert all(
        envelope.change_kind.value != "deleted" for envelope in latest_batch.envelopes
    )
    assert active_remote_id not in {
        envelope.remote_id for envelope in latest_batch.envelopes
    }
    still_active = state_repo.get(
        tenant_id="tenant-1",
        binding_id="msgraph-teams-chat-binding",
        remote_id=active_remote_id,
    )
    assert still_active is not None
    assert still_active.status is KnowledgeRemoteItemStatus.ACTIVE


@pytest.mark.asyncio
async def test_msgraph_teams_chat_explicit_deleted_record_produces_tombstone() -> None:
    fake = _TeamsChatFakeCollaborationSuite()
    fake._snapshot_pages = [
        _snapshot_page(
            items=(_deleted_message(remote_id=_MSG_1, revision=_ETAG_1),),
        ),
    ]
    coordinator, sink, _checkpoint_repo, state_repo, _fake, _integration = (
        _build_coordinator(fake)
    )
    await coordinator.reconcile_once(
        binding_id="msgraph-teams-chat-binding",
        restart=True,
        operation_id="msgraph-teams-chat-absence",
    )
    assert any(
        envelope.change_kind.value == "deleted"
        for batch in sink.calls
        for envelope in batch.envelopes
    )
    deleted_state = state_repo.get(
        tenant_id="tenant-1",
        binding_id="msgraph-teams-chat-binding",
        remote_id=_encode_message_remote_id(message_remote_id=_MSG_1),
    )
    assert deleted_state is not None
    assert deleted_state.status is KnowledgeRemoteItemStatus.DELETED
