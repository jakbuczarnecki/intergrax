# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""End-to-end Microsoft Graph Teams Channel knowledge adapter proof through facade and coordinator."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
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
    MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphTeamsChannelBodyKind,
    MsGraphTeamsChannelImportance,
    MsGraphTeamsChannelMessage,
    MsGraphTeamsChannelMessageKind,
    MsGraphTeamsChannelMessageState,
    MsGraphTeamsChannelMessageType,
    MsGraphTeamsChannelReplyPage,
    MsGraphTeamsChannelRootMessagePage,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_channel import (
    MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
    MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE,
    encode_msgraph_teams_channel_scope_id,
    register_msgraph_teams_channel_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBinding, KnowledgeSourceBindingStatus
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError, VendorKnowledgeErrorCode
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.models import KnowledgeContentMode, KnowledgeSourceScope
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from intergrax.runtime.vendor_knowledge.sync_coordinator import VendorKnowledgeSyncCoordinator
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
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_TEAM_ID = "team-abc-123"
_CHANNEL_ID = "channel-abc-123"
_ROOT_1 = "root-msg-001"
_ROOT_2 = "root-msg-002"
_REPLY_1 = "reply-msg-001"
_REPLY_2 = "reply-msg-002"
_ETAG_ROOT_1 = "etag-root-1"
_ETAG_ROOT_2 = "etag-root-2"
_ETAG_REPLY_1 = "etag-reply-1"
_ETAG_REPLY_2 = "etag-reply-2"
_SECRET_SKIP = "super-secret-skiptoken"
_TS = datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)
_CREATED_TS = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
_QUOTED_TEAM = quote(_TEAM_ID, safe="")
_QUOTED_CHANNEL = quote(_CHANNEL_ID, safe="")
_ROOT_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
    f"{_QUOTED_CHANNEL}/messages?$skiptoken={_SECRET_SKIP}"
)
_REPLY_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
    f"{_QUOTED_CHANNEL}/messages/{quote(_ROOT_1, safe='')}/replies?$skiptoken={_SECRET_SKIP}"
)


def _encode_message_remote_id(
    *,
    message_remote_id: str,
    message_kind: str,
    thread_root_remote_id: str | None = None,
) -> str:
    resolved_thread_root = thread_root_remote_id or message_remote_id
    payload = {
        "schema_version": "msgraph.teams-channel.message-id.v1",
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "thread_root_remote_id": resolved_thread_root,
        "message_kind": message_kind,
        "message_remote_id": message_remote_id,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _active_root(
    *,
    remote_id: str,
    revision: str,
    subject: str,
    body_content: str,
) -> MsGraphTeamsChannelMessage:
    return MsGraphTeamsChannelMessage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        thread_root_remote_id=remote_id,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        remote_id=remote_id,
        revision=revision,
        state=MsGraphTeamsChannelMessageState.ACTIVE,
        message_type=MsGraphTeamsChannelMessageType.MESSAGE,
        importance=MsGraphTeamsChannelImportance.NORMAL,
        created_at=_CREATED_TS,
        last_modified_at=_TS,
        subject=subject,
        body_kind=MsGraphTeamsChannelBodyKind.TEXT,
        body_content=body_content,
    )


def _active_reply(
    *,
    remote_id: str,
    revision: str,
    subject: str,
    body_content: str,
) -> MsGraphTeamsChannelMessage:
    return MsGraphTeamsChannelMessage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        thread_root_remote_id=_ROOT_1,
        message_kind=MsGraphTeamsChannelMessageKind.REPLY,
        remote_id=remote_id,
        revision=revision,
        state=MsGraphTeamsChannelMessageState.ACTIVE,
        message_type=MsGraphTeamsChannelMessageType.MESSAGE,
        importance=MsGraphTeamsChannelImportance.NORMAL,
        created_at=_CREATED_TS,
        last_modified_at=_TS,
        subject=subject,
        body_kind=MsGraphTeamsChannelBodyKind.TEXT,
        body_content=body_content,
    )


def _deleted_root(*, remote_id: str, revision: str) -> MsGraphTeamsChannelMessage:
    return MsGraphTeamsChannelMessage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        thread_root_remote_id=remote_id,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        remote_id=remote_id,
        revision=revision,
        state=MsGraphTeamsChannelMessageState.DELETED,
        message_type=MsGraphTeamsChannelMessageType.MESSAGE,
        importance=MsGraphTeamsChannelImportance.NORMAL,
        created_at=_CREATED_TS,
        last_modified_at=_TS,
        deleted_at=_TS,
    )


def _deleted_reply(*, remote_id: str, revision: str) -> MsGraphTeamsChannelMessage:
    return MsGraphTeamsChannelMessage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        thread_root_remote_id=_ROOT_1,
        message_kind=MsGraphTeamsChannelMessageKind.REPLY,
        remote_id=remote_id,
        revision=revision,
        state=MsGraphTeamsChannelMessageState.DELETED,
        message_type=MsGraphTeamsChannelMessageType.MESSAGE,
        importance=MsGraphTeamsChannelImportance.NORMAL,
        created_at=_CREATED_TS,
        last_modified_at=_TS,
        deleted_at=_TS,
    )


def _root_page(
    *,
    items: tuple[MsGraphTeamsChannelMessage, ...],
    continuation_url: str | None = None,
) -> MsGraphTeamsChannelRootMessagePage:
    continuation = None
    if continuation_url is not None:
        continuation = MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=continuation_url,
        )
    return MsGraphTeamsChannelRootMessagePage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        items=items,
        continuation=continuation,
    )


def _reply_page(
    *,
    items: tuple[MsGraphTeamsChannelMessage, ...],
    root_message_remote_id: str,
    root_message_revision: str,
    continuation_url: str | None = None,
) -> MsGraphTeamsChannelReplyPage:
    continuation = None
    if continuation_url is not None:
        continuation = MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=continuation_url,
        )
    return MsGraphTeamsChannelReplyPage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        root_message_remote_id=root_message_remote_id,
        root_message_revision=root_message_revision,
        items=items,
        continuation=continuation,
    )


class _TeamsChannelFakeCollaborationSuite(CollaborationSuite):
    def __init__(self) -> None:
        self.root_calls: list[dict[str, Any]] = []
        self.reply_calls: list[dict[str, Any]] = []
        self.content_calls: list[dict[str, Any]] = []
        self.forbidden_calls: list[str] = []
        self._root_pages = [
            _root_page(
                items=(
                    _active_root(
                        remote_id=_ROOT_1,
                        revision=_ETAG_ROOT_1,
                        subject="Root one",
                        body_content="root-one-body",
                    ),
                ),
                continuation_url=_ROOT_NEXT_URL,
            ),
            _root_page(
                items=(
                    _deleted_root(remote_id=_ROOT_2, revision=_ETAG_ROOT_2),
                ),
            ),
        ]
        self._reply_pages = {
            (_ROOT_1, _ETAG_ROOT_1): [
                _reply_page(
                    items=(
                        _active_reply(
                            remote_id=_REPLY_1,
                            revision=_ETAG_REPLY_1,
                            subject="Reply one",
                            body_content="reply-one-body",
                        ),
                    ),
                    root_message_remote_id=_ROOT_1,
                    root_message_revision=_ETAG_ROOT_1,
                    continuation_url=_REPLY_NEXT_URL,
                ),
                _reply_page(
                    items=(
                        _deleted_reply(remote_id=_REPLY_2, revision=_ETAG_REPLY_2),
                    ),
                    root_message_remote_id=_ROOT_1,
                    root_message_revision=_ETAG_ROOT_1,
                ),
            ],
            (_ROOT_2, _ETAG_ROOT_2): [
                _reply_page(
                    items=(),
                    root_message_remote_id=_ROOT_2,
                    root_message_revision=_ETAG_ROOT_2,
                ),
            ],
        }
        self._root_pages_backup = list(self._root_pages)
        self._reply_pages_backup = {
            key: list(pages) for key, pages in self._reply_pages.items()
        }
        self._content = {
            (_ROOT_1, _ETAG_ROOT_1): _active_root(
                remote_id=_ROOT_1,
                revision=_ETAG_ROOT_1,
                subject="Root one",
                body_content="root-one-body",
            ),
            (_REPLY_1, _ETAG_REPLY_1): _active_reply(
                remote_id=_REPLY_1,
                revision=_ETAG_REPLY_1,
                subject="Reply one",
                body_content="reply-one-body",
            ),
        }

    def _reset_if_needed(self) -> None:
        if not self._root_pages and self._root_pages_backup:
            self._root_pages = list(self._root_pages_backup)
            self._reply_pages = {
                key: list(pages) for key, pages in self._reply_pages_backup.items()
            }

    def read_teams_channel_root_messages_page_by_reference(
        self,
        *,
        channel,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int,
        max_chars_per_message: int,
    ) -> MsGraphTeamsChannelRootMessagePage:
        if continuation is None:
            self._reset_if_needed()
        self.root_calls.append(
            {
                "channel": channel,
                "continuation": continuation,
                "limit": limit,
                "max_chars_per_message": max_chars_per_message,
            }
        )
        if not self._root_pages:
            raise AssertionError("unexpected root page request")
        return self._root_pages.pop(0)

    def read_teams_channel_replies_page_by_reference(
        self,
        *,
        root_message,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int,
        max_chars_per_message: int,
    ) -> MsGraphTeamsChannelReplyPage:
        self.reply_calls.append(
            {
                "root_message": root_message,
                "continuation": continuation,
                "limit": limit,
                "max_chars_per_message": max_chars_per_message,
            }
        )
        key = (root_message.remote_id, root_message.revision)
        pages = self._reply_pages.get(key)
        if not pages:
            raise AssertionError(f"unexpected reply page request for {key!r}")
        return pages.pop(0)

    def read_teams_channel_message_content(
        self,
        *,
        message,
        max_chars: int,
    ) -> MsGraphTeamsChannelMessage:
        self.content_calls.append({"message": message, "max_chars": max_chars})
        return self._content[(message.remote_id, message.revision)]

    def read_mail_messages_delta_page(self, **kwargs: Any):
        self.forbidden_calls.append("delta")
        raise AssertionError("delta must not be called")

    def read_teams_channel_members_page(self, **kwargs: Any):
        self.forbidden_calls.append("members")
        raise AssertionError("members must not be called")

    def read_teams_channel_hosted_contents_page(self, **kwargs: Any):
        self.forbidden_calls.append("hosted_content")
        raise AssertionError("hosted content must not be called")

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


class _TeamsChannelTestIntegration(Ms365GraphCollaborationSuiteIntegration):
    def _graph_base_url_for_teams_channel_validation(self) -> str:
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
        _ROOT_NEXT_URL,
        _REPLY_NEXT_URL,
        _ETAG_ROOT_1,
        _ETAG_ROOT_2,
        _ETAG_REPLY_1,
        _ETAG_REPLY_2,
        "Authorization",
        "skiptoken",
        "deltatoken",
    )
    for item in forbidden:
        assert item not in blob
    assert "credential_ref" not in blob


def _build_coordinator(fake: _TeamsChannelFakeCollaborationSuite):
    integration = _TeamsChannelTestIntegration.from_client(fake, enabled=True)
    registry = KnowledgeAdapterRegistry()
    register_msgraph_teams_channel_knowledge_adapter(registry)
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
        binding_id="msgraph-teams-channel-binding",
        tenant_id="tenant-1",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
        connection_ref="conn-1",
        safe_display_name="Teams Channel Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id=encode_msgraph_teams_channel_scope_id(
                team_remote_id=_TEAM_ID,
                channel_remote_id=_CHANNEL_ID,
            ),
            remote_scope_type=MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE,
            safe_display_name="General Channel",
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
    )
    return coordinator, sink, checkpoint_repo, state_repo, fake


async def _reconcile_until_complete(coordinator: VendorKnowledgeSyncCoordinator) -> list:
    results = []
    restart = True
    while True:
        result = await coordinator.reconcile_once(
            binding_id="msgraph-teams-channel-binding",
            restart=restart,
        )
        results.append(result)
        assert result.status is KnowledgeSyncRunStatus.COMPLETED
        if not result.has_more:
            break
        restart = False
    return results


@pytest.mark.asyncio
async def test_msgraph_teams_channel_facade_coordinator_reconciliation() -> None:
    coordinator, sink, checkpoint_repo, state_repo, fake = _build_coordinator(
        _TeamsChannelFakeCollaborationSuite()
    )

    results = await _reconcile_until_complete(coordinator)
    assert len(results) == 5
    assert all(result.has_more for result in results[:-1])
    assert results[-1].has_more is False
    assert len(sink.calls) == 5
    assert len({batch.delivery_id for batch in sink.calls}) == 5

    traversal: list[tuple[str, str | None]] = []
    root_parent_id = _encode_message_remote_id(
        message_remote_id=_ROOT_1,
        message_kind="root",
    )
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
            assert descriptor.item_type == "msgraph_teams_channel_message"
            assert envelope.content is not None
            assert envelope.content.mode is KnowledgeContentMode.STRUCTURED_RECORD
            assert envelope.content.structured_record is not None
            assert envelope.content.structured_record["schema"] == (
                "msgraph.teams-channel.message.knowledge.v1"
            )
            if descriptor.metadata is not None:
                assert descriptor.metadata["message_kind"] == (
                    "reply" if descriptor.identity.parent_remote_id else "root"
                )
            if descriptor.identity.parent_remote_id is None:
                traversal.append(("root", descriptor.title))
            else:
                assert descriptor.identity.parent_remote_id == root_parent_id
                traversal.append(("reply", descriptor.title))

    assert traversal == [
        ("root", "Root one"),
        ("reply", "Reply one"),
        ("deleted", _encode_message_remote_id(
            message_remote_id=_REPLY_2,
            message_kind="reply",
            thread_root_remote_id=_ROOT_1,
        )),
        ("deleted", _encode_message_remote_id(
            message_remote_id=_ROOT_2,
            message_kind="root",
        )),
    ]

    checkpoint = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id="msgraph-teams-channel-binding",
    )
    assert checkpoint is not None
    assert checkpoint.cursor is not None
    assert checkpoint.cursor.version == MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION
    assert _SECRET_SKIP not in _public_blob(checkpoint.model_dump(mode="json"))

    assert fake.root_calls[0]["continuation"] is None
    assert fake.root_calls[1]["continuation"] is not None
    assert fake.root_calls[1]["continuation"].url == _ROOT_NEXT_URL
    assert fake.reply_calls[1]["continuation"] is not None
    assert fake.reply_calls[1]["continuation"].url == _REPLY_NEXT_URL
    assert fake.forbidden_calls == []

    for message_kind, message_id in (
        ("root", _ROOT_1),
        ("reply", _REPLY_1),
        ("reply", _REPLY_2),
        ("root", _ROOT_2),
    ):
        state = state_repo.get(
            tenant_id="tenant-1",
            binding_id="msgraph-teams-channel-binding",
            remote_id=_encode_message_remote_id(
                message_remote_id=message_id,
                message_kind=message_kind,
                thread_root_remote_id=_ROOT_1 if message_kind == "reply" else None,
            ),
        )
        assert state is not None
        if message_id in {_REPLY_2, _ROOT_2}:
            assert state.status is KnowledgeRemoteItemStatus.DELETED

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="msgraph-teams-channel-binding")
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert "incremental_changes" in exc_info.value.safe_message

    restart_result = await coordinator.reconcile_once(
        binding_id="msgraph-teams-channel-binding",
        restart=True,
    )
    assert restart_result.has_more is True
    assert fake.root_calls[-1]["continuation"] is None

    public_proof = _public_blob(
        {
            "results": [result.model_dump(mode="json") for result in results],
            "sink": [batch.model_dump(mode="json") for batch in sink.calls],
        }
    )
    _assert_no_secrets(public_proof)
