# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for MsGraphTeamsChatKnowledgeAdapter."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationDependencyError,
)
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
    MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphTeamsChatAttachmentKind,
    MsGraphTeamsChatAttachmentReference,
    MsGraphTeamsChatBodyKind,
    MsGraphTeamsChatContentTooLarge,
    MsGraphTeamsChatImportance,
    MsGraphTeamsChatMention,
    MsGraphTeamsChatMessage,
    MsGraphTeamsChatMessageChanged,
    MsGraphTeamsChatMessageSnapshotPage,
    MsGraphTeamsChatMessageState,
    MsGraphTeamsChatMessageType,
    MsGraphTeamsChatMessageWindow,
    MsGraphTeamsChatReaction,
    MsGraphTeamsForwardedMessageReference,
    MsGraphTeamsIdentity,
    MsGraphTeamsIdentityKind,
)
from intergrax.runtime.vendor_knowledge.adapters.confluence_pages import (
    register_confluence_pages_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.jira_issues import (
    register_jira_issues_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_drive import (
    MSGRAPH_DRIVE_SCOPE_TYPE,
    register_msgraph_drive_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_mail import (
    MSGRAPH_MAIL_SCOPE_TYPE,
    register_msgraph_mail_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_channel import (
    MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE,
    register_msgraph_teams_channel_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_chat import (
    MSGRAPH_TEAMS_CHAT_CURSOR_VERSION,
    MSGRAPH_TEAMS_CHAT_SCOPE_TYPE,
    MsGraphTeamsChatKnowledgeAdapter,
    _MsGraphTeamsChatCursor,
    _MsGraphTeamsChatMessageRevision,
    _MsGraphTeamsChatScope,
    _validate_opaque_revision,
    encode_msgraph_teams_chat_scope_id,
    register_msgraph_teams_chat_knowledge_adapter,
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

_MAILBOX_USER_ID = "user-abc-123"
_CHAT_ID = "chat-abc-123"
_OTHER_MAILBOX = "other-user-456"
_OTHER_CHAT_ID = "other-chat-456"
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
_STRUCTURED_RECORD_SCHEMA = "msgraph.teams-chat.message.knowledge.v1"
_STRUCTURED_RECORD_MIME = "application/vnd.intergrax.msgraph-teams-chat-message+json"
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_CHAT = quote(_CHAT_ID, safe="")
_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/"
    f"{_QUOTED_CHAT}/messages?$skiptoken={_SECRET_SKIP}"
)
_OTHER_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/users/{quote(_OTHER_MAILBOX, safe='')}/chats/"
    f"{quote(_OTHER_CHAT_ID, safe='')}/messages?$skiptoken=other"
)


def _window() -> MsGraphTeamsChatMessageWindow:
    return MsGraphTeamsChatMessageWindow(start_at=_WINDOW_START, end_at=_WINDOW_END)


def _encode_canonical_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _scope_id(
    *,
    mailbox_user_id: str = _MAILBOX_USER_ID,
    chat_remote_id: str = _CHAT_ID,
    window: MsGraphTeamsChatMessageWindow | None = None,
) -> str:
    return encode_msgraph_teams_chat_scope_id(
        mailbox_user_id=mailbox_user_id,
        chat_remote_id=chat_remote_id,
        window=window or _window(),
    )


def _encode_message_identity(
    *,
    message_remote_id: str,
    mailbox_user_id: str = _MAILBOX_USER_ID,
    chat_remote_id: str = _CHAT_ID,
) -> str:
    return _encode_canonical_payload(
        {
            "schema_version": "msgraph.teams-chat.message-id.v1",
            "mailbox_user_id": mailbox_user_id,
            "chat_remote_id": chat_remote_id,
            "message_remote_id": message_remote_id,
        }
    )


def _encode_revision(revision: str) -> str:
    return _encode_canonical_payload(
        {
            "schema_version": "msgraph.teams-chat.revision.v1",
            "revision": revision,
        }
    )


def _source(
    *,
    remote_scope_id: str | None = None,
    remote_scope_type: str = MSGRAPH_TEAMS_CHAT_SCOPE_TYPE,
    parameters: dict[str, Any] | None = None,
    provider_id: str = MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    integration_kind: IntegrationCategory = IntegrationCategory.COLLABORATION_SUITE,
    source_kind: str = MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        scope=KnowledgeSourceScope(
            remote_scope_id=remote_scope_id or _scope_id(),
            remote_scope_type=remote_scope_type,
            safe_display_name="Project Chat",
            parameters=parameters or {},
        ),
    )


def _active_message(
    *,
    remote_id: str = _MSG_1,
    revision: str = _ETAG_1,
    subject: str | None = "Sprint planning",
    body_kind: MsGraphTeamsChatBodyKind = MsGraphTeamsChatBodyKind.TEXT,
    body_content: str = "Message body",
    sender: MsGraphTeamsIdentity | None = None,
    mentions: tuple[MsGraphTeamsChatMention, ...] = (),
    reactions: tuple[MsGraphTeamsChatReaction, ...] = (),
    attachments: tuple[MsGraphTeamsChatAttachmentReference, ...] = (),
    mailbox_user_id: str = _MAILBOX_USER_ID,
    chat_remote_id: str = _CHAT_ID,
) -> MsGraphTeamsChatMessage:
    return MsGraphTeamsChatMessage(
        mailbox_user_id=mailbox_user_id,
        chat_remote_id=chat_remote_id,
        remote_id=remote_id,
        revision=revision,
        state=MsGraphTeamsChatMessageState.ACTIVE,
        message_type=MsGraphTeamsChatMessageType.MESSAGE,
        importance=MsGraphTeamsChatImportance.NORMAL,
        created_at=_CREATED_TS,
        last_modified_at=_TS,
        subject=subject,
        body_kind=body_kind,
        body_content=body_content,
        sender=sender,
        mentions=mentions,
        reactions=reactions,
        attachments=attachments,
    )


def _deleted_message(
    *,
    remote_id: str = _MSG_2,
    revision: str = _ETAG_2,
    mailbox_user_id: str = _MAILBOX_USER_ID,
    chat_remote_id: str = _CHAT_ID,
) -> MsGraphTeamsChatMessage:
    return MsGraphTeamsChatMessage(
        mailbox_user_id=mailbox_user_id,
        chat_remote_id=chat_remote_id,
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
    mailbox_user_id: str = _MAILBOX_USER_ID,
    chat_remote_id: str = _CHAT_ID,
    window: MsGraphTeamsChatMessageWindow | None = None,
) -> MsGraphTeamsChatMessageSnapshotPage:
    continuation = None
    if continuation_url is not None:
        continuation = MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=continuation_url,
        )
    return MsGraphTeamsChatMessageSnapshotPage(
        mailbox_user_id=mailbox_user_id,
        chat_remote_id=chat_remote_id,
        window=window or _window(),
        items=items,
        continuation=continuation,
    )


class _FakeTeamsChatCollaborationSuite(CollaborationSuite):
    def __init__(
        self,
        *,
        snapshot_pages: list[MsGraphTeamsChatMessageSnapshotPage] | None = None,
        content_by_key: dict[tuple[str, str], MsGraphTeamsChatMessage] | None = None,
    ) -> None:
        self._snapshot_pages = list(snapshot_pages or [])
        self._content_by_key = dict(content_by_key or {})
        self.snapshot_calls: list[dict[str, Any]] = []
        self.content_calls: list[dict[str, Any]] = []
        self.forbidden_calls: list[str] = []

    def read_teams_chat_messages_snapshot_page_by_reference(
        self,
        *,
        chat,
        window: MsGraphTeamsChatMessageWindow,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int,
        max_chars_per_message: int,
    ) -> MsGraphTeamsChatMessageSnapshotPage:
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
            raise IntegrationDependencyError("no snapshot pages configured")
        return self._snapshot_pages.pop(0)

    def read_teams_chat_message_content(
        self,
        *,
        message,
        max_chars: int,
    ) -> MsGraphTeamsChatMessage:
        self.content_calls.append({"message": message, "max_chars": max_chars})
        return self._content_by_key[(message.remote_id, message.revision)]

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


class _TeamsChatTestIntegration(Ms365GraphCollaborationSuiteIntegration):
    def _graph_base_url_for_teams_chat_validation(self) -> str:
        return DEFAULT_GRAPH_BASE_URL


def _integration(fake: _FakeTeamsChatCollaborationSuite) -> Ms365GraphCollaborationSuiteIntegration:
    return _TeamsChatTestIntegration.from_client(fake, enabled=True)


def _encode_cursor(payload: dict[str, Any]) -> KnowledgeCursor:
    return KnowledgeCursor(
        value=_encode_canonical_payload(payload),
        version=MSGRAPH_TEAMS_CHAT_CURSOR_VERSION,
    )


def _base_metadata(*, body_kind: str = "text", has_attachments: bool = False) -> dict[str, object]:
    return {
        "message_state": "active",
        "message_type": "message",
        "importance": "normal",
        "body_kind": body_kind,
        "has_attachments": has_attachments,
        "created_at": _CREATED_TS.isoformat(),
        "last_modified_at": _TS.isoformat(),
        "last_edited_at": None,
        "event_detail_type": None,
        "locale": None,
        "attachment_inventory_in_content": True,
        "attachment_binary_content_included": False,
        "hosted_content_included": False,
        "reference_urls_included": False,
    }


def _message_descriptor(
    *,
    message_remote_id: str = _MSG_1,
    revision: str = _ETAG_1,
    title: str = "Sprint planning",
    mailbox_user_id: str = _MAILBOX_USER_ID,
    chat_remote_id: str = _CHAT_ID,
    metadata: dict[str, Any] | None = None,
    metadata_only: bool = False,
    content_available: bool = True,
    item_type: str = "msgraph_teams_chat_message",
    content_mode: KnowledgeContentMode = KnowledgeContentMode.STRUCTURED_RECORD,
    provenance_source_kind: str = MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
) -> KnowledgeItemDescriptor:
    opaque_id = _encode_message_identity(
        message_remote_id=message_remote_id,
        mailbox_user_id=mailbox_user_id,
        chat_remote_id=chat_remote_id,
    )
    base_metadata = _base_metadata()
    resolved_metadata = metadata if metadata_only else {**base_metadata, **(metadata or {})}
    return KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(
            remote_id=opaque_id,
            parent_remote_id=None,
            logical_key=None,
        ),
        revision=KnowledgeItemRevision(
            version=_encode_revision(revision),
            etag=None,
            updated_at=_TS,
        ),
        title=title,
        item_type=item_type,
        content_mode=content_mode,
        content_available=content_available,
        provenance=KnowledgeItemProvenance(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=provenance_source_kind,
            remote_id=opaque_id,
        ),
        metadata=resolved_metadata,
    )


def _assert_invalid_descriptor_boundary(exc_info: pytest.ExceptionInfo[VendorKnowledgeError]) -> None:
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert err.retryable is False
    rendered = f"{err!r} {err.safe_message}"
    for secret in (
        _MAILBOX_USER_ID,
        _CHAT_ID,
        _MSG_1,
        _ETAG_1,
        "Sprint planning",
        "2024-01-01",
        "graph.microsoft.com",
        "Authorization",
    ):
        assert secret not in rendered


async def _fetch_content_invalid_descriptor(item: KnowledgeItemDescriptor) -> pytest.ExceptionInfo[VendorKnowledgeError]:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(
        content_by_key={(_MSG_1, _ETAG_1): _active_message(body_content="x")}
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=item,
        )
    return exc_info


async def test_adapter_identity() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    assert adapter.provider_id == MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    assert adapter.integration_kind is IntegrationCategory.COLLABORATION_SUITE
    assert adapter.source_kind == MSGRAPH_TEAMS_CHAT_SOURCE_KIND
    assert isinstance(adapter, VendorKnowledgeAdapter)


async def test_capabilities_exact_set() -> None:
    caps = MsGraphTeamsChatKnowledgeAdapter().capabilities
    assert caps.full_inventory is True
    assert caps.incremental_changes is False
    assert caps.reconciliation is True
    assert caps.content_fetch is True
    assert caps.binary_content is False
    assert caps.rich_text_content is False
    assert caps.structured_content is True
    assert caps.permissions is False
    assert caps.tombstones is True
    assert caps.remote_versions is True


async def test_registry_registration_and_coexistence() -> None:
    registry = KnowledgeAdapterRegistry()
    chat_adapter = register_msgraph_teams_chat_knowledge_adapter(registry)
    register_msgraph_drive_knowledge_adapter(registry)
    register_msgraph_mail_knowledge_adapter(registry)
    register_msgraph_teams_channel_knowledge_adapter(registry)
    register_confluence_pages_knowledge_adapter(registry)
    register_jira_issues_knowledge_adapter(registry)
    assert isinstance(chat_adapter, MsGraphTeamsChatKnowledgeAdapter)
    assert isinstance(registry.resolve(source=_source()), MsGraphTeamsChatKnowledgeAdapter)


async def test_duplicate_registry_registration_rejected() -> None:
    registry = KnowledgeAdapterRegistry()
    register_msgraph_teams_chat_knowledge_adapter(registry)
    with pytest.raises(ValueError):
        register_msgraph_teams_chat_knowledge_adapter(registry)


async def test_valid_chat_scope_inspect() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    info = await adapter.inspect_scope(integration=_integration(_FakeTeamsChatCollaborationSuite()), source=_source())
    assert info.capabilities.structured_content is True
    assert info.source.scope.remote_scope_type == MSGRAPH_TEAMS_CHAT_SCOPE_TYPE


async def test_encoded_scope_id_hides_raw_ids() -> None:
    scope_id = _scope_id()
    assert _MAILBOX_USER_ID not in scope_id
    assert _CHAT_ID not in scope_id
    padding = "=" * (-len(scope_id) % 4)
    scope = _MsGraphTeamsChatScope.model_validate(
        json.loads(base64.urlsafe_b64decode(scope_id + padding).decode())
    )
    assert repr(scope) == "_MsGraphTeamsChatScope(schema_version='msgraph.teams-chat.scope.v1')"


async def test_scope_encoding_round_trip_and_timezone_normalization() -> None:
    offset = timezone(timedelta(hours=1))
    local_start = datetime(2024, 1, 1, 1, 0, tzinfo=offset)
    local_end = datetime(2024, 2, 1, 1, 0, tzinfo=offset)
    expected_start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    expected_end = datetime(2024, 2, 1, 0, 0, tzinfo=timezone.utc)
    window = MsGraphTeamsChatMessageWindow(start_at=local_start, end_at=local_end)
    scope_id = encode_msgraph_teams_chat_scope_id(
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_remote_id=_CHAT_ID,
        window=window,
    )
    decoded = _MsGraphTeamsChatScope.model_validate(
        json.loads(base64.urlsafe_b64decode(scope_id + "==").decode())
    )
    assert decoded.mailbox_user_id == _MAILBOX_USER_ID
    assert decoded.chat_remote_id == _CHAT_ID
    assert decoded.window_start_at == expected_start
    assert decoded.window_end_at == expected_end


@pytest.mark.parametrize(
    "source",
    [
        _source(provider_id="other"),
        _source(integration_kind=IntegrationCategory.COLLABORATION_SUITE, source_kind="wrong"),
        _source(source_kind=MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND),
        _source(remote_scope_type=MSGRAPH_MAIL_SCOPE_TYPE),
        _source(remote_scope_type=MSGRAPH_DRIVE_SCOPE_TYPE),
        _source(remote_scope_type=MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE),
        _source(parameters={"window": "forbidden"}),
    ],
)
async def test_invalid_scope_rejected(source: KnowledgeSourceRef) -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(
        snapshot_pages=[_snapshot_page(items=(_active_message(),))]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(integration=_integration(fake), source=source, cursor=None, limit=50)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert fake.snapshot_calls == []


@pytest.mark.parametrize(
    "remote_scope_id",
    [
        "not-base64!!!",
        _encode_canonical_payload({"schema_version": "wrong.v1"}),
        _encode_canonical_payload(
            {
                "schema_version": "msgraph.teams-chat.scope.v1",
                "mailbox_user_id": _MAILBOX_USER_ID,
                "chat_remote_id": _CHAT_ID,
                "window_start_at": _WINDOW_START.isoformat(),
                "window_end_at": _WINDOW_END.isoformat(),
                "extra": True,
            }
        ),
        _encode_canonical_payload(
            {
                "schema_version": "msgraph.teams-chat.scope.v1",
                "mailbox_user_id": f" {_MAILBOX_USER_ID} ",
                "chat_remote_id": _CHAT_ID,
                "window_start_at": _WINDOW_START.isoformat(),
                "window_end_at": _WINDOW_END.isoformat(),
            }
        ),
    ],
)
async def test_malformed_scope_payload_rejected(remote_scope_id: str) -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(snapshot_pages=[_snapshot_page(items=(_active_message(),))])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(remote_scope_id=remote_scope_id),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert fake.snapshot_calls == []


async def test_wrong_integration_object_rejected() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(integration=object(), source=_source(), cursor=None, limit=50)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


@pytest.mark.parametrize("limit", [1, 50, 1000])
async def test_limit_accepted_and_provider_clamped(limit: int) -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(
        snapshot_pages=[_snapshot_page(items=(_active_message(),))]
    )
    page = await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=limit)
    assert len(page.changes) == 1
    assert fake.snapshot_calls[0]["limit"] == min(limit, 50)


@pytest.mark.parametrize("limit", [0, -1, 1001, "50", 50.0])
async def test_invalid_limit_rejected(limit: object) -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(snapshot_pages=[_snapshot_page(items=(_active_message(),))])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=limit)  # type: ignore[arg-type]
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert fake.snapshot_calls == []


async def test_first_page_uses_no_continuation() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(
        snapshot_pages=[_snapshot_page(items=(_active_message(),), continuation_url=_NEXT_URL)]
    )
    page = await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    assert fake.snapshot_calls[0]["continuation"] is None
    assert page.has_more is True
    assert page.next_cursor is not None


async def test_continuation_round_trip() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(
        snapshot_pages=[
            _snapshot_page(items=(_active_message(),), continuation_url=_NEXT_URL),
            _snapshot_page(items=(_active_message(remote_id=_MSG_2, revision=_ETAG_2),)),
        ]
    )
    first = await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    second = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=first.next_cursor,
        limit=50,
    )
    assert fake.snapshot_calls[1]["continuation"].url == _NEXT_URL
    assert second.has_more is False
    assert second.next_cursor is None


async def test_active_message_maps_to_upsert_and_deleted_to_tombstone() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(
        snapshot_pages=[
            _snapshot_page(
                items=(
                    _active_message(),
                    _deleted_message(),
                )
            )
        ]
    )
    page = await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    assert page.changes[0].kind is KnowledgeChangeKind.UPSERT
    assert page.changes[0].descriptor is not None
    assert page.changes[0].descriptor.identity.parent_remote_id is None
    assert page.changes[1].kind is KnowledgeChangeKind.DELETED
    assert page.changes[1].descriptor is None


@pytest.mark.parametrize(
    "subject,expected_title",
    [
        ("Planning", "Planning"),
        ("  ", "Teams chat message"),
        (None, "Teams chat message"),
    ],
)
async def test_title_resolution(subject: str | None, expected_title: str) -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(
        snapshot_pages=[_snapshot_page(items=(_active_message(subject=subject),))]
    )
    page = await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    assert page.changes[0].descriptor is not None
    assert page.changes[0].descriptor.title == expected_title


async def test_empty_page_with_continuation_advances() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(
        snapshot_pages=[
            _snapshot_page(items=(), continuation_url=_NEXT_URL),
            _snapshot_page(items=(_active_message(),)),
        ]
    )
    first = await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    assert first.changes == ()
    assert first.has_more is True
    second = await adapter.read_page(
        integration=_integration(fake), source=_source(), cursor=first.next_cursor, limit=50
    )
    assert len(second.changes) == 1
    assert second.has_more is False


async def test_complete_cursor_rejected() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(snapshot_pages=[_snapshot_page(items=(_active_message(),))])
    complete = _encode_cursor(
        {
            "schema_version": MSGRAPH_TEAMS_CHAT_CURSOR_VERSION,
            "mailbox_user_id": _MAILBOX_USER_ID,
            "chat_remote_id": _CHAT_ID,
            "window_start_at": _WINDOW_START.isoformat(),
            "window_end_at": _WINDOW_END.isoformat(),
            "phase": "complete",
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake), source=_source(), cursor=complete, limit=50
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert fake.snapshot_calls == []


async def test_cursor_binding_rejects_mismatched_scope() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(snapshot_pages=[_snapshot_page(items=(_active_message(),))])
    cursor = _encode_cursor(
        {
            "schema_version": MSGRAPH_TEAMS_CHAT_CURSOR_VERSION,
            "mailbox_user_id": _OTHER_MAILBOX,
            "chat_remote_id": _CHAT_ID,
            "window_start_at": _WINDOW_START.isoformat(),
            "window_end_at": _WINDOW_END.isoformat(),
            "phase": "messages",
            "continuation_url": _NEXT_URL,
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(integration=_integration(fake), source=_source(), cursor=cursor, limit=50)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR


async def test_cursor_url_hidden_from_repr_and_errors() -> None:
    cursor = _MsGraphTeamsChatCursor(
        schema_version=MSGRAPH_TEAMS_CHAT_CURSOR_VERSION,
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_remote_id=_CHAT_ID,
        window_start_at=_WINDOW_START,
        window_end_at=_WINDOW_END,
        phase="messages",
        continuation_url=_NEXT_URL,
    )
    rendered = repr(cursor)
    assert _SECRET_SKIP not in rendered
    assert _NEXT_URL not in rendered


async def test_message_descriptor_mapping() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(
        snapshot_pages=[_snapshot_page(items=(_active_message(),))]
    )
    page = await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.item_type == "msgraph_teams_chat_message"
    assert descriptor.content_mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert descriptor.content_available is True
    assert descriptor.identity.remote_id != _MSG_1
    assert set(descriptor.metadata.keys()) == set(_base_metadata().keys())


async def test_forbidden_provider_calls_not_made_during_paging() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(
        snapshot_pages=[_snapshot_page(items=(_active_message(),))]
    )
    await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    assert fake.forbidden_calls == []


async def test_fetch_content_structured_record() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    sender = MsGraphTeamsIdentity(
        identity_kind=MsGraphTeamsIdentityKind.USER,
        remote_id="sender-1",
        display_name="Alice",
    )
    forwarded = MsGraphTeamsForwardedMessageReference(
        original_message_id="orig-msg",
        original_chat_id="orig-chat",
        original_sent_at=_CREATED_TS,
        original_sender=sender,
    )
    attachment = MsGraphTeamsChatAttachmentReference(
        remote_id="att-1",
        attachment_kind=MsGraphTeamsChatAttachmentKind.FORWARDED_MESSAGE_REFERENCE,
        content_type="forwardedMessageReference",
        name="fwd",
        has_thumbnail_url=False,
        forwarded_message=forwarded,
    )
    message = _active_message(
        body_kind=MsGraphTeamsChatBodyKind.HTML,
        body_content="<p>Hello</p>",
        sender=sender,
        mentions=(
            MsGraphTeamsChatMention(
                mention_id=1,
                mention_text="@Alice",
                mentioned=sender,
            ),
        ),
        reactions=(
            MsGraphTeamsChatReaction(
                reaction_type="like",
                display_name="Like",
                created_at=_TS,
                user=sender,
            ),
        ),
        attachments=(attachment,),
    )
    fake = _FakeTeamsChatCollaborationSuite(content_by_key={(_MSG_1, _ETAG_1): message})
    descriptor = _message_descriptor(
        metadata={"body_kind": "html", "has_attachments": True},
    )
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=descriptor,
    )
    assert content.mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert content.mime_type == _STRUCTURED_RECORD_MIME
    record = content.structured_record
    assert record["schema"] == _STRUCTURED_RECORD_SCHEMA
    assert record["sender"]["display_name"] == "Alice"
    assert record["mentions"]
    assert record["reactions"]
    assert record["attachments"]["items"]
    assert "contentUrl" not in json.dumps(record)
    expected_hash = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert content.content_hash == expected_hash


async def test_fetch_content_descriptor_mismatch_rejected_before_integration() -> None:
    exc_info = await _fetch_content_invalid_descriptor(
        _message_descriptor(item_type="wrong_type")
    )
    _assert_invalid_descriptor_boundary(exc_info)


async def test_fetch_permissions_unsupported() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_permissions(
            integration=_integration(fake),
            source=_source(),
            item=_message_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert exc_info.value.retryable is False
    assert fake.forbidden_calls == []
    assert fake.snapshot_calls == []
    assert fake.content_calls == []


def _noncanonical_scope_id(*, start_at: datetime, end_at: datetime) -> str:
    return _encode_canonical_payload(
        {
            "schema_version": "msgraph.teams-chat.scope.v1",
            "mailbox_user_id": _MAILBOX_USER_ID,
            "chat_remote_id": _CHAT_ID,
            "window_start_at": start_at.isoformat(),
            "window_end_at": end_at.isoformat(),
        }
    )


@pytest.mark.parametrize("revision", [" revision", "revision ", "\trevision", "revision\n"])
def test_private_revision_model_rejects_whitespace(revision: str) -> None:
    with pytest.raises(ValidationError):
        _MsGraphTeamsChatMessageRevision(
            schema_version="msgraph.teams-chat.revision.v1",
            revision=revision,
        )


@pytest.mark.parametrize("revision", [" revision", "revision ", "\trevision", "revision\n"])
def test_validate_opaque_revision_rejects_whitespace(revision: str) -> None:
    with pytest.raises(ValueError, match="whitespace"):
        _validate_opaque_revision(revision)


@pytest.mark.parametrize("padding", [" ", "\t", "\n"])
async def test_encoded_descriptor_revision_rejects_whitespace(padding: str) -> None:
    corrupted = _encode_canonical_payload(
        {
            "schema_version": "msgraph.teams-chat.revision.v1",
            "revision": f"{padding}{_ETAG_1}",
        }
    )
    exc_info = await _fetch_content_invalid_descriptor(
        KnowledgeItemDescriptor.model_construct(
            identity=KnowledgeItemIdentity.model_construct(
                remote_id=_encode_message_identity(message_remote_id=_MSG_1),
                parent_remote_id=None,
                logical_key=None,
            ),
            revision=KnowledgeItemRevision.model_construct(
                version=corrupted,
                etag=None,
                updated_at=_TS,
            ),
            title="Sprint planning",
            item_type="msgraph_teams_chat_message",
            content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
            content_available=True,
            provenance=KnowledgeItemProvenance.model_construct(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
                remote_id=_encode_message_identity(message_remote_id=_MSG_1),
            ),
            metadata=_base_metadata(),
        )
    )
    _assert_invalid_descriptor_boundary(exc_info)


@pytest.mark.parametrize("padding", [" ", "\t", "\n"])
async def test_encoded_descriptor_revision_rejects_trailing_whitespace(padding: str) -> None:
    corrupted = _encode_canonical_payload(
        {
            "schema_version": "msgraph.teams-chat.revision.v1",
            "revision": f"{_ETAG_1}{padding}",
        }
    )
    exc_info = await _fetch_content_invalid_descriptor(
        KnowledgeItemDescriptor.model_construct(
            identity=KnowledgeItemIdentity.model_construct(
                remote_id=_encode_message_identity(message_remote_id=_MSG_1),
                parent_remote_id=None,
                logical_key=None,
            ),
            revision=KnowledgeItemRevision.model_construct(
                version=corrupted,
                etag=None,
                updated_at=_TS,
            ),
            title="Sprint planning",
            item_type="msgraph_teams_chat_message",
            content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
            content_available=True,
            provenance=KnowledgeItemProvenance.model_construct(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
                remote_id=_encode_message_identity(message_remote_id=_MSG_1),
            ),
            metadata=_base_metadata(),
        )
    )
    _assert_invalid_descriptor_boundary(exc_info)


async def test_valid_exact_revision_round_trips_unchanged() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    encoded = adapter._encode_revision(_ETAG_1)
    decoded = adapter._decode_revision(encoded)
    assert decoded.revision == _ETAG_1
    assert repr(decoded) == "_MsGraphTeamsChatMessageRevision(schema_version='msgraph.teams-chat.revision.v1')"


async def test_invalid_revision_does_not_invoke_provider() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(
        content_by_key={(_MSG_1, _ETAG_1): _active_message(body_content="x")}
    )
    corrupted = _encode_canonical_payload(
        {
            "schema_version": "msgraph.teams-chat.revision.v1",
            "revision": f" {_ETAG_1}",
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=KnowledgeItemDescriptor.model_construct(
                identity=KnowledgeItemIdentity.model_construct(
                    remote_id=_encode_message_identity(message_remote_id=_MSG_1),
                    parent_remote_id=None,
                    logical_key=None,
                ),
                revision=KnowledgeItemRevision.model_construct(
                    version=corrupted,
                    etag=None,
                    updated_at=_TS,
                ),
                title="Sprint planning",
                item_type="msgraph_teams_chat_message",
                content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
                content_available=True,
                provenance=KnowledgeItemProvenance.model_construct(
                    provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                    source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
                    remote_id=_encode_message_identity(message_remote_id=_MSG_1),
                ),
                metadata=_base_metadata(),
            ),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert fake.content_calls == []
    assert fake.snapshot_calls == []
    assert fake.forbidden_calls == []


async def test_scope_encoder_normalizes_offset_to_utc() -> None:
    offset = timezone(timedelta(hours=1))
    window = MsGraphTeamsChatMessageWindow(
        start_at=datetime(2024, 1, 1, 1, 0, tzinfo=offset),
        end_at=datetime(2024, 2, 1, 1, 0, tzinfo=offset),
    )
    scope_id = encode_msgraph_teams_chat_scope_id(
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_remote_id=_CHAT_ID,
        window=window,
    )
    decoded = _MsGraphTeamsChatScope.model_validate(
        json.loads(base64.urlsafe_b64decode(scope_id + "==").decode())
    )
    assert decoded.window_start_at == datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    assert decoded.window_end_at == datetime(2024, 2, 1, 0, 0, tzinfo=timezone.utc)


async def test_noncanonical_source_payload_rejected() -> None:
    offset = timezone(timedelta(hours=1))
    noncanonical = _noncanonical_scope_id(
        start_at=datetime(2024, 1, 1, 1, 0, tzinfo=offset),
        end_at=datetime(2024, 2, 1, 1, 0, tzinfo=offset),
    )
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(snapshot_pages=[_snapshot_page(items=(_active_message(),))])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(remote_scope_id=noncanonical),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert fake.snapshot_calls == []


async def test_noncanonical_cursor_payload_rejected() -> None:
    offset = timezone(timedelta(hours=1))
    noncanonical_cursor = _encode_cursor(
        {
            "schema_version": MSGRAPH_TEAMS_CHAT_CURSOR_VERSION,
            "mailbox_user_id": _MAILBOX_USER_ID,
            "chat_remote_id": _CHAT_ID,
            "window_start_at": datetime(2024, 1, 1, 1, 0, tzinfo=offset).isoformat(),
            "window_end_at": datetime(2024, 2, 1, 1, 0, tzinfo=offset).isoformat(),
            "phase": "messages",
            "continuation_url": _NEXT_URL,
        }
    )
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(snapshot_pages=[_snapshot_page(items=(_active_message(),))])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=noncanonical_cursor,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert fake.snapshot_calls == []
    assert _SECRET_SKIP not in f"{exc_info.value!r} {exc_info.value.safe_message}"


async def test_naive_scope_timestamp_rejected() -> None:
    naive_scope = _encode_canonical_payload(
        {
            "schema_version": "msgraph.teams-chat.scope.v1",
            "mailbox_user_id": _MAILBOX_USER_ID,
            "chat_remote_id": _CHAT_ID,
            "window_start_at": "2024-01-01T00:00:00",
            "window_end_at": _WINDOW_END.isoformat(),
        }
    )
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(snapshot_pages=[_snapshot_page(items=(_active_message(),))])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(remote_scope_id=naive_scope),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert fake.snapshot_calls == []


async def test_naive_cursor_timestamp_rejected() -> None:
    naive_cursor = _encode_cursor(
        {
            "schema_version": MSGRAPH_TEAMS_CHAT_CURSOR_VERSION,
            "mailbox_user_id": _MAILBOX_USER_ID,
            "chat_remote_id": _CHAT_ID,
            "window_start_at": "2024-01-01T00:00:00",
            "window_end_at": _WINDOW_END.isoformat(),
            "phase": "messages",
            "continuation_url": _NEXT_URL,
        }
    )
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(snapshot_pages=[_snapshot_page(items=(_active_message(),))])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=naive_cursor,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert fake.snapshot_calls == []


async def test_different_actual_window_instant_rejected() -> None:
    offset = timezone(timedelta(hours=1))
    equivalent_noncanonical = _noncanonical_scope_id(
        start_at=datetime(2024, 1, 1, 1, 0, tzinfo=offset),
        end_at=datetime(2024, 2, 1, 1, 0, tzinfo=offset),
    )
    canonical = _scope_id()
    assert equivalent_noncanonical != canonical
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(snapshot_pages=[_snapshot_page(items=(_active_message(),))])
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(remote_scope_id=equivalent_noncanonical),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert fake.snapshot_calls == []


async def test_generated_cursor_stores_utc_values() -> None:
    offset = timezone(timedelta(hours=1))
    window = MsGraphTeamsChatMessageWindow(
        start_at=datetime(2024, 1, 1, 1, 0, tzinfo=offset),
        end_at=datetime(2024, 2, 1, 1, 0, tzinfo=offset),
    )
    scope_id = encode_msgraph_teams_chat_scope_id(
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_remote_id=_CHAT_ID,
        window=window,
    )
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(
        snapshot_pages=[_snapshot_page(items=(_active_message(),), continuation_url=_NEXT_URL, window=window)]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(remote_scope_id=scope_id),
        cursor=None,
        limit=50,
    )
    assert page.next_cursor is not None
    padding = "=" * (-len(page.next_cursor.value) % 4)
    decoded = _MsGraphTeamsChatCursor.model_validate(
        json.loads(base64.urlsafe_b64decode(page.next_cursor.value + padding).decode())
    )
    assert decoded.window_start_at == datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    assert decoded.window_end_at == datetime(2024, 2, 1, 0, 0, tzinfo=timezone.utc)
    assert _MAILBOX_USER_ID not in repr(decoded)
    assert _CHAT_ID not in repr(decoded)
    assert _SECRET_SKIP not in repr(decoded)


async def _fetch_permissions_invalid_descriptor(
    item: KnowledgeItemDescriptor,
) -> pytest.ExceptionInfo[VendorKnowledgeError]:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_permissions(
            integration=_integration(fake),
            source=_source(),
            item=item,
        )
    assert fake.forbidden_calls == []
    assert fake.snapshot_calls == []
    assert fake.content_calls == []
    return exc_info


@pytest.mark.parametrize(
    "item_factory",
    [
        lambda: _message_descriptor(item_type="wrong_type"),
        lambda: _message_descriptor(content_mode=KnowledgeContentMode.BINARY),
        lambda: _message_descriptor(content_available=False),
        lambda: _message_descriptor(mailbox_user_id=_OTHER_MAILBOX),
        lambda: _message_descriptor(chat_remote_id=_OTHER_CHAT_ID),
        lambda: _message_descriptor(provenance_source_kind=MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND),
        lambda: KnowledgeItemDescriptor.model_construct(
            identity=KnowledgeItemIdentity.model_construct(
                remote_id=_encode_message_identity(message_remote_id=_MSG_1),
                parent_remote_id="parent",
                logical_key=None,
            ),
            revision=KnowledgeItemRevision.model_construct(
                version=_encode_revision(_ETAG_1),
                etag=None,
                updated_at=_TS,
            ),
            title="Sprint planning",
            item_type="msgraph_teams_chat_message",
            content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
            content_available=True,
            provenance=KnowledgeItemProvenance.model_construct(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
                remote_id=_encode_message_identity(message_remote_id=_MSG_1),
            ),
            metadata=_base_metadata(),
        ),
        lambda: KnowledgeItemDescriptor.model_construct(
            identity=KnowledgeItemIdentity.model_construct(
                remote_id=_encode_message_identity(message_remote_id=_MSG_1),
                parent_remote_id=None,
                logical_key="logical",
            ),
            revision=KnowledgeItemRevision.model_construct(
                version=_encode_revision(_ETAG_1),
                etag=None,
                updated_at=_TS,
            ),
            title="Sprint planning",
            item_type="msgraph_teams_chat_message",
            content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
            content_available=True,
            provenance=KnowledgeItemProvenance.model_construct(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
                remote_id=_encode_message_identity(message_remote_id=_MSG_1),
            ),
            metadata=_base_metadata(),
        ),
        lambda: _message_descriptor(
            revision=_ETAG_1,
        ).model_copy(
            update={
                "revision": KnowledgeItemRevision.model_construct(
                    version="not-valid-base64!!!",
                    etag=None,
                    updated_at=_TS,
                )
            }
        ),
        lambda: _message_descriptor(
            revision=_ETAG_1,
        ).model_copy(
            update={
                "revision": KnowledgeItemRevision.model_construct(
                    version=_encode_canonical_payload(
                        {
                            "schema_version": "msgraph.teams-chat.revision.v1",
                            "revision": f" {_ETAG_1}",
                        }
                    ),
                    etag=None,
                    updated_at=_TS,
                )
            }
        ),
        lambda: _message_descriptor(metadata={"unexpected_key": True}, metadata_only=True),
        lambda: _message_descriptor(metadata={"message_state": "active"}, metadata_only=True),
        lambda: _message_descriptor(
            metadata={"last_modified_at": datetime(2024, 1, 16, 11, 0, tzinfo=timezone.utc).isoformat()}
        ),
        lambda: KnowledgeItemDescriptor.model_construct(
            identity=KnowledgeItemIdentity.model_construct(
                remote_id=_encode_message_identity(message_remote_id=_MSG_1),
                parent_remote_id=None,
                logical_key=None,
            ),
            revision=KnowledgeItemRevision.model_construct(
                version=_encode_revision(_ETAG_1),
                etag=None,
                updated_at=_TS,
            ),
            title="Sprint planning",
            item_type="msgraph_teams_chat_message",
            content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
            content_available=True,
            provenance=KnowledgeItemProvenance.model_construct(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
                remote_id=_encode_message_identity(message_remote_id=_MSG_1),
                web_url="https://example.test",
            ),
            metadata=_base_metadata(),
        ),
        lambda: KnowledgeItemDescriptor.model_construct(
            identity=KnowledgeItemIdentity.model_construct(
                remote_id=_encode_message_identity(message_remote_id=_MSG_1),
                parent_remote_id=None,
                logical_key=None,
            ),
            revision=KnowledgeItemRevision.model_construct(
                version=_encode_revision(_ETAG_1),
                etag=None,
                updated_at=_TS,
            ),
            title="Sprint planning",
            item_type="msgraph_teams_chat_message",
            content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
            content_available=True,
            provenance=KnowledgeItemProvenance.model_construct(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
                remote_id=_encode_message_identity(message_remote_id=_MSG_1),
                safe_locator="secret-locator",
            ),
            metadata=_base_metadata(),
        ),
        lambda: KnowledgeItemDescriptor.model_construct(
            identity=KnowledgeItemIdentity.model_construct(
                remote_id=_encode_message_identity(message_remote_id=_MSG_1),
                parent_remote_id=None,
                logical_key=None,
            ),
            revision=KnowledgeItemRevision.model_construct(
                version=_encode_revision(_ETAG_1),
                etag="etag",
                updated_at=_TS,
            ),
            title="Sprint planning",
            item_type="msgraph_teams_chat_message",
            content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
            content_available=True,
            provenance=KnowledgeItemProvenance.model_construct(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
                remote_id=_encode_message_identity(message_remote_id=_MSG_1),
            ),
            metadata=_base_metadata(),
        ),
    ],
)
async def test_fetch_permissions_rejects_invalid_descriptor(item_factory) -> None:
    exc_info = await _fetch_permissions_invalid_descriptor(item_factory())
    _assert_invalid_descriptor_boundary(exc_info)


async def test_fetch_content_message_changed_maps_to_dependency_unavailable() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()

    class _ChangedSuite(_FakeTeamsChatCollaborationSuite):
        def read_teams_chat_message_content(self, *, message, max_chars: int):
            raise MsGraphTeamsChatMessageChanged()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(_ChangedSuite()),
            source=_source(),
            item=_message_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    _assert_error_hides_secrets(exc_info.value)


async def test_fetch_content_content_too_large_maps_to_configuration_error() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()

    class _TooLargeSuite(_FakeTeamsChatCollaborationSuite):
        def read_teams_chat_message_content(self, *, message, max_chars: int):
            raise MsGraphTeamsChatContentTooLarge()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(_TooLargeSuite()),
            source=_source(),
            item=_message_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert exc_info.value.retryable is False
    _assert_error_hides_secrets(exc_info.value)


async def test_read_page_malformed_provider_page_maps_to_invalid_provider_response() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    fake = _FakeTeamsChatCollaborationSuite(
        snapshot_pages=[
            MsGraphTeamsChatMessageSnapshotPage.model_construct(
                mailbox_user_id=_MAILBOX_USER_ID,
                chat_remote_id=_CHAT_ID,
                window=_window(),
                items=(_active_message(mailbox_user_id=_OTHER_MAILBOX),),
                continuation=None,
            )
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(integration=_integration(fake), source=_source(), cursor=None, limit=50)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert exc_info.value.retryable is False
    _assert_error_hides_secrets(exc_info.value)


async def test_fetch_content_malformed_exact_content_maps_to_invalid_provider_response() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()
    bad_message = MsGraphTeamsChatMessage.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_remote_id=_CHAT_ID,
        remote_id=_MSG_1,
        revision=_ETAG_1,
        state=MsGraphTeamsChatMessageState.ACTIVE,
        message_type=MsGraphTeamsChatMessageType.MESSAGE,
        importance=MsGraphTeamsChatImportance.NORMAL,
        created_at=_CREATED_TS,
        last_modified_at=_TS,
        body_kind=MsGraphTeamsChatBodyKind.TEXT,
        body_content=None,
    )
    fake = _FakeTeamsChatCollaborationSuite(content_by_key={(_MSG_1, _ETAG_1): bad_message})
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=_message_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert exc_info.value.retryable is False
    _assert_error_hides_secrets(exc_info.value)


def _assert_error_hides_secrets(err: VendorKnowledgeError) -> None:
    rendered = f"{err!r} {err.safe_message}"
    for secret in (
        _MAILBOX_USER_ID,
        _CHAT_ID,
        _MSG_1,
        _ETAG_1,
        _SECRET_SKIP,
        _NEXT_URL,
        "skiptoken",
        "Message body",
        "body-one",
    ):
        assert secret not in rendered


async def test_integration_dependency_error_translated() -> None:
    adapter = MsGraphTeamsChatKnowledgeAdapter()

    class _BrokenSuite(CollaborationSuite):
        def read_teams_chat_messages_snapshot_page_by_reference(self, **kwargs: Any):
            raise IntegrationDependencyError("boom")

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

        def create_event(self, user_id: str, *, subject: str, start: str, end: str, location: str = "", attendees=()):
            raise NotImplementedError

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_BrokenSuite()), source=_source(), cursor=None, limit=50
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True


def test_production_file_has_no_assertions() -> None:
    adapter_path = (
        Path(__file__).resolve().parents[4]
        / "intergrax/runtime/vendor_knowledge/adapters/ms365_graph_teams_chat.py"
    )
    text = adapter_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if " assert " in f" {line} " or stripped.startswith("assert "):
            raise AssertionError(f"assert found in production adapter: {line}")


def test_production_and_tests_have_no_getattr_setattr_hasattr() -> None:
    root = Path(__file__).resolve().parents[4]
    target = root / "intergrax/runtime/vendor_knowledge/adapters/ms365_graph_teams_chat.py"
    text = target.read_text(encoding="utf-8")
    assert "getattr(" not in text
    assert "setattr(" not in text
    assert "hasattr(" not in text
