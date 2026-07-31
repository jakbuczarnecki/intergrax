# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for MsGraphTeamsChannelKnowledgeAdapter."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
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
    DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    MSGRAPH_MAIL_SOURCE_KIND,
    MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphTeamsChannelBodyKind,
    MsGraphTeamsChannelContentTooLarge,
    MsGraphTeamsChannelImportance,
    MsGraphTeamsChannelMessage,
    MsGraphTeamsChannelMessageChanged,
    MsGraphTeamsChannelMessageKind,
    MsGraphTeamsChannelMessageState,
    MsGraphTeamsChannelMessageType,
    MsGraphTeamsChannelReplyPage,
    MsGraphTeamsChannelRootMessagePage,
    MsGraphTeamsChatAttachmentKind,
    MsGraphTeamsChatAttachmentReference,
    MsGraphTeamsChatMention,
    MsGraphTeamsChatReaction,
    MsGraphTeamsIdentity,
    MsGraphTeamsIdentityKind,
)
from intergrax.runtime.vendor_knowledge.adapters.confluence_pages import (
    ConfluencePagesKnowledgeAdapter,
    register_confluence_pages_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.jira_issues import (
    JiraIssuesKnowledgeAdapter,
    register_jira_issues_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_drive import (
    MSGRAPH_DRIVE_SOURCE_KIND,
    MSGRAPH_DRIVE_SCOPE_TYPE,
    MsGraphDriveKnowledgeAdapter,
    register_msgraph_drive_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_mail import (
    MSGRAPH_MAIL_SCOPE_TYPE,
    MsGraphMailKnowledgeAdapter,
    register_msgraph_mail_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_channel import (
    MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
    MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE,
    MsGraphTeamsChannelKnowledgeAdapter,
    encode_msgraph_teams_channel_scope_id,
    register_msgraph_teams_channel_knowledge_adapter,
)
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

_TEAM_ID = "team-abc-123"
_CHANNEL_ID = "channel-abc-123"
_OTHER_TEAM_ID = "other-team-456"
_OTHER_CHANNEL_ID = "other-channel-456"
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
_STRUCTURED_RECORD_SCHEMA = "msgraph.teams-channel.message.knowledge.v1"
_STRUCTURED_RECORD_MIME = "application/vnd.intergrax.msgraph-teams-channel-message+json"
_QUOTED_TEAM = quote(_TEAM_ID, safe="")
_QUOTED_CHANNEL = quote(_CHANNEL_ID, safe="")
_QUOTED_OTHER_TEAM = quote(_OTHER_TEAM_ID, safe="")
_QUOTED_OTHER_CHANNEL = quote(_OTHER_CHANNEL_ID, safe="")
_ROOT_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
    f"{_QUOTED_CHANNEL}/messages?$skiptoken={_SECRET_SKIP}"
)
_REPLY_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
    f"{_QUOTED_CHANNEL}/messages/{quote(_ROOT_1, safe='')}/replies?$skiptoken={_SECRET_SKIP}"
)
_OTHER_ROOT_URL = (
    f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_OTHER_TEAM}/channels/"
    f"{_QUOTED_OTHER_CHANNEL}/messages?$skiptoken=other"
)


def _encode_canonical_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _scope_id(
    *,
    team_remote_id: str = _TEAM_ID,
    channel_remote_id: str = _CHANNEL_ID,
) -> str:
    return encode_msgraph_teams_channel_scope_id(
        team_remote_id=team_remote_id,
        channel_remote_id=channel_remote_id,
    )


def _encode_message_identity(
    *,
    message_remote_id: str,
    message_kind: str,
    thread_root_remote_id: str | None = None,
    team_remote_id: str = _TEAM_ID,
    channel_remote_id: str = _CHANNEL_ID,
) -> str:
    resolved_thread_root = thread_root_remote_id or message_remote_id
    return _encode_canonical_payload(
        {
            "schema_version": "msgraph.teams-channel.message-id.v1",
            "team_remote_id": team_remote_id,
            "channel_remote_id": channel_remote_id,
            "thread_root_remote_id": resolved_thread_root,
            "message_kind": message_kind,
            "message_remote_id": message_remote_id,
        }
    )


def _encode_revision(revision: str) -> str:
    return _encode_canonical_payload(
        {
            "schema_version": "msgraph.teams-channel.revision.v1",
            "revision": revision,
        }
    )


def _source(
    *,
    remote_scope_id: str | None = None,
    remote_scope_type: str = MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE,
    parameters: dict[str, Any] | None = None,
    provider_id: str = MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    integration_kind: IntegrationCategory = IntegrationCategory.COLLABORATION_SUITE,
    source_kind: str = MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        scope=KnowledgeSourceScope(
            remote_scope_id=remote_scope_id or _scope_id(),
            remote_scope_type=remote_scope_type,
            safe_display_name="General",
            parameters=parameters or {},
        ),
    )


def _active_root(
    *,
    remote_id: str = _ROOT_1,
    revision: str = _ETAG_ROOT_1,
    subject: str | None = "Sprint planning",
    body_kind: MsGraphTeamsChannelBodyKind = MsGraphTeamsChannelBodyKind.TEXT,
    body_content: str = "Root body",
    sender: MsGraphTeamsIdentity | None = None,
    mentions: tuple[MsGraphTeamsChatMention, ...] = (),
    reactions: tuple[MsGraphTeamsChatReaction, ...] = (),
    attachments: tuple[MsGraphTeamsChatAttachmentReference, ...] = (),
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
        body_kind=body_kind,
        body_content=body_content,
        sender=sender,
        mentions=mentions,
        reactions=reactions,
        attachments=attachments,
    )


def _active_reply(
    *,
    remote_id: str = _REPLY_1,
    revision: str = _ETAG_REPLY_1,
    thread_root_remote_id: str = _ROOT_1,
    subject: str | None = "Follow-up",
    body_content: str = "Reply body",
) -> MsGraphTeamsChannelMessage:
    return MsGraphTeamsChannelMessage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        thread_root_remote_id=thread_root_remote_id,
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


def _deleted_root(
    *,
    remote_id: str = _ROOT_2,
    revision: str = _ETAG_ROOT_2,
) -> MsGraphTeamsChannelMessage:
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


def _deleted_reply(
    *,
    remote_id: str = _REPLY_2,
    revision: str = _ETAG_REPLY_2,
    thread_root_remote_id: str = _ROOT_1,
) -> MsGraphTeamsChannelMessage:
    return MsGraphTeamsChannelMessage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        thread_root_remote_id=thread_root_remote_id,
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


class _FakeTeamsChannelCollaborationSuite(CollaborationSuite):
    def __init__(
        self,
        *,
        root_pages: list[MsGraphTeamsChannelRootMessagePage] | None = None,
        reply_pages: dict[tuple[str, str], list[MsGraphTeamsChannelReplyPage]] | None = None,
        content_by_key: dict[tuple[str, str], MsGraphTeamsChannelMessage] | None = None,
    ) -> None:
        self._root_pages = list(root_pages or [])
        self._reply_pages = {
            key: list(pages) for key, pages in (reply_pages or {}).items()
        }
        self._content_by_key = dict(content_by_key or {})
        self.root_calls: list[dict[str, Any]] = []
        self.reply_calls: list[dict[str, Any]] = []
        self.content_calls: list[dict[str, Any]] = []
        self.forbidden_calls: list[str] = []

    def read_teams_channel_root_messages_page_by_reference(
        self,
        *,
        channel,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int,
        max_chars_per_message: int,
    ) -> MsGraphTeamsChannelRootMessagePage:
        self.root_calls.append(
            {
                "channel": channel,
                "continuation": continuation,
                "limit": limit,
                "max_chars_per_message": max_chars_per_message,
            }
        )
        if not self._root_pages:
            raise IntegrationDependencyError("no root pages configured")
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
            raise IntegrationDependencyError("no reply pages configured")
        return pages.pop(0)

    def read_teams_channel_message_content(
        self,
        *,
        message,
        max_chars: int,
    ) -> MsGraphTeamsChannelMessage:
        self.content_calls.append({"message": message, "max_chars": max_chars})
        return self._content_by_key[(message.remote_id, message.revision)]

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


def _integration(
    fake: _FakeTeamsChannelCollaborationSuite,
) -> Ms365GraphCollaborationSuiteIntegration:
    return _TeamsChannelTestIntegration.from_client(fake, enabled=True)


def _encode_cursor(payload: dict[str, Any]) -> KnowledgeCursor:
    return KnowledgeCursor(
        value=_encode_canonical_payload(payload),
        version=MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
    )


def _base_metadata(
    *,
    message_kind: str,
    body_kind: str = "text",
    has_attachments: bool = False,
) -> dict[str, object]:
    return {
        "message_state": "active",
        "message_kind": message_kind,
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
    message_remote_id: str = _ROOT_1,
    message_kind: str = "root",
    revision: str = _ETAG_ROOT_1,
    thread_root_remote_id: str | None = None,
    title: str = "Sprint planning",
    team_remote_id: str = _TEAM_ID,
    channel_remote_id: str = _CHANNEL_ID,
    metadata: dict[str, Any] | None = None,
    metadata_only: bool = False,
    content_available: bool = True,
    item_type: str = "msgraph_teams_channel_message",
    content_mode: KnowledgeContentMode = KnowledgeContentMode.STRUCTURED_RECORD,
    provenance_source_kind: str = MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
    parent_remote_id: str | None = None,
) -> KnowledgeItemDescriptor:
    resolved_thread_root = thread_root_remote_id or message_remote_id
    opaque_id = _encode_message_identity(
        message_remote_id=message_remote_id,
        message_kind=message_kind,
        thread_root_remote_id=resolved_thread_root,
        team_remote_id=team_remote_id,
        channel_remote_id=channel_remote_id,
    )
    if parent_remote_id is None and message_kind == "reply":
        parent_remote_id = _encode_message_identity(
            message_remote_id=resolved_thread_root,
            message_kind="root",
            thread_root_remote_id=resolved_thread_root,
            team_remote_id=team_remote_id,
            channel_remote_id=channel_remote_id,
        )
    base_metadata = _base_metadata(message_kind=message_kind)
    if metadata is not None:
        resolved_metadata = metadata if metadata_only else {**base_metadata, **metadata}
    else:
        resolved_metadata = base_metadata
    return KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(
            remote_id=opaque_id,
            parent_remote_id=parent_remote_id,
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


def _assert_invalid_descriptor_boundary(
    exc_info: pytest.ExceptionInfo[VendorKnowledgeError],
) -> None:
    err = exc_info.value
    assert err.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert err.retryable is False
    assert err.__cause__ is None
    rendered = f"{err!r} {err.safe_message}"
    for secret in (
        _TEAM_ID,
        _CHANNEL_ID,
        _ROOT_1,
        _ETAG_ROOT_1,
        "Sprint planning",
        "2024-01-01",
        "graph.microsoft.com",
        "Authorization",
    ):
        assert secret not in rendered


async def _fetch_content_invalid_descriptor(
    item: KnowledgeItemDescriptor,
) -> pytest.ExceptionInfo[VendorKnowledgeError]:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        content_by_key={(_ROOT_1, _ETAG_ROOT_1): _active_root(body_content="x")}
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=item,
        )
    assert fake.content_calls == []
    _assert_invalid_descriptor_boundary(exc_info)
    return exc_info


def _complete_cursor() -> KnowledgeCursor:
    return _encode_cursor(
        {
            "schema_version": MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
            "team_remote_id": _TEAM_ID,
            "channel_remote_id": _CHANNEL_ID,
            "phase": "complete",
        }
    )


# --- 1. Identity, capabilities, registration ---


async def test_adapter_identity() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    assert adapter.provider_id == MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    assert adapter.integration_kind is IntegrationCategory.COLLABORATION_SUITE
    assert adapter.source_kind == MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND


async def test_capabilities_exact_set() -> None:
    caps = MsGraphTeamsChannelKnowledgeAdapter().capabilities
    assert caps.full_inventory is True
    assert caps.incremental_changes is False
    assert caps.content_fetch is True
    assert caps.binary_content is False
    assert caps.rich_text_content is False
    assert caps.structured_content is True
    assert caps.permissions is False
    assert caps.tombstones is True
    assert caps.remote_versions is True
    assert caps.reconciliation is True


async def test_registry_registration_and_coexistence() -> None:
    registry = KnowledgeAdapterRegistry()
    teams_adapter = register_msgraph_teams_channel_knowledge_adapter(registry)
    mail_adapter = register_msgraph_mail_knowledge_adapter(registry)
    drive_adapter = register_msgraph_drive_knowledge_adapter(registry)
    register_jira_issues_knowledge_adapter(registry)
    register_confluence_pages_knowledge_adapter(registry)

    assert isinstance(teams_adapter, MsGraphTeamsChannelKnowledgeAdapter)
    assert isinstance(
        registry.resolve(source=_source()),
        MsGraphTeamsChannelKnowledgeAdapter,
    )
    assert isinstance(
        registry.resolve(
            source=KnowledgeSourceRef(
                tenant_id="tenant-1",
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                source_kind=MSGRAPH_MAIL_SOURCE_KIND,
                scope=KnowledgeSourceScope(
                    remote_scope_id="scope-mail",
                    remote_scope_type=MSGRAPH_MAIL_SCOPE_TYPE,
                    safe_display_name="Inbox",
                    parameters={},
                ),
            )
        ),
        MsGraphMailKnowledgeAdapter,
    )
    assert isinstance(
        registry.resolve(
            source=KnowledgeSourceRef(
                tenant_id="tenant-1",
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                source_kind=MSGRAPH_DRIVE_SOURCE_KIND,
                scope=KnowledgeSourceScope(
                    remote_scope_id="drive-1",
                    remote_scope_type=MSGRAPH_DRIVE_SCOPE_TYPE,
                    safe_display_name="Drive",
                    parameters={},
                ),
            )
        ),
        MsGraphDriveKnowledgeAdapter,
    )
    assert isinstance(
        registry.resolve(
            source=KnowledgeSourceRef(
                tenant_id="tenant-1",
                provider_id="jira",
                integration_kind=IntegrationCategory.ISSUE_TRACKER,
                source_kind="issues",
                scope=KnowledgeSourceScope(
                    remote_scope_id="PROJ",
                    remote_scope_type="jira_project",
                    safe_display_name="Project",
                    parameters={},
                ),
            )
        ),
        JiraIssuesKnowledgeAdapter,
    )
    assert isinstance(
        registry.resolve(
            source=KnowledgeSourceRef(
                tenant_id="tenant-1",
                provider_id="confluence",
                integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
                source_kind="pages",
                scope=KnowledgeSourceScope(
                    remote_scope_id="10000",
                    remote_scope_type="confluence_space",
                    safe_display_name="Space",
                    parameters={},
                ),
            )
        ),
        ConfluencePagesKnowledgeAdapter,
    )
    assert drive_adapter.source_kind == MSGRAPH_DRIVE_SOURCE_KIND
    assert mail_adapter.source_kind == MSGRAPH_MAIL_SOURCE_KIND


async def test_duplicate_registry_registration_rejected() -> None:
    registry = KnowledgeAdapterRegistry()
    register_msgraph_teams_channel_knowledge_adapter(registry)
    with pytest.raises(ValueError):
        register_msgraph_teams_channel_knowledge_adapter(registry)


# --- 2. Scope ---


async def test_valid_channel_scope_inspect() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite()
    info = await adapter.inspect_scope(integration=_integration(fake), source=_source())
    assert info.safe_display_name == "General"
    assert fake.root_calls == []


async def test_encoded_scope_id_hides_raw_ids() -> None:
    encoded = _scope_id()
    assert _TEAM_ID not in encoded
    assert _CHANNEL_ID not in encoded


@pytest.mark.parametrize(
    ("source",),
    [
        (_source(provider_id="jira"),),
        (_source(integration_kind=IntegrationCategory.ISSUE_TRACKER),),
        (_source(source_kind=MSGRAPH_MAIL_SOURCE_KIND),),
        (_source(source_kind=MSGRAPH_DRIVE_SOURCE_KIND),),
        (_source(remote_scope_type="sharepoint"),),
        (_source(parameters={"team": "x"}),),
    ],
)
async def test_invalid_scope_rejected(source: KnowledgeSourceRef) -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(_FakeTeamsChannelCollaborationSuite()),
            source=source,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


@pytest.mark.parametrize(
    ("remote_scope_id",),
    [
        ("not-base64",),
        ("!!!",),
        (_encode_canonical_payload({"bad": 1}),),
        (
            _encode_canonical_payload(
                {
                    "schema_version": "msgraph.teams-channel.scope.v1",
                    "team_remote_id": "",
                    "channel_remote_id": _CHANNEL_ID,
                }
            ),
        ),
        (
            _encode_canonical_payload(
                {
                    "schema_version": "wrong.schema.v1",
                    "team_remote_id": _TEAM_ID,
                    "channel_remote_id": _CHANNEL_ID,
                }
            ),
        ),
    ],
)
async def test_malformed_scope_payload_rejected(remote_scope_id: str) -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(_FakeTeamsChannelCollaborationSuite()),
            source=_source(remote_scope_id=remote_scope_id),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_wrong_integration_object_rejected() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(integration=object(), source=_source())
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


# --- 3. Limits ---


@pytest.mark.parametrize("limit", [1, 50, 100, 1000])
async def test_reply_limit_accepted_and_clamped(limit: int) -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[_root_page(items=(_active_root(),))],
        reply_pages={
            (_ROOT_1, _ETAG_ROOT_1): [
                _reply_page(
                    items=(_active_reply(),),
                    root_message_remote_id=_ROOT_1,
                    root_message_revision=_ETAG_ROOT_1,
                )
            ]
        },
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=limit,
    )
    assert fake.root_calls[0]["limit"] == 1
    reply_cursor = page.next_cursor
    assert reply_cursor is not None
    await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=reply_cursor,
        limit=limit,
    )
    assert fake.reply_calls[0]["limit"] == min(limit, 50)


@pytest.mark.parametrize("limit", [0, 1001, True, False, "10", 10.5, None])
async def test_invalid_limit_rejected(limit: object) -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FakeTeamsChannelCollaborationSuite()),
            source=_source(),
            cursor=None,
            limit=limit,  # type: ignore[arg-type]
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR


# --- 4. Root phase ---


async def test_root_phase_returns_single_upsert_and_reply_cursor() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[
            _root_page(items=(_active_root(),), continuation_url=_ROOT_NEXT_URL),
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    assert fake.root_calls[0]["limit"] == 1
    assert len(page.changes) == 1
    change = page.changes[0]
    assert change.kind is KnowledgeChangeKind.UPSERT
    assert change.descriptor is not None
    assert change.descriptor.title == "Sprint planning"
    assert page.has_more is True
    assert page.next_cursor is not None
    assert page.proposed_checkpoint == page.next_cursor
    assert _SECRET_SKIP not in page.next_cursor.value


async def test_root_phase_deleted_tombstone() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[_root_page(items=(_deleted_root(),))]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    change = page.changes[0]
    assert change.kind is KnowledgeChangeKind.DELETED
    assert change.descriptor is None
    assert page.has_more is True


@pytest.mark.parametrize(
    ("subject", "expected_title"),
    [
        ("Budget Q2", "Budget Q2"),
        (None, "Teams channel post"),
        ("", "Teams channel post"),
        ("   ", "Teams channel post"),
    ],
)
async def test_root_title_resolution(subject: str | None, expected_title: str) -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[_root_page(items=(_active_root(subject=subject),))]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.title == expected_title


async def test_empty_root_page_with_continuation_advances() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[
            _root_page(items=(), continuation_url=_ROOT_NEXT_URL),
            _root_page(items=(_active_root(),)),
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    assert page.changes == ()
    assert page.has_more is True
    assert page.next_cursor is not None


async def test_empty_root_page_without_continuation_completes() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[_root_page(items=(), continuation_url=None)]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    assert page.changes == ()
    assert page.has_more is False
    assert page.next_cursor is None
    assert page.proposed_checkpoint is not None


async def test_root_continuation_resumed_from_roots_cursor() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[_root_page(items=(_active_root(remote_id=_ROOT_2),))]
    )
    cursor = _encode_cursor(
        {
            "schema_version": MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
            "team_remote_id": _TEAM_ID,
            "channel_remote_id": _CHANNEL_ID,
            "phase": "roots",
            "resume_root_continuation_url": _ROOT_NEXT_URL,
        }
    )
    await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=cursor,
        limit=50,
    )
    continuation = fake.root_calls[0]["continuation"]
    assert continuation is not None
    assert continuation.kind == MsGraphKnowledgeContinuationKind.NEXT_PAGE
    assert continuation.url == _ROOT_NEXT_URL


async def test_root_page_more_than_one_item_rejected() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[
            MsGraphTeamsChannelRootMessagePage.model_construct(
                team_remote_id=_TEAM_ID,
                channel_remote_id=_CHANNEL_ID,
                items=(_active_root(), _active_root(remote_id=_ROOT_2)),
                continuation=None,
            )
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


# --- 5. Reply phase ---


async def test_reply_phase_parent_ids_and_continuation() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[_root_page(items=(_active_root(),), continuation_url=_ROOT_NEXT_URL)],
        reply_pages={
            (_ROOT_1, _ETAG_ROOT_1): [
                _reply_page(
                    items=(_active_reply(),),
                    root_message_remote_id=_ROOT_1,
                    root_message_revision=_ETAG_ROOT_1,
                    continuation_url=_REPLY_NEXT_URL,
                )
            ]
        },
    )
    root_page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    reply_page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=root_page.next_cursor,
        limit=75,
    )
    assert fake.reply_calls[0]["limit"] == 50
    descriptor = reply_page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.identity.parent_remote_id == _encode_message_identity(
        message_remote_id=_ROOT_1,
        message_kind="root",
    )
    assert reply_page.has_more is True
    assert reply_page.next_cursor is not None
    continuation = fake.reply_calls[0]["continuation"]
    assert continuation is None


async def test_reply_phase_resumes_with_reply_continuation_url() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        reply_pages={
            (_ROOT_1, _ETAG_ROOT_1): [
                _reply_page(
                    items=(_active_reply(remote_id=_REPLY_2),),
                    root_message_remote_id=_ROOT_1,
                    root_message_revision=_ETAG_ROOT_1,
                )
            ]
        }
    )
    cursor = _encode_cursor(
        {
            "schema_version": MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
            "team_remote_id": _TEAM_ID,
            "channel_remote_id": _CHANNEL_ID,
            "phase": "replies",
            "resume_root_continuation_url": _ROOT_NEXT_URL,
            "root_message_remote_id": _ROOT_1,
            "root_message_revision": _ETAG_ROOT_1,
            "root_message_state": "active",
            "reply_continuation_url": _REPLY_NEXT_URL,
        }
    )
    await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=cursor,
        limit=50,
    )
    continuation = fake.reply_calls[0]["continuation"]
    assert continuation is not None
    assert continuation.url == _REPLY_NEXT_URL


async def test_reply_phase_complete_after_last_root() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        reply_pages={
            (_ROOT_1, _ETAG_ROOT_1): [
                _reply_page(
                    items=(_active_reply(),),
                    root_message_remote_id=_ROOT_1,
                    root_message_revision=_ETAG_ROOT_1,
                )
            ]
        }
    )
    cursor = _encode_cursor(
        {
            "schema_version": MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
            "team_remote_id": _TEAM_ID,
            "channel_remote_id": _CHANNEL_ID,
            "phase": "replies",
            "resume_root_continuation_url": None,
            "root_message_remote_id": _ROOT_1,
            "root_message_revision": _ETAG_ROOT_1,
            "root_message_state": "active",
            "reply_continuation_url": None,
        }
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=cursor,
        limit=50,
    )
    assert page.has_more is False
    assert page.next_cursor is None


async def test_reply_phase_resumes_roots_after_replies_exhausted() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        reply_pages={
            (_ROOT_1, _ETAG_ROOT_1): [
                _reply_page(
                    items=(_active_reply(),),
                    root_message_remote_id=_ROOT_1,
                    root_message_revision=_ETAG_ROOT_1,
                )
            ]
        }
    )
    cursor = _encode_cursor(
        {
            "schema_version": MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
            "team_remote_id": _TEAM_ID,
            "channel_remote_id": _CHANNEL_ID,
            "phase": "replies",
            "resume_root_continuation_url": _ROOT_NEXT_URL,
            "root_message_remote_id": _ROOT_1,
            "root_message_revision": _ETAG_ROOT_1,
            "root_message_state": "active",
            "reply_continuation_url": None,
        }
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=cursor,
        limit=50,
    )
    assert page.has_more is True
    assert page.next_cursor is not None
    assert _ROOT_NEXT_URL not in page.next_cursor.value


async def test_reply_title_fallback() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[_root_page(items=(_active_root(),))],
        reply_pages={
            (_ROOT_1, _ETAG_ROOT_1): [
                _reply_page(
                    items=(_active_reply(subject=None),),
                    root_message_remote_id=_ROOT_1,
                    root_message_revision=_ETAG_ROOT_1,
                )
            ]
        },
    )
    root_page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    reply_page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=root_page.next_cursor,
        limit=50,
    )
    descriptor = reply_page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.title == "Teams channel reply"


async def test_reply_deleted_tombstone() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[_root_page(items=(_active_root(),))],
        reply_pages={
            (_ROOT_1, _ETAG_ROOT_1): [
                _reply_page(
                    items=(_deleted_reply(),),
                    root_message_remote_id=_ROOT_1,
                    root_message_revision=_ETAG_ROOT_1,
                )
            ]
        },
    )
    root_page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    reply_page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=root_page.next_cursor,
        limit=50,
    )
    change = reply_page.changes[0]
    assert change.kind is KnowledgeChangeKind.DELETED
    assert change.descriptor is None


# --- 6. Cursor boundary ---


async def test_complete_cursor_rejected() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FakeTeamsChannelCollaborationSuite()),
            source=_source(),
            cursor=_complete_cursor(),
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR


@pytest.mark.parametrize(
    ("cursor",),
    [
        (KnowledgeCursor(value="not-base64", version=MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION),),
        (KnowledgeCursor(value="!!!", version=MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION),),
        (KnowledgeCursor(value=_encode_cursor({"bad": 1}).value, version="other.v1"),),
        (
            KnowledgeCursor(
                value=_encode_cursor(
                    {
                        "schema_version": MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
                        "team_remote_id": _TEAM_ID,
                        "channel_remote_id": _CHANNEL_ID,
                        "phase": "roots",
                    }
                ).value,
                version=MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
            ),
        ),
        (
            KnowledgeCursor(
                value=_encode_cursor(
                    {
                        "schema_version": MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
                        "team_remote_id": _TEAM_ID,
                        "channel_remote_id": _CHANNEL_ID,
                        "phase": "replies",
                        "resume_root_continuation_url": _ROOT_NEXT_URL,
                    }
                ).value,
                version=MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
            ),
        ),
        (
            KnowledgeCursor(
                value=_encode_cursor(
                    {
                        "schema_version": MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
                        "team_remote_id": _OTHER_TEAM_ID,
                        "channel_remote_id": _OTHER_CHANNEL_ID,
                        "phase": "roots",
                        "resume_root_continuation_url": _OTHER_ROOT_URL,
                    }
                ).value,
                version=MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
            ),
        ),
        (
            KnowledgeCursor(
                value=_encode_cursor(
                    {
                        "schema_version": MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
                        "team_remote_id": _TEAM_ID,
                        "channel_remote_id": _CHANNEL_ID,
                        "phase": "complete",
                        "resume_root_continuation_url": _ROOT_NEXT_URL,
                    }
                ).value,
                version=MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
            ),
        ),
    ],
)
async def test_invalid_cursor_rejected(cursor: KnowledgeCursor) -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FakeTeamsChannelCollaborationSuite()),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR


async def test_cursor_url_hidden_from_repr() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[
            _root_page(items=(_active_root(),), continuation_url=_ROOT_NEXT_URL),
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    assert page.next_cursor is not None
    rendered = repr(page.next_cursor)
    assert _SECRET_SKIP not in rendered
    assert _ROOT_NEXT_URL not in rendered


# --- 7. Descriptor mapping ---


async def test_message_descriptor_mapping() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[_root_page(items=(_active_root(),))]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.identity.parent_remote_id is None
    assert descriptor.title == "Sprint planning"
    assert descriptor.item_type == "msgraph_teams_channel_message"
    assert descriptor.content_mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert descriptor.content_available is True
    assert descriptor.revision.updated_at == _TS
    assert descriptor.metadata is not None
    assert descriptor.metadata["message_state"] == "active"
    assert descriptor.metadata["message_kind"] == "root"
    assert descriptor.metadata["body_kind"] == "text"
    assert descriptor.metadata["attachment_inventory_in_content"] is True
    assert descriptor.metadata["attachment_binary_content_included"] is False
    assert descriptor.metadata["hosted_content_included"] is False
    assert descriptor.metadata["reference_urls_included"] is False
    assert descriptor.provenance.web_url is None
    assert _TEAM_ID not in repr(descriptor.metadata)
    assert _CHANNEL_ID not in repr(descriptor.metadata)
    assert _ETAG_ROOT_1 not in descriptor.revision.version
    assert _ROOT_1 not in descriptor.identity.remote_id


# --- 8. Provider response boundary ---


async def test_wrong_page_type_rejected() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()

    class _BadRootSuite(_FakeTeamsChannelCollaborationSuite):
        def read_teams_channel_root_messages_page_by_reference(self, **kwargs):
            return object()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_BadRootSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_cross_team_item_rejected() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[
            _root_page(items=(_active_root(),)).model_copy(
                update={"items": (_active_root().model_copy(update={"team_remote_id": _OTHER_TEAM_ID}),)}
            )
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_duplicate_reply_remote_id_rejected() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    item = _active_reply()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[_root_page(items=(_active_root(),))],
        reply_pages={
            (_ROOT_1, _ETAG_ROOT_1): [
                MsGraphTeamsChannelReplyPage.model_construct(
                    team_remote_id=_TEAM_ID,
                    channel_remote_id=_CHANNEL_ID,
                    root_message_remote_id=_ROOT_1,
                    root_message_revision=_ETAG_ROOT_1,
                    items=(item, item),
                    continuation=None,
                )
            ]
        },
    )
    root_page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=root_page.next_cursor,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_invalid_continuation_url_rejected() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[
            _root_page(items=(_active_root(),), continuation_url=_OTHER_ROOT_URL),
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(fake),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


# --- 9. Content fetch ---


async def test_fetch_content_structured_record() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    sender = MsGraphTeamsIdentity(
        identity_kind=MsGraphTeamsIdentityKind.USER,
        remote_id="sender-1",
        display_name="Alice",
        tenant_id="tenant-1",
        identity_type="user",
    )
    mention = MsGraphTeamsChatMention(
        mention_id=0,
        mention_text="@Alice",
        mentioned=sender,
    )
    reaction = MsGraphTeamsChatReaction(
        reaction_type="like",
        display_name="Like",
        created_at=_CREATED_TS,
        user=sender,
    )
    attachment = MsGraphTeamsChatAttachmentReference(
        remote_id="att-1",
        attachment_kind=MsGraphTeamsChatAttachmentKind.REFERENCE,
        content_type="reference",
        name="doc.pdf",
        content_url="https://contoso.example/secret-file",
        has_thumbnail_url=False,
    )
    message = _active_root(
        body_kind=MsGraphTeamsChannelBodyKind.HTML,
        body_content="<p>Hello</p>",
        sender=sender,
        mentions=(mention,),
        reactions=(reaction,),
        attachments=(attachment,),
    )
    fake = _FakeTeamsChannelCollaborationSuite(
        content_by_key={(_ROOT_1, _ETAG_ROOT_1): message}
    )
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_message_descriptor(
            metadata=_base_metadata(message_kind="root", body_kind="html", has_attachments=True),
        ),
    )
    assert content.mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert content.mime_type == _STRUCTURED_RECORD_MIME
    record = content.structured_record
    assert record is not None
    assert record["schema"] == _STRUCTURED_RECORD_SCHEMA
    assert record["body"] == {"kind": "html", "content": "<p>Hello</p>"}
    assert record["sender"]["display_name"] == "Alice"
    assert record["mentions"][0]["text"] == "@Alice"
    assert record["reactions"][0]["type"] == "like"
    attachment_item = record["attachments"]["items"][0]
    assert attachment_item["remote_id"] == "att-1"
    assert "content_url" not in attachment_item
    assert record["attachments"]["reference_urls_included"] is False
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    assert content.content_hash == hashlib.sha256(canonical).hexdigest()
    assert fake.content_calls[0]["max_chars"] == DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS


async def test_fetch_content_reply_descriptor() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        content_by_key={
            (_REPLY_1, _ETAG_REPLY_1): _active_reply(body_content="Reply text"),
        }
    )
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_message_descriptor(
            message_remote_id=_REPLY_1,
            message_kind="reply",
            revision=_ETAG_REPLY_1,
            thread_root_remote_id=_ROOT_1,
            title="Follow-up",
            metadata=_base_metadata(message_kind="reply"),
        ),
    )
    assert content.structured_record is not None
    assert content.structured_record["message_kind"] == "reply"


async def test_fetch_content_returns_fresh_dicts() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        content_by_key={(_ROOT_1, _ETAG_ROOT_1): _active_root(body_content="Hello")}
    )
    item = _message_descriptor()
    first = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=item,
    )
    second = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=item,
    )
    assert first.structured_record == second.structured_record
    assert first.structured_record is not second.structured_record


async def test_fetch_content_provider_revision_mismatch() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        content_by_key={
            (_ROOT_1, _ETAG_ROOT_1): _active_root(revision="other-revision"),
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=_message_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.__cause__ is None


async def test_content_too_large_translated() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()

    class _HugeContentSuite(_FakeTeamsChannelCollaborationSuite):
        def read_teams_channel_message_content(self, *, message, max_chars):
            raise MsGraphTeamsChannelContentTooLarge()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(_HugeContentSuite()),
            source=_source(),
            item=_message_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert (
        exc_info.value.safe_message
        == "Microsoft Graph Teams Channel message exceeds the configured content limit"
    )


async def test_message_changed_during_read_translated() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()

    class _ChangedSuite(_FakeTeamsChannelCollaborationSuite):
        def read_teams_channel_message_content(self, *, message, max_chars):
            raise MsGraphTeamsChannelMessageChanged()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(_ChangedSuite()),
            source=_source(),
            item=_message_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE


# --- 10. Invalid descriptor boundary ---


async def test_fetch_content_rejects_wrong_item_type() -> None:
    await _fetch_content_invalid_descriptor(
        _message_descriptor(item_type="msgraph_drive_file"),
    )


async def test_fetch_content_rejects_wrong_content_mode() -> None:
    await _fetch_content_invalid_descriptor(
        _message_descriptor(content_mode=KnowledgeContentMode.BINARY),
    )


async def test_fetch_content_rejects_content_unavailable() -> None:
    await _fetch_content_invalid_descriptor(
        _message_descriptor(content_available=False),
    )


async def test_fetch_content_rejects_wrong_provenance() -> None:
    await _fetch_content_invalid_descriptor(
        _message_descriptor(provenance_source_kind=MSGRAPH_DRIVE_SOURCE_KIND),
    )


async def test_fetch_content_rejects_cross_team_identity() -> None:
    await _fetch_content_invalid_descriptor(
        _message_descriptor(team_remote_id=_OTHER_TEAM_ID),
    )


async def test_fetch_content_rejects_cross_channel_identity() -> None:
    await _fetch_content_invalid_descriptor(
        _message_descriptor(channel_remote_id=_OTHER_CHANNEL_ID),
    )


async def test_fetch_content_rejects_malformed_opaque_identity() -> None:
    item = _message_descriptor()
    broken = KnowledgeItemDescriptor.model_construct(
        identity=KnowledgeItemIdentity(remote_id="not-valid-scope", parent_remote_id=None),
        revision=item.revision,
        title=item.title,
        item_type=item.item_type,
        content_mode=item.content_mode,
        content_available=item.content_available,
        provenance=item.provenance,
        metadata=item.metadata,
    )
    await _fetch_content_invalid_descriptor(broken)


async def test_fetch_content_rejects_empty_revision_version() -> None:
    item = _message_descriptor()
    broken = KnowledgeItemDescriptor.model_construct(
        identity=item.identity,
        revision=KnowledgeItemRevision.model_construct(version="   ", etag=None, updated_at=_TS),
        title=item.title,
        item_type=item.item_type,
        content_mode=item.content_mode,
        content_available=item.content_available,
        provenance=item.provenance,
        metadata=item.metadata,
    )
    await _fetch_content_invalid_descriptor(broken)


async def test_fetch_content_rejects_naive_updated_at() -> None:
    naive = datetime(2024, 1, 1, 11, 0, 0)
    item = _message_descriptor()
    broken = KnowledgeItemDescriptor.model_construct(
        identity=item.identity,
        revision=KnowledgeItemRevision(
            version=item.revision.version,
            etag=None,
            updated_at=naive,
        ),
        title=item.title,
        item_type=item.item_type,
        content_mode=item.content_mode,
        content_available=item.content_available,
        provenance=item.provenance,
        metadata=item.metadata,
    )
    await _fetch_content_invalid_descriptor(broken)


async def test_fetch_content_rejects_logical_key() -> None:
    item = _message_descriptor()
    broken = KnowledgeItemDescriptor.model_construct(
        identity=KnowledgeItemIdentity.model_construct(
            remote_id=item.identity.remote_id,
            parent_remote_id=None,
            logical_key="lk-1",
        ),
        revision=item.revision,
        title=item.title,
        item_type=item.item_type,
        content_mode=item.content_mode,
        content_available=item.content_available,
        provenance=item.provenance,
        metadata=item.metadata,
    )
    await _fetch_content_invalid_descriptor(broken)


async def test_fetch_content_rejects_root_with_parent() -> None:
    await _fetch_content_invalid_descriptor(
        _message_descriptor(
            parent_remote_id=_encode_message_identity(
                message_remote_id=_ROOT_1,
                message_kind="root",
            )
        ),
    )


async def test_fetch_content_rejects_reply_with_wrong_parent() -> None:
    await _fetch_content_invalid_descriptor(
        _message_descriptor(
            message_remote_id=_REPLY_1,
            message_kind="reply",
            revision=_ETAG_REPLY_1,
            thread_root_remote_id=_ROOT_1,
            metadata=_base_metadata(message_kind="reply"),
            parent_remote_id="wrong-parent",
        ),
    )


@pytest.mark.parametrize(
    "metadata_patch",
    [
        {"message_state": "deleted"},
        {"message_kind": "reply"},
        {"importance": "super-urgent"},
        {"body_kind": "markdown"},
        {"has_attachments": "yes"},
        {"attachment_binary_content_included": True},
        {"hosted_content_included": True},
        {"reference_urls_included": True},
        {"attachment_inventory_in_content": False},
        {"created_at": "not-a-date"},
        {"last_modified_at": "2024-01-01T11:00:00"},
        {"unknown_key": True},
    ],
)
async def test_fetch_content_rejects_malformed_metadata(metadata_patch: dict[str, object]) -> None:
    base = _message_descriptor().metadata or {}
    merged = {**base, **metadata_patch}
    await _fetch_content_invalid_descriptor(
        _message_descriptor(metadata=merged, metadata_only=True),
    )


async def test_fetch_content_rejects_last_modified_mismatch() -> None:
    await _fetch_content_invalid_descriptor(
        _message_descriptor(
            metadata={"last_modified_at": "2024-01-01T12:00:00+00:00"},
        ),
    )


async def test_fetch_content_rejects_non_dict_metadata() -> None:
    item = _message_descriptor()
    broken = KnowledgeItemDescriptor.model_construct(
        identity=item.identity,
        revision=item.revision,
        title=item.title,
        item_type=item.item_type,
        content_mode=item.content_mode,
        content_available=item.content_available,
        provenance=item.provenance,
        metadata="not-a-dict",
    )
    await _fetch_content_invalid_descriptor(broken)


async def test_fetch_content_rejects_identity_provenance_mismatch() -> None:
    item = _message_descriptor()
    broken = KnowledgeItemDescriptor.model_construct(
        identity=item.identity,
        revision=item.revision,
        title=item.title,
        item_type=item.item_type,
        content_mode=item.content_mode,
        content_available=item.content_available,
        provenance=KnowledgeItemProvenance.model_construct(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            source_kind=MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
            remote_id="different-remote-id",
        ),
        metadata=item.metadata,
    )
    await _fetch_content_invalid_descriptor(broken)


# --- 11. Errors and permissions ---


async def test_integration_dependency_error_translated() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()

    class _FailingSuite(_FakeTeamsChannelCollaborationSuite):
        def read_teams_channel_root_messages_page_by_reference(self, **kwargs):
            raise IntegrationDependencyError("boom")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert "boom" not in str(exc_info.value)


async def test_integration_configuration_error_translated() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()

    class _FailingSuite(_FakeTeamsChannelCollaborationSuite):
        def read_teams_channel_root_messages_page_by_reference(self, **kwargs):
            raise IntegrationConfigurationError("misconfigured")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR


async def test_fetch_permissions_unsupported() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_permissions(
            integration=_integration(fake),
            source=_source(),
            item=_message_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert fake.content_calls == []


async def test_public_objects_do_not_leak_secrets() -> None:
    adapter = MsGraphTeamsChannelKnowledgeAdapter()
    fake = _FakeTeamsChannelCollaborationSuite(
        root_pages=[
            _root_page(items=(_active_root(),), continuation_url=_ROOT_NEXT_URL),
        ],
        content_by_key={(_ROOT_1, _ETAG_ROOT_1): _active_root(body_content="Hello")},
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    blob = json.dumps(page.model_dump(mode="json"))
    for secret in (_SECRET_SKIP, _ROOT_NEXT_URL, "skiptoken", "Authorization"):
        assert secret not in blob
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_message_descriptor(),
    )
    content_blob = json.dumps(content.model_dump(mode="json"))
    assert _ROOT_NEXT_URL not in content_blob
    assert fake.forbidden_calls == []
