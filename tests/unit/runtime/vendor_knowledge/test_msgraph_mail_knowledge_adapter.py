# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for MsGraphMailKnowledgeAdapter."""

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
    DEFAULT_MAIL_CONTENT_MAX_CHARS,
    MSGRAPH_MAIL_SOURCE_KIND,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphMailAttachment,
    MsGraphMailContentTooLarge,
    MsGraphMailImportance,
    MsGraphMailMessageChange,
    MsGraphMailMessageChangeKind,
    MsGraphMailMessageChanged,
    MsGraphMailMessageContent,
    MsGraphMailMessageDeltaPage,
    MsGraphMailParticipant,
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
    MSGRAPH_MAIL_CURSOR_VERSION,
    MSGRAPH_MAIL_SCOPE_TYPE,
    MsGraphMailKnowledgeAdapter,
    encode_msgraph_mail_folder_scope_id,
    register_msgraph_mail_knowledge_adapter,
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

_MAILBOX_USER_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_FOLDER_ID = "AQMkAGI2TG93AAA="
_OTHER_MAILBOX = "b2c3d4e5-f6a7-8901-bcde-f12345678901"
_OTHER_FOLDER = "AQMkAGI2OTHER="
_SECRET_SKIP = "super-secret-skip-token"
_SECRET_DELTA = "super-secret-delta-token"
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_FOLDER = quote(_FOLDER_ID, safe="")
_NEXT_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
    f"{_QUOTED_FOLDER}/messages/delta?$skiptoken={_SECRET_SKIP}"
)
_DELTA_URL = (
    f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders/"
    f"{_QUOTED_FOLDER}/messages/delta?$deltatoken={_SECRET_DELTA}"
)
_OTHER_MAIL_URL = (
    f"https://graph.microsoft.com/v1.0/users/{quote(_OTHER_MAILBOX, safe='')}/mailFolders/"
    f"{quote(_OTHER_FOLDER, safe='')}/messages/delta?$skiptoken=other"
)
_TS = datetime(2026, 5, 29, 10, 15, 30, tzinfo=timezone.utc)
_STRUCTURED_RECORD_SCHEMA = "msgraph.mail.message.knowledge.v1"
_STRUCTURED_RECORD_MIME = "application/vnd.intergrax.msgraph-mail-message+json"
_REMOVAL_SEMANTICS = "removed_from_synchronized_folder_view"
_MESSAGE_ID = "msg-1"
_CHANGE_KEY = "ck-msg-1"


def _encode_canonical_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _scope_id(
    *,
    mailbox_user_id: str = _MAILBOX_USER_ID,
    folder_id: str = _FOLDER_ID,
) -> str:
    return encode_msgraph_mail_folder_scope_id(
        mailbox_user_id=mailbox_user_id,
        folder_id=folder_id,
    )


def _encode_message_identity(
    *,
    message_id: str = _MESSAGE_ID,
    mailbox_user_id: str = _MAILBOX_USER_ID,
    folder_id: str = _FOLDER_ID,
) -> str:
    return _encode_canonical_payload(
        {
            "schema_version": "msgraph.mail.message-id.v1",
            "mailbox_user_id": mailbox_user_id,
            "folder_id": folder_id,
            "message_id": message_id,
        }
    )


def _encode_revision(change_key: str = _CHANGE_KEY) -> str:
    return _encode_canonical_payload(
        {
            "schema_version": "msgraph.mail.revision.v1",
            "change_key": change_key,
        }
    )


def _source(
    *,
    remote_scope_id: str | None = None,
    remote_scope_type: str = MSGRAPH_MAIL_SCOPE_TYPE,
    parameters: dict[str, Any] | None = None,
    provider_id: str = MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    integration_kind: IntegrationCategory = IntegrationCategory.COLLABORATION_SUITE,
    source_kind: str = MSGRAPH_MAIL_SOURCE_KIND,
) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        scope=KnowledgeSourceScope(
            remote_scope_id=remote_scope_id or _scope_id(),
            remote_scope_type=remote_scope_type,
            safe_display_name="Inbox",
            parameters=parameters or {},
        ),
    )


def _active_message(
    *,
    remote_id: str = _MESSAGE_ID,
    mailbox_user_id: str = _MAILBOX_USER_ID,
    folder_id: str = _FOLDER_ID,
    subject: str | None = "Quarterly report",
    change_key: str = _CHANGE_KEY,
    is_read: bool = True,
    is_draft: bool = False,
    has_attachments: bool = False,
    importance: MsGraphMailImportance = MsGraphMailImportance.NORMAL,
) -> MsGraphMailMessageChange:
    return MsGraphMailMessageChange(
        mailbox_user_id=mailbox_user_id,
        scope_folder_id=folder_id,
        remote_id=remote_id,
        kind=MsGraphMailMessageChangeKind.ACTIVE,
        parent_folder_id=folder_id,
        change_key=change_key,
        conversation_id="conv-1",
        internet_message_id="<msg-1@contoso.com>",
        subject=subject,
        created_at=_TS,
        last_modified_at=_TS,
        received_at=_TS,
        sent_at=_TS,
        is_read=is_read,
        is_draft=is_draft,
        has_attachments=has_attachments,
        importance=importance,
    )


def _removed_message(*, remote_id: str = "gone-1") -> MsGraphMailMessageChange:
    return MsGraphMailMessageChange(
        mailbox_user_id=_MAILBOX_USER_ID,
        scope_folder_id=_FOLDER_ID,
        remote_id=remote_id,
        kind=MsGraphMailMessageChangeKind.REMOVED,
        removed_reason="deleted",
    )


def _delta_page(
    *,
    items: tuple[MsGraphMailMessageChange, ...],
    continuation_kind: MsGraphKnowledgeContinuationKind,
    url: str,
) -> MsGraphMailMessageDeltaPage:
    return MsGraphMailMessageDeltaPage(
        items=items,
        continuation=MsGraphKnowledgeContinuation(kind=continuation_kind, url=url),
    )


def _participant(
    *,
    address: str = "alice@contoso.com",
    display_name: str | None = "Alice",
) -> MsGraphMailParticipant:
    return MsGraphMailParticipant(display_name=display_name, address=address)


def _message_content(
    *,
    remote_id: str = _MESSAGE_ID,
    change_key: str = _CHANGE_KEY,
    body_text: str = "Hello team",
    unique_body_text: str | None = "Hello",
) -> MsGraphMailMessageContent:
    return MsGraphMailMessageContent(
        mailbox_user_id=_MAILBOX_USER_ID,
        remote_id=remote_id,
        parent_folder_id=_FOLDER_ID,
        content_revision=change_key,
        conversation_id="conv-1",
        internet_message_id="<msg-1@contoso.com>",
        subject="Quarterly report",
        body_text=body_text,
        unique_body_text=unique_body_text,
        from_participant=_participant(address="sender@contoso.com", display_name="Sender"),
        sender_participant=_participant(address="sender@contoso.com", display_name="Sender"),
        reply_to=(_participant(address="reply@contoso.com", display_name="Reply"),),
        to_recipients=(_participant(),),
        cc_recipients=(_participant(address="cc@contoso.com", display_name="CC"),),
        bcc_recipients=(),
    )


class _FakeMailCollaborationSuite(CollaborationSuite):
    def __init__(
        self,
        *,
        pages: list[MsGraphMailMessageDeltaPage] | None = None,
        content_by_id: dict[str, MsGraphMailMessageContent] | None = None,
        incremental_page: MsGraphMailMessageDeltaPage | None = None,
    ) -> None:
        self._pages = list(pages or [])
        self._content_by_id = dict(content_by_id or {})
        self._incremental_page = incremental_page
        self.delta_calls: list[dict[str, Any]] = []
        self.content_calls: list[dict[str, Any]] = []
        self.attachment_calls = 0

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
            and self._incremental_page is not None
        ):
            return self._incremental_page
        if not self._pages:
            raise IntegrationDependencyError("no pages configured")
        return self._pages.pop(0)

    def read_mail_message_content(
        self,
        *,
        message: MsGraphMailMessageChange,
        max_chars: int,
    ) -> MsGraphMailMessageContent:
        self.content_calls.append({"message": message, "max_chars": max_chars})
        return self._content_by_id[message.remote_id]

    def read_mail_attachments_page(
        self,
        *,
        message: MsGraphMailMessageChange,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ):
        self.attachment_calls += 1
        raise AssertionError("attachments must not be called")

    def read_mail_file_attachment_content(
        self,
        *,
        message: MsGraphMailMessageChange,
        attachment: MsGraphMailAttachment,
        max_bytes: int,
    ):
        self.attachment_calls += 1
        raise AssertionError("attachments must not be called")

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


class _MailTestIntegration(Ms365GraphCollaborationSuiteIntegration):
    def _graph_base_url_for_mail_messages_validation(self) -> str:
        return DEFAULT_GRAPH_BASE_URL

    def _graph_base_url_for_mail_attachments_validation(self) -> str:
        return DEFAULT_GRAPH_BASE_URL


def _integration(fake: _FakeMailCollaborationSuite) -> Ms365GraphCollaborationSuiteIntegration:
    return _MailTestIntegration.from_client(fake, enabled=True)


def _encode_cursor(payload: dict[str, Any]) -> KnowledgeCursor:
    return KnowledgeCursor(
        value=_encode_canonical_payload(payload),
        version=MSGRAPH_MAIL_CURSOR_VERSION,
    )


def _message_descriptor(
    *,
    message_id: str = _MESSAGE_ID,
    change_key: str = _CHANGE_KEY,
    title: str = "Quarterly report",
    mailbox_user_id: str = _MAILBOX_USER_ID,
    folder_id: str = _FOLDER_ID,
    metadata: dict[str, Any] | None = None,
    metadata_only: bool = False,
    content_available: bool = True,
    item_type: str = "msgraph_mail_message",
    content_mode: KnowledgeContentMode = KnowledgeContentMode.STRUCTURED_RECORD,
    provenance_source_kind: str = MSGRAPH_MAIL_SOURCE_KIND,
) -> KnowledgeItemDescriptor:
    opaque_id = _encode_message_identity(
        message_id=message_id,
        mailbox_user_id=mailbox_user_id,
        folder_id=folder_id,
    )
    base_metadata = {
        "message_state": "active",
        "is_read": True,
        "is_draft": False,
        "has_attachments": False,
        "importance": "normal",
        "attachment_inventory_included": False,
        "attachment_content_included": False,
        "removal_semantics": _REMOVAL_SEMANTICS,
        "created_at": _TS.isoformat(),
        "received_at": _TS.isoformat(),
        "sent_at": _TS.isoformat(),
        "last_modified_at": _TS.isoformat(),
    }
    if metadata is not None:
        resolved_metadata = metadata if metadata_only else {**base_metadata, **metadata}
    else:
        resolved_metadata = base_metadata
    return KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id=opaque_id, parent_remote_id=None),
        revision=KnowledgeItemRevision(
            version=_encode_revision(change_key),
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
    assert err.__cause__ is None
    rendered = f"{err!r} {err.safe_message}"
    for secret in (
        _MAILBOX_USER_ID,
        _FOLDER_ID,
        _MESSAGE_ID,
        _CHANGE_KEY,
        "Quarterly report",
        "alice@contoso.com",
        "2026-05-29",
        "graph.microsoft.com",
        "Authorization",
    ):
        assert secret not in rendered


async def _fetch_content_invalid_descriptor(
    item: KnowledgeItemDescriptor,
) -> pytest.ExceptionInfo[VendorKnowledgeError]:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        content_by_id={_MESSAGE_ID: _message_content(body_text="x")}
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


# --- 1. Identity and capabilities ---


async def test_adapter_identity() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    assert adapter.provider_id == MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    assert adapter.integration_kind is IntegrationCategory.COLLABORATION_SUITE
    assert adapter.source_kind == MSGRAPH_MAIL_SOURCE_KIND


async def test_capabilities_exact_set() -> None:
    caps = MsGraphMailKnowledgeAdapter().capabilities
    assert caps.full_inventory is True
    assert caps.incremental_changes is True
    assert caps.content_fetch is True
    assert caps.binary_content is False
    assert caps.rich_text_content is False
    assert caps.structured_content is True
    assert caps.permissions is False
    assert caps.tombstones is True
    assert caps.remote_versions is True
    assert caps.reconciliation is True


# --- 2. Scope ---


async def test_valid_mail_scope_inspect() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite()
    info = await adapter.inspect_scope(integration=_integration(fake), source=_source())
    assert info.safe_display_name == "Inbox"
    assert fake.delta_calls == []


async def test_encoded_scope_id_hides_raw_ids() -> None:
    encoded = _scope_id()
    assert _MAILBOX_USER_ID not in encoded
    assert _FOLDER_ID not in encoded


@pytest.mark.parametrize(
    ("source",),
    [
        (_source(provider_id="jira"),),
        (_source(integration_kind=IntegrationCategory.ISSUE_TRACKER),),
        (_source(source_kind=MSGRAPH_DRIVE_SOURCE_KIND),),
        (_source(remote_scope_type="sharepoint"),),
        (_source(parameters={"folder": "x"}),),
    ],
)
async def test_invalid_scope_rejected(source: KnowledgeSourceRef) -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(_FakeMailCollaborationSuite()),
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
                    "schema_version": "msgraph.mail.scope.v1",
                    "mailbox_user_id": "",
                    "folder_id": _FOLDER_ID,
                }
            ),
        ),
        (
            _encode_canonical_payload(
                {
                    "schema_version": "wrong.schema.v1",
                    "mailbox_user_id": _MAILBOX_USER_ID,
                    "folder_id": _FOLDER_ID,
                }
            ),
        ),
    ],
)
async def test_malformed_scope_payload_rejected(remote_scope_id: str) -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(_FakeMailCollaborationSuite()),
            source=_source(remote_scope_id=remote_scope_id),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_empty_scope_id_rejected() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    source = KnowledgeSourceRef.model_construct(
        tenant_id="tenant-1",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_MAIL_SOURCE_KIND,
        scope=KnowledgeSourceScope.model_construct(
            remote_scope_id="",
            remote_scope_type=MSGRAPH_MAIL_SCOPE_TYPE,
            safe_display_name="Inbox",
            parameters={},
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(_FakeMailCollaborationSuite()),
            source=source,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_whitespace_scope_id_rejected() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    source = KnowledgeSourceRef.model_construct(
        tenant_id="tenant-1",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_MAIL_SOURCE_KIND,
        scope=KnowledgeSourceScope.model_construct(
            remote_scope_id="   ",
            remote_scope_type=MSGRAPH_MAIL_SCOPE_TYPE,
            safe_display_name="Inbox",
            parameters={},
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(_FakeMailCollaborationSuite()),
            source=source,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_wrong_integration_object_rejected() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(integration=object(), source=_source())
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


# --- 3. Limits ---


@pytest.mark.parametrize("limit", [1, 200, 201, 1000])
async def test_limit_accepted_and_clamped(limit: int) -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        pages=[
            _delta_page(
                items=(_active_message(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=limit,
    )
    assert fake.delta_calls[0]["limit"] == min(limit, 200)


@pytest.mark.parametrize("limit", [0, 1001, True, False, "10", 10.5, None])
async def test_invalid_limit_rejected(limit: object) -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FakeMailCollaborationSuite()),
            source=_source(),
            cursor=None,
            limit=limit,  # type: ignore[arg-type]
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR


# --- 4. Descriptor mapping ---


async def test_message_descriptor_mapping() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        pages=[
            _delta_page(
                items=(_active_message(remote_id="m-1"),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
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
    assert descriptor.title == "Quarterly report"
    assert descriptor.item_type == "msgraph_mail_message"
    assert descriptor.content_mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert descriptor.content_available is True
    assert descriptor.revision.updated_at == _TS
    assert descriptor.metadata is not None
    assert descriptor.metadata["message_state"] == "active"
    assert descriptor.metadata["is_read"] is True
    assert descriptor.metadata["is_draft"] is False
    assert descriptor.metadata["has_attachments"] is False
    assert descriptor.metadata["importance"] == "normal"
    assert descriptor.metadata["attachment_inventory_included"] is False
    assert descriptor.metadata["attachment_content_included"] is False
    assert descriptor.metadata["removal_semantics"] == _REMOVAL_SEMANTICS
    assert descriptor.metadata["created_at"] == _TS.isoformat()
    assert descriptor.metadata["received_at"] == _TS.isoformat()
    assert descriptor.metadata["sent_at"] == _TS.isoformat()
    assert descriptor.metadata["last_modified_at"] == _TS.isoformat()
    assert descriptor.provenance.web_url is None
    assert _MAILBOX_USER_ID not in repr(descriptor.metadata)
    assert _FOLDER_ID not in repr(descriptor.metadata)
    assert _CHANGE_KEY not in descriptor.revision.version
    assert "m-1" not in descriptor.identity.remote_id


@pytest.mark.parametrize(
    ("subject", "expected_title"),
    [
        ("Budget Q2", "Budget Q2"),
        (None, "Mail message"),
        ("", "Mail message"),
        ("   ", "Mail message"),
    ],
)
async def test_message_title_resolution(subject: str | None, expected_title: str) -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        pages=[
            _delta_page(
                items=(_active_message(subject=subject),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
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


# --- 5. Tombstone ---


async def test_tombstone_mapping() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        pages=[
            _delta_page(
                items=(_removed_message(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
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


# --- 6. Paging / cursor ---


async def test_initial_read_next_page_cursor() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        pages=[
            _delta_page(
                items=(_active_message(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            )
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    assert page.has_more is True
    assert page.next_cursor is not None
    assert page.proposed_checkpoint == page.next_cursor
    assert _SECRET_SKIP not in page.next_cursor.value
    assert "skiptoken" not in page.next_cursor.value


async def test_delta_checkpoint_semantics() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        pages=[
            _delta_page(
                items=(_active_message(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    assert page.has_more is False
    assert page.next_cursor is None
    assert page.proposed_checkpoint is not None
    assert _SECRET_DELTA not in page.proposed_checkpoint.value


async def test_empty_page_delta_checkpoint() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        pages=[
            _delta_page(
                items=(),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ]
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    assert page.changes == ()
    assert page.has_more is False
    assert page.proposed_checkpoint is not None


async def test_next_incremental_from_delta_cursor() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        incremental_page=_delta_page(
            items=(_active_message(remote_id="changed-1"),),
            continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=_DELTA_URL,
        )
    )
    cursor = _encode_cursor(
        {
            "schema_version": MSGRAPH_MAIL_CURSOR_VERSION,
            "mailbox_user_id": _MAILBOX_USER_ID,
            "folder_id": _FOLDER_ID,
            "continuation_kind": "delta",
            "continuation_url": _DELTA_URL,
        }
    )
    await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=cursor,
        limit=50,
    )
    continuation = fake.delta_calls[0]["continuation"]
    assert continuation is not None
    assert continuation.kind == MsGraphKnowledgeContinuationKind.DELTA
    assert continuation.url == _DELTA_URL


async def test_cursor_url_hidden_from_repr() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        pages=[
            _delta_page(
                items=(_active_message(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            )
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
    assert _NEXT_URL not in rendered


@pytest.mark.parametrize(
    ("cursor",),
    [
        (KnowledgeCursor(value="not-base64", version=MSGRAPH_MAIL_CURSOR_VERSION),),
        (KnowledgeCursor(value="!!!", version=MSGRAPH_MAIL_CURSOR_VERSION),),
        (KnowledgeCursor(value=_encode_cursor({"bad": 1}).value, version="other.v1"),),
        (
            KnowledgeCursor(
                value=_encode_cursor(
                    {
                        "schema_version": MSGRAPH_MAIL_CURSOR_VERSION,
                        "mailbox_user_id": _MAILBOX_USER_ID,
                        "folder_id": _FOLDER_ID,
                        "continuation_kind": "next_page",
                        "continuation_url": "",
                    }
                ).value,
                version=MSGRAPH_MAIL_CURSOR_VERSION,
            ),
        ),
        (
            KnowledgeCursor(
                value=_encode_cursor(
                    {
                        "schema_version": MSGRAPH_MAIL_CURSOR_VERSION,
                        "mailbox_user_id": _OTHER_MAILBOX,
                        "folder_id": _OTHER_FOLDER,
                        "continuation_kind": "delta",
                        "continuation_url": _OTHER_MAIL_URL,
                    }
                ).value,
                version=MSGRAPH_MAIL_CURSOR_VERSION,
            ),
        ),
    ],
)
async def test_invalid_cursor_rejected(cursor: KnowledgeCursor) -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FakeMailCollaborationSuite()),
            source=_source(),
            cursor=cursor,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR


# --- 7. Provider response boundary ---


async def test_wrong_page_type_rejected() -> None:
    adapter = MsGraphMailKnowledgeAdapter()

    class _BadPageSuite(_FakeMailCollaborationSuite):
        def read_mail_messages_delta_page(self, *, mailbox_user_id, folder_id, continuation=None, limit):
            return object()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_BadPageSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_active_message_missing_change_key_rejected() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    malformed = MsGraphMailMessageChange.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        scope_folder_id=_FOLDER_ID,
        remote_id="bad-1",
        kind=MsGraphMailMessageChangeKind.ACTIVE,
        parent_folder_id=_FOLDER_ID,
        change_key=None,
        last_modified_at=_TS,
        is_read=True,
        is_draft=False,
        has_attachments=False,
        importance=MsGraphMailImportance.NORMAL,
    )
    fake = _FakeMailCollaborationSuite(
        pages=[
            MsGraphMailMessageDeltaPage.model_construct(
                items=(malformed,),
                continuation=MsGraphKnowledgeContinuation(
                    kind=MsGraphKnowledgeContinuationKind.DELTA,
                    url=_DELTA_URL,
                ),
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


async def test_cross_mailbox_item_rejected() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        pages=[
            _delta_page(
                items=(_active_message(mailbox_user_id=_OTHER_MAILBOX),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
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


async def test_duplicate_remote_id_rejected() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    item = _active_message(remote_id="dup-1")
    fake = _FakeMailCollaborationSuite(
        pages=[
            MsGraphMailMessageDeltaPage.model_construct(
                items=(item, item),
                continuation=MsGraphKnowledgeContinuation(
                    kind=MsGraphKnowledgeContinuationKind.DELTA,
                    url=_DELTA_URL,
                ),
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


async def test_invalid_continuation_url_rejected() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        pages=[
            _delta_page(
                items=(_active_message(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_OTHER_MAIL_URL,
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
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR


# --- 8. Content ---


async def test_fetch_content_structured_record() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        content_by_id={_MESSAGE_ID: _message_content(body_text="Hello team")}
    )
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_message_descriptor(),
    )
    assert content.mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert content.mime_type == _STRUCTURED_RECORD_MIME
    record = content.structured_record
    assert record is not None
    assert record["schema"] == _STRUCTURED_RECORD_SCHEMA
    assert record["subject"] == "Quarterly report"
    assert record["conversation_id"] == "conv-1"
    assert record["internet_message_id"] == "<msg-1@contoso.com>"
    assert record["body_text"] == "Hello team"
    assert record["unique_body_text"] == "Hello"
    assert record["from"] == {"display_name": "Sender", "address": "sender@contoso.com"}
    assert record["sender"] == {"display_name": "Sender", "address": "sender@contoso.com"}
    assert record["reply_to"] == [{"display_name": "Reply", "address": "reply@contoso.com"}]
    assert record["to_recipients"] == [{"display_name": "Alice", "address": "alice@contoso.com"}]
    assert record["cc_recipients"] == [{"display_name": "CC", "address": "cc@contoso.com"}]
    assert record["bcc_recipients"] == []
    assert record["created_at"] == _TS.isoformat()
    assert record["last_modified_at"] == _TS.isoformat()
    assert record["received_at"] == _TS.isoformat()
    assert record["sent_at"] == _TS.isoformat()
    assert record["is_read"] is True
    assert record["is_draft"] is False
    assert record["importance"] == "normal"
    assert record["attachments"] == {
        "has_attachments": False,
        "inventory_included": False,
        "binary_content_included": False,
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert content.content_hash == hashlib.sha256(canonical).hexdigest()
    assert fake.content_calls[0]["max_chars"] == DEFAULT_MAIL_CONTENT_MAX_CHARS
    assert fake.content_calls[0]["message"].remote_id == _MESSAGE_ID
    assert fake.content_calls[0]["message"].change_key == _CHANGE_KEY


async def test_fetch_content_returns_fresh_dicts() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        content_by_id={_MESSAGE_ID: _message_content(body_text="Hello team")}
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


# --- 9. Invalid descriptor boundary ---


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


async def test_fetch_content_rejects_cross_mailbox_identity() -> None:
    await _fetch_content_invalid_descriptor(
        _message_descriptor(mailbox_user_id=_OTHER_MAILBOX),
    )


async def test_fetch_content_rejects_cross_folder_identity() -> None:
    await _fetch_content_invalid_descriptor(
        _message_descriptor(folder_id=_OTHER_FOLDER),
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
    naive = datetime(2026, 5, 29, 10, 15, 30)
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


@pytest.mark.parametrize(
    "metadata_patch",
    [
        {"message_state": "removed"},
        {"importance": "urgent"},
        {"is_read": "yes"},
        {"attachment_inventory_included": True},
        {"attachment_content_included": True},
        {"removal_semantics": "deleted"},
        {"created_at": "not-a-date"},
        {"last_modified_at": "2026-05-29T10:15:30"},
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
            metadata={"last_modified_at": "2026-05-29T11:00:00+00:00"},
        ),
    )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("metadata", "not-a-dict"),
        ("revision", "not-a-revision"),
        ("identity", "not-an-identity"),
    ],
)
async def test_fetch_content_rejects_malformed_descriptor_shape(
    field_name: str,
    field_value: object,
) -> None:
    base = _message_descriptor()
    item = KnowledgeItemDescriptor.model_construct(
        identity=base.identity,
        revision=base.revision,
        title=base.title,
        item_type=base.item_type,
        content_mode=base.content_mode,
        content_available=base.content_available,
        provenance=base.provenance,
        metadata=base.metadata,
    )
    object.__setattr__(item, field_name, field_value)
    await _fetch_content_invalid_descriptor(item)


# --- 10. Malformed content boundary ---


async def test_fetch_content_provider_revision_mismatch() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        content_by_id={
            _MESSAGE_ID: _message_content(change_key="other-revision"),
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


async def test_fetch_content_provider_mailbox_mismatch() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        content_by_id={
            _MESSAGE_ID: MsGraphMailMessageContent(
                mailbox_user_id=_OTHER_MAILBOX,
                remote_id=_MESSAGE_ID,
                parent_folder_id=_FOLDER_ID,
                content_revision=_CHANGE_KEY,
                body_text="x",
            )
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


# --- 11. Errors ---


async def test_integration_dependency_error_translated() -> None:
    adapter = MsGraphMailKnowledgeAdapter()

    class _FailingSuite(_FakeMailCollaborationSuite):
        def read_mail_messages_delta_page(self, *, mailbox_user_id, folder_id, continuation=None, limit):
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
    assert exc_info.value.__cause__ is None
    assert "boom" not in str(exc_info.value)
    assert _SECRET_SKIP not in str(exc_info.value)


async def test_integration_configuration_error_translated() -> None:
    adapter = MsGraphMailKnowledgeAdapter()

    class _FailingSuite(_FakeMailCollaborationSuite):
        def read_mail_messages_delta_page(self, *, mailbox_user_id, folder_id, continuation=None, limit):
            raise IntegrationConfigurationError("misconfigured")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert exc_info.value.retryable is False
    assert exc_info.value.__cause__ is None


async def test_unexpected_exception_translated() -> None:
    adapter = MsGraphMailKnowledgeAdapter()

    class _FailingSuite(_FakeMailCollaborationSuite):
        def read_mail_messages_delta_page(self, *, mailbox_user_id, folder_id, continuation=None, limit):
            raise RuntimeError("unexpected")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(_FailingSuite()),
            source=_source(),
            cursor=None,
            limit=50,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.__cause__ is None
    assert "unexpected" not in str(exc_info.value)


async def test_content_too_large_translated() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    huge = "x" * (DEFAULT_MAIL_CONTENT_MAX_CHARS + 1)

    class _HugeContentSuite(_FakeMailCollaborationSuite):
        def read_mail_message_content(self, *, message, max_chars):
            raise MsGraphMailContentTooLarge()

    fake = _HugeContentSuite(
        content_by_id={_MESSAGE_ID: _message_content(body_text=huge)},
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(fake),
            source=_source(),
            item=_message_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert exc_info.value.__cause__ is None
    assert huge not in str(exc_info.value)


async def test_message_changed_during_read_translated() -> None:
    adapter = MsGraphMailKnowledgeAdapter()

    class _ChangedSuite(_FakeMailCollaborationSuite):
        def read_mail_message_content(self, *, message, max_chars):
            raise MsGraphMailMessageChanged()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=_integration(_ChangedSuite()),
            source=_source(),
            item=_message_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.__cause__ is None


async def test_public_objects_do_not_leak_secrets() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        pages=[
            _delta_page(
                items=(_active_message(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_NEXT_URL,
            )
        ],
        content_by_id={_MESSAGE_ID: _message_content()},
    )
    page = await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    blob = json.dumps(page.model_dump(mode="json"))
    for secret in (_SECRET_SKIP, _SECRET_DELTA, _NEXT_URL, "skiptoken", "deltatoken", "Authorization"):
        assert secret not in blob
    content = await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_message_descriptor(),
    )
    content_blob = json.dumps(content.model_dump(mode="json"))
    assert _NEXT_URL not in content_blob
    assert "Hello team" not in repr(page)
    err = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.INVALID_CURSOR,
        safe_message="Microsoft Graph Mail knowledge cursor is invalid",
        provider_id=adapter.provider_id,
        source_kind=adapter.source_kind,
        retryable=False,
    )
    assert _SECRET_SKIP not in repr(err)
    assert _SECRET_SKIP not in str(err)


# --- 12. Attachments and permissions ---


async def test_attachments_never_called() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite(
        pages=[
            _delta_page(
                items=(_active_message(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ],
        content_by_id={_MESSAGE_ID: _message_content()},
    )
    await adapter.read_page(
        integration=_integration(fake),
        source=_source(),
        cursor=None,
        limit=50,
    )
    await adapter.fetch_content(
        integration=_integration(fake),
        source=_source(),
        item=_message_descriptor(),
    )
    assert fake.attachment_calls == 0


async def test_fetch_permissions_unsupported() -> None:
    adapter = MsGraphMailKnowledgeAdapter()
    fake = _FakeMailCollaborationSuite()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_permissions(
            integration=_integration(fake),
            source=_source(),
            item=_message_descriptor(),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert fake.attachment_calls == 0


# --- 13. Registration and facade ---


async def test_registry_registration_and_facade() -> None:
    registry = KnowledgeAdapterRegistry()
    mail_adapter = register_msgraph_mail_knowledge_adapter(registry)
    drive_adapter = register_msgraph_drive_knowledge_adapter(registry)
    register_jira_issues_knowledge_adapter(registry)
    register_confluence_pages_knowledge_adapter(registry)

    assert isinstance(mail_adapter, MsGraphMailKnowledgeAdapter)
    assert isinstance(
        registry.resolve(source=_source()),
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

    fake = _FakeMailCollaborationSuite(
        pages=[
            _delta_page(
                items=(_active_message(),),
                continuation_kind=MsGraphKnowledgeContinuationKind.DELTA,
                url=_DELTA_URL,
            )
        ],
        content_by_id={_MESSAGE_ID: _message_content()},
    )
    integration = _integration(fake)
    source = _source()

    scope_info = await mail_adapter.inspect_scope(integration=integration, source=source)
    assert scope_info.safe_display_name == "Inbox"

    page = await mail_adapter.read_page(
        integration=integration,
        source=source,
        cursor=None,
        limit=50,
    )
    assert page.changes

    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    content = await mail_adapter.fetch_content(
        integration=integration,
        source=source,
        item=descriptor,
    )
    assert content.mode is KnowledgeContentMode.STRUCTURED_RECORD

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await mail_adapter.fetch_permissions(
            integration=integration,
            source=source,
            item=descriptor,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY

    assert drive_adapter.source_kind == MSGRAPH_DRIVE_SOURCE_KIND


async def test_duplicate_registry_registration_rejected() -> None:
    registry = KnowledgeAdapterRegistry()
    register_msgraph_mail_knowledge_adapter(registry)
    with pytest.raises(ValueError):
        register_msgraph_mail_knowledge_adapter(registry)
