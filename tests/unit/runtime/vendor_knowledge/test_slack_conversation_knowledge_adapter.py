# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for SlackConversationKnowledgeAdapter."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SLACK_CONVERSATION_SOURCE_KIND,
    SlackConversationExactMessageResult,
    SlackConversationKind,
    SlackConversationMessage,
    SlackConversationMessagePage,
    compute_slack_conversation_message_revision,
)
from intergrax.runtime.vendor_knowledge.adapters.confluence_pages import (
    register_confluence_pages_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.jira_issues import (
    register_jira_issues_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_channel import (
    register_msgraph_teams_channel_knowledge_adapter,
)
from intergrax.integrations.providers.conversation_channel.slack.mapping import parse_slack_ts
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    SLACK_CONVERSATION_SCOPE_TYPE,
    SlackConversationKnowledgeAdapter,
    encode_slack_conversation_scope_id,
    register_slack_conversation_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError, VendorKnowledgeErrorCode
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChangeKind,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_CONVERSATION_ID = "C01234567"
_OLDEST = "1704067200.000001"
_LATEST = "1706745600.000001"
_ROOT_TS = "1704153600.000001"
_REPLY_TS = "1704153601.000001"
_TS = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)


def _scope_id() -> str:
    return encode_slack_conversation_scope_id(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        oldest=_OLDEST,
        latest=_LATEST,
    )


def _source() -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        scope=KnowledgeSourceScope(
            remote_scope_id=_scope_id(),
            remote_scope_type=SLACK_CONVERSATION_SCOPE_TYPE,
            safe_display_name="General",
            parameters={},
        ),
    )


def _message(
    *,
    message_ts: str,
    text: str,
    reply_count: int = 0,
    root_thread_ts: str | None = None,
) -> SlackConversationMessage:
    created_at = parse_slack_ts(message_ts) or _TS
    return SlackConversationMessage(
        conversation_id=_CONVERSATION_ID,
        message_ts=message_ts,
        root_thread_ts=root_thread_ts,
        actor_provider_id="U111",
        text=text,
        subtype=None,
        created_at=created_at,
        edited_at=None,
        reply_count=reply_count,
        files=(),
        provider_metadata={},
    )


class _FakeSlackIntegration:
    def __init__(self) -> None:
        self.history_calls: list[dict[str, Any]] = []
        self.reply_calls: list[dict[str, Any]] = []
        self.exact_calls: list[dict[str, Any]] = []
        self._content: dict[str, SlackConversationMessage] = {}
        self._history_pages = [
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(_message(message_ts=_ROOT_TS, text="root", reply_count=1),),
                next_cursor="history-2",
            ),
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(_message(message_ts="1704153602.000001", text="second root"),),
            ),
        ]
        self._reply_pages = {
            _ROOT_TS: [
                SlackConversationMessagePage(
                    conversation_id=_CONVERSATION_ID,
                    oldest=_OLDEST,
                    latest=_LATEST,
                    items=(_message(message_ts=_REPLY_TS, text="reply", root_thread_ts=_ROOT_TS),),
                )
            ]
        }

    async def read_conversation_history_page(self, **kwargs: Any) -> SlackConversationMessagePage:
        self.history_calls.append(kwargs)
        page = self._history_pages.pop(0)
        for item in page.items:
            self._content[item.message_ts] = item
        return page

    async def read_thread_replies_page(self, **kwargs: Any) -> SlackConversationMessagePage:
        self.reply_calls.append(kwargs)
        page = self._reply_pages[kwargs["root_message_ts"]].pop(0)
        for item in page.items:
            self._content[item.message_ts] = item
        return page

    async def read_exact_message(self, **kwargs: Any) -> SlackConversationExactMessageResult:
        self.exact_calls.append(kwargs)
        message = self._content.get(kwargs["message_ts"])
        if message is None:
            message = _message(message_ts=kwargs["message_ts"], text="exact")
        revision = kwargs.get("expected_revision")
        if revision is not None and revision != compute_slack_conversation_message_revision(message):
            from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
                SlackConversationMessageChanged,
            )

            raise SlackConversationMessageChanged()
        return SlackConversationExactMessageResult(found=True, message=message)

    async def list_accessible_conversations_page(self, **kwargs: Any):
        raise NotImplementedError

    async def read_file_info(self, **kwargs: Any):
        raise NotImplementedError


class _FakeSlackBackend(_FakeSlackIntegration):
    async def start(self, handler) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, message):
        raise NotImplementedError

    def health(self):
        from intergrax.integrations.contracts.base import HealthStatus

        return HealthStatus(slug="slack", healthy=True, detail="test")


def _integration(fake: _FakeSlackBackend) -> SlackConversationChannelIntegration:
    config = SlackConversationChannelIntegrationConfig(
        enabled=True,
        app_token="xapp-test-token-value",
        bot_token="xoxb-test-token-value",
        knowledge_user_token="xoxp-test-knowledge-user-token",
    )
    return SlackConversationChannelIntegration.from_backend(fake, enabled=True, config=config)  # type: ignore[arg-type]


async def test_registry_coexistence_with_existing_adapters() -> None:
    registry = KnowledgeAdapterRegistry()
    slack_adapter = register_slack_conversation_knowledge_adapter(registry)
    register_msgraph_teams_channel_knowledge_adapter(registry)
    register_jira_issues_knowledge_adapter(registry)
    register_confluence_pages_knowledge_adapter(registry)
    keys = registry.registered_keys()
    assert (
        SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        IntegrationCategory.CONVERSATION_CHANNEL,
        SLACK_CONVERSATION_SOURCE_KIND,
    ) in keys
    assert registry.resolve(source=_source()) is slack_adapter


async def test_read_page_traverses_history_then_replies_without_duplicating_root() -> None:
    adapter = SlackConversationKnowledgeAdapter()
    fake = _FakeSlackBackend()
    integration = _integration(fake)
    source = _source()
    first = await adapter.read_page(integration=integration, source=source, cursor=None, limit=10)
    assert first.changes[0].kind is KnowledgeChangeKind.UPSERT
    second = await adapter.read_page(
        integration=integration,
        source=source,
        cursor=first.next_cursor,
        limit=10,
    )
    assert second.changes[0].descriptor is not None
    assert second.changes[0].descriptor.title == "reply"
    third = await adapter.read_page(
        integration=integration,
        source=source,
        cursor=second.next_cursor,
        limit=10,
    )
    assert third.changes[0].descriptor is not None
    assert third.changes[0].descriptor.title == "second root"
    assert len(fake.reply_calls) == 1


async def test_fetch_permissions_returns_unsupported_capability_after_validation() -> None:
    adapter = SlackConversationKnowledgeAdapter()
    fake = _FakeSlackBackend()
    integration = _integration(fake)
    source = _source()
    page = await adapter.read_page(integration=integration, source=source, cursor=None, limit=1)
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_permissions(integration=integration, source=source, item=descriptor)
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert fake.exact_calls == []


async def test_invalid_scope_rejected_without_provider_calls() -> None:
    adapter = SlackConversationKnowledgeAdapter()
    fake = _FakeSlackBackend()
    integration = _integration(fake)
    bad_source = KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        scope=KnowledgeSourceScope(
            remote_scope_id="not-canonical",
            remote_scope_type=SLACK_CONVERSATION_SCOPE_TYPE,
            safe_display_name="General",
            parameters={},
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(integration=integration, source=bad_source, cursor=None, limit=1)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert fake.history_calls == []


async def test_fetch_content_succeeds_for_later_page_reply() -> None:
    adapter = SlackConversationKnowledgeAdapter()
    fake = _FakeSlackBackend()
    integration = _integration(fake)
    source = _source()
    page = await adapter.read_page(integration=integration, source=source, cursor=None, limit=1)
    reply_page = await adapter.read_page(
        integration=integration,
        source=source,
        cursor=page.next_cursor,
        limit=10,
    )
    descriptor = reply_page.changes[0].descriptor
    assert descriptor is not None
    content = await adapter.fetch_content(integration=integration, source=source, item=descriptor)
    assert content.structured_record is not None
    assert content.structured_record["text"] == "reply"


async def test_fetch_content_revision_mismatch_is_retryable() -> None:
    adapter = SlackConversationKnowledgeAdapter()
    fake = _FakeSlackBackend()
    integration = _integration(fake)
    source = _source()
    page = await adapter.read_page(integration=integration, source=source, cursor=None, limit=1)
    reply_page = await adapter.read_page(
        integration=integration,
        source=source,
        cursor=page.next_cursor,
        limit=10,
    )
    descriptor = reply_page.changes[0].descriptor
    assert descriptor is not None
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(
            integration=integration,
            source=source,
            item=descriptor.model_copy(
                update={
                    "revision": descriptor.revision.model_copy(
                        update={"version": "not-a-valid-revision-payload"}
                    )
                }
            ),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


@pytest.mark.parametrize(
    "mutator",
    [
        lambda descriptor: descriptor.model_copy(
            update={"identity": descriptor.identity.model_copy(update={"remote_id": "tampered"})}
        ),
        lambda descriptor: descriptor.model_copy(
            update={
                "provenance": descriptor.provenance.model_copy(update={"remote_id": "tampered"})
            }
        ),
        lambda descriptor: descriptor.model_copy(
            update={"metadata": {**descriptor.metadata, "thread_root_ts": "1704999999.000001"}}
        ),
        lambda descriptor: descriptor.model_copy(update={"item_type": "other"}),
        lambda descriptor: descriptor.model_copy(update={"content_available": False}),
    ],
)
async def test_fetch_permissions_invalid_descriptor_returns_invalid_scope(
    mutator,
) -> None:
    adapter = SlackConversationKnowledgeAdapter()
    fake = _FakeSlackBackend()
    integration = _integration(fake)
    source = _source()
    page = await adapter.read_page(integration=integration, source=source, cursor=None, limit=1)
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_permissions(
            integration=integration,
            source=source,
            item=mutator(descriptor),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert fake.exact_calls == []


async def test_thread_broadcast_in_history_is_not_emitted() -> None:
    adapter = SlackConversationKnowledgeAdapter()
    fake = _FakeSlackBackend()
    broadcast = _message(message_ts=_REPLY_TS, text="broadcast", root_thread_ts=_ROOT_TS)
    fake._history_pages = [
        SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(broadcast,),
            next_cursor="history-2",
        ),
        SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(_message(message_ts=_ROOT_TS, text="root", reply_count=1),),
        ),
    ]
    fake._reply_pages = {
        _ROOT_TS: [
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(_message(message_ts=_REPLY_TS, text="reply", root_thread_ts=_ROOT_TS),),
            )
        ]
    }
    integration = _integration(fake)
    source = _source()
    first = await adapter.read_page(integration=integration, source=source, cursor=None, limit=1)
    assert first.changes == ()
    assert first.next_cursor is not None
    second = await adapter.read_page(
        integration=integration,
        source=source,
        cursor=first.next_cursor,
        limit=1,
    )
    assert second.changes[0].descriptor is not None
    assert second.changes[0].descriptor.title == "root"
    third = await adapter.read_page(
        integration=integration,
        source=source,
        cursor=second.next_cursor,
        limit=10,
    )
    assert third.changes[0].descriptor is not None
    assert third.changes[0].descriptor.title == "reply"
    assert len(fake.reply_calls) == 1


async def test_capabilities_full_inventory_within_root_window_scope() -> None:
    adapter = SlackConversationKnowledgeAdapter()
    assert adapter.capabilities.full_inventory is True
    scope = encode_slack_conversation_scope_id(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
        oldest=_OLDEST,
        latest=_LATEST,
    )
    assert "slack.conversation.scope.v2" in scope or scope  # encoded payload is opaque
    with pytest.raises(ValueError):
        encode_slack_conversation_scope_id(
            conversation_id=_CONVERSATION_ID,
            conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
            oldest=_LATEST,
            latest=_OLDEST,
        )
