# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""End-to-end Slack conversation knowledge adapter proof through facade and coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
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
from intergrax.integrations.providers.conversation_channel.slack.mapping import (
    parse_slack_ts,
)
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    SLACK_CONVERSATION_SCOPE_TYPE,
    encode_slack_conversation_scope_id,
    register_slack_conversation_knowledge_adapter,
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
from intergrax.runtime.vendor_knowledge.sync_models import KnowledgeSyncRunStatus
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    IdempotentRecordingSink,
    RecordingBindingService,
    durable_reconcile_until_complete,
    durable_reconciliation_coordinator_kwargs,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_CONVERSATION_ID = "C01234567"
_OLDEST = "1704067200.000001"
_LATEST = "1706745600.000001"
_ROOT_TS = "1704153600.000001"
_REPLY_TS = "1704153601.000001"
_EDITED_TS = "1704153602.000001"
_TS = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)


def _message(
    *,
    message_ts: str,
    text: str,
    reply_count: int = 0,
    root_thread_ts: str | None = None,
    edited_at: datetime | None = None,
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
        edited_at=edited_at,
        reply_count=reply_count,
        files=(),
        provider_metadata={},
    )


class _SlackFakeIntegration:
    def __init__(self) -> None:
        self.history_calls: list[dict[str, Any]] = []
        self.reply_calls: list[dict[str, Any]] = []
        self._history_pages = [
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(_message(message_ts=_ROOT_TS, text="root one", reply_count=1),),
                next_cursor="history-2",
            ),
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(
                    _message(
                        message_ts=_EDITED_TS,
                        text="edited body",
                        edited_at=datetime(2024, 1, 3, 12, 0, tzinfo=timezone.utc),
                    ),
                ),
            ),
        ]
        self._history_backup = list(self._history_pages)
        self._reply_pages = [
            SlackConversationMessagePage(
                conversation_id=_CONVERSATION_ID,
                oldest=_OLDEST,
                latest=_LATEST,
                items=(
                    _message(
                        message_ts=_REPLY_TS, text="reply one", root_thread_ts=_ROOT_TS
                    ),
                ),
            )
        ]
        self._reply_backup = list(self._reply_pages)
        self._content: dict[str, SlackConversationMessage] = {}

    def _reset_if_needed(self) -> None:
        if not self._history_pages and self._history_backup:
            self._history_pages = list(self._history_backup)
            self._reply_pages = list(self._reply_backup)

    async def read_conversation_history_page(
        self, **kwargs: Any
    ) -> SlackConversationMessagePage:
        if kwargs.get("cursor") is None:
            self._reset_if_needed()
        self.history_calls.append(kwargs)
        page = self._history_pages.pop(0)
        for item in page.items:
            self._content[item.message_ts] = item
        return page

    async def read_thread_replies_page(
        self, **kwargs: Any
    ) -> SlackConversationMessagePage:
        self.reply_calls.append(kwargs)
        cursor = kwargs.get("cursor")
        if cursor == "reply-page-2":
            page = self._reply_backup[-1]
        else:
            if not self._reply_pages:
                self._reply_pages = list(self._reply_backup)
            page = self._reply_pages.pop(0)
        for item in page.items:
            self._content[item.message_ts] = item
        return page

    async def read_exact_message(
        self, **kwargs: Any
    ) -> SlackConversationExactMessageResult:
        message_ts = kwargs["message_ts"]
        message = self._content.get(message_ts)
        if message is None:
            for page in self._history_pages + self._history_backup:
                for item in page.items:
                    if item.message_ts == message_ts:
                        message = item
                        break
                if message is not None:
                    break
            for pages in self._reply_pages.values():
                for page in pages:
                    for item in page.items:
                        if item.message_ts == message_ts:
                            message = item
                            break
        if message is None:
            message = _message(message_ts=message_ts, text="exact")
        revision = kwargs.get("expected_revision")
        if (
            revision is not None
            and revision != compute_slack_conversation_message_revision(message)
        ):
            from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
                SlackConversationMessageChanged,
            )

            raise SlackConversationMessageChanged()
        return SlackConversationExactMessageResult(found=True, message=message)

    async def list_accessible_conversations_page(self, **kwargs: Any):
        raise NotImplementedError

    async def read_file_info(self, **kwargs: Any):
        raise NotImplementedError

    async def start(self, handler) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, message):
        raise NotImplementedError

    def health(self):
        from intergrax.integrations.contracts.base import HealthStatus

        return HealthStatus(slug="slack", healthy=True, detail="test")


@dataclass
class _SlackResolver:
    integration: SlackConversationChannelIntegration

    def resolve(self, *, source) -> SlackConversationChannelIntegration:
        return self.integration


def _build_coordinator(fake: _SlackFakeIntegration | None = None):
    backend = fake or _SlackFakeIntegration()
    config = SlackConversationChannelIntegrationConfig(
        enabled=True,
        app_token="xapp-test-token-value",
        bot_token="xoxb-test-token-value",
    )
    integration = SlackConversationChannelIntegration.from_backend(
        backend, enabled=True, config=config
    )  # type: ignore[arg-type]
    registry = KnowledgeAdapterRegistry()
    register_slack_conversation_knowledge_adapter(registry)
    facade = VendorKnowledgeFacadeService(
        tenant_id="tenant-1",
        resolver=_SlackResolver(integration=integration),
        adapter_registry=registry,
    )
    document_store = InMemoryDocumentStore()
    lease_repo = DocumentStoreKnowledgeSourceLeaseRepository(document_store)
    checkpoint_repo = DocumentStoreKnowledgeSyncCheckpointRepository(document_store)
    state_repo = DocumentStoreKnowledgeRemoteItemStateRepository(document_store)
    sink = IdempotentRecordingSink()
    binding = KnowledgeSourceBinding(
        binding_id="slack-conversation-binding",
        tenant_id="tenant-1",
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        connection_ref="conn-1",
        safe_display_name="Slack Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id=encode_slack_conversation_scope_id(
                conversation_id=_CONVERSATION_ID,
                conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
                oldest=_OLDEST,
                latest=_LATEST,
            ),
            remote_scope_type=SLACK_CONVERSATION_SCOPE_TYPE,
            safe_display_name="General",
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
    return coordinator, sink, checkpoint_repo, state_repo, backend, integration


async def _reconcile_until_complete(
    coordinator: VendorKnowledgeSyncCoordinator,
) -> list:
    return await durable_reconcile_until_complete(
        coordinator,
        binding_id="slack-conversation-binding",
        operation_id="slack-conversation-recon",
    )


@pytest.mark.asyncio
async def test_slack_conversation_facade_coordinator_reconciliation() -> None:
    coordinator, sink, checkpoint_repo, _, fake, _integration = _build_coordinator(
        _SlackFakeIntegration()
    )
    results = await _reconcile_until_complete(coordinator)
    assert len(results) >= 3
    assert len(sink.calls) == len(results)
    assert all(batch.mode.value == "reconciliation" for batch in sink.calls)
    for batch in sink.calls:
        for envelope in batch.envelopes:
            assert envelope.change_kind.value == "upsert"
            assert envelope.content is not None
            assert envelope.content.mode is KnowledgeContentMode.STRUCTURED_RECORD
            assert envelope.content.structured_record is not None
            assert envelope.content.structured_record["schema"] == (
                "slack.conversation.message.knowledge.v1"
            )
    checkpoint = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id="slack-conversation-binding",
    )
    assert checkpoint is not None
    assert checkpoint.cursor is not None
    assert len(fake.history_calls) >= 2
    assert len(fake.reply_calls) == 1


@pytest.mark.asyncio
async def test_slack_sink_failure_retries_same_page_with_stable_delivery_id() -> None:
    fake = _SlackFakeIntegration()
    single_root = _message(message_ts=_ROOT_TS, text="root one", reply_count=0)
    single_page = SlackConversationMessagePage(
        conversation_id=_CONVERSATION_ID,
        oldest=_OLDEST,
        latest=_LATEST,
        items=(single_root,),
    )
    fake._history_pages = [single_page]
    fake._history_backup = [single_page]
    fake._reply_pages = []
    fake._reply_backup = []
    coordinator, sink, checkpoint_repo, state_repo, _, integration = _build_coordinator(
        fake
    )
    integration_id = id(integration)
    checkpoint_before_attempt = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id="slack-conversation-binding",
    )
    sink.fail_times = 1
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="slack-conversation-binding",
            restart=True,
            operation_id="slack-sink-retry",
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    first_delivery = sink.calls[0].delivery_id
    first_history_calls = len(fake.history_calls)
    checkpoint_after_failure = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id="slack-conversation-binding",
    )
    first_remote_id = (
        sink.calls[0].envelopes[0].remote_id if sink.calls[0].envelopes else None
    )
    state_after_failure = (
        state_repo.get(
            tenant_id="tenant-1",
            binding_id="slack-conversation-binding",
            remote_id=first_remote_id,
        )
        if first_remote_id is not None
        else None
    )
    assert checkpoint_after_failure == checkpoint_before_attempt
    assert state_after_failure is None
    retry_result = await coordinator.reconcile_once(
        binding_id="slack-conversation-binding",
        restart=True,
        operation_id="slack-sink-retry",
    )
    assert retry_result.status is KnowledgeSyncRunStatus.COMPLETED
    assert retry_result.delivery_id == first_delivery
    assert len(sink.calls) == 2
    assert sink.calls[0].delivery_id == sink.calls[1].delivery_id
    assert len(sink.durable_delivery_ids) == 1
    assert len(fake.history_calls) == first_history_calls + 1
    assert fake.history_calls[0] == fake.history_calls[first_history_calls]
    assert id(integration) == integration_id
    checkpoint_after = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id="slack-conversation-binding",
    )
    state_after_retry = (
        state_repo.get(
            tenant_id="tenant-1",
            binding_id="slack-conversation-binding",
            remote_id=first_remote_id,
        )
        if first_remote_id is not None
        else None
    )
    assert checkpoint_after is not None
    assert checkpoint_after != checkpoint_after_failure
    assert state_after_retry is not None


@pytest.mark.asyncio
async def test_slack_absence_does_not_emit_tombstones() -> None:
    fake = _SlackFakeIntegration()
    coordinator, sink, _, _, _, _ = _build_coordinator(fake)
    await _reconcile_until_complete(coordinator)
    first_delivery_count = len(sink.calls)
    fake._history_pages = [
        SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(_message(message_ts="1704153603.000001", text="replacement only"),),
            next_cursor="history-page-absence",
        ),
    ]
    partial = await coordinator.reconcile_once(
        binding_id="slack-conversation-binding",
        restart=True,
        operation_id="slack-absence-partial",
    )
    assert partial.has_more is True
    for batch in sink.calls[first_delivery_count:]:
        for envelope in batch.envelopes:
            assert envelope.change_kind.value != "deleted"


@pytest.mark.asyncio
async def test_slack_edit_changes_revision_and_content() -> None:
    fake = _SlackFakeIntegration()
    config = SlackConversationChannelIntegrationConfig(
        enabled=True,
        app_token="xapp-test-token-value",
        bot_token="xoxb-test-token-value",
    )
    integration = SlackConversationChannelIntegration.from_backend(
        fake, enabled=True, config=config
    )  # type: ignore[arg-type]
    adapter_registry = KnowledgeAdapterRegistry()
    register_slack_conversation_knowledge_adapter(adapter_registry)
    facade = VendorKnowledgeFacadeService(
        tenant_id="tenant-1",
        resolver=_SlackResolver(integration=integration),
        adapter_registry=adapter_registry,
    )
    source = to_source_ref(
        KnowledgeSourceBinding(
            binding_id="slack-conversation-binding",
            tenant_id="tenant-1",
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
            source_kind=SLACK_CONVERSATION_SOURCE_KIND,
            connection_ref="conn-1",
            safe_display_name="Slack Binding",
            scope=KnowledgeSourceScope(
                remote_scope_id=encode_slack_conversation_scope_id(
                    conversation_id=_CONVERSATION_ID,
                    conversation_kind=SlackConversationKind.PUBLIC_CHANNEL,
                    oldest=_OLDEST,
                    latest=_LATEST,
                ),
                remote_scope_type=SLACK_CONVERSATION_SCOPE_TYPE,
                safe_display_name="General",
                parameters={},
            ),
            status=KnowledgeSourceBindingStatus.ACTIVE,
            configuration_version=1,
        )
    )
    page = await facade.read_page(source=source, cursor=None, limit=1)
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    first_revision = descriptor.revision.version
    edited_message = _message(
        message_ts=_ROOT_TS,
        text="edited root",
        reply_count=1,
        edited_at=datetime(2024, 1, 4, 12, 0, tzinfo=timezone.utc),
    )
    fake._content[_ROOT_TS] = edited_message
    edited_page = SlackConversationMessagePage(
        conversation_id=_CONVERSATION_ID,
        oldest=_OLDEST,
        latest=_LATEST,
        items=(edited_message,),
        next_cursor="history-2",
    )
    fake._history_pages[0] = edited_page
    fake._history_backup[0] = edited_page
    page2 = await facade.read_page(source=source, cursor=None, limit=1)
    descriptor2 = page2.changes[0].descriptor
    assert descriptor2 is not None
    assert descriptor2.revision.version != first_revision
    content = await facade.fetch_content(source=source, item=descriptor2)
    assert content.structured_record is not None
    assert content.structured_record["text"] == "edited root"


@pytest.mark.asyncio
async def test_slack_multi_page_thread_proof_without_lost_or_duplicate_replies() -> (
    None
):
    fake = _SlackFakeIntegration()
    reply_two = _message(
        message_ts="1704153604.000001",
        text="reply two",
        root_thread_ts=_ROOT_TS,
    )
    fake._reply_pages = [
        SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(
                _message(
                    message_ts=_REPLY_TS, text="reply one", root_thread_ts=_ROOT_TS
                ),
            ),
            next_cursor="reply-page-2",
        ),
        SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(reply_two,),
        ),
    ]
    fake._reply_backup = list(fake._reply_pages)
    coordinator, sink, checkpoint_repo, state_repo, fake, integration = (
        _build_coordinator(fake)
    )
    results = await _reconcile_until_complete(coordinator)
    reply_timestamps = []
    root_count = 0
    for batch in sink.calls:
        for envelope in batch.envelopes:
            record = envelope.content.structured_record if envelope.content else None
            if record is None:
                continue
            message_ts = record["message"]["message_ts"]
            if message_ts == _ROOT_TS:
                root_count += 1
            if record["thread"]["root_thread_ts"] == _ROOT_TS:
                reply_timestamps.append(message_ts)
    assert root_count == 1
    assert reply_timestamps.count(_REPLY_TS) == 1
    assert reply_timestamps.count("1704153604.000001") == 1
    assert len(fake.reply_calls) == 2
    assert fake.reply_calls[0]["cursor"] is None
    assert fake.reply_calls[1]["cursor"] == "reply-page-2"
    assert results[-1].has_more is False
    assert (
        checkpoint_repo.get(
            tenant_id="tenant-1", binding_id="slack-conversation-binding"
        )
        is not None
    )


@pytest.mark.asyncio
async def test_slack_reply_page_two_sink_failure_retries_safely() -> None:
    fake = _SlackFakeIntegration()
    reply_two_ts = "1704153604.000001"
    history_page_two_ts = "1704153605.000001"
    history_page_one = SlackConversationMessagePage(
        conversation_id=_CONVERSATION_ID,
        oldest=_OLDEST,
        latest=_LATEST,
        items=(_message(message_ts=_ROOT_TS, text="root one", reply_count=1),),
        next_cursor="history-page-2",
    )
    history_page_two = SlackConversationMessagePage(
        conversation_id=_CONVERSATION_ID,
        oldest=_OLDEST,
        latest=_LATEST,
        items=(_message(message_ts=history_page_two_ts, text="root two"),),
    )
    fake._history_pages = [history_page_one, history_page_two]
    fake._history_backup = [history_page_one, history_page_two]
    fake._reply_pages = [
        SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(
                _message(
                    message_ts=_REPLY_TS, text="reply one", root_thread_ts=_ROOT_TS
                ),
            ),
            next_cursor="reply-page-2",
        ),
        SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(
                _message(
                    message_ts=reply_two_ts, text="reply two", root_thread_ts=_ROOT_TS
                ),
            ),
        ),
    ]
    fake._reply_backup = list(fake._reply_pages)
    coordinator, sink, checkpoint_repo, state_repo, fake, integration = (
        _build_coordinator(fake)
    )
    integration_id = id(integration)
    root_result = await coordinator.reconcile_once(
        binding_id="slack-conversation-binding",
        restart=True,
        operation_id="slack-multi-page",
    )
    assert root_result.status is KnowledgeSyncRunStatus.COMPLETED
    reply_one_result = await coordinator.reconcile_once(
        binding_id="slack-conversation-binding",
        restart=False,
        operation_id="slack-multi-page",
        trigger_delivery_id=root_result.delivery_id,
    )
    assert reply_one_result.status is KnowledgeSyncRunStatus.COMPLETED
    checkpoint_after_page_one = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id="slack-conversation-binding",
    )
    sink.fail_times = 1
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="slack-conversation-binding",
            restart=False,
            operation_id="slack-multi-page",
            trigger_delivery_id=reply_one_result.delivery_id,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    checkpoint_after_failure = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id="slack-conversation-binding",
    )
    assert checkpoint_after_failure == checkpoint_after_page_one
    failed_delivery_id = sink.calls[2].delivery_id
    reply_two_remote_id = sink.calls[2].envelopes[0].remote_id
    assert (
        state_repo.get(
            tenant_id="tenant-1",
            binding_id="slack-conversation-binding",
            remote_id=reply_two_remote_id,
        )
        is None
    )
    assert len(fake.history_calls) == 1
    assert len(fake.reply_calls) == 2
    assert fake.reply_calls[1]["cursor"] == "reply-page-2"
    assert not any(
        call.get("cursor") == "history-page-2" for call in fake.history_calls
    )
    retry_result = await coordinator.reconcile_once(
        binding_id="slack-conversation-binding",
        restart=False,
        operation_id="slack-multi-page",
        trigger_delivery_id=reply_one_result.delivery_id,
    )
    assert retry_result.status is KnowledgeSyncRunStatus.COMPLETED
    assert retry_result.delivery_id == failed_delivery_id
    assert len(sink.calls) == 4
    assert len(sink.durable_delivery_ids) == 3
    assert sink.calls[1].envelopes[0].remote_id != sink.calls[3].envelopes[0].remote_id
    assert sink.calls[3].envelopes[0].remote_id == reply_two_remote_id
    assert len(fake.reply_calls) == 3
    assert fake.reply_calls[2]["cursor"] == "reply-page-2"
    assert len(fake.history_calls) == 1
    history_result = await coordinator.reconcile_once(
        binding_id="slack-conversation-binding",
        restart=False,
        operation_id="slack-multi-page",
        trigger_delivery_id=retry_result.delivery_id,
    )
    assert history_result.status is KnowledgeSyncRunStatus.COMPLETED
    assert len(fake.history_calls) == 2
    assert fake.history_calls[1]["cursor"] == "history-page-2"
    delivered_root_timestamps = []
    for batch in sink.calls:
        for envelope in batch.envelopes:
            record = envelope.content.structured_record if envelope.content else None
            if record is None:
                continue
            message_ts = record["message"]["message_ts"]
            if record["thread"]["root_thread_ts"] is None:
                delivered_root_timestamps.append(message_ts)
    assert delivered_root_timestamps.count(_ROOT_TS) == 1
    assert history_page_two_ts in delivered_root_timestamps
    assert id(integration) == integration_id
    assert history_result.has_more is False
    final_checkpoint = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id="slack-conversation-binding",
    )
    assert final_checkpoint is not None
    assert final_checkpoint != checkpoint_after_failure


@pytest.mark.asyncio
async def test_malformed_terminal_history_cursor_blocks_checkpoint_acceptance() -> None:
    from intergrax.integrations.providers.conversation_channel.slack.backend import (
        SlackConversationChannelBackend,
    )

    class _MalformedTerminalHistoryClient:
        async def conversations_history(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs.get("cursor") == "history-page-2":
                return {
                    "ok": True,
                    "messages": [
                        {
                            "ts": _EDITED_TS,
                            "user": "U111",
                            "text": "edited body",
                            "edited": {"ts": "1704153700.000001"},
                        }
                    ],
                    "response_metadata": {"next_cursor": 123},
                }
            return {
                "ok": True,
                "messages": [
                    {
                        "ts": _ROOT_TS,
                        "user": "U111",
                        "text": "root message",
                    }
                ],
                "response_metadata": {"next_cursor": "history-page-2"},
            }

        async def conversations_replies(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("replies should not be requested")

    config = SlackConversationChannelIntegrationConfig(
        enabled=True,
        app_token="xapp-test-token-value",
        bot_token="xoxb-test-token-value",
    )
    backend = SlackConversationChannelBackend(
        config=config,
        web_client=_MalformedTerminalHistoryClient(),
    )
    coordinator, sink, checkpoint_repo, _, _, _ = _build_coordinator(backend)  # type: ignore[arg-type]
    first = await coordinator.reconcile_once(
        binding_id="slack-conversation-binding",
        restart=True,
        operation_id="slack-malformed-terminal",
    )
    assert first.status is KnowledgeSyncRunStatus.COMPLETED
    assert first.has_more is True
    checkpoint_before_terminal = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id="slack-conversation-binding",
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="slack-conversation-binding",
            restart=False,
            operation_id="slack-malformed-terminal",
            trigger_delivery_id=first.delivery_id,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert exc_info.value.retryable is False
    checkpoint_after_failure = checkpoint_repo.get(
        tenant_id="tenant-1",
        binding_id="slack-conversation-binding",
    )
    assert checkpoint_after_failure == checkpoint_before_terminal
    assert len(sink.calls) == 1
