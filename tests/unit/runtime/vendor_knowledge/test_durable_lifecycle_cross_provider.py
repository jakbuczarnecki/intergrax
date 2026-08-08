# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cross-provider durable lifecycle proofs through DocumentStoreDurableKnowledgeSyncSink.

Slack adapter sync unit tests remain baseline-stale (6 failures). Slack durable
proof therefore applies authentic Slack structured-record envelopes through the
generic durable sink. Microsoft Graph Teams Chat proves the full
adapter → coordinator → durable sink path with indexing disabled.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SLACK_CONVERSATION_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_chat import (
    MSGRAPH_TEAMS_CHAT_SCOPE_TYPE,
    encode_msgraph_teams_chat_scope_id,
    register_msgraph_teams_chat_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
)
from intergrax.runtime.vendor_knowledge.durable_materialization import (
    DocumentStoreDurableKnowledgeSyncSink,
    DurableMaterializedItemStatus,
    durable_batch_payload_fingerprint,
)
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.live.ms365_graph.registration import (
    build_msgraph_teams_chat_vendor_knowledge_source_plugin,
)
from intergrax.runtime.vendor_knowledge.live.slack.registration import (
    build_slack_vendor_knowledge_source_plugin,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChangeKind,
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeItemDescriptor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.plugin import VendorKnowledgeMode
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
    KnowledgeSyncBatch,
    KnowledgeSyncEnvelope,
    KnowledgeSyncMode,
    KnowledgeSyncRunStatus,
    KnowledgeSyncSinkReceiptStatus,
)
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    RecordingBindingService,
    durable_reconcile_until_complete,
    durable_reconciliation_coordinator_kwargs,
)
from tests.unit.runtime.vendor_knowledge.test_msgraph_teams_chat_knowledge_sync import (
    _CHAT_ID,
    _ETAG_1,
    _MAILBOX_USER_ID,
    _MSG_1,
    _deleted_message,
    _encode_message_remote_id,
    _snapshot_page,
    _TeamsChatFakeCollaborationSuite,
    _TeamsChatTestIntegration,
    _window,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_SLACK_REMOTE = "C01234567:1704153600.000001"
_SLACK_SCHEMA = "slack.conversation.message.knowledge.v1"


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _slack_source() -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        connection_ref="conn-1",
        scope=KnowledgeSourceScope(
            remote_scope_id="slack-scope-1",
            remote_scope_type="slack.conversation.scope.v2",
            safe_display_name="General",
            parameters={},
        ),
    )


def _slack_envelope(*, text: str, version: str) -> KnowledgeSyncEnvelope:
    return KnowledgeSyncEnvelope(
        change_kind=KnowledgeChangeKind.UPSERT,
        remote_id=_SLACK_REMOTE,
        descriptor=KnowledgeItemDescriptor(
            identity=KnowledgeItemIdentity(remote_id=_SLACK_REMOTE),
            revision=KnowledgeItemRevision(
                version=version,
                updated_at=datetime(2024, 1, 2, 12, 0, tzinfo=UTC),
            ),
            title="Slack message",
            item_type="message",
            content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
            content_available=True,
            provenance=KnowledgeItemProvenance(
                provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
                source_kind=SLACK_CONVERSATION_SOURCE_KIND,
                remote_id=_SLACK_REMOTE,
            ),
            metadata={},
        ),
        content=KnowledgeContent(
            mode=KnowledgeContentMode.STRUCTURED_RECORD,
            structured_record={
                "schema": _SLACK_SCHEMA,
                "provider": "slack",
                "source_kind": "slack_conversation",
                "conversation": {
                    "conversation_id": "C01234567",
                    "safe_display_name": "General",
                },
                "message": {"message_ts": "1704153600.000001"},
                "thread": {"root_thread_ts": None, "reply_count": 0},
                "actor": {"provider_id": "U111"},
                "text": text,
                "timestamps": {
                    "created_at": "2024-01-02T12:00:00+00:00",
                    "edited_at": None,
                },
                "edit_state": {"edited": False},
                "safe_file_inventory": [],
            },
        ),
    )


def _coordinator_kwargs(
    document_store: InMemoryDocumentStore,
    sink: DocumentStoreDurableKnowledgeSyncSink,
):
    state_repo = DocumentStoreKnowledgeRemoteItemStateRepository(document_store)
    kwargs = durable_reconciliation_coordinator_kwargs(
        state_repository=state_repo,
        document_store=document_store,
    )
    kwargs["sink_receipt_inspector"] = sink
    return state_repo, kwargs


def _build_teams_chat_durable(fake: _TeamsChatFakeCollaborationSuite | None = None):
    backend = fake or _TeamsChatFakeCollaborationSuite()
    integration = _TeamsChatTestIntegration.from_client(backend, enabled=True)

    class _GraphResolver:
        def resolve(self, *, source):
            return integration

    registry = KnowledgeAdapterRegistry()
    register_msgraph_teams_chat_knowledge_adapter(registry)
    facade = VendorKnowledgeFacadeService(
        tenant_id="tenant-1",
        resolver=_GraphResolver(),
        adapter_registry=registry,
    )
    document_store = InMemoryDocumentStore()
    sink = DocumentStoreDurableKnowledgeSyncSink(document_store)
    state_repo, recon_kwargs = _coordinator_kwargs(document_store, sink)
    binding = KnowledgeSourceBinding(
        binding_id="teams-chat-durable-binding",
        tenant_id="tenant-1",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
        connection_ref="conn-1",
        safe_display_name="Teams Chat Durable Binding",
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
        lease_repository=DocumentStoreKnowledgeSourceLeaseRepository(document_store),
        checkpoint_repository=DocumentStoreKnowledgeSyncCheckpointRepository(
            document_store
        ),
        item_state_repository=state_repo,
        sink=sink,
        lease_ttl_seconds=30,
        **recon_kwargs,
    )
    return coordinator, sink, state_repo, backend


async def test_slack_durable_materialization_of_structured_records() -> None:
    slack_plugin = build_slack_vendor_knowledge_source_plugin()
    assert slack_plugin.supports(VendorKnowledgeMode.DURABLE)
    assert slack_plugin.supports(VendorKnowledgeMode.INDEXED)

    store = InMemoryDocumentStore()
    sink = DocumentStoreDurableKnowledgeSyncSink(store)
    batch = KnowledgeSyncBatch(
        tenant_id="tenant-1",
        binding_id="slack-durable-binding",
        binding_configuration_version=1,
        source=_slack_source(),
        mode=KnowledgeSyncMode.INCREMENTAL,
        delivery_id=_sha("slack-durable-1"),
        envelopes=(_slack_envelope(text="hello from slack", version="1"),),
        has_more=False,
    )
    await sink.apply_batch(batch=batch)
    await sink.apply_batch(batch=batch)
    item = sink.get_item(
        tenant_id="tenant-1",
        binding_id="slack-durable-binding",
        remote_id=_SLACK_REMOTE,
    )
    assert item is not None
    assert item.status is DurableMaterializedItemStatus.ACTIVE
    assert item.content is not None
    assert item.content.structured_record is not None
    assert item.content.structured_record["schema"] == _SLACK_SCHEMA
    assert item.content.structured_record["text"] == "hello from slack"
    assert "xoxb" not in str(item.model_dump(mode="json"))
    receipt = sink.inspect_receipt(
        tenant_id="tenant-1",
        binding_id="slack-durable-binding",
        delivery_id=batch.delivery_id,
        prepared_batch_payload_fingerprint=durable_batch_payload_fingerprint(batch),
    )
    assert receipt.status is KnowledgeSyncSinkReceiptStatus.APPLIED
    # Slack tombstones are UNPROVEN at the adapter (tombstones=False); absence
    # must not be inferred as deletion by the durable sink.
    assert sink.list_active_remote_ids(
        tenant_id="tenant-1",
        binding_id="slack-durable-binding",
    ) == (_SLACK_REMOTE,)


async def test_teams_chat_durable_materialization_does_not_require_indexing() -> None:
    teams_plugin = build_msgraph_teams_chat_vendor_knowledge_source_plugin()
    assert teams_plugin.supports(VendorKnowledgeMode.DURABLE)
    assert teams_plugin.supports(VendorKnowledgeMode.INDEXED)

    coordinator, sink, _state_repo, _backend = _build_teams_chat_durable()
    results = await durable_reconcile_until_complete(
        coordinator,
        binding_id="teams-chat-durable-binding",
        operation_id="teams-chat-durable-recon",
    )
    assert results
    assert all(result.status is KnowledgeSyncRunStatus.COMPLETED for result in results)
    active = sink.list_active_remote_ids(
        tenant_id="tenant-1",
        binding_id="teams-chat-durable-binding",
    )
    assert len(active) >= 1
    sample = sink.get_item(
        tenant_id="tenant-1",
        binding_id="teams-chat-durable-binding",
        remote_id=active[0],
    )
    assert sample is not None
    assert sample.content is not None
    assert sample.content.mode is KnowledgeContentMode.STRUCTURED_RECORD
    assert sample.content.structured_record is not None
    assert sample.content.structured_record["schema"] == (
        "msgraph.teams-chat.message.knowledge.v1"
    )
    assert type(sink).__name__ == "DocumentStoreDurableKnowledgeSyncSink"


async def test_teams_chat_authoritative_deletion_clears_active_durable_state() -> None:
    fake = _TeamsChatFakeCollaborationSuite()
    fake._snapshot_pages = [
        _snapshot_page(
            items=(_deleted_message(remote_id=_MSG_1, revision=_ETAG_1),),
        ),
    ]
    fake._snapshot_pages_backup = list(fake._snapshot_pages)
    coordinator, sink, state_repo, _ = _build_teams_chat_durable(fake)
    await durable_reconcile_until_complete(
        coordinator,
        binding_id="teams-chat-durable-binding",
        operation_id="teams-chat-durable-delete",
    )
    remote_id = _encode_message_remote_id(message_remote_id=_MSG_1)
    item = sink.get_item(
        tenant_id="tenant-1",
        binding_id="teams-chat-durable-binding",
        remote_id=remote_id,
    )
    assert item is not None
    assert item.status is DurableMaterializedItemStatus.DELETED
    assert item.content is None
    assert remote_id not in sink.list_active_remote_ids(
        tenant_id="tenant-1",
        binding_id="teams-chat-durable-binding",
    )
    state = state_repo.get(
        tenant_id="tenant-1",
        binding_id="teams-chat-durable-binding",
        remote_id=remote_id,
    )
    assert state is not None
    assert state.status is KnowledgeRemoteItemStatus.DELETED
