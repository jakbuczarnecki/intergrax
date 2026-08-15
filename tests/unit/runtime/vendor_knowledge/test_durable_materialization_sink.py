# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit proofs for DocumentStoreDurableKnowledgeSyncSink lifecycle semantics."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime

import pytest

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.durable_materialization import (
    DocumentStoreDurableKnowledgeSyncSink,
    DurableDeliveryReceiptStatus,
    DurableDeliveryReceiptV1,
    DurableMaterializedItemStatus,
    durable_batch_payload_fingerprint,
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
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeSyncBatch,
    KnowledgeSyncEnvelope,
    KnowledgeSyncMode,
    KnowledgeSyncSinkReceiptStatus,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_BINDING_1 = "binding-1"
_BINDING_2 = "binding-2"


def _source(*, tenant_id: str = _TENANT_A, provider_id: str = "provider-x") -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id=tenant_id,
        provider_id=provider_id,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind="example_source",
        connection_ref="conn-1",
        scope=KnowledgeSourceScope(
            remote_scope_id="scope-1",
            remote_scope_type="example.scope.v1",
            safe_display_name="Example",
            parameters={},
        ),
    )


def _descriptor(
    *,
    remote_id: str,
    version: str,
    updated_at: datetime | None = None,
    provider_id: str = "provider-x",
) -> KnowledgeItemDescriptor:
    return KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id=remote_id),
        revision=KnowledgeItemRevision(version=version, updated_at=updated_at),
        title=f"Title {remote_id}",
        item_type="record",
        content_mode=KnowledgeContentMode.STRUCTURED_RECORD,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id=provider_id,
            source_kind="example_source",
            remote_id=remote_id,
        ),
        metadata={},
    )


def _content(remote_id: str, *, text: str) -> KnowledgeContent:
    return KnowledgeContent(
        mode=KnowledgeContentMode.STRUCTURED_RECORD,
        structured_record={
            "schema": "example.record.v1",
            "remote_id": remote_id,
            "text": text,
        },
    )


def _delivery_id(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _batch(
    *,
    tenant_id: str = _TENANT_A,
    binding_id: str = _BINDING_1,
    delivery_seed: str,
    envelopes: tuple[KnowledgeSyncEnvelope, ...],
    provider_id: str = "provider-x",
) -> KnowledgeSyncBatch:
    return KnowledgeSyncBatch(
        tenant_id=tenant_id,
        binding_id=binding_id,
        binding_configuration_version=1,
        source=_source(tenant_id=tenant_id, provider_id=provider_id),
        mode=KnowledgeSyncMode.INCREMENTAL,
        delivery_id=_delivery_id(delivery_seed),
        envelopes=envelopes,
        has_more=False,
    )


async def test_apply_batch_materializes_and_is_idempotent() -> None:
    store = InMemoryDocumentStore()
    sink = DocumentStoreDurableKnowledgeSyncSink(store)
    batch = _batch(
        delivery_seed="d1",
        envelopes=(
            KnowledgeSyncEnvelope(
                change_kind=KnowledgeChangeKind.UPSERT,
                remote_id="item-1",
                descriptor=_descriptor(remote_id="item-1", version="1"),
                content=_content("item-1", text="hello"),
            ),
        ),
    )
    await sink.apply_batch(batch=batch)
    await sink.apply_batch(batch=batch)
    item = sink.get_item(
        tenant_id=_TENANT_A,
        binding_id=_BINDING_1,
        remote_id="item-1",
    )
    assert item is not None
    assert item.status is DurableMaterializedItemStatus.ACTIVE
    assert item.content is not None
    assert item.content.structured_record is not None
    assert item.content.structured_record["text"] == "hello"
    receipt = sink.inspect_receipt(
        tenant_id=_TENANT_A,
        binding_id=_BINDING_1,
        delivery_id=batch.delivery_id,
        prepared_batch_payload_fingerprint=durable_batch_payload_fingerprint(batch),
    )
    assert receipt.status is KnowledgeSyncSinkReceiptStatus.APPLIED


async def test_crash_after_prepare_is_recoverable() -> None:
    store = InMemoryDocumentStore()
    sink = DocumentStoreDurableKnowledgeSyncSink(store)
    batch = _batch(
        delivery_seed="crash-prepare",
        envelopes=(
            KnowledgeSyncEnvelope(
                change_kind=KnowledgeChangeKind.UPSERT,
                remote_id="item-1",
                descriptor=_descriptor(remote_id="item-1", version="1"),
                content=_content("item-1", text="prepared"),
            ),
        ),
    )
    fingerprint = durable_batch_payload_fingerprint(batch)
    # Simulate crash after APPLYING receipt, before item write / APPLIED.
    generation = sink._allocate_generation(tenant_id=_TENANT_A, binding_id=_BINDING_1)
    receipt = DurableDeliveryReceiptV1(
        tenant_id=_TENANT_A,
        binding_id=_BINDING_1,
        binding_configuration_version=1,
        delivery_id=batch.delivery_id,
        status=DurableDeliveryReceiptStatus.APPLYING,
        payload_fingerprint=fingerprint,
        materialization_generation=generation,
        record_version=secrets.token_urlsafe(16),
    )
    assert sink._put_receipt_if_absent(receipt)
    assert (
        sink.get_item(tenant_id=_TENANT_A, binding_id=_BINDING_1, remote_id="item-1")
        is None
    )
    await sink.apply_batch(batch=batch)
    item = sink.get_item(
        tenant_id=_TENANT_A,
        binding_id=_BINDING_1,
        remote_id="item-1",
    )
    assert item is not None
    assert item.status is DurableMaterializedItemStatus.ACTIVE
    inspect = sink.inspect_receipt(
        tenant_id=_TENANT_A,
        binding_id=_BINDING_1,
        delivery_id=batch.delivery_id,
        prepared_batch_payload_fingerprint=fingerprint,
    )
    assert inspect.status is KnowledgeSyncSinkReceiptStatus.APPLIED


async def test_revision_update_and_older_replay_rejected() -> None:
    store = InMemoryDocumentStore()
    sink = DocumentStoreDurableKnowledgeSyncSink(store)
    older = datetime(2024, 1, 1, tzinfo=UTC)
    newer = datetime(2024, 2, 1, tzinfo=UTC)
    first = _batch(
        delivery_seed="rev-1",
        envelopes=(
            KnowledgeSyncEnvelope(
                change_kind=KnowledgeChangeKind.UPSERT,
                remote_id="item-1",
                descriptor=_descriptor(
                    remote_id="item-1", version="1", updated_at=older
                ),
                content=_content("item-1", text="v1"),
            ),
        ),
    )
    second = _batch(
        delivery_seed="rev-2",
        envelopes=(
            KnowledgeSyncEnvelope(
                change_kind=KnowledgeChangeKind.UPSERT,
                remote_id="item-1",
                descriptor=_descriptor(
                    remote_id="item-1", version="2", updated_at=newer
                ),
                content=_content("item-1", text="v2"),
            ),
        ),
    )
    await sink.apply_batch(batch=first)
    await sink.apply_batch(batch=second)
    # Replay older delivery: receipt idempotent, must not clobber newer content.
    await sink.apply_batch(batch=first)
    item = sink.get_item(
        tenant_id=_TENANT_A,
        binding_id=_BINDING_1,
        remote_id="item-1",
    )
    assert item is not None
    assert item.content is not None
    assert item.content.structured_record is not None
    assert item.content.structured_record["text"] == "v2"
    assert item.revision is not None
    assert item.revision.version == "2"


async def test_authoritative_deletion_removes_active_state() -> None:
    store = InMemoryDocumentStore()
    sink = DocumentStoreDurableKnowledgeSyncSink(store)
    upsert = _batch(
        delivery_seed="del-1",
        envelopes=(
            KnowledgeSyncEnvelope(
                change_kind=KnowledgeChangeKind.UPSERT,
                remote_id="item-1",
                descriptor=_descriptor(remote_id="item-1", version="1"),
                content=_content("item-1", text="alive"),
            ),
        ),
    )
    delete = _batch(
        delivery_seed="del-2",
        envelopes=(
            KnowledgeSyncEnvelope(
                change_kind=KnowledgeChangeKind.DELETED,
                remote_id="item-1",
            ),
        ),
    )
    await sink.apply_batch(batch=upsert)
    await sink.apply_batch(batch=delete)
    item = sink.get_item(
        tenant_id=_TENANT_A,
        binding_id=_BINDING_1,
        remote_id="item-1",
    )
    assert item is not None
    assert item.status is DurableMaterializedItemStatus.DELETED
    assert item.content is None
    assert sink.list_active_remote_ids(tenant_id=_TENANT_A, binding_id=_BINDING_1) == ()


async def test_tenant_and_source_isolation() -> None:
    store = InMemoryDocumentStore()
    sink = DocumentStoreDurableKnowledgeSyncSink(store)
    batch_a = _batch(
        tenant_id=_TENANT_A,
        binding_id=_BINDING_1,
        delivery_seed="iso-a",
        envelopes=(
            KnowledgeSyncEnvelope(
                change_kind=KnowledgeChangeKind.UPSERT,
                remote_id="shared-remote",
                descriptor=_descriptor(remote_id="shared-remote", version="1"),
                content=_content("shared-remote", text="tenant-a"),
            ),
        ),
    )
    batch_b = _batch(
        tenant_id=_TENANT_B,
        binding_id=_BINDING_1,
        delivery_seed="iso-b",
        envelopes=(
            KnowledgeSyncEnvelope(
                change_kind=KnowledgeChangeKind.UPSERT,
                remote_id="shared-remote",
                descriptor=_descriptor(remote_id="shared-remote", version="1"),
                content=_content("shared-remote", text="tenant-b"),
            ),
        ),
    )
    batch_binding2 = _batch(
        tenant_id=_TENANT_A,
        binding_id=_BINDING_2,
        delivery_seed="iso-c",
        envelopes=(
            KnowledgeSyncEnvelope(
                change_kind=KnowledgeChangeKind.UPSERT,
                remote_id="shared-remote",
                descriptor=_descriptor(remote_id="shared-remote", version="1"),
                content=_content("shared-remote", text="binding-2"),
            ),
        ),
    )
    await sink.apply_batch(batch=batch_a)
    await sink.apply_batch(batch=batch_b)
    await sink.apply_batch(batch=batch_binding2)
    item_a = sink.get_item(
        tenant_id=_TENANT_A, binding_id=_BINDING_1, remote_id="shared-remote"
    )
    item_b = sink.get_item(
        tenant_id=_TENANT_B, binding_id=_BINDING_1, remote_id="shared-remote"
    )
    item_c = sink.get_item(
        tenant_id=_TENANT_A, binding_id=_BINDING_2, remote_id="shared-remote"
    )
    assert item_a is not None and item_a.content is not None
    assert item_b is not None and item_b.content is not None
    assert item_c is not None and item_c.content is not None
    assert item_a.content.structured_record["text"] == "tenant-a"
    assert item_b.content.structured_record["text"] == "tenant-b"
    assert item_c.content.structured_record["text"] == "binding-2"
    foreign = sink.inspect_receipt(
        tenant_id=_TENANT_B,
        binding_id=_BINDING_1,
        delivery_id=batch_a.delivery_id,
        prepared_batch_payload_fingerprint=durable_batch_payload_fingerprint(batch_a),
    )
    assert foreign.status is KnowledgeSyncSinkReceiptStatus.ABSENT
