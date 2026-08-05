# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for vendor knowledge synchronization models and ports."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChangeKind,
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgePermissions,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
    KnowledgeVisibility,
)
from intergrax.runtime.vendor_knowledge.sync_contracts import (
    KnowledgeRemoteItemStateRepository,
    KnowledgeSourceLeaseRepository,
    KnowledgeSyncCheckpointRepository,
    KnowledgeSyncSink,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeRemoteItemState,
    KnowledgeRemoteItemStatus,
    KnowledgeSourceLeaseToken,
    KnowledgeSyncBatch,
    KnowledgeSyncCheckpoint,
    KnowledgeSyncEnvelope,
    KnowledgeSyncMode,
    KnowledgeSyncRunResult,
    KnowledgeSyncRunStatus,
)
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    KnowledgeSyncPublicationFenceV1,
)
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    IdempotentRecordingSink,
    InMemoryCheckpointRepository,
    InMemoryLeaseRepository,
    InMemoryRemoteItemStateRepository,
    make_descriptor,
)

_DELIVERY = "a" * 64


def _source(*, tenant_id: str = "tenant-1") -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id=tenant_id,
        provider_id="example",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        connection_ref="conn-1",
        scope=KnowledgeSourceScope(
            remote_scope_id="scope-1",
            remote_scope_type="project",
            safe_display_name="Example Project",
        ),
    )


@pytest.mark.unit
def test_sync_models_are_strict_and_frozen() -> None:
    token = KnowledgeSourceLeaseToken(
        tenant_id="tenant-1",
        binding_id="binding-1",
        owner_id="owner-1",
        token="opaque",
    )
    with pytest.raises(ValidationError):
        KnowledgeSourceLeaseToken(
            tenant_id="tenant-1",
            binding_id="binding-1",
            owner_id="owner-1",
            token="opaque",
            extra="nope",  # type: ignore[call-arg]
        )
    with pytest.raises(ValidationError):
        token.tenant_id = "other"  # type: ignore[misc]


@pytest.mark.unit
def test_identifiers_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        KnowledgeSourceLeaseToken(
            tenant_id=" ",
            binding_id="binding-1",
            owner_id="owner-1",
            token="opaque",
        )


@pytest.mark.unit
def test_publication_fence_is_strict_frozen_and_lifecycle_safe() -> None:
    fence = KnowledgeSyncPublicationFenceV1(
        tenant_id="tenant-1",
        binding_id="binding-1",
        lifecycle_revision=1,
        lifecycle_token="opaque-token",
        enabled=True,
        detached=False,
    )
    with pytest.raises(ValidationError):
        fence.lifecycle_revision = 2  # type: ignore[misc]
    with pytest.raises(ValidationError):
        KnowledgeSyncPublicationFenceV1(
            tenant_id="tenant-1",
            binding_id="binding-1",
            lifecycle_revision=True,  # type: ignore[arg-type]
            lifecycle_token="opaque-token",
            enabled=True,
            detached=False,
        )
    with pytest.raises(ValidationError):
        KnowledgeSyncPublicationFenceV1(
            tenant_id="tenant-1",
            binding_id="binding-1",
            lifecycle_revision=1,
            lifecycle_token="opaque-token",
            enabled=True,
            detached=True,
        )
    assert "opaque-token" not in repr(fence)


@pytest.mark.unit
def test_lease_token_hidden_in_repr() -> None:
    token = KnowledgeSourceLeaseToken(
        tenant_id="tenant-1",
        binding_id="binding-1",
        owner_id="owner-1",
        token="super-secret-lease",
    )
    assert "super-secret-lease" not in repr(token)


@pytest.mark.unit
def test_checkpoint_configuration_version_minimum() -> None:
    with pytest.raises(ValidationError):
        KnowledgeSyncCheckpoint(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=0,
            cursor=KnowledgeCursor(value="c1"),
        )
    checkpoint = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value="c1"),
    )
    assert checkpoint.binding_configuration_version == 1


@pytest.mark.unit
def test_active_remote_state_requires_revision() -> None:
    with pytest.raises(ValidationError):
        KnowledgeRemoteItemState(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            provider_id="example",
            source_kind="issues",
            remote_id="item-1",
            status=KnowledgeRemoteItemStatus.ACTIVE,
            revision=None,
            last_delivery_id=_DELIVERY,
        )
    state = KnowledgeRemoteItemState(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        remote_id="item-1",
        status=KnowledgeRemoteItemStatus.DELETED,
        revision=None,
        last_delivery_id=_DELIVERY,
    )
    assert state.revision is None


@pytest.mark.unit
def test_envelope_content_mode_consistency() -> None:
    descriptor = make_descriptor(content_mode=KnowledgeContentMode.STRUCTURED_RECORD)
    with pytest.raises(ValidationError):
        KnowledgeSyncEnvelope(
            change_kind=KnowledgeChangeKind.UPSERT,
            remote_id="item-1",
            descriptor=descriptor,
            content=KnowledgeContent(mode=KnowledgeContentMode.RICH_TEXT, rich_text="x"),
        )


@pytest.mark.unit
def test_tombstone_envelope_rejects_content_and_permissions() -> None:
    with pytest.raises(ValidationError):
        KnowledgeSyncEnvelope(
            change_kind=KnowledgeChangeKind.DELETED,
            remote_id="item-1",
            content=KnowledgeContent(
                mode=KnowledgeContentMode.STRUCTURED_RECORD,
                structured_record={"a": 1},
            ),
        )
    with pytest.raises(ValidationError):
        KnowledgeSyncEnvelope(
            change_kind=KnowledgeChangeKind.REVOKED,
            remote_id="item-1",
            permissions=KnowledgePermissions(visibility=KnowledgeVisibility.TENANT),
        )


@pytest.mark.unit
def test_batch_tenant_source_consistency() -> None:
    with pytest.raises(ValidationError):
        KnowledgeSyncBatch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            source=_source(tenant_id="tenant-2"),
            mode=KnowledgeSyncMode.INCREMENTAL,
            delivery_id=_DELIVERY,
            envelopes=(),
            has_more=False,
        )


@pytest.mark.unit
def test_run_result_completed_and_lease_busy_rules() -> None:
    busy = KnowledgeSyncRunResult(
        status=KnowledgeSyncRunStatus.LEASE_BUSY,
        mode=KnowledgeSyncMode.INCREMENTAL,
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=None,
        changes_count=0,
        active_count=0,
        tombstone_count=0,
        checkpoint_advanced=False,
        has_more=False,
        retryable=True,
    )
    assert busy.retryable is True
    with pytest.raises(ValidationError):
        KnowledgeSyncRunResult(
            status=KnowledgeSyncRunStatus.LEASE_BUSY,
            mode=KnowledgeSyncMode.INCREMENTAL,
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=_DELIVERY,
            changes_count=0,
            active_count=0,
            tombstone_count=0,
            checkpoint_advanced=False,
            has_more=False,
            retryable=True,
        )
    completed = KnowledgeSyncRunResult(
        status=KnowledgeSyncRunStatus.COMPLETED,
        mode=KnowledgeSyncMode.INCREMENTAL,
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=_DELIVERY,
        changes_count=1,
        active_count=1,
        tombstone_count=0,
        checkpoint_advanced=True,
        has_more=False,
        retryable=False,
    )
    assert completed.delivery_id == _DELIVERY
    with pytest.raises(ValidationError):
        KnowledgeSyncRunResult(
            status=KnowledgeSyncRunStatus.COMPLETED,
            mode=KnowledgeSyncMode.INCREMENTAL,
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=None,
            changes_count=0,
            active_count=0,
            tombstone_count=0,
            checkpoint_advanced=False,
            has_more=False,
            retryable=False,
        )


@pytest.mark.unit
def test_sync_protocols_are_runtime_checkable() -> None:
    assert isinstance(InMemoryLeaseRepository(), KnowledgeSourceLeaseRepository)
    assert isinstance(InMemoryCheckpointRepository(), KnowledgeSyncCheckpointRepository)
    assert isinstance(
        InMemoryRemoteItemStateRepository(), KnowledgeRemoteItemStateRepository
    )
    assert isinstance(IdempotentRecordingSink(), KnowledgeSyncSink)
