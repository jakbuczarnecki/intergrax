# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral test fakes for vendor knowledge synchronization coordinator tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
    to_source_ref,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeAdapterCapabilities,
    KnowledgeChange,
    KnowledgeChangeKind,
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgePage,
    KnowledgePermissions,
    KnowledgeScopeInfo,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
    KnowledgeVisibility,
)
from intergrax.runtime.vendor_knowledge.sync_contracts import (
    KnowledgeReconciliationRunConflict,
    KnowledgeSyncCheckpointConflict,
    KnowledgeSyncCorruptState,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeReconciliationRun,
    KnowledgeReconciliationRunCollecting,
    KnowledgeReconciliationRunPhase,
    KnowledgeRemoteItemState,
    KnowledgeRemoteItemStateReceipt,
    KnowledgeRemoteItemStateReceiptStatus,
    KnowledgeRemoteItemStatus,
    KnowledgeSourceLeaseToken,
    KnowledgeSyncBatch,
    KnowledgeSyncCheckpoint,
    KnowledgeSyncSinkReceipt,
    KnowledgeSyncSinkReceiptStatus,
)
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    KnowledgeSyncPublicationFenceV1,
)
from tests.unit.runtime.vendor_knowledge._fakes import make_content


def make_binding(
    *,
    binding_id: str = "binding-1",
    tenant_id: str = "tenant-1",
    provider_id: str = "example",
    integration_kind: IntegrationCategory = IntegrationCategory.ISSUE_TRACKER,
    source_kind: str = "issues",
    connection_ref: str = "conn-1",
    configuration_version: int = 1,
    status: KnowledgeSourceBindingStatus = KnowledgeSourceBindingStatus.ACTIVE,
) -> KnowledgeSourceBinding:
    return KnowledgeSourceBinding(
        binding_id=binding_id,
        tenant_id=tenant_id,
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        connection_ref=connection_ref,
        safe_display_name="Example Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id="scope-1",
            remote_scope_type="project",
            safe_display_name="Example Project",
            parameters={},
        ),
        status=status,
        configuration_version=configuration_version,
    )


def make_descriptor(
    *,
    remote_id: str = "item-1",
    provider_id: str = "example",
    source_kind: str = "issues",
    content_mode: KnowledgeContentMode = KnowledgeContentMode.STRUCTURED_RECORD,
    content_available: bool = True,
    version: str = "1",
) -> KnowledgeItemDescriptor:
    return KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id=remote_id),
        revision=KnowledgeItemRevision(version=version),
        title="Example item",
        item_type="record",
        content_mode=content_mode,
        content_available=content_available,
        provenance=KnowledgeItemProvenance(
            provider_id=provider_id,
            source_kind=source_kind,
            remote_id=remote_id,
        ),
        metadata={"label": "safe"},
    )


def make_change(
    *,
    kind: KnowledgeChangeKind = KnowledgeChangeKind.UPSERT,
    remote_id: str = "item-1",
    descriptor: KnowledgeItemDescriptor | None = None,
) -> KnowledgeChange:
    resolved = descriptor
    if resolved is None and kind in {
        KnowledgeChangeKind.UPSERT,
        KnowledgeChangeKind.METADATA_CHANGED,
        KnowledgeChangeKind.PERMISSIONS_CHANGED,
    }:
        resolved = make_descriptor(remote_id=remote_id)
    return KnowledgeChange(kind=kind, remote_id=remote_id, descriptor=resolved)


def make_page(
    *,
    changes: tuple[KnowledgeChange, ...] | None = None,
    next_cursor: KnowledgeCursor | None = None,
    proposed_checkpoint: KnowledgeCursor | None = None,
    has_more: bool = False,
) -> KnowledgePage:
    resolved_changes = changes
    if resolved_changes is None:
        resolved_changes = (make_change(),)
    return KnowledgePage(
        changes=resolved_changes,
        next_cursor=next_cursor,
        proposed_checkpoint=proposed_checkpoint,
        has_more=has_more,
    )


@dataclass
class InMemoryLeaseRepository:
    held: dict[tuple[str, str], KnowledgeSourceLeaseToken] = field(default_factory=dict)
    acquire_calls: list[dict[str, Any]] = field(default_factory=list)
    release_calls: list[KnowledgeSourceLeaseToken] = field(default_factory=list)
    acquire_error: Exception | None = None
    release_error: Exception | None = None
    force_busy: bool = False
    forced_token: KnowledgeSourceLeaseToken | None = None
    _counter: int = 0

    def acquire(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        owner_id: str,
        ttl_seconds: int,
    ) -> KnowledgeSourceLeaseToken | None:
        self.acquire_calls.append(
            {
                "tenant_id": tenant_id,
                "binding_id": binding_id,
                "owner_id": owner_id,
                "ttl_seconds": ttl_seconds,
            }
        )
        if self.acquire_error is not None:
            raise self.acquire_error
        if self.force_busy:
            return None
        if self.forced_token is not None:
            return self.forced_token
        key = (tenant_id, binding_id)
        if key in self.held:
            return None
        self._counter += 1
        token = KnowledgeSourceLeaseToken(
            tenant_id=tenant_id,
            binding_id=binding_id,
            owner_id=owner_id,
            token=f"lease-{self._counter}",
        )
        self.held[key] = token
        return token

    def release(self, *, lease: KnowledgeSourceLeaseToken) -> None:
        self.release_calls.append(lease)
        if self.release_error is not None:
            raise self.release_error
        key = (lease.tenant_id, lease.binding_id)
        current = self.held.get(key)
        if current is not None and current.token == lease.token:
            del self.held[key]

    def is_owned(self, *, lease: KnowledgeSourceLeaseToken) -> bool:
        current = self.held.get((lease.tenant_id, lease.binding_id))
        return current == lease


@dataclass
class InMemoryCheckpointRepository:
    checkpoints: dict[tuple[str, str], KnowledgeSyncCheckpoint] = field(
        default_factory=dict
    )
    get_calls: list[tuple[str, str]] = field(default_factory=list)
    commit_calls: list[dict[str, Any]] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    get_error: Exception | None = None
    commit_error: Exception | None = None
    fail_commit_times: int = 0
    forced_checkpoint: KnowledgeSyncCheckpoint | None = None

    def get(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSyncCheckpoint | None:
        self.get_calls.append((tenant_id, binding_id))
        if self.get_error is not None:
            raise self.get_error
        if self.forced_checkpoint is not None:
            return self.forced_checkpoint
        return self.checkpoints.get((tenant_id, binding_id))

    def commit(
        self,
        checkpoint: KnowledgeSyncCheckpoint,
        *,
        expected_previous: KnowledgeSyncCheckpoint | None,
        expected_publication_fence: KnowledgeSyncPublicationFenceV1 | None = None,
    ) -> None:
        _ = expected_publication_fence
        self.commit_calls.append(
            {"checkpoint": checkpoint, "expected_previous": expected_previous}
        )
        self.order.append("checkpoint")
        if self.fail_commit_times > 0:
            self.fail_commit_times -= 1
            raise KnowledgeSyncCheckpointConflict("checkpoint conflict")
        if self.commit_error is not None:
            raise self.commit_error
        key = (checkpoint.tenant_id, checkpoint.binding_id)
        current = self.checkpoints.get(key)
        if current != expected_previous:
            raise KnowledgeSyncCheckpointConflict("checkpoint conflict")
        self.checkpoints[key] = checkpoint


@dataclass
class InMemoryRemoteItemStateRepository:
    states: dict[tuple[str, str, str], KnowledgeRemoteItemState] = field(
        default_factory=dict
    )
    applied_delivery_ids: set[str] = field(default_factory=set)
    _delivery_markers: dict[tuple[str, str, str], dict[str, str]] = field(
        default_factory=dict
    )
    apply_calls: list[dict[str, Any]] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    apply_error: Exception | None = None
    get_error: Exception | None = None

    def get(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        remote_id: str,
    ) -> KnowledgeRemoteItemState | None:
        if self.get_error is not None:
            raise self.get_error
        return self.states.get((tenant_id, binding_id, remote_id))

    def apply_batch(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        states: tuple[KnowledgeRemoteItemState, ...],
        prepared_state_mutations_fingerprint: str | None = None,
        expected_publication_fence: KnowledgeSyncPublicationFenceV1 | None = None,
    ) -> None:
        _ = expected_publication_fence
        self.apply_calls.append(
            {
                "tenant_id": tenant_id,
                "binding_id": binding_id,
                "delivery_id": delivery_id,
                "states": states,
                "prepared_state_mutations_fingerprint": prepared_state_mutations_fingerprint,
            }
        )
        self.order.append("state")
        if self.apply_error is not None:
            raise self.apply_error
        marker_key = (tenant_id, binding_id, delivery_id)
        existing = self._delivery_markers.get(marker_key)
        if existing is not None:
            if prepared_state_mutations_fingerprint is not None:
                stored_fp = existing.get("prepared_state_mutations_fingerprint")
                if stored_fp is None:
                    raise KnowledgeSyncCorruptState(
                        "delivery marker cannot prove reconciliation receipt"
                    )
                if stored_fp != prepared_state_mutations_fingerprint:
                    raise KnowledgeSyncCorruptState(
                        "delivery marker prepared_state_mutations_fingerprint mismatch"
                    )
            if delivery_id in self.applied_delivery_ids:
                return
        elif delivery_id in self.applied_delivery_ids:
            return
        for state in states:
            self.states[(tenant_id, binding_id, state.remote_id)] = state
        self.applied_delivery_ids.add(delivery_id)
        if prepared_state_mutations_fingerprint is not None:
            self._delivery_markers[marker_key] = {
                "prepared_state_mutations_fingerprint": prepared_state_mutations_fingerprint,
                "status": "completed",
            }

    def inspect_delivery_receipt(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        prepared_state_mutations_fingerprint: str,
    ) -> KnowledgeRemoteItemStateReceipt:
        marker_key = (tenant_id, binding_id, delivery_id)
        marker = self._delivery_markers.get(marker_key)
        if marker is None:
            if delivery_id not in self.applied_delivery_ids:
                return KnowledgeRemoteItemStateReceipt(
                    status=KnowledgeRemoteItemStateReceiptStatus.ABSENT
                )
            return KnowledgeRemoteItemStateReceipt(
                status=KnowledgeRemoteItemStateReceiptStatus.CONFLICT,
                delivery_id=delivery_id,
                prepared_state_mutations_fingerprint=prepared_state_mutations_fingerprint,
            )
        stored_fp = marker.get("prepared_state_mutations_fingerprint")
        if stored_fp is None or stored_fp != prepared_state_mutations_fingerprint:
            return KnowledgeRemoteItemStateReceipt(
                status=KnowledgeRemoteItemStateReceiptStatus.CONFLICT,
                delivery_id=delivery_id,
                prepared_state_mutations_fingerprint=prepared_state_mutations_fingerprint,
            )
        if marker.get("status") == "applying":
            return KnowledgeRemoteItemStateReceipt(
                status=KnowledgeRemoteItemStateReceiptStatus.APPLYING,
                delivery_id=delivery_id,
                prepared_state_mutations_fingerprint=prepared_state_mutations_fingerprint,
            )
        return KnowledgeRemoteItemStateReceipt(
            status=KnowledgeRemoteItemStateReceiptStatus.COMPLETED,
            delivery_id=delivery_id,
            prepared_state_mutations_fingerprint=prepared_state_mutations_fingerprint,
        )


@dataclass
class RecordingSinkReceiptInspector:
    durable: dict[str, str] = field(default_factory=dict)
    unknown_delivery_ids: set[str] = field(default_factory=set)

    def inspect_receipt(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        prepared_batch_payload_fingerprint: str,
    ) -> KnowledgeSyncSinkReceipt:
        if delivery_id in self.unknown_delivery_ids:
            return KnowledgeSyncSinkReceipt(
                status=KnowledgeSyncSinkReceiptStatus.UNKNOWN,
                delivery_id=delivery_id,
                prepared_batch_payload_fingerprint=prepared_batch_payload_fingerprint,
            )
        stored = self.durable.get(delivery_id)
        if stored is None:
            return KnowledgeSyncSinkReceipt(
                status=KnowledgeSyncSinkReceiptStatus.ABSENT
            )
        if stored != prepared_batch_payload_fingerprint:
            return KnowledgeSyncSinkReceipt(
                status=KnowledgeSyncSinkReceiptStatus.CONFLICT,
                delivery_id=delivery_id,
                prepared_batch_payload_fingerprint=prepared_batch_payload_fingerprint,
            )
        return KnowledgeSyncSinkReceipt(
            status=KnowledgeSyncSinkReceiptStatus.APPLIED,
            delivery_id=delivery_id,
            prepared_batch_payload_fingerprint=prepared_batch_payload_fingerprint,
        )


@dataclass
class IdempotentRecordingSink:
    calls: list[KnowledgeSyncBatch] = field(default_factory=list)
    durable_delivery_ids: list[str] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    apply_error: Exception | None = None
    fail_times: int = 0

    async def apply_batch(self, *, batch: KnowledgeSyncBatch) -> None:
        self.calls.append(batch)
        self.order.append("sink")
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("sink boom")
        if self.apply_error is not None:
            raise self.apply_error
        if batch.delivery_id in self.durable_delivery_ids:
            return
        self.durable_delivery_ids.append(batch.delivery_id)


@dataclass
class RecordingBindingService:
    binding: KnowledgeSourceBinding
    get_calls: list[str] = field(default_factory=list)
    resolve_calls: list[str] = field(default_factory=list)
    get_error: Exception | None = None
    resolve_error: Exception | None = None
    source_override: KnowledgeSourceRef | None = None

    def get(self, binding_id: str) -> KnowledgeSourceBinding:
        self.get_calls.append(binding_id)
        if self.get_error is not None:
            raise self.get_error
        return self.binding

    def resolve_source(self, binding_id: str) -> KnowledgeSourceRef:
        self.resolve_calls.append(binding_id)
        if self.resolve_error is not None:
            raise self.resolve_error
        if self.source_override is not None:
            return self.source_override
        return to_source_ref(self.binding)


@dataclass
class RecordingFacade:
    pages_by_cursor: dict[str | None, KnowledgePage] = field(default_factory=dict)
    default_page: KnowledgePage | None = None
    capabilities: KnowledgeAdapterCapabilities = field(
        default_factory=lambda: KnowledgeAdapterCapabilities(
            full_inventory=True,
            incremental_changes=True,
            content_fetch=True,
            structured_content=True,
            permissions=True,
            tombstones=True,
            reconciliation=True,
        )
    )
    content: KnowledgeContent | None = None
    permissions: KnowledgePermissions | None = None
    inspect_calls: list[KnowledgeSourceRef] = field(default_factory=list)
    read_calls: list[dict[str, Any]] = field(default_factory=list)
    content_calls: list[KnowledgeItemDescriptor] = field(default_factory=list)
    permissions_calls: list[KnowledgeItemDescriptor] = field(default_factory=list)
    inspect_error: Exception | None = None
    read_error: Exception | None = None
    content_error: Exception | None = None
    permissions_error: Exception | None = None
    page_factory: Callable[[KnowledgeCursor | None, int], KnowledgePage] | None = None

    async def inspect_source(self, *, source: KnowledgeSourceRef) -> KnowledgeScopeInfo:
        self.inspect_calls.append(source)
        if self.inspect_error is not None:
            raise self.inspect_error
        return KnowledgeScopeInfo(
            source=source,
            capabilities=self.capabilities,
            safe_display_name="Example scope",
        )

    async def read_page(
        self,
        *,
        source: KnowledgeSourceRef,
        cursor: KnowledgeCursor | None,
        limit: int,
    ) -> KnowledgePage:
        self.read_calls.append({"source": source, "cursor": cursor, "limit": limit})
        if self.read_error is not None:
            raise self.read_error
        if self.page_factory is not None:
            return self.page_factory(cursor, limit)
        key = None if cursor is None else cursor.value
        if key in self.pages_by_cursor:
            return self.pages_by_cursor[key]
        if self.default_page is not None:
            return self.default_page
        return make_page(
            proposed_checkpoint=KnowledgeCursor(value="cursor-1", version="v1"),
        )

    async def fetch_content(
        self,
        *,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        self.content_calls.append(item)
        if self.content_error is not None:
            raise self.content_error
        if self.content is not None:
            return self.content
        return make_content(mode=item.content_mode)

    async def fetch_permissions(
        self,
        *,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        self.permissions_calls.append(item)
        if self.permissions_error is not None:
            raise self.permissions_error
        if self.permissions is not None:
            return self.permissions
        return KnowledgePermissions(visibility=KnowledgeVisibility.TENANT)


def shared_order(
    sink: IdempotentRecordingSink,
    state_repo: InMemoryRemoteItemStateRepository,
    checkpoint_repo: InMemoryCheckpointRepository,
) -> list[str]:
    return [*sink.order, *state_repo.order, *checkpoint_repo.order]


@dataclass
class InMemoryCandidateInventoryRepository:
    state_repository: InMemoryRemoteItemStateRepository
    force_incomplete: bool = False

    def list_active_remote_ids(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        binding_configuration_version: int,
        limit: int,
    ) -> tuple[str, ...]:
        if self.force_incomplete:
            from intergrax.runtime.vendor_knowledge.sync_contracts import (
                KnowledgeCandidateInventoryIncomplete,
            )

            raise KnowledgeCandidateInventoryIncomplete("forced incomplete inventory")
        active: list[str] = []
        for state in self.state_repository.states.values():
            if (
                state.tenant_id == tenant_id
                and state.binding_id == binding_id
                and state.binding_configuration_version == binding_configuration_version
                and state.status is KnowledgeRemoteItemStatus.ACTIVE
            ):
                active.append(state.remote_id)
        ordered = tuple(sorted(set(active)))
        if len(ordered) > limit:
            return ordered[:limit]
        return ordered


@dataclass
class InMemoryReconciliationRunRepository:
    runs: dict[tuple[str, str], KnowledgeReconciliationRun] = field(
        default_factory=dict
    )

    def get(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeReconciliationRun | None:
        return self.runs.get((tenant_id, binding_id))

    def create_initial_run(self, run: KnowledgeReconciliationRun) -> None:
        key = (run.tenant_id, run.binding_id)
        if key in self.runs:
            raise KnowledgeReconciliationRunConflict("create conflict")
        if run.phase is not KnowledgeReconciliationRunPhase.COLLECTING:
            raise KnowledgeSyncCorruptState("initial run must be collecting")
        self.runs[key] = run

    def cas_replace(
        self,
        *,
        expected: KnowledgeReconciliationRun,
        replacement: KnowledgeReconciliationRun,
        expected_publication_fence: KnowledgeSyncPublicationFenceV1 | None = None,
    ) -> None:
        _ = expected_publication_fence
        if expected.phase is KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED:
            raise KnowledgeSyncCorruptState(
                "recovery_required run cannot exit through generic cas_replace"
            )
        current = self.runs.get((expected.tenant_id, expected.binding_id))
        if current != expected:
            raise KnowledgeReconciliationRunConflict("cas conflict")
        self.runs[(expected.tenant_id, expected.binding_id)] = replacement

    def cas_supersede_terminal(
        self,
        *,
        expected: KnowledgeReconciliationRun,
        replacement: KnowledgeReconciliationRunCollecting,
        expected_publication_fence: KnowledgeSyncPublicationFenceV1 | None = None,
    ) -> None:
        _ = expected_publication_fence
        current = self.runs.get((expected.tenant_id, expected.binding_id))
        if current != expected:
            raise KnowledgeReconciliationRunConflict("supersede conflict")
        self.runs[(expected.tenant_id, expected.binding_id)] = replacement

    def cas_recovery(
        self,
        *,
        expected: KnowledgeReconciliationRun,
        replacement: KnowledgeReconciliationRun,
        expected_publication_fence: KnowledgeSyncPublicationFenceV1 | None = None,
    ) -> None:
        _ = expected_publication_fence
        current = self.runs.get((expected.tenant_id, expected.binding_id))
        if current != expected:
            raise KnowledgeReconciliationRunConflict("cas conflict")
        self.runs[(expected.tenant_id, expected.binding_id)] = replacement


def durable_reconciliation_coordinator_kwargs(
    *,
    state_repository: InMemoryRemoteItemStateRepository,
    document_store: object | None = None,
) -> dict[str, object]:
    if document_store is not None:
        from intergrax.runtime.vendor_knowledge.sync_document_store import (
            DocumentStoreKnowledgeReconciliationCandidateInventoryRepository,
            DocumentStoreKnowledgeReconciliationRunRepository,
        )

        candidate_inventory_repository = (
            DocumentStoreKnowledgeReconciliationCandidateInventoryRepository(
                document_store  # type: ignore[arg-type]
            )
        )
        reconciliation_run_repository = (
            DocumentStoreKnowledgeReconciliationRunRepository(
                document_store  # type: ignore[arg-type]
            )
        )
    else:
        candidate_inventory_repository = InMemoryCandidateInventoryRepository(
            state_repository=state_repository
        )
        reconciliation_run_repository = InMemoryReconciliationRunRepository()
    return {
        "reconciliation_run_repository": reconciliation_run_repository,
        "candidate_inventory_repository": candidate_inventory_repository,
        "sink_receipt_inspector": RecordingSinkReceiptInspector(),
    }


async def durable_reconcile_once(
    coordinator: object,
    *,
    binding_id: str,
    operation_id: str,
    restart: bool,
    trigger_delivery_id: str | None = None,
) -> object:
    return await coordinator.reconcile_once(  # type: ignore[attr-defined]
        binding_id=binding_id,
        restart=restart,
        operation_id=operation_id,
        trigger_delivery_id=trigger_delivery_id,
    )


async def durable_reconcile_until_complete(
    coordinator: object,
    *,
    binding_id: str,
    operation_id: str,
) -> list[object]:
    results: list[object] = []
    restart = True
    trigger: str | None = None
    while True:
        result = await durable_reconcile_once(
            coordinator,
            binding_id=binding_id,
            operation_id=operation_id,
            restart=restart,
            trigger_delivery_id=trigger,
        )
        results.append(result)
        if not result.has_more:  # type: ignore[attr-defined]
            break
        restart = False
        trigger = result.delivery_id  # type: ignore[attr-defined]
    return results
