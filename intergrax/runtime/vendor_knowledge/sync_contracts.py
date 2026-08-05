# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Synchronization repository and sink ports for the Vendor Knowledge Facade."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeReconciliationRun,
    KnowledgeRemoteItemState,
    KnowledgeRemoteItemStateReceipt,
    KnowledgeSourceLeaseToken,
    KnowledgeSyncBatch,
    KnowledgeSyncCheckpoint,
    KnowledgeSyncSinkReceipt,
)
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    KnowledgeSyncPublicationFenceV1,
    KnowledgeSyncPublicationPermitV1,
)


class KnowledgeSyncCheckpointConflict(Exception):
    """Optimistic checkpoint compare-and-set conflict."""


class KnowledgeSyncCorruptState(Exception):
    """Durable synchronization state is corrupt or inconsistent."""


class KnowledgeReconciliationRunConflict(Exception):
    """Optimistic reconciliation-run compare-and-set conflict."""


class KnowledgeCandidateInventoryIncomplete(Exception):
    """Active candidate inventory cannot be proven complete for reconciliation."""


@runtime_checkable
class KnowledgeReconciliationCandidateInventoryRepository(Protocol):
    """Provider-neutral active candidate inventory port."""

    def list_active_remote_ids(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        binding_configuration_version: int,
        limit: int,
    ) -> tuple[str, ...]:
        """Return unique UTF-8 ascending active remote IDs up to ``limit``."""
        ...


@runtime_checkable
class KnowledgeSourceLeaseRepository(Protocol):
    """Source-level lease port independent of later queue task claims."""

    def acquire(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        owner_id: str,
        ttl_seconds: int,
    ) -> KnowledgeSourceLeaseToken | None:
        """Return a lease token, or ``None`` when another owner holds the lease."""
        ...

    def release(
        self,
        *,
        lease: KnowledgeSourceLeaseToken,
    ) -> None: ...

    def is_owned(self, *, lease: KnowledgeSourceLeaseToken) -> bool:
        """Return true only while this exact, unexpired lease is authoritative."""
        ...


@runtime_checkable
class KnowledgeSyncCheckpointRepository(Protocol):
    """Durable checkpoint port with optimistic compare-and-set commits."""

    def get(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSyncCheckpoint | None: ...

    def commit(
        self,
        checkpoint: KnowledgeSyncCheckpoint,
        *,
        expected_previous: KnowledgeSyncCheckpoint | None,
        expected_publication_fence: KnowledgeSyncPublicationFenceV1 | None = None,
        publication_permit: KnowledgeSyncPublicationPermitV1 | None = None,
    ) -> None: ...


@runtime_checkable
class KnowledgeRemoteItemStateRepository(Protocol):
    """Page-level remote item state port (no checkpoint or content storage)."""

    def get(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        remote_id: str,
    ) -> KnowledgeRemoteItemState | None: ...

    def apply_batch(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        states: tuple[KnowledgeRemoteItemState, ...],
        prepared_state_mutations_fingerprint: str | None = None,
        expected_publication_fence: KnowledgeSyncPublicationFenceV1 | None = None,
        publication_permit: KnowledgeSyncPublicationPermitV1 | None = None,
    ) -> None: ...

    def inspect_delivery_receipt(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        prepared_state_mutations_fingerprint: str,
    ) -> KnowledgeRemoteItemStateReceipt: ...


@runtime_checkable
class KnowledgeSyncSinkReceiptInspector(Protocol):
    """Read-only sink delivery receipt inspection port."""

    def inspect_receipt(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        prepared_batch_payload_fingerprint: str,
    ) -> KnowledgeSyncSinkReceipt: ...


@runtime_checkable
class KnowledgeReconciliationRunRepository(Protocol):
    """Durable reconciliation-run port with optimistic compare-and-set semantics."""

    def get(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeReconciliationRun | None: ...

    def create_initial_run(self, run: KnowledgeReconciliationRun) -> None: ...

    def cas_replace(
        self,
        *,
        expected: KnowledgeReconciliationRun,
        replacement: KnowledgeReconciliationRun,
        expected_publication_fence: KnowledgeSyncPublicationFenceV1 | None = None,
        publication_permit: KnowledgeSyncPublicationPermitV1 | None = None,
    ) -> None: ...

    def cas_supersede_terminal(
        self,
        *,
        expected: KnowledgeReconciliationRun,
        replacement: KnowledgeReconciliationRun,
        expected_publication_fence: KnowledgeSyncPublicationFenceV1 | None = None,
        publication_permit: KnowledgeSyncPublicationPermitV1 | None = None,
    ) -> None: ...

    def cas_recovery(
        self,
        *,
        expected: KnowledgeReconciliationRun,
        replacement: KnowledgeReconciliationRun,
        expected_publication_fence: KnowledgeSyncPublicationFenceV1 | None = None,
        publication_permit: KnowledgeSyncPublicationPermitV1 | None = None,
    ) -> None: ...


@runtime_checkable
class KnowledgeSyncSink(Protocol):
    """Idempotent durable sink for one synchronized page batch."""

    async def apply_batch(
        self,
        *,
        batch: KnowledgeSyncBatch,
    ) -> None: ...
