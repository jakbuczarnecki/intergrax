# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Synchronization repository and sink ports for the Vendor Knowledge Facade."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeRemoteItemState,
    KnowledgeSourceLeaseToken,
    KnowledgeSyncBatch,
    KnowledgeSyncCheckpoint,
)


class KnowledgeSyncCheckpointConflict(Exception):
    """Optimistic checkpoint compare-and-set conflict."""


class KnowledgeSyncCorruptState(Exception):
    """Durable synchronization state is corrupt or inconsistent."""


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
    ) -> None:
        ...


@runtime_checkable
class KnowledgeSyncCheckpointRepository(Protocol):
    """Durable checkpoint port with optimistic compare-and-set commits."""

    def get(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSyncCheckpoint | None:
        ...

    def commit(
        self,
        checkpoint: KnowledgeSyncCheckpoint,
        *,
        expected_previous: KnowledgeSyncCheckpoint | None,
    ) -> None:
        ...


@runtime_checkable
class KnowledgeRemoteItemStateRepository(Protocol):
    """Page-level remote item state port (no checkpoint or content storage)."""

    def get(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        remote_id: str,
    ) -> KnowledgeRemoteItemState | None:
        ...

    def apply_batch(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        states: tuple[KnowledgeRemoteItemState, ...],
    ) -> None:
        ...


@runtime_checkable
class KnowledgeSyncSink(Protocol):
    """Idempotent durable sink for one synchronized page batch."""

    async def apply_batch(
        self,
        *,
        batch: KnowledgeSyncBatch,
    ) -> None:
        ...
