# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral same-partition atomic batch capability for DocumentStore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)

_MAX_ON_CREATED_OPS = 4


@dataclass(frozen=True, slots=True)
class PartitionPutIfAbsentOnCreated:
    """Insert when missing; executed only after the batch primary insert succeeds."""

    document: DocumentRecord


@dataclass(frozen=True, slots=True)
class PartitionReplaceIfMatchOnCreated:
    """Replace when expected matches; executed only after the batch primary insert succeeds."""

    expected: DocumentRecord
    replacement: DocumentRecord


PartitionOnCreatedOp = PartitionPutIfAbsentOnCreated | PartitionReplaceIfMatchOnCreated


@dataclass(frozen=True, slots=True)
class PartitionAtomicBatch:
    """
    Bounded atomic write batch scoped to one partition.

    Semantics:
    1. ``primary_put_if_absent`` is attempted first.
    2. When it creates a new row, every ``on_created_op`` runs in order.
    3. When it finds an existing row, ``on_created_op`` entries are skipped.
    4. Either all durable effects commit or none commit.
    """

    partition_key: str
    primary_put_if_absent: DocumentRecord
    on_created_ops: tuple[PartitionOnCreatedOp, ...] = ()


@dataclass(frozen=True, slots=True)
class PartitionAtomicBatchResult:
    """Typed outcome for a partition atomic batch."""

    primary_created: bool


def validate_partition_atomic_batch(batch: PartitionAtomicBatch) -> PartitionAtomicBatch:
    if not isinstance(batch.partition_key, str) or not batch.partition_key:
        raise ValueError("partition_atomic_batch_partition_key_invalid")
    primary = batch.primary_put_if_absent
    if primary.partition_key != batch.partition_key:
        raise ValueError("partition_atomic_batch_primary_partition_mismatch")
    if len(batch.on_created_ops) > _MAX_ON_CREATED_OPS:
        raise ValueError("partition_atomic_batch_on_created_ops_exceeded")
    for op in batch.on_created_ops:
        if isinstance(op, PartitionPutIfAbsentOnCreated):
            if op.document.partition_key != batch.partition_key:
                raise ValueError("partition_atomic_batch_on_created_partition_mismatch")
        elif isinstance(op, PartitionReplaceIfMatchOnCreated):
            if op.expected.partition_key != batch.partition_key:
                raise ValueError("partition_atomic_batch_on_created_partition_mismatch")
            if op.replacement.partition_key != batch.partition_key:
                raise ValueError("partition_atomic_batch_on_created_partition_mismatch")
            if op.expected.row_key != op.replacement.row_key:
                raise ValueError("partition_atomic_batch_on_created_row_key_mismatch")
        else:
            raise TypeError("partition_atomic_batch_on_created_op_invalid")
    return batch


@runtime_checkable
class PartitionAtomicDocumentStore(ConditionalDocumentStore, Protocol):
    """
    Optional same-partition atomic batch capability.

    Implementations must guarantee that either every durable effect in the
    batch commits together or none commits. Normal primary-key conflicts
    return ``primary_created=False`` without applying ``on_created_ops``.
    """

    def execute_partition_atomic_batch(
        self,
        batch: PartitionAtomicBatch,
    ) -> PartitionAtomicBatchResult:
        """Execute one bounded atomic batch within a single partition."""
        ...
