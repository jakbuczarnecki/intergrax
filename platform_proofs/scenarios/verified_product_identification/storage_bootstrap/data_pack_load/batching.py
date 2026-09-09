"""Deterministic batch planning for Data Pack storage bootstrap."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapPlan,
    RelationalTargetId,
    VectorTargetId,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    StorageBootstrapIdentityError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.ports import (
    PairedDataPackRecord,
)


def compute_bootstrap_plan(
    *,
    record_count: int,
    batch_size: int,
    relational_target: RelationalTargetId,
    vector_target: VectorTargetId,
) -> BootstrapPlan:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if record_count < 0:
        raise ValueError("record_count must be >= 0")
    if record_count == 0:
        return BootstrapPlan(
            record_count=0,
            batch_size=batch_size,
            batch_count=0,
            final_batch_size=0,
            relational_target=relational_target,
            vector_target=vector_target,
        )
    batch_count = (record_count + batch_size - 1) // batch_size
    remainder = record_count % batch_size
    final_batch_size = remainder if remainder != 0 else batch_size
    return BootstrapPlan(
        record_count=record_count,
        batch_size=batch_size,
        batch_count=batch_count,
        final_batch_size=final_batch_size,
        relational_target=relational_target,
        vector_target=vector_target,
    )


def iter_record_batches(
    records: Iterable[PairedDataPackRecord],
    *,
    batch_size: int,
    start_batch_number: int = 0,
) -> Iterator[tuple[int, tuple[PairedDataPackRecord, ...]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if start_batch_number < 0:
        raise ValueError("start_batch_number must be >= 0")

    buffer: list[PairedDataPackRecord] = []
    previous_global_row_index: int | None = None
    batch_number = start_batch_number

    for record in records:
        current_index = record.relational.global_row_index
        if previous_global_row_index is None:
            if current_index < 0:
                raise StorageBootstrapIdentityError(
                    f"invalid global_row_index {current_index}"
                )
        elif current_index <= previous_global_row_index:
            raise StorageBootstrapIdentityError(
                f"non-ascending global_row_index: previous={previous_global_row_index}, "
                f"current={current_index}"
            )
        elif current_index != previous_global_row_index + 1:
            raise StorageBootstrapIdentityError(
                f"global_row_index gap: previous={previous_global_row_index}, "
                f"current={current_index}"
            )
        previous_global_row_index = current_index
        buffer.append(record)
        if len(buffer) == batch_size:
            yield batch_number, tuple(buffer)
            batch_number += 1
            buffer = []

    if buffer:
        yield batch_number, tuple(buffer)
