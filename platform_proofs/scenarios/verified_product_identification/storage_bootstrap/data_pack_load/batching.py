"""Deterministic batch planning for Data Pack storage bootstrap."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapPlan,
    RelationalTargetId,
    VectorTargetId,
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
    records: Sequence[PairedDataPackRecord],
    *,
    batch_size: int,
) -> Iterator[tuple[int, tuple[PairedDataPackRecord, ...]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    ordered = tuple(sorted(records, key=lambda pair: pair.relational.global_row_index))
    batch_number = 0
    for start in range(0, len(ordered), batch_size):
        yield batch_number, ordered[start : start + batch_size]
        batch_number += 1
