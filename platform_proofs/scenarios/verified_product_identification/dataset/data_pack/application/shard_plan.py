"""Deterministic shard planning for resumable data pack builds."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DataPackShardPlanEntry:
    ordinal: int
    start_row_index: int
    end_row_index_exclusive: int
    expected_record_count: int

    def __post_init__(self) -> None:
        if self.ordinal < 1:
            raise ValueError("ordinal must be >= 1")
        if self.start_row_index < 0:
            raise ValueError("start_row_index must be >= 0")
        if self.end_row_index_exclusive <= self.start_row_index:
            raise ValueError("end_row_index_exclusive must be > start_row_index")
        if self.expected_record_count <= 0:
            raise ValueError("expected_record_count must be > 0")
        if self.expected_record_count != self.end_row_index_exclusive - self.start_row_index:
            raise ValueError("expected_record_count must match row range width")


def plan_data_pack_shards(
    *,
    record_count: int,
    shard_size: int,
) -> tuple[DataPackShardPlanEntry, ...]:
    if record_count <= 0:
        msg = "record_count must be > 0"
        raise ValueError(msg)
    if shard_size <= 0:
        msg = "shard_size must be > 0"
        raise ValueError(msg)

    shard_count = math.ceil(record_count / shard_size)
    entries: list[DataPackShardPlanEntry] = []
    start_row_index = 0
    for ordinal in range(1, shard_count + 1):
        remaining = record_count - start_row_index
        width = min(shard_size, remaining)
        end_row_index_exclusive = start_row_index + width
        entries.append(
            DataPackShardPlanEntry(
                ordinal=ordinal,
                start_row_index=start_row_index,
                end_row_index_exclusive=end_row_index_exclusive,
                expected_record_count=width,
            )
        )
        start_row_index = end_row_index_exclusive
    return tuple(entries)
