"""Lightweight row-group index for bounded Parquet range reads."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow.parquet as pq

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
)


@dataclass(frozen=True, slots=True)
class ParquetRowGroupSpan:
    row_group_index: int
    start_row_index: int
    end_row_index_exclusive: int
    record_count: int


@dataclass(frozen=True, slots=True)
class ParquetRowGroupIndex:
    spans: tuple[ParquetRowGroupSpan, ...]
    total_rows: int


def build_parquet_row_group_index(parquet_file: pq.ParquetFile) -> ParquetRowGroupIndex:
    metadata = parquet_file.metadata
    if metadata is None:
        raise VpiDataPackBuildError("parquet file metadata is unavailable")
    spans: list[ParquetRowGroupSpan] = []
    start_row_index = 0
    for row_group_index in range(parquet_file.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        record_count = row_group.num_rows
        spans.append(
            ParquetRowGroupSpan(
                row_group_index=row_group_index,
                start_row_index=start_row_index,
                end_row_index_exclusive=start_row_index + record_count,
                record_count=record_count,
            )
        )
        start_row_index += record_count
    return ParquetRowGroupIndex(spans=tuple(spans), total_rows=start_row_index)


def spans_for_row_range(
    index: ParquetRowGroupIndex,
    *,
    start_row_index: int,
    end_row_index_exclusive: int,
) -> tuple[ParquetRowGroupSpan, ...]:
    if start_row_index < 0:
        raise ValueError("start_row_index must be >= 0")
    if end_row_index_exclusive <= start_row_index:
        raise ValueError("end_row_index_exclusive must be > start_row_index")
    if end_row_index_exclusive > index.total_rows:
        raise VpiDataPackBuildError(
            f"requested range end {end_row_index_exclusive} exceeds dataset rows {index.total_rows}"
        )
    return tuple(
        span
        for span in index.spans
        if span.end_row_index_exclusive > start_row_index
        and span.start_row_index < end_row_index_exclusive
    )
