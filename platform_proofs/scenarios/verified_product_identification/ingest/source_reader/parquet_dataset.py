"""Streaming parquet dataset reader — bounded memory."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapDataError,
)


@dataclass(frozen=True, slots=True)
class DatasetRow:
    global_row_index: int
    record_json: str


def iter_dataset_rows(
    dataset_path: Path,
    *,
    batch_size: int,
    start_row_index: int = 0,
    start_batch_ordinal: int = 0,
    max_records: int | None = None,
) -> Iterator[tuple[int, tuple[DatasetRow, ...]]]:
    """
    Yield (batch_ordinal, rows) using deterministic global row indices.

    ``batch_ordinal`` is stable for resume: ordinal * batch_size approximates row offset
    only when ``start_row_index`` aligns to batch boundaries; resume uses ``start_row_index``.
    """
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(dataset_path)
    global_index = 0
    batch_ordinal = start_batch_ordinal
    rows_emitted = 0
    buffer: list[DatasetRow] = []

    for batch in parquet_file.iter_batches(columns=["record_json"], batch_size=batch_size):
        column = batch.column(0)
        for index in range(batch.num_rows):
            if global_index < start_row_index:
                global_index += 1
                continue
            if max_records is not None and rows_emitted >= max_records:
                return
            value = column[index].as_py()
            if not isinstance(value, str):
                raise VpiBootstrapDataError("record_json column must contain UTF-8 strings")
            buffer.append(DatasetRow(global_row_index=global_index, record_json=value))
            global_index += 1
            rows_emitted += 1
            if len(buffer) >= batch_size:
                yield batch_ordinal, tuple(buffer)
                batch_ordinal += 1
                buffer = []
    if buffer:
        yield batch_ordinal, tuple(buffer)


def count_rows_to_ingest(
    *,
    start_row_index: int,
    max_records: int | None,
    dataset_record_count: int,
) -> int:
    remaining = max(dataset_record_count - start_row_index, 0)
    if max_records is None:
        return remaining
    return min(max_records, remaining)
