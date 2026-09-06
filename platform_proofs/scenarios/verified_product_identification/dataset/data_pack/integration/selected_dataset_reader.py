"""Efficient deterministic row-range access for canonical selected dataset."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pyarrow.parquet as pq

from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.sample_selection import (
    SelectedDatasetRow,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
)


def read_selected_dataset_range(
    dataset_path: Path,
    *,
    start_row_index: int,
    end_row_index_exclusive: int,
    batch_size: int = 4096,
) -> tuple[SelectedDatasetRow, ...]:
    """Read ``[start_row_index, end_row_index_exclusive)`` without loading the full dataset."""
    if start_row_index < 0:
        raise ValueError("start_row_index must be >= 0")
    if end_row_index_exclusive <= start_row_index:
        raise ValueError("end_row_index_exclusive must be > start_row_index")

    parquet_file = pq.ParquetFile(dataset_path)
    global_row_index = 0
    rows: list[SelectedDatasetRow] = []
    target_count = end_row_index_exclusive - start_row_index

    for batch in parquet_file.iter_batches(columns=["record_json"], batch_size=batch_size):
        column = batch.column(0)
        for index in range(batch.num_rows):
            if global_row_index < start_row_index:
                global_row_index += 1
                continue
            if global_row_index >= end_row_index_exclusive:
                break
            record_json = column[index].as_py()
            if not isinstance(record_json, str):
                raise VpiDataPackBuildError("record_json column must contain UTF-8 strings")
            source_offer = parse_wdc_source_offer_json(record_json)
            rows.append(
                SelectedDatasetRow(
                    global_row_index=global_row_index,
                    record_json=record_json,
                    offer_id=source_offer.offer_id,
                )
            )
            global_row_index += 1
            if len(rows) >= target_count:
                break
        if len(rows) >= target_count:
            break

    if len(rows) != target_count:
        raise VpiDataPackBuildError(
            f"dataset range [{start_row_index}, {end_row_index_exclusive}) yielded "
            f"{len(rows)} rows; expected {target_count}"
        )
    return tuple(rows)


class SelectedDatasetShardReaderPort:
    """Provider-neutral row-range reader for resumable builder."""

    def __init__(self, dataset_path: Path, *, batch_size: int = 4096) -> None:
        self._dataset_path = dataset_path
        self._batch_size = batch_size

    def read_range(
        self,
        start_row_index: int,
        end_row_index_exclusive: int,
    ) -> Iterable[SelectedDatasetRow]:
        return read_selected_dataset_range(
            self._dataset_path,
            start_row_index=start_row_index,
            end_row_index_exclusive=end_row_index_exclusive,
            batch_size=self._batch_size,
        )
