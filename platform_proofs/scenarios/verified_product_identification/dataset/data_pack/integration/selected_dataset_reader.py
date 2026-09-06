"""Efficient deterministic row-range access for canonical selected dataset."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.integration.parquet_row_group_index import (
    ParquetRowGroupIndex,
    ParquetRowGroupSpan,
    build_parquet_row_group_index,
    spans_for_row_range,
)


@dataclass
class RowGroupAccessRecorder:
    """Test seam recording which Parquet row groups were read."""

    accessed_row_groups: list[int] = field(default_factory=list)

    def record(self, row_group_index: int) -> None:
        self.accessed_row_groups.append(row_group_index)


def _parse_record_json(
    record_json: object,
    *,
    global_row_index: int,
) -> SelectedDatasetRow:
    if not isinstance(record_json, str):
        raise VpiDataPackBuildError("record_json column must contain UTF-8 strings")
    source_offer = parse_wdc_source_offer_json(record_json)
    return SelectedDatasetRow(
        global_row_index=global_row_index,
        record_json=record_json,
        offer_id=source_offer.offer_id,
    )


def _rows_from_row_group_slice(
    table,
    *,
    span: ParquetRowGroupSpan,
    slice_start_row_index: int,
    slice_end_row_index_exclusive: int,
) -> list[SelectedDatasetRow]:
    local_start = slice_start_row_index - span.start_row_index
    local_end = slice_end_row_index_exclusive - span.start_row_index
    column = table.column("record_json").slice(local_start, local_end - local_start)
    rows: list[SelectedDatasetRow] = []
    for offset in range(column.length()):
        global_row_index = slice_start_row_index + offset
        rows.append(
            _parse_record_json(
                column[offset].as_py(),
                global_row_index=global_row_index,
            )
        )
    return rows


def read_selected_dataset_range(
    parquet_file: pq.ParquetFile,
    index: ParquetRowGroupIndex,
    *,
    start_row_index: int,
    end_row_index_exclusive: int,
    row_group_access: RowGroupAccessRecorder | None = None,
) -> tuple[SelectedDatasetRow, ...]:
    """Read ``[start_row_index, end_row_index_exclusive)`` using row-group metadata only."""
    if start_row_index < 0:
        raise ValueError("start_row_index must be >= 0")
    if end_row_index_exclusive <= start_row_index:
        raise ValueError("end_row_index_exclusive must be > start_row_index")

    target_count = end_row_index_exclusive - start_row_index
    intersecting_spans = spans_for_row_range(
        index,
        start_row_index=start_row_index,
        end_row_index_exclusive=end_row_index_exclusive,
    )
    rows: list[SelectedDatasetRow] = []
    for span in intersecting_spans:
        slice_start = max(start_row_index, span.start_row_index)
        slice_end = min(end_row_index_exclusive, span.end_row_index_exclusive)
        if row_group_access is not None:
            row_group_access.record(span.row_group_index)
        table = parquet_file.read_row_group(span.row_group_index, columns=["record_json"])
        rows.extend(
            _rows_from_row_group_slice(
                table,
                span=span,
                slice_start_row_index=slice_start,
                slice_end_row_index_exclusive=slice_end,
            )
        )
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

    def __init__(
        self,
        dataset_path: Path,
        *,
        batch_size: int = 4096,
        row_group_access: RowGroupAccessRecorder | None = None,
    ) -> None:
        self._dataset_path = dataset_path
        self._batch_size = batch_size
        self._row_group_access = row_group_access
        self._parquet_file = pq.ParquetFile(dataset_path)
        self._index = build_parquet_row_group_index(self._parquet_file)

    @property
    def row_group_index(self) -> ParquetRowGroupIndex:
        return self._index

    def read_range(
        self,
        start_row_index: int,
        end_row_index_exclusive: int,
    ) -> Iterable[SelectedDatasetRow]:
        return read_selected_dataset_range(
            self._parquet_file,
            self._index,
            start_row_index=start_row_index,
            end_row_index_exclusive=end_row_index_exclusive,
            row_group_access=self._row_group_access,
        )
