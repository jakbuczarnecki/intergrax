"""Row-group-aware selected dataset reader tests."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.integration.parquet_row_group_index import (
    build_parquet_row_group_index,
    spans_for_row_range,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.integration.selected_dataset_reader import (
    RowGroupAccessRecorder,
    SelectedDatasetShardReaderPort,
)

pytestmark = pytest.mark.unit


def _write_multi_row_group_dataset(path: Path, row_count: int, *, row_group_size: int) -> None:
    records = [
        json.dumps({"id": f"offer-{index}", "title": f"Item {index}"})
        for index in range(row_count)
    ]
    table = pa.table({"record_json": records})
    pq.write_table(table, path, row_group_size=row_group_size)


@pytest.fixture
def multi_group_dataset(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "selected.parquet"
    _write_multi_row_group_dataset(dataset_path, row_count=100, row_group_size=10)
    return dataset_path


def test_row_group_index_metadata(multi_group_dataset: Path) -> None:
    parquet_file = pq.ParquetFile(multi_group_dataset)
    index = build_parquet_row_group_index(parquet_file)
    assert index.total_rows == 100
    assert len(index.spans) == 10
    assert index.spans[0].start_row_index == 0
    assert index.spans[-1].end_row_index_exclusive == 100


def test_range_entirely_in_one_row_group(multi_group_dataset: Path) -> None:
    reader = SelectedDatasetShardReaderPort(multi_group_dataset)
    rows = tuple(reader.read_range(12, 18))
    assert len(rows) == 6
    assert rows[0].global_row_index == 12
    assert rows[-1].global_row_index == 17


def test_range_across_two_row_groups(multi_group_dataset: Path) -> None:
    reader = SelectedDatasetShardReaderPort(multi_group_dataset)
    rows = tuple(reader.read_range(8, 12))
    assert len(rows) == 4
    assert [row.global_row_index for row in rows] == [8, 9, 10, 11]


def test_range_starting_mid_group(multi_group_dataset: Path) -> None:
    reader = SelectedDatasetShardReaderPort(multi_group_dataset)
    rows = tuple(reader.read_range(25, 28))
    assert [row.global_row_index for row in rows] == [25, 26, 27]


def test_range_ending_mid_group(multi_group_dataset: Path) -> None:
    reader = SelectedDatasetShardReaderPort(multi_group_dataset)
    rows = tuple(reader.read_range(33, 36))
    assert [row.global_row_index for row in rows] == [33, 34, 35]


def test_range_across_many_groups(multi_group_dataset: Path) -> None:
    reader = SelectedDatasetShardReaderPort(multi_group_dataset)
    rows = tuple(reader.read_range(5, 55))
    assert len(rows) == 50
    assert rows[0].global_row_index == 5
    assert rows[-1].global_row_index == 54


def test_first_row(multi_group_dataset: Path) -> None:
    reader = SelectedDatasetShardReaderPort(multi_group_dataset)
    rows = tuple(reader.read_range(0, 1))
    assert rows[0].global_row_index == 0


def test_last_row(multi_group_dataset: Path) -> None:
    reader = SelectedDatasetShardReaderPort(multi_group_dataset)
    rows = tuple(reader.read_range(99, 100))
    assert rows[0].global_row_index == 99


def test_full_range(multi_group_dataset: Path) -> None:
    reader = SelectedDatasetShardReaderPort(multi_group_dataset)
    rows = tuple(reader.read_range(0, 100))
    assert len(rows) == 100


def test_invalid_range_beyond_dataset(multi_group_dataset: Path) -> None:
    parquet_file = pq.ParquetFile(multi_group_dataset)
    index = build_parquet_row_group_index(parquet_file)
    with pytest.raises(VpiDataPackBuildError, match="exceeds dataset rows"):
        spans_for_row_range(index, start_row_index=90, end_row_index_exclusive=101)


def test_invalid_range_start_not_before_end(multi_group_dataset: Path) -> None:
    reader = SelectedDatasetShardReaderPort(multi_group_dataset)
    with pytest.raises(ValueError, match="end_row_index_exclusive must be > start_row_index"):
        tuple(reader.read_range(5, 5))


def test_late_range_does_not_read_prefix_row_groups(multi_group_dataset: Path) -> None:
    recorder = RowGroupAccessRecorder()
    reader = SelectedDatasetShardReaderPort(multi_group_dataset, row_group_access=recorder)
    rows = tuple(reader.read_range(80, 100))
    assert len(rows) == 20
    assert recorder.accessed_row_groups == [8, 9]
    assert all(group_index >= 8 for group_index in recorder.accessed_row_groups)
