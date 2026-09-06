"""Unit tests for deterministic VPI data pack shard planning."""

from __future__ import annotations

import math

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.shard_plan import (
    plan_data_pack_shards,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DEFAULT_PRODUCTION_SHARD_SIZE,
)

pytestmark = pytest.mark.unit

FULL_DATASET_RECORD_COUNT = 3_770_377


def test_default_production_shard_size_is_5000() -> None:
    assert DEFAULT_PRODUCTION_SHARD_SIZE == 5_000


def test_zero_records_invalid() -> None:
    with pytest.raises(ValueError, match="record_count"):
        plan_data_pack_shards(record_count=0, shard_size=25)


def test_one_record() -> None:
    plan = plan_data_pack_shards(record_count=1, shard_size=25)
    assert len(plan) == 1
    assert plan[0].ordinal == 1
    assert plan[0].start_row_index == 0
    assert plan[0].end_row_index_exclusive == 1
    assert plan[0].expected_record_count == 1


def test_exact_multiple() -> None:
    plan = plan_data_pack_shards(record_count=100, shard_size=25)
    assert len(plan) == 4
    assert [entry.expected_record_count for entry in plan] == [25, 25, 25, 25]


def test_partial_last_shard() -> None:
    plan = plan_data_pack_shards(record_count=120, shard_size=25)
    assert len(plan) == 5
    assert [entry.expected_record_count for entry in plan] == [25, 25, 25, 25, 20]


def test_ranges_have_no_gaps_or_overlap() -> None:
    record_count = 3770377
    shard_size = 25_000
    plan = plan_data_pack_shards(record_count=record_count, shard_size=shard_size)
    assert len(plan) == math.ceil(record_count / shard_size)
    expected_start = 0
    for entry in plan:
        assert entry.ordinal >= 1
        assert entry.start_row_index == expected_start
        expected_start = entry.end_row_index_exclusive
    assert expected_start == record_count


def test_full_production_dataset_plan_shard_size_5000() -> None:
    record_count = FULL_DATASET_RECORD_COUNT
    shard_size = DEFAULT_PRODUCTION_SHARD_SIZE
    plan = plan_data_pack_shards(record_count=record_count, shard_size=shard_size)

    assert len(plan) == 755
    assert plan[0].ordinal == 1
    assert plan[0].start_row_index == 0
    assert plan[0].end_row_index_exclusive == 5_000
    assert plan[0].expected_record_count == 5_000

    assert plan[1].ordinal == 2
    assert plan[1].start_row_index == 5_000
    assert plan[1].end_row_index_exclusive == 10_000
    assert plan[1].expected_record_count == 5_000

    final = plan[-1]
    assert final.ordinal == 755
    assert final.start_row_index == 3_770_000
    assert final.end_row_index_exclusive == record_count
    assert final.expected_record_count == 377

    expected_start = 0
    for entry in plan:
        assert entry.start_row_index == expected_start
        expected_start = entry.end_row_index_exclusive
    assert expected_start == record_count
