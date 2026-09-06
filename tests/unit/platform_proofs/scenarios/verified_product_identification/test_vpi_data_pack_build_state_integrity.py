"""Structural integrity tests for persisted VPI data pack build state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.shard_plan import (
    plan_data_pack_shards,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackBuildState,
    DataPackShardBuildState,
    DataPackShardStatus,
    VPI_DATA_PACK_BUILD_STATE_VERSION,
    build_state_from_json_dict,
    build_state_to_json_dict,
    read_build_state_file,
    write_build_state_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildStateError,
)

pytestmark = pytest.mark.unit


def _pending_shard(
    *,
    ordinal: int,
    start_row_index: int,
    end_row_index_exclusive: int,
    expected_record_count: int,
) -> DataPackShardBuildState:
    return DataPackShardBuildState(
        ordinal=ordinal,
        start_row_index=start_row_index,
        end_row_index_exclusive=end_row_index_exclusive,
        expected_record_count=expected_record_count,
        status=DataPackShardStatus.PENDING,
        relational_relative_path=None,
        embedding_relative_path=None,
        attempt=0,
    )


def _canonical_build_state(
    *,
    expected_record_count: int = 120,
    shard_size: int = 25,
    completed_shards: int = 0,
) -> DataPackBuildState:
    plan = plan_data_pack_shards(record_count=expected_record_count, shard_size=shard_size)
    shards = tuple(
        _pending_shard(
            ordinal=entry.ordinal,
            start_row_index=entry.start_row_index,
            end_row_index_exclusive=entry.end_row_index_exclusive,
            expected_record_count=entry.expected_record_count,
        )
        for entry in plan
    )
    return DataPackBuildState(
        state_version=VPI_DATA_PACK_BUILD_STATE_VERSION,
        build_id="test-build",
        content_identity="abc123",
        expected_record_count=expected_record_count,
        shard_size=shard_size,
        shard_count=len(shards),
        catalog_id="wdc-v2-selected",
        started_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        completed_shards=completed_shards,
        shards=shards,
    )


def _build_state_payload(**overrides: object) -> dict[str, object]:
    payload = build_state_to_json_dict(_canonical_build_state())
    payload.update(overrides)
    return payload


def _shard_payloads(**shard_overrides: object) -> list[dict[str, object]]:
    payload = _build_state_payload()
    shards = payload["shards"]
    assert isinstance(shards, list)
    first = shards[0]
    assert isinstance(first, dict)
    return [{**first, **shard_overrides}]


def test_valid_canonical_plan_passes() -> None:
    state = _canonical_build_state()
    restored = build_state_from_json_dict(build_state_to_json_dict(state))
    assert restored == state


def test_valid_final_short_shard_passes() -> None:
    state = _canonical_build_state(expected_record_count=120, shard_size=25)
    assert tuple(shard.expected_record_count for shard in state.shards) == (25, 25, 25, 25, 20)
    restored = build_state_from_json_dict(build_state_to_json_dict(state))
    assert restored == state


def test_duplicate_ordinal_fails() -> None:
    shards = (
        _pending_shard(ordinal=1, start_row_index=0, end_row_index_exclusive=25, expected_record_count=25),
        _pending_shard(ordinal=1, start_row_index=25, end_row_index_exclusive=50, expected_record_count=25),
        _pending_shard(ordinal=3, start_row_index=50, end_row_index_exclusive=75, expected_record_count=25),
    )
    with pytest.raises(ValueError, match="duplicate shard ordinal"):
        DataPackBuildState(
            state_version=VPI_DATA_PACK_BUILD_STATE_VERSION,
            build_id="test-build",
            content_identity="abc123",
            expected_record_count=75,
            shard_size=25,
            shard_count=3,
            catalog_id="wdc-v2-selected",
            started_at_utc="2026-01-01T00:00:00+00:00",
            updated_at_utc="2026-01-01T00:00:00+00:00",
            completed_shards=0,
            shards=shards,
        )


def test_non_contiguous_ordinal_gap_fails() -> None:
    shards = (
        _pending_shard(ordinal=1, start_row_index=0, end_row_index_exclusive=25, expected_record_count=25),
        _pending_shard(ordinal=2, start_row_index=25, end_row_index_exclusive=50, expected_record_count=25),
        _pending_shard(ordinal=4, start_row_index=50, end_row_index_exclusive=75, expected_record_count=25),
    )
    with pytest.raises(ValueError, match="shard ordinals must be contiguous"):
        DataPackBuildState(
            state_version=VPI_DATA_PACK_BUILD_STATE_VERSION,
            build_id="test-build",
            content_identity="abc123",
            expected_record_count=75,
            shard_size=25,
            shard_count=3,
            catalog_id="wdc-v2-selected",
            started_at_utc="2026-01-01T00:00:00+00:00",
            updated_at_utc="2026-01-01T00:00:00+00:00",
            completed_shards=0,
            shards=shards,
        )


def test_out_of_order_ordinal_fails() -> None:
    shards = (
        _pending_shard(ordinal=1, start_row_index=0, end_row_index_exclusive=25, expected_record_count=25),
        _pending_shard(ordinal=3, start_row_index=25, end_row_index_exclusive=50, expected_record_count=25),
        _pending_shard(ordinal=2, start_row_index=50, end_row_index_exclusive=75, expected_record_count=25),
    )
    with pytest.raises(ValueError, match="shard ordinals must be contiguous"):
        DataPackBuildState(
            state_version=VPI_DATA_PACK_BUILD_STATE_VERSION,
            build_id="test-build",
            content_identity="abc123",
            expected_record_count=75,
            shard_size=25,
            shard_count=3,
            catalog_id="wdc-v2-selected",
            started_at_utc="2026-01-01T00:00:00+00:00",
            updated_at_utc="2026-01-01T00:00:00+00:00",
            completed_shards=0,
            shards=shards,
        )


def test_first_start_not_zero_fails() -> None:
    shards = (
        _pending_shard(ordinal=1, start_row_index=5, end_row_index_exclusive=30, expected_record_count=25),
        _pending_shard(ordinal=2, start_row_index=30, end_row_index_exclusive=55, expected_record_count=25),
    )
    with pytest.raises(ValueError, match="first shard must start at row index 0"):
        DataPackBuildState(
            state_version=VPI_DATA_PACK_BUILD_STATE_VERSION,
            build_id="test-build",
            content_identity="abc123",
            expected_record_count=50,
            shard_size=25,
            shard_count=2,
            catalog_id="wdc-v2-selected",
            started_at_utc="2026-01-01T00:00:00+00:00",
            updated_at_utc="2026-01-01T00:00:00+00:00",
            completed_shards=0,
            shards=shards,
        )


def test_overlapping_ranges_fail() -> None:
    shards = (
        _pending_shard(ordinal=1, start_row_index=0, end_row_index_exclusive=25, expected_record_count=25),
        _pending_shard(ordinal=2, start_row_index=20, end_row_index_exclusive=45, expected_record_count=25),
    )
    with pytest.raises(ValueError, match="shard range overlap"):
        DataPackBuildState(
            state_version=VPI_DATA_PACK_BUILD_STATE_VERSION,
            build_id="test-build",
            content_identity="abc123",
            expected_record_count=45,
            shard_size=25,
            shard_count=2,
            catalog_id="wdc-v2-selected",
            started_at_utc="2026-01-01T00:00:00+00:00",
            updated_at_utc="2026-01-01T00:00:00+00:00",
            completed_shards=0,
            shards=shards,
        )


def test_range_gap_fails() -> None:
    shards = (
        _pending_shard(ordinal=1, start_row_index=0, end_row_index_exclusive=25, expected_record_count=25),
        _pending_shard(ordinal=2, start_row_index=30, end_row_index_exclusive=55, expected_record_count=25),
    )
    with pytest.raises(ValueError, match="shard range gap"):
        DataPackBuildState(
            state_version=VPI_DATA_PACK_BUILD_STATE_VERSION,
            build_id="test-build",
            content_identity="abc123",
            expected_record_count=55,
            shard_size=25,
            shard_count=2,
            catalog_id="wdc-v2-selected",
            started_at_utc="2026-01-01T00:00:00+00:00",
            updated_at_utc="2026-01-01T00:00:00+00:00",
            completed_shards=0,
            shards=shards,
        )


def test_per_shard_range_length_mismatch_fails() -> None:
    shards = (
        _pending_shard(ordinal=1, start_row_index=0, end_row_index_exclusive=25, expected_record_count=24),
    )
    with pytest.raises(ValueError, match="range length 25 does not match expected_record_count 24"):
        DataPackBuildState(
            state_version=VPI_DATA_PACK_BUILD_STATE_VERSION,
            build_id="test-build",
            content_identity="abc123",
            expected_record_count=25,
            shard_size=25,
            shard_count=1,
            catalog_id="wdc-v2-selected",
            started_at_utc="2026-01-01T00:00:00+00:00",
            updated_at_utc="2026-01-01T00:00:00+00:00",
            completed_shards=0,
            shards=shards,
        )


def test_final_end_before_expected_total_fails() -> None:
    shards = (
        _pending_shard(ordinal=1, start_row_index=0, end_row_index_exclusive=119, expected_record_count=119),
    )
    with pytest.raises(ValueError, match="last shard end 119 does not match build expected_record_count 120"):
        DataPackBuildState(
            state_version=VPI_DATA_PACK_BUILD_STATE_VERSION,
            build_id="test-build",
            content_identity="abc123",
            expected_record_count=120,
            shard_size=119,
            shard_count=1,
            catalog_id="wdc-v2-selected",
            started_at_utc="2026-01-01T00:00:00+00:00",
            updated_at_utc="2026-01-01T00:00:00+00:00",
            completed_shards=0,
            shards=shards,
        )


def test_final_end_after_expected_total_fails() -> None:
    shards = (
        _pending_shard(ordinal=1, start_row_index=0, end_row_index_exclusive=121, expected_record_count=121),
    )
    with pytest.raises(ValueError, match="last shard end 121 does not match build expected_record_count 120"):
        DataPackBuildState(
            state_version=VPI_DATA_PACK_BUILD_STATE_VERSION,
            build_id="test-build",
            content_identity="abc123",
            expected_record_count=120,
            shard_size=121,
            shard_count=1,
            catalog_id="wdc-v2-selected",
            started_at_utc="2026-01-01T00:00:00+00:00",
            updated_at_utc="2026-01-01T00:00:00+00:00",
            completed_shards=0,
            shards=shards,
        )


def test_total_shard_count_sum_mismatch_fails() -> None:
    shards = tuple(_canonical_build_state().shards)
    with pytest.raises(ValueError, match="does not match build expected_record_count 121"):
        DataPackBuildState(
            state_version=VPI_DATA_PACK_BUILD_STATE_VERSION,
            build_id="test-build",
            content_identity="abc123",
            expected_record_count=121,
            shard_size=25,
            shard_count=len(shards),
            catalog_id="wdc-v2-selected",
            started_at_utc="2026-01-01T00:00:00+00:00",
            updated_at_utc="2026-01-01T00:00:00+00:00",
            completed_shards=0,
            shards=shards,
        )


def test_completed_shards_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="completed_shards must equal READY shard count"):
        _canonical_build_state(completed_shards=2)


def test_ready_missing_integrity_field_fails() -> None:
    with pytest.raises(ValueError, match="relational_sha256 is required"):
        DataPackShardBuildState(
            ordinal=1,
            start_row_index=0,
            end_row_index_exclusive=25,
            expected_record_count=25,
            status=DataPackShardStatus.READY,
            relational_relative_path="relational/part-000001.parquet",
            embedding_relative_path="embeddings/part-000001.parquet",
            attempt=1,
            relational_sha256=None,
            embedding_sha256="b" * 64,
            relational_source_ref_set_sha256="c" * 64,
            embedding_source_ref_set_sha256="c" * 64,
        )


def test_unsupported_version_fails_with_typed_error() -> None:
    payload = _build_state_payload(state_version="vpi.data_pack.build_state/99")
    with pytest.raises(VpiDataPackBuildStateError, match="unsupported build state version"):
        build_state_from_json_dict(payload)


def test_duplicate_ordinal_via_json_emits_build_state_error() -> None:
    payload = _build_state_payload(
        shard_count=2,
        expected_record_count=50,
        shards=[
            _shard_payloads()[0],
            {**_shard_payloads()[0], "ordinal": 1, "start_row_index": 25, "end_row_index_exclusive": 50},
        ],
    )
    with pytest.raises(VpiDataPackBuildStateError, match="duplicate shard ordinal") as exc_info:
        build_state_from_json_dict(payload)
    assert not isinstance(exc_info.value, ValueError)


def test_overlap_via_json_emits_build_state_error(tmp_path: Path) -> None:
    payload = _build_state_payload(
        shard_count=2,
        expected_record_count=45,
        shards=[
            _shard_payloads()[0],
            {
                **_shard_payloads()[0],
                "ordinal": 2,
                "start_row_index": 20,
                "end_row_index_exclusive": 45,
                "expected_record_count": 25,
            },
        ],
    )
    with pytest.raises(VpiDataPackBuildStateError, match="shard range overlap"):
        build_state_from_json_dict(payload)

    state_path = tmp_path / "build-state.json"
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VpiDataPackBuildStateError, match="shard range overlap"):
        read_build_state_file(state_path)


def test_semantic_invalid_state_via_read_file_emits_build_state_error(tmp_path: Path) -> None:
    state = _canonical_build_state()
    write_build_state_file(tmp_path / "build-state.json", state)
    payload = build_state_to_json_dict(state)
    payload["completed_shards"] = 2
    path = tmp_path / "build-state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VpiDataPackBuildStateError, match="completed_shards must equal READY shard count"):
        read_build_state_file(path)
