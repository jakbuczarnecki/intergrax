"""Unit tests for typed VPI data pack builder state."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.build_state_machine import (
    reset_shard_to_pending,
    transition_shard,
    validate_shard_status_transition,
)
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
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildStateError,
)

pytestmark = pytest.mark.unit


def _pending_shard(ordinal: int = 1) -> DataPackShardBuildState:
    return DataPackShardBuildState(
        ordinal=ordinal,
        start_row_index=0,
        end_row_index_exclusive=25,
        expected_record_count=25,
        status=DataPackShardStatus.PENDING,
        relational_relative_path=None,
        embedding_relative_path=None,
        attempt=0,
    )


def _new_build_state() -> DataPackBuildState:
    plan = plan_data_pack_shards(record_count=120, shard_size=25)
    shards = tuple(
        DataPackShardBuildState(
            ordinal=entry.ordinal,
            start_row_index=entry.start_row_index,
            end_row_index_exclusive=entry.end_row_index_exclusive,
            expected_record_count=entry.expected_record_count,
            status=DataPackShardStatus.PENDING,
            relational_relative_path=None,
            embedding_relative_path=None,
            attempt=0,
        )
        for entry in plan
    )
    return DataPackBuildState(
        state_version=VPI_DATA_PACK_BUILD_STATE_VERSION,
        build_id="test-build",
        content_identity="abc123",
        expected_record_count=120,
        shard_size=25,
        shard_count=len(shards),
        catalog_id="wdc-v2-selected",
        started_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        completed_shards=0,
        shards=shards,
    )


def test_new_build_state_roundtrip() -> None:
    state = _new_build_state()
    payload = build_state_to_json_dict(state)
    restored = build_state_from_json_dict(payload)
    assert restored == state


def test_unknown_field_fails() -> None:
    payload = build_state_to_json_dict(_new_build_state())
    payload["unexpected"] = "value"
    with pytest.raises(VpiDataPackBuildStateError, match="unexpected fields"):
        build_state_from_json_dict(payload)


def test_wrong_type_fails() -> None:
    payload = build_state_to_json_dict(_new_build_state())
    payload["shard_size"] = "25"
    with pytest.raises(VpiDataPackBuildStateError, match="shard_size must be an integer"):
        build_state_from_json_dict(payload)


def test_bool_not_accepted_as_int() -> None:
    payload = build_state_to_json_dict(_new_build_state())
    payload["completed_shards"] = True
    with pytest.raises(VpiDataPackBuildStateError, match="completed_shards must be an integer"):
        build_state_from_json_dict(payload)


def test_invalid_transition_fails() -> None:
    shard = _pending_shard()
    with pytest.raises(VpiDataPackBuildStateError, match="invalid shard status transition"):
        validate_shard_status_transition(shard.status, DataPackShardStatus.READY)


def test_ready_to_non_ready_forbidden() -> None:
    ready = DataPackShardBuildState(
        ordinal=1,
        start_row_index=0,
        end_row_index_exclusive=25,
        expected_record_count=25,
        status=DataPackShardStatus.READY,
        relational_relative_path="relational/part-000001.parquet",
        embedding_relative_path="embeddings/part-000001.parquet",
        attempt=1,
        relational_sha256="a" * 64,
        embedding_sha256="b" * 64,
        relational_source_ref_set_sha256="c" * 64,
        embedding_source_ref_set_sha256="c" * 64,
    )
    with pytest.raises(VpiDataPackBuildStateError, match="READY shard cannot transition"):
        reset_shard_to_pending(ready)


def test_valid_transition_chain() -> None:
    shard = _pending_shard()
    shard = transition_shard(shard, DataPackShardStatus.DERIVING)
    shard = transition_shard(shard, DataPackShardStatus.EMBEDDING)
    shard = transition_shard(shard, DataPackShardStatus.WRITING)
    shard = transition_shard(shard, DataPackShardStatus.VALIDATING)
    shard = transition_shard(
        shard,
        DataPackShardStatus.READY,
        relational_relative_path="relational/part-000001.parquet",
        embedding_relative_path="embeddings/part-000001.parquet",
        relational_sha256="a" * 64,
        embedding_sha256="b" * 64,
        relational_source_ref_set_sha256="c" * 64,
        embedding_source_ref_set_sha256="c" * 64,
    )
    assert shard.status is DataPackShardStatus.READY
