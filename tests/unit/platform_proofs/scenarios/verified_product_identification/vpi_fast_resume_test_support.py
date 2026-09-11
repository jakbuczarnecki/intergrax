"""Test helpers for fast VPI data pack resume boundary tests."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.shard_plan import (
    plan_data_pack_shards,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackBuildState,
    DataPackShardBuildState,
    DataPackShardStatus,
    VPI_DATA_PACK_BUILD_STATE_VERSION,
)


def build_state_with_ready_prefix(
    *,
    record_count: int,
    shard_size: int,
    ready_prefix: int,
    interrupted_status: DataPackShardStatus | None = None,
    content_identity: str = "test-content-identity",
) -> DataPackBuildState:
    plan = plan_data_pack_shards(record_count=record_count, shard_size=shard_size)
    shards: list[DataPackShardBuildState] = []
    for entry in plan:
        if entry.ordinal <= ready_prefix:
            status = DataPackShardStatus.READY
            rel = f"relational/part-{entry.ordinal:06d}.parquet"
            emb = f"embeddings/part-{entry.ordinal:06d}.parquet"
            shards.append(
                DataPackShardBuildState(
                    ordinal=entry.ordinal,
                    start_row_index=entry.start_row_index,
                    end_row_index_exclusive=entry.end_row_index_exclusive,
                    expected_record_count=entry.expected_record_count,
                    status=status,
                    relational_relative_path=rel,
                    embedding_relative_path=emb,
                    attempt=1,
                    relational_sha256="a" * 64,
                    embedding_sha256="b" * 64,
                    relational_source_ref_set_sha256="c" * 64,
                    embedding_source_ref_set_sha256="c" * 64,
                )
            )
        elif interrupted_status is not None and entry.ordinal == ready_prefix + 1:
            shards.append(
                DataPackShardBuildState(
                    ordinal=entry.ordinal,
                    start_row_index=entry.start_row_index,
                    end_row_index_exclusive=entry.end_row_index_exclusive,
                    expected_record_count=entry.expected_record_count,
                    status=interrupted_status,
                    relational_relative_path=None,
                    embedding_relative_path=None,
                    attempt=1,
                )
            )
        else:
            shards.append(
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
            )
    return DataPackBuildState(
        state_version=VPI_DATA_PACK_BUILD_STATE_VERSION,
        build_id="fast-resume-test",
        content_identity=content_identity,
        expected_record_count=record_count,
        shard_size=shard_size,
        shard_count=len(plan),
        catalog_id="wdc-v2-selected",
        started_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        completed_shards=ready_prefix,
        shards=tuple(shards),
    )
