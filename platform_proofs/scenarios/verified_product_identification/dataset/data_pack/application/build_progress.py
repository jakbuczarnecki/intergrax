"""Typed build progress snapshot for operator visibility."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackBuildState,
    DataPackShardStatus,
)


@dataclass(frozen=True, slots=True)
class DataPackBuildProgress:
    ready_shards: int
    total_shards: int
    records_completed: int
    expected_records: int
    remaining_shards: int
    estimated_remaining_seconds: float | None
    average_shard_elapsed_seconds: float | None

    @property
    def percentage(self) -> float:
        if self.expected_records <= 0:
            return 0.0
        return (self.records_completed / self.expected_records) * 100.0


def compute_build_progress(state: DataPackBuildState) -> DataPackBuildProgress:
    records_completed = sum(
        shard.expected_record_count
        for shard in state.shards
        if shard.status is DataPackShardStatus.READY
    )
    elapsed_samples = [
        shard.elapsed_seconds
        for shard in state.shards
        if shard.status is DataPackShardStatus.READY and shard.elapsed_seconds is not None
    ]
    average_shard_elapsed = (
        sum(elapsed_samples) / len(elapsed_samples) if elapsed_samples else None
    )
    remaining_shards = state.shard_count - state.completed_shards
    estimated_remaining = (
        average_shard_elapsed * remaining_shards
        if average_shard_elapsed is not None and remaining_shards > 0
        else None
    )
    return DataPackBuildProgress(
        ready_shards=state.completed_shards,
        total_shards=state.shard_count,
        records_completed=records_completed,
        expected_records=state.expected_record_count,
        remaining_shards=remaining_shards,
        estimated_remaining_seconds=estimated_remaining,
        average_shard_elapsed_seconds=average_shard_elapsed,
    )
