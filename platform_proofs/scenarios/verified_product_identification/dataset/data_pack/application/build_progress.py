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
    return DataPackBuildProgress(
        ready_shards=state.completed_shards,
        total_shards=state.shard_count,
        records_completed=records_completed,
        expected_records=state.expected_record_count,
    )
