"""Typed execution budget for embedding arena stages — provider-neutral."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingArenaExecutionBudget:
    """Resource and stage sizing contract for an arena execution profile."""

    profile_id: str
    stage_a_records: int
    stage_b_records: int
    stage_c_records: int
    max_stage_c_finalists: int
    candidate_timeout_seconds: float
    default_batch_size: int
    fallback_batch_size: int
    batch_sweep_sizes: tuple[int, ...]
    isolate_candidates: bool
    screening_mode: bool
    max_vram_bytes: int | None
    query_latency_repetitions: int
    query_latency_query_count: int

    def __post_init__(self) -> None:
        if self.stage_a_records <= 0:
            msg = "stage_a_records must be > 0"
            raise ValueError(msg)
        if self.stage_a_records > self.stage_b_records:
            msg = "stage_a_records must be <= stage_b_records"
            raise ValueError(msg)
        if self.stage_b_records > self.stage_c_records:
            msg = "stage_b_records must be <= stage_c_records"
            raise ValueError(msg)
        if self.max_stage_c_finalists <= 0:
            msg = "max_stage_c_finalists must be > 0"
            raise ValueError(msg)
        if self.candidate_timeout_seconds <= 0:
            msg = "candidate_timeout_seconds must be > 0"
            raise ValueError(msg)
        if self.default_batch_size <= 0:
            msg = "default_batch_size must be > 0"
            raise ValueError(msg)
        if self.fallback_batch_size <= 0:
            msg = "fallback_batch_size must be > 0"
            raise ValueError(msg)
        if self.query_latency_repetitions <= 0:
            msg = "query_latency_repetitions must be > 0"
            raise ValueError(msg)
        if self.query_latency_query_count <= 0:
            msg = "query_latency_query_count must be > 0"
            raise ValueError(msg)

    @property
    def uses_batch_sweep(self) -> bool:
        return len(self.batch_sweep_sizes) > 0

    def batch_sizes_for_candidate(
        self,
        *,
        fixed_provider_batch_size: int | None,
    ) -> tuple[int, ...]:
        if fixed_provider_batch_size is not None:
            return (fixed_provider_batch_size,)
        if self.uses_batch_sweep:
            return self.batch_sweep_sizes
        return (self.default_batch_size, self.fallback_batch_size)
