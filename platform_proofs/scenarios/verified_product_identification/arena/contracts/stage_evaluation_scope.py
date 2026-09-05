"""Stage-local quality evaluation universe — immutable contract."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.arena.contracts.query_benchmark import (
    EmbeddingArenaQueryCase,
)
from platform_proofs.scenarios.verified_product_identification.arena.sampling.arena_sample import (
    ArenaSampleRecord,
)


def compute_stage_content_fingerprint(records: Sequence[ArenaSampleRecord]) -> str:
    payload = "|".join(record.offer_id for record in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class EmbeddingArenaStageEvaluationScope:
    """One quality-evaluation universe shared by every candidate in a stage."""

    stage_name: str
    records: tuple[ArenaSampleRecord, ...]
    query_cases: tuple[EmbeddingArenaQueryCase, ...]
    offer_index: Mapping[str, int]
    corpus_size: int
    benchmark_version: str
    sample_version: str
    content_fingerprint: str

    def __post_init__(self) -> None:
        if not self.stage_name.strip():
            msg = "stage_name must be non-empty"
            raise ValueError(msg)
        if self.corpus_size != len(self.records):
            msg = "corpus_size must match records length"
            raise ValueError(msg)
        if self.corpus_size <= 0:
            msg = "corpus_size must be > 0"
            raise ValueError(msg)
        if not self.query_cases:
            msg = "query_cases must not be empty"
            raise ValueError(msg)
