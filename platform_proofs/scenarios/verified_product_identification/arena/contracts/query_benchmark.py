"""Query benchmark contracts — benchmark-only ground truth."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    QueryDifficultyClass,
)


@dataclass(frozen=True, slots=True)
class ArenaSourceRef:
    """Arena document identity — offer_id within the fixed sample corpus."""

    offer_id: str
    global_row_index: int


@dataclass(frozen=True, slots=True)
class BenchmarkClusterEvidence:
    """Benchmark-only cluster reference — never used at retrieval runtime."""

    cluster_id: int
    purpose: str


@dataclass(frozen=True, slots=True)
class EmbeddingArenaQueryCase:
    case_id: str
    query_text: str
    difficulty: QueryDifficultyClass
    relevant_source_refs: tuple[ArenaSourceRef, ...]
    provenance: str
    benchmark_only_cluster_evidence: BenchmarkClusterEvidence | None
    hard_negative_offer_ids: tuple[str, ...]
    is_long_input_query: bool

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            msg = "case_id must be non-empty"
            raise ValueError(msg)
        if not self.query_text.strip():
            msg = "query_text must be non-empty"
            raise ValueError(msg)
        if not self.relevant_source_refs:
            msg = "relevant_source_refs must not be empty"
            raise ValueError(msg)
