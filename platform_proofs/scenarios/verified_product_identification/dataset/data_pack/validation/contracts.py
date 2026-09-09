"""Typed validation contracts for full Data Pack validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationStatus,
)


class DataPackValidationPhase(StrEnum):
    ARTIFACT_STRUCTURE = "ARTIFACT_STRUCTURE"
    MANIFEST_IDENTITY = "MANIFEST_IDENTITY"
    SHARD_INDEX = "SHARD_INDEX"
    RELATIONAL_SHARD = "RELATIONAL_SHARD"
    EMBEDDING_SHARD = "EMBEDDING_SHARD"
    CROSS_ARTIFACT_IDENTITY = "CROSS_ARTIFACT_IDENTITY"
    GLOBAL_COVERAGE = "GLOBAL_COVERAGE"
    GLOBAL_DUPLICATES = "GLOBAL_DUPLICATES"
    CHECKSUMS = "CHECKSUMS"
    BUILD_STATE = "BUILD_STATE"
    FINALIZATION = "FINALIZATION"


class DataPackValidationFailureCategory(StrEnum):
    STRUCTURE_FAIL = "STRUCTURE_FAIL"
    MANIFEST_FAIL = "MANIFEST_FAIL"
    SHARD_INDEX_FAIL = "SHARD_INDEX_FAIL"
    SHARD_PAIR_FAIL = "SHARD_PAIR_FAIL"
    RELATIONAL_FAIL = "RELATIONAL_FAIL"
    EMBEDDING_FAIL = "EMBEDDING_FAIL"
    CROSS_IDENTITY_FAIL = "CROSS_IDENTITY_FAIL"
    COVERAGE_FAIL = "COVERAGE_FAIL"
    DUPLICATE_FAIL = "DUPLICATE_FAIL"
    INTEGRITY_FAIL = "INTEGRITY_FAIL"
    BUILD_STATE_FAIL = "BUILD_STATE_FAIL"
    FINALIZATION_FAIL = "FINALIZATION_FAIL"


class DataPackValidationVerdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class PhaseValidationResult:
    phase: DataPackValidationPhase
    checks: tuple[ValidationCheck, ...]
    failure_category: DataPackValidationFailureCategory | None
    continue_validation: bool

    @property
    def passed(self) -> bool:
        return all(check.status is ValidationStatus.PASS for check in self.checks)


@dataclass(frozen=True, slots=True)
class ShardValidationSummary:
    ordinal: int
    relational_record_count: int
    embedding_record_count: int
    global_row_index_start: int
    global_row_index_end_exclusive: int
    passed: bool


@dataclass(frozen=True, slots=True)
class GlobalCoverageSummary:
    expected_record_count: int
    observed_relational_count: int
    observed_embedding_count: int
    shard_count: int
    first_gap_ordinal: int | None
    first_overlap_ordinal: int | None
    final_end_index: int | None
    passed: bool


@dataclass(frozen=True, slots=True)
class DuplicateValidationSummary:
    duplicate_global_row_index_count: int
    duplicate_source_ref_count: int
    duplicate_logical_point_id_count: int
    passed: bool


@dataclass(frozen=True, slots=True)
class DataPackValidationSummary:
    artifact_root: str
    validation_timestamp_utc: str
    expected_record_count: int
    observed_relational_count: int
    observed_embedding_count: int
    relational_shard_count: int
    embedding_shard_count: int
    ready_shard_count: int
    first_invalid_shard_ordinal: int | None
    embedding_dimension: int
    non_finite_vector_count: int
    zero_vector_count: int
    semantic_hash_mismatch_count: int
    finalized_artifact_valid: bool


@dataclass(frozen=True, slots=True)
class FullDataPackValidationReport:
    verdict: DataPackValidationVerdict
    summary: DataPackValidationSummary
    phase_results: tuple[PhaseValidationResult, ...]
    coverage_summary: GlobalCoverageSummary
    duplicate_summary: DuplicateValidationSummary
    shard_summaries: tuple[ShardValidationSummary, ...]

    @property
    def all_checks(self) -> tuple[ValidationCheck, ...]:
        checks: list[ValidationCheck] = []
        for phase_result in self.phase_results:
            checks.extend(phase_result.checks)
        return tuple(checks)
