# © Artur Czarnecki. All rights reserved.

"""Typed performance certification models (immutable, no Any)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class PerformanceWallTimeProvenance(StrEnum):
    MEASURED = "measured"
    HISTORICAL_MEASURED = "historical_measured"
    DERIVED = "derived"
    NOT_AVAILABLE = "not_available"


class QualificationPerformanceOutcome(StrEnum):
    EXCELLENT = "excellent"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class QualificationPerformanceCertificationDecision(StrEnum):
    PASS = "pass"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class TimedWallSeconds:
    seconds: float | None
    provenance: PerformanceWallTimeProvenance
    source_note: str

    def __post_init__(self) -> None:
        if self.provenance is PerformanceWallTimeProvenance.NOT_AVAILABLE:
            if self.seconds is not None:
                raise ValueError("NOT_AVAILABLE wall time must omit seconds")
            return
        if self.seconds is None:
            raise ValueError("seconds required unless provenance is NOT_AVAILABLE")
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("wall seconds must be finite and non-negative")
        if not self.source_note.strip():
            raise ValueError("source_note must be non-empty")


@dataclass(frozen=True, slots=True)
class QualificationBenchmarkRunSample:
    run_id: str
    repetition_index: int
    wall_seconds: float
    total_leaf_work_seconds: float
    critical_path_approximation_seconds: float
    effective_concurrency: float
    scheduler_parallel_efficiency_estimate: float
    slowest_suite_id: str
    slowest_suite_duration_seconds: float
    run_status_pass: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("wall_seconds", self.wall_seconds),
            ("total_leaf_work_seconds", self.total_leaf_work_seconds),
            (
                "critical_path_approximation_seconds",
                self.critical_path_approximation_seconds,
            ),
            ("effective_concurrency", self.effective_concurrency),
            (
                "scheduler_parallel_efficiency_estimate",
                self.scheduler_parallel_efficiency_estimate,
            ),
            (
                "slowest_suite_duration_seconds",
                self.slowest_suite_duration_seconds,
            ),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.repetition_index < 0:
            raise ValueError("repetition_index must be >= 0")
        if not self.slowest_suite_id:
            raise ValueError("slowest_suite_id must be non-empty")


@dataclass(frozen=True, slots=True)
class QualificationSuitePerformanceRow:
    suite_id: str
    duration_seconds: float
    wall_share_percent: float
    exclusive_resource_id: str | None
    legacy_invocation_multiplicity: int | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise ValueError("duration_seconds must be finite and non-negative")
        if not math.isfinite(self.wall_share_percent) or self.wall_share_percent < 0:
            raise ValueError("wall_share_percent must be finite and non-negative")
        if self.legacy_invocation_multiplicity is not None:
            if self.legacy_invocation_multiplicity < 1:
                raise ValueError("legacy_invocation_multiplicity must be >= 1")


@dataclass(frozen=True, slots=True)
class QualificationProfilePerformanceResult:
    profile_id: str
    legacy_logical_subprocess_count: int
    legacy_semantic_unique_leaf_count: int
    canonical_physical_leaf_count: int
    duplicate_execution_eliminated: int
    duplicate_execution_eliminated_percent: float
    max_parallel: int
    canonical_wall: TimedWallSeconds
    legacy_wall: TimedWallSeconds
    speedup_ratio: float | None
    wall_time_reduction_percent: float | None
    critical_path_seconds: float | None
    total_leaf_work_seconds: float | None
    effective_concurrency: float | None
    scheduler_parallel_efficiency_estimate: float | None
    benchmark_samples: tuple[QualificationBenchmarkRunSample, ...]
    slowest_suites: tuple[QualificationSuitePerformanceRow, ...]

    def __post_init__(self) -> None:
        if self.legacy_logical_subprocess_count < 1:
            raise ValueError("legacy_logical_subprocess_count must be >= 1")
        if self.canonical_physical_leaf_count < 1:
            raise ValueError("canonical_physical_leaf_count must be >= 1")
        if self.duplicate_execution_eliminated < 0:
            raise ValueError("duplicate_execution_eliminated must be >= 0")
        if not math.isfinite(self.duplicate_execution_eliminated_percent):
            raise ValueError("duplicate_execution_eliminated_percent must be finite")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")


@dataclass(frozen=True, slots=True)
class QualificationPerformanceEnvironmentEvidence:
    python_version: str
    uv_version: str | None
    platform_system: str
    cpu_logical_count: int
    max_parallel: int

    def __post_init__(self) -> None:
        if self.cpu_logical_count < 1:
            raise ValueError("cpu_logical_count must be >= 1")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")


@dataclass(frozen=True, slots=True)
class QualificationPerformanceCertificationResult:
    schema_version: int
    git_head: str
    semantic_parity_reference: str
    environment: QualificationPerformanceEnvironmentEvidence
    profiles: tuple[QualificationProfilePerformanceResult, ...]
    primary_profile_id: str
    certification_decision: QualificationPerformanceCertificationDecision
    performance_outcome: QualificationPerformanceOutcome | None
    performance_problem_solved: str

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("schema_version must be >= 1")
        if not self.git_head.strip():
            raise ValueError("git_head must be non-empty")
        if not self.primary_profile_id.strip():
            raise ValueError("primary_profile_id must be non-empty")
        if self.performance_problem_solved not in {"YES", "PARTIALLY", "NO"}:
            raise ValueError("performance_problem_solved must be YES, PARTIALLY, or NO")
