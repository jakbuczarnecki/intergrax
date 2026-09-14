# © Artur Czarnecki. All rights reserved.

"""Deterministic JSON serialization for performance certification reports."""

from __future__ import annotations

import json
from typing import TypedDict

from testing_support.execution_qualification.performance.models import (
    QualificationBenchmarkRunSample,
    QualificationPerformanceCertificationResult,
    QualificationPerformanceEnvironmentEvidence,
    QualificationProfilePerformanceResult,
    QualificationSuitePerformanceRow,
    TimedWallSeconds,
)


class _TimedWallPayload(TypedDict):
    seconds: float | None
    provenance: str
    source_note: str


class _EnvironmentPayload(TypedDict):
    python_version: str
    uv_version: str | None
    platform_system: str
    cpu_logical_count: int
    max_parallel: int


class _SamplePayload(TypedDict):
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


class _SuiteRowPayload(TypedDict):
    suite_id: str
    duration_seconds: float
    wall_share_percent: float
    exclusive_resource_id: str | None
    legacy_invocation_multiplicity: int | None


class _ProfilePayload(TypedDict):
    profile_id: str
    legacy_logical_subprocess_count: int
    legacy_semantic_unique_leaf_count: int
    canonical_physical_leaf_count: int
    duplicate_execution_eliminated: int
    duplicate_execution_eliminated_percent: float
    max_parallel: int
    canonical_wall: _TimedWallPayload
    legacy_wall: _TimedWallPayload
    speedup_ratio: float | None
    wall_time_reduction_percent: float | None
    critical_path_seconds: float | None
    total_leaf_work_seconds: float | None
    effective_concurrency: float | None
    scheduler_parallel_efficiency_estimate: float | None
    benchmark_samples: tuple[_SamplePayload, ...]
    slowest_suites: tuple[_SuiteRowPayload, ...]


class _ReportPayload(TypedDict):
    schema_version: int
    git_head: str
    semantic_parity_reference: str
    environment: _EnvironmentPayload
    profiles: tuple[_ProfilePayload, ...]
    primary_profile_id: str
    certification_decision: str
    performance_outcome: str | None
    performance_problem_solved: str


def _timed_wall_payload(wall: TimedWallSeconds) -> _TimedWallPayload:
    return {
        "seconds": wall.seconds,
        "provenance": wall.provenance.value,
        "source_note": wall.source_note,
    }


def _environment_payload(
    environment: QualificationPerformanceEnvironmentEvidence,
) -> _EnvironmentPayload:
    return {
        "python_version": environment.python_version,
        "uv_version": environment.uv_version,
        "platform_system": environment.platform_system,
        "cpu_logical_count": environment.cpu_logical_count,
        "max_parallel": environment.max_parallel,
    }


def _sample_payload(sample: QualificationBenchmarkRunSample) -> _SamplePayload:
    return {
        "run_id": sample.run_id,
        "repetition_index": sample.repetition_index,
        "wall_seconds": sample.wall_seconds,
        "total_leaf_work_seconds": sample.total_leaf_work_seconds,
        "critical_path_approximation_seconds": sample.critical_path_approximation_seconds,
        "effective_concurrency": sample.effective_concurrency,
        "scheduler_parallel_efficiency_estimate": sample.scheduler_parallel_efficiency_estimate,
        "slowest_suite_id": sample.slowest_suite_id,
        "slowest_suite_duration_seconds": sample.slowest_suite_duration_seconds,
        "run_status_pass": sample.run_status_pass,
    }


def _suite_row_payload(row: QualificationSuitePerformanceRow) -> _SuiteRowPayload:
    return {
        "suite_id": row.suite_id,
        "duration_seconds": row.duration_seconds,
        "wall_share_percent": row.wall_share_percent,
        "exclusive_resource_id": row.exclusive_resource_id,
        "legacy_invocation_multiplicity": row.legacy_invocation_multiplicity,
    }


def _profile_payload(profile: QualificationProfilePerformanceResult) -> _ProfilePayload:
    return {
        "profile_id": profile.profile_id,
        "legacy_logical_subprocess_count": profile.legacy_logical_subprocess_count,
        "legacy_semantic_unique_leaf_count": profile.legacy_semantic_unique_leaf_count,
        "canonical_physical_leaf_count": profile.canonical_physical_leaf_count,
        "duplicate_execution_eliminated": profile.duplicate_execution_eliminated,
        "duplicate_execution_eliminated_percent": profile.duplicate_execution_eliminated_percent,
        "max_parallel": profile.max_parallel,
        "canonical_wall": _timed_wall_payload(profile.canonical_wall),
        "legacy_wall": _timed_wall_payload(profile.legacy_wall),
        "speedup_ratio": profile.speedup_ratio,
        "wall_time_reduction_percent": profile.wall_time_reduction_percent,
        "critical_path_seconds": profile.critical_path_seconds,
        "total_leaf_work_seconds": profile.total_leaf_work_seconds,
        "effective_concurrency": profile.effective_concurrency,
        "scheduler_parallel_efficiency_estimate": profile.scheduler_parallel_efficiency_estimate,
        "benchmark_samples": tuple(
            _sample_payload(s) for s in profile.benchmark_samples
        ),
        "slowest_suites": tuple(_suite_row_payload(r) for r in profile.slowest_suites),
    }


def performance_certification_to_payload(
    report: QualificationPerformanceCertificationResult,
) -> _ReportPayload:
    ordered_profiles = tuple(
        sorted(report.profiles, key=lambda row: row.profile_id),
    )
    return {
        "schema_version": report.schema_version,
        "git_head": report.git_head,
        "semantic_parity_reference": report.semantic_parity_reference,
        "environment": _environment_payload(report.environment),
        "profiles": tuple(_profile_payload(p) for p in ordered_profiles),
        "primary_profile_id": report.primary_profile_id,
        "certification_decision": report.certification_decision.value,
        "performance_outcome": (
            report.performance_outcome.value if report.performance_outcome else None
        ),
        "performance_problem_solved": report.performance_problem_solved,
    }


def serialize_performance_certification_report(
    report: QualificationPerformanceCertificationResult,
) -> str:
    payload = performance_certification_to_payload(report)
    return json.dumps(payload, indent=2, sort_keys=True)
