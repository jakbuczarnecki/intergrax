# © Artur Czarnecki. All rights reserved.

"""Qualification performance certification package."""

from testing_support.execution_qualification.performance.certification import (
    MANDATORY_BENCHMARK_PROFILE_IDS,
    PRIMARY_BENCHMARK_PROFILE_ID,
    assemble_certification_report,
    build_default_benchmark_runner,
    build_structural_profile_matrix,
    merge_measured_primary_profile,
    read_git_head,
    resolve_repo_root,
)
from testing_support.execution_qualification.performance.models import (
    PerformanceWallTimeProvenance,
    QualificationPerformanceCertificationDecision,
    QualificationPerformanceCertificationResult,
    QualificationPerformanceOutcome,
    QualificationProfilePerformanceResult,
)
from testing_support.execution_qualification.performance.runner import (
    QualificationPerformanceBenchmarkRunner,
)
from testing_support.execution_qualification.performance.serialization import (
    serialize_performance_certification_report,
)

__all__ = (
    "MANDATORY_BENCHMARK_PROFILE_IDS",
    "PRIMARY_BENCHMARK_PROFILE_ID",
    "PerformanceWallTimeProvenance",
    "QualificationPerformanceBenchmarkRunner",
    "QualificationPerformanceCertificationDecision",
    "QualificationPerformanceCertificationResult",
    "QualificationPerformanceOutcome",
    "QualificationProfilePerformanceResult",
    "assemble_certification_report",
    "build_default_benchmark_runner",
    "build_structural_profile_matrix",
    "merge_measured_primary_profile",
    "read_git_head",
    "resolve_repo_root",
    "serialize_performance_certification_report",
)
