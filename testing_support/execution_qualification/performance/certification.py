# © Artur Czarnecki. All rights reserved.

"""Aggregate performance certification report builder."""

from __future__ import annotations

import subprocess
from pathlib import Path

from testing_support.execution_qualification.catalog.composition import (
    build_default_qualification_catalog,
)
from testing_support.execution_qualification.catalog.profile_builders import (
    NPSC5E_FINAL_PROFILE_ID,
    NPSC5F_FINAL_PROFILE_ID,
    NPSC5F_R1_PROFILE_ID,
    NPSC5F_R2_PROFILE_ID,
    NPSC5F_R3_PROFILE_ID,
    NPSC5F_R4_PROFILE_ID,
)
from testing_support.execution_qualification.performance.environment import (
    collect_performance_environment_evidence,
)
from testing_support.execution_qualification.performance.metrics import (
    classify_performance_outcome,
    classify_performance_problem_solved,
)
from testing_support.execution_qualification.performance.models import (
    QualificationPerformanceCertificationDecision,
    QualificationPerformanceCertificationResult,
    QualificationProfilePerformanceResult,
)
from testing_support.execution_qualification.performance.runner import (
    QualificationPerformanceBenchmarkRunner,
)

SEMANTIC_PARITY_REFERENCE = "GLOBAL SEMANTIC PARITY CERTIFICATION = PASS @ dffe2ae52a6938e0620ae4a3cc5e4be760e7f2f7"

MANDATORY_BENCHMARK_PROFILE_IDS: tuple[str, ...] = (
    NPSC5F_R1_PROFILE_ID,
    NPSC5F_R2_PROFILE_ID,
    NPSC5F_R3_PROFILE_ID,
    NPSC5F_R4_PROFILE_ID,
    NPSC5F_FINAL_PROFILE_ID,
)

OPTIONAL_BENCHMARK_PROFILE_IDS: tuple[str, ...] = (NPSC5E_FINAL_PROFILE_ID,)

PRIMARY_BENCHMARK_PROFILE_ID = NPSC5F_FINAL_PROFILE_ID


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot resolve git HEAD for performance certification")
    return completed.stdout.strip()


def default_artifact_base(repo_root: Path) -> Path:
    return repo_root / ".tmp" / "session" / "qualification-performance"


def build_default_benchmark_runner(
    repo_root: Path | None = None,
    *,
    artifact_base: Path | None = None,
) -> QualificationPerformanceBenchmarkRunner:
    root = repo_root if repo_root is not None else resolve_repo_root()
    base = artifact_base if artifact_base is not None else default_artifact_base(root)
    return QualificationPerformanceBenchmarkRunner(
        repo_root=root,
        catalog=build_default_qualification_catalog(),
        artifact_base=base,
    )


def build_structural_profile_matrix(
    runner: QualificationPerformanceBenchmarkRunner,
    profile_ids: tuple[str, ...],
) -> tuple[QualificationProfilePerformanceResult, ...]:
    return tuple(
        runner.build_static_profile_result(profile_id) for profile_id in profile_ids
    )


def assemble_certification_report(
    *,
    git_head: str,
    environment_max_parallel: int,
    profiles: tuple[QualificationProfilePerformanceResult, ...],
    primary_profile_id: str,
    canonical_run_failed: bool,
) -> QualificationPerformanceCertificationResult:
    if canonical_run_failed:
        return QualificationPerformanceCertificationResult(
            schema_version=1,
            git_head=git_head,
            semantic_parity_reference=SEMANTIC_PARITY_REFERENCE,
            environment=collect_performance_environment_evidence(
                max_parallel=environment_max_parallel,
            ),
            profiles=profiles,
            primary_profile_id=primary_profile_id,
            certification_decision=QualificationPerformanceCertificationDecision.BLOCKED,
            performance_outcome=None,
            performance_problem_solved="PARTIALLY",
        )

    primary = next(
        (row for row in profiles if row.profile_id == primary_profile_id),
        None,
    )
    reduction = primary.wall_time_reduction_percent if primary is not None else None
    canonical_seconds = primary.canonical_wall.seconds if primary is not None else None
    outcome = classify_performance_outcome(reduction)
    solved = classify_performance_problem_solved(canonical_seconds, reduction)
    return QualificationPerformanceCertificationResult(
        schema_version=1,
        git_head=git_head,
        semantic_parity_reference=SEMANTIC_PARITY_REFERENCE,
        environment=collect_performance_environment_evidence(
            max_parallel=environment_max_parallel,
        ),
        profiles=profiles,
        primary_profile_id=primary_profile_id,
        certification_decision=QualificationPerformanceCertificationDecision.PASS,
        performance_outcome=outcome,
        performance_problem_solved=solved,
    )


def merge_measured_primary_profile(
    structural_profiles: tuple[QualificationProfilePerformanceResult, ...],
    measured_primary: QualificationProfilePerformanceResult,
) -> tuple[QualificationProfilePerformanceResult, ...]:
    merged: list[QualificationProfilePerformanceResult] = []
    replaced = False
    for row in structural_profiles:
        if row.profile_id == measured_primary.profile_id:
            merged.append(measured_primary)
            replaced = True
        else:
            merged.append(row)
    if not replaced:
        merged.append(measured_primary)
    return tuple(sorted(merged, key=lambda row: row.profile_id))
