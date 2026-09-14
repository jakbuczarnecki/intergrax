# © Artur Czarnecki. All rights reserved.

"""Thin benchmark runner over catalog + plan runner + pytest subprocess executor."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from testing_support.execution_qualification.catalog.catalog import QualificationCatalog
from testing_support.execution_qualification.configuration import (
    resolve_execution_qualification_max_parallel,
)
from testing_support.execution_qualification.contracts import (
    QualificationRunConfig,
    QualificationRunStatus,
)
from testing_support.execution_qualification.graph_contracts import (
    QualificationExecutionPlan,
    QualificationNodeKind,
    QualificationPlanRunResult,
)
from testing_support.execution_qualification.performance.historical_baselines import (
    historical_legacy_wall_for_profile,
)
from testing_support.execution_qualification.performance.legacy_multiplicity import (
    legacy_execution_multiplicity_for_profile,
    parity_case_for_profile,
)
from testing_support.execution_qualification.performance.metrics import (
    critical_path_max_leaf_approximation,
    duplicate_execution_eliminated_count,
    duplicate_execution_eliminated_percent,
    effective_concurrency,
    optional_reduction_from_timed_walls,
    optional_speedup_from_timed_walls,
    primary_wall_seconds_from_samples,
    scheduler_parallel_efficiency_estimate,
)
from testing_support.execution_qualification.performance.models import (
    PerformanceWallTimeProvenance,
    QualificationBenchmarkRunSample,
    QualificationProfilePerformanceResult,
    QualificationSuitePerformanceRow,
    TimedWallSeconds,
)
from testing_support.execution_qualification.plan_runner import (
    run_qualification_execution_plan,
)

LIVE_SUITE_TIMEOUT_SECONDS = 6 * 3600.0
_SLOWEST_SUITE_LIMIT = 10


@dataclass(frozen=True, slots=True)
class QualificationBenchmarkRunOutcome:
    run_id: str
    repetition_index: int
    wall_seconds: float
    plan_result: QualificationPlanRunResult
    plan: QualificationExecutionPlan


def _suite_metadata(
    plan: QualificationExecutionPlan,
) -> dict[str, tuple[str | None, tuple[str, ...]]]:
    meta: dict[str, tuple[str | None, tuple[str, ...]]] = {}
    for node in plan.ordered_nodes:
        if node.kind is QualificationNodeKind.LEAF_SUITE and node.suite is not None:
            meta[node.node_id] = (
                node.suite.exclusive_resource_id,
                node.suite.pytest_arguments,
            )
    return meta


def _legacy_multiplicity_by_arguments(profile_id: str) -> Mapping[tuple[str, ...], int]:
    legacy = legacy_execution_multiplicity_for_profile(profile_id)
    return {
        row.pytest_arguments: row.legacy_invocation_count
        for row in legacy.per_argument_multiplicity
    }


def _build_run_config(
    repo_root: Path,
    artifact_root: Path,
    *,
    run_id: str,
    max_parallel: int,
) -> QualificationRunConfig:
    return QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=max_parallel,
        run_artifact_root=artifact_root,
        suite_timeout_seconds=LIVE_SUITE_TIMEOUT_SECONDS,
        run_id=run_id,
    )


def _slowest_suite_rows(
    plan_result: QualificationPlanRunResult,
    plan: QualificationExecutionPlan,
    *,
    wall_seconds: float,
    profile_id: str,
    limit: int,
) -> tuple[QualificationSuitePerformanceRow, ...]:
    meta = _suite_metadata(plan)
    legacy_by_args = _legacy_multiplicity_by_arguments(profile_id)
    rows: list[QualificationSuitePerformanceRow] = []
    for receipt in plan_result.suite_receipts:
        exclusive, pytest_args = meta.get(
            receipt.suite_id,
            (None, ()),
        )
        share = 0.0
        if wall_seconds > 0:
            share = (receipt.duration_seconds / wall_seconds) * 100.0
        rows.append(
            QualificationSuitePerformanceRow(
                suite_id=receipt.suite_id,
                duration_seconds=receipt.duration_seconds,
                wall_share_percent=share,
                exclusive_resource_id=exclusive,
                legacy_invocation_multiplicity=legacy_by_args.get(pytest_args),
            ),
        )
    ranked = sorted(
        rows,
        key=lambda row: (-row.duration_seconds, row.suite_id),
    )
    return tuple(ranked[:limit])


def _sample_from_outcome(
    outcome: QualificationBenchmarkRunOutcome,
    *,
    max_parallel: int,
) -> QualificationBenchmarkRunSample:
    durations = tuple(r.duration_seconds for r in outcome.plan_result.suite_receipts)
    total_leaf = sum(durations)
    critical = critical_path_max_leaf_approximation(durations)
    slowest = max(outcome.plan_result.suite_receipts, key=lambda r: r.duration_seconds)
    return QualificationBenchmarkRunSample(
        run_id=outcome.run_id,
        repetition_index=outcome.repetition_index,
        wall_seconds=outcome.wall_seconds,
        total_leaf_work_seconds=total_leaf,
        critical_path_approximation_seconds=critical,
        effective_concurrency=effective_concurrency(total_leaf, outcome.wall_seconds),
        scheduler_parallel_efficiency_estimate=scheduler_parallel_efficiency_estimate(
            total_leaf,
            outcome.wall_seconds,
            max_parallel,
        ),
        slowest_suite_id=slowest.suite_id,
        slowest_suite_duration_seconds=slowest.duration_seconds,
        run_status_pass=outcome.plan_result.status is QualificationRunStatus.PASS,
    )


class QualificationPerformanceBenchmarkRunner:
    def __init__(
        self,
        *,
        repo_root: Path,
        catalog: QualificationCatalog,
        artifact_base: Path,
    ) -> None:
        self._repo_root = repo_root
        self._catalog = catalog
        self._artifact_base = artifact_base

    def run_profile_once(
        self,
        profile_id: str,
        *,
        repetition_index: int,
        max_parallel: int | None = None,
    ) -> QualificationBenchmarkRunOutcome:
        resolved_parallel = resolve_execution_qualification_max_parallel(
            explicit_value=max_parallel,
        )
        compiled = self._catalog.compile_profile(profile_id)
        plan = compiled.plan
        run_id = f"perf-{profile_id}-{repetition_index}"
        artifact_root = self._artifact_base / run_id
        config = _build_run_config(
            self._repo_root,
            artifact_root,
            run_id=run_id,
            max_parallel=resolved_parallel,
        )
        wall_start = time.perf_counter()
        plan_result = run_qualification_execution_plan(plan, config)
        wall_seconds = max(time.perf_counter() - wall_start, 1e-9)
        if len(plan_result.suite_receipts) != len(plan.leaf_suite_ids):
            raise RuntimeError(
                "suite receipt count must equal plan.leaf_suite_ids "
                f"for profile {profile_id!r}"
            )
        return QualificationBenchmarkRunOutcome(
            run_id=run_id,
            repetition_index=repetition_index,
            wall_seconds=wall_seconds,
            plan_result=plan_result,
            plan=plan,
        )

    def build_profile_result_from_outcomes(
        self,
        profile_id: str,
        outcomes: tuple[QualificationBenchmarkRunOutcome, ...],
        *,
        max_parallel: int | None = None,
    ) -> QualificationProfilePerformanceResult:
        if not outcomes:
            raise ValueError("outcomes must be non-empty")
        resolved_parallel = resolve_execution_qualification_max_parallel(
            explicit_value=max_parallel,
        )
        legacy = legacy_execution_multiplicity_for_profile(profile_id)
        parity_case = parity_case_for_profile(profile_id)
        plan = self._catalog.compile_execution_plan(profile_id)
        canonical_count = len(plan.leaf_suite_ids)
        eliminated = duplicate_execution_eliminated_count(
            legacy.legacy_logical_subprocess_count,
            canonical_count,
        )
        eliminated_percent = duplicate_execution_eliminated_percent(
            legacy.legacy_logical_subprocess_count,
            canonical_count,
        )
        samples = tuple(
            _sample_from_outcome(outcome, max_parallel=resolved_parallel)
            for outcome in outcomes
        )
        wall_samples = tuple(sample.wall_seconds for sample in samples)
        primary_wall = primary_wall_seconds_from_samples(wall_samples)
        primary_sample = min(
            samples,
            key=lambda sample: abs(sample.wall_seconds - primary_wall),
        )
        legacy_wall = historical_legacy_wall_for_profile(profile_id)
        canonical_wall = TimedWallSeconds(
            seconds=primary_wall,
            provenance=PerformanceWallTimeProvenance.MEASURED,
            source_note=(
                f"QualificationPerformanceBenchmarkRunner; profile={profile_id}; "
                f"repetitions={len(outcomes)}; max_parallel={resolved_parallel}"
            ),
        )
        speedup = optional_speedup_from_timed_walls(legacy_wall, canonical_wall)
        reduction = optional_reduction_from_timed_walls(legacy_wall, canonical_wall)
        slowest = _slowest_suite_rows(
            outcomes[-1].plan_result,
            plan,
            wall_seconds=primary_wall,
            profile_id=profile_id,
            limit=_SLOWEST_SUITE_LIMIT,
        )
        _ = parity_case  # documents equivalent legacy obligation scope
        return QualificationProfilePerformanceResult(
            profile_id=profile_id,
            legacy_logical_subprocess_count=legacy.legacy_logical_subprocess_count,
            legacy_semantic_unique_leaf_count=legacy.legacy_semantic_unique_leaf_count,
            canonical_physical_leaf_count=canonical_count,
            duplicate_execution_eliminated=eliminated,
            duplicate_execution_eliminated_percent=eliminated_percent,
            max_parallel=resolved_parallel,
            canonical_wall=canonical_wall,
            legacy_wall=legacy_wall,
            speedup_ratio=speedup,
            wall_time_reduction_percent=reduction,
            critical_path_seconds=primary_sample.critical_path_approximation_seconds,
            total_leaf_work_seconds=primary_sample.total_leaf_work_seconds,
            effective_concurrency=primary_sample.effective_concurrency,
            scheduler_parallel_efficiency_estimate=primary_sample.scheduler_parallel_efficiency_estimate,
            benchmark_samples=samples,
            slowest_suites=slowest,
        )

    def build_static_profile_result(
        self, profile_id: str
    ) -> QualificationProfilePerformanceResult:
        legacy = legacy_execution_multiplicity_for_profile(profile_id)
        plan = self._catalog.compile_execution_plan(profile_id)
        canonical_count = len(plan.leaf_suite_ids)
        eliminated = duplicate_execution_eliminated_count(
            legacy.legacy_logical_subprocess_count,
            canonical_count,
        )
        eliminated_percent = duplicate_execution_eliminated_percent(
            legacy.legacy_logical_subprocess_count,
            canonical_count,
        )
        legacy_wall = historical_legacy_wall_for_profile(profile_id)
        canonical_wall = TimedWallSeconds(
            seconds=None,
            provenance=PerformanceWallTimeProvenance.NOT_AVAILABLE,
            source_note="structural profile row; canonical wall not measured in this build",
        )
        return QualificationProfilePerformanceResult(
            profile_id=profile_id,
            legacy_logical_subprocess_count=legacy.legacy_logical_subprocess_count,
            legacy_semantic_unique_leaf_count=legacy.legacy_semantic_unique_leaf_count,
            canonical_physical_leaf_count=canonical_count,
            duplicate_execution_eliminated=eliminated,
            duplicate_execution_eliminated_percent=eliminated_percent,
            max_parallel=resolve_execution_qualification_max_parallel(
                explicit_value=None
            ),
            canonical_wall=canonical_wall,
            legacy_wall=legacy_wall,
            speedup_ratio=None,
            wall_time_reduction_percent=None,
            critical_path_seconds=None,
            total_leaf_work_seconds=None,
            effective_concurrency=None,
            scheduler_parallel_efficiency_estimate=None,
            benchmark_samples=(),
            slowest_suites=(),
        )
