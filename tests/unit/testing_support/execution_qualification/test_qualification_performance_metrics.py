# © Artur Czarnecki. All rights reserved.

import pytest

from testing_support.execution_qualification.catalog.composition import (
    build_default_qualification_catalog,
)
from testing_support.execution_qualification.performance.certification import (
    assemble_certification_report,
    build_default_benchmark_runner,
    build_structural_profile_matrix,
)
from testing_support.execution_qualification.performance.legacy_multiplicity import (
    legacy_execution_multiplicity_for_profile,
)
from testing_support.execution_qualification.performance.metrics import (
    duplicate_execution_eliminated_count,
    duplicate_execution_eliminated_percent,
    optional_reduction_from_timed_walls,
    optional_speedup_from_timed_walls,
    speedup_ratio,
    wall_time_reduction_percent,
    wall_sample_statistics,
)
from testing_support.execution_qualification.performance.models import (
    PerformanceWallTimeProvenance,
    TimedWallSeconds,
)
from testing_support.execution_qualification.performance.serialization import (
    performance_certification_to_payload,
    serialize_performance_certification_report,
)


def test_duplicate_elimination_count() -> None:
    assert duplicate_execution_eliminated_count(384, 43) == 341


def test_duplicate_elimination_percent() -> None:
    value = duplicate_execution_eliminated_percent(384, 43)
    assert 88.0 < value < 89.0


def test_speedup_ratio() -> None:
    assert speedup_ratio(100.0, 50.0) == 2.0


def test_wall_time_reduction_percent() -> None:
    assert wall_time_reduction_percent(892.31, 400.0) == pytest.approx(55.17, rel=0.01)


def test_zero_denominator_speedup_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        speedup_ratio(0.0, 10.0)


def test_provenance_optional_speedup() -> None:
    legacy = TimedWallSeconds(
        seconds=None,
        provenance=PerformanceWallTimeProvenance.NOT_AVAILABLE,
        source_note="missing",
    )
    canonical = TimedWallSeconds(
        seconds=100.0,
        provenance=PerformanceWallTimeProvenance.MEASURED,
        source_note="measured",
    )
    assert optional_speedup_from_timed_walls(legacy, canonical) is None
    assert optional_reduction_from_timed_walls(legacy, canonical) is None


def test_wall_sample_statistics_two_runs() -> None:
    stats = wall_sample_statistics((100.0, 110.0))
    assert stats.median_seconds is None
    assert stats.mean_seconds == 105.0


def test_deterministic_report_serialization() -> None:
    runner = build_default_benchmark_runner()
    structural = build_structural_profile_matrix(
        runner,
        ("npsc5f-final",),
    )
    report = assemble_certification_report(
        git_head="abc123",
        environment_max_parallel=2,
        profiles=structural,
        primary_profile_id="npsc5f-final",
        canonical_run_failed=False,
    )
    first = serialize_performance_certification_report(report)
    second = serialize_performance_certification_report(report)
    assert first == second
    payload = performance_certification_to_payload(report)
    assert payload["schema_version"] == 1


def test_npsc5f_final_legacy_logical_exceeds_canonical() -> None:
    legacy = legacy_execution_multiplicity_for_profile("npsc5f-final")
    plan = build_default_qualification_catalog().compile_execution_plan("npsc5f-final")
    assert legacy.legacy_logical_subprocess_count > len(plan.leaf_suite_ids)
