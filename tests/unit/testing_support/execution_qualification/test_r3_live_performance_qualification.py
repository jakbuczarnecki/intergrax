# © Artur Czarnecki. All rights reserved.

"""Live R3 performance evidence (opt-in via INTERGRAX_R3_LIVE_PERF=1)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from testing_support.execution_qualification.performance_evidence import (
    format_bottleneck_view,
    format_performance_summary,
    format_suite_timing_table,
)
from testing_support.execution_qualification.performance_snapshot import (
    build_suite_timing_rows,
)
from tests.unit.runtime.architecture import (
    test_npsc5e_r3_final_child_fanout_partial_recovery_qualification as r3_final,
)
from tests.unit.runtime.architecture.npsc5e_r3_final_execution_qualification import (
    NPSC5E_R3_EXECUTION_QUALIFICATION_MAX_PARALLEL,
    build_npsc5e_r3_mandatory_projections,
    run_npsc5e_r3_mandatory_qualification_measured,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("INTERGRAX_R3_LIVE_PERF") != "1",
    reason="set INTERGRAX_R3_LIVE_PERF=1 to collect live qualification timings",
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EVIDENCE_DIR = _REPO_ROOT / ".tmp" / "session" / "r3-performance-qualification"

_R3_REPRESENTATIVE_SUBSET: tuple[tuple[str, list[str]], ...] = tuple(
    entry
    for entry in r3_final._MANDATORY_SUITES
    if entry[0]
    in {
        "Terminal",
        "Fan-out",
        "P0A",
        "NPSC-5C",
        "Checkpoint store",
        "Long-running",
    }
)


def _write_evidence(name: str, body: str) -> Path:
    _EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = _EVIDENCE_DIR / name
    path.write_text(body, encoding="utf-8")
    return path


def _record_measured_run(
    prefix: str,
    source: tuple[tuple[str, list[str]], ...],
    measured: object,
) -> None:
    from testing_support.execution_qualification.performance_snapshot import (
        ExecutionQualificationMeasuredRun,
    )

    assert isinstance(measured, ExecutionQualificationMeasuredRun)
    adapted = build_npsc5e_r3_mandatory_projections(source)
    rows = build_suite_timing_rows(adapted, measured.result)
    summary = format_performance_summary(measured.performance)
    table = format_suite_timing_table(rows)
    bottleneck = format_bottleneck_view(rows)
    payload = {
        "version": 1,
        "prefix": prefix,
        "performance": {
            "run_id": measured.performance.run_id,
            "run_status": measured.performance.run_status.value,
            "max_parallel": measured.performance.max_parallel,
            "suite_count": measured.performance.suite_count,
            "wall_duration_seconds": measured.performance.wall_duration_seconds,
            "sum_child_duration_seconds": measured.performance.sum_child_duration_seconds,
            "max_child_duration_seconds": measured.performance.max_child_duration_seconds,
            "observed_overlap_ratio": measured.performance.observed_overlap_ratio,
            "artifact_root": measured.performance.artifact_root,
        },
        "suite_rows": [
            {
                "suite_id": row.suite_id,
                "display_label": row.display_label,
                "status": row.status,
                "outcome_kind": row.outcome_kind,
                "duration_seconds": row.duration_seconds,
                "log_path": row.log_path,
                "child_duration_share": row.child_duration_share,
            }
            for row in rows
        ],
    }
    _write_evidence(f"{prefix}-summary.txt", summary)
    _write_evidence(f"{prefix}-suites.md", table)
    _write_evidence(f"{prefix}-bottleneck.md", bottleneck)
    _write_evidence(f"{prefix}-evidence.json", json.dumps(payload, indent=2, sort_keys=True))


@pytest.mark.parametrize("max_parallel", [1, 2, 3, 4])
def test_r3_representative_subset_concurrency(max_parallel: int) -> None:
    measured = run_npsc5e_r3_mandatory_qualification_measured(
        _R3_REPRESENTATIVE_SUBSET,
        _REPO_ROOT,
        run_id=f"r3-subset-mp{max_parallel}",
        max_parallel=max_parallel,
    )
    _record_measured_run(f"subset-mp{max_parallel}", _R3_REPRESENTATIVE_SUBSET, measured)


def test_r3_full_matrix_production_parallelism() -> None:
    measured = run_npsc5e_r3_mandatory_qualification_measured(
        r3_final._MANDATORY_SUITES,
        _REPO_ROOT,
        run_id="r3-full-mp2-baseline",
        max_parallel=NPSC5E_R3_EXECUTION_QUALIFICATION_MAX_PARALLEL,
    )
    _record_measured_run("full-mp2-baseline", r3_final._MANDATORY_SUITES, measured)


def test_r3_full_matrix_candidate_parallelism_three() -> None:
    if os.environ.get("INTERGRAX_R3_FULL_MP3") != "1":
        pytest.skip("set INTERGRAX_R3_FULL_MP3=1 for full-matrix max_parallel=3 candidate")
    measured = run_npsc5e_r3_mandatory_qualification_measured(
        r3_final._MANDATORY_SUITES,
        _REPO_ROOT,
        run_id="r3-full-mp3-candidate",
        max_parallel=3,
    )
    _record_measured_run("full-mp3-candidate", r3_final._MANDATORY_SUITES, measured)


def test_r3_full_matrix_serial_equivalent_baseline() -> None:
    if os.environ.get("INTERGRAX_R3_SERIAL_BASELINE") != "1":
        pytest.skip("set INTERGRAX_R3_SERIAL_BASELINE=1 for full-matrix max_parallel=1")
    measured = run_npsc5e_r3_mandatory_qualification_measured(
        r3_final._MANDATORY_SUITES,
        _REPO_ROOT,
        run_id="r3-full-mp1-serial",
        max_parallel=1,
    )
    _record_measured_run("full-mp1-serial", r3_final._MANDATORY_SUITES, measured)
