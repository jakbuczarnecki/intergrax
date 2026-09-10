# © Artur Czarnecki. All rights reserved.

"""Typed configuration and manifest projection for NPSC-5E/R3 Final mandatory certification."""

from __future__ import annotations

import uuid
from pathlib import Path

from testing_support.execution_qualification.contracts import (
    ExecutionQualificationRunResult,
    QualificationCoordinatorError,
    QualificationManifestError,
    QualificationRunConfig,
    QualificationRunManifest,
)
from testing_support.execution_qualification.coordinator import (
    validate_and_run,
    validate_and_run_measured,
)
from testing_support.execution_qualification.performance_snapshot import (
    ExecutionQualificationMeasuredRun,
)
from testing_support.execution_qualification.failure_report import assert_execution_qualification_pass
from testing_support.execution_qualification.frozen_pytest_adapter import (
    AdaptedMandatorySuite,
    FrozenPytestSuiteSource,
    adapt_frozen_pytest_suites,
    manifest_from_adapted_suites,
)

NPSC5E_R3_CROSS_DB_EXCLUSIVE_RESOURCE_ID = "npsc5e-r3-cross-db"

NPSC5E_R3_EXECUTION_QUALIFICATION_MAX_PARALLEL = 2

# Full-matrix suites may be long-running; bounded coordinator requires a positive timeout.
NPSC5E_R3_EXECUTION_QUALIFICATION_SUITE_TIMEOUT_SECONDS = 6 * 3600.0

NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID: dict[str, str] = {
    "R1 Final": "npsc5e-r3.mandatory.r1-final",
    "R2 Final": "npsc5e-r3.mandatory.r2-final",
    "R3 implementation gate": "npsc5e-r3.mandatory.r3-implementation-gate",
    "P0A": "npsc5e-r3.mandatory.p0a",
    "DG_001": "npsc5e-r3.mandatory.dg-001",
    "NPSC-5A": "npsc5e-r3.mandatory.npsc-5a",
    "NPSC-5B Final": "npsc5e-r3.mandatory.npsc-5b-final",
    "NPSC-5C": "npsc5e-r3.mandatory.npsc-5c",
    "NPSC-5D Final": "npsc5e-r3.mandatory.npsc-5d-final",
    "HITL R3": "npsc5e-r3.mandatory.hitl-r3",
    "Attempt lifecycle": "npsc5e-r3.mandatory.attempt-lifecycle",
    "Child execution": "npsc5e-r3.mandatory.child-execution",
    "Terminal": "npsc5e-r3.mandatory.terminal",
    "Cancellation": "npsc5e-r3.mandatory.cancellation",
    "Checkpoint store": "npsc5e-r3.mandatory.checkpoint-store",
    "Long-running": "npsc5e-r3.mandatory.long-running",
    "Fan-out": "npsc5e-r3.mandatory.fan-out",
}

NPSC5E_R3_EXCLUSIVE_RESOURCE_BY_LABEL: dict[str, str] = {
    "R3 implementation gate": NPSC5E_R3_CROSS_DB_EXCLUSIVE_RESOURCE_ID,
}


def build_npsc5e_r3_mandatory_projections(
    source: FrozenPytestSuiteSource,
) -> tuple[AdaptedMandatorySuite, ...]:
    return adapt_frozen_pytest_suites(
        source,
        label_to_suite_id=NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID,
        exclusive_resource_by_label=NPSC5E_R3_EXCLUSIVE_RESOURCE_BY_LABEL,
    )


def build_npsc5e_r3_mandatory_manifest(
    source: FrozenPytestSuiteSource,
) -> QualificationRunManifest:
    return manifest_from_adapted_suites(build_npsc5e_r3_mandatory_projections(source))


def label_by_suite_id_from_projections(
    adapted: tuple[AdaptedMandatorySuite, ...],
) -> dict[str, str]:
    return {entry.suite.suite_id: entry.display_label for entry in adapted}


def npsc5e_r3_qualification_run_config(
    repo_root: Path,
    *,
    run_id: str | None = None,
    max_parallel: int | None = None,
) -> QualificationRunConfig:
    resolved_run_id = run_id if run_id is not None else f"npsc5e-r3-{uuid.uuid4().hex}"
    artifact_root = repo_root / "build" / "qualification" / resolved_run_id
    resolved_parallel = (
        max_parallel
        if max_parallel is not None
        else NPSC5E_R3_EXECUTION_QUALIFICATION_MAX_PARALLEL
    )
    return QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=resolved_parallel,
        run_artifact_root=artifact_root,
        suite_timeout_seconds=NPSC5E_R3_EXECUTION_QUALIFICATION_SUITE_TIMEOUT_SECONDS,
        run_id=resolved_run_id,
    )


def run_npsc5e_r3_mandatory_qualification(
    source: FrozenPytestSuiteSource,
    repo_root: Path,
    *,
    run_id: str | None = None,
    max_parallel: int | None = None,
) -> ExecutionQualificationRunResult:
    measured = run_npsc5e_r3_mandatory_qualification_measured(
        source,
        repo_root,
        run_id=run_id,
        max_parallel=max_parallel,
    )
    return measured.result


def run_npsc5e_r3_mandatory_qualification_measured(
    source: FrozenPytestSuiteSource,
    repo_root: Path,
    *,
    run_id: str | None = None,
    max_parallel: int | None = None,
) -> ExecutionQualificationMeasuredRun:
    adapted = build_npsc5e_r3_mandatory_projections(source)
    manifest = manifest_from_adapted_suites(adapted)
    config = npsc5e_r3_qualification_run_config(
        repo_root,
        run_id=run_id,
        max_parallel=max_parallel,
    )
    label_by_suite_id = label_by_suite_id_from_projections(adapted)
    try:
        measured = validate_and_run_measured(manifest, config)
    except QualificationManifestError:
        raise
    except QualificationCoordinatorError as exc:
        raise AssertionError(f"execution qualification infrastructure failure: {exc}") from exc
    assert_execution_qualification_pass(measured.result, label_by_suite_id=label_by_suite_id)
    return measured
