# © Artur Czarnecki. All rights reserved.

"""Phase A source freeze for DS-E2E-15J-L1.R6 (production + R4.R4/R4.R5 proof groups)."""

from __future__ import annotations

import json
from pathlib import Path

from testing_support.decision_e2e.local_ai_incident_qualification import (
    R4R1_EVALUATOR_MAX_ITERATIONS,
    R4R1_MAX_DECISION_REVISIONS,
    R4R1_RUNTIME_VERSION,
    R4R1_SEMANTIC_VERIFICATION,
    R4R1_TEMPERATURE,
    resolve_repository_head_sha,
    semantic_source_groups_for_r4r1,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeCheck,
    SourceFreezeReport,
    SourceFreezeStatus,
)
from testing_support.decision_e2e.local_qualification_session.source_fingerprint import (
    capture_semantic_source_fingerprint,
)
from testing_support.decision_e2e.model_matrix.matrix_artifact_contract import (
    MATRIX_ARTIFACT_SCHEMA_VERSION,
)
from testing_support.decision_e2e.model_matrix.registry import qualification_matrix_version
from testing_support.decision_e2e.natural_alignment.source_freeze import (
    verify_natural_alignment_source_freeze,
)
from testing_support.decision_e2e.scenario_qualification import AI_INCIDENT_SCENARIO_ID

TASK_ID = "DS-E2E-15J-L1.R6"

_BASELINE_RELATIVE = (
    "testing_support/decision_e2e/model_matrix/data/source_freeze_baseline.json"
)

_REQUIRED_GROUPS = (
    "15I",
    "15K-B",
    "O1",
    "O1.R1",
    "O2",
    "QI1",
    "QI2",
    "R4.R4",
    "R4.R5",
)

_FROZEN_CONFIGURATION = {
    "scenario_id": AI_INCIDENT_SCENARIO_ID,
    "matrix_version": qualification_matrix_version(),
    "runtime_version": R4R1_RUNTIME_VERSION.normalized(),
    "temperature": R4R1_TEMPERATURE,
    "evaluator_iterations": R4R1_EVALUATOR_MAX_ITERATIONS,
    "revision_budget": R4R1_MAX_DECISION_REVISIONS,
    "semantic_verification": R4R1_SEMANTIC_VERIFICATION,
    "analysis_schema_version": MATRIX_ARTIFACT_SCHEMA_VERSION,
}


def semantic_source_groups_for_r6() -> dict[str, tuple[str, ...]]:
    return {
        "R6 matrix": (
            "testing_support/decision_e2e/model_matrix/registry.py",
            "testing_support/decision_e2e/model_matrix/profiles.py",
            "testing_support/decision_e2e/model_matrix/qualification_plan.py",
            "testing_support/decision_e2e/model_matrix/qualification_planner.py",
            "testing_support/decision_e2e/model_matrix/analysis.py",
            "testing_support/decision_e2e/model_matrix/source_freeze.py",
            "testing_support/decision_e2e/model_matrix/matrix_artifact_contract.py",
            "testing_support/decision_e2e/model_matrix/model_execution_provider.py",
            "testing_support/decision_e2e/model_matrix/qualification_analysis_strategy.py",
            "testing_support/decision_e2e/model_matrix/qualification_cohort_executor.py",
            "testing_support/decision_e2e/model_matrix/qualification_cohort_failure.py",
            "testing_support/decision_e2e/model_matrix/qualification_execution_pipeline.py",
        ),
    }


def _baseline_path(repo_root: Path) -> Path:
    return repo_root / _BASELINE_RELATIVE


def _group_check_name(group: str) -> str:
    if group == "R4.R4":
        return "group:R4.R4 proof"
    if group == "R4.R5":
        return "r4r5 semantic blob drift"
    return f"group:{group}"


def _append_r6_baseline_checks(
    repo_root: Path,
    checks: list[SourceFreezeCheck],
) -> None:
    baseline_file = _baseline_path(repo_root)
    if not baseline_file.is_file():
        checks.append(
            SourceFreezeCheck(
                name="r6_source_freeze_baseline",
                passed=False,
                detail=f"missing: {_BASELINE_RELATIVE}",
            )
        )
        return
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    frozen_config = baseline.get("configuration")
    if frozen_config != _FROZEN_CONFIGURATION:
        checks.append(
            SourceFreezeCheck(
                name="r6 configuration freeze",
                passed=False,
                detail="configuration mismatch",
            )
        )
    else:
        checks.append(
            SourceFreezeCheck(
                name="r6 configuration freeze",
                passed=True,
                detail="MATCH",
            )
        )
    frozen_blobs = baseline.get("blobs")
    if not isinstance(frozen_blobs, list):
        checks.append(
            SourceFreezeCheck(
                name="r6_source_freeze_baseline",
                passed=False,
                detail="invalid blobs",
            )
        )
        return
    frozen_map = {
        str(item["path"]): str(item["content_hash"])
        for item in frozen_blobs
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and isinstance(item.get("content_hash"), str)
    }
    head = resolve_repository_head_sha(repo_root)
    current = capture_semantic_source_fingerprint(
        repo_root,
        semantic_source_groups=semantic_source_groups_for_r6(),
        repository_head_sha=head,
    )
    current_map = {blob.path: blob.content_hash for blob in current.blobs}
    drift = [path for path, digest in frozen_map.items() if current_map.get(path) != digest]
    checks.append(
        SourceFreezeCheck(
            name="r6 matrix source drift",
            passed=not drift,
            detail="none" if not drift else f"{len(drift)} blob(s) changed",
        )
    )
    expected_fp = baseline.get("semantic_fingerprint")
    if isinstance(expected_fp, str):
        checks.append(
            SourceFreezeCheck(
                name="r6 analysis schema",
                passed=current.semantic_fingerprint() == expected_fp,
                detail="MATCH"
                if current.semantic_fingerprint() == expected_fp
                else "MISMATCH",
            )
        )
    scenario_id = baseline.get("scenario_id")
    checks.append(
        SourceFreezeCheck(
            name="r6 scenario version",
            passed=scenario_id == AI_INCIDENT_SCENARIO_ID,
            detail="MATCH" if scenario_id == AI_INCIDENT_SCENARIO_ID else "MISMATCH",
        )
    )
    matrix_version = baseline.get("matrix_version")
    checks.append(
        SourceFreezeCheck(
            name="r6 matrix version",
            passed=matrix_version == qualification_matrix_version(),
            detail="MATCH" if matrix_version == qualification_matrix_version() else "MISMATCH",
        )
    )


def verify_model_matrix_source_freeze(repo_root: Path) -> SourceFreezeReport:
    """R6 semantic baseline: inherited R4.R5 groups plus matrix identity and configuration."""
    inherited = verify_natural_alignment_source_freeze(repo_root)
    checks: list[SourceFreezeCheck] = list(inherited.checks)
    _append_r6_baseline_checks(repo_root, checks)

    r4r1 = capture_semantic_source_fingerprint(
        repo_root,
        semantic_source_groups=semantic_source_groups_for_r4r1(),
        repository_head_sha=resolve_repository_head_sha(repo_root),
    )
    checks.append(
        SourceFreezeCheck(
            name="r6 runtime source groups",
            passed=bool(r4r1.blobs),
            detail=f"{len(r4r1.blobs)} blob(s)",
        )
    )

    by_name = {check.name: check for check in checks}
    for group in _REQUIRED_GROUPS:
        upstream = _group_check_name(group)
        upstream_check = by_name.get(upstream)
        if upstream_check is None:
            checks.append(
                SourceFreezeCheck(
                    name=f"gate:{group}",
                    passed=False,
                    detail=f"missing upstream check {upstream}",
                )
            )
        elif not upstream_check.passed:
            checks.append(
                SourceFreezeCheck(
                    name=f"gate:{group}",
                    passed=False,
                    detail=f"upstream {upstream} failed",
                )
            )
        else:
            checks.append(
                SourceFreezeCheck(
                    name=f"gate:{group}",
                    passed=True,
                    detail="SATISFIED",
                )
            )

    all_passed = all(item.passed for item in checks)
    return SourceFreezeReport(
        status=SourceFreezeStatus.PASS if all_passed else SourceFreezeStatus.FAIL,
        checks=tuple(checks),
    )


def write_model_matrix_source_freeze_baseline(repo_root: Path) -> Path:
    """Capture R6 matrix source freeze baseline (maintainer bootstrap)."""
    head = resolve_repository_head_sha(repo_root)
    snapshot = capture_semantic_source_fingerprint(
        repo_root,
        semantic_source_groups=semantic_source_groups_for_r6(),
        repository_head_sha=head,
    )
    payload = {
        "task_id": TASK_ID,
        "scenario_id": AI_INCIDENT_SCENARIO_ID,
        "matrix_version": qualification_matrix_version(),
        "configuration": _FROZEN_CONFIGURATION,
        "repository_head_sha": snapshot.repository_head_sha,
        "semantic_fingerprint": snapshot.semantic_fingerprint(),
        "blobs": [
            {
                "path": blob.path,
                "content_hash": blob.content_hash,
                "semantic_group": blob.semantic_group,
            }
            for blob in snapshot.blobs
        ],
    }
    path = _baseline_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "TASK_ID",
    "_REQUIRED_GROUPS",
    "semantic_source_groups_for_r6",
    "verify_model_matrix_source_freeze",
    "write_model_matrix_source_freeze_baseline",
]
