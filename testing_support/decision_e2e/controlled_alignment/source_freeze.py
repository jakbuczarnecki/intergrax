# © Artur Czarnecki. All rights reserved.

"""Semantic source freeze for DS-E2E-15J-L1.R4.R4 (production groups only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from testing_support.decision_e2e.local_ai_incident_qualification import (
    resolve_repository_head_sha,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeCheck,
    SourceFreezeReport,
    SourceFreezeStatus,
)
from testing_support.decision_e2e.local_qualification_session.source_fingerprint import (
    capture_semantic_source_fingerprint,
)

TASK_ID = "DS-E2E-15J-L1.R4.R4"

_BASELINE_RELATIVE = (
    "testing_support/decision_e2e/controlled_alignment/data/source_freeze_baseline.json"
)


def semantic_source_groups_for_r4r4() -> dict[str, tuple[str, ...]]:
    """Frozen production groups required before controlled-alignment stimulus changes."""
    return {
        "15I": (
            "platform_proofs/scenarios/ai_incident_investigation/application/completion_revision_context.py",
        ),
        "15K-B": (
            "platform_proofs/scenarios/ai_incident_investigation/application/completion_alignment.py",
            "platform_proofs/scenarios/ai_incident_investigation/application/completion_alignment_correction.py",
        ),
        "O1": (
            "intergrax/runtime/nexus/tracing/execution/evaluator_model_attempt.py",
            "intergrax/runtime/nexus/tracing/execution/reconciliation_phase.py",
            "intergrax/runtime/observability/qualification_runtime_trace.py",
            "intergrax/runtime/nexus/execution/graph_executor.py",
            "intergrax/runtime/nexus/orchestration/graph_trace_callbacks.py",
            "intergrax/runtime/nexus/orchestration/graph_runner.py",
            "platform_proofs/scenarios/ai_incident_investigation/application/scenario.py",
        ),
        "O1.R1": (
            "intergrax/runtime/observability/qualification_runtime_trace.py",
            "intergrax/runtime/nexus/tracing/execution/reconciliation_phase.py",
        ),
        "O2": (
            "intergrax/runtime/diagnostics/completion_alignment_diag.py",
            "intergrax/runtime/observability/qualification_runtime_trace.py",
            "platform_proofs/scenarios/ai_incident_investigation/application/completion_alignment_telemetry.py",
            "platform_proofs/scenarios/ai_incident_investigation/application/investigator_agent.py",
            "platform_proofs/scenarios/ai_incident_investigation/application/scenario.py",
            "testing_support/decision_e2e/completion_alignment_producer_reachability.py",
            "testing_support/decision_e2e/local_qualification_session/trace_readback.py",
            "testing_support/decision_e2e/local_qualification_session/behavioral_coverage_evidence.py",
        ),
        "QI1": (
            "testing_support/decision_e2e/local_qualification_session/session.py",
            "testing_support/decision_e2e/local_qualification_session/contracts.py",
            "testing_support/decision_e2e/local_qualification_session/run_registry.py",
            "testing_support/decision_e2e/local_qualification_session/finalization.py",
        ),
        "QI2": (
            "testing_support/decision_e2e/local_ai_incident_qualification.py",
        ),
    }


def _baseline_path(repo_root: Path) -> Path:
    return repo_root / _BASELINE_RELATIVE


def write_source_freeze_baseline(repo_root: Path) -> Path:
    """Capture current semantic hashes for frozen groups (maintainer bootstrap)."""
    head = resolve_repository_head_sha(repo_root)
    snapshot = capture_semantic_source_fingerprint(
        repo_root,
        semantic_source_groups=semantic_source_groups_for_r4r4(),
        repository_head_sha=head,
    )
    payload = {
        "task_id": TASK_ID,
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
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def verify_controlled_alignment_source_freeze(repo_root: Path) -> SourceFreezeReport:
    checks: list[SourceFreezeCheck] = []
    baseline_file = _baseline_path(repo_root)
    if not baseline_file.is_file():
        checks.append(
            SourceFreezeCheck(
                name="source_freeze_baseline",
                passed=False,
                detail=f"missing: {_BASELINE_RELATIVE}",
            )
        )
        return SourceFreezeReport(SourceFreezeStatus.FAIL, tuple(checks))

    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    frozen_blobs = baseline.get("blobs")
    if not isinstance(frozen_blobs, list):
        checks.append(
            SourceFreezeCheck(
                name="source_freeze_baseline",
                passed=False,
                detail="invalid blobs",
            )
        )
        return SourceFreezeReport(SourceFreezeStatus.FAIL, tuple(checks))

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
        semantic_source_groups=semantic_source_groups_for_r4r4(),
        repository_head_sha=head,
    )
    current_map = {blob.path: blob.content_hash for blob in current.blobs}
    drift = [path for path, digest in frozen_map.items() if current_map.get(path) != digest]
    checks.append(
        SourceFreezeCheck(
            name="semantic source blob drift",
            passed=not drift,
            detail="none" if not drift else f"{len(drift)} blob(s) changed",
        )
    )
    expected_fp = baseline.get("semantic_fingerprint")
    if isinstance(expected_fp, str):
        checks.append(
            SourceFreezeCheck(
                name="semantic fingerprint",
                passed=current.semantic_fingerprint() == expected_fp,
                detail="MATCH"
                if current.semantic_fingerprint() == expected_fp
                else "MISMATCH",
            )
        )
    for group, group_paths in semantic_source_groups_for_r4r4().items():
        group_drift = bool(drift) and any(path in drift for path in group_paths)
        checks.append(
            SourceFreezeCheck(
                name=f"group:{group}",
                passed=not group_drift,
                detail="DRIFT" if group_drift else "IDENTICAL",
            )
        )

    all_passed = all(item.passed for item in checks)
    return SourceFreezeReport(
        status=SourceFreezeStatus.PASS if all_passed else SourceFreezeStatus.FAIL,
        checks=tuple(checks),
    )


@dataclass(frozen=True, slots=True)
class SourceFreezeGate:
    report: SourceFreezeReport

    @property
    def status(self) -> SourceFreezeStatus:
        return self.report.status


__all__ = [
    "TASK_ID",
    "SourceFreezeGate",
    "semantic_source_groups_for_r4r4",
    "verify_controlled_alignment_source_freeze",
    "write_source_freeze_baseline",
]
