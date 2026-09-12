# © Artur Czarnecki. All rights reserved.

"""Semantic source freeze for DS-E2E-15J-L1.R4.R5 (production + R4.R4 proof)."""

from __future__ import annotations

import json
from pathlib import Path

from testing_support.decision_e2e.controlled_alignment.source_freeze import (
    verify_controlled_alignment_source_freeze,
)
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

TASK_ID = "DS-E2E-15J-L1.R4.R5"

_BASELINE_RELATIVE = (
    "testing_support/decision_e2e/natural_alignment/data/source_freeze_baseline.json"
)


def semantic_source_groups_for_r4r5() -> dict[str, tuple[str, ...]]:
    from testing_support.decision_e2e.controlled_alignment.source_freeze import (
        semantic_source_groups_for_r4r4,
    )

    groups = dict(semantic_source_groups_for_r4r4())
    groups["R4.R4 proof"] = (
        "testing_support/decision_e2e/controlled_alignment/runner.py",
        "testing_support/decision_e2e/controlled_alignment/scenario.py",
        "testing_support/decision_e2e/controlled_alignment/stimulus_llm.py",
        "testing_support/decision_e2e/controlled_alignment/stimulus_state.py",
        "testing_support/decision_e2e/controlled_alignment/evidence.py",
        "testing_support/decision_e2e/controlled_alignment/artifacts.py",
        "testing_support/decision_e2e/controlled_alignment/source_freeze.py",
    )
    return groups


def _baseline_path(repo_root: Path) -> Path:
    return repo_root / _BASELINE_RELATIVE


def write_natural_alignment_source_freeze_baseline(repo_root: Path) -> Path:
    head = resolve_repository_head_sha(repo_root)
    snapshot = capture_semantic_source_fingerprint(
        repo_root,
        semantic_source_groups=semantic_source_groups_for_r4r5(),
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


def verify_natural_alignment_source_freeze(repo_root: Path) -> SourceFreezeReport:
    checks: list[SourceFreezeCheck] = []

    r4r4 = verify_controlled_alignment_source_freeze(repo_root)
    checks.extend(r4r4.checks)

    baseline_file = _baseline_path(repo_root)
    if not baseline_file.is_file():
        checks.append(
            SourceFreezeCheck(
                name="r4r5_source_freeze_baseline",
                passed=False,
                detail=f"missing: {_BASELINE_RELATIVE}",
            )
        )
    else:
        baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
        frozen_blobs = baseline.get("blobs")
        if not isinstance(frozen_blobs, list):
            checks.append(
                SourceFreezeCheck(
                    name="r4r5_source_freeze_baseline",
                    passed=False,
                    detail="invalid blobs",
                )
            )
        else:
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
                semantic_source_groups=semantic_source_groups_for_r4r5(),
                repository_head_sha=head,
            )
            current_map = {blob.path: blob.content_hash for blob in current.blobs}
            drift = [
                path for path, digest in frozen_map.items() if current_map.get(path) != digest
            ]
            checks.append(
                SourceFreezeCheck(
                    name="r4r5 semantic blob drift",
                    passed=not drift,
                    detail="none" if not drift else f"{len(drift)} blob(s) changed",
                )
            )
            for group, group_paths in semantic_source_groups_for_r4r5().items():
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
