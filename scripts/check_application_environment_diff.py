#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CI gate — ApplicationEnvironmentDiff smoke for STRICT product hosts (APP-EVOL-6)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "applications", REPO_ROOT / "agents"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intergrax.applications._shared.environment_diff_wiring import (  # noqa: E402
    build_application_environment_diff,
)
from intergrax.applications._shared.product_manifest_registry import (  # noqa: E402
    iter_strict_product_manifests,
)
from intergrax.applications.contracts.application_environment_diff import DiffRiskLevel  # noqa: E402
from intergrax.applications.contracts.execution_mode import ExecutionMode  # noqa: E402


def main() -> int:
    violations: list[str] = []
    manifests = list(iter_strict_product_manifests())
    if len(manifests) < 2:
        violations.append("need at least two STRICT product manifests for cross-diff smoke")

    for product_id, manifest in manifests:
        env = manifest.resolved_environment()
        self_diff = build_application_environment_diff(manifest, env, manifest, env)
        if self_diff.risk_level is not DiffRiskLevel.LOW:
            violations.append(f"{product_id}: self-diff risk {self_diff.risk_level.value}")
        if self_diff.breaking_changes:
            violations.append(f"{product_id}: self-diff must not report breaking changes")
        if self_diff.left_snapshot_id != self_diff.right_snapshot_id:
            violations.append(f"{product_id}: self-diff snapshot ids must match")

    if len(manifests) >= 2:
        (left_id, left_manifest), (right_id, right_manifest) = manifests[0], manifests[1]
        cross_diff = build_application_environment_diff(
            left_manifest,
            left_manifest.resolved_environment(),
            right_manifest,
            right_manifest.resolved_environment(),
        )
        if not cross_diff.profile_diff.changed:
            violations.append(f"{left_id}->{right_id}: cross-diff profile must differ")
        if cross_diff.risk_level is DiffRiskLevel.LOW:
            violations.append(f"{left_id}->{right_id}: cross-diff risk must not be low")

    sample_manifest, sample_env = manifests[0][1], manifests[0][1].resolved_environment()
    strict_env = sample_env.model_copy(
        update={"execution_mode": ExecutionMode.STRICT},
    )
    relaxed_env = sample_env.model_copy(
        update={"execution_mode": ExecutionMode.BALANCED},
    )
    mode_diff = build_application_environment_diff(
        sample_manifest,
        relaxed_env,
        sample_manifest,
        strict_env,
    )
    if mode_diff.risk_level is not DiffRiskLevel.HIGH:
        violations.append("execution_mode delta must classify as high risk")
    if not any("execution_mode changed" in item for item in mode_diff.breaking_changes):
        violations.append("execution_mode delta must list breaking change")

    if violations:
        print("application environment diff gate: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1

    print("application environment diff gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
