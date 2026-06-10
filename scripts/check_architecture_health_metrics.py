#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-1.2 — architecture health metrics as live signals."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from intergrax.applications._shared.architecture_health_wiring import resolve_architecture_health_wiring  # noqa: E402
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_architecture_health_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable architecture health metrics", file=sys.stderr)
        return 1
    report = wiring.pipeline_report
    if report is None or not report.snapshots:
        print("architecture health pipeline report missing", file=sys.stderr)
        return 1
    summary = report.snapshots[0].report.summary
    if summary.nodes_total <= 0:
        print("architecture health metrics require non-empty capability graph", file=sys.stderr)
        return 1
    print(
        "OK: architecture health metrics "
        f"({len(report.snapshots)} snapshots, nodes={summary.nodes_total})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
