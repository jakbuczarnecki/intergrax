# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q5 cross-domain qualification orchestrator."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_Q5_PACKAGE_DIR = Path(__file__).resolve().parent
for _path in (_REPO_ROOT, _REPO_ROOT / "agents", _REPO_ROOT / "applications"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from scripts.proof.intergrax_proof_environment import load_proof_environment

load_proof_environment(
    proof_package_dir=_Q5_PACKAGE_DIR,
    repository_root=_REPO_ROOT,
)

from intergrax.core.qualification.functional_qualification_reporting import write_qualification_run_report
from intergrax.core.qualification.functional_qualification_runner import run_qualification_plan
from intergrax.core.qualification.functional_qualification_verdict import QualificationVerdict
from tests.system.functional_diagnostics_q5.composition import (
    build_q5_qualification_plan,
    build_q5_qualification_registry,
)

_ARTIFACT_DIR = Path(
    os.environ.get(
        "DIAG_FUNCTIONAL_Q5_ARTIFACT_DIR",
        ".tmp/session/diag-functional-q5",
    ),
)


def main() -> int:
    registry = build_q5_qualification_registry()
    plan = build_q5_qualification_plan()
    report = run_qualification_plan(plan, registry)
    artifact_path = _ARTIFACT_DIR / "qualification-report.json"
    write_qualification_run_report(artifact_path, report)
    domains_passed = sum(
        1 for item in report.plugin_results if item.verdict is QualificationVerdict.PASS
    )
    total_cases = report.aggregate_metrics.total_cases
    summary = {
        "verdict": report.verdict.value,
        "plugins": len(report.plugin_results),
        "domains_passed": domains_passed,
        "cases": total_cases,
    }
    print(json.dumps(summary, indent=2))
    if report.verdict is QualificationVerdict.PASS:
        return 0
    if report.verdict is QualificationVerdict.BLOCKED:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
